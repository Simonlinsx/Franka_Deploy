from __future__ import annotations

import contextlib
import io
import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path

from inspire_mano_pipeline.calibration import PipelineCalibration
from inspire_mano_pipeline.model import HARDWARE_JOINTS
from inspire_mano_pipeline.rh56_stream import (
    REG_ANGLE_ACT,
    REG_ANGLE_SET,
    REG_CURRENT,
    REG_ERROR,
    REG_FORCE_SET,
    REG_POS_ACT,
    REG_SPEED_SET,
    REG_STATUS,
    REG_TEMP,
    SafeRH56Stream,
    StreamState,
    TargetFrame,
)
from realsense_mano_inspire import (
    CURRENT_MONITOR_ONLY_CONFIRMATION,
    NO_HAND_OPEN_CONFIRMATION,
    SIX_DOF_CONFIRMATION,
    build_parser,
    is_safe_six_dof_open1000_profile,
    open_fallback_targets,
    selected_axes_are_calibrated_open,
    validate_args,
)


ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / (
    "examples/inspire_mano_pipeline/"
    "inspire_rh56bfx_right_six_dof_open1000_realtime.json"
)
OLD_SIX_DOF_PROFILE = ROOT / (
    "examples/inspire_mano_pipeline/"
    "inspire_rh56bfx_right_six_dof_commissioning_900_800.json"
)
FIVE_FINGER_PROFILE = ROOT / (
    "examples/inspire_mano_pipeline/"
    "inspire_rh56bfx_right_five_finger_fullrange_realtime200.json"
)


class _SixDofHand:
    def __init__(self) -> None:
        self.angle_targets = (-1,) * 6
        self.angles = (1000,) * 6
        self.positions = (500,) * 6
        self.currents = (0,) * 6
        self.speeds = (1000,) * 6
        self.force_limits = (500,) * 6
        self.current_limits = (1400,) * 6
        self.write_log: list[tuple[int, tuple[int, ...]]] = []
        self.thread_ids: set[int] = set()

    def snapshot(self):
        self.thread_ids.add(threading.get_ident())
        return {
            "angle_targets": self.angle_targets,
            "errors": (0,) * 6,
            "temperatures": (30,) * 6,
            "statuses": (2,) * 6,
            "angles": self.angles,
            "positions": self.positions,
            "currents": self.currents,
            "speeds": self.speeds,
            "force_limits": self.force_limits,
            "current_limits": self.current_limits,
        }

    def write_six_shorts(self, address, values, retries=0):
        del retries
        self.thread_ids.add(threading.get_ident())
        values = tuple(int(value) for value in values)
        self.write_log.append((address, values))
        if address == REG_ANGLE_SET:
            self.angle_targets = values
            self.angles = tuple(
                before if target == -1 else target
                for before, target in zip(self.angles, values)
            )
        elif address == REG_SPEED_SET:
            self.speeds = values
        elif address == REG_FORCE_SET:
            self.force_limits = values
        else:
            raise AssertionError(f"unexpected write: {address}")

    def read_six_shorts(self, address, retries=0):
        del retries
        self.thread_ids.add(threading.get_ident())
        if address == REG_ANGLE_SET:
            return self.angle_targets
        if address == REG_ANGLE_ACT:
            return self.angles
        if address == REG_POS_ACT:
            return self.positions
        if address == REG_CURRENT:
            return self.currents
        if address == REG_SPEED_SET:
            return self.speeds
        if address == REG_FORCE_SET:
            return self.force_limits
        raise AssertionError(f"unexpected six-short read: {address}")

    def read(self, address, length, retries=0):
        del retries
        self.thread_ids.add(threading.get_ident())
        if length != 6:
            raise AssertionError(f"unexpected read length: {length}")
        if address == REG_ERROR:
            return bytes((0,) * 6)
        if address == REG_STATUS:
            return bytes((2,) * 6)
        if address == REG_TEMP:
            return bytes((30,) * 6)
        raise AssertionError(f"unexpected byte read: {address}")


class SixDofProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calibration = PipelineCalibration.load(PROFILE)

    def test_exact_mapping_filter_and_per_axis_speed(self) -> None:
        self.assertTrue(
            is_safe_six_dof_open1000_profile(
                self.calibration, HARDWARE_JOINTS
            )
        )
        self.assertEqual(
            tuple(
                name
                for name in HARDWARE_JOINTS
                if self.calibration.axes[name].enabled
            ),
            HARDWARE_JOINTS,
        )
        self.assertEqual(self.calibration.temporal_filter_axes, HARDWARE_JOINTS)
        self.assertEqual(self.calibration.active_current_policy, "monitor_only")
        self.assertEqual(
            self.calibration.map_qpos((0.05, 0.05, 0.05, 0.08, 0.0, 0.20)),
            (1000,) * 6,
        )
        self.assertEqual(
            self.calibration.map_qpos((0.55, 0.58, 0.47, 0.40, 0.60, 0.95)),
            (0, 0, 0, 0, 0, 900),
        )
        q6 = self.calibration.axes["thumb_rotate"]
        self.assertEqual(q6.map_qpos(0.575), 950)
        self.assertGreaterEqual(q6.map_qpos(0.248), 990)
        self.assertGreaterEqual(q6.map_qpos(0.873), 900)
        self.assertEqual(q6.hardware_speed, 80)
        self.assertEqual(q6.max_rate_units_per_second, 40.0)

    def test_content_gate_rejects_safety_or_axis_mutations(self) -> None:
        q6 = self.calibration.axes["thumb_rotate"]
        mutations = (
            replace(
                self.calibration,
                axes={
                    **self.calibration.axes,
                    "thumb_rotate": replace(q6, command_closed=899),
                },
            ),
            replace(
                self.calibration,
                axes={
                    **self.calibration.axes,
                    "thumb_rotate": replace(q6, hardware_speed=200),
                },
            ),
            replace(
                self.calibration,
                axes={
                    **self.calibration.axes,
                    "thumb_rotate": replace(
                        q6, feedback_to_command_valid_min=840
                    ),
                },
            ),
            replace(self.calibration, speed=201),
            replace(self.calibration, active_current_policy="fault"),
            replace(self.calibration, stream_max_current_ma=401),
            replace(self.calibration, stream_max_total_current_ma=601),
            replace(
                self.calibration,
                temporal_filter_axes=HARDWARE_JOINTS[:-1],
            ),
        )
        for changed in mutations:
            with self.subTest(changed=changed):
                self.assertFalse(
                    is_safe_six_dof_open1000_profile(
                        changed, HARDWARE_JOINTS
                    )
                )
        self.assertFalse(
            is_safe_six_dof_open1000_profile(
                self.calibration, tuple(reversed(HARDWARE_JOINTS))
            )
        )

    def test_hardware_speed_loader_is_strict_and_bounded(self) -> None:
        raw = json.loads(PROFILE.read_text(encoding="utf-8"))
        for invalid in (True, 80.0, "80", 0, 1001):
            changed = json.loads(json.dumps(raw))
            changed["axes"]["thumb_rotate"]["hardware_speed"] = invalid
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "calibration.json"
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    PipelineCalibration.load(path)

    def test_active_current_policy_loader_is_strict(self) -> None:
        raw = json.loads(PROFILE.read_text(encoding="utf-8"))
        for invalid in (True, None, "warn", "fault"):
            changed = json.loads(json.dumps(raw))
            changed["safety"]["active_current_policy"] = invalid
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "calibration.json"
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(invalid=invalid):
                    if invalid == "fault":
                        loaded = PipelineCalibration.load(path)
                        self.assertFalse(
                            is_safe_six_dof_open1000_profile(
                                loaded, HARDWARE_JOINTS
                            )
                        )
                    else:
                        with self.assertRaises(ValueError):
                            PipelineCalibration.load(path)

    def test_open_fallback_and_start_pose_cover_all_six_axes(self) -> None:
        self.assertEqual(
            open_fallback_targets(self.calibration, HARDWARE_JOINTS),
            (1000,) * 6,
        )
        self.assertTrue(
            selected_axes_are_calibrated_open(
                self.calibration, HARDWARE_JOINTS, (1000,) * 6
            )
        )
        self.assertFalse(
            selected_axes_are_calibrated_open(
                self.calibration,
                HARDWARE_JOINTS,
                (1000, 1000, 1000, 1000, 1000, 969),
            )
        )


class SixDofCliGateTests(unittest.TestCase):
    @staticmethod
    def _arguments(profile=PROFILE, *, duration="0"):
        return [
            "--source",
            "realsense",
            "--camera-serial",
            "fake-camera",
            "--device",
            "cuda:0",
            "--calibration",
            str(profile),
            "--retargeter",
            "geometric",
            "--enable-hardware",
            "--confirm-hardware-motion",
            "--confirm-wide-range",
            "RH56_WIDE_RANGE",
            "--confirm-six-dof-motion",
            SIX_DOF_CONFIRMATION,
            "--confirm-current-monitor-only",
            CURRENT_MONITOR_ONLY_CONFIRMATION,
            "--port",
            "/dev/fake-hand",
            "--axes",
            ",".join(HARDWARE_JOINTS),
            "--no-hand-policy",
            "open",
            "--confirm-no-hand-open",
            NO_HAND_OPEN_CONFIRMATION,
            "--duration",
            duration,
        ]

    @staticmethod
    def _validate(arguments) -> None:
        parser = build_parser()
        validate_args(parser.parse_args(arguments), parser)

    def test_exact_tokens_allow_limited_or_unlimited_six_dof(self) -> None:
        for duration in ("20", "0"):
            with self.subTest(duration=duration):
                self._validate(self._arguments(duration=duration))

    def test_each_six_dof_confirmation_is_required(self) -> None:
        cases = (
            ("--confirm-wide-range", "RH56_WIDE_RANGE"),
            ("--confirm-six-dof-motion", SIX_DOF_CONFIRMATION),
            (
                "--confirm-current-monitor-only",
                CURRENT_MONITOR_ONLY_CONFIRMATION,
            ),
            ("--confirm-no-hand-open", NO_HAND_OPEN_CONFIRMATION),
        )
        for flag, value in cases:
            arguments = self._arguments()
            index = arguments.index(flag)
            del arguments[index : index + 2]
            with self.subTest(flag=flag), contextlib.redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit):
                self._validate(arguments)
            arguments = self._arguments()
            arguments[arguments.index(value)] = "WRONG"
            with self.subTest(flag=flag, wrong=True), contextlib.redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit):
                self._validate(arguments)

    def test_old_or_disabled_profiles_are_rejected_before_device_open(self) -> None:
        for profile in (OLD_SIX_DOF_PROFILE, FIVE_FINGER_PROFILE):
            with self.subTest(profile=profile), contextlib.redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit):
                self._validate(self._arguments(profile=profile))


class SixDofStreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calibration = PipelineCalibration.load(PROFILE)

    def test_open_endpoint_seed_saturates_and_q6_uses_speed_80(self) -> None:
        hand = _SixDofHand()

        @contextlib.contextmanager
        def factory():
            yield hand

        stream = SafeRH56Stream(
            self.calibration,
            selected_axes=HARDWARE_JOINTS,
            hand_context_factory=factory,
        )
        stream.start()
        stream.wait_until_ready(2.0)
        self.assertEqual(stream.initial_command_seed, (1000,) * 6)
        self.assertEqual(
            stream.verified_streaming_speeds,
            (200, 200, 200, 200, 200, 80),
        )
        self.assertEqual(stream.verified_device_current_limits, (1400,) * 6)
        self.assertFalse(
            any(
                address == REG_ANGLE_SET and values != (-1,) * 6
                for address, values in hand.write_log
            )
        )
        stream.close()
        self.assertEqual(stream.state, StreamState.STOPPED)
        self.assertTrue(stream.stop_confirmed)
        self.assertTrue(stream.physical_stop_verified)
        self.assertEqual(stream.final_angle_targets, (-1,) * 6)
        self.assertEqual(hand.speeds, (1000,) * 6)

    def test_monitor_only_requires_enabled_device_current_limits(self) -> None:
        hand = _SixDofHand()
        hand.current_limits = (1400, 1400, 1400, 1400, 1400, 0)

        @contextlib.contextmanager
        def factory():
            yield hand

        stream = SafeRH56Stream(
            self.calibration,
            selected_axes=HARDWARE_JOINTS,
            hand_context_factory=factory,
        )
        stream.start()
        with self.assertRaisesRegex(Exception, "enabled device CURRENT_LIMIT"):
            stream.wait_until_ready(2.0)
        stream.close()
        self.assertFalse(stream.ever_active)
        self.assertEqual(stream.motion_write_count, 0)
        self.assertEqual(stream.final_angle_targets, (-1,) * 6)

    def test_selected_six_axis_total_current_606_warns_and_continues(self) -> None:
        class CurrentHand(_SixDofHand):
            def write_six_shorts(self, address, values, retries=0):
                super().write_six_shorts(address, values, retries=retries)
                if address == REG_ANGLE_SET and tuple(values) != (-1,) * 6:
                    self.currents = (101,) * 6
                elif address == REG_ANGLE_SET:
                    self.currents = (0,) * 6

        hand = CurrentHand()

        @contextlib.contextmanager
        def factory():
            yield hand

        fast = replace(
            self.calibration,
            valid_frames_to_arm=1,
            preflight_stability_seconds=0.01,
            control_hz=50.0,
            feedback_hz=50.0,
        )
        stream = SafeRH56Stream(
            fast,
            selected_axes=HARDWARE_JOINTS,
            hand_context_factory=factory,
        )
        stream.start()
        stream.wait_until_ready(2.0)
        now = time.monotonic()
        self.assertTrue(
            stream.submit(
                TargetFrame(
                    targets=(900, 900, 900, 900, 900, 950),
                    captured_at_monotonic=now,
                    frame_number=1,
                    source="mano",
                    depth_evidence_at_monotonic=now,
                )
            )
        )
        deadline = time.monotonic() + 2.0
        while (
            stream.active_current_over_limit_sample_count == 0
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        self.assertEqual(stream.state, StreamState.ACTIVE)
        self.assertIsNone(stream.error)
        self.assertTrue(stream.active_current_threshold_exceeded)
        self.assertGreater(stream.active_current_over_limit_sample_count, 0)
        self.assertGreater(stream.active_current_warning_event_count, 0)
        self.assertEqual(stream.active_peak_abs_currents, (101,) * 6)
        self.assertEqual(stream.active_max_selected_total_current_ma, 606)
        stream.close()
        self.assertEqual(stream.state, StreamState.STOPPED)
        self.assertTrue(stream.physical_stop_verified)
        self.assertEqual(stream.final_angle_targets, (-1,) * 6)

    def test_persistent_post_release_current_keeps_low_settings_and_is_unconfirmed(
        self,
    ) -> None:
        class PersistentCurrentHand(_SixDofHand):
            def write_six_shorts(self, address, values, retries=0):
                super().write_six_shorts(address, values, retries=retries)
                if address == REG_ANGLE_SET and tuple(values) != (-1,) * 6:
                    self.currents = (101,) * 6

        hand = PersistentCurrentHand()

        @contextlib.contextmanager
        def factory():
            yield hand

        fast = replace(
            self.calibration,
            valid_frames_to_arm=1,
            preflight_stability_seconds=0.01,
            control_hz=50.0,
            feedback_hz=50.0,
        )
        stream = SafeRH56Stream(
            fast,
            selected_axes=HARDWARE_JOINTS,
            hand_context_factory=factory,
        )
        stream.start()
        stream.wait_until_ready(2.0)
        now = time.monotonic()
        self.assertTrue(
            stream.submit(
                TargetFrame(
                    targets=(900, 900, 900, 900, 900, 950),
                    captured_at_monotonic=now,
                    frame_number=1,
                    source="mano",
                    depth_evidence_at_monotonic=now,
                )
            )
        )
        deadline = time.monotonic() + 2.0
        while (
            stream.active_current_over_limit_sample_count == 0
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        self.assertEqual(stream.state, StreamState.ACTIVE)
        stream.close()
        self.assertEqual(stream.state, StreamState.STOP_UNCONFIRMED)
        self.assertFalse(stream.stop_confirmed)
        self.assertFalse(stream.physical_stop_verified)
        self.assertIn("physical stop verification failed", stream.stop_reason)
        self.assertEqual(hand.speeds, (200, 200, 200, 200, 200, 80))
        self.assertEqual(hand.force_limits, (80,) * 6)
        self.assertEqual(stream.final_angle_targets, (-1,) * 6)

    def test_device_current_protection_status_still_faults_in_monitor_mode(
        self,
    ) -> None:
        class DeviceProtectedHand(_SixDofHand):
            def __init__(self) -> None:
                super().__init__()
                self.statuses = (2,) * 6

            def write_six_shorts(self, address, values, retries=0):
                super().write_six_shorts(address, values, retries=retries)
                if address == REG_ANGLE_SET and tuple(values) != (-1,) * 6:
                    self.statuses = (5, 1, 1, 1, 1, 1)
                elif address == REG_ANGLE_SET:
                    self.statuses = (2,) * 6

            def read(self, address, length, retries=0):
                if address == REG_STATUS:
                    return bytes(self.statuses)
                return super().read(address, length, retries=retries)

        hand = DeviceProtectedHand()

        @contextlib.contextmanager
        def factory():
            yield hand

        fast = replace(
            self.calibration,
            valid_frames_to_arm=1,
            preflight_stability_seconds=0.01,
            control_hz=50.0,
            feedback_hz=50.0,
        )
        stream = SafeRH56Stream(
            fast,
            selected_axes=HARDWARE_JOINTS,
            hand_context_factory=factory,
        )
        stream.start()
        stream.wait_until_ready(2.0)
        now = time.monotonic()
        self.assertTrue(
            stream.submit(
                TargetFrame(
                    targets=(900, 900, 900, 900, 900, 950),
                    captured_at_monotonic=now,
                    frame_number=1,
                    source="mano",
                    depth_evidence_at_monotonic=now,
                )
            )
        )
        deadline = time.monotonic() + 2.0
        while stream.state not in (
            StreamState.FAULT_LATCHED,
            StreamState.STOP_UNCONFIRMED,
        ) and time.monotonic() < deadline:
            time.sleep(0.01)
        stream.close()
        self.assertEqual(stream.state, StreamState.FAULT_LATCHED)
        self.assertIn("unsafe status 5", str(stream.error))
        self.assertTrue(stream.stop_confirmed)


if __name__ == "__main__":
    unittest.main()
