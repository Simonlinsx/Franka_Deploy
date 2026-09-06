from __future__ import annotations

import contextlib
import io
import sys
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import inspire_rh56_test
from inspire_mano_pipeline.calibration import PipelineCalibration
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
    NO_HAND_OPEN_CONFIRMATION,
    build_parser,
    is_safe_commissioning_profile,
    open_fallback_targets,
    selected_axes_are_calibrated_open,
    validate_args,
)


ROOT = Path(__file__).resolve().parents[3]
OPEN1000_PATH = ROOT / (
    "examples/inspire_mano_pipeline/"
    "inspire_rh56bfx_right_commissioning_open1000.json"
)


class _Open1000FakeHand:
    """In-memory RH56 model; it never constructs or opens a serial transport."""

    def __init__(self) -> None:
        self.angle_targets = (-1,) * 6
        self.angles = (900, 900, 900, 1000, 900, 900)
        self.positions = (500,) * 6
        self.speeds = (500,) * 6
        self.force_limits = (500,) * 6
        self.write_log: list[tuple[int, tuple[int, ...]]] = []

    def snapshot(self):
        return {
            "angle_targets": self.angle_targets,
            "errors": (0,) * 6,
            "temperatures": (25,) * 6,
            "statuses": (2,) * 6,
            "angles": self.angles,
            "currents": (0,) * 6,
            "speeds": self.speeds,
            "force_limits": self.force_limits,
        }

    def write_six_shorts(self, address, values, retries=0):
        del retries
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
        else:  # pragma: no cover - catches future unexpected writes.
            raise AssertionError(f"unexpected write address: {address}")

    def read(self, address, length, retries=0):
        del retries
        if length != 6:
            raise AssertionError(f"unexpected read length: {length}")
        if address == REG_ERROR:
            return bytes((0,) * 6)
        if address == REG_STATUS:
            return bytes((2,) * 6)
        if address == REG_TEMP:
            return bytes((25,) * 6)
        raise AssertionError(f"unexpected read address: {address}")

    def read_six_shorts(self, address, retries=0):
        del retries
        if address == REG_ANGLE_ACT:
            return self.angles
        if address == REG_POS_ACT:
            return self.positions
        if address == REG_ANGLE_SET:
            return self.angle_targets
        if address == REG_SPEED_SET:
            return self.speeds
        if address == REG_FORCE_SET:
            return self.force_limits
        if address == REG_CURRENT:
            return (0,) * 6
        raise AssertionError(f"unexpected six-short address: {address}")


class Open1000CommissioningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calibration = PipelineCalibration.load(OPEN1000_PATH)

    @staticmethod
    def _wait_until(predicate, timeout=1.0):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.002)
        return predicate()

    def test_profile_is_exactly_bounded_for_first_index_motion(self):
        index = self.calibration.axes["index"]
        self.assertTrue(is_safe_commissioning_profile(self.calibration, ("index",)))
        self.assertEqual((index.q_open, index.q_closed), (0.05, 0.65))
        self.assertEqual((index.command_open, index.command_closed), (1000, 800))
        self.assertEqual(index.map_qpos(0.0), 1000)
        self.assertEqual(index.map_qpos(0.05), 1000)
        self.assertEqual(index.map_qpos(0.35), 900)
        self.assertEqual(index.map_qpos(0.65), 800)
        self.assertEqual(index.map_qpos(1.47), 800)
        self.assertEqual(index.max_rate_units_per_second, 80.0)
        self.assertEqual(self.calibration.speed, 80)
        self.assertEqual(self.calibration.force_limit, 80)
        self.assertEqual(
            open_fallback_targets(self.calibration, ("index",)),
            (-1, -1, -1, 1000, -1, -1),
        )
        self.assertTrue(
            selected_axes_are_calibrated_open(
                self.calibration,
                ("index",),
                (0, 0, 0, 1000, 0, 0),
            )
        )
        self.assertFalse(
            selected_axes_are_calibrated_open(
                self.calibration,
                ("index",),
                (0, 0, 0, 969, 0, 0),
            )
        )

    def test_20_second_visible_hardware_cli_passes_without_opening_devices(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--source",
                "realsense",
                "--camera-serial",
                "337322072188",
                "--device",
                "cuda:0",
                "--calibration",
                str(OPEN1000_PATH),
                "--retargeter",
                "geometric",
                "--enable-hardware",
                "--confirm-hardware-motion",
                "--port",
                "/dev/ttyUSB0",
                "--axes",
                "index",
                "--no-hand-policy",
                "open",
                "--confirm-no-hand-open",
                NO_HAND_OPEN_CONFIRMATION,
                "--duration",
                "20",
            ]
        )
        validate_args(args, parser)
        self.assertFalse(args.headless)
        self.assertEqual(args.axes, ("index",))
        self.assertEqual(args.duration, 20.0)

    def test_hardware_rejects_auto_cpu_and_unindexed_cuda(self):
        common = [
            "--source",
            "realsense",
            "--camera-serial",
            "fake-camera",
            "--calibration",
            str(OPEN1000_PATH),
            "--retargeter",
            "geometric",
            "--enable-hardware",
            "--confirm-hardware-motion",
            "--port",
            "/dev/fake-hand",
            "--axes",
            "index",
            "--duration",
            "20",
        ]
        for device_arguments in ([], ["--device", "cpu"], ["--device", "cuda"]):
            with self.subTest(device_arguments=device_arguments):
                parser = build_parser()
                args = parser.parse_args(common + device_arguments)
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
                    SystemExit
                ):
                    validate_args(args, parser)

    def test_tokenless_gate_rejects_relaxed_watchdogs_and_feedback(self):
        unsafe_variants = (
            replace(self.calibration, tracking_timeout_seconds=0.80),
            replace(self.calibration, control_hz=10.0),
            replace(self.calibration, feedback_hz=2.0),
            replace(self.calibration, preflight_max_idle_current_ma=301),
            replace(self.calibration, detector_confidence=0.49),
        )
        for calibration in unsafe_variants:
            with self.subTest(calibration=calibration):
                self.assertFalse(
                    is_safe_commissioning_profile(calibration, ("index",))
                )

    def test_simulated_stream_starts_at_1000_and_fallback_returns_toward_1000(self):
        calibration = replace(
            self.calibration,
            preflight_stability_seconds=0.01,
            control_hz=100.0,
            feedback_hz=25.0,
        )
        fake = _Open1000FakeHand()

        @contextlib.contextmanager
        def factory():
            yield fake

        stream = SafeRH56Stream(
            calibration,
            selected_axes=("index",),
            hand_context_factory=factory,
        )
        stream.start()
        stream.wait_until_ready(1.0)
        try:
            self.assertEqual(stream.latest_angles[3], 1000)
            for frame_number in range(1, 6):
                self.assertTrue(
                    stream.submit(
                        TargetFrame(
                            targets=(-1, -1, -1, 800, -1, -1),
                            captured_at_monotonic=time.monotonic(),
                            frame_number=frame_number,
                            source="mano",
                        )
                    )
                )
                time.sleep(0.02)
            self.assertTrue(
                self._wait_until(
                    lambda: stream.state == StreamState.ACTIVE
                    and stream.last_sent_targets is not None
                    and stream.last_sent_targets[3] < 1000
                )
            )
            closing_value = stream.last_sent_targets[3]
            self.assertTrue(
                stream.submit(
                    TargetFrame(
                        targets=(-1, -1, -1, 1000, -1, -1),
                        captured_at_monotonic=time.monotonic(),
                        frame_number=6,
                        source="fallback_open",
                    )
                )
            )
            self.assertTrue(
                self._wait_until(
                    lambda: stream.last_sent_targets is not None
                    and stream.last_sent_targets[3] > closing_value
                )
            )
            numeric_index_writes = [
                values[3]
                for address, values in fake.write_log
                if address == REG_ANGLE_SET and values != (-1,) * 6
            ]
            self.assertTrue(all(800 <= value <= 1000 for value in numeric_index_writes))
            self.assertEqual(stream.accepted_mano_target_count, 5)
            self.assertEqual(stream.accepted_fallback_open_target_count, 1)
        finally:
            stream.close()
        self.assertTrue(stream.stop_confirmed)
        self.assertEqual(stream.final_angle_targets, (-1,) * 6)


class DisableCliTests(unittest.TestCase):
    class _FakeDisableHand:
        def __init__(self) -> None:
            self.targets = (100, 200, 300, 400, 500, 600)
            self.events: list[tuple] = []

        def snapshot(self):  # pragma: no cover - calling this fails the test.
            raise AssertionError("disable must not wait for a diagnostic snapshot")

        def write_six_shorts(self, address, values, retries=0):
            values = tuple(int(value) for value in values)
            self.events.append(("write", address, values, retries))
            self.targets = values

        def read_six_shorts(self, address, retries=0):
            self.events.append(("read", address, retries))
            return self.targets

    def test_disable_all_reasserts_and_verifies_all_six_minus_one(self):
        hand = self._FakeDisableHand()
        inspire_rh56_test.disable_all(hand)
        self.assertEqual(
            hand.events,
            [
                ("write", inspire_rh56_test.REG_ANGLE_SET, (-1,) * 6, 1),
                ("read", inspire_rh56_test.REG_ANGLE_SET, 1),
                ("write", inspire_rh56_test.REG_ANGLE_SET, (-1,) * 6, 1),
                ("read", inspire_rh56_test.REG_ANGLE_SET, 1),
            ],
        )

    def test_disable_cli_never_constructs_a_real_serial_or_takes_snapshot(self):
        hand = self._FakeDisableHand()
        serial_events: list[tuple] = []

        class FakeLinuxSerial:
            def __init__(self, port, baud, timeout, debug):
                serial_events.append(("construct", port, baud, timeout, debug))

            def __enter__(self):
                serial_events.append(("enter",))
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                del exc_type, exc_value, traceback
                serial_events.append(("exit",))

        argv = [
            "inspire_rh56_test.py",
            "disable",
            "--port",
            "/dev/fake-no-serial",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            inspire_rh56_test, "LinuxSerial", FakeLinuxSerial
        ), mock.patch.object(
            inspire_rh56_test, "RH56Hand", return_value=hand
        ):
            self.assertEqual(inspire_rh56_test.main(), 0)
        self.assertEqual(serial_events[0][0], "construct")
        self.assertEqual(serial_events[-1], ("exit",))
        self.assertEqual(hand.events[0][0], "write")


if __name__ == "__main__":
    unittest.main()
