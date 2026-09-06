from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import pickle
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from inspire_mano_pipeline.calibration import PipelineCalibration
from inspire_mano_pipeline.model import CameraFrame, HARDWARE_JOINTS, ManoDetection
from inspire_mano_pipeline.retargeting import (
    DEFAULT_DEX_ROOT,
    DexInspireRetargeter,
    GeometricInspireRetargeter,
)
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
from inspire_mano_pipeline.wilor_backend import (
    WiLoRBackend,
    canonicalize_for_dex,
    canonicalize_mano,
)
from realsense_mano_inspire import build_parser, validate_args, write_record


ROOT = Path(__file__).resolve().parents[3]
CALIBRATION_PATH = (
    ROOT / "examples/inspire_mano_pipeline/inspire_rh56bfx_right.json"
)
DEX_HUMAN_DATA = (
    DEFAULT_DEX_ROOT / "example/profiling/human_joint_right.pkl"
)
DEX_RETARGETING_AVAILABLE = importlib.util.find_spec("dex_retargeting") is not None


def open_hand_joints() -> np.ndarray:
    joints = np.zeros((21, 3), dtype=np.float32)
    joints[0] = (0.0, 0.0, 0.0)
    joints[1:5] = np.asarray(
        [(0.018, 0.018, 0), (0.036, 0.027, 0), (0.052, 0.035, 0), (0.068, 0.043, 0)]
    )
    for base, x, y in ((5, 0.030, 0.040), (9, 0.010, 0.044), (13, -0.010, 0.042), (17, -0.030, 0.037)):
        joints[base] = (x, y, 0.0)
        joints[base + 1] = (x, y + 0.030, 0.0)
        joints[base + 2] = (x, y + 0.052, 0.0)
        joints[base + 3] = (x, y + 0.070, 0.0)
    return joints


def thumb_flexed_joints(angle_degrees: float = 52.0) -> np.ndarray:
    joints = open_hand_joints()
    angle = np.deg2rad(angle_degrees)
    directions = np.asarray([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)
    joints[2] = joints[1] + 0.020 * directions
    joints[3] = joints[2] + 0.018 * directions
    joints[4] = joints[3] + 0.017 * directions
    return joints


def detection_for(joints: np.ndarray) -> ManoDetection:
    return ManoDetection(
        is_right=True,
        bbox_xyxy=np.asarray([100, 100, 300, 400], dtype=np.float32),
        keypoints_3d_raw=joints.copy(),
        keypoints_3d_canonical=joints.copy(),
        keypoints_2d=np.zeros((21, 2), dtype=np.float32),
        global_orient=np.zeros(3, dtype=np.float32),
        hand_pose=np.zeros((15, 3), dtype=np.float32),
        betas=np.zeros(10, dtype=np.float32),
        vertices=np.zeros((778, 3), dtype=np.float32),
        captured_at_monotonic=time.monotonic(),
        frame_number=1,
    )


class CalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calibration = PipelineCalibration.load(CALIBRATION_PATH)

    def test_qpos_endpoints_and_disabled_thumb_rotation(self) -> None:
        opened = self.calibration.map_qpos([0, 0, 0, 0, 0, 0])
        closed = self.calibration.map_qpos([1.47, 1.47, 1.47, 1.47, 0.6, 1.308])
        self.assertEqual(opened, (900, 900, 900, 900, 900, -1))
        self.assertEqual(closed, (100, 100, 100, 100, 150, -1))

    def test_non_finite_calibration_is_rejected(self) -> None:
        data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        data["safety"]["tracking_timeout_seconds"] = "nan"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finite"):
                PipelineCalibration.load(path)

    def test_string_boolean_is_rejected(self) -> None:
        data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        data["axes"]["index"]["enabled"] = "false"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON boolean"):
                PipelineCalibration.load(path)

    def test_reversed_rh56_direction_is_rejected(self) -> None:
        data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        data["axes"]["index"]["command_open"] = 100
        data["axes"]["index"]["command_closed"] = 900
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "command_open"):
                PipelineCalibration.load(path)

    def test_total_current_limit_defaults_to_legacy_equivalent(self) -> None:
        self.assertEqual(
            self.calibration.stream_max_total_current_ma,
            6 * self.calibration.stream_max_current_ma,
        )

    def test_active_current_policy_defaults_to_fault_and_is_strict(self) -> None:
        self.assertEqual(self.calibration.active_current_policy, "fault")
        data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        for invalid in (True, None, "warn", "monitor"):
            changed = json.loads(json.dumps(data))
            changed["safety"]["active_current_policy"] = invalid
            with self.subTest(value=invalid), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bad.json"
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "active_current_policy"):
                    PipelineCalibration.load(path)

    def test_total_current_limit_is_strict_and_bounded(self) -> None:
        data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        per_axis_limit = data["safety"]["stream_max_current_ma"]
        for invalid in (
            True,
            per_axis_limit - 1,
            6 * per_axis_limit + 1,
        ):
            with self.subTest(value=invalid), tempfile.TemporaryDirectory() as directory:
                data["safety"]["stream_max_total_current_ma"] = invalid
                path = Path(directory) / "bad.json"
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaisesRegex(
                    ValueError,
                    "stream_max_total_current_ma",
                ):
                    PipelineCalibration.load(path)

    def test_geometric_open_hand_is_open(self) -> None:
        output = GeometricInspireRetargeter(self.calibration).retarget(
            detection_for(open_hand_joints())
        )
        np.testing.assert_allclose(output.qpos[:5], np.zeros(5), atol=1e-6)
        self.assertEqual(output.hardware_targets[:5], (900, 900, 900, 900, 900))

    def test_geometric_right_thumb_flex_reaches_closed_endpoint(self) -> None:
        output = GeometricInspireRetargeter(self.calibration).retarget(
            detection_for(thumb_flexed_joints())
        )
        self.assertAlmostEqual(float(output.raw_qpos[4]), 0.60, places=6)
        self.assertAlmostEqual(float(output.qpos[4]), 0.60, places=6)
        self.assertEqual(output.hardware_targets[4], 150)

    def test_geometric_thumb_bend_is_rigid_and_scale_invariant(self) -> None:
        joints = thumb_flexed_joints(angle_degrees=41.0)
        rotation, _ = cv2.Rodrigues(
            np.asarray([0.37, -0.21, 0.44], dtype=np.float32)
        )
        transformed = 1.4 * (joints @ rotation.T) + np.asarray(
            [0.2, -0.3, 0.7], dtype=np.float32
        )
        baseline = GeometricInspireRetargeter(self.calibration).retarget(
            detection_for(joints)
        )
        self.assertGreater(float(baseline.raw_qpos[4]), 0.0)
        self.assertLess(float(baseline.raw_qpos[4]), 0.60)
        actual = GeometricInspireRetargeter(self.calibration).retarget(
            detection_for(transformed)
        )
        self.assertAlmostEqual(
            float(actual.raw_qpos[4]), float(baseline.raw_qpos[4]), places=6
        )

    def test_geometric_thumb_rotation_keeps_tip_opposition_metric(self) -> None:
        joints = thumb_flexed_joints()
        output = GeometricInspireRetargeter(self.calibration).retarget(
            detection_for(joints)
        )
        palm_width = float(np.linalg.norm(joints[5] - joints[17]))
        thumb_to_index = float(np.linalg.norm(joints[4] - joints[5])) / palm_width
        expected = 1.308 * np.clip((1.55 - thumb_to_index) / 1.15, 0.0, 1.0)
        self.assertAlmostEqual(float(output.raw_qpos[5]), expected, places=6)

    def test_geometric_thumb_bend_rejects_degenerate_palm_frame(self) -> None:
        joints = open_hand_joints()
        across = joints[5] - joints[17]
        joints[9] = joints[0] + 0.5 * across
        with self.assertRaisesRegex(ValueError, "palm longitudinal axis"):
            GeometricInspireRetargeter(self.calibration).retarget(
                detection_for(joints)
            )

    def test_geometric_thumb_bend_rejects_degenerate_proximal_segment(self) -> None:
        joints = open_hand_joints()
        joints[2] = joints[1]
        with self.assertRaisesRegex(ValueError, "thumb proximal segment"):
            GeometricInspireRetargeter(self.calibration).retarget(
                detection_for(joints)
            )

    def test_mano_canonicalization_removes_root_pose(self) -> None:
        canonical = open_hand_joints()
        rotation_vector = np.asarray([0.22, -0.31, 0.48], dtype=np.float32)
        rotation, _ = cv2.Rodrigues(rotation_vector)
        translation = np.asarray([0.3, -0.2, 0.8], dtype=np.float32)
        raw = canonical @ rotation.T + translation
        recovered = canonicalize_mano(raw, rotation_vector)
        np.testing.assert_allclose(recovered, canonical, atol=2e-6)

    def test_dex_wrist_frame_is_rigid_transform_invariant(self) -> None:
        canonical = open_hand_joints()
        rotation_vector = np.asarray([-0.41, 0.17, 0.28], dtype=np.float32)
        rotation, _ = cv2.Rodrigues(rotation_vector)
        moved = canonical @ rotation.T + np.asarray([0.2, 0.4, -0.3])
        expected = canonicalize_for_dex(canonical)
        actual = canonicalize_for_dex(moved)
        np.testing.assert_allclose(actual, expected, atol=2e-6)


class DexRetargetingTests(unittest.TestCase):
    @unittest.skipUnless(
        DEFAULT_DEX_ROOT.is_dir() and DEX_RETARGETING_AVAILABLE,
        "optional dex-retargeting package/assets absent",
    )
    def test_inspire_optimizer_builds_and_returns_safe_six_axis_output(self) -> None:
        calibration = PipelineCalibration.load(CALIBRATION_PATH)
        retargeter = DexInspireRetargeter(calibration)
        output = retargeter.retarget(detection_for(open_hand_joints()))
        self.assertEqual(output.qpos.shape, (6,))
        self.assertTrue(np.all(np.isfinite(output.qpos)))
        self.assertTrue(np.all(output.qpos >= 0))
        self.assertEqual(len(output.hardware_targets), 6)
        self.assertEqual(output.hardware_targets[5], -1)

    @unittest.skipUnless(
        DEX_HUMAN_DATA.is_file() and DEX_RETARGETING_AVAILABLE,
        "optional dex-retargeting package/golden data absent",
    )
    def test_official_human_frame_matches_inspire_golden_output(self) -> None:
        calibration = PipelineCalibration.load(CALIBRATION_PATH)
        retargeter = DexInspireRetargeter(calibration)
        with DEX_HUMAN_DATA.open("rb") as handle:
            joints = np.asarray(pickle.load(handle)[0], dtype=np.float32)
        output = retargeter.retarget(detection_for(joints))
        np.testing.assert_allclose(
            output.qpos,
            [0.86951125, 0.71925443, 0.64537317, 0.80320340, 0.0, 0.0],
            atol=2e-5,
        )
        self.assertEqual(output.hardware_targets, (427, 509, 549, 463, 900, -1))
        np.testing.assert_array_equal(
            retargeter._output_indices, [4, 6, 2, 0, 9, 8]
        )


class WiLoRSelectionTests(unittest.TestCase):
    @staticmethod
    def _prediction(is_right: bool, x2: float):
        return {
            "is_right": 1.0 if is_right else 0.0,
            "hand_bbox": [0.0, 0.0, x2, 10.0],
            "wilor_preds": {},
        }

    def test_strict_mode_rejects_right_hand_when_another_hand_is_present(self):
        backend = object.__new__(WiLoRBackend)
        backend.handedness = "right"
        backend.strict_single_hand = True
        right = self._prediction(True, 10.0)
        left = self._prediction(False, 20.0)
        self.assertIsNone(backend._select_prediction([right, left]))
        backend.strict_single_hand = False
        self.assertIs(backend._select_prediction([right, left]), right)


class FakeHand:
    def __init__(self) -> None:
        self.angle_targets = (-1,) * 6
        self.angles = (900,) * 6
        self.positions = (500,) * 6
        self.speeds = (1000,) * 6
        self.force_limits = (500,) * 6
        self.currents = (0,) * 6
        self.write_log = []
        self.thread_ids = set()
        self.closed = False

    def _ensure_open(self):
        if self.closed:
            raise RuntimeError("fake serial is closed")

    def snapshot(self):
        self._ensure_open()
        self.thread_ids.add(threading.get_ident())
        return {
            "angle_targets": self.angle_targets,
            "errors": (0,) * 6,
            "temperatures": (25,) * 6,
            "statuses": (2,) * 6,
            "angles": self.angles,
            "currents": self.currents,
            "speeds": self.speeds,
            "force_limits": self.force_limits,
        }

    def write_six_shorts(self, address, values, retries=0):
        self._ensure_open()
        del retries
        self.thread_ids.add(threading.get_ident())
        values = tuple(values)
        self.write_log.append((address, values))
        if address == REG_ANGLE_SET:
            self.angle_targets = values
            self.angles = tuple(
                old if value == -1 else value
                for old, value in zip(self.angles, values)
            )
        elif address == REG_SPEED_SET:
            self.speeds = values
        elif address == REG_FORCE_SET:
            self.force_limits = values

    def read(self, address, length, retries=0):
        self._ensure_open()
        del retries
        self.thread_ids.add(threading.get_ident())
        self.assert_length(length)
        if address == REG_ERROR:
            return bytes((0,) * 6)
        if address == REG_STATUS:
            return bytes((2,) * 6)
        if address == REG_TEMP:
            return bytes((25,) * 6)
        raise AssertionError(f"unexpected read address: {address}")

    @staticmethod
    def assert_length(length):
        if length != 6:
            raise AssertionError(f"unexpected read length: {length}")

    def read_six_shorts(self, address, retries=0):
        self._ensure_open()
        del retries
        self.thread_ids.add(threading.get_ident())
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
            return self.currents
        raise AssertionError(f"unexpected six-short address: {address}")


class StreamingSafetyTests(unittest.TestCase):
    def test_streaming_settings_are_read_back_before_ready(self) -> None:
        calibration = PipelineCalibration.load(CALIBRATION_PATH)
        fake = FakeHand()
        original_speeds = fake.speeds
        original_forces = fake.force_limits

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
        expected_speeds = (1000, 1000, 1000, calibration.speed, 1000, 1000)
        expected_forces = (500, 500, 500, calibration.force_limit, 500, 500)
        self.assertEqual(stream.verified_streaming_speeds, expected_speeds)
        self.assertEqual(stream.verified_streaming_forces, expected_forces)
        self.assertEqual(fake.speeds, expected_speeds)
        self.assertEqual(fake.force_limits, expected_forces)

        stream.close()
        self.assertTrue(stream.stop_confirmed)
        self.assertEqual(fake.speeds, original_speeds)
        self.assertEqual(fake.force_limits, original_forces)

    def test_streaming_settings_readback_mismatch_fails_closed(self) -> None:
        calibration = PipelineCalibration.load(CALIBRATION_PATH)

        for corrupt_address, label in (
            (REG_SPEED_SET, "SPEED_SET"),
            (REG_FORCE_SET, "FORCE_SET"),
        ):
            with self.subTest(register=label):
                class WrongReadbackHand(FakeHand):
                    def read_six_shorts(self, address, retries=0):
                        values = super().read_six_shorts(address, retries=retries)
                        if address == corrupt_address:
                            wrong = list(values)
                            wrong[3] += 1
                            return tuple(wrong)
                        return values

                fake = WrongReadbackHand()

                @contextlib.contextmanager
                def factory():
                    yield fake

                stream = SafeRH56Stream(
                    calibration,
                    selected_axes=("index",),
                    hand_context_factory=factory,
                )
                stream.start()
                with self.assertRaisesRegex(Exception, f"{label} readback mismatch"):
                    stream.wait_until_ready(1.0)
                stream.close()

                self.assertEqual(stream.state, StreamState.FAULT_LATCHED)
                self.assertTrue(stream.stop_confirmed)
                self.assertFalse(stream.ever_active)
                self.assertEqual(stream.motion_write_count, 0)
                self.assertEqual(stream.final_angle_targets, (-1,) * 6)
                numeric_writes = [
                    values
                    for address, values in fake.write_log
                    if address == REG_ANGLE_SET and values != (-1,) * 6
                ]
                self.assertEqual(numeric_writes, [])

    def test_streaming_current_limit_is_enforced(self) -> None:
        calibration = PipelineCalibration.load(CALIBRATION_PATH)
        fake = FakeHand()
        fake.currents = (0, 0, 0, calibration.stream_max_current_ma + 1, 0, 0)
        stream = SafeRH56Stream(
            calibration,
            selected_axes=("index",),
            hand_context_factory=lambda: None,
        )
        with self.assertRaisesRegex(Exception, "current exceeded"):
            stream._read_safety_feedback(
                fake, (-1, -1, -1, 800, -1, -1), active=True
            )

    def test_selected_axis_total_current_limit_is_enforced(self) -> None:
        base = PipelineCalibration.load(CALIBRATION_PATH)
        calibration = replace(
            base,
            stream_max_current_ma=500,
            stream_max_total_current_ma=500,
        )
        fake = FakeHand()
        fake.currents = (110, 110, 110, 110, 110, 0)
        stream = SafeRH56Stream(
            calibration,
            selected_axes=("pinky", "ring", "middle", "index", "thumb_bend"),
            hand_context_factory=lambda: None,
        )
        with self.assertRaisesRegex(Exception, "total current exceeded 500 mA"):
            stream._read_safety_feedback(
                fake, (800, 800, 800, 800, 850, -1), active=True
            )

    def test_monitor_only_current_policy_records_breach_without_fault(self) -> None:
        base = PipelineCalibration.load(CALIBRATION_PATH)
        calibration = replace(
            base,
            active_current_policy="monitor_only",
            stream_max_current_ma=400,
            stream_max_total_current_ma=600,
        )
        fake = FakeHand()
        fake.currents = (166, 220, 99, 111, 100, 23)
        stream = SafeRH56Stream(
            calibration,
            selected_axes=HARDWARE_JOINTS[:-1],
            hand_context_factory=lambda: None,
        )
        stream._read_safety_feedback(
            fake, (800, 800, 800, 800, 850, 950), active=True
        )
        self.assertIsNone(stream.error)
        self.assertTrue(stream.active_current_threshold_exceeded)
        self.assertEqual(stream.active_current_over_limit_sample_count, 1)
        self.assertEqual(stream.active_current_warning_event_count, 1)
        self.assertEqual(
            stream.active_peak_abs_currents,
            (166, 220, 99, 111, 100, 23),
        )
        self.assertEqual(stream.active_max_selected_total_current_ma, 696)

    def test_monitor_only_policy_keeps_pre_active_idle_current_hard_gate(self) -> None:
        base = PipelineCalibration.load(CALIBRATION_PATH)
        calibration = replace(
            base,
            active_current_policy="monitor_only",
            preflight_max_idle_current_ma=100,
            stream_max_current_ma=400,
            stream_max_total_current_ma=600,
        )
        fake = FakeHand()
        fake.currents = (0, 0, 0, 101, 0, 0)
        stream = SafeRH56Stream(
            calibration,
            selected_axes=("index",),
            hand_context_factory=lambda: None,
        )
        with self.assertRaisesRegex(Exception, "idle limit 100 mA"):
            stream._read_safety_feedback(
                fake, (-1, -1, -1, -1, -1, -1), active=False
            )

    def test_unselected_axis_is_excluded_from_total_current(self) -> None:
        base = PipelineCalibration.load(CALIBRATION_PATH)
        calibration = replace(
            base,
            stream_max_current_ma=1000,
            stream_max_total_current_ma=1000,
        )
        fake = FakeHand()
        fake.currents = (100, 100, 100, 100, 100, 900)
        stream = SafeRH56Stream(
            calibration,
            selected_axes=("pinky", "ring", "middle", "index", "thumb_bend"),
            hand_context_factory=lambda: None,
        )
        stream._read_safety_feedback(
            fake, (800, 800, 800, 800, 850, -1), active=True
        )

    def test_total_current_fault_latches_and_disables_all_axes(self) -> None:
        base = PipelineCalibration.load(CALIBRATION_PATH)
        calibration = replace(
            base,
            tracking_timeout_seconds=0.5,
            valid_frames_to_arm=1,
            control_hz=50.0,
            feedback_hz=20.0,
            stream_max_current_ma=500,
            stream_max_total_current_ma=500,
        )

        class CurrentSpikeAfterMotionHand(FakeHand):
            def write_six_shorts(self, address, values, retries=0):
                super().write_six_shorts(address, values, retries=retries)
                if address == REG_ANGLE_SET and tuple(values) != (-1,) * 6:
                    self.currents = (110, 110, 110, 110, 110, 0)
                elif address == REG_ANGLE_SET:
                    self.currents = (0,) * 6

        fake = CurrentSpikeAfterMotionHand()

        @contextlib.contextmanager
        def factory():
            yield fake

        stream = SafeRH56Stream(
            calibration,
            selected_axes=("pinky", "ring", "middle", "index", "thumb_bend"),
            hand_context_factory=factory,
        )
        stream.start()
        stream.wait_until_ready(1.0)
        self.assertTrue(
            stream.submit(
                TargetFrame(
                    targets=(800, 800, 800, 800, 850, -1),
                    captured_at_monotonic=time.monotonic(),
                    frame_number=1,
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
        self.assertIn("total current exceeded 500 mA", str(stream.error))
        self.assertTrue(stream.stop_confirmed)
        self.assertEqual(stream.final_angle_targets, (-1,) * 6)
        self.assertEqual(fake.write_log[-1], (REG_ANGLE_SET, (-1,) * 6))

    def test_watchdog_latches_and_final_write_is_six_axis_minus_one(self) -> None:
        base = PipelineCalibration.load(CALIBRATION_PATH)
        calibration = replace(
            base,
            tracking_timeout_seconds=0.12,
            valid_frames_to_arm=1,
            control_hz=50.0,
            feedback_hz=20.0,
        )
        fake = FakeHand()

        @contextlib.contextmanager
        def factory():
            try:
                yield fake
            finally:
                fake.closed = True

        stream = SafeRH56Stream(
            calibration,
            selected_axes=("index",),
            hand_context_factory=factory,
        )
        stream.start()
        stream.wait_until_ready(1.0)
        accepted = stream.submit(
            TargetFrame(
                targets=(-1, -1, -1, 800, -1, -1),
                captured_at_monotonic=time.monotonic(),
                frame_number=1,
            )
        )
        self.assertTrue(accepted)
        deadline = time.monotonic() + 2.0
        while stream.state not in (
            StreamState.FAULT_LATCHED,
            StreamState.STOP_UNCONFIRMED,
        ) and time.monotonic() < deadline:
            time.sleep(0.01)
        stream.close()
        self.assertEqual(stream.state, StreamState.FAULT_LATCHED)
        self.assertTrue(stream.stop_confirmed)
        self.assertEqual(stream.final_angle_targets, (-1,) * 6)
        self.assertTrue(fake.closed)
        self.assertEqual(fake.write_log[-1], (REG_ANGLE_SET, (-1,) * 6))
        self.assertTrue(any(address == REG_SPEED_SET for address, _ in fake.write_log))
        self.assertTrue(any(address == REG_FORCE_SET for address, _ in fake.write_log))
        for address, values in fake.write_log:
            if address == REG_ANGLE_SET and values != (-1,) * 6:
                self.assertEqual(values[:3], (-1, -1, -1))
                self.assertEqual(values[4:], (-1, -1))
        self.assertEqual(len(fake.thread_ids), 1)

    def test_context_enter_failure_never_confirms_stop(self) -> None:
        calibration = PipelineCalibration.load(CALIBRATION_PATH)

        @contextlib.contextmanager
        def broken_factory():
            raise OSError("serial unavailable")
            yield  # pragma: no cover

        stream = SafeRH56Stream(
            calibration,
            selected_axes=("index",),
            hand_context_factory=broken_factory,
        )
        stream.start()
        with self.assertRaisesRegex(Exception, "preflight failed"):
            stream.wait_until_ready(1.0)
        stream.close()
        self.assertFalse(stream.stop_confirmed)
        self.assertEqual(stream.state, StreamState.STOP_UNCONFIRMED)

    def test_sparse_valid_frames_do_not_arm(self) -> None:
        base = PipelineCalibration.load(CALIBRATION_PATH)
        calibration = replace(
            base,
            tracking_timeout_seconds=0.50,
            arming_max_frame_gap_seconds=0.10,
            valid_frames_to_arm=2,
            control_hz=50.0,
            feedback_hz=20.0,
            preflight_stability_seconds=0.05,
        )
        fake = FakeHand()

        @contextlib.contextmanager
        def factory():
            try:
                yield fake
            finally:
                fake.closed = True

        stream = SafeRH56Stream(
            calibration,
            selected_axes=("index",),
            hand_context_factory=factory,
        )
        stream.start()
        stream.wait_until_ready(1.0)
        stream.submit(
            TargetFrame(
                targets=(-1, -1, -1, 850, -1, -1),
                captured_at_monotonic=time.monotonic(),
                frame_number=1,
            )
        )
        time.sleep(0.14)
        stream.submit(
            TargetFrame(
                targets=(-1, -1, -1, 840, -1, -1),
                captured_at_monotonic=time.monotonic(),
                frame_number=2,
            )
        )
        time.sleep(0.04)
        self.assertEqual(stream.state, StreamState.WAITING_FOR_TRACKING)
        stream.close()
        self.assertTrue(stream.stop_confirmed)

    def test_stop_during_arming_feedback_cannot_send_motion(self) -> None:
        base = PipelineCalibration.load(CALIBRATION_PATH)
        calibration = replace(
            base,
            valid_frames_to_arm=1,
            control_hz=50.0,
            feedback_hz=20.0,
            preflight_stability_seconds=0.05,
        )
        fake = FakeHand()

        @contextlib.contextmanager
        def factory():
            try:
                yield fake
            finally:
                fake.closed = True

        stream = SafeRH56Stream(
            calibration,
            selected_axes=("index",),
            hand_context_factory=factory,
        )
        entered_feedback = threading.Event()
        release_feedback = threading.Event()
        original_feedback = stream._read_safety_feedback

        def blocking_feedback(hand, sent, active):
            if stream._target_generation > 0 and not entered_feedback.is_set():
                entered_feedback.set()
                if not release_feedback.wait(1.0):
                    raise TimeoutError("test did not release arming feedback")
            return original_feedback(hand, sent, active)

        stream._read_safety_feedback = blocking_feedback
        stream.start()
        stream.wait_until_ready(1.0)
        self.assertTrue(
            stream.submit(
                TargetFrame(
                    targets=(-1, -1, -1, 800, -1, -1),
                    captured_at_monotonic=time.monotonic(),
                    frame_number=1,
                )
            )
        )
        self.assertTrue(entered_feedback.wait(1.0))
        stream.request_stop("test stop during arming feedback")
        self.assertFalse(
            stream.submit(
                TargetFrame(
                    targets=(-1, -1, -1, 810, -1, -1),
                    captured_at_monotonic=time.monotonic(),
                    frame_number=2,
                )
            )
        )
        release_feedback.set()
        stream.close()
        numeric_motion_writes = [
            values
            for address, values in fake.write_log
            if address == REG_ANGLE_SET and values != (-1,) * 6
        ]
        self.assertEqual(numeric_motion_writes, [])
        self.assertFalse(stream.ever_active)
        self.assertEqual(stream.motion_write_count, 0)

    def test_preflight_rejects_selected_axis_outside_commissioning_soft_range(self):
        commissioning = PipelineCalibration.load(
            ROOT
            / "examples/inspire_mano_pipeline/"
            "inspire_rh56bfx_right_commissioning.json"
        )
        fake = FakeHand()
        fake.angles = (900, 900, 900, 0, 900, 900)

        @contextlib.contextmanager
        def factory():
            try:
                yield fake
            finally:
                fake.closed = True

        stream = SafeRH56Stream(
            commissioning,
            selected_axes=("index",),
            hand_context_factory=factory,
        )
        stream.start()
        with self.assertRaisesRegex(Exception, "soft range"):
            stream.wait_until_ready(1.0)
        stream.close()
        self.assertTrue(stream.stop_confirmed)
        self.assertFalse(stream.ever_active)
        self.assertTrue(
            all(
                values == (-1,) * 6
                for address, values in fake.write_log
                if address == REG_ANGLE_SET
            )
        )

    def test_failed_disable_does_not_restore_original_high_force(self) -> None:
        base = PipelineCalibration.load(CALIBRATION_PATH)
        calibration = replace(
            base,
            tracking_timeout_seconds=0.50,
            valid_frames_to_arm=1,
            control_hz=50.0,
            feedback_hz=20.0,
            preflight_stability_seconds=0.05,
        )

        class FailDisableHand(FakeHand):
            fail_disable = False

            def write_six_shorts(self, address, values, retries=0):
                values = tuple(values)
                if (
                    self.fail_disable
                    and address == REG_ANGLE_SET
                    and values == (-1,) * 6
                ):
                    raise RuntimeError("injected final -1 write failure")
                return super().write_six_shorts(address, values, retries=retries)

        fake = FailDisableHand()

        @contextlib.contextmanager
        def factory():
            try:
                yield fake
            finally:
                fake.closed = True

        stream = SafeRH56Stream(
            calibration,
            selected_axes=("index",),
            hand_context_factory=factory,
        )
        stream.start()
        stream.wait_until_ready(1.0)
        self.assertTrue(
            stream.submit(
                TargetFrame(
                    targets=(-1, -1, -1, 800, -1, -1),
                    captured_at_monotonic=time.monotonic(),
                    frame_number=1,
                )
            )
        )
        deadline = time.monotonic() + 1.0
        while stream.motion_write_count == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertGreater(stream.motion_write_count, 0)
        fake.fail_disable = True
        stream.close()
        self.assertEqual(stream.state, StreamState.STOP_UNCONFIRMED)
        self.assertFalse(stream.stop_confirmed)
        self.assertIn("injected final -1 write failure", stream.stop_reason)
        # The selected index must retain the conservative streaming force;
        # restoring the original 500 while a numeric hold remains is unsafe.
        self.assertEqual(fake.force_limits[3], calibration.force_limit)


class HardwareCliGateTests(unittest.TestCase):
    def _parse_and_validate(self, arguments):
        parser = build_parser()
        args = parser.parse_args(arguments)
        validate_args(args, parser)
        return args

    def test_hardware_requires_explicit_calibration_retargeter_axes_and_port(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self._parse_and_validate(
                ["--enable-hardware", "--confirm-hardware-motion"]
            )

    def test_commissioning_index_only_command_passes_gate(self):
        commissioning = ROOT / (
            "examples/inspire_mano_pipeline/"
            "inspire_rh56bfx_right_commissioning.json"
        )
        args = self._parse_and_validate(
            [
                "--enable-hardware",
                "--confirm-hardware-motion",
                "--port",
                "/dev/fake",
                "--camera-serial",
                "fake-camera",
                "--device",
                "cuda:0",
                "--calibration",
                str(commissioning),
                "--retargeter",
                "geometric",
                "--axes",
                "index",
                "--duration",
                "20",
            ]
        )
        self.assertEqual(args.axes, ("index",))

    def test_wide_range_requires_exact_extra_token(self):
        common = [
            "--enable-hardware",
            "--confirm-hardware-motion",
            "--port",
            "/dev/fake",
            "--camera-serial",
            "fake-camera",
            "--device",
            "cuda:0",
            "--calibration",
            str(CALIBRATION_PATH),
            "--retargeter",
            "geometric",
            "--axes",
            "index",
        ]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self._parse_and_validate(common)
        args = self._parse_and_validate(
            common + ["--confirm-wide-range", "RH56_WIDE_RANGE"]
        )
        self.assertEqual(args.confirm_wide_range, "RH56_WIDE_RANGE")

    def test_commissioning_requires_recording_and_bounded_duration(self):
        commissioning = ROOT / (
            "examples/inspire_mano_pipeline/"
            "inspire_rh56bfx_right_commissioning.json"
        )
        common = [
            "--enable-hardware",
            "--confirm-hardware-motion",
            "--port",
            "/dev/fake",
            "--camera-serial",
            "fake-camera",
            "--device",
            "cuda:0",
            "--calibration",
            str(commissioning),
            "--retargeter",
            "geometric",
            "--axes",
            "index",
        ]
        for extra in ([], ["--duration", "61"], ["--duration", "20", "--no-record"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self._parse_and_validate(common + extra)


class RecordingTests(unittest.TestCase):
    def test_tracking_miss_is_recorded_as_a_frame(self):
        frame = CameraFrame(
            color_bgr=np.zeros((4, 4, 3), dtype=np.uint8),
            depth_m=None,
            captured_at_monotonic=12.5,
            frame_number=42,
        )
        handle = io.StringIO()
        write_record(
            handle,
            frame,
            detection=None,
            output=None,
            inference_seconds=0.02,
            tracking_age_seconds=0.8,
        )
        row = json.loads(handle.getvalue())
        self.assertEqual(row["frame_number"], 42)
        self.assertFalse(row["right_hand_detected"])
        self.assertIsNone(row["qpos"])
        self.assertAlmostEqual(row["tracking_age_seconds"], 0.8)


if __name__ == "__main__":
    unittest.main()
