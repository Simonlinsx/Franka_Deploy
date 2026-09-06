from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from inspire_mano_pipeline.calibration import PipelineCalibration
from inspire_mano_pipeline.model import CameraFrame
from inspire_mano_pipeline.retargeting import GeometricInspireRetargeter
from inspire_mano_pipeline.wilor_backend import (
    WiLoRBackend,
    detector_label_for_physical_hand,
)
from realsense_mano_inspire import (
    build_parser,
    validate_args,
    write_run_metadata,
)


ROOT = Path(__file__).resolve().parents[3]
CALIBRATION_PATH = (
    ROOT / "examples/inspire_mano_pipeline/inspire_rh56bfx_right.json"
)


def prediction(detector_is_right: bool, x2: float, wilor_preds=None) -> dict:
    return {
        "is_right": 1.0 if detector_is_right else 0.0,
        "hand_bbox": [0.0, 0.0, x2, 10.0],
        "wilor_preds": {} if wilor_preds is None else wilor_preds,
    }


def selection_backend(input_mirrored: bool, strict: bool = True) -> WiLoRBackend:
    backend = object.__new__(WiLoRBackend)
    backend.physical_handedness = "right"
    backend.handedness = "right"
    backend.input_mirrored = input_mirrored
    backend.detector_handedness = detector_label_for_physical_hand(
        backend.handedness, input_mirrored
    )
    backend.strict_single_hand = strict
    return backend


def asymmetric_hand_joints() -> np.ndarray:
    """Synthetic right-hand MANO joints with non-planar, nonzero curls."""

    joints = np.zeros((21, 3), dtype=np.float32)
    joints[1:5] = np.asarray(
        (
            (-0.018, 0.018, 0.003),
            (-0.032, 0.030, 0.008),
            (-0.041, 0.037, 0.016),
            (-0.045, 0.038, 0.026),
        ),
        dtype=np.float32,
    )
    for base, x, y, curl in (
        (5, 0.030, 0.040, 0.004),
        (9, 0.010, 0.044, 0.007),
        (13, -0.010, 0.042, 0.010),
        (17, -0.030, 0.037, 0.013),
    ):
        joints[base] = (x, y, 0.0)
        joints[base + 1] = (x, y + 0.025, curl)
        joints[base + 2] = (x + 0.004, y + 0.040, curl * 2.0)
        joints[base + 3] = (x + 0.011, y + 0.046, curl * 3.0)
    return joints


def mano_prediction(
    joints: np.ndarray,
    keypoints_2d: np.ndarray,
    vertices: np.ndarray,
    global_orient: np.ndarray,
    hand_pose: np.ndarray,
) -> dict:
    return {
        "pred_keypoints_3d": joints[None, ...],
        "pred_keypoints_2d": keypoints_2d[None, ...],
        "global_orient": global_orient[None, None, ...],
        "hand_pose": hand_pose[None, ...],
        "betas": np.zeros((1, 10), dtype=np.float32),
        "pred_vertices": vertices[None, ...],
    }


def prediction_backend(
    *, input_mirrored: bool, detector_is_right: bool, pred: dict, bbox: list
) -> WiLoRBackend:
    backend = selection_backend(input_mirrored=input_mirrored)
    backend.hand_confidence = 0.5
    backend._mesh_faces = None
    backend._pipeline = Mock()
    backend._pipeline.predict.return_value = [
        prediction(detector_is_right, bbox[2], pred) | {"hand_bbox": bbox}
    ]
    return backend


class PhysicalHandednessSelectionTests(unittest.TestCase):
    def test_detector_label_mapping_is_explicit(self) -> None:
        self.assertEqual(detector_label_for_physical_hand("right", False), "right")
        self.assertEqual(detector_label_for_physical_hand("right", True), "left")
        self.assertEqual(detector_label_for_physical_hand("left", False), "left")
        self.assertEqual(detector_label_for_physical_hand("left", True), "right")

    def test_unmirrored_physical_right_selects_detector_right(self) -> None:
        backend = selection_backend(input_mirrored=False)
        detector_left = prediction(False, 10.0)
        detector_right = prediction(True, 20.0)
        self.assertIs(backend._select_prediction([detector_right]), detector_right)
        self.assertIsNone(backend._select_prediction([detector_left]))

    def test_mirrored_physical_right_selects_detector_left(self) -> None:
        backend = selection_backend(input_mirrored=True)
        detector_left = prediction(False, 10.0)
        detector_right = prediction(True, 20.0)
        self.assertIs(backend._select_prediction([detector_left]), detector_left)
        self.assertIsNone(backend._select_prediction([detector_right]))

    def test_strict_mode_rejects_an_extra_hand_of_either_label(self) -> None:
        detector_right = prediction(True, 10.0)
        detector_left = prediction(False, 20.0)
        backend = selection_backend(input_mirrored=False, strict=True)
        self.assertIsNone(
            backend._select_prediction([detector_right, detector_left])
        )
        backend.strict_single_hand = False
        self.assertIs(
            backend._select_prediction([detector_right, detector_left]),
            detector_right,
        )

    def test_native_class_one_prediction_uses_physical_right_canonicalization(
        self,
    ) -> None:
        raw = np.zeros((21, 3), dtype=np.float32)
        raw[5] = (0.03, 0.04, 0.0)
        raw[9] = (0.0, 0.05, 0.0)
        raw[17] = (-0.03, 0.04, 0.0)
        pred = {
            "pred_keypoints_3d": raw[None, ...],
            "pred_keypoints_2d": np.zeros((1, 21, 2), dtype=np.float32),
            "global_orient": np.zeros((1, 1, 3), dtype=np.float32),
            "hand_pose": np.zeros((1, 15, 3), dtype=np.float32),
            "betas": np.zeros((1, 10), dtype=np.float32),
            "pred_vertices": np.zeros((1, 778, 3), dtype=np.float32),
        }
        backend = selection_backend(input_mirrored=False)
        backend.hand_confidence = 0.5
        backend._pipeline = Mock()
        backend._pipeline.predict.return_value = [prediction(True, 10.0, pred)]
        frame = CameraFrame(
            color_bgr=np.zeros((16, 16, 3), dtype=np.uint8),
            depth_m=None,
            captured_at_monotonic=12.5,
            frame_number=7,
        )

        with patch(
            "inspire_mano_pipeline.wilor_backend.canonicalize_for_dex",
            return_value=raw.copy(),
        ) as canonicalize:
            detection = backend.predict(frame)

        self.assertIsNotNone(detection)
        self.assertTrue(detection.is_right)
        np.testing.assert_array_equal(detection.keypoints_3d_raw, raw)
        np.testing.assert_array_equal(detection.keypoints_3d_canonical, raw)
        self.assertTrue(canonicalize.call_args.kwargs["is_right"])

    def test_original_and_mirrored_images_keep_physical_retarget_semantics(
        self,
    ) -> None:
        """External image mirroring must not mirror the robot-hand target."""

        width = 640
        joints = asymmetric_hand_joints()
        keypoints_2d = np.column_stack(
            (
                np.linspace(120.0, 260.0, 21, dtype=np.float32),
                np.linspace(80.0, 360.0, 21, dtype=np.float32),
            )
        )
        vertices = np.zeros((778, 3), dtype=np.float32)
        vertices[:, 0] = np.linspace(-0.04, 0.05, 778, dtype=np.float32)
        vertices[:, 1] = np.linspace(0.01, 0.09, 778, dtype=np.float32)
        global_orient = np.asarray((0.17, -0.22, 0.31), dtype=np.float32)
        hand_pose = np.linspace(
            -0.3, 0.4, 45, dtype=np.float32
        ).reshape(15, 3)

        original_pred = mano_prediction(
            joints, keypoints_2d, vertices, global_orient, hand_pose
        )
        # This is exactly WiLoR's class-0 post-processing of the same crop:
        # x-reflected 3D geometry and y/z-reflected rotation vectors.  Its 2D
        # outputs remain coordinates in the externally mirrored input image.
        mirrored_joints = joints.copy()
        mirrored_joints[:, 0] *= -1.0
        mirrored_vertices = vertices.copy()
        mirrored_vertices[:, 0] *= -1.0
        mirrored_global_orient = global_orient.copy()
        mirrored_global_orient[1:3] *= -1.0
        mirrored_hand_pose = hand_pose.copy()
        mirrored_hand_pose[:, 1:3] *= -1.0
        mirrored_keypoints_2d = keypoints_2d.copy()
        mirrored_keypoints_2d[:, 0] = width - 1.0 - keypoints_2d[:, 0]
        mirrored_pred = mano_prediction(
            mirrored_joints,
            mirrored_keypoints_2d,
            mirrored_vertices,
            mirrored_global_orient,
            mirrored_hand_pose,
        )

        original_backend = prediction_backend(
            input_mirrored=False,
            detector_is_right=True,
            pred=original_pred,
            bbox=[100.0, 60.0, 280.0, 400.0],
        )
        mirrored_backend = prediction_backend(
            input_mirrored=True,
            detector_is_right=False,
            pred=mirrored_pred,
            bbox=[360.0, 60.0, 540.0, 400.0],
        )
        frame = CameraFrame(
            color_bgr=np.zeros((480, width, 3), dtype=np.uint8),
            depth_m=None,
            captured_at_monotonic=12.5,
            frame_number=7,
        )

        original = original_backend.predict(frame)
        mirrored = mirrored_backend.predict(frame)

        self.assertIsNotNone(original)
        self.assertIsNotNone(mirrored)
        np.testing.assert_allclose(
            mirrored.keypoints_3d_raw, original.keypoints_3d_raw, atol=1e-7
        )
        np.testing.assert_allclose(
            mirrored.keypoints_3d_canonical,
            original.keypoints_3d_canonical,
            atol=1e-6,
        )
        np.testing.assert_allclose(mirrored.vertices, original.vertices, atol=1e-7)
        np.testing.assert_allclose(
            mirrored.global_orient, original.global_orient, atol=1e-7
        )
        np.testing.assert_allclose(
            mirrored.hand_pose, original.hand_pose, atol=1e-7
        )
        np.testing.assert_allclose(
            mirrored.keypoints_2d[:, 0],
            width - 1.0 - original.keypoints_2d[:, 0],
            atol=1e-7,
        )

        calibration = PipelineCalibration.load(CALIBRATION_PATH)
        original_target = GeometricInspireRetargeter(calibration).retarget(
            original
        )
        mirrored_target = GeometricInspireRetargeter(calibration).retarget(
            mirrored
        )
        np.testing.assert_allclose(
            mirrored_target.qpos, original_target.qpos, atol=1e-7
        )
        self.assertEqual(
            mirrored_target.hardware_targets, original_target.hardware_targets
        )


class HandednessMetadataTests(unittest.TestCase):
    def test_mirror_input_remains_static_image_only(self) -> None:
        parser = build_parser()
        args = parser.parse_args(("--source", "realsense", "--mirror-input"))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                validate_args(args, parser)

    def test_metadata_records_physical_detector_and_mirror_conventions(self) -> None:
        calibration = SimpleNamespace(
            tracking_timeout_seconds=0.75,
            valid_frames_to_arm=5,
            min_palm_depth_m=0.15,
            max_palm_depth_m=1.5,
        )
        base_args = {
            "source": "image",
            "width": 848,
            "height": 480,
            "fps": 30,
            "wilor_root": Path("/tmp/wilor"),
            "dex_root": Path("/tmp/dex"),
            "retargeter": "geometric",
            "allow_multiple_right_hands": False,
            "calibration": Path("/tmp/calibration.json"),
            "enable_hardware": False,
            "port": None,
            "hand_id": 1,
            "axes": ("index",),
        }

        with tempfile.TemporaryDirectory() as directory:
            for mirrored, detector_label in ((False, "right"), (True, "left")):
                args = SimpleNamespace(**base_args, mirror_input=mirrored)
                path = Path(directory) / f"metadata-{mirrored}.json"
                write_run_metadata(
                    path,
                    args,
                    calibration,
                    SimpleNamespace(),
                    session_id="test-session",
                )
                model = json.loads(path.read_text(encoding="utf-8"))["model"]
                self.assertEqual(model["physical_handedness"], "right")
                self.assertEqual(model["input_mirrored"], mirrored)
                self.assertEqual(model["wilor_detector_label"], detector_label)
                self.assertTrue(model["strict_single_detected_hand"])


if __name__ == "__main__":
    unittest.main()
