from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from inspire_mano_pipeline.model import CameraFrame, ManoDetection
from inspire_mano_pipeline.visualize import (
    _mesh_triangles_for_image,
    draw_overlay,
)
from inspire_mano_pipeline.wilor_backend import (
    WiLoRBackend,
    _extract_mano_faces,
    project_mano_vertices,
)


def make_detection(
    *,
    keypoints_2d=None,
    vertices_2d=None,
    mesh_faces=None,
) -> ManoDetection:
    joints = np.zeros((21, 3), dtype=np.float32)
    joints[5] = (0.03, 0.04, 0.0)
    joints[9] = (0.0, 0.05, 0.0)
    joints[17] = (-0.03, 0.04, 0.0)
    if keypoints_2d is None:
        keypoints_2d = np.tile(
            np.asarray((20.0, 100.0), dtype=np.float32), (21, 1)
        )
    return ManoDetection(
        is_right=True,
        bbox_xyxy=np.asarray((10, 70, 180, 180), dtype=np.float32),
        keypoints_3d_raw=joints,
        keypoints_3d_canonical=joints,
        keypoints_2d=np.asarray(keypoints_2d, dtype=np.float32),
        global_orient=np.zeros(3, dtype=np.float32),
        hand_pose=np.zeros((15, 3), dtype=np.float32),
        betas=np.zeros(10, dtype=np.float32),
        vertices=np.zeros((778, 3), dtype=np.float32),
        captured_at_monotonic=time.monotonic(),
        frame_number=1,
        vertices_2d=vertices_2d,
        mesh_faces=mesh_faces,
    )


def fake_prediction() -> tuple[dict, np.ndarray]:
    joints = np.zeros((21, 3), dtype=np.float32)
    joints[5] = (0.03, 0.04, 0.0)
    joints[9] = (0.0, 0.05, 0.0)
    joints[17] = (-0.03, 0.04, 0.0)
    vertices = np.zeros((778, 3), dtype=np.float32)
    vertices[:, 2] = 1.0
    vertices[1, 0] = 1.0
    vertices[2, 1] = 1.0
    pred = {
        "pred_keypoints_3d": joints[None, ...],
        "pred_keypoints_2d": np.zeros((1, 21, 2), dtype=np.float32),
        "global_orient": np.zeros((1, 1, 3), dtype=np.float32),
        "hand_pose": np.zeros((1, 15, 3), dtype=np.float32),
        "betas": np.zeros((1, 10), dtype=np.float32),
        "pred_vertices": vertices[None, ...],
        "pred_cam_t_full": np.asarray(((0.0, 0.0, 1.0),), dtype=np.float32),
        "scaled_focal_length": 100.0,
    }
    return pred, joints


def fake_backend(pred: dict, *, with_faces: bool) -> WiLoRBackend:
    backend = object.__new__(WiLoRBackend)
    backend.physical_handedness = "right"
    backend.handedness = "right"
    backend.detector_handedness = "right"
    backend.strict_single_hand = True
    backend.hand_confidence = 0.5
    backend._pipeline = Mock()
    backend._pipeline.predict.return_value = [
        {
            "is_right": 1.0,
            "hand_bbox": [0.0, 0.0, 100.0, 100.0],
            "wilor_preds": pred,
        }
    ]
    if with_faces:
        backend._mesh_faces = np.asarray(((0, 1, 2),), dtype=np.int32)
    return backend


class ManoProjectionTests(unittest.TestCase):
    def test_full_frame_projection_matches_pinhole_camera(self) -> None:
        vertices = np.asarray(
            ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)),
            dtype=np.float32,
        )
        projected = project_mano_vertices(
            vertices,
            translation=((0.0, 0.0, 1.0),),
            focal_length=100.0,
            image_shape=(100, 200),
        )
        np.testing.assert_allclose(
            projected,
            ((100.0, 50.0), (150.0, 50.0), (100.0, 100.0)),
            atol=1e-6,
        )

    def test_projection_rejects_nonfinite_and_behind_camera_vertices(self) -> None:
        bad = np.asarray(((np.nan, 0.0, 1.0),), dtype=np.float32)
        self.assertIsNone(project_mano_vertices(bad, (0, 0, 1), 100, (10, 10)))
        behind = np.asarray(((0.0, 0.0, -2.0),), dtype=np.float32)
        self.assertIsNone(
            project_mano_vertices(behind, (0, 0, 1), 100, (10, 10))
        )

    def test_faces_are_loaded_and_validated_from_wilor_mano(self) -> None:
        faces = np.asarray(((0, 1, 2), (2, 3, 0)), dtype=np.int64)
        pipeline = SimpleNamespace(
            wilor_model=SimpleNamespace(mano=SimpleNamespace(faces=faces))
        )
        loaded = _extract_mano_faces(pipeline, vertex_count=4)
        np.testing.assert_array_equal(loaded, faces)
        self.assertEqual(loaded.dtype, np.int32)
        self.assertFalse(loaded.flags.writeable)
        self.assertIsNone(_extract_mano_faces(pipeline, vertex_count=3))

    def test_backend_attaches_projected_vertices_and_faces(self) -> None:
        pred, joints = fake_prediction()
        backend = fake_backend(pred, with_faces=True)
        frame = CameraFrame(
            color_bgr=np.zeros((100, 200, 3), dtype=np.uint8),
            depth_m=None,
            captured_at_monotonic=1.0,
            frame_number=2,
        )
        with patch(
            "inspire_mano_pipeline.wilor_backend.canonicalize_for_dex",
            return_value=joints,
        ):
            detection = backend.predict(frame)
        self.assertIsNotNone(detection)
        self.assertEqual(detection.vertices_2d.shape, (778, 2))
        np.testing.assert_allclose(detection.vertices_2d[0], (100.0, 50.0))
        np.testing.assert_allclose(detection.vertices_2d[1], (150.0, 50.0))
        np.testing.assert_array_equal(detection.mesh_faces, ((0, 1, 2),))

    def test_object_new_backend_without_faces_or_camera_fields_degrades_cleanly(self) -> None:
        pred, joints = fake_prediction()
        pred.pop("pred_cam_t_full")
        pred.pop("scaled_focal_length")
        backend = fake_backend(pred, with_faces=False)
        frame = CameraFrame(
            color_bgr=np.zeros((20, 20, 3), dtype=np.uint8),
            depth_m=None,
            captured_at_monotonic=1.0,
            frame_number=2,
        )
        with patch(
            "inspire_mano_pipeline.wilor_backend.canonicalize_for_dex",
            return_value=joints,
        ):
            detection = backend.predict(frame)
        self.assertIsNotNone(detection)
        self.assertIsNone(detection.vertices_2d)
        self.assertIsNone(detection.mesh_faces)


class ManoMeshOverlayTests(unittest.TestCase):
    def test_translucent_mesh_and_existing_skeleton_are_both_drawn(self) -> None:
        vertices = np.asarray(
            ((50.0, 80.0), (150.0, 80.0), (100.0, 160.0)),
            dtype=np.float32,
        )
        detection = make_detection(
            vertices_2d=vertices,
            mesh_faces=np.asarray(((0, 1, 2),), dtype=np.int32),
        )
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        overlay = draw_overlay(image, detection, None, inference_fps=20.0)
        self.assertTrue(np.any(overlay[110, 100] != 0))
        self.assertTrue(np.any(overlay[100, 20] != 0))

    def test_invalid_and_fully_offscreen_meshes_are_ignored(self) -> None:
        invalid = make_detection(
            vertices_2d=np.asarray(
                ((np.nan, 0.0), (1.0, 0.0), (0.0, 1.0)),
                dtype=np.float32,
            ),
            mesh_faces=np.asarray(((0, 1, 2),), dtype=np.int32),
        )
        self.assertIsNone(_mesh_triangles_for_image(invalid, (100, 100)))
        offscreen = make_detection(
            vertices_2d=np.asarray(
                ((1000.0, 1000.0), (1100.0, 1000.0), (1000.0, 1100.0)),
                dtype=np.float32,
            ),
            mesh_faces=np.asarray(((0, 1, 2),), dtype=np.int32),
        )
        self.assertIsNone(_mesh_triangles_for_image(offscreen, (100, 100)))
        # Both cases must still retain the legacy skeleton/text overlay.
        for detection in (invalid, offscreen):
            overlay = draw_overlay(
                np.zeros((100, 100, 3), dtype=np.uint8),
                detection,
                None,
                inference_fps=20.0,
            )
            self.assertTrue(np.any(overlay[99, 20] != 0))


if __name__ == "__main__":
    unittest.main()
