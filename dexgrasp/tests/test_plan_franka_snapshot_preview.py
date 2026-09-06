from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_app():
    name = "test_plan_franka_snapshot_preview_app"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "apps" / "plan_franka_snapshot_preview.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


APP = _load_app()


def _snapshot_values():
    return {
        "reference_frame": np.asarray("robot_base"),
        "scene_points": np.asarray([[9.0, 9.0, 9.0]], dtype=np.float32),
        "object_points": np.asarray(
            [[0.60, 0.20, 0.08], [0.61, 0.20, 0.08]], dtype=np.float32
        ),
        "calibration_id": np.asarray("eye-to-hand-test"),
        "camera_serial": np.asarray("337322072188"),
        "frame_id": np.asarray(10),
        "timestamp_s": np.asarray(1.0),
    }


def _snapshot_path(tmp_path):
    path = tmp_path / "grasp.npz"
    np.savez_compressed(path, **_snapshot_values())
    return path


def test_live_scene_replaces_only_snapshot_scene_and_uses_capture_q(tmp_path):
    live_path = tmp_path / "live_scene.npz"
    q_capture = APP.DEFAULT_Q + np.asarray([0.001, 0, 0, 0, 0, 0, 0])
    live_points = np.asarray(
        [[0.60, 0.20, 0.08], [0.61, 0.20, 0.08], [0.30, -0.20, 0.40]],
        dtype=np.float32,
    )
    np.savez_compressed(
        live_path,
        scene_points=live_points,
        reference_frame=np.asarray("robot_base"),
        calibration_id=np.asarray("eye-to-hand-test"),
        camera_serial=np.asarray("337322072188"),
        frame_id=np.asarray(22),
        timestamp_s=np.asarray(2.5),
        capture_q_rad=q_capture,
        capture_q_source=np.asarray("cli_asserted"),
    )

    selected = APP._select_collision_scene(
        _snapshot_values(), _snapshot_path(tmp_path), live_path, None
    )

    np.testing.assert_allclose(selected.points[:3], live_points)
    np.testing.assert_allclose(
        selected.points[3:], _snapshot_values()["object_points"]
    )
    assert not np.any(np.all(selected.points == [9.0, 9.0, 9.0], axis=1))
    np.testing.assert_allclose(selected.robot_return_filter_q, q_capture)
    assert selected.metadata["source_kind"] == "live_scene_npz"
    assert selected.metadata["snapshot_scene_ignored"] is True
    assert selected.metadata["saved_grasp_pose_retained"] is True
    assert selected.metadata["saved_object_cloud_retained"] is True
    assert (
        selected.metadata["robot_return_filter_q_source"]
        == "live_scene_npz:capture_q_rad"
    )
    assert selected.metadata["capture_q_source"] == "cli_asserted"
    assert selected.metadata["capture_q_value_source"] == "live_scene_npz:capture_q_rad"
    assert selected.metadata["frame_id"] == 22


def test_snapshot_scene_defaults_robot_filter_to_reviewed_default_q(tmp_path):
    selected = APP._select_collision_scene(
        _snapshot_values(), _snapshot_path(tmp_path), None, None
    )

    np.testing.assert_allclose(selected.points[0], [9.0, 9.0, 9.0])
    np.testing.assert_allclose(selected.robot_return_filter_q, APP.DEFAULT_Q)
    assert selected.metadata["source_kind"] == "snapshot"
    assert selected.metadata["snapshot_scene_ignored"] is False
    assert selected.metadata["robot_return_filter_q_source"] == "planner_default_q"


def test_explicit_capture_q_has_priority_over_live_metadata(tmp_path):
    live_path = tmp_path / "live_scene.npz"
    np.savez_compressed(
        live_path,
        scene_points=_snapshot_values()["object_points"],
        reference_frame=np.asarray("robot_base"),
        calibration_id=np.asarray("eye-to-hand-test"),
        camera_serial=np.asarray("337322072188"),
        capture_q_rad=APP.DEFAULT_Q,
    )
    explicit = APP.DEFAULT_Q.copy()

    selected = APP._select_collision_scene(
        _snapshot_values(), _snapshot_path(tmp_path), live_path, explicit
    )

    np.testing.assert_allclose(selected.robot_return_filter_q, explicit)
    assert (
        selected.metadata["robot_return_filter_q_source"]
        == "cli:--scene-capture-q-rad"
    )


def test_explicit_capture_q_cannot_override_conflicting_live_metadata(tmp_path):
    live_path = tmp_path / "live_scene.npz"
    np.savez_compressed(
        live_path,
        scene_points=_snapshot_values()["object_points"],
        reference_frame=np.asarray("robot_base"),
        calibration_id=np.asarray("eye-to-hand-test"),
        camera_serial=np.asarray("337322072188"),
        capture_q_rad=APP.DEFAULT_Q,
    )
    explicit = APP.DEFAULT_Q + np.asarray([0.002, 0, 0, 0, 0, 0, 0])

    with pytest.raises(ValueError, match="conflicts with live-scene"):
        APP._select_collision_scene(
            _snapshot_values(), _snapshot_path(tmp_path), live_path, explicit
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reference_frame", "camera", "reference_frame"),
        ("calibration_id", "wrong-calibration", "calibration_id"),
        ("camera_serial", "wrong-camera", "camera_serial"),
    ],
)
def test_live_scene_rejects_frame_or_provenance_mismatch(
    tmp_path, field, value, message
):
    live_path = tmp_path / "live_scene.npz"
    payload = {
        "scene_points": np.asarray([[0.3, 0.0, 0.4]], dtype=np.float32),
        "reference_frame": np.asarray("robot_base"),
        "calibration_id": np.asarray("eye-to-hand-test"),
        "camera_serial": np.asarray("337322072188"),
        "capture_q_rad": APP.DEFAULT_Q,
    }
    payload[field] = np.asarray(value)
    np.savez_compressed(live_path, **payload)

    with pytest.raises(ValueError, match=message):
        APP._select_collision_scene(
            _snapshot_values(), _snapshot_path(tmp_path), live_path, None
        )


def test_conflicting_live_joint_aliases_are_rejected():
    with pytest.raises(ValueError, match="joint metadata conflicts"):
        APP._resolve_robot_return_filter_q(
            {
                "capture_q_rad": APP.DEFAULT_Q,
                "franka_q_rad": APP.DEFAULT_Q
                + np.asarray([0.01, 0, 0, 0, 0, 0, 0]),
            },
            None,
        )


def test_live_scene_rejects_stale_object_geometry(tmp_path):
    live_path = tmp_path / "live_scene.npz"
    np.savez_compressed(
        live_path,
        scene_points=np.asarray([[0.3, 0.0, 0.4]], dtype=np.float32),
        reference_frame=np.asarray("robot_base"),
        calibration_id=np.asarray("eye-to-hand-test"),
        camera_serial=np.asarray("337322072188"),
        capture_q_rad=APP.DEFAULT_Q,
    )

    with pytest.raises(ValueError, match="no longer aligns"):
        APP._select_collision_scene(
            _snapshot_values(), _snapshot_path(tmp_path), live_path, None
        )


def test_robot_return_filter_forwards_capture_q_to_geometry_placement():
    class Sphere:
        radius = 0.1

    class Item:
        name = "fr3_link_test_sc_0"
        geometry = Sphere()

    class Placement:
        rotation = np.eye(3)
        translation = np.zeros(3)

    class GeometryModel:
        geometryObjects = [Item()]

    class GeometryData:
        def __init__(self, unused_model):
            self.oMg = [Placement()]

    class Model:
        @staticmethod
        def createData():
            return object()

    class Pin:
        def __init__(self):
            self.forward_q = None
            self.geometry_q = None

        def forwardKinematics(self, unused_model, unused_data, q):
            self.forward_q = np.asarray(q).copy()

        def updateGeometryPlacements(
            self,
            unused_model,
            unused_data,
            unused_geometry_model,
            unused_geometry_data,
            q,
        ):
            self.geometry_q = np.asarray(q).copy()

    q_capture = APP.DEFAULT_Q + np.asarray([0.002, 0, 0, 0, 0, 0, 0])
    Pin.GeometryData = GeometryData
    pin = Pin()
    remaining, excluded = APP._exclude_initial_robot_returns(
        pin,
        Model(),
        GeometryModel(),
        q_capture,
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        0.02,
    )

    np.testing.assert_allclose(pin.forward_q, q_capture)
    np.testing.assert_allclose(pin.geometry_q, q_capture)
    np.testing.assert_allclose(remaining, [[1.0, 0.0, 0.0]])
    assert excluded == 1
