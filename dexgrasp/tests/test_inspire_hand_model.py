from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
UPSTREAM = ROOT / "third_party/AnyDexGrasp"
HAND_ROOT = UPSTREAM / "generate_mesh_and_pointcloud/inspire_urdf"

from anydex_pipeline.inspire_commissioning import attach_diagnostic_inspire_hand
from anydex_pipeline.inspire_hand_model import (
    OFFICIAL_MOUNTING_RING_OFFSET_M,
    OFFICIAL_SOURCE_MESH_OFFSET_M,
    OFFICIAL_WRIST_CENTER_OFFSET_M,
    InspireHandModel,
    rotation_about_axis,
    rotation_from_rpy,
)
from anydex_pipeline.snapshot import GraspCandidates, VisualizationSnapshot


def _model(*, mesh_resolution="simplified"):
    return InspireHandModel.from_anydex_root(
        UPSTREAM, mesh_resolution=mesh_resolution
    )


def _snapshot():
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = (0.5, 0.1, 0.2)
    grasps = GraspCandidates(
        canonical_poses=pose[None],
        scores=np.asarray([0.8], dtype=np.float32),
        selected_index=0,
        widths_m=np.asarray([0.056], dtype=np.float32),
        depths_m=np.asarray([0.025], dtype=np.float32),
        source_indices=np.asarray([4], dtype=np.int64),
    )
    return VisualizationSnapshot(
        scene_points=np.asarray([[0.4, 0.0, 0.0]], dtype=np.float32),
        scene_colors=np.asarray([[0.3, 0.3, 0.3]], dtype=np.float32),
        object_points=np.asarray([[0.5, 0.1, 0.2]], dtype=np.float32),
        object_colors=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        grasps=grasps,
        reference_frame="robot_base",
        T_reference_camera=np.eye(4),
        frame_id=1,
        timestamp_s=1.0,
        model_name="geometric_demo",
    )


def test_import_is_lazy_and_does_not_load_open3d():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import sys; import anydex_pipeline.inspire_hand_model; "
        "assert 'open3d' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_rotation_helpers_follow_urdf_convention():
    rpy = np.asarray([0.3, -0.2, 0.4])
    rotation = rotation_from_rpy(rpy)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    quarter_turn = rotation_about_axis([0, 0, 2], np.pi / 2)
    np.testing.assert_allclose(quarter_turn @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_official_urdf_topology_and_mapping_order_are_resolved():
    model = _model()
    assert model.root_link == "Link111"
    assert [joint.child for joint in model.joints] == [
        "Link1",
        "Link11",
        "Link2",
        "Link22",
        "Link3",
        "Link33",
        "Link4",
        "Link44",
        "Link5",
        "Link51",
        "Link52",
        "Link53",
    ]
    configuration = model.configuration(4, 0.056)
    assert configuration.grasp_type_name == "Large_Diameter"
    assert configuration.width_key_cm == "5.6"
    assert configuration.joint_positions_rad.shape == (12,)
    assert configuration.actuator_registers.shape == (6,)
    assert not np.allclose(
        configuration.joint_positions_rad[:6],
        configuration.actuator_registers,
    )


def test_fk_and_source_mesh_offset_are_applied_in_the_documented_order():
    model = _model()
    configuration = model.configuration(4, 0.056)
    fk = model.forward_kinematics(configuration.joint_positions_rad)
    assert set(fk) == {link.name for link in model.links}
    for transform in fk.values():
        np.testing.assert_allclose(transform[3], [0, 0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(
            transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-10
        )

    T_reference_hand = np.eye(4)
    T_reference_hand[:3, 3] = (0.4, -0.1, 0.3)
    final = model.link_mesh_transforms(
        T_reference_hand, configuration.joint_positions_rad
    )
    expected_root = T_reference_hand.copy()
    expected_root[:3, 3] += OFFICIAL_SOURCE_MESH_OFFSET_M
    np.testing.assert_allclose(final["Link111"], expected_root, atol=1e-12)


def test_official_source_offset_preserves_the_mounting_ring_datum():
    np.testing.assert_allclose(
        OFFICIAL_SOURCE_MESH_OFFSET_M,
        OFFICIAL_WRIST_CENTER_OFFSET_M + OFFICIAL_MOUNTING_RING_OFFSET_M,
        atol=0.0,
    )
    np.testing.assert_allclose(
        OFFICIAL_MOUNTING_RING_OFFSET_M, [0.0078, 0.0, 0.0], atol=0.0
    )


def test_diagnostic_mapping_adds_hand_pose_but_labels_model_as_non_neural():
    snapshot = attach_diagnostic_inspire_hand(
        _snapshot(),
        grasp_type_id=4,
        mapping=HAND_ROOT / "width_12Dangle_6Dangle.json",
    )
    assert snapshot.grasps.hand_poses.shape == (1, 4, 4)
    assert snapshot.grasps.hand_angles.shape == (1, 6)
    np.testing.assert_array_equal(snapshot.grasps.type_ids, [4])
    assert "diagnostic fixed Inspire type 4" in snapshot.model_name
    assert "not decision-model output" in snapshot.model_name


def test_selected_snapshot_builds_thirteen_separate_link_meshes():
    pytest.importorskip("open3d")
    snapshot = attach_diagnostic_inspire_hand(
        _snapshot(),
        grasp_type_id=4,
        mapping=HAND_ROOT / "width_12Dangle_6Dangle.json",
    )
    meshes = _model(mesh_resolution="simplified").build_selected_snapshot_links(
        snapshot, 0
    )
    assert len(meshes) == 13
    assert sum(len(mesh.triangles) for mesh in meshes) == 162
    for mesh in meshes:
        assert not mesh.is_empty()
        assert np.all(np.isfinite(mesh.get_min_bound()))
        assert np.all(np.isfinite(mesh.get_max_bound()))
