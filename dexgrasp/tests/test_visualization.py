from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anydex_pipeline.snapshot import (  # noqa: E402
    GraspCandidates,
    VisualizationSnapshot,
)
from anydex_pipeline.visualization import (  # noqa: E402
    VisualizationStyle,
    build_open3d_geometries,
    rotation_from_z,
    scores_to_rgb,
    show_snapshot,
)


def _pose(translation, yaw_degrees=0.0):
    yaw = np.deg2rad(yaw_degrees)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    transform[:3, 3] = translation
    return transform


def _snapshot(*, empty=False, with_hand=False):
    if empty:
        poses = np.empty((0, 4, 4), dtype=np.float64)
        scores = np.empty((0,), dtype=np.float32)
        collision_free = np.empty((0,), dtype=np.bool_)
        selected = -1
    else:
        poses = np.stack(
            [
                _pose([0.50, 0.10, 0.20], 0.0),
                _pose([0.60, 0.12, 0.22], 90.0),
                _pose([0.55, 0.08, 0.18], 180.0),
            ]
        )
        scores = np.asarray([0.20, 0.90, 0.75], dtype=np.float32)
        collision_free = np.asarray([True, True, False])
        selected = 1
    hand_poses = None
    if with_hand:
        hand_poses = poses.copy()
        hand_poses[:, 2, 3] += 0.03
    grasps = GraspCandidates(
        canonical_poses=poses,
        scores=scores,
        collision_free=collision_free,
        collision_checked=collision_free.copy(),
        selected_index=selected,
        approach_axis_local=np.asarray([1.0, 0.0, 0.0]),
        hand_poses=hand_poses,
    )
    return VisualizationSnapshot(
        scene_points=np.asarray(
            [[0.40, -0.10, 0.0], [0.70, 0.20, 0.0]], dtype=np.float32
        ),
        scene_colors=np.asarray(
            [[0.2, 0.3, 0.4], [0.5, 0.4, 0.3]], dtype=np.float32
        ),
        object_points=np.asarray(
            [[0.50, 0.10, 0.10], [0.52, 0.11, 0.13]], dtype=np.float32
        ),
        object_colors=np.asarray(
            [[0.8, 0.1, 0.1], [0.9, 0.2, 0.1]], dtype=np.float32
        ),
        grasps=grasps,
        reference_frame="robot_base",
        T_reference_camera=np.eye(4),
        frame_id=5,
        timestamp_s=10.0,
    )


def test_import_is_lazy_and_does_not_load_open3d():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import sys; import anydex_pipeline.visualization; "
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


@pytest.mark.parametrize(
    "direction",
    [
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([0.0, 0.0, -1.0]),
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.2, -0.3, 0.7]),
    ],
)
def test_rotation_from_z_is_a_proper_alignment(direction):
    expected = direction / np.linalg.norm(direction)
    rotation = rotation_from_z(direction)

    np.testing.assert_allclose(rotation @ [0.0, 0.0, 1.0], expected, atol=1e-10)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-10)


def test_score_colors_are_deterministic_clipped_and_bounded():
    scores = np.asarray([-1.0, 0.0, 0.5, 1.0, 3.0])

    first = scores_to_rgb(scores)
    second = scores_to_rgb(scores)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (5, 3)
    assert np.all((first >= 0.0) & (first <= 1.0))
    np.testing.assert_allclose(first[0], first[1])
    np.testing.assert_allclose(first[-1], first[-2])
    assert not np.allclose(first[1], first[3])


def test_geometry_builder_is_headless_and_uses_canonical_plus_x_approach():
    pytest.importorskip("open3d")
    snapshot = _snapshot()
    style = VisualizationStyle(approach_length_m=0.05, max_candidates=10)

    bundle = build_open3d_geometries(snapshot, style)

    assert len(bundle.scene_point_cloud.points) == 2
    assert len(bundle.object_point_cloud.points) == 2
    np.testing.assert_array_equal(bundle.candidate_indices, [1, 0])
    assert len(bundle.grasp_origins) == 2
    assert len(bundle.approach_arrows) == 2
    assert bundle.selected_canonical_frame is not None
    assert bundle.selected_hand_frame is None
    assert bundle.selected_hand_link_meshes == []

    # Candidate 1 has a +90 degree yaw. Canonical +X therefore maps to +Y.
    expected_origin = np.asarray([0.60, 0.12, 0.22])
    expected_start = expected_origin - np.asarray([0.0, 0.05, 0.0])
    np.testing.assert_allclose(bundle.approach_segments[0, 0], expected_start, atol=1e-12)
    np.testing.assert_allclose(bundle.approach_segments[0, 1], expected_origin, atol=1e-12)
    np.testing.assert_allclose(bundle.candidate_colors[0], style.selected_color)

    # Meshes are valid CPU geometry; no Visualizer/create_window is used.
    assert len(bundle.approach_arrows[0].triangles) > 0
    assert np.all(np.isfinite(bundle.approach_arrows[0].get_min_bound()))
    assert len(bundle.geometry_list()) == 2 + 1 + 2 + 2 + 1


def test_selected_rejected_candidate_remains_visible_and_hand_frame_is_distinct():
    pytest.importorskip("open3d")
    snapshot = _snapshot(with_hand=True)
    grasps = snapshot.grasps
    snapshot = VisualizationSnapshot(
        **{
            **snapshot.__dict__,
            "grasps": GraspCandidates(
                canonical_poses=grasps.canonical_poses,
                scores=grasps.scores,
                collision_free=grasps.collision_free,
                collision_checked=grasps.collision_checked,
                selected_index=2,
                approach_axis_local=grasps.approach_axis_local,
                hand_poses=grasps.hand_poses,
            ),
        }
    )
    style = VisualizationStyle(
        max_candidates=1,
        show_collision_rejected=False,
        show_selected_hand_frame=True,
    )

    bundle = build_open3d_geometries(snapshot, style)

    np.testing.assert_array_equal(bundle.candidate_indices, [2])
    assert bundle.selected_canonical_frame is not None
    assert bundle.selected_hand_frame is not None
    np.testing.assert_allclose(bundle.candidate_colors[0], style.selected_color)


def test_empty_candidates_create_clouds_without_pose_geometry():
    pytest.importorskip("open3d")

    bundle = build_open3d_geometries(_snapshot(empty=True))

    assert bundle.candidate_indices.shape == (0,)
    assert bundle.candidate_colors.shape == (0, 3)
    assert bundle.approach_segments.shape == (0, 2, 3)
    assert bundle.grasp_origins == []
    assert bundle.approach_arrows == []
    assert bundle.selected_canonical_frame is None
    assert bundle.selected_hand_link_meshes == []


def test_builder_accepts_prebuilt_selected_hand_link_meshes():
    o3d = pytest.importorskip("open3d")
    palm = o3d.geometry.TriangleMesh.create_box(0.04, 0.03, 0.01)
    finger = o3d.geometry.TriangleMesh.create_cylinder(0.004, 0.03)

    bundle = build_open3d_geometries(
        _snapshot(), selected_hand_link_meshes=[palm, finger]
    )

    assert len(bundle.selected_hand_link_meshes) == 2
    assert bundle.selected_hand_link_meshes[0] is palm
    assert bundle.selected_hand_link_meshes[1] is finger
    assert bundle.geometry_list()[-2] is palm
    assert bundle.geometry_list()[-1] is finger


def test_builder_calls_external_selected_hand_link_mesh_builder_once():
    o3d = pytest.importorskip("open3d")
    snapshot = _snapshot(with_hand=True)
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    calls = []

    def builder(received_snapshot, selected_index):
        calls.append((received_snapshot, selected_index))
        return mesh

    bundle = build_open3d_geometries(
        snapshot, selected_hand_link_mesh_builder=builder
    )

    assert len(calls) == 1
    assert calls[0][0] is snapshot
    assert calls[0][1] == snapshot.grasps.selected_index
    assert len(bundle.selected_hand_link_meshes) == 1
    assert bundle.selected_hand_link_meshes[0] is mesh


def test_hand_mesh_builder_is_not_called_without_a_selected_grasp():
    pytest.importorskip("open3d")

    def unexpected_builder(snapshot, selected_index):
        raise AssertionError("builder must not be called without a selected grasp")

    bundle = build_open3d_geometries(
        _snapshot(empty=True),
        selected_hand_link_mesh_builder=unexpected_builder,
    )

    assert bundle.selected_hand_link_meshes == []


def test_hand_mesh_sources_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_open3d_geometries(
            _snapshot(),
            selected_hand_link_meshes=[],
            selected_hand_link_mesh_builder=lambda snapshot, index: [],
        )


def test_hand_mesh_builder_rejects_non_open3d_results():
    pytest.importorskip("open3d")

    with pytest.raises(TypeError, match="not an open3d.geometry.Geometry"):
        build_open3d_geometries(
            _snapshot(),
            selected_hand_link_mesh_builder=lambda snapshot, index: [object()],
        )


def test_show_snapshot_dispatches_to_the_explicit_window_entrypoint(monkeypatch):
    o3d = pytest.importorskip("open3d")
    called = {}
    selected_link = o3d.geometry.TriangleMesh.create_box(0.01, 0.01, 0.01)

    def fake_draw(geometries, **kwargs):
        called["geometries"] = geometries
        called.update(kwargs)

    monkeypatch.setattr(o3d.visualization, "draw_geometries", fake_draw)
    bundle = show_snapshot(
        _snapshot(),
        window_name="unit-test-window",
        selected_hand_link_meshes=[selected_link],
    )

    assert called["window_name"] == "unit-test-window"
    assert called["geometries"] == bundle.geometry_list()
    assert len(bundle.selected_hand_link_meshes) == 1
    assert bundle.selected_hand_link_meshes[0] is selected_link
