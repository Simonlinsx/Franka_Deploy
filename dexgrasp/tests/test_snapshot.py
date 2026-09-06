from __future__ import annotations

import sys
from dataclasses import replace
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
    load_snapshot_npz,
    save_snapshot_npz,
    validate_snapshot,
)


def _pose(translation, yaw_degrees=0.0):
    yaw = np.deg2rad(yaw_degrees)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def _snapshot() -> VisualizationSnapshot:
    canonical = np.stack(
        [_pose([0.52, 0.03, 0.10]), _pose([0.54, 0.04, 0.11], 90.0)]
    )
    hand = canonical.copy()
    hand[:, :3, 3] += np.asarray([0.0, 0.0, 0.025])
    grasps = GraspCandidates(
        canonical_poses=canonical,
        scores=np.asarray([0.70, 0.93], dtype=np.float32),
        type_ids=np.asarray([2, 6], dtype=np.int32),
        collision_free=np.asarray([True, True]),
        collision_checked=np.asarray([True, True]),
        selected_index=1,
        approach_axis_local=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        hand_poses=hand,
        hand_angles=np.asarray(
            [[1000, 820, 760, 710, 650, 500], [920, 800, 720, 680, 620, 470]],
            dtype=np.float32,
        ),
        widths_m=np.asarray([0.06, 0.08], dtype=np.float32),
        depths_m=np.asarray([0.02, 0.03], dtype=np.float32),
        source_indices=np.asarray([11, 4], dtype=np.int64),
    )
    return VisualizationSnapshot(
        scene_points=np.asarray(
            [[0.45, -0.10, 0.0], [0.65, 0.12, 0.0], [0.50, 0.05, 0.20]],
            dtype=np.float32,
        ),
        scene_colors=np.asarray(
            [[0.1, 0.1, 0.1], [0.4, 0.3, 0.2], [0.2, 0.5, 0.7]],
            dtype=np.float32,
        ),
        object_points=np.asarray(
            [[0.51, 0.03, 0.08], [0.53, 0.04, 0.12]], dtype=np.float32
        ),
        object_colors=np.asarray(
            [[0.8, 0.2, 0.1], [0.9, 0.3, 0.1]], dtype=np.float32
        ),
        grasps=grasps,
        reference_frame="robot_base",
        T_reference_camera=_pose([1.2, 0.35, 0.64], 28.0),
        frame_id=42,
        timestamp_s=1_721_234_567.25,
        calibration_id="eye-to-hand-test",
        camera_serial="337322072188",
        scene_excludes_object=True,
        inference_points=np.asarray(
            [[0.51, 0.03, 0.08], [0.53, 0.04, 0.12], [0.52, 0.04, 0.10]],
            dtype=np.float32,
        ),
        model_name="AnyDexGrasp",
        checkpoint_sha256="a" * 64,
        decision_checkpoint_sha256s=tuple(f"{index + 1:064x}" for index in range(8)),
        official_source_commit="c" * 40,
    )


def test_npz_v2_roundtrip_is_pickle_free(tmp_path):
    expected = _snapshot()
    output = save_snapshot_npz(tmp_path / "capture.data", expected)

    assert output == tmp_path / "capture.data"
    assert output.is_file()
    assert not (tmp_path / "capture.data.npz").exists()
    with np.load(output, allow_pickle=False) as archive:
        assert archive["schema_version"].item() == 2
        assert archive["reference_frame"].item() == "robot_base"
        assert all(archive[name].dtype.kind != "O" for name in archive.files)

    actual = load_snapshot_npz(output)
    assert actual.reference_frame == expected.reference_frame
    assert actual.frame_id == expected.frame_id
    assert actual.timestamp_s == expected.timestamp_s
    assert actual.calibration_id == expected.calibration_id
    assert actual.camera_serial == expected.camera_serial
    assert actual.model_name == expected.model_name
    assert actual.checkpoint_sha256 == expected.checkpoint_sha256
    assert (
        actual.representation_checkpoint_sha256
        == expected.representation_checkpoint_sha256
    )
    assert (
        actual.decision_checkpoint_sha256s
        == expected.decision_checkpoint_sha256s
    )
    assert actual.official_source_commit == expected.official_source_commit
    np.testing.assert_array_equal(
        actual.grasps.collision_checked, expected.grasps.collision_checked
    )
    np.testing.assert_allclose(actual.scene_points, expected.scene_points)
    np.testing.assert_allclose(actual.scene_colors, expected.scene_colors)
    np.testing.assert_allclose(actual.object_points, expected.object_points)
    np.testing.assert_allclose(
        actual.grasps.canonical_poses, expected.grasps.canonical_poses
    )
    np.testing.assert_allclose(actual.grasps.hand_poses, expected.grasps.hand_poses)
    np.testing.assert_allclose(actual.grasps.hand_angles, expected.grasps.hand_angles)
    np.testing.assert_allclose(actual.grasps.widths_m, expected.grasps.widths_m)
    np.testing.assert_allclose(actual.grasps.depths_m, expected.grasps.depths_m)
    np.testing.assert_array_equal(
        actual.grasps.source_indices, expected.grasps.source_indices
    )
    np.testing.assert_allclose(actual.inference_points, expected.inference_points)


def test_empty_candidate_snapshot_roundtrips(tmp_path):
    empty_grasps = GraspCandidates(
        canonical_poses=np.empty((0, 4, 4), dtype=np.float64),
        scores=np.empty((0,), dtype=np.float32),
    )
    snapshot = VisualizationSnapshot(
        scene_points=np.empty((0, 3), dtype=np.float32),
        scene_colors=np.empty((0, 3), dtype=np.float32),
        object_points=np.empty((0, 3), dtype=np.float32),
        object_colors=np.empty((0, 3), dtype=np.float32),
        grasps=empty_grasps,
        reference_frame="robot_base",
        T_reference_camera=np.eye(4),
        frame_id=0,
        timestamp_s=0.0,
    )

    loaded = load_snapshot_npz(save_snapshot_npz(tmp_path / "empty.npz", snapshot))

    assert loaded.grasps.canonical_poses.shape == (0, 4, 4)
    assert loaded.grasps.type_ids.shape == (0,)
    assert loaded.grasps.collision_free.shape == (0,)
    assert loaded.grasps.selected_index == -1
    assert loaded.grasps.hand_poses is None
    assert loaded.inference_points is None


def test_validation_rejects_frame_and_color_errors():
    snapshot = _snapshot()

    bad_color = np.asarray(snapshot.scene_colors).copy()
    bad_color[0, 0] = 1.1
    with pytest.raises(ValueError, match="scene_colors must be RGB"):
        validate_snapshot(replace(snapshot, scene_colors=bad_color))

    reflected = np.asarray(snapshot.T_reference_camera).copy()
    reflected[:3, :3] = np.diag([-1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match=r"determinant \+1"):
        validate_snapshot(replace(snapshot, T_reference_camera=reflected))

    non_finite = np.asarray(snapshot.object_points).copy()
    non_finite[0, 2] = np.nan
    with pytest.raises(ValueError, match="object_points contains non-finite"):
        validate_snapshot(replace(snapshot, object_points=non_finite))


def test_validation_rejects_inconsistent_grasp_arrays_and_pose():
    snapshot = _snapshot()
    grasps = snapshot.grasps

    with pytest.raises(ValueError, match="scores must have shape"):
        validate_snapshot(
            replace(snapshot, grasps=replace(grasps, scores=np.asarray([0.9])))
        )

    with pytest.raises(ValueError, match="selected_index=2"):
        validate_snapshot(replace(snapshot, grasps=replace(grasps, selected_index=2)))

    with pytest.raises(ValueError, match="unit norm"):
        validate_snapshot(
            replace(
                snapshot,
                grasps=replace(
                    grasps, approach_axis_local=np.asarray([2.0, 0.0, 0.0])
                ),
            )
        )

    bad_pose = np.asarray(grasps.canonical_poses).copy()
    bad_pose[0, 3] = [0.0, 0.0, 0.1, 1.0]
    with pytest.raises(ValueError, match="homogeneous bottom row"):
        validate_snapshot(
            replace(snapshot, grasps=replace(grasps, canonical_poses=bad_pose))
        )

    with pytest.raises(ValueError, match="hand_poses must have shape"):
        validate_snapshot(
            replace(
                snapshot,
                grasps=replace(grasps, hand_poses=np.eye(4)[None, ...]),
            )
        )


def test_unknown_metric_candidate_values_use_nan_but_infinity_is_rejected(tmp_path):
    snapshot = _snapshot()
    grasps = replace(
        snapshot.grasps,
        widths_m=np.asarray([np.nan, 0.08], dtype=np.float32),
        depths_m=np.asarray([0.02, np.nan], dtype=np.float32),
    )
    snapshot = replace(snapshot, grasps=grasps)

    loaded = load_snapshot_npz(save_snapshot_npz(tmp_path / "unknown.npz", snapshot))

    assert np.isnan(loaded.grasps.widths_m[0])
    assert np.isnan(loaded.grasps.depths_m[1])
    with pytest.raises(ValueError, match="widths_m contains infinity"):
        validate_snapshot(
            replace(
                snapshot,
                grasps=replace(
                    grasps, widths_m=np.asarray([np.inf, 0.08], dtype=np.float32)
                ),
            )
        )


def test_loader_rejects_unknown_v2_member(tmp_path):
    valid_path = save_snapshot_npz(tmp_path / "valid.npz", _snapshot())
    with np.load(valid_path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]).copy() for name in archive.files}
    payload["typo_field"] = np.asarray(1)
    invalid_path = tmp_path / "unknown.npz"
    np.savez_compressed(invalid_path, **payload)

    with pytest.raises(ValueError, match="unknown v2 keys"):
        load_snapshot_npz(invalid_path)


def test_v1_snapshot_loads_but_legacy_collision_true_is_not_trusted(tmp_path):
    v2_path = save_snapshot_npz(tmp_path / "v2.npz", _snapshot())
    with np.load(v2_path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]).copy() for name in archive.files}
    payload["schema_version"] = np.asarray(1, dtype=np.int32)
    for key in (
        "grasp_collision_checked",
        "representation_checkpoint_sha256",
        "decision_checkpoint_sha256s",
        "official_source_commit",
    ):
        payload.pop(key)
    legacy_path = tmp_path / "legacy-v1.npz"
    np.savez_compressed(legacy_path, **payload)

    loaded = load_snapshot_npz(legacy_path)

    np.testing.assert_array_equal(loaded.grasps.collision_checked, [False, False])
    np.testing.assert_array_equal(loaded.grasps.collision_free, [False, False])
    assert loaded.representation_checkpoint_sha256 == "a" * 64
    assert loaded.decision_checkpoint_sha256s == ()
    assert loaded.official_source_commit == ""


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("checkpoint_sha256", "A" * 64, "64 lowercase hex"),
        ("checkpoint_sha256", "a" * 63, "64 lowercase hex"),
        ("decision_checkpoint_sha256s", ("a" * 64,) * 7, "exactly 8"),
        ("official_source_commit", "not-a-commit", "Git object ID"),
    ],
)
def test_model_provenance_is_strict(field, value, message):
    snapshot = _snapshot()
    changes = {field: value}
    if field == "checkpoint_sha256":
        changes["representation_checkpoint_sha256"] = value
    with pytest.raises(ValueError, match=message):
        validate_snapshot(replace(snapshot, **changes))


def test_collision_free_requires_explicit_boolean_check_evidence():
    snapshot = _snapshot()
    unchecked = replace(
        snapshot.grasps,
        collision_checked=np.asarray([False, True], dtype=np.bool_),
    )
    with pytest.raises(ValueError, match="collision_free=True requires"):
        validate_snapshot(replace(snapshot, grasps=unchecked))
