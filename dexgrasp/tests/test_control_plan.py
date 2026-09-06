from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from anydex_pipeline.control_plan import (
    AdapterGeometry,
    ExecutionConfig,
    StageName,
    build_control_plan,
    eligible_candidate_indices,
    select_eligible_candidate_index,
    validate_rigid_transform,
)
from anydex_pipeline.snapshot import GraspCandidates, VisualizationSnapshot


def _pose(translation=(0.0, 0.0, 0.0), yaw_degrees=0.0):
    yaw = np.deg2rad(yaw_degrees)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def _snapshot(*, model_name="AnyDexGrasp official", selected=0, calibrated=True):
    canonical = _pose([0.60, 0.20, 0.10], yaw_degrees=90.0)
    hand = _pose([0.61, 0.21, 0.13], yaw_degrees=-20.0)
    grasps = GraspCandidates(
        canonical_poses=canonical[None, ...],
        scores=np.asarray([0.9], dtype=np.float32),
        type_ids=np.asarray([4], dtype=np.int32),
        collision_free=np.asarray([True], dtype=np.bool_),
        collision_checked=np.asarray([True], dtype=np.bool_),
        selected_index=selected,
        approach_axis_local=np.asarray([1.0, 0.0, 0.0]),
        hand_poses=hand[None, ...],
        hand_angles=np.asarray([[695, 695, 695, 695, 830, 0]], dtype=np.float32),
    )
    return VisualizationSnapshot(
        scene_points=np.asarray([[0.5, 0.0, 0.0]], dtype=np.float32),
        scene_colors=np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
        object_points=np.asarray([[0.6, 0.2, 0.1]], dtype=np.float32),
        object_colors=np.asarray([[0.9, 0.2, 0.2]], dtype=np.float32),
        grasps=grasps,
        reference_frame="robot_base",
        T_reference_camera=np.eye(4),
        frame_id=3,
        timestamp_s=1.0,
        calibration_id="eye-to-hand-test" if calibrated else "",
        model_name=model_name,
        checkpoint_sha256="a" * 64,
        decision_checkpoint_sha256s=tuple(
            f"{index + 1:064x}" for index in range(8)
        ),
        official_source_commit="c" * 40,
    )


def test_transform_chain_and_stage_order():
    snapshot = _snapshot()
    T_EE_hand = _pose([0.0, 0.0, 0.10], yaw_degrees=30.0)

    plan = build_control_plan(
        snapshot,
        T_EE_hand=T_EE_hand,
        mount_transform_commissioned=True,
        config=ExecutionConfig(enable_thumb_preshape=True),
    )

    expected = snapshot.grasps.hand_poses[0] @ np.linalg.inv(T_EE_hand)
    np.testing.assert_allclose(plan.T_reference_EE_grasp, expected, atol=1e-10)
    assert plan.execution_eligible
    assert [stage.name for stage in plan.stages] == [
        StageName.INSPIRE_OPEN,
        StageName.FRANKA_DEFAULT,
        StageName.FRANKA_PREGRASP,
        StageName.FRANKA_GRASP,
        StageName.EEF_SETTLE_GATE,
        StageName.THUMB_PRESHAPE,
        StageName.INSPIRE_CLOSE,
    ]
    gate_index = [stage.name for stage in plan.stages].index(
        StageName.EEF_SETTLE_GATE
    )
    close_index = [stage.name for stage in plan.stages].index(StageName.INSPIRE_CLOSE)
    assert plan.stages[gate_index].is_verification_gate
    assert gate_index < close_index
    np.testing.assert_array_equal(
        plan.stages[-1].inspire_angles, np.asarray([695, 695, 695, 695, 830, 0])
    )


def test_pregrasp_retreat_uses_canonical_approach_not_hand_axes():
    snapshot = _snapshot()
    distance = 0.12

    plan = build_control_plan(
        snapshot,
        T_EE_hand=np.eye(4),
        mount_transform_commissioned=True,
        config=ExecutionConfig(pregrasp_distance_m=distance),
    )

    # Canonical +X is +Y in robot_base after the 90-degree canonical yaw.
    np.testing.assert_allclose(plan.approach_reference, [0.0, 1.0, 0.0], atol=1e-8)
    retreat = (
        plan.T_reference_EE_pregrasp[:3, 3]
        - plan.T_reference_EE_grasp[:3, 3]
    )
    np.testing.assert_allclose(retreat, [0.0, -distance, 0.0], atol=1e-8)
    np.testing.assert_allclose(
        plan.T_reference_EE_pregrasp[:3, :3],
        plan.T_reference_EE_grasp[:3, :3],
    )


def test_empirical_final_insertion_is_explicit_and_zero_by_default():
    snapshot = _snapshot()
    without_insertion = build_control_plan(
        snapshot,
        T_EE_hand=np.eye(4),
        mount_transform_commissioned=True,
    )
    with_insertion = build_control_plan(
        snapshot,
        T_EE_hand=np.eye(4),
        mount_transform_commissioned=True,
        config=ExecutionConfig(final_insertion_m=0.014),
    )
    delta = (
        with_insertion.T_reference_EE_grasp[:3, 3]
        - without_insertion.T_reference_EE_grasp[:3, 3]
    )
    np.testing.assert_allclose(
        delta, 0.014 * with_insertion.approach_reference, atol=1e-12
    )


def test_uncertainty_and_diagnostic_inputs_fail_closed():
    snapshot = _snapshot()

    with pytest.raises(ValueError, match="T_EE_hand is required"):
        build_control_plan(
            snapshot, T_EE_hand=None, mount_transform_commissioned=True
        )
    with pytest.raises(ValueError, match="mount_transform_commissioned=True"):
        build_control_plan(
            snapshot, T_EE_hand=np.eye(4), mount_transform_commissioned=False
        )

    no_selection = replace(
        snapshot, grasps=replace(snapshot.grasps, selected_index=-1)
    )
    with pytest.raises(ValueError, match="no selected grasp"):
        build_control_plan(
            no_selection, T_EE_hand=np.eye(4), mount_transform_commissioned=True
        )

    diagnostic = replace(
        snapshot,
        model_name=(
            "PCA/OBB commissioning backend (not AnyDexGrasp) + "
            "diagnostic fixed Inspire type 4 (not decision-model output)"
        ),
    )
    with pytest.raises(ValueError, match="planning-only"):
        build_control_plan(
            diagnostic, T_EE_hand=np.eye(4), mount_transform_commissioned=True
        )
    plan = build_control_plan(
        diagnostic,
        T_EE_hand=np.eye(4),
        mount_transform_commissioned=True,
        allow_diagnostic=True,
    )
    assert plan.diagnostic
    assert not plan.execution_eligible
    assert any("planning-only" in reason for reason in plan.execution_blockers)


def test_reference_frame_calibration_and_transform_validation_are_strict():
    snapshot = replace(_snapshot(), reference_frame="camera")
    with pytest.raises(ValueError, match="robot_base"):
        build_control_plan(
            snapshot, T_EE_hand=np.eye(4), mount_transform_commissioned=True
        )

    uncalibrated = _snapshot(calibrated=False)
    plan = build_control_plan(
        uncalibrated, T_EE_hand=np.eye(4), mount_transform_commissioned=True
    )
    assert not plan.execution_eligible
    assert "snapshot has no calibration_id" in plan.execution_blockers

    reflected = np.eye(4)
    reflected[0, 0] = -1.0
    with pytest.raises(ValueError, match=r"determinant must be \+1"):
        validate_rigid_transform(reflected)
    non_finite = np.eye(4)
    non_finite[0, 3] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        validate_rigid_transform(non_finite)


def test_adapter_separates_10mm_mount_plane_from_17p8mm_envelope():
    adapter = AdapterGeometry()

    assert adapter.disk_diameter_m == pytest.approx(0.070)
    assert adapter.spigot_diameter_m == pytest.approx(0.0376)
    assert adapter.mount_plane_offset_m == pytest.approx(0.010)
    assert adapter.total_height_m == pytest.approx(0.0178)
    assert adapter.mount_plane_offset_m != adapter.total_height_m
    assert adapter.mesh_metadata.expected_extents_m == pytest.approx(
        (0.070, 0.070, 0.0178)
    )
    assert not adapter.mesh_metadata.collision_authoritative

    disk, spigot, envelope = adapter.collision_primitives
    assert disk.length_m == pytest.approx(0.010)
    assert spigot.length_m == pytest.approx(0.0078)
    assert envelope.radius_m == pytest.approx(0.035)
    assert envelope.length_m == pytest.approx(0.0178)
    assert envelope.T_adapter_primitive[2, 3] == pytest.approx(0.0089)


def test_top_k_selection_filters_q6_and_collision_instead_of_fixing_index_zero():
    snapshot = _snapshot()
    grasps = snapshot.grasps
    canonical = np.repeat(grasps.canonical_poses, 3, axis=0)
    hand_poses = np.repeat(grasps.hand_poses, 3, axis=0)
    hand_angles = np.repeat(grasps.hand_angles, 3, axis=0)
    hand_angles[:, 5] = (0, 930, 970)
    expanded = replace(
        grasps,
        canonical_poses=canonical,
        hand_poses=hand_poses,
        hand_angles=hand_angles,
        scores=np.asarray([0.99, 0.80, 0.90], dtype=np.float32),
        type_ids=np.asarray([4, 4, 4], dtype=np.int32),
        collision_checked=np.asarray([True, True, True], dtype=np.bool_),
        collision_free=np.asarray([True, True, True], dtype=np.bool_),
        selected_index=0,
    )
    expanded_snapshot = replace(snapshot, grasps=expanded)

    eligible = eligible_candidate_indices(
        expanded_snapshot, thumb_rotate_range=(900, 1000)
    )

    assert eligible == (2, 1)
    assert (
        select_eligible_candidate_index(
            expanded_snapshot, thumb_rotate_range=(900, 1000)
        )
        == 2
    )
    plan = build_control_plan(
        expanded_snapshot,
        T_EE_hand=np.eye(4),
        mount_transform_commissioned=True,
        selected_index=2,
    )
    assert plan.selected_index == 2
    assert plan.execution_eligible


def test_unchecked_collision_result_is_an_execution_blocker():
    snapshot = _snapshot()
    grasps = replace(
        snapshot.grasps,
        collision_checked=np.asarray([False], dtype=np.bool_),
        collision_free=np.asarray([False], dtype=np.bool_),
    )
    plan = build_control_plan(
        replace(snapshot, grasps=grasps),
        T_EE_hand=np.eye(4),
        mount_transform_commissioned=True,
    )
    assert not plan.execution_eligible
    assert "selected grasp has no completed collision check" in plan.execution_blockers
