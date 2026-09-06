from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anydex_pipeline.control_frames import (
    T_MOUNT_SOURCE_AXIS_BASIS,
    compose_T_EE_hand_source,
    compose_T_flange_hand_source,
)
from anydex_pipeline.inspire_hand_model import OFFICIAL_SOURCE_MESH_OFFSET_M
from anydex_pipeline.installed_hand_registration import (
    ICPOptions,
    RegistrationThresholds,
    compose_T_EE_hand,
    decompose_installed_mount,
    register_rigid_hand_model,
    rigid_delta,
    transform_points,
)


ROOT = Path(__file__).resolve().parents[1]
LINK111 = (
    ROOT
    / "third_party/AnyDexGrasp/generate_mesh_and_pointcloud/inspire_urdf"
    / "urdf-five3/meshes/Link111.STL"
)


def _rotation_z(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _asymmetric_model(seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # An asymmetric, hand-sized union with three spatially separated ridges.
    groups = []
    for center, scale, count in (
        ((0.045, 0.000, 0.010), (0.090, 0.050, 0.035), 1800),
        ((0.095, -0.020, 0.030), (0.045, 0.018, 0.025), 800),
        ((0.020, 0.018, -0.010), (0.025, 0.020, 0.018), 600),
    ):
        groups.append(
            rng.uniform(-0.5, 0.5, size=(count, 3)) * np.asarray(scale)
            + np.asarray(center)
        )
    return np.concatenate(groups, axis=0)


def test_static_partial_registration_recovers_known_mount_with_outliers():
    rng = np.random.default_rng(9)
    model = _asymmetric_model()
    true = np.eye(4)
    true[:3, :3] = _rotation_z(-37.0)
    true[:3, 3] = (0.43, -0.12, 0.31)
    visible = model[(model[:, 0] > 0.015) & (model[:, 1] < 0.020)]
    observed = transform_points(true, visible)
    observed += rng.normal(0.0, 0.00020, size=observed.shape)
    clutter = rng.uniform(
        true[:3, 3] + (-0.05, -0.04, -0.03),
        true[:3, 3] + (0.12, 0.04, 0.06),
        size=(len(observed) // 12, 3),
    )
    observed = np.concatenate((observed, clutter), axis=0)

    result = register_rigid_hand_model(
        model,
        observed,
        [
            ("correct", true[:3, :3]),
            ("opposite", true[:3, :3] @ _rotation_z(180.0)),
        ],
        options=ICPOptions(
            correspondence_schedule_m=(0.050, 0.025, 0.010, 0.005),
            iterations_per_stage=30,
            trim_fraction=0.76,
            min_correspondences=150,
        ),
        thresholds=RegistrationThresholds(
            evaluation_inlier_distance_m=0.004,
            min_observed_inlier_ratio=0.70,
            max_observed_p50_m=0.0015,
            max_observed_p90_m=0.006,
        ),
    )
    translation_error, rotation_error = rigid_delta(true, result.T_reference_hand)
    assert result.orientation_candidate_label == "correct"
    assert translation_error < 0.0015
    assert rotation_error < 1.0e-6
    assert result.geometry_gate_passed, result.rejection_reasons


def test_symmetric_geometry_rejects_180_degree_orientation_ambiguity():
    angles = np.linspace(0.0, 2.0 * np.pi, 160, endpoint=False)
    heights = np.linspace(-0.04, 0.04, 20)
    model = np.asarray(
        [
            (0.025 * np.cos(angle), 0.025 * np.sin(angle), height)
            for height in heights
            for angle in angles
        ],
        dtype=np.float64,
    )
    true = np.eye(4)
    true[:3, 3] = (0.50, 0.02, 0.30)
    observed = transform_points(true, model)
    result = register_rigid_hand_model(
        model,
        observed,
        [("zero", np.eye(3)), ("opposite", _rotation_z(180.0))],
        thresholds=RegistrationThresholds(max_multistart_score_gap_ratio=0.98),
    )
    assert result.ambiguous_multistart
    assert not result.geometry_gate_passed
    assert any("orientations" in reason for reason in result.rejection_reasons)


def test_small_exact_surface_patch_cannot_pass_as_full_palm_registration():
    model = _asymmetric_model()
    true = np.eye(4)
    true[:3, :3] = _rotation_z(-42.0)
    true[:3, 3] = (0.48, -0.04, 0.28)
    strip = model[model[:, 0] > np.quantile(model[:, 0], 0.88)]
    observed = transform_points(true, strip)
    result = register_rigid_hand_model(
        model,
        observed,
        [("correct", true[:3, :3])],
        thresholds=RegistrationThresholds(
            evaluation_inlier_distance_m=0.004,
            min_observed_inlier_ratio=0.90,
            max_observed_p50_m=0.002,
            max_observed_p90_m=0.004,
        ),
    )
    assert result.metrics.observed_inlier_ratio > 0.99
    assert result.metrics.observed_p90_m < 0.004
    assert not result.geometry_gate_passed
    assert any("coverage" in reason for reason in result.rejection_reasons)


def test_frame_chain_and_mount_decomposition_round_trip():
    F_T_EE = np.eye(4)
    F_T_EE[:3, 3] = (0.001, -0.002, 0.003)
    T_F_hand = compose_T_flange_hand_source(
        fr3_face_to_rh56_seating_plane_m=0.010,
        assembled_yaw_rad=np.deg2rad(-45.0),
        seating_to_source_origin_m=(0.004, -0.003, 0.034),
    )
    T_EE_hand = compose_T_EE_hand_source(F_T_EE, T_F_hand)
    T_base_EE = np.eye(4)
    T_base_EE[:3, :3] = _rotation_z(22.0)
    T_base_EE[:3, 3] = (0.5, -0.1, 0.4)
    T_base_hand = T_base_EE @ T_EE_hand
    np.testing.assert_allclose(
        compose_T_EE_hand(T_base_EE, T_base_hand), T_EE_hand, atol=1e-12
    )

    decomposed = decompose_installed_mount(
        T_EE_hand,
        F_T_EE,
        fr3_face_to_rh56_seating_plane_m=0.010,
        T_mount_source_axis_basis=T_MOUNT_SOURCE_AXIS_BASIS,
    )
    assert decomposed["assembled_yaw_deg"] == pytest.approx(-45.0)
    np.testing.assert_allclose(
        decomposed["seating_to_hand_source_origin_m"],
        [0.004, -0.003, 0.034],
        atol=1e-12,
    )
    assert decomposed["mount_axis_residual_deg"] < 1.0e-9


def test_link111_is_metres_and_official_source_offset_is_applied_once():
    o3d = pytest.importorskip("open3d")
    mesh = o3d.io.read_triangle_mesh(str(LINK111))
    assert not mesh.is_empty()
    raw_min = np.asarray(mesh.get_min_bound())
    raw_max = np.asarray(mesh.get_max_bound())
    assert np.max(raw_max - raw_min) == pytest.approx(0.128883, abs=2.0e-5)
    source_points = np.asarray(mesh.vertices) + OFFICIAL_SOURCE_MESH_OFFSET_M
    np.testing.assert_allclose(
        source_points.min(axis=0), raw_min + OFFICIAL_SOURCE_MESH_OFFSET_M
    )
    np.testing.assert_allclose(
        source_points.max(axis=0), raw_max + OFFICIAL_SOURCE_MESH_OFFSET_M
    )


def test_registration_rejects_invalid_and_degenerate_clouds():
    model = _asymmetric_model()
    with pytest.raises(ValueError, match="at least 100"):
        register_rigid_hand_model(model, np.zeros((20, 3)), [("I", np.eye(3))])
    with pytest.raises(ValueError, match="finite"):
        bad = model.copy()
        bad[0, 0] = np.nan
        register_rigid_hand_model(bad, model, [("I", np.eye(3))])
    with pytest.raises(ValueError, match="degenerate"):
        line = np.column_stack((np.linspace(0, 1, 200), np.zeros((200, 2))))
        register_rigid_hand_model(model, line, [("I", np.eye(3))])
