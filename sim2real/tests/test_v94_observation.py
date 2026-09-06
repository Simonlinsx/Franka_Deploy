import numpy as np
import pytest

from sim2real.observation.model import (
    MaskedRGBDProjector,
    PolicyHistory,
    PolicyPointFrame,
    PolicyRGBDResolutionAdapter,
    Proprio67Builder,
    infer_fixed_sphere_radius_m,
    pose_from_position_quaternion_wxyz,
    rotation_to_quaternion_wxyz,
)


def test_policy_rgbd_2x_decimation_preserves_aligned_pixels_and_geometry():
    color = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    depth = np.arange(24, dtype=np.uint16).reshape(4, 6) + 1000
    mask = np.zeros((4, 6), dtype=bool)
    mask[2, 4] = True
    source_K = np.asarray(
        [[600.0, 0.0, 3.0], [0.0, 602.0, 2.0], [0.0, 0.0, 1.0]]
    )
    adapter = PolicyRGBDResolutionAdapter(
        camera_K=source_K,
        source_image_size=(6, 4),
        target_image_size=(3, 2),
    )

    adapted = adapter.adapt(
        color_bgr=color,
        depth_raw=depth,
        object_mask=mask,
    )

    np.testing.assert_array_equal(adapted.color_bgr, color[::2, ::2])
    np.testing.assert_array_equal(adapted.depth_raw, depth[::2, ::2])
    np.testing.assert_array_equal(adapted.object_mask, mask[::2, ::2])
    np.testing.assert_allclose(
        adapter.camera_K,
        [[300.0, 0.0, 1.5], [0.0, 301.0, 1.0], [0.0, 0.0, 1.0]],
    )
    source_z = float(depth[2, 4]) * 0.001
    source_xyz = np.asarray(
        [
            (4.0 - source_K[0, 2]) / source_K[0, 0] * source_z,
            (2.0 - source_K[1, 2]) / source_K[1, 1] * source_z,
            source_z,
        ]
    )
    target_xyz = np.asarray(
        [
            (2.0 - adapter.camera_K[0, 2]) / adapter.camera_K[0, 0] * source_z,
            (1.0 - adapter.camera_K[1, 2]) / adapter.camera_K[1, 1] * source_z,
            source_z,
        ]
    )
    np.testing.assert_allclose(target_xyz, source_xyz, atol=1.0e-12, rtol=0.0)
    np.testing.assert_array_equal(
        adapter.mask_to_source_resolution(adapted.object_mask),
        np.repeat(np.repeat(mask[::2, ::2], 2, axis=0), 2, axis=1),
    )


def test_single_active_sphere_dataset_infers_fixed_radius():
    metadata = {
        "ppo_tabletop_domain_randomization": {
            "asset_sampling_weights": [1.0, 0.0],
        },
        "merged_datasets": [
            {
                "name": "sphere60",
                "source_id_start": 0,
                "source_id_count": 1,
            },
            {
                "name": "cylinder70",
                "source_id_start": 1,
                "source_id_count": 1,
            },
        ],
    }
    assert infer_fixed_sphere_radius_m(metadata) == pytest.approx(0.030)


@pytest.mark.parametrize(
    "weights",
    ([0.5, 0.5], [0.0, 1.0]),
)
def test_non_sphere_or_mixed_checkpoint_does_not_infer_sphere(weights):
    metadata = {
        "ppo_tabletop_domain_randomization": {
            "asset_sampling_weights": weights,
        },
        "merged_datasets": [
            {
                "name": "sphere60",
                "source_id_start": 0,
                "source_id_count": 1,
            },
            {
                "name": "cylinder70",
                "source_id_start": 1,
                "source_id_count": 1,
            },
        ],
    }
    assert infer_fixed_sphere_radius_m(metadata) is None


def _projector(
    minimum=1,
    point_feature_dim=6,
    maximum_mask_depth_deviation_m=None,
    **kwargs,
):
    return MaskedRGBDProjector(
        camera_K=np.asarray([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]),
        T_base_camera_optical=np.eye(4),
        image_size=(4, 3),
        depth_range_m=(0.1, 2.0),
        num_points=4,
        minimum_valid_points=minimum,
        point_feature_dim=point_feature_dim,
        maximum_mask_depth_deviation_m=maximum_mask_depth_deviation_m,
        **kwargs,
    )


def test_masked_rgbd_projection_is_row_major_rgb_and_zero_padded():
    color = np.zeros((3, 4, 3), dtype=np.uint8)
    color[0, 1] = [10, 20, 30]
    color[1, 2] = [40, 50, 60]
    depth = np.ones((3, 4), dtype=np.float32)
    mask = np.zeros((3, 4), dtype=bool)
    mask[0, 1] = True
    mask[1, 2] = True
    projector = _projector()
    frame = projector.project(
        color_bgr=color,
        depth_m=depth,
        object_mask=mask,
        T_base_palm_at_capture=np.eye(4),
        captured_at_s=1.0,
        frame_id=7,
    )
    np.testing.assert_allclose(frame.xyzrgb_palm[0, :3], [0.0, -0.5, 1.0])
    np.testing.assert_allclose(frame.xyzrgb_palm[1, :3], [0.5, 0.0, 1.0])
    np.testing.assert_allclose(
        frame.xyzrgb_palm[0, 3:], np.asarray([30, 20, 10]) / 255.0
    )
    np.testing.assert_array_equal(frame.valid, [1, 1, 0, 0])
    np.testing.assert_array_equal(frame.xyzrgb_palm[2:], 0.0)
    provenance = projector.last_effective_object_mask_provenance
    assert provenance.kind == "current_frame_projector_input_from_provider_mask"
    assert provenance.source_frame_id == 7
    assert provenance.area_px == 2
    assert provenance.bbox_xyxy == (1, 0, 2, 1)


def test_xyz_projection_omits_rgb_but_preserves_identical_geometry():
    color = np.full((3, 4, 3), [10, 20, 30], dtype=np.uint8)
    depth = np.ones((3, 4), dtype=np.float32)
    mask = np.zeros((3, 4), dtype=bool)
    mask[0, 1] = True
    mask[1, 2] = True
    common = dict(
        color_bgr=color,
        depth_m=depth,
        object_mask=mask,
        T_base_palm_at_capture=np.eye(4),
        captured_at_s=1.0,
        frame_id=7,
    )
    xyz = _projector(point_feature_dim=3).project(**common)
    xyzrgb = _projector(point_feature_dim=6).project(**common)
    assert xyz.xyzrgb_palm.shape == (4, 3)
    np.testing.assert_array_equal(xyz.xyzrgb_palm, xyzrgb.xyzrgb_palm[:, :3])
    np.testing.assert_array_equal(xyz.valid, xyzrgb.valid)


def test_raw_z16_projection_is_exactly_equivalent_to_metric_depth():
    rng = np.random.default_rng(94)
    color = rng.integers(0, 256, size=(3, 4, 3), dtype=np.uint8)
    depth_raw = np.asarray(
        [
            [0, 99, 100, 101],
            [500, 900, 1500, 1999],
            [2000, 2001, 1100, 750],
        ],
        dtype=np.uint16,
    )
    scale = 0.001
    depth_m = depth_raw.astype(np.float32) * float(scale)
    mask = rng.random((3, 4)) > 0.2
    common = {
        "color_bgr": color,
        "object_mask": mask,
        "T_base_palm_at_capture": np.eye(4),
        "captured_at_s": 3.0,
        "frame_id": 11,
    }
    metric = _projector().project(depth_m=depth_m, **common)
    raw = _projector().project(
        depth_raw=depth_raw,
        depth_scale_m_per_unit=scale,
        **common,
    )
    np.testing.assert_array_equal(raw.xyzrgb_palm, metric.xyzrgb_palm)
    np.testing.assert_array_equal(raw.valid, metric.valid)
    assert raw.source_valid_points == metric.source_valid_points
    assert raw.status == metric.status


def test_semantic_mask_keeps_edge_but_policy_cloud_rejects_mixed_depth():
    color = np.zeros((3, 4, 3), dtype=np.uint8)
    depth = np.ones((3, 4), dtype=np.float32)
    depth[1, 2] = 1.40
    mask = np.zeros((3, 4), dtype=bool)
    mask[0, 0] = True
    mask[0, 1] = True
    mask[1, 2] = True
    projector = _projector(maximum_mask_depth_deviation_m=0.055)

    frame = projector.project(
        color_bgr=color,
        depth_m=depth,
        object_mask=mask,
        T_base_palm_at_capture=np.eye(4),
        captured_at_s=1.0,
        frame_id=1,
    )

    # Projection must not mutate the full semantic mask, while the farther
    # mixed-depth edge pixel is absent from the policy point set.
    assert bool(mask[1, 2])
    assert frame.source_valid_points == 2
    np.testing.assert_array_equal(frame.valid, [1, 1, 0, 0])


def test_support_plane_filters_xyz_without_fragmenting_semantic_cube_mask():
    color = np.zeros((3, 4, 3), dtype=np.uint8)
    depth = np.ones((3, 4), dtype=np.float32)
    mask = np.zeros((3, 4), dtype=bool)
    # One complete rectangular semantic mask: its lower two pixels carry
    # tabletop depth, while the upper two carry measured cube depth.
    mask[0:2, 1:3] = True
    depth[0, 1:3] = 1.20
    projector = MaskedRGBDProjector(
        camera_K=np.asarray(
            [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]
        ),
        T_base_camera_optical=np.eye(4),
        image_size=(4, 3),
        depth_range_m=(0.1, 2.0),
        num_points=4,
        minimum_valid_points=1,
        point_feature_dim=3,
        support_plane_abcd=np.asarray([0.0, 0.0, 1.0, -1.0]),
        support_plane_min_clearance_m=0.05,
    )

    frame = projector.project(
        color_bgr=color,
        depth_m=depth,
        object_mask=mask,
        T_base_palm_at_capture=np.eye(4),
        captured_at_s=1.0,
        frame_id=1,
    )

    assert frame.source_valid_points == 2
    np.testing.assert_array_equal(frame.valid, [1, 1, 0, 0])
    # The binary semantic output shown to the operator remains the complete
    # cube; only the XYZ samples sent to the policy lose tabletop points.
    np.testing.assert_array_equal(projector.last_effective_object_mask, mask)


def test_fixed_sphere_completion_recovers_semantic_top_depth():
    width, height = 80, 60
    K = np.asarray(
        [[600.0, 0.0, 40.0], [0.0, 600.0, 30.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    radius = 0.030
    center = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    rows, columns = np.indices((height, width))
    directions = np.stack(
        (
            (columns - K[0, 2]) / K[0, 0],
            (rows - K[1, 2]) / K[1, 1],
            np.ones((height, width)),
        ),
        axis=-1,
    )
    a = np.sum(directions * directions, axis=-1)
    b = -2.0 * np.einsum("hwc,c->hw", directions, center)
    c = float(center @ center - radius * radius)
    discriminant = b * b - 4.0 * a * c
    mask = discriminant >= 0.0
    depth = np.zeros((height, width), dtype=np.float32)
    depth[mask] = (
        (-b[mask] - np.sqrt(discriminant[mask])) / (2.0 * a[mask])
    ).astype(np.float32)
    # Model the D435 failure in the real image: the RGB semantic silhouette is
    # intact, but its upper edge receives farther background depth.
    corrupted_top = mask & (rows <= int(np.min(rows[mask])) + 6)
    depth[corrupted_top] = np.float32(1.20)
    color = np.zeros((height, width, 3), dtype=np.uint8)

    baseline = MaskedRGBDProjector(
        camera_K=K,
        T_base_camera_optical=np.eye(4),
        image_size=(width, height),
        depth_range_m=(0.1, 2.0),
        num_points=128,
        point_feature_dim=3,
        maximum_mask_depth_deviation_m=0.055,
    ).project(
        color_bgr=color,
        depth_m=depth,
        object_mask=mask,
        T_base_palm_at_capture=np.eye(4),
        captured_at_s=1.0,
        frame_id=1,
    )
    projector = MaskedRGBDProjector(
        camera_K=K,
        T_base_camera_optical=np.eye(4),
        image_size=(width, height),
        depth_range_m=(0.1, 2.0),
        num_points=128,
        point_feature_dim=3,
        maximum_mask_depth_deviation_m=0.055,
        fixed_sphere_completion_radius_m=radius,
    )
    completed = projector.project(
        color_bgr=color,
        depth_m=depth,
        object_mask=mask,
        T_base_palm_at_capture=np.eye(4),
        captured_at_s=1.0,
        frame_id=1,
    )

    assert projector.last_sphere_completion is not None
    assert projector.last_sphere_completion["applied"] is True
    provenance = projector.last_effective_object_mask_provenance
    assert provenance.kind == "current_frame_fixed_sphere_completion_mask"
    assert provenance.source_frame_id == 1
    assert provenance.area_px == int(
        np.count_nonzero(projector.last_effective_object_mask)
    )
    assert provenance.bbox_xyxy is not None
    assert completed.source_valid_points == int(mask.sum())
    assert completed.source_valid_points > baseline.source_valid_points
    completed_xyz = completed.xyzrgb_palm[completed.valid > 0.5]
    baseline_xyz = baseline.xyzrgb_palm[baseline.valid > 0.5]
    assert float(np.min(completed_xyz[:, 1])) < float(np.min(baseline_xyz[:, 1]))
    shell_error = np.abs(np.linalg.norm(completed_xyz - center, axis=1) - radius)
    assert float(np.max(shell_error)) < 1.0e-5


def test_raw_z16_projection_requires_one_valid_depth_representation():
    common = {
        "color_bgr": np.zeros((3, 4, 3), dtype=np.uint8),
        "object_mask": np.ones((3, 4), dtype=bool),
        "T_base_palm_at_capture": np.eye(4),
        "captured_at_s": 1.0,
        "frame_id": 1,
    }
    with pytest.raises(ValueError, match="exactly one"):
        _projector().project(**common)
    with pytest.raises(ValueError, match="exactly one"):
        _projector().project(
            depth_m=np.ones((3, 4), dtype=np.float32),
            depth_raw=np.ones((3, 4), dtype=np.uint16),
            depth_scale_m_per_unit=0.001,
            **common,
        )
    with pytest.raises(ValueError, match="requires depth_scale"):
        _projector().project(depth_raw=np.ones((3, 4), dtype=np.uint16), **common)


def test_too_few_points_reuses_previous_native_palm_without_refreshing_timestamp():
    projector = _projector(minimum=2)
    color = np.zeros((3, 4, 3), dtype=np.uint8)
    depth = np.ones((3, 4), dtype=np.float32)
    good = np.zeros((3, 4), dtype=bool)
    good[0, :2] = True
    first = projector.project(
        color_bgr=color,
        depth_m=depth,
        object_mask=good,
        T_base_palm_at_capture=np.eye(4),
        captured_at_s=1.0,
        frame_id=1,
    )
    bad = np.zeros((3, 4), dtype=bool)
    stale = projector.project(
        color_bgr=color,
        depth_m=depth,
        object_mask=bad,
        T_base_palm_at_capture=np.eye(4),
        captured_at_s=2.0,
        frame_id=2,
    )
    assert stale.status == "stale_palm"
    assert stale.captured_at_s == 1.0
    assert stale.frame_id == 1
    np.testing.assert_array_equal(stale.xyzrgb_palm, first.xyzrgb_palm)
    np.testing.assert_array_equal(projector.last_effective_object_mask, good)
    provenance = projector.last_effective_object_mask_provenance
    assert provenance.kind == "retained_previous_fresh_projector_mask"
    assert provenance.source_frame_id == 1
    assert provenance.source_captured_at_s == 1.0
    assert provenance.area_px == 2
    assert provenance.bbox_xyxy == (0, 0, 1, 0)


def test_stale_palm_cache_is_private_from_fresh_result_mutation():
    projector = _projector(minimum=2)
    color = np.zeros((3, 4, 3), dtype=np.uint8)
    depth = np.ones((3, 4), dtype=np.float32)
    good = np.zeros((3, 4), dtype=bool)
    good[0, :2] = True
    fresh = projector.project(
        color_bgr=color,
        depth_m=depth,
        object_mask=good,
        T_base_palm_at_capture=np.eye(4),
        captured_at_s=1.0,
        frame_id=1,
    )
    expected_points = fresh.xyzrgb_palm.copy()
    expected_valid = fresh.valid.copy()
    fresh.xyzrgb_palm[:] = 123.0
    fresh.valid[:] = 0.0

    stale = projector.project(
        color_bgr=color,
        depth_m=depth,
        object_mask=np.zeros((3, 4), dtype=bool),
        T_base_palm_at_capture=np.eye(4),
        captured_at_s=2.0,
        frame_id=2,
    )
    assert stale.status == "stale_palm"
    np.testing.assert_array_equal(stale.xyzrgb_palm, expected_points)
    np.testing.assert_array_equal(stale.valid, expected_valid)


def test_motion_compensated_fallback_uses_current_mask_without_compounding():
    projector = _projector(
        minimum=2,
        temporal_fallback="motion_compensated",
        temporal_fallback_max_stale_s=0.25,
        temporal_fallback_max_stale_steps=2,
        temporal_fallback_max_image_speed_px_s=100.0,
    )
    color = np.zeros((3, 4, 3), dtype=np.uint8)
    depth = np.ones((3, 4), dtype=np.float32)
    measured_mask = np.zeros((3, 4), dtype=bool)
    measured_mask[0, :2] = True
    measured = projector.project(
        color_bgr=color,
        depth_m=depth,
        object_mask=measured_mask,
        T_base_palm_at_capture=np.eye(4),
        captured_at_s=1.0,
        frame_id=10,
    )

    current_mask = np.zeros((3, 4), dtype=bool)
    current_mask[0, 1] = True
    no_depth = np.zeros_like(depth)
    current_palm = np.eye(4)
    current_palm[0, 3] = 0.1
    predicted1 = projector.project(
        color_bgr=color,
        depth_m=no_depth,
        object_mask=current_mask,
        T_base_palm_at_capture=current_palm,
        captured_at_s=1.05,
        frame_id=11,
    )
    predicted2 = projector.project(
        color_bgr=color,
        depth_m=no_depth,
        object_mask=current_mask,
        T_base_palm_at_capture=current_palm,
        captured_at_s=1.10,
        frame_id=12,
    )

    assert predicted1.status == "motion_compensated"
    assert predicted1.frame_id == 11
    assert predicted1.captured_at_s == pytest.approx(1.05)
    assert predicted1.measured_frame_id == 10
    assert predicted1.measured_at_s == pytest.approx(1.0)
    assert predicted1.fallback_steps == 1
    assert predicted2.fallback_steps == 2
    # Centroid shifts +0.5 px at z=1/fx=2, then current palm subtracts 0.1 m.
    expected_delta_x_palm = 0.25 - 0.1
    selected = measured.valid > 0.5
    np.testing.assert_allclose(
        predicted1.xyzrgb_palm[selected, 0],
        measured.xyzrgb_palm[selected, 0] + expected_delta_x_palm,
    )
    # The second prediction must be identical, not another +0.15 m step.
    np.testing.assert_array_equal(predicted2.xyzrgb_palm, predicted1.xyzrgb_palm)
    provenance = projector.last_effective_object_mask_provenance
    assert provenance.kind == "current_rgb_mask_motion_compensated_from_previous_depth"
    assert provenance.source_frame_id == 12

    expired = projector.project(
        color_bgr=color,
        depth_m=no_depth,
        object_mask=current_mask,
        T_base_palm_at_capture=current_palm,
        captured_at_s=1.15,
        frame_id=13,
    )
    assert expired.status == "invalid_motion_fallback_rejected"
    assert float(np.sum(expired.valid)) == 0.0


def test_motion_compensated_fallback_rejects_empty_fast_and_stale_masks():
    projector = _projector(
        minimum=2,
        temporal_fallback="motion_compensated",
        temporal_fallback_max_stale_s=0.10,
        temporal_fallback_max_image_speed_px_s=5.0,
    )
    color = np.zeros((3, 4, 3), dtype=np.uint8)
    depth = np.ones((3, 4), dtype=np.float32)
    measured_mask = np.zeros((3, 4), dtype=bool)
    measured_mask[0, :2] = True
    projector.project(
        color_bgr=color,
        depth_m=depth,
        object_mask=measured_mask,
        T_base_palm_at_capture=np.eye(4),
        captured_at_s=1.0,
        frame_id=1,
    )
    no_depth = np.zeros_like(depth)
    empty = projector.project(
        color_bgr=color,
        depth_m=no_depth,
        object_mask=np.zeros_like(measured_mask),
        T_base_palm_at_capture=np.eye(4),
        captured_at_s=1.01,
        frame_id=2,
    )
    assert empty.status == "invalid_motion_fallback_rejected"

    shifted = np.zeros_like(measured_mask)
    shifted[0, 2:4] = True
    fast = projector.project(
        color_bgr=color,
        depth_m=no_depth,
        object_mask=shifted,
        T_base_palm_at_capture=np.eye(4),
        captured_at_s=1.02,
        frame_id=3,
    )
    assert fast.status == "invalid_motion_fallback_rejected"

    stale = projector.project(
        color_bgr=color,
        depth_m=no_depth,
        object_mask=measured_mask,
        T_base_palm_at_capture=np.eye(4),
        captured_at_s=1.11,
        frame_id=4,
    )
    assert stale.status == "invalid_motion_fallback_rejected"


def test_history_repeats_initial_then_shifts_oldest_to_newest():
    history = PolicyHistory()
    frame0 = PolicyPointFrame(
        xyzrgb_palm=np.zeros((128, 6), dtype=np.float32),
        valid=np.ones(128, dtype=np.float32),
        captured_at_s=0.0,
        frame_id=0,
        source_valid_points=128,
        status="fresh",
    )
    points, _, proprio = history.append(frame0, np.zeros(67, dtype=np.float32))
    np.testing.assert_array_equal(points, np.zeros((4, 128, 6)))
    np.testing.assert_array_equal(proprio, np.zeros((4, 67)))
    frame1 = PolicyPointFrame(
        xyzrgb_palm=np.ones((128, 6), dtype=np.float32),
        valid=np.ones(128, dtype=np.float32),
        captured_at_s=1.0,
        frame_id=1,
        source_valid_points=128,
        status="fresh",
    )
    points, _, proprio = history.append(frame1, np.ones(67, dtype=np.float32))
    np.testing.assert_array_equal(points[:3], 0.0)
    np.testing.assert_array_equal(points[3], 1.0)
    np.testing.assert_array_equal(proprio[-1], 1.0)


def test_history_supports_xyz_and_rejects_midstream_mode_change():
    history = PolicyHistory(point_feature_dim=3)
    frame = PolicyPointFrame(
        xyzrgb_palm=np.zeros((128, 3), dtype=np.float32),
        valid=np.ones(128, dtype=np.float32),
        captured_at_s=0.0,
        frame_id=0,
        source_valid_points=128,
        status="fresh",
    )
    points, _, _ = history.append(frame, np.zeros(67, dtype=np.float32))
    assert points.shape == (4, 128, 3)
    wrong = PolicyPointFrame(
        xyzrgb_palm=np.zeros((128, 6), dtype=np.float32),
        valid=np.ones(128, dtype=np.float32),
        captured_at_s=1.0,
        frame_id=1,
        source_valid_points=128,
        status="fresh",
    )
    with pytest.raises(ValueError, match="feature mode changed"):
        history.append(wrong, np.zeros(67, dtype=np.float32))


def test_v258_history_supports_eight_xyz_frames_and_96d_proprio():
    history = PolicyHistory(length=8, point_feature_dim=3, proprio_dim=96)
    frame = PolicyPointFrame(
        xyzrgb_palm=np.zeros((128, 3), dtype=np.float32),
        valid=np.ones(128, dtype=np.float32),
        captured_at_s=0.0,
        frame_id=0,
        source_valid_points=128,
        status="fresh",
    )
    points, valid, proprio = history.append(
        frame, np.zeros(96, dtype=np.float32)
    )
    assert points.shape == (8, 128, 3)
    assert valid.shape == (8, 128)
    assert proprio.shape == (8, 96)


def test_v61_history_supports_sixteen_xyz_frames_and_96d_proprio():
    history = PolicyHistory(length=16, point_feature_dim=3, proprio_dim=96)
    frame = PolicyPointFrame(
        xyzrgb_palm=np.zeros((128, 3), dtype=np.float32),
        valid=np.ones(128, dtype=np.float32),
        captured_at_s=0.0,
        frame_id=0,
        source_valid_points=128,
        status="fresh",
    )
    points, valid, proprio = history.append(
        frame, np.zeros(96, dtype=np.float32)
    )
    assert points.shape == (16, 128, 3)
    assert valid.shape == (16, 128)
    assert proprio.shape == (16, 96)


def test_proprio_fingertips_are_expressed_in_palm_frame():
    palm = pose_from_position_quaternion_wxyz(
        [1.0, 2.0, 3.0], [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
    )
    tips_palm = np.asarray([[0.1, 0.0, 0.0]] * 5)
    tips_base = tips_palm @ palm[:3, :3].T + palm[:3, 3]
    builder = Proprio67Builder(q_home_rad=np.zeros(7))
    result = builder.build(
        franka_q_rad=np.zeros(7),
        franka_dq_rad_s=np.zeros(7),
        rh56_virtual_q_policy_order_rad=np.zeros(6),
        rh56_virtual_dq_policy_order_rad_s=np.zeros(6),
        T_base_palm=palm,
        palm_linear_velocity_base_m_s=np.zeros(3),
        palm_angular_velocity_base_rad_s=np.zeros(3),
        fingertip_positions_base_m=tips_base,
    )
    np.testing.assert_allclose(result[39:54].reshape(5, 3), tips_palm, atol=1.0e-7)
    assert result[29] >= 0.0


def test_quaternion_has_stable_positive_w_sign():
    rotation = pose_from_position_quaternion_wxyz([0, 0, 0], [-0.5, 0.5, 0.5, 0.5])[
        :3, :3
    ]
    quaternion = rotation_to_quaternion_wxyz(rotation)
    assert quaternion[0] >= 0.0
    np.testing.assert_allclose(np.linalg.norm(quaternion), 1.0)
