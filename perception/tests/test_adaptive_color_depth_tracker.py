import hashlib
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import MethodType, SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from dynamic_pcd.config import DEFAULT_CONFIG
from dynamic_pcd.segmentation.adaptive_color_depth_tracker import (
    AdaptiveColorDepthTracker,
    TargetAppearanceProbabilityROI,
    _ComponentCandidate,
    _binary_mask_iou,
)
from dynamic_pcd.provider.object_pcd_provider import ObjectPCDProvider
from dynamic_pcd.types import MaskResult
from dynamic_pcd.types import CameraIntrinsics, RGBDFrame
from dynamic_pcd.utils.geometry import enlarge_bbox


def _cfg():
    return {
        "roi_scale": 2.5,
        "lock_roi": False,
        "depth_tolerance": 0.04,
        "max_depth_tolerance": 0.07,
        "depth_spread_margin": 0.005,
        "min_area": 40,
        "morph_kernel": 3,
        "max_area_growth": 1.8,
        "max_area_vs_initial": 2.5,
        "max_bbox_growth": 3.0,
        "lost_after": 3,
        "color_probability_threshold": 0.56,
        "component_min_confidence": 0.62,
        "component_max_center_jump_px": 80.0,
        "min_area_ratio": 0.35,
        "appearance_update_min_confidence": 0.82,
        "appearance_update_alpha": 0.025,
        "appearance_anchor_weight": 0.7,
        "reinit_min_color_score": 0.62,
    }


def _frame(
    target_bbox,
    target_bgr,
    *,
    width=180,
    height=120,
    frame_id=1,
    background_bgr=(45, 45, 45),
    target_depth_mm=700,
    distractor=None,
    timestamp=None,
):
    color = np.empty((height, width, 3), dtype=np.uint8)
    color[:] = background_bgr
    depth = np.full((height, width), target_depth_mm, dtype=np.uint16)
    if target_bbox is not None:
        x1, y1, x2, y2 = target_bbox
        color[y1:y2, x1:x2] = target_bgr
    if distractor is not None:
        bbox, bgr = distractor
        x1, y1, x2, y2 = bbox
        color[y1:y2, x1:x2] = bgr
    return RGBDFrame(
        color_bgr=color,
        depth_raw=depth,
        depth_scale=0.001,
        intrinsics=CameraIntrinsics(
            width=width,
            height=height,
            fx=200.0,
            fy=200.0,
            ppx=(width - 1) / 2.0,
            ppy=(height - 1) / 2.0,
        ),
        timestamp=(
            float(frame_id) / 30.0
            if timestamp is None
            else float(timestamp)
        ),
        frame_id=frame_id,
    )


def _mask(shape, bbox):
    result = np.zeros(shape, dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    result[y1:y2, x1:x2] = 1
    return result


def _paint_object(frame, bbox, bgr, *, depth_mm=None):
    x1, y1, x2, y2 = bbox
    frame.color_bgr[y1:y2, x1:x2] = bgr
    if depth_mm is not None:
        frame.depth_raw[y1:y2, x1:x2] = depth_mm
    return frame


def _occlusion_recovery_cfg(confirm_frames=2):
    cfg = _cfg()
    cfg.update(
        {
            # Exercise the anchor-only recovery path, not a lucky long-baseline
            # KLT match from before the occlusion.
            "optical_flow_enabled": False,
            "lost_recovery_requires_flow": True,
            "occlusion_recovery_enabled": True,
            # Local-only helper. Tests that exercise the deployment category
            # policy enable full-frame recovery explicitly below.
            "occlusion_global_search_enabled": False,
            "occlusion_recovery_confirm_frames": confirm_frames,
            "occlusion_recovery_search_scale": 4.0,
            "occlusion_recovery_max_center_displacement_px": 42.0,
            "occlusion_recovery_color_probability_threshold": 0.60,
            "occlusion_recovery_min_confidence": 0.68,
            "occlusion_recovery_min_score_margin": 0.08,
            "occlusion_recovery_min_area_ratio": 0.65,
            "occlusion_recovery_max_area_ratio": 1.45,
            "occlusion_recovery_min_bbox_area_ratio": 0.65,
            "occlusion_recovery_max_bbox_area_ratio": 1.45,
            "occlusion_recovery_max_depth_jump_m": 0.06,
            "occlusion_recovery_confirm_center_step_px": 4.0,
            "occlusion_recovery_confirm_bbox_iou": 0.75,
            "occlusion_recovery_confirm_area_ratio": 1.15,
            "occlusion_recovery_confirm_depth_step_m": 0.012,
        }
    )
    return cfg


@pytest.mark.parametrize(
    "first,second",
    [
        (
            np.zeros((9, 13), dtype=np.uint8),
            np.zeros((9, 13), dtype=np.uint8),
        ),
        (
            _mask((9, 13), (0, 0, 5, 4)),
            _mask((9, 13), (3, 2, 9, 7)),
        ),
        (
            np.random.default_rng(20260722).integers(
                -2, 3, size=(73, 101), dtype=np.int16
            ),
            np.random.default_rng(20260723).integers(
                -2, 3, size=(73, 101), dtype=np.int16
            ),
        ),
    ],
)
def test_binary_mask_iou_matches_numpy_exactly(first, second):
    first_bool = np.asarray(first) > 0
    second_bool = np.asarray(second) > 0
    expected_first_area = int(first_bool.sum())
    expected_second_area = int(second_bool.sum())
    expected_intersection = int(np.logical_and(first_bool, second_bool).sum())
    expected_union = (
        expected_first_area + expected_second_area - expected_intersection
    )
    expected_iou = (
        0.0
        if expected_union == 0
        else expected_intersection / expected_union
    )

    iou, first_area, second_area = _binary_mask_iou(first, second)

    assert iou == expected_iou
    assert first_area == expected_first_area
    assert second_area == expected_second_area


@pytest.mark.parametrize(
    "bbox",
    [
        (0, 0, 17, 13),
        (163, 107, 180, 120),
    ],
)
def test_adaptive_appearance_crop_is_elementwise_equivalent_at_edges(bbox):
    tracker = AdaptiveColorDepthTracker(_cfg())
    frame = _frame(None, (0, 0, 0))
    frame.color_bgr[:] = np.random.default_rng(41).integers(
        0, 256, size=frame.color_bgr.shape, dtype=np.uint8
    )
    mask = _mask(frame.depth_raw.shape, bbox)
    h_bins = int(tracker.cfg.get("hue_bins", 24))
    s_bins = int(tracker.cfg.get("saturation_bins", 16))
    v_bins = int(tracker.cfg.get("value_bins", 16))
    tracker.adaptive_hs = tracker._renormalize(
        np.arange(1, h_bins * s_bins + 1, dtype=np.float32).reshape(
            h_bins, s_bins
        )
    )
    tracker.adaptive_sv = tracker._renormalize(
        np.arange(1, s_bins * v_bins + 1, dtype=np.float32).reshape(
            s_bins, v_bins
        )
    )
    before_hs = tracker.adaptive_hs.copy()
    before_sv = tracker.adaptive_sv.copy()
    full_hsv = cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2HSV)
    current_hs, current_sv = tracker._histograms(full_hsv, mask)
    alpha = 0.173
    expected_hs = tracker._renormalize(
        (1.0 - alpha) * before_hs + alpha * current_hs
    )
    expected_sv = tracker._renormalize(
        (1.0 - alpha) * before_sv + alpha * current_sv
    )

    tracker._update_adaptive_appearance(frame, mask, alpha_override=alpha)

    np.testing.assert_array_equal(tracker.adaptive_hs, expected_hs)
    np.testing.assert_array_equal(tracker.adaptive_sv, expected_sv)


def test_adaptive_appearance_empty_mask_is_an_exact_noop(monkeypatch):
    tracker = AdaptiveColorDepthTracker(_cfg())
    frame = _frame(None, (0, 0, 0))
    tracker.adaptive_hs = np.full((24, 16), 1.0 / (24 * 16), np.float32)
    tracker.adaptive_sv = np.full((16, 16), 1.0 / (16 * 16), np.float32)
    before_hs = tracker.adaptive_hs.copy()
    before_sv = tracker.adaptive_sv.copy()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("empty adaptive mask should skip BGR-to-HSV")

    monkeypatch.setattr(cv2, "cvtColor", fail_if_called)
    tracker._update_adaptive_appearance(
        frame, np.zeros(frame.depth_raw.shape, dtype=np.uint8)
    )

    np.testing.assert_array_equal(tracker.adaptive_hs, before_hs)
    np.testing.assert_array_equal(tracker.adaptive_sv, before_sv)


def _initialized_appearance_tracker_and_query_frame():
    tracker = AdaptiveColorDepthTracker(_cfg())
    initial_bbox = (54, 30, 104, 86)
    initial = _frame(initial_bbox, (210, 80, 235))
    tracker.initialize(
        initial,
        mask=_mask(initial.depth_raw.shape, initial_bbox),
    )
    query = _frame(None, (0, 0, 0), frame_id=2)
    query.color_bgr[:] = np.random.default_rng(20260811).integers(
        0,
        256,
        size=query.color_bgr.shape,
        dtype=np.uint8,
    )
    root = _mask(query.depth_raw.shape, (24, 14, 146, 106))
    return tracker, query, root


def test_target_appearance_probability_roi_is_exact_for_root_and_subsets():
    tracker, frame, root = _initialized_appearance_tracker_and_query_frame()
    disconnected_subset = np.zeros_like(root)
    disconnected_subset[22:47, 31:65] = 1
    disconnected_subset[68:99, 101:137] = 1
    narrow_subset = np.zeros_like(root)
    narrow_subset[33:91, 73:79] = 1

    evidence = tracker.target_appearance_probability_roi(frame, root)
    assert isinstance(evidence, TargetAppearanceProbabilityROI)

    for query in (root, disconnected_subset, narrow_subset):
        for threshold in (0.17, 0.42, 0.83):
            uncached_stats = tracker.target_appearance_support_stats(
                frame,
                query,
                threshold=threshold,
            )
            cached_stats = tracker.target_appearance_support_stats(
                frame,
                query,
                threshold=threshold,
                probability_roi=evidence,
            )
            assert cached_stats == uncached_stats

            uncached_mask = tracker.target_appearance_supported_mask(
                frame,
                query,
                threshold=threshold,
            )
            cached_mask = tracker.target_appearance_supported_mask(
                frame,
                query,
                threshold=threshold,
                probability_roi=evidence,
            )
            np.testing.assert_array_equal(cached_mask, uncached_mask)


def test_target_appearance_probability_roi_is_immutable_and_digest_bound():
    tracker, frame, root = _initialized_appearance_tracker_and_query_frame()
    evidence = tracker.target_appearance_probability_roi(frame, root)
    assert isinstance(evidence, TargetAppearanceProbabilityROI)

    assert evidence.frame_id == frame.frame_id
    assert evidence.timestamp_f64_bits == tracker._timestamp_f64_bits(
        frame.timestamp
    )
    assert evidence.appearance_revision == tracker.appearance_revision
    assert evidence.image_shape_hw == frame.depth_raw.shape
    assert evidence.root_mask_digest == hashlib.sha256(
        evidence.root_mask_bytes
    ).hexdigest()
    assert evidence.root_mask.base is evidence.root_mask_bytes
    assert evidence.probability.base is evidence.probability_bytes
    assert not evidence.root_mask.flags.writeable
    assert not evidence.probability.flags.writeable
    np.testing.assert_array_equal(evidence.root_mask, root)

    with pytest.raises(ValueError):
        evidence.root_mask[20, 30] = 0
    with pytest.raises(ValueError):
        evidence.probability[0, 0] = 0.0


def test_target_appearance_probability_roi_reuses_once_and_falls_back_safely(
    monkeypatch,
):
    tracker, frame, root = _initialized_appearance_tracker_and_query_frame()
    original = tracker._color_probability
    calls = []

    def counted(hsv):
        calls.append(tuple(hsv.shape))
        return original(hsv)

    monkeypatch.setattr(tracker, "_color_probability", counted)
    evidence = tracker.target_appearance_probability_roi(frame, root)
    assert isinstance(evidence, TargetAppearanceProbabilityROI)
    assert len(calls) == 1

    subset = np.zeros_like(root)
    subset[34:82, 64:108] = 1
    tracker.target_appearance_support_stats(
        frame,
        root,
        probability_roi=evidence,
    )
    tracker.target_appearance_support_stats(
        frame,
        subset,
        probability_roi=evidence,
    )
    tracker.target_appearance_supported_mask(
        frame,
        subset,
        probability_roi=evidence,
    )
    assert len(calls) == 1

    # Timestamp identity is bit-exact, even when the numeric difference is
    # only one representable float step.
    changed_timestamp = _frame(
        None,
        (0, 0, 0),
        frame_id=frame.frame_id,
        timestamp=np.nextafter(frame.timestamp, np.inf),
    )
    tracker.target_appearance_support_stats(
        changed_timestamp,
        subset,
        probability_roi=evidence,
    )
    assert len(calls) == 2

    changed_frame_id = _frame(
        None,
        (0, 0, 0),
        frame_id=frame.frame_id + 1,
        timestamp=frame.timestamp,
    )
    tracker.target_appearance_support_stats(
        changed_frame_id,
        subset,
        probability_roi=evidence,
    )
    assert len(calls) == 3

    outside_root = subset.copy()
    outside_root[3:6, 4:8] = 1
    tracker.target_appearance_support_stats(
        frame,
        outside_root,
        probability_roi=evidence,
    )
    assert len(calls) == 4

    tracker._update_adaptive_appearance(frame, root, alpha_override=0.10)
    tracker.target_appearance_support_stats(
        frame,
        subset,
        probability_roi=evidence,
    )
    assert len(calls) == 5


def test_appearance_revision_is_monotonic_across_update_and_restore():
    tracker = AdaptiveColorDepthTracker(_cfg())
    assert tracker.appearance_revision == 0

    bbox = (54, 30, 104, 86)
    frame = _frame(bbox, (210, 80, 235))
    seed = _mask(frame.depth_raw.shape, bbox)
    tracker.initialize(frame, mask=seed)
    initialized_revision = tracker.appearance_revision
    assert initialized_revision == 1
    snapshot = tracker.snapshot_state()

    tracker._update_adaptive_appearance(
        frame,
        np.zeros_like(seed),
        alpha_override=0.10,
    )
    assert tracker.appearance_revision == initialized_revision

    tracker._update_adaptive_appearance(frame, seed, alpha_override=0.10)
    updated_revision = tracker.appearance_revision
    assert updated_revision == initialized_revision + 1

    tracker.restore_state(snapshot)
    restored_revision = tracker.appearance_revision
    assert restored_revision == updated_revision + 1

    tracker.restore_state(None)
    assert tracker.appearance_revision == restored_revision + 1


def test_empty_target_appearance_probability_root_has_no_cache_evidence():
    tracker, frame, _ = _initialized_appearance_tracker_and_query_frame()
    assert (
        tracker.target_appearance_probability_roi(
            frame,
            np.zeros(frame.depth_raw.shape, dtype=np.uint8),
        )
        is None
    )


def test_initial_appearance_uses_deep_core_but_excludes_full_seed_from_background(
    monkeypatch,
):
    cfg = _cfg()
    cfg.update(
        {
            "appearance_seed_erode": 11,
            "background_exclusion_dilation": 3,
            "background_ring_scale": 1.8,
        }
    )
    tracker = AdaptiveColorDepthTracker(cfg)
    frame = _frame(None, (0, 0, 0))
    bbox = np.asarray([50, 30, 110, 90], dtype=np.int32)
    full_seed = _mask(frame.depth_raw.shape, bbox)
    foreground_core = tracker._appearance_training_mask(
        full_seed,
        bbox,
        box_prompt_only=False,
    )
    assert 20 <= int(foreground_core.sum()) < int(full_seed.sum())

    histogram_masks = []

    def capture_histogram_mask(_hsv, mask):
        histogram_masks.append((np.asarray(mask) > 0).astype(np.uint8))
        return (
            np.full((24, 16), 1.0 / (24 * 16), dtype=np.float32),
            np.full((16, 16), 1.0 / (16 * 16), dtype=np.float32),
        )

    monkeypatch.setattr(tracker, "_histograms", capture_histogram_mask)
    tracker._initialize_appearance(
        frame,
        foreground_core=foreground_core,
        foreground_exclusion=full_seed,
        bbox=bbox,
    )

    assert len(histogram_masks) == 2
    np.testing.assert_array_equal(histogram_masks[0], foreground_core)
    ring_bbox = enlarge_bbox(
        bbox,
        float(cfg["background_ring_scale"]),
        frame.intrinsics.width,
        frame.intrinsics.height,
    )
    ring = _mask(frame.depth_raw.shape, ring_bbox.astype(np.int32))
    excluded = cv2.dilate(
        full_seed,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    expected_background = ring & (excluded == 0).astype(np.uint8)
    assert int(expected_background.sum()) >= 40
    np.testing.assert_array_equal(
        histogram_masks[1], expected_background
    )
    removed_target_rim = np.logical_and(
        full_seed > 0, foreground_core == 0
    )
    assert np.any(removed_target_rim)
    assert not np.any(histogram_masks[1][removed_target_rim])


def test_appearance_histogram_prior_is_invariant_to_identical_sample_count():
    tracker = AdaptiveColorDepthTracker(_cfg())
    frame = _frame(None, (0, 0, 0), background_bgr=(225, 95, 25))
    hsv = cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2HSV)
    small = _mask(frame.depth_raw.shape, (40, 30, 60, 50))
    large = _mask(frame.depth_raw.shape, (30, 20, 90, 80))

    small_hs, small_sv = tracker._histograms(hsv, small)
    large_hs, large_sv = tracker._histograms(hsv, large)

    np.testing.assert_array_equal(small_hs, large_hs)
    np.testing.assert_array_equal(small_sv, large_sv)


@pytest.mark.parametrize(
    "bbox",
    [
        (30, 30, 38, 38),
        (30, 30, 70, 34),
    ],
    ids=("small", "thin"),
)
def test_deep_appearance_seed_erosion_safely_falls_back_for_small_or_thin_masks(
    bbox,
):
    cfg = _cfg()
    cfg["appearance_seed_erode"] = 11
    tracker = AdaptiveColorDepthTracker(cfg)
    seed = _mask((120, 180), bbox)

    core = tracker._appearance_training_mask(
        seed,
        np.asarray(bbox, dtype=np.int32),
        box_prompt_only=False,
    )

    np.testing.assert_array_equal(core, seed)


def test_deployment_appearance_seed_erosion_is_pinned_to_eleven():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "d435_default.yaml"
    )
    deployed = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert DEFAULT_CONFIG["tracker"]["appearance_seed_erode"] == 11
    assert deployed["tracker"]["appearance_seed_erode"] == 11


@pytest.mark.parametrize(
    "invalid_case",
    (
        "empty_core",
        "empty_exclusion",
        "nan_core",
        "nan_exclusion",
        "core_shape",
        "exclusion_shape",
        "core_outside_exclusion",
        "empty_background",
    ),
)
def test_invalid_initial_appearance_masks_cannot_partially_replace_model(
    invalid_case,
):
    tracker = AdaptiveColorDepthTracker(_cfg())
    frame = _frame(None, (0, 0, 0))
    bbox = np.asarray([50, 30, 110, 90], dtype=np.int32)
    core = _mask(frame.depth_raw.shape, (60, 40, 100, 80))
    exclusion = _mask(frame.depth_raw.shape, bbox)
    expected_error = (ValueError, RuntimeError)

    if invalid_case == "empty_core":
        core[:] = 0
    elif invalid_case == "empty_exclusion":
        exclusion[:] = 0
    elif invalid_case == "nan_core":
        core = core.astype(np.float32)
        core[0, 0] = np.nan
    elif invalid_case == "nan_exclusion":
        exclusion = exclusion.astype(np.float32)
        exclusion[0, 0] = np.nan
    elif invalid_case == "core_shape":
        core = core[:-1]
    elif invalid_case == "exclusion_shape":
        exclusion = exclusion[:, :-1]
    elif invalid_case == "core_outside_exclusion":
        core[10, 10] = 1
    elif invalid_case == "empty_background":
        core = np.ones(frame.depth_raw.shape, dtype=np.uint8)
        exclusion = core.copy()
        bbox = np.asarray(
            [0, 0, frame.intrinsics.width, frame.intrinsics.height],
            dtype=np.int32,
        )
    else:  # pragma: no cover - the parametrization is closed above.
        raise AssertionError(invalid_case)

    original = {
        "anchor_hs": np.full((24, 16), 1.0, dtype=np.float32),
        "anchor_sv": np.full((16, 16), 2.0, dtype=np.float32),
        "adaptive_hs": np.full((24, 16), 3.0, dtype=np.float32),
        "adaptive_sv": np.full((16, 16), 4.0, dtype=np.float32),
        "background_hs": np.full((24, 16), 5.0, dtype=np.float32),
        "background_sv": np.full((16, 16), 6.0, dtype=np.float32),
    }
    for name, value in original.items():
        setattr(tracker, name, value.copy())

    with pytest.raises(expected_error):
        tracker._initialize_appearance(
            frame,
            foreground_core=core,
            foreground_exclusion=exclusion,
            bbox=bbox,
        )

    for name, value in original.items():
        np.testing.assert_array_equal(getattr(tracker, name), value)


@pytest.mark.parametrize(
    "target_bgr",
    [
        (210, 80, 235),  # saturated magenta-like object
        (230, 100, 25),  # blue object: proves no hue is hardcoded
        (215, 215, 215),  # low-saturation object uses the learned S/V model
    ],
)
def test_tracks_moving_prompted_object_and_rejects_same_depth_background(
    target_bgr,
):
    tracker = AdaptiveColorDepthTracker(_cfg())
    initial_bbox = (35, 38, 65, 78)
    initial = _frame(initial_bbox, target_bgr)
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, initial_bbox))

    moved_bbox = (49, 41, 79, 81)
    # A different-color object is deliberately at exactly the same depth.
    moved = _frame(
        moved_bbox,
        target_bgr,
        frame_id=2,
        distractor=((92, 35, 125, 80), (20, 220, 30)),
    )
    result = tracker.update(moved)

    assert result.valid, result.message
    assert result.score >= 0.62
    expected = _mask(moved.depth_raw.shape, moved_bbox).astype(bool)
    predicted = result.mask.astype(bool)
    intersection = np.logical_and(expected, predicted).sum()
    union = np.logical_or(expected, predicted).sum()
    assert intersection / union > 0.90
    assert result.mask[35:80, 92:125].sum() == 0
    assert tracker.tracking_status["valid"]
    assert tracker.tracking_status["lost_count"] == 0


def test_semantic_mask_keeps_visible_edge_with_background_depth():
    cfg = _cfg()
    cfg.update(
        {
            "semantic_mask_depth_pixel_gate_enabled": False,
            "component_min_valid_depth_ratio": 0.25,
            "temporal_mask_stability_enabled": False,
        }
    )
    tracker = AdaptiveColorDepthTracker(cfg)
    bbox = (50, 35, 90, 75)
    color = (210, 80, 235)
    initial = _frame(bbox, color)
    seed = _mask(initial.depth_raw.shape, bbox)
    tracker.initialize(initial, mask=seed)

    current = _frame(bbox, color, frame_id=2)
    # RGB still shows the prompted object, but its top edge has the D435
    # background-depth failure observed on the real sphere.
    current.depth_raw[35:43, 50:90] = 1000
    result = tracker.update(current)

    assert result.valid, result.message
    np.testing.assert_array_equal(result.mask, seed)
    assert tracker.state.depth_median == pytest.approx(0.7, abs=1e-6)


def test_initialization_uses_only_commissioned_pointcloud_depth_range():
    cfg = _cfg()
    cfg.update({"z_min": 0.25, "z_max": 1.65})
    tracker = AdaptiveColorDepthTracker(cfg)
    bbox = (50, 35, 90, 75)
    initial = _frame(bbox, (210, 80, 235), target_depth_mm=2500)
    seed = _mask(initial.depth_raw.shape, bbox)
    # The target return is a minority of the semantic mask.  Without the
    # commissioned range the far background becomes the initialization p50.
    initial.depth_raw[35:43, 50:90] = 800

    result = tracker.initialize(initial, mask=seed)

    assert result.valid, result.message
    assert tracker.state is not None
    assert tracker.state.depth_median == pytest.approx(0.8)
    assert "z=0.800m" in result.message


def test_steady_tracking_scales_only_consumed_depth_pixels(monkeypatch):
    """The normal path must not materialize a full-frame float depth image."""

    tracker = AdaptiveColorDepthTracker(_cfg())
    initial_bbox = (35, 38, 65, 78)
    moved_bbox = (38, 39, 68, 79)
    color = (210, 80, 235)
    initial = _frame(initial_bbox, color)
    tracker.initialize(
        initial, mask=_mask(initial.depth_raw.shape, initial_bbox)
    )
    current = _frame(moved_bbox, color, frame_id=2)

    def reject_full_depth(_self):
        raise AssertionError("steady tracker requested full-frame depth_m")

    monkeypatch.setattr(RGBDFrame, "depth_m", property(reject_full_depth))
    result = tracker.update(current)

    assert result.valid, result.message
    assert result.mask.sum() > 0


def test_missing_target_fails_closed_without_publishing_stale_mask():
    tracker = AdaptiveColorDepthTracker(_cfg())
    bbox = (40, 35, 72, 78)
    initial = _frame(bbox, (220, 90, 230))
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, bbox))
    previous_bbox = tracker.state.bbox_xyxy.copy()

    missing = _frame(None, (0, 0, 0), frame_id=2)
    result = tracker.update(missing)

    assert not result.valid
    assert result.mask.sum() == 0
    np.testing.assert_array_equal(result.bbox_xyxy, previous_bbox)
    assert tracker.lost
    assert tracker.confidence == 0.0
    assert tracker.tracking_status["lost_count"] == 1


def test_motion_schmitt_gate_does_not_grow_a_weak_probability_halo(
    monkeypatch,
):
    cfg = _cfg()
    cfg.update(
        {
            "flow_color_probability_floor": 0.42,
            "flow_enter_color_probability_threshold": 0.64,
            "tracked_motion_min_mask_iou": 0.50,
        }
    )
    tracker = AdaptiveColorDepthTracker(cfg)
    bbox = (50, 35, 80, 75)
    frame = _frame(bbox, (210, 80, 235))
    seed = _mask(frame.depth_raw.shape, bbox)
    tracker.initialize(frame, mask=seed)
    flow = SimpleNamespace(
        mask=seed.copy(),
        bbox_xyxy=np.asarray(bbox, dtype=np.int32),
        center_uv=(65.0, 55.0),
        confidence=1.0,
    )
    roi = enlarge_bbox(
        flow.bbox_xyxy,
        float(cfg["roi_scale"]),
        frame.depth_raw.shape[1],
        frame.depth_raw.shape[0],
    )
    x1, y1, x2, y2 = roi.astype(int)
    probability = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
    # Core is strong. The surrounding two-pixel ring is deliberately above
    # the retention threshold but below the new-pixel entry threshold.
    probability[
        bbox[1] - y1 : bbox[3] - y1,
        bbox[0] - x1 : bbox[2] - x1,
    ] = 0.90
    probability[
        bbox[1] - y1 - 2 : bbox[3] - y1 + 2,
        bbox[0] - x1 - 2 : bbox[2] - x1 + 2,
    ] = np.maximum(
        probability[
            bbox[1] - y1 - 2 : bbox[3] - y1 + 2,
            bbox[0] - x1 - 2 : bbox[2] - x1 + 2,
        ],
        0.50,
    )
    monkeypatch.setattr(
        tracker, "_color_probability", lambda _hsv: probability.copy()
    )

    candidate, reason = tracker._best_component(frame, flow)

    assert candidate is not None, reason
    assert int(candidate.mask.sum()) == int(seed.sum())
    np.testing.assert_array_equal(candidate.mask, seed)


def test_motion_contour_gate_rejects_a_large_same_frame_expansion():
    tracker = AdaptiveColorDepthTracker(_cfg())
    predicted = _mask((120, 180), (50, 35, 80, 75))
    expanded = _mask((120, 180), (45, 30, 87, 80))
    candidate = SimpleNamespace(mask=expanded, area=float(expanded.sum()))
    flow = SimpleNamespace(mask=predicted)

    reason = tracker._tracked_motion_consensus_error(candidate, flow)

    assert reason is not None
    assert "area_x" in reason or "IoU" in reason


def test_semantic_motion_gate_accepts_observed_sam2_contour_correction():
    cfg = _cfg()
    cfg.update(
        {
            "tracked_motion_min_area_ratio": 0.60,
            "tracked_motion_max_area_ratio": 1.60,
            "tracked_motion_min_mask_iou": 0.50,
        }
    )
    tracker = AdaptiveColorDepthTracker(cfg)
    predicted = _mask((120, 180), (50, 35, 90, 75))
    # 1.293x reproduces v212-mask-fixed-20260728-014804. Keep the
    # correction spatially attached to the previous semantic contour.
    corrected = predicted.copy()
    corrected[31:35, 50:90] = 1
    corrected[75:82, 50:90] = 1
    corrected[35:75, 90] = 1
    ratio = corrected.sum() / predicted.sum()
    assert ratio == pytest.approx(1.30, abs=0.01)
    candidate = SimpleNamespace(mask=corrected, area=float(corrected.sum()))
    flow = SimpleNamespace(mask=predicted, confidence=0.95)

    reason = tracker._tracked_motion_consensus_error(candidate, flow)

    assert reason is None


def test_no_flow_nudge_above_identity_bound_still_fails_closed():
    cfg = _cfg()
    cfg.update(
        {
            "optical_flow_enabled": False,
            "roi_scale": 4.0,
            "component_max_motion_px": 24.0,
            "component_motion_bbox_diagonal_ratio": 0.80,
            "component_max_motion_cap_px": 100.0,
            "temporal_mask_geometry_max_displacement_px": 0.0,
        }
    )
    tracker = AdaptiveColorDepthTracker(cfg)
    initial_bbox = (25, 35, 55, 75)
    initial = _frame(initial_bbox, (210, 80, 235), frame_id=1)
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, initial_bbox))

    # One exact frame cannot distinguish this from an identical neighbouring
    # instance. Scale-aware relaxation is therefore reserved for reliable KLT.
    moved_bbox = (60, 35, 90, 75)
    moved = _frame(moved_bbox, (210, 80, 235), frame_id=2)
    result = tracker.update(moved)

    assert not result.valid
    assert result.mask.sum() == 0
    assert "motion" in result.message


def test_no_flow_motion_gate_rejects_a_larger_teleport():
    cfg = _cfg()
    cfg.update(
        {
            "optical_flow_enabled": False,
            "roi_scale": 8.0,
            "component_max_motion_px": 24.0,
            "component_motion_bbox_diagonal_ratio": 0.80,
            "component_max_motion_cap_px": 100.0,
        }
    )
    tracker = AdaptiveColorDepthTracker(cfg)
    initial_bbox = (25, 35, 55, 75)
    initial = _frame(initial_bbox, (210, 80, 235), frame_id=1)
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, initial_bbox))

    teleported_bbox = (60, 35, 90, 75)
    teleported = _frame(teleported_bbox, (210, 80, 235), frame_id=2)
    result = tracker.update(teleported)

    assert not result.valid
    assert result.mask.sum() == 0
    assert "motion" in result.message
    assert ">24.0px" in result.message


def test_high_confidence_klt_large_motion_enters_publication_probation(
    monkeypatch,
):
    cfg = _cfg()
    cfg.update(
        {
            "component_max_motion_px": 24.0,
            "component_motion_bbox_diagonal_ratio": 0.80,
            "component_max_motion_cap_px": 100.0,
            "component_flow_motion_min_confidence": 0.45,
            "flow_max_displacement_px": 100.0,
        }
    )
    tracker = AdaptiveColorDepthTracker(cfg)
    initial_bbox = (20, 35, 50, 75)
    initial = _frame(initial_bbox, (210, 80, 235), frame_id=1)
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, initial_bbox))

    moved_bbox = (70, 35, 100, 75)
    moved = _frame(moved_bbox, (210, 80, 235), frame_id=2)
    moved_mask = _mask(moved.depth_raw.shape, moved_bbox)
    flow = _identity_flow(moved_mask, confidence=0.95, tx=50.0)
    flow.message = "KLT synthetic tx=50px"
    monkeypatch.setattr(
        tracker, "_predict_with_optical_flow", lambda _gray: flow
    )

    first = tracker.update(moved)

    assert not first.valid
    assert first.mask.sum() == 0
    assert "motion probation" in first.message
    # The exact candidate advances internal KLT state but never leaks a point
    # cloud. A stable following frame can now resume; provider-level tests own
    # the subsequent multi-frame publication quarantine contract.
    steady = _frame(moved_bbox, (210, 80, 235), frame_id=3)
    identity = _identity_flow(moved_mask, confidence=0.95)
    identity.message = "KLT synthetic steady"
    monkeypatch.setattr(
        tracker, "_predict_with_optical_flow", lambda _gray: identity
    )
    second = tracker.update(steady)
    assert second.valid, second.message
    np.testing.assert_array_equal(second.mask, moved_mask)


def _timestamp_velocity_cfg():
    cfg = _cfg()
    cfg.update(
        {
            "optical_flow_enabled": False,
            "component_motion_model": "timestamp_velocity",
            "component_motion_max_speed_px_s": 2400.0,
            "component_motion_velocity_ema_beta": 0.70,
            "component_motion_max_frame_gap": 3,
            "component_motion_max_timestamp_gap_s": 0.20,
            "component_motion_prediction_residual_bbox_diagonal_ratio": 0.80,
            "component_motion_prediction_residual_min_px": 12.0,
            "component_motion_prediction_residual_max_px": 48.0,
            "occlusion_recovery_motion_max_frame_gap": 6,
            "occlusion_recovery_confirm_frames": 2,
            "occlusion_recovery_confirm_max_frame_gap": 3,
            "occlusion_recovery_confirm_max_timestamp_step_s": 0.20,
            "occlusion_global_search_enabled": True,
            "occlusion_global_search_after_frames": 2,
        }
    )
    return cfg


@pytest.mark.parametrize("step_px", [35, 50, 70])
def test_timestamp_velocity_accepts_fast_first_motion(step_px):
    """30 Hz motion up to the measured 71 px/frame case is not LOST."""

    tracker = AdaptiveColorDepthTracker(_timestamp_velocity_cfg())
    color = (210, 80, 235)
    initial_bbox = (50, 70, 90, 110)
    initial = _frame(
        initial_bbox,
        color,
        width=500,
        height=180,
        frame_id=1,
        timestamp=0.0,
    )
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, initial_bbox))

    moved_bbox = tuple(
        value + (step_px if index % 2 == 0 else 0)
        for index, value in enumerate(initial_bbox)
    )
    moved = _frame(
        moved_bbox,
        color,
        width=500,
        height=180,
        frame_id=2,
        timestamp=1.0 / 30.0,
    )
    result = tracker.update(moved)

    assert result.valid, result.message
    assert "probation" not in result.message
    assert tracker.tracking_status["committed_velocity_samples"] == 1
    assert tracker.tracking_status["committed_velocity_uv_px_s"][0] == pytest.approx(
        step_px * 30.0, rel=0.02
    )


@pytest.mark.parametrize("step_px", [35, 70])
def test_timestamp_velocity_remains_valid_for_continuous_fast_motion(step_px):
    """Velocity prediction must not quarantine every frame of a fast rollout."""

    tracker = AdaptiveColorDepthTracker(_timestamp_velocity_cfg())
    color = (210, 80, 235)
    initial_bbox = (30, 70, 70, 110)
    width = 1400
    initial = _frame(
        initial_bbox,
        color,
        width=width,
        height=180,
        frame_id=1,
        timestamp=0.0,
    )
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, initial_bbox))

    for observation_index in range(1, 16):
        frame_id = observation_index + 1
        x_offset = observation_index * step_px
        bbox = (
            initial_bbox[0] + x_offset,
            initial_bbox[1],
            initial_bbox[2] + x_offset,
            initial_bbox[3],
        )
        result = tracker.update(
            _frame(
                bbox,
                color,
                width=width,
                height=180,
                frame_id=frame_id,
                timestamp=observation_index / 30.0,
            )
        )
        assert result.valid, f"frame={frame_id}: {result.message}"
        assert "probation" not in result.message

    velocity = tracker.tracking_status["committed_velocity_uv_px_s"]
    assert velocity[0] == pytest.approx(step_px * 30.0, rel=0.02)
    assert tracker.tracking_status["committed_velocity_samples"] == 15


def test_timestamp_velocity_accepts_variable_dt_and_frame_gaps():
    tracker = AdaptiveColorDepthTracker(_timestamp_velocity_cfg())
    color = (210, 80, 235)
    initial_bbox = (50, 70, 90, 110)
    initial = _frame(
        initial_bbox,
        color,
        width=700,
        height=180,
        frame_id=1,
        timestamp=0.0,
    )
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, initial_bbox))

    # Constant 2000 px/s sampled with 25, 60 and 100 ms intervals and frame
    # gaps 1, 2 and 3. A fixed px/frame threshold cannot represent this.
    observations = (
        (2, 0.025, 50),
        (4, 0.085, 170),
        (7, 0.185, 370),
    )
    for frame_id, timestamp, x_offset in observations:
        bbox = (
            initial_bbox[0] + x_offset,
            initial_bbox[1],
            initial_bbox[2] + x_offset,
            initial_bbox[3],
        )
        result = tracker.update(
            _frame(
                bbox,
                color,
                width=700,
                height=180,
                frame_id=frame_id,
                timestamp=timestamp,
            )
        )
        assert result.valid, result.message

    velocity = tracker.tracking_status["committed_velocity_uv_px_s"]
    assert velocity[0] == pytest.approx(2000.0, rel=0.02)
    assert abs(velocity[1]) < 1.0


def test_timestamp_velocity_rejects_direction_reversal_residual():
    tracker = AdaptiveColorDepthTracker(_timestamp_velocity_cfg())
    color = (210, 80, 235)
    initial_bbox = (150, 70, 190, 110)
    initial = _frame(
        initial_bbox,
        color,
        width=500,
        height=180,
        frame_id=1,
        timestamp=0.0,
    )
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, initial_bbox))

    forward_bbox = (220, 70, 260, 110)
    forward = tracker.update(
        _frame(
            forward_bbox,
            color,
            width=500,
            height=180,
            frame_id=2,
            timestamp=1.0 / 30.0,
        )
    )
    assert forward.valid, forward.message
    committed_before = tracker.committed_motion

    # Scalar speed is plausible (35 px/frame), but it reverses against a
    # +2100 px/s committed prediction and must not become a new target state.
    reversed_bbox = (185, 70, 225, 110)
    reversed_result = tracker.update(
        _frame(
            reversed_bbox,
            color,
            width=500,
            height=180,
            frame_id=3,
            timestamp=2.0 / 30.0,
        )
    )
    assert not reversed_result.valid
    assert "maneuver recovery pending 1/3" in reversed_result.message
    assert tracker.committed_motion.frame_id == committed_before.frame_id
    assert tracker.committed_motion.center_uv == committed_before.center_uv
    assert tracker.tracking_status["maneuver_pending_count"] == 1

    # A lone conflicting observation is never enough to alter velocity, and
    # a following no-evidence frame clears its tentative maneuver hypothesis.
    missing = tracker.update(
        _frame(
            None,
            color,
            width=500,
            height=180,
            frame_id=4,
            timestamp=3.0 / 30.0,
        )
    )
    assert not missing.valid
    assert tracker.tracking_status["maneuver_pending_count"] == 0
    assert tracker.committed_motion.frame_id == committed_before.frame_id


def test_timestamp_velocity_recovers_a_fast_reversal_in_three_candidates():
    tracker = AdaptiveColorDepthTracker(_timestamp_velocity_cfg())
    color = (210, 80, 235)
    initial_bbox = (250, 70, 290, 110)
    width = 900
    initial = _frame(
        initial_bbox,
        color,
        width=width,
        height=180,
        frame_id=1,
        timestamp=0.0,
    )
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, initial_bbox))

    # Establish a committed +2100 px/s trajectory.
    for observation_index in range(1, 4):
        offset = 70 * observation_index
        bbox = (
            initial_bbox[0] + offset,
            initial_bbox[1],
            initial_bbox[2] + offset,
            initial_bbox[3],
        )
        result = tracker.update(
            _frame(
                bbox,
                color,
                width=width,
                height=180,
                frame_id=observation_index + 1,
                timestamp=observation_index / 30.0,
            )
        )
        assert result.valid, result.message
    assert tracker.committed_motion.velocity_uv_s[0] == pytest.approx(2100.0)

    # Bounce and continue at -2100 px/s. The first two candidates stay closed;
    # the third resets committed velocity from the latest tentative pair.
    invalid_count = 0
    forward_offset = 210
    for reverse_index in range(1, 4):
        offset = forward_offset - 70 * reverse_index
        bbox = (
            initial_bbox[0] + offset,
            initial_bbox[1],
            initial_bbox[2] + offset,
            initial_bbox[3],
        )
        frame_id = 4 + reverse_index
        result = tracker.update(
            _frame(
                bbox,
                color,
                width=width,
                height=180,
                frame_id=frame_id,
                timestamp=(frame_id - 1) / 30.0,
            )
        )
        if reverse_index < 3:
            invalid_count += 1
            assert not result.valid
            assert "maneuver recovery pending" in result.message
            assert tracker.tracking_status["maneuver_pending_count"] == reverse_index
        else:
            assert result.valid, result.message
            assert tracker.tracking_status["temporal_motion_source"] == "maneuver"

    assert invalid_count <= 3
    assert tracker.tracking_status["maneuver_pending_count"] == 0
    assert tracker.committed_motion.velocity_uv_s[0] == pytest.approx(
        -2100.0, rel=0.02
    )

    # Once reset, the new direction remains continuously valid instead of
    # falling back into the old six-frame LOST horizon.
    for continuation_index in range(1, 4):
        offset = -70 * continuation_index
        bbox = (
            initial_bbox[0] + offset,
            initial_bbox[1],
            initial_bbox[2] + offset,
            initial_bbox[3],
        )
        frame_id = 7 + continuation_index
        result = tracker.update(
            _frame(
                bbox,
                color,
                width=width,
                height=180,
                frame_id=frame_id,
                timestamp=(frame_id - 1) / 30.0,
            )
        )
        assert result.valid, f"frame={frame_id}: {result.message}"


def test_timestamp_velocity_no_evidence_does_not_poison_fast_recovery():
    tracker = AdaptiveColorDepthTracker(_timestamp_velocity_cfg())
    color = (210, 80, 235)
    initial_bbox = (50, 70, 90, 110)
    initial = _frame(
        initial_bbox,
        color,
        width=500,
        height=180,
        frame_id=1,
        timestamp=0.0,
    )
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, initial_bbox))

    moved_bbox = (120, 70, 160, 110)
    assert tracker.update(
        _frame(
            moved_bbox,
            color,
            width=500,
            height=180,
            frame_id=2,
            timestamp=1.0 / 30.0,
        )
    ).valid
    committed = tracker.committed_motion

    missing = tracker.update(
        _frame(
            None,
            color,
            width=500,
            height=180,
            frame_id=3,
            timestamp=2.0 / 30.0,
        )
    )
    assert not missing.valid
    assert tracker.committed_motion.frame_id == committed.frame_id
    assert tracker.committed_motion.velocity_uv_s == committed.velocity_uv_s

    first_recovery_bbox = (260, 70, 300, 110)
    pending = tracker.update(
        _frame(
            first_recovery_bbox,
            color,
            width=500,
            height=180,
            frame_id=4,
            timestamp=3.0 / 30.0,
        )
    )
    assert not pending.valid
    assert "recovery pending" in pending.message
    assert tracker.committed_motion.frame_id == committed.frame_id

    second_recovery_bbox = (330, 70, 370, 110)
    recovered = tracker.update(
        _frame(
            second_recovery_bbox,
            color,
            width=500,
            height=180,
            frame_id=5,
            timestamp=4.0 / 30.0,
        )
    )
    assert recovered.valid, recovered.message
    assert tracker.committed_motion.frame_id == 5
    assert tracker.committed_motion.velocity_uv_s[0] == pytest.approx(
        2100.0, rel=0.02
    )


def test_rejected_hand_like_conflict_cannot_change_committed_velocity():
    tracker = AdaptiveColorDepthTracker(_timestamp_velocity_cfg())
    target_color = (210, 80, 235)
    hand_color = (90, 145, 205)
    initial_bbox = (50, 70, 90, 110)
    initial = _frame(
        initial_bbox,
        target_color,
        width=500,
        height=180,
        frame_id=1,
        timestamp=0.0,
    )
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, initial_bbox))
    moved_bbox = (120, 70, 160, 110)
    assert tracker.update(
        _frame(
            moved_bbox,
            target_color,
            width=500,
            height=180,
            frame_id=2,
            timestamp=1.0 / 30.0,
        )
    ).valid
    committed = tracker.committed_motion

    hand = _frame(
        None,
        target_color,
        width=500,
        height=180,
        frame_id=3,
        timestamp=2.0 / 30.0,
    )
    _paint_object(hand, (190, 60, 245, 120), hand_color, depth_mm=590)
    rejected = tracker.update(hand)

    assert not rejected.valid
    assert tracker.committed_motion.frame_id == committed.frame_id
    assert tracker.committed_motion.center_uv == committed.center_uv
    assert tracker.committed_motion.velocity_uv_s == committed.velocity_uv_s


def test_low_confidence_flow_cannot_veto_a_strong_component():
    cfg = _cfg()
    cfg["tracked_motion_min_flow_confidence"] = 0.45
    tracker = AdaptiveColorDepthTracker(cfg)
    predicted = _mask((120, 180), (50, 35, 80, 75))
    expanded = _mask((120, 180), (45, 30, 87, 80))
    candidate = SimpleNamespace(mask=expanded, area=float(expanded.sum()))
    flow = SimpleNamespace(mask=predicted, confidence=0.20)

    reason = tracker._tracked_motion_consensus_error(candidate, flow)

    assert reason is None


def test_reliable_klt_bridges_one_invalid_touch_frame_without_global_scan(
    monkeypatch,
):
    cfg = _occlusion_recovery_cfg(confirm_frames=2)
    cfg.update(
        {
            "optical_flow_enabled": True,
            "occlusion_recovery_flow_grace_frames": 2,
            "component_max_motion_px": 24.0,
            "component_max_motion_cap_px": 100.0,
            "component_flow_motion_min_confidence": 0.45,
            "flow_max_displacement_px": 100.0,
        }
    )
    tracker = AdaptiveColorDepthTracker(cfg)
    initial_bbox = (20, 35, 50, 75)
    initial = _frame(initial_bbox, (210, 80, 235), frame_id=1)
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, initial_bbox))

    moved_bbox = (70, 35, 100, 75)
    moved_mask = _mask(initial.depth_raw.shape, moved_bbox)
    flow = _identity_flow(moved_mask, confidence=0.95, tx=50.0)
    flow.message = "KLT synthetic bridge"
    monkeypatch.setattr(
        tracker, "_predict_with_optical_flow", lambda _gray: flow
    )

    occluded = _frame(None, (210, 80, 235), frame_id=2)
    lost = tracker.update(occluded)
    assert not lost.valid
    assert tracker.tracking_status["lost_count"] == 1

    visible = _frame(moved_bbox, (210, 80, 235), frame_id=3)
    recovered = tracker.update(visible)

    assert recovered.valid, recovered.message
    assert "recovered-flow" in recovered.message
    assert tracker.tracking_status["lost_count"] == 0
    assert tracker.tracking_status["temporal_motion_source"] == "recovery-klt"


def _candidate_from_mask(mask, *, depth_median=0.7):
    ys, xs = np.nonzero(mask)
    return SimpleNamespace(
        mask=mask.astype(np.uint8),
        bbox_xyxy=np.asarray(
            [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
            dtype=np.int32,
        ),
        center_uv=(float(xs.mean()), float(ys.mean())),
        area=float(mask.sum()),
        depth_median=float(depth_median),
        confidence=0.95,
        color_score=0.95,
        search_scope="flow",
    )


def _identity_flow(mask, *, confidence=1.0, tx=0.0, ty=0.0):
    ys, xs = np.nonzero(mask)
    return SimpleNamespace(
        mask=mask.astype(np.uint8),
        bbox_xyxy=np.asarray(
            [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
            dtype=np.int32,
        ),
        center_uv=(float(xs.mean()), float(ys.mean())),
        confidence=float(confidence),
        affine_matrix=np.asarray(
            [[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32
        ),
    )


def _reference_full_frame_stabilize(self, frame, candidate, flow):
    """Pre-ROI implementation retained only as an equivalence oracle."""

    current = np.asarray(candidate.mask) > 0
    predicted = np.asarray(flow.mask) > 0
    raw_iou, _, predicted_area_count = _binary_mask_iou(current, predicted)
    self.last_motion_mask_iou = float(raw_iou)
    self.last_temporal_stabilized = False
    self.last_temporal_motion_source = (
        "geometry"
        if str(getattr(flow, "message", "")).startswith("geometry fallback")
        else "klt"
    )

    def bypass():
        self.temporal_mask_score = current.astype(np.float32)
        return candidate

    if not bool(self.cfg.get("temporal_mask_stability_enabled", True)):
        return bypass()
    if float(flow.confidence) < float(
        self.cfg.get("temporal_mask_min_flow_confidence", 0.45)
    ):
        return bypass()
    predicted_area = float(predicted_area_count)
    if predicted_area <= 0.0:
        return bypass()
    area_ratio = float(candidate.area) / predicted_area
    if (
        raw_iou < float(self.cfg.get("temporal_mask_min_raw_iou", 0.65))
        or area_ratio
        < float(self.cfg.get("temporal_mask_min_raw_area_ratio", 0.85))
        or area_ratio
        > float(self.cfg.get("temporal_mask_max_raw_area_ratio", 1.15))
    ):
        return bypass()

    affine = getattr(flow, "affine_matrix", None)
    previous_score = self.temporal_mask_score
    if (
        affine is None
        or previous_score is None
        or previous_score.shape != current.shape
    ):
        return bypass()
    affine = np.asarray(affine, dtype=np.float32)
    if affine.shape != (2, 3) or not np.isfinite(affine).all():
        return bypass()

    h, w = current.shape
    warped_score = cv2.warpAffine(
        previous_score.astype(np.float32),
        affine,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    current_weight = float(
        np.clip(self.cfg.get("temporal_mask_current_weight", 0.45), 0.0, 1.0)
    )
    score = np.clip(
        (1.0 - current_weight) * warped_score
        + current_weight * current.astype(np.float32),
        0.0,
        1.0,
    )
    stable = score >= float(
        np.clip(self.cfg.get("temporal_mask_threshold", 0.50), 0.0, 1.0)
    )

    retained = stable & ~current
    if np.any(retained):
        tolerance = float(
            self.cfg.get(
                "temporal_mask_retention_depth_tolerance_m",
                self.depth_half_width,
            )
        )
        retained_depth = (
            frame.depth_raw[retained].astype(np.float32)
            * float(frame.depth_scale)
        )
        coherent = (
            np.isfinite(retained_depth)
            & (retained_depth > 0.0)
            & (
                np.abs(retained_depth - float(candidate.depth_median))
                <= tolerance
            )
        )
        if not bool(np.all(coherent)):
            rejected = retained.copy()
            rejected[retained] = ~coherent
            stable[rejected] = False
            score[rejected] = 0.0
        retained = stable & ~current

    max_hold_pixels = max(
        int(self.cfg.get("temporal_mask_max_hold_pixels", 48)),
        int(
            np.ceil(
                float(self.cfg.get("temporal_mask_max_hold_fraction", 0.08))
                * predicted_area
            )
        ),
    )
    if int(retained.sum()) > max_hold_pixels:
        return bypass()

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        stable.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return bypass()
    overlaps = np.bincount(
        labels[current].reshape(-1), minlength=count
    ).astype(np.int64)
    overlaps[0] = 0
    label = int(np.argmax(overlaps))
    if label <= 0 or int(overlaps[label]) == 0:
        return bypass()
    stable = labels == label
    area = float(stats[label, cv2.CC_STAT_AREA])
    if area < float(self.cfg.get("min_area", 80)):
        return bypass()

    bx = int(stats[label, cv2.CC_STAT_LEFT])
    by = int(stats[label, cv2.CC_STAT_TOP])
    bw = int(stats[label, cv2.CC_STAT_WIDTH])
    bh = int(stats[label, cv2.CC_STAT_HEIGHT])
    bbox = np.asarray([bx, by, bx + bw, by + bh], dtype=np.int32)
    center = (float(centroids[label, 0]), float(centroids[label, 1]))
    valid_depths = (
        frame.depth_raw[stable].astype(np.float32) * float(frame.depth_scale)
    )
    valid_depths = valid_depths[
        np.isfinite(valid_depths) & (valid_depths > 0.0)
    ]
    depth_median = (
        float(np.median(valid_depths))
        if valid_depths.size
        else float(candidate.depth_median)
    )

    score[~(stable | current)] = 0.0
    self.temporal_mask_score = score.astype(np.float32)
    self.last_temporal_stabilized = not np.array_equal(stable, current)
    return _ComponentCandidate(
        mask=stable.astype(np.uint8),
        bbox_xyxy=bbox,
        center_uv=center,
        area=area,
        depth_median=depth_median,
        confidence=float(candidate.confidence),
        color_score=float(candidate.color_score),
        search_scope=candidate.search_scope,
    )


def _assert_stabilizer_equivalent(reference, optimized, reference_result, result):
    np.testing.assert_array_equal(result.mask, reference_result.mask)
    np.testing.assert_array_equal(result.bbox_xyxy, reference_result.bbox_xyxy)
    assert result.center_uv == reference_result.center_uv
    assert result.area == reference_result.area
    assert result.depth_median == reference_result.depth_median
    assert result.confidence == reference_result.confidence
    assert result.color_score == reference_result.color_score
    assert result.search_scope == reference_result.search_scope
    np.testing.assert_array_equal(
        optimized.temporal_mask_score, reference.temporal_mask_score
    )
    assert optimized.last_motion_mask_iou == reference.last_motion_mask_iou
    assert (
        optimized.last_temporal_stabilized
        == reference.last_temporal_stabilized
    )
    assert (
        optimized.last_temporal_motion_source
        == reference.last_temporal_motion_source
    )


def _assert_recursive_exact(actual, expected):
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif is_dataclass(expected):
        assert type(actual) is type(expected)
        for field in fields(expected):
            _assert_recursive_exact(
                getattr(actual, field.name), getattr(expected, field.name)
            )
    elif isinstance(expected, tuple):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_recursive_exact(actual_item, expected_item)
    else:
        assert actual == expected


def test_temporal_mask_holds_one_frame_edge_dropout_then_accepts_it():
    cfg = _cfg()
    cfg.update({"temporal_mask_stability_enabled": True})
    tracker = AdaptiveColorDepthTracker(cfg)
    bbox = (50, 35, 80, 75)
    frame = _frame(bbox, (210, 80, 235))
    seed = _mask(frame.depth_raw.shape, bbox)
    tracker.initialize(frame, mask=seed)
    missing_edge = seed.copy()
    missing_edge[35:75, 79] = 0
    candidate = _candidate_from_mask(missing_edge)
    flow = _identity_flow(seed)

    first = tracker._stabilize_tracked_candidate(frame, candidate, flow)
    second = tracker._stabilize_tracked_candidate(frame, candidate, flow)

    np.testing.assert_array_equal(first.mask, seed)
    np.testing.assert_array_equal(second.mask, missing_edge)


def test_temporal_mask_requires_two_frames_for_a_new_edge():
    cfg = _cfg()
    cfg.update({"temporal_mask_stability_enabled": True})
    tracker = AdaptiveColorDepthTracker(cfg)
    bbox = (50, 35, 80, 75)
    frame = _frame(bbox, (210, 80, 235))
    seed = _mask(frame.depth_raw.shape, bbox)
    tracker.initialize(frame, mask=seed)
    expanded = seed.copy()
    expanded[35:75, 80] = 1
    candidate = _candidate_from_mask(expanded)
    flow = _identity_flow(seed)

    first = tracker._stabilize_tracked_candidate(frame, candidate, flow)
    second = tracker._stabilize_tracked_candidate(frame, candidate, flow)

    np.testing.assert_array_equal(first.mask, seed)
    np.testing.assert_array_equal(second.mask, expanded)


def test_temporal_mask_follows_affine_motion_without_image_space_lag():
    cfg = _cfg()
    cfg.update({"temporal_mask_stability_enabled": True})
    tracker = AdaptiveColorDepthTracker(cfg)
    initial_bbox = (50, 35, 80, 75)
    moved_bbox = (56, 39, 86, 79)
    initial = _frame(initial_bbox, (210, 80, 235))
    seed = _mask(initial.depth_raw.shape, initial_bbox)
    moved = _mask(initial.depth_raw.shape, moved_bbox)
    tracker.initialize(initial, mask=seed)
    flow = _identity_flow(moved, tx=6.0, ty=4.0)

    result = tracker._stabilize_tracked_candidate(
        _frame(moved_bbox, (210, 80, 235), frame_id=2),
        _candidate_from_mask(moved),
        flow,
    )

    np.testing.assert_array_equal(result.mask, moved)
    assert result.center_uv == pytest.approx((70.5, 58.5), abs=1e-6)


def test_textureless_geometry_fallback_stabilizes_and_follows_motion():
    cfg = _cfg()
    cfg.update(
        {
            "optical_flow_enabled": False,
            "temporal_mask_stability_enabled": True,
            "temporal_mask_geometry_fallback_enabled": True,
        }
    )
    tracker = AdaptiveColorDepthTracker(cfg)
    initial_bbox = (50, 35, 80, 75)
    moved_bbox = (55, 38, 85, 78)
    color = (210, 80, 235)
    initial = _frame(initial_bbox, color)
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, initial_bbox))

    result = tracker.update(_frame(moved_bbox, color, frame_id=2))

    assert result.valid, result.message
    expected = _mask(initial.depth_raw.shape, moved_bbox)
    np.testing.assert_array_equal(result.mask, expected)
    assert tracker.tracking_status["temporal_motion_source"] == "geometry"
    assert tracker.tracking_status["motion_mask_iou"] > 0.98


def test_temporal_mask_does_not_hold_an_edge_at_wrong_current_depth():
    cfg = _cfg()
    cfg.update({"temporal_mask_stability_enabled": True})
    tracker = AdaptiveColorDepthTracker(cfg)
    bbox = (50, 35, 80, 75)
    frame = _frame(bbox, (210, 80, 235))
    seed = _mask(frame.depth_raw.shape, bbox)
    tracker.initialize(frame, mask=seed)
    missing_edge = seed.copy()
    missing_edge[35:75, 79] = 0
    frame.depth_raw[35:75, 79] = 500

    result = tracker._stabilize_tracked_candidate(
        frame,
        _candidate_from_mask(missing_edge),
        _identity_flow(seed),
    )

    np.testing.assert_array_equal(result.mask, missing_edge)


def test_semantic_temporal_mask_retains_edge_independent_of_depth():
    cfg = _cfg()
    cfg.update(
        {
            "temporal_mask_stability_enabled": True,
            "temporal_mask_retention_depth_gate_enabled": False,
        }
    )
    tracker = AdaptiveColorDepthTracker(cfg)
    bbox = (50, 35, 80, 75)
    frame = _frame(bbox, (210, 80, 235))
    seed = _mask(frame.depth_raw.shape, bbox)
    tracker.initialize(frame, mask=seed)
    missing_edge = seed.copy()
    missing_edge[35:75, 79] = 0
    frame.depth_raw[35:75, 79] = 1000

    result = tracker._stabilize_tracked_candidate(
        frame,
        _candidate_from_mask(missing_edge),
        _identity_flow(seed),
    )

    np.testing.assert_array_equal(result.mask, seed)


def test_temporal_mask_low_confidence_flow_bypasses_hold():
    cfg = _cfg()
    cfg.update({"temporal_mask_stability_enabled": True})
    tracker = AdaptiveColorDepthTracker(cfg)
    bbox = (50, 35, 80, 75)
    frame = _frame(bbox, (210, 80, 235))
    seed = _mask(frame.depth_raw.shape, bbox)
    tracker.initialize(frame, mask=seed)
    missing_edge = seed.copy()
    missing_edge[35:75, 79] = 0

    result = tracker._stabilize_tracked_candidate(
        frame,
        _candidate_from_mask(missing_edge),
        _identity_flow(seed, confidence=0.20),
    )

    np.testing.assert_array_equal(result.mask, missing_edge)
    assert not tracker.tracking_status["temporal_mask_stabilized"]


def _stabilizer_pair(frame, seed, candidate, flow, *, cfg=None, score=None):
    reference = AdaptiveColorDepthTracker(_cfg() if cfg is None else cfg.copy())
    optimized = AdaptiveColorDepthTracker(_cfg() if cfg is None else cfg.copy())
    reference.initialize(frame, mask=seed)
    optimized.initialize(frame, mask=seed)
    if score is not None:
        reference.temporal_mask_score = score.copy()
        optimized.temporal_mask_score = score.copy()
    reference._stabilize_tracked_candidate = MethodType(
        _reference_full_frame_stabilize, reference
    )
    reference_result = reference._stabilize_tracked_candidate(
        frame, candidate, flow
    )
    result = optimized._stabilize_tracked_candidate(frame, candidate, flow)
    _assert_stabilizer_equivalent(reference, optimized, reference_result, result)
    return reference_result, result


def test_temporal_support_roi_is_exact_for_affine_touching_image_edges():
    cfg = _cfg()
    cfg.update({"temporal_mask_stability_enabled": True, "min_area": 10})
    frame = _frame(
        (0, 0, 28, 34),
        (210, 80, 235),
        width=180,
        height=120,
    )
    seed = _mask(frame.depth_raw.shape, (0, 0, 28, 34))
    affine = cv2.getRotationMatrix2D((14.0, 17.0), 7.0, 1.03).astype(
        np.float32
    )
    affine[:, 2] += np.asarray([-3.5, -2.25], dtype=np.float32)
    predicted = cv2.warpAffine(
        seed,
        affine,
        (180, 120),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    flow = _identity_flow(predicted)
    flow.affine_matrix = affine
    _stabilizer_pair(
        frame,
        seed,
        _candidate_from_mask(predicted),
        flow,
        cfg=cfg,
    )


def test_temporal_support_roi_is_exact_for_multiple_soft_components():
    cfg = _cfg()
    cfg.update({"temporal_mask_stability_enabled": True, "min_area": 10})
    frame = _frame((50, 35, 80, 75), (210, 80, 235))
    seed = _mask(frame.depth_raw.shape, (50, 35, 80, 75))
    score = seed.astype(np.float32)
    score[8:13, 145:151] = 0.95
    candidate_mask = seed.copy()
    candidate_mask[35:40, 79] = 0
    _stabilizer_pair(
        frame,
        seed,
        _candidate_from_mask(candidate_mask),
        _identity_flow(seed),
        cfg=cfg,
        score=score,
    )


def test_temporal_support_roi_is_exact_for_retained_depth_rejection():
    cfg = _cfg()
    cfg.update({"temporal_mask_stability_enabled": True})
    frame = _frame((50, 35, 80, 75), (210, 80, 235))
    seed = _mask(frame.depth_raw.shape, (50, 35, 80, 75))
    candidate_mask = seed.copy()
    candidate_mask[35:75, 79] = 0
    frame.depth_raw[35:75, 79] = 500
    _stabilizer_pair(
        frame,
        seed,
        _candidate_from_mask(candidate_mask),
        _identity_flow(seed),
        cfg=cfg,
    )


@pytest.mark.parametrize("flow_confidence", [0.449999, 0.45])
def test_temporal_support_roi_preserves_flow_confidence_bypass_boundary(
    flow_confidence,
):
    cfg = _cfg()
    cfg.update({"temporal_mask_stability_enabled": True})
    frame = _frame((50, 35, 80, 75), (210, 80, 235))
    seed = _mask(frame.depth_raw.shape, (50, 35, 80, 75))
    candidate_mask = seed.copy()
    candidate_mask[35:75, 79] = 0
    _stabilizer_pair(
        frame,
        seed,
        _candidate_from_mask(candidate_mask),
        _identity_flow(seed, confidence=flow_confidence),
        cfg=cfg,
    )


def test_temporal_support_roi_preserves_empty_prediction_bypass():
    cfg = _cfg()
    cfg.update({"temporal_mask_stability_enabled": True})
    frame = _frame((50, 35, 80, 75), (210, 80, 235))
    seed = _mask(frame.depth_raw.shape, (50, 35, 80, 75))
    empty = np.zeros_like(seed)
    flow = SimpleNamespace(
        mask=empty,
        confidence=1.0,
        affine_matrix=np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        ),
    )
    _stabilizer_pair(
        frame,
        seed,
        _candidate_from_mask(seed),
        flow,
        cfg=cfg,
    )


def test_appearance_learning_uses_only_eroded_motion_consensus_core():
    cfg = _cfg()
    tracker = AdaptiveColorDepthTracker(cfg)
    predicted = _mask((120, 180), (50, 35, 80, 75))
    with_fringe = _mask((120, 180), (48, 33, 82, 77))
    candidate = SimpleNamespace(
        mask=with_fringe,
        area=float(with_fringe.sum()),
    )
    flow = SimpleNamespace(mask=predicted, confidence=1.0)

    core = tracker._appearance_update_core(candidate, flow)

    assert core is not None
    assert int(core[:, :48].sum()) == 0
    assert int(core[:, 82:].sum()) == 0
    assert int(core.sum()) < int(predicted.sum())


def test_guarded_redetection_rejects_wrong_object_and_preserves_anchor():
    tracker = AdaptiveColorDepthTracker(_cfg())
    target_bbox = (30, 35, 62, 78)
    target_color = (225, 95, 25)
    initial = _frame(target_bbox, target_color)
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, target_bbox))
    anchor_before = tracker.anchor_hs.copy()

    impostor_bbox = (88, 35, 122, 80)
    impostor = _frame(
        None,
        target_color,
        frame_id=2,
        distractor=(impostor_bbox, (20, 30, 230)),
    )
    result = tracker.reinitialize_with_mask(
        impostor, _mask(impostor.depth_raw.shape, impostor_bbox)
    )

    assert not result.valid
    assert result.mask.sum() == 0
    np.testing.assert_allclose(tracker.anchor_hs, anchor_before)
    np.testing.assert_array_equal(tracker.state.bbox_xyxy, target_bbox)


def _textured_same_color_frame(bbox, frame_id):
    frame = _frame(
        bbox,
        (100, 100, 100),
        frame_id=frame_id,
        background_bgr=(100, 100, 100),
    )
    x1, y1, x2, y2 = bbox
    # All values stay inside the same configured V-histogram bin as the
    # background, so color likelihood cannot distinguish the target.  The
    # deterministic microtexture still gives KLT unique, CPU-cheap features.
    rng = np.random.RandomState(1234)
    texture = rng.randint(96, 109, size=(y2 - y1, x2 - x1), dtype=np.uint8)
    frame.color_bgr[y1:y2, x1:x2] = texture[..., None]
    return frame


@pytest.mark.parametrize("flow_feature_crop_enabled", [False, True])
def test_klt_mask_prediction_tracks_object_with_non_distinctive_base_color(
    flow_feature_crop_enabled,
):
    bbox = (35, 35, 70, 78)
    moved_bbox = (51, 40, 86, 83)
    initial = _textured_same_color_frame(bbox, frame_id=1)
    mask = _mask(initial.depth_raw.shape, bbox)

    without_flow_cfg = _cfg()
    without_flow_cfg["optical_flow_enabled"] = False
    without_flow = AdaptiveColorDepthTracker(without_flow_cfg)
    without_flow.initialize(initial, mask=mask)
    no_flow_result = without_flow.update(
        _textured_same_color_frame(moved_bbox, frame_id=2)
    )
    assert not no_flow_result.valid
    assert no_flow_result.mask.sum() == 0

    flow_cfg = _cfg()
    flow_cfg["flow_feature_crop_enabled"] = flow_feature_crop_enabled
    tracker = AdaptiveColorDepthTracker(flow_cfg)
    tracker.initialize(initial, mask=mask)
    result = tracker.update(_textured_same_color_frame(moved_bbox, frame_id=2))

    assert result.valid, result.message
    expected = _mask(result.mask.shape, moved_bbox).astype(bool)
    predicted = result.mask.astype(bool)
    iou = np.logical_and(expected, predicted).sum() / np.logical_or(
        expected, predicted
    ).sum()
    assert iou > 0.82


def test_klt_feature_crop_preserves_native_full_image_prediction():
    bbox = (35, 35, 70, 78)
    moved_bbox = (51, 40, 86, 83)
    initial = _textured_same_color_frame(bbox, frame_id=1)
    current = _textured_same_color_frame(moved_bbox, frame_id=2)
    current_gray = cv2.cvtColor(current.color_bgr, cv2.COLOR_BGR2GRAY)

    predictions = []
    for crop_enabled in (False, True):
        cfg = _cfg()
        cfg["flow_feature_crop_enabled"] = crop_enabled
        tracker = AdaptiveColorDepthTracker(cfg)
        tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, bbox))
        prediction = tracker._predict_with_optical_flow(current_gray)
        assert prediction is not None
        predictions.append(prediction)

    np.testing.assert_array_equal(predictions[1].mask, predictions[0].mask)
    np.testing.assert_allclose(
        predictions[1].affine_matrix,
        predictions[0].affine_matrix,
        rtol=0.0,
        atol=1.0e-6,
    )
    assert predictions[1].confidence == pytest.approx(predictions[0].confidence)


def test_short_occlusion_is_invalid_then_reacquires_without_model_drift():
    tracker = AdaptiveColorDepthTracker(_cfg())
    bbox = (35, 35, 70, 78)
    initial = _textured_same_color_frame(bbox, frame_id=1)
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, bbox))
    anchor_before = tracker.anchor_hs.copy()

    occluded = _frame(
        None,
        (100, 100, 100),
        frame_id=2,
        background_bgr=(100, 100, 100),
    )
    lost = tracker.update(occluded)
    assert not lost.valid
    assert lost.mask.sum() == 0

    moved_bbox = (49, 40, 84, 83)
    # Reliable KLT bridges the one-frame gap immediately at tracker level. The
    # production provider still quarantines this recovered point cloud for its
    # configured consecutive exact-frame observations.
    recovered = tracker.update(
        _textured_same_color_frame(moved_bbox, frame_id=3)
    )
    assert recovered.valid, recovered.message
    steady = tracker.update(
        _textured_same_color_frame(moved_bbox, frame_id=4)
    )
    assert steady.valid, steady.message
    np.testing.assert_allclose(tracker.anchor_hs, anchor_before)
    assert tracker.tracking_status["lost_count"] == 0


@pytest.mark.parametrize(
    "target_color",
    [
        (210, 80, 235),
        (230, 100, 25),  # blue: recovery must not encode a pink hue prior
        (215, 215, 215),
    ],
)
def test_full_occlusion_without_flow_recovers_after_two_consistent_frames(
    target_color,
):
    tracker = AdaptiveColorDepthTracker(_occlusion_recovery_cfg(confirm_frames=2))
    initial_bbox = (55, 34, 87, 78)
    initial = _frame(initial_bbox, target_color)
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, initial_bbox))
    anchor_before = tracker.anchor_hs.copy()

    # More than one empty frame represents complete occlusion and guarantees
    # there is no accepted current mask or flow-supported identity.
    for frame_id in (2, 3):
        hidden = tracker.update(_frame(None, target_color, frame_id=frame_id))
        assert not hidden.valid
        assert hidden.mask.sum() == 0

    moved_bbox = (70, 38, 102, 82)
    pending = tracker.update(_frame(moved_bbox, target_color, frame_id=4))
    assert not pending.valid
    assert pending.mask.sum() == 0
    assert tracker.tracking_status["recovery_pending_count"] == 1

    recovered = tracker.update(_frame(moved_bbox, target_color, frame_id=5))
    assert recovered.valid, recovered.message
    assert "recovered" in recovered.message
    assert tracker.tracking_status["recovery_pending_count"] == 0
    np.testing.assert_allclose(tracker.anchor_hs, anchor_before)
    expected = _mask(recovered.mask.shape, moved_bbox).astype(bool)
    predicted = recovered.mask.astype(bool)
    iou = np.logical_and(expected, predicted).sum() / np.logical_or(
        expected, predicted
    ).sum()
    assert iou > 0.90


def test_occlusion_recovery_requires_consecutive_spatially_consistent_frames():
    tracker = AdaptiveColorDepthTracker(_occlusion_recovery_cfg(confirm_frames=2))
    color = (225, 95, 25)
    initial_bbox = (65, 35, 97, 78)
    initial = _frame(initial_bbox, color)
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, initial_bbox))
    tracker.update(_frame(None, color, frame_id=2))

    first_bbox = (76, 38, 108, 81)
    first = tracker.update(_frame(first_bbox, color, frame_id=3))
    assert not first.valid
    assert first.mask.sum() == 0
    assert tracker.tracking_status["recovery_pending_count"] == 1

    # Still inside the total recovery radius, but inconsistent with the first
    # hypothesis.  It must restart confirmation instead of accumulating votes.
    shifted_bbox = (54, 30, 86, 73)
    shifted = tracker.update(_frame(shifted_bbox, color, frame_id=4))
    assert not shifted.valid
    assert shifted.mask.sum() == 0
    assert tracker.tracking_status["recovery_pending_count"] == 1

    recovered = tracker.update(_frame(shifted_bbox, color, frame_id=5))
    assert recovered.valid, recovered.message
    np.testing.assert_array_equal(recovered.bbox_xyxy, shifted_bbox)


def test_ambiguous_same_identity_candidates_remain_fail_closed():
    tracker = AdaptiveColorDepthTracker(_occlusion_recovery_cfg(confirm_frames=2))
    color = (230, 100, 25)
    initial_bbox = (74, 36, 104, 78)
    initial = _frame(initial_bbox, color)
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, initial_bbox))
    tracker.update(_frame(None, color, frame_id=2))

    left = (52, 36, 82, 78)
    right = (96, 36, 126, 78)
    for frame_id in (3, 4, 5):
        ambiguous = _frame(None, color, frame_id=frame_id)
        _paint_object(ambiguous, left, color)
        _paint_object(ambiguous, right, color)
        result = tracker.update(ambiguous)
        assert not result.valid
        assert result.mask.sum() == 0
        assert "ambiguous" in result.message
        assert tracker.tracking_status["recovery_pending_count"] == 0


def test_nearer_hand_occlusion_never_becomes_recovery_candidate():
    tracker = AdaptiveColorDepthTracker(_occlusion_recovery_cfg(confirm_frames=2))
    target_color = (230, 100, 25)
    hand_color = (90, 145, 205)
    target_bbox = (62, 34, 96, 80)
    initial = _frame(target_bbox, target_color)
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, target_bbox))

    hand_bbox = (54, 27, 106, 88)
    for frame_id in (2, 3, 4, 5):
        occluded = _frame(None, target_color, frame_id=frame_id)
        _paint_object(occluded, hand_bbox, hand_color, depth_mm=590)
        result = tracker.update(occluded)
        assert not result.valid
        assert result.mask.sum() == 0
        assert tracker.tracking_status["recovery_pending_count"] == 0


def test_large_reappearance_jump_cannot_take_over_after_occlusion():
    cfg = _occlusion_recovery_cfg(confirm_frames=2)
    cfg["occlusion_recovery_max_center_displacement_px"] = 35.0
    tracker = AdaptiveColorDepthTracker(cfg)
    color = (210, 80, 235)
    initial_bbox = (20, 36, 50, 78)
    initial = _frame(initial_bbox, color)
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, initial_bbox))
    tracker.update(_frame(None, color, frame_id=2))

    far_bbox = (105, 36, 135, 78)
    for frame_id in (3, 4, 5):
        result = tracker.update(_frame(far_bbox, color, frame_id=frame_id))
        assert not result.valid
        assert result.mask.sum() == 0
        assert tracker.tracking_status["recovery_pending_count"] == 0


def test_snapshot_restore_preserves_pending_recovery_confirmation():
    tracker = AdaptiveColorDepthTracker(_occlusion_recovery_cfg(confirm_frames=2))
    color = (225, 95, 25)
    initial_bbox = (58, 34, 90, 78)
    initial = _frame(initial_bbox, color)
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, initial_bbox))
    tracker.update(_frame(None, color, frame_id=2))

    recovery_bbox = (70, 38, 102, 82)
    pending = tracker.update(_frame(recovery_bbox, color, frame_id=3))
    assert not pending.valid
    assert tracker.tracking_status["recovery_pending_count"] == 1
    snapshot = tracker.snapshot_state()

    # Mutate the pending state with an inconsistent hypothesis, then roll back.
    other_bbox = (48, 30, 80, 74)
    tracker.update(_frame(other_bbox, color, frame_id=4))
    tracker.restore_state(snapshot)
    assert tracker.tracking_status["recovery_pending_count"] == 1

    recovered = tracker.update(_frame(recovery_bbox, color, frame_id=4))
    assert recovered.valid, recovered.message
    np.testing.assert_array_equal(recovered.bbox_xyxy, recovery_bbox)


def test_snapshot_restore_round_trips_temporal_mask_score():
    tracker = AdaptiveColorDepthTracker(_cfg())
    bbox = (58, 34, 90, 78)
    frame = _frame(bbox, (225, 95, 25))
    seed = _mask(frame.depth_raw.shape, bbox)
    tracker.initialize(frame, mask=seed)
    tracker.temporal_mask_score[40:44, 62:66] = 0.37
    snapshot = tracker.snapshot_state()

    tracker.temporal_mask_score[:] = 0.0
    tracker.restore_state(snapshot)

    np.testing.assert_allclose(
        tracker.temporal_mask_score, snapshot.temporal_mask_score
    )


def test_deployment_defaults_auto_recover_without_flow_after_visibility_returns():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "d435_default.yaml"
    cfg = dict(yaml.safe_load(config_path.read_text(encoding="utf-8"))["tracker"])
    tracker = AdaptiveColorDepthTracker(cfg)
    color = (210, 80, 235)
    bbox = (55, 34, 87, 78)
    initial = _frame(bbox, color)
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, bbox))

    hidden = tracker.update(_frame(None, color, frame_id=2))
    first_visible = tracker.update(_frame(bbox, color, frame_id=3))
    second_visible = tracker.update(_frame(bbox, color, frame_id=4))

    assert not hidden.valid and hidden.mask.sum() == 0
    assert not first_visible.valid and first_visible.mask.sum() == 0
    assert "pending 1/2" in first_visible.message
    assert second_visible.valid, second_visible.message
    assert "recovered" in second_visible.message


def test_deployment_defaults_recover_far_reappearance_with_default_flow_enabled():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "d435_default.yaml"
    cfg = dict(yaml.safe_load(config_path.read_text(encoding="utf-8"))["tracker"])
    assert cfg["optical_flow_enabled"] is True
    assert cfg["occlusion_global_search_enabled"] is True
    tracker = AdaptiveColorDepthTracker(cfg)
    color = (210, 80, 235)
    initial_bbox = (20, 34, 52, 78)
    tracker.initialize(
        _frame(initial_bbox, color, width=220),
        mask=_mask((120, 220), initial_bbox),
    )

    # Two fully hidden frames activate full-image category recovery. The new
    # location is far outside the old 42 px / IoU local gates.
    for frame_id in (2, 3):
        hidden = tracker.update(
            _frame(None, color, width=220, frame_id=frame_id)
        )
        assert not hidden.valid

    far_bbox = (150, 38, 182, 82)
    pending = tracker.update(
        _frame(far_bbox, color, width=220, frame_id=4)
    )
    recovered = tracker.update(
        _frame(far_bbox, color, width=220, frame_id=5)
    )

    assert not pending.valid
    assert "pending 1/2" in pending.message
    assert recovered.valid, recovered.message
    assert "recovered-global" in recovered.message
    np.testing.assert_array_equal(recovered.bbox_xyxy, far_bbox)


def test_gradual_occlusion_cannot_poison_recovery_scale_or_cause_chatter():
    """Reproduce the real log: ~3000 px target shrank to a ~1000 px strip."""

    config_path = Path(__file__).resolve().parents[1] / "configs" / "d435_default.yaml"
    cfg = dict(yaml.safe_load(config_path.read_text(encoding="utf-8"))["tracker"])
    tracker = AdaptiveColorDepthTracker(cfg)
    color = (210, 80, 235)
    initial_bbox = (40, 25, 100, 85)  # 3600 px
    tracker.initialize(
        _frame(initial_bbox, color),
        mask=_mask((120, 180), initial_bbox),
    )

    # A hand advances from the right. Each step is small enough that the old
    # last-frame ratio gate used to accept the strip and poison state.area.
    frame_id = 2
    for width in (52, 45, 38, 32, 27, 23, 20, 18):
        partial_bbox = (40, 25, 40 + width, 85)
        result = tracker.update(_frame(partial_bbox, color, frame_id=frame_id))
        frame_id += 1
        if not result.valid:
            break
    assert tracker.state.area < 0.55 * tracker.state.initial_area

    hidden = tracker.update(_frame(None, color, frame_id=frame_id))
    assert not hidden.valid
    frame_id += 1

    # The same partial strip must never become a recovery hypothesis even if
    # it persists for more than the normal two-hit confirmation period.
    partial_bbox = (40, 25, 58, 85)
    for _ in range(4):
        fragment = tracker.update(
            _frame(partial_bbox, color, frame_id=frame_id)
        )
        frame_id += 1
        assert not fragment.valid
        assert fragment.mask.sum() == 0
        assert tracker.tracking_status["recovery_pending_count"] == 0

    # The complete object is evaluated against initial_area, not the poisoned
    # strip. It recovers in two observations and then remains stable.
    pending = tracker.update(_frame(initial_bbox, color, frame_id=frame_id))
    frame_id += 1
    recovered = tracker.update(_frame(initial_bbox, color, frame_id=frame_id))
    frame_id += 1
    assert not pending.valid
    assert recovered.valid, recovered.message
    assert "recovered-global" in recovered.message
    for _ in range(12):
        stable = tracker.update(_frame(initial_bbox, color, frame_id=frame_id))
        frame_id += 1
        assert stable.valid, stable.message


def _paint_fragmented_rectangle(frame, bbox, color, band_width=10, gap=5):
    x1, y1, x2, y2 = bbox
    x = x1
    while x < x2:
        end = min(x2, x + band_width)
        frame.color_bgr[y1:y2, x:end] = color
        x = end + gap
    return frame


def test_global_recovery_merges_nearby_same_depth_fragments_of_one_object():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "d435_default.yaml"
    cfg = dict(yaml.safe_load(config_path.read_text(encoding="utf-8"))["tracker"])
    tracker = AdaptiveColorDepthTracker(cfg)
    color = (210, 80, 235)
    bbox = (50, 25, 110, 85)
    tracker.initialize(
        _frame(bbox, color), mask=_mask((120, 180), bbox)
    )
    tracker.update(_frame(None, color, frame_id=2))
    tracker.update(_frame(None, color, frame_id=3))

    first = _paint_fragmented_rectangle(
        _frame(None, color, frame_id=4), bbox, color
    )
    second = _paint_fragmented_rectangle(
        _frame(None, color, frame_id=5), bbox, color
    )
    pending = tracker.update(first)
    recovered = tracker.update(second)

    assert not pending.valid
    assert recovered.valid, recovered.message
    assert "recovered-global" in recovered.message
    # The published mask is the depth-supported object hull, not one stripe.
    assert int(recovered.mask.sum()) >= int(0.85 * 60 * 60)


def test_global_fragment_merge_keeps_two_distant_instances_ambiguous():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "d435_default.yaml"
    cfg = dict(yaml.safe_load(config_path.read_text(encoding="utf-8"))["tracker"])
    tracker = AdaptiveColorDepthTracker(cfg)
    color = (230, 100, 25)
    initial_bbox = (60, 25, 120, 85)
    tracker.initialize(
        _frame(initial_bbox, color),
        mask=_mask((120, 180), initial_bbox),
    )
    tracker.update(_frame(None, color, frame_id=2))
    tracker.update(_frame(None, color, frame_id=3))

    left = (3, 25, 63, 85)
    right = (117, 25, 177, 85)
    for frame_id in (4, 5, 6):
        frame = _frame(None, color, frame_id=frame_id)
        _paint_fragmented_rectangle(frame, left, color)
        _paint_fragmented_rectangle(frame, right, color)
        result = tracker.update(frame)
        assert not result.valid
        assert result.mask.sum() == 0
        assert "ambiguous" in result.message
        assert tracker.tracking_status["recovery_pending_count"] == 0


def test_global_fragment_merge_does_not_bridge_a_closer_same_color_hand():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "d435_default.yaml"
    cfg = dict(yaml.safe_load(config_path.read_text(encoding="utf-8"))["tracker"])
    tracker = AdaptiveColorDepthTracker(cfg)
    color = (210, 80, 235)
    bbox = (50, 25, 110, 85)
    tracker.initialize(
        _frame(bbox, color), mask=_mask((120, 180), bbox)
    )
    tracker.update(_frame(None, color, frame_id=2))
    tracker.update(_frame(None, color, frame_id=3))

    for frame_id in (4, 5, 6):
        frame = _frame(None, color, frame_id=frame_id)
        _paint_fragmented_rectangle(frame, bbox, color)
        # Same colour and spatially close enough to enter the dilation group,
        # but 11 cm closer than the target depth.
        _paint_object(frame, (112, 30, 126, 80), color, depth_mm=590)
        result = tracker.update(frame)
        assert not result.valid
        assert result.mask.sum() == 0
        assert tracker.tracking_status["recovery_pending_count"] == 0


def test_recovery_confirmation_tolerates_one_depth_dropout_frame():
    cfg = _occlusion_recovery_cfg(confirm_frames=2)
    cfg["occlusion_recovery_confirm_max_frame_gap"] = 3
    tracker = AdaptiveColorDepthTracker(cfg)
    color = (230, 100, 25)
    initial_bbox = (55, 34, 87, 78)
    tracker.initialize(
        _frame(initial_bbox, color),
        mask=_mask((120, 180), initial_bbox),
    )
    tracker.update(_frame(None, color, frame_id=2))

    recovery_bbox = (66, 38, 98, 82)
    pending = tracker.update(_frame(recovery_bbox, color, frame_id=3))
    assert not pending.valid
    assert tracker.tracking_status["recovery_pending_count"] == 1

    depth_dropout = _frame(recovery_bbox, color, frame_id=4)
    depth_dropout.depth_raw[38:82, 66:98] = 0
    depth_dropout.depth_m[38:82, 66:98] = 0.0
    missing = tracker.update(depth_dropout)
    assert not missing.valid
    assert tracker.tracking_status["recovery_pending_count"] == 1

    recovered = tracker.update(_frame(recovery_bbox, color, frame_id=5))
    assert recovered.valid, recovered.message


def test_lost_recovery_does_not_require_a_stale_klt_result(monkeypatch):
    config_path = Path(__file__).resolve().parents[1] / "configs" / "d435_default.yaml"
    cfg = dict(yaml.safe_load(config_path.read_text(encoding="utf-8"))["tracker"])
    tracker = AdaptiveColorDepthTracker(cfg)
    color = (210, 80, 235)
    bbox = (55, 34, 87, 78)
    tracker.initialize(
        _frame(bbox, color), mask=_mask((120, 180), bbox)
    )
    tracker.update(_frame(None, color, frame_id=2))

    def stale_flow_must_not_run(_current_gray):
        raise AssertionError("LOST recovery incorrectly trusted frozen KLT")

    monkeypatch.setattr(tracker, "_predict_with_optical_flow", stale_flow_must_not_run)
    pending = tracker.update(_frame(bbox, color, frame_id=3))
    recovered = tracker.update(_frame(bbox, color, frame_id=4))

    assert not pending.valid
    assert recovered.valid, recovered.message


def test_lost_mode_recovery_scan_has_headroom_over_20_hz():
    cfg = _occlusion_recovery_cfg(confirm_frames=2)
    cfg["occlusion_global_search_enabled"] = True
    cfg["occlusion_global_search_after_frames"] = 2
    cfg["min_area"] = 200
    tracker = AdaptiveColorDepthTracker(cfg)
    color = (210, 80, 235)
    bbox = (380, 190, 450, 280)
    initial = _frame(bbox, color, width=848, height=480)
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, bbox))

    durations = []
    for frame_id in range(2, 32):
        hidden = _frame(
            None,
            color,
            width=848,
            height=480,
            frame_id=frame_id,
        )
        started = time.perf_counter()
        result = tracker.update(hidden)
        durations.append(time.perf_counter() - started)
        assert not result.valid
        assert result.mask.sum() == 0

    assert float(np.mean(durations)) < 0.025
    assert float(np.percentile(durations, 95)) < 0.050


def test_lost_global_scan_stays_fast_with_many_coloured_scene_fragments():
    """Full-frame LOST search must not scale as components x image pixels."""

    cfg = _occlusion_recovery_cfg(confirm_frames=2)
    cfg.update(
        {
            "occlusion_global_search_enabled": True,
            "occlusion_global_search_after_frames": 1,
            "occlusion_fragment_merge_enabled": True,
            "min_area": 200,
        }
    )
    tracker = AdaptiveColorDepthTracker(cfg)
    color = (210, 80, 235)
    bbox = (380, 190, 450, 280)
    initial = _frame(bbox, color, width=848, height=480)
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, bbox))
    tracker.update(_frame(None, color, width=848, height=480, frame_id=2))

    durations = []
    for frame_id in range(3, 13):
        noisy = _frame(
            None,
            color,
            width=848,
            height=480,
            frame_id=frame_id,
        )
        # Hundreds of separated 5x5 highlights survive the morphology pass.
        # None is large enough to be the target, but the old merge loop ran a
        # full-frame connected-components pass once for every highlight.
        for y in range(4, 476, 16):
            for x in range(4, 844, 16):
                noisy.color_bgr[y : y + 5, x : x + 5] = color
        started = time.perf_counter()
        result = tracker.update(noisy)
        durations.append(time.perf_counter() - started)
        assert not result.valid
        assert result.mask.sum() == 0

    # Even this deliberately adversarial full-frame scene leaves enough of a
    # 50 ms frame budget for camera capture and point-cloud extraction.
    assert float(np.mean(durations)) < 0.035
    assert float(np.percentile(durations, 95)) < 0.050


def test_identical_neighbour_cannot_take_over_after_large_one_frame_jump():
    tracker = AdaptiveColorDepthTracker(_cfg())
    target_color = (210, 80, 235)
    initial_bbox = (35, 38, 65, 78)
    initial = _frame(initial_bbox, target_color)
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, initial_bbox))

    # The original instance vanishes and an indistinguishable same-depth
    # object appears next to it.  Identity is unobservable, so fail closed.
    impostor_bbox = (65, 38, 95, 78)
    replacement = _frame(
        None,
        target_color,
        frame_id=2,
        distractor=(impostor_bbox, target_color),
    )
    result = tracker.update(replacement)

    assert not result.valid
    assert result.mask.sum() == 0
    assert "motion" in result.message


def test_guarded_redetection_rejects_far_or_different_depth_same_color_object():
    target_color = (210, 80, 235)
    initial_bbox = (20, 35, 50, 75)
    initial = _frame(initial_bbox, target_color)

    far_tracker = AdaptiveColorDepthTracker(_cfg())
    far_tracker.initialize(
        initial, mask=_mask(initial.depth_raw.shape, initial_bbox)
    )
    far_bbox = (140, 35, 170, 75)
    far_frame = _frame(far_bbox, target_color, frame_id=2)
    far_result = far_tracker.reinitialize_with_mask(
        far_frame, _mask(far_frame.depth_raw.shape, far_bbox)
    )
    assert not far_result.valid
    assert "center jump" in far_result.message

    depth_cfg = _cfg()
    depth_cfg["reinit_max_center_jump_px"] = 0.0
    depth_tracker = AdaptiveColorDepthTracker(depth_cfg)
    depth_tracker.initialize(
        initial, mask=_mask(initial.depth_raw.shape, initial_bbox)
    )
    deep_frame = _frame(
        initial_bbox,
        target_color,
        frame_id=2,
        target_depth_mm=1000,
    )
    depth_result = depth_tracker.reinitialize_with_mask(
        deep_frame, _mask(deep_frame.depth_raw.shape, initial_bbox)
    )
    assert not depth_result.valid
    assert "depth jump" in depth_result.message


def test_category_redetection_can_explicitly_relocate_across_camera_depth():
    cfg = _cfg()
    cfg["reinit_max_center_jump_px"] = 0.0
    tracker = AdaptiveColorDepthTracker(cfg)
    target_color = (210, 80, 235)
    bbox = (20, 35, 50, 75)
    initial = _frame(bbox, target_color)
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, bbox))

    relocated = _frame(
        bbox,
        target_color,
        frame_id=2,
        target_depth_mm=1100,
    )
    result = tracker.reinitialize_with_mask(
        relocated,
        _mask(relocated.depth_raw.shape, bbox),
        max_depth_jump_m=0.0,
    )

    assert result.valid, result.message
    assert tracker.state.depth_median == pytest.approx(1.1)


def test_tracker_rejects_gradual_shrink_below_initial_identity_size():
    cfg = _cfg()
    cfg.update(
        {
            "optical_flow_enabled": False,
            "min_area_vs_initial": 0.25,
            "min_bbox_area_vs_initial": 0.25,
        }
    )
    tracker = AdaptiveColorDepthTracker(cfg)
    color = (210, 80, 235)
    boxes = [
        (60, 35, 100, 75),
        (66, 41, 94, 69),
        (70, 45, 90, 65),
        (73, 48, 87, 62),
    ]
    initial = _frame(boxes[0], color)
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, boxes[0]))

    for frame_id, bbox in enumerate(boxes[1:3], start=2):
        result = tracker.update(_frame(bbox, color, frame_id=frame_id))
        assert result.valid, result.message

    rejected = tracker.update(_frame(boxes[3], color, frame_id=4))
    assert not rejected.valid
    assert rejected.mask.sum() == 0
    assert "area vs init" in rejected.message or "bbox vs init" in rejected.message


def test_real_v94_rgbd_replay_matches_full_frame_reference_exactly():
    capture = (
        Path(__file__).resolve().parents[2]
        / "dexgrasp"
        / "runs"
        / "v94_raw_rgbd_mask_sam_off_20260722_01.npz"
    )
    if not capture.is_file():
        pytest.skip("local camera-only V94 replay capture is unavailable")

    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "d435_default.yaml"
    )
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))["tracker"]
    with np.load(capture, allow_pickle=False) as archive:
        rgb = archive["rgb"]
        depth = archive["depth_raw"]
        masks = archive["object_mask"]
        scales = archive["frame_depth_scale_m_per_unit"]
        timestamps = archive["camera_timestamp_s"]
        frame_ids = archive["camera_frame_id"]
        camera_k = archive["camera_K"]

    intrinsics = CameraIntrinsics(
        width=int(rgb.shape[2]),
        height=int(rgb.shape[1]),
        fx=float(camera_k[0, 0]),
        fy=float(camera_k[1, 1]),
        ppx=float(camera_k[0, 2]),
        ppy=float(camera_k[1, 2]),
    )
    frames = [
        RGBDFrame(
            color_bgr=rgb[index],
            depth_raw=depth[index],
            depth_scale=float(scales[index]),
            intrinsics=intrinsics,
            timestamp=float(timestamps[index]),
            frame_id=int(frame_ids[index]),
        )
        for index in range(len(rgb))
    ]

    reference = AdaptiveColorDepthTracker(cfg.copy())
    optimized = AdaptiveColorDepthTracker(cfg.copy())
    reference.initialize(frames[0], mask=masks[0])
    optimized.initialize(frames[0], mask=masks[0])
    reference._stabilize_tracked_candidate = MethodType(
        _reference_full_frame_stabilize, reference
    )

    for frame in frames[1:]:
        reference_result = reference.update(frame)
        result = optimized.update(frame)
        assert result.valid == reference_result.valid
        assert result.score == reference_result.score
        assert result.message == reference_result.message
        assert result.source == reference_result.source
        np.testing.assert_array_equal(result.mask, reference_result.mask)
        np.testing.assert_array_equal(
            result.bbox_xyxy, reference_result.bbox_xyxy
        )
        _assert_recursive_exact(
            optimized.snapshot_state(), reference.snapshot_state()
        )


def test_high_rate_tracker_stage_has_large_margin_over_20_hz():
    """Offline tracker-stage budget; camera/visualizer are intentionally absent."""

    cfg = _cfg()
    cfg["min_area"] = 200
    width, height = 848, 480
    bbox = (380, 190, 450, 280)
    tracker = AdaptiveColorDepthTracker(cfg)
    initial = _frame(
        bbox,
        (210, 80, 235),
        width=width,
        height=height,
    )
    tracker.initialize(initial, mask=_mask(initial.depth_raw.shape, bbox))

    durations = []
    for frame_id in range(2, 32):
        dx = min(10, frame_id // 4)
        moved = (bbox[0] + dx, bbox[1], bbox[2] + dx, bbox[3])
        frame = _frame(
            moved,
            (210, 80, 235),
            width=width,
            height=height,
            frame_id=frame_id,
        )
        started = time.perf_counter()
        result = tracker.update(frame)
        durations.append(time.perf_counter() - started)
        assert result.valid, result.message

    # 50 ms is the complete 20 Hz frame budget.  This test covers only the
    # segmentation stage and therefore requires comfortable headroom.
    assert float(np.mean(durations)) < 0.025
    assert float(np.percentile(durations, 95)) < 0.050


def test_provider_accepts_external_prompt_mask_and_resets_target_state():
    bbox = (30, 28, 62, 72)
    frame = _frame(bbox, (220, 90, 230))
    prompt_mask = _mask(frame.depth_raw.shape, bbox)

    class FakeTracker:
        def __init__(self):
            self.received = None

        def initialize(self, _frame, mask=None, bbox_xyxy=None):
            self.received = (mask.copy(), bbox_xyxy.copy())
            return MaskResult(
                mask=mask.copy(),
                bbox_xyxy=bbox_xyxy.copy(),
                score=1.0,
                valid=True,
            )

    class FakeHistory:
        def __init__(self):
            self.reset_called = False

        def reset(self):
            self.reset_called = True

    provider = ObjectPCDProvider.__new__(ObjectPCDProvider)
    provider.tracker = FakeTracker()
    provider.history = FakeHistory()
    provider.last_mask_result = None
    provider.last_packet = object()
    provider.recovery_template = np.ones((2, 2), dtype=np.uint8)
    provider.recovery_template_mask = np.ones((2, 2), dtype=np.uint8)
    provider.recovery_template_bbox = np.asarray([1, 1, 3, 3])
    provider.identity_hist = np.ones((2, 2), dtype=np.float32)
    provider.identity_depth_median = 0.7
    provider.identity_center_uv = (1.0, 2.0)
    provider.identity_initial_area = 10.0
    provider._log_mask_result = lambda *args, **kwargs: None
    provider._save_debug_mask = lambda *args, **kwargs: None
    observed = []
    provider._update_recovery_template = (
        lambda _frame, _result, source: observed.append(source)
    )

    assert provider.initialize_from_mask(
        frame, prompt_mask, source="grounding_sam2"
    )
    received_mask, received_bbox = provider.tracker.received
    np.testing.assert_array_equal(received_mask, prompt_mask)
    np.testing.assert_array_equal(received_bbox, bbox)
    assert provider.history.reset_called
    assert provider.last_packet is None
    assert provider.identity_hist is None
    assert provider.recovery_template is None
    assert observed == ["grounding_sam2_tracker_init"]
