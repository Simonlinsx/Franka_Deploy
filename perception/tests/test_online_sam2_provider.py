from __future__ import annotations

import copy
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import cv2

from dynamic_pcd.config import load_config
from dynamic_pcd.apps.realtime_masked_pcd import validate_online_sam2_config
from dynamic_pcd.pointcloud.extractor import ExtractedObjectPCD
from dynamic_pcd.provider.object_pcd_provider import (
    ObjectPCDProvider,
    _BoundaryExitContinuity,
    _validated_runtime_frame_timeout_ms,
)
from dynamic_pcd.segmentation.adaptive_color_depth_tracker import (
    AdaptiveColorDepthTracker,
)
from dynamic_pcd.segmentation.sam2_video_protocol import SAM2VideoResult
from dynamic_pcd.tracking.history import PCDHistoryBuffer
from dynamic_pcd.types import CameraIntrinsics, MaskResult, RGBDFrame
from dynamic_pcd.utils.geometry import bbox_from_mask, erode_mask, largest_component

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOYED_CONFIG = REPO_ROOT / "configs" / "d435_default.yaml"


def _frame(frame_id: int) -> RGBDFrame:
    color = np.zeros((24, 32, 3), dtype=np.uint8)
    color[..., 0] = frame_id
    return RGBDFrame(
        color_bgr=color,
        depth_raw=np.full((24, 32), 700, dtype=np.uint16),
        depth_scale=0.001,
        intrinsics=CameraIntrinsics(
            width=32,
            height=24,
            fx=50.0,
            fy=50.0,
            ppx=16.0,
            ppy=12.0,
        ),
        timestamp=float(frame_id) / 30.0,
        frame_id=frame_id,
    )


def _mask(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    value = np.zeros((24, 32), dtype=np.uint8)
    value[y1:y2, x1:x2] = 1
    return value


def test_guarded_raw_component_cleanup_removes_only_subthreshold_speckles():
    raw = np.zeros((80, 100), dtype=np.uint8)
    raw[20:45, 30:55] = 1  # 625-pixel target component.
    raw[48:53, 60:65] = 1  # Separate 25-pixel visible target piece.
    raw[2, 3] = 1
    raw[70:72, 90:92] = 1

    clean = ObjectPCDProvider._remove_subthreshold_raw_components(
        raw, min_area=20
    )

    expected = np.zeros_like(raw)
    expected[20:45, 30:55] = 1
    expected[48:53, 60:65] = 1
    assert np.array_equal(clean, expected)
    assert clean.dtype == np.uint8
    assert clean.flags.c_contiguous
    assert cv2.connectedComponents(clean, connectivity=8)[0] - 1 == 2


def test_policy_semantic_side_channel_does_not_change_guarded_publication():
    provider = object.__new__(ObjectPCDProvider)
    semantic = _mask(5, 6, 12, 14)
    semantic.setflags(write=False)
    provider._current_policy_semantic_mask = semantic
    provider._current_policy_semantic_source = (
        "online_sam2_exact_current_non_authoritative"
    )
    public = MaskResult(
        mask=np.zeros_like(semantic),
        bbox_xyxy=np.zeros(4, dtype=np.int32),
        score=0.0,
        valid=False,
        message="guarded depth gate rejected current publication",
        source="online_sam2_video",
    )

    attached = provider._attach_policy_semantic_mask(public)

    assert attached is public
    assert attached.valid is False
    assert int(np.count_nonzero(attached.mask)) == 0
    assert attached.policy_semantic_valid is True
    assert attached.policy_semantic_source.endswith("non_authoritative")
    np.testing.assert_array_equal(attached.policy_semantic_mask, semantic)
    attached.policy_semantic_mask[:] = 0
    assert int(np.count_nonzero(semantic)) > 0


def test_policy_semantic_side_channel_is_empty_without_exact_current_evidence():
    provider = object.__new__(ObjectPCDProvider)
    provider._current_policy_semantic_mask = None
    public = _mask_result(_mask(5, 6, 12, 14), valid=True)
    attached = provider._attach_policy_semantic_mask(public)
    assert attached.valid is True
    assert attached.policy_semantic_mask is None
    assert attached.policy_semantic_valid is False
    assert attached.policy_semantic_source == ""


def test_guarded_raw_component_cleanup_is_empty_when_no_component_can_publish():
    raw = np.zeros((40, 50), dtype=np.uint8)
    raw[3:6, 4:7] = 1
    raw[20:23, 30:34] = 1

    clean = ObjectPCDProvider._remove_subthreshold_raw_components(
        raw, min_area=20
    )

    assert not np.any(clean)


def test_guarded_raw_component_cleanup_preserves_all_eligible_bytes():
    raw = np.zeros((40, 50), dtype=np.uint8)
    raw[5:15, 7:17] = 1
    raw[20:25, 25:30] = 1

    clean = ObjectPCDProvider._remove_subthreshold_raw_components(
        raw, min_area=20
    )

    assert np.array_equal(clean, raw)


def test_integer_translation_overlap_counts_matches_warp_affine_randomized():
    """All integer signs/clipping cases remain bit/count exact to OpenCV."""

    rng = np.random.default_rng(20260811)
    shapes = ((1, 1), (3, 7), (24, 32), (59, 83))
    explicit_shifts = (
        (-100, -100),
        (-12, 0),
        (0, -12),
        (-3, 5),
        (0, 0),
        (4, -7),
        (12, 0),
        (0, 12),
        (100, 100),
    )
    for height, width in shapes:
        for trial in range(32):
            reference = (rng.random((height, width)) < 0.22).astype(np.uint8)
            candidate = (rng.random((height, width)) < 0.28).astype(np.uint8)
            shifts = explicit_shifts + (
                (
                    int(rng.integers(-2 * width - 1, 2 * width + 2)),
                    int(rng.integers(-2 * height - 1, 2 * height + 2)),
                ),
            )
            for dx, dy in shifts:
                warped = cv2.warpAffine(
                    reference,
                    np.asarray(
                        [[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]],
                        dtype=np.float32,
                    ),
                    (width, height),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                expected = (
                    int(np.count_nonzero(warped)),
                    int(np.count_nonzero((warped > 0) & (candidate > 0))),
                )
                actual = ObjectPCDProvider._integer_translation_overlap_counts(
                    reference,
                    candidate,
                    np.asarray([dx, dy], dtype=np.int32),
                )
                assert actual == expected, (
                    (height, width),
                    trial,
                    (dx, dy),
                    actual,
                    expected,
                )


def _legacy_best_broad_visible_anchor_alignment(
    provider,
    anchor_mask,
    candidate_mask,
    strict_mask,
    *,
    radius_px,
):
    """Pre-correlation exact search retained as a regression oracle."""

    anchor = (np.asarray(anchor_mask) > 0).astype(np.uint8)
    candidate = (np.asarray(candidate_mask) > 0).astype(np.uint8)
    strict = (np.asarray(strict_mask) > 0).astype(np.uint8)
    if anchor.shape != candidate.shape or candidate.shape != strict.shape:
        return None, None
    anchor_bbox = bbox_from_mask(anchor, min_area=1)
    anchor_center = provider._binary_mask_centroid(anchor, anchor_bbox)
    strict_center = provider._binary_mask_centroid(strict)
    anchor_pixels = int(np.count_nonzero(anchor))
    if (
        anchor_bbox is None
        or anchor_center is None
        or strict_center is None
        or anchor_pixels <= 0
    ):
        return None, None
    base_shift = np.rint(strict_center - anchor_center).astype(np.int64)
    best_shift = None
    best_overlap = -1
    best_residual = float("inf")
    radius = max(0, int(radius_px))
    for dy in range(int(base_shift[1]) - radius, int(base_shift[1]) + radius + 1):
        for dx in range(
            int(base_shift[0]) - radius,
            int(base_shift[0]) + radius + 1,
        ):
            shift = np.asarray([dx, dy], dtype=np.int64)
            translated_pixels, overlap = (
                provider._integer_translation_overlap_counts(
                    anchor,
                    candidate,
                    shift,
                    anchor_bbox,
                )
            )
            if translated_pixels != anchor_pixels:
                continue
            residual = float(np.sum((shift - base_shift) ** 2))
            if overlap > best_overlap or (
                overlap == best_overlap and residual < best_residual
            ):
                best_overlap = int(overlap)
                best_residual = residual
                best_shift = shift.copy()
    if best_shift is None:
        return None, None
    return provider._translate_binary_mask_integer(anchor, best_shift), best_shift


def test_broad_visible_anchor_native_alignment_matches_exact_search_property():
    """Random interior/edge/tie lattices match the legacy integer oracle."""

    provider = object.__new__(ObjectPCDProvider)
    rng = np.random.default_rng(20260811)
    shapes_and_trials = (((24, 32), 80), ((79, 113), 80), ((480, 848), 4))
    for (height, width), trials in shapes_and_trials:
        for _trial in range(trials):
            anchor = np.zeros((height, width), dtype=np.uint8)
            anchor_width = int(rng.integers(1, min(23, width) + 1))
            anchor_height = int(rng.integers(1, min(19, height) + 1))
            x1 = int(rng.integers(0, width - anchor_width + 1))
            y1 = int(rng.integers(0, height - anchor_height + 1))
            patch = (
                rng.random((anchor_height, anchor_width)) > 0.35
            ).astype(np.uint8)
            if not np.any(patch):
                patch[0, 0] = 1
            anchor[
                y1 : y1 + anchor_height,
                x1 : x1 + anchor_width,
            ] = patch
            candidate = (rng.random((height, width)) < 0.025).astype(np.uint8)
            true_shift = np.asarray(
                [int(rng.integers(-9, 10)), int(rng.integers(-9, 10))],
                dtype=np.int64,
            )
            translated = provider._translate_binary_mask_integer(
                anchor, true_shift
            )
            candidate = np.maximum(
                candidate,
                translated * (rng.random((height, width)) > 0.18),
            )
            strict = translated * (rng.random((height, width)) > 0.45)
            if not np.any(strict):
                strict = translated.copy()
            if not np.any(strict):
                strict[height // 2, width // 2] = 1
            radius = int(rng.integers(0, 13))

            expected_mask, expected_shift = (
                _legacy_best_broad_visible_anchor_alignment(
                    provider,
                    anchor,
                    candidate,
                    strict,
                    radius_px=radius,
                )
            )
            actual_mask, actual_shift, _reason = (
                provider._best_broad_visible_anchor_alignment(
                    anchor,
                    candidate,
                    strict,
                    radius_px=radius,
                )
            )
            assert (expected_shift is None) == (actual_shift is None)
            if expected_shift is not None:
                np.testing.assert_array_equal(actual_shift, expected_shift)
                np.testing.assert_array_equal(actual_mask, expected_mask)


def test_broad_visible_anchor_alignment_preserves_boundary_and_tie_order():
    """Unclipped bounds and legacy dy/dx tie ordering stay deterministic."""

    provider = object.__new__(ObjectPCDProvider)
    anchor = np.zeros((24, 32), dtype=np.uint8)
    anchor[12, 16] = 1
    strict = anchor.copy()
    candidate = np.zeros_like(anchor)
    candidate[12, 15] = 1
    candidate[12, 17] = 1
    aligned, shift, reason = provider._best_broad_visible_anchor_alignment(
        anchor, candidate, strict, radius_px=1
    )
    assert aligned is not None, reason
    np.testing.assert_array_equal(shift, np.asarray([-1, 0], dtype=np.int64))
    assert aligned[12, 15] == 1

    # With radius zero the centroid-requested placement clips a wide anchor;
    # no partial anchor is allowed to become identity/scale evidence.
    boundary_anchor = np.zeros_like(anchor)
    boundary_anchor[8:12, 10:20] = 1
    boundary_strict = np.zeros_like(anchor)
    boundary_strict[9, 0] = 1
    rejected, rejected_shift, rejected_reason = (
        provider._best_broad_visible_anchor_alignment(
            boundary_anchor,
            boundary_strict,
            boundary_strict,
            radius_px=0,
        )
    )
    assert rejected is None and rejected_shift is None
    assert "no unclipped bounded alignment" in rejected_reason


@pytest.mark.parametrize(
    ("correlations", "reason_fragment"),
    (
        (np.full((3, 3), np.nan, dtype=np.float32), "non-finite"),
        (np.full((3, 3), np.inf, dtype=np.float32), "non-finite"),
        (np.full((3, 3), -1.0, dtype=np.float32), "out of range"),
        (np.full((3, 3), 5.0, dtype=np.float32), "out of range"),
        (np.zeros((2, 3), dtype=np.float32), "shape changed"),
    ),
)
def test_broad_visible_anchor_alignment_fails_closed_on_bad_correlation(
    monkeypatch,
    correlations,
    reason_fragment,
):
    """Malformed native correlation can never mint an alignment proof."""

    provider = object.__new__(ObjectPCDProvider)
    anchor = np.zeros((24, 32), dtype=np.uint8)
    anchor[10:12, 15:17] = 1  # four pixels: 5 is above the valid range.
    candidate = anchor.copy()
    strict = anchor.copy()
    monkeypatch.setattr(
        cv2,
        "matchTemplate",
        lambda *_args, **_kwargs: correlations.copy(),
    )
    aligned, shift, reason = provider._best_broad_visible_anchor_alignment(
        anchor,
        candidate,
        strict,
        radius_px=1,
    )
    assert aligned is None and shift is None
    assert reason_fragment in reason


def test_broad_visible_anchor_alignment_uses_exact_search_above_envelope(
    monkeypatch,
):
    """Larger-than-848x480 images bypass float correlation entirely."""

    provider = object.__new__(ObjectPCDProvider)
    anchor = np.zeros((481, 848), dtype=np.uint8)
    anchor[200:207, 300:309] = 1
    candidate = provider._translate_binary_mask_integer(
        anchor, np.asarray([2, -1], dtype=np.int64)
    )
    strict = candidate.copy()

    def _forbidden_native_path(*_args, **_kwargs):
        raise AssertionError("native correlation exceeded its reviewed envelope")

    monkeypatch.setattr(cv2, "matchTemplate", _forbidden_native_path)
    aligned, shift, reason = provider._best_broad_visible_anchor_alignment(
        anchor,
        candidate,
        strict,
        radius_px=3,
    )
    assert aligned is not None, reason
    np.testing.assert_array_equal(shift, np.asarray([2, -1], dtype=np.int64))
    np.testing.assert_array_equal(aligned, candidate)


def _boundary_seed_fine_frame(frame_id, *, target_bbox=None, hand_bbox=None):
    color = np.full((120, 180, 3), 45, dtype=np.uint8)
    if target_bbox is not None:
        x1, y1, x2, y2 = target_bbox
        color[y1:y2, x1:x2] = (20, 20, 205)
    if hand_bbox is not None:
        x1, y1, x2, y2 = hand_bbox
        color[y1:y2, x1:x2] = (75, 145, 210)
    return RGBDFrame(
        color_bgr=color,
        depth_raw=np.full((120, 180), 900, dtype=np.uint16),
        depth_scale=0.001,
        intrinsics=CameraIntrinsics(
            width=180,
            height=120,
            fx=200.0,
            fy=200.0,
            ppx=89.5,
            ppy=59.5,
        ),
        timestamp=float(frame_id) / 30.0,
        frame_id=frame_id,
    )


def test_boundary_seed_fine_core_selects_target_from_hand_union(monkeypatch):
    """A clipped red seed can select red target pixels, never the skin union."""

    provider, _tracker = _guarded_primary_provider(
        monkeypatch,
        [],
        {},
        _large_mask((40, 30, 80, 70)),
    )
    try:
        seed_frame = _boundary_seed_fine_frame(
            20, target_bbox=(0, 35, 26, 75)
        )
        seed = np.zeros((120, 180), dtype=np.uint8)
        seed[35:75, 0:26] = 1
        model = provider._build_boundary_seed_fine_appearance_model(
            seed_frame, seed
        )
        assert model is not None
        provider._boundary_seed_fine_appearance_model = model
        provider._guarded_v2_bootstrap_phase = "expired"
        provider._guarded_v2_bootstrap_expiry_kind = "boundary_seed"

        current = _boundary_seed_fine_frame(
            101,
            target_bbox=(80, 35, 106, 75),
            hand_bbox=(55, 20, 82, 90),
        )
        raw = np.zeros((120, 180), dtype=np.uint8)
        raw[20:90, 55:82] = 1
        raw[35:75, 80:106] = 1
        raw_evidence = provider._defer_full_target_geometry_commit(current, raw)
        assert raw_evidence is not None
        candidate, reason = provider._boundary_seed_fine_visible_core(
            current, raw_evidence
        )
        assert candidate is not None, reason
        # The hand overwrites the left two target columns in the RGB image;
        # publish only the still-visible target part, never hallucinated
        # pixels behind the occluder.
        assert int(np.count_nonzero(candidate.mask)) == 24 * 40
        np.testing.assert_array_equal(
            candidate.geometry.bbox_xyxy,
            np.asarray([82, 35, 106, 75], dtype=np.int32),
        )
        proof = provider._boundary_seed_fine_visible_core_frame_proof
        assert proof is not None
        anchor_fraction = np.asarray(
            [proof.anchor_fraction_bits], dtype=np.uint64
        ).view(np.float64)[0]
        assert anchor_fraction == pytest.approx(24.0 / 26.0)
    finally:
        provider.stop()


def test_boundary_seed_fine_core_expands_only_same_strict_component(monkeypatch):
    """A bounded second target surface is recovered, detached hand is not."""

    provider, _tracker = _guarded_primary_provider(
        monkeypatch,
        [],
        {},
        _large_mask((40, 30, 80, 70)),
    )
    try:
        seed_frame = _boundary_seed_fine_frame(
            20, target_bbox=(0, 35, 26, 75)
        )
        seed = np.zeros((120, 180), dtype=np.uint8)
        seed[35:75, 0:26] = 1
        model = provider._build_boundary_seed_fine_appearance_model(
            seed_frame, seed
        )
        assert model is not None
        provider._boundary_seed_fine_appearance_model = model
        provider._guarded_v2_bootstrap_phase = "expired"
        provider._guarded_v2_bootstrap_expiry_kind = "boundary_seed"

        current = _boundary_seed_fine_frame(
            110,
            target_bbox=(80, 35, 106, 75),
            hand_bbox=(45, 20, 72, 90),
        )
        # The minimal synthetic seed has only foreground/background bins, so
        # install one distinct intermediate Lab bin for the second surface.
        # The strict red body and skin bins remain untouched.
        relaxed_colour = (255, 255, 0)
        current.color_bgr[31:35, 80:106] = relaxed_colour
        relaxed_lab = cv2.cvtColor(
            np.asarray([[relaxed_colour]], dtype=np.uint8), cv2.COLOR_BGR2LAB
        )[0, 0]
        a_index = min(model.bins - 1, int(relaxed_lab[1]) * model.bins // 256)
        b_index = min(model.bins - 1, int(relaxed_lab[2]) * model.bins // 256)
        likelihood = np.array(model.likelihood_table, copy=True)
        likelihood[a_index, b_index] = 0.50
        likelihood_bytes = np.ascontiguousarray(
            likelihood, dtype=np.float32
        ).tobytes(order="C")
        import hashlib

        model = replace(
            model,
            likelihood_table=np.frombuffer(
                likelihood_bytes, dtype=np.float32
            ).reshape(likelihood.shape),
            likelihood_table_digest=hashlib.sha256(
                likelihood_bytes
            ).hexdigest(),
        )
        provider._boundary_seed_fine_appearance_model = model

        raw = np.zeros((120, 180), dtype=np.uint8)
        raw[20:90, 45:72] = 1
        raw[31:75, 80:106] = 1
        raw_evidence = provider._defer_full_target_geometry_commit(current, raw)
        assert raw_evidence is not None
        candidate, reason = provider._boundary_seed_fine_visible_core(
            current, raw_evidence
        )
        assert candidate is not None, reason
        proof = provider._boundary_seed_fine_visible_core_frame_proof
        assert proof is not None and proof.relaxed_expansion
        assert not np.any(candidate.mask[20:90, 45:72])
        assert np.all(candidate.mask[35:75, 80:106])
        assert np.all(candidate.mask[31:35, 80:106])

        provider.cfg["tracker"][
            "boundary_seed_fine_relaxed_max_area_ratio"
        ] = 1.34
        changed, changed_reason = provider._boundary_seed_fine_visible_core(
            current, raw_evidence
        )
        assert changed is None
        assert "proof/config changed" in changed_reason
    finally:
        provider.stop()


def test_boundary_seed_fine_relaxed_hand_union_falls_back_to_strict(monkeypatch):
    """A relaxed component above the bounded expansion ratio is never used."""

    provider, _tracker = _guarded_primary_provider(
        monkeypatch,
        [],
        {},
        _large_mask((40, 30, 80, 70)),
    )
    try:
        seed_frame = _boundary_seed_fine_frame(
            20, target_bbox=(0, 35, 26, 75)
        )
        seed = np.zeros((120, 180), dtype=np.uint8)
        seed[35:75, 0:26] = 1
        model = provider._build_boundary_seed_fine_appearance_model(
            seed_frame, seed
        )
        assert model is not None

        # Make the synthetic skin bin relaxed-supported but not strict.  It
        # touches the red target and therefore produces one large union; the
        # expansion ratio gate must preserve only the original strict target.
        hand_colour = (75, 145, 210)
        hand_lab = cv2.cvtColor(
            np.asarray([[hand_colour]], dtype=np.uint8), cv2.COLOR_BGR2LAB
        )[0, 0]
        a_index = min(model.bins - 1, int(hand_lab[1]) * model.bins // 256)
        b_index = min(model.bins - 1, int(hand_lab[2]) * model.bins // 256)
        likelihood = np.array(model.likelihood_table, copy=True)
        likelihood[a_index, b_index] = 0.50
        likelihood_bytes = np.ascontiguousarray(
            likelihood, dtype=np.float32
        ).tobytes(order="C")
        import hashlib

        model = replace(
            model,
            likelihood_table=np.frombuffer(
                likelihood_bytes, dtype=np.float32
            ).reshape(likelihood.shape),
            likelihood_table_digest=hashlib.sha256(
                likelihood_bytes
            ).hexdigest(),
        )
        provider._boundary_seed_fine_appearance_model = model
        provider._guarded_v2_bootstrap_phase = "expired"
        provider._guarded_v2_bootstrap_expiry_kind = "boundary_seed"

        current = _boundary_seed_fine_frame(
            102,
            target_bbox=(80, 35, 106, 75),
            hand_bbox=(55, 20, 82, 90),
        )
        raw = np.zeros((120, 180), dtype=np.uint8)
        raw[20:90, 55:82] = 1
        raw[35:75, 80:106] = 1
        raw_evidence = provider._defer_full_target_geometry_commit(current, raw)
        assert raw_evidence is not None
        candidate, reason = provider._boundary_seed_fine_visible_core(
            current, raw_evidence
        )
        assert candidate is not None, reason
        proof = provider._boundary_seed_fine_visible_core_frame_proof
        assert proof is not None and not proof.relaxed_expansion
        assert int(np.count_nonzero(candidate.mask)) == 24 * 40
        assert not np.any(candidate.mask[20:90, 55:82])
    finally:
        provider.stop()


def test_boundary_seed_fine_core_rejects_hand_when_target_hidden(monkeypatch):
    """A colour fragment with no clipped-seed support cannot publish as target."""

    provider, _tracker = _guarded_primary_provider(
        monkeypatch,
        [],
        {},
        _large_mask((40, 30, 80, 70)),
    )
    try:
        seed_frame = _boundary_seed_fine_frame(
            20, target_bbox=(0, 35, 26, 75)
        )
        seed = np.zeros((120, 180), dtype=np.uint8)
        seed[35:75, 0:26] = 1
        provider._boundary_seed_fine_appearance_model = (
            provider._build_boundary_seed_fine_appearance_model(seed_frame, seed)
        )
        assert provider._boundary_seed_fine_appearance_model is not None
        provider._guarded_v2_bootstrap_phase = "expired"
        provider._guarded_v2_bootstrap_expiry_kind = "boundary_seed"

        hidden = _boundary_seed_fine_frame(
            96, hand_bbox=(70, 20, 112, 92)
        )
        raw = np.zeros((120, 180), dtype=np.uint8)
        raw[20:92, 70:112] = 1
        raw_evidence = provider._defer_full_target_geometry_commit(hidden, raw)
        assert raw_evidence is not None
        candidate, reason = provider._boundary_seed_fine_visible_core(
            hidden, raw_evidence
        )
        assert candidate is None
        assert "empty" in reason or "overlap rejected" in reason
        assert provider._boundary_seed_fine_visible_core_frame_proof is None
    finally:
        provider.stop()


def test_boundary_seed_fine_core_stays_active_until_rearmed_bootstrap_commissions(
    monkeypatch,
):
    """Formal recovery must not disable the selector one tick too early."""

    provider, _tracker = _guarded_primary_provider(
        monkeypatch,
        [],
        {},
        _large_mask((40, 30, 80, 70)),
    )
    try:
        seed_frame = _boundary_seed_fine_frame(
            20, target_bbox=(0, 35, 26, 75)
        )
        seed = np.zeros((120, 180), dtype=np.uint8)
        seed[35:75, 0:26] = 1
        provider._boundary_seed_fine_appearance_model = (
            provider._build_boundary_seed_fine_appearance_model(seed_frame, seed)
        )
        assert provider._boundary_seed_fine_appearance_model is not None
        provider._guarded_v2_bootstrap_boundary_rearm_count = 1
        provider._guarded_v2_bootstrap_commission_count = 0
        provider._guarded_v2_bootstrap_phase = "expired"
        provider._guarded_v2_bootstrap_expiry_kind = "other"

        current = _boundary_seed_fine_frame(
            105,
            target_bbox=(90, 35, 110, 75),
            hand_bbox=(64, 20, 92, 90),
        )
        raw = np.zeros((120, 180), dtype=np.uint8)
        raw[20:90, 64:92] = 1
        raw[35:75, 90:110] = 1
        raw_evidence = provider._defer_full_target_geometry_commit(current, raw)
        assert raw_evidence is not None
        candidate, reason = provider._boundary_seed_fine_visible_core(
            current, raw_evidence
        )
        assert candidate is not None, reason
        np.testing.assert_array_equal(
            candidate.geometry.bbox_xyxy,
            np.asarray([92, 35, 110, 75], dtype=np.int32),
        )

        provider._guarded_v2_bootstrap_commission_count = 1
        provider._begin_frame_mask_evidence(current)
        raw_evidence = provider._defer_full_target_geometry_commit(current, raw)
        assert raw_evidence is not None
        candidate, reason = provider._boundary_seed_fine_visible_core(
            current, raw_evidence
        )
        assert candidate is None
        assert "inactive" in reason
    finally:
        provider.stop()


def test_boundary_seed_fine_core_is_lost_only_after_commission(monkeypatch):
    """A clipped single-mode model must not crop clean commissioned tracking."""

    provider, _tracker = _guarded_primary_provider(
        monkeypatch,
        [],
        {},
        _large_mask((40, 30, 80, 70)),
    )
    try:
        seed_frame = _boundary_seed_fine_frame(
            20, target_bbox=(0, 35, 26, 75)
        )
        seed = np.zeros((120, 180), dtype=np.uint8)
        seed[35:75, 0:26] = 1
        provider._boundary_seed_fine_appearance_model = (
            provider._build_boundary_seed_fine_appearance_model(seed_frame, seed)
        )
        assert provider._boundary_seed_fine_appearance_model is not None
        provider._guarded_v2_bootstrap_boundary_rearm_count = 1
        provider._guarded_v2_bootstrap_commission_count = 1
        provider._guarded_v2_bootstrap_phase = "commissioned"

        current = _boundary_seed_fine_frame(
            123,
            target_bbox=(90, 35, 110, 75),
            hand_bbox=(64, 20, 92, 90),
        )
        raw = np.zeros((120, 180), dtype=np.uint8)
        raw[20:90, 64:92] = 1
        raw[35:75, 90:110] = 1

        provider._tracking_committed = True
        raw_evidence = provider._defer_full_target_geometry_commit(current, raw)
        assert raw_evidence is not None
        candidate, reason = provider._boundary_seed_fine_visible_core(
            current, raw_evidence
        )
        assert candidate is None
        assert "LOST-only after commission" in reason

        # The same immutable model remains available after an actual LOST
        # transition, where the raw union is no longer robot-facing authority.
        provider._tracking_committed = False
        provider._begin_frame_mask_evidence(current)
        raw_evidence = provider._defer_full_target_geometry_commit(current, raw)
        assert raw_evidence is not None
        candidate, reason = provider._boundary_seed_fine_visible_core(
            current, raw_evidence
        )
        assert candidate is not None, reason
        np.testing.assert_array_equal(
            candidate.geometry.bbox_xyxy,
            np.asarray([92, 35, 110, 75], dtype=np.int32),
        )
    finally:
        provider.stop()


def _full_frame_mask_gate_reference(provider, frame, mask):
    """Pre-ROI mask-only evidence algorithm retained as an equivalence oracle."""

    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    if mask_u8.shape != frame.depth_raw.shape:
        return (
            False,
            (
                f"mask shape {mask_u8.shape} differs from RGB-D "
                f"{frame.depth_raw.shape}"
            ),
            np.zeros(3, dtype=np.float32),
            0,
        )

    point_cfg = dict(provider.cfg.get("pointcloud", {}))
    point_cfg.update(provider.extractor.cfg)
    erode_kernel = int(point_cfg.get("erode_kernel", 3))
    if erode_kernel > 0:
        mask_u8 = erode_mask(mask_u8, erode_kernel)
    mask_u8 = largest_component(mask_u8, min_area=20)

    stride = max(1, int(point_cfg.get("stride", 1)))
    sampled_mask = mask_u8[::stride, ::stride].astype(bool)
    v_idx, u_idx = np.nonzero(sampled_mask)
    if len(v_idx) < 10:
        return (
            False,
            "too few valid depth pixels after mask",
            np.zeros(3, dtype=np.float32),
            0,
        )

    depth_raw = np.asarray(frame.depth_raw)[::stride, ::stride]
    z = depth_raw[v_idx, u_idx].astype(np.float32) * float(frame.depth_scale)
    camera_cfg = provider.cfg.get("camera", {})
    z_min = float(point_cfg.get("z_min", camera_cfg.get("z_min", 0.0)))
    z_max = float(point_cfg.get("z_max", camera_cfg.get("z_max", 10.0)))
    depth_valid = np.isfinite(z) & (z > z_min) & (z < z_max)
    if int(np.count_nonzero(depth_valid)) < 10:
        return (
            False,
            "too few valid depth pixels after mask",
            np.zeros(3, dtype=np.float32),
            0,
        )

    z = z[depth_valid]
    u = u_idx[depth_valid].astype(np.float32) * stride
    v = v_idx[depth_valid].astype(np.float32) * stride
    intrinsics = frame.intrinsics
    if not (
        np.isfinite(float(intrinsics.fx))
        and np.isfinite(float(intrinsics.fy))
        and float(intrinsics.fx) != 0.0
        and float(intrinsics.fy) != 0.0
    ):
        return (
            False,
            "invalid camera intrinsics",
            np.zeros(3, dtype=np.float32),
            0,
        )
    x = (u - float(intrinsics.ppx)) / float(intrinsics.fx) * z
    y = (v - float(intrinsics.ppy)) / float(intrinsics.fy) * z
    points_camera = np.stack([x, y, z], axis=1).astype(np.float32)

    transform = np.asarray(provider.extractor.T_base_camera, dtype=np.float32)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        return (
            False,
            "invalid camera-to-reference transform",
            np.zeros(3, dtype=np.float32),
            0,
        )
    points_reference = points_camera @ transform[:3, :3].T + transform[:3, 3]
    workspace_min = point_cfg.get("workspace_min")
    workspace_max = point_cfg.get("workspace_max")
    finite = np.all(np.isfinite(points_reference), axis=1)
    if workspace_min is not None and workspace_max is not None:
        minimum = np.asarray(workspace_min, dtype=np.float32).reshape(3)
        maximum = np.asarray(workspace_max, dtype=np.float32).reshape(3)
        finite &= np.all(points_reference >= minimum[None, :], axis=1)
        finite &= np.all(points_reference <= maximum[None, :], axis=1)
    points_reference = points_reference[finite]
    if len(points_reference) < 10:
        return (
            False,
            "too few points after workspace crop",
            np.zeros(3, dtype=np.float32),
            0,
        )
    return (
        True,
        f"mask gate points={len(points_reference)}",
        np.median(points_reference, axis=0).astype(np.float32),
        len(points_reference),
    )


def _large_mask(bbox) -> np.ndarray:
    value = np.zeros((120, 180), dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    value[y1:y2, x1:x2] = 1
    return value


def _large_ellipse_mask(bbox) -> np.ndarray:
    value = np.zeros((120, 180), dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    center_x = 0.5 * (x1 + x2 - 1)
    center_y = 0.5 * (y1 + y2 - 1)
    radius_x = max(0.5, 0.5 * (x2 - x1))
    radius_y = max(0.5, 0.5 * (y2 - y1))
    yy, xx = np.ogrid[:120, :180]
    inside = ((xx - center_x) / radius_x) ** 2 + (
        (yy - center_y) / radius_y
    ) ** 2 <= 1.0
    value[inside] = 1
    return value


def _moving_mask_support_frame(
    frame_id: int, mask: np.ndarray, support_fraction: float
) -> RGBDFrame:
    frame = _large_frame(frame_id)
    frame.color_bgr[:] = 45
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    supported = int(np.floor(len(xs) * float(support_fraction)))
    if supported > 0:
        # Spread supported pixels through the silhouette so this models the
        # reviewed ball's shading/blur, not one synthetic appendage branch.
        positions = np.linspace(0, len(xs) - 1, supported, dtype=np.int64)
        frame.color_bgr[ys[positions], xs[positions]] = (210, 80, 235)
    return frame


def _large_frame(
    frame_id: int,
    *,
    contaminant_bbox=None,
    contaminant_bgr=(200, 200, 200),
) -> RGBDFrame:
    color = np.full((120, 180, 3), 45, dtype=np.uint8)
    color[30:70, 40:80] = (210, 80, 235)
    if contaminant_bbox is not None:
        x1, y1, x2, y2 = contaminant_bbox
        color[y1:y2, x1:x2] = contaminant_bgr
    return RGBDFrame(
        color_bgr=color,
        depth_raw=np.full((120, 180), 800, dtype=np.uint16),
        depth_scale=0.001,
        intrinsics=CameraIntrinsics(
            width=180,
            height=120,
            fx=200.0,
            fy=200.0,
            ppx=89.5,
            ppy=59.5,
        ),
        timestamp=float(frame_id) / 30.0,
        frame_id=frame_id,
    )


def _assert_adaptive_snapshot_equal(first, second) -> None:
    assert first.state is not None and second.state is not None
    for name in (
        "depth_median",
        "center_uv",
        "area",
        "initial_area",
        "initial_bbox_area",
        "lost_count",
        "valid",
    ):
        assert getattr(first.state, name) == getattr(second.state, name)
    np.testing.assert_array_equal(first.state.bbox_xyxy, second.state.bbox_xyxy)
    np.testing.assert_array_equal(first.state.last_mask, second.state.last_mask)
    for name in (
        "fixed_bbox_xyxy",
        "anchor_hs",
        "anchor_sv",
        "adaptive_hs",
        "adaptive_sv",
        "background_hs",
        "background_sv",
        "previous_gray",
        "temporal_mask_score",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert first.depth_half_width == second.depth_half_width
    assert first.confidence == second.confidence
    assert first.appearance_freeze_remaining == second.appearance_freeze_remaining
    assert first.motion_mask_iou == second.motion_mask_iou
    assert first.temporal_mask_stabilized == second.temporal_mask_stabilized
    assert first.temporal_motion_source == second.temporal_motion_source


def _mask_result(
    mask: np.ndarray, *, valid: bool = True, message: str = "fake tracker"
) -> MaskResult:
    ys, xs = np.nonzero(mask)
    bbox = (
        np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.int32)
        if len(xs)
        else np.asarray([0, 0, 1, 1], dtype=np.int32)
    )
    return MaskResult(
        mask=mask.copy(),
        bbox_xyxy=bbox,
        score=1.0 if valid else 0.0,
        valid=valid,
        message=message,
    )


def _video_result(
    frame_id: int,
    mask: np.ndarray,
    *,
    valid: bool = True,
    message: str = "fake online SAM2",
    internal_frame_idx: int | None = None,
) -> SAM2VideoResult:
    return SAM2VideoResult(
        mask=mask.copy() if valid else np.zeros_like(mask),
        valid=valid,
        frame_id=frame_id,
        message=message,
        timings_ms={"service_total": 3.25},
        internal_frame_idx=(
            int(frame_id) if internal_frame_idx is None else int(internal_frame_idx)
        ),
        mask_area=int(mask.sum()) if valid else 0,
    )


class _FakeCamera:
    device_serial = "fake"

    def __init__(self, frames, lifecycle=None):
        self.frames = list(frames)
        self.index = 0
        self.started = False
        self.stopped = False
        self.lifecycle = lifecycle
        self.requested_timeouts_ms = []

    def start(self):
        if self.lifecycle is not None:
            self.lifecycle.append("camera_start")
        self.started = True

    def stop(self):
        self.stopped = True

    def get_frame(self, *, timeout_ms):
        self.requested_timeouts_ms.append(timeout_ms)
        value = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return value


class _FakeTracker:
    manages_identity = True
    confidence = 1.0

    def __init__(self, update_results):
        self.update_results = dict(update_results)
        self.initialized = False
        self.state = None
        self.token = "new"
        self.restore_calls = 0
        self.initialize_calls = []
        self.reinit_masks = []
        self.reject_reinit_masks = []
        self.update_started = None
        self.online_started = None
        self.overlapped = False

    def _set_state(self, result: MaskResult) -> None:
        self.state = SimpleNamespace(
            lost_count=0 if result.valid else 1,
            last_mask=result.mask.copy(),
            bbox_xyxy=result.bbox_xyxy.copy(),
            valid=result.valid,
        )

    def initialize(self, _frame, mask, bbox_xyxy):
        self.initialize_calls.append(
            (int(_frame.frame_id), (np.asarray(mask) > 0).astype(np.uint8))
        )
        self.initialized = True
        result = _mask_result(mask, message="fake initialized")
        result.bbox_xyxy = np.asarray(bbox_xyxy, dtype=np.int32).copy()
        self.token = "initialized"
        self._set_state(result)
        return result

    def snapshot_state(self):
        return copy.deepcopy((self.token, self.state))

    def restore_state(self, snapshot):
        self.restore_calls += 1
        self.token, self.state = copy.deepcopy(snapshot)

    def update(self, frame):
        if self.update_started is not None:
            self.update_started.set()
            self.overlapped = self.online_started.wait(timeout=0.5)
        result = copy.deepcopy(self.update_results[frame.frame_id])
        self.token = f"adaptive-f{frame.frame_id}"
        self._set_state(result)
        return result

    def reinitialize_with_mask(self, frame, mask, **_kwargs):
        mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
        self.reinit_masks.append((frame.frame_id, mask_u8.copy()))
        self.token = f"sam2-mutated-f{frame.frame_id}"
        if any(np.array_equal(mask_u8, value) for value in self.reject_reinit_masks):
            result = _mask_result(
                np.zeros_like(mask_u8), valid=False, message="strict gate rejected"
            )
        else:
            result = _mask_result(mask_u8, message="strict gate accepted")
        self._set_state(result)
        return result


class _IdentityAwareFakeTracker(_FakeTracker):
    """Small deterministic owner with immutable magenta-target appearance."""

    def __init__(self, update_results):
        super().__init__(update_results)
        self.cfg = copy.deepcopy(load_config()["tracker"])

    def snapshot_state(self):
        return copy.deepcopy(SimpleNamespace(token=self.token, state=self.state))

    def restore_state(self, snapshot):
        self.restore_calls += 1
        self.token = str(snapshot.token)
        self.state = copy.deepcopy(snapshot.state)

    def target_appearance_support_stats(self, frame, mask, *, threshold=None):
        selected = np.asarray(mask) > 0
        count = int(selected.sum())
        if count == 0:
            return (0.0, 0.0, 0)
        pixels = np.asarray(frame.color_bgr)[selected]
        target = (pixels[:, 0] >= 180) & (pixels[:, 1] <= 120) & (pixels[:, 2] >= 180)
        probabilities = np.where(target, 0.95, 0.10).astype(np.float32)
        support_threshold = 0.50 if threshold is None else float(threshold)
        return (
            float(probabilities.mean()),
            float(np.mean(probabilities >= support_threshold)),
            count,
        )

    def target_appearance_supported_mask(self, frame, mask, *, threshold=None):
        selected = np.asarray(mask) > 0
        output = np.zeros_like(np.asarray(mask), dtype=np.uint8)
        if not np.any(selected):
            return output
        pixels = np.asarray(frame.color_bgr)
        target = (
            (pixels[..., 0] >= 180) & (pixels[..., 1] <= 120) & (pixels[..., 2] >= 180)
        )
        output[np.logical_and(selected, target)] = 1
        return output


def _moving_target_frame(
    frame_id: int,
    target_bbox,
    *,
    contaminant_bbox=None,
) -> RGBDFrame:
    frame = _large_frame(frame_id)
    frame.color_bgr[:] = 45
    x1, y1, x2, y2 = target_bbox
    frame.color_bgr[y1:y2, x1:x2] = (210, 80, 235)
    if contaminant_bbox is not None:
        x1, y1, x2, y2 = contaminant_bbox
        frame.color_bgr[y1:y2, x1:x2] = (200, 200, 200)
    return frame


def _boundary_exit_appearance_frame(
    frame_id: int,
    bbox,
    *,
    distributed: bool,
) -> RGBDFrame:
    """Build a boundary target with weak appearance only in its extra side.

    ``distributed=True`` models the reviewed shaded/blurred ball: aggregate
    target support is healthy and its one connected support lattice spans the
    whole raw silhouette, while the rightmost third alone remains below the
    appendage floor.  ``False`` models a gray one-sided appendage whose target
    appearance ends at the aligned-clean boundary.
    """

    frame = _large_frame(frame_id)
    frame.color_bgr[:] = 45
    x1, y1, x2, y2 = (int(value) for value in bbox)
    yy, xx = np.indices(frame.depth_raw.shape)
    candidate = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
    clean_edge = x2 - 10
    if distributed:
        horizontal_lattice = ((yy - y1) % 4) == 0
        dense_clean = (xx < clean_edge) & (((xx - x1) % 4) != 3)
        sparse_extra = (xx >= clean_edge) & (((xx - clean_edge) % 4) == 0)
        supported = candidate & (horizontal_lattice | dense_clean | sparse_extra)
        # Make one four-connected component whose bbox is the exact raw bbox.
        supported |= candidate & (
            (xx == x1) | (xx == x2 - 1) | (yy == y1) | (yy == y2 - 1)
        )
    else:
        supported = candidate & (xx < clean_edge)
    frame.color_bgr[supported] = (210, 80, 235)
    return frame


def _install_visible_partial_boundary_history(
    provider,
    bbox,
    *,
    frame_id: int = 1,
    velocity_xy=(0.0, 0.0),
) -> None:
    """Install one immutable, confirmed, non-authoritative partial owner."""

    mask = _large_mask(bbox)
    center = provider._binary_mask_centroid(mask)
    assert center is not None
    bbox_array = np.asarray(bbox, dtype=np.int32)
    area = float(np.count_nonzero(mask))
    provider._visible_exact_partial_continuity = SimpleNamespace(
        frame_id=int(frame_id),
        timestamp=float(frame_id) / 30.0,
        mask=mask.copy(),
        mask_digest=provider._publication_mask_digest(mask),
        core_bbox_xyxy=bbox_array.copy(),
        core_center_xy=np.asarray(center, dtype=np.float64).copy(),
        core_area=area,
        bbox_xyxy=bbox_array.copy(),
        center_xy=np.asarray(center, dtype=np.float64).copy(),
        area=area,
        bbox_area=float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
        depth_median=0.8,
        center_velocity_px_s=np.asarray(velocity_xy, dtype=np.float64),
        velocity_samples=1,
        hits=2,
        evidence_kind="sanitized_target_core",
        sanitized_handoff_eligible=True,
        sanitized_anchor_bbox_xyxy=bbox_array.copy(),
        sanitized_anchor_center_xy=np.asarray(center, dtype=np.float64).copy(),
        sanitized_anchor_area=area,
    )
    # Mirror the production state after several safe partial publications:
    # full/trusted authority was intentionally not advanced.
    provider._trusted_online_sam2_publication = None
    provider._trusted_online_sam2_continuity_intact = False
    provider._trusted_online_sam2_continuity_reason = (
        "test visible-partial chain revoked trusted authority"
    )


class _FakeExtractor:
    def extract(self, _frame, mask):
        valid = bool(np.asarray(mask).any())
        points = (
            np.ones((8, 3), dtype=np.float32)
            if valid
            else np.zeros((0, 3), dtype=np.float32)
        )
        return ExtractedObjectPCD(
            points=points,
            colors=points.copy(),
            policy_points=points.copy(),
            reference_points=points.copy(),
            center=(
                np.asarray([0.5, 0.0, 0.2], dtype=np.float32)
                if valid
                else np.zeros(3, dtype=np.float32)
            ),
            valid=valid,
            message="fake points" if valid else "empty",
        )


class _FakeHistory:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1
        return None

    def update(self, points, center, _timestamp):
        return SimpleNamespace(
            pcd_history=points[None],
            center=center.copy(),
            velocity=np.zeros(3, dtype=np.float32),
            valid=True,
            message="fake history",
        )


class _FakeOnlineSAM2Manager:
    def __init__(
        self,
        track_results=None,
        lifecycle=None,
        initialize_results=None,
    ):
        self.track_results = dict(track_results or {})
        self.initialize_results = dict(initialize_results or {})
        self.lifecycle = lifecycle
        self.calls = []
        self.thread_ids = []
        self.track_started = None
        self.update_started = None
        self.overlapped = False
        self.closed = False
        self.track_wait_events = {}
        self.track_entered_events = {}
        self.track_finished_events = {}

    def _record(self, operation, *values):
        self.calls.append((operation, *values))
        self.thread_ids.append(threading.get_ident())

    def start(self):
        if self.lifecycle is not None:
            self.lifecycle.append("online_sam2_start")
        self._record("start")
        return {
            "service": "dynamic-pcd-online-sam2",
            "protocol_version": 1,
            "device": "fake",
            "image_size": 512,
            "vos_optimized": True,
            "vos_compile_mode": "max-autotune-no-cudagraphs",
            "vos_compile_cuda_graphs": False,
            "vos_component_compile_modes": {
                "image_encoder": "max-autotune-no-cudagraphs",
                "memory_encoder": "max-autotune-no-cudagraphs",
                "memory_attention": "max-autotune-no-cudagraphs",
                "sam_prompt_encoder": "max-autotune-no-cudagraphs",
                "sam_mask_decoder": "max-autotune-no-cudagraphs",
            },
            "vos_component_compile_dynamic": {
                "image_encoder": False,
                "memory_encoder": False,
                "memory_attention": True,
                "sam_prompt_encoder": False,
                "sam_mask_decoder": False,
            },
            "vos_memory_attention_rope_grid_hw": [32, 32],
            "vos_memory_attention_rope_expected_tokens": 1024,
            "vos_memory_attention_rope_cache_count": 8,
            "vos_memory_attention_rope_cache_token_counts": [1024] * 8,
            "vos_memory_attention_rope_caches_verified": True,
            "initialized": False,
            "compile_prewarm_required": True,
            "compile_prewarm_completed": True,
            "compile_prewarm_contract": (
                "initialize_box_track_reset_explicit_mask_track_reset_v1"
            ),
            "compile_prewarm_shape_hw": [480, 848],
            "compile_prewarm_ms": 1.0,
            "compile_prewarm_initialize_box_ms": 0.1,
            "compile_prewarm_track_ms": 0.1,
            "compile_prewarm_initialize_mask_ms": 0.1,
            "compile_prewarm_mask_track_ms": 0.1,
        }

    def initialize(self, image_bgr, mask, frame_id, **_kwargs):
        self._record(
            "initialize", int(frame_id), image_bgr.copy(), np.asarray(mask).copy()
        )
        value = self.initialize_results.get(int(frame_id))
        if isinstance(value, BaseException):
            raise value
        if value is not None:
            return copy.deepcopy(value)
        return _video_result(int(frame_id), np.asarray(mask))

    def initialize_box(self, image_bgr, bbox_xyxy, frame_id, **_kwargs):
        bbox = np.asarray(bbox_xyxy, dtype=np.int32)
        self._record("initialize_box", int(frame_id), image_bgr.copy(), bbox.copy())
        x1, y1, x2, y2 = bbox.tolist()
        mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
        return _video_result(int(frame_id), mask)

    def track(self, image_bgr, frame_id, **_kwargs):
        self._record("track", int(frame_id), image_bgr.copy())
        entered = self.track_entered_events.get(int(frame_id))
        if entered is not None:
            entered.set()
        if self.track_started is not None:
            self.track_started.set()
            self.overlapped = self.update_started.wait(timeout=0.5)
        wait_event = self.track_wait_events.get(int(frame_id))
        if wait_event is not None and not wait_event.wait(timeout=1.0):
            raise TimeoutError("fake track was not released")
        try:
            value = self.track_results[int(frame_id)]
            if isinstance(value, BaseException):
                raise value
            return copy.deepcopy(value)
        finally:
            finished = self.track_finished_events.get(int(frame_id))
            if finished is not None:
                finished.set()

    def reset(self, **_kwargs):
        self._record("reset")
        return {"status": "ok"}

    def close(self):
        self._record("close")
        self.closed = True


class _InitialSeedAppearanceTracker:
    """Deterministic immutable-appearance oracle for the one-shot seed gate."""

    def __init__(self, supported_mask, *, restore_error=None):
        self.supported_mask = np.asarray(supported_mask, dtype=np.uint8).copy()
        self.restore_error = restore_error
        self.state_token = "uninitialized"
        self.initialize_masks = []
        self.restore_calls = 0

    def snapshot_state(self):
        return self.state_token

    def restore_state(self, snapshot):
        self.restore_calls += 1
        if self.restore_error is not None:
            raise self.restore_error
        self.state_token = snapshot

    def initialize(self, frame, mask, bbox_xyxy):
        mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
        self.initialize_masks.append(mask_u8.copy())
        self.state_token = f"temporary-{int(frame.frame_id)}"
        return _mask_result(mask_u8)

    def target_appearance_probability_roi(self, _frame, _root):
        if not self.state_token.startswith("temporary-"):
            raise AssertionError("appearance queried before temporary init")
        return object()

    def target_appearance_supported_mask(
        self,
        _frame,
        _mask,
        *,
        threshold=None,
        probability_roi=None,
    ):
        assert threshold == pytest.approx(0.10)
        assert probability_roi is not None
        return self.supported_mask.copy()


def _initial_seed_frame(frame_id=1, *, shape=(80, 96)):
    height, width = shape
    return RGBDFrame(
        color_bgr=np.full((height, width, 3), 90, dtype=np.uint8),
        depth_raw=np.full((height, width), 800, dtype=np.uint16),
        depth_scale=0.001,
        intrinsics=CameraIntrinsics(
            width=width,
            height=height,
            fx=100.0,
            fy=100.0,
            ppx=width / 2.0,
            ppy=height / 2.0,
        ),
        timestamp=float(frame_id) / 30.0,
        frame_id=int(frame_id),
    )


def _initial_seed_provider(tracker):
    provider = object.__new__(ObjectPCDProvider)
    provider.tracker = tracker
    provider.online_sam2_cfg = {
        "trusted_min_valid_depth_ratio": 0.35,
        "trusted_max_depth_spread_m": 0.18,
    }
    return provider


def test_initial_sam_halo_cleanup_removes_only_boundary_connected_low_probability():
    frame = _initial_seed_frame()
    raw = np.zeros(frame.depth_raw.shape, dtype=np.uint8)
    raw[20:60, 24:64] = 1
    supported = np.zeros_like(raw)
    supported[21:59, 25:63] = 1
    tracker = _InitialSeedAppearanceTracker(supported)
    provider = _initial_seed_provider(tracker)

    cleaned = provider._initial_sam_halo_cleanup(
        frame, raw, source="online_sam2_box"
    )

    np.testing.assert_array_equal(cleaned, supported)
    assert not cleaned.flags.writeable
    assert tracker.state_token == "uninitialized"
    assert tracker.restore_calls == 1
    np.testing.assert_array_equal(tracker.initialize_masks, [raw])
    evidence = provider.last_initial_sam_halo_cleanup_evidence
    assert evidence.applied
    assert evidence.raw_area_px == 1600
    assert evidence.clean_area_px == 1444
    assert evidence.deleted_area_px == 156
    assert evidence.raw_mask_digest != evidence.clean_mask_digest
    # The source75-like boundary cleanup improves precision without an
    # indiscriminate erosion: every retained pixel has learned P>=0.10.
    assert evidence.clean_area_px / evidence.raw_area_px == pytest.approx(0.9025)


@pytest.mark.parametrize("case", ["fast", "red", "heavy"])
def test_initial_sam_halo_cleanup_small_deletion_is_bit_exact_noop(case):
    del case
    frame = _initial_seed_frame()
    raw = np.zeros(frame.depth_raw.shape, dtype=np.uint8)
    raw[20:60, 24:64] = 1
    supported = raw.copy()
    supported[20, 24:34] = 0
    tracker = _InitialSeedAppearanceTracker(supported)
    provider = _initial_seed_provider(tracker)

    output = provider._initial_sam_halo_cleanup(frame, raw, source="sam2")

    np.testing.assert_array_equal(output, raw)
    assert output.tobytes() == raw.tobytes()
    assert not provider.last_initial_sam_halo_cleanup_evidence.applied
    assert "deletion 10px<" in provider.last_initial_sam_halo_cleanup_evidence.status


def test_initial_sam_halo_cleanup_rejects_disconnected_rolling_proposal():
    frame = _initial_seed_frame()
    raw = np.zeros(frame.depth_raw.shape, dtype=np.uint8)
    raw[20:60, 24:64] = 1
    supported = raw.copy()
    supported[39, 24:64] = 0
    tracker = _InitialSeedAppearanceTracker(supported)
    provider = _initial_seed_provider(tracker)

    output = provider._initial_sam_halo_cleanup(
        frame, raw, source="online_sam2_box"
    )

    np.testing.assert_array_equal(output, raw)
    assert not provider.last_initial_sam_halo_cleanup_evidence.applied
    assert "not exactly one connected component" in (
        provider.last_initial_sam_halo_cleanup_evidence.status
    )


def test_initial_sam_halo_cleanup_preserves_camera_boundary_object():
    frame = _initial_seed_frame()
    raw = np.zeros(frame.depth_raw.shape, dtype=np.uint8)
    raw[18:58, 0:40] = 1
    supported = np.zeros_like(raw)
    supported[19:57, 1:39] = 1
    tracker = _InitialSeedAppearanceTracker(supported)
    provider = _initial_seed_provider(tracker)

    output = provider._initial_sam_halo_cleanup(
        frame, raw, source="online_sam2_box"
    )

    np.testing.assert_array_equal(output, raw)
    assert tracker.initialize_masks == []
    assert "camera boundary" in provider.last_initial_sam_halo_cleanup_evidence.status


def test_initial_sam_halo_cleanup_restore_failure_aborts_instead_of_committing():
    frame = _initial_seed_frame()
    raw = np.zeros(frame.depth_raw.shape, dtype=np.uint8)
    raw[20:60, 24:64] = 1
    supported = np.zeros_like(raw)
    supported[21:59, 25:63] = 1
    tracker = _InitialSeedAppearanceTracker(
        supported, restore_error=RuntimeError("synthetic restore failure")
    )
    provider = _initial_seed_provider(tracker)

    with pytest.raises(RuntimeError, match="synthetic restore failure"):
        provider._initial_sam_halo_cleanup(
            frame, raw, source="online_sam2_box"
        )
    assert tracker.state_token == "temporary-1"


@pytest.mark.parametrize("interrupt_type", (KeyboardInterrupt, SystemExit))
def test_initial_sam_halo_cleanup_preserves_interrupt_when_restore_also_fails(
    monkeypatch, interrupt_type
):
    frame = _initial_seed_frame()
    raw = np.zeros(frame.depth_raw.shape, dtype=np.uint8)
    raw[20:60, 24:64] = 1
    supported = np.ones_like(raw)
    restore_error = RuntimeError("synthetic restore double fault")
    tracker = _InitialSeedAppearanceTracker(
        supported, restore_error=restore_error
    )
    provider = _initial_seed_provider(tracker)

    def interrupted_initialize(*_args, **_kwargs):
        tracker.state_token = "temporary-interrupted"
        raise interrupt_type()

    monkeypatch.setattr(tracker, "initialize", interrupted_initialize)
    with pytest.raises(interrupt_type) as raised:
        provider._initial_sam_halo_cleanup(
            frame, raw, source="online_sam2_box"
        )

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "restore double fault" in str(raised.value.__cause__)
    assert tracker.restore_calls == 1


def test_initial_sam_halo_cleanup_restores_real_adaptive_tracker_snapshot():
    """The appearance probe may change only the monotonic ABA revision."""

    tracker = AdaptiveColorDepthTracker(
        {
            "min_area": 20,
            "depth_tolerance": 0.04,
            "max_depth_tolerance": 0.08,
            "flow_color_probability_floor": 0.42,
        }
    )
    initial_color = np.full((80, 96, 3), 30, dtype=np.uint8)
    initial_color[15:35, 10:30] = (0, 220, 0)
    initial_frame = _initial_seed_frame(frame_id=0)
    initial_frame = replace(initial_frame, color_bgr=initial_color)
    initial_mask = np.zeros(initial_frame.depth_raw.shape, dtype=np.uint8)
    initial_mask[15:35, 10:30] = 1
    tracker.initialize(initial_frame, mask=initial_mask)

    before = tracker.snapshot_state()
    revision_before = tracker.appearance_revision
    cleanup_color = np.full((80, 96, 3), 30, dtype=np.uint8)
    cleanup_color[20:60, 24:64] = (0, 220, 0)
    cleanup_frame = replace(
        _initial_seed_frame(frame_id=1), color_bgr=cleanup_color
    )
    raw = np.zeros(cleanup_frame.depth_raw.shape, dtype=np.uint8)
    raw[19:61, 23:65] = 1
    provider = _initial_seed_provider(tracker)

    cleaned = provider._initial_sam_halo_cleanup(
        cleanup_frame, raw, source="online_sam2_box"
    )
    after = tracker.snapshot_state()

    def assert_snapshot_value_equal(first, second):
        if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
            np.testing.assert_array_equal(first, second)
            return
        fields = getattr(first, "__dataclass_fields__", None)
        if fields is not None:
            assert type(first) is type(second)
            for name in fields:
                assert_snapshot_value_equal(
                    getattr(first, name), getattr(second, name)
                )
            return
        if isinstance(first, (tuple, list)):
            assert type(first) is type(second)
            assert len(first) == len(second)
            for first_item, second_item in zip(first, second):
                assert_snapshot_value_equal(first_item, second_item)
            return
        assert first == second

    assert_snapshot_value_equal(before, after)
    assert tracker.appearance_revision == revision_before + 2
    assert provider.last_initial_sam_halo_cleanup_evidence.applied
    assert int(cleaned.sum()) == 1600


def test_initial_sam_halo_cleanup_is_not_a_general_mask_or_recovery_path(
    monkeypatch,
):
    frame = _frame(1)
    seed = _mask(4, 5, 13, 15)
    manager = _FakeOnlineSAM2Manager()
    tracker = _FakeTracker({})
    provider, _camera = _provider(monkeypatch, [frame], tracker, manager, start=True)
    calls = []
    monkeypatch.setattr(
        provider,
        "_initial_sam_halo_cleanup",
        lambda *_args, **_kwargs: calls.append("called"),
    )
    try:
        assert provider.initialize_from_mask(frame, seed, source="external_prompt")
        assert calls == []
    finally:
        provider.stop()


def test_initial_sam_cleanup_canonical_bytes_bind_every_initial_owner(
    monkeypatch,
):
    frame = _frame(1)
    raw = _mask(4, 5, 13, 15)
    canonical = raw.copy()
    canonical[5, 4] = 0
    manager = _FakeOnlineSAM2Manager(
        {
            2: _video_result(2, raw),
            3: _video_result(3, raw),
        }
    )
    tracker = _FakeTracker({})
    provider, _camera = _provider(monkeypatch, [frame], tracker, manager, start=True)
    provider.online_sam2_cfg["require_for_bbox_init"] = True

    def canonical_cleanup(*_args, **_kwargs):
        frozen, _mask_bytes, _digest = provider._frozen_binary_mask(canonical)
        return frozen

    monkeypatch.setattr(provider, "_initial_sam_halo_cleanup", canonical_cleanup)
    try:
        assert provider.initialize_from_bbox(
            frame, np.asarray([4, 5, 13, 15], dtype=np.int32)
        )
        remote_seed = [call for call in manager.calls if call[0] == "initialize"][-1][
            3
        ]
        local_seed = tracker.initialize_calls[-1][1]
        np.testing.assert_array_equal(remote_seed, canonical)
        np.testing.assert_array_equal(local_seed, canonical)
        np.testing.assert_array_equal(provider.last_mask_result.mask, canonical)
        np.testing.assert_array_equal(
            provider._last_final_clean_publication.mask, canonical
        )
        assert provider._full_target_geometry_authority.area == float(
            canonical.sum()
        )
        assert provider._full_target_geometry_authority.bbox_area == 90.0
        np.testing.assert_array_equal(
            provider._trusted_online_sam2_publication.mask, canonical
        )
        assert provider._target_initial_area == float(canonical.sum())
        assert provider.last_bbox_initialization_evidence.mask_area_px == int(
            canonical.sum()
        )
    finally:
        provider.stop()


def test_initial_sam_remote_seed_is_reset_if_local_commit_fails(monkeypatch):
    frame = _frame(1)
    raw = _mask(4, 5, 13, 15)
    manager = _FakeOnlineSAM2Manager(
        {
            2: _video_result(2, raw),
            3: _video_result(3, raw),
        }
    )
    tracker = _FakeTracker({})
    provider, _camera = _provider(monkeypatch, [frame], tracker, manager, start=True)
    provider.online_sam2_cfg["require_for_bbox_init"] = True
    history_resets = provider.history.reset_calls
    monkeypatch.setattr(provider, "initialize_from_mask", lambda *_a, **_k: False)
    try:
        assert not provider.initialize_from_bbox(
            frame, np.asarray([4, 5, 13, 15], dtype=np.int32)
        )
        assert [call[0] for call in manager.calls][-2:] == ["initialize", "reset"]
        assert not provider._online_sam2_initialized
        assert not provider._online_sam2_seed_pending
        assert provider.history.reset_calls == history_resets
        assert provider.last_bbox_initialization_evidence is None
    finally:
        provider.stop()


def test_initial_sam_remote_exact_seed_byte_mismatch_fails_before_local_commit(
    monkeypatch,
):
    frame = _frame(1)
    raw = _mask(4, 5, 13, 15)
    mismatched = raw.copy()
    mismatched[5, 4] = 0
    manager = _FakeOnlineSAM2Manager(
        {
            2: _video_result(2, raw),
            3: _video_result(3, raw),
        },
        initialize_results={1: _video_result(1, mismatched)},
    )
    tracker = _FakeTracker({})
    provider, _camera = _provider(monkeypatch, [frame], tracker, manager, start=True)
    provider.online_sam2_cfg["require_for_bbox_init"] = True
    try:
        assert not provider.initialize_from_bbox(
            frame, np.asarray([4, 5, 13, 15], dtype=np.int32)
        )
        assert tracker.initialize_calls == []
        assert [call[0] for call in manager.calls][-2:] == ["initialize", "reset"]
        assert "exact_seed_match=False" in (
            provider.last_online_sam2_bbox_prewarm_evidence.status
        )
        assert provider.last_bbox_initialization_evidence is None
    finally:
        provider.stop()


def test_bbox_remote_seed_is_reset_once_when_depth_refine_fails(monkeypatch):
    frame = _frame(1)
    manager = _FakeOnlineSAM2Manager()
    provider, _camera = _provider(
        monkeypatch, [frame], _FakeTracker({}), manager, start=True
    )
    provider.online_sam2_cfg["require_for_bbox_init"] = True
    monkeypatch.setattr(
        provider,
        "_refine_online_sam2_box_mask_with_depth",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic depth-refine failure")
        ),
    )
    try:
        assert not provider.initialize_from_bbox(
            frame, np.asarray([4, 5, 13, 15], dtype=np.int32)
        )
        operations = [call[0] for call in manager.calls]
        assert operations == ["start", "initialize_box", "reset"]
        assert operations.count("reset") == 1
        assert provider.last_bbox_initialization_evidence is None
        assert not provider._online_sam2_initialized
    finally:
        provider.stop()


def test_bbox_live_but_semantically_invalid_remote_seed_is_reset(monkeypatch):
    frame = _frame(1)
    manager = _FakeOnlineSAM2Manager()
    empty = np.zeros(frame.depth_raw.shape, dtype=np.uint8)

    def invalid_initialize_box(image_bgr, bbox_xyxy, frame_id, **_kwargs):
        manager._record(
            "initialize_box",
            int(frame_id),
            image_bgr.copy(),
            np.asarray(bbox_xyxy, dtype=np.int32).copy(),
        )
        return SAM2VideoResult(
            mask=empty.copy(),
            valid=False,
            frame_id=int(frame_id),
            message="synthetic negative object score",
            timings_ms={"service_total": 1.0},
            internal_frame_idx=0,
            object_score=-1.0,
            mask_area=0,
        )

    monkeypatch.setattr(manager, "initialize_box", invalid_initialize_box)
    provider, _camera = _provider(
        monkeypatch, [frame], _FakeTracker({}), manager, start=True
    )
    provider.online_sam2_cfg["require_for_bbox_init"] = True
    try:
        assert not provider.initialize_from_bbox(
            frame, np.asarray([4, 5, 13, 15], dtype=np.int32)
        )
        operations = [call[0] for call in manager.calls]
        assert operations == ["start", "initialize_box", "reset"]
        assert operations.count("reset") == 1
        assert not provider._online_sam2_initialized
    finally:
        provider.stop()


def test_bbox_keyboard_interrupt_resets_remote_once_then_reraises(monkeypatch):
    frame = _frame(1)
    manager = _FakeOnlineSAM2Manager()
    provider, _camera = _provider(
        monkeypatch, [frame], _FakeTracker({}), manager, start=True
    )
    provider.online_sam2_cfg["require_for_bbox_init"] = True
    monkeypatch.setattr(
        provider,
        "_initial_sam_halo_cleanup",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    try:
        with pytest.raises(KeyboardInterrupt):
            provider.initialize_from_bbox(
                frame, np.asarray([4, 5, 13, 15], dtype=np.int32)
            )
        operations = [call[0] for call in manager.calls]
        assert operations == ["start", "initialize_box", "reset"]
        assert operations.count("reset") == 1
        assert provider.last_bbox_initialization_evidence is None
        assert not provider._online_sam2_initialized
    finally:
        provider.stop()


def test_bbox_early_failure_clears_previous_initialization_evidence(monkeypatch):
    frame = _frame(1)
    manager = _FakeOnlineSAM2Manager()
    provider, _camera = _provider(
        monkeypatch, [frame], _FakeTracker({}), manager, start=True
    )
    provider.online_sam2_cfg["require_for_bbox_init"] = True
    provider.last_bbox_initialization_evidence = SimpleNamespace(valid=True)
    provider.last_initial_sam_halo_cleanup_evidence = SimpleNamespace(
        attempted=True, applied=True, status="stale success"
    )
    monkeypatch.setattr(
        provider,
        "_initialize_bbox_with_online_sam2",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic early failure")
        ),
    )
    try:
        assert not provider.initialize_from_bbox(
            frame, np.asarray([4, 5, 13, 15], dtype=np.int32)
        )
        assert provider.last_bbox_initialization_evidence is None
        cleanup = provider.last_initial_sam_halo_cleanup_evidence
        assert not cleanup.attempted
        assert not cleanup.applied
        assert cleanup.status == "bbox_initialization_not_reached"
        assert [call[0] for call in manager.calls] == ["start"]
    finally:
        provider.stop()


def test_initialize_from_mask_rolls_back_local_canonical_byte_tamper(monkeypatch):
    frame = _frame(1)
    seed = _mask(4, 5, 13, 15)
    manager = _FakeOnlineSAM2Manager()
    tracker = _FakeTracker({})
    provider, _camera = _provider(monkeypatch, [frame], tracker, manager, start=True)
    history_resets = provider.history.reset_calls

    def tampering_initialize(_frame_value, mask, bbox_xyxy):
        tampered = (np.asarray(mask) > 0).astype(np.uint8)
        tampered[5, 4] = 0
        tracker.token = "tampered"
        result = _mask_result(tampered, message="synthetic tamper")
        result.bbox_xyxy = np.asarray(bbox_xyxy, dtype=np.int32).copy()
        tracker._set_state(result)
        return result

    monkeypatch.setattr(tracker, "initialize", tampering_initialize)
    try:
        assert not provider.initialize_from_mask(frame, seed)
        assert tracker.token == "new"
        assert tracker.state is None
        assert provider.history.reset_calls == history_resets
        assert provider.last_mask_result is None
        assert provider.last_packet is None
        assert [call[0] for call in manager.calls] == ["start"]
    finally:
        provider.stop()


def _provider(
    monkeypatch,
    frames,
    tracker,
    manager,
    *,
    lifecycle=None,
    start=True,
    runtime_frame_timeout_ms=None,
    recovery_publication_mode=None,
):
    import dynamic_pcd.provider.object_pcd_provider as provider_module

    camera = _FakeCamera(frames, lifecycle=lifecycle)
    monkeypatch.setattr(provider_module, "RealSenseCamera", lambda _cfg: camera)
    monkeypatch.setattr(
        provider_module, "AdaptiveColorDepthTracker", lambda _cfg: tracker
    )
    monkeypatch.setattr(
        provider_module,
        "ObjectPointCloudExtractor",
        lambda _cfg, T_base_camera=None: _FakeExtractor(),
    )
    monkeypatch.setattr(
        provider_module,
        "PCDHistoryBuffer",
        lambda history_len, max_center_jump: _FakeHistory(),
    )

    camera_overrides = {"serial": None}
    if runtime_frame_timeout_ms is not None:
        camera_overrides["runtime_frame_timeout_ms"] = runtime_frame_timeout_ms
    tracker_overrides = {
        "mode": "adaptive_color_depth",
        "recovery_enabled": False,
    }
    if recovery_publication_mode is not None:
        tracker_overrides["recovery_publication_mode"] = recovery_publication_mode
    cfg = load_config(
        overrides={
            "camera": camera_overrides,
            "tracker": tracker_overrides,
            "sam2": {"enabled": False},
            "online_sam2": {
                "enabled": True,
                "service_addr": "tcp://127.0.0.1:15558",
                "service_autostart": False,
                "startup_timeout_s": 1.0,
                "request_timeout_s": 0.5,
                "frame_wait_timeout_s": 0.03,
            },
            "extrinsics": {
                "calibration_file": None,
                "require_calibration": False,
                "require_quality_pass": False,
                "strict_camera_serial": False,
            },
            "pointcloud": {"remove_outliers": False},
        }
    )
    factory_calls = []

    def factory(*args, **kwargs):
        factory_calls.append((args, kwargs))
        return manager

    provider = ObjectPCDProvider(cfg, online_sam2_manager_factory=factory)
    if start:
        provider.start()
    assert factory_calls
    return provider, camera


def test_required_first_frame_uses_online_sam2_box_prompt(monkeypatch):
    frame = _frame(1)
    box_mask = _mask(4, 5, 13, 15)
    manager = _FakeOnlineSAM2Manager(
        {
            2: _video_result(2, box_mask),
            3: _video_result(3, box_mask),
        }
    )
    tracker = _FakeTracker({})
    provider, _camera = _provider(monkeypatch, [frame], tracker, manager, start=True)
    provider.online_sam2_cfg["require_for_bbox_init"] = True

    try:
        assert provider.initialize_from_bbox(
            frame, np.asarray([4, 5, 13, 15], dtype=np.int32)
        )
        operations = [call[0] for call in manager.calls]
        assert operations[:5] == [
            "start",
            "initialize_box",
            "track",
            "track",
            "initialize",
        ]
        # The duplicate-frame result only warms temporal propagate. The exact
        # accepted seed is restored on its original frame before local state
        # can be committed.
        assert manager.calls[2][1] == 2
        assert manager.calls[3][1] == 3
        # Replace the raw box-prompt temporal seed with the exact mask accepted
        # after current-frame RGB-D foreground refinement.
        assert manager.calls[4][1] == 1
        np.testing.assert_array_equal(manager.calls[4][3], box_mask)
        box_call = manager.calls[1]
        np.testing.assert_array_equal(
            box_call[3], np.asarray([4, 5, 13, 15], dtype=np.int32)
        )
        assert provider.last_bbox_initialization_evidence.source == ("online_sam2_box")
        assert provider.last_bbox_initialization_evidence.mask_area_px == 90
        prewarm = provider.last_online_sam2_bbox_prewarm_evidence
        assert prewarm.enabled and prewarm.attempted and prewarm.passed
        assert prewarm.seed_frame_id == 1
        assert prewarm.discarded_track_frame_id == 3
        assert prewarm.discarded_track_frame_ids == (2, 3)
        assert prewarm.attempt_count == 2
        assert prewarm.achieved_stable_tracks == 2
        assert prewarm.required_stable_tracks == 2
        assert prewarm.status == (
            "discarded_tracks_stable_then_exact_seed_restored"
        )
    finally:
        provider.stop()


def test_bbox_prewarm_track_failure_is_atomic_and_fail_closed(monkeypatch):
    frame = _frame(1)
    manager = _FakeOnlineSAM2Manager({2: RuntimeError("synthetic cold-track failure")})
    tracker = _FakeTracker({})
    provider, _camera = _provider(monkeypatch, [frame], tracker, manager, start=True)
    provider.online_sam2_cfg["require_for_bbox_init"] = True
    history_resets = provider.history.reset_calls

    try:
        assert not provider.initialize_from_bbox(
            frame, np.asarray([4, 5, 13, 15], dtype=np.int32)
        )
        assert [call[0] for call in manager.calls] == [
            "start",
            "initialize_box",
            "track",
            "reset",
        ]
        assert not tracker.initialized
        assert tracker.token == "new"
        assert provider.history.reset_calls == history_resets
        assert provider.last_mask_result is None
        assert provider.last_packet is None
        assert provider.last_bbox_initialization_evidence is None
        evidence = provider.last_online_sam2_bbox_prewarm_evidence
        assert evidence.attempted and not evidence.passed
        assert "synthetic cold-track failure" in evidence.status
    finally:
        provider.stop()


def test_bbox_exact_reseed_failure_is_atomic_and_fail_closed(monkeypatch):
    frame = _frame(1)
    box_mask = _mask(4, 5, 13, 15)
    manager = _FakeOnlineSAM2Manager(
        {
            2: _video_result(2, box_mask),
            3: _video_result(3, box_mask),
        },
        initialize_results={1: RuntimeError("synthetic exact-reseed failure")},
    )
    tracker = _FakeTracker({})
    provider, _camera = _provider(monkeypatch, [frame], tracker, manager, start=True)
    provider.online_sam2_cfg["require_for_bbox_init"] = True
    history_resets = provider.history.reset_calls

    try:
        assert not provider.initialize_from_bbox(
            frame, np.asarray([4, 5, 13, 15], dtype=np.int32)
        )
        assert [call[0] for call in manager.calls] == [
            "start",
            "initialize_box",
            "track",
            "track",
            "initialize",
            "reset",
        ]
        assert not tracker.initialized
        assert provider.history.reset_calls == history_resets
        assert provider.last_mask_result is None
        assert provider.last_bbox_initialization_evidence is None
        evidence = provider.last_online_sam2_bbox_prewarm_evidence
        assert evidence.attempted and not evidence.passed
        assert "synthetic exact-reseed failure" in evidence.status
    finally:
        provider.stop()


def test_bbox_exact_mask_reseed_rejects_nonempty_semantically_invalid_result(
    monkeypatch,
):
    frame = _frame(1)
    box_mask = _mask(4, 5, 13, 15)
    # In production SAM2, an explicit nonempty mask is a direct-mask output
    # with object_score=+10. A nonempty but semantically invalid result is an
    # invariant breach and must remain fail-closed.
    explicit_mask_seed = SAM2VideoResult(
        mask=box_mask.copy(),
        valid=False,
        frame_id=1,
        message="semantic object score below threshold",
        timings_ms={"service_total": 3.25},
        internal_frame_idx=0,
        object_score=-0.25,
        mask_area=int(box_mask.sum()),
    )
    manager = _FakeOnlineSAM2Manager(
        {
            2: _video_result(2, box_mask),
            3: _video_result(3, box_mask),
        },
        initialize_results={1: explicit_mask_seed},
    )
    tracker = _FakeTracker({})
    provider, _camera = _provider(monkeypatch, [frame], tracker, manager, start=True)
    provider.online_sam2_cfg["require_for_bbox_init"] = True

    try:
        assert not provider.initialize_from_bbox(
            frame, np.asarray([4, 5, 13, 15], dtype=np.int32)
        )
        evidence = provider.last_online_sam2_bbox_prewarm_evidence
        assert not evidence.passed
        assert "session_alive=True" in evidence.status
        assert "semantic_valid=False" in evidence.status
        assert "mask_area=90" in evidence.status
        assert "object_score=-0.25" in evidence.status
        assert not provider._online_sam2_initialized
        # No local target was committed, so the provisional remote session is
        # reset instead of scheduling a retry against a nonexistent owner.
        assert not provider._online_sam2_seed_pending
        assert manager.calls[-1][0] == "reset"
    finally:
        provider.stop()


def test_bbox_exact_reseed_empty_mask_reports_complete_diagnostics(monkeypatch):
    frame = _frame(1)
    box_mask = _mask(4, 5, 13, 15)
    empty = np.zeros_like(box_mask)
    empty_seed = SAM2VideoResult(
        mask=empty,
        valid=False,
        frame_id=1,
        message="synthetic empty explicit-mask output",
        timings_ms={"service_total": 3.25},
        internal_frame_idx=0,
        object_score=-0.5,
        mask_area=0,
    )
    manager = _FakeOnlineSAM2Manager(
        {
            2: _video_result(2, box_mask),
            3: _video_result(3, box_mask),
        },
        initialize_results={1: empty_seed},
    )
    tracker = _FakeTracker({})
    provider, _camera = _provider(monkeypatch, [frame], tracker, manager, start=True)
    provider.online_sam2_cfg["require_for_bbox_init"] = True

    try:
        assert not provider.initialize_from_bbox(
            frame, np.asarray([4, 5, 13, 15], dtype=np.int32)
        )
        status = provider.last_online_sam2_bbox_prewarm_evidence.status
        assert "seed_input_area=90" in status
        assert "session_alive=True" in status
        assert "semantic_valid=False" in status
        assert "mask_area=0" in status
        assert "reported_mask_area=0" in status
        assert "object_score=-0.5" in status
        assert "synthetic empty explicit-mask output" in status
        assert not tracker.initialized
    finally:
        provider.stop()


def test_bbox_prewarm_can_be_disabled_for_historical_ab(monkeypatch):
    frame = _frame(1)
    manager = _FakeOnlineSAM2Manager()
    tracker = _FakeTracker({})
    provider, _camera = _provider(monkeypatch, [frame], tracker, manager, start=True)
    provider.online_sam2_cfg["require_for_bbox_init"] = True
    provider.online_sam2_cfg["bbox_init_discarded_track_prewarm"] = False

    try:
        assert provider.initialize_from_bbox(
            frame, np.asarray([4, 5, 13, 15], dtype=np.int32)
        )
        assert [call[0] for call in manager.calls] == [
            "start",
            "initialize_box",
            "initialize",
        ]
        evidence = provider.last_online_sam2_bbox_prewarm_evidence
        assert not evidence.enabled
        assert not evidence.attempted
        assert not evidence.passed
        assert evidence.status == "disabled_ab_path"
    finally:
        provider.stop()


def test_bbox_prewarm_consumes_first_cold_track_before_rollout(monkeypatch):
    seed = _mask(4, 5, 13, 15)
    current = _mask(5, 5, 14, 15)
    release_cold_track = threading.Event()
    manager = _FakeOnlineSAM2Manager(
        {
            2: _video_result(2, current),
            3: _video_result(3, current),
            4: _video_result(4, current),
        }
    )
    manager.track_wait_events[2] = release_cold_track
    provider, _camera = _provider(
        monkeypatch,
        [_frame(2)],
        _FakeTracker({2: _mask_result(current)}),
        manager,
        start=True,
        recovery_publication_mode="unified_three_evidence",
    )
    provider.online_sam2_cfg["require_for_bbox_init"] = True
    provider.online_sam2_cfg["mask_publication_mode"] = "semantic_sam2"
    release_timer = threading.Timer(0.060, release_cold_track.set)
    release_timer.start()

    try:
        assert provider.initialize_from_bbox(
            _frame(1), np.asarray([4, 5, 13, 15], dtype=np.int32)
        )
        # The 60 ms first propagate happened only in camera-only init. The
        # first formal tick uses the restored frame-1 seed and publishes its
        # exact frame-2 result without deadline/busy fallout.
        frame, result = provider.step_mask_only()
        assert frame.frame_id == 2
        assert result.valid
        np.testing.assert_array_equal(result.mask, current)
        assert provider._online_sam2_deadline_miss_count == 0
        assert provider._online_sam2_busy_skip_count == 0
        assert [call[0] for call in manager.calls] == [
            "start",
            "initialize_box",
            "track",
            "track",
            "track",
            "initialize",
            "track",
        ]
        assert provider.last_online_sam2_bbox_prewarm_evidence.total_ms >= 50.0
        assert provider.last_online_sam2_bbox_prewarm_evidence.attempt_count == 3
        assert (
            provider.last_online_sam2_bbox_prewarm_evidence.achieved_stable_tracks
            == 2
        )
        # The dummy result was not committed to any robot-facing owner.
        assert provider._trusted_online_sam2_publication.frame_id == 1
    finally:
        release_timer.cancel()
        release_cold_track.set()
        provider.stop()


def test_bbox_prewarm_fails_closed_without_consecutive_hot_tracks(monkeypatch):
    frame = _frame(1)
    box_mask = _mask(4, 5, 13, 15)
    manager = _FakeOnlineSAM2Manager(
        {
            frame_id: _video_result(frame_id, box_mask)
            for frame_id in range(2, 8)
        }
    )
    tracker = _FakeTracker({})
    provider, _camera = _provider(monkeypatch, [frame], tracker, manager, start=True)
    provider.online_sam2_cfg["require_for_bbox_init"] = True
    real_timed_call = provider._timed_online_call

    def force_slow_tracks(call, *args, **kwargs):
        outcome = real_timed_call(call, *args, **kwargs)
        if getattr(call, "__name__", "") == "track":
            return outcome[0], 41.0, outcome[2]
        return outcome

    monkeypatch.setattr(provider, "_timed_online_call", force_slow_tracks)
    history_resets = provider.history.reset_calls

    try:
        assert not provider.initialize_from_bbox(
            frame, np.asarray([4, 5, 13, 15], dtype=np.int32)
        )
        assert [call[0] for call in manager.calls] == [
            "start",
            "initialize_box",
            "track",
            "track",
            "track",
            "track",
            "track",
            "track",
            "reset",
        ]
        assert not tracker.initialized
        assert provider.history.reset_calls == history_resets
        assert provider.last_mask_result is None
        assert provider.last_packet is None
        evidence = provider.last_online_sam2_bbox_prewarm_evidence
        assert evidence.attempt_count == 6
        assert evidence.achieved_stable_tracks == 0
        assert evidence.discarded_track_rpc_ms_values == (41.0,) * 6
        assert not evidence.passed
        assert "did not reach consecutive hot latency" in evidence.status
    finally:
        provider.stop()


def test_first_sam2_mask_is_clipped_to_tight_operator_prompt(monkeypatch):
    frame = _frame(1)
    manager = _FakeOnlineSAM2Manager()
    provider, _camera = _provider(
        monkeypatch, [frame], _FakeTracker({}), manager, start=True
    )
    prompt = np.asarray([8, 6, 16, 14], dtype=np.int32)
    leaked = _mask(3, 2, 23, 21)
    result = _mask_result(leaked, message="SAM2 leaked beyond tight prompt")

    try:
        refined = provider._refine_online_sam2_box_mask_with_depth(
            frame,
            result,
            prompt_bbox_xyxy=prompt,
        )
        ys, xs = np.nonzero(refined.mask)
        margin = int(provider.online_sam2_cfg["bbox_init_clip_margin_px"])
        assert int(xs.min()) >= int(prompt[0]) - margin
        assert int(ys.min()) >= int(prompt[1]) - margin
        assert int(xs.max()) < int(prompt[2]) + margin
        assert int(ys.max()) < int(prompt[3]) + margin
        assert int(refined.mask.sum()) < int(leaked.sum())
    finally:
        provider.stop()


def test_deployed_first_sam2_mask_keeps_semantic_edge_with_mixed_depth(monkeypatch):
    frame = _frame(1)
    depth_raw = frame.depth_raw.copy()
    depth_raw[5:7, 8:16] = 1100
    frame = RGBDFrame(
        color_bgr=frame.color_bgr,
        depth_raw=depth_raw,
        depth_scale=frame.depth_scale,
        intrinsics=frame.intrinsics,
        timestamp=frame.timestamp,
        frame_id=frame.frame_id,
    )
    semantic = _mask(8, 5, 16, 14)
    provider, _camera = _provider(
        monkeypatch, [frame], _FakeTracker({}), _FakeOnlineSAM2Manager(), start=True
    )

    try:
        assert provider.online_sam2_cfg["bbox_init_depth_refine_enabled"] is False
        result = provider._refine_online_sam2_box_mask_with_depth(
            frame,
            _mask_result(semantic),
            prompt_bbox_xyxy=np.asarray([8, 5, 16, 14], dtype=np.int32),
        )
        np.testing.assert_array_equal(result.mask, semantic)
    finally:
        provider.stop()


@pytest.mark.parametrize(
    ("configured_timeout_ms", "expected_timeout_ms"),
    ((None, 1000), (100, 100)),
)
def test_mask_pipeline_passes_bounded_runtime_frame_timeout(
    monkeypatch,
    configured_timeout_ms,
    expected_timeout_ms,
):
    seed = _mask(4, 5, 12, 15)
    adaptive = _mask(5, 5, 13, 15)
    provider, camera = _provider(
        monkeypatch,
        [_frame(2)],
        _FakeTracker({2: _mask_result(adaptive)}),
        _FakeOnlineSAM2Manager(),
        runtime_frame_timeout_ms=configured_timeout_ms,
    )
    assert provider.initialize_from_mask(_frame(1), seed)

    try:
        frame, _result = provider.step_mask_only()
        assert frame.frame_id == 2
        assert provider.runtime_frame_timeout_ms == expected_timeout_ms
        assert camera.requested_timeouts_ms == [expected_timeout_ms]
    finally:
        provider.stop()


def test_deployed_d435_runtime_frame_timeout_is_100ms():
    cfg = load_config(str(DEPLOYED_CONFIG))
    assert cfg["camera"]["runtime_frame_timeout_ms"] == 100
    assert _validated_runtime_frame_timeout_ms(cfg["camera"]) == 100


@pytest.mark.parametrize(
    "value",
    (True, False, None, 0, -1, 1001, 100.0, "100", np.nan),
)
def test_runtime_frame_timeout_rejects_non_integer_or_out_of_range(value):
    with pytest.raises(
        ValueError,
        match=r"camera\.runtime_frame_timeout_ms must be an integer in 1\.\.1000",
    ):
        _validated_runtime_frame_timeout_ms({"runtime_frame_timeout_ms": value})


def test_online_sam2_is_enabled_by_default_and_exposed_by_cli():
    for cfg in (load_config(), load_config(str(DEPLOYED_CONFIG))):
        online_cfg = cfg["online_sam2"]
        assert cfg["tracker"]["interactive_roi_padding_px"] == 0
        assert online_cfg["enabled"] is True
        assert online_cfg["vos_optimized"] is True
        assert online_cfg["vos_compile_mode"] == (
            "max-autotune-no-cudagraphs"
        )
        assert online_cfg["guarded_v2_semantic_primary"] is True
        assert online_cfg["guarded_v2_bootstrap_commission_enabled"] is True
        assert online_cfg["guarded_v2_bootstrap_max_frame_delta"] == 12
        assert online_cfg["guarded_v2_bootstrap_max_elapsed_s"] == 0.40
        assert online_cfg["guarded_v2_bootstrap_candidate_max_frame_gap"] == 3
        assert online_cfg["guarded_v2_bootstrap_candidate_max_timestamp_gap_s"] == 0.20
        assert online_cfg["guarded_v2_bootstrap_stable_max_area_ratio"] == 1.05
        assert online_cfg["guarded_v2_bootstrap_stable_max_extent_ratio"] == 1.08
        assert (
            online_cfg[
                "guarded_v2_bootstrap_max_registered_contraction_fraction"
            ]
            == 0.05
        )
        assert online_cfg["guarded_v2_bootstrap_min_supported_extent_ratio"] == 0.90
        assert (
            online_cfg["guarded_v2_bootstrap_max_supported_centroid_offset_ratio"]
            == 0.075
        )
        assert (
            online_cfg[
                "guarded_v2_bootstrap_partial_seed_max_supported_centroid_offset_ratio"
            ]
            == 0.10
        )
        assert (
            online_cfg["guarded_v2_bootstrap_max_unsupported_bbox_margin_ratio"]
            == 0.075
        )
        assert online_cfg["bbox_init_depth_refine_enabled"] is False
        assert online_cfg["bbox_init_clip_to_prompt"] is True
        assert online_cfg["bbox_init_clip_margin_px"] == 2
        assert cfg["tracker"][
            "publication_full_target_min_aligned_coverage"
        ] == pytest.approx(0.90)
        assert cfg["tracker"][
            "publication_full_scale_transition_min_normalized_coverage"
        ] == pytest.approx(0.85)
        assert cfg["tracker"][
            "publication_full_target_min_appearance_support_ratio"
        ] == pytest.approx(0.90)
        assert cfg["tracker"][
            "publication_boundary_exit_continuation_min_appearance_support_ratio"
        ] == pytest.approx(0.55)
        assert online_cfg["tracked_frame_wait_timeout_s"] == 0.045
        assert online_cfg["frame_wait_timeout_s"] == 0.045
        assert online_cfg["bbox_init_discarded_track_prewarm"] is True
        assert online_cfg["bbox_init_prewarm_max_attempts"] == 6
        assert online_cfg["bbox_init_prewarm_required_stable_tracks"] == 2
        assert online_cfg["bbox_init_prewarm_max_rpc_ms"] == 40.0
        assert online_cfg["tracked_publication_source"] == "sam2"
        assert online_cfg["tracked_min_area_ratio"] == 0.60
        assert online_cfg["tracked_max_area_ratio"] == 1.60
        assert online_cfg["tracked_min_iou"] == 0.40
        assert online_cfg["tracked_min_initial_bbox_area_ratio"] == 0.45
        assert online_cfg["tracked_max_initial_bbox_area_ratio"] == 1.35
        assert online_cfg["tracked_extra_core_dilation_px"] == 2
        assert online_cfg["tracked_extra_appearance_min_pixels"] == 24
        assert online_cfg["tracked_extra_appearance_min_fraction"] == 0.04
        assert online_cfg["tracked_extra_appearance_probability_threshold"] == 0.42
        assert online_cfg["tracked_extra_min_appearance_support_ratio"] == 0.65
        assert online_cfg["tracked_extra_min_appearance_mean"] == 0.45
        assert online_cfg["trusted_max_center_speed_px_s"] == 2400.0
        assert online_cfg["trusted_max_log_area_rate_s"] == 12.0
        assert online_cfg["recovery_min_area_ratio"] == 0.40
        assert online_cfg["recovery_max_area_ratio"] == 1.80
        assert online_cfg["recovery_min_bbox_area_ratio"] == 0.35
        assert online_cfg["recovery_max_bbox_area_ratio"] == 2.20
        assert online_cfg["recovery_min_valid_depth_ratio"] == 0.35
        assert online_cfg["recovery_max_depth_spread_m"] == 0.18
        assert online_cfg["recovery_confirm_frames"] == 2
        assert online_cfg["recovery_confirm_max_frame_gap"] == 3
        assert cfg["tracker"]["recovery_publish_confirm_frames"] == 3
        assert cfg["tracker"]["recovery_publish_confirm_max_area_ratio"] == 1.35

    deployed_cfg = load_config(str(DEPLOYED_CONFIG))
    assert deployed_cfg["tracker"]["recovery_publication_mode"] == (
        "unified_three_evidence"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dynamic_pcd.apps.realtime_masked_pcd",
            "--help",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=15.0,
    )
    assert "--online_sam2" in result.stdout
    assert "--no_online_sam2" in result.stdout
    assert "--online_sam2_image_size" in result.stdout
    assert "--online_sam2_frame_wait_ms" in result.stdout


@pytest.mark.parametrize(
    ("value", "expected_flag"),
    ((True, "--vos-optimized"), (False, "--no-vos-optimized")),
)
def test_online_sam2_launcher_identity_includes_vos_optimized_mode(
    value: bool, expected_flag: str
):
    provider = object.__new__(ObjectPCDProvider)
    provider.cfg = {"camera": {"width": 848, "height": 480}}
    provider.online_sam2_cfg = {
        "checkpoint": "/tmp/model.pt",
        "vos_optimized": value,
    }

    args = provider._online_sam2_launcher_args()

    assert expected_flag in args
    assert args[args.index("--vos-compile-mode") + 1] == (
        "max-autotune-no-cudagraphs"
    )
    assert args[args.index("--prewarm-input-width") + 1] == "848"
    assert args[args.index("--prewarm-input-height") + 1] == "480"
    other = (
        "--no-vos-optimized"
        if expected_flag == "--vos-optimized"
        else "--vos-optimized"
    )
    assert other not in args


@pytest.mark.parametrize("value", [None, 0, 1, "true", np.bool_(True)])
def test_online_sam2_vos_optimized_config_requires_a_boolean(value):
    cfg = dict(load_config()["online_sam2"])
    cfg["vos_optimized"] = value

    with pytest.raises(
        ValueError, match="online_sam2.vos_optimized must be true or false"
    ):
        validate_online_sam2_config(cfg)


@pytest.mark.parametrize("value", [None, "", "reduce-overhead", 1])
def test_online_sam2_config_rejects_unknown_vos_compile_mode(value):
    cfg = dict(load_config()["online_sam2"])
    cfg["vos_compile_mode"] = value

    with pytest.raises(
        ValueError, match="online_sam2.vos_compile_mode must be one of"
    ):
        validate_online_sam2_config(cfg)


def test_online_sam2_production_rejects_diagnostic_cudagraph_mode():
    cfg = dict(load_config()["online_sam2"])
    cfg["vos_compile_mode"] = "max-autotune"

    with pytest.raises(
        ValueError, match="optimized production runtime requires"
    ):
        validate_online_sam2_config(cfg)


def test_provider_cold_starts_online_sam2_before_camera_readiness(monkeypatch):
    lifecycle = []
    tracker = _FakeTracker({})
    manager = _FakeOnlineSAM2Manager(lifecycle=lifecycle)

    provider, camera = _provider(
        monkeypatch,
        [_frame(1)],
        tracker,
        manager,
        lifecycle=lifecycle,
    )
    try:
        assert lifecycle == ["online_sam2_start", "camera_start"]
        assert manager.calls[0][0] == "start"
        assert provider._online_sam2_service_health["vos_optimized"] is True
        assert camera.started
    finally:
        provider.stop()


def test_provider_rejects_reused_service_with_wrong_vos_optimized_identity(
    monkeypatch,
):
    class WrongIdentityManager(_FakeOnlineSAM2Manager):
        def start(self):
            health = super().start()
            health["vos_optimized"] = False
            return health

    manager = WrongIdentityManager()
    provider, camera = _provider(
        monkeypatch,
        [_frame(1)],
        _FakeTracker({}),
        manager,
        start=True,
    )
    try:
        assert provider.online_sam2_cfg["vos_optimized"] is True
        assert provider._online_sam2_ready is False
        assert provider._online_sam2_service_health is None
        assert manager.closed is True
        assert camera.started is True
    finally:
        provider.stop()


def test_camera_start_failure_closes_prestarted_online_service(monkeypatch):
    lifecycle = []
    tracker = _FakeTracker({})
    manager = _FakeOnlineSAM2Manager(lifecycle=lifecycle)
    provider, camera = _provider(
        monkeypatch,
        [_frame(1)],
        tracker,
        manager,
        lifecycle=lifecycle,
        start=False,
    )

    def fail_camera_start():
        lifecycle.append("camera_start")
        camera.started = True
        raise RuntimeError("scripted camera readiness failure")

    camera.start = fail_camera_start
    with pytest.raises(RuntimeError, match="scripted camera readiness failure"):
        provider.start()

    assert lifecycle == ["online_sam2_start", "camera_start"]
    assert camera.stopped
    assert manager.closed
    assert provider._provider_stopped
    assert provider._online_sam2_executor is None


def test_initial_prompt_seeds_but_same_target_reacquire_waits_for_probation(
    monkeypatch,
):
    first = _mask(5, 5, 13, 15)
    reacquired = _mask(15, 6, 23, 16)
    tracker = _FakeTracker({})
    manager = _FakeOnlineSAM2Manager()
    provider, _camera = _provider(monkeypatch, [_frame(3)], tracker, manager)

    assert provider.initialize_from_mask(_frame(1), first, source="grounded_sam_prompt")
    assert provider.reinitialize_same_target_from_mask(
        _frame(2), reacquired, source="prompt_reacquire_test"
    )

    initialize_calls = [call for call in manager.calls if call[0] == "initialize"]
    assert [call[1] for call in initialize_calls] == [1]
    np.testing.assert_array_equal(initialize_calls[0][3], first)
    assert not provider.tracking_committed
    assert provider.recovery_probation_active
    assert provider._online_sam2_initialized
    assert not provider._online_sam2_seed_pending
    provider.stop()


def test_initialize_from_mask_failure_restores_publication_provenance(
    monkeypatch,
):
    first = _mask(5, 5, 13, 15)
    replacement = _mask(15, 6, 23, 16)
    provider, _camera = _provider(
        monkeypatch,
        [],
        _FakeTracker({}),
        _FakeOnlineSAM2Manager(),
    )
    assert provider.initialize_from_mask(_frame(1), first)
    mask_result_before = provider.last_mask_result
    authority_before = copy.deepcopy(provider._full_target_geometry_authority)
    provider._last_output_mask_source = "sentinel_output"
    provider._last_online_sam2_status = "sentinel_online"
    provider._last_online_sam2_area = 123
    provider._last_online_sam2_geometry_reason = "sentinel_geometry"
    provider._last_online_sam2_reinit_reason = "sentinel_reinit"
    provider._last_online_sam2_exact_result = True
    provider._current_recovery_evidence_outcome = "candidate"
    provider._last_publication_guard_source = "sentinel_guard_source"
    provider._last_publication_guard_status = "sentinel_guard_status"
    provider._last_publication_guard_known_partial = True
    provider._last_publication_guard_partial_reason = "sentinel_partial"

    def fail_after_target_reset(*_args, **_kwargs):
        raise RuntimeError("scripted recovery-template failure")

    monkeypatch.setattr(provider, "_update_recovery_template", fail_after_target_reset)
    try:
        assert not provider.initialize_from_mask(_frame(2), replacement)
        assert provider.last_mask_result is mask_result_before
        assert provider._last_output_mask_source == "sentinel_output"
        assert provider._last_online_sam2_status == "sentinel_online"
        assert provider._last_online_sam2_area == 123
        assert provider._last_online_sam2_geometry_reason == ("sentinel_geometry")
        assert provider._last_online_sam2_reinit_reason == "sentinel_reinit"
        assert provider._last_online_sam2_exact_result is True
        assert provider._current_recovery_evidence_outcome == "candidate"
        assert provider._last_publication_guard_source == ("sentinel_guard_source")
        assert provider._last_publication_guard_status == ("sentinel_guard_status")
        assert provider._last_publication_guard_known_partial is True
        assert provider._last_publication_guard_partial_reason == ("sentinel_partial")
        assert provider._full_target_geometry_authority.frame_id == (
            authority_before.frame_id
        )
        assert provider._full_target_geometry_authority.area == pytest.approx(
            authority_before.area
        )
    finally:
        provider.stop()


def test_rejected_prompt_reacquire_never_reseeds_temporal_identity(monkeypatch):
    first = _mask(5, 5, 13, 15)
    wrong_instance = _mask(20, 8, 28, 18)
    tracker = _FakeTracker({})
    tracker.reject_reinit_masks.append(wrong_instance)
    manager = _FakeOnlineSAM2Manager()
    provider, _camera = _provider(monkeypatch, [_frame(3)], tracker, manager)
    assert provider.initialize_from_mask(_frame(1), first)

    assert not provider.reinitialize_same_target_from_mask(
        _frame(2), wrong_instance, source="rejected_prompt"
    )

    initialize_calls = [call for call in manager.calls if call[0] == "initialize"]
    assert [call[1] for call in initialize_calls] == [1]
    np.testing.assert_array_equal(initialize_calls[0][3], first)
    provider.stop()


def test_online_sam2_track_overlaps_adaptive_update_and_uses_strict_gate(
    monkeypatch,
):
    seed = _mask(4, 5, 12, 15)
    adaptive = _mask(5, 5, 13, 15)
    temporal = _mask(6, 5, 14, 15)
    update_started = threading.Event()
    track_started = threading.Event()
    tracker = _FakeTracker({2: _mask_result(adaptive)})
    tracker.update_started = update_started
    tracker.online_started = track_started
    manager = _FakeOnlineSAM2Manager({2: _video_result(2, temporal)})
    manager.update_started = update_started
    manager.track_started = track_started
    provider, _camera = _provider(monkeypatch, [_frame(2)], tracker, manager)
    assert provider.initialize_from_mask(_frame(1), seed)

    _frame_out, result, _obj, packet = provider.step()

    assert tracker.overlapped and manager.overlapped
    assert tracker.reinit_masks[-1][0] == 2
    np.testing.assert_array_equal(tracker.reinit_masks[-1][1], temporal)
    # The exact-current SAM2 contour becomes the public policy mask only after
    # it passes adaptive spatial consensus plus the immutable identity/depth
    # gate.  Its guarded reinit is observational and cannot mutate the
    # independent adaptive owner used on the next frame.
    np.testing.assert_array_equal(result.mask, temporal)
    assert tracker.restore_calls == 1
    assert tracker.token == "adaptive-f2"
    assert result.valid and packet.valid
    assert packet.debug["mask_source"] == "online_sam2_tracked"
    assert provider.last_timings_ms["sam2"] > 0.0
    assert packet.debug["online_sam2_exact_this_frame"] is True
    assert packet.debug["online_sam2_step_count"] == 1
    assert packet.debug["online_sam2_eligible_count"] == 1
    assert packet.debug["online_sam2_submit_count"] == 1
    assert packet.debug["online_sam2_busy_skip_count"] == 0
    assert packet.debug["online_sam2_exact_count"] == 1
    assert packet.debug["online_sam2_exact_valid_count"] == 1
    assert packet.debug["online_sam2_exact_frame_ratio"] == 1.0
    assert packet.debug["online_sam2_submit_frame_ratio"] == 1.0
    assert packet.debug["online_sam2_deadline_miss_count"] == 0
    assert packet.debug["online_sam2_late_count"] == 0
    provider.stop()


def test_mask_only_step_reuses_tracker_sam_gates_without_building_packet(
    monkeypatch,
):
    seed = _mask(4, 5, 12, 15)
    adaptive = _mask(5, 5, 13, 15)
    temporal = _mask(6, 5, 14, 15)
    tracker = _FakeTracker({2: _mask_result(adaptive)})
    manager = _FakeOnlineSAM2Manager({2: _video_result(2, temporal)})
    provider, _camera = _provider(monkeypatch, [_frame(2)], tracker, manager)
    assert provider.initialize_from_mask(_frame(1), seed)

    extractor_calls = 0
    history_calls = 0

    def forbidden_extract(*_args, **_kwargs):
        nonlocal extractor_calls
        extractor_calls += 1
        raise AssertionError("mask-only step called extractor.extract")

    def forbidden_history_update(*_args, **_kwargs):
        nonlocal history_calls
        history_calls += 1
        raise AssertionError("mask-only step updated provider PCD history")

    provider.extractor.extract = forbidden_extract
    provider.history.update = forbidden_history_update
    provider.last_packet = object()

    frame_out, result = provider.step_mask_only()

    assert frame_out.frame_id == 2
    # The mask-only preflight publishes the same accepted SAM2 contour as the
    # full point-cloud path; it merely skips point extraction/history writes.
    np.testing.assert_array_equal(result.mask, temporal)
    assert result.valid
    assert extractor_calls == 0
    assert history_calls == 0
    assert provider.last_packet is None
    assert provider.last_mask_result is result
    assert provider.last_timings_ms["pcd"] == 0.0
    assert provider.last_timings_ms["mask_gate"] >= 0.0
    assert provider.last_timings_ms["total"] >= provider.last_timings_ms["camera"]
    assert [call[1] for call in manager.calls if call[0] == "track"] == [2]
    assert provider._last_online_sam2_exact_result
    provider.stop()


def test_semantic_sam2_mask_mode_ignores_depth_and_adaptive_tracker(
    monkeypatch,
):
    seed = _mask(4, 5, 14, 17)
    # A hand-shaped occluder can split one visible object into two pieces.
    # Keep both pieces from the one SAM2 object id; do not take only the
    # largest connected component.
    visible = np.zeros_like(seed)
    visible[5:17, 4:8] = 1
    visible[5:17, 11:14] = 1
    frame = _frame(2)
    frame.depth_raw[:] = 0
    tracker = _FakeTracker({})
    manager = _FakeOnlineSAM2Manager({2: _video_result(2, visible)})
    provider, _camera = _provider(monkeypatch, [frame], tracker, manager)
    assert provider.initialize_from_mask(_frame(1), seed)
    provider.online_sam2_cfg["mask_publication_mode"] = "semantic_sam2"

    def forbidden_update(_frame_value):
        raise AssertionError("semantic SAM2 mode called adaptive tracker")

    tracker.update = forbidden_update
    try:
        frame_out, result = provider.step_mask_only()
        assert frame_out.frame_id == 2
        assert result.valid
        np.testing.assert_array_equal(result.mask, visible)
        assert result.source == "online_sam2_video"
        assert provider.last_timings_ms["tracker"] == 0.0
        assert provider.last_timings_ms["mask_gate"] == 0.0
    finally:
        provider.stop()


def _guarded_primary_provider(monkeypatch, frames, track_results, initial_mask):
    tracker = _IdentityAwareFakeTracker({})
    manager = _FakeOnlineSAM2Manager(track_results)
    provider, _camera = _provider(
        monkeypatch,
        frames,
        tracker,
        manager,
        recovery_publication_mode="unified_three_evidence",
    )
    provider.online_sam2_cfg["mask_publication_mode"] = "guarded_sam2_primary"
    assert provider.initialize_from_mask(
        _moving_target_frame(1, (40, 30, 80, 70)), initial_mask
    )

    def forbidden_update(_frame_value):
        raise AssertionError("guarded v2 called adaptive tracker.update")

    tracker.update = forbidden_update
    return provider, tracker


@pytest.mark.parametrize(
    ("setting", "invalid_value"),
    (
        ("guarded_v2_bootstrap_min_supported_extent_ratio", np.nan),
        ("guarded_v2_bootstrap_min_supported_extent_ratio", 0.899),
        ("guarded_v2_bootstrap_min_supported_extent_ratio", 1.001),
        ("guarded_v2_bootstrap_max_supported_centroid_offset_ratio", np.nan),
        ("guarded_v2_bootstrap_max_supported_centroid_offset_ratio", -0.001),
        ("guarded_v2_bootstrap_max_supported_centroid_offset_ratio", 1.001),
        (
            "guarded_v2_bootstrap_partial_seed_max_supported_centroid_offset_ratio",
            np.nan,
        ),
        (
            "guarded_v2_bootstrap_partial_seed_max_supported_centroid_offset_ratio",
            0.074,
        ),
        (
            "guarded_v2_bootstrap_partial_seed_max_supported_centroid_offset_ratio",
            1.001,
        ),
        ("guarded_v2_bootstrap_max_unsupported_bbox_margin_ratio", np.nan),
        ("guarded_v2_bootstrap_max_unsupported_bbox_margin_ratio", -0.001),
        ("guarded_v2_bootstrap_max_unsupported_bbox_margin_ratio", 1.001),
        (
            "guarded_v2_bootstrap_max_registered_contraction_fraction",
            np.nan,
        ),
        (
            "guarded_v2_bootstrap_max_registered_contraction_fraction",
            -0.001,
        ),
        (
            "guarded_v2_bootstrap_max_registered_contraction_fraction",
            0.101,
        ),
    ),
)
def test_guarded_primary_bootstrap_spatial_bounds_fail_when_armed(
    monkeypatch, setting, invalid_value
):
    tracker = _IdentityAwareFakeTracker({})
    provider, _camera = _provider(
        monkeypatch,
        [],
        tracker,
        _FakeOnlineSAM2Manager(),
        recovery_publication_mode="unified_three_evidence",
    )
    provider.online_sam2_cfg["mask_publication_mode"] = "guarded_sam2_primary"
    provider.online_sam2_cfg[setting] = invalid_value
    seed = _large_mask((40, 30, 80, 70))
    try:
        assert not provider.initialize_from_mask(
            _moving_target_frame(1, (40, 30, 80, 70)), seed
        )
        with pytest.raises(
            ValueError, match="online_sam2 guarded_v2 bootstrap bounds are invalid"
        ):
            provider._guarded_v2_bootstrap_limits()
    finally:
        provider.stop()


def _commit_bootstrap_raw_for_test(
    provider,
    frame,
    raw_mask,
    *,
    final_mask=None,
    eligibility_kind=None,
    prior_undercoverage=False,
    wrong_digest=False,
    output_source="online_sam2_guarded_primary",
    online_status="guarded_primary_trusted: test",
):
    """Drive the post-final-boundary bootstrap hook with explicit provenance."""

    final = raw_mask if final_mask is None else final_mask
    provider._online_sam2_step_count += 1
    provider._defer_full_target_geometry_commit(frame, raw_mask)
    if prior_undercoverage:
        provider._mark_pending_full_target_geometry_ineligible(
            frame,
            "test stale provisional scale",
            eligibility_kind="full_scale_undercoverage",
        )
    if eligibility_kind is not None:
        provider._mark_pending_full_target_geometry_ineligible(
            frame,
            "test known partial",
            eligibility_kind=eligibility_kind,
        )
    pending = provider._pending_full_target_geometry_commit
    assert pending is not None
    if wrong_digest:
        pending.mask_digest = "0" * 64
    provider._last_output_mask_source = output_source
    provider._last_publication_guard_source = output_source
    provider._last_online_sam2_status = str(online_status)
    provider._last_publication_guard_status = f"accepted: source={output_source}; test"
    provider._last_publication_guard_frame_id = int(frame.frame_id)
    provider._last_online_sam2_exact_result = True
    provider._last_publication_guard_known_partial = bool(
        not pending.raw_exact_eligible
    )
    provider._last_publication_guard_partial_reason = (
        pending.eligibility_reason
        if provider._last_publication_guard_known_partial
        else "none"
    )
    result = _mask_result(final)
    result.source = (
        "online_sam2_video"
        if output_source == "online_sam2_guarded_primary"
        else output_source
    )
    obj = provider._mask_only_publication_evidence(frame, final)
    provider._commit_final_publication_histories(frame, result, obj)
    return result


def test_guarded_primary_steady_mask_bypasses_adaptive_update(monkeypatch):
    target = _large_mask((40, 30, 80, 70))
    current = _moving_target_frame(2, (40, 30, 80, 70))
    provider, tracker = _guarded_primary_provider(
        monkeypatch,
        [current],
        {2: _video_result(2, target)},
        target,
    )
    token_before = tracker.token
    try:
        frame_out, result = provider.step_mask_only()
        assert frame_out.frame_id == 2
        assert result.valid
        np.testing.assert_array_equal(result.mask, target)
        assert provider.last_timings_ms["tracker"] == 0.0
        assert tracker.token == token_before
        assert provider._last_final_clean_publication.frame_id == 2
        final_history = provider._last_final_clean_publication
        assert final_history.mask_evidence is not None
        assert final_history.mask is final_history.mask_evidence.mask
        assert not final_history.mask.flags.writeable
        assert provider._full_target_geometry_authority.frame_id == 2
        assert provider._full_target_geometry_authority.area == pytest.approx(
            float(target.sum())
        )
        np.testing.assert_array_equal(
            provider._last_final_clean_publication.mask, target
        )
    finally:
        provider.stop()


def test_guarded_primary_bootstrap_commissions_stable_raw_exact_pair_once(
    monkeypatch,
):
    """A blurred oversized seed is replaced only by two adjacent raw masks."""

    blurred_seed = _large_mask((20, 10, 110, 100))
    exact4 = _large_mask((70, 35, 125, 90))
    exact5 = _large_mask((72, 34, 127, 89))
    frames = [
        _moving_target_frame(4, (70, 35, 125, 90)),
        _moving_target_frame(5, (72, 34, 127, 89)),
    ]
    provider, tracker = _guarded_primary_provider(
        monkeypatch,
        frames,
        {4: _video_result(4, exact4), 5: _video_result(5, exact5)},
        blurred_seed,
    )
    provider.cfg["tracker"]["recovery_enabled"] = True
    seed_authority_area = provider._full_target_geometry_authority.area
    try:
        assert provider.guarded_v2_bootstrap_state["phase"] == "provisional"
        result4 = provider.step_mask_only()[1]
        assert result4.valid
        assert provider.guarded_v2_bootstrap_state["phase"] == "provisional"
        assert provider.guarded_v2_bootstrap_state["candidate_frame_id"] == 4

        result5 = provider.step_mask_only()[1]
        assert result5.valid
        np.testing.assert_array_equal(result5.mask, exact5)
        state = provider.guarded_v2_bootstrap_state
        assert state["phase"] == "commissioned", state
        assert state["commissioned_frame_id"] == 5
        assert state["commission_count"] == 1
        assert seed_authority_area > float(exact5.sum())
        assert provider._target_initial_area == pytest.approx(float(exact5.sum()))
        assert provider._target_initial_bbox_area == pytest.approx(float(55 * 55))
        assert provider._last_final_clean_publication.frame_id == 5
        assert provider._last_final_clean_publication.motion_samples == 0
        assert provider._full_target_geometry_authority.frame_id == 5
        assert provider._full_target_geometry_authority.scale_samples == 0
        assert provider._trusted_online_sam2_publication.frame_id == 5
        assert provider._trusted_online_sam2_publication.velocity_samples == 0
        trusted_evidence = (
            provider._trusted_online_sam2_publication.mask_evidence
        )
        assert trusted_evidence is not None
        assert provider._trusted_online_sam2_publication.mask is (
            trusted_evidence.mask
        )
        assert provider._history_mask_evidence(
            provider._trusted_online_sam2_publication
        ) is trusted_evidence
        exact_anchor = provider._lost_reappearance_full_anchor(exact5.shape)
        assert exact_anchor[0] is trusted_evidence.mask, exact_anchor[-1]
        np.testing.assert_array_equal(
            provider._trusted_online_sam2_publication.mask, exact5
        )
        assert tracker.initialize_calls[0][0] == 1
        assert tracker.initialize_calls[-1][0] == 5
        assert len(tracker.initialize_calls) == 2
        np.testing.assert_array_equal(tracker.initialize_calls[-1][1], exact5)
        assert provider.recovery_template_bbox.tolist() == [72, 34, 127, 89]
        # Rebuilding local frozen state must not restart/reseed remote SAM2.
        assert [
            call[1]
            for call in provider.online_sam2_manager.calls
            if call[0] == "initialize"
        ] == [1]

        # Once commissioned, even another perfect raw frame cannot rebase it.
        provider._commit_final_publication_histories(
            _moving_target_frame(6, (74, 34, 129, 89)),
            _mask_result(_large_mask((74, 34, 129, 89))),
            provider._mask_only_publication_evidence(
                _moving_target_frame(6, (74, 34, 129, 89)),
                _large_mask((74, 34, 129, 89)),
            ),
        )
        assert len(tracker.initialize_calls) == 2
        assert provider.guarded_v2_bootstrap_state["phase"] == "commissioned"
    finally:
        provider.stop()


def test_guarded_primary_bootstrap_accepts_bounded_fast_scale_quantization(
    monkeypatch,
):
    """A clean 43x43 -> 44x44 exact pair commissions at fast entry.

    This is the 1.047x area change observed in the real fast-ball replay.  The
    pair still has to cross every raw-exact, immutable-appearance, depth,
    motion, dominant-component and final-publication gate; this test changes
    only the last bounded stability comparison.
    """

    # Keep the provisional blur inside the independent predicted-area and
    # trusted scale-rate guards.  This fixture isolates only the 43x43 ->
    # 44x44 pair's final bootstrap stability comparison.
    seed = _large_mask((35, 20, 110, 95))
    exact4 = _large_mask((70, 35, 113, 78))
    exact5 = _large_mask((71, 35, 115, 79))
    frames = [
        _moving_target_frame(4, (70, 35, 113, 78)),
        _moving_target_frame(5, (71, 35, 115, 79)),
    ]
    provider, tracker = _guarded_primary_provider(
        monkeypatch,
        frames,
        {4: _video_result(4, exact4), 5: _video_result(5, exact5)},
        seed,
    )
    try:
        assert float(exact5.sum()) / float(exact4.sum()) == pytest.approx(
            1936.0 / 1849.0
        )
        assert provider.step_mask_only()[1].valid
        state4 = provider.guarded_v2_bootstrap_state
        assert state4["phase"] == "provisional"
        assert state4["candidate_frame_id"] == 4

        assert provider.step_mask_only()[1].valid
        state5 = provider.guarded_v2_bootstrap_state
        assert state5["phase"] == "commissioned"
        assert state5["commissioned_frame_id"] == 5
        assert state5["commission_count"] == 1
        assert "area_x=1.047" in state5["status"] or "two bounded stable" in state5[
            "status"
        ]
        assert len(tracker.initialize_calls) == 2
        np.testing.assert_array_equal(tracker.initialize_calls[-1][1], exact5)
    finally:
        provider.stop()


def test_guarded_primary_bootstrap_commissions_across_one_transient_gap(
    monkeypatch,
):
    """One fail-closed deadline miss may bridge an otherwise stable pair."""

    blurred_seed = _large_mask((20, 10, 110, 100))
    exact4 = _large_mask((70, 35, 125, 90))
    exact7 = _large_mask((72, 34, 127, 89))
    release_frame5 = threading.Event()
    finished_frame5 = threading.Event()
    manager = _FakeOnlineSAM2Manager(
        {
            4: _video_result(4, exact4),
            5: _video_result(5, exact4),
            7: _video_result(7, exact7),
        }
    )
    manager.track_wait_events[5] = release_frame5
    manager.track_finished_events[5] = finished_frame5
    tracker = _IdentityAwareFakeTracker({})
    provider, _camera = _provider(
        monkeypatch,
        [
            _moving_target_frame(4, (70, 35, 125, 90)),
            _moving_target_frame(5, (70, 35, 125, 90)),
            _moving_target_frame(7, (72, 34, 127, 89)),
        ],
        tracker,
        manager,
        recovery_publication_mode="unified_three_evidence",
    )
    provider.online_sam2_cfg["mask_publication_mode"] = "guarded_sam2_primary"
    provider.online_sam2_cfg["semantic_frame_wait_timeout_s"] = 0.005
    assert provider.initialize_from_mask(
        _moving_target_frame(1, (40, 30, 80, 70)), blurred_seed
    )
    try:
        output4 = provider.step()
        assert output4[3].valid, (
            output4[1].message,
            output4[2].message,
            output4[3].message,
        )
        assert provider.guarded_v2_bootstrap_state["candidate_frame_id"] == 4

        output5 = provider.step()
        assert not output5[1].valid
        assert not output5[2].valid
        assert not output5[3].valid
        paused = provider.guarded_v2_bootstrap_state
        assert paused["phase"] == "provisional"
        assert paused["candidate_frame_id"] == 4
        assert paused["transient_gap_frame_id"] == 5
        assert paused["transient_gap_count"] == 1

        release_frame5.set()
        assert finished_frame5.wait(timeout=1.0)
        output7 = provider.step()
        assert output7[1].valid and output7[2].valid and output7[3].valid
        np.testing.assert_array_equal(output7[1].mask, exact7)
        state = provider.guarded_v2_bootstrap_state
        assert state["phase"] == "commissioned", state
        assert state["commissioned_frame_id"] == 7
        assert state["commission_count"] == 1
        assert state["transient_gap_frame_id"] is None
        assert state["transient_gap_count"] == 1
        assert len(tracker.initialize_calls) == 2
    finally:
        release_frame5.set()
        provider.stop()


@pytest.mark.parametrize("failure", ("second_gap", "candidate_timeout"))
def test_guarded_primary_bootstrap_transient_gap_stays_strict(
    monkeypatch, failure
):
    """A pause is one-shot and cannot cross the existing candidate bounds."""

    seed = _large_mask((20, 10, 110, 100))
    exact = _large_mask((70, 35, 125, 90))
    provider, tracker = _guarded_primary_provider(monkeypatch, [], {}, seed)
    try:
        frame2 = _moving_target_frame(2, (70, 35, 125, 90))
        _commit_bootstrap_raw_for_test(provider, frame2, exact)
        assert provider.guarded_v2_bootstrap_state["candidate_frame_id"] == 2

        def observe_transient(frame):
            provider._online_sam2_step_count += 1
            provider._current_recovery_evidence_outcome = (
                "transient_inference_unavailable"
            )
            empty = np.zeros_like(exact)
            result = _mask_result(
                empty,
                valid=False,
                message="scripted transient inference miss",
            )
            result.source = "online_sam2_video"
            obj = provider._mask_only_publication_evidence(frame, empty)
            provider._commit_final_publication_histories(frame, result, obj)

        if failure == "candidate_timeout":
            # Still inside the 12-frame/0.40s seed window, but outside the
            # existing 3-frame/0.20s candidate continuation window.
            observe_transient(_moving_target_frame(6, (70, 35, 125, 90)))
        else:
            observe_transient(_moving_target_frame(3, (70, 35, 125, 90)))
            paused = provider.guarded_v2_bootstrap_state
            assert paused["phase"] == "provisional"
            assert paused["transient_gap_count"] == 1
            observe_transient(_moving_target_frame(4, (70, 35, 125, 90)))

        state = provider.guarded_v2_bootstrap_state
        assert state["phase"] == "expired"
        assert state["candidate_frame_id"] is None
        assert state["transient_gap_frame_id"] is None
        assert len(tracker.initialize_calls) == 1
    finally:
        provider.stop()


@pytest.mark.parametrize(
    "rejection",
    (
        "wrong_digest",
        "sanitized",
        "generic_partial",
        "boundary",
        "late",
        "nan_appearance",
        "forged_compact_sanitized",
        "forged_compact_recovery",
    ),
)
def test_guarded_primary_bootstrap_rejects_ineligible_evidence(monkeypatch, rejection):
    seed = _large_mask((30, 20, 120, 105))
    normal = _large_mask((65, 35, 120, 90))
    bbox = (65, 35, 120, 90)
    frame_id = 20 if rejection == "late" else 4
    if rejection == "boundary":
        normal = _large_mask((130, 35, 180, 90))
        bbox = (130, 35, 180, 90)
    frame = _moving_target_frame(frame_id, bbox)
    provider, tracker = _guarded_primary_provider(monkeypatch, [], {}, seed)
    try:
        kwargs = {}
        if rejection == "wrong_digest":
            kwargs["wrong_digest"] = True
        elif rejection == "sanitized":
            kwargs["final_mask"] = _large_mask((65, 35, 115, 90))
            kwargs["eligibility_kind"] = "known_partial"
            kwargs["output_source"] = "publication_sanitized_target_core"
        elif rejection == "generic_partial":
            kwargs["eligibility_kind"] = "known_partial"
        elif rejection == "forged_compact_sanitized":
            kwargs["final_mask"] = _large_mask((65, 35, 115, 90))
            kwargs["eligibility_kind"] = "bootstrap_compact_undercoverage"
            kwargs["output_source"] = "publication_sanitized_target_core"
        elif rejection == "forged_compact_recovery":
            kwargs["eligibility_kind"] = "bootstrap_compact_undercoverage"
            provider._current_recovery_evidence_outcome = "candidate"
        elif rejection == "nan_appearance":
            tracker.target_appearance_support_stats = lambda *_args, **_kwargs: (
                float("nan"),
                1.0,
                100,
            )
        _commit_bootstrap_raw_for_test(provider, frame, normal, **kwargs)

        assert provider.guarded_v2_bootstrap_state["phase"] != "commissioned"
        assert provider.guarded_v2_bootstrap_state["candidate_frame_id"] is None
        assert len(tracker.initialize_calls) == 1
        if rejection in ("boundary", "late", "forged_compact_recovery"):
            assert provider.guarded_v2_bootstrap_state["phase"] == "expired"
    finally:
        provider.stop()


def test_guarded_primary_bootstrap_rejects_stable_hand_union(monkeypatch):
    blurred_seed = _large_mask((30, 20, 100, 85))
    target = _large_mask((40, 30, 80, 70))
    # This smaller appendage used to pass extent+centroid alone:
    # extent_x=40/46=.870 and centroid offset_x=-3/46=-.065.
    hand = _large_mask((80, 30, 86, 70))
    union = np.maximum(target, hand)
    frames = [
        _moving_target_frame(
            frame_id,
            (40, 30, 80, 70),
            contaminant_bbox=(80, 30, 86, 70),
        )
        for frame_id in (4, 5)
    ]
    provider, tracker = _guarded_primary_provider(
        monkeypatch,
        frames,
        {4: _video_result(4, union), 5: _video_result(5, union)},
        blurred_seed,
    )
    authority_before = copy.deepcopy(provider._full_target_geometry_authority)
    trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)
    try:
        accepted, detail = provider._appearance_supported_silhouette_extent_accepts(
            frames[0], union, threshold=0.42
        )
        assert not accepted
        assert "margins=(0.000,0.130" in detail

        # A target-coloured outlier at the hand tip cannot enlarge the support
        # bbox because the discriminator measures only the largest component.
        original_supported_mask = tracker.target_appearance_supported_mask

        def supported_with_isolated_tip(frame, mask, *, threshold=None):
            supported = original_supported_mask(frame, mask, threshold=threshold)
            supported[50, 85] = 1
            return supported

        tracker.target_appearance_supported_mask = supported_with_isolated_tip
        spoofed, spoofed_detail = (
            provider._appearance_supported_silhouette_extent_accepts(
                frames[0], union, threshold=0.42
            )
        )
        assert not spoofed
        assert "margins=(0.000,0.130" in spoofed_detail
        tracker.target_appearance_supported_mask = original_supported_mask

        outputs = [provider.step_mask_only()[1] for _ in frames]
        assert all(result.valid for result in outputs)
        for result in outputs:
            assert np.all(np.logical_or(result.mask == 0, target > 0))
            assert np.count_nonzero(result.mask) > 0
            assert np.count_nonzero(np.logical_and(result.mask > 0, hand > 0)) == 0
            assert result.source == "publication_sanitized_target_core"
        assert provider._last_publication_guard_status.startswith("sanitized:")
        assert provider._last_publication_guard_known_partial
        assert "appearance" in provider._last_publication_guard_partial_reason
        assert provider.guarded_v2_bootstrap_state["phase"] == "provisional"
        assert provider.guarded_v2_bootstrap_state["candidate_frame_id"] is None
        assert len(tracker.initialize_calls) == 1
        assert provider._full_target_geometry_authority.frame_id == (
            authority_before.frame_id
        )
        assert provider._trusted_online_sam2_publication.frame_id == (
            trusted_before.frame_id
        )
    finally:
        provider.stop()


def test_partial_seed_centroid_bound_is_bootstrap_only(monkeypatch):
    """Reviewed 0.086 shading offset needs 0.10 only before commission."""

    candidate = _large_mask((50, 30, 90, 70))
    frame = _moving_target_frame(4, (50, 30, 90, 70))
    # Preserve a connected full-extent border, but remove lower-centre target
    # appearance.  The support centroid offset is ~0.086: outside the normal
    # 0.075 publication bound and inside the provisional-seed 0.10 bound.
    frame.color_bgr[50:66, 55:85] = (120, 120, 120)
    provider, _tracker = _guarded_primary_provider(
        monkeypatch, [], {}, candidate
    )
    try:
        normal_ok, normal_detail = (
            provider._appearance_supported_silhouette_extent_accepts(
                frame, candidate, threshold=0.42
            )
        )
        startup_ok, startup_detail = (
            provider._appearance_supported_silhouette_extent_accepts(
                frame,
                candidate,
                threshold=0.42,
                max_centroid_offset_ratio=0.10,
            )
        )
        assert not normal_ok
        assert "/0.075" in normal_detail
        assert startup_ok, startup_detail
        assert "/0.100" in startup_detail
    finally:
        provider.stop()


def test_guarded_primary_bootstrap_accepts_real_rgb_like_distributed_support(
    monkeypatch,
):
    """A partially visible seed may expand to a fully supported real ball."""

    # The seed is a smaller hand-occluded fragment with the same aspect as the
    # later full silhouette.  Candidate area remains inside the independent
    # 2.5x initialization bound; no generic expansion bypass is introduced.
    partial_seed = _large_mask((40, 30, 80, 90))  # 40x60
    # Dimensions and appearance geometry reproduce the reviewed source44/45
    # measurements while staying clear of this fixture's image boundary.
    masks = {
        4: _large_mask((65, 35, 113, 102)),  # 48x67
        5: _large_mask((65, 36, 113, 100)),  # 48x64, 1.047x contraction
    }

    def shaded_ball_frame(frame_id, mask):
        frame = _moving_target_frame(frame_id, (0, 0, 0, 0))
        ys, xs = np.nonzero(mask)
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
        supported = (mask > 0).astype(np.uint8)
        frame.color_bgr[mask > 0] = (120, 120, 120)
        if frame_id == 4:
            # source44-like: support=.793, extent=(.979,1),
            # centroid offset=(-.052,-.060).
            supported[y1:y2, x2 - 1 : x2] = 0
            supported[y1 + 38 : y1 + 63, x1 + 20 : x1 + 44] = 0
        else:
            # source45-like dominant component: extent=(.938,1), with
            # aggregate support=.829 across the contracted silhouette.
            supported[y1:y2, x2 - 3 : x2] = 0
            supported[y1 + 45 : y1 + 54, x1 + 8 : x1 + 45] = 0
        frame.color_bgr[supported > 0] = (210, 80, 235)
        return frame, supported

    fixtures = [
        shaded_ball_frame(frame_id, masks[frame_id]) for frame_id in (4, 5)
    ]
    frames = [frame for frame, _supported in fixtures]
    provider, tracker = _guarded_primary_provider(
        monkeypatch,
        frames,
        {frame_id: _video_result(frame_id, masks[frame_id]) for frame_id in masks},
        partial_seed,
    )
    authority_before = copy.deepcopy(provider._full_target_geometry_authority)
    trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)
    try:
        support_details = []
        expected_support = (0.7926, 0.8291)
        for (frame, _supported), mask, expected in zip(
            fixtures, masks.values(), expected_support
        ):
            stats = tracker.target_appearance_support_stats(frame, mask, threshold=0.42)
            assert stats[1] == pytest.approx(expected, abs=0.0006)
            accepted, detail = provider._appearance_supported_silhouette_extent_accepts(
                frame, mask, threshold=0.42
            )
            assert accepted, detail
            support_details.append(detail)

        assert "extent=(0.979,1.000)" in support_details[0]
        assert "centroid_offset=(-0.052,-0.060)" in support_details[0]
        assert "margins=(0.000,0.021,0.000,0.000)" in support_details[0]
        assert "extent=(0.938,1.000)" in support_details[1]

        output4 = provider.step_mask_only()[1]
        assert output4.valid
        np.testing.assert_array_equal(output4.mask, masks[4])
        assert output4.source == "online_sam2_video"
        assert provider.guarded_v2_bootstrap_state["phase"] == "provisional"
        assert provider.guarded_v2_bootstrap_state["candidate_frame_id"] == 4
        assert "stable candidate 1/2" in provider.guarded_v2_bootstrap_state["status"]
        assert provider._bbox_isotropic_scale_consistency(
            trusted_before.bbox_xyxy,
            provider._guarded_v2_bootstrap_hypothesis.bbox_xyxy,
        ) >= 0.85
        assert "startup compact" in provider._last_publication_guard_partial_reason
        assert provider._full_target_geometry_authority.frame_id == (
            authority_before.frame_id
        )
        assert provider._trusted_online_sam2_publication.frame_id == (
            trusted_before.frame_id
        )
        assert len(tracker.initialize_calls) == 1

        output5 = provider.step_mask_only()[1]
        assert output5.valid
        np.testing.assert_array_equal(output5.mask, masks[5])
        assert output5.source == "online_sam2_video"
        state = provider.guarded_v2_bootstrap_state
        assert state["phase"] == "commissioned", state
        assert state["commissioned_frame_id"] == 5
        assert state["commission_count"] == 1
        assert len(tracker.initialize_calls) == 2
        np.testing.assert_array_equal(tracker.initialize_calls[-1][1], masks[5])
    finally:
        provider.stop()


def test_guarded_primary_bootstrap_rejects_translated_erosion_with_noise(
    monkeypatch,
):
    seed = _large_mask((25, 15, 125, 105))
    first = _large_mask((50, 30, 110, 90))
    eroded = _large_mask((52, 30, 112, 87))
    eroded[60, 52] = 0
    # Registered loss=181/3600=5.028%, while 12 leading-edge pixels keep the
    # total area ratio below 1.05.  A little gain cannot hide contraction over
    # the independent bootstrap-only 5% limit.
    eroded[50:62, 112] = 1
    provider, tracker = _guarded_primary_provider(monkeypatch, [], {}, seed)
    try:
        frame4 = _moving_target_frame(4, (50, 30, 110, 90))
        frame5 = _moving_target_frame(5, (52, 30, 113, 87))
        frame5.color_bgr[eroded > 0] = (210, 80, 235)
        _commit_bootstrap_raw_for_test(
            provider,
            frame4,
            first,
            prior_undercoverage=True,
        )
        previous = provider._guarded_v2_bootstrap_hypothesis
        assert previous is not None
        current_center = provider._binary_mask_centroid(eroded)
        assert current_center is not None
        assert int(np.rint(current_center[0] - previous.center_xy[0])) == 2
        _commit_bootstrap_raw_for_test(
            provider,
            frame5,
            eroded,
            prior_undercoverage=True,
        )
        assert provider.guarded_v2_bootstrap_state["phase"] == "provisional"
        assert "instability" in provider.guarded_v2_bootstrap_state["status"]
        assert len(tracker.initialize_calls) == 1
    finally:
        provider.stop()


def test_guarded_primary_bootstrap_mask_only_exception_rolls_back_candidate(
    monkeypatch,
):
    seed = _large_mask((25, 15, 125, 105))
    first = _large_mask((60, 35, 115, 90))
    unstable = _large_mask((61, 35, 121, 90))
    provider, tracker = _guarded_primary_provider(monkeypatch, [], {}, seed)
    try:
        frame4 = _moving_target_frame(4, (60, 35, 115, 90))
        _commit_bootstrap_raw_for_test(
            provider, frame4, first, prior_undercoverage=True
        )
        candidate_before = copy.deepcopy(provider._guarded_v2_bootstrap_hypothesis)
        status_before = provider._guarded_v2_bootstrap_status
        tracker_before = tracker.snapshot_state()
        authority_before = copy.deepcopy(provider._full_target_geometry_authority)
        final_before = copy.deepcopy(provider._last_final_clean_publication)
        original_confirm = provider._confirm_full_target_scale_transition

        def fail_after_observation(*_args, **_kwargs):
            raise RuntimeError("scripted post-observation failure")

        provider._confirm_full_target_scale_transition = fail_after_observation
        frame5 = _moving_target_frame(5, (61, 35, 121, 90))
        with pytest.raises(RuntimeError, match="post-observation failure"):
            _commit_bootstrap_raw_for_test(
                provider, frame5, unstable, prior_undercoverage=True
            )

        restored = provider._guarded_v2_bootstrap_hypothesis
        assert restored is not None and candidate_before is not None
        assert restored.frame_id == candidate_before.frame_id == 4
        np.testing.assert_array_equal(restored.mask, candidate_before.mask)
        assert provider._guarded_v2_bootstrap_status == status_before
        assert tracker.token == tracker_before.token
        np.testing.assert_array_equal(
            tracker.state.last_mask, tracker_before.state.last_mask
        )
        assert provider._full_target_geometry_authority.frame_id == (
            authority_before.frame_id
        )
        assert provider._last_final_clean_publication.frame_id == (
            final_before.frame_id
        )
        assert provider._pending_full_target_geometry_commit is None

        # The failed step increment remains visible, so frame 4 cannot pair
        # across it with the next valid observation.
        provider._confirm_full_target_scale_transition = original_confirm
        frame6 = _moving_target_frame(6, (60, 35, 115, 90))
        _commit_bootstrap_raw_for_test(
            provider, frame6, first, prior_undercoverage=True
        )
        assert provider.guarded_v2_bootstrap_state["phase"] == "provisional"
        assert provider.guarded_v2_bootstrap_state["candidate_frame_id"] == 6
        assert len(tracker.initialize_calls) == 1
    finally:
        provider.stop()


def test_guarded_primary_full_step_bypasses_adaptive_update(monkeypatch):
    target = _large_mask((40, 30, 80, 70))
    current = _moving_target_frame(2, (40, 30, 80, 70))
    provider, tracker = _guarded_primary_provider(
        monkeypatch,
        [current],
        {2: _video_result(2, target)},
        target,
    )
    token_before = tracker.token
    try:
        frame_out, result, obj, packet = provider.step()
        assert frame_out.frame_id == 2
        assert result.valid and obj.valid and packet.valid
        np.testing.assert_array_equal(result.mask, target)
        assert provider.last_timings_ms["tracker"] == 0.0
        assert tracker.token == token_before
    finally:
        provider.stop()


@pytest.mark.parametrize("commit_case", ["changed_mask", "cross_frame"])
def test_guarded_primary_pending_raw_exact_token_is_frame_and_mask_bound(
    monkeypatch, commit_case
):
    raw = _large_mask((40, 30, 80, 70))
    changed = _large_mask((42, 30, 82, 70))
    provider, tracker = _guarded_primary_provider(monkeypatch, [], {}, raw)
    authority_before = copy.deepcopy(provider._full_target_geometry_authority)
    trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)
    try:
        token_frame = _moving_target_frame(2, (40, 30, 80, 70))
        provider._defer_full_target_geometry_commit(
            token_frame,
            raw,
            source="online_sam2_video",
            raw_exact_eligible=True,
        )
        if commit_case == "changed_mask":
            final_frame = token_frame
            final_mask = changed
        else:
            final_frame = _moving_target_frame(3, (40, 30, 80, 70))
            final_mask = raw
        final_result = _mask_result(final_mask)
        final_obj = provider._mask_only_publication_evidence(final_frame, final_mask)

        provider._commit_final_publication_histories(
            final_frame, final_result, final_obj
        )

        assert provider._pending_full_target_geometry_commit is None
        assert provider._full_target_geometry_authority.frame_id == (
            authority_before.frame_id
        )
        assert provider._trusted_online_sam2_publication.frame_id == (
            trusted_before.frame_id
        )
    finally:
        provider.stop()


def test_guarded_primary_transient_deadline_and_busy_gap_resume_directly(
    monkeypatch,
):
    """A local scheduling hole fails closed without revoking identity.

    Frame 2 misses the bounded exact-frame deadline and leaves its inference
    in flight.  Frame 3 therefore observes the expected latest-only busy
    condition.  Neither is visual contradiction, so frame 4 can publish in
    one step once its exact mask passes the normal trusted temporal gates.
    """

    target1 = _large_mask((40, 30, 80, 70))
    target4 = _large_mask((42, 30, 82, 70))
    release_frame2 = threading.Event()
    finished_frame2 = threading.Event()
    manager = _FakeOnlineSAM2Manager(
        {
            2: _video_result(2, target1),
            4: _video_result(4, target4),
        }
    )
    manager.track_wait_events[2] = release_frame2
    manager.track_finished_events[2] = finished_frame2
    tracker = _IdentityAwareFakeTracker({})
    provider, _camera = _provider(
        monkeypatch,
        [
            _moving_target_frame(2, (40, 30, 80, 70)),
            _moving_target_frame(3, (40, 30, 80, 70)),
            _moving_target_frame(4, (42, 30, 82, 70)),
        ],
        tracker,
        manager,
        recovery_publication_mode="unified_three_evidence",
    )
    provider.online_sam2_cfg["mask_publication_mode"] = "guarded_sam2_primary"
    assert provider.initialize_from_mask(
        _moving_target_frame(1, (40, 30, 80, 70)), target1
    )
    resets_before = provider.history.reset_calls

    try:
        output2 = provider.step()
        output3 = provider.step()

        for output in (output2, output3):
            assert not output[1].valid
            assert not output[2].valid
            assert not output[3].valid
            assert "transient inference unavailability" in output[1].message
        assert provider.tracking_committed
        assert provider._trusted_online_sam2_continuity_intact
        assert provider._trusted_online_sam2_publication.frame_id == 1
        assert provider._last_final_clean_publication.frame_id == 1
        assert provider._online_sam2_recovery_hypothesis is None
        assert provider._recovery_publication_hypothesis is None
        assert provider.history.reset_calls == resets_before
        assert provider._online_sam2_deadline_miss_count == 1
        assert provider._online_sam2_busy_skip_count == 1

        release_frame2.set()
        assert finished_frame2.wait(timeout=1.0)
        output4 = provider.step()

        assert output4[1].valid and output4[2].valid and output4[3].valid
        np.testing.assert_array_equal(output4[1].mask, target4)
        assert provider.tracking_committed
        assert provider._trusted_online_sam2_continuity_intact
        assert provider._trusted_online_sam2_publication.frame_id == 4
        assert provider._last_final_clean_publication.frame_id == 4
        assert provider._trusted_online_sam2_publication.velocity_samples == 0
        assert provider._last_final_clean_publication.motion_samples == 0
        assert not provider._trusted_online_sam2_transient_gap_active
        assert provider._online_sam2_recovery_hypothesis is None
        assert provider._recovery_publication_hypothesis is None
        assert "recovery pending" not in output4[1].message.lower()
        assert provider.history.reset_calls == resets_before
    finally:
        release_frame2.set()
        provider.stop()


def test_guarded_primary_previous_frame_grace_avoids_second_busy_hole(
    monkeypatch,
):
    old_target = _large_mask((40, 30, 80, 70))
    current_target = _large_mask((42, 30, 82, 70))
    release_frame2 = threading.Event()
    finished_frame2 = threading.Event()
    manager = _FakeOnlineSAM2Manager(
        {
            2: _video_result(2, old_target),
            3: _video_result(3, current_target),
        }
    )
    manager.track_wait_events[2] = release_frame2
    manager.track_finished_events[2] = finished_frame2
    tracker = _IdentityAwareFakeTracker({})
    provider, _camera = _provider(
        monkeypatch,
        [
            _moving_target_frame(2, (40, 30, 80, 70)),
            _moving_target_frame(3, (40, 30, 80, 70)),
        ],
        tracker,
        manager,
        recovery_publication_mode="unified_three_evidence",
    )
    provider.online_sam2_cfg["mask_publication_mode"] = "guarded_sam2_primary"
    provider.online_sam2_cfg["semantic_previous_frame_grace_s"] = 0.015
    assert provider.initialize_from_mask(
        _moving_target_frame(1, (40, 30, 80, 70)), old_target
    )
    release_timer = None

    try:
        output2 = provider.step()
        assert not output2[1].valid
        # The production tail which motivated this path needed about 11--12
        # ms more service time at the next 20 Hz tick.  Its stale pixels must
        # be retired, never published as the current frame.
        release_timer = threading.Timer(0.012, release_frame2.set)
        release_timer.start()

        output3 = provider.step()
        assert finished_frame2.wait(timeout=1.0)
        assert output3[1].valid and output3[2].valid and output3[3].valid
        np.testing.assert_array_equal(output3[1].mask, current_target)
        assert provider._online_sam2_deadline_miss_count == 1
        assert provider._online_sam2_busy_skip_count == 0
        assert provider._online_sam2_previous_grace_hit_count == 1
        assert provider._trusted_online_sam2_publication.frame_id == 3
        assert not provider._trusted_online_sam2_transient_gap_active
    finally:
        if release_timer is not None:
            release_timer.cancel()
        release_frame2.set()
        provider.stop()


def test_guarded_primary_transient_gap_rebases_accelerating_motion_history(
    monkeypatch,
):
    """A resumed exact frame is an anchor, not an across-gap velocity sample.

    The target moves far to the right while frame 2 is late, then reverses by
    60 px on frame 5.  Both visible displacements satisfy the configured hard
    speed gate.  Estimating velocity from frame 1 to frame 4 would nevertheless
    predict frame 5 about 97 px away and falsely exceed the 72 px residual cap.
    """

    target1 = _large_mask((20, 30, 40, 70))
    target4 = _large_mask((130, 30, 150, 70))
    target5 = _large_mask((70, 30, 90, 70))
    release_frame2 = threading.Event()
    finished_frame2 = threading.Event()
    manager = _FakeOnlineSAM2Manager(
        {
            2: _video_result(2, target1),
            4: _video_result(4, target4),
            5: _video_result(5, target5),
        }
    )
    manager.track_wait_events[2] = release_frame2
    manager.track_finished_events[2] = finished_frame2
    tracker = _IdentityAwareFakeTracker({})
    provider, _camera = _provider(
        monkeypatch,
        [
            _moving_target_frame(2, (20, 30, 40, 70)),
            _moving_target_frame(3, (20, 30, 40, 70)),
            _moving_target_frame(4, (130, 30, 150, 70)),
            _moving_target_frame(5, (70, 30, 90, 70)),
        ],
        tracker,
        manager,
        recovery_publication_mode="unified_three_evidence",
    )
    provider.online_sam2_cfg["mask_publication_mode"] = "guarded_sam2_primary"
    assert provider.initialize_from_mask(
        _moving_target_frame(1, (20, 30, 40, 70)), target1
    )

    try:
        assert not provider.step()[1].valid
        assert not provider.step()[1].valid
        release_frame2.set()
        assert finished_frame2.wait(timeout=1.0)

        output4 = provider.step()
        assert output4[1].valid
        assert provider._trusted_online_sam2_publication.velocity_samples == 0
        assert provider._last_final_clean_publication.motion_samples == 0

        output5 = provider.step()
        assert output5[1].valid and output5[2].valid and output5[3].valid
        np.testing.assert_array_equal(output5[1].mask, target5)
        assert provider._trusted_online_sam2_publication.frame_id == 5
        assert provider._trusted_online_sam2_publication.velocity_samples == 1
    finally:
        release_frame2.set()
        provider.stop()


def test_guarded_primary_transient_gap_skips_stale_velocity_prediction(
    monkeypatch,
):
    """Established velocity cannot become authority across an inference hole."""

    target1 = _large_mask((20, 30, 40, 70))
    target2 = _large_mask((40, 30, 60, 70))
    target5 = _large_mask((20, 30, 40, 70))
    release_frame3 = threading.Event()
    finished_frame3 = threading.Event()
    manager = _FakeOnlineSAM2Manager(
        {
            2: _video_result(2, target2),
            3: _video_result(3, target2),
            5: _video_result(5, target5),
        }
    )
    manager.track_wait_events[3] = release_frame3
    manager.track_finished_events[3] = finished_frame3
    tracker = _IdentityAwareFakeTracker({})
    provider, _camera = _provider(
        monkeypatch,
        [
            _moving_target_frame(2, (40, 30, 60, 70)),
            _moving_target_frame(3, (40, 30, 60, 70)),
            _moving_target_frame(4, (40, 30, 60, 70)),
            _moving_target_frame(5, (20, 30, 40, 70)),
        ],
        tracker,
        manager,
        recovery_publication_mode="unified_three_evidence",
    )
    provider.online_sam2_cfg["mask_publication_mode"] = "guarded_sam2_primary"
    assert provider.initialize_from_mask(
        _moving_target_frame(1, (20, 30, 40, 70)), target1
    )

    try:
        output2 = provider.step()
        assert output2[1].valid
        assert provider._trusted_online_sam2_publication.velocity_samples == 1

        assert not provider.step()[1].valid
        assert not provider.step()[1].valid
        assert provider._trusted_online_sam2_transient_gap_active
        release_frame3.set()
        assert finished_frame3.wait(timeout=1.0)

        output5 = provider.step()
        assert output5[1].valid and output5[2].valid and output5[3].valid
        np.testing.assert_array_equal(output5[1].mask, target5)
        assert provider._trusted_online_sam2_publication.frame_id == 5
        assert provider._trusted_online_sam2_publication.velocity_samples == 0
        assert not provider._trusted_online_sam2_transient_gap_active
    finally:
        release_frame3.set()
        provider.stop()


def test_guarded_primary_sanitizes_hand_without_tracker_mutation(monkeypatch):
    target = _large_mask((40, 30, 80, 70))
    hand = _large_mask((80, 30, 90, 70))
    union = np.maximum(target, hand)
    current = _moving_target_frame(
        2, (40, 30, 80, 70), contaminant_bbox=(80, 30, 90, 70)
    )
    provider, tracker = _guarded_primary_provider(
        monkeypatch,
        [current],
        {2: _video_result(2, union)},
        target,
    )
    token_before = tracker.token
    full_authority_before = copy.deepcopy(provider._full_target_geometry_authority)
    trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)
    try:
        _frame_out, result = provider.step_mask_only()
        assert result.valid
        np.testing.assert_array_equal(result.mask, target)
        assert "foreign appendage removed" in result.message
        assert "semantic_primary_no_tracker_mutation" in result.message
        assert tracker.token == token_before
        np.testing.assert_array_equal(
            provider._last_final_clean_publication.mask, target
        )
        # The published visible core may move forward, but neither the raw
        # hand-contaminated mask nor its sanitized replacement can teach the
        # full-target recovery scale authority.
        assert provider._full_target_geometry_authority.frame_id == (
            full_authority_before.frame_id
        )
        assert provider._full_target_geometry_authority.area == pytest.approx(
            full_authority_before.area
        )
        assert provider._trusted_online_sam2_publication.frame_id == (
            trusted_before.frame_id
        )
        np.testing.assert_array_equal(
            provider._trusted_online_sam2_publication.mask,
            trusted_before.mask,
        )
        assert provider._full_target_scale_transition_hypothesis is None
        assert provider._trusted_online_sam2_transient_gap_active
    finally:
        provider.stop()


def test_guarded_primary_exact_split_partial_only_advances_final_clean(
    monkeypatch,
):
    full = _large_mask((40, 30, 80, 70))
    visible = full.copy()
    visible[30:70, 57:63] = 0  # 15% known-missing aligned silhouette.
    provider, _tracker = _guarded_primary_provider(
        monkeypatch,
        [
            _moving_target_frame(2, (40, 30, 80, 70)),
            _moving_target_frame(3, (40, 30, 80, 70)),
            _moving_target_frame(4, (40, 30, 80, 70)),
        ],
        {
            2: _video_result(2, visible),
            3: _video_result(3, visible),
            4: _video_result(4, full),
        },
        full,
    )
    full_before = copy.deepcopy(provider._full_target_geometry_authority)
    trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)
    try:
        _frame2, result2 = provider.step_mask_only()
        assert result2.valid
        np.testing.assert_array_equal(result2.mask, visible)
        assert provider._last_final_clean_publication.frame_id == 2
        assert provider._full_target_geometry_authority.frame_id == (
            full_before.frame_id
        )
        assert provider._trusted_online_sam2_publication.frame_id == (
            trusted_before.frame_id
        )
        assert provider._last_publication_guard_known_partial
        assert "coverage=0.850<0.900" in (
            provider._last_publication_guard_partial_reason
        )
        assert provider._trusted_online_sam2_transient_gap_active
        assert provider._full_target_scale_transition_hypothesis is None

        _frame3, result3 = provider.step_mask_only()
        assert result3.valid, (
            provider.online_sam2_status,
            provider._last_publication_guard_status,
            provider._last_visible_exact_partial_reason,
            result3.message,
        )
        np.testing.assert_array_equal(result3.mask, visible)
        assert provider._last_final_clean_publication.frame_id == 3
        assert provider._full_target_geometry_authority.frame_id == (
            full_before.frame_id
        )
        assert provider._trusted_online_sam2_publication.frame_id == (
            trusted_before.frame_id
        )
        assert provider._full_target_scale_transition_hypothesis is None

        _frame4, result4 = provider.step_mask_only()
        assert result4.valid
        np.testing.assert_array_equal(result4.mask, full)
        assert provider._last_final_clean_publication.frame_id == 4
        assert provider._full_target_geometry_authority.frame_id == 4
        assert provider._full_target_geometry_authority.scale_samples == 0
        assert provider._trusted_online_sam2_publication.frame_id == 4
        assert provider._trusted_online_sam2_publication.velocity_samples == 0
        assert not provider._last_publication_guard_known_partial
        assert not provider._trusted_online_sam2_transient_gap_active
    finally:
        provider.stop()


def test_guarded_primary_sustained_fast_shrink_updates_scale_authority(
    monkeypatch,
):
    initial_bbox = (40, 30, 80, 70)
    shrinking_bboxes = {
        2: (75, 32, 111, 68),
        3: (105, 45, 137, 77),
        4: (125, 60, 155, 90),
        5: (130, 68, 158, 96),
    }
    initial = _large_mask(initial_bbox)
    shrinking = {
        frame_id: _large_mask(bbox) for frame_id, bbox in shrinking_bboxes.items()
    }
    provider, _tracker = _guarded_primary_provider(
        monkeypatch,
        [
            _moving_target_frame(frame_id, shrinking_bboxes[frame_id])
            for frame_id in sorted(shrinking_bboxes)
        ],
        {
            frame_id: _video_result(frame_id, shrinking[frame_id])
            for frame_id in shrinking
        },
        initial,
    )
    try:
        _frame2, result2 = provider.step_mask_only()
        assert result2.valid
        np.testing.assert_array_equal(result2.mask, shrinking[2])
        assert provider._full_target_geometry_authority.frame_id == 1
        assert provider._full_target_scale_transition_hypothesis.frame_id == 2

        _frame3, result3 = provider.step_mask_only()
        assert result3.valid
        np.testing.assert_array_equal(result3.mask, shrinking[3])
        assert provider._full_target_geometry_authority.frame_id == 3
        assert provider._full_target_geometry_authority.area == pytest.approx(
            float(shrinking[3].sum())
        )
        assert provider._full_target_geometry_authority.log_area_rate_s == (
            pytest.approx(
                np.log(shrinking[3].sum() / shrinking[2].sum()) / (1.0 / 30.0)
            )
        )
        assert provider._full_target_geometry_authority.scale_samples == 1
        assert provider._trusted_online_sam2_publication.frame_id == 3
        assert provider._full_target_scale_transition_hypothesis is None

        _frame4, result4 = provider.step_mask_only()
        assert result4.valid
        np.testing.assert_array_equal(result4.mask, shrinking[4])
        assert provider._full_target_geometry_authority.frame_id == 3
        assert provider._full_target_scale_transition_hypothesis.frame_id == 4

        _frame5, result5 = provider.step_mask_only()
        assert result5.valid
        np.testing.assert_array_equal(result5.mask, shrinking[5])
        assert provider._full_target_geometry_authority.frame_id == 5
        assert provider._full_target_geometry_authority.area == pytest.approx(
            float(shrinking[5].sum())
        )
        assert provider._full_target_geometry_authority.log_area_rate_s == (
            pytest.approx(
                np.log(shrinking[5].sum() / shrinking[4].sum()) / (1.0 / 30.0)
            )
        )
        assert provider._full_target_geometry_authority.scale_samples == 2
        assert provider._trusted_online_sam2_publication.frame_id == 5
        assert provider._full_target_scale_transition_hypothesis is None

        recovery_bbox = (132, 72, 158, 98)
        recovery_mask = _large_mask(recovery_bbox)
        recovery_frame = _moving_target_frame(6, recovery_bbox)
        recovery_geometry, reason = provider._online_mask_geometry(
            recovery_frame, recovery_mask
        )
        assert recovery_geometry is not None, reason
        (
            recovery_ok,
            recovery_reason,
        ) = provider._guarded_primary_recovery_geometry_accepts(
            recovery_frame, recovery_geometry
        )
        assert recovery_ok, recovery_reason
    finally:
        provider.stop()


def test_guarded_primary_scale_transition_has_dedicated_shape_floor(
    monkeypatch,
):
    full_bbox = (40, 30, 80, 70)
    shrinking_bboxes = {
        # Persistent bbox isotropy is 34/38=.895: deliberately below the
        # normal 0.90 partial-authority floor, but above the transition 0.85.
        2: (80, 31, 114, 69),
        # Relative to the unchanged 40x40 full anchor this is 30/34=.882.
        3: (120, 33, 150, 67),
    }
    full = _large_mask(full_bbox)
    shrinking = {
        frame_id: _large_mask(bbox) for frame_id, bbox in shrinking_bboxes.items()
    }
    provider, _tracker = _guarded_primary_provider(
        monkeypatch,
        [
            _moving_target_frame(frame_id, shrinking_bboxes[frame_id])
            for frame_id in sorted(shrinking_bboxes)
        ],
        {
            frame_id: _video_result(frame_id, shrinking[frame_id])
            for frame_id in shrinking
        },
        full,
    )
    try:
        _frame2, result2 = provider.step_mask_only()
        assert result2.valid
        assert provider._full_target_geometry_authority.frame_id == 1
        assert provider._full_target_scale_transition_hypothesis.frame_id == 2
        assert "normalized=0.895" in provider._last_publication_guard_status

        _frame3, result3 = provider.step_mask_only()
        assert result3.valid
        np.testing.assert_array_equal(result3.mask, shrinking[3])
        assert provider._full_target_geometry_authority.frame_id == 3
        assert provider._trusted_online_sam2_publication.frame_id == 3
        assert provider._full_target_scale_transition_hypothesis is None
        assert "normalized=0.882" in provider._last_publication_guard_status
    finally:
        provider.stop()


def test_guarded_primary_progressive_connected_partial_cannot_teach_scale(
    monkeypatch,
):
    full_bbox = (40, 30, 80, 70)
    partial_bboxes = {
        2: (40, 30, 74, 70),  # 85% of the persistent full silhouette.
        3: (40, 30, 71, 70),  # 91% of frame 2, but only 77.5% of full.
    }
    full = _large_mask(full_bbox)
    partials = {
        frame_id: _large_mask(bbox) for frame_id, bbox in partial_bboxes.items()
    }
    provider, _tracker = _guarded_primary_provider(
        monkeypatch,
        [
            _moving_target_frame(frame_id, partial_bboxes[frame_id])
            for frame_id in sorted(partial_bboxes)
        ],
        {
            frame_id: _video_result(frame_id, partials[frame_id])
            for frame_id in partials
        },
        full,
    )
    authority_before = copy.deepcopy(provider._full_target_geometry_authority)
    trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)
    try:
        _frame2, result2 = provider.step_mask_only()
        assert result2.valid
        assert provider._full_target_scale_transition_hypothesis is None
        assert provider._full_target_geometry_authority.frame_id == (
            authority_before.frame_id
        )
        assert "insufficient object translation" in (
            provider._last_publication_guard_status
        )

        _frame3, result3 = provider.step_mask_only()
        assert result3.valid
        np.testing.assert_array_equal(result3.mask, partials[3])
        assert provider._full_target_scale_transition_hypothesis is None
        assert provider._full_target_geometry_authority.frame_id == (
            authority_before.frame_id
        )
        assert provider._full_target_geometry_authority.area == pytest.approx(
            authority_before.area
        )
        assert provider._trusted_online_sam2_publication.frame_id == (
            trusted_before.frame_id
        )
        assert provider._visible_exact_partial_continuity.frame_id == 3
        assert provider._visible_exact_partial_continuity.hits == 2
        assert "persistent=0.775" in (provider._last_publication_guard_status)
    finally:
        provider.stop()


def test_guarded_primary_concentric_isotropic_erosion_cannot_teach_scale(
    monkeypatch,
):
    full_bbox = (40, 30, 80, 70)
    eroded_bboxes = {
        2: (42, 32, 78, 68),
        3: (44, 34, 76, 66),
    }
    full = _large_mask(full_bbox)
    eroded = {frame_id: _large_mask(bbox) for frame_id, bbox in eroded_bboxes.items()}
    provider, _tracker = _guarded_primary_provider(
        monkeypatch,
        [
            _moving_target_frame(frame_id, eroded_bboxes[frame_id])
            for frame_id in sorted(eroded_bboxes)
        ],
        {frame_id: _video_result(frame_id, eroded[frame_id]) for frame_id in eroded},
        full,
    )
    authority_before = copy.deepcopy(provider._full_target_geometry_authority)
    trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)
    try:
        _frame2, result2 = provider.step_mask_only()
        _frame3, result3 = provider.step_mask_only()

        # A hand can occlude an annulus while leaving a clean, centred target
        # core.  Shape normalization alone reports 1.0 for this 40->36->32
        # erosion, so it must remain visible-partial evidence only.
        assert result2.valid and result3.valid
        assert provider._full_target_geometry_authority.frame_id == (
            authority_before.frame_id
        )
        assert provider._full_target_geometry_authority.area == pytest.approx(
            authority_before.area
        )
        assert provider._trusted_online_sam2_publication.frame_id == (
            trusted_before.frame_id
        )
        assert "insufficient object translation" in (
            provider._last_publication_guard_status
        )
    finally:
        provider.stop()


def test_guarded_primary_real_fast_scale_trace_never_enters_recovery(
    monkeypatch,
):
    # Reviewed source65..72 ball trace from the pinned fast-entry RGB replay.
    # Its blur axis changes while the area shrinks, so persistent bbox
    # isotropy intentionally does not classify every frame as a new full mask.
    trace = {
        65: ((40, 40, 87, 96), 1.00),
        66: ((49, 46, 94, 95), 0.95),
        68: ((63, 27, 110, 71), 0.86),
        69: ((70, 24, 117, 67), 0.80),
        71: ((86, 34, 131, 81), 0.83),
        72: ((92, 45, 137, 92), 0.82),
    }
    masks = {
        frame_id: _large_ellipse_mask(bbox)
        for frame_id, (bbox, _support) in trace.items()
    }
    frames = [
        _moving_mask_support_frame(frame_id, masks[frame_id], support)
        for frame_id, (_bbox, support) in trace.items()
        if frame_id != 65
    ]
    tracker = _IdentityAwareFakeTracker({})
    manager = _FakeOnlineSAM2Manager(
        {
            frame_id: _video_result(frame_id, masks[frame_id])
            for frame_id in trace
            if frame_id != 65
        }
    )
    provider, _camera = _provider(
        monkeypatch,
        frames,
        tracker,
        manager,
        recovery_publication_mode="unified_three_evidence",
    )
    provider.online_sam2_cfg["mask_publication_mode"] = "guarded_sam2_primary"
    seed = _moving_mask_support_frame(65, masks[65], 1.0)
    assert provider.initialize_from_mask(seed, masks[65])
    initial_trusted = copy.deepcopy(provider._trusted_online_sam2_publication)
    try:
        outputs = [provider.step_mask_only() for _ in frames]

        assert all(result.valid for _frame_out, result in outputs)
        assert [frame.frame_id for frame, _result in outputs] == [
            66,
            68,
            69,
            71,
            72,
        ]
        assert provider.tracking_committed
        assert provider._trusted_online_sam2_continuity_intact
        assert provider._last_final_clean_publication.frame_id == 72
        # Partial-exact continuity is policy evidence only: unstable blur
        # shape never launders the old full/trusted geometry authority.
        assert provider._trusted_online_sam2_publication.frame_id == (
            initial_trusted.frame_id
        )
        bootstrap = provider.guarded_v2_bootstrap_state
        assert bootstrap["phase"] == "expired"
        assert "anisotropic to the provisional seed" in bootstrap["status"]
        assert provider._visible_exact_partial_continuity.frame_id == 72
        assert provider._visible_exact_partial_continuity.hits >= 2
        assert provider._last_online_sam2_status.startswith(
            "guarded_primary_visible_partial_continuation:"
        )
    finally:
        provider.stop()


def test_guarded_primary_source104_core_keeps_owner_for_source105(
    monkeypatch,
):
    """A 943/1610 core may continue only through its 943/986 core chain."""

    initial = _large_mask((40, 30, 80, 70))
    provider, tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)

    raw104 = _large_mask((80, 40, 126, 75))
    ys, xs = np.nonzero(raw104 > 0)
    center_x = float(np.mean(xs))
    center_y = float(np.mean(ys))
    radius_x = 0.5 * float(np.ptp(xs) + 1)
    radius_y = 0.5 * float(np.ptp(ys) + 1)
    rank = np.minimum(
        np.abs(xs - center_x) / radius_x,
        np.abs(ys - center_y) / radius_y,
    )
    order = np.argsort(rank, kind="stable")
    core104 = np.zeros_like(raw104)
    core104[ys[order[:943]], xs[order[:943]]] = 1
    aligned_core102 = core104.copy()
    remaining = order[943:]
    aligned_core102[ys[remaining[:43]], xs[remaining[:43]]] = 1
    assert int(raw104.sum()) == 1610
    assert int(core104.sum()) == 943
    assert int(aligned_core102.sum()) == 986

    def shift_right(mask, pixels):
        shifted = np.zeros_like(mask)
        if pixels >= 0:
            shifted[:, pixels:] = mask[:, : mask.shape[1] - pixels]
        else:
            shifted[:, :pixels] = mask[:, -pixels:]
        return shifted

    core102 = shift_right(aligned_core102, -4)
    raw102 = shift_right(raw104, -4)
    core102_bbox = np.asarray([76, 40, 122, 75], dtype=np.int32)
    core102_center = provider._binary_mask_centroid(core102)
    raw102_center = provider._binary_mask_centroid(raw102)
    assert core102_center is not None and raw102_center is not None
    provider._visible_exact_partial_continuity = SimpleNamespace(
        frame_id=102,
        timestamp=102.0 / 30.0,
        mask=core102.copy(),
        mask_digest=provider._publication_mask_digest(core102),
        core_bbox_xyxy=core102_bbox.copy(),
        core_center_xy=core102_center.copy(),
        core_area=986.0,
        bbox_xyxy=core102_bbox.copy(),
        center_xy=raw102_center.copy(),
        area=float(raw102.sum()),
        bbox_area=float((122 - 76) * (75 - 40)),
        depth_median=0.8,
        center_velocity_px_s=np.asarray([60.0, 0.0]),
        velocity_samples=10,
        hits=11,
        evidence_kind="sanitized_target_core",
        sanitized_handoff_eligible=True,
        sanitized_anchor_bbox_xyxy=core102_bbox.copy(),
        sanitized_anchor_center_xy=core102_center.copy(),
        sanitized_anchor_area=986.0,
    )
    authority_before = copy.deepcopy(provider._full_target_geometry_authority)
    trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)
    frame104 = _large_frame(104)
    frame104.color_bgr[:] = 45
    frame104.color_bgr[core104 > 0] = (210, 80, 235)
    provider._defer_full_target_geometry_commit(frame104, raw104)
    provider._mark_pending_full_target_geometry_ineligible(
        frame104, "artifact source104 sanitizer-owned core"
    )
    pending = provider._pending_full_target_geometry_commit
    assert pending is not None
    provider._last_output_mask_source = "publication_sanitized_target_core"
    provider._last_online_sam2_status = (
        "guarded_primary_visible_partial_continuation: artifact source104"
    )
    try:
        expected_digest = pending.mask_digest
        pending.mask_digest = "0" * 64
        assert not provider._commit_visible_exact_partial_continuity(
            frame104,
            _mask_result(core104, message="wrong raw digest"),
            pending,
        )
        pending.mask_digest = expected_digest

        original_stats = tracker.target_appearance_support_stats
        tracker.target_appearance_support_stats = lambda *_args, **_kwargs: (
            float("nan"),
            1.0,
            943,
        )
        assert not provider._commit_visible_exact_partial_continuity(
            frame104,
            _mask_result(core104, message="non-finite scorer"),
            pending,
        )
        tracker.target_appearance_support_stats = original_stats

        previous = provider._visible_exact_partial_continuity
        previous.frame_id = 100
        previous.timestamp = 100.0 / 30.0
        assert not provider._commit_visible_exact_partial_continuity(
            frame104,
            _mask_result(core104, message="stale core owner"),
            pending,
        )
        previous.frame_id = 102
        previous.timestamp = 102.0 / 30.0

        assert provider._commit_visible_exact_partial_continuity(
            frame104,
            _mask_result(core104, message="source104 sanitized core"),
            pending,
        )
        state = provider._visible_exact_partial_continuity
        assert state.frame_id == 104
        assert state.evidence_kind == "sanitized_target_core"
        assert state.core_area == pytest.approx(943.0)
        assert state.area == pytest.approx(1610.0)
        assert state.hits == 12

        raw105 = shift_right(raw104, 2)
        frame105 = _moving_mask_support_frame(105, raw105, 0.64)
        geometry105, reason105 = provider._online_mask_geometry(frame105, raw105)
        assert geometry105 is not None, reason105
        accepted105, detail105 = provider._visible_exact_partial_candidate_agrees(
            frame105,
            geometry105,
            allow_boundary_support=True,
        )
        assert accepted105, detail105
        assert provider._full_target_geometry_authority.frame_id == (
            authority_before.frame_id
        )
        assert provider._trusted_online_sam2_publication.frame_id == (
            trusted_before.frame_id
        )
    finally:
        provider.stop()


def _install_confirmed_sanitized_partial_owner(provider):
    """Install the reviewed source51-style non-authoritative core owner."""

    previous = _large_mask((60, 30, 100, 70))
    frame = _moving_target_frame(51, (60, 30, 100, 70))
    assert provider._commit_final_clean_publication(
        frame, previous, rebase_motion_history=True
    )
    center = provider._binary_mask_centroid(previous)
    assert center is not None
    bbox = np.asarray([60, 30, 100, 70], dtype=np.int32)
    provider._visible_exact_partial_continuity = SimpleNamespace(
        frame_id=51,
        timestamp=51.0 / 30.0,
        mask=previous.copy(),
        mask_digest=provider._publication_mask_digest(previous),
        core_bbox_xyxy=bbox.copy(),
        core_center_xy=center.copy(),
        core_area=1600.0,
        bbox_xyxy=bbox.copy(),
        center_xy=center.copy(),
        area=1600.0,
        bbox_area=1600.0,
        depth_median=0.8,
        center_velocity_px_s=np.asarray([30.0, 0.0], dtype=np.float64),
        velocity_samples=2,
        hits=3,
        evidence_kind="sanitized_target_core",
        sanitized_handoff_eligible=True,
        sanitized_anchor_bbox_xyxy=bbox.copy(),
        sanitized_anchor_center_xy=center.copy(),
        sanitized_anchor_area=1600.0,
    )
    return previous


def _install_confirmed_raw_partial_owner(
    provider,
    *,
    frame_id=48,
    bbox=(68, 30, 108, 70),
):
    """Install a confirmed, non-authoritative raw-exact predecessor."""

    mask = _large_mask(bbox)
    center = provider._binary_mask_centroid(mask)
    assert center is not None
    bbox_array = np.asarray(bbox, dtype=np.int32)
    provider._visible_exact_partial_continuity = SimpleNamespace(
        frame_id=int(frame_id),
        timestamp=float(frame_id) / 30.0,
        mask=mask.copy(),
        mask_digest=provider._publication_mask_digest(mask),
        core_bbox_xyxy=bbox_array.copy(),
        core_center_xy=center.copy(),
        core_area=float(mask.sum()),
        bbox_xyxy=bbox_array.copy(),
        center_xy=center.copy(),
        area=float(mask.sum()),
        bbox_area=float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
        depth_median=0.8,
        center_velocity_px_s=np.asarray([30.0, 0.0], dtype=np.float64),
        velocity_samples=3,
        hits=2,
        evidence_kind="raw_exact",
        sanitized_handoff_eligible=False,
        sanitized_anchor_bbox_xyxy=None,
        sanitized_anchor_center_xy=None,
        sanitized_anchor_area=None,
    )
    return provider._visible_exact_partial_continuity


def _bind_current_sanitized_evidence(provider, frame, raw, final):
    """Return exact provider-owned raw/final tokens for one focused test."""

    raw_evidence = provider._defer_full_target_geometry_commit(frame, raw)
    assert raw_evidence is not None
    provider._mark_pending_full_target_geometry_ineligible(
        frame, "focused sanitized partial evidence"
    )
    final_evidence, reason = provider._register_frame_mask_evidence(
        frame,
        final,
        evidence_kind="sanitized_target_core",
        authority_eligible=False,
    )
    assert final_evidence is not None, reason
    pending = provider._pending_full_target_geometry_commit
    assert pending is not None
    return pending, final_evidence.mask


def _exact_token_mask_result(mask, *, message):
    """Test adapter which deliberately preserves evidence object identity."""

    result = _mask_result(mask, message=message)
    result.mask = mask
    return result


def _install_broad_visible_core_probability_map(tracker, probability):
    """Install one deterministic immutable-appearance probability image."""

    probability = np.asarray(probability, dtype=np.float32).copy()

    def support_stats(_frame, mask, *, threshold=None):
        selected = np.asarray(mask) > 0
        values = probability[selected]
        if values.size == 0:
            return 0.0, 0.0, 0
        threshold_value = 0.50 if threshold is None else float(threshold)
        return (
            float(np.mean(values)),
            float(np.mean(values >= threshold_value)),
            int(values.size),
        )

    def supported_mask(_frame, mask, *, threshold=None):
        threshold_value = 0.50 if threshold is None else float(threshold)
        return np.logical_and(
            np.asarray(mask) > 0,
            probability >= threshold_value,
        ).astype(np.uint8)

    tracker.target_appearance_support_stats = support_stats
    tracker.target_appearance_supported_mask = supported_mask


def _stage_broad_visible_core_case(monkeypatch, *, variant="accepted"):
    """Bind one exact raw/strict/fixed-anchor broad-core transaction."""

    initial = _large_mask((40, 30, 80, 70))
    provider, tracker = _guarded_primary_provider(
        monkeypatch, [], {}, initial
    )
    provider._guarded_v2_bootstrap_phase = "commissioned"

    anchor_bbox = (60, 30, 100, 70)
    if variant == "boundary_anchor":
        anchor_bbox = (4, 30, 44, 70)
    previous = _install_confirmed_raw_partial_owner(
        provider,
        frame_id=51,
        bbox=anchor_bbox,
    )
    # This raw-exact frame is an adjacent bridge inside an already-confirmed
    # sanitizer lineage.  It may seed one new broad episode, but may not
    # create sanitizer authority on its own.
    previous.sanitized_handoff_eligible = True
    previous.hits = 3

    frame = _large_frame(52)
    if variant == "low_overlap":
        # At the best bounded alignment only 444/1600=.278 of the fixed
        # anchor overlaps this candidate.  Candidate coverage remains high,
        # so the fixed-anchor scale test is the isolated rejection.
        broad = _large_mask((74, 39, 86, 78))  # 468px.
        strict = _large_mask((77, 45, 83, 55))  # 60px.
        raw = _large_mask((69, 34, 91, 83))
    elif variant == "boundary_anchor":
        # Keep current raw/broad/strict evidence well inside the image so the
        # rejection is attributable only to the stored pre-occlusion anchor.
        broad = _large_mask((20, 39, 72, 61))
        strict = _large_mask((30, 41, 46, 59))
        raw = _large_mask((17, 25, 77, 75))
    else:
        # The complete broad candidate is 52x22=1144px.  A 40x40 anchor can
        # cover only 40x22=880px, so publishing candidate&anchor would lose
        # 264 independently supported target pixels.
        broad = _large_mask((60, 39, 112, 61))
        strict = _large_mask((76, 41, 92, 59))  # 288px.
        raw = _large_mask((55, 25, 117, 75))

    probability = np.full(raw.shape, 0.05, dtype=np.float32)
    probability[broad > 0] = 0.30
    if variant != "strict_empty":
        strict_probability = strict
        if variant == "deep_seedable":
            # Keep the final strict publication small while the independent
            # p=.42 identity witness covers 704/3100=.227 of raw, matching
            # the source222 seed contract.
            strict_probability = _large_mask((68, 39, 100, 61))
        probability[strict_probability > 0] = 0.95
    _install_broad_visible_core_probability_map(tracker, probability)

    raw_evidence = provider._defer_full_target_geometry_commit(frame, raw)
    assert raw_evidence is not None
    strict_evidence, strict_reason = provider._register_frame_mask_evidence(
        frame,
        strict,
        evidence_kind="sanitized_target_core",
        authority_eligible=False,
    )
    assert strict_evidence is not None, strict_reason
    staged, stage_reason = provider._stage_broad_visible_core(
        frame,
        raw_evidence,
        strict_evidence,
        previous,
    )
    pending = provider._pending_full_target_geometry_commit
    assert pending is not None
    return SimpleNamespace(
        provider=provider,
        tracker=tracker,
        frame=frame,
        previous=previous,
        raw_evidence=raw_evidence,
        strict_evidence=strict_evidence,
        pending=pending,
        staged=staged,
        stage_reason=stage_reason,
        expected_broad=broad,
    )


def _stage_deep_occlusion_broad_pair_case(monkeypatch, *, variant="accepted"):
    """Stage a source224->225-like fixed-anchor deep visible-core pair."""

    initial = _large_mask((60, 30, 100, 70))
    provider, tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    previous = _install_confirmed_raw_partial_owner(
        provider, frame_id=60, bbox=(64, 38, 96, 62)
    )
    previous.sanitized_handoff_eligible = True
    previous.hits = 5
    previous.evidence_kind = "sanitized_target_core"
    fixed_anchor = _large_mask((60, 30, 100, 70))
    previous.broad_visible_anchor_mask = fixed_anchor.copy()
    previous.broad_visible_anchor_digest = provider._publication_mask_digest(
        fixed_anchor
    )
    previous.broad_visible_anchor_bbox_xyxy = np.array(
        [60, 30, 100, 70], dtype=np.int32
    )
    previous.broad_visible_anchor_center_xy = np.array(
        [79.5, 49.5], dtype=np.float64
    )
    previous.broad_visible_anchor_area = 1600.0
    previous.broad_visible_anchor_frame_id = 1

    frame_a = _large_frame(61)
    raw_a = _large_mask((55, 25, 105, 65))  # 2000px.
    broad_a = _large_mask((62, 38, 98, 58))  # 720px=.36 raw/.45 anchor.
    strict_a = _large_mask((67, 39, 91, 57))  # 432px=.216 raw.
    final_strict_a = _large_mask((68, 40, 90, 56))
    probability_a = np.full(raw_a.shape, 0.05, dtype=np.float32)
    probability_a[broad_a > 0] = 0.30
    probability_a[strict_a > 0] = 0.95
    _install_broad_visible_core_probability_map(tracker, probability_a)
    raw_a_evidence = provider._defer_full_target_geometry_commit(frame_a, raw_a)
    assert raw_a_evidence is not None
    strict_a_evidence, reason = provider._register_frame_mask_evidence(
        frame_a,
        final_strict_a,
        evidence_kind="sanitized_target_core",
        authority_eligible=False,
    )
    assert strict_a_evidence is not None, reason
    first, first_reason = provider._stage_broad_visible_core(
        frame_a, raw_a_evidence, strict_a_evidence, previous
    )
    assert first is None
    assert "deep occlusion broad seed armed" in first_reason
    seed = provider._deep_occlusion_broad_seed
    assert seed is not None

    # The first frame remains on the existing strict path.  Bind the seed to
    # the exact strict owner which would cross that frame's final boundary.
    owner_a = copy.copy(previous)
    owner_a.frame_id = frame_a.frame_id
    owner_a.timestamp = frame_a.timestamp
    owner_a.mask = strict_a_evidence.mask
    owner_a.mask_digest = strict_a_evidence.mask_digest
    owner_a.mask_evidence = strict_a_evidence
    owner_a.core_bbox_xyxy = strict_a_evidence.geometry.bbox_xyxy.copy()
    owner_a.core_center_xy = strict_a_evidence.geometry.centroid_xy.copy()
    owner_a.core_area = strict_a_evidence.geometry.area
    owner_a.bbox_xyxy = raw_a_evidence.geometry.bbox_xyxy.copy()
    owner_a.center_xy = raw_a_evidence.geometry.centroid_xy.copy()
    owner_a.area = raw_a_evidence.geometry.area
    owner_a.bbox_area = raw_a_evidence.geometry.bbox_area
    owner_a.depth_median = raw_a_evidence.geometry.depth_median
    owner_a.hits = previous.hits + 1
    owner_a.evidence_kind = "sanitized_target_core"
    owner_a.sanitized_handoff_eligible = True
    owner_a.deep_occlusion_broad_confirmed = False
    provider._visible_exact_partial_continuity = owner_a
    provider._bind_deep_occlusion_seed_to_committed_owner(frame_a, owner_a)
    assert provider._deep_occlusion_broad_seed is not None

    frame_b = _large_frame(62)
    raw_b = _large_mask((57, 25, 107, 65))
    broad_b = _large_mask((64, 38, 100, 58))
    strict_b = _large_mask((69, 39, 93, 57))
    final_strict_b = _large_mask((70, 40, 92, 56))
    if variant == "absent_ratios":
        # Stable but hand-dominated: pair overlap is high, yet the independent
        # target-appearance/raw fractions match reviewed complete absence.
        broad_b = _large_mask((67, 40, 87, 58))
        strict_b = _large_mask((71, 43, 83, 55))
        final_strict_b = strict_b.copy()
    elif variant == "boundary":
        raw_b = _large_mask((4, 25, 54, 65))
        broad_b = _large_mask((8, 38, 44, 58))
        strict_b = _large_mask((13, 39, 37, 57))
        final_strict_b = _large_mask((14, 40, 36, 56))
    probability_b = np.full(raw_b.shape, 0.05, dtype=np.float32)
    probability_b[broad_b > 0] = 0.30
    probability_b[strict_b > 0] = 0.95
    _install_broad_visible_core_probability_map(tracker, probability_b)
    raw_b_evidence = provider._defer_full_target_geometry_commit(frame_b, raw_b)
    assert raw_b_evidence is not None
    strict_b_evidence, reason = provider._register_frame_mask_evidence(
        frame_b,
        final_strict_b,
        evidence_kind="sanitized_target_core",
        authority_eligible=False,
    )
    assert strict_b_evidence is not None, reason
    staged, stage_reason = provider._stage_broad_visible_core(
        frame_b, raw_b_evidence, strict_b_evidence, owner_a
    )
    return SimpleNamespace(
        provider=provider,
        tracker=tracker,
        previous=owner_a,
        frame=frame_b,
        pending=provider._pending_full_target_geometry_commit,
        raw_evidence=raw_b_evidence,
        staged=staged,
        stage_reason=stage_reason,
        fixed_anchor=fixed_anchor,
        authority_before=copy.deepcopy(provider._full_target_geometry_authority),
        trusted_before=copy.deepcopy(provider._trusted_online_sam2_publication),
    )


def _stage_lost_reappearance_visible_core_case(
    monkeypatch,
    *,
    variant="accepted",
):
    """Bind a source312->314-like two-frame LOST core transaction."""

    # The fixed full-target anchor is deliberately larger than the first
    # visible reappearance fragment.  It stays byte-identical through both
    # frames and is never allowed to become a moving/shrinking ruler.
    full_anchor = _large_mask((40, 30, 76, 68))
    provider, tracker = _guarded_primary_provider(
        monkeypatch,
        [],
        {},
        full_anchor,
    )
    # The historical guarded-primary fixture predates immutable frame-mask
    # evidence and therefore initializes the trusted/full publication without
    # the raw-exact token required by the LOST reappearance anchor.  Bind the
    # exact initialization frame here instead of weakening the production
    # anchor contract for tests.
    anchor_frame = _moving_target_frame(1, (40, 30, 80, 70))
    anchor_evidence, anchor_reason = provider._register_frame_mask_evidence(
        anchor_frame,
        full_anchor,
        evidence_kind="raw_exact",
        authority_eligible=True,
    )
    assert anchor_evidence is not None, anchor_reason
    trusted_anchor = provider._trusted_online_sam2_publication
    assert trusted_anchor is not None
    trusted_anchor.mask = anchor_evidence.mask
    trusted_anchor.mask_evidence = anchor_evidence
    provider._guarded_v2_bootstrap_phase = "commissioned"
    provider._trusted_online_sam2_continuity_intact = False
    provider._trusted_online_sam2_continuity_reason = (
        "focused LOST reappearance fixture"
    )
    provider._tracking_committed = False

    # Seed: raw 21x33, p=.20 core 17x32, strict core 16x31.  This is the
    # compact first reappearance geometry of reviewed source312.
    seed_frame = _large_frame(312)
    seed_raw = _large_mask((50, 36, 71, 69))
    seed_broad = _large_mask((54, 37, 71, 69))
    seed_strict = _large_mask((55, 37, 71, 68))
    if variant == "pair_growth":
        # Keep appearance support just above the independent 0.65 identity
        # floor, so a dense second fixed-anchor-consistent core can isolate
        # the 2.5x pair growth limit without first failing raw-bbox IoU.
        seed_broad = _large_mask((56, 38, 71, 68))
        seed_broad[37, 56] = 1
        seed_strict = seed_broad.copy()
    seed_probability = np.full(seed_raw.shape, 0.05, dtype=np.float32)
    seed_probability[seed_broad > 0] = 0.30
    seed_probability[seed_strict > 0] = 0.95
    _install_broad_visible_core_probability_map(tracker, seed_probability)
    seed_raw_evidence = provider._defer_full_target_geometry_commit(
        seed_frame, seed_raw
    )
    assert seed_raw_evidence is not None
    seed_geometry = seed_raw_evidence.geometry
    seed_identity_ok, seed_identity_reason = (
        provider._online_sam2_independent_identity_accepts(
            seed_frame, seed_geometry
        )
    )
    assert seed_identity_ok, seed_identity_reason
    seed_geometry_ok, seed_geometry_reason = (
        provider._guarded_primary_recovery_geometry_accepts(
            seed_frame, seed_geometry
        )
    )
    assert seed_geometry_ok, seed_geometry_reason
    confirmed, hits, confirmation = provider._confirm_online_recovery(
        seed_frame.frame_id,
        seed_geometry,
        timestamp=seed_frame.timestamp,
        source="guarded_sam2_primary",
    )
    assert not confirmed and hits == 1, confirmation
    seed_reason = provider._arm_lost_reappearance_visible_core_seed(
        seed_frame, seed_raw_evidence, hits
    )
    seed = provider._lost_reappearance_visible_core_seed
    assert seed is not None, seed_reason
    seed_snapshot = copy.deepcopy(seed)
    recovery_hit1 = copy.deepcopy(provider._online_sam2_recovery_hypothesis)

    current_frame = _large_frame(314)
    current_raw = _large_mask((45, 31, 71, 68))
    current_broad = current_raw.copy()
    current_strict = _large_mask((46, 32, 70, 67))
    if variant == "low_broad_raw":
        current_raw = _large_mask((40, 30, 76, 68))
        current_broad = _large_mask((50, 39, 66, 59))
        current_strict = _large_mask((50, 39, 66, 59))
    elif variant == "low_strict_raw":
        current_raw = _large_mask((40, 30, 76, 68))
        current_broad = current_raw.copy()
        current_strict = _large_mask((50, 39, 66, 59))
    elif variant == "hand_union":
        target = _large_mask((45, 31, 71, 68))
        hand = _large_mask((71, 38, 87, 61))
        current_raw = np.maximum(target, hand)
        current_broad = current_raw.copy()
        current_strict = _large_mask((46, 32, 70, 67))
    elif variant == "appearance_empty":
        current_broad = np.zeros_like(current_raw)
        current_strict = np.zeros_like(current_raw)
    elif variant == "appearance_absent":
        current_broad = None
        current_strict = None
    elif variant == "low_anchor_coverage":
        current_raw = _large_mask((51, 41, 69, 59))
        current_broad = _large_mask((52, 42, 67, 57))
        current_strict = _large_mask((53, 43, 67, 57))
    elif variant == "low_candidate_coverage":
        target = _large_mask((45, 31, 71, 68))
        # Keep the foreign union farther than the bounded anchor alignment
        # radius, so no allowed shift can inflate candidate/anchor to 0.90.
        foreign = _large_mask((86, 39, 101, 60))
        current_raw = np.maximum(target, foreign)
        current_broad = current_raw.copy()
        current_strict = _large_mask((46, 32, 70, 67))
    elif variant == "boundary":
        current_raw = _large_mask((4, 31, 30, 68))
        current_broad = current_raw.copy()
        current_strict = _large_mask((5, 32, 29, 67))
    elif variant == "pair_bbox":
        current_raw = _large_mask((62, 36, 83, 69))
        current_broad = _large_mask((62, 36, 79, 68))
        current_strict = _large_mask((62, 37, 78, 68))
    elif variant == "pair_growth":
        current_raw = _large_mask((45, 33, 77, 69))
        current_broad = current_raw.copy()
        current_strict = current_raw.copy()

    if variant in ("low_valid_depth", "high_depth_spread"):
        ys, xs = np.nonzero(current_broad)
        if variant == "low_valid_depth":
            current_frame.depth_raw[ys, xs] = 0
            min_valid = float(
                provider.cfg["tracker"][
                    "recovery_publish_min_valid_depth_ratio"
                ]
            )
            valid_pixels = max(1, int(np.floor((min_valid - 0.01) * len(xs))))
            current_frame.depth_raw[ys[:valid_pixels], xs[:valid_pixels]] = 800
        else:
            midpoint = len(xs) // 2
            current_frame.depth_raw[ys[:midpoint], xs[:midpoint]] = 700
            current_frame.depth_raw[ys[midpoint:], xs[midpoint:]] = 1000

    current_probability = np.full(current_raw.shape, 0.05, dtype=np.float32)
    if current_broad is None:
        tracker.target_appearance_supported_mask = (
            lambda _frame, _mask, *, threshold=None: None
        )
    else:
        current_probability[current_broad > 0] = 0.30
        current_probability[current_strict > 0] = 0.95
        _install_broad_visible_core_probability_map(tracker, current_probability)
    current_raw_evidence = provider._defer_full_target_geometry_commit(
        current_frame, current_raw
    )
    assert current_raw_evidence is not None
    current_geometry = current_raw_evidence.geometry
    staged, stage_reason = provider._stage_lost_reappearance_visible_core(
        current_frame,
        current_raw_evidence,
        current_geometry,
    )
    pending = provider._pending_full_target_geometry_commit
    assert pending is not None
    return SimpleNamespace(
        provider=provider,
        tracker=tracker,
        seed_frame=seed_frame,
        seed_raw=seed_raw,
        seed_broad=seed_broad,
        seed_strict=seed_strict,
        seed_snapshot=seed_snapshot,
        recovery_hit1=recovery_hit1,
        current_frame=current_frame,
        current_raw=current_raw,
        current_broad=current_broad,
        current_strict=current_strict,
        current_raw_evidence=current_raw_evidence,
        current_geometry=current_geometry,
        pending=pending,
        staged=staged,
        stage_reason=stage_reason,
    )


def _internal_highlight_split_mask():
    """One full-extent target split into body/highlight components."""

    mask = np.zeros((120, 180), dtype=np.uint8)
    mask[40:70, 70:106] = 1  # 1080px body.
    mask[30:38, 94:110] = 1  # 128px internal highlight.
    return mask


def _fragmented_tail_masks(*, below_raw_majority: bool):
    """Build reviewed-tail geometry without encoding one production frame.

    ``raw`` is the containing exact-SAM silhouette.  ``aligned`` is the
    immutable, motion-aligned target support from the current frame.  The
    final sanitizer result contains a large body and a disconnected highlight
    which both remain inside that clean support.  The two variants exercise
    the two real tail regimes: a normal 0.54 raw retention and a recovery-only
    0.498 raw retention.  In both cases the final mask retains at least 90% of
    the aligned clean target, so raw retention alone must not be mistaken for
    target disappearance.
    """

    raw = _large_mask((70, 30, 110, 70))  # 1600px containing silhouette.
    final = np.zeros_like(raw)
    if below_raw_majority:
        # 775px body + 22px disconnected highlight = 797px.  The exact-SAM
        # silhouette has one more non-target pixel than the visible target,
        # so 797/1600=.498125 is deliberately just below one half.
        final[42:67, 72:103] = 1
        final[32:37, 104:108] = 1
        final[37:39, 107] = 1
        aligned = final.copy()
        aligned[39:41, 72:108] = 1
        aligned[41, 72:83] = 1
        expected_final = 797
        expected_aligned = 880
    else:
        # 832px body + 32px disconnected highlight = 864px.  The aligned
        # clean support adds only the object's fragmented transition band.
        final[42:68, 72:104] = 1
        final[32:40, 104:108] = 1
        aligned = final.copy()
        aligned[39:42, 72:104] = 1
        expected_final = 864
        expected_aligned = 960

    assert int(raw.sum()) == 1600
    assert int(final.sum()) == expected_final
    assert int(aligned.sum()) == expected_aligned
    assert not np.any(np.logical_and(final > 0, aligned == 0))
    assert float(final.sum()) / float(aligned.sum()) >= 0.90
    final_bbox = np.asarray(np.nonzero(final > 0))
    aligned_bbox = np.asarray(np.nonzero(aligned > 0))
    final_extent = np.ptp(final_bbox, axis=1) + 1
    aligned_extent = np.ptp(aligned_bbox, axis=1) + 1
    assert float(np.min(final_extent / aligned_extent)) >= 0.90
    return raw, aligned, final


def test_sanitized_tail_owner_uses_proven_final_guard_alignment(monkeypatch):
    """A safe published core advances across a one-predictor pixel miss."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    _install_confirmed_sanitized_partial_owner(provider)
    previous = provider._visible_exact_partial_continuity
    frame = _moving_target_frame(52, (62, 30, 102, 70))
    raw = _large_mask((62, 30, 102, 70))
    final = raw.copy()
    final[:, 101:102] = 0
    pending, final_token = _bind_current_sanitized_evidence(
        provider, frame, raw, final
    )
    provider._last_publication_guard_frame_id = frame.frame_id
    provider._last_publication_guard_aligned_clean = raw.copy()
    provider._last_output_mask_source = "publication_sanitized_target_core"
    provider._last_online_sam2_status = (
        "guarded_primary_visible_partial_continuation: reviewed boundary tail"
    )
    provider._last_online_sam2_exact_result = True
    monkeypatch.setattr(
        provider,
        "_aligned_committed_clean_mask",
        lambda *_args, **_kwargs: (
            previous.mask.copy(),
            "test deliberately stale partial-owner predictor",
        ),
    )
    try:
        # The ordinary exact-raw subset proof already passes.  The separate
        # current final-guard proof is nevertheless needed to authorize the
        # chain helper's cached-alignment fallback for x=100.
        safe_subset, subset_reason = (
            provider._sanitized_final_partial_subset_accepts(
                frame, final_token, pending.mask
            )
        )
        assert safe_subset, subset_reason
        assert np.any(
            np.logical_and(final_token > 0, np.asarray(previous.mask) == 0)
        )

        assert provider._commit_visible_exact_partial_continuity(
            frame,
            _exact_token_mask_result(
                final_token, message="reviewed boundary tail alignment"
            ),
            pending,
        )
        owner = provider._visible_exact_partial_continuity
        assert owner.frame_id == 52
        assert owner.hits == previous.hits + 1
        assert owner.evidence_kind == "sanitized_target_core"
        assert "final-guard aligned-clean fallback" in (
            provider._last_sanitized_partial_chain_reason
        )
    finally:
        provider.stop()


def test_sanitized_tail_owner_reuses_matching_final_clean_alignment(monkeypatch):
    """A 3/3 recovery core survives the next frame's predictor mismatch.

    The partial owner and the provider-owned final-clean history contain the
    same immutable mask bytes.  Their independent velocity predictors may
    nevertheless differ by a few pixels on the next RGB-D frame.  Only the
    matching final-clean alignment may repair that disagreement; the ordinary
    chain remains fail-closed when the explicit proof is not requested.
    """

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    _install_confirmed_sanitized_partial_owner(provider)
    previous = provider._visible_exact_partial_continuity
    frame = _moving_target_frame(52, (62, 30, 102, 70))
    raw = _large_mask((62, 30, 104, 70))
    final = _large_mask((62, 30, 102, 70))
    pending, final_token = _bind_current_sanitized_evidence(
        provider, frame, raw, final
    )

    def align_current(_frame, _raw, *, recovery_reference=None):
        if recovery_reference is not None:
            return (
                previous.mask.copy(),
                "focused partial-owner predictor misses two columns",
            )
        return final_token.copy(), "focused matching final-clean predictor"

    monkeypatch.setattr(provider, "_aligned_committed_clean_mask", align_current)
    try:
        accepted_without_proof, reason_without_proof = (
            provider._sanitized_partial_chain_accepts(
                frame,
                final_token,
                provider._pending_full_target_geometry_commit.mask,
                previous,
            )
        )
        assert not accepted_without_proof
        assert "outside aligned predecessor" in reason_without_proof

        accepted, reason = provider._sanitized_partial_chain_accepts(
            frame,
            final_token,
            provider._pending_full_target_geometry_commit.mask,
            previous,
            allow_current_clean_alignment=True,
        )
        assert accepted, reason
        assert "current exact final-clean alignment fallback" in reason

        # The final-history boundary must consume the exact same proof.  A
        # previous implementation accepted the robot-facing mask above but
        # then cleared its owner here, so the following sampled frame fell
        # back into LOST.  Deliberately disable the separate final-guard
        # fallback: this assertion depends on the byte-matched final-clean
        # alignment, not on a second implicit exception.
        monkeypatch.setattr(
            provider,
            "_aligned_clean_visible_core_handoff_accepts",
            lambda *_args, **_kwargs: (False, "focused guard unavailable"),
        )
        provider._last_output_mask_source = (
            "publication_sanitized_target_core"
        )
        provider._last_online_sam2_status = (
            "guarded_primary_visible_partial_sanitized_continuation: "
            "focused matching final-clean owner"
        )
        provider._last_online_sam2_exact_result = True
        committed = provider._commit_visible_exact_partial_continuity(
            frame,
            _exact_token_mask_result(
                final_token, message="matching final-clean owner"
            ),
            pending,
        )
        assert committed
        owner = provider._visible_exact_partial_continuity
        assert owner.frame_id == frame.frame_id
        assert owner.hits == previous.hits + 1
        assert owner.evidence_kind == "sanitized_target_core"
        assert owner.sanitized_handoff_eligible
    finally:
        provider.stop()


def test_commissioned_trusted_sanitizer_seeds_confirmed_partial_owner(monkeypatch):
    """Trusted exact geometry plus one safe final core avoids false LOST."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    provider._visible_exact_partial_continuity = None
    trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)
    full_before = copy.deepcopy(provider._full_target_geometry_authority)
    frame = _moving_target_frame(2, (42, 30, 82, 70))
    frame.color_bgr[30:70, 78:82] = 45
    raw = _large_mask((42, 30, 82, 70))
    final = _large_mask((42, 30, 78, 70))
    pending, final_token = _bind_current_sanitized_evidence(
        provider, frame, raw, final
    )
    provider._last_publication_guard_frame_id = frame.frame_id
    provider._last_publication_guard_aligned_clean = raw.copy()
    provider._last_output_mask_source = "publication_sanitized_target_core"
    provider._last_online_sam2_status = (
        "guarded_primary_trusted: focused commissioned handoff"
    )
    provider._last_online_sam2_exact_result = True
    try:
        assert provider._commit_visible_exact_partial_continuity(
            frame,
            _exact_token_mask_result(
                final_token, message="trusted sanitized handoff"
            ),
            pending,
        )
        owner = provider._visible_exact_partial_continuity
        assert owner.frame_id == 2
        assert owner.hits == 2
        assert owner.evidence_kind == "sanitized_target_core"
        assert owner.sanitized_handoff_eligible
        assert provider._trusted_online_sam2_publication.frame_id == (
            trusted_before.frame_id
        )
        assert provider._full_target_geometry_authority.frame_id == (
            full_before.frame_id
        )
    finally:
        provider.stop()


def test_commissioned_sanitized_commit_trace_keeps_translated_core_fresh(
    monkeypatch,
):
    """Source47/50-style final provenance carries into source53/54."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    provider._visible_exact_partial_continuity = None
    provider._trusted_online_sam2_publication.frame_id = 45
    provider._trusted_online_sam2_publication.timestamp = 45.0 / 30.0
    authority_before = copy.deepcopy(provider._full_target_geometry_authority)
    trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)

    raw47 = _large_mask((42, 20, 82, 60))
    final47 = raw47.copy()
    # 1342/1600=.839, matching the reviewed first sanitized handoff: below
    # ordinary raw retention while preserving the full target extent.
    final47[45:51, 43:82] = 0
    final47[51, 43:67] = 0
    frame47 = _moving_target_frame(47, (42, 20, 82, 60))
    pending47, token47 = _bind_current_sanitized_evidence(
        provider, frame47, raw47, final47
    )
    provider._last_publication_guard_frame_id = frame47.frame_id
    provider._last_publication_guard_aligned_clean = raw47.copy()
    provider._last_publication_guard_source = "online_sam2_guarded_primary"
    provider._last_output_mask_source = "publication_sanitized_target_core"
    provider._last_online_sam2_status = (
        "guarded_primary_trusted: trusted dt=0.067s speed=821.9px/s "
        "log_area_rate=0.16/s"
    )
    provider._last_online_sam2_exact_result = True
    provider._tracking_committed = True

    try:
        assert not pending47.raw_exact_eligible
        obj47 = provider._mask_only_publication_evidence(frame47, token47)
        provider._commit_final_publication_histories(
            frame47,
            _exact_token_mask_result(
                token47, message="reviewed source47 sanitized publication"
            ),
            obj47,
        )
        owner47 = provider._visible_exact_partial_continuity
        assert owner47 is not None
        assert owner47.frame_id == 47
        assert owner47.hits == 2
        assert owner47.evidence_kind == "sanitized_target_core"

        # Translate the exact source47 core, then retain three quarters of
        # its pixels. The bbox extent remains intact, so this is translated
        # visibility contraction rather than concentric/cumulative erosion.
        aligned50 = np.zeros_like(final47)
        aligned50[:110, 12:] = final47[10:, :168]
        final50 = aligned50.copy()
        ys50, xs50 = np.nonzero(final50)
        final50[ys50[::4], xs50[::4]] = 0
        raw50 = _large_mask((54, 10, 94, 50))
        frame50 = _moving_target_frame(50, (54, 10, 94, 50))
        pending50, token50 = _bind_current_sanitized_evidence(
            provider, frame50, raw50, final50
        )
        provider._last_publication_guard_frame_id = frame50.frame_id
        provider._last_publication_guard_aligned_clean = aligned50.copy()
        provider._last_publication_guard_source = "online_sam2_guarded_primary"
        provider._last_output_mask_source = "publication_sanitized_target_core"
        provider._last_online_sam2_status = (
            "guarded_primary_trusted: trusted dt=0.167s speed=572.8px/s "
            "log_area_rate=1.08/s"
        )
        provider._last_online_sam2_exact_result = True
        original_alignment = provider._aligned_committed_clean_mask
        monkeypatch.setattr(
            provider,
            "_aligned_committed_clean_mask",
            lambda frame, *_args, **_kwargs: (
                (aligned50.copy(), "reviewed source50 aligned core")
                if int(frame.frame_id) == 50
                else original_alignment(frame, *_args, **_kwargs)
            ),
        )
        obj50 = provider._mask_only_publication_evidence(frame50, token50)
        provider._commit_final_publication_histories(
            frame50,
            _exact_token_mask_result(
                token50, message="reviewed source50 translated contraction"
            ),
            obj50,
        )
        owner50 = provider._visible_exact_partial_continuity
        assert owner50 is not None
        assert owner50.frame_id == 50
        assert owner50.hits == 3
        assert owner50.evidence_kind == "sanitized_target_core"
        assert "prior_retention=0.750" in (
            provider._last_sanitized_partial_chain_reason
        )
        assert "step translated shrink" in (
            provider._last_sanitized_partial_chain_reason
        )
        assert "anchor translated shrink" in (
            provider._last_sanitized_partial_chain_reason
        )
        assert provider._full_target_geometry_authority.frame_id == (
            authority_before.frame_id
        )
        assert provider._trusted_online_sam2_publication.frame_id == (
            trusted_before.frame_id
        )

        for frame_id, bbox in (
            (53, (66, 0, 106, 40)),
            (54, (70, 2, 110, 42)),
        ):
            frame = _moving_target_frame(frame_id, bbox)
            mask = _large_mask(bbox)
            provider._defer_full_target_geometry_commit(frame, mask)
            provider._last_output_mask_source = "online_sam2_guarded_primary"
            provider._last_online_sam2_exact_result = True
            prepared = provider._prepare_guarded_sam2_primary_mask(
                frame,
                _exact_token_mask_result(
                    provider._pending_full_target_geometry_commit.mask,
                    message=f"reviewed source{frame_id} exact mask",
                ),
            )
            assert prepared.valid
            assert provider._last_online_sam2_status.startswith(
                "guarded_primary_visible_partial_continuation:"
            )
            obj = provider._mask_only_publication_evidence(frame, prepared.mask)
            published, obj = provider._apply_recovery_publication_gate(
                frame, prepared, obj
            )
            assert published.valid and obj.valid
            assert provider._recovery_publication_hypothesis is None
            provider._commit_final_publication_histories(frame, published, obj)

        assert provider._visible_exact_partial_continuity.frame_id == 54
    finally:
        provider.stop()


def test_commissioned_sanitized_owner_keeps_source53_54_out_of_recovery(
    monkeypatch,
):
    """Reviewed fast motion continues after the source45 owner goes stale."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    provider._visible_exact_partial_continuity = None
    trusted = provider._trusted_online_sam2_publication
    trusted.frame_id = 45
    trusted.timestamp = 45.0 / 30.0

    seed_frame = _moving_target_frame(51, (50, 30, 90, 70))
    seed_frame.color_bgr[30:70, 86:90] = 45
    seed_raw = _large_mask((50, 30, 90, 70))
    seed_final = _large_mask((50, 30, 86, 70))
    pending, seed_token = _bind_current_sanitized_evidence(
        provider, seed_frame, seed_raw, seed_final
    )
    provider._last_publication_guard_frame_id = seed_frame.frame_id
    provider._last_publication_guard_aligned_clean = seed_raw.copy()
    provider._last_output_mask_source = "publication_sanitized_target_core"
    provider._last_online_sam2_status = (
        "guarded_primary_trusted: reviewed source51 handoff"
    )
    provider._last_online_sam2_exact_result = True
    try:
        assert provider._commit_visible_exact_partial_continuity(
            seed_frame,
            _exact_token_mask_result(seed_token, message="reviewed source51 core"),
            pending,
        )
        assert provider._commit_final_clean_publication(
            seed_frame, seed_token, rebase_motion_history=True
        )

        for frame_id, bbox in (
            (53, (60, 32, 100, 72)),
            (54, (68, 35, 108, 75)),
        ):
            frame = _moving_target_frame(frame_id, bbox)
            mask = _large_mask(bbox)
            provider._defer_full_target_geometry_commit(frame, mask)
            provider._last_output_mask_source = "online_sam2_guarded_primary"
            provider._last_online_sam2_exact_result = True
            prepared = provider._prepare_guarded_sam2_primary_mask(
                frame,
                _exact_token_mask_result(
                    provider._pending_full_target_geometry_commit.mask,
                    message=f"reviewed source{frame_id} exact mask",
                ),
            )
            assert prepared.valid
            assert provider._last_online_sam2_status.startswith(
                "guarded_primary_visible_partial_continuation:"
            )
            obj = provider._mask_only_publication_evidence(frame, prepared.mask)
            published, obj = provider._apply_recovery_publication_gate(
                frame, prepared, obj
            )
            assert published.valid and obj.valid
            assert provider.tracking_committed
            assert provider._recovery_publication_hypothesis is None
            provider._commit_final_publication_histories(frame, published, obj)

        assert provider._visible_exact_partial_continuity.frame_id == 54
        assert provider._last_final_clean_publication.frame_id == 54
    finally:
        provider.stop()


@pytest.mark.parametrize(
    "fault",
    ("precommission", "not_exact", "broken_trusted", "remote", "hand", "absent"),
)
def test_trusted_sanitizer_partial_seed_remains_fail_closed(monkeypatch, fault):
    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    provider._visible_exact_partial_continuity = None
    bbox = (130, 30, 170, 70) if fault == "remote" else (42, 30, 82, 70)
    frame = _moving_target_frame(2, bbox)
    raw = _large_mask(bbox)
    final = raw.copy()
    final[:, bbox[2] - 4 : bbox[2]] = 0
    if fault == "hand":
        frame.color_bgr[final > 0] = (200, 200, 200)
    elif fault == "absent":
        final[:] = 0
    if fault == "absent":
        provider._defer_full_target_geometry_commit(frame, raw)
        provider._mark_pending_full_target_geometry_ineligible(
            frame, "focused absent sanitized evidence"
        )
        pending = provider._pending_full_target_geometry_commit
        final_token = final
    else:
        pending, final_token = _bind_current_sanitized_evidence(
            provider, frame, raw, final
        )
    assert pending is not None
    provider._last_publication_guard_frame_id = frame.frame_id
    provider._last_publication_guard_aligned_clean = raw.copy()
    provider._last_output_mask_source = "publication_sanitized_target_core"
    provider._last_online_sam2_status = (
        "guarded_primary_trusted: negative commissioned handoff"
    )
    provider._last_online_sam2_exact_result = True
    if fault == "precommission":
        provider._guarded_v2_bootstrap_phase = "provisional"
    elif fault == "not_exact":
        provider._last_online_sam2_exact_result = False
    elif fault == "broken_trusted":
        provider._trusted_online_sam2_continuity_intact = False
    try:
        assert not provider._commit_visible_exact_partial_continuity(
            frame,
            _exact_token_mask_result(
                final_token, message=f"negative trusted handoff {fault}"
            ),
            pending,
        )
        assert provider._visible_exact_partial_continuity is None
    finally:
        provider.stop()


def test_fragmented_tail_core_uses_aligned_clean_not_raw_retention(
    monkeypatch,
):
    """A reviewed 0.54 tail advances only the partial continuity owner."""

    initial = _large_mask((40, 30, 80, 70))
    provider, tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    previous = _install_confirmed_raw_partial_owner(
        provider,
        frame_id=118,
        bbox=(70, 30, 110, 70),
    )
    authority_before = copy.deepcopy(provider._full_target_geometry_authority)
    trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)
    frame = _moving_target_frame(119, (70, 30, 110, 70))
    raw, aligned, final = _fragmented_tail_masks(below_raw_majority=False)
    pending, final_token = _bind_current_sanitized_evidence(
        provider, frame, raw, final
    )
    provider._last_publication_guard_frame_id = frame.frame_id
    provider._last_publication_guard_aligned_clean = aligned.copy()
    provider._last_output_mask_source = "publication_sanitized_target_core"
    provider._last_online_sam2_status = (
        "guarded_primary_visible_partial_continuation: reviewed boundary tail"
    )
    provider._last_online_sam2_exact_result = True
    try:
        mean, supported, count = tracker.target_appearance_support_stats(
            frame, final_token, threshold=0.42
        )
        assert (mean, supported, count) == pytest.approx((0.95, 1.0, 864))
        assert float(final_token.sum()) / float(pending.mask.sum()) == pytest.approx(
            0.54
        )
        assert float(final_token.sum()) / float(aligned.sum()) == pytest.approx(0.90)

        assert provider._commit_visible_exact_partial_continuity(
            frame,
            _exact_token_mask_result(
                final_token, message="fragmented reviewed boundary tail"
            ),
            pending,
        )
        owner = provider._visible_exact_partial_continuity
        assert owner is not previous
        assert owner.frame_id == 119
        assert owner.hits == 3
        assert owner.evidence_kind == "sanitized_target_core"
        assert owner.sanitized_handoff_eligible
        # This exception carries visible target motion only.  It must never
        # rewrite the full-object ruler or trusted exact-SAM history.
        assert provider._full_target_geometry_authority.frame_id == (
            authority_before.frame_id
        )
        assert provider._full_target_geometry_authority.area == pytest.approx(
            authority_before.area
        )
        assert provider._trusted_online_sam2_publication.frame_id == (
            trusted_before.frame_id
        )
        np.testing.assert_array_equal(
            provider._trusted_online_sam2_publication.mask,
            trusted_before.mask,
        )
    finally:
        provider.stop()


def test_fragmented_tail_one_outboard_rh56_pixel_fails_closed(monkeypatch):
    """Even one final pixel outside aligned clean target support is foreign."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    previous = _install_confirmed_raw_partial_owner(
        provider,
        frame_id=118,
        bbox=(70, 30, 110, 70),
    )
    frame = _moving_target_frame(119, (70, 30, 110, 70))
    raw, aligned, final = _fragmented_tail_masks(below_raw_majority=False)
    final = final.copy()
    final[50, 109] = 1  # In raw exact SAM, outside immutable clean target.
    pending, final_token = _bind_current_sanitized_evidence(
        provider, frame, raw, final
    )
    provider._last_publication_guard_frame_id = frame.frame_id
    provider._last_publication_guard_aligned_clean = aligned.copy()
    provider._last_output_mask_source = "publication_sanitized_target_core"
    provider._last_online_sam2_status = (
        "guarded_primary_visible_partial_continuation: outboard RH56 branch"
    )
    provider._last_online_sam2_exact_result = True
    try:
        assert not provider._commit_visible_exact_partial_continuity(
            frame,
            _exact_token_mask_result(
                final_token, message="one-pixel outboard RH56 branch"
            ),
            pending,
        )
        assert provider._visible_exact_partial_continuity is previous
    finally:
        provider.stop()


def test_sanitized_internal_highlight_components_commit_partial_owner(
    monkeypatch,
):
    """A 0.894 dominant split inside the target envelope is not a hand."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    previous = _install_confirmed_raw_partial_owner(provider)
    authority_before = copy.deepcopy(provider._full_target_geometry_authority)
    trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)
    frame = _moving_target_frame(50, (70, 30, 110, 70))
    raw = _large_mask((70, 30, 110, 70))
    final = _internal_highlight_split_mask()
    pending, final_token = _bind_current_sanitized_evidence(
        provider, frame, raw, final
    )
    provider._last_publication_guard_frame_id = frame.frame_id
    provider._last_publication_guard_aligned_clean = raw.copy()
    provider._last_output_mask_source = "publication_sanitized_target_core"
    provider._last_online_sam2_status = "guarded_primary_trusted: focused frame"
    provider._last_online_sam2_exact_result = True
    try:
        ordinary, ordinary_reason = (
            provider._sanitized_final_partial_subset_accepts(
                frame, final_token, pending.mask
            )
        )
        assert not ordinary
        assert ordinary_reason.startswith("sanitized core dominant=0.894<0.900")
        assert provider._commit_visible_exact_partial_continuity(
            frame,
            _exact_token_mask_result(
                final_token, message="internal highlight core"
            ),
            pending,
        )
        owner = provider._visible_exact_partial_continuity
        assert owner is not previous
        assert owner.frame_id == 50
        assert owner.hits == 3
        assert owner.evidence_kind == "sanitized_target_core"
        assert owner.sanitized_handoff_eligible
        assert provider._full_target_geometry_authority.frame_id == (
            authority_before.frame_id
        )
        assert provider._trusted_online_sam2_publication.frame_id == (
            trusted_before.frame_id
        )
    finally:
        provider.stop()


@pytest.mark.parametrize("inherits_sanitizer_lineage", (False, True))
def test_final_sanitized_handoff_reuses_only_confirmed_raw_lineage(
    monkeypatch,
    inherits_sanitizer_lineage,
):
    """A raw frame may bridge a confirmed sanitizer chain, not create one.

    The steady repair gate consumes ``sanitized_handoff_eligible`` even when
    the immediately preceding finally-published frame is raw exact.  The
    final-history transaction must consume the same structured lineage or it
    can expose a safe core while leaving its owner on a stale frame.  A plain
    raw owner without that inherited bit remains ineligible.
    """

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    previous = _install_confirmed_raw_partial_owner(provider)
    previous.hits = 34
    previous.sanitized_handoff_eligible = bool(inherits_sanitizer_lineage)
    previous.sanitized_anchor_bbox_xyxy = previous.core_bbox_xyxy.copy()
    previous.sanitized_anchor_center_xy = previous.core_center_xy.copy()
    previous.sanitized_anchor_area = float(previous.core_area)

    frame = _moving_target_frame(50, (70, 30, 110, 70))
    raw = _large_mask((70, 30, 110, 70))
    final = np.zeros_like(raw)
    final[45:65, 80:100] = 1
    pending, final_token = _bind_current_sanitized_evidence(
        provider, frame, raw, final
    )
    provider._last_publication_guard_frame_id = frame.frame_id
    provider._last_publication_guard_aligned_clean = raw.copy()
    provider._last_output_mask_source = "publication_sanitized_target_core"
    provider._last_online_sam2_status = (
        "guarded_primary_visible_partial_sanitized_continuation: "
        "confirmed raw-lineage bridge"
    )
    provider._last_online_sam2_exact_result = True

    aligned_calls = []
    chain_calls = []
    monkeypatch.setattr(
        provider,
        "_sanitized_final_partial_subset_accepts",
        lambda *_args, **_kwargs: (False, "focused low-retention core"),
    )
    monkeypatch.setattr(
        provider,
        "_compact_same_target_internal_ensemble_accepts",
        lambda *_args, **_kwargs: (False, "focused non-majority core"),
    )
    monkeypatch.setattr(
        provider,
        "_strict_majority_binary_subset_accepts",
        lambda *_args, **_kwargs: (False, "focused non-majority core"),
    )

    def accept_aligned(*_args, **_kwargs):
        aligned_calls.append(True)
        # Force the lower-retention lineage-chain fallback.  The ordinary
        # aligned-clean subset gate deliberately rejects source134-like cores
        # before the stricter motion/anchor chain proof admits them.
        return False, "focused aligned-clean retention miss"

    def accept_chain(*_args, **_kwargs):
        chain_calls.append(True)
        return True, "focused confirmed lineage chain"

    monkeypatch.setattr(
        provider, "_aligned_clean_visible_core_handoff_accepts", accept_aligned
    )
    monkeypatch.setattr(provider, "_sanitized_partial_chain_accepts", accept_chain)
    try:
        committed = provider._commit_visible_exact_partial_continuity(
            frame,
            _exact_token_mask_result(
                final_token, message="confirmed raw-lineage sanitizer bridge"
            ),
            pending,
        )
        assert committed is bool(inherits_sanitizer_lineage)
        if inherits_sanitizer_lineage:
            assert aligned_calls == [True]
            assert chain_calls == [True]
            owner = provider._visible_exact_partial_continuity
            assert owner is not previous
            assert owner.frame_id == 50
            assert owner.hits == 35
            assert owner.evidence_kind == "sanitized_target_core"
            assert owner.sanitized_handoff_eligible
            assert owner.sanitized_anchor_area == pytest.approx(previous.core_area)
        else:
            assert aligned_calls == []
            assert chain_calls == []
            assert provider._visible_exact_partial_continuity is previous
    finally:
        provider.stop()


def test_sanitized_outboard_hand_component_cannot_commit_partial_owner(
    monkeypatch,
):
    """The same topology is rejected when its second island leaves the target."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    previous = _install_confirmed_raw_partial_owner(provider)
    frame = _moving_target_frame(50, (70, 30, 110, 70))
    raw = _large_mask((70, 30, 110, 70))
    final = _internal_highlight_split_mask()
    pending, final_token = _bind_current_sanitized_evidence(
        provider, frame, raw, final
    )
    # The aligned immutable target is only the lower body.  The upper-right
    # component now represents an outboard finger branch, despite having the
    # same area/colour/topology as the accepted internal-highlight case.
    aligned = np.zeros_like(raw)
    aligned[40:70, 70:106] = 1
    provider._last_publication_guard_frame_id = frame.frame_id
    provider._last_publication_guard_aligned_clean = aligned
    provider._last_output_mask_source = "publication_sanitized_target_core"
    provider._last_online_sam2_status = "guarded_primary_trusted: focused frame"
    provider._last_online_sam2_exact_result = True
    try:
        assert not provider._commit_visible_exact_partial_continuity(
            frame,
            _exact_token_mask_result(final_token, message="outboard hand island"),
            pending,
        )
        assert provider._visible_exact_partial_continuity is previous
    finally:
        provider.stop()


@pytest.mark.parametrize(
    "output_source",
    ("publication_sanitized_target_core", "recovery_committed"),
)
def test_failed_sanitized_commit_preserves_owner_without_refresh(
    monkeypatch,
    output_source,
):
    """A safe publication cannot extend a rejected continuity owner clock."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    previous = _install_confirmed_raw_partial_owner(provider)
    previous_snapshot = copy.deepcopy(previous)
    frame = _moving_target_frame(50, (70, 30, 110, 70))
    raw = _large_mask((70, 30, 110, 70))
    final = _internal_highlight_split_mask()
    pending, final_token = _bind_current_sanitized_evidence(
        provider, frame, raw, final
    )
    aligned = np.zeros_like(raw)
    aligned[40:70, 70:106] = 1
    provider._last_publication_guard_frame_id = frame.frame_id
    provider._last_publication_guard_aligned_clean = aligned
    provider._last_publication_guard_source = "online_sam2_guarded_primary"
    provider._last_output_mask_source = output_source
    provider._last_online_sam2_status = "guarded_primary_trusted: focused frame"
    provider._last_online_sam2_exact_result = True
    provider._tracking_committed = True
    obj = provider._mask_only_publication_evidence(frame, final_token)
    try:
        provider._commit_final_publication_histories(
            frame,
            _exact_token_mask_result(
                final_token, message="safe but owner-ineligible core"
            ),
            obj,
        )
        owner = provider._visible_exact_partial_continuity
        assert owner is previous
        assert owner.frame_id == previous_snapshot.frame_id
        assert owner.timestamp == previous_snapshot.timestamp
        assert owner.hits == previous_snapshot.hits
        assert owner.velocity_samples == previous_snapshot.velocity_samples
        assert owner.mask_digest == previous_snapshot.mask_digest
        np.testing.assert_array_equal(owner.mask, previous_snapshot.mask)
        # No hidden freshness extension: the unchanged frame-48 owner expires
        # under the normal three-frame limit at source52.
        later = _moving_target_frame(52, (74, 30, 114, 70))
        later_geometry, reason = provider._online_mask_geometry(
            later, _large_mask((74, 30, 114, 70))
        )
        assert later_geometry is not None, reason
        accepted, detail = provider._visible_exact_partial_candidate_agrees(
            later, later_geometry
        )
        assert not accepted
        assert "visible-partial gap frame=4/3" in detail
    finally:
        provider.stop()


def test_formally_recovered_boundary_fine_lost_owner_uses_published_core(
    monkeypatch,
):
    """A 3/3-recovered fine owner may continue, but only by its core bytes."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    try:
        previous = _install_confirmed_raw_partial_owner(provider)
        previous.evidence_kind = "lost_reappearance_visible_core"
        provider._tracking_committed = True
        provider._guarded_v2_bootstrap_boundary_rearm_count = 1
        frame = _moving_target_frame(50, (70, 30, 110, 70))
        current = _large_mask((70, 30, 110, 70))
        geometry, reason = provider._online_mask_geometry(frame, current)
        assert geometry is not None, reason

        accepted, detail = provider._visible_exact_partial_candidate_agrees(
            frame,
            geometry,
            previous_geometry_kind="published_core",
        )
        assert accepted, detail

        rejected, detail = provider._visible_exact_partial_candidate_agrees(
            frame,
            geometry,
            previous_geometry_kind="motion_envelope",
        )
        assert not rejected
        assert "evidence kind is not trusted" in detail
    finally:
        provider.stop()


def test_recovery_compact_strict_majority_seeds_partial_only(monkeypatch):
    """A 0.553 compact recovery core seeds hits=2, never full/trusted."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    provider._visible_exact_partial_continuity = None
    authority_before = copy.deepcopy(provider._full_target_geometry_authority)
    trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)
    frame = _moving_target_frame(117, (70, 30, 110, 70))
    raw = _large_mask((70, 30, 110, 70))
    final = np.zeros_like(raw)
    final[40:70, 70:95] = 1  # 750px body.
    final[30:39, 95:110] = 1  # 135px highlight; 885/1600=.553.
    try:
        (
            recovered_frame,
            pending,
            published,
            published_obj,
            _aligned,
            final_token,
        ) = _complete_guarded_recovery_for_masks(
            monkeypatch,
            provider,
            raw=raw,
            aligned=final,
            final=final,
        )
        ordinary, ordinary_reason = provider._sanitized_final_partial_subset_accepts(
            recovered_frame, final_token, pending.mask
        )
        assert not ordinary
        assert ordinary_reason.startswith("sanitized core retention=0.553")
        provider._commit_final_publication_histories(
            recovered_frame, published, published_obj
        )
        owner = provider._visible_exact_partial_continuity
        assert owner.frame_id == recovered_frame.frame_id
        assert owner.hits == 2
        assert owner.evidence_kind == "recovery_sanitized_target_core"
        assert owner.sanitized_handoff_eligible
        assert provider._full_target_geometry_authority.frame_id == (
            authority_before.frame_id
        )
        assert provider._trusted_online_sam2_publication.frame_id == (
            trusted_before.frame_id
        )
    finally:
        provider.stop()


def test_recovery_local_visible_core_seeds_owner_without_stale_full_scale(
    monkeypatch,
):
    """A bound 3/3 160/400 core is not remeasured against stale 1600px."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    authority_before = copy.deepcopy(provider._full_target_geometry_authority)
    trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)
    try:
        local_reference = _large_mask((70, 40, 90, 60))  # 400px.
        _arm_exact_two_sam_then_same_frame_rgbd_recovery(
            provider, local_reference
        )
        local_proof = provider._recovery_publication_hypothesis
        assert local_proof is not None and local_proof.hits == 2
        # This focused helper seeds raw bytes by default.  The production
        # source135 hypothesis carried the same immutable bytes as an already
        # sanitized target core, which enables only the lower contraction
        # floor while leaving the ordinary growth/motion/depth bounds intact.
        local_proof.mask_evidence_kind = "sanitized_target_core"

        frame = _moving_target_frame(118, (70, 30, 110, 70))
        raw = _large_mask((70, 30, 110, 70))  # stale/full envelope: 1600px.
        core = _large_mask((76, 40, 84, 60))  # 160px = .40 local, .10 full.
        pending, core_token = _bind_current_sanitized_evidence(
            provider, frame, raw, core
        )
        monkeypatch.setattr(
            provider,
            "_aligned_committed_clean_mask",
            lambda *_args, **_kwargs: (
                local_reference.copy(),
                "focused recovery-local alignment",
            ),
        )
        accepted, detail = (
            provider._guarded_recovery_sanitized_visible_core_accepts(
                frame,
                _exact_token_mask_result(
                    core_token, message="focused recovery-local core"
                ),
                raw_exact_mask=pending.mask,
                local_reference=local_proof,
            )
        )
        assert accepted, detail
        assert pending.eligibility_kind == "recovery_local_visible_core"
        assert pending.recovery_local_core_pixels == 160
        assert pending.recovery_local_reference_pixels == 400
        np.testing.assert_array_equal(
            pending.recovery_local_aligned_reference_mask, local_reference
        )
        assert pending.recovery_local_aligned_reference_digest == (
            provider._publication_mask_digest(local_reference)
        )

        # The ordinary final guard still contains the current core, but its
        # old full silhouette is intentionally too large for the generic .15
        # scale check.  Only the immutable local proof above may supply that
        # denominator at the final history boundary.
        provider._last_publication_guard_frame_id = frame.frame_id
        provider._last_publication_guard_aligned_clean = raw.copy()
        provider._last_output_mask_source = "publication_sanitized_target_core"
        provider._last_online_sam2_status = (
            "guarded_primary_visible_partial_sanitized_continuation: "
            "focused later RGB-D"
        )
        provider._last_online_sam2_exact_result = True
        provider._current_recovery_evidence_outcome = "candidate"
        monkeypatch.setattr(
            provider,
            "_publication_candidate_agrees_with_committed_clean_mask",
            lambda *_args, **_kwargs: (True, "focused final guard"),
        )
        result = _exact_token_mask_result(
            core_token, message="later distinct RGB-D local core"
        )
        result.source = "online_sam2_video"
        obj = provider.extractor.extract(frame, core_token)
        published, published_obj = provider._apply_recovery_publication_gate(
            frame, result, obj
        )
        assert published.valid and published_obj.valid
        provider._commit_final_publication_histories(
            frame, published, published_obj
        )
        owner = provider._visible_exact_partial_continuity
        assert owner is not None
        assert owner.frame_id == frame.frame_id
        assert owner.hits == 2
        assert owner.evidence_kind == "recovery_sanitized_target_core"
        assert owner.sanitized_handoff_eligible
        assert owner.sanitized_anchor_phase == "fixed"
        assert owner.sanitized_anchor_area == pytest.approx(400.0)
        assert owner.sanitized_anchor_frame_id == local_proof.frame_id
        np.testing.assert_array_equal(
            owner.sanitized_anchor_mask, local_reference
        )
        assert provider._full_target_geometry_authority.frame_id == (
            authority_before.frame_id
        )
        assert provider._trusted_online_sam2_publication.frame_id == (
            trusted_before.frame_id
        )
    finally:
        provider.stop()


def test_fixed_anchor_broad_visible_core_stages_complete_candidate(monkeypatch):
    """The fixed anchor proves identity but never clips robot-facing pixels."""

    case = _stage_broad_visible_core_case(monkeypatch)
    try:
        assert case.staged is not None, case.stage_reason
        assert case.staged.valid
        assert case.staged.source == "publication_sanitized_target_core"
        np.testing.assert_array_equal(case.staged.mask, case.expected_broad)
        assert int(np.count_nonzero(case.staged.mask)) == 1144

        proof = case.pending.broad_visible_core_proof
        assert proof is not None
        aligned = np.asarray(proof.aligned_anchor_mask) > 0
        candidate = np.asarray(case.staged.mask) > 0
        intersection = np.logical_and(candidate, aligned)
        assert int(np.count_nonzero(aligned)) == 1600
        assert int(np.count_nonzero(intersection)) == 880
        assert int(np.count_nonzero(candidate)) > int(
            np.count_nonzero(intersection)
        )
        assert np.any(np.logical_and(candidate, np.logical_not(aligned)))
        assert case.pending.eligibility_kind == "sanitizer_broad_visible_core"
        assert not case.pending.raw_exact_eligible
        assert not case.pending.bootstrap_eligible

        accepted, reason = case.provider._broad_visible_core_proof_accepts(
            case.frame,
            case.staged.mask,
            case.pending.mask,
            case.pending,
            case.previous,
        )
        assert accepted, reason
    finally:
        case.provider.stop()


def test_deep_occlusion_broad_requires_two_frames_then_commits_partial_only(
    monkeypatch,
):
    """The first fragment is only a seed; the second fresh core may publish."""

    case = _stage_deep_occlusion_broad_pair_case(monkeypatch)
    try:
        assert case.staged is not None, case.stage_reason
        assert case.staged.valid
        proof = case.pending.broad_visible_core_proof
        assert proof is not None and proof.deep_occlusion_mode
        assert proof.deep_seed is not None
        assert int(np.count_nonzero(case.staged.mask)) == 720
        assert case.pending.eligibility_kind == "sanitizer_broad_visible_core"
        assert not case.pending.raw_exact_eligible
        assert not case.pending.bootstrap_eligible
        accepted, reason = case.provider._broad_visible_core_proof_accepts(
            case.frame,
            case.staged.mask,
            case.pending.mask,
            case.pending,
            case.previous,
        )
        assert accepted, reason

        case.provider._last_output_mask_source = (
            "publication_sanitized_target_core"
        )
        case.provider._last_online_sam2_status = (
            "guarded_primary_visible_partial_sanitized_continuation: focused deep"
        )
        case.provider._last_online_sam2_exact_result = True
        case.provider._last_publication_guard_frame_id = case.frame.frame_id
        case.provider._last_publication_guard_aligned_clean = (
            proof.aligned_anchor_mask.copy()
        )
        committed = case.provider._commit_visible_exact_partial_continuity(
            case.frame, case.staged, case.pending
        )
        assert committed
        owner = case.provider._visible_exact_partial_continuity
        assert owner.frame_id == case.frame.frame_id
        assert owner.deep_occlusion_broad_confirmed
        assert owner.evidence_kind == "sanitized_target_core"
        assert owner.sanitized_handoff_eligible
        np.testing.assert_array_equal(
            owner.broad_visible_anchor_mask, case.fixed_anchor
        )
        assert owner.broad_visible_anchor_area == pytest.approx(1600.0)
        assert case.provider._full_target_geometry_authority.frame_id == (
            case.authority_before.frame_id
        )
        assert case.provider._trusted_online_sam2_publication.frame_id == (
            case.trusted_before.frame_id
        )

        # A confirmed owner may continue only with another freshly computed
        # exact-frame candidate; it does not reuse the first transaction seed.
        frame_c = _large_frame(63)
        raw_c = _large_mask((59, 25, 109, 65))
        # De-occlusion grows back above the ordinary fixed-anchor floor.  A
        # confirmed deep owner must still consume the fresh pair rather than
        # being rejected by the standard-branch seed invariant.
        broad_c = _large_mask((64, 37, 106, 61))
        pure_strict_c = _large_mask((71, 39, 95, 57))
        strict_c = _large_mask((72, 40, 94, 56))
        probability_c = np.full(raw_c.shape, 0.05, dtype=np.float32)
        probability_c[broad_c > 0] = 0.30
        probability_c[pure_strict_c > 0] = 0.95
        _install_broad_visible_core_probability_map(
            case.tracker, probability_c
        )
        raw_c_evidence = case.provider._defer_full_target_geometry_commit(
            frame_c, raw_c
        )
        assert raw_c_evidence is not None
        raw_c_result = _exact_token_mask_result(
            raw_c_evidence.mask,
            message="confirmed deep current exact raw",
        )
        raw_c_result.source = "online_sam2_video"
        case.provider._tracking_committed = True
        case.provider._last_online_sam2_exact_result = True
        case.provider._online_sam2_initialized = True
        case.provider._online_sam2_seed_pending = False
        case.provider._online_sam2_failure_reported = False
        monkeypatch.setattr(
            case.provider,
            "_sanitize_publication_target_core",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("legacy strict sanitizer must not run first")
            ),
        )
        continued, continued_reason = (
            case.provider._guarded_visible_partial_sanitized_continuation(
                frame_c,
                raw_c_result,
                owner,
                raw_failure_kind="shape_only",
            )
        )
        assert continued is not None, continued_reason
        continued_proof = (
            case.provider._pending_full_target_geometry_commit.broad_visible_core_proof
        )
        assert continued_proof.deep_occlusion_mode
        assert continued_proof.deep_seed is None
        accepted, reason = case.provider._broad_visible_core_proof_accepts(
            frame_c,
            continued.mask,
            raw_c_evidence.mask,
            case.provider._pending_full_target_geometry_commit,
            owner,
        )
        assert accepted, reason
    finally:
        case.provider.stop()


def test_deep_occlusion_broad_reuses_sealed_stage_without_recomputing_cv_or_pcd(
    monkeypatch,
):
    """Later boundaries validate bytes/state but never rerun stage CV/PCD."""

    case = _stage_deep_occlusion_broad_pair_case(monkeypatch)
    try:
        assert case.staged is not None, case.stage_reason
        assert case.pending.deep_occlusion_broad_stage_seal is not None
        assert case.provider._deep_occlusion_broad_registry_seal(case.frame) is (
            case.pending.deep_occlusion_broad_stage_seal
        )
        with monkeypatch.context() as sealed:
            sealed.setattr(
                case.provider,
                "_deep_occlusion_broad_components",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    AssertionError("deep components were recomputed after stage")
                ),
            )
            sealed.setattr(
                case.provider,
                "_deep_occlusion_broad_pair",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    AssertionError("deep pair was recomputed after stage")
                ),
            )
            for _ in range(3):
                accepted, reason = case.provider._broad_visible_core_proof_accepts(
                    case.frame,
                    case.staged.mask,
                    case.pending.mask,
                    case.pending,
                    case.previous,
                )
                assert accepted, reason
    finally:
        case.provider.stop()


@pytest.mark.parametrize(
    ("variant", "reason_fragment"),
    (
        ("absent_ratios", "individual evidence rejected"),
        ("boundary", "touches effective boundary"),
    ),
)
def test_deep_occlusion_broad_rejects_stable_absence_and_boundary(
    monkeypatch, variant, reason_fragment
):
    """A stable hand mask cannot pass without independent target evidence."""

    case = _stage_deep_occlusion_broad_pair_case(monkeypatch, variant=variant)
    try:
        assert case.staged is None
        assert reason_fragment in case.stage_reason
        assert case.pending.broad_visible_core_proof is None
        assert case.provider._deep_occlusion_broad_seed is None
    finally:
        case.provider.stop()


@pytest.mark.parametrize(
    "fault",
    (
        "seed_bytes",
        "pair_bytes",
        "anchor_bytes",
        "owner_digest",
        "depth_bits",
        "config",
        "seal_removed",
        "registry_removed",
        "frame_depth",
        "pending_kind",
        "proof_removed",
    ),
)
def test_deep_occlusion_broad_final_revalidation_rejects_tampering(
    monkeypatch, fault
):
    """All first/current-frame and fixed-anchor facts remain fail-closed."""

    case = _stage_deep_occlusion_broad_pair_case(monkeypatch)
    try:
        assert case.staged is not None, case.stage_reason
        proof = case.pending.broad_visible_core_proof
        assert proof is not None and proof.deep_seed is not None
        if fault == "seed_bytes":
            replacement = np.asarray(proof.deep_seed.candidate_mask).copy()
            replacement[39, 63] ^= 1
            proof.deep_seed.candidate_mask = replacement
        elif fault == "pair_bytes":
            replacement = np.asarray(proof.deep_pair_aligned_seed_mask).copy()
            replacement[39, 65] ^= 1
            proof.deep_pair_aligned_seed_mask = replacement
        elif fault == "anchor_bytes":
            replacement = np.asarray(proof.anchor_mask).copy()
            replacement[30, 60] ^= 1
            proof.anchor_mask = replacement
        elif fault == "owner_digest":
            case.previous.mask_digest = "0" * 64
        elif fault == "depth_bits":
            proof.deep_pair_depth_step_bits ^= 1
        elif fault == "config":
            proof.deep_config_fingerprint = "0" * 64
        elif fault == "seal_removed":
            case.pending.deep_occlusion_broad_stage_seal = None
        elif fault == "registry_removed":
            case.provider._deep_occlusion_broad_stage_registry = {}
        elif fault == "frame_depth":
            case.frame.depth_raw[40, 70] += 1
        elif fault == "pending_kind":
            case.pending.eligibility_kind = "known_partial"
        elif fault == "proof_removed":
            case.pending.broad_visible_core_proof = None
        accepted, _reason = case.provider._broad_visible_core_proof_accepts(
            case.frame,
            case.staged.mask,
            case.pending.mask,
            case.pending,
            case.previous,
        )
        assert not accepted
    finally:
        case.provider.stop()


def test_deep_occlusion_broad_rejects_nonprojectable_current_core(monkeypatch):
    """A 2-D fragment never publishes without fresh usable depth/points."""

    initial = _large_mask((60, 30, 100, 70))
    provider, tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    try:
        frame = _large_frame(70)
        raw = _large_mask((55, 25, 105, 65))
        broad = _large_mask((62, 38, 98, 58))
        strict = _large_mask((67, 39, 91, 57))
        frame.depth_raw[broad > 0] = 0
        probability = np.full(raw.shape, 0.05, dtype=np.float32)
        probability[broad > 0] = 0.30
        probability[strict > 0] = 0.95
        _install_broad_visible_core_probability_map(tracker, probability)
        components, reason = provider._deep_occlusion_broad_components(
            frame, raw, initial
        )
        assert components is None
        assert "RGB-D quality rejected" in reason or "geometry rejected" in reason
    finally:
        provider.stop()


@pytest.mark.parametrize(
    ("variant", "reason_fragment"),
    (
        ("low_overlap", "fixed-anchor proof rejected"),
        ("boundary_anchor", "anchor touches effective boundary"),
        ("strict_empty", "strict/raw subset contract failed"),
    ),
)
def test_fixed_anchor_broad_visible_core_stage_rejects_unproven_candidate(
    monkeypatch,
    variant,
    reason_fragment,
):
    """Absent-scale, boundary, or empty strict evidence remains fail-closed."""

    case = _stage_broad_visible_core_case(monkeypatch, variant=variant)
    try:
        assert case.staged is None
        assert reason_fragment in case.stage_reason
        assert case.pending.broad_visible_core_proof is None
        # A failed optional broad candidate must not revoke the exact raw
        # token.  The caller may still try the ordinary strict-core path.
        assert case.pending.raw_exact_eligible
        assert case.pending.bootstrap_eligible
    finally:
        case.provider.stop()


@pytest.mark.parametrize(
    ("fault", "reason_fragment"),
    (
        ("candidate_digest", "raw/final binding changed"),
        ("raw_digest", "raw/final binding changed"),
        ("raw_identity", "raw/final binding changed"),
        ("final_identity", "raw/final binding changed"),
        ("strict_digest", "broad/strict appearance bytes changed"),
        ("owner_bytes", "owner bytes changed"),
        ("anchor_digest", "anchor changed"),
        ("anchor_bytes", "anchor changed"),
        ("aligned_digest", "alignment proof changed"),
        ("aligned_bytes", "alignment proof changed"),
        ("pending_timestamp", "frame token changed"),
        ("proof_timestamp", "frame token changed"),
    ),
)
def test_fixed_anchor_broad_visible_core_final_validator_rejects_tampering(
    monkeypatch,
    fault,
    reason_fragment,
):
    """Every staged raw/strict/anchor fact is rechecked at final history."""

    case = _stage_broad_visible_core_case(monkeypatch)
    try:
        assert case.staged is not None, case.stage_reason
        proof = case.pending.broad_visible_core_proof
        assert proof is not None
        raw_mask = case.pending.mask
        final_mask = case.staged.mask
        frame = case.frame

        if fault == "candidate_digest":
            proof.candidate_mask_digest = "0" * 64
        elif fault == "raw_digest":
            case.pending.mask_digest = "0" * 64
        elif fault == "raw_identity":
            raw_mask = case.pending.mask.copy()
        elif fault == "final_identity":
            final_mask = case.staged.mask.copy()
        elif fault == "strict_digest":
            proof.strict_mask_digest = "0" * 64
        elif fault == "owner_bytes":
            changed = case.previous.mask.copy()
            changed[30, 60] = 0
            case.previous.mask = changed
        elif fault == "anchor_digest":
            proof.anchor_mask_digest = "0" * 64
        elif fault == "anchor_bytes":
            changed = proof.anchor_mask.copy()
            changed[30, 60] = 0
            proof.anchor_mask = changed
        elif fault == "aligned_digest":
            proof.aligned_anchor_mask_digest = "0" * 64
        elif fault == "aligned_bytes":
            changed = proof.aligned_anchor_mask.copy()
            ys, xs = np.nonzero(changed)
            changed[int(ys[0]), int(xs[0])] = 0
            proof.aligned_anchor_mask = changed
        elif fault == "pending_timestamp":
            case.pending.timestamp = float(
                np.nextafter(case.pending.timestamp, np.inf)
            )
        elif fault == "proof_timestamp":
            proof.timestamp_bits ^= 1
        else:  # pragma: no cover - parametrization is exhaustive.
            raise AssertionError(f"unknown broad-core fault {fault!r}")

        accepted, reason = case.provider._broad_visible_core_proof_accepts(
            frame,
            final_mask,
            raw_mask,
            case.pending,
            case.previous,
        )
        assert not accepted
        assert reason_fragment in reason
        assert case.provider._visible_exact_partial_continuity is case.previous
        assert case.previous.frame_id == 51
        assert case.previous.hits == 3
    finally:
        case.provider.stop()


def test_fixed_anchor_broad_visible_core_final_commit_keeps_authorities_and_anchor(
    monkeypatch,
):
    """A valid broad frame advances only partial motion with a fixed anchor."""

    case = _stage_broad_visible_core_case(monkeypatch)
    provider = case.provider
    try:
        assert case.staged is not None, case.stage_reason
        proof = case.pending.broad_visible_core_proof
        assert proof is not None
        full_before = copy.deepcopy(provider._full_target_geometry_authority)
        trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)
        previous_hits = int(case.previous.hits)
        fixed_anchor = np.asarray(proof.anchor_mask).copy()
        fixed_anchor_digest = str(proof.anchor_mask_digest)
        fixed_anchor_area = int(proof.anchor_pixels)
        fixed_anchor_frame = int(proof.anchor_frame_id)

        provider._last_output_mask_source = "publication_sanitized_target_core"
        provider._last_publication_guard_source = (
            "publication_sanitized_target_core"
        )
        provider._last_publication_guard_status = (
            "accepted: fixed-anchor broad visible core test"
        )
        provider._last_online_sam2_status = (
            "guarded_primary_visible_partial_sanitized_continuation: "
            "fixed-anchor broad visible core test"
        )
        provider._last_online_sam2_exact_result = True
        provider._tracking_committed = True
        obj = provider.extractor.extract(case.frame, case.staged.mask)
        assert obj.valid

        provider._commit_final_publication_histories(
            case.frame,
            case.staged,
            obj,
        )

        owner = provider._visible_exact_partial_continuity
        assert owner is not None and owner is not case.previous
        assert owner.frame_id == case.frame.frame_id
        assert owner.hits == previous_hits + 1
        assert owner.evidence_kind == "sanitized_target_core"
        np.testing.assert_array_equal(owner.mask, case.expected_broad)
        np.testing.assert_array_equal(
            owner.broad_visible_anchor_mask,
            fixed_anchor,
        )
        assert owner.broad_visible_anchor_digest == fixed_anchor_digest
        assert owner.broad_visible_anchor_area == pytest.approx(
            float(fixed_anchor_area)
        )
        assert owner.broad_visible_anchor_frame_id == fixed_anchor_frame

        full_after = provider._full_target_geometry_authority
        assert full_after.frame_id == full_before.frame_id
        assert full_after.timestamp == full_before.timestamp
        assert full_after.area == pytest.approx(full_before.area)
        assert full_after.bbox_area == pytest.approx(full_before.bbox_area)

        trusted_after = provider._trusted_online_sam2_publication
        assert trusted_after.frame_id == trusted_before.frame_id
        assert trusted_after.timestamp == trusted_before.timestamp
        assert trusted_after.area == pytest.approx(trusted_before.area)
        np.testing.assert_array_equal(trusted_after.mask, trusted_before.mask)
    finally:
        provider.stop()


def test_standard_broad_frame_seeds_next_deep_occlusion_pair(monkeypatch):
    """An accepted source222-like broad frame may confirm source224."""

    case = _stage_broad_visible_core_case(monkeypatch, variant="deep_seedable")
    provider = case.provider
    try:
        assert case.staged is not None, case.stage_reason
        seed = provider._deep_occlusion_broad_seed
        assert seed is not None
        assert seed.frame_id == case.frame.frame_id
        # The first broad frame still uses the ordinary proof and publication
        # path.  Its optional deep seed must be rebound to the owner that
        # actually crosses the final boundary, not the pre-final strict core.
        proof = case.pending.broad_visible_core_proof
        assert proof is not None and not proof.deep_occlusion_mode
        provider._last_output_mask_source = "publication_sanitized_target_core"
        provider._last_publication_guard_source = (
            "publication_sanitized_target_core"
        )
        provider._last_publication_guard_status = "accepted: standard broad seed"
        provider._last_online_sam2_status = (
            "guarded_primary_visible_partial_sanitized_continuation: "
            "standard broad seed"
        )
        provider._last_online_sam2_exact_result = True
        provider._tracking_committed = True
        obj = provider.extractor.extract(case.frame, case.staged.mask)
        assert obj.valid
        provider._commit_final_publication_histories(
            case.frame, case.staged, obj
        )
        owner = provider._visible_exact_partial_continuity
        rebound = provider._deep_occlusion_broad_seed
        assert owner is not None and rebound is not None
        assert rebound.expected_owner_frame_id == owner.frame_id
        assert rebound.expected_owner_hits == owner.hits
        assert rebound.expected_owner_mask_digest == owner.mask_digest

        # The next exact frame is below the ordinary 0.50 fixed-anchor scale
        # floor but remains above every independent deep-occlusion floor.  It
        # must consume the previous accepted broad seed rather than arm too
        # late and lose the owner.
        frame_b = _large_frame(53)
        raw_b = _large_mask((55, 25, 105, 65))  # 2000px.
        broad_b = _large_mask((62, 38, 98, 58))  # 720px/.45 anchor.
        pure_strict_b = _large_mask((67, 39, 91, 57))  # 432px.
        final_strict_b = _large_mask((68, 40, 90, 56))
        probability_b = np.full(raw_b.shape, 0.05, dtype=np.float32)
        probability_b[broad_b > 0] = 0.30
        probability_b[pure_strict_b > 0] = 0.95
        _install_broad_visible_core_probability_map(
            case.tracker, probability_b
        )
        raw_b_evidence = provider._defer_full_target_geometry_commit(
            frame_b, raw_b
        )
        strict_b_evidence, strict_reason = (
            provider._register_frame_mask_evidence(
                frame_b,
                final_strict_b,
                evidence_kind="sanitized_target_core",
                authority_eligible=False,
            )
        )
        assert strict_b_evidence is not None, strict_reason
        staged, stage_reason = provider._stage_broad_visible_core(
            frame_b, raw_b_evidence, strict_b_evidence, owner
        )
        assert staged is not None, stage_reason
        second_proof = (
            provider._pending_full_target_geometry_commit.broad_visible_core_proof
        )
        assert second_proof is not None and second_proof.deep_occlusion_mode
        assert second_proof.deep_seed is rebound
        assert "deep broad pair accepted" in stage_reason

        # The production replay source222->224 transition retains 242/426=
        # .568 of the
        # first visible core.  This is distinct from complete absence because
        # both frames independently satisfy all four appearance/fixed-anchor
        # floors before the pair metric is evaluated.
        assert provider._deep_occlusion_broad_limits()[5] == pytest.approx(0.56)
    finally:
        provider.stop()


def test_lost_reappearance_visible_core_requires_two_exact_frames(monkeypatch):
    """Source312 arms only a seed; source314 stages one partial-only mask."""

    case = _stage_lost_reappearance_visible_core_case(monkeypatch)
    try:
        seed = case.seed_snapshot
        assert seed.frame_id == case.seed_frame.frame_id
        assert seed.candidate_pixels == int(np.count_nonzero(case.seed_broad))
        assert seed.strict_pixels == int(np.count_nonzero(case.seed_strict))
        assert case.recovery_hit1 is not None
        assert case.recovery_hit1.hits == 1
        assert case.recovery_hit1.source == "guarded_sam2_primary"

        assert case.staged is not None, case.stage_reason
        assert case.staged.valid
        assert case.staged.source == "lost_reappearance_visible_core"
        np.testing.assert_array_equal(case.staged.mask, case.current_broad)
        assert case.pending.eligibility_kind == "lost_reappearance_visible_core"
        assert not case.pending.raw_exact_eligible
        assert not case.pending.bootstrap_eligible
        proof = case.pending.lost_reappearance_visible_core_proof
        assert proof is not None
        assert proof.seed.frame_id == case.seed_frame.frame_id

        # This fixture deliberately follows the real source312->314 shape:
        # candidate bboxes alone have IoU<.60, while the containing raw exact
        # silhouettes have IoU>.60.  The pair gate uses raw geometry and the
        # p=.20 cores independently prove containment/identity.
        assert case.provider._bbox_iou(
            proof.seed.candidate_bbox_xyxy,
            case.current_geometry.bbox_xyxy,
        ) < 0.60
        assert case.provider._bbox_iou(
            proof.seed.raw_bbox_xyxy,
            case.current_geometry.bbox_xyxy,
        ) >= 0.60
        accepted, reason = (
            case.provider._lost_reappearance_visible_core_proof_accepts(
                case.current_frame,
                case.staged.mask,
                case.pending.mask,
                case.pending,
                require_rgbd_attachment=False,
            )
        )
        assert accepted, reason
        # Two exact images have seeded same-frame RGB-D probation, but no
        # robot-facing owner or formally committed recovery exists yet.
        assert case.provider._visible_exact_partial_continuity is None
        assert not case.provider.tracking_committed
        recovery = case.provider._recovery_publication_hypothesis
        assert recovery is not None
        assert recovery.hits == 2
        assert recovery.evidence_kind == "two_exact_sam_seed"
        assert case.provider._pending_recovery_publication_commit is None
    finally:
        case.provider.stop()


@pytest.mark.parametrize(
    ("variant", "reason_fragment"),
    (
        ("low_broad_raw", "current core rejected"),
        ("low_strict_raw", "current core rejected"),
        ("hand_union", "fixed-anchor overlap rejected"),
        ("appearance_absent", "appearance evidence is unavailable"),
        ("appearance_empty", "current core rejected"),
        ("low_anchor_coverage", "fixed-anchor overlap rejected"),
        ("low_candidate_coverage", "fixed-anchor overlap rejected"),
        ("boundary", "current core rejected"),
        ("pair_bbox", "pair rejected"),
        ("pair_growth", "pair rejected"),
    ),
)
def test_lost_reappearance_visible_core_rejects_unproven_second_frame(
    monkeypatch,
    variant,
    reason_fragment,
):
    """Weak source305/306, hand union, absence, edge, and bad pair stay shut."""

    case = _stage_lost_reappearance_visible_core_case(
        monkeypatch,
        variant=variant,
    )
    try:
        assert case.staged is None
        assert reason_fragment.lower() in case.stage_reason.lower()
        raw_pixels = int(np.count_nonzero(case.current_raw))
        if variant == "low_broad_raw":
            assert np.count_nonzero(case.current_broad) / raw_pixels < 0.50
            assert np.count_nonzero(case.current_strict) / raw_pixels < 0.50
        elif variant == "low_strict_raw":
            assert np.count_nonzero(case.current_broad) / raw_pixels >= 0.50
            assert np.count_nonzero(case.current_strict) / raw_pixels < 0.50
        elif variant in ("low_anchor_coverage", "low_candidate_coverage"):
            anchor = case.provider._trusted_online_sam2_publication.mask
            radius = case.provider._lost_reappearance_visible_core_limits()[6]
            aligned, _shift, detail = (
                case.provider._best_broad_visible_anchor_alignment(
                    anchor,
                    case.current_broad,
                    case.current_strict,
                    radius_px=radius,
                )
            )
            assert aligned is not None, detail
            overlap = int(
                np.count_nonzero(
                    np.logical_and(aligned > 0, case.current_broad > 0)
                )
            )
            candidate_coverage = overlap / np.count_nonzero(case.current_broad)
            anchor_coverage = overlap / np.count_nonzero(aligned)
            if variant == "low_anchor_coverage":
                assert candidate_coverage >= 0.90
                assert anchor_coverage < 0.20
            else:
                assert candidate_coverage < 0.90
                assert anchor_coverage >= 0.20
        elif variant == "boundary":
            assert not case.provider._mask_is_effective_interior(
                case.current_broad
            )
        assert case.pending.lost_reappearance_visible_core_proof is None
        assert case.provider._visible_exact_partial_continuity is None
        assert not case.provider.tracking_committed
        assert case.provider._pending_recovery_publication_commit is None
    finally:
        case.provider.stop()


@pytest.mark.parametrize(
    ("fault", "reason_fragment"),
    (
        ("broad_digest", "raw/final evidence changed"),
        ("raw_digest", "raw/final evidence changed"),
        ("raw_identity", "raw/final evidence changed"),
        ("broad_identity", "raw/final evidence changed"),
        ("strict_digest", "current mask/anchor bytes changed"),
        ("aligned_anchor_digest", "current mask/anchor bytes changed"),
        ("full_anchor_digest", "current mask/anchor bytes changed"),
        ("pending_timestamp", "current frame token changed"),
        ("proof_timestamp", "current frame token changed"),
        ("current_generation", "current frame token changed"),
        ("seed_timestamp", "first-frame seed changed"),
        ("seed_generation", "first-frame seed changed"),
        ("seed_raw_digest", "first-frame seed changed"),
        ("seed_broad_digest", "first-frame seed changed"),
        ("seed_strict_digest", "first-frame seed changed"),
        ("seed_anchor_digest", "first-frame seed changed"),
        ("anchor_timestamp", "current proof changed"),
        ("anchor_generation", "current proof changed"),
        ("anchor_owner", "current proof changed"),
        ("recovery_owner", "recovery/RGB-D attachment changed"),
        ("recovery_digest", "recovery/RGB-D attachment changed"),
    ),
)
def test_lost_reappearance_visible_core_final_validator_rejects_tampering(
    monkeypatch,
    fault,
    reason_fragment,
):
    """Every raw/core/anchor/frame/generation owner is revalidated."""

    case = _stage_lost_reappearance_visible_core_case(monkeypatch)
    provider = case.provider
    try:
        assert case.staged is not None, case.stage_reason
        proof = case.pending.lost_reappearance_visible_core_proof
        assert proof is not None
        raw_mask = case.pending.mask
        final_mask = case.staged.mask

        if fault == "broad_digest":
            proof.candidate_mask_digest = "0" * 64
        elif fault == "raw_digest":
            case.pending.mask_digest = "0" * 64
        elif fault == "raw_identity":
            raw_mask = raw_mask.copy()
        elif fault == "broad_identity":
            final_mask = final_mask.copy()
        elif fault == "strict_digest":
            proof.strict_mask_digest = "0" * 64
        elif fault == "aligned_anchor_digest":
            proof.aligned_anchor_mask_digest = "0" * 64
        elif fault == "full_anchor_digest":
            proof.anchor_mask_digest = "0" * 64
        elif fault == "pending_timestamp":
            case.pending.timestamp = float(
                np.nextafter(case.pending.timestamp, np.inf)
            )
        elif fault == "proof_timestamp":
            proof.timestamp_bits ^= 1
        elif fault == "current_generation":
            provider._target_generation += 1
        elif fault == "seed_timestamp":
            provider._lost_reappearance_visible_core_seed.timestamp_bits ^= 1
        elif fault == "seed_generation":
            provider._lost_reappearance_visible_core_seed.target_generation += 1
        elif fault == "seed_raw_digest":
            provider._lost_reappearance_visible_core_seed.raw_mask_digest = "0" * 64
        elif fault == "seed_broad_digest":
            provider._lost_reappearance_visible_core_seed.candidate_mask_digest = (
                "0" * 64
            )
        elif fault == "seed_strict_digest":
            provider._lost_reappearance_visible_core_seed.strict_mask_digest = (
                "0" * 64
            )
        elif fault == "seed_anchor_digest":
            provider._lost_reappearance_visible_core_seed.anchor_mask_digest = (
                "0" * 64
            )
        elif fault == "anchor_timestamp":
            trusted = provider._trusted_online_sam2_publication
            trusted.timestamp = float(np.nextafter(trusted.timestamp, np.inf))
        elif fault == "anchor_generation":
            trusted = provider._trusted_online_sam2_publication
            trusted.mask_evidence = replace(
                trusted.mask_evidence,
                target_generation=trusted.mask_evidence.target_generation + 1,
            )
        elif fault == "anchor_owner":
            provider._full_target_geometry_authority.frame_id += 1
        elif fault == "recovery_owner":
            provider._recovery_publication_hypothesis.hits = 1
        elif fault == "recovery_digest":
            provider._recovery_publication_hypothesis.mask_digest = "0" * 64
        else:  # pragma: no cover - parametrization is exhaustive.
            raise AssertionError(f"unknown LOST visible-core fault {fault!r}")

        accepted, reason = provider._lost_reappearance_visible_core_proof_accepts(
            case.current_frame,
            final_mask,
            raw_mask,
            case.pending,
            require_rgbd_attachment=False,
        )
        assert not accepted
        assert reason_fragment in reason
        assert provider._visible_exact_partial_continuity is None
        assert not provider.tracking_committed
        assert provider._pending_recovery_publication_commit is None
    finally:
        provider.stop()


@pytest.mark.parametrize(
    ("fault", "reason_fragment"),
    (
        ("trusted_equal_copy", "current proof changed"),
        ("full_bbox_area", "current proof changed"),
    ),
)
def test_lost_reappearance_visible_core_rejects_full_anchor_owner_tampering(
    monkeypatch,
    fault,
    reason_fragment,
):
    """Equal bytes are not owner identity; full bbox scale stays immutable."""

    case = _stage_lost_reappearance_visible_core_case(monkeypatch)
    provider = case.provider
    try:
        assert case.staged is not None, case.stage_reason
        if fault == "trusted_equal_copy":
            trusted = provider._trusted_online_sam2_publication
            trusted.mask = trusted.mask.copy()
        elif fault == "full_bbox_area":
            provider._full_target_geometry_authority.bbox_area += 1.0
        else:  # pragma: no cover - parametrization is exhaustive.
            raise AssertionError(f"unknown full-anchor fault {fault!r}")

        accepted, reason = provider._lost_reappearance_visible_core_proof_accepts(
            case.current_frame,
            case.staged.mask,
            case.pending.mask,
            case.pending,
            require_rgbd_attachment=False,
        )
        assert not accepted
        assert reason_fragment in reason
        assert provider._visible_exact_partial_continuity is None
        assert not provider.tracking_committed
    finally:
        provider.stop()


@pytest.mark.parametrize(
    ("variant", "quality_name"),
    (
        ("low_valid_depth", "valid_depth"),
        ("high_depth_spread", "depth_spread"),
    ),
)
def test_lost_reappearance_visible_core_attach_rejects_bad_rgbd_quality(
    monkeypatch,
    variant,
    quality_name,
):
    """Special same-frame attachment enforces ordinary recovery RGB-D gates."""

    case = _stage_lost_reappearance_visible_core_case(
        monkeypatch,
        variant=variant,
    )
    provider = case.provider
    try:
        assert case.staged is not None, case.stage_reason
        token = provider._lookup_frame_mask_evidence(
            case.current_frame, case.staged.mask
        )
        assert token is not None
        if variant == "low_valid_depth":
            threshold = float(
                provider.cfg["tracker"][
                    "recovery_publish_min_valid_depth_ratio"
                ]
            )
            assert token.geometry.valid_depth_ratio < threshold
        else:
            threshold = float(
                provider.cfg["tracker"][
                    "recovery_publish_max_depth_spread_m"
                ]
            )
            assert token.geometry.depth_spread > threshold

        recovery_before = provider._recovery_publication_hypothesis
        assert recovery_before is not None
        assert recovery_before.evidence_kind == "two_exact_sam_seed"
        obj = provider.extractor.extract(case.current_frame, case.staged.mask)
        assert obj.valid
        attached, reason = provider._attach_lost_reappearance_visible_core_rgbd(
            case.current_frame,
            case.staged,
            obj,
            case.pending,
        )
        assert not attached
        assert "RGB-D quality rejected" in reason
        assert quality_name in reason
        assert provider._recovery_publication_hypothesis is recovery_before
        assert recovery_before.evidence_kind == "two_exact_sam_seed"
        assert provider._visible_exact_partial_continuity is None
        assert not provider.tracking_committed
    finally:
        provider.stop()


def test_lost_reappearance_visible_core_commit_is_partial_only(monkeypatch):
    """The p=.20 core advances no full/trusted/bootstrap/tracker authority."""

    case = _stage_lost_reappearance_visible_core_case(monkeypatch)
    provider = case.provider
    try:
        assert case.staged is not None, case.stage_reason
        proof = case.pending.lost_reappearance_visible_core_proof
        assert proof is not None
        full_before = copy.deepcopy(provider._full_target_geometry_authority)
        trusted_owner_before = provider._trusted_online_sam2_publication
        trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)
        final_before = copy.deepcopy(provider._last_final_clean_publication)
        bootstrap_phase_before = provider._guarded_v2_bootstrap_phase
        bootstrap_hypothesis_before = copy.deepcopy(
            provider._guarded_v2_bootstrap_hypothesis
        )
        bootstrap_count_before = provider._guarded_v2_bootstrap_commission_count
        tracker_before = case.tracker.snapshot_state()

        obj = provider.extractor.extract(case.current_frame, case.staged.mask)
        assert obj.valid
        attached, reason = provider._attach_lost_reappearance_visible_core_rgbd(
            case.current_frame,
            case.staged,
            obj,
            case.pending,
        )
        assert attached, reason
        accepted, reason = provider._lost_reappearance_visible_core_proof_accepts(
            case.current_frame,
            case.staged.mask,
            case.pending.mask,
            case.pending,
            require_rgbd_attachment=True,
        )
        assert accepted, reason

        provider._commit_final_publication_histories(
            case.current_frame,
            case.staged,
            obj,
        )

        owner = provider._visible_exact_partial_continuity
        assert owner is not None
        assert owner.frame_id == case.current_frame.frame_id
        assert owner.hits == 2
        assert owner.evidence_kind == "lost_reappearance_visible_core"
        assert not owner.sanitized_handoff_eligible
        np.testing.assert_array_equal(owner.mask, case.current_broad)
        np.testing.assert_array_equal(
            owner.broad_visible_anchor_mask, proof.anchor_mask
        )
        assert owner.broad_visible_anchor_digest == proof.anchor_mask_digest

        # This transaction publishes one partial observation while staying
        # LOST.  The normal 3-evidence recovery token remains attached for a
        # later distinct RGB-D frame; it is not silently promoted here.
        assert not provider.tracking_committed
        recovery = provider._recovery_publication_hypothesis
        assert recovery is not None
        assert recovery.hits == 2
        assert recovery.evidence_kind == "two_exact_sam_seed_with_rgbd"
        assert provider._pending_recovery_publication_commit is None
        assert provider._pending_full_target_geometry_commit is None
        assert provider._lost_reappearance_visible_core_seed is None

        full_after = provider._full_target_geometry_authority
        assert full_after.frame_id == full_before.frame_id
        assert full_after.timestamp == full_before.timestamp
        assert full_after.area == pytest.approx(full_before.area)
        trusted_after = provider._trusted_online_sam2_publication
        assert trusted_after is trusted_owner_before
        assert trusted_after.frame_id == trusted_before.frame_id
        assert trusted_after.timestamp == trusted_before.timestamp
        assert trusted_after.mask_evidence.mask_digest == (
            trusted_before.mask_evidence.mask_digest
        )
        np.testing.assert_array_equal(trusted_after.mask, trusted_before.mask)
        assert provider._last_final_clean_publication.frame_id == (
            final_before.frame_id
        )
        assert provider._last_final_clean_publication.timestamp == (
            final_before.timestamp
        )
        assert provider._guarded_v2_bootstrap_phase == bootstrap_phase_before
        assert provider._guarded_v2_bootstrap_hypothesis == (
            bootstrap_hypothesis_before
        )
        assert provider._guarded_v2_bootstrap_commission_count == (
            bootstrap_count_before
        )
        tracker_after = case.tracker.snapshot_state()
        assert tracker_after.token == tracker_before.token
        np.testing.assert_array_equal(
            tracker_after.state.last_mask, tracker_before.state.last_mask
        )
        np.testing.assert_array_equal(
            tracker_after.state.bbox_xyxy, tracker_before.state.bbox_xyxy
        )
        assert tracker_after.state.lost_count == tracker_before.state.lost_count
        assert tracker_after.state.valid == tracker_before.state.valid
    finally:
        provider.stop()


def test_boundary_fine_lost_commit_rearms_provisional_bootstrap(monkeypatch):
    """A sealed fine-core pair survives into the later RGB-D recovery tick."""

    case = _stage_lost_reappearance_visible_core_case(monkeypatch)
    provider = case.provider
    try:
        assert case.staged is not None, case.stage_reason
        proof = case.pending.lost_reappearance_visible_core_proof
        live_seed = provider._lost_reappearance_visible_core_seed
        assert proof is not None and live_seed is not None
        # The generic fixture exercises the same immutable two-frame
        # transaction.  Mark its two copies as the boundary-fine subtype and
        # isolate this test to the final bootstrap handoff; the pixel-level
        # fine selector/revalidator has dedicated tests above.
        proof.seed.boundary_seed_fine_core = True
        live_seed.boundary_seed_fine_core = True
        provider._guarded_v2_bootstrap_phase = "expired"
        provider._guarded_v2_bootstrap_expiry_kind = "boundary_seed"
        provider._guarded_v2_bootstrap_commission_count = 0
        provider._guarded_v2_bootstrap_boundary_rearm_count = 0
        monkeypatch.setattr(
            provider,
            "_boundary_seed_fine_recovery_accepts",
            lambda *_args, **_kwargs: (True, "focused sealed fine proof"),
        )

        obj = provider.extractor.extract(case.current_frame, case.staged.mask)
        assert obj.valid
        attached, reason = provider._attach_lost_reappearance_visible_core_rgbd(
            case.current_frame,
            case.staged,
            obj,
            case.pending,
        )
        assert attached, reason
        provider._commit_final_publication_histories(
            case.current_frame,
            case.staged,
            obj,
        )

        state = provider.guarded_v2_bootstrap_state
        assert state["phase"] == "provisional"
        assert state["seed_frame_id"] == case.current_frame.frame_id
        assert state["boundary_rearm_count"] == 1
        assert state["commission_count"] == 0
        assert "sealed boundary-seed fine visible core" in state["status"]
        owner = provider._visible_exact_partial_continuity
        assert owner is not None
        assert owner.evidence_kind == "lost_reappearance_visible_core"
        expected_velocity = (
            np.asarray(owner.core_center_xy, dtype=np.float64)
            - np.asarray(proof.seed.candidate_center_xy, dtype=np.float64)
        ) / (case.current_frame.timestamp - case.seed_frame.timestamp)
        np.testing.assert_allclose(owner.center_velocity_px_s, expected_velocity)
        assert not provider.tracking_committed
    finally:
        provider.stop()


def test_lost_reappearance_visible_core_target_reset_drops_transaction(
    monkeypatch,
):
    """A target reset cannot carry a seed/proof into the next generation."""

    case = _stage_lost_reappearance_visible_core_case(monkeypatch)
    try:
        assert case.provider._lost_reappearance_visible_core_seed is not None
        assert case.pending.lost_reappearance_visible_core_proof is not None
        case.provider._reset_target_models()
        assert case.provider._lost_reappearance_visible_core_seed is None
        assert case.provider._pending_full_target_geometry_commit is None
        assert case.provider._recovery_publication_hypothesis is None
        assert case.provider._visible_exact_partial_continuity is None
        assert not case.provider.tracking_committed
    finally:
        case.provider.stop()


def test_lost_reappearance_visible_core_keeps_normal_three_evidence_recovery(
    monkeypatch,
):
    """The partial tick and ordinary later-RGB-D 3/3 path coexist."""

    case = _stage_lost_reappearance_visible_core_case(monkeypatch)
    provider = case.provider
    try:
        assert case.staged is not None, case.stage_reason
        obj = provider.extractor.extract(case.current_frame, case.staged.mask)
        attached, reason = provider._attach_lost_reappearance_visible_core_rgbd(
            case.current_frame, case.staged, obj, case.pending
        )
        assert attached, reason
        provider._commit_final_publication_histories(
            case.current_frame, case.staged, obj
        )
        partial_owner = provider._visible_exact_partial_continuity
        assert partial_owner is not None
        recovery = provider._recovery_publication_hypothesis
        assert recovery is not None
        assert recovery.evidence_kind == "two_exact_sam_seed_with_rgbd"

        # A third, distinct RGB-D tuple continues through the unchanged
        # unified recovery gate.  Its exact raw envelope and sanitized core
        # are intentionally identical to the second reviewed observation.
        later = _large_frame(315)
        pending, final_token = _bind_current_sanitized_evidence(
            provider,
            later,
            case.current_raw,
            case.current_broad,
        )
        provider._last_publication_guard_frame_id = later.frame_id
        provider._last_publication_guard_aligned_clean = (
            case.current_broad.copy()
        )
        provider._last_output_mask_source = "publication_sanitized_target_core"
        provider._last_online_sam2_status = (
            "guarded_primary_visible_partial_sanitized_continuation: "
            "later distinct RGB-D after LOST visible core"
        )
        provider._last_online_sam2_exact_result = True
        provider._current_recovery_evidence_outcome = "candidate"
        monkeypatch.setattr(
            provider,
            "_publication_candidate_agrees_with_committed_clean_mask",
            lambda *_args, **_kwargs: (True, "focused later clean candidate"),
        )
        result = _exact_token_mask_result(
            final_token,
            message="later distinct RGB-D after LOST visible core",
        )
        result.source = "online_sam2_video"
        later_obj = provider.extractor.extract(later, final_token)
        published, published_obj = provider._apply_recovery_publication_gate(
            later, result, later_obj
        )
        assert published.valid and published_obj.valid
        assert "recovery publication pending final commit 3/3" in published.message
        recovery_proof = provider._pending_recovery_publication_commit
        assert recovery_proof is not None
        assert recovery_proof.hits == 3
        assert recovery_proof.frame_id == later.frame_id
        assert recovery_proof.evidence_kind == "two_exact_sam_plus_later_rgbd"
        assert not provider.tracking_committed
        assert provider._visible_exact_partial_continuity is partial_owner

        provider._commit_final_publication_histories(
            later, published, published_obj
        )
        assert provider.tracking_committed
        assert provider._pending_recovery_publication_commit is None
        assert provider._recovery_publication_hypothesis is None
        assert provider._lost_reappearance_visible_core_seed is None
        assert pending is not provider._pending_full_target_geometry_commit
    finally:
        provider.stop()


def test_lost_reappearance_visible_core_late_commit_failure_rolls_back(
    monkeypatch,
):
    """A late exception cannot leak partial/full/tracker ownership."""

    case = _stage_lost_reappearance_visible_core_case(monkeypatch)
    provider = case.provider
    try:
        assert case.staged is not None, case.stage_reason
        obj = provider.extractor.extract(case.current_frame, case.staged.mask)
        attached, reason = provider._attach_lost_reappearance_visible_core_rgbd(
            case.current_frame, case.staged, obj, case.pending
        )
        assert attached, reason

        full_before = copy.deepcopy(provider._full_target_geometry_authority)
        trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)
        final_before = copy.deepcopy(provider._last_final_clean_publication)
        recovery_before = copy.deepcopy(provider._recovery_publication_hypothesis)
        tracker_before = case.tracker.snapshot_state()
        original = provider._commit_lost_reappearance_visible_core

        def fail_after_partial_commit(frame, mask_result, object_pcd):
            assert original(frame, mask_result, object_pcd)
            provider._full_target_geometry_authority.frame_id = 99999
            case.tracker.token = "leaked-lost-reappearance-owner"
            raise RuntimeError("focused late LOST commit failure")

        monkeypatch.setattr(
            provider,
            "_commit_lost_reappearance_visible_core",
            fail_after_partial_commit,
        )
        with pytest.raises(RuntimeError, match="focused late LOST commit failure"):
            provider._commit_final_publication_histories(
                case.current_frame, case.staged, obj
            )

        assert provider._visible_exact_partial_continuity is None
        assert not provider.tracking_committed
        assert provider._pending_full_target_geometry_commit is None
        assert provider._pending_recovery_publication_commit is None
        assert provider._lost_reappearance_visible_core_seed is not None
        full_after = provider._full_target_geometry_authority
        assert full_after.frame_id == full_before.frame_id
        assert full_after.timestamp == full_before.timestamp
        assert full_after.area == pytest.approx(full_before.area)
        trusted_after = provider._trusted_online_sam2_publication
        assert trusted_after.frame_id == trusted_before.frame_id
        assert trusted_after.timestamp == trusted_before.timestamp
        np.testing.assert_array_equal(trusted_after.mask, trusted_before.mask)
        assert provider._last_final_clean_publication.frame_id == (
            final_before.frame_id
        )
        recovery_after = provider._recovery_publication_hypothesis
        assert recovery_after.frame_id == recovery_before.frame_id
        assert recovery_after.hits == recovery_before.hits
        assert recovery_after.evidence_kind == recovery_before.evidence_kind
        assert recovery_after.mask_digest == recovery_before.mask_digest
        tracker_after = case.tracker.snapshot_state()
        assert tracker_after.token == tracker_before.token
        np.testing.assert_array_equal(
            tracker_after.state.last_mask, tracker_before.state.last_mask
        )
    finally:
        provider.stop()


def test_full_packet_broad_partial_commit_failure_restores_pcd_history_atomically(
    monkeypatch,
):
    """A late partial-owner fault cannot leave the current PCD tick behind."""

    case = _stage_broad_visible_core_case(monkeypatch)
    provider = case.provider
    try:
        assert case.staged is not None, case.stage_reason
        assert case.pending.broad_visible_core_proof is not None
        assert provider.guarded_v2_bootstrap_state["phase"] == "commissioned"

        # The common provider fixture deliberately uses a stateless fake
        # history.  Replace only that adapter with the production buffer and
        # seed two distinguishable observations, so this test can prove the
        # exact deque bytes, centers, timestamps, and last center are restored.
        history = PCDHistoryBuffer(history_len=4, max_center_jump=1.0)
        first_points = (
            np.arange(24, dtype=np.float32).reshape(8, 3) / 100.0
        )
        second_points = first_points + np.float32(0.25)
        first_center = np.asarray([0.47, -0.02, 0.20], dtype=np.float32)
        second_center = np.asarray([0.49, -0.01, 0.20], dtype=np.float32)
        assert history.update(first_points, first_center, 1.25).valid
        assert history.update(second_points, second_center, 1.50).valid
        provider.history = history

        points_before = tuple(value.copy() for value in history.points)
        centers_before = tuple(value.copy() for value in history.centers)
        timestamps_before = tuple(history.timestamps)
        last_center_before = history.last_valid_center.copy()
        history_len_before = len(history.points)

        final_clean_before = copy.deepcopy(provider._last_final_clean_publication)
        full_before = copy.deepcopy(provider._full_target_geometry_authority)
        trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)
        partial_before = copy.deepcopy(provider._visible_exact_partial_continuity)

        provider._last_output_mask_source = "publication_sanitized_target_core"
        provider._last_online_sam2_status = (
            "guarded_primary_visible_partial_sanitized_continuation: "
            "fixed-anchor broad visible core packet rollback test"
        )
        provider._last_online_sam2_exact_result = True
        provider._tracking_committed = True
        monkeypatch.setattr(
            provider,
            "_run_semantic_sam2_mask_pipeline",
            lambda **_kwargs: SimpleNamespace(
                frame=case.frame,
                mask_result=case.staged,
                camera_ms=0.0,
                tracker_ms=0.0,
                online_sam2_ms=0.0,
                sam2_reinit_ms=0.0,
            ),
        )

        original_partial_commit = provider._commit_visible_exact_partial_continuity
        observed_post_update = {}

        def fail_after_real_partial_commit(frame, mask_result, pending):
            # This is the real broad/partial history implementation.  Throw
            # only after it has consumed and refreshed the partial owner.
            committed = original_partial_commit(frame, mask_result, pending)
            assert committed
            observed_post_update["history_len"] = len(provider.history.points)
            observed_post_update["partial_frame"] = (
                provider._visible_exact_partial_continuity.frame_id
            )
            raise RuntimeError("injected broad partial-history failure")

        monkeypatch.setattr(
            provider,
            "_commit_visible_exact_partial_continuity",
            fail_after_real_partial_commit,
        )

        with pytest.raises(
            RuntimeError, match="injected broad partial-history failure"
        ):
            provider.step()

        # Prove PCDHistoryBuffer.update completed before the injected final
        # history fault, rather than testing an earlier acquisition/gate exit.
        assert observed_post_update == {
            "history_len": history_len_before + 1,
            "partial_frame": case.frame.frame_id,
        }

        restored = provider.history
        assert isinstance(restored, PCDHistoryBuffer)
        assert len(restored.points) == history_len_before
        assert len(restored.centers) == len(centers_before)
        assert tuple(restored.timestamps) == timestamps_before
        assert restored.last_valid_center.tobytes() == last_center_before.tobytes()
        for actual, expected in zip(restored.points, points_before):
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            assert actual.tobytes() == expected.tobytes()
        for actual, expected in zip(restored.centers, centers_before):
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            assert actual.tobytes() == expected.tobytes()

        # The late commit temporarily writes final-clean and partial history;
        # neither it nor the full/trusted authority may survive rollback.
        final_clean_after = provider._last_final_clean_publication
        assert final_clean_after.frame_id == final_clean_before.frame_id
        assert final_clean_after.timestamp == final_clean_before.timestamp
        assert final_clean_after.mask.tobytes() == final_clean_before.mask.tobytes()

        full_after = provider._full_target_geometry_authority
        assert full_after.frame_id == full_before.frame_id
        assert full_after.timestamp == full_before.timestamp
        assert full_after.area == full_before.area
        assert full_after.bbox_area == full_before.bbox_area

        trusted_after = provider._trusted_online_sam2_publication
        assert trusted_after.frame_id == trusted_before.frame_id
        assert trusted_after.timestamp == trusted_before.timestamp
        assert trusted_after.area == trusted_before.area
        assert trusted_after.mask.tobytes() == trusted_before.mask.tobytes()

        partial_after = provider._visible_exact_partial_continuity
        assert partial_after.frame_id == partial_before.frame_id
        assert partial_after.timestamp == partial_before.timestamp
        assert partial_after.hits == partial_before.hits
        assert partial_after.mask_digest == partial_before.mask_digest
        assert partial_after.mask.tobytes() == partial_before.mask.tobytes()
    finally:
        provider.stop()


def _arm_exact_two_sam_then_same_frame_rgbd_recovery(provider, clean_mask):
    """Build the real unified proof prefix without forging a source label.

    Frames 116 and 117 are two distinct exact SAM observations.  Frame 117
    then crosses the provider's RGB-D adapter only to attach 3-D evidence; it
    is deliberately not counted as a third frame.  The caller must still
    supply one later distinct RGB-D frame before anything can publish.
    """

    provider._break_trusted_online_sam2_continuity("focused synthetic LOST")
    provider._tracking_committed = False
    provider._uncommitted_frames = 0
    provider._visible_exact_partial_continuity = None
    provider._recovery_visible_partial_handoff_frame_id = None
    provider._recovery_publication_hypothesis = None
    provider._online_sam2_recovery_hypothesis = None

    frames = [
        _moving_target_frame(frame_id, (70, 30, 110, 70))
        for frame_id in (116, 117)
    ]
    last_geometry = None
    second_exact_token = None
    for index, frame in enumerate(frames, start=1):
        if index == 2:
            # This is the exact provider-owned immutable mask which the real
            # guarded primary path registers before carrying its two-SAM
            # recovery evidence into the same-frame RGB-D adapter.
            second_exact_token = provider._defer_full_target_geometry_commit(
                frame, clean_mask
            )
            assert second_exact_token is not None
            geometry = second_exact_token.geometry
        else:
            geometry, reason = provider._online_mask_geometry(frame, clean_mask)
            assert geometry is not None, reason
        confirmed, hits, _detail = provider._confirm_online_recovery(
            frame.frame_id,
            geometry,
            timestamp=frame.timestamp,
            source="guarded_sam2_primary",
        )
        assert hits == index
        assert confirmed is (index == 2)
        last_geometry = geometry

    evidence = provider._online_sam2_recovery_hypothesis
    assert evidence is not None and evidence.hits == 2
    provider._seed_unified_publication_from_online_sam2(
        frames[-1], last_geometry, evidence
    )
    two_sam_seed = provider._recovery_publication_hypothesis
    assert two_sam_seed is not None
    assert two_sam_seed.evidence_kind == "two_exact_sam_seed"
    assert two_sam_seed.hits == 2
    assert two_sam_seed.frame_id == frames[-1].frame_id
    assert two_sam_seed.target_generation == provider._target_generation
    assert two_sam_seed.timestamp_bits == provider._frame_timestamp_bits(
        frames[-1].timestamp
    )
    assert two_sam_seed.mask_digest == second_exact_token.mask_digest
    provider._online_sam2_recovery_hypothesis = None
    provider._current_recovery_evidence_outcome = "candidate"
    provider._last_online_sam2_status = "guarded_primary_two_sam_confirmed"
    provider._last_online_sam2_exact_result = True
    provider._last_output_mask_source = "online_sam2_guarded_primary"

    assert second_exact_token is not None
    same_frame_result = _exact_token_mask_result(
        second_exact_token.mask,
        message="second exact SAM awaiting later RGB-D",
    )
    same_frame_result.source = "online_sam2_video"
    same_frame_obj = provider.extractor.extract(
        frames[-1], second_exact_token.mask
    )
    pending_mask, pending_obj, recovered = provider._gate_recovery_publication(
        frames[-1], same_frame_result, same_frame_obj
    )
    assert not pending_mask.valid and not pending_obj.valid and not recovered
    assert "awaiting one distinct RGB-D frame" in pending_mask.message
    attached = provider._recovery_publication_hypothesis
    assert attached is not None
    assert attached.evidence_kind == "two_exact_sam_seed_with_rgbd"
    assert attached.hits == 2
    assert attached.frame_id == frames[-1].frame_id
    assert attached.target_generation == provider._target_generation
    assert attached.timestamp_bits == provider._frame_timestamp_bits(
        frames[-1].timestamp
    )
    assert attached.mask_digest == second_exact_token.mask_digest
    assert not provider.tracking_committed
    assert provider._visible_exact_partial_continuity is None
    return frames[-1]


def _complete_guarded_recovery_for_masks(
    monkeypatch,
    provider,
    *,
    raw,
    aligned,
    final,
):
    """Complete a bound 2-SAM + later-RGBD recovery for exact masks."""

    _arm_exact_two_sam_then_same_frame_rgbd_recovery(provider, aligned)
    frame = _moving_target_frame(118, (70, 30, 110, 70))
    pending, final_token = _bind_current_sanitized_evidence(
        provider, frame, raw, final
    )
    provider._last_publication_guard_frame_id = frame.frame_id
    provider._last_publication_guard_aligned_clean = aligned.copy()
    provider._last_output_mask_source = "publication_sanitized_target_core"
    provider._last_online_sam2_status = (
        "guarded_primary_visible_partial_sanitized_continuation: "
        "later distinct RGB-D"
    )
    provider._last_online_sam2_exact_result = True
    provider._current_recovery_evidence_outcome = "candidate"
    # The test isolates the recovery handoff and partial-owner boundary.  The
    # independent publication-geometry guard has its own exhaustive suite.
    monkeypatch.setattr(
        provider,
        "_publication_candidate_agrees_with_committed_clean_mask",
        lambda *_args, **_kwargs: (True, "focused clean-aligned candidate"),
    )
    final_result = _exact_token_mask_result(
        final_token, message="later distinct RGB-D fragmented target core"
    )
    final_result.source = "online_sam2_video"
    final_obj = provider.extractor.extract(frame, final_token)
    published, published_obj = provider._apply_recovery_publication_gate(
        frame, final_result, final_obj
    )
    assert published.valid and published_obj.valid
    assert "recovery publication pending final commit 3/3" in published.message
    # The exact proof exists here, but ownership must not cross until the
    # final returned mask/object history transaction revalidates the token.
    assert not provider.tracking_committed
    assert provider._last_output_mask_source == "recovery_pending_final_commit"
    proof = provider._pending_recovery_publication_commit
    final_evidence = provider._lookup_frame_mask_evidence(frame, final_token)
    assert proof is not None and final_evidence is not None
    assert proof.evidence_kind == "two_exact_sam_plus_later_rgbd"
    assert proof.hits == 3
    assert proof.frame_id == frame.frame_id
    assert proof.timestamp_bits == provider._frame_timestamp_bits(frame.timestamp)
    assert proof.target_generation == provider._target_generation
    assert proof.mask_digest == final_evidence.mask_digest
    return frame, pending, published, published_obj, aligned, final_token


def _complete_raw_exact_guarded_recovery(monkeypatch, provider, mask):
    """Complete the formal 2-SAM + later-RGBD proof with raw exact bytes."""

    _arm_exact_two_sam_then_same_frame_rgbd_recovery(provider, mask)
    frame = _moving_target_frame(118, (70, 30, 110, 70))
    raw_evidence = provider._defer_full_target_geometry_commit(frame, mask)
    assert raw_evidence is not None
    provider._last_output_mask_source = "online_sam2_guarded_primary"
    provider._last_online_sam2_status = "guarded_primary_later_rgbd_candidate"
    provider._last_online_sam2_exact_result = True
    provider._current_recovery_evidence_outcome = "candidate"
    monkeypatch.setattr(
        provider,
        "_publication_candidate_agrees_with_committed_clean_mask",
        lambda *_args, **_kwargs: (True, "focused raw exact recovery"),
    )
    result = _exact_token_mask_result(
        raw_evidence.mask,
        message="later distinct RGB-D raw exact recovery",
    )
    result.source = "online_sam2_video"
    obj = provider.extractor.extract(frame, raw_evidence.mask)
    published, published_obj = provider._apply_recovery_publication_gate(
        frame, result, obj
    )
    assert published.valid and published_obj.valid
    assert provider._pending_recovery_publication_commit is not None
    pending = provider._pending_full_target_geometry_commit
    assert pending is not None
    return frame, pending, published, published_obj, raw_evidence.mask


def test_boundary_clipped_bootstrap_rearms_once_after_raw_interior_recovery(
    monkeypatch,
):
    """A clipped entry commissions only after interior recovery + two raws."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    try:
        boundary_frame = _moving_target_frame(2, (0, 30, 40, 70))
        boundary_mask = _large_mask((0, 30, 40, 70))
        _commit_bootstrap_raw_for_test(provider, boundary_frame, boundary_mask)
        assert provider.guarded_v2_bootstrap_state["phase"] == "expired"
        assert provider.guarded_v2_bootstrap_state["status"] == (
            "expired: raw candidate touches the effective image boundary"
        )

        recovered = _large_mask((70, 30, 110, 70))
        (
            recovery_frame,
            _pending,
            published,
            published_obj,
            recovery_token,
        ) = _complete_raw_exact_guarded_recovery(
            monkeypatch, provider, recovered
        )
        provider._commit_final_publication_histories(
            recovery_frame, published, published_obj
        )
        state = provider.guarded_v2_bootstrap_state
        assert state["phase"] == "provisional"
        assert state["seed_frame_id"] == recovery_frame.frame_id
        assert state["candidate_frame_id"] is None
        assert state["commission_count"] == 0
        assert state["boundary_rearm_count"] == 1
        assert "rearmed once from confirmed interior recovery" in state["status"]

        provider._current_recovery_evidence_outcome = "no_evidence"
        for frame_id, shift in ((119, 1), (120, 2)):
            mask = np.roll(recovery_token, shift, axis=1)
            frame = _moving_target_frame(
                frame_id,
                (70 + shift, 30, 110 + shift, 70),
            )
            _commit_bootstrap_raw_for_test(
                provider,
                frame,
                mask,
                eligibility_kind="bootstrap_recovery_visible_partial",
                online_status=(
                    "guarded_primary_visible_partial_continuation: "
                    "focused exact-current continuation"
                ),
            )

        state = provider.guarded_v2_bootstrap_state
        assert state["phase"] == "commissioned"
        assert state["commissioned_frame_id"] == 120
        assert state["commission_count"] == 1
        assert state["boundary_rearm_count"] == 1
    finally:
        provider.stop()


def test_boundary_clipped_bootstrap_rejects_sanitized_recovery_commit(
    monkeypatch,
):
    """A clipped seed cannot let a hand-like sanitizer core self-confirm."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    try:
        boundary_frame = _moving_target_frame(2, (0, 30, 40, 70))
        boundary_mask = _large_mask((0, 30, 40, 70))
        _commit_bootstrap_raw_for_test(provider, boundary_frame, boundary_mask)
        state = provider.guarded_v2_bootstrap_state
        assert state["phase"] == "expired"
        assert state["expiry_kind"] == "boundary_seed"

        recovered = _large_mask((70, 30, 110, 70))
        _arm_exact_two_sam_then_same_frame_rgbd_recovery(provider, recovered)
        final_frame = _moving_target_frame(118, (70, 30, 110, 70))
        raw = np.maximum(recovered, _large_mask((110, 20, 145, 80)))
        sanitized = recovered.copy()
        pending, sanitized_token = _bind_current_sanitized_evidence(
            provider, final_frame, raw, sanitized
        )
        authority_before = copy.deepcopy(provider._full_target_geometry_authority)
        clean_before = copy.deepcopy(provider._last_final_clean_publication)
        provider._last_online_sam2_exact_result = True
        provider._current_recovery_evidence_outcome = "candidate"
        result = _exact_token_mask_result(
            sanitized_token,
            message="sanitized recovery from a clipped startup seed",
        )
        result.source = "online_sam2_video"
        obj = provider.extractor.extract(final_frame, sanitized_token)

        public_mask, public_obj, recovered_now = provider._gate_recovery_publication(
            final_frame, result, obj
        )

        assert not recovered_now
        assert not public_mask.valid and not public_obj.valid
        assert "boundary-clipped bootstrap recovery remains fail-closed" in (
            public_mask.message
        )
        assert provider._pending_recovery_publication_commit is None
        assert provider._recovery_publication_hypothesis is None
        assert not provider.tracking_committed
        assert provider.guarded_v2_bootstrap_state["phase"] == "expired"
        assert provider.guarded_v2_bootstrap_state["boundary_rearm_count"] == 0
        assert pending.mask_digest == provider._publication_mask_digest(raw)
        assert provider._full_target_geometry_authority.frame_id == (
            authority_before.frame_id
        )
        assert provider._last_final_clean_publication.frame_id == (
            clean_before.frame_id
        )
    finally:
        provider.stop()


def test_bootstrap_recovery_partial_kind_cannot_be_forged(monkeypatch):
    """Ordinary visible-partial evidence stays bootstrap-ineligible."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    try:
        frame = _moving_target_frame(2, (41, 30, 81, 70))
        raw = _large_mask((41, 30, 81, 70))
        _commit_bootstrap_raw_for_test(
            provider,
            frame,
            raw,
            eligibility_kind="bootstrap_recovery_visible_partial",
            online_status=(
                "guarded_primary_visible_partial_continuation: forged test"
            ),
        )
        state = provider.guarded_v2_bootstrap_state
        assert state["phase"] == "provisional"
        assert state["candidate_frame_id"] is None
        assert state["boundary_rearm_count"] == 0
        assert "no bootstrap-eligible raw exact token" in state["status"]
    finally:
        provider.stop()


def _complete_fragmented_tail_recovery(
    monkeypatch,
    provider,
    *,
    below_raw_majority: bool,
):
    """Complete the bound recovery for one reviewed fragmented-tail ratio."""

    raw, aligned, final = _fragmented_tail_masks(
        below_raw_majority=below_raw_majority
    )
    return _complete_guarded_recovery_for_masks(
        monkeypatch,
        provider,
        raw=raw,
        aligned=aligned,
        final=final,
    )


def test_recovery_below_raw_majority_requires_bound_three_evidence_handoff(
    monkeypatch,
):
    """A .498 raw subset may seed hits=2 only after exact 2-SAM+RGB-D."""

    initial = _large_mask((40, 30, 80, 70))
    provider, tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    authority_before = copy.deepcopy(provider._full_target_geometry_authority)
    trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)
    try:
        (
            frame,
            pending,
            published,
            published_obj,
            aligned,
            final_token,
        ) = _complete_fragmented_tail_recovery(
            monkeypatch,
            provider,
            below_raw_majority=True,
        )
        mean, supported, count = tracker.target_appearance_support_stats(
            frame, final_token, threshold=0.42
        )
        assert (mean, supported, count) == pytest.approx((0.95, 1.0, 797))
        assert float(final_token.sum()) / float(pending.mask.sum()) == pytest.approx(
            0.498125
        )
        assert float(final_token.sum()) / float(aligned.sum()) > 0.90

        provider._commit_final_publication_histories(
            frame, published, published_obj
        )
        owner = provider._visible_exact_partial_continuity
        assert owner is not None
        assert owner.frame_id == frame.frame_id
        assert owner.hits == 2
        assert owner.evidence_kind == "recovery_sanitized_target_core"
        assert owner.sanitized_handoff_eligible
        assert provider._full_target_geometry_authority.frame_id == (
            authority_before.frame_id
        )
        assert provider._trusted_online_sam2_publication.frame_id == (
            trusted_before.frame_id
        )
    finally:
        provider.stop()


def test_forged_recovery_label_and_frame_marker_cannot_create_owner(monkeypatch):
    """A source string/frame integer is not the completed recovery proof."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    provider._visible_exact_partial_continuity = None
    frame = _moving_target_frame(118, (70, 30, 110, 70))
    raw, aligned, final = _fragmented_tail_masks(below_raw_majority=False)
    pending, final_token = _bind_current_sanitized_evidence(
        provider, frame, raw, final
    )
    provider._last_publication_guard_frame_id = frame.frame_id
    provider._last_publication_guard_aligned_clean = aligned.copy()
    provider._last_output_mask_source = "recovery_committed"
    provider._recovery_visible_partial_handoff_frame_id = frame.frame_id
    provider._last_online_sam2_exact_result = True
    try:
        assert not provider._commit_visible_exact_partial_continuity(
            frame,
            _exact_token_mask_result(
                final_token, message="forged recovery source and frame marker"
            ),
            pending,
        )
        assert provider._visible_exact_partial_continuity is None
    finally:
        provider.stop()


@pytest.mark.parametrize("fault", ("frame", "target_generation", "digest"))
def test_completed_recovery_proof_is_exact_frame_generation_digest_bound(
    monkeypatch,
    fault,
):
    """A completed proof cannot authorize a neighboring or mutated token."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    try:
        frame, pending, published, _obj, _aligned, final_token = (
            _complete_fragmented_tail_recovery(
                monkeypatch,
                provider,
                below_raw_majority=False,
            )
        )
        proof = provider._pending_recovery_publication_commit
        assert proof is not None
        corrupt = SimpleNamespace(
            frame_id=int(proof.frame_id),
            timestamp_bits=int(proof.timestamp_bits),
            target_generation=int(proof.target_generation),
            mask_digest=str(proof.mask_digest),
            hits=int(proof.hits),
            evidence_kind=str(proof.evidence_kind),
        )
        if fault == "frame":
            corrupt.frame_id -= 1
        elif fault == "target_generation":
            corrupt.target_generation += 1
        else:
            corrupt.mask_digest = "0" * 64
        provider._pending_recovery_publication_commit = corrupt
        assert not provider._commit_visible_exact_partial_continuity(
            frame,
            _exact_token_mask_result(
                final_token, message=f"stale recovery {fault} proof"
            ),
            pending,
        )
        assert provider._visible_exact_partial_continuity is None
        assert published.valid
    finally:
        provider.stop()


def test_two_sam_proof_plus_full_occlusion_remains_fail_closed(monkeypatch):
    """Exact SAM history is not permission to publish through occlusion."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    _raw, aligned, _final = _fragmented_tail_masks(below_raw_majority=True)
    try:
        _arm_exact_two_sam_then_same_frame_rgbd_recovery(provider, aligned)
        frame = _moving_target_frame(118, (70, 30, 110, 70))
        empty = np.zeros_like(aligned)
        missing = _mask_result(
            empty,
            valid=False,
            message="exact SAM empty under full RH56 occlusion",
        )
        missing_obj = provider.extractor.extract(frame, empty)
        published, published_obj = provider._apply_recovery_publication_gate(
            frame, missing, missing_obj
        )
        assert not published.valid and not published_obj.valid
        assert not provider.tracking_committed
        assert provider._visible_exact_partial_continuity is None
        assert provider._recovery_visible_partial_handoff_frame_id is None
    finally:
        provider.stop()


def _integrated_guarded_recovery_provider(monkeypatch):
    """Build a real exact-empty -> 2-SAM -> later-RGBD step sequence."""

    target1 = _large_mask((40, 30, 80, 70))
    target3 = _large_mask((42, 30, 82, 70))
    target4 = _large_mask((43, 30, 83, 70))
    target5 = _large_mask((44, 30, 84, 70))
    empty = np.zeros_like(target1)
    tracker = _IdentityAwareFakeTracker(
        {
            2: _mask_result(target1, message="clean adaptive"),
            3: _mask_result(target3, message="adaptive visible again"),
            4: _mask_result(target4, message="adaptive visible again"),
            5: _mask_result(target5, message="later RGB-D candidate"),
        }
    )
    manager = _FakeOnlineSAM2Manager(
        {
            2: _video_result(2, empty, valid=False, message="occluded"),
            3: _video_result(3, target3),
            4: _video_result(4, target4),
            5: _video_result(5, target5),
        }
    )
    frames = [
        _moving_target_frame(2, (40, 30, 80, 70)),
        _moving_target_frame(3, (42, 30, 82, 70)),
        _moving_target_frame(4, (43, 30, 83, 70)),
        _moving_target_frame(5, (44, 30, 84, 70)),
    ]
    provider, _camera = _provider(
        monkeypatch,
        frames,
        tracker,
        manager,
        recovery_publication_mode="unified_three_evidence",
    )
    provider.online_sam2_cfg["mask_publication_mode"] = "guarded_sam2_primary"
    assert provider.initialize_from_mask(
        _moving_target_frame(1, (40, 30, 80, 70)), target1
    )
    return provider, target5


@pytest.mark.parametrize("fault_stage", ("final_history", "packet"))
def test_guarded_recovery_final_commit_failure_is_atomic(
    monkeypatch,
    fault_stage,
):
    """A post-proof exception restores every pre-frame recovery owner."""

    provider, _target5 = _integrated_guarded_recovery_provider(monkeypatch)
    try:
        prefix = [provider.step() for _ in range(3)]
        assert all(not result.valid for _, result, _, _ in prefix)
        tracking_before = provider.tracking_committed
        history_before = copy.deepcopy(provider.history)
        packet_before = provider.last_packet
        partial_before = copy.deepcopy(provider._visible_exact_partial_continuity)
        proof_before = copy.deepcopy(provider._pending_recovery_publication_commit)
        marker_before = provider._recovery_visible_partial_handoff_frame_id
        assert not tracking_before
        assert proof_before is None and marker_before is None

        if fault_stage == "final_history":
            original = provider._commit_final_publication_histories_impl

            def fail_after_final_history(frame, mask_result, obj):
                original(frame, mask_result, obj)
                raise RuntimeError("injected final-history failure")

            monkeypatch.setattr(
                provider,
                "_commit_final_publication_histories_impl",
                fail_after_final_history,
            )
        else:
            monkeypatch.setattr(
                provider,
                "_packet_debug",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    RuntimeError("injected packet failure")
                ),
            )

        with pytest.raises(RuntimeError, match=f"injected {fault_stage.replace('_', '-')}"):
            provider.step()

        assert provider.tracking_committed is tracking_before
        assert provider.history.reset_calls == history_before.reset_calls
        assert provider.last_packet is packet_before
        assert provider._visible_exact_partial_continuity == partial_before
        assert provider._pending_recovery_publication_commit == proof_before
        assert provider._recovery_visible_partial_handoff_frame_id == marker_before
    finally:
        provider.stop()


def test_guarded_recovery_mask_only_final_commit_consumes_exact_proof(
    monkeypatch,
):
    """Successful mask-only publication consumes proof and seeds ownership."""

    provider, target5 = _integrated_guarded_recovery_provider(monkeypatch)
    try:
        outputs = [provider.step_mask_only()[1] for _ in range(4)]
        assert [result.valid for result in outputs] == [False, False, False, True]
        np.testing.assert_array_equal(outputs[-1].mask, target5)
        assert "recovery publication stable 3/3" in outputs[-1].message
        assert provider.tracking_committed
        owner = provider._visible_exact_partial_continuity
        assert owner is not None
        assert owner.frame_id == 5
        assert owner.hits >= 2
        assert owner.evidence_kind.startswith("recovery_")
        assert provider._pending_recovery_publication_commit is None
        assert provider._recovery_visible_partial_handoff_frame_id is None
        assert provider._last_output_mask_source == "recovery_committed"
        assert provider.last_packet is None
    finally:
        provider.stop()


def _shape_conflict_candidate():
    core = _large_mask((62, 30, 102, 70))
    hand = _large_mask((102, 40, 110, 60))
    raw = np.logical_or(core > 0, hand > 0).astype(np.uint8)
    frame = _moving_target_frame(
        53,
        (62, 30, 102, 70),
        contaminant_bbox=(102, 40, 110, 60),
    )
    return frame, core, hand, raw


def test_guarded_primary_source53_clean_raw_uses_partial_only_shape_floor(
    monkeypatch,
):
    """The reviewed 0.834 clean raw mask publishes without LOST or crop."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    provider._break_trusted_online_sam2_continuity("stale full owner")
    _install_confirmed_sanitized_partial_owner(provider)
    trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)
    full_before = copy.deepcopy(provider._full_target_geometry_authority)
    frame = _moving_target_frame(53, (62, 30, 102, 70))
    raw = _large_mask((62, 30, 102, 70))
    provider._defer_full_target_geometry_commit(frame, raw)
    provider._last_output_mask_source = "online_sam2_guarded_primary"
    provider._last_online_sam2_exact_result = True
    # Source51 was finally published through the target-core sanitizer.  That
    # revokes full/trusted derivatives, but is not an RPC/empty input gap: the
    # confirmed exact partial owner is the continuity authority for source53.
    provider._trusted_online_sam2_transient_gap_active = True
    original_coverage = provider._scale_normalized_reference_coverage
    monkeypatch.setattr(
        provider,
        "_scale_normalized_reference_coverage",
        lambda *_args, **_kwargs: (
            0.834,
            "reviewed source53 bbox_scale=1.244 retained=1553/1862",
        ),
    )
    try:
        prepared = provider._prepare_guarded_sam2_primary_mask(
            frame,
            _exact_token_mask_result(
                provider._pending_full_target_geometry_commit.mask,
                message="reviewed clean source53 exact raw",
            ),
        )
        assert prepared.valid
        np.testing.assert_array_equal(prepared.mask, raw)
        assert provider._last_online_sam2_status.startswith(
            "guarded_primary_visible_partial_continuation:"
        )
        assert "non-authoritative partial-only shape=0.834/0.830" in (
            provider._last_online_sam2_status
        )
        assert not provider._pending_full_target_geometry_commit.raw_exact_eligible

        # Restore the real shape function for the independent final guard.
        monkeypatch.setattr(
            provider, "_scale_normalized_reference_coverage", original_coverage
        )
        obj = provider._mask_only_publication_evidence(frame, prepared.mask)
        published, obj = provider._apply_recovery_publication_gate(
            frame, prepared, obj
        )
        assert published.valid and obj.valid
        np.testing.assert_array_equal(published.mask, raw)
        assert provider._recovery_publication_hypothesis is None
        provider._commit_final_publication_histories(frame, published, obj)
        owner = provider._visible_exact_partial_continuity
        assert owner.frame_id == 53
        assert owner.evidence_kind == "raw_exact"
        assert provider._trusted_online_sam2_publication.frame_id == (
            trusted_before.frame_id
        )
        assert provider._full_target_geometry_authority.frame_id == (
            full_before.frame_id
        )
    finally:
        provider.stop()


def test_guarded_partial_repair_marks_sanitized_publication_source(monkeypatch):
    """A repaired exact mask reaches history as a sanitizer-owned core."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    provider._break_trusted_online_sam2_continuity("focused partial repair")
    _install_confirmed_sanitized_partial_owner(provider)
    frame, core, _hand, raw = _shape_conflict_candidate()
    provider._defer_full_target_geometry_commit(frame, raw)
    provider._last_output_mask_source = "online_sam2_guarded_primary"
    provider._last_online_sam2_exact_result = True

    def reject_raw(*_args, **_kwargs):
        provider._last_visible_exact_partial_failure_kind = "shape_only"
        return False, "focused raw contains a hand appendage"

    repaired = _mask_result(
        core,
        message="focused sanitizer retained the visible target core",
    )
    repaired.source = "publication_sanitized_target_core"
    monkeypatch.setattr(
        provider,
        "_visible_exact_partial_candidate_agrees",
        reject_raw,
    )
    monkeypatch.setattr(
        provider,
        "_guarded_visible_partial_sanitized_continuation",
        lambda *_args, **_kwargs: (repaired, "focused repaired core"),
    )
    try:
        prepared = provider._prepare_guarded_sam2_primary_mask(
            frame,
            _exact_token_mask_result(
                provider._pending_full_target_geometry_commit.mask,
                message="focused raw exact SAM envelope",
            ),
        )
        assert prepared.valid
        np.testing.assert_array_equal(prepared.mask, core)
        assert provider._last_output_mask_source == (
            "publication_sanitized_target_core"
        )
        assert provider._last_online_sam2_status.startswith(
            "guarded_primary_visible_partial_sanitized_continuation:"
        )
    finally:
        provider.stop()


def test_guarded_primary_shape_only_conflict_publishes_only_sanitized_core(
    monkeypatch,
):
    """The 0.83 raw admission still lets the final guard remove a hand."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    provider._break_trusted_online_sam2_continuity("test partial handoff")
    _install_confirmed_sanitized_partial_owner(provider)
    trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)
    full_before = copy.deepcopy(provider._full_target_geometry_authority)
    frame, core, hand, raw = _shape_conflict_candidate()
    provider._defer_full_target_geometry_commit(frame, raw)
    provider._last_output_mask_source = "online_sam2_guarded_primary"
    provider._last_online_sam2_exact_result = True
    original_coverage = provider._scale_normalized_reference_coverage
    monkeypatch.setattr(
        provider,
        "_scale_normalized_reference_coverage",
        lambda *_args, **_kwargs: (0.840, "reviewed source53 shape conflict"),
    )
    try:
        prepared = provider._prepare_guarded_sam2_primary_mask(
            frame,
            _exact_token_mask_result(
                provider._pending_full_target_geometry_commit.mask,
                message="exact raw with hand appendage",
            ),
        )
        assert prepared.valid
        np.testing.assert_array_equal(prepared.mask, raw)
        assert provider._last_online_sam2_status.startswith(
            "guarded_primary_visible_partial_continuation:"
        )
        assert provider._pending_full_target_geometry_commit is not None
        assert not provider._pending_full_target_geometry_commit.raw_exact_eligible

        # The production final guard still owns robot-facing publication.
        monkeypatch.setattr(
            provider,
            "_scale_normalized_reference_coverage",
            original_coverage,
        )
        obj = provider._mask_only_publication_evidence(frame, prepared.mask)
        published, obj = provider._apply_recovery_publication_gate(
            frame, prepared, obj
        )
        assert published.valid and obj.valid
        np.testing.assert_array_equal(published.mask, core)
        assert not np.any(np.logical_and(published.mask > 0, hand > 0))
        assert provider._last_output_mask_source == (
            "publication_sanitized_target_core"
        )
        provider._commit_final_publication_histories(frame, published, obj)
        partial = provider._visible_exact_partial_continuity
        assert partial.frame_id == 53
        assert partial.hits == 4
        assert partial.evidence_kind == "sanitized_target_core"
        assert provider._trusted_online_sam2_publication.frame_id == (
            trusted_before.frame_id
        )
        assert provider._full_target_geometry_authority.frame_id == (
            full_before.frame_id
        )
    finally:
        provider.stop()


@pytest.mark.parametrize("fault", ("multi_component", "rpc_gap", "identity", "jump"))
def test_guarded_primary_partial_only_shape_floor_keeps_faults_closed(
    monkeypatch, fault
):
    """The 0.83 floor cannot bypass topology, gap, identity or motion gates."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    provider._break_trusted_online_sam2_continuity("stale full owner")
    _install_confirmed_sanitized_partial_owner(provider)
    bbox = (170, 30, 210, 70) if fault == "jump" else (62, 30, 102, 70)
    frame = _moving_target_frame(53, bbox)
    raw = _large_mask(bbox)
    detached = np.zeros_like(raw)
    if fault == "multi_component":
        detached[40:60, 104:114] = 1
        raw = np.logical_or(raw > 0, detached > 0).astype(np.uint8)
        frame.color_bgr[detached > 0] = (210, 80, 235)
    elif fault == "rpc_gap":
        provider._online_sam2_initialized = False
        provider._online_sam2_seed_pending = True
        provider._online_sam2_failure_reported = True
    elif fault == "identity":
        frame.color_bgr[:] = 45

    provider._defer_full_target_geometry_commit(frame, raw)
    provider._last_output_mask_source = "online_sam2_guarded_primary"
    provider._last_online_sam2_exact_result = True
    monkeypatch.setattr(
        provider,
        "_scale_normalized_reference_coverage",
        lambda *_args, **_kwargs: (0.834, f"negative {fault} shape"),
    )
    try:
        prepared = provider._prepare_guarded_sam2_primary_mask(
            frame,
            _exact_token_mask_result(
                provider._pending_full_target_geometry_commit.mask,
                message=f"negative {fault} exact raw",
            ),
        )
        assert not provider._last_online_sam2_status.startswith(
            "guarded_primary_visible_partial_continuation:"
        )
        assert not prepared.valid
    finally:
        provider.stop()


def test_guarded_primary_recovery_commit_hands_off_without_restarting_probation(
    monkeypatch,
):
    """A 3/3 recovery seeds hits=2; its next safe exact frame stays open."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    provider._break_trusted_online_sam2_continuity("synthetic LOST")
    provider._tracking_committed = False
    trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)
    full_before = copy.deepcopy(provider._full_target_geometry_authority)
    try:
        recovered_mask = _large_mask((70, 30, 110, 70))
        (
            recovered_frame,
            _pending,
            recovered_result,
            recovered_obj,
            _aligned,
            _final_token,
        ) = _complete_guarded_recovery_for_masks(
            monkeypatch,
            provider,
            raw=recovered_mask,
            aligned=recovered_mask,
            final=recovered_mask,
        )
        provider._commit_final_publication_histories(
            recovered_frame, recovered_result, recovered_obj
        )
        partial = provider._visible_exact_partial_continuity
        assert partial.frame_id == recovered_frame.frame_id
        assert partial.hits == 2
        assert partial.evidence_kind == "recovery_raw_exact_partial"
        assert provider._trusted_online_sam2_publication.frame_id == (
            trusted_before.frame_id
        )
        assert provider._full_target_geometry_authority.frame_id == (
            full_before.frame_id
        )

        next_frame_id = recovered_frame.frame_id + 1
        frame5 = _moving_target_frame(next_frame_id, (71, 30, 111, 70))
        mask5 = _large_mask((71, 30, 111, 70))
        provider._defer_full_target_geometry_commit(frame5, mask5)
        provider._last_output_mask_source = "online_sam2_guarded_primary"
        provider._last_online_sam2_exact_result = True
        prepared5 = provider._prepare_guarded_sam2_primary_mask(
            frame5, _mask_result(mask5, message="post-recovery exact frame")
        )
        assert prepared5.valid
        assert provider._last_online_sam2_status.startswith(
            "guarded_primary_visible_partial_continuation:"
        )
        assert not provider._pending_full_target_geometry_commit.raw_exact_eligible
        obj5 = provider._mask_only_publication_evidence(frame5, prepared5.mask)
        result5, obj5 = provider._apply_recovery_publication_gate(
            frame5, prepared5, obj5
        )
        assert result5.valid and obj5.valid
        assert provider._recovery_publication_hypothesis is None
        provider._commit_final_publication_histories(frame5, result5, obj5)
        assert provider._visible_exact_partial_continuity.frame_id == next_frame_id
        assert provider._visible_exact_partial_continuity.hits == 3
        assert provider._trusted_online_sam2_publication.frame_id == (
            trusted_before.frame_id
        )
        assert provider._full_target_geometry_authority.frame_id == (
            full_before.frame_id
        )
    finally:
        provider.stop()


def test_recovery_source_label_or_corrupt_token_cannot_seed_partial_owner(
    monkeypatch,
):
    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    frame = _moving_target_frame(4, (43, 30, 83, 70))
    mask = _large_mask((43, 30, 83, 70))
    provider._defer_full_target_geometry_commit(frame, mask)
    provider._mark_pending_full_target_geometry_ineligible(frame, "test recovery")
    pending = provider._pending_full_target_geometry_commit
    provider._last_output_mask_source = "recovery_committed"
    provider._last_online_sam2_exact_result = True
    try:
        # A source string is not the frame-scoped 3/3 recovery proof.
        assert not provider._commit_visible_exact_partial_continuity(
            frame, _mask_result(mask), pending
        )
        provider._recovery_visible_partial_handoff_frame_id = frame.frame_id
        original = copy.deepcopy(pending)
        corruptions = (
            ("mask_digest", "0" * 64),
            ("timestamp", float(frame.timestamp) + 0.01),
            ("frame_id", int(frame.frame_id) - 1),
            ("target_generation", int(pending.target_generation) + 1),
            ("source", "stale_online_sam2_video"),
        )
        for field, value in corruptions:
            setattr(pending, field, value)
            assert not provider._commit_visible_exact_partial_continuity(
                frame, _mask_result(mask), pending
            ), field
            setattr(pending, field, copy.deepcopy(getattr(original, field)))
        pending.mask = pending.mask.copy()
        pending.mask[30, 43] = 0
        assert not provider._commit_visible_exact_partial_continuity(
            frame, _mask_result(mask), pending
        )
        pending.mask = original.mask.copy()
        provider._last_online_sam2_exact_result = False
        assert not provider._commit_visible_exact_partial_continuity(
            frame, _mask_result(mask), pending
        )
        assert provider._visible_exact_partial_continuity is None
    finally:
        provider.stop()


def _bind_source140_like_external_occluder_case(provider, *, outside_bridge):
    """Bind exact 124 -> 70px source140-style target/occluder evidence.

    The visible target core is a concentric contraction of the confirmed
    predecessor.  In the positive variant the unsupported component which
    explains all 54 lost target pixels continues outside the predecessor and
    expands the exact SAM envelope to 949px.  The negative variant keeps the
    same unsupported 54px erosion wholly inside the predecessor.
    """

    _install_confirmed_sanitized_partial_owner(provider)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    previous = provider._visible_exact_partial_continuity

    aligned = np.zeros((120, 180), dtype=np.uint8)
    aligned[40:51, 80:91] = 1  # 121px body.
    aligned[44:47, 91] = 1  # 3px source-like visible shoulder.
    core = np.zeros_like(aligned)
    core[40:47, 80:90] = 1
    raw = aligned.copy()
    if outside_bridge:
        # The first rectangle overlaps 44 predecessor pixels, adding 781px;
        # the final row adds 44px: 124 + 781 + 44 == 949px.
        raw[47:72, 80:113] = 1
        raw[72, 80:124] = 1

    assert int(aligned.sum()) == 124
    assert int(core.sum()) == 70
    assert int(raw.sum()) == (949 if outside_bridge else 124)
    assert int(np.logical_and(aligned > 0, core == 0).sum()) == 54

    previous_center = provider._binary_mask_centroid(aligned)
    previous_bbox = np.asarray([80, 40, 92, 51], dtype=np.int32)
    assert previous_center is not None
    previous.mask = aligned.copy()
    previous.mask_digest = provider._publication_mask_digest(aligned)
    previous.core_bbox_xyxy = previous_bbox.copy()
    previous.core_center_xy = previous_center.copy()
    previous.core_area = 124.0
    previous.bbox_xyxy = previous_bbox.copy()
    previous.center_xy = previous_center.copy()
    previous.area = 124.0
    previous.bbox_area = float(12 * 11)
    previous.center_velocity_px_s = np.zeros(2, dtype=np.float64)
    previous.velocity_samples = 3
    previous.hits = 3
    previous.sanitized_anchor_bbox_xyxy = previous_bbox.copy()
    previous.sanitized_anchor_center_xy = previous_center.copy()
    previous.sanitized_anchor_area = 132.0

    frame = _large_frame(52)
    frame.color_bgr[:] = 45
    frame.color_bgr[core > 0] = (210, 80, 235)
    pending, core_token = _bind_current_sanitized_evidence(
        provider, frame, raw, core
    )
    return frame, previous, aligned, pending, core_token


def test_external_occluder_invasion_accepts_source140_like_visible_core(
    monkeypatch,
):
    """A confirmed 70/132 core may continue through an outside hand branch."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    (
        frame,
        previous,
        aligned,
        pending,
        core_token,
    ) = _bind_source140_like_external_occluder_case(
        provider, outside_bridge=True
    )
    monkeypatch.setattr(
        provider,
        "_aligned_committed_clean_mask",
        lambda *_args, **_kwargs: (
            aligned.copy(),
            "focused source140 exact predecessor alignment",
        ),
    )
    try:
        accepted, reason = provider._external_occluder_invasion_accepts(
            frame, core_token, pending.mask, aligned, previous
        )
        assert accepted, reason
        assert "external low-appearance occluder invasion" in reason
        assert "core/anchor=70/132px (0.530)" in reason
        assert "loss raw/foreign/bridge=1.000/1.000/1.000" in reason

        accepted, reason = provider._sanitized_partial_chain_accepts(
            frame,
            core_token,
            pending.mask,
            previous,
            allow_external_occluder_contraction=True,
        )
        assert accepted, reason
        assert "external low-appearance occluder invasion" in reason
    finally:
        provider.stop()


def test_external_occluder_invasion_rejects_internal_concentric_erosion(
    monkeypatch,
):
    """An unsupported internal annulus has no outside occluder bridge."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    (
        frame,
        previous,
        aligned,
        pending,
        core_token,
    ) = _bind_source140_like_external_occluder_case(
        provider, outside_bridge=False
    )
    try:
        accepted, reason = provider._external_occluder_invasion_accepts(
            frame, core_token, pending.mask, aligned, previous
        )
        assert not accepted
        assert "external-occluder outside bridge=0/54px" in reason
    finally:
        provider.stop()


def test_external_occluder_invasion_rejects_core_below_fixed_anchor_floor(
    monkeypatch,
):
    """An outside bridge cannot make a sub-anchor sliver publishable."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    (
        frame,
        previous,
        aligned,
        pending,
        core_token,
    ) = _bind_source140_like_external_occluder_case(
        provider, outside_bridge=True
    )
    previous.sanitized_anchor_area = 500.0  # 70/500=.14 < configured .15.
    try:
        accepted, reason = provider._external_occluder_invasion_accepts(
            frame, core_token, pending.mask, aligned, previous
        )
        assert not accepted
        assert "external-occluder anchor fraction=0.140<0.150" == reason
    finally:
        provider.stop()


def test_external_occluder_invasion_requires_confirmed_sanitizer_owner(
    monkeypatch,
):
    """Two observations are insufficient for the narrow invasion exception."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    (
        frame,
        previous,
        aligned,
        pending,
        core_token,
    ) = _bind_source140_like_external_occluder_case(
        provider, outside_bridge=True
    )
    previous.hits = 2
    try:
        accepted, reason = provider._external_occluder_invasion_accepts(
            frame, core_token, pending.mask, aligned, previous
        )
        assert not accepted
        assert reason == "confirmed commissioned sanitizer owner is unavailable"
    finally:
        provider.stop()


def _tracking_contraction_case(
    monkeypatch,
    *,
    internal_erosion=False,
    arm_immediately=True,
    hand_dominated_raw=False,
):
    """Create one fixed-anchor owner and arm source147-like contraction."""

    broad = _stage_broad_visible_core_case(monkeypatch)
    provider = broad.provider
    provider._last_output_mask_source = "publication_sanitized_target_core"
    provider._last_publication_guard_source = "online_sam2_guarded_primary"
    provider._last_publication_guard_status = "accepted focused broad owner"
    provider._last_online_sam2_status = (
        "guarded_primary_visible_partial_sanitized_continuation: focused"
    )
    provider._last_online_sam2_exact_result = True
    broad_obj = provider._mask_only_publication_evidence(
        broad.frame, broad.staged.mask
    )
    provider._commit_final_publication_histories(
        broad.frame, broad.staged, broad_obj
    )
    previous = provider._visible_exact_partial_continuity
    assert previous is not None
    assert previous.broad_visible_anchor_mask is not None
    trusted = provider._trusted_online_sam2_publication
    if provider._history_mask_evidence(trusted) is None:
        trusted_frame = _large_frame(int(trusted.frame_id))
        trusted_token, trusted_reason = provider._register_frame_mask_evidence(
            trusted_frame,
            trusted.mask,
            evidence_kind="raw_exact",
            authority_eligible=True,
        )
        assert trusted_token is not None, trusted_reason
        trusted.mask = trusted_token.mask
        trusted.mask_evidence = trusted_token
    assert provider._history_mask_evidence(
        provider._last_final_clean_publication
    ) is not None
    assert provider._history_mask_evidence(
        provider._trusted_online_sam2_publication
    ) is not None
    assert provider._full_target_geometry_authority is not None
    assert provider._history_mask_evidence(previous) is not None
    # The narrow external-occluder proof uses the same fixed episode anchor.
    previous.sanitized_anchor_area = float(previous.broad_visible_anchor_area)

    aligned = np.asarray(previous.mask, dtype=np.uint8).copy()
    first_core = np.zeros_like(aligned)
    first_core[39:49, 60:100] = 1
    first_raw = aligned.copy()
    if not internal_erosion:
        if hand_dominated_raw:
            # A large low-target-appearance hand union leaves the target core
            # below the ordinary core/raw floor while preserving an interior,
            # externally bridged exact SAM envelope.
            first_raw[17:100, 17:140] = 1
        else:
            first_raw[61:75, 60:112] = 1
    probability = np.full(aligned.shape, 0.05, dtype=np.float32)
    probability[first_core > 0] = 0.95
    _install_broad_visible_core_probability_map(broad.tracker, probability)
    first_frame = _large_frame(53)
    first_frame.color_bgr[:] = 45
    first_frame.color_bgr[first_core > 0] = (210, 80, 235)
    first_raw_evidence = provider._defer_full_target_geometry_commit(
        first_frame, first_raw
    )
    first_core_evidence, reason = provider._register_frame_mask_evidence(
        first_frame,
        first_core,
        evidence_kind="sanitized_target_core",
        authority_eligible=False,
    )
    assert first_core_evidence is not None, reason
    original_alignment = provider._aligned_committed_clean_mask
    monkeypatch.setattr(
        provider,
        "_aligned_committed_clean_mask",
        lambda frame, *_args, **_kwargs: (
            (aligned.copy(), "focused fixed predecessor")
            if int(frame.frame_id) in (53, 54)
            else original_alignment(frame, *_args, **_kwargs)
        ),
    )
    first_core_result = _exact_token_mask_result(
        first_core_evidence.mask, message="first exact contraction core"
    )
    if arm_immediately:
        armed, arm_reason = provider._arm_tracking_contraction_visible_core(
            first_frame,
            first_raw_evidence,
            first_core_result,
            previous,
        )
    else:
        armed, arm_reason = False, "not armed by fixture"
    return SimpleNamespace(
        provider=provider,
        tracker=broad.tracker,
        previous=previous,
        aligned=aligned,
        first_frame=first_frame,
        first_core=first_core_evidence.mask,
        first_core_result=first_core_result,
        first_raw=first_raw_evidence.mask,
        first_raw_evidence=first_raw_evidence,
        armed=armed,
        arm_reason=arm_reason,
    )


def test_tracking_contraction_first_frame_is_empty_and_preserves_owners(monkeypatch):
    case = _tracking_contraction_case(monkeypatch)
    provider = case.provider
    try:
        assert case.armed, case.arm_reason
        owners = (
            provider._last_final_clean_publication,
            provider._full_target_geometry_authority,
            provider._trusted_online_sam2_publication,
            provider._visible_exact_partial_continuity,
        )
        invalid = provider._invalid_public_mask(
            case.first_frame,
            _mask_result(case.first_raw),
            "first contraction frame is fail closed",
        )
        obj = provider._mask_only_publication_evidence(
            case.first_frame, invalid.mask
        )
        published, published_obj = provider._apply_recovery_publication_gate(
            case.first_frame, invalid, obj, mask_only=True
        )
        assert not published.valid and not published_obj.valid
        assert int(np.count_nonzero(published.mask)) == 0
        assert not provider.tracking_committed
        assert provider._tracking_contraction_visible_core_seed is not None
        assert owners == (
            provider._last_final_clean_publication,
            provider._full_target_geometry_authority,
            provider._trusted_online_sam2_publication,
            provider._visible_exact_partial_continuity,
        )
    finally:
        provider.stop()


def test_tracking_contraction_internal_erosion_cannot_arm(monkeypatch):
    case = _tracking_contraction_case(monkeypatch, internal_erosion=True)
    try:
        assert not case.armed
        assert "external occluder rejected" in case.arm_reason
        assert case.provider._tracking_contraction_visible_core_seed is None
        assert case.provider.tracking_committed
    finally:
        case.provider.stop()


def test_tracking_contraction_external_hand_union_can_be_below_core_raw_floor(
    monkeypatch,
):
    case = _tracking_contraction_case(monkeypatch, hand_dominated_raw=True)
    provider = case.provider
    try:
        assert case.armed, case.arm_reason
        seed = provider._tracking_contraction_visible_core_seed
        assert seed is not None
        floor = provider._tracking_contraction_visible_core_limits()[1]
        assert seed.core_pixels / int(np.count_nonzero(seed.raw_mask)) < floor
        assert seed.initial_external_occluder_proven

        # The distinct frame may remain hand-dominated.  It is accepted only
        # through the sealed first-frame external proof plus pair/fixed-anchor
        # checks, never by treating the hand union as target scale.
        second_core = np.zeros_like(case.first_core)
        second_core[39:48, 60:100] = 1
        probability = np.full(second_core.shape, 0.05, dtype=np.float32)
        probability[second_core > 0] = 0.95
        _install_broad_visible_core_probability_map(case.tracker, probability)
        second_frame = _large_frame(54)
        second_frame.color_bgr[:] = 45
        second_frame.color_bgr[second_core > 0] = (210, 80, 235)
        second_raw = np.zeros_like(second_core)
        second_raw[17:100, 17:140] = 1
        second_evidence = provider._defer_full_target_geometry_commit(
            second_frame, second_raw
        )
        second_result = _exact_token_mask_result(
            second_evidence.mask,
            message="second hand-dominated exact envelope",
        )
        second_result.source = "online_sam2_video"
        staged, reason = provider._stage_tracking_contraction_visible_core(
            second_frame, second_result, second_evidence
        )
        assert staged is not None, reason
        np.testing.assert_array_equal(staged.mask, second_core)
        assert int(np.count_nonzero(staged.mask)) / int(
            np.count_nonzero(second_evidence.mask)
        ) < floor
    finally:
        provider.stop()


def test_tracking_contraction_accepts_only_fixed_inherited_sanitizer_raw_owner(
    monkeypatch,
):
    case = _tracking_contraction_case(monkeypatch, arm_immediately=False)
    provider = case.provider
    try:
        previous = case.previous
        assert previous.sanitized_anchor_mask is not None
        assert previous.sanitized_anchor_phase == "fixed"
        previous.evidence_kind = "raw_exact"
        armed, reason = provider._arm_tracking_contraction_visible_core(
            case.first_frame,
            case.first_raw_evidence,
            case.first_core_result,
            previous,
        )
        assert armed, reason
        assert provider._tracking_contraction_visible_core_seed is not None
    finally:
        provider.stop()


def test_tracking_contraction_rejects_unfixed_raw_owner_lineage(monkeypatch):
    case = _tracking_contraction_case(monkeypatch, arm_immediately=False)
    provider = case.provider
    try:
        previous = case.previous
        previous.evidence_kind = "raw_exact"
        previous.sanitized_anchor_phase = "armed"
        armed, reason = provider._arm_tracking_contraction_visible_core(
            case.first_frame,
            case.first_raw_evidence,
            case.first_core_result,
            previous,
        )
        assert not armed
        assert "sanitizer owner is unavailable" in reason
        assert provider._tracking_contraction_visible_core_seed is None
        assert provider.tracking_committed
    finally:
        provider.stop()


def test_tracking_contraction_reuses_fixed_recovery_sanitizer_anchor(monkeypatch):
    case = _tracking_contraction_case(monkeypatch, arm_immediately=False)
    provider = case.provider
    try:
        previous = case.previous
        previous.evidence_kind = "recovery_sanitized_target_core"
        previous.hits = 2
        for name in (
            "broad_visible_anchor_mask",
            "broad_visible_anchor_digest",
            "broad_visible_anchor_bbox_xyxy",
            "broad_visible_anchor_center_xy",
            "broad_visible_anchor_area",
            "broad_visible_anchor_frame_id",
        ):
            setattr(previous, name, None)
        assert previous.sanitized_anchor_phase == "fixed"
        armed, reason = provider._arm_tracking_contraction_visible_core(
            case.first_frame,
            case.first_raw_evidence,
            case.first_core_result,
            previous,
        )
        assert armed, reason
        seed = provider._tracking_contraction_visible_core_seed
        assert seed is not None
        assert seed.fixed_anchor_frame_id == previous.sanitized_anchor_frame_id
        np.testing.assert_array_equal(
            seed.fixed_anchor_mask, previous.sanitized_anchor_mask
        )
    finally:
        provider.stop()


def test_tracking_contraction_equal_area_relocation_is_not_erosion(monkeypatch):
    """A moving occluder may expose a different equal-area target fragment."""

    case = _tracking_contraction_case(monkeypatch)
    provider = case.provider
    try:
        assert case.armed, case.arm_reason
        second_core = np.zeros_like(case.first_core)
        # The sanitizer's fixed predecessor clips one edge, leaving a 400px
        # square-like core with only 50% best overlap against the horizontal
        # 400px seed.
        second_core[34:59, 60:80] = 1
        probability = np.full(second_core.shape, 0.05, dtype=np.float32)
        probability[second_core > 0] = 0.95
        _install_broad_visible_core_probability_map(case.tracker, probability)
        second_frame = _large_frame(54)
        second_frame.color_bgr[:] = 45
        second_frame.color_bgr[second_core > 0] = (210, 80, 235)
        seed = provider._tracking_contraction_visible_core_seed
        assert seed is not None
        x1, y1, x2, y2 = (int(v) for v in seed.raw_bbox_xyxy)
        second_raw = second_core.copy()
        second_raw[y1, x1:x2] = 1
        second_raw[y2 - 1, x1:x2] = 1
        second_raw[y1:y2, x1] = 1
        second_raw[y1:y2, x2 - 1] = 1
        evidence = provider._defer_full_target_geometry_commit(
            second_frame, second_raw
        )
        result = _exact_token_mask_result(
            evidence.mask, message="equal-area relocated visible target core"
        )
        result.source = "online_sam2_video"
        staged, reason = provider._stage_tracking_contraction_visible_core(
            second_frame, result, evidence
        )
        assert staged is not None, reason
        proof = provider._pending_full_target_geometry_commit
        assert proof is not None
        proof = proof.tracking_contraction_visible_core_proof
        assert proof is not None and proof.relocated_equal_area_pair
        retention = np.asarray(
            [proof.pair_retention_bits], dtype=np.uint64
        ).view(np.float64)[0]
        assert retention < provider._tracking_contraction_visible_core_limits()[3]
        assert int(np.count_nonzero(staged.mask)) == case.first_core.sum()
        obj = provider._mask_only_publication_evidence(
            second_frame, staged.mask, preserve_sanitized_core=True
        )
        published, published_obj = provider._apply_recovery_publication_gate(
            second_frame, staged, obj, mask_only=True
        )
        assert published.valid and published_obj.valid
        provider._commit_final_publication_histories(
            second_frame, published, published_obj
        )
        owner = provider._visible_exact_partial_continuity
        assert owner is not None
        assert owner.frame_id == second_frame.frame_id
        assert owner.evidence_kind == "tracking_contraction_visible_core"
        assert not owner.sanitized_handoff_eligible
    finally:
        provider.stop()


def test_tracking_contraction_empty_frame_clears_seed(monkeypatch):
    case = _tracking_contraction_case(monkeypatch)
    provider = case.provider
    try:
        assert case.armed
        empty_frame = _large_frame(54)
        empty = np.zeros_like(case.first_core)
        result = provider._prepare_guarded_sam2_primary_mask(
            empty_frame,
            _mask_result(empty, message="complete absent source245"),
        )
        assert not result.valid
        assert provider._tracking_contraction_visible_core_seed is None
    finally:
        provider.stop()


def test_tracking_contraction_seed_tamper_fails_before_second_frame(monkeypatch):
    case = _tracking_contraction_case(monkeypatch)
    provider = case.provider
    try:
        seed = provider._tracking_contraction_visible_core_seed
        assert seed is not None
        seed.core_pixels += 1
        second_frame = _large_frame(54)
        raw_evidence = provider._defer_full_target_geometry_commit(
            second_frame, case.first_raw
        )
        raw_result = _exact_token_mask_result(
            raw_evidence.mask, message="tampered-seed second frame"
        )
        raw_result.source = "online_sam2_video"
        staged, reason = provider._stage_tracking_contraction_visible_core(
            second_frame, raw_result, raw_evidence
        )
        assert staged is None
        assert "authority changed" in reason
        assert provider._tracking_contraction_visible_core_seed is None
    finally:
        provider.stop()


def test_tracking_contraction_source137_to138_reuses_sealed_first_external_proof(
    monkeypatch,
):
    """The second stable core must not re-explain old loss against predecessor."""

    case = _tracking_contraction_case(monkeypatch)
    provider = case.provider
    try:
        assert case.armed, case.arm_reason

        def forbidden_repeat(*_args, **_kwargs):
            raise AssertionError(
                "source138 repeated source137's predecessor-loss proof"
            )

        # Source137 already crossed the strict raw/foreign/bridge proof while
        # arming.  Source138 owns a new exact pair/fixed-anchor/depth/appearance
        # proof, not a second interpretation of source137's historical loss.
        monkeypatch.setattr(
            provider, "_external_occluder_invasion_accepts", forbidden_repeat
        )
        second_core = np.zeros_like(case.first_core)
        second_core[39:48, 60:100] = 1  # 360/400=.90 retained pair.
        probability = np.full(second_core.shape, 0.05, dtype=np.float32)
        probability[second_core > 0] = 0.95
        _install_broad_visible_core_probability_map(case.tracker, probability)
        second_frame = _large_frame(54)
        second_frame.color_bgr[:] = 45
        second_frame.color_bgr[second_core > 0] = (210, 80, 235)
        seed = provider._tracking_contraction_visible_core_seed
        assert seed is not None
        x1, y1, x2, y2 = (int(v) for v in seed.raw_bbox_xyxy)
        second_raw = second_core.copy()
        # Preserve the reviewed raw-envelope bbox without making the synthetic
        # hand union so dense that the independent core/raw identity floor,
        # rather than pair retention, becomes the deciding gate.
        second_raw[y1, x1:x2] = 1
        second_raw[y2 - 1, x1:x2] = 1
        second_raw[y1:y2, x1] = 1
        second_raw[y1:y2, x2 - 1] = 1
        raw_evidence = provider._defer_full_target_geometry_commit(
            second_frame, second_raw
        )
        raw_result = _exact_token_mask_result(
            raw_evidence.mask, message="source138 stable exact contraction raw"
        )
        raw_result.source = "online_sam2_video"
        staged, reason = provider._stage_tracking_contraction_visible_core(
            second_frame, raw_result, raw_evidence
        )
        assert staged is not None, reason
        np.testing.assert_array_equal(staged.mask, second_core)

        obj = provider._mask_only_publication_evidence(
            second_frame, staged.mask, preserve_sanitized_core=True
        )
        published, published_obj = provider._apply_recovery_publication_gate(
            second_frame, staged, obj, mask_only=True
        )
        assert published.valid and published_obj.valid
        provider._commit_final_publication_histories(
            second_frame, published, published_obj
        )
        owner = provider._visible_exact_partial_continuity
        assert owner is not None and owner.frame_id == second_frame.frame_id
        np.testing.assert_array_equal(owner.mask, second_core)
    finally:
        provider.stop()


@pytest.mark.parametrize(
    ("retained_rows", "accepted"),
    ((7, True), (6, False)),
)
def test_tracking_contraction_reviewed_external_pair_floor(
    monkeypatch, retained_rows, accepted
):
    """A sealed hand contraction admits 70%, but not a 60% eroding pair."""

    case = _tracking_contraction_case(monkeypatch)
    provider = case.provider
    try:
        assert case.armed, case.arm_reason
        provider.cfg["tracker"]["tracking_contraction_min_pair_retention"] = 0.70
        second_core = np.zeros_like(case.first_core)
        second_core[39 : 39 + retained_rows, 60:100] = 1
        probability = np.full(second_core.shape, 0.05, dtype=np.float32)
        probability[second_core > 0] = 0.95
        _install_broad_visible_core_probability_map(case.tracker, probability)
        second_frame = _large_frame(54)
        second_frame.color_bgr[:] = 45
        second_frame.color_bgr[second_core > 0] = (210, 80, 235)
        seed = provider._tracking_contraction_visible_core_seed
        assert seed is not None
        x1, y1, x2, y2 = (int(v) for v in seed.raw_bbox_xyxy)
        second_raw = second_core.copy()
        second_raw[y1, x1:x2] = 1
        second_raw[y2 - 1, x1:x2] = 1
        second_raw[y1:y2, x1] = 1
        second_raw[y1:y2, x2 - 1] = 1
        raw_evidence = provider._defer_full_target_geometry_commit(
            second_frame, second_raw
        )
        raw_result = _exact_token_mask_result(
            raw_evidence.mask, message="reviewed moving-hand contraction pair"
        )
        raw_result.source = "online_sam2_video"
        staged, reason = provider._stage_tracking_contraction_visible_core(
            second_frame, raw_result, raw_evidence
        )
        assert (staged is not None) is accepted, reason
        if accepted:
            np.testing.assert_array_equal(staged.mask, second_core)
        else:
            assert "retention=0.600/0.700" in reason
            assert provider._tracking_contraction_visible_core_seed is None
    finally:
        provider.stop()


def test_safe_fixed_anchor_broad_precedes_tracking_contraction_arm(monkeypatch):
    """A complete broad proof must win before the fail-closed two-frame arm."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    frame, previous, _aligned, pending, strict_token = (
        _bind_source140_like_external_occluder_case(
            provider, outside_bridge=True
        )
    )
    raw_result = _exact_token_mask_result(
        pending.mask, message="source135 exact raw"
    )
    raw_result.source = "online_sam2_video"
    strict_result = _exact_token_mask_result(
        strict_token, message="source135 strict identity witness"
    )
    provider._last_visible_exact_partial_failure_kind = "appearance_only"
    provider._last_online_sam2_exact_result = True
    provider._online_sam2_initialized = True
    provider._online_sam2_seed_pending = False
    provider._online_sam2_failure_reported = False
    provider._online_sam2_recovery_hypothesis = None
    provider._recovery_publication_hypothesis = None
    monkeypatch.setattr(
        provider,
        "_sanitize_publication_target_core",
        lambda *_args, **_kwargs: (strict_result, "strict current core"),
    )
    monkeypatch.setattr(
        provider,
        "_online_sam2_independent_identity_accepts",
        lambda *_args, **_kwargs: (True, "strict identity accepted"),
    )
    monkeypatch.setattr(
        provider,
        "_stage_broad_visible_core",
        lambda *_args, **_kwargs: (strict_result, "complete fixed-anchor proof"),
    )

    def forbidden_arm(*_args, **_kwargs):
        raise AssertionError("contraction armed before a valid broad proof")

    monkeypatch.setattr(
        provider, "_arm_tracking_contraction_visible_core", forbidden_arm
    )
    try:
        repaired, reason = (
            provider._guarded_visible_partial_sanitized_continuation(
                frame, raw_result, previous
            )
        )
        assert repaired is strict_result
        assert "fixed-anchor broad core" in reason
        assert provider.tracking_committed
        assert provider._tracking_contraction_visible_core_seed is None

        monkeypatch.setattr(
            provider,
            "_stage_broad_visible_core",
            lambda *_args, **_kwargs: (None, "fixed-anchor proof rejected"),
        )
        monkeypatch.setattr(
            provider,
            "_arm_tracking_contraction_visible_core",
            lambda *_args, **_kwargs: (True, "two-frame contraction seeded"),
        )
        repaired, reason = (
            provider._guarded_visible_partial_sanitized_continuation(
                frame, raw_result, previous
            )
        )
        assert repaired is None
        assert reason == (
            "tracking contraction armed: two-frame contraction seeded"
        )
    finally:
        provider.stop()


def test_structured_hard_shrink_can_only_arm_two_frame_contraction(monkeypatch):
    """A hard raw envelope may arm, but never publish, one strict core."""

    case = _tracking_contraction_case(monkeypatch, arm_immediately=False)
    provider = case.provider
    raw_result = _exact_token_mask_result(
        case.first_raw_evidence.mask,
        message="hard hand-union raw exact envelope",
    )
    raw_result.source = "online_sam2_video"
    try:
        provider._last_online_sam2_exact_result = True
        provider._online_sam2_initialized = True
        provider._online_sam2_seed_pending = False
        provider._online_sam2_failure_reported = False
        provider._online_sam2_recovery_hypothesis = None
        provider._recovery_publication_hypothesis = None
        raw_geometry, geometry_reason = provider._online_mask_geometry(
            case.first_frame, case.first_raw_evidence.mask
        )
        assert raw_geometry is not None, geometry_reason
        provider._visible_exact_partial_candidate_agrees(
            case.first_frame, raw_geometry
        )
        report = provider._last_visible_exact_partial_gate_report
        assert report is not None
        report = replace(
            report,
            accepted=False,
            failure_kind="hard_reject",
            violations=("log_bbox_rate",),
        )
        monkeypatch.setattr(
            provider,
            "_sanitize_publication_target_core",
            lambda *_args, **_kwargs: (
                case.first_core_result,
                "strict current target core",
            ),
        )
        monkeypatch.setattr(
            provider,
            "_online_sam2_independent_identity_accepts",
            lambda *_args, **_kwargs: (True, "strict identity accepted"),
        )
        monkeypatch.setattr(
            provider,
            "_stage_broad_visible_core",
            lambda *_args, **_kwargs: (None, "fixed-anchor broad rejected"),
        )

        repaired, reason = (
            provider._guarded_visible_partial_sanitized_continuation(
                case.first_frame,
                raw_result,
                case.previous,
                raw_failure_kind="hard_reject",
                raw_failure_report=report,
            )
        )

        assert repaired is None
        assert "armed two-frame tracking contraction" in reason
        assert provider._tracking_contraction_visible_core_seed is not None
        assert not provider.tracking_committed
        # The first exact frame is evidence only.  No publication authority
        # or old pixels may be returned before a distinct second frame.
        assert provider._visible_exact_partial_continuity is case.previous
        assert provider._last_final_clean_publication is not None
    finally:
        provider.stop()


def test_non_pixel_hard_failure_cannot_reach_contraction_arm(monkeypatch):
    case = _tracking_contraction_case(monkeypatch, arm_immediately=False)
    provider = case.provider
    raw_result = _exact_token_mask_result(
        case.first_raw_evidence.mask, message="non-pixel hard rejection"
    )
    raw_result.source = "online_sam2_video"
    try:
        raw_geometry, geometry_reason = provider._online_mask_geometry(
            case.first_frame, case.first_raw_evidence.mask
        )
        assert raw_geometry is not None, geometry_reason
        provider._visible_exact_partial_candidate_agrees(
            case.first_frame, raw_geometry
        )
        report = replace(
            provider._last_visible_exact_partial_gate_report,
            accepted=False,
            failure_kind="hard_reject",
            violations=("depth",),
        )

        def forbidden_arm(*_args, **_kwargs):
            raise AssertionError("non-pixel hard failure reached contraction arm")

        monkeypatch.setattr(
            provider, "_arm_tracking_contraction_visible_core", forbidden_arm
        )
        repaired, reason = (
            provider._guarded_visible_partial_sanitized_continuation(
                case.first_frame,
                raw_result,
                case.previous,
                raw_failure_kind="hard_reject",
                raw_failure_report=report,
            )
        )
        assert repaired is None
        assert "not repairable" in reason
        assert provider._tracking_contraction_visible_core_seed is None
        assert provider.tracking_committed
    finally:
        provider.stop()


def test_sanitized_fallback_uses_frozen_raw_failure_after_core_gate_mutation(
    monkeypatch,
):
    """A failed core counterfactual must not erase a raw appearance failure."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    frame, previous, _aligned, pending, strict_token = (
        _bind_source140_like_external_occluder_case(
            provider, outside_bridge=True
        )
    )
    raw_result = _exact_token_mask_result(
        pending.mask, message="source135 exact raw"
    )
    raw_result.source = "online_sam2_video"
    strict_result = _exact_token_mask_result(
        strict_token, message="source135 strict identity witness"
    )
    provider._last_online_sam2_exact_result = True
    provider._online_sam2_initialized = True
    provider._online_sam2_seed_pending = False
    provider._online_sam2_failure_reported = False
    provider._online_sam2_recovery_hypothesis = None
    provider._recovery_publication_hypothesis = None
    monkeypatch.setattr(
        provider,
        "_sanitize_publication_target_core",
        lambda *_args, **_kwargs: (strict_result, "strict current core"),
    )
    monkeypatch.setattr(
        provider,
        "_online_sam2_independent_identity_accepts",
        lambda *_args, **_kwargs: (True, "strict identity accepted"),
    )
    monkeypatch.setattr(
        provider,
        "_stage_broad_visible_core",
        lambda *_args, **_kwargs: (strict_result, "complete fixed-anchor proof"),
    )
    try:
        # This models the source-decomposition core recheck overwriting the
        # legacy diagnostic cache after the immutable raw report recorded an
        # appearance-only failure.
        provider._last_visible_exact_partial_failure_kind = "hard_reject"
        rejected, rejected_reason = (
            provider._guarded_visible_partial_sanitized_continuation(
                frame, raw_result, previous
            )
        )
        assert rejected is None
        assert "hard_reject" in rejected_reason

        repaired, reason = (
            provider._guarded_visible_partial_sanitized_continuation(
                frame,
                raw_result,
                previous,
                raw_failure_kind="appearance_only",
            )
        )
        assert repaired is strict_result
        assert "repaired appearance_only with fixed-anchor broad core" in reason
    finally:
        provider.stop()


def _owner_aligned_external_projection_case(
    monkeypatch,
    *,
    outside_bridge=True,
    zero_default_depth=True,
    boundary=False,
):
    """Build a source224-like 20px default and 70px owner-aligned strict core."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    frame, previous, aligned, pending, expected_core = (
        _bind_source140_like_external_occluder_case(
            provider, outside_bridge=outside_bridge
        )
    )
    default_core = np.zeros_like(expected_core)
    default_core[40:42, 80:90] = 1
    if zero_default_depth:
        frame.depth_raw[default_core > 0] = 0
    default_token, token_reason = provider._register_frame_mask_evidence(
        frame,
        default_core,
        evidence_kind="sanitized_target_core",
        authority_eligible=False,
    )
    assert default_token is not None, token_reason
    if boundary:
        provider.cfg["tracker"]["publication_boundary_exit_margin_px"] = 100
    monkeypatch.setattr(
        provider,
        "_aligned_committed_clean_mask",
        lambda *_args, **_kwargs: (
            aligned.copy(),
            "focused partial-owner velocity alignment",
        ),
    )
    raw_evidence = provider._current_raw_evidence_matches_pending(
        frame, pending.mask, pending
    )
    assert raw_evidence is not None
    candidate, reason, default_unprojectable = (
        provider._stage_owner_aligned_external_visible_core(
            frame,
            raw_evidence,
            _exact_token_mask_result(
                default_token.mask, message="generic final-clean strict core"
            ),
            previous,
        )
    )
    return SimpleNamespace(
        provider=provider,
        frame=frame,
        previous=previous,
        pending=pending,
        expected_core=expected_core,
        default_core=default_token.mask,
        candidate=candidate,
        reason=reason,
        default_unprojectable=default_unprojectable,
    )


def test_owner_aligned_external_core_rescues_depth_and_commits_atomically(
    monkeypatch,
):
    """The larger strict core remains non-authoritative and advances its owner."""

    case = _owner_aligned_external_projection_case(monkeypatch)
    provider = case.provider
    try:
        assert case.default_unprojectable
        assert case.candidate is not None, case.reason
        assert np.count_nonzero(case.default_core) == 20
        assert np.count_nonzero(case.candidate.mask) == 70
        assert case.pending.eligibility_kind == "sanitizer_external_visible_core"
        assert case.pending.sanitizer_external_occluder_proof_required
        assert case.pending.sanitizer_external_occluder_proof is not None
        seal = case.pending.sanitizer_external_occluder_stage_seal
        assert seal is not None and seal.staged and seal.required

        full_before = copy.deepcopy(provider._full_target_geometry_authority)
        trusted_before = provider._trusted_online_sam2_publication
        provider._last_output_mask_source = "publication_sanitized_target_core"
        provider._last_online_sam2_status = (
            "guarded_primary_visible_partial_sanitized_continuation: "
            "source224 owner-aligned strict external core"
        )
        provider._last_online_sam2_exact_result = True
        obj = provider._mask_only_publication_evidence(
            case.frame,
            case.candidate.mask,
            preserve_sanitized_core=True,
        )
        assert obj.valid
        published, published_obj = provider._apply_recovery_publication_gate(
            case.frame, case.candidate, obj, mask_only=True
        )
        assert published.valid and published_obj.valid
        provider._commit_final_publication_histories(
            case.frame, published, published_obj
        )
        owner = provider._visible_exact_partial_continuity
        assert owner is not None and owner is not case.previous
        assert owner.frame_id == case.frame.frame_id
        assert owner.hits == case.previous.hits + 1
        np.testing.assert_array_equal(owner.mask, case.expected_core)
        assert (
            provider._full_target_geometry_authority.frame_id
            == full_before.frame_id
        )
        assert provider._trusted_online_sam2_publication is trusted_before
        assert provider.tracking_committed
        assert (
            provider._sanitizer_external_occluder_registry_seal(case.frame)
            is None
        )
    finally:
        provider.stop()


def test_prepared_sanitized_core_uses_preserving_first_extraction(monkeypatch):
    """Mask-only and full packet entry points must not re-erode a 2-D core."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    frame = _large_frame(52)
    raw = np.zeros((120, 180), dtype=np.uint8)
    raw[38:46, 78:114] = 1
    fragmented = np.zeros_like(raw)
    fragmented[40:42, 80:111] = 1  # 62 px; erode3 clears it.
    raw_evidence = provider._defer_full_target_geometry_commit(frame, raw)
    core_evidence, core_reason = provider._register_frame_mask_evidence(
        frame,
        fragmented,
        evidence_kind="sanitized_target_core",
        authority_eligible=False,
    )
    assert raw_evidence is not None
    assert core_evidence is not None, core_reason
    provider._mark_pending_full_target_geometry_ineligible(
        frame, "prepared exact-frame sanitizer core"
    )
    provider._last_output_mask_source = "publication_sanitized_target_core"
    provider._last_online_sam2_status = (
        "guarded_primary_visible_partial_sanitized_continuation: "
        "prepared fragmented core"
    )
    result = _exact_token_mask_result(
        core_evidence.mask, message="prepared fragmented sanitizer core"
    )
    result.source = "publication_sanitized_target_core"
    tracked = SimpleNamespace(
        frame=frame,
        mask_result=result,
        camera_ms=0.0,
        tracker_ms=0.0,
        online_sam2_ms=0.0,
        sam2_reinit_ms=0.0,
    )
    try:
        assert provider._current_mask_is_pre_sanitized_core(
            frame, core_evidence.mask
        )
        ordinary = provider._mask_only_publication_evidence(
            frame, core_evidence.mask
        )
        preserved = provider._mask_only_publication_evidence(
            frame,
            core_evidence.mask,
            preserve_sanitized_core=True,
        )
        assert not ordinary.valid
        assert ordinary.message == "too few valid depth pixels after mask"
        assert preserved.valid
        assert preserved.message == "mask gate points=62"

        monkeypatch.setattr(
            provider,
            "_run_semantic_sam2_mask_pipeline",
            lambda *, guarded=False: tracked,
        )

        class FirstExtractionObserved(RuntimeError):
            pass

        mask_only_preserve = []

        def observe_mask_only(
            _frame, _mask, *, preserve_sanitized_core=False
        ):
            mask_only_preserve.append(bool(preserve_sanitized_core))
            raise FirstExtractionObserved("mask-only first extraction")

        monkeypatch.setattr(
            provider, "_mask_only_publication_evidence", observe_mask_only
        )
        with pytest.raises(FirstExtractionObserved, match="mask-only"):
            provider._step_mask_only_impl()
        assert mask_only_preserve == [True]

        full_preserve = []

        def observe_full(_frame, _mask, *, preserve_sanitized_core=False):
            full_preserve.append(bool(preserve_sanitized_core))
            raise FirstExtractionObserved("full first extraction")

        monkeypatch.setattr(provider, "_extract_publication_object", observe_full)
        with pytest.raises(FirstExtractionObserved, match="full"):
            provider._step_impl()
        assert full_preserve == [True]

        # Equal external bytes and the containing raw token have no sanitizer
        # provenance and therefore keep the ordinary morphology.
        assert not provider._current_mask_is_pre_sanitized_core(
            frame, np.asarray(core_evidence.mask).copy()
        )
        assert not provider._current_mask_is_pre_sanitized_core(
            frame, raw_evidence.mask
        )
    finally:
        provider.stop()


@pytest.mark.parametrize(
    "variant",
    ("default_projects", "internal_erosion", "boundary", "complete_absent"),
)
def test_owner_aligned_external_core_negative_paths_fail_closed(
    monkeypatch, variant
):
    """No fallback exists for healthy, absent, internal or boundary evidence."""

    case = _owner_aligned_external_projection_case(
        monkeypatch,
        outside_bridge=variant != "internal_erosion",
        zero_default_depth=variant
        not in ("default_projects", "complete_absent"),
        boundary=variant == "boundary",
    )
    provider = case.provider
    try:
        if variant == "complete_absent":
            raw_evidence = provider._current_raw_evidence_matches_pending(
                case.frame, case.pending.mask, case.pending
            )
            assert raw_evidence is not None
            empty = np.zeros_like(case.default_core)
            candidate, reason, attempted = (
                provider._stage_owner_aligned_external_visible_core(
                    case.frame,
                    raw_evidence,
                    _mask_result(empty, message="complete absent exact SAM"),
                    case.previous,
                )
            )
            assert candidate is None and not attempted
            assert "unbound" in reason
            return
        assert case.candidate is None
        if variant == "default_projects":
            assert not case.default_unprojectable
            assert case.reason == "default strict core already projects"
        elif variant == "internal_erosion":
            assert case.default_unprojectable
            assert "no mandatory external-occluder transaction" in case.reason
            assert case.pending.sanitizer_external_occluder_proof is None
        else:
            assert case.default_unprojectable
            assert "disabled at image boundaries" in case.reason
            seal = case.pending.sanitizer_external_occluder_stage_seal
            assert seal is not None and not seal.staged
    finally:
        provider.stop()


@pytest.mark.parametrize("tamper", ("proof_removed", "required_flip", "owner"))
def test_owner_aligned_external_guard_rejects_transaction_tamper(
    monkeypatch, tamper
):
    """The special final guard consumes the exact monotonic owner seal."""

    case = _owner_aligned_external_projection_case(monkeypatch)
    provider = case.provider
    try:
        assert case.candidate is not None, case.reason
        provider._last_output_mask_source = "publication_sanitized_target_core"
        provider._last_online_sam2_status = (
            "guarded_primary_visible_partial_sanitized_continuation: "
            "tampered source224 owner-aligned strict external core"
        )
        provider._last_online_sam2_exact_result = True
        if tamper == "proof_removed":
            case.pending.sanitizer_external_occluder_proof = None
        elif tamper == "required_flip":
            case.pending.sanitizer_external_occluder_proof_required = False
        else:
            case.previous.hits += 1
        obj = provider._mask_only_publication_evidence(
            case.frame,
            case.candidate.mask,
            preserve_sanitized_core=True,
        )
        published, published_obj = provider._apply_recovery_publication_gate(
            case.frame, case.candidate, obj, mask_only=True
        )
        assert not published.valid and not published_obj.valid
        assert int(np.count_nonzero(published.mask)) == 0
        assert provider._visible_exact_partial_continuity is None
    finally:
        provider.stop()


@pytest.mark.parametrize(
    "tamper",
    (
        "seal_only_removed",
        "all_pending_stage_fields_and_eligibility_cleared",
        "source_flipped",
    ),
)
def test_provider_external_stage_registry_blocks_post_apply_pending_bypass(
    monkeypatch, tamper
):
    """The provider latch survives every mutable pending-field rewrite."""

    case = _owner_aligned_external_projection_case(monkeypatch)
    provider = case.provider
    try:
        assert case.candidate is not None, case.reason
        provider._last_output_mask_source = "publication_sanitized_target_core"
        provider._last_online_sam2_status = (
            "guarded_primary_visible_partial_sanitized_continuation: "
            "provider-latched source224 external core"
        )
        provider._last_online_sam2_exact_result = True
        obj = provider._mask_only_publication_evidence(
            case.frame,
            case.candidate.mask,
            preserve_sanitized_core=True,
        )
        published, published_obj = provider._apply_recovery_publication_gate(
            case.frame, case.candidate, obj, mask_only=True
        )
        assert published.valid and published_obj.valid
        registry_seal = provider._sanitizer_external_occluder_registry_seal(
            case.frame
        )
        assert registry_seal is not None
        assert registry_seal is case.pending.sanitizer_external_occluder_stage_seal
        final_before = provider._last_final_clean_publication
        owner_before = provider._visible_exact_partial_continuity

        if tamper == "seal_only_removed":
            case.pending.sanitizer_external_occluder_stage_seal = None
        elif tamper == "all_pending_stage_fields_and_eligibility_cleared":
            case.pending.sanitizer_external_occluder_stage_seal = None
            case.pending.sanitizer_external_occluder_proof = None
            case.pending.sanitizer_external_occluder_proof_required = False
            case.pending.eligibility_kind = "known_partial"
        else:
            case.pending.source = "forged_online_sam2_video"

        with pytest.raises(RuntimeError, match="sealed sanitizer external proof"):
            provider._commit_final_publication_histories(
                case.frame, published, published_obj
            )
        assert provider._last_final_clean_publication is final_before
        assert provider._visible_exact_partial_continuity is owner_before
        assert provider._pending_full_target_geometry_commit is None
        assert (
            provider._sanitizer_external_occluder_registry_seal(case.frame)
            is None
        )
    finally:
        provider.stop()


def test_external_stage_snapshot_restore_drops_registry_and_fails_closed(
    monkeypatch,
):
    """A deep-copied pending seal cannot survive without its live registry."""

    case = _owner_aligned_external_projection_case(monkeypatch)
    provider = case.provider
    try:
        assert case.candidate is not None, case.reason
        snapshot = provider._guarded_v2_bootstrap_transaction_snapshot(
            force=True
        )
        assert snapshot is not None
        provider._restore_guarded_v2_bootstrap_transaction(snapshot)
        restored = provider._pending_full_target_geometry_commit
        assert restored is not None and restored is not case.pending
        assert (
            provider._sanitizer_external_occluder_registry_seal(case.frame)
            is None
        )
        accepted, reason = (
            provider._sanitizer_external_occluder_stage_seal_accepts(
                case.frame,
                restored.sanitizer_external_occluder_stage_seal.core_evidence.mask,
                restored,
                provider._visible_exact_partial_continuity,
            )
        )
        assert not accepted
        assert "provider latch disappeared" in reason
    finally:
        provider.stop()


@pytest.mark.parametrize(
    "tamper",
    ("initial_flag", "initial_metrics"),
)
def test_tracking_contraction_sealed_initial_external_proof_tamper_fails_closed(
    monkeypatch, tamper
):
    case = _tracking_contraction_case(monkeypatch)
    provider = case.provider
    try:
        assert case.armed, case.arm_reason
        seed = provider._tracking_contraction_visible_core_seed
        assert seed is not None
        if tamper == "initial_flag":
            seed.initial_external_occluder_proven = False
        else:
            object.__setattr__(
                seed.initial_external_occluder_proof,
                "bridged_loss_pixels",
                seed.initial_external_occluder_proof.bridged_loss_pixels - 1,
            )
        second_frame = _large_frame(54)
        raw_evidence = provider._defer_full_target_geometry_commit(
            second_frame, case.first_raw
        )
        raw_result = _exact_token_mask_result(
            raw_evidence.mask, message="tampered initial proof"
        )
        raw_result.source = "online_sam2_video"
        staged, reason = provider._stage_tracking_contraction_visible_core(
            second_frame, raw_result, raw_evidence
        )
        assert staged is None
        assert "authority changed" in reason
        assert provider._tracking_contraction_visible_core_seed is None
    finally:
        provider.stop()


def test_tracking_contraction_second_core_is_partial_only_then_needs_later_rgbd(
    monkeypatch,
):
    case = _tracking_contraction_case(monkeypatch)
    provider = case.provider
    try:
        assert case.armed, case.arm_reason
        owners_before = (
            provider._last_final_clean_publication,
            provider._full_target_geometry_authority,
            provider._trusted_online_sam2_publication,
        )
        anchor_before = case.previous.broad_visible_anchor_mask.copy()
        anchor_digest_before = case.previous.broad_visible_anchor_digest
        anchor_frame_before = case.previous.broad_visible_anchor_frame_id

        second_core = np.zeros_like(case.first_core)
        second_core[39:48, 60:100] = 1  # 360/400=.90 fresh pair.
        probability = np.full(second_core.shape, 0.05, dtype=np.float32)
        probability[second_core > 0] = 0.95
        _install_broad_visible_core_probability_map(case.tracker, probability)
        second_frame = _large_frame(54)
        second_frame.color_bgr[:] = 45
        second_frame.color_bgr[second_core > 0] = (210, 80, 235)
        second_raw_evidence = provider._defer_full_target_geometry_commit(
            second_frame, case.first_raw
        )
        second_raw_result = _exact_token_mask_result(
            second_raw_evidence.mask, message="second exact contraction raw"
        )
        second_raw_result.source = "online_sam2_video"
        staged, stage_reason = provider._stage_tracking_contraction_visible_core(
            second_frame, second_raw_result, second_raw_evidence
        )
        assert staged is not None, stage_reason
        np.testing.assert_array_equal(staged.mask, second_core)

        obj = provider._mask_only_publication_evidence(
            second_frame, staged.mask, preserve_sanitized_core=True
        )
        published, published_obj = provider._apply_recovery_publication_gate(
            second_frame, staged, obj, mask_only=True
        )
        assert published.valid and published_obj.valid
        provider._commit_final_publication_histories(
            second_frame, published, published_obj
        )
        owner = provider._visible_exact_partial_continuity
        assert owner.frame_id == second_frame.frame_id
        assert owner.hits == 2
        assert owner.evidence_kind == "tracking_contraction_visible_core"
        assert not owner.sanitized_handoff_eligible
        np.testing.assert_array_equal(owner.mask, second_core)
        np.testing.assert_array_equal(owner.broad_visible_anchor_mask, anchor_before)
        assert owner.broad_visible_anchor_digest == anchor_digest_before
        assert owner.broad_visible_anchor_frame_id == anchor_frame_before
        assert owners_before == (
            provider._last_final_clean_publication,
            provider._full_target_geometry_authority,
            provider._trusted_online_sam2_publication,
        )
        assert not provider.tracking_committed
        recovery = provider._recovery_publication_hypothesis
        assert recovery.hits == 2
        assert recovery.evidence_kind == "two_exact_sam_seed_with_rgbd"
        assert not provider._guarded_recovery_hypothesis_bound(
            second_frame, recovery, phase="later_rgbd"
        )
        assert provider._pending_recovery_publication_commit is None

        later_frame = _large_frame(55)
        later_evidence = provider._defer_full_target_geometry_commit(
            later_frame, second_core
        )
        later_result = _exact_token_mask_result(
            later_evidence.mask, message="distinct later RGB-D"
        )
        later_result.source = "online_sam2_video"
        later_obj = provider._mask_only_publication_evidence(
            later_frame, later_evidence.mask, preserve_sanitized_core=True
        )
        later_result, later_obj, recovered = provider._gate_recovery_publication(
            later_frame, later_result, later_obj
        )
        assert recovered
        assert later_result.valid and later_obj.valid
        assert provider._pending_recovery_publication_commit is not None
    finally:
        provider.stop()


def test_external_occluder_proof_is_revalidated_at_final_history_commit(
    monkeypatch,
):
    """Final history consumes current pixels, not an earlier proof result."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    (
        frame,
        previous,
        aligned,
        pending,
        core_token,
    ) = _bind_source140_like_external_occluder_case(
        provider, outside_bridge=True
    )
    provider._last_publication_guard_frame_id = frame.frame_id
    provider._last_publication_guard_aligned_clean = aligned.copy()
    provider._last_output_mask_source = "publication_sanitized_target_core"
    provider._last_online_sam2_status = (
        "guarded_primary_visible_partial_sanitized_continuation: "
        "source140-like external occluder"
    )
    provider._last_online_sam2_exact_result = True
    monkeypatch.setattr(
        provider,
        "_aligned_committed_clean_mask",
        lambda *_args, **_kwargs: (
            aligned.copy(),
            "focused source140 exact predecessor alignment",
        ),
    )
    try:
        accepted, reason = provider._external_occluder_invasion_accepts(
            frame, core_token, pending.mask, aligned, previous
        )
        assert accepted, reason

        # Replacing only the final guard with the raw envelope leaves the exact
        # current-frame raw/core tokens unchanged but removes every pixel which
        # is outside the aligned predecessor.  Final history must recompute and
        # reject rather than trusting the earlier successful helper call.
        provider._last_publication_guard_aligned_clean = pending.mask.copy()
        invalidated, invalidated_reason = (
            provider._external_occluder_invasion_accepts(
                frame,
                core_token,
                pending.mask,
                provider._last_publication_guard_aligned_clean,
                previous,
            )
        )
        assert not invalidated, invalidated_reason
        assert "external-occluder outside bridge=0/879px" in invalidated_reason
        assert not provider._commit_visible_exact_partial_continuity(
            frame,
            _exact_token_mask_result(
                core_token, message="external bridge disappeared before commit"
            ),
            pending,
        )
        assert provider._visible_exact_partial_continuity is previous

        provider._last_publication_guard_aligned_clean = aligned.copy()
        assert provider._commit_visible_exact_partial_continuity(
            frame,
            _exact_token_mask_result(
                core_token, message="external bridge revalidated at commit"
            ),
            pending,
        )
        owner = provider._visible_exact_partial_continuity
        assert owner.frame_id == frame.frame_id
        assert owner.hits == 4
        assert owner.evidence_kind == "sanitized_target_core"
        assert owner.sanitized_anchor_area == pytest.approx(132.0)
    finally:
        provider.stop()


def _prepared_sanitized_external_pipeline_case(
    monkeypatch,
    *,
    fail_second_proof_capture=False,
):
    """Run the real source224 prepare path, including its external proof."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(
        monkeypatch, [], {}, initial
    )
    frame, previous, aligned, pending, core_token = (
        _bind_source140_like_external_occluder_case(
            provider, outside_bridge=True
        )
    )
    raw_result = _exact_token_mask_result(
        pending.mask, message="source224 exact raw"
    )
    raw_result.source = "online_sam2_video"
    strict_result = _exact_token_mask_result(
        core_token, message="source224 strict external core"
    )
    strict_result.source = "publication_sanitized_target_core"
    provider._last_visible_exact_partial_failure_kind = "shape_only"
    provider._last_online_sam2_exact_result = True
    provider._online_sam2_initialized = True
    provider._online_sam2_seed_pending = False
    provider._online_sam2_failure_reported = False
    provider._online_sam2_recovery_hypothesis = None
    provider._recovery_publication_hypothesis = None
    monkeypatch.setattr(
        provider,
        "_sanitize_publication_target_core",
        lambda *_args, **_kwargs: (
            strict_result,
            "source224 strict sanitizer",
        ),
    )
    monkeypatch.setattr(
        provider,
        "_online_sam2_independent_identity_accepts",
        lambda *_args, **_kwargs: (True, "source224 identity accepted"),
    )
    monkeypatch.setattr(
        provider,
        "_stage_broad_visible_core",
        lambda *_args, **_kwargs: (None, "source224 broad core rejected"),
    )
    monkeypatch.setattr(
        provider,
        "_arm_tracking_contraction_visible_core",
        lambda *_args, **_kwargs: (False, "source224 contraction not armed"),
    )
    monkeypatch.setattr(
        provider,
        "_aligned_committed_clean_mask",
        lambda *_args, **_kwargs: (
            aligned.copy(),
            "source224 prepare-time predecessor alignment",
        ),
    )
    if fail_second_proof_capture:
        original_capture = provider._capture_external_occluder_invasion_proof
        capture_count = 0

        def fail_stage_capture(*args, **kwargs):
            nonlocal capture_count
            capture_count += 1
            if capture_count == 1:
                return original_capture(*args, **kwargs)
            return None, "forced second capture failure"

        monkeypatch.setattr(
            provider,
            "_capture_external_occluder_invasion_proof",
            fail_stage_capture,
        )
    repaired, reason = (
        provider._guarded_visible_partial_sanitized_continuation(
            frame, raw_result, previous
        )
    )
    return SimpleNamespace(
        provider=provider,
        frame=frame,
        previous=previous,
        aligned=aligned,
        pending=pending,
        core_token=core_token,
        repaired=repaired,
        reason=reason,
    )


def _apply_prepared_sanitized_external_case(case):
    provider = case.provider
    provider._last_output_mask_source = "publication_sanitized_target_core"
    provider._last_online_sam2_status = (
        "guarded_primary_visible_partial_sanitized_continuation: "
        "source224 prepared strict external core"
    )
    provider._last_online_sam2_exact_result = True
    obj = provider._mask_only_publication_evidence(
        case.frame,
        case.repaired.mask,
        preserve_sanitized_core=True,
    )
    assert obj.valid
    published, published_obj = provider._apply_recovery_publication_gate(
        case.frame, case.repaired, obj, mask_only=True
    )
    assert published.valid and published_obj.valid
    return published, published_obj


def test_real_prepare_path_seals_external_proof_and_advances_source224_owner(
    monkeypatch,
):
    """A valid prepared external core and its owner commit in one frame."""

    case = _prepared_sanitized_external_pipeline_case(monkeypatch)
    provider = case.provider
    try:
        assert case.repaired is not None, case.reason
        np.testing.assert_array_equal(case.repaired.mask, case.core_token)
        assert "external low-appearance occluder invasion" in case.reason
        assert case.pending.eligibility_kind == (
            "sanitizer_external_visible_core"
        )
        assert case.pending.sanitizer_external_occluder_proof_required
        assert case.pending.sanitizer_external_occluder_proof is not None
        seal = case.pending.sanitizer_external_occluder_stage_seal
        assert seal is not None and seal.required and seal.staged
        assert (
            provider._sanitizer_external_occluder_registry_seal(case.frame)
            is seal
        )

        published, published_obj = _apply_prepared_sanitized_external_case(
            case
        )
        # Recreate the source224 ordering hazard: the generic alignment is no
        # longer the exact prepare-time predecessor.  The sealed proof, not a
        # second alignment interpretation, must advance the owner.
        monkeypatch.setattr(
            provider,
            "_aligned_committed_clean_mask",
            lambda *_args, **_kwargs: (
                case.pending.mask.copy(),
                "post-final-clean raw-envelope alignment",
            ),
        )
        provider._last_publication_guard_aligned_clean = (
            case.pending.mask.copy()
        )
        provider._commit_final_publication_histories(
            case.frame, published, published_obj
        )
        owner = provider._visible_exact_partial_continuity
        assert owner is not None and owner is not case.previous
        assert owner.frame_id == case.frame.frame_id
        assert owner.hits == case.previous.hits + 1
        assert owner.core_area == pytest.approx(70.0)
        np.testing.assert_array_equal(owner.mask, case.core_token)
        assert (
            provider._sanitizer_external_occluder_registry_seal(case.frame)
            is None
        )
    finally:
        provider.stop()


def test_real_prepare_path_external_stage_failure_returns_no_valid_core(
    monkeypatch,
):
    """A mandatory proof which cannot be sealed never becomes robot-facing."""

    case = _prepared_sanitized_external_pipeline_case(
        monkeypatch, fail_second_proof_capture=True
    )
    provider = case.provider
    try:
        assert case.repaired is None
        assert "could not be sealed" in case.reason
        assert "forced second capture failure" in case.reason
        assert case.pending.sanitizer_external_occluder_proof_required
        assert case.pending.sanitizer_external_occluder_proof is None
        seal = case.pending.sanitizer_external_occluder_stage_seal
        assert seal is not None and seal.required and not seal.staged
        assert (
            provider._sanitizer_external_occluder_registry_seal(case.frame)
            is seal
        )
    finally:
        provider.stop()


@pytest.mark.parametrize(
    "tamper",
    ("proof", "required", "seal", "registry", "commit_false"),
)
def test_real_prepare_path_external_owner_failure_rolls_back_whole_frame(
    monkeypatch,
    tamper,
):
    """No valid external core may survive without its same-frame owner."""

    case = _prepared_sanitized_external_pipeline_case(monkeypatch)
    provider = case.provider
    try:
        assert case.repaired is not None, case.reason
        published, published_obj = _apply_prepared_sanitized_external_case(
            case
        )
        final_before = provider._last_final_clean_publication
        owner_before = provider._visible_exact_partial_continuity
        if tamper == "proof":
            case.pending.sanitizer_external_occluder_proof = None
        elif tamper == "required":
            case.pending.sanitizer_external_occluder_proof_required = False
        elif tamper == "seal":
            case.pending.sanitizer_external_occluder_stage_seal = None
        elif tamper == "registry":
            provider._revoke_sanitizer_external_occluder_stage_registry()
        else:
            monkeypatch.setattr(
                provider,
                "_commit_visible_exact_partial_continuity",
                lambda *_args, **_kwargs: False,
            )

        with pytest.raises(
            RuntimeError, match="sealed sanitizer external proof"
        ):
            provider._commit_final_publication_histories(
                case.frame, published, published_obj
            )
        assert provider._last_final_clean_publication is final_before
        assert provider._visible_exact_partial_continuity is owner_before
        assert provider._pending_full_target_geometry_commit is None
        assert (
            provider._sanitizer_external_occluder_registry_seal(case.frame)
            is None
        )
    finally:
        provider.stop()


def test_source222_to224_sealed_external_proof_atomically_advances_partial_owner(
    monkeypatch,
):
    """Prepare-time raw alignment survives generic final-clean replacement."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    (
        frame,
        previous,
        aligned,
        pending,
        core_token,
    ) = _bind_source140_like_external_occluder_case(
        provider, outside_bridge=True
    )
    provider._last_output_mask_source = "publication_sanitized_target_core"
    provider._last_online_sam2_status = (
        "guarded_primary_visible_partial_sanitized_continuation: "
        "source224 strict external core"
    )
    provider._last_online_sam2_exact_result = True
    provider._last_publication_guard_frame_id = frame.frame_id
    provider._last_publication_guard_aligned_clean = aligned.copy()
    try:
        pending.sanitizer_external_occluder_proof_required = (
            provider._sanitizer_external_occluder_proof_is_required(
                core_token, pending.mask, aligned, previous
            )
        )
        assert pending.sanitizer_external_occluder_proof_required
        staged, reason = provider._stage_sanitizer_external_occluder_proof(
            frame, core_token, pending, previous, aligned
        )
        assert staged, reason
        assert pending.sanitizer_external_occluder_proof is not None

        # This is the source224 bug: generic final-clean advances before the
        # source222 partial owner.  Even if a later helper would now derive a
        # different alignment from the tiny final mask, the exact prepare-time
        # raw-envelope proof remains frame/owner bound.
        provider._last_publication_guard_aligned_clean = pending.mask.copy()
        result = _exact_token_mask_result(
            core_token, message="source224 sealed strict sanitizer core"
        )
        obj = provider._mask_only_publication_evidence(
            frame, core_token, preserve_sanitized_core=True
        )
        provider._commit_final_publication_histories(frame, result, obj)
        owner = provider._visible_exact_partial_continuity
        assert owner is not None
        assert owner is not previous
        assert owner.frame_id == frame.frame_id
        assert owner.hits == previous.hits + 1
        assert owner.evidence_kind == "sanitized_target_core"
        np.testing.assert_array_equal(owner.mask, core_token)
    finally:
        provider.stop()


def test_sealed_external_proof_tamper_rolls_back_final_clean_and_partial_owner(
    monkeypatch,
):
    """A source224 proof failure cannot leave a valid mask without its owner."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    (
        frame,
        previous,
        aligned,
        pending,
        core_token,
    ) = _bind_source140_like_external_occluder_case(
        provider, outside_bridge=True
    )
    provider._last_output_mask_source = "publication_sanitized_target_core"
    provider._last_online_sam2_status = (
        "guarded_primary_visible_partial_sanitized_continuation: "
        "source224 strict external core"
    )
    provider._last_online_sam2_exact_result = True
    provider._last_publication_guard_frame_id = frame.frame_id
    provider._last_publication_guard_aligned_clean = aligned.copy()
    try:
        pending.sanitizer_external_occluder_proof_required = (
            provider._sanitizer_external_occluder_proof_is_required(
                core_token, pending.mask, aligned, previous
            )
        )
        assert pending.sanitizer_external_occluder_proof_required
        staged, reason = provider._stage_sanitizer_external_occluder_proof(
            frame, core_token, pending, previous, aligned
        )
        assert staged, reason
        proof = pending.sanitizer_external_occluder_proof
        assert proof is not None
        final_before = provider._last_final_clean_publication
        owner_before = provider._visible_exact_partial_continuity
        object.__setattr__(
            proof, "bridged_loss_pixels", int(proof.bridged_loss_pixels) - 1
        )
        result = _exact_token_mask_result(
            core_token, message="tampered source224 sealed proof"
        )
        obj = provider._mask_only_publication_evidence(
            frame, core_token, preserve_sanitized_core=True
        )
        with pytest.raises(RuntimeError, match="sealed sanitizer external proof"):
            provider._commit_final_publication_histories(frame, result, obj)
        assert provider._last_final_clean_publication is final_before
        assert provider._visible_exact_partial_continuity is owner_before
    finally:
        provider.stop()


@pytest.mark.parametrize(
    "tamper",
    ("proof_removed", "required_flip", "aligned_mask_bytes"),
)
def test_sanitizer_external_stage_seal_cannot_be_bypassed(monkeypatch, tamper):
    """The monotonic prepare seal survives public pending-field tampering."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    frame, previous, aligned, pending, core_token = (
        _bind_source140_like_external_occluder_case(
            provider, outside_bridge=True
        )
    )
    provider._last_output_mask_source = "publication_sanitized_target_core"
    provider._last_online_sam2_status = (
        "guarded_primary_visible_partial_sanitized_continuation: "
        "source224 monotonic-seal tamper"
    )
    provider._last_online_sam2_exact_result = True
    provider._last_publication_guard_frame_id = frame.frame_id
    provider._last_publication_guard_aligned_clean = aligned.copy()
    try:
        pending.sanitizer_external_occluder_proof_required = True
        staged, reason = provider._stage_sanitizer_external_occluder_proof(
            frame, core_token, pending, previous, aligned
        )
        assert staged, reason
        proof = pending.sanitizer_external_occluder_proof
        seal = pending.sanitizer_external_occluder_stage_seal
        assert proof is not None and seal is not None
        final_before = provider._last_final_clean_publication
        owner_before = provider._visible_exact_partial_continuity

        if tamper == "proof_removed":
            pending.sanitizer_external_occluder_proof = None
        elif tamper == "required_flip":
            pending.sanitizer_external_occluder_proof_required = False
        else:
            proof.aligned_predecessor_mask.setflags(write=True)
            proof.aligned_predecessor_mask[40, 80] ^= np.uint8(1)
            proof.aligned_predecessor_mask.setflags(write=False)

        result = _exact_token_mask_result(
            core_token, message=f"source224 stage-seal {tamper}"
        )
        obj = provider._mask_only_publication_evidence(
            frame, core_token, preserve_sanitized_core=True
        )
        with pytest.raises(RuntimeError, match="sealed sanitizer external proof"):
            provider._commit_final_publication_histories(frame, result, obj)
        assert provider._last_final_clean_publication is final_before
        assert provider._visible_exact_partial_continuity is owner_before
    finally:
        provider.stop()


def test_internal_erosion_requires_but_cannot_stage_external_owner_proof(
    monkeypatch,
):
    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    frame, previous, aligned, pending, core_token = (
        _bind_source140_like_external_occluder_case(
            provider, outside_bridge=False
        )
    )
    try:
        assert not provider._sanitizer_external_occluder_proof_is_required(
            core_token, pending.mask, aligned, previous
        )
        staged, reason = provider._stage_sanitizer_external_occluder_proof(
            frame, core_token, pending, previous, aligned
        )
        assert not staged
        assert "outside bridge" in reason
        assert pending.sanitizer_external_occluder_proof is None
    finally:
        provider.stop()


def test_apply_gate_missing_mandatory_external_proof_is_frame_local_invalid(
    monkeypatch,
):
    """A failed generic sanitizer seal drops one frame, not the stream."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(
        monkeypatch, [], {}, initial
    )
    frame, previous, aligned, pending, core_token = (
        _bind_source140_like_external_occluder_case(
            provider, outside_bridge=True
        )
    )
    raw_result = _exact_token_mask_result(
        pending.mask, message="focused unsealed raw candidate"
    )
    raw_result.source = "online_sam2_video"
    sanitized_result = _exact_token_mask_result(
        core_token, message="focused unsealed sanitizer core"
    )
    sanitized_result.source = "publication_sanitized_target_core"
    provider._last_output_mask_source = "online_sam2_guarded_primary"
    provider._last_online_sam2_exact_result = True
    provider._last_publication_guard_aligned_clean = aligned.copy()
    monkeypatch.setattr(
        provider,
        "_publication_candidate_agrees_with_committed_clean_mask",
        lambda *_args, **_kwargs: (False, "focused raw appendage reject"),
    )
    monkeypatch.setattr(
        provider,
        "_unified_two_sam_may_enter_long_gap_probation",
        lambda *_args, **_kwargs: (False, "focused no probation"),
    )
    monkeypatch.setattr(
        provider,
        "_sanitize_publication_target_core",
        lambda *_args, **_kwargs: (
            sanitized_result,
            "focused strict sanitizer",
        ),
    )
    monkeypatch.setattr(
        provider,
        "_sanitizer_external_occluder_proof_is_required",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        provider,
        "_stage_sanitizer_external_occluder_proof",
        lambda *_args, **_kwargs: (False, "focused proof unavailable"),
    )
    try:
        raw_obj = provider._mask_only_publication_evidence(
            frame, pending.mask
        )
        assert raw_obj.valid
        published, published_obj = provider._apply_recovery_publication_gate(
            frame, raw_result, raw_obj, mask_only=True
        )
        assert not published.valid
        assert not published_obj.valid
        assert "mandatory external occluder proof was not sealed" in (
            published.message
        )
        assert "focused proof unavailable" in published.message
        assert provider._pending_full_target_geometry_commit is None
        assert provider._visible_exact_partial_continuity is None
        assert provider._sanitizer_external_occluder_registry_seal(frame) is None
        assert provider._last_output_mask_source == (
            "publication_foreign_appendage_rejected"
        )
        assert previous is not provider._visible_exact_partial_continuity
    finally:
        provider.stop()


def test_guarded_primary_low_retention_core_cannot_chain_stationary_erosion(
    monkeypatch,
):
    """The 0.55 continuation exception cannot accumulate core erosion."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    raw = _large_mask((70, 35, 110, 75))
    ys, xs = np.nonzero(raw > 0)
    center_x = float(np.mean(xs))
    center_y = float(np.mean(ys))
    radius_x = 0.5 * float(np.ptp(xs) + 1)
    radius_y = 0.5 * float(np.ptp(ys) + 1)
    rank = np.minimum(
        np.abs(xs - center_x) / radius_x,
        np.abs(ys - center_y) / radius_y,
    )
    order = np.argsort(rank, kind="stable")
    current_core = np.zeros_like(raw)
    previous_core = np.zeros_like(raw)
    current_core[ys[order[:900]], xs[order[:900]]] = 1
    previous_core[ys[order[:1000]], xs[order[:1000]]] = 1
    bbox = np.asarray([70, 35, 110, 75], dtype=np.int32)
    center = provider._binary_mask_centroid(raw)
    previous_core_center = provider._binary_mask_centroid(previous_core)
    assert center is not None and previous_core_center is not None
    previous = SimpleNamespace(
        frame_id=103,
        timestamp=103.0 / 30.0,
        mask=previous_core.copy(),
        mask_digest=provider._publication_mask_digest(previous_core),
        core_bbox_xyxy=bbox.copy(),
        core_center_xy=previous_core_center.copy(),
        core_area=1000.0,
        bbox_xyxy=bbox.copy(),
        center_xy=center.copy(),
        area=1600.0,
        bbox_area=1600.0,
        depth_median=0.8,
        center_velocity_px_s=np.zeros(2, dtype=np.float64),
        velocity_samples=8,
        hits=9,
        evidence_kind="sanitized_target_core",
        sanitized_handoff_eligible=True,
        # The persistent anchor records the original un-eroded core even
        # though each individual 1000 -> 900 step looks superficially mild.
        sanitized_anchor_bbox_xyxy=bbox.copy(),
        sanitized_anchor_center_xy=center.copy(),
        sanitized_anchor_area=1600.0,
    )
    provider._visible_exact_partial_continuity = previous
    authority_before = copy.deepcopy(provider._full_target_geometry_authority)
    trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)
    frame104 = _large_frame(104)
    frame104.color_bgr[:] = 45
    frame104.color_bgr[current_core > 0] = (210, 80, 235)
    provider._defer_full_target_geometry_commit(frame104, raw)
    provider._mark_pending_full_target_geometry_ineligible(
        frame104, "stationary progressive core erosion"
    )
    pending = provider._pending_full_target_geometry_commit
    assert pending is not None
    provider._last_output_mask_source = "publication_sanitized_target_core"
    provider._last_online_sam2_status = (
        "guarded_primary_visible_partial_continuation: stationary erosion"
    )
    try:
        assert 0.55 < float(current_core.sum()) / float(raw.sum()) < 0.60
        assert not provider._commit_visible_exact_partial_continuity(
            frame104,
            _mask_result(current_core, message="stationary eroded core"),
            pending,
        )
        assert provider._visible_exact_partial_continuity is previous
        assert provider._full_target_geometry_authority.frame_id == (
            authority_before.frame_id
        )
        assert provider._trusted_online_sam2_publication.frame_id == (
            trusted_before.frame_id
        )
    finally:
        provider.stop()


def test_guarded_primary_late_sanitized_core_hands_off_to_boundary(
    monkeypatch,
):
    """Reviewed source93..123 late-exit semantics stay policy-visible.

    Source99 contains a small low-appearance branch and is safely reduced to
    an immutable-appearance target core.  Source104 is sanitizer-owned, then
    source105 crosses below the ordinary 0.65 publication floor while staying
    above the bounded 0.60 partial-admission floor.  It must follow normal
    partial continuity and the final sanitizer instead of entering stale-scale
    recovery.  Later low-support raw masks remain sanitizer-only while the
    core carries motion into a confirmed right-edge exit.  Full/trusted
    authority remains the strong source93 seed throughout.
    """

    seed_bbox = (92, 45, 137, 90)
    seed = _large_ellipse_mask(seed_bbox)

    def connected_support_frame(frame_id, mask, support_fraction):
        frame = _large_frame(frame_id)
        frame.color_bgr[:] = 45
        ys, xs = np.nonzero(np.asarray(mask) > 0)
        supported = int(np.floor(len(xs) * float(support_fraction)))
        if supported > 0:
            center_x = float(np.mean(xs))
            center_y = float(np.mean(ys))
            radius_x = max(1.0, 0.5 * float(np.ptp(xs) + 1))
            radius_y = max(1.0, 0.5 * float(np.ptp(ys) + 1))
            # A thick connected plus-shaped support region keeps the target's
            # full x/y extent while matching the reviewed aggregate support.
            # It survives the mask-only 3x3 erosion unlike isolated pixels.
            rank = np.minimum(
                np.abs(xs - center_x) / radius_x,
                np.abs(ys - center_y) / radius_y,
            )
            positions = np.argsort(rank, kind="stable")[:supported]
            frame.color_bgr[ys[positions], xs[positions]] = (
                210,
                80,
                235,
            )
        return frame

    trace = {
        95: ((95, 45, 140, 90), 0.77),
        96: ((97, 45, 142, 90), 0.77),
        98: ((100, 45, 145, 90), 0.765),
        101: ((105, 45, 150, 90), 0.75),
        102: ((107, 45, 152, 90), 0.74),
        104: ((109, 45, 154, 90), 0.620),
        105: ((110, 45, 155, 90), 0.640),
        107: ((112, 45, 157, 90), 0.646),
        108: ((113, 45, 158, 90), 0.644),
        110: ((115, 45, 160, 90), 0.643),
        111: ((116, 45, 161, 90), 0.642),
        113: ((118, 45, 163, 90), 0.634),
        114: ((118, 45, 163, 90), 0.640),
        116: ((118, 45, 163, 90), 0.650),
        117: ((118, 45, 163, 90), 0.647),
        119: ((120, 45, 165, 90), 0.616),
        120: ((122, 45, 167, 90), 0.601),
        122: ((125, 45, 170, 90), 0.585),
        123: ((127, 45, 172, 90), 0.580),
    }
    masks = {
        frame_id: _large_ellipse_mask(bbox)
        for frame_id, (bbox, _support) in trace.items()
    }
    source99_core = _large_ellipse_mask((102, 45, 147, 90))
    source99_branch = _large_mask((98, 55, 102, 75))
    masks[99] = np.maximum(source99_core, source99_branch)

    frames = []
    for frame_id in (95, 96, 98):
        _bbox, support = trace[frame_id]
        frames.append(connected_support_frame(frame_id, masks[frame_id], support))
    # The target core is unambiguously magenta; only the connected 80px
    # branch is gray and must be removed before it can own continuity.
    frames.append(connected_support_frame(99, source99_core, 1.0))
    for frame_id in trace:
        if frame_id <= 98:
            continue
        _bbox, support = trace[frame_id]
        frames.append(connected_support_frame(frame_id, masks[frame_id], support))

    tracker = _IdentityAwareFakeTracker({})
    manager = _FakeOnlineSAM2Manager(
        {frame_id: _video_result(frame_id, mask) for frame_id, mask in masks.items()}
    )
    provider, _camera = _provider(
        monkeypatch,
        frames,
        tracker,
        manager,
        recovery_publication_mode="unified_three_evidence",
    )
    provider.online_sam2_cfg["mask_publication_mode"] = "guarded_sam2_primary"
    seed_frame = _moving_mask_support_frame(93, seed, 1.0)
    assert provider.initialize_from_mask(seed_frame, seed)
    authority_before = copy.deepcopy(provider._full_target_geometry_authority)
    trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)
    try:
        outputs = {}
        statuses = {}
        for _ in frames:
            frame, result = provider.step_mask_only()
            outputs[frame.frame_id] = result
            statuses[frame.frame_id] = (
                provider._last_online_sam2_status,
                provider._last_publication_guard_status,
                provider._last_output_mask_source,
            )

        assert all(result.valid for result in outputs.values()), [
            (
                frame_id,
                result.message,
                statuses[frame_id],
            )
            for frame_id, result in outputs.items()
            if not result.valid
        ]
        assert outputs[99].valid
        assert not np.any(np.logical_and(outputs[99].mask > 0, source99_branch > 0))
        assert not np.array_equal(outputs[99].mask, masks[99])
        assert statuses[104][2] == "publication_sanitized_target_core"
        assert statuses[105][0].startswith(
            "guarded_primary_visible_partial_continuation:"
        )
        assert statuses[105][2] == "publication_sanitized_target_core"

        # Every below-normal-support frame, including the confirmed exit, is
        # admitted only to the pixelwise immutable-appearance sanitizer.  Its
        # raw mask, including every unsupported pixel, is never robot-facing.
        for frame_id in (
            104,
            105,
            107,
            108,
            110,
            111,
            113,
            114,
            117,
            119,
            120,
            122,
            123,
        ):
            assert not np.array_equal(outputs[frame_id].mask, masks[frame_id])
            selected = outputs[frame_id].mask > 0
            pixels = frames[
                [frame.frame_id for frame in frames].index(frame_id)
            ].color_bgr[selected]
            assert len(pixels) > 0
            assert np.all(pixels[:, 0] >= 180)
            assert np.all(pixels[:, 1] <= 120)
            assert np.all(pixels[:, 2] >= 180)

        boundary = provider._boundary_exit_continuity
        assert boundary is not None
        assert boundary.frame_id == 123
        assert boundary.hits >= 4
        assert boundary.edges == ("right",)
        assert boundary.mask_evidence is not None
        assert boundary.mask is boundary.mask_evidence.mask
        assert boundary.mask_digest == boundary.mask_evidence.mask_digest
        assert not boundary.mask.flags.writeable
        assert provider._full_target_geometry_authority.frame_id == (
            authority_before.frame_id
        )
        assert provider._trusted_online_sam2_publication.frame_id == (
            trusted_before.frame_id
        )
        assert provider._last_online_sam2_status.startswith(
            "guarded_primary_boundary_continuation:"
        )
    finally:
        provider.stop()


@pytest.mark.parametrize("invalid_floor", [np.nan, -0.01, 1.01])
def test_guarded_primary_scale_transition_floor_must_be_finite_unit_interval(
    monkeypatch, invalid_floor
):
    initial_bbox = (40, 30, 80, 70)
    shrink_bbox = (80, 32, 116, 68)
    initial = _large_mask(initial_bbox)
    shrink = _large_mask(shrink_bbox)
    provider, tracker = _guarded_primary_provider(
        monkeypatch,
        [_moving_target_frame(2, shrink_bbox)],
        {2: _video_result(2, shrink)},
        initial,
    )
    provider.cfg["tracker"][
        "publication_full_scale_transition_min_normalized_coverage"
    ] = invalid_floor
    try:
        with pytest.raises(
            ValueError,
            match=(
                "publication_full_scale_transition_min_normalized_coverage "
                "must be finite and in 0..1"
            ),
        ):
            provider.step_mask_only()
    finally:
        provider.stop()


def test_guarded_primary_complete_occlusion_clears_scale_transition(
    monkeypatch,
):
    initial_bbox = (40, 30, 80, 70)
    first_shrink_bbox = (80, 32, 116, 68)
    post_occlusion_bbox = (120, 34, 152, 66)
    initial = _large_mask(initial_bbox)
    first_shrink = _large_mask(first_shrink_bbox)
    post_occlusion = _large_mask(post_occlusion_bbox)
    empty = np.zeros_like(initial)
    provider, _tracker = _guarded_primary_provider(
        monkeypatch,
        [
            _moving_target_frame(2, first_shrink_bbox),
            _moving_target_frame(3, (0, 0, 0, 0)),
            _moving_target_frame(4, post_occlusion_bbox),
        ],
        {
            2: _video_result(2, first_shrink),
            3: _video_result(3, empty, valid=False),
            4: _video_result(4, post_occlusion),
        },
        initial,
    )
    try:
        _frame2, result2 = provider.step_mask_only()
        assert result2.valid
        assert provider._full_target_scale_transition_hypothesis.frame_id == 2
        assert provider._full_target_geometry_authority.frame_id == 1

        _frame3, result3 = provider.step_mask_only()
        assert not result3.valid
        assert provider._full_target_scale_transition_hypothesis is None
        assert provider._full_target_geometry_authority.frame_id == 1
        assert not provider._trusted_online_sam2_continuity_intact

        _frame4, result4 = provider.step_mask_only()
        assert not result4.valid
        assert provider._full_target_scale_transition_hypothesis is None
        assert provider._full_target_geometry_authority.frame_id == 1
        assert provider._trusted_online_sam2_publication.frame_id == 1
    finally:
        provider.stop()


def test_guarded_primary_trusted_time_limit_tolerates_device_roundoff(
    monkeypatch,
):
    target_bbox = (40, 30, 80, 70)
    target = _large_mask(target_bbox)
    current = _moving_target_frame(45, target_bbox)
    current.timestamp = 1000.0 + 45.0 / 30.0
    provider, _tracker = _guarded_primary_provider(
        monkeypatch,
        [current],
        {45: _video_result(45, target)},
        target,
    )
    try:
        trusted = provider._trusted_online_sam2_publication
        trusted.frame_id = 39
        trusted.timestamp = 1000.0 + 39.0 / 30.0
        dt_s = current.timestamp - trusted.timestamp
        assert dt_s > 0.20
        assert dt_s == pytest.approx(0.20, abs=1.0e-6)

        geometry, reason = provider._online_mask_geometry(current, target)
        assert geometry is not None, reason
        accepted, detail = provider._trusted_online_sam2_candidate_agrees(
            current, geometry
        )
        assert accepted, detail
    finally:
        provider.stop()


@pytest.mark.parametrize("clean_case", ["mild_shape", "fast_translation"])
def test_guarded_primary_clean_geometry_remains_full_authority_eligible(
    monkeypatch, clean_case
):
    initial = _large_mask((40, 30, 80, 70))
    if clean_case == "mild_shape":
        current_bbox = (41, 30, 79, 70)
    else:
        current_bbox = (100, 30, 140, 70)
    current_mask = _large_mask(current_bbox)
    provider, _tracker = _guarded_primary_provider(
        monkeypatch,
        [_moving_target_frame(2, current_bbox)],
        {2: _video_result(2, current_mask)},
        initial,
    )
    try:
        _frame2, result = provider.step_mask_only()
        assert result.valid
        np.testing.assert_array_equal(result.mask, current_mask)
        assert not provider._last_publication_guard_known_partial
        assert provider._full_target_geometry_authority.frame_id == 2
        assert provider._trusted_online_sam2_publication.frame_id == 2
        assert "full_target_coverage=" in (provider._last_publication_guard_status)
    finally:
        provider.stop()


def test_guarded_primary_sanitizes_geometry_only_giant_appendage_read_only(
    monkeypatch,
):
    """A same-colour giant mask cannot skip sanitization via appearance."""

    target = _large_mask((40, 30, 80, 70))
    giant = _large_mask((0, 0, 120, 100))
    current = _large_frame(2)
    current.color_bgr[:] = 45
    current.color_bgr[giant.astype(bool)] = (210, 80, 235)
    provider, tracker = _guarded_primary_provider(
        monkeypatch,
        [current],
        {2: _video_result(2, giant)},
        target,
    )
    token_before = tracker.token
    final_before = copy.deepcopy(provider._last_final_clean_publication)
    try:
        result = provider._prepare_guarded_sam2_primary_mask(
            current, _mask_result(giant)
        )

        # First recovery evidence remains fail-closed, but the candidate held
        # for comparison is the clean visible core, not the giant aggregate.
        assert not result.valid
        assert provider._last_online_sam2_status == (
            "guarded_primary_recovery_pending_1"
        )
        hypothesis = provider._online_sam2_recovery_hypothesis
        assert hypothesis is not None
        assert hypothesis.area == pytest.approx(float(target.sum()))
        assert tuple(
            (hypothesis.bbox_xyxy[2:] - hypothesis.bbox_xyxy[:2]).tolist()
        ) == (40, 40)
        assert provider._last_online_sam2_area == int(target.sum())
        assert tracker.token == token_before
        assert provider._last_final_clean_publication.frame_id == (
            final_before.frame_id
        )
        np.testing.assert_array_equal(
            provider._last_final_clean_publication.mask,
            final_before.mask,
        )
    finally:
        provider.stop()


def test_guarded_primary_geometry_sanitizer_rejects_far_same_colour_instance(
    monkeypatch,
):
    target = _large_mask((10, 30, 50, 70))
    far = _large_mask((130, 30, 170, 70))
    current = _moving_target_frame(2, (130, 30, 170, 70))
    provider, tracker = _guarded_primary_provider(
        monkeypatch,
        [current],
        {2: _video_result(2, far)},
        target,
    )
    token_before = tracker.token
    try:
        result = provider._prepare_guarded_sam2_primary_mask(current, _mask_result(far))

        assert not result.valid
        assert provider._online_sam2_recovery_hypothesis is None
        assert provider._last_online_sam2_status == (
            "guarded_primary_recovery_rejected"
        )
        assert tracker.token == token_before
        assert provider._last_final_clean_publication.frame_id == 1
        np.testing.assert_array_equal(
            provider._last_final_clean_publication.mask, target
        )
    finally:
        provider.stop()


@pytest.mark.parametrize("scorer_state", ["missing", "returns_none"])
def test_guarded_primary_significant_extra_requires_appearance_scorer(
    monkeypatch, scorer_state
):
    target = _large_mask((40, 30, 80, 70))
    appendage = _large_mask((80, 30, 105, 70))
    candidate_mask = target | appendage
    current = _large_frame(2)
    current.color_bgr[candidate_mask.astype(bool)] = (210, 80, 235)
    provider, tracker = _guarded_primary_provider(monkeypatch, [], {}, target)
    try:
        if scorer_state == "missing":
            tracker.target_appearance_support_stats = None
        else:
            tracker.target_appearance_support_stats = lambda *_args, **_kwargs: None
        candidate = _mask_result(candidate_mask)

        (
            accepted_v2,
            detail_v2,
        ) = provider._publication_candidate_agrees_with_committed_clean_mask(
            current, candidate
        )
        assert not accepted_v2
        assert "appearance model unavailable" in detail_v2

        provider.online_sam2_cfg["mask_publication_mode"] = "adaptive_fusion"
        (
            accepted_legacy,
            detail_legacy,
        ) = provider._publication_candidate_agrees_with_committed_clean_mask(
            current, candidate
        )
        assert accepted_legacy, detail_legacy
    finally:
        provider.stop()


@pytest.mark.parametrize(
    ("extra_bbox", "scorer_state"),
    [
        (None, "missing"),
        (None, "returns_none"),
        (None, "returns_nan"),
        ((80, 40, 82, 45), "returns_none"),
        ((80, 40, 82, 45), "returns_nan"),
    ],
)
def test_guarded_primary_small_or_no_extra_fails_closed_without_finite_scorer(
    monkeypatch, extra_bbox, scorer_state
):
    target_bbox = (40, 30, 80, 70)
    target = _large_mask(target_bbox)
    candidate = target.copy()
    if extra_bbox is not None:
        candidate |= _large_mask(extra_bbox)
    current = _moving_target_frame(2, target_bbox)
    provider, tracker = _guarded_primary_provider(
        monkeypatch,
        [current],
        {2: _video_result(2, candidate)},
        target,
    )
    authority_before = copy.deepcopy(provider._full_target_geometry_authority)
    if scorer_state == "missing":
        tracker.target_appearance_support_stats = None
    elif scorer_state == "returns_none":
        tracker.target_appearance_support_stats = lambda *_args, **_kwargs: None
    else:
        tracker.target_appearance_support_stats = lambda _frame, mask, **_kwargs: (
            float("nan"),
            float("nan"),
            int(np.count_nonzero(mask)),
        )
    try:
        _frame2, result = provider.step_mask_only()

        assert not result.valid
        assert provider._full_target_geometry_authority.frame_id == (
            authority_before.frame_id
        )
        assert "appearance" in result.message
    finally:
        provider.stop()


def test_guarded_primary_in_footprint_hand_cannot_teach_authority(
    monkeypatch,
):
    full_bbox = (40, 30, 80, 70)
    target_bbox = (40, 30, 70, 70)
    hand_bbox = (70, 35, 80, 65)
    full = _large_mask(full_bbox)
    target = _large_mask(target_bbox)
    hand = _large_mask(hand_bbox)
    contaminated = target | hand
    current = _moving_target_frame(2, target_bbox, contaminant_bbox=hand_bbox)
    provider, _tracker = _guarded_primary_provider(
        monkeypatch,
        [current],
        {2: _video_result(2, contaminated)},
        full,
    )
    authority_before = copy.deepcopy(provider._full_target_geometry_authority)
    trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)
    try:
        _frame2, result = provider.step_mask_only()

        # The visible target may remain policy-facing, but 80% immutable
        # support cannot promote a mask containing a 300px hand replacement
        # to full/trusted geometry authority.
        assert result.valid
        assert provider._full_target_geometry_authority.frame_id == (
            authority_before.frame_id
        )
        assert provider._trusted_online_sam2_publication.frame_id == (
            trusted_before.frame_id
        )
        assert provider._last_publication_guard_known_partial
        assert "appearance support" in (provider._last_publication_guard_partial_reason)
    finally:
        provider.stop()


def test_guarded_primary_keeps_clean_outward_boundary_exit(monkeypatch):
    previous = _large_mask((130, 40, 170, 80))
    exiting = _large_mask((150, 40, 180, 80))
    current = _moving_target_frame(2, (150, 40, 180, 80))
    provider, tracker = _guarded_primary_provider(
        monkeypatch,
        [current],
        {2: _video_result(2, exiting)},
        previous,
    )
    try:
        provider._last_output_mask_source = "online_sam2_guarded_primary"
        accepted, reason = provider._outward_boundary_clipping_accepts(
            current, _mask_result(exiting)
        )

        assert accepted, reason
        assert "edge=right" in reason
        assert tracker.token == "initialized"
    finally:
        provider.stop()


def test_guarded_primary_keeps_raw_exact_boundary_exit_with_distributed_appearance(
    monkeypatch,
):
    previous = _large_mask((130, 40, 170, 80))
    exiting = _large_mask((150, 40, 180, 80))
    current = _boundary_exit_appearance_frame(
        2, (150, 40, 180, 80), distributed=True
    )
    provider, _tracker = _guarded_primary_provider(
        monkeypatch,
        [current],
        {2: _video_result(2, exiting)},
        previous,
    )
    aligned_clean = _large_mask((150, 40, 170, 80))
    monkeypatch.setattr(
        provider,
        "_aligned_committed_clean_mask",
        lambda *_args, **_kwargs: (
            aligned_clean.copy(),
            "test clipped boundary alignment",
        ),
    )
    initial_full = copy.deepcopy(provider._full_target_geometry_authority)
    initial_trusted = copy.deepcopy(provider._trusted_online_sam2_publication)
    try:
        frame, result = provider.step_mask_only()

        assert frame.frame_id == 2
        assert result.valid
        assert result.source == "online_sam2_video"
        np.testing.assert_array_equal(result.mask, exiting)
        assert "raw_exact_boundary_visible_partial accepted" in (
            provider._last_publication_guard_status
        )
        assert "spatial appearance accepted" in (
            provider._last_publication_guard_status
        )
        assert provider._last_publication_guard_known_partial
        assert "outward-boundary" in provider._last_publication_guard_partial_reason
        # This exception is policy-facing visible evidence only.  It cannot
        # promote the clipped frame to full geometry or trusted dynamics.
        assert provider._full_target_geometry_authority.frame_id == (
            initial_full.frame_id
        )
        assert provider._trusted_online_sam2_publication.frame_id == (
            initial_trusted.frame_id
        )
    finally:
        provider.stop()


def test_guarded_primary_keeps_last_same_edge_boundary_sliver(monkeypatch):
    """A confirmed exit may shrink at about 30/s for its last visible tick."""

    previous = _large_mask((130, 40, 170, 80))
    reference = np.zeros((120, 180), dtype=np.uint8)
    cv2.ellipse(reference, (168, 60), (11, 20), 0, 0, 360, 1, -1)
    final_sliver = np.zeros_like(reference)
    cv2.ellipse(final_sliver, (173, 60), (6, 13), 0, 0, 360, 1, -1)
    current = _boundary_exit_appearance_frame(
        3, (167, 47, 180, 74), distributed=True
    )
    provider, _tracker = _guarded_primary_provider(
        monkeypatch,
        [current],
        {3: _video_result(3, final_sliver)},
        previous,
    )
    provider.cfg["tracker"][
        "publication_boundary_exit_continuation_max_log_shrink_rate_s"
    ] = 36.0
    try:
        reference_frame = _boundary_exit_appearance_frame(
            2, (157, 40, 180, 81), distributed=True
        )
        reference_geometry, reference_reason = provider._online_mask_geometry(
            reference_frame, reference
        )
        assert reference_geometry is not None, reference_reason
        provider._boundary_exit_continuity = _BoundaryExitContinuity(
            frame_id=2,
            timestamp=reference_frame.timestamp,
            mask=reference.copy(),
            mask_digest=provider._publication_mask_digest(reference),
            bbox_xyxy=reference_geometry.bbox_xyxy.copy(),
            center_xy=reference_geometry.centroid_xy.copy(),
            area=float(reference_geometry.area),
            bbox_area=float(reference_geometry.bbox_area),
            depth_median=float(reference_geometry.depth_median),
            edges=("right",),
            hits=3,
        )

        frame3, result3 = provider.step_mask_only()
        assert frame3.frame_id == 3
        assert result3.valid, (
            provider.online_sam2_status,
            provider._last_publication_guard_status,
            provider._last_visible_exact_partial_reason,
            result3.message,
        )
        np.testing.assert_array_equal(result3.mask, final_sliver)
        assert "boundary" in provider.online_sam2_status

        # The exception remains shrink-only.  A same-edge hand/growth branch
        # cannot consume the larger shrink budget.
        state = provider._boundary_exit_continuity
        assert state is not None
        growing = _large_mask((150, 30, 180, 90))
        growth_frame = _boundary_exit_appearance_frame(
            4, (150, 30, 180, 90), distributed=True
        )
        geometry, reason = provider._online_mask_geometry(growth_frame, growing)
        assert geometry is not None, reason
        accepted, reason, _edges = provider._boundary_exit_continuation_accepts(
            growth_frame, geometry
        )
        assert not accepted
        assert any(
            fragment in reason
            for fragment in ("growth", "area_rate", "did not continue outward")
        )
    finally:
        provider.stop()


def test_guarded_primary_first_boundary_raw_uses_visible_partial_without_trusted(
    monkeypatch,
):
    previous_bbox = (130, 40, 170, 80)
    previous = _large_mask(previous_bbox)
    exiting = _large_mask((150, 40, 180, 80))
    current = _boundary_exit_appearance_frame(
        2, (150, 40, 180, 80), distributed=True
    )
    provider, _tracker = _guarded_primary_provider(
        monkeypatch,
        [current],
        {2: _video_result(2, exiting)},
        previous,
    )
    _install_visible_partial_boundary_history(
        provider,
        previous_bbox,
        velocity_xy=(450.0, 0.0),
    )
    aligned_clean = _large_mask((150, 40, 170, 80))
    monkeypatch.setattr(
        provider,
        "_aligned_committed_clean_mask",
        lambda *_args, **_kwargs: (
            aligned_clean.copy(),
            "test clipped visible-partial alignment",
        ),
    )
    initial_full = copy.deepcopy(provider._full_target_geometry_authority)
    try:
        frame, result = provider.step_mask_only()

        assert frame.frame_id == 2
        assert result.valid
        assert result.source == "online_sam2_video"
        np.testing.assert_array_equal(result.mask, exiting)
        assert "first raw outward boundary exit accepted" in (
            provider._last_publication_guard_status
        )
        assert "raw_exact_boundary_visible_partial accepted" in (
            provider._last_publication_guard_status
        )
        assert provider._last_publication_guard_known_partial
        assert provider._full_target_geometry_authority.frame_id == (
            initial_full.frame_id
        )
        assert provider._trusted_online_sam2_publication is None
        assert provider._guarded_v2_bootstrap_phase != "commissioned"
    finally:
        provider.stop()


@pytest.mark.parametrize("rejection", ("low_appearance", "remote", "growing"))
def test_guarded_primary_visible_partial_boundary_fallback_rejects_unsafe_raw(
    monkeypatch,
    rejection,
):
    if rejection == "remote":
        previous_bbox = (50, 40, 90, 80)
        current_bbox = (150, 40, 180, 80)
        distributed = True
        aligned_bbox = (150, 40, 170, 80)
        velocity = (0.0, 0.0)
    elif rejection == "growing":
        previous_bbox = (145, 40, 175, 80)
        current_bbox = (148, 37, 180, 83)
        distributed = True
        aligned_bbox = (148, 37, 170, 83)
        velocity = (120.0, 0.0)
    else:
        previous_bbox = (130, 40, 170, 80)
        current_bbox = (150, 40, 180, 80)
        distributed = False
        aligned_bbox = (150, 40, 170, 80)
        velocity = (450.0, 0.0)
    previous = _large_mask(previous_bbox)
    candidate = _large_mask(current_bbox)
    current = _boundary_exit_appearance_frame(
        2, current_bbox, distributed=distributed
    )
    provider, _tracker = _guarded_primary_provider(
        monkeypatch,
        [current],
        {2: _video_result(2, candidate)},
        previous,
    )
    _install_visible_partial_boundary_history(
        provider,
        previous_bbox,
        velocity_xy=velocity,
    )
    aligned_clean = _large_mask(aligned_bbox)
    monkeypatch.setattr(
        provider,
        "_aligned_committed_clean_mask",
        lambda *_args, **_kwargs: (
            aligned_clean.copy(),
            "test unsafe visible-partial boundary alignment",
        ),
    )
    try:
        _frame2, result = provider.step_mask_only()

        assert not (
            result.valid
            and result.source == "online_sam2_video"
            and np.array_equal(result.mask, candidate)
        )
        assert "raw_exact_boundary_visible_partial accepted" not in (
            provider._last_publication_guard_status
        )
        assert provider._trusted_online_sam2_publication is None
        if rejection == "low_appearance":
            assert "appearance" in provider._last_publication_guard_status
        elif rejection == "growing":
            assert "grew area/bbox" in (
                provider._last_publication_guard_status
                + " "
                + provider._last_online_sam2_status
                + " "
                + result.message
            )
        else:
            assert "speed" in (
                provider._last_publication_guard_status
                + " "
                + provider._last_online_sam2_status
                + " "
                + result.message
            )
    finally:
        provider.stop()


def test_guarded_primary_boundary_exit_still_sanitizes_one_sided_appendage(
    monkeypatch,
):
    previous = _large_mask((130, 40, 170, 80))
    contaminated = _large_mask((150, 40, 180, 80))
    current = _boundary_exit_appearance_frame(
        2, (150, 40, 180, 80), distributed=False
    )
    provider, _tracker = _guarded_primary_provider(
        monkeypatch,
        [current],
        {2: _video_result(2, contaminated)},
        previous,
    )
    aligned_clean = _large_mask((150, 40, 170, 80))
    monkeypatch.setattr(
        provider,
        "_aligned_committed_clean_mask",
        lambda *_args, **_kwargs: (
            aligned_clean.copy(),
            "test clipped boundary alignment",
        ),
    )
    try:
        _frame2, result = provider.step_mask_only()

        assert result.valid
        assert result.source == "publication_sanitized_target_core"
        assert not np.array_equal(result.mask, contaminated)
        assert not np.any(result.mask[:, 170:180])
        assert "appearance" in provider._last_publication_guard_status
        assert "raw_exact_boundary_visible_partial accepted" not in (
            provider._last_publication_guard_status
        )
    finally:
        provider.stop()


def test_guarded_primary_nonboundary_cannot_use_raw_boundary_extent_exception(
    monkeypatch,
):
    previous = _large_mask((80, 40, 120, 80))
    candidate = _large_mask((100, 40, 130, 80))
    current = _boundary_exit_appearance_frame(
        2, (100, 40, 130, 80), distributed=True
    )
    provider, _tracker = _guarded_primary_provider(
        monkeypatch,
        [current],
        {2: _video_result(2, candidate)},
        previous,
    )
    aligned_clean = _large_mask((100, 40, 120, 80))
    monkeypatch.setattr(
        provider,
        "_aligned_committed_clean_mask",
        lambda *_args, **_kwargs: (
            aligned_clean.copy(),
            "test nonboundary alignment",
        ),
    )
    try:
        _frame2, result = provider.step_mask_only()

        assert result.valid
        assert result.source == "publication_sanitized_target_core"
        assert not np.array_equal(result.mask, candidate)
        assert "raw_exact_boundary_visible_partial accepted" not in (
            provider._last_publication_guard_status
        )
        assert "candidate does not touch an image boundary" in (
            provider._last_publication_guard_status
        )
    finally:
        provider.stop()


def test_guarded_primary_boundary_extent_exception_requires_current_exact_token(
    monkeypatch,
):
    previous = _large_mask((130, 40, 170, 80))
    exiting = _large_mask((150, 40, 180, 80))
    current = _boundary_exit_appearance_frame(
        2, (150, 40, 180, 80), distributed=True
    )
    provider, _tracker = _guarded_primary_provider(
        monkeypatch,
        [],
        {},
        previous,
    )
    aligned_clean = _large_mask((150, 40, 170, 80))
    monkeypatch.setattr(
        provider,
        "_aligned_committed_clean_mask",
        lambda *_args, **_kwargs: (
            aligned_clean.copy(),
            "test clipped boundary alignment",
        ),
    )
    provider._last_output_mask_source = "online_sam2_guarded_primary"
    provider._last_online_sam2_exact_result = True
    try:
        accepted_boundary, boundary_reason = (
            provider._outward_boundary_clipping_accepts(
                current, _mask_result(exiting)
            )
        )
        assert accepted_boundary, boundary_reason
        assert provider._pending_full_target_geometry_commit is None

        accepted, detail = (
            provider._publication_candidate_agrees_with_committed_clean_mask(
                current, _mask_result(exiting)
            )
        )

        assert not accepted
        assert "raw_exact_boundary_visible_partial accepted" not in detail
        assert "appearance=" in detail
    finally:
        provider.stop()


def test_guarded_primary_boundary_partial_only_advances_visible_history(
    monkeypatch,
):
    full = _large_mask((130, 40, 170, 80))
    clipped = _large_mask((150, 40, 180, 80))
    reentered = full.copy()
    provider, _tracker = _guarded_primary_provider(
        monkeypatch,
        [
            _moving_target_frame(2, (150, 40, 180, 80)),
            _moving_target_frame(3, (130, 40, 170, 80)),
        ],
        {
            2: _video_result(2, clipped),
            3: _video_result(3, reentered),
        },
        full,
    )
    initial_full = copy.deepcopy(provider._full_target_geometry_authority)
    initial_trusted = copy.deepcopy(provider._trusted_online_sam2_publication)
    try:
        _frame2, result2 = provider.step_mask_only()
        assert result2.valid
        np.testing.assert_array_equal(result2.mask, clipped)
        assert provider._last_final_clean_publication.frame_id == 2
        assert provider._full_target_geometry_authority.frame_id == (
            initial_full.frame_id
        )
        assert provider._trusted_online_sam2_publication.frame_id == (
            initial_trusted.frame_id
        )
        assert provider._last_publication_guard_known_partial
        assert "boundary" in provider._last_publication_guard_partial_reason
        assert provider._trusted_online_sam2_transient_gap_active

        _frame3, result3 = provider.step_mask_only()
        assert result3.valid
        np.testing.assert_array_equal(result3.mask, reentered)
        assert provider._full_target_geometry_authority.frame_id == 3
        assert provider._trusted_online_sam2_publication.frame_id == 3
        assert provider._trusted_online_sam2_publication.velocity_samples == 0
        assert not provider._trusted_online_sam2_transient_gap_active
    finally:
        provider.stop()


def test_guarded_primary_boundary_continuation_publishes_only_supported_sliver(
    monkeypatch,
):
    full = _large_mask((130, 40, 170, 80))
    boundary_bboxes = {
        2: (145, 40, 175, 80),
        3: (150, 40, 178, 80),
        4: (154, 40, 180, 80),
        5: (156, 40, 180, 80),
        6: (158, 40, 180, 80),
        7: (160, 40, 180, 80),
        8: (162, 40, 180, 80),
    }
    masks = {frame_id: _large_mask(bbox) for frame_id, bbox in boundary_bboxes.items()}
    frames = []
    for frame_id, bbox in boundary_bboxes.items():
        frame = _moving_target_frame(frame_id, bbox)
        if frame_id >= 4:
            # Only 57.5% of the exact SAM silhouette has immutable target
            # colour.  This is below the normal boundary support floor (0.60)
            # while its mean probability remains above the unchanged global
            # identity floor (0.45).
            x1, _y1, x2, y2 = bbox
            frame.color_bgr[y2 - 17 : y2, x1:x2] = 45
        frames.append(frame)
    frames.append(_moving_target_frame(9, (0, 0, 0, 0)))
    stale_bbox = (164, 40, 180, 80)
    stale_mask = _large_mask(stale_bbox)
    stale_frame = _moving_target_frame(10, stale_bbox)
    stale_frame.color_bgr[63:80, 164:180] = 45
    frames.append(stale_frame)
    empty = np.zeros_like(full)
    track_results = {
        frame_id: _video_result(frame_id, mask) for frame_id, mask in masks.items()
    }
    track_results[9] = _video_result(9, empty, valid=False)
    track_results[10] = _video_result(10, stale_mask)
    provider, _tracker = _guarded_primary_provider(
        monkeypatch, frames, track_results, full
    )
    initial_full = copy.deepcopy(provider._full_target_geometry_authority)
    initial_trusted = copy.deepcopy(provider._trusted_online_sam2_publication)
    try:
        for expected_frame_id in (2, 3):
            frame, result = provider.step_mask_only()
            assert frame.frame_id == expected_frame_id
            assert result.valid
            np.testing.assert_array_equal(result.mask, masks[expected_frame_id])

        # Once immutable support occupies only one side of the raw boundary
        # silhouette, the raw bytes are never robot-facing.  The visible
        # appearance core remains publishable while it has enough continuity.
        for expected_frame_id in range(4, 8):
            frame, result = provider.step_mask_only()
            assert frame.frame_id == expected_frame_id
            assert result.valid
            assert result.source == "publication_sanitized_target_core"
            assert not np.array_equal(result.mask, masks[expected_frame_id])
            selected = result.mask > 0
            pixels = frame.color_bgr[selected]
            assert len(pixels) > 0
            assert np.all(pixels[:, 0] >= 180)
            assert np.all(pixels[:, 1] <= 120)
            assert np.all(pixels[:, 2] >= 180)

        # The final tiny sliver cannot prove a safe supported core and fails
        # closed instead of exposing the unsupported raw silhouette.
        frame8, result8 = provider.step_mask_only()
        assert frame8.frame_id == 8
        assert not result8.valid
        assert not np.any(result8.mask)
        assert provider._boundary_exit_continuity is None
        assert provider._last_final_clean_publication.frame_id == 7
        assert provider._full_target_geometry_authority.frame_id == (
            initial_full.frame_id
        )
        assert provider._trusted_online_sam2_publication.frame_id == (
            initial_trusted.frame_id
        )

        # A real empty SAM observation cannot manufacture an absent mask.
        frame9, result9 = provider.step_mask_only()
        assert frame9.frame_id == 9
        assert not result9.valid
        assert provider._boundary_exit_continuity is None
        assert provider._pending_boundary_exit_continuity is None

        # The pre-occlusion edge token is gone.  A later low-support mask must
        # start ordinary fail-closed recovery, never resume stale boundary
        # continuation across the invalid frame.
        frame10, result10 = provider.step_mask_only()
        assert frame10.frame_id == 10
        assert not result10.valid
        assert "boundary_continuation" not in (provider._last_online_sam2_status)
        assert provider._boundary_exit_continuity is None
    finally:
        provider.stop()


def test_guarded_primary_boundary_continuation_rejects_low_support_takeover(
    monkeypatch,
):
    full = _large_mask((130, 40, 170, 80))
    bboxes = {
        2: (145, 40, 175, 80),
        3: (150, 40, 178, 80),
        4: (154, 40, 180, 80),
    }
    masks = {frame_id: _large_mask(bbox) for frame_id, bbox in bboxes.items()}
    frames = [_moving_target_frame(frame_id, bbox) for frame_id, bbox in bboxes.items()]
    # 52.5% support still has mean probability >0.45 in the deterministic
    # scorer.  Mean alone must not let a hand-coloured takeover continue.
    frames[-1].color_bgr[40 + 21 : 80, 154:180] = 45
    provider, tracker = _guarded_primary_provider(
        monkeypatch,
        frames,
        {frame_id: _video_result(frame_id, mask) for frame_id, mask in masks.items()},
        full,
    )
    try:
        assert provider.step_mask_only()[1].valid
        assert provider.step_mask_only()[1].valid
        tracker.target_appearance_supported_mask = (
            lambda frame, mask, threshold=0.42: np.zeros_like(mask, dtype=np.uint8)
        )
        _frame4, result4 = provider.step_mask_only()

        assert not result4.valid
        assert provider._boundary_exit_continuity is None
        assert "support" in result4.message
    finally:
        provider.stop()


@pytest.mark.parametrize(
    (
        "reviewed_source",
        "supported_fraction",
        "previous_bbox",
        "candidate_bbox",
        "touches_boundary",
    ),
    (
        (117, 0.637, (120, 40, 155, 80), (125, 40, 160, 80), False),
        (119, 0.588, (130, 40, 170, 80), (150, 40, 180, 80), True),
        (120, 0.580, (130, 40, 170, 80), (150, 40, 180, 80), True),
        (122, 0.569, (130, 40, 170, 80), (150, 40, 180, 80), True),
        (123, 0.568, (130, 40, 170, 80), (150, 40, 180, 80), True),
    ),
)
def test_guarded_primary_reviewed_right_boundary_tail_reaches_final_sanitizer(
    monkeypatch,
    reviewed_source,
    supported_fraction,
    previous_bbox,
    candidate_bbox,
    touches_boundary,
):
    """The reviewed tail stays eligible only through its confirmed core."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    _install_visible_partial_boundary_history(
        provider,
        previous_bbox,
        frame_id=reviewed_source - 1,
        velocity_xy=(
            30.0 * (candidate_bbox[0] - previous_bbox[0]),
            0.0,
        ),
    )
    candidate = _large_mask(candidate_bbox)
    frame = _moving_target_frame(reviewed_source, candidate_bbox)
    geometry, reason = provider._online_mask_geometry(frame, candidate)
    assert geometry is not None, reason
    monkeypatch.setattr(
        provider,
        "_immutable_target_appearance_stats",
        lambda *_args, **_kwargs: (
            (0.60, supported_fraction, int(candidate.sum())),
            "reviewed source appearance",
        ),
    )
    try:
        accepted, detail = provider._visible_exact_partial_candidate_agrees(
            frame,
            geometry,
            allow_boundary_support=True,
        )

        assert accepted, detail
        assert f"boundary_support={touches_boundary}" in detail
        assert "sanitized_handoff=True" in detail
        # This is admission to the final appearance sanitizer, not raw-mask
        # publication or a promotion back to full/trusted authority.
        assert provider._trusted_online_sam2_publication is None
        assert not provider._trusted_online_sam2_continuity_intact
    finally:
        provider.stop()


def test_guarded_primary_confirmed_boundary_core_floor_remains_fail_closed(
    monkeypatch,
):
    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    _install_visible_partial_boundary_history(
        provider,
        (130, 40, 170, 80),
        frame_id=118,
        velocity_xy=(600.0, 0.0),
    )
    boundary = _large_mask((150, 40, 180, 80))
    boundary_frame = _moving_target_frame(119, (150, 40, 180, 80))
    boundary_geometry, reason = provider._online_mask_geometry(
        boundary_frame, boundary
    )
    assert boundary_geometry is not None, reason
    try:
        monkeypatch.setattr(
            provider,
            "_immutable_target_appearance_stats",
            lambda *_args, **_kwargs: (
                (0.60, 0.549, int(boundary.sum())),
                "below confirmed boundary core floor",
            ),
        )
        accepted, detail = provider._visible_exact_partial_candidate_agrees(
            boundary_frame,
            boundary_geometry,
            allow_boundary_support=True,
        )
        assert not accepted
        assert "0.549" in detail and "0.550" in detail

        # An un-sanitized raw owner has not earned the confirmed-core floor.
        provider._visible_exact_partial_continuity.evidence_kind = "raw_exact"
        provider._visible_exact_partial_continuity.sanitized_handoff_eligible = False
        monkeypatch.setattr(
            provider,
            "_immutable_target_appearance_stats",
            lambda *_args, **_kwargs: (
                (0.60, 0.568, int(boundary.sum())),
                "raw owner retains boundary entry floor",
            ),
        )
        accepted, detail = provider._visible_exact_partial_candidate_agrees(
            boundary_frame,
            boundary_geometry,
            allow_boundary_support=True,
        )
        assert not accepted
        assert "0.568" in detail and "0.600" in detail
    finally:
        provider.stop()


def test_guarded_primary_boundary_extra_hand_is_checked_before_exception(
    monkeypatch,
):
    previous = _large_mask((130, 40, 170, 80))
    target = _large_mask((150, 40, 180, 80))
    hand = _large_mask((140, 45, 150, 75))
    contaminated = target | hand
    current = _moving_target_frame(
        2,
        (150, 40, 180, 80),
        contaminant_bbox=(140, 45, 150, 75),
    )
    provider, _tracker = _guarded_primary_provider(
        monkeypatch,
        [current],
        {2: _video_result(2, contaminated)},
        previous,
    )
    initial_full = copy.deepcopy(provider._full_target_geometry_authority)
    try:
        _frame2, result = provider.step_mask_only()

        # Sanitizing to the clean target is acceptable; publishing the exact
        # 300px gray branch or establishing an edge token from it is not.
        if result.valid:
            assert not np.any(np.logical_and(result.mask > 0, hand > 0))
        assert not np.array_equal(result.mask, contaminated)
        assert provider._boundary_exit_continuity is None
        assert provider._full_target_geometry_authority.frame_id == (
            initial_full.frame_id
        )
    finally:
        provider.stop()


def test_guarded_primary_keeps_outward_exit_inside_effective_border(
    monkeypatch,
):
    previous = _large_mask((120, 40, 160, 80))
    exiting = _large_mask((140, 40, 168, 80))
    current = _moving_target_frame(2, (140, 40, 168, 80))
    provider, tracker = _guarded_primary_provider(
        monkeypatch,
        [current],
        {2: _video_result(2, exiting)},
        previous,
    )
    try:
        provider._last_output_mask_source = "online_sam2_guarded_primary"
        accepted, reason = provider._outward_boundary_clipping_accepts(
            current, _mask_result(exiting)
        )

        assert accepted, reason
        assert "edge=right" in reason
        assert tracker.token == "initialized"
    finally:
        provider.stop()


def test_guarded_primary_boundary_exit_does_not_exempt_hand_growth(
    monkeypatch,
):
    previous = _large_mask((130, 40, 170, 80))
    hand = _large_mask((170, 30, 180, 90))
    contaminated = np.maximum(previous, hand)
    current = _moving_target_frame(
        2,
        (130, 40, 170, 80),
        contaminant_bbox=(170, 30, 180, 90),
    )
    provider, _tracker = _guarded_primary_provider(
        monkeypatch,
        [current],
        {2: _video_result(2, contaminated)},
        previous,
    )
    try:
        provider._last_output_mask_source = "online_sam2_guarded_primary"
        accepted, reason = provider._outward_boundary_clipping_accepts(
            current, _mask_result(contaminated)
        )

        assert not accepted
        assert "grew area/bbox" in reason
    finally:
        provider.stop()


def test_guarded_primary_boundary_exit_does_not_exempt_hand_identity(
    monkeypatch,
):
    previous = _large_mask((130, 40, 170, 80))
    hand = _large_mask((150, 40, 180, 80))
    current = _moving_target_frame(
        2,
        (130, 40, 170, 80),
        contaminant_bbox=(150, 40, 180, 80),
    )
    # Remove target-colour pixels: this is a same-size hand takeover at the
    # effective edge, not a growing appendage.
    current.color_bgr[40:80, 130:150] = 45
    provider, _tracker = _guarded_primary_provider(
        monkeypatch,
        [current],
        {2: _video_result(2, hand)},
        previous,
    )
    try:
        provider._last_output_mask_source = "online_sam2_guarded_primary"
        accepted, reason = provider._outward_boundary_clipping_accepts(
            current, _mask_result(hand)
        )

        assert not accepted
        assert "boundary target appearance" in reason
    finally:
        provider.stop()


def test_guarded_primary_keeps_visible_core_under_inside_occlusion(
    monkeypatch,
):
    target = _large_mask((40, 30, 80, 70))
    visible = _large_mask((40, 30, 65, 70))
    current = _large_frame(2)
    current.color_bgr[:] = 45
    current.color_bgr[30:70, 40:65] = (210, 80, 235)
    current.color_bgr[30:70, 65:80] = (200, 200, 200)
    provider, tracker = _guarded_primary_provider(
        monkeypatch,
        [current],
        {2: _video_result(2, target)},
        target,
    )
    token_before = tracker.token
    try:
        _frame_out, result = provider.step_mask_only()
        assert result.valid
        np.testing.assert_array_equal(result.mask, visible)
        assert "appearance_core=True" in result.message
        assert tracker.token == token_before
    finally:
        provider.stop()


def test_guarded_primary_reject_does_not_advance_final_clean(monkeypatch):
    target = _large_mask((40, 30, 80, 70))
    current = _large_frame(
        2,
        contaminant_bbox=(40, 30, 80, 70),
        contaminant_bgr=(200, 200, 200),
    )
    current.color_bgr[30:70, 40:80] = (200, 200, 200)
    provider, _tracker = _guarded_primary_provider(
        monkeypatch,
        [current],
        {2: _video_result(2, target)},
        target,
    )
    try:
        _frame_out, result = provider.step_mask_only()
        assert not result.valid
        assert provider._last_final_clean_publication.frame_id == 1
        np.testing.assert_array_equal(
            provider._last_final_clean_publication.mask, target
        )
    finally:
        provider.stop()


def test_guarded_primary_recovery_uses_final_clean_dynamic_scale(
    monkeypatch,
):
    initial = _large_mask((20, 10, 100, 90))
    clean = _large_mask((40, 30, 80, 70))
    candidate = _large_mask((41, 31, 79, 69))
    provider, tracker = _guarded_primary_provider(
        monkeypatch,
        [],
        {},
        initial,
    )
    token_before = tracker.token
    try:
        clean_frame = _moving_target_frame(2, (40, 30, 80, 70))
        candidate_frame = _moving_target_frame(3, (41, 31, 79, 69))
        assert provider._commit_final_clean_publication(clean_frame, clean)
        # This test models an unsanitized exact-SAM publication, so both the
        # visible policy history and the full-target scale authority advance.
        assert provider._commit_full_target_geometry_authority(clean_frame, clean)
        assert provider._commit_trusted_online_sam2_publication(clean_frame, clean)
        provider._break_trusted_online_sam2_continuity("test timeout")
        geometry, reason = provider._online_mask_geometry(candidate_frame, candidate)
        assert geometry is not None, reason
        legacy_ok, legacy_reason = provider._online_recovery_geometry_accepts(geometry)
        (
            dynamic_ok,
            dynamic_reason,
        ) = provider._guarded_primary_recovery_geometry_accepts(
            candidate_frame, geometry
        )
        assert not legacy_ok
        assert "recovery area ratio" in legacy_reason
        assert dynamic_ok, dynamic_reason
        assert "final_clean" in dynamic_reason
        assert tracker.token == token_before
    finally:
        provider.stop()


@pytest.mark.parametrize(
    ("visible_bbox", "case_name"),
    [
        ((40, 30, 50, 70), "static_occlusion_core"),
        ((40, 30, 55, 70), "heavy_hand_occlusion_core"),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_guarded_primary_visible_core_cannot_shrink_recovery_scale_authority(
    monkeypatch, visible_bbox, case_name
):
    del case_name
    full = _large_mask((40, 30, 80, 70))
    visible = _large_mask(visible_bbox)
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, full)
    try:
        initial_authority = copy.deepcopy(provider._full_target_geometry_authority)
        occluded_frame = _moving_target_frame(2, visible_bbox)

        # This is the state transition made after a safely sanitized partial
        # publication: policy motion follows the visible pixels.
        assert provider._commit_final_clean_publication(occluded_frame, visible)
        assert provider._last_final_clean_publication.area == pytest.approx(
            float(visible.sum())
        )

        # It must not alter the separate full-object scale ruler.
        assert provider._full_target_geometry_authority.frame_id == (
            initial_authority.frame_id
        )
        assert provider._full_target_geometry_authority.area == pytest.approx(
            float(full.sum())
        )

        provider._break_trusted_online_sam2_continuity("test occlusion")
        reappearance_frame = _moving_target_frame(3, (40, 30, 80, 70))
        geometry, reason = provider._online_mask_geometry(reappearance_frame, full)
        assert geometry is not None, reason
        accepted, detail = provider._guarded_primary_recovery_geometry_accepts(
            reappearance_frame, geometry
        )
        assert accepted, detail
        assert "scale_ref=full_target_geometry" in detail
        assert "area_x=1.000" in detail
    finally:
        provider.stop()


@pytest.mark.parametrize("stale_log_area_rate", [-6.0, 6.0])
def test_guarded_primary_long_gap_zeros_stale_scale_derivative(
    monkeypatch, stale_log_area_rate
):
    target = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, target)
    try:
        authority = provider._full_target_geometry_authority
        authority.timestamp = 0.10
        authority.log_area_rate_s = stale_log_area_rate
        authority.scale_samples = 2
        recovery_frame = _moving_target_frame(30, (40, 30, 80, 70))
        recovery_frame.timestamp = 1.10
        geometry, reason = provider._online_mask_geometry(recovery_frame, target)
        assert geometry is not None, reason

        accepted, detail = provider._guarded_primary_recovery_geometry_accepts(
            recovery_frame, geometry
        )

        assert accepted, detail
        assert "scale_ref=full_target_geometry" in detail
        assert "scale_rate=zeroed_stale" in detail
        assert "area_x=1.000" in detail
    finally:
        provider.stop()


def test_guarded_primary_recovery_uses_centroid_and_hard_speed_gate(
    monkeypatch,
):
    target = _large_mask((40, 30, 80, 70))
    first = _large_mask((40, 30, 70, 70))
    first[30, 79] = 1
    second = _large_mask((50, 30, 80, 70))
    second[30, 40] = 1
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, target)
    try:
        frame_first = _moving_target_frame(2, (40, 30, 80, 70))
        frame_second = _moving_target_frame(3, (40, 30, 80, 70))
        frame_first.timestamp = 1.0
        frame_second.timestamp = 1.001
        first_geometry, _ = provider._online_mask_geometry(frame_first, first)
        second_geometry, _ = provider._online_mask_geometry(frame_second, second)
        assert first_geometry is not None and second_geometry is not None
        assert np.array_equal(first_geometry.bbox_xyxy, second_geometry.bbox_xyxy)
        confirmed, hits, _ = provider._confirm_online_recovery(
            2,
            first_geometry,
            timestamp=frame_first.timestamp,
            source="guarded_sam2_primary",
        )
        assert not confirmed and hits == 1
        confirmed, hits, reason = provider._confirm_online_recovery(
            3,
            second_geometry,
            timestamp=frame_second.timestamp,
            source="guarded_sam2_primary",
        )
        assert not confirmed and hits == 0
        assert "explicit conflict" in reason
        assert "speed=" in reason
    finally:
        provider.stop()


def test_guarded_primary_stale_final_clean_alignment_fails_closed(
    monkeypatch,
):
    target = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, target)
    try:
        stale_frame = _moving_target_frame(20, (40, 30, 80, 70))
        aligned, alignment_reason = provider._aligned_committed_clean_mask(
            stale_frame, target
        )
        assert aligned is None
        assert "outside the bounded device-time window" in alignment_reason
        (
            accepted,
            guard_reason,
        ) = provider._publication_candidate_agrees_with_committed_clean_mask(
            stale_frame, _mask_result(target)
        )
        assert not accepted
        assert "guarded final-clean alignment unavailable" in guard_reason
    finally:
        provider.stop()


def test_guarded_primary_empty_then_requires_two_sam_and_later_rgbd(
    monkeypatch,
):
    target = _large_mask((40, 30, 80, 70))
    frames = [
        _moving_target_frame(frame_id, (40, 30, 80, 70)) for frame_id in (2, 3, 4, 5)
    ]
    results = {
        2: _video_result(2, target, valid=False),
        3: _video_result(3, target),
        4: _video_result(4, target),
        5: _video_result(5, target),
    }
    provider, tracker = _guarded_primary_provider(monkeypatch, frames, results, target)
    token_before = tracker.token
    try:
        outputs = [provider.step_mask_only()[1] for _ in frames]
        assert [result.valid for result in outputs] == [
            False,
            False,
            False,
            True,
        ], "\n".join(result.message for result in outputs)
        assert "pending 1/2" in outputs[1].message
        assert "awaiting one distinct RGB-D frame" in outputs[2].message
        assert provider._last_final_clean_publication.frame_id == 5
        assert tracker.token == token_before
    finally:
        provider.stop()


def test_guarded_primary_later_rgbd_uses_fresh_provisional_sam_clean(
    monkeypatch,
):
    """A stale final clean cannot deadlock an otherwise complete recovery."""

    target = _large_mask((40, 30, 80, 70))
    frame_ids = (2, 4, 6, 8)
    frames = [
        _moving_target_frame(frame_id, (40, 30, 80, 70)) for frame_id in frame_ids
    ]
    results = {
        2: _video_result(2, target, valid=False),
        4: _video_result(4, target),
        6: _video_result(6, target),
        8: _video_result(8, target),
    }
    provider, tracker = _guarded_primary_provider(monkeypatch, frames, results, target)
    token_before = tracker.token
    try:
        outputs = [provider.step_mask_only()[1] for _ in frames]
        assert [result.valid for result in outputs] == [
            False,
            False,
            False,
            True,
        ], "\n".join(result.message for result in outputs)
        assert provider._last_final_clean_publication.frame_id == 8
        assert "recovery publication stable 3/3" in outputs[-1].message
        assert "alignment=recovery_exact_sam2" in (
            provider._last_publication_guard_status
        )
        assert tracker.token == token_before
    finally:
        provider.stop()


def test_guarded_primary_long_gap_two_sam_then_later_rgbd_recovers(
    monkeypatch,
):
    """A real LOST interval cannot deadlock the complete 2-SAM+RGB-D chain."""

    initial = _large_mask((40, 30, 80, 70))
    recovered = _large_mask((130, 30, 170, 70))
    frames = [
        _moving_target_frame(frame_id, (130, 30, 170, 70)) for frame_id in (30, 31, 32)
    ]
    for frame, timestamp in zip(frames, (1.00, 1.05, 1.10)):
        frame.timestamp = timestamp
    provider, tracker = _guarded_primary_provider(
        monkeypatch,
        frames,
        {
            30: _video_result(30, recovered),
            31: _video_result(31, recovered),
            32: _video_result(32, recovered),
        },
        initial,
    )
    token_before = tracker.token
    try:
        outputs = [provider.step_mask_only()[1] for _ in frames]
        assert [result.valid for result in outputs] == [
            False,
            False,
            True,
        ], "\n".join(result.message for result in outputs)
        assert "awaiting one distinct RGB-D frame" in outputs[1].message
        assert "recovery publication stable 3/3" in outputs[2].message
        np.testing.assert_array_equal(outputs[2].mask, recovered)
        assert provider._last_final_clean_publication.frame_id == 32
        assert tracker.token == token_before
    finally:
        provider.stop()


@pytest.mark.parametrize("conflict_kind", ["hand_growth", "teleport"])
def test_guarded_primary_long_gap_conflict_never_gets_probation_bypass(
    monkeypatch,
    conflict_kind,
):
    """Stale alignment never turns hand growth or a teleport into recovery."""

    initial = _large_mask((40, 30, 80, 70))
    if conflict_kind == "hand_growth":
        conflict = initial | _large_mask((75, 10, 175, 100))
        frames = [
            _moving_target_frame(
                frame_id,
                (40, 30, 80, 70),
                contaminant_bbox=(75, 10, 175, 100),
            )
            for frame_id in (30, 31, 32)
        ]
    else:
        conflict = _large_mask((155, 30, 175, 70))
        frames = [
            _moving_target_frame(frame_id, (155, 30, 175, 70))
            for frame_id in (30, 31, 32)
        ]
        # Make the reacquisition displacement cap explicit so this remains a
        # focused unit test if the synthetic image dimensions change.
    for frame, timestamp in zip(frames, (1.00, 1.05, 1.10)):
        frame.timestamp = timestamp
    provider, tracker = _guarded_primary_provider(
        monkeypatch,
        frames,
        {
            30: _video_result(30, conflict),
            31: _video_result(31, conflict),
            32: _video_result(32, conflict),
        },
        initial,
    )
    if conflict_kind == "teleport":
        provider.online_sam2_cfg[
            "trusted_recovery_max_displacement_image_diagonal_ratio"
        ] = 0.40
    token_before = tracker.token
    try:
        outputs = [provider.step_mask_only()[1] for _ in frames]
        assert not any(result.valid for result in outputs)
        assert provider._recovery_publication_hypothesis is None
        assert provider._online_sam2_recovery_hypothesis is None
        assert provider._last_final_clean_publication.frame_id == 1
        assert tracker.token == token_before
        combined = "\n".join(result.message for result in outputs)
        assert "probation_bypass" not in combined
        if conflict_kind == "hand_growth":
            assert "area ratio" in combined or "bbox ratio" in combined
        else:
            assert "center residual" in combined
    finally:
        provider.stop()


@pytest.mark.parametrize("third_kind", ["hand", "far_same_colour"])
def test_guarded_primary_third_rgbd_conflict_clears_provisional_clean(
    monkeypatch,
    third_kind,
):
    """The provisional SAM clean never lets a conflicting third frame pass."""

    target = _large_mask((40, 30, 80, 70))
    third_mask = target.copy()
    frames = [
        _moving_target_frame(frame_id, (40, 30, 80, 70)) for frame_id in (2, 4, 6)
    ]
    if third_kind == "hand":
        third = _large_frame(8)
        third.color_bgr[30:70, 40:80] = (200, 200, 200)
    else:
        third_mask = _large_mask((120, 30, 160, 70))
        third = _moving_target_frame(8, (120, 30, 160, 70))
        # Make the third exact-frame candidate physically impossible even
        # though it has the same target colour and shape.
        third.timestamp = frames[-1].timestamp + 0.01
    frames.append(third)
    results = {
        2: _video_result(2, target, valid=False),
        4: _video_result(4, target),
        6: _video_result(6, target),
        8: _video_result(8, third_mask),
    }
    provider, tracker = _guarded_primary_provider(monkeypatch, frames, results, target)
    token_before = tracker.token
    try:
        outputs = [provider.step_mask_only()[1] for _ in frames]
        assert not any(result.valid for result in outputs)
        assert provider._recovery_publication_hypothesis is None
        assert provider._online_sam2_recovery_hypothesis is None
        assert provider._last_final_clean_publication.frame_id == 1
        assert tracker.token == token_before
        if third_kind == "hand":
            assert "appearance" in outputs[-1].message
        else:
            assert "speed" in outputs[-1].message
    finally:
        provider.stop()


def test_guarded_primary_motionless_bootstrap_allows_blurred_seed_scale(
    monkeypatch,
):
    """The first exact mask may be smaller than a motion-blurred prompt."""

    initial = _large_mask((10, 10, 110, 110))
    candidate = _large_mask((80, 40, 135, 95))
    provider, tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    token_before = tracker.token
    try:
        frame = _moving_target_frame(3, (80, 40, 135, 95))
        geometry, reason = provider._online_mask_geometry(frame, candidate)
        assert geometry is not None, reason
        accepted, detail = provider._guarded_primary_recovery_geometry_accepts(
            frame, geometry
        )
        assert accepted, detail
        assert "area_x=0.302" in detail
        assert provider._last_final_clean_publication.motion_samples == 0
        assert tracker.token == token_before
    finally:
        provider.stop()


def test_guarded_primary_occlusion_recovery_publishes_only_visible_core(
    monkeypatch,
):
    target = _large_mask((40, 30, 80, 70))
    visible = _large_mask((40, 30, 65, 70))
    frames = [_moving_target_frame(2, (40, 30, 80, 70))]
    for frame_id in (3, 4, 5):
        frame = _large_frame(frame_id)
        frame.color_bgr[:] = 45
        frame.color_bgr[30:70, 40:65] = (210, 80, 235)
        frame.color_bgr[30:70, 65:80] = (200, 200, 200)
        frames.append(frame)
    results = {
        2: _video_result(2, target, valid=False),
        3: _video_result(3, target),
        4: _video_result(4, target),
        5: _video_result(5, target),
    }
    provider, tracker = _guarded_primary_provider(monkeypatch, frames, results, target)
    token_before = tracker.token
    try:
        outputs = [provider.step_mask_only()[1] for _ in frames]
        assert [result.valid for result in outputs] == [
            False,
            False,
            False,
            True,
        ], "\n".join(result.message for result in outputs)
        np.testing.assert_array_equal(outputs[-1].mask, visible)
        assert "appearance_core=True" in outputs[-1].message
        assert tracker.token == token_before
        assert provider._last_final_clean_publication.frame_id == 5
    finally:
        provider.stop()


def test_mask_only_publication_evidence_bbox_roi_is_elementwise_equivalent(
    monkeypatch,
):
    height, width = 64, 80
    rows, cols = np.indices((height, width))
    depth_raw = (650 + ((3 * cols + 5 * rows) % 151)).astype(np.uint16)
    depth_raw[(7 * cols + rows) % 29 == 0] = 0
    frame = RGBDFrame(
        color_bgr=np.zeros((height, width, 3), dtype=np.uint8),
        depth_raw=depth_raw,
        depth_scale=0.001,
        intrinsics=CameraIntrinsics(
            width=width,
            height=height,
            fx=91.0,
            fy=93.0,
            ppx=39.5,
            ppy=31.5,
        ),
        timestamp=1.0,
        frame_id=30,
    )
    provider, _camera = _provider(
        monkeypatch,
        [frame],
        _FakeTracker({}),
        _FakeOnlineSAM2Manager(),
        start=False,
    )
    provider.extractor.T_base_camera = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.12],
            [0.0, 1.0, 0.0, -0.04],
            [0.0, 0.0, 1.0, 0.02],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    empty = np.zeros((height, width), dtype=np.uint8)
    interior = empty.copy()
    interior[17:48, 23:62] = 1
    top_left_edge = empty.copy()
    top_left_edge[:27, :34] = 1
    bottom_right_edge = empty.copy()
    bottom_right_edge[38:, 48:] = 1
    multiple_components = empty.copy()
    multiple_components[3:17, 5:24] = 1
    multiple_components[29:57, 42:73] = 1
    equal_components = empty.copy()
    equal_components[4:22, 3:25] = 1
    equal_components[36:54, 50:72] = 1
    erodes_to_empty = empty.copy()
    erodes_to_empty[20:22, 20:42] = 1
    masks = {
        "empty": empty,
        "interior": interior,
        "top_left_edge": top_left_edge,
        "bottom_right_edge": bottom_right_edge,
        "multiple_components": multiple_components,
        "equal_components": equal_components,
        "erodes_to_empty": erodes_to_empty,
    }

    try:
        for erode_kernel in (0, 1, 3, 5):
            for stride in (1, 2, 3):
                provider.extractor.cfg = {
                    "erode_kernel": erode_kernel,
                    "stride": stride,
                    "z_min": 0.50,
                    "z_max": 0.90,
                }
                for case, mask in masks.items():
                    (
                        expected_valid,
                        expected_message,
                        expected_center,
                        _count,
                    ) = _full_frame_mask_gate_reference(provider, frame, mask)
                    actual = provider._mask_only_publication_evidence(frame, mask)
                    context = (
                        f"case={case}, erode_kernel={erode_kernel}, " f"stride={stride}"
                    )
                    assert actual.valid is expected_valid, context
                    assert actual.message == expected_message, context
                    np.testing.assert_array_equal(
                        actual.center, expected_center, err_msg=context
                    )
                    assert actual.points.shape == (0, 3), context
                    assert actual.reference_points.shape == (0, 3), context
    finally:
        provider.stop()


def test_mask_only_step_fails_closed_when_mask_has_no_usable_depth(
    monkeypatch,
):
    seed = _mask(4, 5, 12, 15)
    adaptive = _mask(5, 5, 13, 15)
    current = _frame(2)
    current.depth_raw[adaptive.astype(bool)] = 0
    tracker = _FakeTracker({2: _mask_result(adaptive)})
    manager = _FakeOnlineSAM2Manager()
    provider, _camera = _provider(monkeypatch, [current], tracker, manager)
    assert provider.initialize_from_mask(_frame(1), seed)

    provider.extractor.extract = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("mask-only step called extractor.extract")
    )
    resets_before = provider.history.reset_calls

    _frame_out, result = provider.step_mask_only()

    assert not result.valid
    assert int(result.mask.sum()) == 0
    assert "too few valid depth pixels" in result.message
    assert not provider.tracking_committed
    assert provider.history.reset_calls == resets_before + 1
    assert provider.last_packet is None
    provider.stop()


def test_mask_only_step_keeps_shared_three_frame_recovery_quarantine(
    monkeypatch,
):
    seed = _mask(4, 5, 12, 15)
    recovered = _mask(7, 5, 15, 15)
    tracker = _FakeTracker(
        {
            2: _mask_result(recovered, message="tentative recovery"),
            3: _mask_result(recovered, message="tentative recovery"),
            4: _mask_result(recovered, message="tentative recovery"),
        }
    )
    manager = _FakeOnlineSAM2Manager()
    frames = [_frame(2), _frame(3), _frame(4)]
    # Consecutive accepted frames may be 133 ms apart after the D435 skips
    # queued sensor frames. Spatially/depth-consistent recovery evidence must
    # remain confirmable below the formal 200 ms publication-stall bound.
    for frame, timestamp in zip(frames, (0.200, 0.333, 0.466)):
        frame.timestamp = timestamp
    provider, _camera = _provider(monkeypatch, frames, tracker, manager)
    assert provider.initialize_from_mask(_frame(1), seed)
    assert provider.reinitialize_same_target_from_mask(
        _frame(1), recovered, source="test_recovery"
    )
    # This test isolates the provider-owned final publication gate; online SAM
    # fusion is covered independently above.
    provider._online_sam2_ready = False
    provider._online_sam2_initialized = False

    provider.extractor.extract = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("mask-only step called extractor.extract")
    )
    history_updates = 0

    def forbidden_history_update(*_args, **_kwargs):
        nonlocal history_updates
        history_updates += 1
        raise AssertionError("mask-only step updated provider PCD history")

    provider.history.update = forbidden_history_update
    results = [provider.step_mask_only()[1] for _ in range(3)]

    assert [result.valid for result in results] == [False, False, True]
    assert all(int(result.mask.sum()) == 0 for result in results[:2])
    assert "publication pending 1/3" in results[0].message.lower()
    assert "publication pending 2/3" in results[1].message.lower()
    assert "publication stable 3/3" in results[2].message.lower()
    np.testing.assert_array_equal(results[2].mask, recovered)
    assert provider.tracking_committed
    assert history_updates == 0
    assert provider.last_packet is None
    provider.stop()


def test_tracked_output_uses_sam2_when_exact_and_adaptive_on_temporal_miss(
    monkeypatch,
):
    seed = _mask(4, 5, 12, 15)
    adaptive = _mask(5, 5, 13, 15)  # 80 px
    temporal = _mask(4, 4, 14, 14)  # 100 px, legal 1.25x contour
    tracker = _FakeTracker(
        {
            2: _mask_result(adaptive),
            3: _mask_result(adaptive),
            4: _mask_result(adaptive),
        }
    )
    manager = _FakeOnlineSAM2Manager(
        {
            2: _video_result(2, temporal),
            3: _video_result(
                3,
                np.zeros_like(seed),
                valid=False,
                message="temporary temporal miss",
                internal_frame_idx=2,
            ),
            4: _video_result(4, temporal),
        }
    )
    provider, _camera = _provider(
        monkeypatch, [_frame(2), _frame(3), _frame(4)], tracker, manager
    )
    assert provider.initialize_from_mask(_frame(1), seed)

    outputs = [provider.step() for _ in range(3)]

    # Exact-empty temporal evidence fail-closes the occluded frame. The first
    # later silhouette is quarantined rather than immediately reviving a
    # possibly hand/background-locked adaptive track.
    expected_masks = (
        temporal,
        np.zeros_like(adaptive),
        np.zeros_like(temporal),
    )
    expected_sources = (
        "online_sam2_tracked",
        "none",
        # The minimal fake tracker has no immutable appearance scorer, so the
        # first post-empty SAM mask is rejected rather than counted.  The
        # production identity-aware path is covered by the 2-SAM+1-RGB-D test.
        "none",
    )
    for (
        index,
        ((_frame_out, result, obj, packet), expected_mask, expected_source),
    ) in enumerate(zip(outputs, expected_masks, expected_sources)):
        np.testing.assert_array_equal(result.mask, expected_mask)
        assert (obj.valid and packet.valid) is (index == 0)
        assert packet.debug["mask_source"] == expected_source
    assert [call[1] for call in manager.calls if call[0] == "track"] == [
        2,
        3,
        4,
    ]
    provider.stop()


def test_hand_union_rejected_without_overriding_valid_adaptive(
    monkeypatch,
):
    target_mask = _large_mask((40, 30, 80, 70))
    contaminant_bbox = (80, 25, 120, 75)
    online_mask = target_mask | _large_mask(contaminant_bbox)
    assert int(target_mask.sum()) == 1600
    assert int(online_mask.sum()) == 3600

    tracker = AdaptiveColorDepthTracker(load_config()["tracker"])
    manager = _FakeOnlineSAM2Manager()
    current = _large_frame(
        2,
        contaminant_bbox=contaminant_bbox,
    )
    provider, _camera = _provider(monkeypatch, [current], tracker, manager)
    provider.cfg["tracker"]["recovery_enabled"] = True
    assert provider.initialize_from_mask(_large_frame(1), target_mask)

    adaptive_result = tracker.update(current)
    assert adaptive_result.valid, adaptive_result.message
    assert int(adaptive_result.mask.sum()) == 1600

    # The sample-count-invariant appearance prior now rejects this physical
    # target+hand signature at the raw tracker gate.  The provider consensus
    # remains defense in depth and must still preserve the clean adaptive
    # result without mutating appearance/state.
    before_raw_gate = tracker.snapshot_state()
    raw_result = tracker.reinitialize_with_mask(current, online_mask)
    assert not raw_result.valid
    assert raw_result.mask.sum() == 0
    assert "color=" in raw_result.message
    tracker.restore_state(before_raw_gate)

    state_before = tracker.snapshot_state()
    template_before = copy.deepcopy(provider.recovery_template)
    template_mask_before = provider.recovery_template_mask.copy()
    resets_before = provider.history.reset_calls
    result, recovered = provider._fuse_online_sam2_candidate(
        current,
        adaptive_result,
        _video_result(2, online_mask),
    )

    assert not recovered
    np.testing.assert_array_equal(result.mask, adaptive_result.mask)
    _assert_adaptive_snapshot_equal(state_before, tracker.snapshot_state())
    np.testing.assert_array_equal(provider.recovery_template, template_before)
    np.testing.assert_array_equal(provider.recovery_template_mask, template_mask_before)
    assert provider.history.reset_calls == resets_before
    provider.stop()


def test_small_low_appearance_hand_appendage_falls_back_to_clean_adaptive(
    monkeypatch,
):
    """Catch the observed ball-majority mask with a thin neutral finger.

    Whole-mask area, overlap and mean appearance all remain plausible in this
    case.  Only the SAM-only branch has the wrong immutable appearance, so it
    must not enter either the policy mask or the next adaptive state.
    """

    target_mask = _large_mask((40, 30, 80, 70))
    hand_bbox = (80, 45, 94, 55)
    online_mask = target_mask | _large_mask(hand_bbox)
    assert int(target_mask.sum()) == 1600
    assert int(online_mask.sum()) == 1740

    tracker = AdaptiveColorDepthTracker(load_config()["tracker"])
    manager = _FakeOnlineSAM2Manager()
    current = _large_frame(2, contaminant_bbox=hand_bbox)
    provider, _camera = _provider(monkeypatch, [current], tracker, manager)
    assert provider.initialize_from_mask(_large_frame(1), target_mask)

    adaptive_result = tracker.update(current)
    assert adaptive_result.valid, adaptive_result.message
    assert int(adaptive_result.mask.sum()) == 1600

    # Reinitializing from the complete union still passes the historical
    # whole-mask appearance gate, which is why the branch-specific guard is
    # required.
    before_raw_gate = tracker.snapshot_state()
    raw_result = tracker.reinitialize_with_mask(current, online_mask)
    assert raw_result.valid, raw_result.message
    tracker.restore_state(before_raw_gate)

    state_before = tracker.snapshot_state()
    result, recovered = provider._fuse_online_sam2_candidate(
        current,
        adaptive_result,
        _video_result(2, online_mask),
    )

    assert not recovered
    np.testing.assert_array_equal(result.mask, adaptive_result.mask)
    assert "foreign appendage" in provider._last_online_sam2_status
    _assert_adaptive_snapshot_equal(state_before, tracker.snapshot_state())
    provider.stop()


def test_disagreeing_adaptive_fails_closed_until_two_sam_plus_one_rgbd(
    monkeypatch,
):
    """A teleport enters probation; two exact SAM frames never publish alone."""

    seed = _large_mask((5, 30, 25, 50))
    # Match the reviewed fast-entry recording (~2134 px/s) while remaining
    # inside the commissioned 2400 px/s identity-motion envelope.
    sam2 = _large_mask((75, 30, 95, 50))  # 2100 px/s from seed.
    sam3 = _large_mask((85, 30, 105, 50))
    sam4 = _large_mask((95, 30, 115, 50))
    wrong_adaptive = _large_mask((5, 5, 75, 75))
    empty = np.zeros_like(seed)
    tracker = _IdentityAwareFakeTracker(
        {
            2: _mask_result(wrong_adaptive, message="wrong large adaptive"),
            3: _mask_result(empty, valid=False, message="adaptive lost"),
            4: _mask_result(sam4, message="tentative correct adaptive"),
        }
    )
    manager = _FakeOnlineSAM2Manager(
        {
            2: _video_result(2, sam2),
            3: _video_result(3, sam3),
            4: _video_result(4, sam4),
        }
    )
    frames = [
        _moving_target_frame(2, (75, 30, 95, 50)),
        _moving_target_frame(3, (85, 30, 105, 50)),
        _moving_target_frame(4, (95, 30, 115, 50)),
    ]
    provider, _camera = _provider(
        monkeypatch,
        frames,
        tracker,
        manager,
        recovery_publication_mode="unified_three_evidence",
    )
    provider.online_sam2_cfg["mask_publication_mode"] = "guarded_sam2_primary"
    assert provider.initialize_from_mask(_moving_target_frame(1, (5, 30, 25, 50)), seed)
    provider._break_trusted_online_sam2_continuity("test entered LOST")
    provider._tracking_committed = False
    template_before = copy.deepcopy(provider.recovery_template)

    output2 = provider.step()
    snapshot_after_first = tracker.snapshot_state()
    template_after_first = copy.deepcopy(provider.recovery_template)
    initialize_count_after_first = [call[0] for call in manager.calls].count(
        "initialize"
    )
    output3 = provider.step()
    output4 = provider.step()

    assert not output2[1].valid and not output2[3].valid
    assert not output3[1].valid and not output3[3].valid
    assert output4[1].valid and output4[3].valid
    np.testing.assert_array_equal(output4[1].mask, sam4)
    # Guarded-primary recovery never mutates the adaptive owner: the single
    # tentative SAM did not become adaptive/template/remote-session memory.
    np.testing.assert_array_equal(snapshot_after_first.state.last_mask, seed)
    assert snapshot_after_first.state.valid is True
    assert snapshot_after_first.token == "initialized"
    if template_before is None:
        assert template_after_first is None
    else:
        np.testing.assert_array_equal(template_after_first, template_before)
    assert initialize_count_after_first == 1
    provider.stop()


def test_trusted_sam_owner_accepts_fast_translation_and_gradual_scale(
    monkeypatch,
):
    """Published SAM history, not a swollen adaptive area, owns consensus."""

    seed = _large_mask((5, 30, 25, 50))
    sam2 = _large_mask((65, 28, 89, 52))
    sam3 = _large_mask((120, 26, 148, 54))
    wrong_adaptive = _large_mask((5, 5, 105, 95))
    tracker = _IdentityAwareFakeTracker(
        {
            2: _mask_result(sam2, message="correct fast adaptive"),
            3: _mask_result(wrong_adaptive, message="swollen stale adaptive"),
        }
    )
    manager = _FakeOnlineSAM2Manager(
        {2: _video_result(2, sam2), 3: _video_result(3, sam3)}
    )
    frames = [
        _moving_target_frame(2, (65, 28, 89, 52)),
        _moving_target_frame(3, (120, 26, 148, 54)),
    ]
    provider, _camera = _provider(monkeypatch, frames, tracker, manager)
    assert provider.initialize_from_mask(_moving_target_frame(1, (5, 30, 25, 50)), seed)

    output2 = provider.step()
    output3 = provider.step()

    assert output2[1].valid and output2[3].valid
    assert output3[1].valid and output3[3].valid
    np.testing.assert_array_equal(output2[1].mask, sam2)
    np.testing.assert_array_equal(output3[1].mask, sam3)
    assert output3[3].debug["mask_source"] == ("online_sam2_trusted_override")
    assert "trusted_override" in provider._last_online_sam2_status
    provider.stop()


def test_continuous_trusted_sam_seamlessly_replaces_invalid_adaptive(
    monkeypatch,
):
    target1 = _large_mask((40, 30, 80, 70))
    target2 = _large_mask((45, 30, 85, 70))
    empty = np.zeros_like(target1)
    tracker = _IdentityAwareFakeTracker(
        {2: _mask_result(empty, valid=False, message="adaptive lost")}
    )
    manager = _FakeOnlineSAM2Manager({2: _video_result(2, target2)})
    provider, _camera = _provider(
        monkeypatch,
        [_moving_target_frame(2, (45, 30, 85, 70))],
        tracker,
        manager,
        recovery_publication_mode="unified_three_evidence",
    )
    assert provider.initialize_from_mask(
        _moving_target_frame(1, (40, 30, 80, 70)), target1
    )

    _frame_out, result, obj, packet = provider.step()

    assert result.valid and obj.valid and packet.valid
    np.testing.assert_array_equal(result.mask, target2)
    assert packet.debug["mask_source"] == "online_sam2_trusted_override"
    assert packet.debug["online_sam2_trusted_continuity_intact"] is True
    assert provider._trusted_online_sam2_publication.frame_id == 2
    provider.stop()


def test_exact_empty_requires_two_sam_plus_later_rgbd_before_recovery(
    monkeypatch,
):
    target1 = _large_mask((40, 30, 80, 70))
    target3 = _large_mask((42, 30, 82, 70))
    target4 = _large_mask((43, 30, 83, 70))
    target5 = _large_mask((44, 30, 84, 70))
    empty = np.zeros_like(target1)
    tracker = _IdentityAwareFakeTracker(
        {
            2: _mask_result(target1, message="clean adaptive"),
            3: _mask_result(target3, message="adaptive visible again"),
            4: _mask_result(target4, message="adaptive visible again"),
            5: _mask_result(target5, message="later RGB-D candidate"),
        }
    )
    manager = _FakeOnlineSAM2Manager(
        {
            2: _video_result(2, empty, valid=False, message="occluded"),
            3: _video_result(3, target3),
            4: _video_result(4, target4),
            5: _video_result(5, target5),
        }
    )
    frames = [
        _moving_target_frame(2, (40, 30, 80, 70)),
        _moving_target_frame(3, (42, 30, 82, 70)),
        _moving_target_frame(4, (43, 30, 83, 70)),
        _moving_target_frame(5, (44, 30, 84, 70)),
    ]
    provider, _camera = _provider(
        monkeypatch,
        frames,
        tracker,
        manager,
        recovery_publication_mode="unified_three_evidence",
    )
    provider.online_sam2_cfg["mask_publication_mode"] = "guarded_sam2_primary"
    assert provider.initialize_from_mask(
        _moving_target_frame(1, (40, 30, 80, 70)), target1
    )

    output2 = provider.step()
    output3 = provider.step()
    output4 = provider.step()
    assert not output2[1].valid and not output2[3].valid
    assert not output3[1].valid and not output3[3].valid
    assert not output4[1].valid and not output4[3].valid
    assert provider._trusted_online_sam2_continuity_intact is False
    assert provider._trusted_online_sam2_publication.frame_id == 1

    output5 = provider.step()
    assert output5[1].valid and output5[3].valid
    np.testing.assert_array_equal(output5[1].mask, target5)
    # Recovery is deliberately non-authoritative: the complete three-frame
    # proof may reopen robot-facing visibility, but it cannot silently repair
    # the older trusted/full exact-SAM ruler.
    assert provider._trusted_online_sam2_continuity_intact is False
    assert provider._trusted_online_sam2_publication.frame_id == 1
    partial = provider._visible_exact_partial_continuity
    assert partial is not None
    assert partial.frame_id == 5
    assert partial.hits == 2
    assert partial.evidence_kind == "recovery_raw_exact_partial"
    provider.stop()


def test_rpc_break_blocks_single_frame_trusted_takeover(monkeypatch):
    target1 = _large_mask((40, 30, 80, 70))
    target2 = _large_mask((42, 30, 82, 70))
    empty = np.zeros_like(target1)
    tracker = _IdentityAwareFakeTracker({})
    provider, _camera = _provider(
        monkeypatch,
        [_moving_target_frame(2, (42, 30, 82, 70))],
        tracker,
        _FakeOnlineSAM2Manager(),
        recovery_publication_mode="unified_three_evidence",
    )
    assert provider.initialize_from_mask(
        _moving_target_frame(1, (40, 30, 80, 70)), target1
    )
    provider._mark_online_sam2_rpc_failure("synthetic disconnect")

    result, recovered = provider._fuse_online_sam2_candidate(
        _moving_target_frame(2, (42, 30, 82, 70)),
        _mask_result(empty, valid=False, message="adaptive lost"),
        _video_result(2, target2),
    )

    assert not result.valid and not recovered
    assert provider._trusted_online_sam2_continuity_intact is False
    assert provider._online_sam2_recovery_hypothesis is not None
    assert provider._online_sam2_recovery_hypothesis.hits == 1
    assert "continuity is broken" in provider._last_online_sam2_status
    provider.stop()


def test_late_exact_empty_also_breaks_trusted_continuity(monkeypatch):
    target = _large_mask((40, 30, 80, 70))
    tracker = _IdentityAwareFakeTracker({})
    provider, _camera = _provider(
        monkeypatch,
        [_moving_target_frame(2, (40, 30, 80, 70))],
        tracker,
        _FakeOnlineSAM2Manager(),
    )
    assert provider.initialize_from_mask(
        _moving_target_frame(1, (40, 30, 80, 70)), target
    )
    completed = Future()
    completed.set_result(
        (_video_result(2, np.zeros_like(target), valid=False), 3.0, None)
    )
    provider._online_sam2_future = completed
    provider._online_sam2_future_kind = "track"
    provider._online_sam2_future_frame_id = 2

    provider._harvest_completed_online_future()

    assert provider._trusted_online_sam2_continuity_intact is False
    assert "late exact SAM2 empty" in (provider._trusted_online_sam2_continuity_reason)
    assert provider._trusted_online_sam2_publication.frame_id == 1
    provider.stop()


def test_pure_adaptive_recovery_does_not_repair_trusted_sam_continuity(
    monkeypatch,
):
    target = _large_mask((40, 30, 80, 70))
    tracker = _IdentityAwareFakeTracker({})
    provider, _camera = _provider(
        monkeypatch,
        [_moving_target_frame(2, (40, 30, 80, 70))],
        tracker,
        _FakeOnlineSAM2Manager(),
        recovery_publication_mode="unified_three_evidence",
    )
    assert provider.initialize_from_mask(
        _moving_target_frame(1, (40, 30, 80, 70)), target
    )
    provider._break_trusted_online_sam2_continuity("synthetic exact empty")
    provider._tracking_committed = False
    outputs = []
    for frame_id in (2, 3, 4):
        frame = _moving_target_frame(frame_id, (40, 30, 80, 70))
        provider._last_output_mask_source = "adaptive"
        provider._current_recovery_evidence_outcome = "candidate"
        outputs.append(
            provider._apply_recovery_publication_gate(
                frame,
                _mask_result(target, message="pure adaptive recovery"),
                provider.extractor.extract(frame, target),
            )
        )

    assert not outputs[0][0].valid
    assert not outputs[1][0].valid
    assert outputs[2][0].valid and outputs[2][1].valid
    assert provider.tracking_committed
    assert provider._trusted_online_sam2_continuity_intact is False
    assert provider._trusted_online_sam2_publication.frame_id == 1
    provider.stop()


def test_stale_trusted_mask_does_not_start_unbounded_registration(
    monkeypatch,
):
    target = _large_mask((40, 30, 80, 70))
    tracker = _IdentityAwareFakeTracker({})
    provider, _camera = _provider(
        monkeypatch,
        [_moving_target_frame(100, (40, 30, 80, 70))],
        tracker,
        _FakeOnlineSAM2Manager(),
    )
    assert provider.initialize_from_mask(
        _moving_target_frame(1, (40, 30, 80, 70)), target
    )

    def unexpected_registration(*_args, **_kwargs):
        raise AssertionError("stale trusted mask entered overlap registration")

    monkeypatch.setattr(
        provider, "_mask_overlap_registration_shift", unexpected_registration
    )
    frame100 = _moving_target_frame(100, (40, 30, 80, 70))
    geometry, reason = provider._online_mask_geometry(frame100, target)
    assert geometry is not None, reason
    accepted, reason = provider._online_sam2_independent_identity_accepts(
        frame100, geometry
    )
    assert accepted, reason
    provider.stop()


def test_hand_sam_candidate_never_recovers_or_advances_trusted_history(
    monkeypatch,
):
    target = _large_mask((40, 30, 80, 70))
    hand_bbox = (80, 45, 100, 60)
    hand_union = target | _large_mask(hand_bbox)
    tracker = _IdentityAwareFakeTracker({2: _mask_result(target)})
    manager = _FakeOnlineSAM2Manager({2: _video_result(2, hand_union)})
    frame2 = _moving_target_frame(2, (40, 30, 80, 70), contaminant_bbox=hand_bbox)
    provider, _camera = _provider(monkeypatch, [frame2], tracker, manager)
    assert provider.initialize_from_mask(
        _moving_target_frame(1, (40, 30, 80, 70)), target
    )

    _frame_out, result, obj, packet = provider.step()

    assert result.valid and obj.valid and packet.valid
    np.testing.assert_array_equal(result.mask, target)
    assert packet.debug["mask_source"] == "adaptive"
    assert "identity_rejected" in provider._last_online_sam2_status
    assert provider._online_sam2_recovery_hypothesis is None
    assert provider._trusted_online_sam2_publication.frame_id == 1
    provider.stop()


def test_final_publication_guard_sanitizes_adaptive_async_pending_hand_branch(
    monkeypatch,
):
    """Keep visible target pixels while SAM2 is pending and remove the hand."""

    target_mask = _large_mask((40, 30, 80, 70))
    hand_bbox = (80, 45, 94, 55)
    contaminated = target_mask | _large_mask(hand_bbox)
    tracker = AdaptiveColorDepthTracker(load_config()["tracker"])
    provider, _camera = _provider(
        monkeypatch,
        [_large_frame(2, contaminant_bbox=hand_bbox)],
        tracker,
        _FakeOnlineSAM2Manager(),
    )
    try:
        assert provider.initialize_from_mask(_large_frame(1), target_mask)
        committed_before = copy.deepcopy(provider._last_committed_tracker_snapshot)

        current = _large_frame(2, contaminant_bbox=hand_bbox)
        # Emulate the exact failure observed in deployment: adaptive has
        # already swallowed a thin neutral finger while the async semantic
        # result is not yet available, so the adaptive candidate would
        # otherwise be the final source for this frame.
        candidate = tracker.reinitialize_with_mask(current, contaminated)
        assert candidate.valid, candidate.message
        provider._last_output_mask_source = "adaptive"
        provider._last_online_sam2_status = "tracked_async_pending"
        obj = provider.extractor.extract(current, candidate.mask)

        public_mask, public_obj = provider._apply_recovery_publication_gate(
            current, candidate, obj
        )

        assert public_mask.valid and public_obj.valid
        assert np.all(public_mask.mask[target_mask.astype(bool)] == 1)
        # The 2 px registered-core tolerance may retain only the immediately
        # adjacent boundary, never the full 14 px finger branch.
        assert int(public_mask.mask[45:55, 80:94].sum()) <= 20
        assert "foreign appendage removed" in public_mask.message
        assert provider.tracking_committed
        assert tracker.state is not None and tracker.state.valid
        assert provider._last_publication_guard_source == "adaptive"
        assert "sanitized: source=adaptive" in (provider._last_publication_guard_status)
        assert provider._last_online_sam2_status == "tracked_async_pending"
        debug = provider._packet_debug(public_mask, public_obj)
        assert debug["publication_guard_source"] == "adaptive"
        assert "sanitized: source=adaptive" in (debug["publication_guard_status"])
        assert "appearance=" in debug["publication_guard_status"]
        _assert_adaptive_snapshot_equal(
            committed_before, provider._last_committed_tracker_snapshot
        )
    finally:
        provider.stop()


def test_periodic_sam2_reinit_side_effects_wait_for_final_publication(
    monkeypatch,
):
    """A rejected image-SAM contour cannot poison template/temporal state."""

    target_mask = _large_mask((40, 30, 80, 70))
    hand_bbox = (80, 45, 94, 55)
    contaminated = target_mask | _large_mask(hand_bbox)
    tracker = AdaptiveColorDepthTracker(load_config()["tracker"])
    current = _large_frame(2, contaminant_bbox=hand_bbox)
    provider, _camera = _provider(
        monkeypatch,
        [current],
        tracker,
        _FakeOnlineSAM2Manager(),
    )
    try:
        assert provider.initialize_from_mask(_large_frame(1), target_mask)
        provider.online_sam2_enabled = False
        provider.cfg["tracker"]["mode"] = "sam2_reinit"
        provider.cfg["sam2"]["reinit_every"] = 1
        provider.sam2 = SimpleNamespace(
            available=lambda: True,
            segment_with_box=lambda *_args, **_kwargs: _mask_result(
                contaminated, message="periodic SAM swallowed hand"
            ),
        )

        template_commits = []
        asynchronous_seeds = []
        synchronous_seeds = []
        monkeypatch.setattr(
            provider,
            "_update_recovery_template",
            lambda frame, result, source: template_commits.append(
                (frame.frame_id, result.mask.copy(), source)
            ),
        )
        monkeypatch.setattr(
            provider,
            "_queue_online_sam2_seed_after_inflight",
            lambda frame, mask, reason: asynchronous_seeds.append(
                (frame.frame_id, np.asarray(mask).copy(), reason)
            )
            or True,
        )
        monkeypatch.setattr(
            provider,
            "_seed_online_sam2",
            lambda frame, mask, reason: synchronous_seeds.append(
                (frame.frame_id, np.asarray(mask).copy(), reason)
            )
            or (True, 0.0),
        )

        tracked = provider._run_mask_pipeline()
        assert tracked.mask_result.valid
        # Before the final mask+RGB-D boundary, no persistent owner is taught.
        assert template_commits == []
        assert asynchronous_seeds == []
        assert synchronous_seeds == []

        obj = provider.extractor.extract(tracked.frame, tracked.mask_result.mask)
        public_mask, public_obj = provider._apply_recovery_publication_gate(
            tracked.frame, tracked.mask_result, obj
        )
        assert public_mask.valid and public_obj.valid
        assert "foreign appendage removed" in public_mask.message
        assert len(template_commits) == 1
        assert len(asynchronous_seeds) == 1
        assert synchronous_seeds == []
        np.testing.assert_array_equal(template_commits[0][1], public_mask.mask)
        np.testing.assert_array_equal(asynchronous_seeds[0][1], public_mask.mask)
    finally:
        provider.stop()


def test_final_publication_guard_rejects_lost_recovery_thin_hand_branch(
    monkeypatch,
):
    """LOST recovery cannot bypass the source-agnostic appendage guard."""

    target_mask = _large_mask((40, 30, 80, 70))
    hand_bbox = (80, 45, 94, 55)
    contaminated = target_mask | _large_mask(hand_bbox)
    tracker = AdaptiveColorDepthTracker(load_config()["tracker"])
    provider, _camera = _provider(
        monkeypatch,
        [_large_frame(2, contaminant_bbox=hand_bbox)],
        tracker,
        _FakeOnlineSAM2Manager(),
        recovery_publication_mode="unified_three_evidence",
    )
    try:
        assert provider.initialize_from_mask(_large_frame(1), target_mask)
        committed_before = copy.deepcopy(provider._last_committed_tracker_snapshot)
        current = _large_frame(2, contaminant_bbox=hand_bbox)
        candidate = tracker.reinitialize_with_mask(current, contaminated)
        assert candidate.valid, candidate.message

        provider._tracking_committed = False
        provider._uncommitted_frames = 1
        provider._recovery_tracker_tentative = True
        provider._last_output_mask_source = "online_sam2_recovery"
        obj = provider.extractor.extract(current, candidate.mask)
        public_mask, public_obj = provider._apply_recovery_publication_gate(
            current, candidate, obj
        )

        assert not public_mask.valid and not public_obj.valid
        assert public_mask.mask.sum() == 0
        assert "publication foreign appendage" in public_mask.message
        assert not provider.tracking_committed
        assert provider._recovery_publication_hypothesis is None
        assert provider._online_sam2_recovery_hypothesis is None
        assert tracker.state is not None and not tracker.state.valid
        assert provider._last_publication_guard_source == ("online_sam2_recovery")
        assert "rejected: source=online_sam2_recovery" in (
            provider._last_publication_guard_status
        )
        _assert_adaptive_snapshot_equal(
            committed_before, provider._last_committed_tracker_snapshot
        )
    finally:
        provider.stop()


def test_final_publication_guard_accepts_clean_1800px_s_translation(
    monkeypatch,
):
    """Committed time prediction preserves a clean 60 px/frame target."""

    initial_mask = _large_mask((40, 30, 80, 70))
    moved_mask = _large_mask((100, 30, 140, 70))
    tracker = AdaptiveColorDepthTracker(load_config()["tracker"])
    provider, _camera = _provider(
        monkeypatch,
        [_large_frame(2)],
        tracker,
        _FakeOnlineSAM2Manager(),
    )
    try:
        assert provider.initialize_from_mask(_large_frame(1), initial_mask)
        assert tracker.committed_motion is not None
        tracker.committed_motion.velocity_uv_s = (1800.0, 0.0)
        tracker.committed_motion.velocity_samples = 1
        provider._capture_committed_tracker_snapshot()

        current = _large_frame(2)
        current.color_bgr[30:70, 40:80] = 45
        current.color_bgr[30:70, 100:140] = (210, 80, 235)
        candidate = _mask_result(
            moved_mask, message="clean fast translated adaptive mask"
        )
        provider._last_output_mask_source = "adaptive"

        (
            guard_ok,
            guard_reason,
        ) = provider._publication_candidate_agrees_with_committed_clean_mask(
            current, candidate
        )
        assert guard_ok, guard_reason
        assert "alignment=final_clean" in guard_reason
        assert "shift=(60.0,0.0)px" in guard_reason

        obj = provider.extractor.extract(current, candidate.mask)
        public_mask, public_obj = provider._apply_recovery_publication_gate(
            current, candidate, obj
        )
        assert public_mask.valid and public_obj.valid
        np.testing.assert_array_equal(public_mask.mask, moved_mask)
        assert provider.tracking_committed
        assert "accepted: source=adaptive" in (provider._last_publication_guard_status)
    finally:
        provider.stop()


def test_final_publication_guard_accepts_visible_split_partial_occlusion(
    monkeypatch,
):
    """A hand-shaped gap may split the object without suppressing visibility."""

    target_mask = _large_mask((40, 30, 80, 70))
    visible = target_mask.copy()
    visible[30:70, 57:63] = 0
    tracker = AdaptiveColorDepthTracker(load_config()["tracker"])
    provider, _camera = _provider(
        monkeypatch,
        [_large_frame(2)],
        tracker,
        _FakeOnlineSAM2Manager(),
    )
    try:
        assert provider.initialize_from_mask(_large_frame(1), target_mask)
        current = _large_frame(2)
        candidate = _mask_result(
            visible, message="visible target split by partial occlusion"
        )
        provider._last_output_mask_source = "online_sam2_tracked"
        obj = provider.extractor.extract(current, candidate.mask)
        public_mask, public_obj = provider._apply_recovery_publication_gate(
            current, candidate, obj
        )
        assert public_mask.valid and public_obj.valid
        np.testing.assert_array_equal(public_mask.mask, visible)
        assert "accepted: source=online_sam2_tracked" in (
            provider._last_publication_guard_status
        )
    finally:
        provider.stop()


def test_final_guard_extra_region_failure_keeps_full_clean_intersection(
    monkeypatch,
):
    """A bad appendage must not pixelwise erode the healthy target body."""

    target = _large_mask((40, 30, 80, 70))
    appendage = _large_mask((80, 30, 90, 70))
    candidate_mask = np.maximum(target, appendage)
    provider, _tracker = _guarded_primary_provider(
        monkeypatch, [], {}, target
    )
    current = _large_frame(2)
    monkeypatch.setattr(
        provider,
        "_aligned_committed_clean_mask",
        lambda *_args, **_kwargs: (target.copy(), "focused clean alignment"),
    )

    def deterministic_stats(_frame, mask, *, threshold):
        del threshold
        current_mask = (np.asarray(mask) > 0).astype(np.uint8)
        if np.array_equal(current_mask, appendage):
            return (0.05, 0.05, int(current_mask.sum())), "extra is foreign"
        if np.array_equal(current_mask, candidate_mask):
            return (0.75, 0.75, int(current_mask.sum())), "candidate is target-majority"
        if np.array_equal(current_mask, target):
            return (0.80, 0.80, int(current_mask.sum())), "retained target is healthy"
        raise AssertionError(f"unexpected appearance mask area={int(current_mask.sum())}")

    monkeypatch.setattr(
        provider, "_immutable_target_appearance_stats", deterministic_stats
    )
    monkeypatch.setattr(
        provider,
        "_immutable_target_appearance_supported_mask",
        lambda *_args, **_kwargs: pytest.fail(
            "extra-region rejection must not force pixelwise appearance core"
        ),
    )
    try:
        candidate = _mask_result(candidate_mask, message="target plus foreign branch")
        provider._last_output_mask_source = "online_sam2_video"
        obj = provider.extractor.extract(current, candidate.mask)
        public_mask, public_obj = provider._apply_recovery_publication_gate(
            current, candidate, obj
        )

        assert public_mask.valid and public_obj.valid
        assert public_mask.source == "publication_sanitized_target_core"
        np.testing.assert_array_equal(public_mask.mask, target)
        assert provider._last_publication_guard_failure_kind == (
            "extra_region_identity_failed"
        )
    finally:
        provider.stop()


def test_recovery_sanitizer_prefers_explicit_fresh_reference_over_cached_clean(
    monkeypatch,
):
    """Later RGB-D recovery cannot fall back to the older clean publication."""

    old_clean = _large_mask((30, 30, 70, 70))
    fresh_recovery = _large_mask((100, 30, 140, 70))
    candidate_mask = np.maximum(
        fresh_recovery,
        _large_mask((140, 30, 150, 70)),
    )
    provider, _tracker = _guarded_primary_provider(
        monkeypatch, [], {}, old_clean
    )
    current = _large_frame(8)
    recovery_reference = SimpleNamespace(mask=fresh_recovery)
    calls = []

    def aligned(_frame, _candidate, *, recovery_reference=None):
        calls.append(recovery_reference)
        if recovery_reference is not None:
            return fresh_recovery.copy(), "fresh recovery-local alignment"
        return old_clean.copy(), "stale final-clean alignment"

    monkeypatch.setattr(provider, "_aligned_committed_clean_mask", aligned)
    monkeypatch.setattr(
        provider,
        "_immutable_target_appearance_stats",
        lambda _frame, mask, *, threshold: (
            (0.90, 0.90, int(np.count_nonzero(mask))),
            f"healthy at threshold={threshold}",
        ),
    )
    # Deliberately poison the same-frame cache: the explicit recovery token
    # must still win.
    provider._last_publication_guard_frame_id = current.frame_id
    provider._last_publication_guard_aligned_clean = old_clean.copy()
    try:
        sanitized, reason = provider._sanitize_publication_target_core(
            current,
            _mask_result(candidate_mask),
            recovery_reference=recovery_reference,
        )

        assert sanitized is not None, reason
        assert calls == [recovery_reference]
        np.testing.assert_array_equal(sanitized.mask, fresh_recovery)
        assert "sanitized=1600px" in reason
    finally:
        provider.stop()


def test_recovery_local_small_visible_core_is_partial_only(monkeypatch):
    """A small identity-clean occlusion core bypasses only full-scale size."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    local_mask = _large_mask((70, 30, 110, 70))
    _arm_exact_two_sam_then_same_frame_rgbd_recovery(provider, local_mask)
    local_reference = provider._recovery_publication_hypothesis
    assert local_reference is not None
    frame = _moving_target_frame(118, (70, 30, 110, 70))
    raw_evidence = provider._defer_full_target_geometry_commit(frame, local_mask)
    assert raw_evidence is not None
    core = np.zeros_like(local_mask)
    core[40:70, 70:80] = 1  # 300/1600=.1875 of the fresh local target.
    core_evidence, reason = provider._register_frame_mask_evidence(
        frame,
        core,
        evidence_kind="sanitized_target_core",
        authority_eligible=False,
    )
    assert core_evidence is not None, reason
    monkeypatch.setattr(
        provider,
        "_aligned_committed_clean_mask",
        lambda *_args, **_kwargs: (local_mask.copy(), "focused local alignment"),
    )
    monkeypatch.setattr(
        provider,
        "_immutable_target_appearance_stats",
        lambda _frame, mask, *, threshold: (
            (0.90, 0.90, int(np.count_nonzero(mask))),
            f"focused identity threshold={threshold}",
        ),
    )
    authority_before = copy.deepcopy(provider._full_target_geometry_authority)
    trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)
    try:
        accepted, detail = (
            provider._guarded_recovery_sanitized_visible_core_accepts(
                frame,
                _exact_token_mask_result(
                    core_evidence.mask, message="small visible object core"
                ),
                raw_exact_mask=raw_evidence.mask,
                local_reference=local_reference,
            )
        )
        assert accepted, detail
        assert "fraction=0.188" in detail
        pending = provider._pending_full_target_geometry_commit
        assert pending is not None
        assert pending.mask is raw_evidence.mask
        assert pending.mask_digest == raw_evidence.mask_digest
        assert not pending.raw_exact_eligible
        assert not pending.bootstrap_eligible
        assert pending.eligibility_kind == "recovery_local_visible_core"
        assert pending.recovery_local_visible_core_digest == (
            core_evidence.mask_digest
        )
        assert pending.recovery_local_reference_digest == (
            local_reference.mask_digest
        )
        assert pending.recovery_local_core_pixels == 300
        assert pending.recovery_local_reference_pixels == 1600
        assert provider._full_target_geometry_authority.frame_id == (
            authority_before.frame_id
        )
        assert provider._trusted_online_sam2_publication.frame_id == (
            trusted_before.frame_id
        )
    finally:
        provider.stop()


def test_recovery_local_visible_core_requires_bound_subset(monkeypatch):
    """A forged sanitizer token outside raw SAM cannot use partial recovery."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    local_mask = _large_mask((70, 30, 110, 70))
    _arm_exact_two_sam_then_same_frame_rgbd_recovery(provider, local_mask)
    local_reference = provider._recovery_publication_hypothesis
    frame = _moving_target_frame(118, (70, 30, 110, 70))
    raw_evidence = provider._defer_full_target_geometry_commit(frame, local_mask)
    outside = _large_mask((120, 30, 130, 60))
    outside_evidence, reason = provider._register_frame_mask_evidence(
        frame,
        outside,
        evidence_kind="sanitized_target_core",
        authority_eligible=False,
    )
    assert outside_evidence is not None, reason
    try:
        accepted, detail = (
            provider._guarded_recovery_sanitized_visible_core_accepts(
                frame,
                _exact_token_mask_result(
                    outside_evidence.mask, message="forged outboard core"
                ),
                raw_exact_mask=raw_evidence.mask,
                local_reference=local_reference,
            )
        )
        assert not accepted
        assert "not a strict subset" in detail
        assert provider._pending_full_target_geometry_commit.raw_exact_eligible
    finally:
        provider.stop()


def test_recovery_three_of_three_allows_deeper_bound_partial_occlusion(
    monkeypatch,
):
    """The later core may shrink below the full-mask 0.741 area floor."""

    initial = _large_mask((40, 30, 80, 70))
    provider, _tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    previous_core = _large_mask((70, 30, 90, 50))  # 400px.
    _arm_exact_two_sam_then_same_frame_rgbd_recovery(provider, previous_core)
    previous = provider._recovery_publication_hypothesis
    assert previous is not None
    previous.mask_evidence_kind = "sanitized_target_core"

    frame = _moving_target_frame(118, (70, 30, 90, 50))
    raw = previous_core.copy()
    raw_evidence = provider._defer_full_target_geometry_commit(frame, raw)
    assert raw_evidence is not None
    later_core = np.zeros_like(raw)
    later_core[32:47, 72:82] = 1  # 150/400=.375, IoU=.375.
    later_evidence, reason = provider._register_frame_mask_evidence(
        frame,
        later_core,
        evidence_kind="sanitized_target_core",
        authority_eligible=False,
    )
    assert later_evidence is not None, reason
    provider._mark_pending_full_target_geometry_ineligible(
        frame, "focused deeper partial occlusion"
    )
    provider._last_online_sam2_exact_result = True
    provider._current_recovery_evidence_outcome = "candidate"
    result = _exact_token_mask_result(
        later_evidence.mask,
        message="later distinct RGB-D deeper visible core",
    )
    result.source = "online_sam2_video"
    obj = provider.extractor.extract(frame, later_evidence.mask)
    try:
        public_mask, public_obj, recovered = provider._gate_recovery_publication(
            frame, result, obj
        )
        assert recovered
        assert public_mask.valid and public_obj.valid
        assert "pending final commit 3/3" in public_mask.message
        assert provider._pending_recovery_publication_commit is not None
        assert provider._recovery_publication_hypothesis.mask_evidence_kind == (
            "sanitized_target_core"
        )
    finally:
        provider.stop()


def test_final_publication_guard_rejects_low_appearance_occluder_inside_core(
    monkeypatch,
):
    """Complete occlusion fails closed even with no geometric appendage."""

    target_mask = _large_mask((40, 30, 80, 70))
    hand_inside_core = _large_mask((45, 35, 75, 65))
    tracker = AdaptiveColorDepthTracker(load_config()["tracker"])
    provider, _camera = _provider(
        monkeypatch,
        [_large_frame(2)],
        tracker,
        _FakeOnlineSAM2Manager(),
    )
    try:
        assert provider.initialize_from_mask(_large_frame(1), target_mask)
        current = _large_frame(2)
        current.color_bgr[30:70, 40:80] = 45
        candidate = _mask_result(
            hand_inside_core, message="occluding hand inside target extent"
        )
        provider._last_output_mask_source = "adaptive"
        obj = provider.extractor.extract(current, candidate.mask)
        public_mask, public_obj = provider._apply_recovery_publication_gate(
            current, candidate, obj
        )
        assert not public_mask.valid and not public_obj.valid
        assert public_mask.mask.sum() == 0
        assert "candidate appearance" in public_mask.message
        assert "visible appearance core=0px" in public_mask.message
        assert provider._last_publication_guard_failure_kind == (
            "whole_candidate_identity_failed"
        )
        assert not provider.tracking_committed
    finally:
        provider.stop()


def test_final_publication_guard_accepts_target_colored_shape_expansion(
    monkeypatch,
):
    """A real target deformation is retained when added pixels match target."""

    initial = _large_mask((40, 30, 80, 70))
    expanded = _large_mask((38, 28, 82, 72))
    tracker = AdaptiveColorDepthTracker(load_config()["tracker"])
    provider, _camera = _provider(
        monkeypatch,
        [_large_frame(2)],
        tracker,
        _FakeOnlineSAM2Manager(),
    )
    try:
        assert provider.initialize_from_mask(_large_frame(1), initial)
        current = _large_frame(2)
        current.color_bgr[expanded.astype(bool)] = (210, 80, 235)
        candidate = _mask_result(
            expanded, message="target-colored perspective expansion"
        )
        provider._last_output_mask_source = "adaptive"
        obj = provider.extractor.extract(current, candidate.mask)
        public_mask, public_obj = provider._apply_recovery_publication_gate(
            current, candidate, obj
        )
        assert public_mask.valid and public_obj.valid
        np.testing.assert_array_equal(public_mask.mask, expanded)
        assert "accepted: source=adaptive" in (provider._last_publication_guard_status)
    finally:
        provider.stop()


def test_final_publication_guard_accepts_measured_fast_entry_from_rest(
    monkeypatch,
):
    """The measured 71 px/30 Hz entry case stays inside the guard envelope."""

    initial = _large_mask((40, 30, 80, 70))
    moved = _large_mask((111, 30, 151, 70))
    tracker = AdaptiveColorDepthTracker(load_config()["tracker"])
    provider, _camera = _provider(
        monkeypatch,
        [_large_frame(2)],
        tracker,
        _FakeOnlineSAM2Manager(),
    )
    try:
        assert provider.initialize_from_mask(_large_frame(1), initial)
        current = _large_frame(2)
        current.color_bgr[30:70, 40:80] = 45
        current.color_bgr[30:70, 111:151] = (210, 80, 235)
        candidate = _mask_result(moved, message="71 px fast entry")
        provider._last_output_mask_source = "adaptive"
        ok, reason = provider._publication_candidate_agrees_with_committed_clean_mask(
            current, candidate
        )
        assert ok, reason
        assert "mask_overlap shift=(71.0,0.0)px" in reason
    finally:
        provider.stop()


@pytest.mark.parametrize(
    ("contaminant_bbox", "expected_area"),
    [
        ((80, 25, 120, 75), 3600),  # 2.25x the immutable target area
        ((80, 20, 128, 80), 4480),  # 2.80x, still below the old 3x gate
    ],
)
def test_lost_hand_union_fails_closed_without_state_or_template_pollution(
    monkeypatch,
    contaminant_bbox,
    expected_area,
):
    target_mask = _large_mask((40, 30, 80, 70))
    online_mask = target_mask | _large_mask(contaminant_bbox)
    assert int(online_mask.sum()) == expected_area

    tracker = AdaptiveColorDepthTracker(load_config()["tracker"])
    manager = _FakeOnlineSAM2Manager()
    current = _large_frame(2, contaminant_bbox=contaminant_bbox)
    provider, _camera = _provider(monkeypatch, [current], tracker, manager)
    provider.cfg["tracker"]["recovery_enabled"] = True
    assert provider.initialize_from_mask(_large_frame(1), target_mask)
    adaptive_result = tracker._reject(current, "synthetic complete occlusion")
    assert not adaptive_result.valid

    # Deep-core initialization plus a sample-count-invariant prior now rejects
    # both contaminated masks at the tracker gate, including the observed 2.8x
    # case.  Keep the provider assertions below as defense-in-depth coverage.
    before_raw_gate = tracker.snapshot_state()
    raw_result = tracker.reinitialize_with_mask(current, online_mask)
    assert not raw_result.valid
    assert raw_result.mask.sum() == 0
    assert "color=" in raw_result.message
    tracker.restore_state(before_raw_gate)

    state_before = tracker.snapshot_state()
    template_before = provider.recovery_template.copy()
    template_mask_before = provider.recovery_template_mask.copy()
    resets_before = provider.history.reset_calls
    result, recovered = provider._fuse_online_sam2_candidate(
        current,
        adaptive_result,
        _video_result(2, online_mask),
    )

    assert not result.valid and not recovered
    assert result.mask.sum() == 0
    assert provider._online_sam2_recovery_hypothesis is None
    _assert_adaptive_snapshot_equal(state_before, tracker.snapshot_state())
    np.testing.assert_array_equal(provider.recovery_template, template_before)
    np.testing.assert_array_equal(provider.recovery_template_mask, template_mask_before)
    assert provider.history.reset_calls == resets_before
    provider.stop()


@pytest.mark.parametrize("bad_depth", ["low_valid_ratio", "wide_spread"])
def test_lost_online_recovery_rejects_bad_rgbd_quality_before_commit(
    monkeypatch,
    bad_depth,
):
    target_mask = _large_mask((40, 30, 80, 70))
    current = _large_frame(2)
    if bad_depth == "low_valid_ratio":
        current.depth_raw[target_mask.astype(bool)] = 0
        current.depth_raw[30:40, 40:80] = 800  # 25% valid, below 35%.
    else:
        current.depth_raw[30:70, 40:60] = 700
        current.depth_raw[30:70, 60:80] = 950  # p95-p5 = 0.25m.

    tracker = AdaptiveColorDepthTracker(load_config()["tracker"])
    manager = _FakeOnlineSAM2Manager()
    provider, _camera = _provider(monkeypatch, [current], tracker, manager)
    assert provider.initialize_from_mask(_large_frame(1), target_mask)
    adaptive_result = tracker._reject(current, "synthetic complete occlusion")

    # Median-only validation accepts both cases. The new provider gate adds
    # valid-depth support and robust p5/p95 spread without object-specific RGB.
    before_raw_gate = tracker.snapshot_state()
    raw_result = tracker.reinitialize_with_mask(current, target_mask)
    assert raw_result.valid, raw_result.message
    tracker.restore_state(before_raw_gate)
    state_before = tracker.snapshot_state()

    result, recovered = provider._fuse_online_sam2_candidate(
        current,
        adaptive_result,
        _video_result(2, target_mask),
    )

    assert not result.valid and not recovered
    assert provider._online_sam2_recovery_hypothesis is None
    assert provider._last_online_sam2_status.startswith("recovery_geometry_rejected:")
    if bad_depth == "low_valid_ratio":
        assert "valid depth" in provider._last_online_sam2_status
    else:
        assert "depth spread" in provider._last_online_sam2_status
    assert provider._last_online_sam2_geometry_reason in (
        provider._last_online_sam2_status
    )
    _assert_adaptive_snapshot_equal(state_before, tracker.snapshot_state())
    provider.stop()


def test_rejected_online_sam2_mask_rolls_back_gate_and_keeps_adaptive_result(
    monkeypatch,
):
    seed = _mask(4, 5, 12, 15)
    adaptive = _mask(5, 5, 13, 15)
    wrong_instance = _mask(20, 8, 28, 18)
    tracker = _FakeTracker({2: _mask_result(adaptive)})
    tracker.reject_reinit_masks.append(wrong_instance)
    manager = _FakeOnlineSAM2Manager({2: _video_result(2, wrong_instance)})
    provider, _camera = _provider(monkeypatch, [_frame(2)], tracker, manager)
    assert provider.initialize_from_mask(_frame(1), seed)

    _frame_out, result, _obj, packet = provider.step()

    # Spatial consensus rejects this disjoint candidate before it can mutate
    # the tracker, so no rollback call is needed.
    assert tracker.restore_calls == 0
    assert tracker.token == "adaptive-f2"
    np.testing.assert_array_equal(result.mask, adaptive)
    assert result.valid and packet.valid
    provider.stop()


def test_online_sam2_recovery_stays_closed_through_provider_probation(
    monkeypatch,
):
    seed = _mask(4, 5, 12, 15)
    empty = np.zeros_like(seed)
    recovered = _mask(7, 5, 15, 15)
    tracker = _FakeTracker(
        {
            2: _mask_result(empty, valid=False, message="adaptive lost"),
            3: _mask_result(empty, valid=False, message="adaptive lost"),
            4: _mask_result(recovered, message="adaptive recovered stable"),
            5: _mask_result(recovered, message="adaptive recovered stable"),
        }
    )
    manager = _FakeOnlineSAM2Manager(
        {
            2: _video_result(2, recovered),
            3: _video_result(3, recovered),
            4: _video_result(4, recovered),
            5: _video_result(5, recovered),
        }
    )
    provider, _camera = _provider(
        monkeypatch,
        [_frame(2), _frame(3), _frame(4), _frame(5)],
        tracker,
        manager,
        recovery_publication_mode="legacy_double_confirm",
    )
    provider.cfg["tracker"]["recovery_enabled"] = True
    assert provider.initialize_from_mask(_frame(1), seed)
    initial_template = provider.recovery_template.copy()
    initial_template_mask = provider.recovery_template_mask.copy()
    resets_after_init = provider.history.reset_calls

    _frame2, result_f2, _obj2, packet_f2 = provider.step()

    assert not result_f2.valid and not packet_f2.valid
    assert result_f2.mask.sum() == 0
    assert "pending 1/2" in result_f2.message
    assert tracker.token == "adaptive-f2"
    assert tracker.restore_calls == 1
    assert provider.history.reset_calls == resets_after_init + 1
    np.testing.assert_array_equal(provider.recovery_template, initial_template)
    np.testing.assert_array_equal(
        provider.recovery_template_mask, initial_template_mask
    )

    _frame3, result_f3, obj_f3, packet_f3 = provider.step()

    assert not result_f3.valid and not obj_f3.valid and not packet_f3.valid
    assert result_f3.mask.sum() == 0
    assert "publication pending 1/3" in result_f3.message.lower()
    assert tracker.token == "sam2-mutated-f3"
    assert not provider.tracking_committed

    _frame4, result_f4, obj_f4, packet_f4 = provider.step()
    assert not result_f4.valid and not obj_f4.valid and not packet_f4.valid
    assert "publication pending 2/3" in result_f4.message.lower()

    _frame5, result_f5, obj_f5, packet_f5 = provider.step()
    np.testing.assert_array_equal(result_f5.mask, recovered)
    assert result_f5.valid and obj_f5.valid and packet_f5.valid
    assert "publication stable 3/3" in result_f5.message.lower()
    assert provider.tracking_committed
    assert provider.history.reset_calls == resets_after_init + 2
    np.testing.assert_array_equal(provider.recovery_template, initial_template)
    np.testing.assert_array_equal(
        provider.recovery_template_mask, initial_template_mask
    )
    provider.stop()


def test_two_frame_tentative_recovery_then_loss_never_publishes(
    monkeypatch,
):
    seed = _mask(4, 5, 12, 15)
    empty = np.zeros_like(seed)
    recovered = _mask(7, 5, 15, 15)
    tracker = _FakeTracker(
        {
            2: _mask_result(empty, valid=False, message="adaptive lost"),
            3: _mask_result(empty, valid=False, message="adaptive lost"),
            4: _mask_result(recovered, message="tentative adaptive"),
            5: _mask_result(empty, valid=False, message="lost again"),
        }
    )
    manager = _FakeOnlineSAM2Manager(
        {
            2: _video_result(2, recovered),
            3: _video_result(3, recovered),
            4: _video_result(4, recovered),
            5: _video_result(
                5,
                empty,
                valid=False,
                message="occluded again",
                internal_frame_idx=4,
            ),
        }
    )
    provider, _camera = _provider(
        monkeypatch,
        [_frame(2), _frame(3), _frame(4), _frame(5)],
        tracker,
        manager,
    )
    assert provider.initialize_from_mask(_frame(1), seed)

    outputs = [provider.step() for _ in range(4)]

    assert all(not result.valid for _, result, _, _ in outputs)
    assert all(not obj.valid for _, _, obj, _ in outputs)
    assert all(not packet.valid for _, _, _, packet in outputs)
    assert all(packet.center is None for _, _, _, packet in outputs)
    assert not provider.tracking_committed
    assert provider._recovery_publication_hypothesis is None
    provider.stop()


def test_failed_probation_rolls_real_adaptive_tracker_back_to_committed_anchor():
    target_mask = _large_mask((40, 30, 80, 70))
    wrong_mask = _large_mask((82, 30, 122, 70))
    target_color = (210, 80, 235)
    tracker = AdaptiveColorDepthTracker(load_config()["tracker"])
    initial = _large_frame(1)
    tracker.initialize(initial, mask=target_mask)
    committed_bbox = tracker.state.bbox_xyxy.copy()
    committed_anchor = tracker.anchor_hs.copy()
    committed_adaptive = tracker.adaptive_hs.copy()

    provider = ObjectPCDProvider.__new__(ObjectPCDProvider)
    provider.tracker = tracker
    provider._tracking_committed = True
    provider._uncommitted_frames = 0
    provider._last_committed_tracker_snapshot = None
    provider._capture_committed_tracker_snapshot()

    wrong_frame = _large_frame(
        2,
        contaminant_bbox=(82, 30, 122, 70),
        contaminant_bgr=target_color,
    )
    tentative = tracker.reinitialize_with_mask(wrong_frame, wrong_mask)
    assert tentative.valid, tentative.message
    assert not np.array_equal(tracker.state.bbox_xyxy, committed_bbox)
    # Re-detection is provisional and must not learn before provider commit.
    np.testing.assert_array_equal(tracker.anchor_hs, committed_anchor)
    np.testing.assert_array_equal(tracker.adaptive_hs, committed_adaptive)

    provider._tracking_committed = False
    provider._uncommitted_frames = 2
    assert provider._restore_committed_tracker_as_lost()

    np.testing.assert_array_equal(tracker.state.bbox_xyxy, committed_bbox)
    np.testing.assert_array_equal(tracker.anchor_hs, committed_anchor)
    np.testing.assert_array_equal(tracker.adaptive_hs, committed_adaptive)
    assert not tracker.state.valid
    assert tracker.state.lost_count == 2
    assert tracker.previous_gray is None


def test_rpc_failure_uses_adaptive_then_reseeds_from_next_current_frame(
    monkeypatch,
):
    seed = _mask(4, 5, 12, 15)
    adaptive_f2 = _mask(5, 5, 13, 15)
    adaptive_f3 = _mask(12, 6, 20, 16)
    tracker = _FakeTracker(
        {
            2: _mask_result(adaptive_f2),
            3: _mask_result(adaptive_f3),
        }
    )
    manager = _FakeOnlineSAM2Manager(
        {
            2: _video_result(
                2,
                np.zeros_like(seed),
                valid=False,
                message="SAM2 video RPC failed closed: timeout",
                internal_frame_idx=-1,
            )
        }
    )
    provider, _camera = _provider(monkeypatch, [_frame(2), _frame(3)], tracker, manager)
    assert provider.initialize_from_mask(_frame(1), seed)

    _f2, result_f2, _o2, packet_f2 = provider.step()
    calls_after_failure = list(manager.calls)
    _f3, result_f3, _o3, packet_f3 = provider.step()

    np.testing.assert_array_equal(result_f2.mask, adaptive_f2)
    np.testing.assert_array_equal(result_f3.mask, adaptive_f3)
    assert packet_f2.valid and packet_f3.valid
    assert [call[0:2] for call in calls_after_failure if call[0] != "start"] == [
        ("initialize", 1),
        ("track", 2),
    ]
    initialize_calls = [call for call in manager.calls if call[0] == "initialize"]
    assert [call[1] for call in initialize_calls] == [1, 3]
    np.testing.assert_array_equal(initialize_calls[-1][3], adaptive_f3)
    assert provider._online_sam2_initialized or (
        provider._online_sam2_future_kind == "initialize"
        and provider._online_sam2_future_frame_id == 3
    )
    provider.stop()


def test_normal_empty_temporal_mask_keeps_session_but_quarantines_reappearance(
    monkeypatch,
):
    seed = _mask(4, 5, 12, 15)
    adaptive_f2 = _mask(5, 5, 13, 15)
    adaptive_f3 = _mask(6, 5, 14, 15)
    temporal_f3 = _mask(7, 5, 15, 15)
    tracker = _FakeTracker(
        {
            2: _mask_result(adaptive_f2),
            3: _mask_result(adaptive_f3),
        }
    )
    manager = _FakeOnlineSAM2Manager(
        {
            # A legitimate propagated empty mask during complete occlusion is
            # not an RPC/session failure: its internal frame index advances.
            2: _video_result(
                2,
                np.zeros_like(seed),
                valid=False,
                message="target temporarily absent",
                internal_frame_idx=1,
            ),
            3: _video_result(3, temporal_f3, internal_frame_idx=2),
        }
    )
    provider, _camera = _provider(monkeypatch, [_frame(2), _frame(3)], tracker, manager)
    assert provider.initialize_from_mask(_frame(1), seed)

    provider.step()
    _frame_out, result_f3, _obj, packet_f3 = provider.step()

    operations = [call[0:2] for call in manager.calls if call[0] != "start"]
    assert operations == [
        ("initialize", 1),
        ("track", 2),
        ("track", 3),
    ]
    assert not result_f3.valid and not packet_f3.valid
    assert int(result_f3.mask.sum()) == 0
    assert "continuity" in result_f3.message
    provider.stop()


def test_slow_online_result_never_blocks_or_applies_an_old_mask_to_new_depth(
    monkeypatch,
):
    seed = _mask(4, 5, 12, 15)
    adaptive_f2 = _mask(5, 5, 13, 15)
    adaptive_f3 = _mask(6, 5, 14, 15)
    stale_temporal_f2 = _mask(18, 7, 26, 17)
    temporal_f3 = _mask(7, 5, 15, 15)
    tracker = _FakeTracker(
        {
            2: _mask_result(adaptive_f2),
            3: _mask_result(adaptive_f3),
        }
    )
    release_f2 = threading.Event()
    entered_f2 = threading.Event()
    finished_f2 = threading.Event()
    manager = _FakeOnlineSAM2Manager(
        {
            2: _video_result(2, stale_temporal_f2),
            3: _video_result(3, temporal_f3),
        }
    )
    manager.track_wait_events[2] = release_f2
    manager.track_entered_events[2] = entered_f2
    manager.track_finished_events[2] = finished_f2
    provider, _camera = _provider(monkeypatch, [_frame(2), _frame(3)], tracker, manager)
    assert provider.initialize_from_mask(_frame(1), seed)

    started = time.perf_counter()
    _frame2, result_f2, _obj2, packet_f2 = provider.step()
    elapsed_s = time.perf_counter() - started

    assert entered_f2.is_set()
    assert elapsed_s < 0.15
    np.testing.assert_array_equal(result_f2.mask, adaptive_f2)
    assert packet_f2.valid

    release_f2.set()
    assert finished_f2.wait(timeout=0.5)
    _frame3, result_f3, _obj3, packet_f3 = provider.step()

    assert [call[1] for call in manager.calls if call[0] == "track"] == [2, 3]
    assert not any(
        np.array_equal(mask, stale_temporal_f2)
        for _frame_id, mask in tracker.reinit_masks
    )
    np.testing.assert_array_equal(result_f3.mask, temporal_f3)
    assert packet_f3.valid
    provider.stop()


def test_missing_exact_online_frame_preserves_recovery_hypothesis(
    monkeypatch,
):
    seed = _mask(4, 5, 12, 15)
    tracker = _FakeTracker({})
    manager = _FakeOnlineSAM2Manager()
    provider, _camera = _provider(monkeypatch, [_frame(3)], tracker, manager)
    assert provider.initialize_from_mask(_frame(1), seed)
    hypothesis = SimpleNamespace(
        frame_id=2,
        bbox_xyxy=np.asarray([4, 5, 12, 15], dtype=np.int32),
        area=float(seed.sum()),
        depth_median=0.7,
        hits=1,
    )
    provider._online_sam2_recovery_hypothesis = hypothesis
    provider._online_sam2_future = None
    invalid = _mask_result(
        np.zeros_like(seed), valid=False, message="adaptive temporarily lost"
    )

    result, elapsed_ms, recovered = provider._resolve_online_sam2_for_frame(
        _frame(3), invalid
    )

    assert result is invalid
    assert elapsed_ms == 0.0 and not recovered
    assert provider._online_sam2_recovery_hypothesis is hypothesis
    provider.stop()


def test_online_deadline_miss_is_not_negative_recovery_evidence(
    monkeypatch,
):
    seed = _mask(4, 5, 12, 15)
    tracker = _FakeTracker({})
    manager = _FakeOnlineSAM2Manager()
    provider, _camera = _provider(monkeypatch, [_frame(3)], tracker, manager)
    assert provider.initialize_from_mask(_frame(1), seed)
    hypothesis = SimpleNamespace(
        frame_id=2,
        bbox_xyxy=np.asarray([4, 5, 12, 15], dtype=np.int32),
        area=float(seed.sum()),
        depth_median=0.7,
        hits=1,
    )
    provider._online_sam2_recovery_hypothesis = hypothesis
    provider._online_sam2_future = Future()
    provider._online_sam2_future_kind = "track"
    provider._online_sam2_future_frame_id = 3
    provider._online_sam2_future_started = time.perf_counter() - 1.0
    invalid = _mask_result(
        np.zeros_like(seed), valid=False, message="adaptive temporarily lost"
    )

    result, elapsed_ms, recovered = provider._resolve_online_sam2_for_frame(
        _frame(3), invalid
    )

    assert result is invalid
    assert elapsed_ms == 0.0 and not recovered
    assert provider._online_sam2_recovery_hypothesis is hypothesis
    assert provider._last_online_sam2_status == "deadline_miss"
    assert provider._online_sam2_deadline_miss_count == 1
    provider._online_sam2_future.cancel()
    provider._clear_online_future()
    provider.stop()


def test_valid_adaptive_track_waits_only_the_configured_same_frame_sam2_window(
    monkeypatch,
):
    seed = _mask(4, 5, 12, 15)
    adaptive = _mask_result(seed, valid=True, message="adaptive valid")
    tracker = _FakeTracker({})
    manager = _FakeOnlineSAM2Manager()
    provider, _camera = _provider(monkeypatch, [_frame(3)], tracker, manager)
    assert provider.initialize_from_mask(_frame(1), seed)
    provider._online_sam2_future = Future()
    provider._online_sam2_future_kind = "track"
    provider._online_sam2_future_frame_id = 3
    provider._online_sam2_future_started = time.perf_counter()

    started = time.perf_counter()
    result, elapsed_ms, recovered = provider._resolve_online_sam2_for_frame(
        _frame(3), adaptive
    )
    elapsed_s = time.perf_counter() - started

    assert result is adaptive
    assert elapsed_ms == 0.0 and not recovered
    assert 0.020 <= elapsed_s < 0.080
    assert provider._last_online_sam2_status == "tracked_async_pending"
    assert provider._online_sam2_deadline_miss_count == 0
    provider._online_sam2_future.cancel()
    provider._clear_online_future()
    provider.stop()


def test_valid_adaptive_wait_budget_is_measured_from_submission(monkeypatch):
    class RecordingPendingFuture(Future):
        def __init__(self):
            super().__init__()
            self.observed_timeout = None

        def result(self, timeout=None):
            self.observed_timeout = timeout
            raise FutureTimeoutError()

    seed = _mask(4, 5, 12, 15)
    adaptive = _mask_result(seed, valid=True, message="adaptive valid")
    tracker = _FakeTracker({})
    manager = _FakeOnlineSAM2Manager()
    provider, _camera = _provider(monkeypatch, [_frame(3)], tracker, manager)
    assert provider.initialize_from_mask(_frame(1), seed)
    future = RecordingPendingFuture()
    provider._online_sam2_future = future
    provider._online_sam2_future_kind = "track"
    provider._online_sam2_future_frame_id = 3
    provider._online_sam2_future_started = time.perf_counter() - 0.010

    result, elapsed_ms, recovered = provider._resolve_online_sam2_for_frame(
        _frame(3), adaptive
    )

    assert result is adaptive
    assert elapsed_ms == 0.0 and not recovered
    # Ten milliseconds elapsed since submission, leaving approximately 35 ms
    # of the configured 45 ms exact-frame SAM2 publication window.
    assert future.observed_timeout == pytest.approx(0.035, abs=0.005)
    assert provider._last_online_sam2_status == "tracked_async_pending"
    provider._clear_online_future()
    provider.stop()


def test_online_mask_geometry_scales_only_masked_depth(monkeypatch):
    frame = _frame(3)
    mask = _mask(5, 5, 13, 15)
    provider = object.__new__(ObjectPCDProvider)
    provider.cfg = {}

    def reject_full_depth(_self):
        raise AssertionError("online geometry requested full-frame depth_m")

    monkeypatch.setattr(RGBDFrame, "depth_m", property(reject_full_depth))
    geometry, reason = provider._compute_online_mask_geometry(frame, mask)

    assert geometry is not None, reason
    assert geometry.area == float(mask.sum())
    assert geometry.valid_depth_ratio == 1.0
    assert geometry.depth_median == pytest.approx(0.7)


def test_online_mask_geometry_uses_formal_pointcloud_depth_range():
    frame = _frame(3)
    mask = _mask(5, 5, 13, 15)
    depth_raw = np.asarray(frame.depth_raw).copy()
    selected_y, selected_x = np.nonzero(mask)
    # Half the semantic mask is valid target depth; the other half is far
    # background that the formal point-cloud extractor must reject.
    split = selected_y.size // 2
    depth_raw[selected_y[:split], selected_x[:split]] = 800
    depth_raw[selected_y[split:], selected_x[split:]] = 2500
    frame = replace(frame, depth_raw=depth_raw, depth_scale=0.001)

    provider = object.__new__(ObjectPCDProvider)
    provider.cfg = {
        "camera": {"z_min": 0.25, "z_max": 1.65},
        "pointcloud": {},
    }
    geometry, reason = provider._compute_online_mask_geometry(frame, mask)

    assert geometry is not None, reason
    assert geometry.valid_depth_ratio == pytest.approx(0.5)
    assert geometry.depth_median == pytest.approx(0.8)
    assert geometry.depth_spread == pytest.approx(0.0)


def _frame_evidence_provider(frame: RGBDFrame) -> ObjectPCDProvider:
    provider = object.__new__(ObjectPCDProvider)
    provider._target_generation = 7
    provider._frame_mask_evidence_scope = None
    provider._frame_mask_evidence_registry = {}
    provider._pending_full_target_geometry_commit = None
    provider._begin_frame_mask_evidence(frame)
    return provider


class _ProviderAppearanceProbabilityCacheTracker:
    """Record provider cache plumbing without duplicating tracker validation."""

    def __init__(self):
        self.cache_token = object()
        self.build_calls = []
        self.stats_calls = []
        self.mask_calls = []

    def target_appearance_probability_roi(self, frame, root_mask):
        self.build_calls.append((int(frame.frame_id), np.asarray(root_mask).copy()))
        return self.cache_token

    def target_appearance_support_stats(
        self, frame, mask, *, threshold=None, probability_roi=None
    ):
        selected = np.asarray(mask) > 0
        self.stats_calls.append((int(frame.frame_id), probability_roi))
        return (0.9, 0.8, int(selected.sum()))

    def target_appearance_supported_mask(
        self, frame, mask, *, threshold=None, probability_roi=None
    ):
        self.mask_calls.append((int(frame.frame_id), probability_roi))
        return (np.asarray(mask) > 0).astype(np.uint8)


def test_raw_frame_evidence_reuses_one_appearance_probability_for_subsets():
    frame = _frame(3)
    raw_mask = _mask(5, 5, 15, 17)
    provider = _frame_evidence_provider(frame)
    tracker = _ProviderAppearanceProbabilityCacheTracker()
    provider.tracker = tracker

    evidence = provider._defer_full_target_geometry_commit(frame, raw_mask)
    assert evidence is not None
    assert evidence.appearance_probability_roi is tracker.cache_token
    assert len(tracker.build_calls) == 1

    subset = raw_mask.copy()
    subset[5:8, 5:15] = 0
    stats, reason = provider._immutable_target_appearance_stats(
        frame, subset, threshold=0.42
    )
    assert stats == (0.9, 0.8, int(subset.sum())), reason
    supported = provider._immutable_target_appearance_supported_mask(
        frame, subset, threshold=0.42
    )
    np.testing.assert_array_equal(supported, subset)
    assert tracker.stats_calls == [(3, tracker.cache_token)]
    assert tracker.mask_calls == [(3, tracker.cache_token)]
    assert provider._appearance_probability_roi_build_count == 1
    assert provider._appearance_probability_roi_query_count == 2
    assert provider._appearance_probability_roi_hit_count == 2

    # A sanitizer token never builds or replaces the raw-root cache.
    sanitized, sanitized_reason = provider._register_frame_mask_evidence(
        frame,
        subset,
        evidence_kind="sanitized_target_core",
        authority_eligible=False,
    )
    assert sanitized is not None, sanitized_reason
    assert sanitized.appearance_probability_roi is None
    assert len(tracker.build_calls) == 1

    # Exact timestamp scope mismatch falls back to the legacy scorer call.
    changed_timestamp = copy.copy(frame)
    changed_timestamp.timestamp = float(np.nextafter(frame.timestamp, np.inf))
    fallback, fallback_reason = provider._immutable_target_appearance_stats(
        changed_timestamp, subset, threshold=0.42
    )
    assert fallback is not None, fallback_reason
    assert tracker.stats_calls[-1] == (3, None)


def test_exact_frame_appearance_queries_are_bit_exact_memoized_and_fail_safe():
    """Only immutable exact tokens deduplicate repeated gate computations."""

    frame = _frame(3)
    raw_mask = _mask(5, 5, 15, 17)
    provider = _frame_evidence_provider(frame)
    tracker = _ProviderAppearanceProbabilityCacheTracker()
    provider.tracker = tracker

    raw_evidence = provider._defer_full_target_geometry_commit(frame, raw_mask)
    assert raw_evidence is not None
    subset = raw_mask.copy()
    subset[5:8, 5:15] = 0
    sanitized, sanitized_reason = provider._register_frame_mask_evidence(
        frame,
        subset,
        evidence_kind="sanitized_target_core",
        authority_eligible=False,
    )
    assert sanitized is not None, sanitized_reason

    first_stats, first_reason = provider._immutable_target_appearance_stats(
        frame, sanitized.mask, threshold=0.42
    )
    first_supported = provider._immutable_target_appearance_supported_mask(
        frame, sanitized.mask, threshold=0.42
    )
    second_stats, second_reason = provider._immutable_target_appearance_stats(
        frame, sanitized.mask, threshold=0.42
    )
    second_supported = provider._immutable_target_appearance_supported_mask(
        frame, sanitized.mask, threshold=0.42
    )

    assert second_stats == first_stats
    assert second_reason == first_reason
    assert first_supported is second_supported
    assert first_supported is not None
    assert not first_supported.flags.writeable
    np.testing.assert_array_equal(first_supported, subset)
    assert first_supported.dtype == subset.dtype
    assert first_supported.tobytes(order="C") == subset.tobytes(order="C")
    assert len(tracker.stats_calls) == 1
    assert len(tracker.mask_calls) == 1
    # The existing raw-probability diagnostics retain one query/hit per
    # logical gate request even when the downstream CPU reduction is memoized.
    assert provider._appearance_probability_roi_query_count == 4
    assert provider._appearance_probability_roi_hit_count == 4

    # Equal bytes in caller-owned mutable storage must not inherit the exact
    # provider token or its cached gate decision.
    external_equal_copy = sanitized.mask.copy()
    copied_stats, copied_reason = provider._immutable_target_appearance_stats(
        frame, external_equal_copy, threshold=0.42
    )
    copied_supported = provider._immutable_target_appearance_supported_mask(
        frame, external_equal_copy, threshold=0.42
    )
    assert copied_stats == first_stats
    assert copied_reason == first_reason
    np.testing.assert_array_equal(copied_supported, first_supported)
    assert len(tracker.stats_calls) == 2
    assert len(tracker.mask_calls) == 2

    provider._begin_frame_mask_evidence(_frame(4))
    assert provider._frame_appearance_stats_cache == {}
    assert provider._frame_appearance_supported_mask_cache == {}


def test_frame_mask_evidence_reuses_one_compute_and_preserves_geometry_bytes(
    monkeypatch,
):
    frame = _frame(3)
    mask = _mask(5, 5, 13, 15)
    provider = _frame_evidence_provider(frame)
    original = ObjectPCDProvider._compute_online_mask_geometry
    calls = []

    def counted(current_frame, current_mask):
        calls.append((int(current_frame.frame_id), id(current_mask)))
        return original(provider, current_frame, current_mask)

    monkeypatch.setattr(provider, "_compute_online_mask_geometry", counted)
    evidence, reason = provider._register_frame_mask_evidence(
        frame,
        mask,
        evidence_kind="raw_exact",
        authority_eligible=True,
    )
    assert evidence is not None, reason
    assert len(calls) == 1
    assert evidence.mask.dtype == np.uint8
    assert evidence.mask.flags.c_contiguous
    assert not evidence.mask.flags.writeable
    storage_owner = evidence.mask
    while isinstance(storage_owner, np.ndarray) and storage_owner.base is not None:
        storage_owner = storage_owner.base
    assert storage_owner is evidence.mask_bytes
    assert evidence.mask.tobytes() == np.ascontiguousarray(mask).tobytes()

    first, first_reason = provider._online_mask_geometry(frame, evidence.mask)
    second, second_reason = provider._online_mask_geometry(frame, evidence.mask)
    assert first is second is evidence.geometry
    assert first_reason == second_reason == "ok"
    assert len(calls) == 1

    uncached, uncached_reason = original(provider, frame, mask)
    assert uncached is not None, uncached_reason
    np.testing.assert_array_equal(first.mask, uncached.mask)
    np.testing.assert_array_equal(first.bbox_xyxy, uncached.bbox_xyxy)
    np.testing.assert_array_equal(first.centroid_xy, uncached.centroid_xy)
    assert first.area == uncached.area
    assert first.bbox_area == uncached.bbox_area
    assert first.valid_depth_ratio == uncached.valid_depth_ratio
    assert first.depth_median == uncached.depth_median
    assert first.depth_spread == uncached.depth_spread


def test_frame_mask_evidence_external_copy_mutation_and_scope_mismatch_miss(
    monkeypatch,
):
    frame = _frame(4)
    mask = _mask(6, 4, 15, 17)
    provider = _frame_evidence_provider(frame)
    original = ObjectPCDProvider._compute_online_mask_geometry
    calls = []

    def counted(current_frame, current_mask):
        calls.append((int(current_frame.frame_id), id(current_mask)))
        return original(provider, current_frame, current_mask)

    monkeypatch.setattr(provider, "_compute_online_mask_geometry", counted)
    evidence, reason = provider._register_frame_mask_evidence(
        frame,
        mask,
        evidence_kind="raw_exact",
        authority_eligible=True,
    )
    assert evidence is not None, reason
    assert len(calls) == 1

    # Equal bytes are not evidence: only the immutable provider-owned object
    # identity may hit the registry.
    equal_copy = evidence.mask.copy()
    copied_geometry, copied_reason = provider._online_mask_geometry(frame, equal_copy)
    assert copied_geometry is not None, copied_reason
    assert len(calls) == 2
    equal_copy[0, 0] = 1
    mutated_geometry, mutated_reason = provider._online_mask_geometry(frame, equal_copy)
    assert mutated_geometry is not None, mutated_reason
    assert len(calls) == 3
    with pytest.raises(ValueError):
        evidence.mask[0, 0] = 1

    changed_timestamp = copy.copy(frame)
    changed_timestamp.timestamp = float(np.nextafter(frame.timestamp, np.inf))
    provider._online_mask_geometry(changed_timestamp, evidence.mask)
    assert len(calls) == 4
    provider._target_generation += 1
    provider._online_mask_geometry(frame, evidence.mask)
    assert len(calls) == 5


def test_sanitized_frame_evidence_is_distinct_and_never_raw_authority():
    frame = _frame(5)
    mask = _mask(7, 5, 16, 18)
    provider = _frame_evidence_provider(frame)
    raw, raw_reason = provider._register_frame_mask_evidence(
        frame,
        mask,
        evidence_kind="raw_exact",
        authority_eligible=True,
    )
    sanitized, sanitized_reason = provider._register_frame_mask_evidence(
        frame,
        mask,
        evidence_kind="sanitized_target_core",
        authority_eligible=False,
    )
    assert raw is not None, raw_reason
    assert sanitized is not None, sanitized_reason
    assert raw.mask is not sanitized.mask
    assert raw.mask_digest == sanitized.mask_digest
    assert raw.authority_eligible and not sanitized.authority_eligible

    reused = provider._defer_full_target_geometry_commit(
        frame,
        sanitized.mask,
        raw_exact_eligible=True,
    )
    assert reused is sanitized
    pending = provider._pending_full_target_geometry_commit
    assert pending is not None
    assert pending.mask is sanitized.mask
    assert not pending.raw_exact_eligible
    assert not pending.bootstrap_eligible
    assert pending.eligibility_kind == "known_partial"

    provider._full_target_geometry_authority = None
    provider._trusted_online_sam2_publication = None
    assert not provider._commit_full_target_geometry_authority(
        frame, sanitized.mask
    )
    assert provider._full_target_geometry_authority is None
    assert not provider._commit_trusted_online_sam2_publication(
        frame, sanitized.mask
    )
    assert provider._trusted_online_sam2_publication is None


def test_frame_mask_evidence_registry_is_revoked_on_public_step_exception():
    frame = _frame(6)
    provider = _frame_evidence_provider(frame)
    evidence, reason = provider._register_frame_mask_evidence(
        frame,
        _mask(8, 6, 17, 19),
        evidence_kind="raw_exact",
        authority_eligible=True,
    )
    assert evidence is not None, reason
    assert provider._frame_mask_evidence_registry

    def fail_step():
        raise RuntimeError("synthetic provider failure")

    provider._step_mask_only_impl = fail_step
    with pytest.raises(RuntimeError, match="synthetic provider failure"):
        provider.step_mask_only()

    assert provider._pending_full_target_geometry_commit is None
    assert provider._frame_mask_evidence_scope is None
    assert provider._frame_mask_evidence_registry == {}


def test_frame_mask_evidence_registry_is_revoked_on_rollback():
    frame = _frame(6)
    provider = _frame_evidence_provider(frame)
    evidence, reason = provider._register_frame_mask_evidence(
        frame,
        _mask(8, 6, 17, 19),
        evidence_kind="raw_exact",
        authority_eligible=True,
    )
    assert evidence is not None, reason
    assert provider._frame_mask_evidence_registry

    provider._last_final_clean_publication = None
    provider._full_target_geometry_authority = None
    provider._trusted_online_sam2_publication = None
    provider._full_target_scale_transition_hypothesis = None
    provider._visible_exact_partial_continuity = None
    provider._boundary_exit_continuity = None
    provider._trusted_online_sam2_continuity_intact = False
    provider._trusted_online_sam2_continuity_reason = "test"
    provider._trusted_online_sam2_transient_gap_active = False
    snapshot = provider._publication_authority_snapshot()
    provider._restore_publication_authority_snapshot(snapshot)

    assert provider._frame_mask_evidence_scope is None
    assert provider._frame_mask_evidence_registry == {}


def test_online_recovery_can_confirm_across_two_missed_inference_frames(
    monkeypatch,
):
    seed = _mask(4, 5, 12, 15)
    tracker = _FakeTracker({})
    manager = _FakeOnlineSAM2Manager()
    provider, _camera = _provider(monkeypatch, [_frame(5)], tracker, manager)
    assert provider.initialize_from_mask(_frame(1), seed)
    geometry2, reason2 = provider._online_mask_geometry(_frame(2), seed)
    geometry5, reason5 = provider._online_mask_geometry(_frame(5), seed)
    assert geometry2 is not None, reason2
    assert geometry5 is not None, reason5

    confirmed2, hits2, _ = provider._confirm_online_recovery(2, geometry2)
    confirmed5, hits5, _ = provider._confirm_online_recovery(5, geometry5)

    assert not confirmed2 and hits2 == 1
    assert confirmed5 and hits5 == 2
    provider.stop()


def test_unified_recovery_counts_sam2_two_plus_one_distinct_provider_frame(
    monkeypatch,
):
    """Fast motion with little bbox IoU still commits in three total frames."""

    seed = _large_mask((5, 30, 25, 50))
    recovered2 = _large_mask((10, 30, 30, 50))
    # Recorded fast-entry data reaches about 2134 px/s. Exercise essentially
    # that rate with zero bbox overlap, not merely ordinary hand-held motion.
    recovered3 = _large_mask((80, 30, 100, 50))  # 2100 px/s.
    recovered4 = _large_mask((150, 30, 170, 50))
    empty = np.zeros_like(seed)
    tracker = _IdentityAwareFakeTracker(
        {
            2: _mask_result(empty, valid=False, message="adaptive lost"),
            3: _mask_result(empty, valid=False, message="adaptive lost"),
            4: _mask_result(recovered4, message="tentative fast track"),
        }
    )
    manager = _FakeOnlineSAM2Manager(
        {
            2: _video_result(2, recovered2),
            3: _video_result(3, recovered3),
            4: _video_result(4, recovered4),
        }
    )
    provider, _camera = _provider(
        monkeypatch,
        [
            _moving_target_frame(2, (10, 30, 30, 50)),
            _moving_target_frame(3, (80, 30, 100, 50)),
            _moving_target_frame(4, (150, 30, 170, 50)),
        ],
        tracker,
        manager,
        recovery_publication_mode="unified_three_evidence",
    )
    provider.online_sam2_cfg["mask_publication_mode"] = "guarded_sam2_primary"
    assert provider.initialize_from_mask(
        _moving_target_frame(1, (5, 30, 25, 50)), seed
    )
    provider._break_trusted_online_sam2_continuity("test entered LOST")
    provider._tracking_committed = False

    outputs = [provider.step() for _ in range(3)]

    assert [result.valid for _, result, _, _ in outputs] == [
        False,
        False,
        True,
    ], "\n".join(result.message for _, result, _, _ in outputs)
    assert "pending 1/2" in outputs[0][1].message
    assert "pending 2/3" in outputs[1][1].message.lower()
    assert "stable 3/3" in outputs[2][1].message.lower()
    np.testing.assert_array_equal(outputs[2][1].mask, recovered4)
    assert provider.tracking_committed
    provider.stop()


def test_unified_recovery_pauses_on_exact_empty_without_losing_evidence(
    monkeypatch,
):
    seed = _large_mask((40, 30, 80, 70))
    recovered = _large_mask((43, 30, 83, 70))
    empty = np.zeros_like(seed)
    tracker = _IdentityAwareFakeTracker(
        {
            2: _mask_result(empty, valid=False, message="adaptive lost"),
            3: _mask_result(empty, valid=False, message="adaptive lost"),
            4: _mask_result(empty, valid=False, message="adaptive lost"),
            5: _mask_result(recovered, message="tentative recovery"),
        }
    )
    manager = _FakeOnlineSAM2Manager(
        {
            2: _video_result(2, recovered),
            3: _video_result(3, empty, valid=False, message="occluded"),
            4: _video_result(4, recovered),
            5: _video_result(5, recovered),
        }
    )
    provider, _camera = _provider(
        monkeypatch,
        [
            _moving_target_frame(frame_id, (43, 30, 83, 70))
            for frame_id in (2, 3, 4, 5)
        ],
        tracker,
        manager,
        recovery_publication_mode="unified_three_evidence",
    )
    provider.online_sam2_cfg["mask_publication_mode"] = "guarded_sam2_primary"
    assert provider.initialize_from_mask(
        _moving_target_frame(1, (40, 30, 80, 70)), seed
    )
    provider._break_trusted_online_sam2_continuity("test entered LOST")
    provider._tracking_committed = False

    outputs = [provider.step() for _ in range(4)]

    assert [result.valid for _, result, _, _ in outputs] == [
        False,
        False,
        False,
        True,
    ], "\n".join(result.message for _, result, _, _ in outputs)
    assert "no evidence; probation paused" in outputs[1][1].message
    assert "pending 2/3" in outputs[2][1].message.lower()
    assert "stable 3/3" in outputs[3][1].message.lower()
    provider.stop()


def test_unified_no_evidence_expires_after_three_frame_freshness_window(
    monkeypatch,
):
    seed = _mask(4, 5, 12, 15)
    empty = np.zeros_like(seed)
    tracker = _FakeTracker(
        {
            frame_id: _mask_result(empty, valid=False, message="adaptive lost")
            for frame_id in range(2, 7)
        }
    )
    manager = _FakeOnlineSAM2Manager(
        {
            2: _video_result(2, seed),
            **{
                frame_id: _video_result(
                    frame_id,
                    empty,
                    valid=False,
                    message="occluded",
                )
                for frame_id in range(3, 7)
            },
        }
    )
    provider, _camera = _provider(
        monkeypatch,
        [_frame(frame_id) for frame_id in range(2, 7)],
        tracker,
        manager,
        recovery_publication_mode="unified_three_evidence",
    )
    assert provider.initialize_from_mask(_frame(1), seed)

    provider.step()  # Exact SAM2 evidence at frame 2.
    for _ in range(3):
        provider.step()  # Frames 3..5 remain inside max frame gap 3.
    assert provider._online_sam2_recovery_hypothesis is not None

    _frame6, result6, _obj6, packet6 = provider.step()

    assert not result6.valid and not packet6.valid
    assert "probation expired" in result6.message
    assert provider._online_sam2_recovery_hypothesis is None
    assert provider._recovery_publication_hypothesis is None
    provider.stop()


def test_unified_online_recovery_explicit_conflict_clears_hypothesis(
    monkeypatch,
):
    seed = _mask(4, 5, 12, 15)
    conflicting = _mask(3, 3, 15, 17)  # 2.1x area, beyond pair gate.
    provider, _camera = _provider(
        monkeypatch,
        [_frame(3)],
        _FakeTracker({}),
        _FakeOnlineSAM2Manager(),
        recovery_publication_mode="unified_three_evidence",
    )
    assert provider.initialize_from_mask(_frame(1), seed)
    geometry2, reason2 = provider._online_mask_geometry(_frame(2), seed)
    geometry3, reason3 = provider._online_mask_geometry(_frame(3), conflicting)
    assert geometry2 is not None, reason2
    assert geometry3 is not None, reason3

    confirmed2, hits2, _ = provider._confirm_online_recovery(
        2, geometry2, timestamp=_frame(2).timestamp
    )
    confirmed3, hits3, detail3 = provider._confirm_online_recovery(
        3, geometry3, timestamp=_frame(3).timestamp
    )

    assert not confirmed2 and hits2 == 1
    assert not confirmed3 and hits3 == 0
    assert "explicit conflict" in detail3
    assert provider._online_sam2_recovery_hypothesis is None
    provider.stop()


def test_unified_online_recovery_depth_conflict_clears_immediately(
    monkeypatch,
):
    seed = _mask(4, 5, 12, 15)
    provider, _camera = _provider(
        monkeypatch,
        [_frame(3)],
        _FakeTracker({}),
        _FakeOnlineSAM2Manager(),
        recovery_publication_mode="unified_three_evidence",
    )
    assert provider.initialize_from_mask(_frame(1), seed)
    frame2 = _frame(2)
    frame3 = _frame(3)
    frame3.depth_raw[:] = 900  # 0.20 m jump in 1/30 s is conflicting.
    geometry2, reason2 = provider._online_mask_geometry(frame2, seed)
    geometry3, reason3 = provider._online_mask_geometry(frame3, seed)
    assert geometry2 is not None, reason2
    assert geometry3 is not None, reason3

    provider._confirm_online_recovery(2, geometry2, timestamp=frame2.timestamp)
    confirmed3, hits3, detail3 = provider._confirm_online_recovery(
        3, geometry3, timestamp=frame3.timestamp
    )

    assert not confirmed3 and hits3 == 0
    assert "depth_step" in detail3
    assert provider._online_sam2_recovery_hypothesis is None
    provider.stop()


def test_online_sam2_lifecycle_stays_on_one_worker_and_stop_cleans_it(
    monkeypatch,
):
    main_thread = threading.get_ident()
    seed = _mask(4, 5, 12, 15)
    temporal = _mask(5, 5, 13, 15)
    tracker = _FakeTracker({2: _mask_result(temporal)})
    manager = _FakeOnlineSAM2Manager({2: _video_result(2, temporal)})
    provider, camera = _provider(monkeypatch, [_frame(2)], tracker, manager)
    assert provider.initialize_from_mask(_frame(1), seed)
    provider.step()

    provider.stop()
    provider.stop()  # Cleanup is idempotent, including an externally-owned service.

    operations = [call[0] for call in manager.calls]
    assert operations == ["start", "initialize", "track", "reset", "close"]
    assert len(set(manager.thread_ids)) == 1
    assert manager.thread_ids[0] != main_thread
    assert manager.closed and camera.stopped
    assert provider._online_sam2_executor is None


def test_perception_rate_ticks_without_publisher_but_publish_rate_does_not():
    from dynamic_pcd.apps import realtime_masked_pcd as app

    class Monitor:
        def __init__(self):
            self.values = []

        def tick(self, valid):
            self.values.append(bool(valid))

    perception_monitor = Monitor()
    publish_monitor = Monitor()

    app.record_packet_rates(
        packet_valid=True,
        perception_monitor=perception_monitor,
        publish_monitor=publish_monitor,
        published=False,
    )
    app.record_packet_rates(
        packet_valid=False,
        perception_monitor=perception_monitor,
        publish_monitor=publish_monitor,
        published=False,
    )
    app.record_packet_rates(
        packet_valid=True,
        perception_monitor=perception_monitor,
        publish_monitor=publish_monitor,
        published=True,
    )

    assert perception_monitor.values == [True, False, True]
    assert publish_monitor.values == [True]


def _source_decomposed_partial_case(
    monkeypatch,
    *,
    frame_id=52,
    appendage_bbox=(98, 48, 150, 52),
    core_bbox=(60, 30, 100, 70),
    supported_core=None,
    normalized_raw=None,
    sanitized_lineage=True,
    stage=True,
):
    """Build one exact raw-envelope violation with an owner-aligned clean core."""

    initial = _large_mask((40, 30, 80, 70))
    provider, tracker = _guarded_primary_provider(monkeypatch, [], {}, initial)
    provider._guarded_v2_bootstrap_phase = "commissioned"
    provider._break_trusted_online_sam2_continuity("focused source decomposition")
    previous = _install_confirmed_raw_partial_owner(
        provider, frame_id=51, bbox=core_bbox
    )
    previous.sanitized_handoff_eligible = bool(sanitized_lineage)
    previous.sanitized_anchor_mask = None
    previous.sanitized_anchor_digest = None
    previous.sanitized_anchor_frame_id = None
    previous.sanitized_anchor_phase = "fixed"
    previous.broad_visible_anchor_mask = None
    previous.broad_visible_anchor_digest = None
    previous.broad_visible_anchor_bbox_xyxy = None
    previous.broad_visible_anchor_center_xy = None
    previous.broad_visible_anchor_area = None
    previous.broad_visible_anchor_frame_id = None
    owner_frame = _moving_target_frame(51, core_bbox)
    if sanitized_lineage:
        previous.mask_evidence = None
    else:
        provider._begin_frame_mask_evidence(owner_frame)
        owner_evidence, owner_reason = provider._register_frame_mask_evidence(
            owner_frame,
            previous.mask,
            evidence_kind="raw_exact",
            authority_eligible=True,
        )
        assert owner_evidence is not None, owner_reason
        previous.mask = owner_evidence.mask
        previous.mask_digest = owner_evidence.mask_digest
        previous.mask_evidence = owner_evidence
    assert provider._commit_final_clean_publication(
        owner_frame, previous.mask, rebase_motion_history=True
    )

    core = _large_mask(core_bbox)
    appendage = _large_mask(appendage_bbox)
    raw = np.logical_or(core > 0, appendage > 0).astype(np.uint8)
    frame = _moving_target_frame(
        frame_id, core_bbox, contaminant_bbox=appendage_bbox
    )
    probability = np.full(raw.shape, 0.05, dtype=np.float32)
    probability[(core if supported_core is None else supported_core) > 0] = 0.95
    _install_broad_visible_core_probability_map(tracker, probability)
    if normalized_raw is not None:
        real_coverage = provider._scale_normalized_reference_coverage

        def focused_coverage(reference, reference_bbox, candidate, candidate_bbox):
            if int(np.count_nonzero(candidate)) == int(np.count_nonzero(raw)):
                return float(normalized_raw), "focused source147 raw shape"
            return real_coverage(reference, reference_bbox, candidate, candidate_bbox)

        monkeypatch.setattr(
            provider, "_scale_normalized_reference_coverage", focused_coverage
        )

    raw_evidence = provider._defer_full_target_geometry_commit(frame, raw)
    assert raw_evidence is not None
    provider._last_output_mask_source = "online_sam2_guarded_primary"
    provider._last_online_sam2_exact_result = True
    provider._online_sam2_initialized = True
    provider._online_sam2_seed_pending = False
    provider._online_sam2_failure_reported = False
    provider._online_sam2_recovery_hypothesis = None
    provider._recovery_publication_hypothesis = None
    raw_result = _exact_token_mask_result(
        raw_evidence.mask, message="focused exact raw envelope"
    )
    raw_result.source = "online_sam2_video"
    raw_ok, raw_reason = provider._visible_exact_partial_candidate_agrees(
        frame, raw_evidence.geometry, allow_boundary_support=True
    )
    raw_report = provider._last_visible_exact_partial_gate_report
    repaired = None
    repair_reason = "source decomposition not staged"
    if stage:
        repaired, repair_reason = (
            provider._guarded_visible_partial_source_decomposed_continuation(
                frame, raw_result, previous, raw_report
            )
        )
    return SimpleNamespace(
        provider=provider,
        tracker=tracker,
        frame=frame,
        previous=previous,
        core=core,
        raw=raw,
        raw_evidence=raw_evidence,
        raw_result=raw_result,
        raw_ok=raw_ok,
        raw_reason=raw_reason,
        raw_report=raw_report,
        repaired=repaired,
        repair_reason=repair_reason,
    )


def _publish_source_decomposed_case(case):
    provider = case.provider
    assert case.repaired is not None, case.repair_reason
    provider._last_output_mask_source = "publication_sanitized_target_core"
    provider._last_online_sam2_status = (
        "guarded_primary_visible_partial_sanitized_continuation: "
        "source_decomposed focused"
    )
    obj = provider._mask_only_publication_evidence(
        case.frame, case.repaired.mask, preserve_sanitized_core=True
    )
    assert obj.valid
    published, published_obj = provider._apply_recovery_publication_gate(
        case.frame, case.repaired, obj, mask_only=True
    )
    assert published.valid and published_obj.valid
    return published, published_obj


def test_source326_bbox_violation_decomposes_to_original_gate_clean_core(
    monkeypatch,
):
    case = _source_decomposed_partial_case(monkeypatch)
    provider = case.provider
    try:
        assert not case.raw_ok
        assert case.raw_report.violations == ("log_bbox_rate",)
        assert case.repaired is not None, case.repair_reason
        np.testing.assert_array_equal(case.repaired.mask, case.core)
        pending = provider._pending_full_target_geometry_commit
        assert pending.eligibility_kind == "source_decomposed_visible_core"
        proof = pending.source_decomposed_visible_core_proof
        seal = pending.source_decomposed_visible_core_stage_seal
        assert proof is not None and proof.core_report.accepted
        assert proof.core_report.violations == ()
        assert not proof.external_occluder_required
        assert proof.external_occluder_proof is None
        assert seal is not None
        assert provider._source_decomposed_visible_core_registry_seal(case.frame) is seal
        assert pending.sanitizer_external_occluder_stage_seal is None
        assert provider._sanitizer_external_occluder_registry_seal(case.frame) is None
    finally:
        provider.stop()


def test_source326_raw_owner_needs_no_prior_sanitizer_lineage(monkeypatch):
    """A confirmed raw owner may start the strictly sealed core episode."""

    case = _source_decomposed_partial_case(
        monkeypatch,
        sanitized_lineage=False,
    )
    provider = case.provider
    try:
        assert not case.raw_ok
        assert case.raw_report.violations == ("log_bbox_rate",)
        assert case.previous.evidence_kind == "raw_exact"
        assert not case.previous.sanitized_handoff_eligible
        assert case.repaired is not None, case.repair_reason
        np.testing.assert_array_equal(case.repaired.mask, case.core)
        pending = provider._pending_full_target_geometry_commit
        assert pending.eligibility_kind == "source_decomposed_visible_core"
        assert pending.source_decomposed_visible_core_proof is not None
        assert provider._source_decomposed_visible_core_registry_seal(
            case.frame
        ) is not None
    finally:
        provider.stop()


def test_source_decomposition_rechecks_core_against_published_core_geometry(
    monkeypatch,
):
    """A hand-contaminated prior raw envelope is audit-only for the core."""

    case = _source_decomposed_partial_case(monkeypatch, stage=False)
    provider = case.provider
    try:
        # The previously published mask/core is still the 40x40 target, but
        # its admission-only raw envelope contains a much larger hand branch.
        # The current raw envelope must fail against that motion envelope;
        # the current sanitized core must be judged against the previous
        # published core rather than inheriting the hand geometry.
        case.previous.bbox_xyxy = np.asarray([40, 20, 160, 80], dtype=np.int32)
        case.previous.center_xy = np.asarray([100.0, 50.0], dtype=np.float64)
        case.previous.area = 2200.0
        case.previous.bbox_area = 7200.0
        raw_ok, _ = provider._visible_exact_partial_candidate_agrees(
            case.frame,
            case.raw_evidence.geometry,
            allow_boundary_support=True,
        )
        report = provider._last_visible_exact_partial_gate_report
        assert not raw_ok
        assert "log_bbox_rate" in report.violations

        repaired, reason = (
            provider._guarded_visible_partial_source_decomposed_continuation(
                case.frame, case.raw_result, case.previous, report
            )
        )
        assert repaired is not None, reason
        np.testing.assert_array_equal(repaired.mask, case.core)
        proof = (
            provider._pending_full_target_geometry_commit
            .source_decomposed_visible_core_proof
        )
        assert proof.raw_report.previous_geometry_kind == "motion_envelope"
        assert proof.core_report.previous_geometry_kind == "published_core"
        assert proof.core_report.accepted
    finally:
        provider.stop()


def test_source_decomposition_accepts_only_sealed_fixed_sanitizer_anchor(
    monkeypatch,
):
    """A formal sanitizer recovery may supply the fixed non-ratcheting ruler."""

    case = _source_decomposed_partial_case(monkeypatch, stage=False)
    provider = case.provider
    try:
        previous = case.previous
        previous.evidence_kind = "recovery_sanitized_target_core"
        # A recovery_sanitized owner already crossed two exact SAM frames and
        # a later distinct RGB-D frame; its partial-owner convention is hits=2.
        previous.hits = 2
        previous.sanitized_handoff_eligible = True
        anchor = provider._bytes_backed_binary_mask(previous.mask)
        anchor_bbox = bbox_from_mask(anchor, min_area=1)
        anchor_center = provider._binary_mask_centroid(anchor, anchor_bbox)
        previous.sanitized_anchor_mask = anchor
        previous.sanitized_anchor_digest = provider._publication_mask_digest(anchor)
        previous.sanitized_anchor_bbox_xyxy = np.asarray(
            anchor_bbox, dtype=np.int32
        )
        previous.sanitized_anchor_center_xy = np.asarray(
            anchor_center, dtype=np.float64
        )
        previous.sanitized_anchor_area = float(np.count_nonzero(anchor))
        previous.sanitized_anchor_frame_id = int(previous.frame_id)
        previous.sanitized_anchor_phase = "fixed"
        previous.broad_visible_anchor_mask = None
        previous.broad_visible_anchor_digest = None
        previous.broad_visible_anchor_bbox_xyxy = None
        previous.broad_visible_anchor_center_xy = None
        previous.broad_visible_anchor_area = None
        previous.broad_visible_anchor_frame_id = None

        raw_ok, _ = provider._visible_exact_partial_candidate_agrees(
            case.frame,
            case.raw_evidence.geometry,
            allow_boundary_support=True,
        )
        report = provider._last_visible_exact_partial_gate_report
        assert not raw_ok
        repaired, reason = (
            provider._guarded_visible_partial_source_decomposed_continuation(
                case.frame, case.raw_result, previous, report
            )
        )
        assert repaired is not None, reason
        proof = (
            provider._pending_full_target_geometry_commit
            .source_decomposed_visible_core_proof
        )
        assert proof.fixed_anchor_frame_id == previous.frame_id
        np.testing.assert_array_equal(proof.fixed_anchor_mask, anchor)

        previous.sanitized_anchor_mask = np.asarray(anchor).copy()
        rejected, rejected_frame, rejected_reason = (
            provider._source_decomposition_fixed_anchor_from_previous(previous)
        )
        assert rejected is None and rejected_frame is None
        assert "token" in rejected_reason
    finally:
        provider.stop()


def test_source_decomposition_decouples_only_stable_core_from_reversing_raw(
    monkeypatch,
):
    """A moving hand may reverse the raw centroid, not the retained target."""

    case = _source_decomposed_partial_case(monkeypatch, stage=False)
    provider = case.provider
    try:
        previous = case.previous
        previous.evidence_kind = "source_decomposed_sanitized_target_core"
        previous.hits = 3
        previous.sanitized_handoff_eligible = True
        current_core = _large_mask((61, 30, 101, 70))
        left_hand = _large_mask((10, 48, 60, 52))
        current_raw = np.logical_or(
            current_core > 0, left_hand > 0
        ).astype(np.uint8)
        frame = _moving_target_frame(
            52, (61, 30, 101, 70), contaminant_bbox=(10, 48, 60, 52)
        )
        provider._begin_frame_mask_evidence(frame)
        _pending, current_core_token = _bind_current_sanitized_evidence(
            provider, frame, current_raw, current_core
        )
        monkeypatch.setattr(
            provider,
            "_aligned_committed_clean_mask",
            lambda *_args, **_kwargs: (
                current_core_token,
                "focused stable published-core alignment",
            ),
        )

        rejected, rejected_reason = provider._sanitized_partial_chain_accepts(
            frame,
            current_core_token,
            provider._pending_full_target_geometry_commit.mask,
            previous,
            allow_raw_exact_predecessor=True,
        )
        assert not rejected
        assert "reversed raw-envelope motion" in rejected_reason

        decoupled = []
        accepted, accepted_reason = provider._sanitized_partial_chain_accepts(
            frame,
            current_core_token,
            provider._pending_full_target_geometry_commit.mask,
            previous,
            allow_raw_exact_predecessor=True,
            allow_stable_core_raw_motion_decoupling=True,
            stable_core_raw_motion_decoupled_out=decoupled,
        )
        assert accepted, accepted_reason
        assert decoupled == [True]
        assert "stable published-core/raw-envelope" in accepted_reason

        # The exception is not an erosion shortcut: below 90% immediate-core
        # retention the same reversing raw envelope remains fail-closed.
        eroded = _large_mask((64, 34, 98, 67))
        provider._begin_frame_mask_evidence(frame)
        _pending, eroded_token = _bind_current_sanitized_evidence(
            provider, frame, current_raw, eroded
        )
        monkeypatch.setattr(
            provider,
            "_aligned_committed_clean_mask",
            lambda *_args, **_kwargs: (
                current_core,
                "focused eroded published-core alignment",
            ),
        )
        rejected_eroded, _ = provider._sanitized_partial_chain_accepts(
            frame,
            eroded_token,
            provider._pending_full_target_geometry_commit.mask,
            previous,
            allow_raw_exact_predecessor=True,
            allow_stable_core_raw_motion_decoupling=True,
        )
        assert not rejected_eroded
    finally:
        provider.stop()


def test_hard_raw_failure_may_use_only_bound_fixed_anchor_broad_fallback(
    monkeypatch,
):
    """A failed strict decomposition may still use the stronger broad seal."""

    case = _source_decomposed_partial_case(monkeypatch, stage=False)
    provider = case.provider
    try:
        assert not case.raw_ok
        assert case.raw_report.failure_kind == "hard_reject"
        repaired, reason = (
            provider._guarded_visible_partial_sanitized_continuation(
                case.frame,
                case.raw_result,
                case.previous,
                raw_failure_kind="hard_reject",
                raw_failure_report=case.raw_report,
            )
        )
        assert repaired is not None, reason
        assert "fixed-anchor broad core" in reason
        pending = provider._pending_full_target_geometry_commit
        assert pending.broad_visible_core_proof is not None
        assert not pending.raw_exact_eligible
        assert not pending.bootstrap_eligible

        # The source label alone is not permission to reinterpret a hard
        # failure.  Without the exact frozen report the same helper refuses.
        provider._begin_frame_mask_evidence(case.frame)
        raw_evidence = provider._defer_full_target_geometry_commit(
            case.frame, case.raw
        )
        forged = _exact_token_mask_result(
            raw_evidence.mask, message="hard failure without report"
        )
        rejected, rejected_reason = (
            provider._guarded_visible_partial_sanitized_continuation(
                case.frame,
                forged,
                case.previous,
                raw_failure_kind="hard_reject",
            )
        )
        assert rejected is None
        assert "not repairable" in rejected_reason
    finally:
        provider.stop()


def test_source147_shape_only_uses_same_gate_and_exact_removed_partition(
    monkeypatch,
):
    case = _source_decomposed_partial_case(
        monkeypatch,
        frame_id=54,
        appendage_bbox=(98, 48, 112, 52),
        normalized_raw=0.80,
    )
    provider = case.provider
    try:
        assert not case.raw_ok
        assert case.raw_report.violations == ("shape_normalized",)
        assert case.repaired is not None, case.repair_reason
        proof = provider._pending_full_target_geometry_commit.source_decomposed_visible_core_proof
        removed = np.logical_and(case.raw > 0, case.core == 0).astype(np.uint8)
        assert proof.removed_pixels == int(np.count_nonzero(removed))
        assert proof.removed_mask_digest == provider._publication_mask_digest(removed)
        np.testing.assert_array_equal(
            case.raw > 0, np.logical_or(case.repaired.mask > 0, removed > 0)
        )
    finally:
        provider.stop()


def test_source330_commit_writes_core_not_raw_motion_geometry(monkeypatch):
    first = _source_decomposed_partial_case(monkeypatch)
    provider = first.provider
    try:
        published, published_obj = _publish_source_decomposed_case(first)
        full_before = copy.deepcopy(provider._full_target_geometry_authority)
        trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)
        provider._commit_final_publication_histories(
            first.frame, published, published_obj
        )
        owner = provider._visible_exact_partial_continuity
        assert owner.evidence_kind == "source_decomposed_sanitized_target_core"
        assert owner.area == pytest.approx(float(first.core.sum()))
        assert owner.bbox_area == pytest.approx(1600.0)
        assert owner.bbox_area < float(first.raw_evidence.geometry.bbox_area)
        assert provider._full_target_geometry_authority.frame_id == full_before.frame_id
        assert provider._trusted_online_sam2_publication.frame_id == trusted_before.frame_id

        # The same large raw bbox must still be rejected relative to the core
        # owner.  If raw audit geometry had become the owner this would be a
        # zero-rate raw continuation and the hand appendage would escape.
        second_frame = _moving_target_frame(
            53, (60, 30, 100, 70), contaminant_bbox=(98, 48, 150, 52)
        )
        second_evidence = provider._defer_full_target_geometry_commit(
            second_frame, first.raw
        )
        second_ok, _ = provider._visible_exact_partial_candidate_agrees(
            second_frame,
            second_evidence.geometry,
            allow_boundary_support=True,
        )
        second_report = provider._last_visible_exact_partial_gate_report
        assert not second_ok
        assert second_report.violations == ("log_bbox_rate",)
    finally:
        provider.stop()


@pytest.mark.parametrize("fault", ("gap", "depth", "nonfinite"))
def test_source_decomposition_rejects_nonpixel_raw_failures(monkeypatch, fault):
    frame_id = 55 if fault == "gap" else 52
    case = _source_decomposed_partial_case(
        monkeypatch, frame_id=frame_id, stage=False
    )
    provider = case.provider
    try:
        # Rebuild only the raw report for the requested non-pixel failure.  The
        # helper must reject before another sanitizer transaction is staged.
        if fault == "depth":
            case.frame.depth_raw[case.raw > 0] = 1000
            provider._begin_frame_mask_evidence(case.frame)
            case.raw_evidence = provider._defer_full_target_geometry_commit(
                case.frame, case.raw
            )
        elif fault == "nonfinite":
            case.previous.depth_median = float("nan")
        raw_ok, _ = provider._visible_exact_partial_candidate_agrees(
            case.frame,
            case.raw_evidence.geometry,
            allow_boundary_support=True,
        )
        report = provider._last_visible_exact_partial_gate_report
        assert not raw_ok
        expected = {"gap", "depth_step", "nonfinite"}
        assert set(report.violations) & expected
        repaired, reason = (
            provider._guarded_visible_partial_source_decomposed_continuation(
                case.frame,
                _exact_token_mask_result(
                    case.raw_evidence.mask, message="non-pixel raw failure"
                ),
                case.previous,
                report,
            )
        )
        assert repaired is None
        assert "gap/depth/nonfinite" in reason
        assert provider._source_decomposed_visible_core_registry_seal(case.frame) is None
    finally:
        provider.stop()


def test_source_decomposition_rejects_internal_erosion_and_absent(monkeypatch):
    tiny = _large_mask((76, 45, 84, 55))
    case = _source_decomposed_partial_case(monkeypatch, supported_core=tiny)
    provider = case.provider
    try:
        assert case.repaired is None
        assert (
            "failed original partial gate" in case.repair_reason
            or "fixed-anchor chain rejected" in case.repair_reason
            or "strict sanitizer rejected" in case.repair_reason
        )
        assert provider._source_decomposed_visible_core_registry_seal(case.frame) is None

        absent_frame = _moving_target_frame(53, (60, 30, 100, 70))
        absent = np.zeros_like(case.raw)
        result = provider._prepare_guarded_sam2_primary_mask(
            absent_frame, _mask_result(absent, message="reviewed absent source353")
        )
        assert not result.valid
        assert provider._source_decomposed_visible_core_registry_seal(absent_frame) is None
    finally:
        provider.stop()


@pytest.mark.parametrize(
    "tamper", ("proof", "owner", "registry", "frame_color", "frame_depth")
)
def test_source_decomposed_proof_tamper_rolls_back_atomic_owner(
    monkeypatch, tamper
):
    case = _source_decomposed_partial_case(monkeypatch)
    provider = case.provider
    try:
        published, published_obj = _publish_source_decomposed_case(case)
        pending = provider._pending_full_target_geometry_commit
        final_before = provider._last_final_clean_publication
        owner_before = provider._visible_exact_partial_continuity
        if tamper == "proof":
            pending.source_decomposed_visible_core_proof = replace(
                pending.source_decomposed_visible_core_proof,
                removed_pixels=(
                    pending.source_decomposed_visible_core_proof.removed_pixels + 1
                ),
            )
        elif tamper == "owner":
            owner_before.hits += 1
        elif tamper == "registry":
            provider._revoke_source_decomposed_visible_core_stage_registry()
        elif tamper == "frame_color":
            yy, xx = np.argwhere(case.raw > 0)[0]
            case.frame.color_bgr[yy, xx, 0] ^= np.uint8(1)
        else:
            yy, xx = np.argwhere(case.raw > 0)[0]
            case.frame.depth_raw[yy, xx] += np.uint16(1)
        with pytest.raises(RuntimeError, match="source-decomposed"):
            provider._commit_final_publication_histories(
                case.frame, published, published_obj
            )
        assert provider._last_final_clean_publication is final_before
        assert provider._visible_exact_partial_continuity is owner_before
        assert provider._pending_full_target_geometry_commit is None
    finally:
        provider.stop()


def test_source_decomposed_joint_pending_clear_cannot_erase_provider_latch(
    monkeypatch,
):
    """Clearing every mutable pending marker must still fail via registry."""

    case = _source_decomposed_partial_case(monkeypatch)
    provider = case.provider
    try:
        published, published_obj = _publish_source_decomposed_case(case)
        pending = provider._pending_full_target_geometry_commit
        final_before = provider._last_final_clean_publication
        owner_before = provider._visible_exact_partial_continuity
        assert provider._source_decomposed_visible_core_registry_seal(
            case.frame
        ) is not None
        pending.source_decomposed_visible_core_proof = None
        pending.source_decomposed_visible_core_stage_seal = None
        pending.eligibility_kind = "known_partial"
        pending.eligibility_reason = "joint pending-field tamper"
        pending.raw_exact_eligible = False
        pending.bootstrap_eligible = False
        with pytest.raises(RuntimeError, match="source-decomposed"):
            provider._commit_final_publication_histories(
                case.frame, published, published_obj
            )
        assert provider._last_final_clean_publication is final_before
        assert provider._visible_exact_partial_continuity is owner_before
        assert provider._pending_full_target_geometry_commit is None
        assert provider._source_decomposed_visible_core_registry_seal(
            case.frame
        ) is None
    finally:
        provider.stop()


@pytest.mark.parametrize("tamper", ("raw_equal_copy", "core_equal_copy"))
def test_source_decomposed_equal_copy_cannot_replace_exact_evidence(
    monkeypatch, tamper
):
    case = _source_decomposed_partial_case(monkeypatch)
    provider = case.provider
    try:
        pending = provider._pending_full_target_geometry_commit
        final_mask = case.repaired.mask
        if tamper == "raw_equal_copy":
            pending.mask = np.asarray(pending.mask).copy()
        else:
            final_mask = np.asarray(final_mask).copy()
        accepted, reason = provider._source_decomposed_visible_core_proof_accepts(
            case.frame,
            final_mask,
            pending,
            case.previous,
        )
        assert not accepted
        assert "changed" in reason or "disappeared" in reason
    finally:
        provider.stop()


def test_source_decomposed_seal_ignores_unread_background_rgbd(monkeypatch):
    """Pixels outside exact raw evidence cannot affect this mask decision."""

    case = _source_decomposed_partial_case(monkeypatch)
    provider = case.provider
    try:
        assert case.raw[0, 0] == 0
        case.frame.color_bgr[0, 0, 0] ^= np.uint8(1)
        case.frame.depth_raw[0, 0] += np.uint16(1)
        pending = provider._pending_full_target_geometry_commit
        accepted, reason = provider._source_decomposed_visible_core_proof_accepts(
            case.frame,
            case.repaired.mask,
            pending,
            case.previous,
        )
        assert accepted, reason
    finally:
        provider.stop()


def test_source_decomposed_registry_is_revoked_on_new_frame_and_reset(monkeypatch):
    case = _source_decomposed_partial_case(monkeypatch)
    provider = case.provider
    try:
        assert provider._source_decomposed_visible_core_registry_seal(
            case.frame
        ) is not None
        next_frame = _moving_target_frame(53, (60, 30, 100, 70))
        provider._begin_frame_mask_evidence(next_frame)
        assert provider._source_decomposed_visible_core_registry_seal(
            case.frame
        ) is None
        assert provider._source_decomposed_visible_core_registry_seal(
            next_frame
        ) is None
        provider._clear_frame_mask_evidence()
        assert provider._source_decomposed_visible_core_registry_seal(
            next_frame
        ) is None
    finally:
        provider.stop()


def _source_decomposition_external_only_case(
    monkeypatch, *, formal_recovery_owner=False
):
    supported = _large_mask((61, 31, 99, 68))  # 87.9% concentric retention
    case = _source_decomposed_partial_case(
        monkeypatch, supported_core=supported, stage=False
    )
    provider = case.provider
    previous = case.previous
    previous.hits = 2 if formal_recovery_owner else 3
    previous.evidence_kind = (
        "recovery_sanitized_target_core"
        if formal_recovery_owner
        else "sanitized_target_core"
    )
    previous.sanitized_anchor_mask = previous.mask.copy()
    previous.sanitized_anchor_digest = provider._publication_mask_digest(
        previous.sanitized_anchor_mask
    )
    previous.sanitized_anchor_bbox_xyxy = previous.bbox_xyxy.copy()
    previous.sanitized_anchor_center_xy = previous.center_xy.copy()
    previous.sanitized_anchor_area = float(np.count_nonzero(previous.mask))
    previous.sanitized_anchor_frame_id = int(previous.frame_id)
    previous.broad_visible_anchor_mask = previous.mask.copy()
    previous.broad_visible_anchor_digest = provider._publication_mask_digest(
        previous.broad_visible_anchor_mask
    )
    previous.broad_visible_anchor_bbox_xyxy = previous.bbox_xyxy.copy()
    previous.broad_visible_anchor_center_xy = previous.center_xy.copy()
    previous.broad_visible_anchor_area = float(np.count_nonzero(previous.mask))
    previous.broad_visible_anchor_frame_id = int(previous.frame_id)
    raw_ok, _ = provider._visible_exact_partial_candidate_agrees(
        case.frame, case.raw_evidence.geometry, allow_boundary_support=True
    )
    assert not raw_ok
    report = provider._last_visible_exact_partial_gate_report
    case.repaired, case.repair_reason = (
        provider._guarded_visible_partial_source_decomposed_continuation(
            case.frame, case.raw_result, previous, report
        )
    )
    return case


def test_source_decomposition_external_only_chain_must_be_sealed(monkeypatch):
    """Any external proof that changes chain rejection must be mandatory."""

    case = _source_decomposition_external_only_case(monkeypatch)
    provider = case.provider
    try:
        assert case.repaired is not None, case.repair_reason
        assert "external low-appearance occluder" in case.repair_reason
        pending = provider._pending_full_target_geometry_commit
        proof = pending.source_decomposed_visible_core_proof
        assert proof.external_occluder_required
        assert proof.external_occluder_proof is not None
        assert pending.sanitizer_external_occluder_stage_seal is not None
        assert provider._sanitizer_external_occluder_registry_seal(
            case.frame
        ) is pending.sanitizer_external_occluder_stage_seal
        full_before = copy.deepcopy(provider._full_target_geometry_authority)
        trusted_before = copy.deepcopy(provider._trusted_online_sam2_publication)
        published, published_obj = _publish_source_decomposed_case(case)
        provider._commit_final_publication_histories(
            case.frame, published, published_obj
        )
        owner = provider._visible_exact_partial_continuity
        assert owner.evidence_kind == "source_decomposed_sanitized_target_core"
        np.testing.assert_array_equal(owner.mask, case.repaired.mask)
        assert owner.area == pytest.approx(float(np.count_nonzero(case.repaired.mask)))
        assert provider._full_target_geometry_authority.frame_id == full_before.frame_id
        assert provider._trusted_online_sam2_publication.frame_id == trusted_before.frame_id
        assert provider._source_decomposed_visible_core_registry_seal(
            case.frame
        ) is None
        assert provider._sanitizer_external_occluder_registry_seal(
            case.frame
        ) is None
    finally:
        provider.stop()


def test_formal_recovery_hits2_may_use_sealed_external_source_decomposition(
    monkeypatch,
):
    """The 3/3 recovery owner does not need an extra fourth hit."""

    case = _source_decomposition_external_only_case(
        monkeypatch, formal_recovery_owner=True
    )
    provider = case.provider
    try:
        assert case.repaired is not None, case.repair_reason
        pending = provider._pending_full_target_geometry_commit
        proof = pending.source_decomposed_visible_core_proof
        assert proof.external_occluder_required
        assert pending.sanitizer_external_occluder_stage_seal is not None
    finally:
        provider.stop()


def test_source_decomposition_does_not_seal_unused_external_acceptance(
    monkeypatch,
):
    """A true detector result grants nothing when the ordinary chain passes."""

    case = _source_decomposed_partial_case(monkeypatch, stage=False)
    provider = case.provider
    try:
        monkeypatch.setattr(
            provider,
            "_external_occluder_invasion_accepts",
            lambda *args, **kwargs: (True, "focused unused external detector"),
        )
        repaired, reason = (
            provider._guarded_visible_partial_source_decomposed_continuation(
                case.frame,
                case.raw_result,
                case.previous,
                case.raw_report,
            )
        )
        assert repaired is not None, reason
        pending = provider._pending_full_target_geometry_commit
        proof = pending.source_decomposed_visible_core_proof
        assert not proof.external_occluder_required
        assert proof.external_occluder_proof is None
        assert pending.sanitizer_external_occluder_proof is None
        assert pending.sanitizer_external_occluder_stage_seal is None
        assert provider._sanitizer_external_occluder_registry_seal(
            case.frame
        ) is None
    finally:
        provider.stop()


@pytest.mark.parametrize("tamper", ("proof", "registry", "joint_clear"))
def test_source_decomposition_external_seal_tamper_rolls_back_atomically(
    monkeypatch, tamper
):
    case = _source_decomposition_external_only_case(monkeypatch)
    provider = case.provider
    try:
        published, published_obj = _publish_source_decomposed_case(case)
        pending = provider._pending_full_target_geometry_commit
        final_before = provider._last_final_clean_publication
        owner_before = provider._visible_exact_partial_continuity
        if tamper in ("proof", "joint_clear"):
            pending.sanitizer_external_occluder_proof = None
        if tamper in ("registry", "joint_clear"):
            provider._revoke_sanitizer_external_occluder_stage_registry()
        if tamper == "joint_clear":
            pending.sanitizer_external_occluder_proof_required = False
            pending.sanitizer_external_occluder_stage_seal = None
        with pytest.raises(RuntimeError, match="source-decomposed"):
            provider._commit_final_publication_histories(
                case.frame, published, published_obj
            )
        assert provider._last_final_clean_publication is final_before
        assert provider._visible_exact_partial_continuity is owner_before
        assert provider._pending_full_target_geometry_commit is None
        assert provider._source_decomposed_visible_core_registry_seal(
            case.frame
        ) is None
        assert provider._sanitizer_external_occluder_registry_seal(
            case.frame
        ) is None
    finally:
        provider.stop()
