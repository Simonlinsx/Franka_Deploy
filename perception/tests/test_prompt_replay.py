import numpy as np
import pytest

from dynamic_pcd.segmentation.prompt_replay import (
    RecentRGBDFrameBuffer,
    RGBDReplayWindow,
    replay_prompt_mask,
)
from dynamic_pcd.types import CameraIntrinsics, MaskResult, RGBDFrame


def _frame(frame_id, bbox=(12, 15, 34, 43), *, timestamp=None, color=(220, 80, 235)):
    height, width = 64, 96
    image = np.full((height, width, 3), 35, dtype=np.uint8)
    depth = np.full((height, width), 900, dtype=np.uint16)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        image[y1:y2, x1:x2] = color
        depth[y1:y2, x1:x2] = 700
    return RGBDFrame(
        color_bgr=image,
        depth_raw=depth,
        depth_scale=0.001,
        intrinsics=CameraIntrinsics(
            width=width,
            height=height,
            fx=120.0,
            fy=120.0,
            ppx=(width - 1) / 2.0,
            ppy=(height - 1) / 2.0,
        ),
        timestamp=(float(frame_id) / 30.0 if timestamp is None else float(timestamp)),
        frame_id=int(frame_id),
    )


def _mask(frame, bbox):
    mask = np.zeros(frame.depth_raw.shape, dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    mask[y1:y2, x1:x2] = 1
    return mask


def _tracker_cfg():
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
        "component_max_motion_px": 24.0,
        "min_area_ratio": 0.35,
        "appearance_update_min_confidence": 0.82,
        "appearance_update_alpha": 0.025,
        "appearance_anchor_weight": 0.7,
        "reinit_min_color_score": 0.62,
        "optical_flow_enabled": True,
    }


def test_recent_buffer_copies_capture_arrays_and_evicts_by_time():
    buffer = RecentRGBDFrameBuffer(retention_s=2.0, capacity_frames=20)
    first = _frame(1, timestamp=0.0)
    original_pixel = first.color_bgr[0, 0].copy()
    buffer.append(first)
    first.color_bgr[:] = 255

    np.testing.assert_array_equal(buffer.get(1).color_bgr[0, 0], original_pixel)
    buffer.append(_frame(2, timestamp=0.5))
    buffer.append(_frame(3, timestamp=2.1))

    assert buffer.get(1) is None
    assert buffer.frame_ids == (2, 3)
    assert buffer.head.frame_id == 3


def test_recent_buffer_rejects_duplicate_or_out_of_order_frame_ids():
    buffer = RecentRGBDFrameBuffer(retention_s=2.0, capacity_frames=10)
    buffer.append(_frame(5))
    with pytest.raises(ValueError, match="strictly increasing"):
        buffer.append(_frame(5))
    with pytest.raises(ValueError, match="strictly increasing"):
        buffer.append(_frame(4))


def test_replay_window_uniformly_samples_at_most_twelve_and_keeps_endpoints():
    buffer = RecentRGBDFrameBuffer(retention_s=20.0, capacity_frames=40)
    for frame_id in range(1, 31):
        buffer.append(_frame(frame_id))

    window = buffer.replay_window(5, max_frames=12)

    assert window is not None
    assert window.available_frame_count == 26
    assert len(window.frames) == 12
    assert window.sampled_frame_ids[0] == 5
    assert window.sampled_frame_ids[-1] == 30
    assert all(
        right > left
        for left, right in zip(
            window.sampled_frame_ids, window.sampled_frame_ids[1:]
        )
    )


def test_replay_returns_exact_head_mask_and_strict_update_order():
    buffer = RecentRGBDFrameBuffer(retention_s=10.0, capacity_frames=20)
    for frame_id in range(1, 8):
        buffer.append(_frame(frame_id))
    window = buffer.replay_window(1, max_frames=4)
    calls = []

    class RecordingTracker:
        def __init__(self, cfg):
            assert cfg["generic"] is True

        def initialize(self, frame, mask=None):
            calls.append(("initialize", frame.frame_id))
            return MaskResult(
                mask=mask.copy(),
                bbox_xyxy=np.asarray([12, 15, 34, 43]),
                score=1.0,
                valid=True,
                message="source",
            )

        def update(self, frame):
            calls.append(("update", frame.frame_id))
            mask = np.zeros(frame.depth_raw.shape, dtype=np.uint8)
            mask[15:43, 12:34] = 1
            return MaskResult(
                mask=mask,
                bbox_xyxy=np.asarray([12, 15, 34, 43]),
                score=0.9,
                valid=True,
                message="tracked",
            )

    result = replay_prompt_mask(
        window,
        _mask(window.frames[0], (12, 15, 34, 43)),
        {"generic": True},
        tracker_factory=RecordingTracker,
    )

    assert result.valid
    assert result.target_frame_id == buffer.head.frame_id == 7
    assert result.sampled_frame_ids == window.sampled_frame_ids
    assert [frame_id for _, frame_id in calls] == list(window.sampled_frame_ids)
    assert all(
        right > left
        for left, right in zip(
            result.sampled_frame_ids, result.sampled_frame_ids[1:]
        )
    )
    assert result.mask.sum() > 0


def test_any_invalid_intermediate_update_fails_closed_and_stops_replay():
    buffer = RecentRGBDFrameBuffer(retention_s=10.0, capacity_frames=20)
    for frame_id in range(1, 7):
        buffer.append(_frame(frame_id))
    window = buffer.replay_window(1, max_frames=6)
    updated = []

    class FailingTracker:
        def __init__(self, _cfg):
            pass

        def initialize(self, frame, mask=None):
            return MaskResult(mask, np.asarray([12, 15, 34, 43]), 1.0, True)

        def update(self, frame):
            updated.append(frame.frame_id)
            valid = frame.frame_id != 3
            mask = np.ones(frame.depth_raw.shape, dtype=np.uint8) if valid else np.zeros(frame.depth_raw.shape, dtype=np.uint8)
            return MaskResult(
                mask,
                np.asarray([12, 15, 34, 43]),
                0.8 if valid else 0.0,
                valid,
                "visible" if valid else "complete occlusion",
            )

    result = replay_prompt_mask(
        window,
        _mask(window.frames[0], (12, 15, 34, 43)),
        {},
        tracker_factory=FailingTracker,
    )

    assert not result.valid
    assert result.failure_code == "tracker_update_invalid"
    assert result.mask.shape == buffer.head.depth_raw.shape
    assert result.mask.sum() == 0
    assert updated == [2, 3]
    assert "complete occlusion" in result.message


def test_real_adaptive_replay_tracks_generic_gradual_motion_to_head():
    buffer = RecentRGBDFrameBuffer(retention_s=10.0, capacity_frames=20)
    boxes = []
    for frame_id in range(1, 7):
        bbox = (12 + 2 * frame_id, 15 + frame_id, 34 + 2 * frame_id, 43 + frame_id)
        boxes.append(bbox)
        buffer.append(_frame(frame_id, bbox=bbox, color=(225, 90, 35)))
    window = buffer.replay_window(1, max_frames=12)

    result = replay_prompt_mask(
        window,
        _mask(window.frames[0], boxes[0]),
        _tracker_cfg(),
    )

    assert result.valid, result.message
    expected = _mask(window.frames[-1], boxes[-1]).astype(bool)
    predicted = result.mask.astype(bool)
    intersection = np.logical_and(expected, predicted).sum()
    union = np.logical_or(expected, predicted).sum()
    assert intersection / union > 0.85
    assert result.target_frame_id == 6


def test_real_adaptive_replay_does_not_bridge_complete_occlusion():
    visible_bbox = (18, 16, 42, 46)
    frames = (
        _frame(1, bbox=visible_bbox),
        _frame(2, bbox=None),
        _frame(3, bbox=(42, 16, 66, 46)),
    )
    window = RGBDReplayWindow(
        frames=frames,
        source_frame_id=1,
        head_frame_id=3,
        available_frame_count=3,
    )

    result = replay_prompt_mask(
        window,
        _mask(frames[0], visible_bbox),
        _tracker_cfg(),
    )

    assert not result.valid
    assert result.failure_code == "tracker_update_invalid"
    assert result.mask.sum() == 0


def test_evicted_source_never_produces_a_replay_window():
    buffer = RecentRGBDFrameBuffer(retention_s=0.5, capacity_frames=20)
    buffer.append(_frame(1, timestamp=0.0))
    buffer.append(_frame(2, timestamp=0.6))

    assert buffer.replay_window(1, max_frames=12) is None
    assert buffer.head.frame_id == 2
