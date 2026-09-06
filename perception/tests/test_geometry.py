import numpy as np
import pytest
from queue import SimpleQueue

from dynamic_pcd.provider.object_pcd_provider import ObjectPCDProvider
from dynamic_pcd.segmentation.roi_depth_tracker import ROIDepthTracker
from dynamic_pcd.segmentation.sam2_image import _select_best_mask
from dynamic_pcd.types import CameraIntrinsics, MaskResult, RGBDFrame
from dynamic_pcd.utils.geometry import bbox_from_mask, enlarge_bbox, bbox_to_mask
from dynamic_pcd.utils.vis import Open3DLiveViewer


def test_bbox_roundtrip():
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[10:20, 30:50] = 1
    bbox = bbox_from_mask(mask)
    assert bbox.tolist() == [30, 10, 50, 20]
    mask2 = bbox_to_mask(bbox, mask.shape)
    assert mask2.sum() == 200


@pytest.mark.parametrize("dtype", [np.uint8, np.bool_, np.float32])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_bbox_fast_path_is_equivalent_to_nonzero_coordinate_bounds(
    dtype,
    noncontiguous,
):
    base = np.zeros((48, 72), dtype=dtype)
    base[7:31, 19:54] = 1
    if np.issubdtype(dtype, np.floating):
        # Preserve the former ``astype(bool)`` semantics for unusual mask
        # values too: negative and NaN pixels are foreground.
        base[5, 60] = -2.0
        base[40, 3] = np.nan
    mask = base[:, ::-1] if noncontiguous else base

    ys, xs = np.where(mask.astype(bool))
    expected = np.asarray(
        [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
        dtype=np.int32,
    )

    np.testing.assert_array_equal(bbox_from_mask(mask), expected)


def test_bbox_fast_path_preserves_minimum_area_rejection():
    mask = np.zeros((30, 40), dtype=np.uint8)
    mask[3:5, 8:11] = 255

    assert bbox_from_mask(mask, min_area=7) is None
    np.testing.assert_array_equal(
        bbox_from_mask(mask, min_area=6),
        np.asarray([8, 3, 11, 5], dtype=np.int32),
    )


def test_enlarge_bbox():
    bbox = np.array([30, 10, 50, 20])
    e = enlarge_bbox(bbox, 2.0, width=120, height=100)
    assert e[0] < bbox[0]
    assert e[2] > bbox[2]


def test_sam2_selection_rejects_oversized_mask():
    large = np.zeros((80, 80), dtype=np.uint8)
    large[5:70, 5:70] = 1
    tight = np.zeros((80, 80), dtype=np.uint8)
    tight[20:30, 20:30] = 1
    masks = np.stack([large, tight], axis=0)
    scores = np.array([0.99, 0.80], dtype=np.float32)
    box = np.array([20, 20, 30, 30], dtype=np.int32)

    mask, bbox, score, valid, message = _select_best_mask(
        masks,
        scores,
        box,
        (80, 80),
        {
            "score_threshold": 0.0,
            "min_area": 1,
            "max_mask_area_ratio": 4.0,
            "max_bbox_area_ratio": 4.0,
            "keep_largest_component": False,
        },
    )

    assert valid
    assert score == scores[1]
    assert mask.sum() == 100
    assert bbox.tolist() == [20, 20, 30, 30]
    assert "mask_too_large" in message


def test_roi_depth_tracker_rejects_area_jump():
    intr = CameraIntrinsics(width=80, height=80, fx=100.0, fy=100.0, ppx=40.0, ppy=40.0)

    def make_frame(depth_raw: np.ndarray, frame_id: int) -> RGBDFrame:
        return RGBDFrame(
            color_bgr=np.zeros((80, 80, 3), dtype=np.uint8),
            depth_raw=depth_raw,
            depth_scale=0.001,
            intrinsics=intr,
            timestamp=float(frame_id),
            frame_id=frame_id,
        )

    init_depth = np.zeros((80, 80), dtype=np.uint16)
    init_depth[20:30, 20:30] = 1000
    init_mask = np.zeros((80, 80), dtype=np.uint8)
    init_mask[20:30, 20:30] = 1
    tracker = ROIDepthTracker(
        {
            "roi_scale": 3.0,
            "depth_tolerance": 0.02,
            "min_area": 1,
            "morph_kernel": 1,
            "lost_after": 8,
            "max_area_growth": 2.5,
        }
    )
    tracker.initialize(make_frame(init_depth, 0), mask=init_mask)

    update_depth = np.zeros((80, 80), dtype=np.uint16)
    update_depth[10:40, 10:40] = 1000
    res = tracker.update(make_frame(update_depth, 1))

    assert not res.valid
    assert res.mask.sum() == 100
    assert "reject area jump" in res.message


def test_roi_depth_tracker_rejects_growth_vs_initial():
    intr = CameraIntrinsics(width=80, height=80, fx=100.0, fy=100.0, ppx=40.0, ppy=40.0)

    def make_frame(depth_raw: np.ndarray, frame_id: int) -> RGBDFrame:
        return RGBDFrame(
            color_bgr=np.zeros((80, 80, 3), dtype=np.uint8),
            depth_raw=depth_raw,
            depth_scale=0.001,
            intrinsics=intr,
            timestamp=float(frame_id),
            frame_id=frame_id,
        )

    init_depth = np.zeros((80, 80), dtype=np.uint16)
    init_depth[20:30, 20:30] = 1000
    init_mask = np.zeros((80, 80), dtype=np.uint8)
    init_mask[20:30, 20:30] = 1
    tracker = ROIDepthTracker(
        {
            "roi_scale": 4.0,
            "depth_tolerance": 0.02,
            "min_area": 1,
            "morph_kernel": 1,
            "lost_after": 8,
            "max_area_growth": 10.0,
            "max_area_vs_initial": 2.0,
            "max_bbox_growth": 10.0,
        }
    )
    tracker.initialize(make_frame(init_depth, 0), mask=init_mask)

    update_depth = np.zeros((80, 80), dtype=np.uint16)
    update_depth[15:40, 15:40] = 1000
    res = tracker.update(make_frame(update_depth, 1))

    assert not res.valid
    assert res.mask.sum() == 100
    assert "reject area vs init" in res.message


def test_roi_depth_tracker_reinit_preserves_initial_size():
    intr = CameraIntrinsics(width=80, height=80, fx=100.0, fy=100.0, ppx=40.0, ppy=40.0)
    depth = np.full((80, 80), 1000, dtype=np.uint16)
    frame = RGBDFrame(
        color_bgr=np.zeros((80, 80, 3), dtype=np.uint8),
        depth_raw=depth,
        depth_scale=0.001,
        intrinsics=intr,
        timestamp=0.0,
        frame_id=0,
    )
    init_mask = np.zeros((80, 80), dtype=np.uint8)
    init_mask[20:30, 20:30] = 1
    reinit_mask = np.zeros((80, 80), dtype=np.uint8)
    reinit_mask[20:40, 20:40] = 1
    tracker = ROIDepthTracker({})

    tracker.initialize(frame, mask=init_mask)
    tracker.reinitialize_with_mask(frame, reinit_mask)

    assert tracker.state.initial_area == 100
    assert tracker.state.area == 400


def test_roi_depth_tracker_rejects_oversized_reinit():
    intr = CameraIntrinsics(width=80, height=80, fx=100.0, fy=100.0, ppx=40.0, ppy=40.0)
    depth = np.full((80, 80), 1000, dtype=np.uint16)
    frame = RGBDFrame(
        color_bgr=np.zeros((80, 80, 3), dtype=np.uint8),
        depth_raw=depth,
        depth_scale=0.001,
        intrinsics=intr,
        timestamp=0.0,
        frame_id=0,
    )
    init_mask = np.zeros((80, 80), dtype=np.uint8)
    init_mask[20:30, 20:30] = 1
    reinit_mask = np.zeros((80, 80), dtype=np.uint8)
    reinit_mask[10:40, 10:40] = 1
    tracker = ROIDepthTracker({"max_area_vs_initial": 2.0, "max_bbox_growth": 3.0})

    tracker.initialize(frame, mask=init_mask)
    res = tracker.reinitialize_with_mask(frame, reinit_mask)

    assert not res.valid
    assert res.mask.sum() == 100
    assert tracker.state.area == 100
    assert "reject reinit area vs init" in res.message


def test_roi_depth_tracker_stays_initialized_when_lost():
    intr = CameraIntrinsics(width=80, height=80, fx=100.0, fy=100.0, ppx=40.0, ppy=40.0)

    def make_frame(depth_raw: np.ndarray, frame_id: int) -> RGBDFrame:
        return RGBDFrame(
            color_bgr=np.zeros((80, 80, 3), dtype=np.uint8),
            depth_raw=depth_raw,
            depth_scale=0.001,
            intrinsics=intr,
            timestamp=float(frame_id),
            frame_id=frame_id,
        )

    init_depth = np.zeros((80, 80), dtype=np.uint16)
    init_depth[20:30, 20:30] = 1000
    init_mask = np.zeros((80, 80), dtype=np.uint8)
    init_mask[20:30, 20:30] = 1
    tracker = ROIDepthTracker(
        {
            "roi_scale": 1.5,
            "depth_tolerance": 0.01,
            "min_area": 1,
            "morph_kernel": 1,
            "lost_after": 0,
        }
    )

    tracker.initialize(make_frame(init_depth, 0), mask=init_mask)
    res = tracker.update(make_frame(np.zeros((80, 80), dtype=np.uint16), 1))

    assert not res.valid
    assert not tracker.state.valid
    assert tracker.initialized


def test_roi_lock_toggle_anchors_at_current_bbox():
    tracker = ROIDepthTracker({"lock_roi": False})
    tracker.state = type("State", (), {})()
    tracker.state.bbox_xyxy = np.array([12, 23, 45, 67], dtype=np.int32)
    tracker.fixed_bbox_xyxy = np.array([1, 2, 3, 4], dtype=np.int32)

    assert tracker.toggle_roi_lock()
    assert tracker.roi_locked
    assert tracker.fixed_bbox_xyxy.tolist() == [12, 23, 45, 67]

    assert not tracker.toggle_roi_lock()
    assert not tracker.roi_locked


def test_open3d_command_callback_only_queues_and_drains_fifo():
    # Construct without opening a real Open3D window; command queue behavior is
    # independent of rendering and must remain safe in headless CI.
    viewer = Open3DLiveViewer.__new__(Open3DLiveViewer)
    viewer._command_queue = SimpleQueue()
    select_cb = viewer._make_command_callback("T", "reselect_target")
    pause_cb = viewer._make_command_callback("P", "toggle_pause")

    assert select_cb(None) is False
    assert pause_cb(None) is False
    assert viewer.drain_commands() == ["reselect_target", "toggle_pause"]
    assert viewer.drain_commands() == []


def test_identity_rejects_different_color_candidate():
    intr = CameraIntrinsics(width=80, height=80, fx=100.0, fy=100.0, ppx=40.0, ppy=40.0)

    def make_frame(color_bgr: np.ndarray) -> RGBDFrame:
        return RGBDFrame(
            color_bgr=color_bgr,
            depth_raw=np.full((80, 80), 1000, dtype=np.uint16),
            depth_scale=0.001,
            intrinsics=intr,
            timestamp=0.0,
            frame_id=0,
        )

    provider = ObjectPCDProvider.__new__(ObjectPCDProvider)
    provider.cfg = {
        "tracker": {
            "identity_enabled": True,
            "identity_max_hist_dist": 0.2,
            "identity_max_center_jump_px": 0.0,
            "identity_max_depth_jump": 0.0,
            "identity_max_area_ratio": 10.0,
            "identity_update_alpha": 0.0,
        }
    }
    provider.identity_hist = None
    provider.identity_depth_median = 0.0
    provider.identity_center_uv = None
    provider.identity_initial_area = 0.0

    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[20:40, 20:40] = 1
    bbox = np.array([20, 20, 40, 40], dtype=np.int32)
    init_color = np.zeros((80, 80, 3), dtype=np.uint8)
    init_color[20:40, 20:40] = [0, 0, 255]
    candidate_color = np.zeros((80, 80, 3), dtype=np.uint8)
    candidate_color[20:40, 20:40] = [255, 0, 0]

    init_res = MaskResult(mask=mask, bbox_xyxy=bbox, score=1.0, valid=True)
    candidate_res = MaskResult(mask=mask, bbox_xyxy=bbox, score=1.0, valid=True)
    provider._update_identity_model(make_frame(init_color), init_res, "test")
    ok, reason = provider._identity_accepts_mask(make_frame(candidate_color), candidate_res, "candidate")

    assert not ok
    assert "hist_dist" in reason


def test_successful_manual_reselect_clears_old_identity_and_packet():
    intr = CameraIntrinsics(width=40, height=30, fx=100.0, fy=100.0, ppx=20.0, ppy=15.0)
    frame = RGBDFrame(
        color_bgr=np.zeros((30, 40, 3), dtype=np.uint8),
        depth_raw=np.full((30, 40), 1000, dtype=np.uint16),
        depth_scale=0.001,
        intrinsics=intr,
        timestamp=0.0,
        frame_id=0,
    )
    bbox = np.array([10, 8, 20, 18], dtype=np.int32)
    mask = bbox_to_mask(bbox, frame.depth_raw.shape)
    mask_res = MaskResult(mask=mask, bbox_xyxy=bbox, score=1.0, valid=True)

    class FakeTracker:
        def initialize(self, _frame, mask=None, bbox_xyxy=None):
            return MaskResult(mask=mask.copy(), bbox_xyxy=bbox_xyxy.copy(), score=1.0, valid=True)

    class FakeHistory:
        def __init__(self):
            self.reset_called = False

        def reset(self):
            self.reset_called = True

    provider = ObjectPCDProvider.__new__(ObjectPCDProvider)
    provider.cfg = {"tracker": {"init_method": "box"}}
    provider.sam2 = None
    provider.tracker = FakeTracker()
    provider.history = FakeHistory()
    provider.last_mask_result = None
    old_packet = object()
    provider.last_packet = old_packet
    provider.recovery_template = np.ones((2, 2), dtype=np.uint8)
    provider.recovery_template_mask = np.ones((2, 2), dtype=np.uint8)
    provider.recovery_template_bbox = np.array([1, 1, 3, 3])
    provider.identity_hist = np.ones((2, 2), dtype=np.float32)
    provider.identity_depth_median = 0.7
    provider.identity_center_uv = (4.0, 5.0)
    provider.identity_initial_area = 42.0
    provider._fallback_initialize = lambda _frame, _bbox: mask_res
    provider._log_mask_result = lambda *args, **kwargs: None
    provider._save_debug_mask = lambda *args, **kwargs: None

    observed = []

    def observe_new_target_state(_frame, _mask_res, source):
        observed.append(
            (
                source,
                provider.identity_hist,
                provider.identity_depth_median,
                provider.identity_center_uv,
                provider.identity_initial_area,
                provider.recovery_template,
                provider.last_packet,
            )
        )

    provider._update_recovery_template = observe_new_target_state

    assert provider.initialize_from_bbox(frame, bbox)
    assert provider.history.reset_called
    assert observed == [("tracker_init", None, 0.0, None, 0.0, None, None)]
    assert provider.last_packet is None


def test_cancelled_manual_reselect_preserves_old_target(monkeypatch):
    frame = object()
    old_packet = object()
    provider = ObjectPCDProvider.__new__(ObjectPCDProvider)
    provider.camera = type("Camera", (), {"get_frame": lambda self: frame})()
    provider.last_packet = old_packet
    provider.identity_hist = np.ones((2, 2), dtype=np.float32)

    monkeypatch.setattr(
        "dynamic_pcd.provider.object_pcd_provider.select_roi_bbox",
        lambda _frame: None,
    )

    assert not provider.select_and_initialize()
    assert provider.last_packet is old_packet
    assert provider.identity_hist is not None
