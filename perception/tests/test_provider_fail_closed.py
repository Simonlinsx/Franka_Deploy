import numpy as np

from dynamic_pcd.pointcloud.extractor import ExtractedObjectPCD
from dynamic_pcd.provider.object_pcd_provider import ObjectPCDProvider
from dynamic_pcd.types import (
    CameraIntrinsics,
    MaskResult,
    ObjectPCDPacket,
    RGBDFrame,
)


def _frame(frame_id=2):
    return RGBDFrame(
        color_bgr=np.zeros((24, 32, 3), dtype=np.uint8),
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
        timestamp=float(frame_id),
        frame_id=frame_id,
    )


def _old_packet():
    points = np.ones((4, 3), dtype=np.float32)
    return ObjectPCDPacket(
        pcd_current=points,
        pcd_history=points[None],
        center=np.ones(3, dtype=np.float32),
        velocity=np.ones(3, dtype=np.float32),
        bbox_xyxy=np.asarray([1, 2, 8, 9], dtype=np.float32),
        timestamp=1.0,
        frame_id=1,
        valid=True,
        pcd_reference=points.copy(),
    )


def test_invalid_frame_never_carries_previous_object_payload():
    frame = _frame()

    class Camera:
        device_serial = "fake"

        def get_frame(self, *, timeout_ms):
            assert timeout_ms == 1000
            return frame

    class Tracker:
        initialized = True
        manages_identity = True
        state = type(
            "State",
            (),
            {
                "lost_count": 1,
                "last_mask": np.ones(frame.depth_raw.shape, dtype=np.uint8),
                "bbox_xyxy": np.asarray([3, 4, 10, 12], dtype=np.int32),
            },
        )()

        def snapshot_state(self):
            return None

        def update(self, _frame):
            return MaskResult(
                mask=np.zeros(frame.depth_raw.shape, dtype=np.uint8),
                bbox_xyxy=self.state.bbox_xyxy.copy(),
                score=0.0,
                valid=False,
                message="lost",
            )

    class Extractor:
        def extract(self, _frame, _mask):
            empty = np.zeros((0, 3), dtype=np.float32)
            return ExtractedObjectPCD(
                points=empty,
                colors=empty,
                policy_points=empty,
                reference_points=empty,
                center=np.zeros(3, dtype=np.float32),
                valid=False,
                message="empty",
            )

    provider = ObjectPCDProvider.__new__(ObjectPCDProvider)
    provider.cfg = {"tracker": {"recovery_enabled": False}}
    provider.online_sam2_cfg = {}
    provider._online_sam2_service_health = None
    provider.camera = Camera()
    provider.tracker = Tracker()
    provider.sam2 = None
    provider.extractor = Extractor()
    provider.frame_counter = 0
    provider.last_packet = _old_packet()
    provider.last_mask_result = None
    provider.last_timings_ms = {}
    provider.reference_frame = "robot_base"
    provider.point_frame = "robot_base"
    provider.extrinsics = type(
        "Extrinsics", (), {"calibration_id": "test", "T_base_camera": np.eye(4)}
    )()

    _, _, _, packet = provider.step()

    assert not packet.valid
    assert packet.timestamp == frame.timestamp
    assert packet.pcd_current is None
    assert packet.pcd_history is None
    assert packet.pcd_reference is None
    assert packet.center is None
    assert packet.velocity is None
    assert packet.bbox_xyxy is None
    assert packet.debug["tracker_search_bbox_xyxy"] == [3, 4, 10, 12]


def test_invalid_history_never_carries_current_object_payload():
    frame = _frame()
    mask = np.zeros(frame.depth_raw.shape, dtype=np.uint8)
    mask[4:12, 6:16] = 1
    bbox = np.asarray([6, 4, 16, 12], dtype=np.int32)
    points = np.ones((4, 3), dtype=np.float32)

    class Camera:
        device_serial = "fake"

        def get_frame(self, *, timeout_ms):
            assert timeout_ms == 1000
            return frame

    class Tracker:
        initialized = True
        manages_identity = True
        state = type("State", (), {"lost_count": 0})()

        def snapshot_state(self):
            return None

        def update(self, _frame):
            return MaskResult(
                mask=mask,
                bbox_xyxy=bbox,
                score=1.0,
                valid=True,
                message="tracked",
            )

    class Extractor:
        def extract(self, _frame, _mask):
            return ExtractedObjectPCD(
                points=points,
                colors=points,
                policy_points=points,
                reference_points=points.copy(),
                center=np.ones(3, dtype=np.float32),
                valid=True,
                message="object",
            )

    class History:
        def update(self, _points, _center, _timestamp):
            return type(
                "RejectedHistory",
                (),
                {
                    "pcd_history": points[None],
                    "center": np.ones(3, dtype=np.float32),
                    "velocity": np.ones(3, dtype=np.float32),
                    "valid": False,
                    "message": "center jump",
                },
            )()

    provider = ObjectPCDProvider.__new__(ObjectPCDProvider)
    provider.cfg = {"tracker": {"recovery_enabled": False}}
    provider.online_sam2_cfg = {}
    provider._online_sam2_service_health = None
    provider.camera = Camera()
    provider.tracker = Tracker()
    provider.sam2 = None
    provider.extractor = Extractor()
    provider.history = History()
    provider.frame_counter = 0
    provider.last_packet = _old_packet()
    provider.last_mask_result = None
    provider.last_timings_ms = {}
    provider.reference_frame = "robot_base"
    provider.point_frame = "robot_base"
    provider.extrinsics = type(
        "Extrinsics", (), {"calibration_id": "test", "T_base_camera": np.eye(4)}
    )()

    _, _, _, packet = provider.step()

    assert not packet.valid
    assert "invalid history" in packet.message
    assert packet.pcd_current is None
    assert packet.pcd_history is None
    assert packet.pcd_reference is None
    assert packet.center is None
    assert packet.velocity is None
    assert packet.bbox_xyxy is None


def test_bbox_recovery_without_segmenter_is_invalid_and_empty():
    provider = ObjectPCDProvider.__new__(ObjectPCDProvider)
    provider.sam2 = None
    provider.last_mask_result = MaskResult(
        mask=np.ones((24, 32), dtype=np.uint8),
        bbox_xyxy=np.asarray([1, 2, 8, 9], dtype=np.int32),
        score=1.0,
        valid=True,
        message="old frame",
    )

    result, elapsed_ms = provider._reinit_from_bbox(
        _frame(), np.asarray([2, 3, 12, 14]), "test"
    )

    assert elapsed_ms == 0.0
    assert not result.valid
    assert result.mask.sum() == 0
    assert "no segmentation backend" in result.message


def test_rejected_async_prompt_does_not_invalidate_a_healthy_tracker():
    frame = _frame()
    prompt_mask = np.ones(frame.depth_raw.shape, dtype=np.uint8)

    class Tracker:
        def __init__(self):
            self.state = {"valid": True, "token": "healthy"}

        def snapshot_state(self):
            return dict(self.state)

        def restore_state(self, value):
            self.state = dict(value)

        def reinitialize_with_mask(self, _frame, _mask):
            self.state = {"valid": False, "token": "mutated"}
            return MaskResult(
                mask=np.zeros(frame.depth_raw.shape, dtype=np.uint8),
                bbox_xyxy=np.asarray([1, 2, 8, 9], dtype=np.int32),
                score=0.0,
                valid=False,
                message="stale semantic mask",
            )

    provider = ObjectPCDProvider.__new__(ObjectPCDProvider)
    provider.tracker = Tracker()

    assert not provider.reinitialize_same_target_from_mask(frame, prompt_mask)
    assert provider.tracker.state == {"valid": True, "token": "healthy"}


def test_new_target_initialization_rolls_back_every_state_after_late_error():
    frame = _frame()
    prompt_mask = np.zeros(frame.depth_raw.shape, dtype=np.uint8)
    prompt_mask[5:15, 8:20] = 1
    prompt_bbox = np.asarray([8, 5, 20, 15], dtype=np.int32)

    class Tracker:
        def __init__(self):
            self.state = {"token": "old-target"}

        def snapshot_state(self):
            return dict(self.state)

        def restore_state(self, value):
            self.state = dict(value)

        def initialize(self, _frame, mask=None, bbox_xyxy=None):
            self.state = {"token": "partially-initialized"}
            return MaskResult(
                mask=mask.copy(),
                bbox_xyxy=bbox_xyxy.copy(),
                score=1.0,
                valid=True,
                message="new target",
            )

    class History:
        def __init__(self):
            self.reset_called = False

        def reset(self):
            self.reset_called = True

    provider = ObjectPCDProvider.__new__(ObjectPCDProvider)
    provider.tracker = Tracker()
    provider.history = History()
    old_mask_result = MaskResult(
        mask=np.ones(frame.depth_raw.shape, dtype=np.uint8),
        bbox_xyxy=np.asarray([1, 2, 8, 9], dtype=np.int32),
        score=0.8,
        valid=True,
        message="old target",
    )
    old_packet = object()
    provider.last_mask_result = old_mask_result
    provider.last_packet = old_packet
    provider.recovery_template = np.full((2, 2), 3, dtype=np.uint8)
    provider.recovery_template_mask = np.full((2, 2), 4, dtype=np.uint8)
    provider.recovery_template_bbox = np.asarray([1, 1, 3, 3], dtype=np.int32)
    provider.identity_hist = np.full((2, 2), 0.25, dtype=np.float32)
    provider.identity_depth_median = 0.7
    provider.identity_center_uv = (4.0, 5.0)
    provider.identity_initial_area = 42.0
    provider._log_mask_result = lambda *args, **kwargs: None
    provider._save_debug_mask = lambda *args, **kwargs: None

    def fail_after_state_reset(*_args, **_kwargs):
        raise RuntimeError("late recovery-template failure")

    provider._update_recovery_template = fail_after_state_reset

    assert not provider.initialize_from_mask(
        frame,
        prompt_mask,
        bbox_xyxy=prompt_bbox,
        source="test_prompt",
    )
    assert provider.tracker.state == {"token": "old-target"}
    assert not provider.history.reset_called
    assert provider.last_mask_result is old_mask_result
    assert provider.last_packet is old_packet
    np.testing.assert_array_equal(
        provider.recovery_template, np.full((2, 2), 3, dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        provider.recovery_template_mask, np.full((2, 2), 4, dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        provider.recovery_template_bbox, np.asarray([1, 1, 3, 3])
    )
    np.testing.assert_array_equal(
        provider.identity_hist, np.full((2, 2), 0.25, dtype=np.float32)
    )
    assert provider.identity_depth_median == 0.7
    assert provider.identity_center_uv == (4.0, 5.0)
    assert provider.identity_initial_area == 42.0
