from __future__ import annotations

import inspect
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from sim2real.observation.camera_profile import (
    TabletopReleaseHeightGateConfig,
    ThrownRolloutTriggerConfig,
)
from sim2real.runtime.v94_live_observation_owner import (
    ProductionV94PolicyTickSourceFactory,
    _ThrownObjectRolloutTrigger,
)
from sim2real.observation.model import PolicyRGBDResolutionAdapter


def test_production_source_factory_accepts_the_sealed_rollout_trigger() -> None:
    signature = inspect.signature(ProductionV94PolicyTickSourceFactory.__init__)
    assert "rollout_trigger_config" in signature.parameters
    parameter = signature.parameters["rollout_trigger_config"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


def _config() -> ThrownRolloutTriggerConfig:
    return ThrownRolloutTriggerConfig(
        mode="object_motion_or_entry",
        maximum_wait_s=12.0,
        poll_interval_s=0.001,
        minimum_mask_area_px=20,
        absent_arm_frames=3,
        stable_arm_frames=3,
        stable_max_centroid_speed_px_s=80.0,
        trigger_min_centroid_speed_px_s=120.0,
        trigger_min_displacement_px=6.0,
        minimum_area_ratio=0.25,
        maximum_area_ratio=4.0,
    )


def _ramp_config() -> ThrownRolloutTriggerConfig:
    return ThrownRolloutTriggerConfig(
        mode="object_motion_then_tabletop_height",
        maximum_wait_s=20.0,
        poll_interval_s=0.001,
        minimum_mask_area_px=20,
        absent_arm_frames=999,
        stable_arm_frames=3,
        stable_max_centroid_speed_px_s=10.0,
        trigger_min_centroid_speed_px_s=25.0,
        trigger_min_displacement_px=0.3,
        minimum_area_ratio=0.75,
        maximum_area_ratio=1.35,
        allow_absent_entry=False,
        display_name="Object motion trigger",
        ready_instruction="release the object down the ramp now",
        tabletop_release_height_gate=TabletopReleaseHeightGateConfig(
            reference_center_base_z_m=0.052,
            tolerance_m=0.015,
            confirm_frames=3,
            minimum_valid_points=16,
        ),
    )


def _square(x0: int, y0: int = 10) -> np.ndarray:
    mask = np.zeros((48, 80), dtype=np.bool_)
    mask[y0 : y0 + 6, x0 : x0 + 6] = True
    return mask


class _TriggerClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _TriggerCameraOwner:
    is_open = True

    def __init__(self, masks: list[np.ndarray], clock: _TriggerClock) -> None:
        self._masks = masks
        self._index = 0
        self._clock = clock

    def snapshot(self):
        mask = self._masks[self._index]
        ys, xs = np.nonzero(mask)
        bbox = np.asarray(
            [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.int32
        )
        return SimpleNamespace(
            frame_id=self._index + 1,
            timestamp_s=1.0 + 0.05 * self._index,
            mask=mask,
            mask_valid=True,
            policy_semantic_mask=None,
            policy_semantic_valid=False,
            color_bgr=np.zeros((*mask.shape, 3), dtype=np.uint8),
            requested_object_mask_mode="guarded",
            effective_object_mask_mode="guarded_v2",
            effective_mask_publication_mode="adaptive_fusion",
            effective_recovery_publication_mode="unified_three_evidence",
            object_mask_area_px=int(np.count_nonzero(mask)),
            object_mask_bbox_xyxy=bbox,
            mask_source="test",
            message="test",
            online_sam2_status="tracked",
        )

    def wait_for_frame_change(self, _frame_id: int, *, timeout_s: float) -> bool:
        self._clock.now += max(float(timeout_s), 0.05)
        if self._index + 1 < len(self._masks):
            self._index += 1
            return True
        return False


class _TriggerSink:
    def __init__(self) -> None:
        self.samples = []

    def try_publish(self, sample) -> bool:
        self.samples.append(sample)
        return True


def _trigger_factory(masks: list[np.ndarray], config: ThrownRolloutTriggerConfig):
    factory = object.__new__(ProductionV94PolicyTickSourceFactory)
    clock = _TriggerClock()
    sink = _TriggerSink()
    factory.rollout_trigger_config = config
    factory._camera_warmed = True
    factory.camera_owner = _TriggerCameraOwner(masks, clock)
    factory._monotonic = clock
    factory._sleep = lambda seconds: setattr(clock, "now", clock.now + seconds)
    factory.live_visualizer = sink
    factory.point_feature_dim = 3
    factory._trigger_visualization_sequence = 0
    factory._trigger_visualization_points = np.zeros((128, 3), dtype=np.float32)
    factory._trigger_visualization_valid = np.zeros(128, dtype=np.float32)
    factory._trigger_visualization_transform = np.eye(4, dtype=np.float64)
    factory._rollout_trigger_result = None
    return factory, clock, sink


def test_production_trigger_records_camera_only_frames_before_robot_owners() -> None:
    factory, _clock, sink = _trigger_factory(
        [_square(x) for x in (10, 11, 12, 19)], _config()
    )
    result = factory.wait_for_rollout_trigger(
        object(),
        hard_deadline_monotonic_s=10.0,
        stop_requested=threading.Event(),
    )
    assert result is not None and result["detected"] is True
    assert len(sink.samples) == 4
    assert all(sample.source_valid_points == 0 for sample in sink.samples)
    assert np.array_equal(sink.samples[-1].object_mask, _square(19))


def test_production_trigger_timeout_is_explicit_and_keeps_diagnostics() -> None:
    config = _config()
    config = ThrownRolloutTriggerConfig(
        **{**config.__dict__, "maximum_wait_s": 0.20}
    )
    factory, _clock, sink = _trigger_factory([_square(10)] * 8, config)
    with pytest.raises(Exception, match="throw trigger timed out"):
        factory.wait_for_rollout_trigger(
            object(),
            hard_deadline_monotonic_s=10.0,
            stop_requested=threading.Event(),
        )
    diagnostics = factory.rollout_trigger_diagnostics
    assert diagnostics["detected"] is False
    assert diagnostics["timed_out"] is True
    assert diagnostics["timeout_kind"] == "configured_trigger_wait"
    assert diagnostics["armed"] is True
    assert diagnostics["observed_frames"] >= 3
    assert sink.samples


def test_production_split_trigger_detects_only_after_ready_rebaseline() -> None:
    factory, clock, sink = _trigger_factory(
        [_square(x) for x in (10, 11, 12, 13, 20)], _config()
    )
    prepared = factory.prepare_rollout_trigger(
        object(),
        hard_deadline_monotonic_s=10.0,
        stop_requested=threading.Event(),
    )
    assert prepared is not None and prepared["armed"] is True
    assert prepared["detected"] is False
    assert len(sink.samples) == 3

    result = factory.detect_rollout_trigger(
        object(),
        hard_deadline_monotonic_s=10.0,
        stop_requested=threading.Event(),
    )
    assert result is not None and result["detected"] is True
    assert result["frame_id"] == 5
    assert result["active_franka_owner_at_detection"] is True
    assert result["active_rh56_owner_at_detection"] is True
    assert result["armed_ready_to_detection_s"] > 0.0
    assert result["owner_policy_preparation_s"] >= 0.0
    assert len(sink.samples) == 5
    assert clock.now < 10.0


def test_tabletop_ramp_motion_waits_for_three_fresh_flat_height_frames() -> None:
    factory, clock, sink = _trigger_factory(
        [_square(x) for x in (10, 10, 10, 10, 12, 14, 16, 18, 20)],
        _ramp_config(),
    )
    height_by_frame = {
        5: 0.082,
        6: 0.070,
        7: 0.064,
        8: 0.054,
        9: 0.051,
    }
    factory._tabletop_release_center_base_z = (
        lambda *, camera, mask: (
            None
            if camera.frame_id not in height_by_frame
            else (height_by_frame[camera.frame_id], 64)
        )
    )
    prepared = factory.prepare_rollout_trigger(
        object(),
        hard_deadline_monotonic_s=10.0,
        stop_requested=threading.Event(),
    )
    assert prepared is not None and prepared["armed"] is True

    result = factory.detect_rollout_trigger(
        object(),
        hard_deadline_monotonic_s=10.0,
        stop_requested=threading.Event(),
    )
    assert result is not None and result["detected"] is True
    assert result["reason"] == "motion_then_tabletop_height_confirmed"
    assert result["motion_detected_frame_id"] == 5
    assert result["frame_id"] == 9
    assert result["tabletop_height_confirmed_frames"] == 3
    assert result["tabletop_center_base_z_m"] == pytest.approx(0.051)
    assert result["tabletop_height_delta_m"] == pytest.approx(-0.001)
    assert result["tabletop_height_source_points"] == 64
    assert len(sink.samples) == 9
    assert clock.now < 10.0


def test_tabletop_ramp_height_confirmation_is_consecutive_and_fresh() -> None:
    factory, _clock, _sink = _trigger_factory(
        [_square(x) for x in (10, 10, 10, 10, 12, 14, 16, 18, 20, 22, 24)],
        _ramp_config(),
    )
    height_by_frame = {
        5: 0.080,
        6: 0.060,
        7: 0.055,
        8: None,
        9: 0.054,
        10: 0.053,
        11: 0.052,
    }
    factory._tabletop_release_center_base_z = (
        lambda *, camera, mask: (
            None
            if height_by_frame.get(camera.frame_id) is None
            else (height_by_frame[camera.frame_id], 48)
        )
    )
    factory.prepare_rollout_trigger(
        object(),
        hard_deadline_monotonic_s=10.0,
        stop_requested=threading.Event(),
    )
    result = factory.detect_rollout_trigger(
        object(),
        hard_deadline_monotonic_s=10.0,
        stop_requested=threading.Event(),
    )
    assert result is not None
    assert result["frame_id"] == 11
    assert result["tabletop_height_confirmed_frames"] == 3


def test_tabletop_ramp_never_releases_while_object_remains_elevated() -> None:
    config = _ramp_config()
    config = ThrownRolloutTriggerConfig(
        **{**config.__dict__, "maximum_wait_s": 0.50}
    )
    factory, _clock, _sink = _trigger_factory(
        [_square(x) for x in (10, 10, 10, 10, 12, 14, 16, 18, 20, 22)],
        config,
    )
    factory._tabletop_release_center_base_z = (
        lambda *, camera, mask: (0.082, 64)
    )
    factory.prepare_rollout_trigger(
        object(),
        hard_deadline_monotonic_s=10.0,
        stop_requested=threading.Event(),
    )
    with pytest.raises(Exception, match="throw trigger timed out"):
        factory.detect_rollout_trigger(
            object(),
            hard_deadline_monotonic_s=10.0,
            stop_requested=threading.Event(),
        )
    diagnostics = factory.rollout_trigger_diagnostics
    assert diagnostics["detected"] is False
    assert diagnostics["motion_detected_frame_id"] == 5
    assert diagnostics["tabletop_height_confirmed_frames"] == 0
    assert diagnostics["last_tabletop_center_base_z_m"] == pytest.approx(0.082)


def test_tabletop_ramp_height_uses_fresh_calibrated_policy_projection() -> None:
    factory = object.__new__(ProductionV94PolicyTickSourceFactory)
    factory.rollout_trigger_config = _ramp_config()
    camera_K = np.asarray(
        [[100.0, 0.0, 39.5], [0.0, 100.0, 23.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[2, 3] = -0.948
    factory.contract = SimpleNamespace(
        T_base_camera_optical=transform,
        depth_range_m=(0.25, 2.0),
    )
    factory._tabletop_release_rgbd_adapter = PolicyRGBDResolutionAdapter(
        camera_K=camera_K,
        source_image_size=(80, 48),
        target_image_size=(80, 48),
    )
    factory._tabletop_release_projector = None
    factory.fixed_sphere_completion_radius_m = None
    factory.camera_owner = SimpleNamespace(
        support_plane_abcd=None,
        support_plane_min_clearance_m=0.0,
    )
    mask = np.zeros((48, 80), dtype=np.bool_)
    mask[12:32, 20:40] = True
    camera = SimpleNamespace(
        color_bgr=np.zeros((48, 80, 3), dtype=np.uint8),
        depth_m=np.ones((48, 80), dtype=np.float32),
        depth_raw=None,
        depth_scale_m_per_unit=0.001,
        timestamp_s=1.0,
        frame_id=7,
    )
    measured = factory._tabletop_release_center_base_z(
        camera=camera,
        mask=mask,
    )
    assert measured is not None
    center_z, source_points = measured
    assert center_z == pytest.approx(0.052, abs=1.0e-7)
    assert source_points == 400


def test_stable_held_object_arms_then_one_fast_exact_frame_triggers() -> None:
    trigger = _ThrownObjectRolloutTrigger(_config())
    decisions = [
        trigger.observe(
            frame_id=index + 1,
            timestamp_s=1.0 + 0.05 * index,
            mask=_square(x),
            mask_kind="guarded",
        )
        for index, x in enumerate((10, 11, 12, 19))
    ]

    assert [item.armed_now for item in decisions] == [False, False, True, False]
    assert decisions[-1].detected is True
    assert decisions[-1].reason == "stable_to_fast_motion"
    assert decisions[-1].centroid_displacement_px == 7.0
    assert decisions[-1].centroid_speed_px_s == pytest.approx(140.0)


def test_tabletop_trigger_ignores_empty_entry_and_releases_on_low_speed_motion() -> None:
    config = ThrownRolloutTriggerConfig(
        mode="object_motion_only",
        maximum_wait_s=20.0,
        poll_interval_s=0.001,
        minimum_mask_area_px=20,
        absent_arm_frames=999,
        stable_arm_frames=3,
        stable_max_centroid_speed_px_s=10.0,
        trigger_min_centroid_speed_px_s=25.0,
        trigger_min_displacement_px=0.3,
        minimum_area_ratio=0.75,
        maximum_area_ratio=1.35,
        allow_absent_entry=False,
        display_name="Object motion trigger",
        ready_instruction="move the object now",
    )
    trigger = _ThrownObjectRolloutTrigger(config)
    empty = np.zeros((48, 80), dtype=np.bool_)
    for index in range(5):
        decision = trigger.observe(
            frame_id=index + 1,
            timestamp_s=1.0 + 0.05 * index,
            mask=empty,
            mask_kind="none",
        )
        assert decision.armed_now is False
        assert decision.detected is False

    stable = []
    for index in range(3):
        stable.append(
            trigger.observe(
                frame_id=6 + index,
                timestamp_s=1.25 + 0.05 * index,
                mask=_square(10),
                mask_kind="guarded",
            )
        )
    assert stable[-1].armed_now is True
    moving = trigger.observe(
        frame_id=9,
        timestamp_s=1.40,
        mask=_square(12),
        mask_kind="guarded",
    )
    assert moving.detected is True
    assert moving.reason == "stable_to_fast_motion"
    assert moving.centroid_speed_px_s == pytest.approx(40.0)


def test_empty_field_arms_then_first_exact_object_entry_triggers() -> None:
    trigger = _ThrownObjectRolloutTrigger(_config())
    empty = np.zeros((48, 80), dtype=np.bool_)
    for index in range(3):
        decision = trigger.observe(
            frame_id=index + 1,
            timestamp_s=2.0 + 0.05 * index,
            mask=empty,
            mask_kind="none",
        )
    assert decision.armed_now is True

    detected = trigger.observe(
        frame_id=4,
        timestamp_s=2.15,
        mask=_square(30),
        mask_kind="exact_current_semantic_non_authoritative",
    )
    assert detected.detected is True
    assert detected.reason == "absent_to_visible_entry"
    assert detected.mask_area_px == 36


def test_unstable_or_area_inconsistent_masks_never_arm_or_trigger() -> None:
    trigger = _ThrownObjectRolloutTrigger(_config())
    decisions = []
    for index, x in enumerate((10, 20, 10, 20, 10)):
        decisions.append(
            trigger.observe(
                frame_id=index + 1,
                timestamp_s=3.0 + 0.05 * index,
                mask=_square(x),
                mask_kind="guarded",
            )
        )
    assert not any(item.armed_now or item.detected for item in decisions)
