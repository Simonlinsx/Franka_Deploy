from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading

import numpy as np
import pytest

from sim2real.deployment.bundle import DeployBundle
from robot_control.franka.session import (
    FrankaPoseRing,
    FrankaPoseSample,
)
from sim2real.observation.live_preview import _CameraSample
from robot_control.rh56.actuator import RH56SafetyFeedback
from sim2real.contracts.v94 import V94Contract
from sim2real.runtime.v94_live_observation_owner import (
    D435ObjectCameraOwner,
    PersistentV94ObservationOwner,
    ProductionV94PolicyTickSourceFactory,
    V94LiveObservationError,
)
from sim2real.observation.model import Proprio67Builder
from sim2real.runtime.v94_policy_tick_source import V94RecoverableObservationHold


BUNDLE = Path(__file__).resolve().parents[2] / "data/test_fixtures/sim2real/deploy.zip"


class _Clock:
    def __init__(self, monotonic_s=10.0, realtime_s=100.0):
        self.monotonic_s = float(monotonic_s)
        self.realtime_s = float(realtime_s)

    def monotonic(self):
        return self.monotonic_s

    def realtime(self):
        return self.realtime_s


class _CameraSource:
    def __init__(self, sample, *, pointcloud_temporal_fallback_config=None):
        self.sample = sample
        self.pointcloud_temporal_fallback_config = dict(
            pointcloud_temporal_fallback_config or {}
        )

    def snapshot(self):
        return self.sample

    def wait_for_frame_change(self, frame_id, *, timeout_s):
        return self.sample.frame_id != frame_id


class _HandHistory:
    def __init__(self, samples):
        self.samples = tuple(samples)
        self.calls = 0

    def require_fresh_feedback_history(self, *, minimum_samples, maximum_age_s):
        self.calls += 1
        assert len(self.samples) >= minimum_samples
        assert maximum_age_s > 0.0
        return type(
            "Snapshot",
            (),
            {
                "samples": tuple(
                    type("Timestamped", (), {"feedback": sample})()
                    for sample in self.samples
                ),
                "fresh": True,
            },
        )()


class _Fingertips:
    def positions_base(self, *, angle_act_register_order, T_base_palm):
        assert np.asarray(angle_act_register_order).shape == (6,)
        return np.repeat(np.asarray(T_base_palm)[None, :3, 3], 5, axis=0)


def _feedback(captured, angles):
    return RH56SafetyFeedback(
        captured_monotonic_s=captured,
        positions=(0,) * 6,
        angles=tuple(angles),
        forces_g=(0,) * 6,
        currents_ma=(0,) * 6,
        errors=(0,) * 6,
        statuses=(2,) * 6,
        temperatures_c=(25,) * 6,
    )


def _camera(
    contract,
    *,
    timestamp_s=99.995,
    frame_id=7,
    mask_valid=True,
    mask_extent=4,
):
    height, width = contract.camera_height, contract.camera_width
    color = np.zeros((height, width, 3), dtype=np.uint8)
    stop = 100 + int(mask_extent)
    color[100:stop, 100:stop] = [10, 20, 30]
    depth = np.ones((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)
    mask[100:stop, 100:stop] = True
    return _CameraSample(
        color_bgr=color,
        depth_m=depth,
        mask=mask,
        camera_K=contract.camera_K.copy(),
        distortion=np.zeros(5, dtype=np.float64),
        distortion_model="distortion.inverse_brown_conrady",
        depth_scale_m_per_unit=contract.depth_scale_m_per_unit,
        timestamp_s=timestamp_s,
        frame_id=frame_id,
        mask_valid=mask_valid,
        message="ok",
        provider_timings_ms=np.zeros(6, dtype=np.float64),
        retrieved_at_s=timestamp_s + 0.005,
        timestamp_domain="timestamp_domain.global_time",
        color_depth_timestamp_skew_s=0.0001,
        color_depth_epoch_timestamp_skew_s=0.0001,
        rejected_timestamp_skew_frames=0,
        last_rejected_color_depth_skew_s=0.0,
        published_at_s=timestamp_s + 0.010,
        dropped_queued_framesets=0,
        sensor_frame_number=frame_id,
        mask_source="adaptive",
        online_sam2_status="tracked_async_pending",
        object_mask_area_px=int(np.count_nonzero(mask)) if mask_valid else 0,
        object_mask_bbox_xyxy=np.asarray(
            [100, 100, stop - 1, stop - 1] if mask_valid else [0, 0, 0, 0],
            dtype=np.int32,
        ),
        requested_object_mask_mode="guarded",
        effective_object_mask_mode="guarded_v2",
        effective_mask_publication_mode="guarded_sam2_primary",
        effective_recovery_publication_mode="unified_three_evidence",
    )


def _pose(
    cycle,
    realtime_s,
    monotonic_s,
    q,
    T,
    *,
    status_flags=0,
    shaper_q_d=None,
):
    return FrankaPoseSample(
        cycle=cycle,
        realtime_s=realtime_s,
        monotonic_s=monotonic_s,
        q_rad=np.asarray(q, dtype=np.float64),
        dq_rad_s=np.zeros(7),
        T_base_eef=np.asarray(T, dtype=np.float64),
        status_flags=status_flags,
        shaper_q_d_rad=shaper_q_d,
    )


def _provider(
    *,
    mask_valid=True,
    visualization_sink=None,
    hand_dt_s=0.010,
    maximum_velocity_dt_s=0.10,
    latest_franka_status_flags=0,
    latest_shaper_q_d=None,
    policy_rgbd_resolution="848x480",
    mask_extent=4,
    object_mask_mode="guarded",
    pointcloud_temporal_fallback_config=None,
):
    contract = V94Contract.from_bundle(DeployBundle(BUNDLE))
    clock = _Clock()
    ring = FrankaPoseRing(capacity=8)
    first_T = np.eye(4)
    first_T[0, 3] = 0.40
    latest_T = first_T.copy()
    latest_T[0, 3] += 0.001
    q0 = contract.q_home_rad.astype(np.float64)
    ring.publish(_pose(1, 99.990, 9.990, q0, first_T))
    ring.publish(
        _pose(
            2,
            100.000,
            10.000,
            q0,
            latest_T,
            status_flags=latest_franka_status_flags,
            shaper_q_d=latest_shaper_q_d,
        )
    )
    hand = _HandHistory(
        [
            _feedback(
                10.000 - float(hand_dt_s),
                [1000, 1000, 1000, 1000, 1000, 1000],
            ),
            _feedback(10.000, [1000, 990, 980, 970, 960, 950]),
        ]
    )
    effective_object_mask_mode = (
        "guarded_v2" if object_mask_mode == "guarded" else object_mask_mode
    )
    provider_publication_mode = (
        "semantic_sam2"
        if object_mask_mode == "legacy"
        else (
            "guarded_sam2_primary"
            if object_mask_mode in ("guarded", "guarded_v2")
            else "adaptive_fusion"
        )
    )
    recovery_publication_mode = (
        "semantic_sam2_direct"
        if object_mask_mode == "legacy"
        else (
            "unified_three_evidence"
            if object_mask_mode in ("guarded", "guarded_v2")
            else "legacy_double_confirm"
        )
    )
    camera = _CameraSource(
        replace(
            _camera(contract, mask_valid=mask_valid, mask_extent=mask_extent),
            requested_object_mask_mode=object_mask_mode,
            effective_object_mask_mode=effective_object_mask_mode,
            effective_mask_publication_mode=provider_publication_mode,
            effective_recovery_publication_mode=recovery_publication_mode,
        ),
        pointcloud_temporal_fallback_config=(
            pointcloud_temporal_fallback_config
        ),
    )
    builder = Proprio67Builder(
        q_home_rad=contract.q_home_rad,
        q_hand_close_rad=contract.q_hand_close_rad,
    )
    provider = PersistentV94ObservationOwner(
        contract=contract,
        franka_pose_ring=ring,
        expected_F_T_EE=np.eye(4),
        rh56_feedback_source=hand,
        camera_source=camera,
        fingertip_model=_Fingertips(),
        proprio_builder=builder,
        visualization_sink=visualization_sink,
        policy_rgbd_resolution=policy_rgbd_resolution,
        requested_object_mask_mode=object_mask_mode,
        effective_object_mask_mode=effective_object_mask_mode,
        maximum_velocity_dt_s=maximum_velocity_dt_s,
        monotonic=clock.monotonic,
        realtime=clock.realtime,
    )
    return provider, hand, camera


def test_424x240_policy_projection_uses_decimated_mask_and_scaled_intrinsics():
    provider, _hand, _camera_source = _provider(
        policy_rgbd_resolution="424x240",
        mask_extent=8,
    )

    observation = provider.snapshot(
        sequence=1,
        now_monotonic_s=10.0,
        hard_deadline_monotonic_s=11.0,
    )

    assert provider._pointcloud_projector.width == 424
    assert provider._pointcloud_projector.height == 240
    np.testing.assert_allclose(
        provider._pointcloud_projector.K[:2],
        provider.contract.camera_K[:2] * np.asarray([[0.5], [0.5]]),
    )
    assert observation.pointcloud_status == "fresh"
    assert observation.source_valid_points == 16


def test_ten_hz_hand_feedback_jitter_uses_commissioned_150ms_window():
    provider, _hand, _camera_source = _provider(
        hand_dt_s=0.1113,
        maximum_velocity_dt_s=0.150,
    )
    observation = provider.snapshot(
        sequence=1,
        now_monotonic_s=10.0,
        hard_deadline_monotonic_s=11.0,
    )
    assert observation.pointcloud_status == "fresh"

    stale_window, _hand, _camera_source = _provider(
        hand_dt_s=0.1113,
        maximum_velocity_dt_s=0.100,
    )
    with pytest.raises(
        V94LiveObservationError,
        match="unsafe RH56 finite-difference interval",
    ):
        stale_window.snapshot(
            sequence=1,
            now_monotonic_s=10.0,
            hard_deadline_monotonic_s=11.0,
        )


def test_snapshot_is_repeatable_and_uses_only_owner_histories():
    provider, hand, _camera_source = _provider()
    first = provider.snapshot(
        sequence=1,
        now_monotonic_s=10.0,
        hard_deadline_monotonic_s=11.0,
    )
    second = provider.snapshot(
        sequence=1,
        now_monotonic_s=10.0,
        hard_deadline_monotonic_s=11.0,
    )

    np.testing.assert_array_equal(
        first.pointcloud_xyzrgb_palm, second.pointcloud_xyzrgb_palm
    )
    np.testing.assert_array_equal(first.pointcloud_valid, second.pointcloud_valid)
    np.testing.assert_array_equal(first.proprio_prefix54, second.proprio_prefix54)
    np.testing.assert_array_equal(first.measured_franka_q_rad, second.measured_franka_q_rad)
    assert first.camera_frame_id == second.camera_frame_id == 7
    assert first.pointcloud_status == "fresh"
    assert first.source_valid_points == 16
    assert hand.calls == 2
    # q-q_home is exactly zero and the Franka dq comes directly from the
    # active-handle state rather than another Robot read.
    np.testing.assert_array_equal(first.proprio_prefix54[:14], np.zeros(14))


def test_repeated_exact_camera_frame_reuses_private_rgbd_and_projection_caches():
    provider, _hand, _camera_source = _provider(
        policy_rgbd_resolution="424x240",
        mask_extent=8,
    )
    adaptation_calls = 0
    projection_calls = 0
    original_adapt = provider._policy_rgbd_adapter.adapt
    original_project = provider._pointcloud_projector.project

    def counted_adapt(**kwargs):
        nonlocal adaptation_calls
        adaptation_calls += 1
        return original_adapt(**kwargs)

    def counted_project(**kwargs):
        nonlocal projection_calls
        projection_calls += 1
        return original_project(**kwargs)

    provider._policy_rgbd_adapter.adapt = counted_adapt
    provider._pointcloud_projector.project = counted_project

    first = provider.snapshot(
        sequence=1,
        now_monotonic_s=10.0,
        hard_deadline_monotonic_s=11.0,
    )
    second = provider.snapshot(
        sequence=2,
        now_monotonic_s=10.0,
        hard_deadline_monotonic_s=11.0,
    )

    assert adaptation_calls == 1
    assert projection_calls == 1
    np.testing.assert_array_equal(
        first.pointcloud_xyzrgb_palm, second.pointcloud_xyzrgb_palm
    )
    np.testing.assert_array_equal(first.pointcloud_valid, second.pointcloud_valid)

    adapted_cache = provider._adapted_policy_rgbd_cache
    projected_cache = provider._projected_pointcloud_cache
    assert adapted_cache is not None
    assert projected_cache is not None
    for array in (
        adapted_cache.policy_rgbd.color_bgr,
        adapted_cache.policy_rgbd.object_mask,
        adapted_cache.policy_rgbd.depth_m,
        projected_cache.T_base_palm_at_capture,
        projected_cache.point_frame.xyzrgb_palm,
        projected_cache.point_frame.valid,
    ):
        assert array is not None
        assert not array.flags.writeable

    latency = provider.diagnostics_snapshot["perception_latency"]
    assert latency["sample_count"] == 2
    assert latency["rgbd_adaptation_cache"] == {
        "hits": 1,
        "misses": 1,
        "last_hit": True,
    }
    assert latency["pointcloud_projection_cache"] == {
        "hits": 1,
        "misses": 1,
        "last_hit": True,
    }
    assert latency["last_camera_frame_id"] == 7
    assert latency["last_pointcloud_frame_id"] == 7
    assert "capture_to_retrieval" in latency["last_stage_s"]
    assert "pointcloud_projection_compute" in latency["last_stage_s"]
    assert set(latency["last_provider_stage_ms"]) == {
        "camera",
        "tracker",
        "sam2",
        "sam2_reinit",
        "mask_gate",
        "total",
    }


def test_new_camera_sample_with_reused_numeric_id_cannot_hit_frame_cache():
    provider, _hand, camera_source = _provider(
        policy_rgbd_resolution="424x240",
        mask_extent=8,
    )
    provider.snapshot(
        sequence=1,
        now_monotonic_s=10.0,
        hard_deadline_monotonic_s=11.0,
    )

    # A restarted/malformed producer could reuse an integer id.  Exact camera
    # publication identity, not the numeric id alone, owns both cache entries.
    camera_source.sample = replace(
        camera_source.sample,
        color_bgr=np.full_like(camera_source.sample.color_bgr, 64),
        published_at_s=camera_source.sample.published_at_s + 0.001,
    )
    provider.snapshot(
        sequence=2,
        now_monotonic_s=10.0,
        hard_deadline_monotonic_s=11.0,
    )

    latency = provider.diagnostics_snapshot["perception_latency"]
    assert latency["rgbd_adaptation_cache"]["hits"] == 0
    assert latency["rgbd_adaptation_cache"]["misses"] == 2
    assert latency["pointcloud_projection_cache"]["hits"] == 0
    assert latency["pointcloud_projection_cache"]["misses"] == 2


def test_full_packet_provider_timing_keeps_pcd_stage_name():
    provider, _hand, camera_source = _provider(
        policy_rgbd_resolution="424x240",
        mask_extent=8,
    )
    camera_source.sample = replace(
        camera_source.sample,
        provider_output_mode="full_packet",
        provider_timings_ms=np.arange(6, dtype=np.float64),
    )

    provider.snapshot(
        sequence=1,
        now_monotonic_s=10.0,
        hard_deadline_monotonic_s=11.0,
    )

    stages = provider.diagnostics_snapshot["perception_latency"][
        "last_provider_stage_ms"
    ]
    assert "pcd" in stages
    assert "mask_gate" not in stages
    assert stages["pcd"] == pytest.approx(4.0)


def test_native_resolution_cache_detaches_mutable_camera_buffers():
    provider, _hand, camera_source = _provider(
        policy_rgbd_resolution="848x480",
        mask_extent=8,
    )
    original_color = camera_source.sample.color_bgr.copy()
    original_mask = camera_source.sample.mask.copy()

    provider.snapshot(
        sequence=1,
        now_monotonic_s=10.0,
        hard_deadline_monotonic_s=11.0,
    )
    cached = provider._adapted_policy_rgbd_cache
    assert cached is not None
    assert not np.shares_memory(
        cached.policy_rgbd.color_bgr, camera_source.sample.color_bgr
    )
    assert not np.shares_memory(
        cached.policy_rgbd.object_mask, camera_source.sample.mask
    )

    camera_source.sample.color_bgr[:] = 255
    camera_source.sample.mask[:] = False
    np.testing.assert_array_equal(cached.policy_rgbd.color_bgr, original_color)
    np.testing.assert_array_equal(cached.policy_rgbd.object_mask, original_mask)
    assert not cached.policy_rgbd.color_bgr.flags.writeable
    assert not cached.policy_rgbd.object_mask.flags.writeable


def test_snapshot_forwards_same_sample_shaper_qd_without_reconstruction():
    q_d = np.asarray(
        [0.10, -0.20, 0.30, -1.10, 0.05, 1.40, 0.40], dtype=np.float32
    )
    provider, _hand, _camera_source = _provider(latest_shaper_q_d=q_d)

    observation = provider.snapshot(
        sequence=1,
        now_monotonic_s=10.0,
        hard_deadline_monotonic_s=11.0,
    )

    np.testing.assert_array_equal(observation.shaper_q_d_rad, q_d)
    assert not observation.shaper_q_d_rad.flags.writeable


@pytest.mark.parametrize("status_flags", [1 << 1, 1 << 3, (1 << 1) | (1 << 3)])
def test_native_franka_contact_flags_hold_only_the_arm_target(status_flags):
    provider, _hand, _camera_source = _provider(
        latest_franka_status_flags=status_flags
    )
    observation = provider.snapshot(
        sequence=1,
        now_monotonic_s=10.0,
        hard_deadline_monotonic_s=11.0,
    )
    assert observation.hold_arm_target is True


def test_clear_native_franka_status_does_not_hold_arm_target():
    provider, _hand, _camera_source = _provider(latest_franka_status_flags=0)
    observation = provider.snapshot(
        sequence=1,
        now_monotonic_s=10.0,
        hard_deadline_monotonic_s=11.0,
    )
    assert observation.hold_arm_target is False


def test_franka_state_age_soft_and_hard_staleness_both_hold_without_stage():
    provider, _hand, _camera_source = _provider()
    provider.maximum_rh56_age_s = 0.150
    clock = provider._monotonic.__self__

    clock.monotonic_s = 10.0256
    with pytest.raises(
        V94RecoverableObservationHold,
        match="franka_state_action_age_limit",
    ):
        provider.snapshot(
            sequence=1,
            now_monotonic_s=clock.monotonic_s,
            hard_deadline_monotonic_s=11.0,
        )

    clock.monotonic_s = 10.0501
    with pytest.raises(
        V94RecoverableObservationHold,
        match="franka_state_hard_age_retry",
    ):
        provider.snapshot(
            sequence=1,
            now_monotonic_s=clock.monotonic_s,
            hard_deadline_monotonic_s=11.0,
        )


def test_visualization_receives_same_frame_final_policy_mask_and_points():
    class Sink:
        def __init__(self):
            self.samples = []

        def try_publish(self, sample):
            self.samples.append(sample)
            return True

    sink = Sink()
    provider, _hand, camera_source = _provider(visualization_sink=sink)
    observation = provider.snapshot(
        sequence=3,
        now_monotonic_s=10.0,
        hard_deadline_monotonic_s=11.0,
    )

    assert len(sink.samples) == 1
    sample = sink.samples[0]
    assert sample.sequence == 3
    assert sample.frame_id == observation.camera_frame_id == 7
    assert sample.captured_realtime_s == observation.pointcloud_captured_realtime_s
    np.testing.assert_array_equal(sample.object_mask, camera_source.sample.mask)
    np.testing.assert_array_equal(
        sample.pointcloud_xyzrgb_palm, observation.pointcloud_xyzrgb_palm
    )
    np.testing.assert_array_equal(sample.pointcloud_valid, observation.pointcloud_valid)
    assert sample.source_valid_points == observation.source_valid_points == 16


def test_visualization_failure_cannot_change_policy_observation():
    class FailingSink:
        def try_publish(self, _sample):
            raise RuntimeError("viewer closed")

    provider, _hand, _camera_source = _provider(
        visualization_sink=FailingSink()
    )
    observation = provider.snapshot(
        sequence=1,
        now_monotonic_s=10.0,
        hard_deadline_monotonic_s=11.0,
    )
    assert observation.pointcloud_status == "fresh"
    assert int(np.count_nonzero(observation.pointcloud_valid)) == 16


def test_invalid_mask_without_previous_cloud_is_a_recoverable_hold():
    provider, _hand, _camera_source = _provider(mask_valid=False)
    with pytest.raises(
        V94RecoverableObservationHold,
        match="object_pointcloud_transient_invalid",
    ):
        provider.snapshot(
            sequence=1,
            now_monotonic_s=10.0,
            hard_deadline_monotonic_s=11.0,
        )


def test_thrown_mode_uses_current_semantic_mask_for_bounded_motion_cloud():
    provider, _hand, camera_source = _provider(
        pointcloud_temporal_fallback_config={
            "temporal_fallback": "motion_compensated",
            "temporal_fallback_max_stale_s": 0.25,
            "temporal_fallback_max_stale_steps": 5,
            "temporal_fallback_max_image_speed_px_s": 2400.0,
        }
    )
    measured = provider.snapshot(
        sequence=1,
        now_monotonic_s=10.0,
        hard_deadline_monotonic_s=11.0,
    )
    assert measured.pointcloud_status == "fresh"

    semantic = camera_source.sample.mask.copy()
    camera_source.sample = replace(
        camera_source.sample,
        frame_id=8,
        sensor_frame_number=8,
        timestamp_s=99.996,
        retrieved_at_s=100.001,
        published_at_s=100.006,
        depth_m=np.zeros_like(camera_source.sample.depth_m),
        mask=np.zeros_like(camera_source.sample.mask),
        mask_valid=False,
        policy_semantic_mask=semantic,
        policy_semantic_valid=True,
        policy_semantic_source="online_sam2_exact_current_non_authoritative",
    )
    predicted = provider.snapshot(
        sequence=2,
        now_monotonic_s=10.0,
        hard_deadline_monotonic_s=11.0,
    )
    assert predicted.pointcloud_status == "motion_compensated"
    assert predicted.camera_frame_id == 8
    assert predicted.pointcloud_source_frame_id == 8
    diagnostics = provider.diagnostics_snapshot
    assert diagnostics["fresh_pointcloud_count"] == 1
    assert diagnostics["motion_compensated_pointcloud_count"] == 1


@pytest.mark.parametrize(
    ("mask_valid", "remaining_mask_points"),
    [(False, 0), (True, 1)],
)
def test_invalid_new_mask_reuses_previous_native_palm_cloud_and_metadata(
    mask_valid,
    remaining_mask_points,
):
    class Sink:
        def __init__(self):
            self.samples = []

        def try_publish(self, sample):
            self.samples.append(sample)
            return True

    sink = Sink()
    provider, _hand, camera_source = _provider(
        visualization_sink=sink,
        object_mask_mode="legacy",
    )
    fresh = provider.snapshot(
        sequence=1,
        now_monotonic_s=10.0,
        hard_deadline_monotonic_s=11.0,
    )
    fresh_points = fresh.pointcloud_xyzrgb_palm.copy()
    fresh_valid = fresh.pointcloud_valid.copy()
    assert len(sink.samples) == 1
    fresh_visual = sink.samples[0]

    # Move the arm before the next camera capture.  The fallback must still be
    # byte-for-byte the old capture-time palm cloud; it must not be transformed
    # into this newer palm frame.
    previous_pose = provider.franka_pose_ring.snapshot()[-1]
    moved_T = previous_pose.T_base_eef.copy()
    moved_T[0, 3] += 0.100
    provider.franka_pose_ring.publish(
        _pose(
            3,
            100.001,
            10.001,
            previous_pose.q_rad,
            moved_T,
        )
    )
    current_color = np.full_like(camera_source.sample.color_bgr, 255)
    current_mask = np.zeros_like(camera_source.sample.mask)
    if remaining_mask_points:
        current_mask[50, 50] = True
    camera_source.sample = replace(
        camera_source.sample,
        color_bgr=current_color,
        mask=current_mask,
        timestamp_s=100.001,
        retrieved_at_s=100.006,
        published_at_s=100.011,
        frame_id=8,
        sensor_frame_number=8,
        mask_valid=mask_valid,
    )

    stale = provider.snapshot(
        sequence=2,
        now_monotonic_s=10.0,
        hard_deadline_monotonic_s=11.0,
    )
    assert stale.pointcloud_status == "stale_palm"
    assert stale.source_valid_points == remaining_mask_points
    assert fresh.camera_frame_id == fresh.pointcloud_source_frame_id == 7
    assert stale.camera_frame_id == 8
    assert stale.pointcloud_source_frame_id == 7
    assert (
        stale.pointcloud_captured_realtime_s
        == fresh.pointcloud_captured_realtime_s
        == 99.995
    )
    np.testing.assert_array_equal(stale.pointcloud_xyzrgb_palm, fresh_points)
    np.testing.assert_array_equal(stale.pointcloud_valid, fresh_valid)

    # The lossy viewer must still publish the exact fresh source that produced
    # this retained policy cloud, not the newer empty/too-small mask, color, or
    # palm pose.
    assert len(sink.samples) == 2
    stale_visual = sink.samples[1]
    assert stale_visual.frame_id == fresh_visual.frame_id == 7
    assert stale_visual.captured_realtime_s == fresh_visual.captured_realtime_s
    assert stale_visual.source_valid_points == fresh_visual.source_valid_points == 16
    assert (
        fresh_visual.projector_effective_policy_mask_provenance
        == "current_frame_projector_input_from_provider_mask"
    )
    assert (
        stale_visual.projector_effective_policy_mask_provenance
        == "retained_previous_fresh_projector_mask"
    )
    assert stale_visual.projector_effective_policy_mask_source_frame_id == 7
    assert stale_visual.projector_effective_policy_mask_area_px == 16
    assert stale_visual.projector_effective_policy_mask_bbox_xyxy == (
        100,
        100,
        103,
        103,
    )
    np.testing.assert_array_equal(stale_visual.color_bgr, fresh_visual.color_bgr)
    np.testing.assert_array_equal(stale_visual.object_mask, fresh_visual.object_mask)
    np.testing.assert_array_equal(
        stale_visual.T_base_palm_at_capture,
        fresh_visual.T_base_palm_at_capture,
    )
    assert not np.array_equal(stale_visual.color_bgr, current_color)
    assert not np.array_equal(stale_visual.object_mask, current_mask)


def test_perception_diagnostics_distinguish_advancing_invalid_camera_frames():
    provider, _hand, camera_source = _provider(object_mask_mode="legacy")
    provider.snapshot(
        sequence=1,
        now_monotonic_s=10.0,
        hard_deadline_monotonic_s=11.0,
    )

    empty_mask = np.zeros_like(camera_source.sample.mask)
    for sequence, frame_id in ((2, 8), (3, 9)):
        camera_source.sample = replace(
            camera_source.sample,
            frame_id=frame_id,
            sensor_frame_number=frame_id,
            mask=empty_mask,
            mask_valid=False,
            message="tracker mask invalid during bounded recovery",
        )
        stale = provider.snapshot(
            sequence=sequence,
            now_monotonic_s=10.0,
            hard_deadline_monotonic_s=11.0,
        )
        assert stale.pointcloud_status == "stale_palm"
        assert stale.camera_frame_id == frame_id
        assert stale.pointcloud_source_frame_id == 7

    diagnostics = provider.diagnostics_snapshot
    assert diagnostics["snapshot_attempt_count"] == 3
    assert diagnostics["fresh_pointcloud_count"] == 1
    assert diagnostics["stale_palm_recovery_count"] == 2
    assert diagnostics["last_camera_frame_id"] == 9
    assert diagnostics["last_camera_sensor_frame_number"] == 9
    assert diagnostics["last_camera_mask_valid"] is False
    assert diagnostics["last_camera_mask_area_px"] == 0
    assert diagnostics["configured_requested_object_mask_mode"] == "legacy"
    assert diagnostics["configured_effective_object_mask_mode"] == "legacy"
    assert diagnostics["stale_palm_policy"] == "legacy_explicit_compatibility"
    assert diagnostics["stale_palm_fail_closed_hold_count"] == 0
    assert diagnostics["requested_object_mask_mode"] == "legacy"
    assert diagnostics["effective_object_mask_mode"] == "legacy"
    assert (
        diagnostics["effective_provider_mask_publication_mode"]
        == "semantic_sam2"
    )
    assert diagnostics["provider_published_mask_source"] == "adaptive"
    assert diagnostics["provider_online_sam2_status"] == "tracked_async_pending"
    assert diagnostics["last_pointcloud_status"] == "stale_palm"
    assert diagnostics["last_pointcloud_frame_id"] == 7
    assert diagnostics["projector_effective_policy_mask"] == {
        "provenance": "retained_previous_fresh_projector_mask",
        "source_frame_id": 7,
        "source_captured_realtime_s": 99.995,
        "coordinate_space": "policy_rgbd_pixels",
        "area_px": 16,
        "bbox_xyxy": [100, 100, 103, 103],
    }
    stale_episode = diagnostics["stale_palm_episode"]
    assert stale_episode["active"] is True
    assert stale_episode["first_camera_frame_id"] == 8
    assert stale_episode["last_camera_frame_id"] == 9
    assert stale_episode["camera_frame_changes"] == 1
    assert "latest_camera_frame=9" in provider.diagnostics_summary
    assert "pointcloud_frame=7" in provider.diagnostics_summary


@pytest.mark.parametrize("object_mask_mode", ("guarded", "guarded_v2", "guarded_v1"))
def test_guarded_modes_fail_closed_on_stale_palm_before_policy_stage(
    object_mask_mode,
):
    class Sink:
        def __init__(self):
            self.samples = []

        def try_publish(self, sample):
            self.samples.append(sample)
            return True

    sink = Sink()
    provider, _hand, camera_source = _provider(
        visualization_sink=sink,
        object_mask_mode=object_mask_mode,
    )
    provider.snapshot(
        sequence=1,
        now_monotonic_s=10.0,
        hard_deadline_monotonic_s=11.0,
    )
    camera_source.sample = replace(
        camera_source.sample,
        frame_id=8,
        sensor_frame_number=8,
        mask=np.zeros_like(camera_source.sample.mask),
        mask_valid=False,
        message="guarded tracker recovery pending",
    )

    with pytest.raises(
        V94RecoverableObservationHold,
        match=(
            "guarded_stale_palm_fail_closed:"
            f"requested_mode={object_mask_mode}"
        ),
    ):
        provider.snapshot(
            sequence=2,
            now_monotonic_s=10.0,
            hard_deadline_monotonic_s=11.0,
        )

    # Only the fresh result reaches the lossy side channel. The retained cloud
    # is observable in diagnostics, but cannot become a policy observation.
    assert len(sink.samples) == 1
    diagnostics = provider.diagnostics_snapshot
    assert diagnostics["stale_palm_recovery_count"] == 1
    assert diagnostics["stale_palm_fail_closed_hold_count"] == 1
    assert diagnostics["stale_palm_policy"] == (
        "recoverable_fail_closed_no_policy_stage"
    )
    assert "guarded_stale_palm_fail_closed" in (
        diagnostics["last_recoverable_hold_reason"]
    )


def test_legacy_stale_palm_compatibility_cannot_be_enabled_implicitly():
    contract = V94Contract.from_bundle(DeployBundle(BUNDLE))
    provider, hand, camera_source = _provider()
    with pytest.raises(
        ValueError,
        match="legacy stale-palm compatibility requires an explicit legacy request",
    ):
        PersistentV94ObservationOwner(
            contract=contract,
            franka_pose_ring=provider.franka_pose_ring,
            expected_F_T_EE=np.eye(4),
            rh56_feedback_source=hand,
            camera_source=camera_source,
            requested_object_mask_mode="guarded_v2",
            effective_object_mask_mode="legacy",
        )


def test_pose_capture_skew_fails_closed():
    provider, _hand, camera_source = _provider()
    camera_source.sample = replace(camera_source.sample, timestamp_s=99.0)
    with pytest.raises(V94LiveObservationError, match="pose skew"):
        provider.snapshot(
            sequence=1,
            now_monotonic_s=10.0,
            hard_deadline_monotonic_s=11.0,
        )


def test_d435_owner_factory_is_inert_and_cleanup_is_single_owner():
    contract = V94Contract.from_bundle(DeployBundle(BUNDLE))
    sample = _camera(contract)
    calls = []

    class Provider:
        def stop(self):
            calls.append("provider_stop")

    provider = Provider()

    def factory():
        calls.append("factory")
        return provider

    def worker(_provider, stop, latest, output_mode):
        calls.append(f"worker:{output_mode}")
        latest.put(sample)
        stop.wait(1.0)

    owner = D435ObjectCameraOwner(
        provider_factory=factory,
        worker=worker,
        join_timeout_s=1.0,
    )
    assert calls == []
    owner.open()
    assert owner.snapshot() is sample
    owner.close()
    assert calls == ["factory", "worker:mask_only", "provider_stop"]
    owner.close()


def test_prewarmed_d435_handoff_is_exact_bound_single_use_and_idempotent():
    import sim2real.runtime.v94_live_observation_owner as owner_module

    contract = V94Contract.from_bundle(DeployBundle(BUNDLE))
    sample = _camera(contract, frame_id=8, timestamp_s=100.025)
    calls = []

    class Provider:
        def stop(self):
            calls.append("provider_stop")

    def worker(_provider, stop, latest, output_mode):
        assert output_mode == "mask_only"
        latest.put(sample)
        stop.wait(1.0)

    owner = D435ObjectCameraOwner(
        provider_factory=Provider,
        worker=worker,
        join_timeout_s=1.0,
    )
    owner.open()
    binding = owner_module._PrewarmedD435Binding(
        roi_xywh=(90, 80, 20, 20),
        object_mask_mode="guarded",
        pcd_config_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        preflight_sha256="c" * 64,
        camera_serial=contract.camera_serial,
        calibration_id=contract.calibration_id,
        last_preflight_frame_id=7,
        last_preflight_timestamp_s=100.0,
        adopted_frame_id=8,
        adopted_timestamp_s=100.025,
    )
    handoff = owner_module.PrewarmedD435CameraHandoff(
        _seal=owner_module._PREWARMED_D435_HANDOFF_SEAL,
        owner=owner,
        binding=binding,
    )

    with pytest.raises(
        V94LiveObservationError,
        match="differs from the current ROI/artifacts",
    ):
        handoff.consume(
            contract=contract,
            roi_xywh=binding.roi_xywh,
            object_mask_mode="guarded",
            pcd_config_sha256="d" * 64,
            checkpoint_sha256=binding.checkpoint_sha256,
            preflight_sha256=binding.preflight_sha256,
        )

    assert handoff.consume(
        contract=contract,
        roi_xywh=binding.roi_xywh,
        object_mask_mode="guarded",
        pcd_config_sha256=binding.pcd_config_sha256,
        checkpoint_sha256=binding.checkpoint_sha256,
        preflight_sha256=binding.preflight_sha256,
    ) is owner
    with pytest.raises(V94LiveObservationError, match="already consumed"):
        handoff.consume(
            contract=contract,
            roi_xywh=binding.roi_xywh,
            object_mask_mode="guarded",
            pcd_config_sha256=binding.pcd_config_sha256,
            checkpoint_sha256=binding.checkpoint_sha256,
            preflight_sha256=binding.preflight_sha256,
        )

    handoff.close()
    handoff.close()
    assert calls == ["provider_stop"]


def test_pre_runtime_resync_waits_past_stale_last_value_for_new_frame():
    contract = V94Contract.from_bundle(DeployBundle(BUNDLE))
    clock = _Clock(monotonic_s=10.0)
    stale = replace(
        _camera(contract, frame_id=8, timestamp_s=100.0),
        retrieved_monotonic_s=9.45,
    )
    fresh = replace(
        _camera(contract, frame_id=9, timestamp_s=100.017),
        retrieved_monotonic_s=9.99,
    )
    stale_published = threading.Event()
    allow_fresh = threading.Event()

    class Provider:
        def stop(self):
            pass

    def worker(_provider, stop, latest, _output_mode):
        latest.put(stale)
        stale_published.set()
        allow_fresh.wait(1.0)
        latest.put(fresh)
        stop.wait(1.0)

    owner = D435ObjectCameraOwner(
        provider_factory=Provider,
        worker=worker,
        maximum_publication_stall_s=0.40,
        monotonic=clock.monotonic,
    )
    owner.open()
    assert stale_published.wait(0.2)
    timer = threading.Timer(0.01, allow_fresh.set)
    timer.start()
    try:
        synchronized = owner.wait_for_next_fresh_publication(timeout_s=0.40)
    finally:
        timer.cancel()
        allow_fresh.set()
        owner.close()
    assert synchronized is fresh
    assert synchronized.frame_id == 9


def test_pre_runtime_resync_remains_bounded_when_camera_never_recovers():
    contract = V94Contract.from_bundle(DeployBundle(BUNDLE))
    published = threading.Event()

    class Provider:
        def stop(self):
            pass

    def worker(_provider, stop, latest, _output_mode):
        latest.put(_camera(contract, frame_id=8, timestamp_s=100.0))
        published.set()
        stop.wait(1.0)

    owner = D435ObjectCameraOwner(
        provider_factory=Provider,
        worker=worker,
        maximum_publication_stall_s=0.40,
    )
    owner.open()
    assert published.wait(0.2)
    try:
        with pytest.raises(
            V94LiveObservationError,
            match="bounded pre-runtime resync",
        ):
            owner.wait_for_next_fresh_publication(timeout_s=0.02)
    finally:
        owner.close()


def test_preopened_source_factory_resyncs_without_reopening_camera():
    contract = V94Contract.from_bundle(DeployBundle(BUNDLE))
    sample = _camera(contract, frame_id=9, timestamp_s=100.017)
    calls = []

    class Owner:
        is_open = True
        maximum_publication_stall_s = 0.40

        def open(self):
            raise AssertionError("preopened D435 owner must not be reopened")

        def wait_for_next_fresh_publication(self, *, timeout_s):
            calls.append(timeout_s)
            return sample

    factory = object.__new__(ProductionV94PolicyTickSourceFactory)
    factory._camera_warmed = False
    factory._closed = False
    factory._camera_owner_preopened = True
    factory.camera_owner = Owner()
    factory.contract = contract
    factory.live_visualizer = None
    factory.camera_poll_interval_s = 0.001
    factory._monotonic = lambda: 10.0
    factory._sleep = lambda _seconds: None
    factory._feedback_mapper = object()
    factory._fingertip_model = object()
    factory._proprio_builder = object()

    factory.open_and_warm_camera(
        object(),
        hard_deadline_monotonic_s=11.0,
    )

    assert calls == [pytest.approx(0.40)]
    assert factory._warm_frame_id == 9
    assert factory._camera_warmed is True


def test_d435_owner_reports_provider_stage_before_actuator_watchdog():
    contract = V94Contract.from_bundle(DeployBundle(BUNDLE))
    clock = _Clock(monotonic_s=10.0)
    sample = replace(
        _camera(contract),
        retrieved_monotonic_s=9.799,
        message="adaptive tracker lost object",
        mask_valid=False,
        mask=np.zeros(
            (contract.camera_height, contract.camera_width),
            dtype=bool,
        ),
    )

    class Provider:
        last_pipeline_stage = "camera_get_frame"
        last_pipeline_frame_id = None
        last_pipeline_stage_started_monotonic_s = 9.800

        def stop(self):
            pass

    def worker(_provider, stop, latest, _output_mode):
        latest.put(sample)
        stop.wait(1.0)

    owner = D435ObjectCameraOwner(
        provider_factory=Provider,
        worker=worker,
        maximum_publication_stall_s=0.20,
        monotonic=clock.monotonic,
    )
    owner.open()
    with pytest.raises(
        V94LiveObservationError,
        match=(
            r"D435 formal publication stalled: .*last_frame=7.*"
            r"mask_valid=False.*provider_stage='camera_get_frame'"
        ),
    ):
        owner.snapshot()
    diagnostics = owner.diagnostics_snapshot
    assert diagnostics["last_frame_id"] == 7
    assert diagnostics["publication_age_s"] == pytest.approx(0.201)
    assert "provider_stage='camera_get_frame'" in diagnostics["provider_progress"]
    owner.close()


def test_d435_owner_allows_short_210ms_transport_jitter_but_not_400ms_stall():
    contract = V94Contract.from_bundle(DeployBundle(BUNDLE))
    clock = _Clock(monotonic_s=10.0)
    sample = replace(
        _camera(contract),
        retrieved_monotonic_s=9.790201,
    )

    class Provider:
        def stop(self):
            pass

    def worker(_provider, stop, latest, _output_mode):
        latest.put(sample)
        stop.wait(1.0)

    owner = D435ObjectCameraOwner(
        provider_factory=Provider,
        worker=worker,
        maximum_publication_stall_s=0.40,
        monotonic=clock.monotonic,
    )
    owner.open()
    assert owner.snapshot().frame_id == sample.frame_id
    clock.monotonic_s = 10.191
    with pytest.raises(V94LiveObservationError, match="formal publication stalled"):
        owner.snapshot()
    owner.close()


def test_d435_owner_diagnostics_retain_bounded_capture_retry_evidence():
    contract = V94Contract.from_bundle(DeployBundle(BUNDLE))
    clock = _Clock(monotonic_s=10.0)
    sample = replace(
        _camera(contract),
        retrieved_monotonic_s=9.950,
    )

    class Camera:
        device_serial = "test-serial"
        device_name = "test-d435"
        device_firmware_version = "5.13.0.55"
        device_usb_type_descriptor = "3.2"
        sdk_version = "2.58.2"
        runtime_frame_wait_timeouts = 2
        rejected_transport_stale_frames = 3
        rejected_timestamp_skew_frames = 4
        rejected_non_increasing_frames = 5
        dropped_queued_framesets = 6
        drained_pending_framesets = 7
        last_accepted_sensor_frame_number = 88
        last_accepted_depth_sensor_frame_number = 89
        last_frame_wait_started_monotonic_s = 9.990
        last_frame_wait_failed_monotonic_s = 9.995

    class Provider:
        camera = Camera()
        last_pipeline_stage = "camera_get_frame"
        last_pipeline_frame_id = 7
        last_pipeline_stage_started_monotonic_s = 9.990
        _online_sam2_eligible_count = 10
        _online_sam2_busy_skip_count = 2
        _online_sam2_exact_count = 8

        def stop(self):
            pass

    def worker(_provider, stop, latest, _output_mode):
        latest.put(sample)
        retryable = RuntimeError("SDK UVC capture deadline")
        latest.note_transient(retryable)
        stop.wait(1.0)

    owner = D435ObjectCameraOwner(
        provider_factory=Provider,
        worker=worker,
        maximum_publication_stall_s=0.20,
        monotonic=clock.monotonic,
    )
    owner.open()
    assert owner.snapshot().frame_id == 7
    diagnostics = owner.diagnostics_snapshot
    assert diagnostics["worker"]["transient_error_count"] == 1
    assert diagnostics["worker"]["consecutive_transient_error_count"] == 1
    assert "UVC capture deadline" in diagnostics["worker"]["last_transient_error"]
    assert diagnostics["transport"]["device_serial"] == "test-serial"
    assert diagnostics["transport"]["device_firmware_version"] == "5.13.0.55"
    assert diagnostics["transport"]["sdk_version"] == "2.58.2"
    assert diagnostics["transport"]["runtime_frame_wait_timeouts"] == 2
    assert diagnostics["online_sam2"]["eligible_count"] == 10
    assert diagnostics["online_sam2"]["busy_skip_count"] == 2
    assert diagnostics["online_sam2"]["exact_coverage_fraction"] == pytest.approx(
        0.8
    )
    assert diagnostics["transport"]["last_frame_wait_failed_age_s"] == (
        pytest.approx(0.005)
    )
    owner.close()


def test_d435_owner_test_adapter_fallback_clock_resets_on_new_frame():
    contract = V94Contract.from_bundle(DeployBundle(BUNDLE))
    clock = _Clock(monotonic_s=10.0)
    samples = [_camera(contract, frame_id=7)]

    class Provider:
        def stop(self):
            pass

    def worker(_provider, stop, latest, _output_mode):
        while not stop.is_set():
            latest.put(samples[-1])
            stop.wait(0.001)

    owner = D435ObjectCameraOwner(
        provider_factory=Provider,
        worker=worker,
        maximum_publication_stall_s=0.20,
        monotonic=clock.monotonic,
    )
    owner.open()
    assert owner.snapshot().frame_id == 7
    clock.monotonic_s = 10.199
    assert owner.snapshot().frame_id == 7

    samples.append(_camera(contract, frame_id=8))
    assert owner.wait_for_frame_change(7, timeout_s=0.1)
    clock.monotonic_s = 10.398
    assert owner.snapshot().frame_id == 8
    owner.close()


def test_d435_owner_wraps_background_failure_with_provider_progress():
    class Provider:
        last_pipeline_stage = "tracker_update"
        last_pipeline_frame_id = 12
        last_pipeline_stage_started_monotonic_s = 9.950

        def stop(self):
            pass

    def worker(_provider, _stop, latest, _output_mode):
        latest.fail(RuntimeError("synthetic camera timeout"))

    clock = _Clock(monotonic_s=10.0)
    owner = D435ObjectCameraOwner(
        provider_factory=Provider,
        worker=worker,
        monotonic=clock.monotonic,
    )
    owner.open()
    with pytest.raises(
        V94LiveObservationError,
        match=(
            r"D435 background provider failed; "
            r"provider_stage='tracker_update'.*synthetic camera timeout"
        ),
    ):
        owner.snapshot()
    owner.close()
