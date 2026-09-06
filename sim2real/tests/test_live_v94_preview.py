from collections import deque
from dataclasses import replace
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from sim2real.observation.live_preview import (
    ACTION_COMPUTE_RESERVE_WARMUP_FACTOR,
    DEFAULT_MAX_CAMERA_DROPOUT_S,
    DEFAULT_RH56_READ_RATE_HZ,
    MAX_POLICY_ACTIONS_PER_CAMERA_FRAME,
    _CameraFrameWatchdog,
    _CameraPolicyReuseGuard,
    _CameraSample,
    _ComputeThreadGuard,
    _DeadlineMetrics,
    _FreshnessFrameGate,
    _HostClockSample,
    _HostClockGuard,
    _Latest,
    _apply_online_sam2_live_override,
    _calibrated_action_compute_reserve_s,
    _camera_rejection_diagnostic,
    _capture_pose_is_ready,
    _record_camera_rejection_diagnostic,
    _camera_worker,
    _deadline_metrics,
    _freshness_budget_metrics,
    _hand_worker,
    _hold_for_unusable_initial_object_observation,
    _hold_for_transient_camera_staleness,
    _immutable_owned_array,
    _nearest_pose,
    _policy_reuse_hold_diagnostic,
    _require_deadline,
    _require_preview_complete,
    _require_runtime_frame_timeout_contract,
    _require_safe_bbox_initialization,
    _require_sensor_age,
    _reset_policy_continuity,
    _run_policy_preview_forward,
    _shutdown_readonly_resources,
    _stack_camera_rejection_diagnostics,
    _stack_policy_reuse_hold_diagnostics,
    _stack_audit_records,
    _start_and_initialize_provider,
    _validate_camera_contract,
    _wait_for_camera_replacement,
    build_parser,
)
from sim2real.v94_kinematics import KinematicVelocityTracker
from sim2real.observation.model import PolicyHistory, PolicyPointFrame


def test_immutable_owned_array_transfers_owned_storage_and_copies_views():
    owned = np.arange(12, dtype=np.uint8).reshape(3, 4).copy()
    published = _immutable_owned_array(owned, dtype=np.uint8)
    assert published is owned
    assert not published.flags.writeable

    source = np.arange(24, dtype=np.float32).reshape(4, 6)
    view = source[:, ::2]
    copied = _immutable_owned_array(view, dtype=np.float32)
    assert copied.flags.owndata
    assert copied.flags.c_contiguous
    assert not np.shares_memory(copied, source)
    assert not copied.flags.writeable
    np.testing.assert_array_equal(copied, view)


def _camera_sample(*, camera_K=None, depth_scale=0.001, distortion=None):
    return _CameraSample(
        color_bgr=np.zeros((2, 3, 3), dtype=np.uint8),
        depth_m=np.ones((2, 3), dtype=np.float32),
        mask=np.ones((2, 3), dtype=bool),
        camera_K=(
            np.asarray(camera_K, dtype=np.float64)
            if camera_K is not None
            else np.asarray([[600.0, 0.0, 1.0], [0.0, 601.0, 0.5], [0.0, 0.0, 1.0]])
        ),
        distortion=(
            np.asarray(distortion, dtype=np.float64)
            if distortion is not None
            else np.zeros(5, dtype=np.float64)
        ),
        distortion_model="distortion.inverse_brown_conrady",
        depth_scale_m_per_unit=depth_scale,
        timestamp_s=1.0,
        frame_id=4,
        mask_valid=True,
        message="ok",
        provider_timings_ms=np.zeros(6, dtype=np.float64),
        retrieved_at_s=1.01,
        timestamp_domain="timestamp_domain.global_time",
        color_depth_timestamp_skew_s=0.0001,
        color_depth_epoch_timestamp_skew_s=0.0002,
        rejected_timestamp_skew_frames=0,
        last_rejected_color_depth_skew_s=0.0,
        published_at_s=1.02,
        dropped_queued_framesets=0,
        sensor_frame_number=10,
    )


def test_background_failure_is_rethrown_even_after_a_value_was_published():
    latest = _Latest()
    latest.put("one-good-sample")
    latest.fail(ValueError("reader died"))
    with pytest.raises(RuntimeError, match="reader died"):
        latest.get()
    with pytest.raises(RuntimeError, match="reader died"):
        latest.raise_if_failed()


def test_latest_retains_retryable_camera_evidence_and_resets_episode_on_publish():
    latest = _Latest()
    latest.note_transient(RuntimeError("USB capture timeout one"))
    latest.note_transient(RuntimeError("USB capture timeout two"))

    during = latest.diagnostics_snapshot()
    assert during["publication_count"] == 0
    assert during["transient_error_count"] == 2
    assert during["consecutive_transient_error_count"] == 2
    assert "USB capture timeout two" in during["last_transient_error"]
    assert during["transient_episode_started_monotonic_s"] is not None

    latest.put("recovered-frame")
    after = latest.diagnostics_snapshot()
    assert after["publication_count"] == 1
    assert after["transient_error_count"] == 2
    assert after["consecutive_transient_error_count"] == 0
    assert after["transient_episode_started_monotonic_s"] is None
    assert latest.get() == "recovered-frame"


def test_latest_frame_wait_does_not_miss_an_already_published_replacement():
    latest = _Latest()
    first = _camera_sample()
    replacement = replace(first, frame_id=first.frame_id + 1)
    latest.put(first)

    # Re-publishing the blocked frame is not a replacement.
    latest.put(replace(first))
    assert not latest.wait_for_frame_change(first.frame_id, timeout_s=0.001)

    # Publication can race ahead of the waiter.  The condition predicate must
    # still observe it immediately rather than sleeping until timeout.
    latest.put(replacement)
    assert latest.wait_for_frame_change(first.frame_id, timeout_s=0.100)


def test_latest_frame_wait_wakes_and_rethrows_background_failure():
    latest = _Latest()
    latest.put(_camera_sample())
    waiter_started = threading.Event()
    failures = []

    def wait_for_camera() -> None:
        waiter_started.set()
        try:
            latest.wait_for_frame_change(4, timeout_s=1.0)
        except BaseException as exc:
            failures.append(exc)

    waiter = threading.Thread(target=wait_for_camera)
    waiter.start()
    assert waiter_started.wait(timeout=1.0)
    latest.fail(OSError("camera disconnected while waiting"))
    waiter.join(timeout=1.0)

    assert not waiter.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    assert "camera disconnected while waiting" in str(failures[0])


def test_startup_capture_waits_until_pose_history_matches_strict_skew():
    history = deque(
        [
            (10.000, np.eye(4)),
            (10.016, np.eye(4)),
            (10.033, np.eye(4)),
        ]
    )
    assert not _capture_pose_is_ready(history, 9.970, 0.025)
    assert _capture_pose_is_ready(history, 9.976, 0.025)
    assert _capture_pose_is_ready(history, 10.057, 0.025)
    assert not _capture_pose_is_ready(history, 10.059, 0.025)
    assert not _capture_pose_is_ready(deque(), 10.0, 0.025)
    with pytest.raises(ValueError, match="finite and positive"):
        _capture_pose_is_ready(history, 10.0, 0.0)
    with pytest.raises(ValueError, match="timestamp"):
        _capture_pose_is_ready(history, float("nan"), 0.025)


def test_camera_replacement_wait_is_bounded_by_two_ticks_and_dropout():
    observed_timeouts = []

    class RecordingLatest:
        def wait_for_frame_change(self, frame_id, *, timeout_s):
            observed_timeouts.append((frame_id, timeout_s))
            return False

    latest = RecordingLatest()
    assert not _wait_for_camera_replacement(
        latest,
        blocked_frame_id=41,
        control_dt_s=1.0 / 60.0,
        maximum_dropout_s=0.500,
    )
    assert observed_timeouts[-1] == pytest.approx((41, 2.0 / 60.0))

    assert not _wait_for_camera_replacement(
        latest,
        blocked_frame_id=42,
        control_dt_s=1.0 / 60.0,
        maximum_dropout_s=0.010,
    )
    assert observed_timeouts[-1] == pytest.approx((42, 0.010))


def test_preview_completion_rejects_zero_steps_and_late_worker_failure():
    camera = _Latest()
    hand = _Latest()
    with pytest.raises(RuntimeError, match="zero valid policy steps"):
        _require_preview_complete(camera, hand, [])

    camera.fail(OSError("camera disconnected"))
    with pytest.raises(RuntimeError, match="camera disconnected"):
        _require_preview_complete(camera, hand, [{"step": np.asarray(0)}])


def test_deadline_metrics_and_fail_closed_limit():
    metrics = _deadline_metrics(
        scheduled_tick_monotonic_s=10.0,
        tick_started_monotonic_s=10.005,
        tick_finished_monotonic_s=10.026,
        control_dt_s=0.016,
    )
    assert metrics.start_lateness_s == pytest.approx(0.005)
    assert metrics.computation_s == pytest.approx(0.021)
    assert metrics.next_deadline_overrun_s == pytest.approx(0.010)
    _require_deadline(metrics, maximum_overrun_s=0.011)
    with pytest.raises(RuntimeError, match="deadline overrun"):
        _require_deadline(metrics, maximum_overrun_s=0.009)


def test_host_clock_guard_records_stable_dual_clock_diagnostics():
    guard = _HostClockGuard(
        maximum_offset_jump_s=0.010,
        start_realtime_s=100.0,
        start_monotonic_s=10.0,
    )
    sample = guard.sample(realtime_s=100.105, monotonic_s=10.100)
    assert sample.realtime_s == pytest.approx(100.105)
    assert sample.monotonic_s == pytest.approx(10.100)
    assert sample.pair_span_s == 0.0
    assert sample.realtime_minus_monotonic_s == pytest.approx(90.005)
    assert sample.offset_delta_from_start_s == pytest.approx(0.005)
    assert sample.offset_delta_from_previous_s == pytest.approx(0.005)


def test_host_clock_guard_fails_on_previous_or_start_offset_jump():
    previous_guard = _HostClockGuard(
        maximum_offset_jump_s=0.010,
        start_realtime_s=100.0,
        start_monotonic_s=10.0,
    )
    previous_guard.sample(realtime_s=100.094, monotonic_s=10.100)
    with pytest.raises(RuntimeError, match="host realtime discontinuity"):
        previous_guard.sample(realtime_s=100.206, monotonic_s=10.200)

    start_guard = _HostClockGuard(
        maximum_offset_jump_s=0.010,
        start_realtime_s=100.0,
        start_monotonic_s=10.0,
    )
    start_guard.sample(realtime_s=100.106, monotonic_s=10.100)
    with pytest.raises(RuntimeError, match="host realtime discontinuity"):
        start_guard.sample(realtime_s=100.211, monotonic_s=10.200)


def test_host_clock_pair_retries_preemption_and_uses_conservative_time(monkeypatch):
    monotonic_values = iter([10.0, 10.020, 20.0, 20.001])
    realtime_values = iter([100.0, 200.0])
    monkeypatch.setattr(
        "sim2real.observation.live_preview.time.monotonic",
        lambda: next(monotonic_values),
    )
    monkeypatch.setattr(
        "sim2real.observation.live_preview.time.time", lambda: next(realtime_values)
    )

    realtime_s, monotonic_s, pair_span_s = _HostClockGuard._read_pair()

    assert pair_span_s == pytest.approx(0.001)
    assert monotonic_s == pytest.approx(20.001)
    assert realtime_s == pytest.approx(200.001)


def test_first_policy_forward_is_warmup_only_and_next_output_is_accepted():
    outputs = iter(["cold-output", "formal-output"])
    policy = SimpleNamespace(act=lambda *_args: next(outputs))
    inputs = np.zeros(1, dtype=np.float32)
    cold, cold_elapsed = _run_policy_preview_forward(
        policy, inputs, inputs, inputs, accept_output=False
    )
    formal, formal_elapsed = _run_policy_preview_forward(
        policy, inputs, inputs, inputs, accept_output=True
    )
    assert cold is None
    assert formal == "formal-output"
    assert cold_elapsed >= 0.0
    assert formal_elapsed >= 0.0


def test_calibrated_compute_reserve_is_conservative_and_uncapped():
    assert _calibrated_action_compute_reserve_s(
        minimum_reserve_s=0.020,
        warmup_observation_action_s=0.003,
    ) == pytest.approx(0.020)
    assert _calibrated_action_compute_reserve_s(
        minimum_reserve_s=0.020,
        warmup_observation_action_s=0.030,
    ) == pytest.approx(0.030 * ACTION_COMPUTE_RESERVE_WARMUP_FACTOR)
    with pytest.raises(ValueError, match="invalid"):
        _calibrated_action_compute_reserve_s(
            minimum_reserve_s=0.0,
            warmup_observation_action_s=0.003,
        )


def test_freshness_budget_rejects_near_expiry_without_widening_age_limit():
    accepted = _freshness_budget_metrics(
        pointcloud_age_s=0.079,
        maximum_age_s=0.100,
        required_reserve_s=0.020,
    )
    assert accepted.eligible
    assert accepted.remaining_s == pytest.approx(0.021)

    rejected = _freshness_budget_metrics(
        pointcloud_age_s=0.081,
        maximum_age_s=0.100,
        required_reserve_s=0.020,
    )
    assert not rejected.eligible
    assert rejected.remaining_s == pytest.approx(0.019)
    assert rejected.required_reserve_s == pytest.approx(0.020)


def test_budget_rejected_frame_cannot_be_reused_as_a_logical_tick():
    gate = _FreshnessFrameGate()
    gate.reject(41)
    assert gate.rejected_frames == 1
    assert gate.waiting_for_newer_frame(41)
    assert gate.waiting_for_newer_frame(41)
    assert gate.wait_ticks == 2

    # Only a changed provider frame id clears the hold.  The caller resets its
    # periodic schedule during each True result, so no rejected tick is caught
    # up or appended to PolicyHistory.
    assert not gate.waiting_for_newer_frame(42)
    assert gate.blocked_frame_id is None


def test_short_rejection_gap_preserves_normal_30_to_60_hz_continuity():
    gate = _FreshnessFrameGate()
    tracker = KinematicVelocityTracker(maximum_dt_s=0.100)
    history = PolicyHistory()
    identity = np.eye(4)
    hand = np.zeros(6)
    tracker.update(
        captured_at_s=10.0,
        T_base_palm=identity,
        hand_q_policy_order_rad=hand,
    )
    first_point = PolicyPointFrame(
        xyzrgb_palm=np.zeros((128, 6), dtype=np.float32),
        valid=np.ones(128, dtype=np.float32),
        captured_at_s=9.98,
        frame_id=40,
        source_valid_points=128,
        status="fresh",
    )
    history.append(first_point, np.zeros(67, dtype=np.float32))
    gate.reject(41)
    assert gate.waiting_for_newer_frame(41)
    assert not gate.waiting_for_newer_frame(42)

    # One missing 30 Hz frame does not erase the four logical 60 Hz history or
    # the finite-difference baseline when the accepted-state gap remains safe.
    assert not gate.resolve_replacement(
        42,
        policy_state_gap_s=0.050,
        maximum_continuity_gap_s=0.100,
    )
    assert gate.rebootstrap_count == 0
    assert not gate.rejection_episode_active
    assert not gate.replacement_ready
    velocity = tracker.update(
        captured_at_s=10.050,
        T_base_palm=identity,
        hand_q_policy_order_rad=hand,
    )
    assert velocity.dt_s == pytest.approx(0.050)
    second_point = replace(
        first_point,
        xyzrgb_palm=np.ones((128, 6), dtype=np.float32),
        frame_id=42,
    )
    point_history, _, _ = history.append(
        second_point, np.ones(67, dtype=np.float32)
    )
    # Preserved bootstrap history shifts by one formal 60 Hz observation.
    np.testing.assert_array_equal(point_history[:3], 0.0)
    np.testing.assert_array_equal(point_history[3], 1.0)
    # The replacement can be reused on the intervening 60 Hz tick.
    assert not gate.waiting_for_newer_frame(42)


def test_long_or_contaminated_rejection_gap_requires_one_rebootstrap():
    long_gap = _FreshnessFrameGate()
    tracker = KinematicVelocityTracker(maximum_dt_s=0.100)
    history = PolicyHistory()
    tracker.update(
        captured_at_s=10.0,
        T_base_palm=np.eye(4),
        hand_q_policy_order_rad=np.zeros(6),
    )
    first_point = PolicyPointFrame(
        xyzrgb_palm=np.zeros((128, 6), dtype=np.float32),
        valid=np.ones(128, dtype=np.float32),
        captured_at_s=9.98,
        frame_id=7,
        source_valid_points=128,
        status="fresh",
    )
    history.append(first_point, np.zeros(67, dtype=np.float32))
    long_gap.reject(7)
    assert not long_gap.waiting_for_newer_frame(8)
    assert long_gap.resolve_replacement(
        8,
        policy_state_gap_s=1.5692768,
        maximum_continuity_gap_s=0.100,
    )
    _reset_policy_continuity(history=history, velocity_tracker=tracker)
    recovered_velocity = tracker.update(
        captured_at_s=11.5692768,
        T_base_palm=np.eye(4),
        hand_q_policy_order_rad=np.zeros(6),
    )
    assert recovered_velocity.dt_s is None
    recovered_point = replace(
        first_point,
        xyzrgb_palm=np.ones((128, 6), dtype=np.float32),
        frame_id=8,
    )
    recovered_history, _, _ = history.append(
        recovered_point, np.ones(67, dtype=np.float32)
    )
    np.testing.assert_array_equal(recovered_history, 1.0)
    assert long_gap.rebootstrap_count == 1
    assert not long_gap.resolve_replacement(
        8,
        policy_state_gap_s=0.0,
        maximum_continuity_gap_s=0.100,
    )

    contaminated = _FreshnessFrameGate()
    contaminated.reject(10, force_rebootstrap=True)
    assert not contaminated.waiting_for_newer_frame(11)
    assert contaminated.resolve_replacement(
        11,
        policy_state_gap_s=0.033,
        maximum_continuity_gap_s=0.100,
    )


def test_rejection_diagnostics_distinguish_same_frame_from_new_stale_stream():
    first_camera = replace(
        _camera_sample(),
        frame_id=41,
        sensor_frame_number=141,
        timestamp_s=100.000,
        retrieved_at_s=100.040,
        published_at_s=100.095,
        mask_valid=False,
        message="tracker LOST / recovery pending",
        object_mask_area_px=0,
        object_mask_bbox_xyxy=np.zeros(4, dtype=np.int32),
    )
    first = _camera_rejection_diagnostic(
        first_camera,
        observed_at_s=100.105,
        reason="capture_age_rejected",
        blocked_frame_id_before=None,
        previous_diagnostic_frame_id=None,
        required_action_compute_reserve_s=0.012,
        freshness_budget_remaining_s=-0.005,
        observation_action_compute_s=0.008,
    )
    same = _camera_rejection_diagnostic(
        first_camera,
        observed_at_s=100.121,
        reason="same_frame_capture_age_wait",
        blocked_frame_id_before=41,
        previous_diagnostic_frame_id=41,
    )
    next_camera = replace(
        first_camera,
        frame_id=42,
        sensor_frame_number=142,
        timestamp_s=100.033,
        retrieved_at_s=100.075,
        published_at_s=100.136,
    )
    changed_but_stale = _camera_rejection_diagnostic(
        next_camera,
        observed_at_s=100.141,
        reason="capture_age_rejected",
        blocked_frame_id_before=41,
        previous_diagnostic_frame_id=41,
    )

    assert not first["same_frame_as_blocked"]
    assert same["same_frame_as_blocked"]
    assert not same["new_frame_since_previous_diagnostic"]
    assert changed_but_stale["new_frame_since_previous_diagnostic"]
    assert changed_but_stale["capture_to_publication_s"] == pytest.approx(0.103)
    assert not changed_but_stale["camera_mask_valid"]
    assert "LOST" in changed_but_stale["camera_message"]

    payload = _stack_camera_rejection_diagnostics(
        [first, same, changed_but_stale]
    )
    assert int(payload["camera_rejection_diagnostic_schema_version"]) == 2
    assert int(payload["camera_rejection_diagnostic_count"]) == 3
    assert payload["camera_rejection_required_action_compute_reserve_s"][0] == (
        pytest.approx(0.012)
    )
    assert payload["camera_rejection_freshness_budget_remaining_s"][0] == (
        pytest.approx(-0.005)
    )
    assert payload["camera_rejection_observation_action_compute_s"][0] == (
        pytest.approx(0.008)
    )
    assert np.isnan(
        payload["camera_rejection_observation_action_compute_s"][1]
    )
    assert payload["camera_rejection_camera_message"].dtype != object
    np.testing.assert_array_equal(
        payload["camera_rejection_same_frame_as_blocked"],
        [False, True, False],
    )


def test_camera_policy_reuse_guard_allows_two_actions_then_requires_hold():
    guard = _CameraPolicyReuseGuard()

    assert guard.accept(41) == 1
    assert not guard.must_hold(41)
    assert guard.accept(41) == 2
    assert guard.must_hold(41)
    # A side-effect-free query cannot turn the held tick into action evidence.
    assert guard.accepted_action_count == 2
    assert guard.record_hold(41) == MAX_POLICY_ACTIONS_PER_CAMERA_FRAME
    assert guard.hold_count == 1
    assert guard.accepted_action_count == 2
    with pytest.raises(RuntimeError, match="exceed the per-camera-frame"):
        guard.accept(41)

    assert guard.accept(42) == 1
    assert guard.accepted_action_count == 3
    with pytest.raises(RuntimeError, match="regressed"):
        guard.must_hold(41)


def test_policy_reuse_hold_is_separate_no_action_no_state_mutation_evidence():
    camera = replace(
        _camera_sample(),
        frame_id=41,
        timestamp_s=100.000,
        retrieved_at_s=100.010,
        published_at_s=100.020,
    )
    observation_clock = _HostClockSample(
        realtime_s=100.030,
        monotonic_s=10.000,
        pair_span_s=0.0001,
        realtime_minus_monotonic_s=90.030,
        offset_delta_from_start_s=0.0,
        offset_delta_from_previous_s=0.0,
    )
    hold_clock = replace(
        observation_clock,
        realtime_s=100.040,
        monotonic_s=10.010,
        realtime_minus_monotonic_s=90.030,
    )
    timing = _DeadlineMetrics(
        scheduled_tick_monotonic_s=10.000,
        tick_started_monotonic_s=10.001,
        tick_finished_monotonic_s=10.011,
        start_lateness_s=0.001,
        computation_s=0.010,
        next_deadline_overrun_s=0.0,
    )
    record = _policy_reuse_hold_diagnostic(
        camera,
        hold_index=0,
        pointcloud_frame_id=41,
        pointcloud_capture_timestamp_s=100.000,
        pointcloud_retrieved_at_s=100.010,
        pointcloud_published_at_s=100.020,
        observation_clock=observation_clock,
        hold_clock=hold_clock,
        timing=timing,
        previous_action_tick_started_monotonic_s=9.984,
        camera_is_new=False,
        action_uses_before_hold=2,
        maximum_policy_actions_per_camera_frame=2,
    )

    assert record["reason"] == "camera_frame_policy_reuse_limit"
    assert record["pointcloud_frame_id"] == 41
    assert record["action_uses_before_hold"] == 2
    assert record["new_policy_action_evidence"] is False
    assert record["policy_state_mutated"] is False
    assert record["write"] is False
    assert record["waits_for_new_camera_frame"] is True

    payload = _stack_policy_reuse_hold_diagnostics([record])
    assert int(payload["policy_hold_schema_version"]) == 1
    assert int(payload["policy_hold_count"]) == 1
    np.testing.assert_array_equal(
        payload["policy_hold_new_policy_action_evidence"], [False]
    )
    np.testing.assert_array_equal(
        payload["policy_hold_policy_state_mutated"], [False]
    )
    assert all(value.dtype != object for value in payload.values())

    empty = _stack_policy_reuse_hold_diagnostics([])
    assert int(empty["policy_hold_schema_version"]) == 1
    assert int(empty["policy_hold_count"]) == 0
    with pytest.raises(ValueError, match="inconsistent fields"):
        _stack_policy_reuse_hold_diagnostics([record, {"reason": "bad"}])


def test_rejection_diagnostics_are_complete_but_quiet_suppresses_stderr(capsys):
    camera = replace(
        _camera_sample(),
        frame_id=41,
        timestamp_s=100.0,
        retrieved_at_s=100.04,
        published_at_s=100.09,
    )
    records = []
    for index in range(5):
        _record_camera_rejection_diagnostic(
            records,
            camera,
            observed_at_s=100.101 + index * 0.016,
            reason="same_frame_capture_age_wait",
            blocked_frame_id_before=41,
            quiet=True,
            print_every_n=2,
        )
    assert capsys.readouterr().err == ""
    assert len(records) == 5
    payload = _stack_camera_rejection_diagnostics(records)
    assert int(payload["camera_rejection_diagnostic_count"]) == 5


def test_nonquiet_rejection_diagnostics_print_first_changes_and_heartbeat(capsys):
    camera = replace(
        _camera_sample(),
        frame_id=41,
        timestamp_s=100.0,
        retrieved_at_s=100.04,
        published_at_s=100.09,
    )
    records = []
    calls = [
        (camera, "capture_age_rejected", None),
        (camera, "same_frame_capture_age_wait", 41),
        (camera, "same_frame_capture_age_wait", 41),
        (camera, "same_frame_capture_age_wait", 41),
        (replace(camera, frame_id=42), "capture_age_rejected", 41),
    ]
    for index, (sample, reason, blocked) in enumerate(calls):
        _record_camera_rejection_diagnostic(
            records,
            sample,
            observed_at_s=100.101 + index * 0.016,
            reason=reason,
            blocked_frame_id_before=blocked,
            quiet=False,
            print_every_n=3,
        )
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 4
    assert '"diagnostic_index": 0' in lines[0]
    assert '"diagnostic_index": 1' in lines[1]
    assert '"diagnostic_index": 3' in lines[2]
    assert '"diagnostic_index": 4' in lines[3]
    assert len(records) == 5


def test_transient_camera_stale_frame_holds_and_new_frame_releases():
    gate = _FreshnessFrameGate()
    watchdog = _CameraFrameWatchdog()

    # This is the measured failure shape: 0.624 ms over the strict limit.  It
    # is held rather than treated as a process failure, so a caller has no
    # reason to invoke projector/history/policy for this tick.
    assert _hold_for_transient_camera_staleness(
        gate=gate,
        watchdog=watchdog,
        frame_id=41,
        age_s=0.100624,
        monotonic_s=10.000,
        maximum_age_s=0.100,
        maximum_future_skew_s=0.010,
        maximum_dropout_s=0.500,
    )
    assert gate.blocked_frame_id == 41
    assert gate.transient_stale_rejected_frames == 1
    assert gate.budget_rejected_frames == 0

    # Polling the exact rejected frame remains a hold and does not count a
    # second rejection.
    assert _hold_for_transient_camera_staleness(
        gate=gate,
        watchdog=watchdog,
        frame_id=41,
        age_s=0.116,
        monotonic_s=10.016,
        maximum_age_s=0.100,
        maximum_future_skew_s=0.010,
        maximum_dropout_s=0.500,
    )
    assert gate.transient_stale_rejected_frames == 1

    # A new, fresh 30 Hz frame releases the hold.  Reusing that same fresh
    # frame on the intervening 60 Hz tick remains valid.
    assert not _hold_for_transient_camera_staleness(
        gate=gate,
        watchdog=watchdog,
        frame_id=42,
        age_s=0.010,
        monotonic_s=10.033,
        maximum_age_s=0.100,
        maximum_future_skew_s=0.010,
        maximum_dropout_s=0.500,
    )
    assert gate.blocked_frame_id is None
    assert not _hold_for_transient_camera_staleness(
        gate=gate,
        watchdog=watchdog,
        frame_id=42,
        age_s=0.026,
        monotonic_s=10.049,
        maximum_age_s=0.100,
        maximum_future_skew_s=0.010,
        maximum_dropout_s=0.500,
    )


def test_camera_future_timestamp_and_hard_dropout_remain_fatal():
    gate = _FreshnessFrameGate()
    watchdog = _CameraFrameWatchdog()
    with pytest.raises(RuntimeError, match="future"):
        _hold_for_transient_camera_staleness(
            gate=gate,
            watchdog=watchdog,
            frame_id=7,
            age_s=-0.011,
            monotonic_s=20.0,
            maximum_age_s=0.100,
            maximum_future_skew_s=0.010,
            maximum_dropout_s=0.500,
        )
    # Future timestamps fail before they can alter watchdog/gate state.
    assert watchdog.last_frame_id is None
    assert gate.blocked_frame_id is None

    assert not _hold_for_transient_camera_staleness(
        gate=gate,
        watchdog=watchdog,
        frame_id=7,
        age_s=0.010,
        monotonic_s=20.0,
        maximum_age_s=0.100,
        maximum_future_skew_s=0.010,
        maximum_dropout_s=0.500,
    )
    with pytest.raises(RuntimeError, match="hard dropout"):
        _hold_for_transient_camera_staleness(
            gate=gate,
            watchdog=watchdog,
            frame_id=7,
            age_s=0.511,
            monotonic_s=20.501,
            maximum_age_s=0.100,
            maximum_future_skew_s=0.010,
            maximum_dropout_s=0.500,
        )


def test_unusable_initial_object_observation_waits_for_different_frame():
    gate = _FreshnessFrameGate()

    assert _hold_for_unusable_initial_object_observation(
        gate=gate,
        frame_id=17,
        usable=False,
    )
    assert gate.blocked_frame_id == 17
    assert gate.unusable_initial_observation_rejected_frames == 1
    assert gate.forced_rebootstrap_pending

    # The ordinary frame gate owns repeated observations and releases only
    # after the producer publishes a different frame.
    assert gate.waiting_for_newer_frame(17)
    assert not gate.waiting_for_newer_frame(18)
    assert gate.replacement_ready
    assert not _hold_for_unusable_initial_object_observation(
        gate=gate,
        frame_id=18,
        usable=True,
    )
    assert gate.unusable_initial_observation_rejected_frames == 1


def test_cli_defaults_keep_strict_camera_age_and_explicit_compute_limits():
    args = build_parser().parse_args([])
    assert args.max_camera_age_s == pytest.approx(0.100)
    assert args.max_camera_dropout_s == pytest.approx(DEFAULT_MAX_CAMERA_DROPOUT_S)
    assert args.min_action_compute_reserve_s == pytest.approx(0.012)
    assert args.compute_threads == 1
    assert args.rh56_read_rate_hz == pytest.approx(DEFAULT_RH56_READ_RATE_HZ)
    assert args.disable_online_sam2 is False
    assert args.provider_output_mode == "mask_only"

    ab_args = build_parser().parse_args(["--disable-online-sam2"])
    assert ab_args.disable_online_sam2 is True


def test_hand_worker_rate_limits_compact_reads_without_changing_freshness(
    monkeypatch,
):
    class Reader:
        def __init__(self):
            self.started = 0
            self.closed = 0
            self.reads = 0

        def start(self):
            self.started += 1

        def read(self):
            self.reads += 1
            zeros = np.zeros(6, dtype=np.int16)
            return SimpleNamespace(
                angle_targets=np.full(6, -1, dtype=np.int16),
                angles=zeros.copy(),
                positions=zeros.copy(),
                forces=zeros.copy(),
                currents=zeros.copy(),
                errors=np.zeros(6, dtype=np.uint8),
                statuses=np.zeros(6, dtype=np.uint8),
                temperatures_c=np.full(6, 25, dtype=np.uint8),
                captured_at_s=123.0,
            )

        def close(self):
            self.closed += 1

    class StopAfterFirstRateWait:
        def __init__(self):
            self.waits = []

        def is_set(self):
            return False

        def wait(self, timeout):
            self.waits.append(float(timeout))
            return True

        def set(self):
            raise AssertionError("successful read must not set the stop event")

    monotonic_values = iter([10.0, 10.010])
    monkeypatch.setattr(
        "sim2real.observation.live_preview.time.monotonic",
        lambda: next(monotonic_values),
    )
    reader = Reader()
    stop = StopAfterFirstRateWait()
    latest = _Latest()
    mapper = SimpleNamespace(
        map=lambda _angles: SimpleNamespace(
            q_policy_order_rad=np.zeros(6, dtype=np.float32)
        )
    )

    _hand_worker(reader, mapper, stop, latest, read_rate_hz=30.0)

    assert reader.started == 1
    assert reader.reads == 1
    assert reader.closed == 1
    assert stop.waits == pytest.approx([1.0 / 30.0 - 0.010])
    sample = latest.get()
    assert sample.timestamp_s == pytest.approx(123.0)
    np.testing.assert_array_equal(sample.angle_targets, -1)
    np.testing.assert_array_equal(sample.dq_policy_rad_s, 0.0)


def test_hand_worker_rejects_invalid_rate_before_opening_reader():
    class Reader:
        def start(self):
            raise AssertionError("invalid configuration must not open serial")

    stop = threading.Event()
    latest = _Latest()
    _hand_worker(
        Reader(),
        SimpleNamespace(),
        stop,
        latest,
        read_rate_hz=float("nan"),
    )
    assert stop.is_set()
    with pytest.raises(RuntimeError, match="finite and positive"):
        latest.raise_if_failed()


def test_online_sam2_live_override_is_disable_only_and_auditable():
    enabled_config = {
        "online_sam2": {"enabled": True, "service_addr": "test-address"},
        "sam2": {"enabled": True},
        "tracker": {"mode": "adaptive_color_depth"},
    }
    unchanged = _apply_online_sam2_live_override(
        enabled_config, disable_online_sam2=False
    )
    assert unchanged.source_config_enabled is True
    assert unchanged.disable_override_requested is False
    assert unchanged.effective_config_enabled is True
    assert unchanged.mode == "enabled_by_config"
    assert enabled_config["online_sam2"]["enabled"] is True
    assert unchanged.requested_object_mask_mode is None
    assert unchanged.source_mask_publication_mode == "adaptive_fusion"
    assert unchanged.effective_mask_publication_mode == "adaptive_fusion"

    guarded_config = {
        "online_sam2": {
            "enabled": True,
            "mask_publication_mode": "semantic_sam2",
        }
    }
    guarded = _apply_online_sam2_live_override(
        guarded_config,
        disable_online_sam2=False,
        object_mask_mode="guarded",
    )
    assert guarded.requested_object_mask_mode == "guarded"
    assert guarded.source_mask_publication_mode == "semantic_sam2"
    assert guarded.effective_mask_publication_mode == "guarded_sam2_primary"
    assert guarded.effective_recovery_publication_mode == "unified_three_evidence"
    assert guarded.effective_object_mask_mode == "guarded_v2"
    assert (
        guarded_config["online_sam2"]["mask_publication_mode"]
        == "guarded_sam2_primary"
    )
    assert (
        guarded_config["tracker"]["recovery_publication_mode"]
        == "unified_three_evidence"
    )

    guarded_v1_config = {
        "online_sam2": {
            "enabled": True,
            "mask_publication_mode": "adaptive_fusion",
        }
    }
    guarded_v1 = _apply_online_sam2_live_override(
        guarded_v1_config,
        disable_online_sam2=False,
        object_mask_mode="guarded_v1",
    )
    assert guarded_v1.requested_object_mask_mode == "guarded_v1"
    assert guarded_v1.effective_mask_publication_mode == "adaptive_fusion"
    assert guarded_v1.effective_object_mask_mode == "guarded_v1"
    assert (
        guarded_v1_config["tracker"]["recovery_publication_mode"]
        == "legacy_double_confirm"
    )

    legacy_config = {
        "online_sam2": {
            "enabled": True,
            "mask_publication_mode": "adaptive_fusion",
        }
    }
    legacy = _apply_online_sam2_live_override(
        legacy_config,
        disable_online_sam2=False,
        object_mask_mode="legacy",
    )
    assert legacy.requested_object_mask_mode == "legacy"
    assert legacy.source_mask_publication_mode == "adaptive_fusion"
    assert legacy.effective_mask_publication_mode == "semantic_sam2"
    assert legacy.effective_recovery_publication_mode == "semantic_sam2_direct"
    assert legacy.effective_object_mask_mode == "legacy"
    assert legacy_config["online_sam2"]["mask_publication_mode"] == "semantic_sam2"

    with pytest.raises(ValueError, match="guarded.*legacy"):
        _apply_online_sam2_live_override(
            {"online_sam2": {"enabled": True}},
            disable_online_sam2=False,
            object_mask_mode="unknown",
        )

    disabled = _apply_online_sam2_live_override(
        enabled_config, disable_online_sam2=True
    )
    assert disabled.source_config_enabled is True
    assert disabled.disable_override_requested is True
    assert disabled.effective_config_enabled is False
    assert disabled.mode == "disabled_by_live_cli"
    assert enabled_config["online_sam2"]["enabled"] is False
    assert enabled_config["online_sam2"]["require_for_bbox_init"] is False
    assert (
        enabled_config["online_sam2"]["mask_publication_mode"]
        == "adaptive_fusion"
    )
    assert enabled_config["sam2"]["enabled"] is True
    assert enabled_config["tracker"]["mode"] == "adaptive_color_depth"

    production_disable = {
        "online_sam2": {
            "enabled": True,
            "require_for_bbox_init": True,
            "mask_publication_mode": "semantic_sam2",
        },
        "tracker": {"mode": "adaptive_color_depth"},
    }
    selection = _apply_online_sam2_live_override(
        production_disable,
        disable_online_sam2=True,
        object_mask_mode="guarded",
    )
    assert selection.effective_config_enabled is False
    assert selection.effective_object_mask_mode == "adaptive_only"
    assert production_disable["online_sam2"] == {
        "enabled": False,
        "require_for_bbox_init": False,
        "mask_publication_mode": "adaptive_fusion",
    }

    with pytest.raises(ValueError, match="incompatible.*legacy"):
        _apply_online_sam2_live_override(
            {"online_sam2": {"enabled": True}},
            disable_online_sam2=True,
            object_mask_mode="legacy",
        )

    source_disabled = _apply_online_sam2_live_override(
        {"online_sam2": {"enabled": False}}, disable_online_sam2=False
    )
    assert source_disabled.mode == "disabled_by_config"

    with pytest.raises(ValueError, match="must be a mapping"):
        _apply_online_sam2_live_override(
            {"online_sam2": None}, disable_online_sam2=True
        )


def test_formal_runtime_frame_timeout_contract_is_exact_and_preconstruction():
    _require_runtime_frame_timeout_contract(
        {"camera": {"runtime_frame_timeout_ms": 100}},
        expected_timeout_ms=100,
    )
    for config in (
        {"camera": {}},
        {"camera": {"runtime_frame_timeout_ms": 101}},
        {"camera": {"runtime_frame_timeout_ms": 100.0}},
        {"camera": {"runtime_frame_timeout_ms": True}},
        {},
    ):
        with pytest.raises(RuntimeError, match="formal D435 runtime frame timeout|camera mapping"):
            _require_runtime_frame_timeout_contract(
                config,
                expected_timeout_ms=100,
            )


def test_compute_thread_guard_verifies_and_restores_process_settings():
    import cv2
    from threadpoolctl import threadpool_info

    opencv_before = int(cv2.getNumThreads())
    blas_before = {
        value["prefix"]: int(value["num_threads"])
        for value in threadpool_info()
        if value.get("user_api") == "blas"
    }
    guard = _ComputeThreadGuard(1).start()
    try:
        assert guard.opencv_threads_active == 1
        assert guard.blas_pools_active
        assert all(pool["num_threads"] == 1 for pool in guard.blas_pools_active)
    finally:
        guard.close()

    assert int(cv2.getNumThreads()) == opencv_before
    blas_after = {
        value["prefix"]: int(value["num_threads"])
        for value in threadpool_info()
        if value.get("user_api") == "blas"
    }
    assert blas_after == blas_before


def test_sensor_age_rejects_stale_and_future_samples():
    _require_sensor_age(
        "camera", 0.01, maximum_age_s=0.10, maximum_future_skew_s=0.05
    )
    with pytest.raises(RuntimeError, match="stale"):
        _require_sensor_age(
            "camera", 0.11, maximum_age_s=0.10, maximum_future_skew_s=0.05
        )
    with pytest.raises(RuntimeError, match="future"):
        _require_sensor_age(
            "camera", -0.06, maximum_age_s=0.10, maximum_future_skew_s=0.05
        )


def test_camera_contract_checks_intrinsics_scale_and_unmodelled_distortion():
    expected_K = np.asarray([[600.0, 0.0, 1.0], [0.0, 601.0, 0.5], [0.0, 0.0, 1.0]])
    contract = SimpleNamespace(
        camera_K=expected_K,
        depth_scale_m_per_unit=0.001,
    )
    _validate_camera_contract(_camera_sample(), contract)
    _validate_camera_contract(
        replace(
            _camera_sample(),
            color_depth_timestamp_skew_s=0.005476,
        ),
        contract,
    )
    with pytest.raises(RuntimeError, match="not synchronized"):
        _validate_camera_contract(
            replace(
                _camera_sample(),
                color_depth_timestamp_skew_s=0.008001,
            ),
            contract,
        )

    bad_K = expected_K.copy()
    bad_K[0, 0] += 0.1
    with pytest.raises(RuntimeError, match="intrinsics differ"):
        _validate_camera_contract(_camera_sample(camera_K=bad_K), contract)
    with pytest.raises(RuntimeError, match="depth scale differs"):
        _validate_camera_contract(_camera_sample(depth_scale=0.002), contract)
    with pytest.raises(RuntimeError, match="distortion"):
        _validate_camera_contract(
            _camera_sample(distortion=[0.01, 0.0, 0.0, 0.0, 0.0]), contract
        )


def test_nearest_pose_returns_the_auditable_skew():
    first = np.eye(4)
    second = np.eye(4)
    second[0, 3] = 0.1
    pose, skew = _nearest_pose(
        deque([(1.0, first), (1.02, second)]),
        timestamp_s=1.018,
        maximum_delta_s=0.01,
    )
    np.testing.assert_array_equal(pose, second)
    assert skew == pytest.approx(0.002)
    with pytest.raises(RuntimeError, match="pose skew"):
        _nearest_pose(
            deque([(1.0, first)]),
            timestamp_s=1.1,
            maximum_delta_s=0.01,
        )


def test_audit_payload_stacks_exact_model_inputs_without_pickle_objects():
    record = {
        "pointcloud_history_xyzrgb_palm": np.zeros((4, 128, 6), np.float32),
        "pointcloud_valid_history": np.ones((4, 128), np.float32),
        "proprio_history67": np.zeros((4, 67), np.float32),
        "raw_policy_action13": np.zeros(13, np.float32),
        "pointcloud_status": np.asarray("fresh"),
    }
    payload = _stack_audit_records([record, record])
    assert payload["pointcloud_history_xyzrgb_palm"].shape == (2, 4, 128, 6)
    assert payload["pointcloud_valid_history"].shape == (2, 4, 128)
    assert payload["proprio_history67"].shape == (2, 4, 67)
    assert payload["raw_policy_action13"].shape == (2, 13)
    assert payload["pointcloud_status"].dtype.kind == "U"
    assert payload["audit_schema_version"].item() == 1
    assert payload["hardware_writes"].item() is False
    assert (
        payload["hardware_writes_semantics"].item()
        == "robot_actuator_or_register_commands_only"
    )
    assert payload["robot_command_writes"].item() is False
    assert payload["camera_configuration_writes"].item() is True
    assert all(value.dtype != object for value in payload.values())

    with pytest.raises(RuntimeError, match="zero valid policy steps"):
        _stack_audit_records([])
    with pytest.raises(ValueError, match="inconsistent fields"):
        _stack_audit_records([record, {"different": np.asarray(1)}])


class _FakeProvider:
    def __init__(
        self,
        *,
        start_error=None,
        select_result=True,
        bbox_result=True,
        initialization_evidence=None,
    ):
        self.start_error = start_error
        self.select_result = select_result
        self.bbox_result = bbox_result
        self.start_calls = 0
        self.stop_calls = 0
        self.camera = SimpleNamespace(get_frame=lambda: "frame")
        self.last_bbox_initialization_evidence = (
            SimpleNamespace(
                source="grabcut",
                valid=True,
                mask_area_px=72,
                mask_bbox_area_px=100,
                prompt_bbox_area_px=120,
                mask_bbox_xyxy=np.asarray([10, 10, 20, 20], dtype=np.int32),
                prompt_bbox_xyxy=np.asarray([2, 2, 28, 28], dtype=np.int32),
                solid_bbox_mask=False,
                full_prompt_box_mask=False,
            )
            if initialization_evidence is None
            else initialization_evidence
        )

    def start(self):
        self.start_calls += 1
        if self.start_error is not None:
            raise self.start_error

    def stop(self):
        self.stop_calls += 1

    def select_and_initialize(self):
        return self.select_result

    def initialize_from_bbox(self, frame, bbox):
        assert frame == "frame"
        np.testing.assert_array_equal(bbox, [1, 2, 4, 6])
        return self.bbox_result


class _SequencedROIProvider(_FakeProvider):
    def __init__(self, *, frame_ids, bbox_results, initialization_evidence=None):
        super().__init__(initialization_evidence=initialization_evidence)
        self._frames = iter(
            SimpleNamespace(frame_id=int(frame_id)) for frame_id in frame_ids
        )
        self._bbox_results = iter(bool(result) for result in bbox_results)
        self.frame_read_calls = 0
        self.bbox_frame_ids = []
        self.camera = SimpleNamespace(get_frame=self._get_frame)

    def _get_frame(self):
        self.frame_read_calls += 1
        return next(self._frames)

    def initialize_from_bbox(self, frame, bbox):
        np.testing.assert_array_equal(bbox, [1, 2, 4, 6])
        self.bbox_frame_ids.append(int(frame.frame_id))
        return next(self._bbox_results)


@pytest.mark.parametrize(
    ("provider", "roi", "message"),
    [
        (_FakeProvider(start_error=OSError("start failed")), None, "start failed"),
        (_FakeProvider(select_result=False), None, "cancelled or failed"),
        (_FakeProvider(), (-1, 2, 3, 4), "--roi"),
        (_FakeProvider(bbox_result=False), (1, 2, 3, 4), "ROI initialization"),
    ],
)
def test_provider_initialization_stops_on_every_failure(provider, roi, message):
    with pytest.raises((OSError, RuntimeError, ValueError), match=message):
        _start_and_initialize_provider(provider, roi)
    assert provider.stop_calls == 1


def test_provider_initialization_leaves_successful_provider_running():
    provider = _FakeProvider()
    assert _start_and_initialize_provider(provider, (1, 2, 3, 4)) is provider
    assert provider.start_calls == 1
    assert provider.stop_calls == 0


def test_fixed_roi_initialization_retries_only_on_three_fresh_frames():
    provider = _SequencedROIProvider(
        frame_ids=[10, 10, 9, 11, 12],
        bbox_results=[False, False, True],
    )

    assert _start_and_initialize_provider(provider, (1, 2, 3, 4)) is provider
    assert provider.frame_read_calls == 5
    assert provider.bbox_frame_ids == [10, 11, 12]
    assert provider.stop_calls == 0


def test_fixed_roi_initialization_recovers_after_one_retryable_camera_timeout():
    class RetryableTimeout(RuntimeError):
        retryable_camera_timeout = True

    provider = _SequencedROIProvider(
        frame_ids=[50, 51],
        bbox_results=[False, True],
    )
    original_get_frame = provider.camera.get_frame
    calls = 0

    def get_frame():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RetryableTimeout("fixed 0.100s capture deadline")
        return original_get_frame()

    provider.camera.get_frame = get_frame

    assert _start_and_initialize_provider(provider, (1, 2, 3, 4)) is provider
    assert calls == 3
    assert provider.frame_read_calls == 2
    assert provider.bbox_frame_ids == [50, 51]
    assert provider.stop_calls == 0


def test_fixed_roi_initialization_fails_after_bounded_retryable_timeouts():
    class RetryableTimeout(RuntimeError):
        retryable_camera_timeout = True

    provider = _FakeProvider()
    calls = 0

    def get_frame():
        nonlocal calls
        calls += 1
        raise RetryableTimeout("fixed 0.100s capture deadline")

    provider.camera.get_frame = get_frame

    with pytest.raises(
        RuntimeError, match="exceeded 3 retryable camera timeouts"
    ):
        _start_and_initialize_provider(provider, (1, 2, 3, 4))
    assert calls == 4
    assert provider.stop_calls == 1


def test_fixed_roi_initialization_stops_after_three_failed_fresh_attempts():
    provider = _SequencedROIProvider(
        frame_ids=[20, 21, 22],
        bbox_results=[False, False, False],
    )

    with pytest.raises(
        RuntimeError, match="failed after 3 fresh-frame attempts"
    ):
        _start_and_initialize_provider(provider, (1, 2, 3, 4))
    assert provider.bbox_frame_ids == [20, 21, 22]
    assert provider.stop_calls == 1


def test_fixed_roi_initialization_duplicate_frames_do_not_spend_attempts():
    provider = _SequencedROIProvider(
        frame_ids=[30] * 12,
        bbox_results=[False],
    )

    with pytest.raises(
        RuntimeError, match="could not obtain enough fresh increasing camera frames"
    ):
        _start_and_initialize_provider(provider, (1, 2, 3, 4))
    assert provider.frame_read_calls == 12
    assert provider.bbox_frame_ids == [30]
    assert provider.stop_calls == 1


def test_fixed_roi_initialization_does_not_retry_an_unsafe_success():
    provider = _SequencedROIProvider(
        frame_ids=[40, 41, 42],
        bbox_results=[True, True, True],
        initialization_evidence=SimpleNamespace(
            source="box",
            valid=True,
            mask_area_px=100,
            mask_bbox_area_px=100,
            prompt_bbox_area_px=100,
            solid_bbox_mask=True,
            full_prompt_box_mask=True,
        ),
    )

    with pytest.raises(RuntimeError, match="unsafe box fallback"):
        _start_and_initialize_provider(provider, (1, 2, 3, 4))
    assert provider.frame_read_calls == 1
    assert provider.bbox_frame_ids == [40]
    assert provider.stop_calls == 1


@pytest.mark.parametrize(
    "evidence",
    [
        None,
        SimpleNamespace(
            source="box_fallback_from_grabcut",
            valid=True,
            mask_area_px=3900,
            mask_bbox_area_px=3900,
            prompt_bbox_area_px=3900,
            solid_bbox_mask=True,
            full_prompt_box_mask=True,
        ),
        SimpleNamespace(
            source="grabcut",
            valid=True,
            mask_area_px=3900,
            mask_bbox_area_px=3900,
            prompt_bbox_area_px=3900,
            solid_bbox_mask=True,
            full_prompt_box_mask=True,
        ),
    ],
)
def test_live_roi_initialization_rejects_missing_box_or_solid_mask_evidence(
    evidence,
):
    provider = _FakeProvider(initialization_evidence=evidence)
    if evidence is None:
        # ``None`` is normally replaced by the fake's safe default; explicitly
        # exercise a provider that failed to retain structured provenance.
        provider.last_bbox_initialization_evidence = None
    with pytest.raises(RuntimeError, match="(structured mask evidence|box fallback|solid-box)"):
        _start_and_initialize_provider(provider, (1, 2, 3, 4))
    assert provider.stop_calls == 1


def test_live_roi_initialization_allows_non_rectangular_grabcut_object_mask():
    provider = _FakeProvider(
        initialization_evidence=SimpleNamespace(
            source="grabcut",
            valid=True,
            mask_area_px=2480,
            mask_bbox_area_px=3900,
            prompt_bbox_area_px=4096,
            mask_bbox_xyxy=np.asarray([10, 10, 74, 74], dtype=np.int32),
            prompt_bbox_xyxy=np.asarray([2, 2, 82, 82], dtype=np.int32),
            solid_bbox_mask=False,
            full_prompt_box_mask=False,
        )
    )
    _require_safe_bbox_initialization(provider)
    assert _start_and_initialize_provider(provider, (1, 2, 3, 4)) is provider
    assert provider.stop_calls == 0


def test_live_roi_initialization_rejects_mask_touching_prompt_boundary():
    provider = _FakeProvider(
        initialization_evidence=SimpleNamespace(
            source="grabcut",
            valid=True,
            mask_area_px=900,
            mask_bbox_area_px=1200,
            prompt_bbox_area_px=4096,
            mask_bbox_xyxy=np.asarray([20, 20, 60, 50], dtype=np.int32),
            prompt_bbox_xyxy=np.asarray([4, 4, 60, 60], dtype=np.int32),
            solid_bbox_mask=False,
            full_prompt_box_mask=False,
        )
    )

    with pytest.raises(RuntimeError, match="margins=.*required>=8px"):
        _require_safe_bbox_initialization(provider)


def test_live_roi_initialization_allows_sam2_to_extend_beyond_prompt():
    provider = _FakeProvider(
        initialization_evidence=SimpleNamespace(
            source="online_sam2_box",
            valid=True,
            mask_area_px=3622,
            mask_bbox_area_px=6177,
            prompt_bbox_area_px=8010,
            mask_bbox_xyxy=np.asarray([355, 245, 426, 332], dtype=np.int32),
            prompt_bbox_xyxy=np.asarray([348, 242, 438, 331], dtype=np.int32),
            solid_bbox_mask=False,
            full_prompt_box_mask=False,
        )
    )
    provider.cfg = {
        "camera": {"width": 848, "height": 480},
        "sam2": {"max_mask_area_ratio": 4.0, "max_bbox_area_ratio": 6.0},
    }

    _require_safe_bbox_initialization(provider)


def test_live_roi_initialization_rejects_sam2_at_camera_boundary():
    provider = _FakeProvider(
        initialization_evidence=SimpleNamespace(
            source="online_sam2_box",
            valid=True,
            mask_area_px=1200,
            mask_bbox_area_px=1800,
            prompt_bbox_area_px=2400,
            mask_bbox_xyxy=np.asarray([0, 20, 45, 60], dtype=np.int32),
            prompt_bbox_xyxy=np.asarray([0, 15, 50, 63], dtype=np.int32),
            solid_bbox_mask=False,
            full_prompt_box_mask=False,
        )
    )
    provider.cfg = {
        "camera": {"width": 848, "height": 480},
        "sam2": {"max_mask_area_ratio": 4.0, "max_bbox_area_ratio": 6.0},
    }

    with pytest.raises(RuntimeError, match="camera image boundary"):
        _require_safe_bbox_initialization(provider)


def test_camera_worker_discards_exactly_one_cold_provider_frame():
    class Provider:
        def __init__(self):
            self.calls = 0
            self.last_timings_ms = {}

        def step(self):
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("test finished")
            intrinsics = SimpleNamespace(
                fx=600.0,
                fy=601.0,
                ppx=1.0,
                ppy=0.5,
                distortion=np.zeros(5),
                model="distortion.inverse_brown_conrady",
            )
            frame = SimpleNamespace(
                intrinsics=intrinsics,
                color_bgr=np.zeros((2, 3, 3), dtype=np.uint8),
                depth_raw=np.full((2, 3), 1000, dtype=np.uint16),
                depth_m=np.ones((2, 3), dtype=np.float32),
                depth_scale=0.001,
                timestamp=float(self.calls),
                frame_id=self.calls,
                retrieved_at_s=float(self.calls) + 0.01,
                timestamp_domain="timestamp_domain.global_time",
                color_depth_timestamp_skew_s=0.0,
                color_depth_epoch_timestamp_skew_s=0.0,
                rejected_timestamp_skew_frames=0,
                last_rejected_color_depth_skew_s=None,
                dropped_queued_framesets=0,
                sensor_frame_number=self.calls,
            )
            mask = SimpleNamespace(
                mask=np.ones((2, 3), dtype=np.uint8), valid=True, message="ok"
            )
            return frame, mask, None, None

    class Sink:
        def __init__(self):
            self.values = []
            self.errors = []

        def put(self, value):
            self.values.append(value)

        def fail(self, error):
            self.errors.append(error)

    provider = Provider()
    stop = threading.Event()
    sink = Sink()
    _camera_worker(provider, stop, sink)
    assert [sample.frame_id for sample in sink.values] == [2]
    assert len(sink.errors) == 1
    assert "index=3, stage=formal" in str(sink.errors[0])
    assert stop.is_set()


def test_camera_worker_uses_reviewed_mask_only_provider_path():
    class Provider:
        def __init__(self):
            self.calls = 0
            self.full_calls = 0
            self.last_timings_ms = {"mask_gate": 12.5, "pcd": 99.0}

        def step(self):
            self.full_calls += 1
            raise AssertionError("full packet path must not run")

        def step_mask_only(self):
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("test finished")
            intrinsics = SimpleNamespace(
                fx=600.0,
                fy=601.0,
                ppx=1.0,
                ppy=0.5,
                distortion=np.zeros(5),
                model="distortion.inverse_brown_conrady",
            )
            frame = SimpleNamespace(
                intrinsics=intrinsics,
                color_bgr=np.zeros((2, 3, 3), dtype=np.uint8),
                depth_raw=np.full((2, 3), 1000, dtype=np.uint16),
                depth_m=np.ones((2, 3), dtype=np.float32),
                depth_scale=0.001,
                timestamp=float(self.calls),
                frame_id=self.calls,
                retrieved_at_s=float(self.calls) + 0.01,
                timestamp_domain="timestamp_domain.global_time",
                color_depth_timestamp_skew_s=0.0,
                color_depth_epoch_timestamp_skew_s=0.0,
                rejected_timestamp_skew_frames=0,
                last_rejected_color_depth_skew_s=None,
                dropped_queued_framesets=0,
                sensor_frame_number=self.calls,
            )
            mask = SimpleNamespace(
                mask=np.ones((2, 3), dtype=np.uint8), valid=True, message="ok"
            )
            return frame, mask

    class Sink:
        def __init__(self):
            self.values = []
            self.errors = []

        def put(self, value):
            self.values.append(value)

        def fail(self, error):
            self.errors.append(error)

    provider = Provider()
    stop = threading.Event()
    sink = Sink()
    _camera_worker(provider, stop, sink, "mask_only")
    assert provider.full_calls == 0
    assert [sample.frame_id for sample in sink.values] == [2]
    assert sink.values[0].depth_m is None
    np.testing.assert_array_equal(sink.values[0].depth_raw, 1000)
    assert sink.values[0].depth_raw.dtype == np.uint16
    assert not sink.values[0].depth_raw.flags.writeable
    assert sink.values[0].provider_output_mode == "mask_only"
    assert sink.values[0].provider_timings_ms[4] == pytest.approx(12.5)
    assert len(sink.errors) == 1
    assert "output_mode=mask_only" in str(sink.errors[0])


def test_mask_only_camera_worker_recovers_after_one_retryable_timeout():
    class RetryableTimeout(RuntimeError):
        retryable_camera_timeout = True

    class Provider:
        def __init__(self):
            self.calls = 0
            self.last_timings_ms = {}

        def step_mask_only(self):
            self.calls += 1
            if self.calls == 2:
                raise RetryableTimeout("fixed 0.100s capture deadline")
            if self.calls == 4:
                raise RuntimeError("test finished")
            intrinsics = SimpleNamespace(
                fx=600.0,
                fy=601.0,
                ppx=1.0,
                ppy=0.5,
                distortion=np.zeros(5),
                model="distortion.inverse_brown_conrady",
            )
            frame = SimpleNamespace(
                intrinsics=intrinsics,
                color_bgr=np.zeros((2, 3, 3), dtype=np.uint8),
                depth_raw=np.full((2, 3), 1000, dtype=np.uint16),
                depth_m=np.ones((2, 3), dtype=np.float32),
                depth_scale=0.001,
                timestamp=float(self.calls),
                frame_id=self.calls,
                retrieved_at_s=float(self.calls) + 0.01,
                timestamp_domain="timestamp_domain.global_time",
                color_depth_timestamp_skew_s=0.0,
                color_depth_epoch_timestamp_skew_s=0.0,
                rejected_timestamp_skew_frames=0,
                last_rejected_color_depth_skew_s=None,
                dropped_queued_framesets=0,
                sensor_frame_number=self.calls,
            )
            mask = SimpleNamespace(
                mask=np.ones((2, 3), dtype=np.uint8),
                valid=True,
                message="ok",
            )
            return frame, mask

    class Sink:
        def __init__(self):
            self.values = []
            self.errors = []
            self.transients = []

        def put(self, value):
            self.values.append(value)

        def fail(self, error):
            self.errors.append(error)

        def note_transient(self, error):
            self.transients.append(error)

    provider = Provider()
    stop = threading.Event()
    sink = Sink()
    _camera_worker(provider, stop, sink, "mask_only")

    # Call 1 is the cold discard, call 2 is a bounded retryable timeout, and
    # call 3 is still the formal stage and is published normally.
    assert provider.calls == 4
    assert [sample.frame_id for sample in sink.values] == [3]
    assert len(sink.errors) == 1
    assert len(sink.transients) == 1
    assert "fixed 0.100s" in str(sink.transients[0])
    assert "index=4, stage=formal" in str(sink.errors[0])
    assert "fixed 0.100s" not in str(sink.errors[0])
    assert stop.is_set()


def test_mask_only_camera_worker_continuous_timeouts_wait_for_external_stop():
    class RetryableTimeout(RuntimeError):
        retryable_camera_timeout = True

    entered = threading.Event()

    class Provider:
        def __init__(self):
            self.calls = 0
            self.last_timings_ms = {}

        def step_mask_only(self):
            self.calls += 1
            entered.set()
            raise RetryableTimeout("fixed 0.100s capture deadline")

    class Sink:
        def __init__(self):
            self.values = []
            self.errors = []

        def put(self, value):
            self.values.append(value)

        def fail(self, error):
            self.errors.append(error)

    provider = Provider()
    stop = threading.Event()
    sink = Sink()
    worker = threading.Thread(
        target=_camera_worker,
        args=(provider, stop, sink, "mask_only"),
        daemon=True,
    )
    worker.start()
    assert entered.wait(1.0)
    stop.set()
    worker.join(1.0)

    assert not worker.is_alive()
    assert provider.calls >= 1
    assert sink.values == []
    assert sink.errors == []


def test_mask_only_camera_worker_nonretryable_error_still_fails():
    class Provider:
        last_timings_ms = {}

        @staticmethod
        def step_mask_only():
            raise RuntimeError("fatal camera contract")

    class Sink:
        def __init__(self):
            self.values = []
            self.errors = []

        def put(self, value):
            self.values.append(value)

        def fail(self, error):
            self.errors.append(error)

    stop = threading.Event()
    sink = Sink()
    _camera_worker(Provider(), stop, sink, "mask_only")

    assert sink.values == []
    assert len(sink.errors) == 1
    assert "fatal camera contract" in str(sink.errors[0])
    assert stop.is_set()


def test_full_packet_camera_worker_does_not_retry_timeout_marker():
    class RetryableTimeout(RuntimeError):
        retryable_camera_timeout = True

    class Provider:
        last_timings_ms = {}

        @staticmethod
        def step():
            raise RetryableTimeout("mask-only retry marker")

    class Sink:
        def __init__(self):
            self.values = []
            self.errors = []

        def put(self, value):
            self.values.append(value)

        def fail(self, error):
            self.errors.append(error)

    stop = threading.Event()
    sink = Sink()
    _camera_worker(Provider(), stop, sink, "full_packet")

    assert sink.values == []
    assert len(sink.errors) == 1
    assert "mask-only retry marker" in str(sink.errors[0])
    assert stop.is_set()


def test_cleanup_never_stops_provider_while_camera_thread_is_alive():
    class FakeThread:
        def __init__(self, name, alive):
            self.name = name
            self.alive = alive
            self.joined = []

        def join(self, timeout):
            self.joined.append(timeout)

        def is_alive(self):
            return self.alive

    class Closable:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    class Stoppable:
        def __init__(self):
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1

    camera = FakeThread("v94-camera-readonly", True)
    franka = Closable()
    provider = Stoppable()
    stop = threading.Event()
    with pytest.raises(RuntimeError, match="provider.stop was not called"):
        _shutdown_readonly_resources(
            stop=stop,
            threads=[camera],
            camera_thread=camera,
            franka_reader=franka,
            provider=provider,
            join_timeout_s=0.0,
        )
    assert stop.is_set()
    assert camera.joined == [0.0]
    assert franka.close_calls == 1
    assert provider.stop_calls == 0


def test_cleanup_stops_provider_only_after_all_threads_are_dead():
    class FakeThread:
        name = "v94-camera-readonly"

        def __init__(self):
            self.joined = False

        def join(self, timeout):
            self.joined = True

        def is_alive(self):
            return False

    camera = FakeThread()
    franka = SimpleNamespace(close=lambda: None)
    provider_calls = []
    provider = SimpleNamespace(stop=lambda: provider_calls.append("stop"))
    stop = threading.Event()
    _shutdown_readonly_resources(
        stop=stop,
        threads=[camera],
        camera_thread=camera,
        franka_reader=franka,
        provider=provider,
        join_timeout_s=0.0,
    )
    assert camera.joined
    assert provider_calls == ["stop"]
