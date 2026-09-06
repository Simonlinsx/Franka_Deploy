from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from sim2real.closed_loop_core import (
    AUTHORIZATION_SCOPE,
    ClosedLoopCommand,
    ClosedLoopSafetyGate,
    ExecutedActionLedger,
    MotionAuthorization,
    SafetyState,
)
from robot_control.rh56.actuator import (
    RH56ActuationPreflight,
    RH56ActuatorState,
    RH56SafetyFeedback,
    RH56StopReport,
    RH56TransactionalActuator,
    _issue_supervised_rh56_preflight,
)
from robot_control.rh56.watchdog import (
    RH56OwnedSession,
    RH56WatchdogOwner,
    RH56WatchdogOwnerError,
    RH56WatchdogOwnerState,
    _advance_periodic_deadline,
)


_REQUIRED_RH56_CHECKS = (
    "evidence_physical_deadman_acceptance",
    "evidence_physical_emergency_stop_acceptance",
    "evidence_rh56_single_owner_serial_session",
    "evidence_rh56_full_six_axis_and_fingertip_fk",
    "evidence_rh56_full_thumb_rotation_range",
    "evidence_rh56_60hz_write_readback",
    "evidence_rh56_fault_disable_and_verified_stop",
    "evidence_dual_device_same_sequence_ack_transaction",
    "evidence_dual_device_rate_and_max_gap",
    "evidence_metric_rh56_sustained_command_rate_hz",
    "evidence_metric_dual_device_sustained_ack_rate_hz",
    "evidence_metric_dual_device_max_interaction_gap_s",
    "evidence_metric_policy_command_watchdog_timeout_s",
)


class _FakeClock:
    def __init__(self, value=10.0):
        self._lock = threading.Lock()
        self._value = float(value)

    def __call__(self):
        with self._lock:
            return self._value

    def advance(self, seconds):
        with self._lock:
            self._value += float(seconds)


class _FakeTransport:
    hardware_backed = False
    baud_rate = 115200
    transport_name = "watchdog-owner-fake"

    def __init__(
        self,
        clock,
        *,
        event_log=None,
        block_disable=None,
        block_feedback_on_call=None,
        feedback_block_started=None,
        feedback_block_release=None,
        feedback_angles_script=None,
    ):
        self.clock = clock
        self.event_log = [] if event_log is None else event_log
        self.block_disable = block_disable
        self.block_feedback_on_call = block_feedback_on_call
        self.feedback_block_started = feedback_block_started
        self.feedback_block_release = feedback_block_release
        self.feedback_angles_script = (
            None
            if feedback_angles_script is None
            else tuple(
                tuple(int(value) for value in sample)
                for sample in feedback_angles_script
            )
        )
        self.is_open = False
        self.targets = (-1,) * 6
        self.open_calls = 0
        self.close_calls = 0
        self.writes = []
        self.feedback_calls = 0
        self.feedback_capture_times = []
        self.transport_thread_ids = []

    def _record(self, name, value=None):
        ident = threading.get_ident()
        self.transport_thread_ids.append(ident)
        self.event_log.append((name, value, ident))

    def open(self):
        self._record("open")
        self.open_calls += 1
        self.is_open = True
        return self

    def close(self):
        self._record("close")
        self.close_calls += 1
        self.is_open = False

    def write_angle_set(self, values, *, deadline_monotonic_s):
        assert self.is_open
        assert self.clock() <= deadline_monotonic_s
        targets = tuple(int(v) for v in values)
        self._record("write", targets)
        self.writes.append(targets)
        if targets == (-1,) * 6 and self.block_disable is not None:
            self.block_disable.wait()
        self.clock.advance(0.0005)
        self.targets = targets

    def read_angle_set(self, *, deadline_monotonic_s):
        assert self.is_open
        assert self.clock() <= deadline_monotonic_s
        self._record("read_targets")
        self.clock.advance(0.0005)
        return self.targets

    def read_safety_feedback(self, *, deadline_monotonic_s):
        assert self.is_open
        assert self.clock() <= deadline_monotonic_s
        self._record("feedback")
        self.feedback_calls += 1
        if self.feedback_calls == self.block_feedback_on_call:
            if self.feedback_block_started is not None:
                self.feedback_block_started.set()
            if self.feedback_block_release is not None:
                self.feedback_block_release.wait()
        self.clock.advance(0.0005)
        self.feedback_capture_times.append(self.clock())
        angles = (500,) * 6
        if self.feedback_angles_script:
            index = min(
                self.feedback_calls - 1,
                len(self.feedback_angles_script) - 1,
            )
            angles = self.feedback_angles_script[index]
        return RH56SafetyFeedback(
            captured_monotonic_s=self.clock(),
            positions=angles,
            angles=angles,
            forces_g=(0,) * 6,
            currents_ma=(0,) * 6,
            errors=(0,) * 6,
            statuses=(2,) * 6,
            temperatures_c=(25,) * 6,
        )


class _PersistentReadFailureTransport(_FakeTransport):
    def read_safety_feedback(self, *, deadline_monotonic_s):
        assert self.is_open
        assert self.clock() <= deadline_monotonic_s
        self._record("feedback_failure")
        self.feedback_calls += 1
        raise TimeoutError("synthetic persistent compact-read timeout")


def _preflight():
    metrics = {
        "evidence_metric_rh56_sustained_command_rate_hz": 60.0,
        "evidence_metric_dual_device_sustained_ack_rate_hz": 60.0,
        "evidence_metric_dual_device_max_interaction_gap_s": 1.0 / 30.0,
        "evidence_metric_policy_command_watchdog_timeout_s": 0.05,
    }
    return RH56ActuationPreflight.from_report(
        {
            "result": "PASS",
            "readiness_level": "C2_BOUNDED_CLOSED_LOOP",
            "offline_only": True,
            "device_access": False,
            "hardware_writes": False,
            "robot_command_writes": False,
            "arming_state": "DISARMED",
            "motion_authorization_created": False,
            "physical_motion_authorized": False,
            "future_authorization_scope_required": AUTHORIZATION_SCOPE,
            "eligible_for_operator_authorization": True,
            "run_id": "owner-commissioning",
            "failed_checks": [],
            "blockers": [],
            "inputs": {
                "sha256": {"commissioning_profile_sha256": "a" * 64}
            },
            "checks": [
                {
                    "code": code,
                    "passed": True,
                    **({"actual": metrics[code]} if code in metrics else {}),
                }
                for code in _REQUIRED_RH56_CHECKS
            ],
        }
    )


def _authorization():
    return MotionAuthorization(
        run_id="owner-execution",
        authorization_id="owner-authorization",
        issued_monotonic_s=9.0,
        expires_monotonic_s=100.0,
    )


def _gate(authorization):
    gate = ClosedLoopSafetyGate()
    gate.update_interlocks(deadman_asserted=True, estop_healthy=True)
    gate.arm(
        authorization,
        run_id=authorization.run_id,
        now_monotonic_s=10.0,
        franka_rest_verified=True,
        rh56_disabled_verified=True,
    )
    return gate


class _SessionFactory:
    def __init__(
        self,
        clock,
        transport,
        gate,
        *,
        maximum_feedback_age_s=0.025,
        supervised=False,
        supervised_inter_command_watchdog_timeout_s=0.05,
    ):
        self.clock = clock
        self.transport = transport
        self.gate = gate
        self.maximum_feedback_age_s = maximum_feedback_age_s
        self.supervised = supervised
        self.supervised_inter_command_watchdog_timeout_s = (
            supervised_inter_command_watchdog_timeout_s
        )
        self.actuator = None
        self.calls = 0
        self.thread_ids = []

    def __call__(self):
        self.calls += 1
        self.thread_ids.append(threading.get_ident())
        opened = self.transport.open()
        assert opened is self.transport
        try:
            actuator = RH56TransactionalActuator(
                self.transport,
                command_watchdog_timeout_s=0.05,
                supervised_inter_command_watchdog_timeout_s=(
                    self.supervised_inter_command_watchdog_timeout_s
                ),
                maximum_feedback_age_s=self.maximum_feedback_age_s,
                stop_timeout_s=0.25,
                stop_verify_samples=2,
                stop_verify_interval_s=0.0,
                monotonic=self.clock,
                sleep=self.clock.advance,
            )
            actuator.claim_for_current_thread()
            if self.supervised:
                actuator.arm_supervised(
                    preflight=_issue_supervised_rh56_preflight(
                        run_id="owner-execution",
                        confirmed_permit_sha256="b" * 64,
                        commissioning_profile_sha256="a" * 64,
                        watchdog_timeout_s=0.05,
                        inter_command_watchdog_timeout_s=(
                            self.supervised_inter_command_watchdog_timeout_s
                        ),
                    ),
                    authorization=_authorization(),
                    safety_gate=self.gate,
                    run_id="owner-execution",
                    now_monotonic_s=self.clock(),
                )
            else:
                actuator.arm(
                    preflight=_preflight(),
                    authorization=_authorization(),
                    safety_gate=self.gate,
                    run_id="owner-execution",
                    now_monotonic_s=self.clock(),
                )
            self.actuator = actuator
            return RH56OwnedSession(
                actuator=actuator,
                close=self.transport.close,
            )
        except BaseException:
            self.transport.close()
            raise


def _owner(
    *,
    clock=None,
    transport=None,
    ledger=None,
    event_log=None,
    callback=None,
    block_disable=None,
    supervised_target_only=False,
    feedback_period_s=1.0 / 30.0,
    feedback_hard_age_s=0.050,
    block_feedback_on_call=None,
    feedback_block_started=None,
    feedback_block_release=None,
    feedback_angles_script=None,
    supervised_inter_command_watchdog_timeout_s=0.05,
    tracking_significant_gap_units=None,
    tracking_min_progress_units=None,
    tracking_timeout_s=None,
):
    clock = _FakeClock() if clock is None else clock
    ledger = (
        ExecutedActionLedger(maximum_commit_latency_s=0.05)
        if ledger is None
        else ledger
    )
    transport = (
        _FakeTransport(
            clock,
            event_log=event_log,
            block_disable=block_disable,
            block_feedback_on_call=block_feedback_on_call,
            feedback_block_started=feedback_block_started,
            feedback_block_release=feedback_block_release,
            feedback_angles_script=feedback_angles_script,
        )
        if transport is None
        else transport
    )
    authorization = _authorization()
    gate = _gate(authorization)
    factory = _SessionFactory(
        clock,
        transport,
        gate,
        maximum_feedback_age_s=(
            feedback_hard_age_s if supervised_target_only else 0.025
        ),
        supervised=supervised_target_only,
        supervised_inter_command_watchdog_timeout_s=(
            supervised_inter_command_watchdog_timeout_s
        ),
    )
    owner_kwargs = {}
    if tracking_significant_gap_units is not None:
        owner_kwargs["tracking_significant_gap_units"] = (
            tracking_significant_gap_units
        )
    if tracking_min_progress_units is not None:
        owner_kwargs["tracking_min_progress_units"] = tracking_min_progress_units
    if tracking_timeout_s is not None:
        owner_kwargs["tracking_timeout_s"] = tracking_timeout_s
    owner = RH56WatchdogOwner(
        factory,
        ledger,
        fault_callback=callback,
        monotonic=clock,
        realtime=lambda: 1_750_000_000.0 + clock(),
        startup_timeout_s=0.5,
        response_timeout_s=0.5,
        join_timeout_s=0.5,
        supervised_target_only=supervised_target_only,
        feedback_period_s=feedback_period_s,
        feedback_hard_age_s=feedback_hard_age_s,
        **owner_kwargs,
    )
    return owner, factory, transport, ledger, gate, clock


def _command(ledger, clock, *, sequence=1, rh56_target=960):
    action = np.full(13, 0.25, dtype=np.float32)
    return ClosedLoopCommand(
        sequence=sequence,
        produced_monotonic_s=clock(),
        observation_realtime_s=1_750_000_000.0 + clock(),
        previous_executed_action13_used=ledger.previous_executed_action13(),
        raw_policy_action13=action,
        executed_policy_action13=action,
        franka_target_q_rad=np.zeros(7),
        rh56_angle_set_register_order=np.full(
            6, int(rh56_target), dtype=np.int32
        ),
    )


def _wait_until(predicate, *, timeout_s=0.5):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return bool(predicate())


@pytest.mark.parametrize(
    ("deadline_s", "now_s", "expected_s"),
    (
        (10.05, 10.06, 10.10),
        (10.05, 10.171, 10.20),
        (10.05, 10.05, 10.10),
    ),
)
def test_periodic_feedback_deadline_keeps_absolute_cadence(
    deadline_s, now_s, expected_s
):
    assert _advance_periodic_deadline(
        deadline_s,
        period_s=0.05,
        now_s=now_s,
    ) == pytest.approx(expected_s)


def test_constructor_is_hardware_inert_and_module_has_no_execute_cli():
    owner, factory, transport, _ledger, _gate_value, _clock = _owner()

    assert owner.state is RH56WatchdogOwnerState.CREATED
    assert owner.independent_watchdog_active is False
    assert owner.fault_callback_configured is False
    assert owner.supervised_target_only is False
    assert owner.first_command_readiness_verified is False
    assert factory.calls == 0
    assert transport.open_calls == 0
    assert transport.transport_thread_ids == []

    import robot_control.rh56.watchdog as module

    assert not hasattr(module, "main")
    assert not hasattr(module, "build_parser")


def test_supervised_target_queues_behind_inflight_feedback_without_overlapping_io():
    feedback_started = threading.Event()
    release_feedback = threading.Event()
    owner, _factory, transport, ledger, _gate_value, clock = _owner(
        callback=lambda _fault: None,
        supervised_target_only=True,
        block_feedback_on_call=3,
        feedback_block_started=feedback_started,
        feedback_block_release=release_feedback,
    )
    owner.start()
    assert owner.first_command_readiness_verified is True
    clock.advance(0.034)
    assert feedback_started.wait(0.2)

    command = _command(ledger, clock)
    ledger.stage(command, now_monotonic_s=clock())
    assert ledger.acknowledge(
        "franka", sequence=command.sequence, now_monotonic_s=clock()
    ) is False
    outcome = []
    waiter = threading.Thread(target=lambda: outcome.append(owner.execute(command)))
    waiter.start()
    time.sleep(0.010)
    assert waiter.is_alive()
    assert outcome == []
    assert not [value for value in transport.writes if value != (-1,) * 6]

    release_feedback.set()
    waiter.join(timeout=0.2)
    assert not waiter.is_alive()
    assert [receipt.sequence for receipt in outcome] == [1]
    assert [value for value in transport.writes if value != (-1,) * 6] == [
        (960,) * 6
    ]
    assert owner.stop_and_close().rh56_disabled_verified is True


def test_supervised_submit_needs_ledger_stage_but_not_parent_handshake():
    owner, _factory, transport, ledger, _gate_value, clock = _owner(
        callback=lambda _fault: None,
        supervised_target_only=True,
    )
    owner.start()
    command = _command(ledger, clock)
    with pytest.raises(RH56WatchdogOwnerError, match="not staged"):
        owner.submit(command)

    assert not [value for value in transport.writes if value != (-1,) * 6]
    ledger.stage(command, now_monotonic_s=clock())
    assert ledger.acknowledge(
        "franka", sequence=command.sequence, now_monotonic_s=clock()
    ) is False
    assert owner.execute(command).sequence == 1
    assert [value for value in transport.writes if value != (-1,) * 6] == [
        (960,) * 6
    ]
    assert owner.stop_and_close().rh56_disabled_verified is True


def test_persistent_bootstrap_read_failure_never_opens_ready_gate_or_writes_numeric():
    clock = _FakeClock()
    transport = _PersistentReadFailureTransport(clock)
    owner, _factory, transport, _ledger, _gate_value, _clock = _owner(
        clock=clock,
        transport=transport,
        callback=lambda _fault: None,
        supervised_target_only=True,
    )

    with pytest.raises(RH56WatchdogOwnerError, match="failed during startup"):
        owner.start()

    assert owner.first_command_readiness_verified is False
    assert not [value for value in transport.writes if value != (-1,) * 6]
    result = owner.stop_and_close()
    assert result.worker_stopped is True
    assert result.fault is not None
    assert "persistent compact-read timeout" in result.fault.reason


def test_fault_stop_error_survives_later_empty_finally_update_until_verified():
    owner, _factory, _transport, _ledger, _gate_value, clock = _owner()
    owner._publish_fault("synthetic owner fault")
    owner._finish_fault_stop(
        stop_report=None,
        stop_error=RuntimeError("original stop failure"),
    )
    owner._finish_fault_stop(stop_report=None, stop_error=None)

    fault = owner.fault_snapshot
    assert fault is not None
    assert fault.stop_confirmed is False
    assert fault.stop_error == "original stop failure"

    owner._finish_fault_stop(
        stop_report=RH56StopReport(
            verified=True,
            disable_passes_verified=2,
            feedback_samples_verified=2,
            completed_monotonic_s=clock(),
        ),
        stop_error=None,
    )
    fault = owner.fault_snapshot
    assert fault is not None
    assert fault.stop_confirmed is True
    assert fault.stop_error is None


def test_supervised_target_only_acks_without_per_command_complete_feedback():
    owner, _factory, transport, ledger, _gate_value, clock = _owner(
        callback=lambda _fault: None,
        supervised_target_only=True,
    )
    owner.start()
    assert owner.supervised_target_only is True
    assert owner.feedback_period_s == pytest.approx(1.0 / 30.0)
    assert owner.feedback_hard_age_s == pytest.approx(0.050)
    assert transport.feedback_calls == 2

    command = _command(ledger, clock)
    ledger.stage(command, now_monotonic_s=clock())
    assert ledger.acknowledge(
        "franka", sequence=command.sequence, now_monotonic_s=clock()
    ) is False
    receipt = owner.execute(command)

    assert receipt.sequence == 1
    assert ledger.last_committed_sequence == 1
    assert transport.feedback_calls == 2
    assert [event[0] for event in transport.event_log].count("read_targets") == 0

    result = owner.stop_and_close()
    assert result.rh56_disabled_verified is True


def test_supervised_due_feedback_does_not_delay_target_ticket_completion():
    feedback_started = threading.Event()
    release_feedback = threading.Event()
    owner, _factory, transport, ledger, _gate_value, clock = _owner(
        callback=lambda _fault: None,
        supervised_target_only=True,
        block_feedback_on_call=3,
        feedback_block_started=feedback_started,
        feedback_block_release=release_feedback,
    )
    owner.start()
    clock.advance(0.034)
    command = _command(ledger, clock)
    ledger.stage(command, now_monotonic_s=clock())
    assert ledger.acknowledge(
        "franka", sequence=command.sequence, now_monotonic_s=clock()
    ) is False

    receipt = owner.execute(command)
    assert receipt.sequence == 1
    assert ledger.last_committed_sequence == 1
    assert feedback_started.wait(0.2)
    # The worker is blocked in the independently scheduled complete feedback,
    # but the verified target ticket was already returned to the 60 Hz path.
    assert owner.state is RH56WatchdogOwnerState.RUNNING

    release_feedback.set()
    assert _wait_until(lambda: transport.feedback_calls == 3)
    assert _wait_until(
        lambda: len(owner.feedback_history_snapshot(maximum_age_s=0.050).samples)
        == 3
    )
    result = owner.stop_and_close()
    assert result.rh56_disabled_verified is True


def test_supervised_six_target_acks_schedule_about_three_feedback_samples():
    owner, _factory, transport, ledger, _gate_value, clock = _owner(
        callback=lambda _fault: None,
        supervised_target_only=True,
    )
    owner.start()
    assert transport.feedback_calls == 2

    receipts = []
    for sequence in range(1, 7):
        clock.advance(1.0 / 60.0)
        command = _command(ledger, clock, sequence=sequence)
        ledger.stage(command, now_monotonic_s=clock())
        assert ledger.acknowledge(
            "franka", sequence=sequence, now_monotonic_s=clock()
        ) is False
        receipt = owner.execute(command)
        assert receipt.sequence == sequence
        receipts.append(receipt)

    assert ledger.last_committed_sequence == 6
    assert _wait_until(lambda: transport.feedback_calls == 5)
    numeric_writes = [
        values for values in transport.writes if values != (-1,) * 6
    ]
    result = owner.stop_and_close()
    # The 60 Hz ledger heartbeat remains sequence-exact, but a sample-held
    # physical target is not rewritten: repeatedly writing the same ANGLE_SET
    # can restart motion planning in the RH56 firmware.
    assert len(numeric_writes) == 1
    assert [receipt.numeric_write_performed for receipt in receipts] == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    # Two bootstrap samples plus three independently scheduled ~30 Hz samples.
    assert len(owner.feedback_history_snapshot(maximum_age_s=0.050).samples) == 5
    assert result.rh56_disabled_verified is True


def test_supervised_twenty_hz_targets_and_feedback_keep_absolute_cadence():
    owner, _factory, transport, ledger, _gate_value, clock = _owner(
        callback=lambda _fault: None,
        supervised_target_only=True,
        feedback_period_s=0.050,
        feedback_hard_age_s=0.150,
        supervised_inter_command_watchdog_timeout_s=0.125,
    )
    owner.start()
    first_policy_tick_s = clock()

    for sequence in range(1, 41):
        scheduled_tick_s = first_policy_tick_s + sequence * 0.050
        clock.advance(max(0.0, scheduled_tick_s - clock()))
        command = _command(
            ledger,
            clock,
            sequence=sequence,
            rh56_target=500 + sequence,
        )
        ledger.stage(command, now_monotonic_s=clock())
        assert ledger.acknowledge(
            "franka", sequence=sequence, now_monotonic_s=clock()
        ) is False
        assert owner.execute(command).sequence == sequence

    assert ledger.last_committed_sequence == 40
    assert _wait_until(lambda: transport.feedback_calls == 42)
    scheduled_captures = transport.feedback_capture_times[2:]
    assert len(scheduled_captures) == 40
    intervals = np.diff(np.asarray(scheduled_captures, dtype=np.float64))
    np.testing.assert_allclose(intervals, 0.050, atol=1.0e-12, rtol=0.0)
    assert scheduled_captures[-1] - scheduled_captures[0] == pytest.approx(
        39 * 0.050
    )
    assert owner.stop_and_close().rh56_disabled_verified is True


def test_supervised_stationary_angle_act_under_large_target_gap_faults_and_disables():
    callback_seen = threading.Event()
    owner, _factory, transport, ledger, _gate_value, clock = _owner(
        callback=lambda _fault: callback_seen.set(),
        supervised_target_only=True,
        feedback_period_s=0.020,
        feedback_hard_age_s=0.150,
        feedback_angles_script=[(1000,) * 6] * 16,
        supervised_inter_command_watchdog_timeout_s=0.125,
        tracking_significant_gap_units=40,
        tracking_min_progress_units=5,
        tracking_timeout_s=0.080,
    )
    owner.start()
    command = _command(ledger, clock, rh56_target=900)
    ledger.stage(command, now_monotonic_s=clock())
    assert ledger.acknowledge(
        "franka", sequence=command.sequence, now_monotonic_s=clock()
    ) is False
    receipt = owner.execute(command)
    assert receipt.numeric_write_performed is True

    for _ in range(5):
        clock.advance(0.021)
        time.sleep(0.025)
        if callback_seen.is_set():
            break

    assert callback_seen.wait(0.5)
    snapshot = owner.physical_tracking_snapshot()
    assert snapshot.verdict == "failed"
    assert any(snapshot.challenged_axes)
    result = owner.stop_and_close()
    assert result.fault is not None
    assert "ANGLE_ACT" in result.fault.reason
    assert "progress" in result.fault.reason
    assert result.rh56_disabled_verified is True
    assert transport.writes[-2:] == [(-1,) * 6, (-1,) * 6]


def test_supervised_fresh_angle_act_progress_creates_physical_tracking_proof():
    owner, _factory, transport, ledger, _gate_value, clock = _owner(
        callback=lambda _fault: None,
        supervised_target_only=True,
        feedback_period_s=0.020,
        feedback_hard_age_s=0.150,
        feedback_angles_script=[
            (1000,) * 6,
            (1000,) * 6,
            (980,) * 6,
            (960,) * 6,
        ],
        supervised_inter_command_watchdog_timeout_s=0.125,
        tracking_significant_gap_units=40,
        tracking_min_progress_units=5,
        tracking_timeout_s=0.080,
    )
    owner.start()
    command = _command(ledger, clock, rh56_target=900)
    ledger.stage(command, now_monotonic_s=clock())
    assert ledger.acknowledge(
        "franka", sequence=command.sequence, now_monotonic_s=clock()
    ) is False
    assert owner.execute(command).numeric_write_performed is True

    clock.advance(0.021)
    assert _wait_until(lambda: transport.feedback_calls >= 3)
    snapshot = owner.physical_tracking_snapshot()
    result = owner.stop_and_close()
    assert snapshot.verdict == "verified"
    assert all(snapshot.verified_axes)
    assert snapshot.latest_feedback_fresh is True
    assert result.rh56_disabled_verified is True


def test_verified_axis_may_stop_short_of_target_after_grasp_contact():
    owner, _factory, _transport, _ledger, _gate_value, clock = _owner(
        supervised_target_only=True,
        tracking_significant_gap_units=40,
        tracking_min_progress_units=5,
        tracking_timeout_s=0.080,
    )
    initial = RH56SafetyFeedback(
        captured_monotonic_s=clock(),
        positions=(100,) * 6,
        angles=(1000,) * 6,
        forces_g=(0,) * 6,
        currents_ma=(0,) * 6,
        errors=(0,) * 6,
        statuses=(2,) * 6,
        temperatures_c=(25,) * 6,
    )
    owner._cache_feedback(initial)
    owner._register_physical_tracking_target(
        (900, 1000, 1000, 1000, 1000, 1000),
        now_monotonic_s=clock(),
    )
    clock.advance(0.020)
    progressed = RH56SafetyFeedback(
        captured_monotonic_s=clock(),
        positions=(120, 100, 100, 100, 100, 100),
        angles=(980, 1000, 1000, 1000, 1000, 1000),
        forces_g=(0,) * 6,
        currents_ma=(40, 0, 0, 0, 0, 0),
        errors=(0,) * 6,
        statuses=(1, 2, 2, 2, 2, 2),
        temperatures_c=(25,) * 6,
    )
    owner._cache_feedback(progressed)
    assert owner.physical_tracking_snapshot().verified_axes[0] is True

    # The motor has already proved it can follow commands.  A later closure
    # target may be blocked by the grasped object while all safety feedback
    # remains healthy; that expected contact hold must not become a no-motion
    # fault merely because ANGLE_ACT cannot reach the free-space target.
    owner._register_physical_tracking_target(
        (800, 1000, 1000, 1000, 1000, 1000),
        now_monotonic_s=clock(),
    )
    clock.advance(0.100)
    stationary_contact_hold = RH56SafetyFeedback(
        captured_monotonic_s=clock(),
        positions=progressed.positions,
        angles=progressed.angles,
        forces_g=(20, 0, 0, 0, 0, 0),
        currents_ma=(50, 0, 0, 0, 0, 0),
        errors=(0,) * 6,
        statuses=(1, 2, 2, 2, 2, 2),
        temperatures_c=(25,) * 6,
    )
    owner._cache_feedback(stationary_contact_hold)

    snapshot = owner.physical_tracking_snapshot()
    assert snapshot.verdict == "verified"
    assert snapshot.verified_axes[0] is True
    assert snapshot.active_challenge_axes[0] is False
    assert snapshot.failed_axes[0] is False
    assert snapshot.failure_reason is None


def test_withdrawn_axis_does_not_erase_other_axis_physical_tracking_proof():
    owner, _factory, _transport, _ledger, _gate_value, clock = _owner(
        supervised_target_only=True,
        tracking_significant_gap_units=40,
        tracking_min_progress_units=5,
        tracking_timeout_s=0.200,
    )
    initial = RH56SafetyFeedback(
        captured_monotonic_s=clock(),
        positions=(100,) * 6,
        angles=(1000,) * 6,
        forces_g=(0,) * 6,
        currents_ma=(0,) * 6,
        errors=(0,) * 6,
        statuses=(2,) * 6,
        temperatures_c=(25,) * 6,
    )
    owner._cache_feedback(initial)
    owner._register_physical_tracking_target(
        (900, 900, 1000, 1000, 1000, 1000),
        now_monotonic_s=clock(),
    )
    clock.advance(0.020)
    progressed = RH56SafetyFeedback(
        captured_monotonic_s=clock(),
        positions=(120, 100, 100, 100, 100, 100),
        angles=(980, 1000, 1000, 1000, 1000, 1000),
        forces_g=(0,) * 6,
        currents_ma=(0,) * 6,
        errors=(0,) * 6,
        statuses=(1,) * 6,
        temperatures_c=(25,) * 6,
    )
    owner._cache_feedback(progressed)
    # Axis 1 was challenged but the later policy target returned inside the
    # 40-unit threshold before that axis moved.  Axis 0 retains real progress.
    owner._register_physical_tracking_target(
        (900, 980, 1000, 1000, 1000, 1000),
        now_monotonic_s=clock(),
    )

    snapshot = owner.physical_tracking_snapshot()
    assert snapshot.challenged_axes[:2] == (True, True)
    assert snapshot.verified_axes[:2] == (True, False)
    assert snapshot.active_challenge_axes[1] is False
    assert snapshot.verdict == "verified"


def test_only_withdrawn_challenge_is_not_reported_as_verified():
    owner, _factory, _transport, _ledger, _gate_value, clock = _owner(
        supervised_target_only=True,
        tracking_significant_gap_units=40,
        tracking_min_progress_units=5,
        tracking_timeout_s=0.200,
    )
    initial = RH56SafetyFeedback(
        captured_monotonic_s=clock(),
        positions=(100,) * 6,
        angles=(1000,) * 6,
        forces_g=(0,) * 6,
        currents_ma=(0,) * 6,
        errors=(0,) * 6,
        statuses=(2,) * 6,
        temperatures_c=(25,) * 6,
    )
    owner._cache_feedback(initial)
    owner._register_physical_tracking_target(
        (900, 1000, 1000, 1000, 1000, 1000),
        now_monotonic_s=clock(),
    )
    owner._register_physical_tracking_target(
        (980, 1000, 1000, 1000, 1000, 1000),
        now_monotonic_s=clock(),
    )

    snapshot = owner.physical_tracking_snapshot()
    assert snapshot.challenged_axes[0] is True
    assert snapshot.verified_axes[0] is False
    assert snapshot.active_challenge_axes[0] is False
    assert snapshot.verdict == "challenged_withdrawn_unverified"


def test_streamed_target_moving_toward_actual_does_not_open_false_challenge():
    owner, _factory, _transport, _ledger, _gate_value, clock = _owner(
        supervised_target_only=True,
        tracking_significant_gap_units=50,
        tracking_min_progress_units=3,
        tracking_timeout_s=0.200,
    )
    initial = RH56SafetyFeedback(
        captured_monotonic_s=clock(),
        positions=(100,) * 6,
        angles=(100, 1000, 1000, 1000, 1000, 1000),
        forces_g=(0,) * 6,
        currents_ma=(0,) * 6,
        errors=(0,) * 6,
        statuses=(2,) * 6,
        temperatures_c=(25,) * 6,
    )
    owner._cache_feedback(initial)
    owner._register_physical_tracking_target(
        (100, 1000, 1000, 1000, 1000, 1000),
        now_monotonic_s=clock(),
    )
    endpoint_advanced = RH56SafetyFeedback(
        captured_monotonic_s=clock(),
        positions=(0, 100, 100, 100, 100, 100),
        angles=(0, 1000, 1000, 1000, 1000, 1000),
        forces_g=(0,) * 6,
        currents_ma=(0,) * 6,
        errors=(0,) * 6,
        statuses=(1, 2, 2, 2, 2, 2),
        temperatures_c=(25,) * 6,
    )
    owner._cache_feedback(endpoint_advanced)
    owner._register_physical_tracking_target(
        (75, 1000, 1000, 1000, 1000, 1000),
        now_monotonic_s=clock(),
    )

    snapshot = owner.physical_tracking_snapshot()
    assert snapshot.challenged_axes[0] is False
    assert snapshot.verdict == "not_exercised"


def test_physical_tracking_challenge_threshold_is_inclusive():
    owner, _factory, _transport, _ledger, _gate_value, clock = _owner(
        supervised_target_only=True,
        tracking_significant_gap_units=50,
        tracking_min_progress_units=3,
        tracking_timeout_s=0.200,
    )
    initial = RH56SafetyFeedback(
        captured_monotonic_s=clock(),
        positions=(100,) * 6,
        angles=(1000,) * 6,
        forces_g=(0,) * 6,
        currents_ma=(0,) * 6,
        errors=(0,) * 6,
        statuses=(2,) * 6,
        temperatures_c=(25,) * 6,
    )
    owner._cache_feedback(initial)

    owner._register_physical_tracking_target(
        (951, 1000, 1000, 1000, 1000, 1000),
        now_monotonic_s=clock(),
    )
    below = owner.physical_tracking_snapshot()
    assert below.challenged_axes[0] is False
    assert below.active_challenge_axes[0] is False
    assert below.verdict == "not_exercised"

    owner._register_physical_tracking_target(
        (950, 1000, 1000, 1000, 1000, 1000),
        now_monotonic_s=clock(),
    )
    at_threshold = owner.physical_tracking_snapshot()
    assert at_threshold.challenged_axes[0] is True
    assert at_threshold.active_challenge_axes[0] is True
    assert at_threshold.verdict == "challenged_unverified"


def test_force_act_contact_proves_tracking_without_free_space_motion():
    owner, _factory, _transport, _ledger, _gate_value, clock = _owner(
        supervised_target_only=True,
        tracking_significant_gap_units=50,
        tracking_min_progress_units=3,
        tracking_timeout_s=0.200,
    )
    initial = RH56SafetyFeedback(
        captured_monotonic_s=clock(),
        positions=(100,) * 6,
        angles=(1000,) * 6,
        forces_g=(-40,) * 6,
        currents_ma=(0,) * 6,
        errors=(0,) * 6,
        statuses=(1,) * 6,
        temperatures_c=(25,) * 6,
    )
    owner._cache_feedback(initial)
    owner._register_physical_tracking_target(
        (1000, 1000, 1000, 500, 1000, 1000),
        now_monotonic_s=clock(),
    )
    clock.advance(0.100)
    grasp_contact = RH56SafetyFeedback(
        captured_monotonic_s=clock(),
        positions=initial.positions,
        angles=initial.angles,
        forces_g=(-40, -40, -40, 420, -40, -40),
        currents_ma=(0, 0, 0, 30, 0, 0),
        errors=(0,) * 6,
        statuses=(1, 1, 1, 0, 1, 1),
        temperatures_c=(25,) * 6,
    )
    owner._cache_feedback(grasp_contact)

    snapshot = owner.physical_tracking_snapshot()
    assert snapshot.verdict == "verified"
    assert snapshot.verified_axes[3] is True
    assert snapshot.contact_verified_axes[3] is True
    assert snapshot.failed_axes[3] is False


def test_streamed_closure_preserves_precontact_force_baseline_until_challenge():
    owner, _factory, _transport, _ledger, _gate_value, clock = _owner(
        supervised_target_only=True,
        tracking_significant_gap_units=50,
        tracking_min_progress_units=3,
        tracking_timeout_s=0.200,
    )
    initial = RH56SafetyFeedback(
        captured_monotonic_s=clock(),
        positions=(100,) * 6,
        angles=(1000,) * 6,
        forces_g=(0, 0, 0, 0, -8, 0),
        currents_ma=(0,) * 6,
        errors=(0,) * 6,
        statuses=(2,) * 6,
        temperatures_c=(25,) * 6,
    )
    owner._cache_feedback(initial)
    owner._register_physical_tracking_target(
        (1000,) * 6, now_monotonic_s=clock()
    )
    # Thumb closure starts while its ANGLE_ACT gap is still below the formal
    # 50-unit challenge threshold.
    owner._register_physical_tracking_target(
        (1000, 1000, 1000, 1000, 998, 1000),
        now_monotonic_s=clock(),
    )
    prechallenge_contact = RH56SafetyFeedback(
        captured_monotonic_s=clock(),
        positions=initial.positions,
        angles=initial.angles,
        forces_g=(0, 0, 0, 0, 112, 0),
        currents_ma=(0,) * 6,
        errors=(0,) * 6,
        statuses=(1,) * 6,
        temperatures_c=(25,) * 6,
    )
    owner._cache_feedback(prechallenge_contact)
    owner._register_physical_tracking_target(
        (1000, 1000, 1000, 1000, 949, 1000),
        now_monotonic_s=clock(),
    )
    assert owner.physical_tracking_snapshot().active_challenge_axes[4] is True

    clock.advance(0.050)
    force_controlled_hold = RH56SafetyFeedback(
        captured_monotonic_s=clock(),
        positions=initial.positions,
        angles=initial.angles,
        forces_g=(0, 0, 0, 0, 224, 0),
        currents_ma=(0, 0, 0, 0, 50, 0),
        errors=(0,) * 6,
        statuses=(1,) * 6,
        temperatures_c=(25,) * 6,
    )
    owner._cache_feedback(force_controlled_hold)

    snapshot = owner.physical_tracking_snapshot()
    assert snapshot.verified_axes[4] is True
    assert snapshot.contact_verified_axes[4] is True
    assert snapshot.failed_axes[4] is False


def test_static_force_offset_without_force_change_does_not_hide_no_motion():
    owner, _factory, _transport, _ledger, _gate_value, clock = _owner(
        supervised_target_only=True,
        tracking_significant_gap_units=50,
        tracking_min_progress_units=3,
        tracking_timeout_s=0.080,
    )
    initial = RH56SafetyFeedback(
        captured_monotonic_s=clock(),
        positions=(100,) * 6,
        angles=(1000,) * 6,
        forces_g=(300,) * 6,
        currents_ma=(0,) * 6,
        errors=(0,) * 6,
        statuses=(1,) * 6,
        temperatures_c=(25,) * 6,
    )
    owner._cache_feedback(initial)
    owner._register_physical_tracking_target(
        (900, 1000, 1000, 1000, 1000, 1000),
        now_monotonic_s=clock(),
    )
    clock.advance(0.081)
    unchanged = RH56SafetyFeedback(
        captured_monotonic_s=clock(),
        positions=initial.positions,
        angles=initial.angles,
        forces_g=initial.forces_g,
        currents_ma=initial.currents_ma,
        errors=initial.errors,
        statuses=initial.statuses,
        temperatures_c=initial.temperatures_c,
    )
    with pytest.raises(RH56WatchdogOwnerError, match="made no.*progress"):
        owner._cache_feedback(unchanged)


def test_supervised_final_settle_holds_without_rewriting_until_progress():
    owner, _factory, transport, ledger, _gate_value, clock = _owner(
        callback=lambda _fault: None,
        supervised_target_only=True,
        feedback_period_s=0.020,
        feedback_hard_age_s=0.150,
        feedback_angles_script=[
            (1000,) * 6,
            (1000,) * 6,
            (1000,) * 6,
            (980,) * 6,
        ],
        # Deliberately shorter than the tracking window: the explicit final
        # settle must keep only the sample-hold heartbeat alive.
        supervised_inter_command_watchdog_timeout_s=0.050,
        tracking_significant_gap_units=40,
        tracking_min_progress_units=5,
        tracking_timeout_s=0.200,
    )
    owner.start()
    command = _command(ledger, clock, rh56_target=900)
    ledger.stage(command, now_monotonic_s=clock())
    assert ledger.acknowledge(
        "franka", sequence=command.sequence, now_monotonic_s=clock()
    ) is False
    assert owner.execute(command).numeric_write_performed is True
    numeric_writes_before_settle = [
        values for values in transport.writes if values != (-1,) * 6
    ]

    settle_deadline = owner.begin_final_tracking_settle()
    assert settle_deadline > clock()
    assert owner.final_tracking_settle_active is True
    with pytest.raises(RH56WatchdogOwnerError, match="already active"):
        owner.begin_final_tracking_settle()
    clock.advance(0.021)
    assert _wait_until(lambda: transport.feedback_calls >= 3)
    assert owner.physical_tracking_snapshot().verdict == "challenged_unverified"
    clock.advance(0.021)
    assert _wait_until(
        lambda: owner.physical_tracking_snapshot().verdict == "verified"
    )

    assert ledger.last_committed_sequence == 1
    assert [
        values for values in transport.writes if values != (-1,) * 6
    ] == numeric_writes_before_settle == [(900,) * 6]
    result = owner.stop_and_close()
    assert result.rh56_disabled_verified is True


def test_supervised_final_settle_preserves_no_motion_fault_deadline():
    callback_seen = threading.Event()
    owner, _factory, transport, ledger, _gate_value, clock = _owner(
        callback=lambda _fault: callback_seen.set(),
        supervised_target_only=True,
        feedback_period_s=0.020,
        feedback_hard_age_s=0.150,
        feedback_angles_script=[(1000,) * 6] * 20,
        supervised_inter_command_watchdog_timeout_s=0.050,
        tracking_significant_gap_units=40,
        tracking_min_progress_units=5,
        tracking_timeout_s=0.080,
    )
    owner.start()
    command = _command(ledger, clock, rh56_target=900)
    ledger.stage(command, now_monotonic_s=clock())
    assert ledger.acknowledge(
        "franka", sequence=command.sequence, now_monotonic_s=clock()
    ) is False
    assert owner.execute(command).numeric_write_performed is True
    owner.begin_final_tracking_settle()

    for _ in range(5):
        clock.advance(0.021)
        time.sleep(0.025)
        if callback_seen.is_set():
            break

    assert callback_seen.wait(0.5)
    result = owner.stop_and_close()
    assert result.fault is not None
    assert "physical tracking made no ANGLE_ACT/POS_ACT progress" in (
        result.fault.reason
    )
    assert "command heartbeat deadline" not in result.fault.reason
    assert result.rh56_disabled_verified is True
    # The settle itself performs no second target write.  The only later
    # numeric value is the measured-pose hold required by terminal stop.
    assert [
        values for values in transport.writes if values != (-1,) * 6
    ] == [(900,) * 6, (1000,) * 6]


def test_supervised_feedback_hard_age_faults_before_late_poll_can_hide_gap():
    callback_seen = threading.Event()
    owner, _factory, transport, _ledger, _gate_value, clock = _owner(
        callback=lambda _fault: callback_seen.set(),
        supervised_target_only=True,
        feedback_hard_age_s=0.045,
    )
    owner.start()
    command = _command(_ledger, clock)
    _ledger.stage(command, now_monotonic_s=clock())
    assert _ledger.acknowledge(
        "franka", sequence=command.sequence, now_monotonic_s=clock()
    ) is False
    assert owner.execute(command).sequence == 1
    clock.advance(0.046)

    assert callback_seen.wait(0.5)
    assert owner.physical_tracking_snapshot().verdict != "verified"
    result = owner.stop_and_close()
    assert result.fault is not None
    assert "feedback hard-age deadline expired" in result.fault.reason
    assert result.rh56_disabled_verified is True
    assert transport.writes[-2:] == [(-1,) * 6, (-1,) * 6]


def test_worker_owns_open_bootstrap_execute_disable_close_and_cache_is_read_only():
    callbacks = []
    owner, factory, transport, ledger, _gate_value, clock = _owner(
        callback=callbacks.append
    )
    main_ident = threading.get_ident()

    owner.start()
    assert owner.state is RH56WatchdogOwnerState.RUNNING
    assert owner.independent_watchdog_active is True
    assert owner.fault_callback_configured is True
    assert factory.calls == 1
    assert factory.thread_ids == [owner.owner_ident]
    assert owner.owner_ident != main_ident

    calls_before_snapshot = len(transport.event_log)
    bootstrap = owner.require_fresh_feedback_history(minimum_samples=2)
    assert len(bootstrap.samples) == 2
    assert bootstrap.fresh is True
    assert bootstrap.latest_age_s <= bootstrap.maximum_age_s
    assert bootstrap.samples[-1].captured_realtime_s == pytest.approx(
        1_750_000_000.0
        + bootstrap.samples[-1].captured_monotonic_s
    )
    assert len(transport.event_log) == calls_before_snapshot

    command = _command(ledger, clock)
    ledger.stage(command, now_monotonic_s=clock())
    assert not ledger.acknowledge(
        "franka", sequence=1, now_monotonic_s=clock()
    )
    receipt = owner.execute(command)
    assert receipt.sequence == 1
    assert ledger.last_committed_sequence == 1
    after = owner.require_fresh_feedback_history(minimum_samples=2)
    assert len(after.samples) == 3

    result = owner.stop_and_close()
    assert result.worker_stopped is True
    assert result.rh56_disabled_verified is True
    assert result.disable_attempted is True
    assert result.close_attempted is True
    assert result.serial_closed is True
    assert callbacks == []
    assert transport.writes[0] == (960,) * 6
    assert transport.writes[-2:] == [(-1,) * 6, (-1,) * 6]
    assert transport.event_log[-1][0] == "close"
    assert set(transport.transport_thread_ids) == {owner.owner_ident}


def test_fake_clock_watchdog_disables_while_policy_inference_is_blocked_forever():
    event_log = []
    inference_started = threading.Event()
    release_inference = threading.Event()
    callback_seen = threading.Event()
    franka_stop_requested = threading.Event()
    callback_faults = []
    callback_gate = []

    def blocked_inference():
        inference_started.set()
        release_inference.wait()

    owner_ref = {}

    def on_fault(fault):
        callback_faults.append(fault)
        event_log.append(("fault_callback", None, threading.get_ident()))
        callback_gate[0].latch_fault(f"RH56 owner: {fault.reason}")
        franka_stop_requested.set()
        callback_seen.set()

    owner, _factory, transport, _ledger, gate, clock = _owner(
        event_log=event_log,
        callback=on_fault,
    )
    owner_ref["owner"] = owner
    callback_gate.append(gate)
    owner.start()

    policy_thread = threading.Thread(target=blocked_inference, daemon=True)
    policy_thread.start()
    assert inference_started.wait(0.2)
    clock.advance(0.051)

    assert callback_seen.wait(0.5)
    assert owner.fault_event.wait(0.5)
    result = owner.stop_and_close()
    assert policy_thread.is_alive()
    assert franka_stop_requested.is_set()
    assert gate.state is SafetyState.FAULT_LATCHED
    assert result.rh56_disabled_verified is True
    assert result.serial_closed is True
    assert transport.writes[-2:] == [(-1,) * 6, (-1,) * 6]
    callback_index = next(
        index for index, event in enumerate(event_log) if event[0] == "fault_callback"
    )
    first_disable_index = next(
        index
        for index, event in enumerate(event_log)
        if event[0] == "write" and event[1] == (-1,) * 6
    )
    assert callback_index < first_disable_index
    assert callback_faults[0].reason.startswith("RH56 independent owner watchdog")

    release_inference.set()
    policy_thread.join(timeout=0.2)


def test_supervised_owner_honors_configured_125ms_inter_command_deadline():
    callback_seen = threading.Event()
    owner, factory, _transport, ledger, _gate_value, clock = _owner(
        callback=lambda _fault: callback_seen.set(),
        supervised_target_only=True,
        feedback_hard_age_s=0.075,
        supervised_inter_command_watchdog_timeout_s=0.125,
    )
    owner.start()
    command = _command(ledger, clock)
    ledger.stage(command, now_monotonic_s=clock())
    assert ledger.acknowledge("franka", sequence=1, now_monotonic_s=clock()) is False
    receipt = owner.execute(command)
    assert factory.actuator is not None
    assert factory.actuator.policy_watchdog_deadline_monotonic_s == pytest.approx(
        receipt.acknowledged_monotonic_s + 0.125
    )

    clock.advance(0.065)
    time.sleep(0.020)
    assert callback_seen.is_set() is False
    assert owner.state is RH56WatchdogOwnerState.RUNNING

    clock.advance(0.061)
    assert callback_seen.wait(0.5)
    result = owner.stop_and_close()
    assert result.fault is not None
    assert "command heartbeat deadline" in result.fault.reason
    assert result.rh56_disabled_verified is True


def test_watchdog_faults_ledger_when_franka_is_the_only_partial_ack():
    callback_seen = threading.Event()
    owner, _factory, transport, ledger, _gate_value, clock = _owner(
        callback=lambda _fault: callback_seen.set()
    )
    owner.start()
    command = _command(ledger, clock)
    ledger.stage(command, now_monotonic_s=clock())
    assert ledger.acknowledge(
        "franka", sequence=command.sequence, now_monotonic_s=clock()
    ) is False
    assert ledger.pending_sequence == 1

    # The main policy thread stalls before submit; the owner still owns the
    # prior heartbeat and can fail the exact partial sequence independently.
    clock.advance(0.051)
    assert callback_seen.wait(0.5)
    result = owner.stop_and_close()

    assert ledger.last_committed_sequence == 0
    assert ledger.pending_sequence == 1
    assert ledger.fault_reason is not None
    assert "rh56 failed sequence 1" in ledger.fault_reason
    assert result.fault is not None
    assert result.fault.pending_sequence == 1
    assert result.fault.stop_confirmed is True
    assert result.fault.stop_error is None
    assert result.rh56_disabled_verified is True
    assert transport.writes[-2:] == [(-1,) * 6, (-1,) * 6]


def test_stop_request_wakes_owner_with_static_fake_clock():
    owner, _factory, transport, _ledger, _gate_value, _clock = _owner(
        callback=lambda _fault: None
    )
    owner.start()
    started = time.monotonic()
    result = owner.stop_and_close(timeout_s=0.5)
    elapsed = time.monotonic() - started

    assert elapsed < 0.2
    assert result.rh56_disabled_verified is True
    assert transport.writes[-2:] == [(-1,) * 6, (-1,) * 6]


def test_stop_join_is_bounded_if_a_fake_serial_driver_violates_its_deadline():
    release_disable = threading.Event()
    owner, _factory, _transport, _ledger, _gate_value, _clock = _owner(
        callback=lambda _fault: None,
        block_disable=release_disable,
    )
    owner.start()

    started = time.monotonic()
    with pytest.raises(RH56WatchdogOwnerError, match="bounded join timeout"):
        owner.stop_and_close(timeout_s=0.02)
    assert time.monotonic() - started < 0.2

    release_disable.set()
    result = owner.stop_and_close(timeout_s=0.5)
    assert result.worker_stopped is True
    assert result.serial_closed is True


def test_actuator_exact_deadline_enforcement_is_owner_only_and_fail_closed():
    clock = _FakeClock()
    transport = _FakeTransport(clock)
    gate = _gate(_authorization())
    transport.open()
    actuator = RH56TransactionalActuator(
        transport,
        command_watchdog_timeout_s=0.05,
        stop_timeout_s=0.25,
        stop_verify_samples=2,
        stop_verify_interval_s=0.0,
        monotonic=clock,
        sleep=clock.advance,
    )
    actuator.claim_for_current_thread()
    actuator.arm(
        preflight=_preflight(),
        authorization=_authorization(),
        safety_gate=gate,
        run_id="owner-execution",
        now_monotonic_s=clock(),
    )
    deadline = actuator.policy_watchdog_deadline_monotonic_s
    assert deadline == pytest.approx(clock() + 0.05)
    assert actuator.enforce_policy_watchdog(
        now_monotonic_s=deadline - 1.0e-9
    ) is None
    report = actuator.enforce_policy_watchdog(now_monotonic_s=deadline)

    assert report is not None and report.verified is True
    assert actuator.state is RH56ActuatorState.FAULT_LATCHED
    assert transport.writes[-2:] == [(-1,) * 6, (-1,) * 6]
    transport.close()
