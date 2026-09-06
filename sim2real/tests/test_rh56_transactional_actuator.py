from __future__ import annotations

from collections import deque
import threading

import numpy as np
import pytest

from sim2real.closed_loop_core import (
    AUTHORIZATION_SCOPE,
    ClosedLoopCommand,
    ClosedLoopSafetyGate,
    ExecutedActionLedger,
    MotionAuthorization,
)
from robot_control.rh56.actuator import (
    RH56ActuationPreflight,
    RH56ActuatorState,
    RH56SafetyFeedback,
    RH56StopUnconfirmed,
    RH56TargetReceipt,
    RH56TransactionalActuator,
    RH56TransactionalError,
    RH56_SUPERVISED_EXCHANGE_TIMEOUT_S,
    RH56_STOP_RELEASE_SCHEDULING_MARGIN_S,
    RH56_SUPERVISED_WRITE_RESPONSE_GRACE_S,
    _issue_supervised_rh56_preflight,
    estimate_compact_transaction_wire_rate,
)
from sim2real.contracts.v94 import INITIAL_PREVIOUS_ACTION13


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


class _Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += float(seconds)


class _FakeTransport:
    """Deterministic in-memory transport; it never imports or opens serial."""

    hardware_backed = False
    baud_rate = 115200
    transport_name = "deterministic-fake-rh56"

    def __init__(
        self,
        clock: _Clock,
        *,
        write_duration_s: float = 0.001,
        read_target_duration_s: float = 0.001,
        feedback_duration_s: float = 0.001,
    ) -> None:
        self.clock = clock
        self.write_duration_s = float(write_duration_s)
        self.read_target_duration_s = float(read_target_duration_s)
        self.feedback_duration_s = float(feedback_duration_s)
        self.targets = (-1,) * 6
        self.calls: list[tuple[str, object]] = []
        self.write_behaviors: deque[object] = deque()
        self.read_target_behaviors: deque[object] = deque()
        self.feedback_behaviors: deque[object] = deque()
        self.write_deadlines: list[float] = []
        self.read_target_deadlines: list[float] = []
        self.feedback_deadlines: list[float] = []

    def _advance(self, duration: float, deadline: float, behavior: object) -> None:
        if behavior == "timeout":
            self.clock.value = max(self.clock.value, deadline + 0.001)
            raise TimeoutError("scripted transport timeout")
        self.clock.advance(duration)
        if self.clock.value > deadline:
            raise TimeoutError("transport operation exceeded deadline")
        if isinstance(behavior, BaseException):
            raise behavior

    @staticmethod
    def _next(queue: deque[object]) -> object:
        return queue.popleft() if queue else None

    def write_angle_set(self, values, *, deadline_monotonic_s):
        values = tuple(values)
        self.calls.append(("write", values))
        self.write_deadlines.append(float(deadline_monotonic_s))
        behavior = self._next(self.write_behaviors)
        if behavior == "apply_then_raise":
            self.clock.advance(self.write_duration_s)
            self.targets = values
            raise TimeoutError("scripted missing write response after apply")
        if behavior == "apply_then_timeout":
            self.clock.value = max(
                self.clock.value,
                float(deadline_monotonic_s),
            )
            self.targets = values
            raise TimeoutError(
                "scripted missing write response at the complete ACK deadline"
            )
        self._advance(self.write_duration_s, deadline_monotonic_s, behavior)
        self.targets = values

    def read_angle_set(self, *, deadline_monotonic_s):
        self.calls.append(("read_targets", None))
        self.read_target_deadlines.append(float(deadline_monotonic_s))
        behavior = self._next(self.read_target_behaviors)
        self._advance(self.read_target_duration_s, deadline_monotonic_s, behavior)
        return self.targets if behavior is None else behavior

    def _default_feedback(self) -> RH56SafetyFeedback:
        return RH56SafetyFeedback(
            captured_monotonic_s=self.clock(),
            positions=(500,) * 6,
            angles=(500,) * 6,
            forces_g=(0,) * 6,
            currents_ma=(0,) * 6,
            errors=(0,) * 6,
            statuses=(2,) * 6,
            temperatures_c=(25,) * 6,
        )

    def read_safety_feedback(self, *, deadline_monotonic_s):
        self.calls.append(("feedback", None))
        self.feedback_deadlines.append(float(deadline_monotonic_s))
        behavior = self._next(self.feedback_behaviors)
        self._advance(self.feedback_duration_s, deadline_monotonic_s, behavior)
        if behavior is None:
            return self._default_feedback()
        if callable(behavior):
            return behavior()
        return behavior


def _preflight_report(**overrides):
    metric_actual = {
        "evidence_metric_rh56_sustained_command_rate_hz": 60.0,
        "evidence_metric_dual_device_sustained_ack_rate_hz": 60.0,
        "evidence_metric_dual_device_max_interaction_gap_s": 1.0 / 30.0,
        "evidence_metric_policy_command_watchdog_timeout_s": 0.05,
    }
    report = {
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
        "run_id": "commissioning-run-1",
        "failed_checks": [],
        "blockers": [],
        "inputs": {
            "sha256": {
                "commissioning_profile_sha256": "a" * 64,
            }
        },
        "checks": [
            {
                "code": code,
                "passed": True,
                **(
                    {"actual": metric_actual[code]}
                    if code in metric_actual
                    else {}
                ),
            }
            for code in _REQUIRED_RH56_CHECKS
        ],
    }
    report.update(overrides)
    return report


def _authorization(*, authorization_id="operator-approval-1"):
    return MotionAuthorization(
        run_id="execution-run-1",
        authorization_id=authorization_id,
        issued_monotonic_s=9.0,
        expires_monotonic_s=20.0,
    )


def _armed_gate(authorization=None):
    authorization = _authorization() if authorization is None else authorization
    gate = ClosedLoopSafetyGate()
    gate.update_interlocks(deadman_asserted=True, estop_healthy=True)
    gate.arm(
        authorization,
        run_id="execution-run-1",
        now_monotonic_s=10.0,
        franka_rest_verified=True,
        rh56_disabled_verified=True,
    )
    return gate


def _actuator(clock=None, transport=None, **kwargs):
    clock = _Clock() if clock is None else clock
    transport = _FakeTransport(clock) if transport is None else transport
    actuator = RH56TransactionalActuator(
        transport,
        monotonic=clock,
        sleep=clock.advance,
        **kwargs,
    )
    actuator.claim_for_current_thread()
    return actuator, transport, clock


def _arm(actuator, clock, *, gate=None, authorization=None, preflight=None):
    authorization = _authorization() if authorization is None else authorization
    gate = _armed_gate(authorization) if gate is None else gate
    preflight = (
        RH56ActuationPreflight.from_report(_preflight_report())
        if preflight is None
        else preflight
    )
    actuator.arm(
        preflight=preflight,
        authorization=authorization,
        safety_gate=gate,
        run_id="execution-run-1",
        now_monotonic_s=clock(),
    )
    return gate


def _arm_supervised(
    actuator,
    clock,
    *,
    gate=None,
    authorization=None,
    inter_command_watchdog_timeout_s=None,
):
    authorization = _authorization() if authorization is None else authorization
    gate = _armed_gate(authorization) if gate is None else gate
    actuator.arm_supervised(
        preflight=_issue_supervised_rh56_preflight(
            run_id="execution-run-1",
            confirmed_permit_sha256="b" * 64,
            commissioning_profile_sha256="a" * 64,
            watchdog_timeout_s=0.05,
            inter_command_watchdog_timeout_s=(
                inter_command_watchdog_timeout_s
            ),
        ),
        authorization=authorization,
        safety_gate=gate,
        run_id="execution-run-1",
        now_monotonic_s=clock(),
    )
    return gate


def _command(sequence, produced, previous, value=0.25):
    action = np.full(13, value, dtype=np.float32)
    return ClosedLoopCommand(
        sequence=sequence,
        produced_monotonic_s=produced,
        observation_realtime_s=1_750_000_000.0 + produced,
        previous_executed_action13_used=previous,
        raw_policy_action13=action,
        executed_policy_action13=action,
        franka_target_q_rad=np.full(7, 0.1),
        rh56_angle_set_register_order=np.asarray([990, 980, 970, 960, 950, 940]),
    )


def _stage(ledger, clock, sequence=1, value=0.25):
    command = _command(
        sequence,
        clock(),
        ledger.previous_executed_action13(),
        value=value,
    )
    ledger.stage(command, now_monotonic_s=clock())
    return command


def test_preflight_permit_requires_exact_disarmed_offline_pass():
    permit = RH56ActuationPreflight.from_report(_preflight_report())
    assert permit.commissioning_run_id == "commissioning-run-1"
    assert len(permit.report_sha256) == 64
    with pytest.raises(TypeError, match="must be created from a verified report"):
        RH56ActuationPreflight(
            commissioning_run_id="forged",
            report_sha256="0" * 64,
            commissioning_profile_sha256="0" * 64,
            verified_command_watchdog_timeout_s=0.05,
            verified_rh56_command_rate_hz=60.0,
            verified_dual_ack_rate_hz=60.0,
            verified_dual_ack_max_gap_s=1.0 / 30.0,
        )

    for edit in (
        {"result": "FAIL"},
        {"physical_motion_authorized": True},
        {"device_access": True},
        {"failed_checks": ["rh56_60hz_write_readback"]},
        {"run_id": ""},
    ):
        with pytest.raises(ValueError, match="not eligible"):
            RH56ActuationPreflight.from_report(_preflight_report(**edit))

    missing_check = _preflight_report()
    missing_check["checks"] = missing_check["checks"][:-1]
    with pytest.raises(ValueError, match="policy_command_watchdog"):
        RH56ActuationPreflight.from_report(missing_check)

    weak_rate = _preflight_report()
    next(
        item
        for item in weak_rate["checks"]
        if item["code"] == "evidence_metric_rh56_sustained_command_rate_hz"
    )["actual"] = 59.9
    with pytest.raises(ValueError, match="sustained_command_rate"):
        RH56ActuationPreflight.from_report(weak_rate)


def test_default_disarmed_refuses_transport_and_does_not_fail_ledger():
    actuator, transport, clock = _actuator()
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.05)
    command = _stage(ledger, clock)

    with pytest.raises(RH56TransactionalError, match="disarmed"):
        actuator.execute(command, action_ledger=ledger)
    assert transport.calls == []
    assert actuator.state is RH56ActuatorState.DISARMED
    assert ledger.fault_reason is None


def test_arm_requires_same_live_authorization_as_safety_gate():
    actuator, transport, clock = _actuator()
    gate_authorization = _authorization(authorization_id="gate-authorization")
    gate = _armed_gate(gate_authorization)
    other_authorization = _authorization(authorization_id="different-authorization")

    with pytest.raises(RH56TransactionalError, match="does not match"):
        _arm(
            actuator,
            clock,
            gate=gate,
            authorization=other_authorization,
        )
    assert actuator.state is RH56ActuatorState.DISARMED
    assert transport.calls == []


def test_arm_requires_exact_preflight_watchdog_binding():
    actuator, transport, clock = _actuator(command_watchdog_timeout_s=0.04)
    with pytest.raises(RH56TransactionalError, match="watchdog differs"):
        _arm(actuator, clock)
    assert actuator.state is RH56ActuatorState.DISARMED
    assert transport.calls == []


def test_success_acknowledges_only_after_exact_readback_and_feedback():
    actuator, transport, clock = _actuator()
    _arm(actuator, clock)
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.05)
    command = _stage(ledger, clock)

    receipt = actuator.execute(command, action_ledger=ledger)
    assert receipt.dual_device_commit_completed is False
    assert ledger.last_committed_sequence == 0
    np.testing.assert_array_equal(
        ledger.previous_executed_action13(), INITIAL_PREVIOUS_ACTION13
    )
    assert [name for name, _ in transport.calls] == [
        "write",
        "read_targets",
        "feedback",
    ]
    assert receipt.exact_readback == tuple(
        command.rh56_angle_set_register_order.tolist()
    )
    assert ledger.acknowledge("franka", sequence=1, now_monotonic_s=clock())
    assert ledger.last_committed_sequence == 1


def test_success_completes_dual_ack_when_franka_already_acknowledged():
    actuator, _transport, clock = _actuator()
    _arm(actuator, clock)
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.05)
    command = _stage(ledger, clock)
    assert not ledger.acknowledge("franka", sequence=1, now_monotonic_s=clock())

    receipt = actuator.execute(command, action_ledger=ledger)
    assert receipt.dual_device_commit_completed is True
    assert ledger.last_committed_sequence == 1
    np.testing.assert_array_equal(
        ledger.previous_executed_action13(), command.executed_policy_action13
    )


def test_supervised_target_transaction_uses_single_acknowledged_write():
    actuator, transport, clock = _actuator()
    _arm_supervised(actuator, clock)
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.05)
    command = _stage(ledger, clock)
    assert ledger.acknowledge(
        "franka", sequence=command.sequence, now_monotonic_s=clock()
    ) is False

    receipt = actuator.execute_target_and_ack(command, action_ledger=ledger)

    assert isinstance(receipt, RH56TargetReceipt)
    assert receipt.sequence == 1
    assert receipt.exact_target == tuple(
        command.rh56_angle_set_register_order.tolist()
    )
    assert receipt.exact_readback == receipt.exact_target
    assert receipt.write_response_received is True
    assert receipt.application_verified_by_readback is False
    assert receipt.dual_device_commit_completed is True
    assert ledger.last_committed_sequence == 1
    assert [name for name, _value in transport.calls] == ["write"]


def test_supervised_inter_command_watchdog_is_distinct_from_command_freshness():
    actuator, _transport, clock = _actuator(
        command_watchdog_timeout_s=0.05,
        supervised_inter_command_watchdog_timeout_s=0.500,
    )
    _arm_supervised(
        actuator,
        clock,
        inter_command_watchdog_timeout_s=0.500,
    )
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.05)
    first = _stage(ledger, clock)
    assert ledger.acknowledge("franka", sequence=1, now_monotonic_s=clock()) is False
    first_receipt = actuator.execute_target_and_ack(first, action_ledger=ledger)
    first_ack = first_receipt.acknowledged_monotonic_s

    assert actuator.policy_watchdog_deadline_monotonic_s == pytest.approx(
        first_ack + 0.500
    )
    # Regress the real 126 ms observation/USB scheduling gap: a newly
    # timestamped command remains admissible because inter-command sample
    # hold is separate from new-command freshness.
    clock.advance(0.126)
    second = _stage(ledger, clock, sequence=2)
    assert ledger.acknowledge("franka", sequence=2, now_monotonic_s=clock()) is False
    assert actuator.execute_target_and_ack(second, action_ledger=ledger).sequence == 2


def test_supervised_new_command_still_expires_fifty_ms_after_production():
    actuator, _transport, clock = _actuator(
        command_watchdog_timeout_s=0.05,
        supervised_inter_command_watchdog_timeout_s=0.500,
    )
    _arm_supervised(
        actuator,
        clock,
        inter_command_watchdog_timeout_s=0.500,
    )
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.05)
    first = _stage(ledger, clock)
    assert ledger.acknowledge("franka", sequence=1, now_monotonic_s=clock()) is False
    actuator.execute_target_and_ack(first, action_ledger=ledger)

    second = _stage(ledger, clock, sequence=2)
    assert ledger.acknowledge("franka", sequence=2, now_monotonic_s=clock()) is False
    clock.advance(0.051)
    with pytest.raises(RH56TransactionalError, match="missed deadline"):
        actuator.execute_target_and_ack(second, action_ledger=ledger)


def test_supervised_inter_command_watchdog_must_match_sealed_preflight():
    actuator, _transport, clock = _actuator(
        command_watchdog_timeout_s=0.05,
        supervised_inter_command_watchdog_timeout_s=0.500,
    )
    with pytest.raises(RH56TransactionalError, match="inter-command watchdog"):
        _arm_supervised(
            actuator,
            clock,
            inter_command_watchdog_timeout_s=0.100,
        )


def test_supervised_inter_command_watchdog_rejects_above_bounded_hold_ceiling():
    with pytest.raises(ValueError, match="exceeds 0.500s"):
        _actuator(
            command_watchdog_timeout_s=0.050,
            supervised_inter_command_watchdog_timeout_s=0.501,
        )


def test_formal_arm_cannot_use_supervised_target_transaction():
    actuator, transport, clock = _actuator()
    _arm(actuator, clock)
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.05)
    command = _stage(ledger, clock)

    with pytest.raises(RH56TransactionalError, match="requires arm_supervised") as raised:
        actuator.execute_target_and_ack(command, action_ledger=ledger)

    message = str(raised.value)
    assert "sequence=1" in message
    # NumPy 2 renders scalar tuple members as ``np.int32(...)`` while the
    # production error deliberately normalizes register values to plain
    # Python integers.  Compare the normalized wire values on every NumPy
    # version.
    exact_target = tuple(
        int(value) for value in command.rh56_angle_set_register_order
    )
    assert f"exact_target={exact_target}" in message
    assert "phase=PRECHECK" in message
    assert "APPLY_UNKNOWN=false" in message
    assert "RETRY_FORBIDDEN" in message
    assert transport.calls == []
    assert ledger.fault_reason is None
    assert actuator.state is RH56ActuatorState.ARMED

    # The formal method and its feedback exchange remain unchanged.
    actuator.execute(command, action_ledger=ledger)
    assert [name for name, _value in transport.calls] == [
        "write",
        "read_targets",
        "feedback",
    ]


@pytest.mark.parametrize(
    ("write_duration_s", "read_duration_s", "failure_phase"),
    (
        (RH56_SUPERVISED_WRITE_RESPONSE_GRACE_S, 0.001, None),
        (0.001, RH56_SUPERVISED_EXCHANGE_TIMEOUT_S - 0.002, None),
        (
            RH56_SUPERVISED_WRITE_RESPONSE_GRACE_S + 0.000001,
            0.001,
            "ANGLE_SET_EXACT_READBACK",
        ),
    ),
)
def test_supervised_target_has_bounded_write_grace_and_original_command_deadline(
    write_duration_s, read_duration_s, failure_phase
):
    clock = _Clock()
    transport = _FakeTransport(
        clock,
        write_duration_s=write_duration_s,
        read_target_duration_s=read_duration_s,
    )
    actuator, _transport, _clock = _actuator(clock, transport)
    _arm_supervised(actuator, clock)
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.05)
    command = _stage(ledger, clock)

    if failure_phase is None:
        receipt = actuator.execute_target_and_ack(command, action_ledger=ledger)
        assert receipt.exact_readback == receipt.exact_target
        assert receipt.application_verified_by_readback is False
        assert [name for name, _value in transport.calls] == ["write"]
        assert "feedback" not in [name for name, _value in transport.calls]
        return

    with pytest.raises(RH56TransactionalError) as raised:
        actuator.execute_target_and_ack(command, action_ledger=ledger)
    message = str(raised.value)
    assert f"phase={failure_phase}" in message
    assert "APPLY_UNKNOWN=true" in message
    assert "RETRY_FORBIDDEN" in message
    assert actuator.state is RH56ActuatorState.FAULT_LATCHED
    assert actuator.stop_confirmed is False
    target = tuple(command.rh56_angle_set_register_order.tolist())
    assert [
        value for name, value in transport.calls if name == "write" and value == target
    ] == [target]
    assert "feedback" not in [name for name, _value in transport.calls]


def test_missing_write_response_can_be_verified_by_one_exact_readback():
    actuator, transport, clock = _actuator()
    _arm_supervised(actuator, clock)
    transport.write_behaviors.append("apply_then_raise")
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.05)
    command = _stage(ledger, clock)
    target = tuple(command.rh56_angle_set_register_order.tolist())
    assert ledger.acknowledge(
        "franka", sequence=command.sequence, now_monotonic_s=clock()
    ) is False

    receipt = actuator.execute_target_and_ack(command, action_ledger=ledger)

    assert receipt.write_response_received is False
    assert receipt.application_verified_by_readback is True
    assert receipt.exact_target == target
    assert receipt.exact_readback == target
    assert receipt.dual_device_commit_completed is True
    assert ledger.last_committed_sequence == 1
    assert [name for name, _value in transport.calls] == ["write", "read_targets"]
    assert [
        value for name, value in transport.calls if name == "write" and value == target
    ] == [target]


def test_missing_write_response_without_application_is_apply_unknown_no_retry():
    actuator, transport, clock = _actuator()
    _arm_supervised(actuator, clock)
    transport.write_behaviors.append("timeout")
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.05)
    command = _stage(ledger, clock)
    target = tuple(command.rh56_angle_set_register_order.tolist())

    with pytest.raises(RH56TransactionalError) as raised:
        actuator.execute_target_and_ack(command, action_ledger=ledger)

    message = str(raised.value)
    assert "phase=ANGLE_SET_EXACT_READBACK" in message
    assert "APPLY_UNKNOWN=true" in message
    assert f"exact_target={target}" in message
    assert "RETRY_FORBIDDEN" in message
    assert [name for name, _value in transport.calls] == ["write", "read_targets"]
    assert [value for name, value in transport.calls if name == "write"] == [target]
    assert ledger.last_committed_sequence == 0
    assert actuator.state is RH56ActuatorState.FAULT_LATCHED
    assert actuator.stop_confirmed is False


def test_missing_write_response_and_failed_readback_is_apply_unknown_no_retry():
    actuator, transport, clock = _actuator()
    _arm_supervised(actuator, clock)
    transport.write_behaviors.append("apply_then_raise")
    transport.read_target_behaviors.append("timeout")
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.05)
    command = _stage(ledger, clock)
    target = tuple(command.rh56_angle_set_register_order.tolist())

    with pytest.raises(RH56TransactionalError) as raised:
        actuator.execute_target_and_ack(command, action_ledger=ledger)

    message = str(raised.value)
    assert "write response missing" in message
    assert "exact verification readback failed" in message
    assert "phase=ANGLE_SET_EXACT_READBACK" in message
    assert "APPLY_UNKNOWN=true" in message
    assert "RETRY_FORBIDDEN" in message
    assert [name for name, _value in transport.calls] == ["write", "read_targets"]
    assert [value for name, value in transport.calls if name == "write"] == [target]
    assert ledger.last_committed_sequence == 0
    assert actuator.state is RH56ActuatorState.FAULT_LATCHED
    assert actuator.stop_confirmed is False


def test_supervised_missing_ack_wrong_readback_is_terminal_and_never_retries_target():
    actuator, transport, clock = _actuator()
    _arm_supervised(actuator, clock)
    transport.write_behaviors.append("apply_then_raise")
    transport.read_target_behaviors.append((1, 2, 3, 4, 5, 6))
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.05)
    command = _stage(ledger, clock)
    target = tuple(command.rh56_angle_set_register_order.tolist())

    with pytest.raises(RH56TransactionalError) as raised:
        actuator.execute_target_and_ack(command, action_ledger=ledger)

    message = str(raised.value)
    assert "sequence=1" in message
    assert f"exact_target={target}" in message
    assert "phase=ANGLE_SET_EXACT_READBACK" in message
    assert "APPLY_UNKNOWN=true" in message
    assert "RETRY_FORBIDDEN" in message
    assert [value for name, value in transport.calls if name == "write"] == [target]
    assert [name for name, _value in transport.calls] == ["write", "read_targets"]
    assert ledger.last_committed_sequence == 0
    assert ledger.fault_reason is not None
    assert actuator.stop_confirmed is False

    # The lifecycle owner performs this only after publishing the fault edge.
    report = actuator.fault_and_disable("owner observed target transaction fault")
    assert report.verified is True
    assert actuator.stop_confirmed is True


def test_poll_safety_can_defer_cleanup_for_supervised_owner_fault_ordering():
    actuator, transport, clock = _actuator()
    _arm_supervised(actuator, clock)
    transport.feedback_behaviors.append(OSError("synthetic feedback failure"))

    with pytest.raises(RH56TransactionalError, match="synthetic feedback failure"):
        actuator.poll_safety(defer_failure_cleanup=True)

    assert actuator.state is RH56ActuatorState.FAULT_LATCHED
    assert actuator.stop_confirmed is False
    assert [name for name, _value in transport.calls] == ["feedback"]

    report = actuator.fault_and_disable("owner published fault and stopped Franka")
    assert report.verified is True
    assert actuator.stop_confirmed is True


def test_supervised_pre_first_feedback_uses_bounded_freshness_budget_without_write():
    actuator, transport, clock = _actuator(maximum_feedback_age_s=0.075)
    _arm_supervised(actuator, clock)
    started = clock()

    feedback = actuator.poll_safety(defer_failure_cleanup=True)

    assert feedback.statuses == (2,) * 6
    assert transport.feedback_deadlines == pytest.approx([started + 0.075])
    assert not any(name == "write" for name, _value in transport.calls)


def test_poll_safety_default_still_performs_synchronous_cleanup():
    actuator, transport, clock = _actuator()
    _arm(actuator, clock)
    transport.feedback_behaviors.append(OSError("synthetic formal poll failure"))

    with pytest.raises(RH56TransactionalError, match="synthetic formal poll failure"):
        actuator.poll_safety()

    assert actuator.state is RH56ActuatorState.FAULT_LATCHED
    assert actuator.stop_confirmed is True
    writes = [value for name, value in transport.calls if name == "write"]
    assert writes[-2:] == [(-1,) * 6, (-1,) * 6]


def test_wrong_readback_latches_ledger_and_runs_verified_stop():
    actuator, transport, clock = _actuator()
    _arm(actuator, clock)
    transport.read_target_behaviors.append((1, 2, 3, 4, 5, 6))
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.05)
    command = _stage(ledger, clock)

    with pytest.raises(RH56TransactionalError, match="readback mismatch"):
        actuator.execute(command, action_ledger=ledger)
    assert actuator.state is RH56ActuatorState.FAULT_LATCHED
    assert actuator.stop_confirmed is True
    assert "readback mismatch" in ledger.fault_reason
    disable_writes = [
        value
        for name, value in transport.calls
        if name == "write" and value == (-1,) * 6
    ]
    assert len(disable_writes) == 2


def test_partial_readback_is_transport_fault_and_never_acknowledges():
    actuator, transport, clock = _actuator()
    _arm(actuator, clock)
    transport.read_target_behaviors.append((1, 2, 3))
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.05)
    command = _stage(ledger, clock)

    with pytest.raises(RH56TransactionalError, match="expected six"):
        actuator.execute(command, action_ledger=ledger)
    assert ledger.last_committed_sequence == 0
    assert actuator.state is RH56ActuatorState.FAULT_LATCHED
    assert actuator.stop_confirmed is True


def test_transport_timeout_latches_and_attempts_both_disable_passes():
    actuator, transport, clock = _actuator()
    _arm(actuator, clock)
    transport.write_behaviors.append("timeout")
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.05)
    command = _stage(ledger, clock)

    with pytest.raises(RH56TransactionalError, match="timeout"):
        actuator.execute(command, action_ledger=ledger)
    writes = [value for name, value in transport.calls if name == "write"]
    assert writes[0] == tuple(command.rh56_angle_set_register_order.tolist())
    assert writes[1:] == [(500,) * 6, (-1,) * 6, (-1,) * 6]
    assert actuator.state is RH56ActuatorState.FAULT_LATCHED


def test_stale_sequence_never_writes_numeric_target_and_is_terminal():
    actuator, transport, clock = _actuator()
    _arm(actuator, clock)
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.05)
    first = _stage(ledger, clock, sequence=1)
    ledger.acknowledge("franka", sequence=1, now_monotonic_s=clock())
    actuator.execute(first, action_ledger=ledger)
    second = _stage(ledger, clock, sequence=2, value=0.5)
    call_count = len(transport.calls)

    with pytest.raises(RH56TransactionalError, match="stale/out-of-order"):
        actuator.execute(first, action_ledger=ledger)
    new_writes = [
        value for name, value in transport.calls[call_count:] if name == "write"
    ]
    assert new_writes == [(500,) * 6, (-1,) * 6, (-1,) * 6]
    assert tuple(first.rh56_angle_set_register_order.tolist()) not in new_writes
    assert ledger.pending_sequence == second.sequence
    assert ledger.fault_reason is not None


@pytest.mark.parametrize(
    ("feedback", "message"),
    [
        (
            dict(errors=(1, 0, 0, 0, 0, 0)),
            "ERROR is nonzero",
        ),
        (
            dict(statuses=(2, 2, 2, 2, 2, 7)),
            "fault status 7",
        ),
        (
            dict(currents_ma=(0, 0, 0, 1500, 0, 0)),
            "running current exceeded",
        ),
        (
            dict(temperatures_c=(25, 25, 25, 25, 60, 25)),
            "temperature reached",
        ),
    ],
)
def test_feedback_safety_faults_are_latched_before_ack(feedback, message):
    actuator, transport, clock = _actuator()
    _arm(actuator, clock)

    def bad_feedback():
        values = dict(
            captured_monotonic_s=clock(),
            positions=(500,) * 6,
            angles=(500,) * 6,
            forces_g=(0,) * 6,
            currents_ma=(0,) * 6,
            errors=(0,) * 6,
            statuses=(2,) * 6,
            temperatures_c=(25,) * 6,
        )
        values.update(feedback)
        return RH56SafetyFeedback(**values)

    transport.feedback_behaviors.append(bad_feedback)
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.05)
    command = _stage(ledger, clock)
    with pytest.raises(RH56TransactionalError, match=message):
        actuator.execute(command, action_ledger=ledger)
    assert ledger.last_committed_sequence == 0
    assert actuator.stop_confirmed is True


def test_stale_feedback_timestamp_is_rejected_and_stopped():
    actuator, transport, clock = _actuator(maximum_feedback_age_s=0.01)
    _arm(actuator, clock)

    def stale_feedback():
        return RH56SafetyFeedback(
            captured_monotonic_s=clock() - 0.02,
            positions=(500,) * 6,
            angles=(500,) * 6,
            forces_g=(0,) * 6,
            currents_ma=(0,) * 6,
            errors=(0,) * 6,
            statuses=(2,) * 6,
            temperatures_c=(25,) * 6,
        )

    transport.feedback_behaviors.append(stale_feedback)
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.05)
    command = _stage(ledger, clock)
    with pytest.raises(RH56TransactionalError, match="feedback is stale"):
        actuator.execute(command, action_ledger=ledger)
    assert actuator.state is RH56ActuatorState.FAULT_LATCHED


def test_policy_watchdog_expiry_latches_and_disables():
    actuator, transport, clock = _actuator(command_watchdog_timeout_s=0.05)
    _arm(actuator, clock)
    clock.advance(0.051)

    with pytest.raises(RH56TransactionalError, match="watchdog expired"):
        actuator.poll_safety()
    writes = [value for name, value in transport.calls if name == "write"]
    assert writes == [(500,) * 6, (-1,) * 6, (-1,) * 6]
    assert actuator.state is RH56ActuatorState.FAULT_LATCHED


def test_feedback_poll_cannot_extend_missing_policy_command_watchdog():
    actuator, transport, clock = _actuator(command_watchdog_timeout_s=0.05)
    _arm(actuator, clock)
    clock.advance(0.020)
    actuator.poll_safety()
    clock.advance(0.030)

    with pytest.raises(RH56TransactionalError, match="policy command watchdog"):
        actuator.poll_safety()
    writes = [value for name, value in transport.calls if name == "write"]
    assert writes == [(500,) * 6, (-1,) * 6, (-1,) * 6]
    assert actuator.state is RH56ActuatorState.FAULT_LATCHED


def test_normal_shutdown_holds_measured_angle_then_disables_after_stable_tail():
    actuator, transport, clock = _actuator()
    _arm(actuator, clock)
    started = clock()

    report = actuator.disable_and_verify()
    assert report.verified is True
    assert report.disable_passes_verified == 2
    assert report.feedback_samples_verified == 3
    assert actuator.state is RH56ActuatorState.DISARMED
    assert actuator.stop_confirmed is True
    assert report.completed_monotonic_s - started >= 0.200
    writes = [value for name, value in transport.calls if name == "write"]
    assert writes == [(500,) * 6, (-1,) * 6, (-1,) * 6]
    # seed + three in-place-hold samples + three wholly post-disable samples
    assert [name for name, _ in transport.calls].count("feedback") == 7
    assert transport.targets == (-1,) * 6


def test_five_second_stop_reserves_complete_bounded_release_proof_tail():
    actuator, transport, clock = _actuator(
        stop_timeout_s=5.0,
        stop_verify_samples=3,
        stop_verify_interval_s=0.02,
    )
    _arm(actuator, clock)
    started = clock()

    report = actuator.disable_and_verify()

    assert report.verified is True
    expected_release_reserve_s = (
        2.0
        * (
            RH56_SUPERVISED_WRITE_RESPONSE_GRACE_S
            + RH56_SUPERVISED_EXCHANGE_TIMEOUT_S
        )
        + 3.0 * 3.0 * RH56_SUPERVISED_EXCHANGE_TIMEOUT_S
        + 2.0 * 0.02
        + RH56_STOP_RELEASE_SCHEDULING_MARGIN_S
    )
    assert expected_release_reserve_s == pytest.approx(0.730)
    settle_deadline = started + 5.0 - expected_release_reserve_s
    overall_deadline = started + 5.0

    # Seed/hold and all rolling-settle reads are forbidden from borrowing the
    # release reserve.  Both disable passes and the wholly fresh final proof
    # retain the overall stop deadline.
    assert transport.read_target_deadlines[:4] == pytest.approx(
        [settle_deadline] * 4
    )
    assert transport.feedback_deadlines[:4] == pytest.approx(
        [settle_deadline] * 4
    )
    assert transport.read_target_deadlines[4:] == pytest.approx(
        [overall_deadline] * 5
    )
    assert transport.feedback_deadlines[4:] == pytest.approx(
        [overall_deadline] * 3
    )


def test_stop_missing_write_acks_are_verified_without_resending_each_pass():
    actuator, transport, clock = _actuator()
    _arm(actuator, clock)
    transport.write_behaviors.extend(["apply_then_raise"] * 3)

    report = actuator.disable_and_verify()

    assert report.verified is True
    writes = [value for name, value in transport.calls if name == "write"]
    assert writes == [(500,) * 6, (-1,) * 6, (-1,) * 6]
    assert transport.targets == (-1,) * 6
    assert actuator.stop_confirmed is True


def test_stop_full_write_ack_timeout_uses_readback_without_numeric_resend():
    actuator, transport, clock = _actuator(
        stop_timeout_s=5.0,
        stop_verify_interval_s=0.02,
    )
    _arm(actuator, clock)
    started = clock()
    transport.write_behaviors.append("apply_then_timeout")

    report = actuator.disable_and_verify()

    assert report.verified is True
    writes = [value for name, value in transport.calls if name == "write"]
    assert writes == [(500,) * 6, (-1,) * 6, (-1,) * 6]
    # Seed feedback takes 1 ms, then the missing hold ACK consumes exactly its
    # independent 20 ms grace.  The next operation is the read-only proof, not
    # another numeric hold write.
    assert transport.write_deadlines[0] == pytest.approx(
        started + 0.001 + RH56_SUPERVISED_WRITE_RESPONSE_GRACE_S
    )
    assert transport.calls[:3] == [
        ("feedback", None),
        ("write", (500,) * 6),
        ("read_targets", None),
    ]
    assert transport.targets == (-1,) * 6


def test_stop_hold_write_timeout_is_short_and_both_disable_passes_remain():
    actuator, transport, clock = _actuator(
        stop_timeout_s=0.5,
        stop_verify_interval_s=0.02,
    )
    _arm(actuator, clock)
    started = clock()
    # The hold request may or may not have reached the hand.  It must not be
    # retransmitted, and its short ACK timeout must leave time for both
    # explicit -1 disable passes.
    transport.write_behaviors.append("timeout")

    with pytest.raises(RH56StopUnconfirmed, match="in-place hold/settle"):
        actuator.disable_and_verify()

    writes = [value for name, value in transport.calls if name == "write"]
    assert writes == [(500,) * 6, (-1,) * 6, (-1,) * 6]
    assert clock() - started < 0.1
    assert transport.targets == (-1,) * 6


def test_bad_stop_seed_feedback_still_attempts_both_disable_passes():
    actuator, transport, clock = _actuator(
        stop_timeout_s=0.5,
        stop_verify_interval_s=0.02,
    )
    _arm(actuator, clock)
    started = clock()
    transport.feedback_behaviors.append(
        OSError("synthetic stop seed feedback failure")
    )

    with pytest.raises(RH56StopUnconfirmed, match="stop seed feedback failure"):
        actuator.disable_and_verify()

    writes = [value for name, value in transport.calls if name == "write"]
    assert writes == [(-1,) * 6, (-1,) * 6]
    assert clock() - started < 0.1
    assert transport.targets == (-1,) * 6


def test_disarmed_shutdown_cannot_be_used_as_write_bypass():
    actuator, transport, _clock = _actuator()
    with pytest.raises(RH56TransactionalError, match="unauthorised disable"):
        actuator.disable_and_verify()
    assert transport.calls == []


def test_stop_failure_is_sticky_even_when_second_disable_pass_succeeds():
    actuator, transport, clock = _actuator()
    _arm(actuator, clock)
    # hold readback + three rolling samples, then fail disable pass 1 only
    transport.read_target_behaviors.extend(
        [None, None, None, None, (0,) * 6]
    )

    with pytest.raises(RH56StopUnconfirmed, match="disable pass 1"):
        actuator.disable_and_verify()
    writes = [value for name, value in transport.calls if name == "write"]
    assert writes == [(500,) * 6, (-1,) * 6, (-1,) * 6]
    assert actuator.state is RH56ActuatorState.FAULT_LATCHED
    assert actuator.stop_confirmed is False


def test_stop_rolling_tail_waits_through_transient_nonidle_status():
    actuator, transport, clock = _actuator(
        stop_timeout_s=0.8,
        stop_verify_interval_s=0.05,
    )
    _arm(actuator, clock)

    samples = (
        (500, 1),  # hold seed: motion is allowed before the hold is committed
        (500, 1),
        (500, 1),
        (500, 2),
        (500, 2),
        (500, 2),
    )
    for value, status in samples:
        transport.feedback_behaviors.append(
            lambda value=value, status=status: RH56SafetyFeedback(
                captured_monotonic_s=clock(),
                positions=(value,) * 6,
                angles=(value,) * 6,
                forces_g=(0,) * 6,
                currents_ma=(0,) * 6,
                errors=(0,) * 6,
                statuses=(status,) * 6,
                temperatures_c=(25,) * 6,
            )
        )

    report = actuator.disable_and_verify()
    assert report.verified is True
    writes = [value for name, value in transport.calls if name == "write"]
    assert writes == [(500,) * 6, (-1,) * 6, (-1,) * 6]
    assert [name for name, _ in transport.calls].count("feedback") == 7
    assert actuator.stop_confirmed is True


def test_stop_allows_running_current_during_hold_but_requires_idle_post_disable():
    actuator, transport, clock = _actuator(
        maximum_running_axis_current_ma=400,
        stop_max_axis_current_ma=100,
    )
    _arm(actuator, clock)

    def feedback(*, current, status):
        return lambda: RH56SafetyFeedback(
            captured_monotonic_s=clock(),
            positions=(500,) * 6,
            angles=(500,) * 6,
            forces_g=(0,) * 6,
            currents_ma=(current,) * 6,
            errors=(0,) * 6,
            statuses=(status,) * 6,
            temperatures_c=(25,) * 6,
        )

    # The 133 mA seed/hold samples are legitimate in-place braking under the
    # 400 mA running cap.  Only the three samples captured after both -1
    # readbacks must satisfy the strict 100 mA idle cap.
    transport.feedback_behaviors.extend(
        [feedback(current=133, status=1)] * 4
        + [feedback(current=0, status=2)] * 3
    )

    report = actuator.disable_and_verify()
    assert report.verified is True
    assert report.feedback_samples_verified == 3
    assert [name for name, _ in transport.calls].count("feedback") == 7
    assert transport.targets == (-1,) * 6


def test_stop_has_separate_transient_current_cap_without_weakening_running_cap():
    actuator, transport, clock = _actuator(
        maximum_running_axis_current_ma=400,
        stop_settle_max_axis_current_ma=800,
        stop_max_axis_current_ma=100,
    )
    _arm(actuator, clock)

    def feedback(*, currents, status):
        return lambda: RH56SafetyFeedback(
            captured_monotonic_s=clock(),
            positions=(500,) * 6,
            angles=(500,) * 6,
            forces_g=(0,) * 6,
            currents_ma=currents,
            errors=(0,) * 6,
            statuses=(status,) * 6,
            temperatures_c=(25,) * 6,
        )

    # Replay the peak pattern from the two completed hardware runs.  It is
    # accepted only after the stop path has taken ownership; active command
    # feedback remains independently bounded to 400 mA.  The entirely fresh
    # final proof still has to be idle and below 100 mA.
    stop_transient = (24, 256, 229, 64, 429, 0)
    transport.feedback_behaviors.extend(
        [feedback(currents=stop_transient, status=1)] * 4
        # Even after -1 is written, firmware can need a short braking
        # transition before producing the strict idle tail.
        + [feedback(currents=stop_transient, status=2)] * 2
        + [feedback(currents=(0,) * 6, status=2)] * 3
    )

    report = actuator.disable_and_verify()
    assert report.verified is True
    assert actuator.maximum_running_axis_current_ma == 400
    assert actuator.stop_settle_max_axis_current_ma == 800
    assert actuator.stop_max_axis_current_ma == 100
    assert transport.targets == (-1,) * 6


def test_installed_running_current_vector_is_allowed_below_1000ma_host_trip():
    actuator, transport, clock = _actuator(
        maximum_running_axis_current_ma=1000,
        stop_settle_max_axis_current_ma=1000,
        stop_max_axis_current_ma=100,
    )
    _arm(actuator, clock)
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.05)
    command = _stage(ledger, clock)
    ledger.acknowledge("franka", sequence=1, now_monotonic_s=clock())
    transport.feedback_behaviors.append(
        lambda: RH56SafetyFeedback(
            captured_monotonic_s=clock(),
            positions=(500,) * 6,
            angles=(500,) * 6,
            forces_g=(0,) * 6,
            currents_ma=(150, 0, 13, 0, 420, 42),
            errors=(0,) * 6,
            statuses=(1,) * 6,
            temperatures_c=(38, 38, 42, 38, 34, 36),
        )
    )

    receipt = actuator.execute(command, action_ledger=ledger)

    assert receipt.dual_device_commit_completed is True
    assert actuator.state is RH56ActuatorState.ARMED


def test_running_and_stop_transition_caps_reject_values_above_1000ma():
    actuator, transport, clock = _actuator(
        maximum_running_axis_current_ma=1000,
        stop_settle_max_axis_current_ma=1000,
        stop_max_axis_current_ma=100,
    )
    _arm(actuator, clock)
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.05)
    command = _stage(ledger, clock)
    ledger.acknowledge("franka", sequence=1, now_monotonic_s=clock())

    transport.feedback_behaviors.append(
        lambda: RH56SafetyFeedback(
            captured_monotonic_s=clock(),
            positions=(500,) * 6,
            angles=(500,) * 6,
            forces_g=(0,) * 6,
            currents_ma=(0, 0, 0, 0, 1001, 0),
            errors=(0,) * 6,
            statuses=(1,) * 6,
            temperatures_c=(25,) * 6,
        )
    )

    with pytest.raises(
        RH56TransactionalError,
        match="running current exceeded.*1000mA",
    ):
        actuator.execute(command, action_ledger=ledger)
    assert actuator.state is RH56ActuatorState.FAULT_LATCHED


def test_stop_rejects_current_above_separate_transition_cap():
    actuator, transport, clock = _actuator(
        maximum_running_axis_current_ma=400,
        stop_settle_max_axis_current_ma=800,
        stop_max_axis_current_ma=100,
    )
    _arm(actuator, clock)
    transport.feedback_behaviors.append(
        lambda: RH56SafetyFeedback(
            captured_monotonic_s=clock(),
            positions=(500,) * 6,
            angles=(500,) * 6,
            forces_g=(0,) * 6,
            currents_ma=(0, 0, 0, 0, 801, 0),
            errors=(0,) * 6,
            statuses=(1,) * 6,
            temperatures_c=(25,) * 6,
        )
    )

    with pytest.raises(
        RH56StopUnconfirmed,
        match="stop hold seed current exceeded per-axis bound 800mA",
    ):
        actuator.disable_and_verify()
    writes = [value for name, value in transport.calls if name == "write"]
    assert writes == [(-1,) * 6, (-1,) * 6]
    assert actuator.stop_confirmed is False


def test_stop_does_not_count_high_current_idle_samples_in_post_disable_proof():
    actuator, transport, clock = _actuator(
        maximum_running_axis_current_ma=400,
        stop_settle_max_axis_current_ma=800,
        stop_max_axis_current_ma=100,
        stop_timeout_s=0.8,
        stop_verify_interval_s=0.05,
    )
    _arm(actuator, clock)

    def feedback(*, current, status):
        return lambda: RH56SafetyFeedback(
            captured_monotonic_s=clock(),
            positions=(500,) * 6,
            angles=(500,) * 6,
            forces_g=(0,) * 6,
            currents_ma=(current,) * 6,
            errors=(0,) * 6,
            statuses=(status,) * 6,
            temperatures_c=(25,) * 6,
        )

    transport.feedback_behaviors.extend(
        [feedback(current=133, status=1)] * 4
        + [feedback(current=429, status=2)] * 20
    )

    with pytest.raises(
        RH56StopUnconfirmed,
        match="latest idle current exceeds per-axis bound 100mA",
    ):
        actuator.disable_and_verify()
    assert actuator.stop_confirmed is False
    assert transport.targets == (-1,) * 6


def test_stop_rejects_fault_status_during_numeric_hold_phase():
    actuator, transport, clock = _actuator()
    _arm(actuator, clock)
    transport.feedback_behaviors.append(
        lambda: RH56SafetyFeedback(
            captured_monotonic_s=clock(),
            positions=(500,) * 6,
            angles=(500,) * 6,
            forces_g=(0,) * 6,
            currents_ma=(0,) * 6,
            errors=(0,) * 6,
            statuses=(5, 2, 2, 2, 2, 2),
            temperatures_c=(25,) * 6,
        )
    )

    with pytest.raises(
        RH56StopUnconfirmed,
        match="stop hold seed actuator fault status remains",
    ):
        actuator.disable_and_verify()
    assert actuator.stop_confirmed is False
    assert transport.targets == (-1,) * 6


def test_stop_rejects_fault_status_in_post_disable_tail():
    actuator, transport, clock = _actuator()
    _arm(actuator, clock)

    def feedback(*, status):
        return lambda: RH56SafetyFeedback(
            captured_monotonic_s=clock(),
            positions=(500,) * 6,
            angles=(500,) * 6,
            forces_g=(0,) * 6,
            currents_ma=(0,) * 6,
            errors=(0,) * 6,
            statuses=(status,) * 6,
            temperatures_c=(25,) * 6,
        )

    transport.feedback_behaviors.extend(
        [feedback(status=2)] * 4 + [feedback(status=6)]
    )

    with pytest.raises(
        RH56StopUnconfirmed,
        match="post-disable stable sample 1 actuator fault status remains",
    ):
        actuator.disable_and_verify()
    assert actuator.stop_confirmed is False
    assert transport.targets == (-1,) * 6


def test_stop_rolling_tail_uses_timeout_and_rejects_continuous_motion():
    actuator, transport, clock = _actuator(
        stop_timeout_s=0.45,
        stop_verify_interval_s=0.05,
    )
    _arm(actuator, clock)
    started = clock()

    for value in range(500, 540):
        transport.feedback_behaviors.append(
            lambda value=value: RH56SafetyFeedback(
                captured_monotonic_s=clock(),
                positions=(value * 2,) * 6,
                angles=(value,) * 6,
                forces_g=(0,) * 6,
                currents_ma=(0,) * 6,
                errors=(0,) * 6,
                statuses=(2,) * 6,
                temperatures_c=(25,) * 6,
            )
        )

    with pytest.raises(RH56StopUnconfirmed, match="still moving") as raised:
        actuator.disable_and_verify()
    assert "observed=" in str(raised.value)
    assert clock() - started >= 0.20
    writes = [value for name, value in transport.calls if name == "write"]
    assert writes[0] == (500,) * 6
    assert writes[-2:] == [(-1,) * 6, (-1,) * 6]
    assert actuator.state is RH56ActuatorState.FAULT_LATCHED


def test_stop_hold_uses_exact_measured_angle_vector_before_release():
    actuator, transport, clock = _actuator()
    _arm(actuator, clock)
    measured = (901, 902, 903, 904, 905, 906)
    transport.feedback_behaviors.append(
        lambda: RH56SafetyFeedback(
            captured_monotonic_s=clock(),
            positions=(300, 301, 302, 303, 304, 305),
            angles=measured,
            forces_g=(0,) * 6,
            currents_ma=(0,) * 6,
            errors=(0,) * 6,
            statuses=(1,) * 6,
            temperatures_c=(25,) * 6,
        )
    )

    report = actuator.disable_and_verify()
    assert report.verified is True
    writes = [value for name, value in transport.calls if name == "write"]
    assert writes == [measured, (-1,) * 6, (-1,) * 6]


def test_stop_hold_converts_thumb_rotation_feedback_to_command_domain():
    actuator, transport, clock = _actuator(
        feedback_to_command_offset_units=(0, 0, 0, 0, 0, 15),
        stop_hold_command_minimum=(0, 0, 0, 0, 0, 900),
        stop_hold_command_maximum=(1000,) * 6,
    )
    _arm(actuator, clock)
    measured = (500, 500, 500, 500, 500, 980)

    def stable_feedback():
        return RH56SafetyFeedback(
            captured_monotonic_s=clock(),
            positions=measured,
            angles=measured,
            forces_g=(0,) * 6,
            currents_ma=(0,) * 6,
            errors=(0,) * 6,
            statuses=(2,) * 6,
            temperatures_c=(25,) * 6,
        )

    transport.feedback_behaviors.extend([stable_feedback] * 5)
    report = actuator.disable_and_verify()

    assert report.verified is True
    writes = [value for name, value in transport.calls if name == "write"]
    assert writes == [
        (500, 500, 500, 500, 500, 995),
        (-1,) * 6,
        (-1,) * 6,
    ]


def test_stop_hold_clamps_small_calibrated_q6_lower_endpoint_overshoot():
    actuator, transport, clock = _actuator(
        feedback_to_command_offset_units=(0, 0, 0, 0, 0, 15),
        stop_hold_command_minimum=(0, 0, 0, 0, 0, 416),
        stop_hold_command_maximum=(1000,) * 6,
    )
    _arm(actuator, clock)
    measured = (500, 500, 500, 500, 500, 395)

    def stable_feedback():
        return RH56SafetyFeedback(
            captured_monotonic_s=clock(),
            positions=measured,
            angles=measured,
            forces_g=(0,) * 6,
            currents_ma=(0,) * 6,
            errors=(0,) * 6,
            statuses=(2,) * 6,
            temperatures_c=(25,) * 6,
        )

    transport.feedback_behaviors.extend([stable_feedback] * 5)
    report = actuator.disable_and_verify()

    assert report.verified is True
    writes = [value for name, value in transport.calls if name == "write"]
    assert writes == [
        (500, 500, 500, 500, 500, 416),
        (-1,) * 6,
        (-1,) * 6,
    ]


def test_stop_hold_rejects_q6_lower_excursion_beyond_calibration_offset():
    actuator, transport, clock = _actuator(
        feedback_to_command_offset_units=(0, 0, 0, 0, 0, 15),
        stop_hold_command_minimum=(0, 0, 0, 0, 0, 416),
        stop_hold_command_maximum=(1000,) * 6,
    )
    _arm(actuator, clock)
    measured = (500, 500, 500, 500, 500, 385)
    transport.feedback_behaviors.append(
        lambda: RH56SafetyFeedback(
            captured_monotonic_s=clock(),
            positions=measured,
            angles=measured,
            forces_g=(0,) * 6,
            currents_ma=(0,) * 6,
            errors=(0,) * 6,
            statuses=(2,) * 6,
            temperatures_c=(25,) * 6,
        )
    )

    with pytest.raises(RH56StopUnconfirmed, match="outside calibrated"):
        actuator.disable_and_verify()
    writes = [value for name, value in transport.calls if name == "write"]
    assert writes == [(-1,) * 6, (-1,) * 6]


def test_stop_hold_readback_failure_still_attempts_both_disable_passes():
    actuator, transport, clock = _actuator()
    _arm(actuator, clock)
    transport.read_target_behaviors.append((499,) * 6)

    with pytest.raises(RH56StopUnconfirmed, match="hold readback mismatch"):
        actuator.disable_and_verify()
    writes = [value for name, value in transport.calls if name == "write"]
    assert writes == [(500,) * 6, (-1,) * 6, (-1,) * 6]
    assert transport.targets == (-1,) * 6
    assert actuator.state is RH56ActuatorState.FAULT_LATCHED


def test_stop_invalid_measured_angle_never_becomes_a_numeric_hold():
    actuator, transport, clock = _actuator()
    _arm(actuator, clock)
    transport.feedback_behaviors.append(
        lambda: RH56SafetyFeedback(
            captured_monotonic_s=clock(),
            positions=(500,) * 6,
            angles=(-1, 500, 500, 500, 500, 500),
            forces_g=(0,) * 6,
            currents_ma=(0,) * 6,
            errors=(0,) * 6,
            statuses=(2,) * 6,
            temperatures_c=(25,) * 6,
        )
    )

    with pytest.raises(RH56StopUnconfirmed, match=r"stop hold ANGLE_ACT\[0\]"):
        actuator.disable_and_verify()
    writes = [value for name, value in transport.calls if name == "write"]
    assert writes == [(-1,) * 6, (-1,) * 6]
    assert actuator.state is RH56ActuatorState.FAULT_LATCHED


def test_stop_waits_for_fresh_stable_tail_after_release_transient():
    actuator, transport, clock = _actuator()
    _arm(actuator, clock)

    def feedback(value):
        return lambda: RH56SafetyFeedback(
            captured_monotonic_s=clock(),
            positions=(value,) * 6,
            angles=(value,) * 6,
            forces_g=(0,) * 6,
            currents_ma=(0,) * 6,
            errors=(0,) * 6,
            statuses=(2,) * 6,
            temperatures_c=(25,) * 6,
        )

    # seed + three stable held samples + one moving post-disable sample,
    # followed by the fake's stable default feedback.
    transport.feedback_behaviors.extend(
        [feedback(500), feedback(500), feedback(500), feedback(500), feedback(504)]
    )

    report = actuator.disable_and_verify()
    assert report.verified is True
    assert transport.targets == (-1,) * 6
    assert actuator.stop_confirmed is True
    assert [name for name, _ in transport.calls].count("feedback") == 8


def test_five_second_stop_budget_accepts_proven_long_transitional_tail():
    actuator, transport, clock = _actuator(
        stop_timeout_s=5.0,
        stop_verify_interval_s=0.02,
    )
    _arm(actuator, clock)
    started = clock()

    def delayed_idle_feedback():
        idle = clock() - started >= 2.7
        return RH56SafetyFeedback(
            captured_monotonic_s=clock(),
            positions=(500,) * 6,
            angles=(500,) * 6,
            forces_g=(0,) * 6,
            currents_ma=(0,) * 6,
            errors=(0,) * 6,
            statuses=((2,) * 6 if idle else (0, 2, 2, 2, 2, 2)),
            temperatures_c=(25,) * 6,
        )

    transport.feedback_behaviors.extend([delayed_idle_feedback] * 256)

    report = actuator.disable_and_verify()

    assert report.verified is True
    assert 2.7 <= report.completed_monotonic_s - started < 5.0
    assert actuator.stop_confirmed is True
    assert transport.targets == (-1,) * 6


def test_stop_rejects_continuous_motion_in_wholly_post_disable_tail():
    actuator, transport, clock = _actuator(
        stop_timeout_s=0.8,
        stop_verify_interval_s=0.05,
    )
    _arm(actuator, clock)

    def feedback(value):
        return lambda: RH56SafetyFeedback(
            captured_monotonic_s=clock(),
            positions=(value,) * 6,
            angles=(value,) * 6,
            forces_g=(0,) * 6,
            currents_ma=(0,) * 6,
            errors=(0,) * 6,
            statuses=(2,) * 6,
            temperatures_c=(25,) * 6,
        )

    transport.feedback_behaviors.extend(
        [feedback(500)] * 4
        + [feedback(value) for value in range(504, 640, 8)]
    )

    with pytest.raises(
        RH56StopUnconfirmed,
        match="disabled target did not reach a stable idle tail",
    ):
        actuator.disable_and_verify()
    assert transport.targets == (-1,) * 6
    assert actuator.stop_confirmed is False
    assert actuator.state is RH56ActuatorState.FAULT_LATCHED


def test_non_owner_thread_cannot_touch_transport():
    actuator, transport, clock = _actuator()
    _arm(actuator, clock)
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.05)
    command = _stage(ledger, clock)
    errors = []

    def intruder():
        try:
            actuator.execute(command, action_ledger=ledger)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=intruder)
    thread.start()
    thread.join()
    assert len(errors) == 1
    assert "non-owner" in str(errors[0])
    assert transport.calls == []
    actuator.disable_and_verify()


def test_wire_only_estimate_at_115200_is_not_hardware_60hz_evidence():
    estimate = estimate_compact_transaction_wire_rate(115200)
    assert estimate.wire_bytes_per_transaction == 158
    assert estimate.request_response_exchanges == 4
    assert estimate.ideal_wire_time_s == pytest.approx(1580.0 / 115200.0)
    assert estimate.ideal_wire_only_max_rate_hz == pytest.approx(72.911392405)
    assert estimate.ideal_wire_utilization_at_60hz == pytest.approx(0.8229166667)
    assert estimate.wire_only_60hz_possible is True
    assert estimate.real_hardware_60hz_verified is False
    assert estimate.feasibility == "UNKNOWN_UNTIL_HARDWARE_COMMISSIONING"


def test_deterministic_wire_time_fake_can_exceed_60hz_but_claim_stays_unknown():
    clock = _Clock()
    write_s = 29 * 10 / 115200.0
    read_target_s = 29 * 10 / 115200.0
    feedback_s = (41 + 59) * 10 / 115200.0
    transport = _FakeTransport(
        clock,
        write_duration_s=write_s,
        read_target_duration_s=read_target_s,
        feedback_duration_s=feedback_s,
    )
    actuator, _transport, _clock = _actuator(clock, transport)
    _arm(actuator, clock)
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.05)
    started = clock()
    for sequence in range(1, 61):
        command = _stage(
            ledger,
            clock,
            sequence=sequence,
            value=sequence / 100.0,
        )
        ledger.acknowledge("franka", sequence=sequence, now_monotonic_s=clock())
        receipt = actuator.execute(command, action_ledger=ledger)
        assert receipt.dual_device_commit_completed is True
    elapsed = clock() - started
    assert 60.0 / elapsed == pytest.approx(72.911392405)
    assert estimate_compact_transaction_wire_rate().real_hardware_60hz_verified is False


def test_watchdog_cannot_be_configured_above_commissioning_limit():
    clock = _Clock()
    transport = _FakeTransport(clock)
    with pytest.raises(ValueError, match="must not exceed"):
        RH56TransactionalActuator(
            transport,
            command_watchdog_timeout_s=0.051,
            monotonic=clock,
        )
