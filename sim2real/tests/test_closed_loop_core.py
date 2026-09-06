import threading

import numpy as np
import pytest

from sim2real.closed_loop_core import (
    ClosedLoopCommand,
    ClosedLoopProtocolError,
    ClosedLoopSafetyGate,
    ExecutedActionLedger,
    MotionAuthorization,
    PolicyCommandSampleHold,
    SafetyState,
    SingleThreadOwner,
    TransactionalV94ActionMapper,
)
from sim2real.contracts.v94 import (
    INITIAL_PREVIOUS_ACTION13,
    Q_HAND_SEMANTIC_CLOSE_RAD,
)
from sim2real.contracts.actions import QD_G015_FRANKA_ACTION_CONTRACT_ID
from sim2real.contracts.actions import inward_float32_joint_limits


FRANKA_PHYSICAL_LIMITS_F64 = np.asarray(
    [
        [-2.7437, 2.7437],
        [-1.7837, 1.7837],
        [-2.9007, 2.9007],
        [-3.0421, -0.1518],
        [-2.8065, 2.8065],
        [0.5445, 4.5169],
        [-3.0159, 3.0159],
    ],
    dtype=np.float64,
)
FRANKA_JOINT_MARGIN_RAD = 0.05
FRANKA_SAFE_LIMITS_F64 = FRANKA_PHYSICAL_LIMITS_F64.copy()
FRANKA_SAFE_LIMITS_F64[:, 0] += FRANKA_JOINT_MARGIN_RAD
FRANKA_SAFE_LIMITS_F64[:, 1] -= FRANKA_JOINT_MARGIN_RAD


def _authorization() -> MotionAuthorization:
    return MotionAuthorization(
        run_id="run-1",
        authorization_id="operator-approval-1",
        issued_monotonic_s=10.0,
        expires_monotonic_s=20.0,
    )


def _command(
    sequence: int,
    *,
    produced: float,
    previous: np.ndarray,
    executed_value: float,
) -> ClosedLoopCommand:
    executed = np.full(13, executed_value, dtype=np.float32)
    return ClosedLoopCommand(
        sequence=sequence,
        produced_monotonic_s=produced,
        observation_realtime_s=1_750_000_000.0 + produced,
        previous_executed_action13_used=previous,
        raw_policy_action13=executed,
        executed_policy_action13=executed,
        franka_target_q_rad=np.full(7, 0.1 * sequence),
        rh56_angle_set_register_order=np.full(6, 1000 - sequence),
    )


def _transactional_mapper(**kwargs) -> TransactionalV94ActionMapper:
    options = dict(
        initial_arm_target_q_rad=np.zeros(7),
        initial_hand_target_q_policy_order_rad=np.zeros(6),
        joint_limits_rad=np.asarray([[-1.0, 1.0]] * 7),
        joint_limit_margin_rad=0.1,
        control_dt_s=1.0 / 60.0,
        q_hand_close_rad=Q_HAND_SEMANTIC_CLOSE_RAD,
    )
    options.update(kwargs)
    return TransactionalV94ActionMapper(**options)


def test_exact_target_replay_bypasses_action_remap_but_commits_action_chain():
    mapper = _transactional_mapper()
    action = np.linspace(-1.0, 1.0, 13, dtype=np.float32)
    arm = np.asarray([0.01, -0.55, 0.02, -0.80, 0.03, 0.62, 0.72])
    registers = np.asarray([1000, 750, 500, 250, 0, 0], dtype=np.int32)
    proposal = mapper.propose_exact_targets(
        1,
        action,
        franka_target_q_rad=arm,
        rh56_angle_set_register_order=registers,
        measured_q_rad=np.zeros(7),
    )
    np.testing.assert_allclose(proposal.mapped.franka_target_q_rad, arm)
    np.testing.assert_array_equal(
        proposal.mapped.rh56_angle_set_register_order, registers
    )
    np.testing.assert_array_equal(proposal.mapped.executed_policy_action13, action)

    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)
    command = ClosedLoopCommand(
        sequence=1,
        produced_monotonic_s=1.0,
        observation_realtime_s=1.0,
        previous_executed_action13_used=INITIAL_PREVIOUS_ACTION13,
        raw_policy_action13=action,
        executed_policy_action13=action,
        franka_target_q_rad=arm,
        rh56_angle_set_register_order=registers,
        hold_arm_target=False,
    )
    ledger.stage(command, now_monotonic_s=1.0)
    ledger.acknowledge("franka", sequence=1, now_monotonic_s=1.1)
    ledger.acknowledge("rh56", sequence=1, now_monotonic_s=1.2)
    mapper.commit(proposal, action_ledger=ledger)
    committed_arm, committed_hand = mapper.committed_targets()
    np.testing.assert_allclose(committed_arm, arm)
    assert committed_hand[0] == pytest.approx(mapper._q_hand_close[0])


def _command_from_proposal(proposal) -> ClosedLoopCommand:
    mapped = proposal.mapped
    return ClosedLoopCommand(
        sequence=proposal.sequence,
        produced_monotonic_s=1.0,
        observation_realtime_s=2.0,
        previous_executed_action13_used=INITIAL_PREVIOUS_ACTION13,
        raw_policy_action13=mapped.raw_policy_action13,
        executed_policy_action13=mapped.executed_policy_action13,
        franka_target_q_rad=mapped.franka_target_q_rad,
        rh56_angle_set_register_order=mapped.rh56_angle_set_register_order,
    )


def test_safety_gate_starts_disarmed_and_needs_live_interlocks():
    gate = ClosedLoopSafetyGate()
    assert gate.state is SafetyState.DISARMED
    with pytest.raises(ClosedLoopProtocolError, match="deadman"):
        gate.arm(
            _authorization(),
            run_id="run-1",
            now_monotonic_s=11.0,
            franka_rest_verified=True,
            rh56_disabled_verified=True,
        )
    gate.update_interlocks(deadman_asserted=True, estop_healthy=True)
    gate.arm(
        _authorization(),
        run_id="run-1",
        now_monotonic_s=11.0,
        franka_rest_verified=True,
        rh56_disabled_verified=True,
    )
    assert gate.state is SafetyState.ARMED
    gate.require_motion(run_id="run-1", now_monotonic_s=12.0)


@pytest.mark.parametrize(
    ("deadman", "estop", "message"),
    [(False, True, "deadman"), (True, False, "emergency stop")],
)
def test_interlock_loss_latches_terminal_fault(deadman, estop, message):
    gate = ClosedLoopSafetyGate()
    gate.update_interlocks(deadman_asserted=True, estop_healthy=True)
    gate.arm(
        _authorization(),
        run_id="run-1",
        now_monotonic_s=11.0,
        franka_rest_verified=True,
        rh56_disabled_verified=True,
    )
    with pytest.raises(ClosedLoopProtocolError, match=message):
        gate.update_interlocks(deadman_asserted=deadman, estop_healthy=estop)
    assert gate.state is SafetyState.FAULT_LATCHED
    with pytest.raises(ClosedLoopProtocolError, match="fault-latched"):
        gate.disarm_after_verified_stop(
            run_id="run-1",
            franka_stop_verified=True,
            rh56_disabled_verified=True,
        )


def test_authorization_expiry_latches_instead_of_silently_holding():
    gate = ClosedLoopSafetyGate()
    gate.update_interlocks(deadman_asserted=True, estop_healthy=True)
    gate.arm(
        _authorization(),
        run_id="run-1",
        now_monotonic_s=11.0,
        franka_rest_verified=True,
        rh56_disabled_verified=True,
    )
    with pytest.raises(ClosedLoopProtocolError, match="expired"):
        gate.require_motion(run_id="run-1", now_monotonic_s=20.0)
    assert gate.state is SafetyState.FAULT_LATCHED


def test_clean_disarm_requires_both_devices_to_have_verified_stop():
    gate = ClosedLoopSafetyGate()
    gate.update_interlocks(deadman_asserted=True, estop_healthy=True)
    gate.arm(
        _authorization(),
        run_id="run-1",
        now_monotonic_s=11.0,
        franka_rest_verified=True,
        rh56_disabled_verified=True,
    )
    with pytest.raises(ClosedLoopProtocolError, match="both Franka stop"):
        gate.disarm_after_verified_stop(
            run_id="run-1",
            franka_stop_verified=True,
            rh56_disabled_verified=False,
        )
    assert gate.state is SafetyState.FAULT_LATCHED


def test_clean_disarm_consumes_authorization():
    gate = ClosedLoopSafetyGate()
    authorization = _authorization()
    gate.update_interlocks(deadman_asserted=True, estop_healthy=True)
    gate.arm(
        authorization,
        run_id="run-1",
        now_monotonic_s=11.0,
        franka_rest_verified=True,
        rh56_disabled_verified=True,
    )
    gate.disarm_after_verified_stop(
        run_id="run-1",
        franka_stop_verified=True,
        rh56_disabled_verified=True,
    )
    assert gate.state is SafetyState.DISARMED
    with pytest.raises(ClosedLoopProtocolError, match="already consumed"):
        gate.arm(
            authorization,
            run_id="run-1",
            now_monotonic_s=12.0,
            franka_rest_verified=True,
            rh56_disabled_verified=True,
        )


def test_sample_hold_reuses_same_command_at_1khz_and_rejects_stale_data():
    command = _command(
        1,
        produced=1.0,
        previous=INITIAL_PREVIOUS_ACTION13,
        executed_value=0.25,
    )
    hold = PolicyCommandSampleHold()
    hold.publish(command)
    assert hold.sample(now_monotonic_s=1.001, maximum_age_s=0.04) is command
    assert hold.sample(now_monotonic_s=1.002, maximum_age_s=0.04) is command
    with pytest.raises(ClosedLoopProtocolError, match="stale"):
        hold.sample(now_monotonic_s=1.05, maximum_age_s=0.04)


def test_sample_hold_rejects_skipped_policy_sequence():
    hold = PolicyCommandSampleHold()
    first = _command(
        1,
        produced=1.0,
        previous=INITIAL_PREVIOUS_ACTION13,
        executed_value=0.1,
    )
    hold.publish(first)
    with pytest.raises(ClosedLoopProtocolError, match="sequence gap"):
        hold.publish(
            _command(
                3,
                produced=1.01,
                previous=first.executed_policy_action13,
                executed_value=0.2,
            )
        )


def test_previous_action_advances_only_after_both_actuators_acknowledge():
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.02)
    first = _command(
        1,
        produced=1.0,
        previous=ledger.previous_executed_action13(),
        executed_value=0.25,
    )
    ledger.stage(first, now_monotonic_s=1.001)
    assert not ledger.acknowledge(
        "franka", sequence=1, now_monotonic_s=1.002
    )
    np.testing.assert_array_equal(
        ledger.previous_executed_action13(), INITIAL_PREVIOUS_ACTION13
    )
    with pytest.raises(ClosedLoopProtocolError, match="another is uncommitted"):
        ledger.stage(
            _command(
                2,
                produced=1.003,
                previous=first.executed_policy_action13,
                executed_value=0.5,
            ),
            now_monotonic_s=1.003,
        )
    assert "another is uncommitted" in ledger.fault_reason


def test_dual_ack_commits_previous_action():
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.02)
    first = _command(
        1,
        produced=1.0,
        previous=ledger.previous_executed_action13(),
        executed_value=0.25,
    )
    ledger.stage(first, now_monotonic_s=1.001)
    assert not ledger.acknowledge(
        "franka", sequence=1, now_monotonic_s=1.002
    )
    assert ledger.acknowledge("rh56", sequence=1, now_monotonic_s=1.004)
    np.testing.assert_array_equal(
        ledger.previous_executed_action13(), first.executed_policy_action13
    )
    assert ledger.last_committed_sequence == 1


def test_ledger_rejects_policy_observation_using_uncommitted_action():
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.02)
    with pytest.raises(ClosedLoopProtocolError, match="uncommitted previous"):
        ledger.stage(
            _command(
                1,
                produced=1.0,
                previous=np.zeros(13),
                executed_value=0.25,
            ),
            now_monotonic_s=1.001,
        )


def test_partial_commit_timeout_is_sticky():
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.01)
    command = _command(
        1,
        produced=1.0,
        previous=ledger.previous_executed_action13(),
        executed_value=0.25,
    )
    ledger.stage(command, now_monotonic_s=1.001)
    ledger.acknowledge("franka", sequence=1, now_monotonic_s=1.002)
    with pytest.raises(ClosedLoopProtocolError, match="missing acknowledgements=rh56"):
        ledger.require_commit_deadline(now_monotonic_s=1.011)
    with pytest.raises(ClosedLoopProtocolError, match="fault is latched"):
        ledger.acknowledge("rh56", sequence=1, now_monotonic_s=1.012)


def test_consumer_ack_query_is_atomic_and_sequence_exact():
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.01)
    command = _command(
        1,
        produced=1.0,
        previous=ledger.previous_executed_action13(),
        executed_value=0.25,
    )
    ledger.stage(command, now_monotonic_s=1.001)

    assert ledger.consumer_acknowledged("franka", sequence=1) is False
    assert ledger.consumer_acknowledged("rh56", sequence=1) is False
    ledger.acknowledge("franka", sequence=1, now_monotonic_s=1.002)
    assert ledger.consumer_acknowledged("franka", sequence=1) is True
    assert ledger.consumer_acknowledged("rh56", sequence=1) is False
    with pytest.raises(ClosedLoopProtocolError, match="neither pending"):
        ledger.consumer_acknowledged("franka", sequence=2)

    ledger.acknowledge("rh56", sequence=1, now_monotonic_s=1.003)
    assert ledger.consumer_acknowledged("franka", sequence=1) is True
    assert ledger.consumer_acknowledged("rh56", sequence=1) is True


def test_command_payload_is_immutable_after_construction():
    raw = np.full(13, 0.25, dtype=np.float32)
    command = ClosedLoopCommand(
        sequence=1,
        produced_monotonic_s=1.0,
        observation_realtime_s=2.0,
        previous_executed_action13_used=INITIAL_PREVIOUS_ACTION13,
        raw_policy_action13=raw,
        executed_policy_action13=raw,
        franka_target_q_rad=np.zeros(7),
        rh56_angle_set_register_order=np.full(6, 1000),
    )
    raw[:] = -1.0
    np.testing.assert_array_equal(command.raw_policy_action13, np.full(13, 0.25))
    with pytest.raises(ValueError):
        command.franka_target_q_rad[0] = 1.0


def test_single_thread_owner_rejects_second_thread():
    owner = SingleThreadOwner("RH56 serial")
    owner.claim_for_current_thread()
    errors = []

    def intruder():
        try:
            owner.require_current_thread()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=intruder)
    thread.start()
    thread.join()
    assert len(errors) == 1
    assert isinstance(errors[0], ClosedLoopProtocolError)
    assert "non-owner" in str(errors[0])


def test_transactional_mapper_proposal_does_not_mutate_and_can_be_discarded():
    mapper = _transactional_mapper()
    initial_arm, initial_hand = mapper.committed_targets()
    proposal = mapper.propose(
        1,
        np.ones(13, dtype=np.float32),
        measured_q_rad=np.zeros(7),
    )
    proposed_arm = proposal.mapped.franka_target_q_rad
    assert np.max(np.abs(proposed_arm - initial_arm)) > 0.0
    arm, hand = mapper.committed_targets()
    np.testing.assert_array_equal(arm, initial_arm)
    np.testing.assert_array_equal(hand, initial_hand)
    mapper.discard_unstaged(proposal)
    replacement = mapper.propose(
        1,
        -np.ones(13, dtype=np.float32),
        measured_q_rad=np.zeros(7),
    )
    assert replacement.sequence == 1
    arm, hand = mapper.committed_targets()
    np.testing.assert_array_equal(arm, initial_arm)
    np.testing.assert_array_equal(hand, initial_hand)


def test_transactional_mapper_commits_only_after_dual_ack():
    mapper = _transactional_mapper()
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.02)
    proposal = mapper.propose(
        1,
        np.ones(13, dtype=np.float32),
        measured_q_rad=np.zeros(7),
    )
    command = _command_from_proposal(proposal)
    ledger.stage(command, now_monotonic_s=1.001)
    with pytest.raises(ClosedLoopProtocolError, match="dual-acknowledged"):
        mapper.commit(proposal, action_ledger=ledger)
    initial_arm, _ = mapper.committed_targets()
    np.testing.assert_array_equal(initial_arm, np.zeros(7))
    ledger.acknowledge("franka", sequence=1, now_monotonic_s=1.002)
    with pytest.raises(ClosedLoopProtocolError, match="dual-acknowledged"):
        mapper.commit(proposal, action_ledger=ledger)
    ledger.acknowledge("rh56", sequence=1, now_monotonic_s=1.003)
    mapper.commit(proposal, action_ledger=ledger)
    committed_arm, committed_hand = mapper.committed_targets()
    np.testing.assert_array_equal(
        committed_arm, proposal.mapped.franka_target_q_rad
    )
    np.testing.assert_array_equal(
        committed_hand, proposal.mapped.rh56_target_q_policy_order_rad
    )
    assert mapper.last_committed_sequence == 1


def test_transactional_mapper_requires_exact_pending_proposal_identity():
    mapper = _transactional_mapper()
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.02)
    proposal = mapper.propose(
        1,
        np.zeros(13, dtype=np.float32),
        measured_q_rad=np.zeros(7),
    )
    command = _command_from_proposal(proposal)
    ledger.stage(command, now_monotonic_s=1.001)
    ledger.acknowledge("franka", sequence=1, now_monotonic_s=1.002)
    ledger.acknowledge("rh56", sequence=1, now_monotonic_s=1.003)
    impostor = type(proposal)(
        sequence=proposal.sequence,
        mapped=proposal.mapped,
        prior_arm_target_q_rad=proposal.prior_arm_target_q_rad,
        prior_hand_target_q_policy_order_rad=(
            proposal.prior_hand_target_q_policy_order_rad
        ),
        arm_target_rate_rad_s=proposal.arm_target_rate_rad_s,
        hand_target_rate_rad_s=proposal.hand_target_rate_rad_s,
    )
    with pytest.raises(ClosedLoopProtocolError, match="exact pending proposal"):
        mapper.commit(impostor, action_ledger=ledger)
    assert "exact pending proposal" in mapper.fault_reason


def test_transactional_mapper_contracts_joint_limits_by_fci_margin():
    mapper = _transactional_mapper(
        initial_arm_target_q_rad=np.full(7, 0.899, dtype=np.float32)
    )
    proposal = mapper.propose(
        1,
        np.ones(13, dtype=np.float32),
        measured_q_rad=np.full(7, 0.899, dtype=np.float32),
    )
    assert np.all(proposal.mapped.franka_target_q_rad <= 0.9)
    np.testing.assert_allclose(
        proposal.mapped.franka_target_q_rad, 0.9, atol=1.0e-7
    )


@pytest.mark.parametrize("side,action_value", [(0, -1.0), (1, 1.0)])
def test_transactional_mapper_margin_and_float32_rounding_stay_inside_native_limits(
    side,
    action_value,
):
    expected = inward_float32_joint_limits(
        FRANKA_PHYSICAL_LIMITS_F64,
        margin_rad=FRANKA_JOINT_MARGIN_RAD,
    )
    midpoint = np.mean(FRANKA_SAFE_LIMITS_F64, axis=1)
    mapper = TransactionalV94ActionMapper(
        initial_arm_target_q_rad=midpoint,
        initial_hand_target_q_policy_order_rad=np.zeros(6),
        joint_limits_rad=FRANKA_PHYSICAL_LIMITS_F64,
        joint_limit_margin_rad=FRANKA_JOINT_MARGIN_RAD,
        control_dt_s=1.0 / 20.0,
        q_hand_close_rad=Q_HAND_SEMANTIC_CLOSE_RAD,
        franka_action_contract_id=QD_G015_FRANKA_ACTION_CONTRACT_ID,
    )
    action = np.asarray([action_value] * 7 + [-1.0] * 6, dtype=np.float32)
    q_d = FRANKA_SAFE_LIMITS_F64[:, side] + (-1.0 if side == 0 else 1.0)
    proposal = mapper.propose(
        1,
        action,
        measured_q_rad=midpoint,
        shaper_q_d_rad=q_d,
    )

    target = proposal.mapped.franka_target_q_rad
    np.testing.assert_array_equal(target, expected[:, side])
    assert np.all(target.astype(np.float64) >= FRANKA_SAFE_LIMITS_F64[:, 0])
    assert np.all(target.astype(np.float64) <= FRANKA_SAFE_LIMITS_F64[:, 1])


@pytest.mark.parametrize("side", [0, 1])
def test_exact_replay_canonicalizes_only_the_nearest_quantized_boundary(side):
    expected = inward_float32_joint_limits(
        FRANKA_PHYSICAL_LIMITS_F64,
        margin_rad=FRANKA_JOINT_MARGIN_RAD,
    )
    rounded = FRANKA_SAFE_LIMITS_F64.astype(np.float32)
    midpoint = np.mean(FRANKA_SAFE_LIMITS_F64, axis=1)
    mapper = TransactionalV94ActionMapper(
        initial_arm_target_q_rad=midpoint,
        initial_hand_target_q_policy_order_rad=np.zeros(6),
        joint_limits_rad=FRANKA_PHYSICAL_LIMITS_F64,
        joint_limit_margin_rad=FRANKA_JOINT_MARGIN_RAD,
        control_dt_s=1.0 / 20.0,
        q_hand_close_rad=Q_HAND_SEMANTIC_CLOSE_RAD,
    )
    proposal = mapper.propose_exact_targets(
        1,
        np.zeros(13, dtype=np.float32),
        franka_target_q_rad=rounded[:, side],
        rh56_angle_set_register_order=np.full(6, 1000, dtype=np.int32),
        measured_q_rad=midpoint,
    )
    np.testing.assert_array_equal(
        proposal.mapped.franka_target_q_rad,
        expected[:, side],
    )
    assert proposal.mapped.clipped is True
    assert "Franka float32 boundary canonicalized inward" in proposal.mapped.reasons


@pytest.mark.parametrize("side", [0, 1])
def test_exact_replay_rejects_the_next_float32_beyond_quantized_boundary(side):
    rounded = FRANKA_SAFE_LIMITS_F64.astype(np.float32)
    midpoint = np.mean(FRANKA_SAFE_LIMITS_F64, axis=1)
    direction = np.float32(-np.inf if side == 0 else np.inf)
    for axis in range(7):
        mapper = TransactionalV94ActionMapper(
            initial_arm_target_q_rad=midpoint,
            initial_hand_target_q_policy_order_rad=np.zeros(6),
            joint_limits_rad=FRANKA_PHYSICAL_LIMITS_F64,
            joint_limit_margin_rad=FRANKA_JOINT_MARGIN_RAD,
            control_dt_s=1.0 / 20.0,
            q_hand_close_rad=Q_HAND_SEMANTIC_CLOSE_RAD,
        )
        target = midpoint.astype(np.float32)
        target[axis] = np.nextafter(rounded[axis, side], direction)
        with pytest.raises(
            ClosedLoopProtocolError,
            match="exact Franka replay target is outside joint limits",
        ):
            mapper.propose_exact_targets(
                1,
                np.zeros(13, dtype=np.float32),
                franka_target_q_rad=target,
                rh56_angle_set_register_order=np.full(6, 1000, dtype=np.int32),
                measured_q_rad=midpoint,
            )


def test_transactional_mapper_rejects_uncommissioned_policy_target_rate():
    mapper = _transactional_mapper(
        commissioned_max_arm_target_rate_rad_s=0.05
    )
    with pytest.raises(ClosedLoopProtocolError, match="0.180000"):
        mapper.propose(
            1,
            np.ones(13, dtype=np.float32),
            measured_q_rad=np.zeros(7),
        )
    assert "Franka target rate exceeds commissioned" in mapper.fault_reason
    arm, hand = mapper.committed_targets()
    np.testing.assert_array_equal(arm, np.zeros(7))
    np.testing.assert_array_equal(hand, np.zeros(6))


def test_transactional_mapper_rate_guard_is_optional_for_offline_analysis():
    mapper = _transactional_mapper(
        commissioned_max_arm_target_rate_rad_s=None
    )
    proposal = mapper.propose(
        1,
        np.ones(13, dtype=np.float32),
        measured_q_rad=np.zeros(7),
    )
    np.testing.assert_allclose(
        proposal.arm_target_rate_rad_s, 0.18, atol=1.0e-6
    )


def test_qd_g015_transactional_mapper_uses_qd_and_skips_target_rate_guard():
    mapper = _transactional_mapper(
        control_dt_s=0.05,
        commissioned_max_arm_target_rate_rad_s=0.05,
        franka_action_contract_id=QD_G015_FRANKA_ACTION_CONTRACT_ID,
    )
    proposal = mapper.propose(
        1,
        np.asarray([1.0] * 7 + [-1.0] * 6, dtype=np.float32),
        measured_q_rad=np.full(7, -0.8),
        shaper_q_d_rad=np.full(7, 0.2),
    )

    np.testing.assert_allclose(
        proposal.mapped.franka_target_q_rad, 0.26, atol=1.0e-7
    )
    assert np.max(np.abs(proposal.arm_target_rate_rad_s)) > 0.05
    assert mapper.fault_reason is None


def test_qd_g015_transactional_mapper_missing_qd_latches_fault():
    mapper = _transactional_mapper(
        franka_action_contract_id=QD_G015_FRANKA_ACTION_CONTRACT_ID,
    )
    with pytest.raises(ClosedLoopProtocolError, match="requires shaper_q_d_rad"):
        mapper.propose(
            1,
            np.zeros(13, dtype=np.float32),
            measured_q_rad=np.zeros(7),
        )
    assert "requires shaper_q_d_rad" in mapper.fault_reason
