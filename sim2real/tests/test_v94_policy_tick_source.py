from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sim2real.runtime.bounded_c2_runtime import BoundedPolicyTickHold
from sim2real.action_replay import (
    ReplayPolicyOutput,
    ReplayActionSequence,
    TransactionalReplayActionPolicy,
)
from sim2real.closed_loop_core import (
    ClosedLoopCommand,
    ExecutedActionLedger,
    TransactionalV94ActionMapper,
)
from sim2real.deployment.bundle import DeployBundle, load_checkpoint_safely
from sim2real.policy import (
    QD_G015_ACTION_CONTROLLER_CONTRACT_ID,
    RollingStudentPolicy,
)
from sim2real.contracts.v94 import (
    INITIAL_PREVIOUS_ACTION13,
    Q_HAND_SEMANTIC_CLOSE_RAD,
    V94Contract,
)
from sim2real.contracts.actions import TransactionalRH56HardwareCommandShaper
from sim2real.runtime.v94_policy_tick_source import (
    CAMERA_REUSE_HOLD_REASON,
    EXACT_REPLAY_ARM_HOLD_REASON,
    FRANKA_ARRIVAL_GATE_HOLD_REASON,
    FRANKA_STATE_STALE_HOLD_REASON,
    OBJECT_POINTCLOUD_STALE_HOLD_REASON,
    OBSERVATION_SNAPSHOT_CONTRACT,
    POLICY_TRANSACTIONAL_STATE_CONTRACT,
    TransactionalBoundedV94PolicyTickSource,
    V94FreshActuatorSnapshot,
    V94PolicyObservation,
    V94PolicyTickSourceError,
    V94RecoverableObservationHold,
)

WORKSPACE = Path(__file__).resolve().parents[2]
SHADOW21 = (
    WORKSPACE
    / "dexgrasp/runs/v94_live_readonly_clean_dkms_power_on_reuse_guard_20260722_21.npz"
)


class _Clock:
    def __init__(self, *, monotonic=10.0, realtime=1000.0):
        self.monotonic_s = float(monotonic)
        self.realtime_s = float(realtime)

    def monotonic(self):
        return self.monotonic_s

    def realtime(self):
        return self.realtime_s

    def advance(self, seconds):
        self.monotonic_s += float(seconds)
        self.realtime_s += float(seconds)


class _Provider:
    transactional_snapshot_contract = OBSERVATION_SNAPSHOT_CONTRACT

    def __init__(self, observations):
        self.observations = list(observations)
        self.calls = []
        self.index = 0

    def snapshot(
        self,
        *,
        sequence,
        now_monotonic_s,
        hard_deadline_monotonic_s,
    ):
        assert now_monotonic_s < hard_deadline_monotonic_s
        self.calls.append(int(sequence))
        if self.index >= len(self.observations):
            return self.observations[-1]
        result = self.observations[self.index]
        self.index += 1
        if isinstance(result, BaseException):
            raise result
        return result


class _Policy:
    transactional_state_contract = POLICY_TRANSACTIONAL_STATE_CONTRACT

    def __init__(
        self,
        actions,
        *,
        clock=None,
        advance_s=0.0,
        point_feature_dim=6,
        history_length=4,
        proprio_dim=67,
        controller_contract_id=None,
    ):
        self.actions = [np.asarray(value, dtype=np.float32) for value in actions]
        self.clock = clock
        self.advance_s = float(advance_s)
        self.inputs = []
        self.point_feature_dim = int(point_feature_dim)
        self.history_length = int(history_length)
        self.proprio_dim = int(proprio_dim)
        if controller_contract_id is not None:
            self.action_controller = SimpleNamespace(
                contract_id=str(controller_contract_id)
            )
            self.initial_previous_action13 = np.zeros(13, dtype=np.float32)

    def act(self, pointcloud, valid, proprio):
        self.inputs.append(
            (
                np.asarray(pointcloud).copy(),
                np.asarray(valid).copy(),
                np.asarray(proprio).copy(),
            )
        )
        if self.clock is not None and self.advance_s:
            self.clock.advance(self.advance_s)
        index = min(len(self.inputs) - 1, len(self.actions) - 1)
        return SimpleNamespace(action13=self.actions[index].copy())


class _FeedbackPolicy(_Policy):
    def __init__(self, actions):
        super().__init__(actions)
        self.feedback_commits = []

    def commit_rh56_feedback_after_dual_ack(self, sequence, feedback):
        self.feedback_commits.append((int(sequence), dict(feedback)))


class _OccludedClosurePolicy(_Policy):
    is_replay_policy = True

    def __init__(self):
        super().__init__([np.zeros(13, dtype=np.float32)])
        self._proposal = None
        self.committed_sequences = []

    def action_for_sequence(self, sequence, pointcloud, valid, proprio):
        self.inputs.append(
            (
                np.asarray(pointcloud).copy(),
                np.asarray(valid).copy(),
                np.asarray(proprio).copy(),
            )
        )
        self._proposal = int(sequence)
        return ReplayPolicyOutput(
            action13=np.zeros(13, dtype=np.float32),
            exact_franka_target_q_rad=np.zeros(7, dtype=np.float32),
            exact_rh56_angle_set_register_order=np.full(
                6, 1000, dtype=np.int32
            ),
            bypass_rh56_host_slew=True,
            replay_frame_index=0,
        )

    def action_for_occluded_closure(
        self, sequence, *, observation_hold_reason
    ):
        assert observation_hold_reason.startswith(
            "object_pointcloud_transient_invalid:"
        )
        if self.committed_sequences != [1]:
            return None
        self._proposal = int(sequence)
        return ReplayPolicyOutput(
            action13=np.full(13, 0.1, dtype=np.float32),
            exact_franka_target_q_rad=np.zeros(7, dtype=np.float32),
            exact_rh56_angle_set_register_order=np.full(
                6, 900, dtype=np.int32
            ),
            bypass_rh56_host_slew=True,
            replay_frame_index=1,
        )

    def action_for_frozen_post_closure(
        self, sequence, *, observation_hold_reason
    ):
        assert observation_hold_reason.startswith(
            "object_pointcloud_transient_invalid:"
        )
        if self.committed_sequences != [1, 2]:
            return None
        self._proposal = int(sequence)
        return ReplayPolicyOutput(
            action13=np.full(13, 0.2, dtype=np.float32),
            exact_franka_target_q_rad=np.full(
                7, 0.01, dtype=np.float32
            ),
            exact_rh56_angle_set_register_order=np.full(
                6, 800, dtype=np.int32
            ),
            bypass_rh56_host_slew=True,
            replay_frame_index=2,
        )

    def commit_replay_proposal(self, sequence):
        assert self._proposal == int(sequence)
        self.committed_sequences.append(int(sequence))
        self._proposal = None

    def discard_replay_proposal(self, sequence):
        assert self._proposal == int(sequence)
        self._proposal = None


class _OccludedClearanceLiftPolicy(_OccludedClosurePolicy):
    def action_for_occluded_closure(
        self, sequence, *, observation_hold_reason
    ):
        assert observation_hold_reason.startswith(
            "object_pointcloud_transient_invalid:"
        )
        if self.committed_sequences != [1]:
            return None
        self._proposal = int(sequence)
        target = np.zeros(7, dtype=np.float32)
        target[1] = np.float32(0.005)
        return ReplayPolicyOutput(
            action13=np.full(13, 0.1, dtype=np.float32),
            exact_franka_target_q_rad=target,
            exact_rh56_angle_set_register_order=np.full(
                6, 900, dtype=np.int32
            ),
            bypass_rh56_host_slew=True,
            replay_frame_index=1,
            occluded_clearance_lift=True,
        )


def _observation(
    clock,
    *,
    frame_id=1,
    age_s=0.01,
    prefix_value=0.0,
    measured_q=None,
    franka_age_s=0.0,
    hold_arm_target=False,
    pointcloud_status="fresh",
    pointcloud_source_frame_id=None,
    point_feature_dim=6,
    controller_state29=None,
    shaper_q_d=None,
):
    points = np.zeros((128, point_feature_dim), dtype=np.float32)
    points[:, 0] = np.linspace(-0.02, 0.02, 128, dtype=np.float32)
    points[:, 2] = np.float32(0.10)
    valid = np.ones(128, dtype=np.float32)
    return V94PolicyObservation(
        pointcloud_xyzrgb_palm=points,
        pointcloud_valid=valid,
        proprio_prefix54=np.full(54, prefix_value, dtype=np.float32),
        measured_franka_q_rad=(
            np.zeros(7, dtype=np.float32)
            if measured_q is None
            else np.asarray(measured_q, dtype=np.float32)
        ),
        franka_state_captured_monotonic_s=(clock.monotonic_s - float(franka_age_s)),
        pointcloud_captured_realtime_s=clock.realtime_s - float(age_s),
        observation_realtime_s=clock.realtime_s,
        camera_frame_id=frame_id,
        hold_arm_target=hold_arm_target,
        controller_state29=controller_state29,
        shaper_q_d_rad=shaper_q_d,
        pointcloud_status=pointcloud_status,
        source_valid_points=128,
        pointcloud_source_frame_id=pointcloud_source_frame_id,
    )


def _mapper(
    *,
    arm=None,
    hand=None,
    joint_limits=None,
    q_close=None,
    controller_contract_id=None,
):
    qd_contract = controller_contract_id == QD_G015_ACTION_CONTROLLER_CONTRACT_ID
    return TransactionalV94ActionMapper(
        initial_arm_target_q_rad=(
            np.zeros(7, dtype=np.float32)
            if arm is None
            else np.asarray(arm, dtype=np.float32)
        ),
        initial_hand_target_q_policy_order_rad=(
            np.zeros(6, dtype=np.float32)
            if hand is None
            else np.asarray(hand, dtype=np.float32)
        ),
        joint_limits_rad=(
            np.asarray([[-2.0, 2.0]] * 7, dtype=np.float32)
            if joint_limits is None
            else np.asarray(joint_limits, dtype=np.float32)
        ),
        control_dt_s=1.0 / 60.0,
        q_hand_close_rad=(
            Q_HAND_SEMANTIC_CLOSE_RAD
            if q_close is None
            else np.asarray(q_close, dtype=np.float32)
        ),
        franka_action_contract_id=(
            "v94_inspire_semantic_13d"
            if controller_contract_id is None
            else controller_contract_id
        ),
        arm_raw_gain_rad=0.15 if qd_contract else 0.015,
        target_filter_alpha=0.40 if qd_contract else 0.20,
        maximum_arm_target_step_rad=0.0 if qd_contract else 0.015,
    )


def _source(
    provider,
    policy,
    mapper,
    clock,
    *,
    rh56_command_shaper=None,
    maximum_actions_per_camera_frame=2,
    startup_non_actuated_policy_steps=0,
    maximum_object_pointcloud_age_s=0.20,
    franka_arrival_gate_enabled=False,
    franka_arrival_gate_tolerance_rad=0.005,
    franka_arrival_gate_timeout_s=0.350,
    policy_io_recorder=None,
):
    kwargs = {}
    if rh56_command_shaper is not None:
        kwargs["rh56_hardware_command_shaper"] = rh56_command_shaper
    return TransactionalBoundedV94PolicyTickSource(
        observation_provider=provider,
        policy=policy,
        action_mapper=mapper,
        maximum_actions_per_camera_frame=maximum_actions_per_camera_frame,
        startup_non_actuated_policy_steps=(
            startup_non_actuated_policy_steps
        ),
        maximum_object_pointcloud_age_s=maximum_object_pointcloud_age_s,
        franka_arrival_gate_enabled=franka_arrival_gate_enabled,
        franka_arrival_gate_tolerance_rad=(
            franka_arrival_gate_tolerance_rad
        ),
        franka_arrival_gate_timeout_s=franka_arrival_gate_timeout_s,
        policy_io_recorder=policy_io_recorder,
        monotonic=clock.monotonic,
        realtime=clock.realtime,
        **kwargs,
    )


class _PolicyIOProbe:
    def __init__(self, *, raises=False):
        self.raises = bool(raises)
        self.records = []

    def record_tick(self, **payload):
        if self.raises:
            raise RuntimeError("diagnostic recorder failure")
        self.records.append(
            {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in payload.items()
            }
        )
        return True


def _prepare(source, ledger, clock):
    snapshot = ledger.committed_snapshot()
    return source.prepare(
        sequence=snapshot.sequence + 1,
        previous_executed_action13=snapshot.executed_policy_action13,
        now_monotonic_s=clock.monotonic(),
        hard_deadline_monotonic_s=clock.monotonic() + 1.0,
    )


def _dual_commit(source, ledger, command, clock):
    assert isinstance(command, ClosedLoopCommand)
    ledger.stage(command, now_monotonic_s=clock.monotonic())
    assert not ledger.acknowledge(
        "franka", sequence=command.sequence, now_monotonic_s=clock.monotonic()
    )
    assert ledger.acknowledge(
        "rh56", sequence=command.sequence, now_monotonic_s=clock.monotonic()
    )
    source.commit(command, action_ledger=ledger)


def test_previous_action_and_history_advance_only_after_exact_dual_commit():
    clock = _Clock()
    observations = [
        _observation(clock, frame_id=1, prefix_value=1.0),
        _observation(clock, frame_id=2, prefix_value=2.0),
    ]
    first_action = np.linspace(-0.5, 0.5, 13, dtype=np.float32)
    second_action = np.linspace(0.4, -0.4, 13, dtype=np.float32)
    policy = _Policy([first_action, second_action])
    mapper = _mapper()
    source = _source(_Provider(observations), policy, mapper, clock)
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)

    first = _prepare(source, ledger, clock)
    assert isinstance(first, ClosedLoopCommand)
    assert source.committed_history() is None
    assert mapper.last_committed_sequence == 0
    np.testing.assert_array_equal(
        policy.inputs[0][2][:, 54:67],
        np.repeat(INITIAL_PREVIOUS_ACTION13[None, :], 4, axis=0),
    )
    _dual_commit(source, ledger, first, clock)
    assert source.last_committed_sequence == mapper.last_committed_sequence == 1

    clock.advance(1.0 / 60.0)
    second = _prepare(source, ledger, clock)
    assert isinstance(second, ClosedLoopCommand)
    np.testing.assert_array_equal(
        policy.inputs[1][2][-1, 54:67], first.executed_policy_action13
    )
    # The prior three frames still contain the action that was valid when
    # those logical observations were captured.
    np.testing.assert_array_equal(
        policy.inputs[1][2][:-1, 54:67],
        np.repeat(INITIAL_PREVIOUS_ACTION13[None, :], 3, axis=0),
    )
    assert source.last_committed_sequence == 1
    assert mapper.last_committed_sequence == 1
    _dual_commit(source, ledger, second, clock)
    assert source.last_committed_sequence == mapper.last_committed_sequence == 2


def test_rh56_feedback_callback_is_bound_to_exact_dual_committed_sequence():
    clock = _Clock()
    policy = _FeedbackPolicy([np.zeros(13, dtype=np.float32)])
    source = _source(
        _Provider([_observation(clock, frame_id=1)]),
        policy,
        _mapper(),
        clock,
    )
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)
    command = _prepare(source, ledger, clock)
    _dual_commit(source, ledger, command, clock)
    feedback = {
        "fresh": True,
        "forces_g": [293, 502, 289, 50, -5, -50],
        "errors": [0, 0, 0, 0, 0, 0],
    }
    source.commit_rh56_feedback_after_dual_ack(1, feedback)
    assert policy.feedback_commits == [(1, feedback)]

    with pytest.raises(
        V94PolicyTickSourceError,
        match="feedback callback differs from the committed sequence",
    ):
        source.commit_rh56_feedback_after_dual_ack(2, feedback)


def test_v258_history_bootstraps_eight_frames_with_same_sample_controller_state():
    clock = _Clock()
    controller = np.zeros(29, dtype=np.float32)
    controller[0:7] = np.float32(0.25)
    controller[28] = np.float32(1.0)
    observation = _observation(
        clock,
        frame_id=1,
        point_feature_dim=3,
        controller_state29=controller,
    )
    policy = _Policy(
        [np.zeros(13, dtype=np.float32)],
        point_feature_dim=3,
        history_length=8,
        proprio_dim=96,
    )
    source = _source(_Provider([observation]), policy, _mapper(), clock)
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)

    command = _prepare(source, ledger, clock)
    assert isinstance(command, ClosedLoopCommand)
    points, valid, proprio = policy.inputs[0]
    assert points.shape == (8, 128, 3)
    assert valid.shape == (8, 128)
    assert proprio.shape == (8, 96)
    np.testing.assert_array_equal(
        proprio[:, 67:96], np.repeat(controller[None, :], 8, axis=0)
    )


def test_v61_history_bootstraps_sixteen_frames_with_same_controller_state():
    clock = _Clock()
    controller = np.zeros(29, dtype=np.float32)
    controller[0:7] = np.float32(0.25)
    controller[28] = np.float32(1.0)
    observation = _observation(
        clock,
        frame_id=1,
        point_feature_dim=3,
        controller_state29=controller,
    )
    policy = _Policy(
        [np.zeros(13, dtype=np.float32)],
        point_feature_dim=3,
        history_length=16,
        proprio_dim=96,
    )
    source = _source(_Provider([observation]), policy, _mapper(), clock)
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)

    command = _prepare(source, ledger, clock)
    assert isinstance(command, ClosedLoopCommand)
    points, valid, proprio = policy.inputs[0]
    assert points.shape == (16, 128, 3)
    assert valid.shape == (16, 128)
    assert proprio.shape == (16, 96)
    np.testing.assert_array_equal(
        proprio[:, 67:96], np.repeat(controller[None, :], 16, axis=0)
    )


@pytest.mark.parametrize("proprio_dim", [67, 96])
def test_qd_g015_v75_v76_use_zero_reset_and_same_snapshot_qd(proprio_dim):
    clock = _Clock()
    q_d = np.asarray(
        [0.20, -0.30, 0.10, -1.00, 0.05, 1.20, 0.30], dtype=np.float32
    )
    measured = q_d - np.float32(0.01)
    controller = None
    if proprio_dim == 96:
        controller = np.zeros(29, dtype=np.float32)
        # This is the *previously held* q_cmd state supplied by the coherent
        # snapshot.  A new all-ones action must not overwrite it before the
        # policy forward is constructed.
        controller[:7] = np.linspace(-0.5, 0.5, 7, dtype=np.float32)
        controller[7:14] = np.float32(0.2)
        controller[28] = np.float32(1.0)
    observation = _observation(
        clock,
        frame_id=1,
        measured_q=measured,
        point_feature_dim=3,
        controller_state29=controller,
        shaper_q_d=q_d,
    )
    policy = _Policy(
        [np.ones(13, dtype=np.float32)],
        point_feature_dim=3,
        history_length=8,
        proprio_dim=proprio_dim,
        controller_contract_id=QD_G015_ACTION_CONTROLLER_CONTRACT_ID,
    )
    mapper = _mapper(
        arm=measured,
        controller_contract_id=QD_G015_ACTION_CONTROLLER_CONTRACT_ID,
    )
    source = _source(_Provider([observation]), policy, mapper, clock)
    ledger = ExecutedActionLedger(
        maximum_commit_latency_s=1.0,
        initial_previous_action13=np.zeros(13, dtype=np.float32),
    )

    command = _prepare(source, ledger, clock)

    assert isinstance(command, ClosedLoopCommand)
    points, _valid, proprio = policy.inputs[0]
    assert points.shape == (8, 128, 3)
    assert proprio.shape == (8, proprio_dim)
    np.testing.assert_array_equal(
        proprio[:, 54:67], np.zeros((8, 13), dtype=np.float32)
    )
    if controller is not None:
        np.testing.assert_array_equal(
            proprio[:, 67:96], np.repeat(controller[None, :], 8, axis=0)
        )
    np.testing.assert_allclose(
        command.franka_target_q_rad,
        q_d + np.float32(0.06),
        atol=2.0e-7,
        rtol=0.0,
    )


def test_qd_g015_four_startup_ticks_advance_only_policy_history_then_tick5_executes():
    clock = _Clock()
    observations = [
        _observation(
            clock,
            frame_id=index + 1,
            prefix_value=float(index + 1),
            age_s=-(index * 0.05),
            franka_age_s=-(index * 0.05),
            point_feature_dim=3,
            pointcloud_source_frame_id=index + 1,
            shaper_q_d=np.zeros(7, dtype=np.float32),
        )
        for index in range(5)
    ]
    actions = [
        np.full(13, np.float32(0.1 * (index + 1)), dtype=np.float32)
        for index in range(5)
    ]
    policy = _Policy(
        actions,
        point_feature_dim=3,
        history_length=8,
        proprio_dim=67,
        controller_contract_id=QD_G015_ACTION_CONTROLLER_CONTRACT_ID,
    )
    mapper = _mapper(
        controller_contract_id=QD_G015_ACTION_CONTROLLER_CONTRACT_ID,
    )
    source = _source(
        _Provider(observations),
        policy,
        mapper,
        clock,
        maximum_actions_per_camera_frame=3,
        startup_non_actuated_policy_steps=4,
    )
    ledger = ExecutedActionLedger(
        maximum_commit_latency_s=1.0,
        initial_previous_action13=np.zeros(13, dtype=np.float32),
    )

    for logical_index in range(4):
        primed = source.prime_non_actuated_startup_tick(
            logical_index=logical_index,
            now_monotonic_s=clock.monotonic(),
            hard_deadline_monotonic_s=clock.monotonic() + 1.0,
        )
        assert primed.logical_index == logical_index
        np.testing.assert_array_equal(
            primed.previous_policy_action13_used,
            np.zeros(13, dtype=np.float32)
            if logical_index == 0
            else actions[logical_index - 1],
        )
        np.testing.assert_array_equal(
            primed.accepted_policy_action13,
            actions[logical_index],
        )
        # No mapper proposal, hardware ledger stage, or hardware sequence is
        # allowed during the simulator-aligned first four logical ticks.
        assert mapper.last_committed_sequence == 0
        assert mapper.pending_sequence is None
        assert ledger.last_committed_sequence == 0
        assert ledger.pending_sequence is None
        clock.advance(0.05)

    assert source.last_committed_sequence == 0
    assert len(source.startup_non_actuated_records) == 4
    history = source.committed_history()
    assert history is not None
    np.testing.assert_array_equal(
        history[2][:, 0],
        np.asarray([1, 1, 1, 1, 1, 2, 3, 4], dtype=np.float32),
    )

    command = _prepare(source, ledger, clock)
    assert isinstance(command, ClosedLoopCommand)
    assert command.sequence == 1
    np.testing.assert_array_equal(
        command.previous_executed_action13_used,
        np.zeros(13, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        command.previous_policy_action13_used,
        actions[3],
    )
    np.testing.assert_array_equal(command.raw_policy_action13, actions[4])

    # The fifth policy input has the exact simulator startup layout: four
    # copies of obs0 followed by obs1..obs4, and its current previous-action
    # slot is action3.  Only action4 is mapped for the first hardware command.
    tick5_proprio = policy.inputs[4][2]
    np.testing.assert_array_equal(
        tick5_proprio[:, 0],
        np.asarray([1, 1, 1, 1, 2, 3, 4, 5], dtype=np.float32),
    )
    expected_previous = np.vstack(
        [
            np.zeros((4, 13), dtype=np.float32),
            actions[0],
            actions[1],
            actions[2],
            actions[3],
        ]
    )
    np.testing.assert_array_equal(tick5_proprio[:, 54:67], expected_previous)
    assert mapper.pending_sequence == 1
    assert ledger.pending_sequence is None


def test_policy_io_records_startup_then_only_dual_committed_hardware_tick():
    clock = _Clock()
    observations = [
        _observation(
            clock,
            frame_id=index + 1,
            prefix_value=float(index + 1),
            age_s=-(index * 0.05),
            franka_age_s=-(index * 0.05),
            point_feature_dim=3,
            pointcloud_source_frame_id=index + 1,
            shaper_q_d=np.zeros(7, dtype=np.float32),
        )
        for index in range(5)
    ]
    actions = [
        np.full(13, np.float32(0.05 * (index + 1)), dtype=np.float32)
        for index in range(5)
    ]
    recorder = _PolicyIOProbe()
    source = _source(
        _Provider(observations),
        _Policy(
            actions,
            point_feature_dim=3,
            history_length=8,
            proprio_dim=67,
            controller_contract_id=QD_G015_ACTION_CONTROLLER_CONTRACT_ID,
        ),
        _mapper(controller_contract_id=QD_G015_ACTION_CONTROLLER_CONTRACT_ID),
        clock,
        maximum_actions_per_camera_frame=3,
        startup_non_actuated_policy_steps=4,
        policy_io_recorder=recorder,
    )
    ledger = ExecutedActionLedger(
        maximum_commit_latency_s=1.0,
        initial_previous_action13=np.zeros(13, dtype=np.float32),
    )

    for logical_index in range(4):
        source.prime_non_actuated_startup_tick(
            logical_index=logical_index,
            now_monotonic_s=clock.monotonic(),
            hard_deadline_monotonic_s=clock.monotonic() + 1.0,
        )
        clock.advance(0.05)

    assert [item["logical_policy_step"] for item in recorder.records] == [0, 1, 2, 3]
    assert all(item["hardware_sequence"] == -1 for item in recorder.records)
    command = _prepare(source, ledger, clock)
    assert isinstance(command, ClosedLoopCommand)
    # A proposal is not an accepted real policy tick until both devices ACK.
    assert len(recorder.records) == 4
    _dual_commit(source, ledger, command, clock)
    assert len(recorder.records) == 5
    final = recorder.records[-1]
    assert final["logical_policy_step"] == 4
    assert final["hardware_sequence"] == 1
    assert final["startup_non_actuated"] is False
    assert final["hardware_command_valid"] is True
    np.testing.assert_array_equal(
        final["previous_policy_action13"], actions[3]
    )
    np.testing.assert_array_equal(
        final["previous_ledger_action13"], np.zeros(13, dtype=np.float32)
    )


def test_policy_io_diagnostic_exception_cannot_reject_dual_commit():
    clock = _Clock()
    source = _source(
        _Provider([_observation(clock, frame_id=1)]),
        _Policy([np.zeros(13, dtype=np.float32)]),
        _mapper(),
        clock,
        policy_io_recorder=_PolicyIOProbe(raises=True),
    )
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)
    command = _prepare(source, ledger, clock)
    _dual_commit(source, ledger, command, clock)
    assert source.last_committed_sequence == 1
    assert source.fault_reason is None


def test_qd_g015_rejects_legacy_previous_action_reset_and_missing_qd():
    clock = _Clock()
    policy = _Policy(
        [np.zeros(13, dtype=np.float32)],
        point_feature_dim=3,
        history_length=8,
        proprio_dim=67,
        controller_contract_id=QD_G015_ACTION_CONTROLLER_CONTRACT_ID,
    )
    source = _source(
        _Provider([_observation(clock, point_feature_dim=3)]),
        policy,
        _mapper(controller_contract_id=QD_G015_ACTION_CONTROLLER_CONTRACT_ID),
        clock,
    )
    legacy_ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)
    with pytest.raises(V94PolicyTickSourceError, match="initial previous action"):
        _prepare(source, legacy_ledger, clock)

    # Use a fresh source because the first mismatch is deliberately terminal.
    source = _source(
        _Provider([_observation(clock, point_feature_dim=3)]),
        policy,
        _mapper(controller_contract_id=QD_G015_ACTION_CONTROLLER_CONTRACT_ID),
        clock,
    )
    zero_ledger = ExecutedActionLedger(
        maximum_commit_latency_s=1.0,
        initial_previous_action13=np.zeros(13, dtype=np.float32),
    )
    with pytest.raises(V94PolicyTickSourceError, match="shaper_q_d_rad"):
        _prepare(source, zero_ledger, clock)


def test_closed_loop_arrival_gate_holds_without_a_second_policy_inference():
    clock = _Clock()
    provider = _Provider([_observation(clock, frame_id=1)])
    policy = _Policy(
        [
            np.asarray([1.0] * 7 + [0.0] * 6, dtype=np.float32),
            np.asarray([-1.0] * 7 + [0.0] * 6, dtype=np.float32),
        ]
    )
    source = _source(
        provider,
        policy,
        _mapper(),
        clock,
        maximum_actions_per_camera_frame=6,
        franka_arrival_gate_enabled=True,
        franka_arrival_gate_tolerance_rad=0.001,
        franka_arrival_gate_timeout_s=0.200,
    )
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)

    first = _prepare(source, ledger, clock)
    assert isinstance(first, ClosedLoopCommand)
    _dual_commit(source, ledger, first, clock)
    assert len(policy.inputs) == 1

    clock.advance(0.050)
    provider.observations.append(
        _observation(clock, frame_id=2, measured_q=np.zeros(7))
    )
    held = _prepare(source, ledger, clock)
    assert isinstance(held, BoundedPolicyTickHold)
    assert held.reason == FRANKA_ARRIVAL_GATE_HOLD_REASON
    assert held.sequence == 2
    assert len(policy.inputs) == 1
    assert ledger.last_committed_sequence == 1

    clock.advance(0.010)
    provider.observations.append(
        _observation(
            clock,
            frame_id=3,
            measured_q=first.franka_target_q_rad,
        )
    )
    second = _prepare(source, ledger, clock)
    assert isinstance(second, ClosedLoopCommand)
    assert second.sequence == 2
    assert len(policy.inputs) == 2
    gate = source.diagnostics_snapshot["franka_arrival_gate"]
    assert gate["hold_count"] == 1
    assert gate["arrived_target_count"] == 1
    assert gate["last_error_rad"] == pytest.approx(0.0)


def test_closed_loop_arrival_gate_times_out_with_axis_error():
    clock = _Clock()
    provider = _Provider([_observation(clock, frame_id=1)])
    policy = _Policy(
        [np.asarray([1.0] * 7 + [0.0] * 6, dtype=np.float32)]
    )
    source = _source(
        provider,
        policy,
        _mapper(),
        clock,
        franka_arrival_gate_enabled=True,
        franka_arrival_gate_tolerance_rad=0.001,
        franka_arrival_gate_timeout_s=0.100,
    )
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)
    first = _prepare(source, ledger, clock)
    _dual_commit(source, ledger, first, clock)

    clock.advance(0.101)
    provider.observations.append(
        _observation(clock, frame_id=2, measured_q=np.zeros(7))
    )
    with pytest.raises(V94PolicyTickSourceError, match="arrival gate timed out"):
        _prepare(source, ledger, clock)
    assert source.fault_reason is not None


def test_closed_loop_arrival_gate_latches_per_axis_target_crossings():
    clock = _Clock()
    provider = _Provider([_observation(clock, frame_id=1)])
    policy = _Policy(
        [np.asarray([1.0] * 7 + [0.0] * 6, dtype=np.float32)]
    )
    source = _source(
        provider,
        policy,
        _mapper(),
        clock,
        franka_arrival_gate_enabled=True,
        franka_arrival_gate_tolerance_rad=0.001,
        franka_arrival_gate_timeout_s=0.200,
    )
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)
    first = _prepare(source, ledger, clock)
    _dual_commit(source, ledger, first, clock)

    target = np.asarray(first.franka_target_q_rad, dtype=np.float32)
    only_first_axis_arrived = np.zeros(7, dtype=np.float32)
    only_first_axis_arrived[0] = target[0]
    clock.advance(0.020)
    assert not source.check_franka_arrival_gate(
        only_first_axis_arrived,
        now_monotonic_s=clock.monotonic(),
    )

    # Axis 1 has moved away again, but its earlier arrival stays latched while
    # the remaining six joints reach their targets at a different instant.
    remaining_axes_arrived = target.copy()
    remaining_axes_arrived[0] = 0.0
    clock.advance(0.020)
    assert source.check_franka_arrival_gate(
        remaining_axes_arrived,
        now_monotonic_s=clock.monotonic(),
    )
    gate = source.diagnostics_snapshot["franka_arrival_gate"]
    assert gate["arrived_target_count"] == 1
    assert gate["target_pending"] is False


def test_xyz_policy_receives_four_frame_128x3_history():
    clock = _Clock()
    policy = _Policy([np.zeros(13, dtype=np.float32)], point_feature_dim=3)
    source = _source(
        _Provider([_observation(clock, point_feature_dim=3)]),
        policy,
        _mapper(),
        clock,
    )
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)
    command = _prepare(source, ledger, clock)
    assert isinstance(command, ClosedLoopCommand)
    assert policy.inputs[0][0].shape == (4, 128, 3)


def test_replay_actions_advance_by_committed_sequence_not_prepare_attempt():
    clock = _Clock()
    first = np.linspace(-0.6, 0.6, 13, dtype=np.float32)
    second = -first
    replay = TransactionalReplayActionPolicy(
        ReplayActionSequence(
            actions13=np.stack([first, second]),
            sha256="0" * 64,
            source_format="test",
            declared_policy_rate_hz=60.0,
        ),
        point_feature_dim=6,
        selected_steps=2,
    )
    provider = _Provider(
        [
            V94RecoverableObservationHold(
                reason="synthetic camera wait", camera_frame_id=1
            ),
            _observation(clock, frame_id=2),
            _observation(clock, frame_id=3),
        ]
    )
    source = _source(provider, replay, _mapper(), clock)
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)

    held = _prepare(source, ledger, clock)
    assert isinstance(held, BoundedPolicyTickHold)
    command1 = _prepare(source, ledger, clock)
    np.testing.assert_array_equal(command1.raw_policy_action13, first)
    _dual_commit(source, ledger, command1, clock)
    clock.advance(1.0 / 60.0)
    command2 = _prepare(source, ledger, clock)
    np.testing.assert_array_equal(command2.raw_policy_action13, second)


def test_replay_bundle_exact_targets_bypass_action_mapper():
    clock = _Clock()
    action = np.linspace(-1.0, 1.0, 13, dtype=np.float32)
    exact_arm = np.asarray(
        [0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07], dtype=np.float32
    )
    exact_hand = np.asarray([1000, 800, 600, 400, 200, 0], dtype=np.int32)
    replay = TransactionalReplayActionPolicy(
        ReplayActionSequence(
            actions13=action[None, :],
            sha256="0" * 64,
            source_format="test_exact_targets",
            declared_policy_rate_hz=60.0,
            recorded_franka_target_q_rad=exact_arm[None, :],
            recorded_rh56_angle_set_register_order=exact_hand[None, :],
        ),
        point_feature_dim=6,
        selected_steps=1,
    )
    source = _source(
        _Provider([_observation(clock)]), replay, _mapper(), clock
    )
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)
    command = _prepare(source, ledger, clock)
    np.testing.assert_array_equal(command.raw_policy_action13, action)
    np.testing.assert_allclose(command.franka_target_q_rad, exact_arm)
    np.testing.assert_array_equal(
        command.rh56_angle_set_register_order, exact_hand
    )


def test_exact_arrival_replay_arm_hold_does_not_skip_unsent_waypoint():
    clock = _Clock()
    targets = np.zeros((3, 7), dtype=np.float32)
    targets[1, :] = np.float32(0.012)
    targets[2, :] = np.float32(0.024)
    replay = TransactionalReplayActionPolicy(
        ReplayActionSequence(
            actions13=np.zeros((3, 13), dtype=np.float32),
            sha256="0" * 64,
            source_format="test_exact_arrival_targets",
            declared_policy_rate_hz=20.0,
            recorded_franka_target_q_rad=targets,
            recorded_rh56_angle_set_register_order=np.full(
                (3, 6), 1000, dtype=np.int32
            ),
        ),
        point_feature_dim=6,
        selected_steps=3,
        arrival_gated=True,
        arrival_tolerance_rad=0.005,
        q_home_rad=np.zeros(7, dtype=np.float32),
    )
    source = _source(
        _Provider(
            [
                _observation(clock, frame_id=1),
                _observation(clock, frame_id=2, hold_arm_target=True),
                _observation(clock, frame_id=3),
                _observation(
                    clock,
                    frame_id=4,
                    prefix_value=0.012,
                    measured_q=targets[1],
                ),
            ]
        ),
        replay,
        _mapper(),
        clock,
    )
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)

    first = _prepare(source, ledger, clock)
    np.testing.assert_array_equal(first.franka_target_q_rad, targets[0])
    _dual_commit(source, ledger, first, clock)

    held = _prepare(source, ledger, clock)
    assert isinstance(held, BoundedPolicyTickHold)
    assert held.reason == EXACT_REPLAY_ARM_HOLD_REASON
    assert source.pending_sequence is None
    assert replay.completed_replay_frames == 1

    second = _prepare(source, ledger, clock)
    np.testing.assert_array_equal(second.franka_target_q_rad, targets[1])
    _dual_commit(source, ledger, second, clock)
    third = _prepare(source, ledger, clock)
    np.testing.assert_array_equal(third.franka_target_q_rad, targets[2])
    assert float(
        np.max(np.abs(third.franka_target_q_rad - second.franka_target_q_rad))
    ) == pytest.approx(0.012)


def test_rh56_hardware_target_is_latest_only_20hz_hold_without_changing_policy_chain():
    clock = _Clock()
    observations = [
        _observation(
            clock,
            frame_id=index + 1,
            franka_age_s=-(index * (1.0 / 60.0)),
        )
        for index in range(4)
    ]
    actions = [
        np.asarray([0.0] * 7 + [value] * 6, dtype=np.float32)
        for value in (1.0, -1.0, -1.0, 1.0)
    ]
    shaped_policy = _Policy(actions)
    baseline_policy = _Policy(actions)
    shaper = TransactionalRH56HardwareCommandShaper(
        initial_angle_set_register_order=np.full(6, 1000, dtype=np.int32),
        policy_rate_hz=60.0,
        hardware_rate_hz=20.0,
        maximum_register_delta_per_update=np.full(6, 1000, dtype=np.int32),
    )
    shaped = _source(
        _Provider(observations),
        shaped_policy,
        _mapper(),
        clock,
        rh56_command_shaper=shaper,
    )
    baseline = _source(
        _Provider(observations),
        baseline_policy,
        _mapper(),
        clock,
    )
    shaped_ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)
    baseline_ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)
    shaped_commands = []
    baseline_commands = []

    for _ in range(4):
        shaped_command = _prepare(shaped, shaped_ledger, clock)
        baseline_command = _prepare(baseline, baseline_ledger, clock)
        shaped_commands.append(shaped_command)
        baseline_commands.append(baseline_command)
        _dual_commit(shaped, shaped_ledger, shaped_command, clock)
        _dual_commit(baseline, baseline_ledger, baseline_command, clock)
        clock.advance(1.0 / 60.0)

    # The policy and its previous-action chain remain exactly the transferred
    # 60 Hz contract.  Only the physical RH56 register target is sample-held.
    for shaped_command, baseline_command in zip(shaped_commands, baseline_commands):
        np.testing.assert_array_equal(
            shaped_command.raw_policy_action13,
            baseline_command.raw_policy_action13,
        )
        np.testing.assert_array_equal(
            shaped_command.executed_policy_action13,
            baseline_command.executed_policy_action13,
        )
        np.testing.assert_array_equal(
            shaped_command.previous_executed_action13_used,
            baseline_command.previous_executed_action13_used,
        )

    first_target = shaped_commands[0].rh56_angle_set_register_order
    np.testing.assert_array_equal(
        shaped_commands[1].rh56_angle_set_register_order,
        first_target,
    )
    np.testing.assert_array_equal(
        shaped_commands[2].rh56_angle_set_register_order,
        first_target,
    )
    # The next 20 Hz release uses sequence 4's newest proposal, not either
    # stale intermediate request from sequences 2 and 3.
    np.testing.assert_array_equal(
        shaped_commands[3].rh56_angle_set_register_order,
        baseline_commands[3].rh56_angle_set_register_order,
    )
    assert not np.array_equal(
        shaped_commands[3].rh56_angle_set_register_order,
        first_target,
    )


def test_partial_ack_abort_does_not_pollute_history_or_mapper_commit():
    clock = _Clock()
    policy = _Policy([np.zeros(13, dtype=np.float32)])
    mapper = _mapper()
    source = _source(_Provider([_observation(clock)]), policy, mapper, clock)
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)
    command = _prepare(source, ledger, clock)
    assert isinstance(command, ClosedLoopCommand)
    ledger.stage(command, now_monotonic_s=clock.monotonic())
    assert not ledger.acknowledge(
        "franka", sequence=1, now_monotonic_s=clock.monotonic()
    )

    source.abort(command, reason="synthetic RH56 failure after Franka ACK")

    assert source.committed_history() is None
    assert source.last_committed_sequence == 0
    assert mapper.last_committed_sequence == 0
    assert mapper.pending_sequence == 1
    assert source.fault_reason is not None
    with pytest.raises(V94PolicyTickSourceError, match="fault is latched"):
        _prepare(source, ledger, clock)


def test_third_same_camera_frame_is_explicit_hold_without_stage_or_policy_call():
    clock = _Clock()
    observations = [
        _observation(clock, frame_id=41, prefix_value=1.0),
        _observation(
            clock,
            frame_id=41,
            prefix_value=2.0,
            franka_age_s=-(1.0 / 60.0),
        ),
        _observation(
            clock,
            frame_id=41,
            prefix_value=3.0,
            franka_age_s=-(2.0 / 60.0),
        ),
        _observation(
            clock,
            frame_id=42,
            prefix_value=4.0,
            franka_age_s=-(2.0 / 60.0 + 0.001),
        ),
    ]
    policy = _Policy([np.zeros(13, dtype=np.float32)] * 3)
    mapper = _mapper()
    source = _source(_Provider(observations), policy, mapper, clock)
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)

    for _ in range(2):
        command = _prepare(source, ledger, clock)
        _dual_commit(source, ledger, command, clock)
        clock.advance(1.0 / 60.0)

    history_before = source.committed_history()
    previous_before = ledger.previous_executed_action13()
    held = _prepare(source, ledger, clock)
    assert isinstance(held, BoundedPolicyTickHold)
    assert held.sequence == 3
    assert held.camera_frame_id == 41
    assert held.reason == CAMERA_REUSE_HOLD_REASON
    assert held.policy_state_mutated is False
    assert held.action_staged is False
    assert len(policy.inputs) == 2
    assert mapper.pending_sequence is None
    assert ledger.pending_sequence is None
    np.testing.assert_array_equal(ledger.previous_executed_action13(), previous_before)
    for before, after in zip(history_before, source.committed_history()):
        np.testing.assert_array_equal(after, before)

    clock.advance(source.hold_retry_interval_s)
    third = _prepare(source, ledger, clock)
    assert isinstance(third, ClosedLoopCommand)
    assert third.sequence == 3
    assert len(policy.inputs) == 3


def test_fresh_camera_frame_can_bridge_six_policy_ticks_before_reuse_hold():
    clock = _Clock()
    observations = [
        _observation(
            clock,
            frame_id=51,
            prefix_value=float(index),
            franka_age_s=-(index * (1.0 / 60.0)),
        )
        for index in range(7)
    ]
    policy = _Policy([np.zeros(13, dtype=np.float32)] * 6)
    mapper = _mapper()
    source = _source(
        _Provider(observations),
        policy,
        mapper,
        clock,
        maximum_actions_per_camera_frame=6,
    )
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)

    for _ in range(6):
        command = _prepare(source, ledger, clock)
        assert isinstance(command, ClosedLoopCommand)
        _dual_commit(source, ledger, command, clock)
        clock.advance(1.0 / 60.0)

    held = _prepare(source, ledger, clock)
    assert isinstance(held, BoundedPolicyTickHold)
    assert held.sequence == 7
    assert held.camera_frame_id == 51
    assert held.reason == CAMERA_REUSE_HOLD_REASON
    assert source.last_committed_sequence == 6
    assert len(policy.inputs) == 6
    assert ledger.pending_sequence is None


def test_pointcloud_expiring_during_inference_holds_and_rolls_back_candidate():
    clock = _Clock()
    observation = _observation(clock, frame_id=8, age_s=0.199)
    policy = _Policy(
        [np.zeros(13, dtype=np.float32)],
        clock=clock,
        advance_s=0.002,
    )
    mapper = _mapper()
    source = _source(_Provider([observation]), policy, mapper, clock)
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)

    held = _prepare(source, ledger, clock)

    assert isinstance(held, BoundedPolicyTickHold)
    assert held.reason == OBJECT_POINTCLOUD_STALE_HOLD_REASON
    assert len(policy.inputs) == 1
    assert source.committed_history() is None
    assert source.pending_sequence is None
    assert mapper.pending_sequence is None
    assert ledger.pending_sequence is None


def test_fresh_pointcloud_just_over_100ms_is_accepted_in_200ms_window():
    clock = _Clock()
    observation = _observation(clock, frame_id=9, age_s=0.103)
    source = _source(
        _Provider([observation]),
        _Policy([np.zeros(13, dtype=np.float32)]),
        _mapper(),
        clock,
    )
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)

    command = _prepare(source, ledger, clock)

    assert isinstance(command, ClosedLoopCommand)
    assert source.pending_sequence == 1
    _dual_commit(source, ledger, command, clock)
    assert source.last_committed_frame_id == 9


def test_advancing_camera_frames_do_not_hide_stale_cloud_reuse_limit():
    clock = _Clock()
    observations = [
        _observation(
            clock,
            frame_id=100 + index,
            age_s=12.0,
            franka_age_s=-(index * (1.0 / 60.0)),
            pointcloud_status="stale_palm",
            pointcloud_source_frame_id=7,
        )
        for index in range(3)
    ]
    policy = _Policy([np.zeros(13, dtype=np.float32)] * 2)
    source = _source(
        _Provider(observations),
        policy,
        _mapper(),
        clock,
        maximum_actions_per_camera_frame=2,
    )
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)

    for expected_frame in (100, 101):
        command = _prepare(source, ledger, clock)
        assert isinstance(command, ClosedLoopCommand)
        _dual_commit(source, ledger, command, clock)
        assert source.last_committed_frame_id == expected_frame
        clock.advance(1.0 / 60.0)

    held = _prepare(source, ledger, clock)
    assert isinstance(held, BoundedPolicyTickHold)
    assert held.reason == CAMERA_REUSE_HOLD_REASON
    assert held.camera_frame_id == 102
    assert len(policy.inputs) == 2
    assert source.hold_count == 1
    assert observations[0].pointcloud_source_frame_id == 7
    assert observations[1].pointcloud_source_frame_id == 7
    assert observations[2].pointcloud_source_frame_id == 7


def test_stale_palm_cannot_hide_a_frozen_current_camera_frame():
    clock = _Clock()
    observations = [
        _observation(
            clock,
            frame_id=100,
            age_s=12.0,
            franka_age_s=-(index * (1.0 / 60.0)),
            pointcloud_status="stale_palm",
            pointcloud_source_frame_id=7,
        )
        for index in range(3)
    ]
    policy = _Policy([np.zeros(13, dtype=np.float32)] * 2)
    source = _source(
        _Provider(observations),
        policy,
        _mapper(),
        clock,
        maximum_actions_per_camera_frame=2,
    )
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)

    for _ in range(2):
        command = _prepare(source, ledger, clock)
        assert isinstance(command, ClosedLoopCommand)
        _dual_commit(source, ledger, command, clock)
        clock.advance(1.0 / 60.0)

    held = _prepare(source, ledger, clock)
    assert isinstance(held, BoundedPolicyTickHold)
    assert held.reason == CAMERA_REUSE_HOLD_REASON
    assert held.camera_frame_id == 100
    assert len(policy.inputs) == 2


def test_franka_state_between_action_and_hard_age_holds_without_policy_or_stage():
    clock = _Clock()
    observation = _observation(clock, frame_id=18, franka_age_s=0.0256)
    policy = _Policy([np.zeros(13, dtype=np.float32)])
    mapper = _mapper()
    source = _source(_Provider([observation]), policy, mapper, clock)
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)

    held = _prepare(source, ledger, clock)

    assert isinstance(held, BoundedPolicyTickHold)
    assert held.reason == FRANKA_STATE_STALE_HOLD_REASON
    assert len(policy.inputs) == 0
    assert source.committed_history() is None
    assert source.pending_sequence is None
    assert mapper.pending_sequence is None
    assert ledger.pending_sequence is None


def test_franka_state_expiring_during_inference_holds_before_mapping():
    clock = _Clock()
    observation = _observation(clock, frame_id=19, franka_age_s=0.024)
    policy = _Policy(
        [np.zeros(13, dtype=np.float32)],
        clock=clock,
        advance_s=0.002,
    )
    mapper = _mapper()
    source = _source(_Provider([observation]), policy, mapper, clock)
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)

    held = _prepare(source, ledger, clock)

    assert isinstance(held, BoundedPolicyTickHold)
    assert held.reason == FRANKA_STATE_STALE_HOLD_REASON
    assert len(policy.inputs) == 1
    assert source.committed_history() is None
    assert source.pending_sequence is None
    assert mapper.pending_sequence is None
    assert ledger.pending_sequence is None


def test_franka_state_beyond_hard_age_holds_without_policy_or_stage():
    clock = _Clock()
    observation = _observation(clock, frame_id=20, franka_age_s=0.0501)
    source = _source(
        _Provider([observation]),
        _Policy([np.zeros(13, dtype=np.float32)]),
        _mapper(),
        clock,
    )
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)

    held = _prepare(source, ledger, clock)

    assert isinstance(held, BoundedPolicyTickHold)
    assert held.reason == FRANKA_STATE_STALE_HOLD_REASON
    assert source.pending_sequence is None
    assert ledger.pending_sequence is None


def test_franka_state_beyond_hard_age_retries_same_sequence_on_fresh_state():
    clock = _Clock()
    source = _source(
        _Provider(
            [
                _observation(clock, frame_id=20, franka_age_s=0.058967),
                _observation(
                    clock,
                    frame_id=21,
                    franka_age_s=-0.010,
                ),
            ]
        ),
        _Policy([np.zeros(13, dtype=np.float32)]),
        _mapper(),
        clock,
    )
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)

    held = _prepare(source, ledger, clock)
    assert isinstance(held, BoundedPolicyTickHold)
    assert held.sequence == 1
    assert source.pending_sequence is None
    assert ledger.pending_sequence is None

    clock.advance(source.hold_retry_interval_s)
    recovered = _prepare(source, ledger, clock)
    assert isinstance(recovered, ClosedLoopCommand)
    assert recovered.sequence == 1


def test_transient_invalid_object_frame_holds_then_new_frame_recovers():
    clock = _Clock()
    transient = V94RecoverableObservationHold(
        camera_frame_id=17,
        reason="object_pointcloud_transient_invalid:source_points=0",
    )
    policy = _Policy([np.zeros(13, dtype=np.float32)])
    mapper = _mapper()
    source = _source(
        _Provider([transient, _observation(clock, frame_id=18)]),
        policy,
        mapper,
        clock,
    )
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)

    held = _prepare(source, ledger, clock)
    assert isinstance(held, BoundedPolicyTickHold)
    assert held.sequence == 1
    assert held.camera_frame_id == 17
    assert "object_pointcloud_transient_invalid" in held.reason
    assert source.fault_reason is None
    assert source.pending_sequence is None
    assert mapper.pending_sequence is None
    assert ledger.pending_sequence is None
    assert policy.inputs == []

    clock.advance(source.hold_retry_interval_s)
    recovered = _prepare(source, ledger, clock)
    assert isinstance(recovered, ClosedLoopCommand)
    assert recovered.sequence == 1
    assert len(policy.inputs) == 1


def test_occluded_closure_advances_hand_only_without_consuming_stale_history():
    clock = _Clock()
    closure_transient = V94RecoverableObservationHold(
        camera_frame_id=2,
        reason=(
            "object_pointcloud_transient_invalid:"
            "status=invalid_motion_fallback_rejected"
        ),
    )
    lift_transient = V94RecoverableObservationHold(
        camera_frame_id=3,
        reason=(
            "object_pointcloud_transient_invalid:"
            "status=invalid_motion_fallback_rejected"
        ),
        fresh_actuator_snapshot=V94FreshActuatorSnapshot(
            measured_franka_q_rad=np.zeros(7, dtype=np.float32),
            franka_state_captured_monotonic_s=clock.monotonic_s + 0.10,
            observation_realtime_s=clock.realtime_s + 0.10,
            hold_arm_target=False,
        ),
    )
    policy = _OccludedClosurePolicy()
    mapper = _mapper()
    source = _source(
        _Provider(
            [
                _observation(clock, frame_id=1),
                closure_transient,
                lift_transient,
                closure_transient,
            ]
        ),
        policy,
        mapper,
        clock,
    )
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)

    first = _prepare(source, ledger, clock)
    assert isinstance(first, ClosedLoopCommand)
    _dual_commit(source, ledger, first, clock)
    history_before = source.committed_history()
    assert history_before is not None
    assert source.last_committed_frame_id == 1

    clock.advance(0.05)
    completion = _prepare(source, ledger, clock)
    assert isinstance(completion, ClosedLoopCommand)
    assert completion.hold_arm_target is True
    np.testing.assert_array_equal(
        completion.franka_target_q_rad, first.franka_target_q_rad
    )
    np.testing.assert_array_equal(
        completion.rh56_angle_set_register_order,
        np.full(6, 900, dtype=np.int32),
    )
    assert source.last_committed_sequence == 1
    assert source.last_committed_frame_id == 1
    _dual_commit(source, ledger, completion, clock)
    assert source.last_committed_sequence == 2
    assert source.last_committed_frame_id == 1
    history_after = source.committed_history()
    assert history_after is not None
    for before, after in zip(history_before, history_after):
        np.testing.assert_array_equal(after, before)
    assert policy.committed_sequences == [1, 2]

    # A fresh Franka owner snapshot may advance the already frozen lift, but
    # it still consumes no stale object history.
    clock.advance(0.05)
    frozen_lift = _prepare(source, ledger, clock)
    assert isinstance(frozen_lift, ClosedLoopCommand)
    assert frozen_lift.hold_arm_target is False
    np.testing.assert_array_equal(
        frozen_lift.franka_target_q_rad,
        np.full(7, 0.01, dtype=np.float32),
    )
    _dual_commit(source, ledger, frozen_lift, clock)
    assert source.last_committed_sequence == 3
    assert source.last_committed_frame_id == 3
    history_after_lift = source.committed_history()
    assert history_after_lift is not None
    for before, after in zip(history_before, history_after_lift):
        np.testing.assert_array_equal(after, before)

    # The fake policy has no further reviewed post-closure step.  The same
    # invalid observation therefore returns to the ordinary no-stage hold.
    clock.advance(0.05)
    held = _prepare(source, ledger, clock)
    assert isinstance(held, BoundedPolicyTickHold)
    assert source.pending_sequence is None
    assert mapper.pending_sequence is None


def test_occluded_closure_clearance_lift_uses_fresh_franka_without_stale_cloud():
    clock = _Clock()
    clearance_transient = V94RecoverableObservationHold(
        camera_frame_id=2,
        reason=(
            "object_pointcloud_transient_invalid:"
            "status=invalid_motion_fallback_rejected"
        ),
        fresh_actuator_snapshot=V94FreshActuatorSnapshot(
            measured_franka_q_rad=np.zeros(7, dtype=np.float32),
            franka_state_captured_monotonic_s=clock.monotonic_s + 0.05,
            observation_realtime_s=clock.realtime_s + 0.05,
            hold_arm_target=False,
        ),
    )
    policy = _OccludedClearanceLiftPolicy()
    mapper = _mapper()
    source = _source(
        _Provider(
            [_observation(clock, frame_id=1), clearance_transient]
        ),
        policy,
        mapper,
        clock,
    )
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)
    first = _prepare(source, ledger, clock)
    assert isinstance(first, ClosedLoopCommand)
    _dual_commit(source, ledger, first, clock)
    history_before = source.committed_history()
    assert history_before is not None

    clock.advance(0.05)
    completion = _prepare(source, ledger, clock)
    assert isinstance(completion, ClosedLoopCommand)
    assert completion.hold_arm_target is False
    assert completion.franka_target_q_rad[1] == pytest.approx(0.005)
    _dual_commit(source, ledger, completion, clock)
    history_after = source.committed_history()
    assert history_after is not None
    for before, after in zip(history_before, history_after):
        np.testing.assert_array_equal(after, before)
    assert source.last_committed_frame_id == 1
    assert policy.committed_sequences == [1, 2]


def test_hold_arm_is_explicit_in_command_and_preserves_prior_arm_target():
    clock = _Clock()
    initial_arm = np.full(7, 0.2, dtype=np.float32)
    observation = _observation(
        clock,
        frame_id=1,
        measured_q=initial_arm,
        hold_arm_target=True,
    )
    action = np.asarray([1.0] * 7 + [0.0] * 6, dtype=np.float32)
    source = _source(
        _Provider([observation]),
        _Policy([action]),
        _mapper(arm=initial_arm),
        clock,
    )
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)

    command = _prepare(source, ledger, clock)

    assert isinstance(command, ClosedLoopCommand)
    assert command.hold_arm_target is True
    np.testing.assert_allclose(
        command.franka_target_q_rad, initial_arm, atol=2.0e-8, rtol=0.0
    )


def test_uncertified_stateful_policy_and_mutating_provider_fail_closed():
    clock = _Clock()
    provider = _Provider([_observation(clock)])
    policy = _Policy([np.zeros(13, dtype=np.float32)])
    mapper = _mapper()
    policy.transactional_state_contract = "persistent_hidden_state_without_rollback"
    with pytest.raises(ValueError, match="cannot be rolled back"):
        _source(provider, policy, mapper, clock)

    policy.transactional_state_contract = POLICY_TRANSACTIONAL_STATE_CONTRACT
    provider.transactional_snapshot_contract = "mutates_velocity_history_on_snapshot"
    with pytest.raises(ValueError, match="not certified side-effect-free"):
        _source(provider, policy, _mapper(), clock)


@pytest.fixture(scope="module")
def _bundle_assets():
    bundle = DeployBundle(WORKSPACE / "data/test_fixtures/sim2real/deploy.zip")
    bundle.verify()
    contract = V94Contract.from_bundle(bundle)
    policy = RollingStudentPolicy(load_checkpoint_safely(bundle.checkpoint_bytes()))
    closed = bundle.load_npz(
        "alignment/student_closed_loop/closed_loop_observation_action_pairs.npz"
    )
    return bundle, contract, policy, closed


def test_shadow21_first_real_observation_produces_recorded_action_and_targets(
    _bundle_assets,
):
    _bundle, contract, policy, _closed = _bundle_assets
    with np.load(SHADOW21, allow_pickle=False) as archive:
        shadow = {name: archive[name].copy() for name in archive.files}
    index = 0
    clock = _Clock(
        monotonic=float(shadow["host_action_monotonic_s"][index]),
        realtime=float(shadow["host_action_realtime_s"][index]),
    )
    observation = V94PolicyObservation(
        pointcloud_xyzrgb_palm=shadow["pointcloud_current_xyzrgb_palm"][index],
        pointcloud_valid=shadow["pointcloud_current_valid"][index],
        proprio_prefix54=shadow["proprio_current67"][index, :54],
        measured_franka_q_rad=shadow["franka_q_rad"][index],
        franka_state_captured_monotonic_s=clock.monotonic_s,
        pointcloud_captured_realtime_s=float(shadow["pointcloud_timestamp_s"][index]),
        observation_realtime_s=float(shadow["host_observation_realtime_s"][index]),
        camera_frame_id=int(shadow["pointcloud_frame_id"][index]),
        hold_arm_target=False,
        pointcloud_status=str(shadow["pointcloud_status"][index]),
        source_valid_points=int(shadow["pointcloud_source_valid_points"][index]),
    )
    mapper = _mapper(
        arm=shadow["franka_q_rad"][index],
        hand=shadow["rh56_virtual_q_policy_order_rad"][index],
        joint_limits=contract.joint_limits_rad,
        q_close=contract.q_hand_close_rad,
    )
    source = _source(_Provider([observation]), policy, mapper, clock)
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)

    command = _prepare(source, ledger, clock)

    assert isinstance(command, ClosedLoopCommand)
    np.testing.assert_allclose(
        command.raw_policy_action13,
        shadow["raw_policy_action13"][index],
        atol=2.0e-6,
        rtol=0.0,
    )
    np.testing.assert_array_equal(
        command.previous_executed_action13_used,
        shadow["proprio_current67"][index, 54:67],
    )
    np.testing.assert_allclose(
        command.executed_policy_action13,
        shadow["executed_policy_action13"][index],
        atol=2.0e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        command.franka_target_q_rad,
        shadow["franka_target_q_rad"][index],
        atol=5.0e-7,
        rtol=0.0,
    )
    np.testing.assert_array_equal(
        command.rh56_angle_set_register_order,
        shadow["rh56_proposed_angle_set_register_order"][index],
    )


def test_packaged_closed_loop_replay_matches_policy_mapper_and_previous_chain(
    _bundle_assets,
):
    _bundle, contract, policy, closed = _bundle_assets
    count = int(closed["episode_step"].shape[0])
    clock = _Clock()
    observations = []
    for index in range(count):
        sample_realtime_s = clock.realtime_s + index * (1.0 / 60.0)
        observations.append(
            V94PolicyObservation(
                pointcloud_xyzrgb_palm=closed["current_pointcloud_xyzrgb_palm"][index],
                pointcloud_valid=closed["current_pointcloud_valid"][index],
                proprio_prefix54=closed["current_proprio67"][index, :54],
                measured_franka_q_rad=closed["franka_measured_q_rad"][index],
                franka_state_captured_monotonic_s=(
                    clock.monotonic_s + index * (1.0 / 60.0)
                ),
                pointcloud_captured_realtime_s=sample_realtime_s,
                observation_realtime_s=sample_realtime_s,
                camera_frame_id=index,
                hold_arm_target=False,
                pointcloud_status="packaged_closed_loop",
                source_valid_points=int(
                    np.sum(closed["current_pointcloud_valid"][index])
                ),
            )
        )
    mapper = _mapper(
        arm=closed["franka_measured_q_rad"][0],
        hand=closed["rh56_virtual_q_policy_order_rad"][0],
        joint_limits=contract.joint_limits_rad,
        q_close=contract.q_hand_close_rad,
    )
    # The transferred closed-loop observation chain contains the simulator's
    # executed (GPU) action in the next proprio frame.  Replay those exact
    # recorded actions here so this test isolates the transaction/target
    # mapping.  The preceding shadow21 test exercises the real NumPy policy.
    recorded_policy = _Policy(closed["policy_action13"])
    source = _source(_Provider(observations), recorded_policy, mapper, clock)
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)

    for index in range(count):
        command = _prepare(source, ledger, clock)
        assert isinstance(command, ClosedLoopCommand)
        np.testing.assert_allclose(
            command.raw_policy_action13,
            closed["policy_action13"][index],
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_array_equal(
            command.previous_executed_action13_used,
            closed["current_proprio67"][index, 54:67],
        )
        np.testing.assert_allclose(
            command.franka_target_q_rad,
            closed["franka_target_q_rad"][index],
            atol=5.0e-7,
            rtol=0.0,
        )
        np.testing.assert_array_equal(
            command.rh56_angle_set_register_order,
            closed["rh56_angle_set_register_order"][index],
        )
        _dual_commit(source, ledger, command, clock)
        clock.advance(1.0 / 60.0)

    assert source.last_committed_sequence == count
    assert mapper.last_committed_sequence == count
    assert ledger.last_committed_sequence == count


def test_packaged_rh56_targets_are_released_exactly_at_20hz(_bundle_assets):
    _bundle, _contract, _policy, closed = _bundle_assets
    np.testing.assert_array_equal(
        closed["episode_step"],
        np.arange(1, 12, dtype=np.int64),
    )
    desired_targets = np.asarray(
        closed["rh56_angle_set_register_order"],
        dtype=np.int32,
    )
    envelope = np.asarray([137, 143, 158, 158, 251, 120], dtype=np.int32)
    shaper = TransactionalRH56HardwareCommandShaper(
        initial_angle_set_register_order=np.full(6, 1000, dtype=np.int32),
        policy_rate_hz=60.0,
        hardware_rate_hz=20.0,
        maximum_register_delta_per_update=envelope,
        minimum_angle_set_register_order=(0, 0, 0, 0, 0, 416),
        maximum_angle_set_register_order=(1000,) * 6,
    )
    latest_release_indices = (0, 0, 0, 3, 3, 3, 6, 6, 6, 9, 9)
    release_sequences = {1, 4, 7, 10}

    for sequence, (desired, release_index) in enumerate(
        zip(desired_targets, latest_release_indices),
        start=1,
    ):
        proposal = shaper.propose(sequence, desired)
        np.testing.assert_array_equal(
            proposal.angle_set_register_order,
            desired_targets[release_index],
        )
        assert proposal.hardware_update_due is (sequence in release_sequences)
        if proposal.hardware_update_due:
            np.testing.assert_array_equal(
                proposal.angle_set_register_order,
                proposal.bounded_desired_angle_set_register_order,
            )
        shaper.commit(proposal)

    np.testing.assert_array_equal(
        shaper.committed_target(),
        desired_targets[9],
    )
