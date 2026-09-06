import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sim2real.closed_loop_core import (
    ClosedLoopCommand,
    ClosedLoopSafetyGate,
    ExecutedActionLedger,
    MotionAuthorization,
    SafetyState,
)
from robot_control.franka.session import (
    CommissionedFrankaEnvelope,
    FrankaJointTarget,
    FrankaPersistentSession,
    FrankaPersistentSessionError,
    FrankaSessionMode,
    FrankaSessionState,
    FrankaTargetSampleHold,
    FrankaTargetSource,
    SUPERVISED_BOOTSTRAP_READ_TO_WRITE_DEADLINE_S,
    _issue_supervised_franka_preflight_token,
    load_commissioned_franka_envelope,
    load_experimental_supervised_franka_envelope,
    verified_franka_preflight_token_from_report,
)
from sim2real.contracts.actions import (
    LEGACY_FRANKA_ACTION_CONTRACT_ID,
    QD_G015_FRANKA_ACTION_CONTRACT_ID,
)
from sim2real.contracts.v94 import INITIAL_PREVIOUS_ACTION13


class _Clock:
    def __init__(self, value=1.0):
        self.value = float(value)

    def __call__(self):
        return self.value


def _profile_payload(**session_overrides):
    session = {
        "initial_target_tolerance_rad": 0.01,
        "target_reached_tolerance_rad": 0.005,
        "control_period_min_s": 0.0005,
        "control_period_max_s": 0.002,
        "policy_command_max_age_s": 0.1,
        "read_to_write_deadline_s": 0.0004,
        "minimum_control_success_rate": 0.9,
        "maximum_session_duration_s": 1.0,
        "stop_maximum_velocity_rad_s": 0.01,
        "stop_consecutive_samples": 2,
        "stop_maximum_samples": 3,
        "pose_ring_capacity": 8,
        "allow_one_initial_zero_period": True,
        "F_T_EE_tolerance": 1.0e-8,
        "mass_tolerance_kg": 1.0e-5,
        "center_of_mass_tolerance_m": 1.0e-6,
        "inertia_tolerance_kg_m2": 1.0e-6,
        "expected_external_load": {
            "mass_kg": 0.0,
            "F_x_Cload_m": [0.0, 0.0, 0.0],
            "inertia_kg_m2": np.zeros((3, 3)).tolist(),
        },
    }
    session.update(session_overrides)
    return {
        "schema_version": 1,
        "franka": {
            "joint_limits_rad": [[-2.0, 2.0]] * 7,
            "joint_limit_margin_rad": 0.1,
            "default_max_joint_velocity_rad_s": 1.0,
            "online_max_joint_acceleration_rad_s2": 100.0,
            "online_max_joint_jerk_rad_s3": 100000.0,
            "online_max_tracking_error_rad": 0.1,
            "expected_F_T_EE": np.eye(4).tolist(),
            "expected_end_effector": {
                "mass_kg": 0.6,
                "F_x_Cee_m": [0.0, 0.0, 0.05],
                "inertia_kg_m2": np.diag([0.01, 0.01, 0.005]).tolist(),
            },
            "persistent_session": session,
        },
    }


def _write_profile(tmp_path: Path, payload=None):
    path = tmp_path / "commissioning.json"
    path.write_text(json.dumps(payload or _profile_payload()), encoding="utf-8")
    return path


_REQUIRED_CHECKS = (
    "evidence_installed_payload_dynamics",
    "evidence_installed_fr3_adapter_rh56_collision_model",
    "evidence_physical_deadman_acceptance",
    "evidence_physical_emergency_stop_acceptance",
    "evidence_franka_persistent_fci_1khz_session",
    "evidence_franka_single_owner_state_source",
    "evidence_franka_online_velocity_acceleration_jerk_limits",
    "evidence_franka_command_watchdog_and_fault_stop",
    "evidence_franka_tracking_and_collision_monitoring",
    "evidence_metric_franka_control_loop_rate_hz",
    "evidence_metric_policy_command_watchdog_timeout_s",
)


def _report(envelope, mode=FrankaSessionMode.C1_COMMISSIONING):
    readiness = (
        "C2_BOUNDED_CLOSED_LOOP"
        if mode is FrankaSessionMode.C2_V94_POLICY
        else "C1_SUPERVISED_CONTROL_COMMISSIONING"
    )
    return {
        "result": "PASS",
        "readiness_level": readiness,
        "offline_only": True,
        "device_access": False,
        "hardware_writes": False,
        "robot_command_writes": False,
        "arming_state": "DISARMED",
        "motion_authorization_created": False,
        "physical_motion_authorized": False,
        "future_authorization_scope_required": "v94_closed_loop_franka_rh56",
        "eligible_for_operator_authorization": True,
        "run_id": "commissioning-evidence-run",
        "inputs": {
            "sha256": {
                "commissioning_profile_sha256": envelope.profile_sha256,
            }
        },
        "checks": [
            {
                "code": code,
                "passed": True,
                **(
                    {"actual": envelope.policy_command_max_age_s}
                    if code == "evidence_metric_policy_command_watchdog_timeout_s"
                    else {}
                ),
            }
            for code in _REQUIRED_CHECKS
        ],
        "failed_checks": [],
        "blockers": [],
    }


def _authorization():
    return MotionAuthorization(
        run_id="execution-run",
        authorization_id="explicit-user-authorization",
        issued_monotonic_s=0.5,
        expires_monotonic_s=2.0,
    )


def _armed_gate(authorization):
    gate = ClosedLoopSafetyGate()
    gate.update_interlocks(deadman_asserted=True, estop_healthy=True)
    gate.arm(
        authorization,
        run_id="execution-run",
        now_monotonic_s=1.0,
        franka_rest_verified=True,
        rh56_disabled_verified=True,
    )
    return gate


def _token(envelope, mode=FrankaSessionMode.C1_COMMISSIONING):
    return verified_franka_preflight_token_from_report(
        _report(envelope, mode),
        envelope=envelope,
        run_id="execution-run",
        stage=mode,
        issued_monotonic_s=0.5,
        expires_monotonic_s=2.0,
    )


def _state(*, q=None, dq=None, mode="move", contact=False, success=1.0):
    flags7 = np.zeros(7)
    if contact:
        flags7[0] = 1.0
    return SimpleNamespace(
        q=np.zeros(7) if q is None else np.asarray(q, dtype=np.float64),
        dq=np.zeros(7) if dq is None else np.asarray(dq, dtype=np.float64),
        O_T_EE=np.eye(4).reshape(16, order="F"),
        F_T_EE=np.eye(4).reshape(16, order="F"),
        m_ee=0.6,
        F_x_Cee=np.asarray([0.0, 0.0, 0.05]),
        I_ee=np.diag([0.01, 0.01, 0.005]).reshape(9, order="F"),
        m_load=0.0,
        F_x_Cload=np.zeros(3),
        I_load=np.zeros((3, 3)).reshape(9, order="F"),
        m_total=0.6,
        robot_mode=mode,
        current_errors={},
        joint_contact=flags7,
        joint_collision=np.zeros(7),
        cartesian_contact=np.zeros(6),
        cartesian_collision=np.zeros(6),
        control_command_success_rate=success,
    )


class _FakeControl:
    def __init__(self, backend, clock, states, on_read=None):
        self.backend = backend
        self.clock = clock
        self.states = list(states)
        self.on_read = on_read
        self.reads = 0
        self.writes = []
        self.bootstrap_writes = []
        self.finishes = []

    def read_once(self):
        self.backend.events.append("active_read")
        self.reads += 1
        self.clock.value += 0.001
        if self.on_read is not None:
            self.on_read(self.reads)
        state = self.states[min(self.reads - 1, len(self.states) - 1)]
        return state, 0.0 if self.reads == 1 else 0.001

    def write_once(self, q_rad, *, sequence):
        self.backend.events.append("active_write")
        self.backend.write_attempts += 1
        if self.backend.fail_write_on == self.backend.write_attempts:
            raise RuntimeError("synthetic active write fault")
        self.writes.append((int(sequence), np.asarray(q_rad).copy()))

    def write_bootstrap_hold(self, q_rad):
        self.backend.events.append("bootstrap_write")
        self.backend.write_attempts += 1
        if self.backend.fail_write_on == self.backend.write_attempts:
            raise RuntimeError("synthetic active write fault")
        self.bootstrap_writes.append(np.asarray(q_rad).copy())

    def finish(self, q_rad):
        self.backend.events.append("finish")
        self.finishes.append(np.asarray(q_rad).copy())


class _FakeBackend:
    def __init__(
        self,
        clock,
        states=None,
        stop_states=None,
        on_read=None,
        fail_write_on=None,
    ):
        self.clock = clock
        self.events = []
        self.fail_write_on = fail_write_on
        self.write_attempts = 0
        self.control = _FakeControl(
            self,
            clock,
            states or [_state()],
            on_read=on_read,
        )
        self.stop_states = list(stop_states or [_state(mode="idle")])
        self.stop_reads = 0
        self.start_count = 0
        self.stop_count = 0
        self.close_count = 0

    def start_joint_position_session(self):
        self.events.append("start")
        self.start_count += 1
        return self.control

    def request_stop(self):
        self.events.append("stop")
        self.stop_count += 1

    def read_post_stop_state(self):
        self.events.append("post_stop_read")
        value = self.stop_states[min(self.stop_reads, len(self.stop_states) - 1)]
        self.stop_reads += 1
        return value

    def close(self):
        self.events.append("close")
        self.close_count += 1


def _c1_target(produced=1.0, q=None, sequence=1):
    return FrankaJointTarget(
        sequence=sequence,
        produced_monotonic_s=produced,
        target_q_rad=np.zeros(7) if q is None else q,
        source=FrankaTargetSource.C1_DETERMINISTIC,
        source_id=f"probe-{sequence}",
    )


def _c2_command(produced=1.0, sequence=1):
    action = np.zeros(13, dtype=np.float32)
    return ClosedLoopCommand(
        sequence=sequence,
        produced_monotonic_s=produced,
        observation_realtime_s=100.0,
        previous_executed_action13_used=INITIAL_PREVIOUS_ACTION13,
        raw_policy_action13=action,
        executed_policy_action13=action,
        franka_target_q_rad=np.zeros(7),
        rh56_angle_set_register_order=np.full(6, 1000),
    )


def _supervised_fixture(
    tmp_path: Path,
    *,
    clock: _Clock,
    franka_action_contract_id: str = LEGACY_FRANKA_ACTION_CONTRACT_ID,
    maximum_episode_delta_rad=0.05,
    enable_measured_hold_bootstrap: bool = True,
):
    del tmp_path
    workspace = Path(__file__).resolve().parents[2]
    profile_path = workspace / "dexgrasp/configs/fr3_rh56_v94_commissioning.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    reference = np.asarray(profile["franka"]["default_q_rad"], dtype=np.float64)
    envelope = load_experimental_supervised_franka_envelope(profile_path)
    authorization = _authorization()
    ledger = ExecutedActionLedger(
        maximum_commit_latency_s=0.05,
        initial_previous_action13=(
            np.zeros(13, dtype=np.float32)
            if franka_action_contract_id == QD_G015_FRANKA_ACTION_CONTRACT_ID
            else INITIAL_PREVIOUS_ACTION13
        ),
    )
    backend = _FakeBackend(
        clock,
        states=[_state(q=reference)],
    )
    session = FrankaPersistentSession(
        run_id="execution-run",
        mode=FrankaSessionMode.SUPERVISED_V94,
        envelope=envelope,
        target_hold=FrankaTargetSampleHold(),
        safety_gate=_armed_gate(authorization),
        backend_factory=lambda: backend,
        action_ledger=ledger,
        enable_c2_measured_hold_bootstrap=enable_measured_hold_bootstrap,
        supervised_reference_q_rad=reference,
        supervised_maximum_start_error_rad=0.01,
        supervised_maximum_tick_target_delta_rad=0.0031,
        supervised_maximum_episode_delta_rad=maximum_episode_delta_rad,
        franka_action_contract_id=franka_action_contract_id,
        supervised_static_provenance_prevalidated=True,
        monotonic=clock,
        realtime=lambda: 100.0 + clock.value,
    )
    token = _issue_supervised_franka_preflight_token(
        envelope=envelope,
        run_id="execution-run",
        confirmed_permit_sha256="a" * 64,
        issued_monotonic_s=0.5,
        expires_monotonic_s=2.0,
    )
    return session, backend, authorization, token


def test_qd_g015_supervised_fallback_accepts_disabled_episode_guard(tmp_path):
    clock = _Clock()
    session, backend, authorization, token = _supervised_fixture(
        tmp_path,
        clock=clock,
        franka_action_contract_id=QD_G015_FRANKA_ACTION_CONTRACT_ID,
        maximum_episode_delta_rad=None,
        enable_measured_hold_bootstrap=False,
    )
    reference = session.supervised_reference_q_rad
    assert reference is not None
    qd_target = reference.copy()
    qd_target[0] += 1.22
    assert abs(qd_target[0] - reference[0]) > 1.21
    assert qd_target[0] < session.envelope.safe_joint_upper_rad[0]
    previous = np.zeros(13, dtype=np.float32)
    action = np.zeros(13, dtype=np.float32)
    command = ClosedLoopCommand(
        sequence=1,
        produced_monotonic_s=clock.value,
        observation_realtime_s=100.0,
        previous_executed_action13_used=previous,
        raw_policy_action13=action,
        executed_policy_action13=action,
        franka_target_q_rad=qd_target,
        rh56_angle_set_register_order=np.full(6, 1000),
    )
    assert session.action_ledger is not None
    session.action_ledger.stage(command, now_monotonic_s=clock.value)
    assert (
        session.action_ledger.acknowledge(
            "rh56", sequence=1, now_monotonic_s=clock.value
        )
        is False
    )
    session.target_hold.publish(
        FrankaJointTarget.from_closed_loop_command(
            command,
            source=FrankaTargetSource.SUPERVISED_V94,
        )
    )

    telemetry = session.run(
        authorization=authorization,
        preflight_token=token,
        maximum_cycles=1,
    )

    assert session.franka_action_contract_id == QD_G015_FRANKA_ACTION_CONTRACT_ID
    assert session.supervised_maximum_episode_delta_rad is None
    assert telemetry.franka_ack_count == 1
    assert session.action_ledger.last_committed_sequence == 1
    assert [sequence for sequence, _q in backend.control.writes] == [1]
    assert telemetry.maximum_target_following_error_rad > 1.21


def test_legacy_supervised_fallback_still_requires_positive_episode_guard(tmp_path):
    with pytest.raises(ValueError, match="supervised_maximum_episode_delta_rad"):
        _supervised_fixture(
            tmp_path,
            clock=_Clock(),
            franka_action_contract_id=LEGACY_FRANKA_ACTION_CONTRACT_ID,
            maximum_episode_delta_rad=None,
        )


def test_legacy_supervised_fallback_retains_tick_target_guard(tmp_path):
    clock = _Clock()
    session, _backend, authorization, token = _supervised_fixture(
        tmp_path,
        clock=clock,
        enable_measured_hold_bootstrap=False,
    )
    reference = session.supervised_reference_q_rad
    assert reference is not None
    legacy_target = reference.copy()
    legacy_target[0] += 0.004
    action = np.zeros(13, dtype=np.float32)
    command = ClosedLoopCommand(
        sequence=1,
        produced_monotonic_s=clock.value,
        observation_realtime_s=100.0,
        previous_executed_action13_used=INITIAL_PREVIOUS_ACTION13,
        raw_policy_action13=action,
        executed_policy_action13=action,
        franka_target_q_rad=legacy_target,
        rh56_angle_set_register_order=np.full(6, 1000),
    )
    assert session.action_ledger is not None
    session.action_ledger.stage(command, now_monotonic_s=clock.value)
    assert (
        session.action_ledger.acknowledge(
            "rh56", sequence=1, now_monotonic_s=clock.value
        )
        is False
    )
    session.target_hold.publish(
        FrankaJointTarget.from_closed_loop_command(
            command,
            source=FrankaTargetSource.SUPERVISED_V94,
        )
    )

    with pytest.raises(
        FrankaPersistentSessionError,
        match="supervised policy target tick delta exceeded",
    ):
        session.run(
            authorization=authorization,
            preflight_token=token,
            maximum_cycles=1,
        )


def test_supervised_active_loop_consumes_prevalidated_static_proof(tmp_path):
    clock = _Clock()
    session, backend, authorization, token = _supervised_fixture(
        tmp_path, clock=clock
    )
    session._validate_static_provenance = lambda _state: pytest.fail(
        "heavy static provenance must stay outside the active FCI window"
    )

    telemetry = session.run(
        authorization=authorization,
        preflight_token=token,
        maximum_cycles=1,
    )

    assert telemetry.static_provenance_verified is True
    assert telemetry.active_read_count == telemetry.active_write_count == 1
    assert telemetry.bootstrap_hold_write_count == 1
    assert len(backend.control.bootstrap_writes) == 1


def test_supervised_envelope_keeps_native_servo_period_bound(tmp_path):
    clock = _Clock()
    session, _backend, _authorization, _token_value = _supervised_fixture(
        tmp_path, clock=clock
    )

    assert session.envelope.control_period_max_s == pytest.approx(0.002)


def test_supervised_period_fault_records_the_failing_sample(tmp_path):
    clock = _Clock()
    session, backend, authorization, token = _supervised_fixture(
        tmp_path, clock=clock
    )
    original_read_once = backend.control.read_once

    def read_once_with_long_second_gap():
        state, dt = original_read_once()
        if backend.control.reads == 2:
            return state, 0.005
        return state, dt

    backend.control.read_once = read_once_with_long_second_gap

    with pytest.raises(
        FrankaPersistentSessionError,
        match=r"control period 0\.005000000s is outside commissioned bounds",
    ):
        session.run(
            authorization=authorization,
            preflight_token=token,
            maximum_cycles=2,
        )

    assert session.last_telemetry is not None
    assert session.last_telemetry.maximum_control_period_s == pytest.approx(0.005)
    assert session.last_telemetry.active_read_count == 2
    assert session.last_telemetry.active_write_count == 1


def test_supervised_first_bootstrap_uses_1p5ms_then_steady_uses_0p8ms(tmp_path):
    clock = _Clock()
    session, backend, authorization, token = _supervised_fixture(
        tmp_path, clock=clock
    )
    original = session._validate_dynamic_state
    delays = iter((0.0012, 0.0009))

    def delayed_dynamic_state(*args, **kwargs):
        result = original(*args, **kwargs)
        clock.value += next(delays)
        return result

    session._validate_dynamic_state = delayed_dynamic_state

    with pytest.raises(
        FrankaPersistentSessionError,
        match=r"actual=0\.000900000s limit=0\.000800000s",
    ):
        session.run(
            authorization=authorization,
            preflight_token=token,
            maximum_cycles=2,
        )

    assert SUPERVISED_BOOTSTRAP_READ_TO_WRITE_DEADLINE_S == pytest.approx(0.0015)
    assert session.envelope.read_to_write_deadline_s == pytest.approx(0.0008)
    assert backend.control.reads == 2
    assert len(backend.control.bootstrap_writes) == 1
    assert session.last_telemetry is not None
    assert session.last_telemetry.active_write_count == 1
    assert session.last_telemetry.maximum_read_to_write_s == pytest.approx(0.0012)
    assert session.last_telemetry.stop_requested is True
    assert session.last_telemetry.stop_verified is True


def test_supervised_first_cycle_overrun_records_actual_before_any_write(tmp_path):
    clock = _Clock()
    session, backend, authorization, token = _supervised_fixture(
        tmp_path, clock=clock
    )
    original = session._validate_dynamic_state

    def delayed_dynamic_state(*args, **kwargs):
        result = original(*args, **kwargs)
        clock.value += 0.0016
        return result

    session._validate_dynamic_state = delayed_dynamic_state

    with pytest.raises(
        FrankaPersistentSessionError,
        match=r"actual=0\.001600000s limit=0\.001500000s",
    ):
        session.run(
            authorization=authorization,
            preflight_token=token,
            maximum_cycles=1,
        )

    assert backend.control.bootstrap_writes == []
    assert backend.control.writes == []
    assert session.last_telemetry is not None
    assert session.last_telemetry.active_write_count == 0
    assert session.last_telemetry.maximum_read_to_write_s == pytest.approx(0.0016)


def test_current_v94_profile_is_rejected_without_inventing_online_limits():
    workspace = Path(__file__).resolve().parents[2]
    source = workspace / "dexgrasp/configs/fr3_rh56_v94_commissioning.json"
    with pytest.raises(ValueError, match="persistent_session|acceleration"):
        load_commissioned_franka_envelope(source)


def test_envelope_cannot_be_constructed_or_fill_null_limits_by_default(tmp_path):
    with pytest.raises(TypeError, match="exact profile"):
        CommissionedFrankaEnvelope(_seal=object())
    payload = _profile_payload()
    payload["franka"]["online_max_joint_jerk_rad_s3"] = None
    with pytest.raises(ValueError, match="jerk"):
        load_commissioned_franka_envelope(_write_profile(tmp_path, payload))


def test_token_rejects_failed_or_profile_unbound_report(tmp_path):
    envelope = load_commissioned_franka_envelope(_write_profile(tmp_path))
    failed = _report(envelope)
    failed["result"] = "FAIL"
    with pytest.raises(ValueError, match="result"):
        verified_franka_preflight_token_from_report(
            failed,
            envelope=envelope,
            run_id="execution-run",
            stage=FrankaSessionMode.C1_COMMISSIONING,
            issued_monotonic_s=0.5,
            expires_monotonic_s=2.0,
        )
    unbound = _report(envelope)
    unbound["inputs"]["sha256"]["commissioning_profile_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="commissioning_profile_sha256"):
        verified_franka_preflight_token_from_report(
            unbound,
            envelope=envelope,
            run_id="execution-run",
            stage=FrankaSessionMode.C1_COMMISSIONING,
            issued_monotonic_s=0.5,
            expires_monotonic_s=2.0,
        )


def test_default_session_has_no_device_factory_and_opens_nothing(tmp_path):
    envelope = load_commissioned_franka_envelope(_write_profile(tmp_path))
    authorization = _authorization()
    hold = FrankaTargetSampleHold()
    hold.publish(_c1_target())
    session = FrankaPersistentSession(
        run_id="execution-run",
        mode=FrankaSessionMode.C1_COMMISSIONING,
        envelope=envelope,
        target_hold=hold,
        safety_gate=_armed_gate(authorization),
        monotonic=_Clock(),
    )
    with pytest.raises(FrankaPersistentSessionError, match="device access remains disabled"):
        session.run(
            authorization=authorization,
            preflight_token=_token(envelope),
            maximum_cycles=1,
        )
    assert session.state is FrankaSessionState.DISABLED


def test_stale_target_fails_before_backend_factory_is_called(tmp_path):
    envelope = load_commissioned_franka_envelope(_write_profile(tmp_path))
    authorization = _authorization()
    hold = FrankaTargetSampleHold()
    hold.publish(_c1_target(produced=0.0))
    calls = []
    session = FrankaPersistentSession(
        run_id="execution-run",
        mode=FrankaSessionMode.C1_COMMISSIONING,
        envelope=envelope,
        target_hold=hold,
        safety_gate=_armed_gate(authorization),
        backend_factory=lambda: calls.append(True),
        monotonic=_Clock(),
    )
    with pytest.raises(FrankaPersistentSessionError, match="stale"):
        session.run(
            authorization=authorization,
            preflight_token=_token(envelope),
            maximum_cycles=1,
        )
    assert calls == []
    assert session.state is FrankaSessionState.DISABLED


def test_c1_persistent_session_uses_one_read_per_cycle_and_publishes_pose_ring(tmp_path):
    envelope = load_commissioned_franka_envelope(_write_profile(tmp_path))
    authorization = _authorization()
    clock = _Clock()
    hold = FrankaTargetSampleHold()
    hold.publish(_c1_target())
    backend = _FakeBackend(clock)
    session = FrankaPersistentSession(
        run_id="execution-run",
        mode=FrankaSessionMode.C1_COMMISSIONING,
        envelope=envelope,
        target_hold=hold,
        safety_gate=_armed_gate(authorization),
        backend_factory=lambda: backend,
        monotonic=clock,
        realtime=lambda: 100.0 + clock.value,
    )
    telemetry = session.run(
        authorization=authorization,
        preflight_token=_token(envelope),
        maximum_cycles=3,
    )
    assert telemetry.state is FrankaSessionState.STOPPED
    assert telemetry.active_read_count == telemetry.active_write_count == 3
    assert telemetry.pose_publish_count == 3
    assert telemetry.static_provenance_verified is True
    assert backend.control.reads == 3
    assert len(backend.control.writes) == 3
    assert len(backend.control.finishes) == 1
    assert backend.start_count == backend.stop_count == backend.close_count == 1
    assert telemetry.stop_verified is True
    assert telemetry.stop_verification_samples == 2
    samples = session.pose_ring.snapshot()
    assert [sample.cycle for sample in samples] == [1, 2, 3]
    assert all(not sample.q_rad.flags.writeable for sample in samples)


def test_static_tool_and_load_provenance_is_checked_before_first_write(tmp_path):
    envelope = load_commissioned_franka_envelope(_write_profile(tmp_path))
    authorization = _authorization()
    clock = _Clock()
    hold = FrankaTargetSampleHold()
    hold.publish(_c1_target())
    bad_state = _state()
    bad_state.m_total = 0.7
    backend = _FakeBackend(clock, states=[bad_state])
    session = FrankaPersistentSession(
        run_id="execution-run",
        mode=FrankaSessionMode.C1_COMMISSIONING,
        envelope=envelope,
        target_hold=hold,
        safety_gate=_armed_gate(authorization),
        backend_factory=lambda: backend,
        monotonic=clock,
    )
    with pytest.raises(FrankaPersistentSessionError, match="total mass"):
        session.run(
            authorization=authorization,
            preflight_token=_token(envelope),
            maximum_cycles=1,
        )
    assert backend.control.reads == 1
    assert backend.control.writes == []
    assert session.last_telemetry.static_provenance_verified is False


def test_c2_acknowledges_exact_sequence_only_once_across_sample_hold_cycles(tmp_path):
    envelope = load_commissioned_franka_envelope(_write_profile(tmp_path))
    authorization = _authorization()
    clock = _Clock(1.001)
    command = _c2_command()
    hold = FrankaTargetSampleHold()
    hold.publish(FrankaJointTarget.from_closed_loop_command(command))
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.1)
    ledger.stage(command, now_monotonic_s=1.0)
    assert ledger.acknowledge("rh56", sequence=1, now_monotonic_s=1.001) is False
    backend = _FakeBackend(clock)
    session = FrankaPersistentSession(
        run_id="execution-run",
        mode=FrankaSessionMode.C2_V94_POLICY,
        envelope=envelope,
        target_hold=hold,
        safety_gate=_armed_gate(authorization),
        backend_factory=lambda: backend,
        action_ledger=ledger,
        monotonic=clock,
    )
    telemetry = session.run(
        authorization=authorization,
        preflight_token=_token(envelope, FrankaSessionMode.C2_V94_POLICY),
        maximum_cycles=3,
    )
    assert telemetry.franka_ack_count == 1
    assert ledger.last_committed_sequence == 1
    assert [sequence for sequence, _q in backend.control.writes] == [1, 1, 1]


def test_c2_measured_hold_bootstrap_publishes_pose_then_sequence_one_takes_over(
    tmp_path,
):
    envelope = load_commissioned_franka_envelope(_write_profile(tmp_path))
    authorization = _authorization()
    clock = _Clock()
    hold = FrankaTargetSampleHold()
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.1)
    command = _c2_command(produced=1.0, sequence=1)

    def publish_first_policy_target(read_count):
        if read_count != 2:
            return
        ledger.stage(command, now_monotonic_s=clock.value)
        assert (
            ledger.acknowledge(
                "rh56", sequence=1, now_monotonic_s=clock.value
            )
            is False
        )
        hold.publish(FrankaJointTarget.from_closed_loop_command(command))

    backend = _FakeBackend(clock, on_read=publish_first_policy_target)
    session = FrankaPersistentSession(
        run_id="execution-run",
        mode=FrankaSessionMode.C2_V94_POLICY,
        envelope=envelope,
        target_hold=hold,
        safety_gate=_armed_gate(authorization),
        backend_factory=lambda: backend,
        action_ledger=ledger,
        enable_c2_measured_hold_bootstrap=True,
        monotonic=clock,
        realtime=lambda: 100.0 + clock.value,
    )

    telemetry = session.run(
        authorization=authorization,
        preflight_token=_token(envelope, FrankaSessionMode.C2_V94_POLICY),
        maximum_cycles=2,
    )

    assert telemetry.bootstrap_hold_write_count == 1
    assert telemetry.active_read_count == telemetry.active_write_count == 2
    assert len(backend.control.bootstrap_writes) == 1
    np.testing.assert_array_equal(backend.control.bootstrap_writes[0], np.zeros(7))
    assert [sequence for sequence, _q in backend.control.writes] == [1]
    assert ledger.last_committed_sequence == 1
    assert telemetry.franka_ack_count == 1
    assert session.c2_bootstrap_ready is True
    assert len(session.pose_ring.snapshot()) == 2


def test_c2_requires_exact_staged_transaction_before_backend_open(tmp_path):
    envelope = load_commissioned_franka_envelope(_write_profile(tmp_path))
    authorization = _authorization()
    command = _c2_command()
    hold = FrankaTargetSampleHold()
    hold.publish(FrankaJointTarget.from_closed_loop_command(command))
    calls = []
    session = FrankaPersistentSession(
        run_id="execution-run",
        mode=FrankaSessionMode.C2_V94_POLICY,
        envelope=envelope,
        target_hold=hold,
        safety_gate=_armed_gate(authorization),
        backend_factory=lambda: calls.append(True),
        action_ledger=ExecutedActionLedger(maximum_commit_latency_s=0.1),
        monotonic=_Clock(),
    )
    with pytest.raises(FrankaPersistentSessionError, match="pending dual-device"):
        session.run(
            authorization=authorization,
            preflight_token=_token(envelope, FrankaSessionMode.C2_V94_POLICY),
            maximum_cycles=1,
        )
    assert calls == []
    assert session.state is FrankaSessionState.DISABLED


def test_c2_runtime_fault_is_latched_into_shared_action_ledger(tmp_path):
    envelope = load_commissioned_franka_envelope(_write_profile(tmp_path))
    authorization = _authorization()
    clock = _Clock()
    command = _c2_command()
    hold = FrankaTargetSampleHold()
    hold.publish(FrankaJointTarget.from_closed_loop_command(command))
    ledger = ExecutedActionLedger(maximum_commit_latency_s=0.1)
    ledger.stage(command, now_monotonic_s=1.0)
    backend = _FakeBackend(clock, states=[_state(contact=True)])
    session = FrankaPersistentSession(
        run_id="execution-run",
        mode=FrankaSessionMode.C2_V94_POLICY,
        envelope=envelope,
        target_hold=hold,
        safety_gate=_armed_gate(authorization),
        backend_factory=lambda: backend,
        action_ledger=ledger,
        monotonic=clock,
    )
    with pytest.raises(FrankaPersistentSessionError, match="joint_contact"):
        session.run(
            authorization=authorization,
            preflight_token=_token(envelope, FrankaSessionMode.C2_V94_POLICY),
            maximum_cycles=1,
        )
    assert ledger.fault_reason is not None
    assert "franka failed sequence 1" in ledger.fault_reason
    assert backend.control.writes == []


def test_contact_fault_latches_gate_requests_stop_and_never_writes(tmp_path):
    envelope = load_commissioned_franka_envelope(_write_profile(tmp_path))
    authorization = _authorization()
    gate = _armed_gate(authorization)
    clock = _Clock()
    hold = FrankaTargetSampleHold()
    hold.publish(_c1_target())
    backend = _FakeBackend(clock, states=[_state(contact=True)])
    session = FrankaPersistentSession(
        run_id="execution-run",
        mode=FrankaSessionMode.C1_COMMISSIONING,
        envelope=envelope,
        target_hold=hold,
        safety_gate=gate,
        backend_factory=lambda: backend,
        monotonic=clock,
    )
    with pytest.raises(FrankaPersistentSessionError, match="joint_contact"):
        session.run(
            authorization=authorization,
            preflight_token=_token(envelope),
            maximum_cycles=2,
        )
    assert session.state is FrankaSessionState.FAULT_LATCHED
    assert gate.state is SafetyState.FAULT_LATCHED
    assert backend.control.reads == 1
    assert backend.control.writes == []
    assert backend.control.finishes == []
    assert backend.events[-4:] == [
        "stop",
        "post_stop_read",
        "post_stop_read",
        "close",
    ]
    assert backend.stop_count == backend.close_count == 1
    assert session.last_telemetry.stop_verified is True


def test_interlock_fault_during_cycle_is_rechecked_before_write(tmp_path):
    envelope = load_commissioned_franka_envelope(_write_profile(tmp_path))
    authorization = _authorization()
    gate = _armed_gate(authorization)
    clock = _Clock()
    hold = FrankaTargetSampleHold()
    hold.publish(_c1_target())

    def latch_after_read(read_count):
        if read_count == 1:
            gate.latch_fault("synthetic interlock supervisor fault")

    backend = _FakeBackend(clock, on_read=latch_after_read)
    session = FrankaPersistentSession(
        run_id="execution-run",
        mode=FrankaSessionMode.C1_COMMISSIONING,
        envelope=envelope,
        target_hold=hold,
        safety_gate=gate,
        backend_factory=lambda: backend,
        monotonic=clock,
    )
    with pytest.raises(FrankaPersistentSessionError, match="fault is latched"):
        session.run(
            authorization=authorization,
            preflight_token=_token(envelope),
            maximum_cycles=1,
        )
    assert backend.control.reads == 1
    assert backend.control.writes == []
    assert backend.control.finishes == []
    assert backend.events[-4:] == [
        "stop",
        "post_stop_read",
        "post_stop_read",
        "close",
    ]


def test_deadman_fault_after_prior_write_stops_without_finish(tmp_path):
    envelope = load_commissioned_franka_envelope(_write_profile(tmp_path))
    authorization = _authorization()
    gate = _armed_gate(authorization)
    clock = _Clock()
    hold = FrankaTargetSampleHold()
    hold.publish(_c1_target())

    def release_deadman_after_second_read(read_count):
        if read_count == 2:
            gate.update_interlocks(
                deadman_asserted=False,
                estop_healthy=True,
            )

    backend = _FakeBackend(clock, on_read=release_deadman_after_second_read)
    session = FrankaPersistentSession(
        run_id="execution-run",
        mode=FrankaSessionMode.C1_COMMISSIONING,
        envelope=envelope,
        target_hold=hold,
        safety_gate=gate,
        backend_factory=lambda: backend,
        monotonic=clock,
    )
    with pytest.raises(FrankaPersistentSessionError, match="deadman"):
        session.run(
            authorization=authorization,
            preflight_token=_token(envelope),
            maximum_cycles=3,
        )
    assert len(backend.control.writes) == 1
    assert backend.control.finishes == []
    assert backend.events[-4:] == [
        "stop",
        "post_stop_read",
        "post_stop_read",
        "close",
    ]


def test_active_write_fault_after_prior_write_stops_without_finish(tmp_path):
    envelope = load_commissioned_franka_envelope(_write_profile(tmp_path))
    authorization = _authorization()
    gate = _armed_gate(authorization)
    clock = _Clock()
    hold = FrankaTargetSampleHold()
    hold.publish(_c1_target())
    backend = _FakeBackend(clock, fail_write_on=2)
    session = FrankaPersistentSession(
        run_id="execution-run",
        mode=FrankaSessionMode.C1_COMMISSIONING,
        envelope=envelope,
        target_hold=hold,
        safety_gate=gate,
        backend_factory=lambda: backend,
        monotonic=clock,
    )
    with pytest.raises(FrankaPersistentSessionError, match="active write fault"):
        session.run(
            authorization=authorization,
            preflight_token=_token(envelope),
            maximum_cycles=3,
        )
    assert len(backend.control.writes) == 1
    assert backend.write_attempts == 2
    assert backend.control.finishes == []
    assert backend.events[-4:] == [
        "stop",
        "post_stop_read",
        "post_stop_read",
        "close",
    ]


def test_policy_target_rate_over_limit_faults_instead_of_silently_reshaping(tmp_path):
    envelope = load_commissioned_franka_envelope(_write_profile(tmp_path))
    authorization = _authorization()
    gate = _armed_gate(authorization)
    clock = _Clock()
    hold = FrankaTargetSampleHold()
    hold.publish(_c1_target())

    def publish_fast_second(read_count):
        if read_count == 2:
            hold.publish(
                _c1_target(
                    produced=clock.value,
                    q=np.full(7, 0.1),
                    sequence=2,
                )
            )

    backend = _FakeBackend(clock, on_read=publish_fast_second)
    session = FrankaPersistentSession(
        run_id="execution-run",
        mode=FrankaSessionMode.C1_COMMISSIONING,
        envelope=envelope,
        target_hold=hold,
        safety_gate=gate,
        backend_factory=lambda: backend,
        monotonic=clock,
    )
    with pytest.raises(FrankaPersistentSessionError, match="target rate"):
        session.run(
            authorization=authorization,
            preflight_token=_token(envelope),
            maximum_cycles=3,
        )
    assert len(backend.control.writes) == 1
    assert session.last_telemetry.maximum_target_rate_rad_s > 1.0


def test_commissioned_shaper_bounds_velocity_acceleration_jerk_and_target_lag(tmp_path):
    envelope = load_commissioned_franka_envelope(_write_profile(tmp_path))
    authorization = _authorization()
    clock = _Clock()
    hold = FrankaTargetSampleHold()
    hold.publish(_c1_target())

    def publish_bounded_second(read_count):
        if read_count == 2:
            hold.publish(
                _c1_target(
                    produced=clock.value,
                    q=np.full(7, 0.001),
                    sequence=2,
                )
            )

    backend = _FakeBackend(clock, on_read=publish_bounded_second)
    session = FrankaPersistentSession(
        run_id="execution-run",
        mode=FrankaSessionMode.C1_COMMISSIONING,
        envelope=envelope,
        target_hold=hold,
        safety_gate=_armed_gate(authorization),
        backend_factory=lambda: backend,
        monotonic=clock,
    )
    telemetry = session.run(
        authorization=authorization,
        preflight_token=_token(envelope),
        maximum_cycles=4,
    )
    assert 0.0 < telemetry.maximum_command_velocity_rad_s <= 1.0
    assert telemetry.maximum_command_acceleration_rad_s2 <= 100.0 + 1.0e-9
    assert telemetry.maximum_command_jerk_rad_s3 <= 100000.0 + 1.0e-7
    assert telemetry.maximum_target_following_error_rad <= 0.005
    assert telemetry.maximum_target_rate_rad_s <= 1.0


@pytest.mark.parametrize(
    ("states", "message"),
    [
        (
            [_state(dq=np.zeros(7)), _state(dq=np.full(7, 0.2))],
            "acceleration",
        ),
        (
            [
                _state(dq=np.zeros(7)),
                _state(dq=np.full(7, 0.09)),
                _state(dq=np.zeros(7)),
            ],
            "jerk",
        ),
        (
            [_state(q=np.zeros(7)), _state(q=np.full(7, 0.2))],
            "tracking error",
        ),
    ],
)
def test_measured_acceleration_jerk_and_tracking_fault_before_next_write(
    tmp_path, states, message
):
    envelope = load_commissioned_franka_envelope(_write_profile(tmp_path))
    authorization = _authorization()
    clock = _Clock()
    hold = FrankaTargetSampleHold()
    hold.publish(_c1_target())
    backend = _FakeBackend(clock, states=states)
    session = FrankaPersistentSession(
        run_id="execution-run",
        mode=FrankaSessionMode.C1_COMMISSIONING,
        envelope=envelope,
        target_hold=hold,
        safety_gate=_armed_gate(authorization),
        backend_factory=lambda: backend,
        monotonic=clock,
    )
    with pytest.raises(FrankaPersistentSessionError, match=message):
        session.run(
            authorization=authorization,
            preflight_token=_token(envelope),
            maximum_cycles=len(states),
        )
    assert len(backend.control.writes) == len(states) - 1


def test_target_age_is_rechecked_after_blocking_active_read(tmp_path):
    envelope = load_commissioned_franka_envelope(_write_profile(tmp_path))
    authorization = _authorization()
    clock = _Clock()
    hold = FrankaTargetSampleHold()
    hold.publish(_c1_target())

    def make_target_stale(read_count):
        if read_count == 1:
            clock.value = 1.101

    backend = _FakeBackend(clock, on_read=make_target_stale)
    session = FrankaPersistentSession(
        run_id="execution-run",
        mode=FrankaSessionMode.C1_COMMISSIONING,
        envelope=envelope,
        target_hold=hold,
        safety_gate=_armed_gate(authorization),
        backend_factory=lambda: backend,
        monotonic=clock,
    )
    with pytest.raises(FrankaPersistentSessionError, match="stale"):
        session.run(
            authorization=authorization,
            preflight_token=_token(envelope),
            maximum_cycles=1,
        )
    assert backend.control.reads == 1
    assert backend.control.writes == []


def test_stop_must_be_verified_with_consecutive_fresh_idle_samples(tmp_path):
    envelope = load_commissioned_franka_envelope(_write_profile(tmp_path))
    authorization = _authorization()
    gate = _armed_gate(authorization)
    clock = _Clock()
    hold = FrankaTargetSampleHold()
    hold.publish(_c1_target())
    moving = _state(mode="move", dq=np.full(7, 0.02))
    backend = _FakeBackend(clock, stop_states=[moving, moving, moving])
    session = FrankaPersistentSession(
        run_id="execution-run",
        mode=FrankaSessionMode.C1_COMMISSIONING,
        envelope=envelope,
        target_hold=hold,
        safety_gate=gate,
        backend_factory=lambda: backend,
        monotonic=clock,
    )
    with pytest.raises(FrankaPersistentSessionError, match="stop was not verified"):
        session.run(
            authorization=authorization,
            preflight_token=_token(envelope),
            maximum_cycles=1,
        )
    assert gate.state is SafetyState.FAULT_LATCHED
    assert session.last_telemetry.stop_verified is False
    assert session.last_telemetry.stop_verification_samples == 3


@pytest.mark.parametrize(
    ("state_kwargs", "message"),
    [
        ({"mode": "reflex"}, "mode became unsafe"),
        ({"success": 0.5}, "success rate"),
        ({"dq": np.full(7, 1.1)}, "velocity"),
    ],
)
def test_per_cycle_mode_communication_and_velocity_checks_fail_closed(
    tmp_path, state_kwargs, message
):
    envelope = load_commissioned_franka_envelope(_write_profile(tmp_path))
    authorization = _authorization()
    clock = _Clock()
    hold = FrankaTargetSampleHold()
    hold.publish(_c1_target())
    backend = _FakeBackend(clock, states=[_state(**state_kwargs)])
    session = FrankaPersistentSession(
        run_id="execution-run",
        mode=FrankaSessionMode.C1_COMMISSIONING,
        envelope=envelope,
        target_hold=hold,
        safety_gate=_armed_gate(authorization),
        backend_factory=lambda: backend,
        monotonic=clock,
    )
    with pytest.raises(FrankaPersistentSessionError, match=message):
        session.run(
            authorization=authorization,
            preflight_token=_token(envelope),
            maximum_cycles=1,
        )
    assert backend.control.writes == []
