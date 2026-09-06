"""Offline integration guards for the supervised native Franka session path.

All device boundaries in this module are fakes.  In particular, the native
session below owns no process, socket, robot, camera, or serial descriptor.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from sim2real.runtime.bounded_c2_runtime import (
    BoundedC2HardwareRuntimeError,
    BoundedV94C2RuntimeFactory,
    SupervisedV94RuntimeFactory,
    _seal_supervised_v94_admission,
)
from sim2real.closed_loop_core import ClosedLoopSafetyGate, MotionAuthorization
from sim2real.contracts.actions import QD_G015_FRANKA_ACTION_CONTRACT_ID
from robot_control.franka.session import (
    FrankaPersistentSession,
    FrankaPoseSample,
    FrankaPoseRing,
    FrankaTargetSampleHold,
    _issue_supervised_franka_preflight_token,
    load_experimental_supervised_franka_envelope,
)
from robot_control.rh56.actuator import (
    _issue_supervised_rh56_preflight,
)
from sim2real.tests.test_bounded_c2_orchestrator import _admission
from sim2real.tests.test_bounded_c2_runtime import (
    _ManagedFakeTransport,
    _Source,
    _TransportFactory,
    _actuator_factory,
)


class _SupervisedSafetySupervisor:
    physical_interlocks_configured = False
    operator_supervision_confirmed = True
    independent_command_watchdogs_configured = True
    verified_dual_device_stop_supported = True

    def require_motion(self, admission, *, now_monotonic_s):
        admission.safety_gate.require_motion(
            run_id=admission.run_id,
            now_monotonic_s=now_monotonic_s,
        )


def _supervised_admission(
    *,
    steps: int = 1,
    franka_action_contract_id: str = "v94_inspire_semantic_13d",
    initial_previous_action13=None,
):
    workspace = Path(__file__).resolve().parents[2]
    profile_path = workspace / "dexgrasp/configs/fr3_rh56_v94_commissioning.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    envelope = load_experimental_supervised_franka_envelope(profile_path)
    now = time.monotonic()
    run_id = "native-supervised-offline-test"
    authorization = MotionAuthorization(
        run_id=run_id,
        authorization_id="native-supervised-offline-authorization",
        issued_monotonic_s=now - 1.0,
        expires_monotonic_s=now + 5.0,
    )
    gate = ClosedLoopSafetyGate()
    gate.update_interlocks(deadman_asserted=True, estop_healthy=True)
    gate.arm(
        authorization,
        run_id=run_id,
        now_monotonic_s=now,
        franka_rest_verified=True,
        rh56_disabled_verified=True,
    )
    permit_sha256 = "a" * 64
    franka_token = _issue_supervised_franka_preflight_token(
        envelope=envelope,
        run_id=run_id,
        confirmed_permit_sha256=permit_sha256,
        issued_monotonic_s=now - 1.0,
        expires_monotonic_s=authorization.expires_monotonic_s,
    )
    rh56_token = _issue_supervised_rh56_preflight(
        run_id=run_id,
        confirmed_permit_sha256=permit_sha256,
        commissioning_profile_sha256=envelope.profile_sha256,
        watchdog_timeout_s=0.05,
    )
    kwargs = {}
    if initial_previous_action13 is not None:
        kwargs["initial_previous_action13"] = initial_previous_action13
    maximum_episode_delta_rad = (
        None
        if franka_action_contract_id == QD_G015_FRANKA_ACTION_CONTRACT_ID
        else 0.05
    )
    return _seal_supervised_v94_admission(
        run_id=run_id,
        requested_policy_steps=steps,
        policy_rate_hz=60.0,
        hard_deadline_monotonic_s=now + 3.0,
        authorization=authorization,
        franka_preflight_token=franka_token,
        rh56_preflight=rh56_token,
        franka_envelope=envelope,
        safety_gate=gate,
        franka_reference_q_rad=profile["franka"]["default_q_rad"],
        maximum_start_error_rad=0.01,
        maximum_tick_target_delta_rad=0.0031,
        maximum_episode_delta_rad=maximum_episode_delta_rad,
        franka_static_provenance_prevalidated=True,
        franka_action_contract_id=franka_action_contract_id,
        **kwargs,
    )


def test_supervised_admission_target_count_boundary_is_seven_hundred_twenty() -> None:
    admission = _supervised_admission(steps=720)
    assert admission.requested_policy_steps == 720
    with pytest.raises(ValueError, match="policy-step bound exceeds 720"):
        _supervised_admission(steps=721)


def test_qd_g015_admission_seeds_the_dual_ack_ledger_with_zero_action() -> None:
    admission = _supervised_admission(
        franka_action_contract_id=QD_G015_FRANKA_ACTION_CONTRACT_ID,
        initial_previous_action13=np.zeros(13, dtype=np.float32),
    )
    native_factory = _NativeSessionFactory()
    runtime = _supervised_factory(native_factory=native_factory)(admission)
    assert admission.maximum_episode_delta_rad is None
    np.testing.assert_array_equal(
        runtime.action_ledger.previous_executed_action13(),
        np.zeros(13, dtype=np.float32),
    )


def test_qd_g015_admission_constructs_python_fallback_with_disabled_episode() -> None:
    admission = _supervised_admission(
        franka_action_contract_id=QD_G015_FRANKA_ACTION_CONTRACT_ID,
        initial_previous_action13=np.zeros(13, dtype=np.float32),
    )
    transport = _ManagedFakeTransport(time.monotonic)
    factory = SupervisedV94RuntimeFactory(
        franka_backend_factory=lambda: None,
        rh56_transport_factory=_TransportFactory(transport),
        rh56_actuator_factory=_actuator_factory(time.monotonic),
        policy_tick_source=_Source(),
        live_safety_supervisor=_SupervisedSafetySupervisor(),
        franka_stop_join_timeout_s=1.0,
        commit_poll_interval_s=0.0001,
        monotonic=time.monotonic,
        realtime=time.time,
        sleep=time.sleep,
    )

    runtime = factory(admission)

    assert isinstance(runtime.franka_session, FrankaPersistentSession)
    assert (
        runtime.franka_session.franka_action_contract_id
        == QD_G015_FRANKA_ACTION_CONTRACT_ID
    )
    assert runtime.franka_session.supervised_maximum_episode_delta_rad is None


def test_legacy_admission_retains_the_home_centered_episode_envelope() -> None:
    admission = _supervised_admission()
    assert admission.maximum_episode_delta_rad == pytest.approx(0.05)


def test_qd_g015_admission_rejects_a_legacy_previous_action_reset() -> None:
    with pytest.raises(ValueError, match="requires zero previous action"):
        _supervised_admission(
            franka_action_contract_id=QD_G015_FRANKA_ACTION_CONTRACT_ID,
            initial_previous_action13=np.asarray(
                [0.0] * 7 + [-1.0] * 6, dtype=np.float32
            ),
        )


class _FakeNativeSession:
    """Minimal independent-owner surface consumed by the combined runtime."""

    def __init__(
        self,
        *,
        admission,
        action_ledger,
        events=None,
        ack_delay_s: float = 0.0,
        fail_before_ack: bool = False,
    ):
        self.target_hold = FrankaTargetSampleHold()
        self.pose_ring = FrankaPoseRing(capacity=8)
        self.c2_bootstrap_ready = True
        self.envelope = admission.franka_envelope
        self.action_ledger = action_ledger
        self.events = [] if events is None else events
        self.ack_delay_s = float(ack_delay_s)
        self.fail_before_ack = bool(fail_before_ack)
        self.last_telemetry = SimpleNamespace(
            stop_requested=False,
            stop_verified=False,
        )
        self._stop = threading.Event()
        self.target_seen = None
        self.franka_ack_monotonic_s = None

    def run(self, *, authorization, preflight_token):
        assert authorization is not None
        assert preflight_token is not None
        target = self.target_hold.peek()
        if target is None:
            raise AssertionError("runtime did not publish into native target_hold")
        self.target_seen = target
        self.events.append("franka_target_received")
        if self.ack_delay_s > 0.0:
            time.sleep(self.ack_delay_s)
        if self.fail_before_ack:
            self.events.append("franka_failed_before_ack")
            self.last_telemetry = SimpleNamespace(
                stop_requested=True,
                stop_verified=True,
                fault_reason="synthetic Franka failure before ACK",
            )
            raise RuntimeError("synthetic Franka failure before ACK")
        self.franka_ack_monotonic_s = time.monotonic()
        self.action_ledger.acknowledge(
            "franka",
            sequence=target.sequence,
            now_monotonic_s=self.franka_ack_monotonic_s,
        )
        self.events.append("franka_ack")
        if not self._stop.wait(timeout=2.0):
            raise AssertionError("test runtime did not request native stop")
        self.last_telemetry = SimpleNamespace(
            stop_requested=True,
            stop_verified=True,
        )
        return self.last_telemetry

    def request_clean_stop(self):
        self._stop.set()


class _NativeSessionFactory:
    def __init__(
        self,
        *,
        events=None,
        ack_delay_s: float = 0.0,
        fail_before_ack: bool = False,
    ):
        self.calls = []
        self.events = [] if events is None else events
        self.ack_delay_s = float(ack_delay_s)
        self.fail_before_ack = bool(fail_before_ack)

    def __call__(self, *, admission, action_ledger):
        session = _FakeNativeSession(
            admission=admission,
            action_ledger=action_ledger,
            events=self.events,
            ack_delay_s=self.ack_delay_s,
            fail_before_ack=self.fail_before_ack,
        )
        self.calls.append((admission, action_ledger, session))
        return session


class _BootstrapNativeSession:
    """Fake already-bootstrapped independent owner for production-path tests."""

    def __init__(self, *, admission, action_ledger, events):
        self.target_hold = FrankaTargetSampleHold()
        self.pose_ring = FrankaPoseRing(capacity=8)
        started = time.monotonic()
        for cycle, offset in ((1, 0.0), (2, 0.101)):
            self.pose_ring.publish(
                FrankaPoseSample(
                    cycle=cycle,
                    realtime_s=time.time() + offset,
                    monotonic_s=started + offset,
                    q_rad=np.zeros(7),
                    dq_rad_s=np.zeros(7),
                    T_base_eef=np.eye(4),
                )
            )
        self.c2_bootstrap_ready = True
        self.envelope = admission.franka_envelope
        self.action_ledger = action_ledger
        self.events = events
        self.last_telemetry = SimpleNamespace(
            stop_requested=False,
            stop_verified=False,
        )
        self._stop = threading.Event()

    def run(self, *, authorization, preflight_token):
        assert authorization is not None
        assert preflight_token is not None
        while self.target_hold.peek() is None:
            if self._stop.wait(timeout=0.0005):
                self.last_telemetry = SimpleNamespace(
                    stop_requested=True,
                    stop_verified=True,
                )
                return self.last_telemetry
        target = self.target_hold.peek()
        assert target is not None
        self.events.append("franka_target_received")
        time.sleep(0.010)
        self.action_ledger.acknowledge(
            "franka",
            sequence=target.sequence,
            now_monotonic_s=time.monotonic(),
        )
        self.events.append("franka_ack")
        if not self._stop.wait(timeout=2.0):
            raise AssertionError("test runtime did not request native stop")
        self.last_telemetry = SimpleNamespace(
            stop_requested=True,
            stop_verified=True,
        )
        return self.last_telemetry

    def request_clean_stop(self):
        self._stop.set()


class _BootstrapNativeFactory:
    def __init__(self, events):
        self.events = events
        self.sessions = []

    def __call__(self, *, admission, action_ledger):
        session = _BootstrapNativeSession(
            admission=admission,
            action_ledger=action_ledger,
            events=self.events,
        )
        self.sessions.append(session)
        return session


class _OwnerAwareSource(_Source):
    def __init__(self, events):
        super().__init__()
        self.events = events
        self.owner = None

    def prepare(self, **kwargs):
        assert self.owner is not None
        self.events.append("policy_prepare")
        return super().prepare(**kwargs)


class _OwnerAwareSourceFactory:
    construction_is_inert = True

    def __init__(self, source, events):
        self.source = source
        self.events = events
        self.closed = False

    def open_and_warm_camera(self, admission, *, hard_deadline_monotonic_s):
        assert admission.hard_deadline_monotonic_s == hard_deadline_monotonic_s
        self.events.append("camera_warm")

    def wait_for_camera_pose_alignment(
        self,
        *,
        franka_session,
        hard_deadline_monotonic_s,
    ):
        assert franka_session.c2_bootstrap_ready
        assert hard_deadline_monotonic_s > time.monotonic()
        self.events.append("camera_pose_aligned")

    def __call__(
        self,
        *,
        franka_session,
        rh56_feedback_source,
        hard_deadline_monotonic_s,
    ):
        assert franka_session.c2_bootstrap_ready
        assert hard_deadline_monotonic_s > time.monotonic()
        self.source.owner = rh56_feedback_source
        self.events.append("policy_source_build")
        return self.source

    def close(self):
        self.closed = True
        self.events.append("camera_close")


class _FakeRH56Owner:
    independent_watchdog_active = False
    fault_callback_configured = False
    first_command_readiness_verified = False

    def __init__(self, *, action_ledger, fault_callback, events):
        self.action_ledger = action_ledger
        self.fault_callback = fault_callback
        self.events = events
        self.stopped = False

    def start(self):
        self.independent_watchdog_active = True
        self.fault_callback_configured = callable(self.fault_callback)
        self.first_command_readiness_verified = True
        self.events.append("rh56_owner_ready")

    def submit(self, command):
        self.events.append("rh56_submit")
        self.action_ledger.acknowledge(
            "rh56",
            sequence=command.sequence,
            now_monotonic_s=time.monotonic(),
        )
        return SimpleNamespace(sequence=command.sequence)

    def execute(self, command, *, timeout_s=None):
        del timeout_s
        assert self.action_ledger.consumer_acknowledged(
            "franka",
            sequence=command.sequence,
        )
        self.events.append("rh56_execute")
        return self.submit(command)

    def raise_if_faulted(self):
        return None

    def feedback_history_snapshot(self, *, maximum_age_s=None):
        assert maximum_age_s is None or maximum_age_s > 0.0
        return SimpleNamespace(samples=(), fresh=True)

    def require_fresh_feedback_history(self, *, minimum_samples, maximum_age_s):
        assert minimum_samples >= 1
        assert maximum_age_s > 0.0
        return SimpleNamespace(samples=("feedback",) * minimum_samples, fresh=True)

    def request_stop(self):
        self.stopped = True

    def stop_and_close(self, *, timeout_s=None):
        assert timeout_s is None or timeout_s > 0.0
        self.stopped = True
        return SimpleNamespace(
            worker_stopped=True,
            rh56_disabled_verified=True,
            disable_attempted=True,
            close_attempted=True,
            serial_closed=True,
            stop_report=None,
            fault=None,
        )


class _SettlingFakeRH56Owner(_FakeRH56Owner):
    """Owner fake whose challenged hand becomes verified during final hold."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_settle_started = False
        self.final_settle_snapshot_calls = 0
        self.execute_count = 0

    def execute(self, command, *, timeout_s=None):
        self.execute_count += 1
        return super().execute(command, timeout_s=timeout_s)

    def physical_tracking_snapshot(self):
        if self.final_settle_started:
            self.final_settle_snapshot_calls += 1
        verified = (
            self.final_settle_started
            and self.final_settle_snapshot_calls >= 2
        )
        return SimpleNamespace(
            verdict="verified" if verified else "challenged_unverified",
            challenged_axes=(True, False, False, False, False, False),
            verified_axes=(verified, False, False, False, False, False),
            failed_axes=(False,) * 6,
            active_challenge_axes=(not verified, False, False, False, False, False),
            latest_feedback_fresh=True,
            latest_feedback_age_s=0.0,
            latest_target=(900,) * 6,
            latest_angles=((995,) + (1000,) * 5) if verified else (1000,) * 6,
            latest_positions=((995,) + (1000,) * 5) if verified else (1000,) * 6,
            tracking_significant_gap_units=30,
            tracking_min_progress_units=3,
            tracking_timeout_s=0.75,
            failure_reason=None,
        )

    def begin_final_tracking_settle(self, *, timeout_s=None):
        assert timeout_s is None or timeout_s > 0.0
        assert self.action_ledger.pending_sequence is None
        assert self.action_ledger.last_committed_sequence == 1
        self.final_settle_started = True
        self.events.append("rh56_final_tracking_settle")
        return time.monotonic() + 0.5


class _OrderedTransport(_ManagedFakeTransport):
    def __init__(self, events):
        super().__init__(time.monotonic)
        self.events = events
        self.numeric_writes = []

    def write_angle_set(self, values, *, deadline_monotonic_s):
        target = tuple(int(value) for value in values)
        if target != (-1,) * 6:
            self.numeric_writes.append(target)
            self.events.append("rh56_numeric_write")
        return super().write_angle_set(
            target,
            deadline_monotonic_s=deadline_monotonic_s,
        )


def _supervised_factory(*, native_factory, transport=None, source=None):
    transport = transport or _ManagedFakeTransport(time.monotonic)
    return SupervisedV94RuntimeFactory(
        supervised_franka_session_factory=native_factory,
        rh56_transport_factory=_TransportFactory(transport),
        rh56_actuator_factory=_actuator_factory(time.monotonic),
        policy_tick_source=_Source() if source is None else source,
        live_safety_supervisor=_SupervisedSafetySupervisor(),
        franka_stop_join_timeout_s=1.0,
        commit_poll_interval_s=0.0001,
        monotonic=time.monotonic,
        realtime=time.time,
        sleep=time.sleep,
    )


class _CommitFailingSource(_Source):
    def commit(self, command, *, action_ledger):
        assert action_ledger.last_committed_sequence == command.sequence
        raise RuntimeError("synthetic source commit failure after dual ACK")


class _StopAfterCommitSource(_Source):
    def __init__(self, stop_requested):
        super().__init__()
        self.stop_requested = stop_requested

    def commit(self, command, *, action_ledger):
        super().commit(command, action_ledger=action_ledger)
        self.stop_requested.set()


def test_supervised_factory_adopts_native_sessions_exact_target_hold():
    admission = _supervised_admission()
    native_factory = _NativeSessionFactory()
    runtime = _supervised_factory(
        native_factory=native_factory,
    )(admission)

    assert len(native_factory.calls) == 1
    created_admission, created_ledger, native_session = native_factory.calls[0]
    assert created_admission is admission
    assert created_ledger is runtime.action_ledger
    assert runtime.franka_session is native_session
    assert runtime.target_hold is native_session.target_hold


def test_formal_factory_rejects_a_native_only_franka_factory(tmp_path):
    admission, _gate, _shadow = _admission(tmp_path, steps=1)
    native_factory = _NativeSessionFactory()
    transport = _ManagedFakeTransport(lambda: 10.0)
    factory = BoundedV94C2RuntimeFactory(
        supervised_franka_session_factory=native_factory,
        rh56_transport_factory=_TransportFactory(transport),
        rh56_actuator_factory=_actuator_factory(lambda: 10.0),
        policy_tick_source=_Source(),
        live_safety_supervisor=_SupervisedSafetySupervisor(),
        franka_stop_join_timeout_s=1.0,
        commit_poll_interval_s=0.0001,
        monotonic=lambda: 10.0,
        realtime=lambda: 1010.0,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(
        ValueError,
        match="formal bounded C2 runtime requires a Franka backend factory",
    ):
        factory(admission)

    assert native_factory.calls == []


def test_supervised_final_action_is_held_for_one_policy_period_and_recorded():
    admission = _supervised_admission()
    native_factory = _NativeSessionFactory()
    runtime = _supervised_factory(
        native_factory=native_factory,
    )(admission)
    native_session = native_factory.calls[0][2]

    try:
        result = runtime.run(
            maximum_policy_steps=1,
            hard_deadline_monotonic_s=admission.hard_deadline_monotonic_s,
            stop_requested=threading.Event(),
        )
        returned_monotonic_s = time.monotonic()
    finally:
        proof = runtime.stop_and_verify()

    assert result["completed_policy_steps"] == 1
    assert result["final_policy_hold_completed"] is True
    assert result["final_policy_hold_s"] == pytest.approx(1.0 / 60.0)
    assert native_session.franka_ack_monotonic_s is not None
    assert (
        returned_monotonic_s - native_session.franka_ack_monotonic_s
        >= 1.0 / 60.0 - 1.0e-4
    )
    assert native_session.target_seen is runtime.target_hold.peek()
    assert result["committed_commands"][-1]["sequence"] == 1
    assert proof.franka_stop_verified is True
    assert proof.rh56_disabled_verified is True


def test_dual_ack_evidence_survives_policy_source_commit_failure():
    admission = _supervised_admission()
    native_factory = _NativeSessionFactory()
    runtime = _supervised_factory(
        native_factory=native_factory,
        source=_CommitFailingSource(),
    )(admission)

    try:
        with pytest.raises(
            BoundedC2HardwareRuntimeError,
            match="source commit failure after dual ACK",
        ):
            runtime.run(
                maximum_policy_steps=1,
                hard_deadline_monotonic_s=admission.hard_deadline_monotonic_s,
                stop_requested=threading.Event(),
            )
    finally:
        proof = runtime.stop_and_verify()

    assert runtime.action_ledger.last_committed_sequence == 1
    assert runtime.failure_pending_command is None
    evidence = runtime.committed_commands_snapshot
    assert len(evidence) == 1
    assert evidence[0]["sequence"] == 1
    assert evidence[0]["dual_ack_completed"] is True
    assert evidence[0]["policy_source_commit_completed"] is False
    assert proof.franka_stop_verified is True
    assert proof.rh56_disabled_verified is True


def test_dual_ack_evidence_survives_stop_during_final_policy_hold():
    admission = _supervised_admission()
    native_factory = _NativeSessionFactory()
    external_stop = threading.Event()
    runtime = _supervised_factory(
        native_factory=native_factory,
        source=_StopAfterCommitSource(external_stop),
    )(admission)

    try:
        with pytest.raises(
            BoundedC2HardwareRuntimeError,
            match="stop requested during final 60 Hz policy-target hold",
        ):
            runtime.run(
                maximum_policy_steps=1,
                hard_deadline_monotonic_s=admission.hard_deadline_monotonic_s,
                stop_requested=external_stop,
            )
    finally:
        proof = runtime.stop_and_verify()

    assert runtime.action_ledger.last_committed_sequence == 1
    assert runtime.failure_pending_command is None
    evidence = runtime.committed_commands_snapshot
    assert len(evidence) == 1
    assert evidence[0]["sequence"] == 1
    assert evidence[0]["dual_ack_completed"] is True
    assert evidence[0]["policy_source_commit_completed"] is True
    assert proof.franka_stop_verified is True
    assert proof.rh56_disabled_verified is True


def test_supervised_rh56_numeric_write_occurs_only_after_same_sequence_franka_ack():
    admission = _supervised_admission()
    events = []
    native_factory = _NativeSessionFactory(
        events=events,
        ack_delay_s=0.020,
    )
    transport = _OrderedTransport(events)
    runtime = _supervised_factory(
        native_factory=native_factory,
        transport=transport,
    )(admission)

    try:
        result = runtime.run(
            maximum_policy_steps=1,
            hard_deadline_monotonic_s=admission.hard_deadline_monotonic_s,
            stop_requested=threading.Event(),
        )
        numeric_writes_before_stop = list(transport.numeric_writes)
    finally:
        proof = runtime.stop_and_verify()

    assert result["completed_policy_steps"] == 1
    assert numeric_writes_before_stop == [(901,) * 6]
    assert events.index("franka_target_received") < events.index("franka_ack")
    assert events.index("franka_ack") < events.index("rh56_numeric_write")
    assert proof.franka_stop_verified is True
    assert proof.rh56_disabled_verified is True


def test_supervised_franka_failure_before_ack_performs_zero_rh56_numeric_writes():
    admission = _supervised_admission()
    events = []
    native_factory = _NativeSessionFactory(
        events=events,
        ack_delay_s=0.010,
        fail_before_ack=True,
    )
    transport = _OrderedTransport(events)
    runtime = _supervised_factory(
        native_factory=native_factory,
        transport=transport,
    )(admission)

    try:
        with pytest.raises(
            BoundedC2HardwareRuntimeError,
            match="Franka.*before.*ACK|Franka owner failed|franka failed sequence",
        ):
            runtime.run(
                maximum_policy_steps=1,
                hard_deadline_monotonic_s=admission.hard_deadline_monotonic_s,
                stop_requested=threading.Event(),
            )
        numeric_writes_before_stop = list(transport.numeric_writes)
    finally:
        proof = runtime.stop_and_verify()

    assert events[:2] == ["franka_target_received", "franka_failed_before_ack"]
    assert numeric_writes_before_stop == []
    assert runtime.action_ledger.last_committed_sequence == 0
    assert runtime.action_ledger.pending_sequence == 1
    assert "franka failed sequence 1" in runtime.action_ledger.fault_reason
    assert proof.franka_stop_verified is True
    assert proof.rh56_disabled_verified is True


def test_production_owner_path_orders_franka_ack_before_rh56():
    admission = _supervised_admission()
    events = []
    native_factory = _BootstrapNativeFactory(events)
    source = _OwnerAwareSource(events)
    source_factory = _OwnerAwareSourceFactory(source, events)
    owners = []

    def owner_factory(*, admission, action_ledger, fault_callback):
        del admission
        owner = _FakeRH56Owner(
            action_ledger=action_ledger,
            fault_callback=fault_callback,
            events=events,
        )
        owners.append(owner)
        return owner

    factory = SupervisedV94RuntimeFactory(
        supervised_franka_session_factory=native_factory,
        rh56_watchdog_owner_factory=owner_factory,
        policy_tick_source_factory=source_factory,
        live_safety_supervisor=_SupervisedSafetySupervisor(),
        franka_stop_join_timeout_s=1.0,
        commit_poll_interval_s=0.0001,
        monotonic=time.monotonic,
        realtime=time.time,
        sleep=time.sleep,
    )
    runtime = factory(admission)

    try:
        result = runtime.run(
            maximum_policy_steps=1,
            hard_deadline_monotonic_s=admission.hard_deadline_monotonic_s,
            stop_requested=threading.Event(),
        )
    finally:
        proof = runtime.stop_and_verify()

    assert result["completed_policy_steps"] == 1
    assert len(owners) == 1
    assert events.index("rh56_owner_ready") < events.index("policy_source_build")
    assert events.index("policy_source_build") < events.index("policy_prepare")
    assert events.index("policy_prepare") < events.index("franka_target_received")
    assert events.index("franka_target_received") < events.index("franka_ack")
    assert events.index("franka_ack") < events.index("rh56_execute")
    assert source_factory.closed is True
    assert proof.franka_stop_verified is True
    assert proof.rh56_disabled_verified is True


def test_short_supervised_run_waits_for_tracking_without_an_extra_policy_action():
    admission = _supervised_admission()
    events = []
    native_factory = _BootstrapNativeFactory(events)
    source = _OwnerAwareSource(events)
    source_factory = _OwnerAwareSourceFactory(source, events)
    owners = []

    def owner_factory(*, admission, action_ledger, fault_callback):
        del admission
        owner = _SettlingFakeRH56Owner(
            action_ledger=action_ledger,
            fault_callback=fault_callback,
            events=events,
        )
        owners.append(owner)
        return owner

    runtime = SupervisedV94RuntimeFactory(
        supervised_franka_session_factory=native_factory,
        rh56_watchdog_owner_factory=owner_factory,
        policy_tick_source_factory=source_factory,
        live_safety_supervisor=_SupervisedSafetySupervisor(),
        franka_stop_join_timeout_s=1.0,
        commit_poll_interval_s=0.0001,
        monotonic=time.monotonic,
        realtime=time.time,
        sleep=time.sleep,
    )(admission)

    try:
        result = runtime.run(
            maximum_policy_steps=1,
            hard_deadline_monotonic_s=admission.hard_deadline_monotonic_s,
            stop_requested=threading.Event(),
        )
        # The final hand-only settle must take the arm through its native
        # clean-stop path instead of waiting long enough to trip the policy
        # inter-target watchdog.
        assert native_factory.sessions[0]._stop.is_set()
    finally:
        proof = runtime.stop_and_verify()

    assert result["completed_policy_steps"] == 1
    assert result["last_dual_ack_sequence"] == 1
    assert result["rh56_physical_tracking"]["verdict"] == "verified"
    assert result["final_rh56_tracking_settle_s"] > 0.0
    assert owners[0].execute_count == 1
    assert events.count("policy_prepare") == 1
    assert events.count("rh56_execute") == 1
    assert events.count("rh56_final_tracking_settle") == 1
    assert runtime.action_ledger.last_committed_sequence == 1
    assert proof.franka_stop_verified is True
    assert proof.rh56_disabled_verified is True
