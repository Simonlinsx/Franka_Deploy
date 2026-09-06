from __future__ import annotations

import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from sim2real.runtime.bounded_c2_orchestrator import (
    BoundedC2AdmissionError,
    BoundedC2Orchestrator,
    BoundedC2RunError,
    BoundedC2State,
)
from sim2real.runtime.bounded_c2_runtime import (
    BoundedC2HardwareRuntimeError,
    BoundedPolicyTickHold,
    BoundedV94C2Runtime,
    BoundedV94C2RuntimeFactory,
    LinuxRH56TransportFactory,
)
from sim2real.closed_loop_core import (
    ClosedLoopCommand,
    ClosedLoopProtocolError,
    SafetyState,
)
from robot_control.rh56.actuator import (
    RH56SafetyFeedback,
    RH56TransactionalActuator,
)
from robot_control.rh56.watchdog import RH56OwnerFault, RH56OwnerStopResult
from sim2real.tests.test_bounded_c2_orchestrator import _admission


class _Clock:
    def __init__(self, value=10.0):
        self._value = float(value)
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            return self._value

    def advance(self, seconds):
        with self._lock:
            self._value += float(seconds)

    def set(self, value):
        with self._lock:
            self._value = float(value)


def _state(*, mode="move"):
    return SimpleNamespace(
        q=np.zeros(7),
        dq=np.zeros(7),
        O_T_EE=np.eye(4).reshape(16, order="F"),
        F_T_EE=np.eye(4).reshape(16, order="F"),
        m_ee=0.6,
        F_x_Cee=np.asarray([0.0, 0.0, 0.05]),
        I_ee=np.diag([0.01, 0.01, 0.005]).reshape(9, order="F"),
        m_load=0.0,
        F_x_Cload=np.zeros(3),
        I_load=np.zeros(9),
        m_total=0.6,
        robot_mode=mode,
        current_errors={},
        joint_contact=np.zeros(7),
        joint_collision=np.zeros(7),
        cartesian_contact=np.zeros(6),
        cartesian_collision=np.zeros(6),
        control_command_success_rate=1.0,
    )


class _Control:
    def __init__(
        self,
        backend,
        clock,
        *,
        fail_first_write=False,
        initial_read_delay_s=0.0,
    ):
        self.backend = backend
        self.clock = clock
        self.fail_first_write = bool(fail_first_write)
        self.initial_read_delay_s = float(initial_read_delay_s)
        self.read_count = 0
        self.write_count = 0
        self.bootstrap_write_count = 0
        self.writes = []
        self.finishes = 0

    def read_once(self):
        # Mirror, rather than accelerate, the commissioned 1 kHz cadence so
        # the fake target-age clock cannot outrun wall-time scheduling.
        if self.read_count == 0 and self.initial_read_delay_s > 0.0:
            time.sleep(self.initial_read_delay_s)
        time.sleep(0.001)
        self.read_count += 1
        self.clock.advance(0.001)
        return _state(), 0.0 if self.read_count == 1 else 0.001

    def write_once(self, q_rad, *, sequence):
        self.write_count += 1
        if self.fail_first_write and self.write_count == 1:
            # Give the main RH56 owner enough wall time to complete its exact
            # write/readback/feedback acknowledgement first.
            time.sleep(0.020)
            raise RuntimeError("synthetic Franka first-write failure")
        self.writes.append((int(sequence), tuple(float(v) for v in q_rad)))
        self.backend.first_write.set()

    def write_bootstrap_hold(self, q_rad):
        self.bootstrap_write_count += 1
        self.backend.bootstrap_write.set()
        assert np.asarray(q_rad).shape == (7,)

    def finish(self, q_rad):
        del q_rad
        self.finishes += 1


class _Backend:
    def __init__(
        self,
        clock,
        *,
        fail_first_write=False,
        initial_read_delay_s=0.0,
    ):
        self.clock = clock
        self.first_write = threading.Event()
        self.bootstrap_write = threading.Event()
        self.control = _Control(
            self,
            clock,
            fail_first_write=fail_first_write,
            initial_read_delay_s=initial_read_delay_s,
        )
        self.factory_calls = 0
        self.start_calls = 0
        self.stop_calls = 0
        self.stop_reads = 0
        self.close_calls = 0

    def factory(self):
        self.factory_calls += 1
        return self

    def start_joint_position_session(self):
        self.start_calls += 1
        return self.control

    def request_stop(self):
        self.stop_calls += 1

    def read_post_stop_state(self):
        self.stop_reads += 1
        return _state(mode="idle")

    def close(self):
        self.close_calls += 1


class _ManagedFakeTransport:
    hardware_backed = False
    baud_rate = 115200
    transport_name = "managed-fake-rh56"

    def __init__(self, clock, *, fail_numeric_after_franka_ack=None):
        self.clock = clock
        self.fail_numeric_after_franka_ack = fail_numeric_after_franka_ack
        self.is_open = False
        self.open_calls = 0
        self.close_calls = 0
        self.targets = (-1,) * 6
        self.writes = []
        self.read_target_calls = 0
        self.feedback_calls = 0
        self.numeric_write_completed = threading.Event()

    def open(self):
        self.open_calls += 1
        self.is_open = True
        return self

    def close(self):
        self.close_calls += 1
        self.is_open = False

    def write_angle_set(self, values, *, deadline_monotonic_s):
        assert self.is_open
        assert self.clock() <= deadline_monotonic_s
        targets = tuple(int(v) for v in values)
        self.writes.append(targets)
        if targets != (-1,) * 6:
            waiter = self.fail_numeric_after_franka_ack
            if waiter is not None:
                assert waiter.wait(timeout=1.0)
                raise OSError("synthetic RH56 numeric-write failure")
            self.numeric_write_completed.set()
        self.targets = targets

    def read_angle_set(self, *, deadline_monotonic_s):
        assert self.is_open
        assert self.clock() <= deadline_monotonic_s
        self.read_target_calls += 1
        return self.targets

    def read_safety_feedback(self, *, deadline_monotonic_s):
        assert self.is_open
        assert self.clock() <= deadline_monotonic_s
        self.feedback_calls += 1
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


class _TransportFactory:
    construction_is_inert = True

    def __init__(self, transport, gate_to_fault=None):
        self.transport = transport
        self.gate_to_fault = gate_to_fault
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.gate_to_fault is not None:
            self.gate_to_fault.latch_fault("fault during inert transport construction")
        return self.transport


class _Supervisor:
    physical_interlocks_configured = True
    independent_command_watchdogs_configured = True
    verified_dual_device_stop_supported = True

    def __init__(self):
        self.calls = 0

    def require_motion(self, admission, *, now_monotonic_s):
        self.calls += 1
        admission.safety_gate.require_motion(
            run_id=admission.run_id,
            now_monotonic_s=now_monotonic_s,
        )


class _Source:
    def __init__(self):
        self.prepared = []
        self.committed = []
        self.aborted = []

    def prepare(
        self,
        *,
        sequence,
        previous_executed_action13,
        now_monotonic_s,
        hard_deadline_monotonic_s,
    ):
        assert now_monotonic_s < hard_deadline_monotonic_s
        action = np.full(13, sequence / 10.0, dtype=np.float32)
        command = ClosedLoopCommand(
            sequence=sequence,
            produced_monotonic_s=now_monotonic_s,
            observation_realtime_s=1000.0 + now_monotonic_s,
            previous_executed_action13_used=previous_executed_action13,
            raw_policy_action13=action,
            executed_policy_action13=action,
            franka_target_q_rad=np.zeros(7),
            rh56_angle_set_register_order=np.asarray(
                [900 + sequence] * 6,
                dtype=np.int32,
            ),
        )
        self.prepared.append(command)
        return command

    def commit(self, command, *, action_ledger):
        assert action_ledger.last_committed_sequence == command.sequence
        assert action_ledger.pending_sequence is None
        self.committed.append(command.sequence)

    def abort(self, command, *, reason):
        self.aborted.append((command.sequence, str(reason)))


class _HoldOnceSource(_Source):
    def __init__(self, hold_sequence):
        super().__init__()
        self.hold_sequence = int(hold_sequence)
        self.hold_returned = False
        self.prepare_sequences = []

    def prepare(self, **kwargs):
        sequence = int(kwargs["sequence"])
        self.prepare_sequences.append(sequence)
        if sequence == self.hold_sequence and not self.hold_returned:
            self.hold_returned = True
            now = float(kwargs["now_monotonic_s"])
            return BoundedPolicyTickHold(
                sequence=sequence,
                observed_monotonic_s=now,
                retry_not_before_monotonic_s=now,
                camera_frame_id=41,
                reason="camera_frame_policy_reuse_limit",
            )
        return super().prepare(**kwargs)


class _PersistentHoldSource(_Source):
    def __init__(self):
        super().__init__()
        self.prepare_sequences = []
        self.hold_count = 0

    def prepare(self, **kwargs):
        sequence = int(kwargs["sequence"])
        now = float(kwargs["now_monotonic_s"])
        self.prepare_sequences.append(sequence)
        self.hold_count += 1
        return BoundedPolicyTickHold(
            sequence=sequence,
            observed_monotonic_s=now,
            retry_not_before_monotonic_s=now + 0.002,
            camera_frame_id=100 + self.hold_count // 2,
            reason="object_pointcloud_transient_invalid",
        )

    @property
    def diagnostics_snapshot(self):
        return {
            "last_pointcloud_status": "stale_palm",
            "last_camera_frame_id": 109,
            "last_pointcloud_frame_id": 100,
        }

    @property
    def diagnostics_summary(self):
        return (
            "perception=pointcloud_status=stale_palm,"
            "pointcloud_frame=100,latest_camera_frame=109"
        )


class _FakeWatchdogOwner:
    independent_watchdog_active = False
    fault_callback_configured = False

    def __init__(
        self,
        *,
        admission,
        action_ledger,
        fault_callback,
        clock,
        events,
    ):
        self.admission = admission
        self.action_ledger = action_ledger
        self.fault_callback = fault_callback
        self.clock = clock
        self.events = events
        self.started = False
        self.stopped = False
        self.first_command_readiness_verified = False

    def start(self):
        self.events.append("rh56_owner_start")
        self.started = True
        self.independent_watchdog_active = True
        self.fault_callback_configured = callable(self.fault_callback)
        self.first_command_readiness_verified = True
        return self

    def submit(self, command):
        assert self.started and not self.stopped
        self.events.append(f"rh56_submit_{command.sequence}")
        self.action_ledger.acknowledge(
            "rh56",
            sequence=command.sequence,
            now_monotonic_s=self.clock(),
        )
        return SimpleNamespace(sequence=command.sequence)

    def execute(self, command, *, timeout_s=None):
        del timeout_s
        return self.submit(command)

    def raise_if_faulted(self):
        return None

    def feedback_history_snapshot(self, *, maximum_age_s):
        assert maximum_age_s > 0.0
        return SimpleNamespace(samples=("immutable-feedback",), fresh=True)

    def require_fresh_feedback_history(self, *, minimum_samples, maximum_age_s):
        assert minimum_samples >= 1
        assert maximum_age_s > 0.0
        return SimpleNamespace(
            samples=tuple("immutable-feedback" for _ in range(minimum_samples)),
            fresh=True,
        )

    def request_stop(self):
        self.events.append("rh56_owner_stop_requested")
        self.stopped = True

    def stop_and_close(self, timeout_s=None):
        assert timeout_s is None or timeout_s > 0.0
        self.events.append("rh56_owner_stop_closed")
        self.stopped = True
        return SimpleNamespace(rh56_disabled_verified=True)


class _FakeManagedSourceFactory:
    construction_is_inert = True

    def __init__(self, source, backend, events):
        self.source = source
        self.backend = backend
        self.events = events
        self.closed = False

    def open_and_warm_camera(self, admission, *, hard_deadline_monotonic_s):
        assert admission.hard_deadline_monotonic_s == hard_deadline_monotonic_s
        assert self.backend.factory_calls == 0
        self.events.append("camera_warm")

    def __call__(
        self,
        *,
        franka_session,
        rh56_feedback_source,
        hard_deadline_monotonic_s,
    ):
        assert hard_deadline_monotonic_s > 0.0
        assert franka_session.c2_bootstrap_ready
        assert len(franka_session.pose_ring.snapshot()) >= 2
        assert self.backend.control.bootstrap_write_count >= 2
        snapshot = rh56_feedback_source.feedback_history_snapshot(
            maximum_age_s=0.025
        )
        assert snapshot.fresh is True
        self.events.append("policy_source_build")
        return self.source

    def wait_for_camera_pose_alignment(
        self,
        *,
        franka_session,
        hard_deadline_monotonic_s,
    ):
        assert hard_deadline_monotonic_s > 0.0
        assert franka_session.c2_bootstrap_ready
        assert len(franka_session.pose_ring.snapshot()) >= 2
        self.events.append("camera_pose_aligned")

    def close(self):
        self.events.append("camera_close")
        self.closed = True


class _FakeTriggeredManagedSourceFactory(_FakeManagedSourceFactory):
    def __init__(self, source, backend, events, clock):
        super().__init__(source, backend, events)
        self.clock = clock
        self._trigger = {"enabled": True}

    def wait_for_rollout_trigger(
        self,
        admission,
        *,
        hard_deadline_monotonic_s,
        stop_requested,
    ):
        assert admission.hard_deadline_monotonic_s == hard_deadline_monotonic_s
        assert stop_requested.is_set() is False
        assert self.backend.factory_calls == 0
        self.events.append("throw_detected_camera_only")
        self._trigger = {
            "enabled": True,
            "detected": True,
            "detected_monotonic_s": self.clock(),
            "active_franka_owner_at_detection": False,
            "active_rh56_owner_at_detection": False,
        }
        return dict(self._trigger)

    @property
    def rollout_trigger_diagnostics(self):
        return dict(self._trigger)


class _FakePreparedTriggeredManagedSourceFactory(
    _FakeManagedSourceFactory
):
    def __init__(self, source, backend, events, clock):
        super().__init__(source, backend, events)
        self.clock = clock
        self.rollout_trigger_config = object()
        self._trigger = {"enabled": True}

    def prepare_rollout_trigger(
        self,
        admission,
        *,
        hard_deadline_monotonic_s,
        stop_requested,
    ):
        assert admission.hard_deadline_monotonic_s == hard_deadline_monotonic_s
        assert stop_requested.is_set() is False
        assert self.backend.factory_calls == 0
        self.events.append("throw_preparing_camera_only")
        return {"enabled": True, "armed": True, "detected": False}

    def detect_rollout_trigger(
        self,
        admission,
        *,
        hard_deadline_monotonic_s,
        stop_requested,
    ):
        assert hard_deadline_monotonic_s <= admission.hard_deadline_monotonic_s
        assert stop_requested.is_set() is False
        assert self.backend.factory_calls == 1
        self.events.append("throw_detected_owners_ready")
        self._trigger = {
            "enabled": True,
            "detected": True,
            "detected_monotonic_s": self.clock(),
            "active_franka_owner_at_detection": True,
            "active_rh56_owner_at_detection": True,
        }
        return dict(self._trigger)

    @property
    def rollout_trigger_diagnostics(self):
        return dict(self._trigger)


def _actuator_factory(clock):
    calls = []

    def build(transport, admission):
        assert transport.is_open
        calls.append((transport, admission))
        return RH56TransactionalActuator(
            transport,
            command_watchdog_timeout_s=(
                admission.rh56_preflight.verified_command_watchdog_timeout_s
            ),
            maximum_feedback_age_s=0.025,
            maximum_running_axis_current_ma=1400,
            maximum_temperature_c=60,
            stop_timeout_s=0.2,
            stop_verify_samples=2,
            stop_verify_interval_s=0.0,
            stop_max_axis_current_ma=100,
            stop_max_angle_drift_units=2,
            stop_max_position_drift_units=3,
            monotonic=clock,
            sleep=lambda _seconds: None,
        )

    build.calls = calls
    return build


def _runtime_factory(
    admission,
    clock,
    *,
    backend=None,
    transport=None,
    transport_factory=None,
    source=None,
    supervisor=None,
    maximum_consecutive_no_stage_hold_s=0.025,
    sleep=time.sleep,
):
    backend = backend or _Backend(clock)
    transport = transport or _ManagedFakeTransport(clock)
    transport_factory = transport_factory or _TransportFactory(transport)
    source = source or _Source()
    supervisor = supervisor or _Supervisor()
    actuator_factory = _actuator_factory(clock)
    factory = BoundedV94C2RuntimeFactory(
        franka_backend_factory=backend.factory,
        rh56_transport_factory=transport_factory,
        rh56_actuator_factory=actuator_factory,
        policy_tick_source=source,
        live_safety_supervisor=supervisor,
        franka_stop_join_timeout_s=1.0,
        commit_poll_interval_s=0.0001,
        maximum_consecutive_no_stage_hold_s=(
            maximum_consecutive_no_stage_hold_s
        ),
        monotonic=clock,
        realtime=lambda: 1000.0 + clock(),
        sleep=sleep,
    )
    return SimpleNamespace(
        factory=factory,
        backend=backend,
        transport=transport,
        transport_factory=transport_factory,
        source=source,
        supervisor=supervisor,
        actuator_factory=actuator_factory,
    )


def test_linux_transport_factory_is_inert_even_when_called(monkeypatch):
    # Other commissioning tests legitimately import the command module during
    # collection.  Isolate this assertion from collection order so it checks
    # only whether constructing/calling this factory imports the RH56 API.
    monkeypatch.delitem(sys.modules, "examples.inspire_rh56_test", raising=False)
    loads = []
    factory = LinuxRH56TransportFactory(
        "/dev/not-opened-by-test",
        module_loader=lambda name: loads.append(name),
    )
    assert "examples.inspire_rh56_test" not in sys.modules
    transport = factory()
    assert loads == []
    assert transport.is_open is False
    assert factory.construction_is_inert is True


def test_no_stage_hold_bound_must_precede_effective_rh56_watchdog_with_margin(
    tmp_path,
):
    admission, _gate, _shadow = _admission(tmp_path, steps=1)
    clock = _Clock()
    deps = _runtime_factory(
        admission,
        clock,
        maximum_consecutive_no_stage_hold_s=0.030,
    )

    with pytest.raises(
        ValueError,
        match=(
            r"plus one policy period must leave at least 10% margin.*"
            r"hold=0\.030000s policy_period=0\.016667s "
            r"watchdog=0\.050000s maximum=0\.028333s"
        ),
    ):
        deps.factory(admission)

    assert deps.transport_factory.calls == 0
    assert deps.backend.factory_calls == 0


def test_gate_monotonicity_is_per_owner_thread_not_cross_thread_arrival(tmp_path):
    admission, gate, _shadow = _admission(tmp_path, steps=1)
    gate.require_motion(run_id=admission.run_id, now_monotonic_s=10.2)
    outcomes = []

    def worker():
        # This sample is older than the main owner's high-water mark but is
        # the worker owner's first sample, so concurrent arrival is valid.
        gate.require_motion(run_id=admission.run_id, now_monotonic_s=10.1)
        outcomes.append("cross-thread-valid")
        try:
            gate.require_motion(run_id=admission.run_id, now_monotonic_s=10.09)
        except ClosedLoopProtocolError as exc:
            outcomes.append(str(exc))
        else:
            outcomes.append("same-thread-not-rejected")

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert outcomes == ["cross-thread-valid", "motion authorization clock regressed"]
    assert gate.state is SafetyState.FAULT_LATCHED


def test_successful_k_is_exact_dual_ack_count_and_both_devices_stop(tmp_path):
    admission, gate, _shadow = _admission(tmp_path, steps=2)
    clock = _Clock()
    deps = _runtime_factory(admission, clock)
    created = []

    def create(value):
        runtime = deps.factory(value)
        created.append(runtime)
        return runtime

    orchestrator = BoundedC2Orchestrator(
        admission,
        runtime_factory=create,
        monotonic=clock,
    )
    assert deps.transport_factory.calls == 0
    assert deps.transport.open_calls == 0
    assert deps.backend.factory_calls == 0

    result = orchestrator.run()

    assert result == {
        "completed_policy_steps": 2,
        "last_dual_ack_sequence": 2,
        "requested_policy_steps": 2,
        "stopped_early": False,
        "no_stage_policy_hold_count": 0,
        "no_stage_policy_hold_counts_by_reason": {},
        "last_no_stage_policy_hold": None,
        "maximum_consecutive_no_stage_hold_count": 0,
        "maximum_consecutive_no_stage_hold_duration_s": 0.0,
        "maximum_consecutive_no_stage_hold_s": 0.025,
        "no_stage_hold_effective_rh56_watchdog_s": 0.05,
        "policy_schedule": {
            "mode": "legacy_minimum_period",
            "skipped_slot_count": 0,
            "max_release_lateness_s": 0.0,
        },
        "startup_non_actuated_completed_steps": 0,
        "startup_non_actuated_policy_ticks": [],
        "counting_basis": "executed_action_ledger_dual_ack_commit",
    }
    runtime = created[0]
    assert runtime.action_ledger.last_committed_sequence == 2
    assert runtime.action_ledger.pending_sequence is None
    assert deps.source.committed == [1, 2]
    assert (
        deps.source.prepared[1].produced_monotonic_s
        - deps.source.prepared[0].produced_monotonic_s
        >= 1.0 / admission.policy_rate_hz - 1.0e-12
    )
    np.testing.assert_array_equal(
        runtime.action_ledger.previous_executed_action13(),
        deps.source.prepared[-1].executed_policy_action13,
    )
    assert deps.transport.writes[:2] == [(901,) * 6, (902,) * 6]
    assert deps.transport.writes[-2:] == [(-1,) * 6, (-1,) * 6]
    assert deps.transport.open_calls == deps.transport.close_calls == 1
    assert deps.backend.factory_calls == 1
    assert deps.backend.stop_calls == deps.backend.close_calls == 1
    assert runtime.stop_errors == ()
    assert orchestrator.state is BoundedC2State.STOPPED_VERIFIED
    assert gate.state is SafetyState.DISARMED


def test_owner_path_warms_camera_then_bootstraps_state_before_sequence_one(tmp_path):
    admission, gate, _shadow = _admission(tmp_path, steps=1)
    clock = _Clock()
    events = []
    # FCI startup deliberately exceeds the RH56 50 ms heartbeat.  The RH56
    # owner must not even be constructed/armed until measured-hold bootstrap
    # and camera/pose alignment are complete.
    backend = _Backend(clock, initial_read_delay_s=0.060)
    source = _Source()
    source_factory = _FakeManagedSourceFactory(source, backend, events)
    owners = []
    owner_factory_elapsed_s = []
    test_started = time.monotonic()

    def owner_factory(*, admission, action_ledger, fault_callback):
        owner_factory_elapsed_s.append(time.monotonic() - test_started)
        owner = _FakeWatchdogOwner(
            admission=admission,
            action_ledger=action_ledger,
            fault_callback=fault_callback,
            clock=clock,
            events=events,
        )
        owners.append(owner)
        return owner

    runtime_factory = BoundedV94C2RuntimeFactory(
        franka_backend_factory=backend.factory,
        rh56_watchdog_owner_factory=owner_factory,
        policy_tick_source_factory=source_factory,
        live_safety_supervisor=_Supervisor(),
        franka_stop_join_timeout_s=1.0,
        commit_poll_interval_s=0.0001,
        monotonic=clock,
        realtime=lambda: 1000.0 + clock(),
        sleep=time.sleep,
    )
    created = []
    orchestrator = BoundedC2Orchestrator(
        admission,
        runtime_factory=lambda value: created.append(runtime_factory(value))
        or created[-1],
        monotonic=clock,
    )

    result = orchestrator.run()

    assert result["completed_policy_steps"] == 1
    assert owner_factory_elapsed_s[0] >= 0.050
    assert events.index("camera_warm") < events.index("rh56_owner_start")
    assert events.index("camera_warm") < events.index("camera_pose_aligned")
    assert events.index("camera_pose_aligned") < events.index("rh56_owner_start")
    assert events.index("rh56_owner_start") < events.index("policy_source_build")
    assert events.index("policy_source_build") < events.index("rh56_submit_1")
    assert backend.control.bootstrap_write_count >= 2
    assert backend.control.writes
    assert {sequence for sequence, _q in backend.control.writes} == {1}
    assert created[0].franka_telemetry.bootstrap_hold_write_count >= 2
    assert source.committed == [1]
    assert owners[0].stopped is True
    assert source_factory.closed is True
    assert events[-1] == "camera_close"
    assert gate.state is SafetyState.DISARMED


def test_throw_trigger_is_camera_only_and_precedes_both_actuator_owners(tmp_path):
    admission, gate, _shadow = _admission(tmp_path, steps=1)
    clock = _Clock()
    events = []
    backend = _Backend(clock)
    source = _Source()
    source_factory = _FakeTriggeredManagedSourceFactory(
        source, backend, events, clock
    )

    def owner_factory(*, admission, action_ledger, fault_callback):
        return _FakeWatchdogOwner(
            admission=admission,
            action_ledger=action_ledger,
            fault_callback=fault_callback,
            clock=clock,
            events=events,
        )

    runtime_factory = BoundedV94C2RuntimeFactory(
        franka_backend_factory=backend.factory,
        rh56_watchdog_owner_factory=owner_factory,
        policy_tick_source_factory=source_factory,
        live_safety_supervisor=_Supervisor(),
        franka_stop_join_timeout_s=1.0,
        commit_poll_interval_s=0.0001,
        monotonic=clock,
        realtime=lambda: 1000.0 + clock(),
        sleep=time.sleep,
    )
    orchestrator = BoundedC2Orchestrator(
        admission,
        runtime_factory=runtime_factory,
        monotonic=clock,
    )

    result = orchestrator.run()

    assert result["completed_policy_steps"] == 1
    assert result["rollout_trigger"]["detected"] is True
    assert result["rollout_trigger"]["active_franka_owner_at_detection"] is False
    assert result["rollout_trigger"]["active_rh56_owner_at_detection"] is False
    assert events.index("camera_warm") < events.index("throw_detected_camera_only")
    assert events.index("throw_detected_camera_only") < events.index("rh56_owner_start")
    assert gate.state is SafetyState.DISARMED


def test_split_throw_trigger_prepares_before_owners_and_detects_after_source_ready(
    tmp_path,
):
    admission, gate, _shadow = _admission(tmp_path, steps=1)
    clock = _Clock()
    events = []
    backend = _Backend(clock)
    source = _Source()
    source_factory = _FakePreparedTriggeredManagedSourceFactory(
        source, backend, events, clock
    )

    def owner_factory(*, admission, action_ledger, fault_callback):
        return _FakeWatchdogOwner(
            admission=admission,
            action_ledger=action_ledger,
            fault_callback=fault_callback,
            clock=clock,
            events=events,
        )

    runtime_factory = BoundedV94C2RuntimeFactory(
        franka_backend_factory=backend.factory,
        rh56_watchdog_owner_factory=owner_factory,
        policy_tick_source_factory=source_factory,
        live_safety_supervisor=_Supervisor(),
        franka_stop_join_timeout_s=1.0,
        commit_poll_interval_s=0.0001,
        monotonic=clock,
        realtime=lambda: 1000.0 + clock(),
        sleep=time.sleep,
    )
    result = BoundedC2Orchestrator(
        admission,
        runtime_factory=runtime_factory,
        monotonic=clock,
    ).run()

    assert result["completed_policy_steps"] == 1
    assert result["rollout_trigger"]["active_franka_owner_at_detection"] is True
    assert result["rollout_trigger"]["active_rh56_owner_at_detection"] is True
    assert events.index("camera_warm") < events.index(
        "throw_preparing_camera_only"
    )
    assert events.index("throw_preparing_camera_only") < events.index(
        "camera_pose_aligned"
    )
    assert events.index("camera_pose_aligned") < events.index(
        "rh56_owner_start"
    )
    assert events.index("rh56_owner_start") < events.index(
        "policy_source_build"
    )
    assert events.index("policy_source_build") < events.index(
        "throw_detected_owners_ready"
    )
    assert events.index("throw_detected_owners_ready") < events.index(
        "rh56_submit_1"
    )
    detected_index = events.index("throw_detected_owners_ready")
    assert events[detected_index + 1] == "rh56_submit_1"
    assert gate.state is SafetyState.DISARMED


def test_runtime_diagnostics_do_not_expose_removed_command_window_handshake(tmp_path):
    admission, _gate, _shadow = _admission(tmp_path, steps=1)
    clock = _Clock()
    deps = _runtime_factory(admission, clock)
    runtime: BoundedV94C2Runtime = deps.factory(admission)
    diagnostics = runtime.runtime_diagnostics_snapshot
    assert not any(key.startswith("rh56_command_window") for key in diagnostics)


def test_20hz_phase_clock_does_not_drift_or_issue_catch_up_bursts(tmp_path):
    admission, _gate, _shadow = _admission(tmp_path, steps=1)
    clock = _Clock()
    runtime = _runtime_factory(admission, clock).factory(admission)
    period = 0.05

    # Variable work duration does not turn the schedule into ``now + 50 ms``.
    release = runtime._advance_phase_locked_policy_release(
        10.0,
        policy_period_s=period,
        completed_at_monotonic_s=10.012,
    )
    assert release == pytest.approx(10.05)
    release = runtime._advance_phase_locked_policy_release(
        release,
        policy_period_s=period,
        completed_at_monotonic_s=10.071,
    )
    assert release == pytest.approx(10.10)

    # If work crosses the following release, skip that slot.  Never publish a
    # burst to catch up, and never permanently shift the original phase grid.
    release = runtime._advance_phase_locked_policy_release(
        release,
        policy_period_s=period,
        completed_at_monotonic_s=10.170,
    )
    assert release == pytest.approx(10.20)
    diagnostics = runtime.runtime_diagnostics_snapshot["policy_schedule"]
    assert diagnostics["skipped_slot_count"] == 1
    assert diagnostics["max_release_lateness_s"] == pytest.approx(0.070)


def test_no_stage_camera_hold_retries_same_sequence_without_consuming_k(tmp_path):
    admission, gate, _shadow = _admission(tmp_path, steps=2)
    clock = _Clock()
    # Exercise the runtime hold branch before the fake 1 kHz owner starts; the
    # source-level tests above cover the real third-use-after-two-commits rule.
    source = _HoldOnceSource(hold_sequence=1)
    deps = _runtime_factory(admission, clock, source=source)
    created = []

    orchestrator = BoundedC2Orchestrator(
        admission,
        runtime_factory=lambda value: created.append(deps.factory(value))
        or created[-1],
        monotonic=clock,
    )

    result = orchestrator.run()

    assert result["completed_policy_steps"] == 2
    assert result["last_dual_ack_sequence"] == 2
    assert result["no_stage_policy_hold_count"] == 1
    assert result["no_stage_policy_hold_counts_by_reason"] == {
        "camera_frame_policy_reuse_limit": 1
    }
    last_hold = result["last_no_stage_policy_hold"]
    assert last_hold["sequence"] == 1
    assert last_hold["reason"] == "camera_frame_policy_reuse_limit"
    assert last_hold["camera_frame_id"] == 41
    assert last_hold["scheduled_retry_delay_s"] == 0.0
    assert last_hold["consecutive_index"] == 1
    assert last_hold["streak_first_camera_frame_id"] == 41
    assert last_hold["streak_last_camera_frame_id"] == 41
    assert last_hold["streak_frame_changes"] == 0
    assert result["maximum_consecutive_no_stage_hold_count"] == 1
    assert result["maximum_consecutive_no_stage_hold_duration_s"] >= 0.0
    assert result["maximum_consecutive_no_stage_hold_s"] == 0.025
    assert result["no_stage_hold_effective_rh56_watchdog_s"] == 0.05
    assert source.prepare_sequences == [1, 1, 2]
    assert source.committed == [1, 2]
    assert len(source.prepared) == 2
    assert (
        source.prepared[1].produced_monotonic_s
        - source.prepared[0].produced_monotonic_s
        >= 1.0 / admission.policy_rate_hz - 1.0e-12
    )
    runtime = created[0]
    assert runtime.action_ledger.last_committed_sequence == 2
    assert runtime.action_ledger.pending_sequence is None
    assert runtime._no_stage_policy_hold_deadline_monotonic_s() is None
    assert deps.transport.writes[:2] == [(901,) * 6, (902,) * 6]
    assert gate.state is SafetyState.DISARMED


def test_repeated_no_stage_holds_cannot_renew_first_observed_deadline(tmp_path):
    admission, _gate, _shadow = _admission(tmp_path, steps=1)
    clock = _Clock()
    deps = _runtime_factory(
        admission,
        clock,
        maximum_consecutive_no_stage_hold_s=0.020,
    )
    runtime = deps.factory(admission)
    first = BoundedPolicyTickHold(
        sequence=1,
        observed_monotonic_s=10.000,
        retry_not_before_monotonic_s=11.000,
        camera_frame_id=70,
        reason="first_missing_frame",
    )
    runtime._record_no_stage_policy_hold(first)
    # A far-future retry suggestion is not elapsed hold time.
    runtime._require_no_stage_policy_hold_within_bound(10.010)

    repeated = BoundedPolicyTickHold(
        sequence=1,
        observed_monotonic_s=10.019,
        retry_not_before_monotonic_s=20.000,
        camera_frame_id=71,
        reason="still_missing_frame",
    )
    runtime._record_no_stage_policy_hold(repeated)

    with pytest.raises(
        BoundedC2HardwareRuntimeError,
        match=(
            r"duration=0\.020000s.*sequence=1.*camera_frame_id=71.*"
            r"reason=still_missing_frame.*streak_first_camera_frame_id=70.*"
            r"streak_last_camera_frame_id=71.*streak_frame_changes=1"
        ),
    ):
        runtime._require_no_stage_policy_hold_within_bound(10.020)

    diagnostics = runtime.runtime_diagnostics_snapshot
    assert diagnostics["maximum_consecutive_no_stage_hold_duration_s"] == (
        pytest.approx(0.020)
    )
    assert diagnostics["last_no_stage_policy_hold"]["consecutive_duration_s"] == (
        pytest.approx(0.019)
    )
    assert runtime.action_ledger.last_committed_sequence == 0
    assert runtime.action_ledger.pending_sequence is None


def test_persistent_no_stage_hold_fails_before_actuator_watchdog_and_dual_stops(
    tmp_path,
):
    admission, gate, _shadow = _admission(tmp_path, steps=1)
    clock = _Clock()
    source = _PersistentHoldSource()
    maximum_hold_s = 0.010
    assert (
        maximum_hold_s
        < admission.rh56_preflight.verified_command_watchdog_timeout_s
    )
    events = []
    backend = _Backend(clock)
    source_factory = _FakeManagedSourceFactory(source, backend, events)
    owners = []

    def owner_factory(*, admission, action_ledger, fault_callback):
        owner = _FakeWatchdogOwner(
            admission=admission,
            action_ledger=action_ledger,
            fault_callback=fault_callback,
            clock=clock,
            events=events,
        )
        owners.append(owner)
        return owner

    runtime_factory = BoundedV94C2RuntimeFactory(
        franka_backend_factory=backend.factory,
        rh56_watchdog_owner_factory=owner_factory,
        policy_tick_source_factory=source_factory,
        live_safety_supervisor=_Supervisor(),
        franka_stop_join_timeout_s=1.0,
        commit_poll_interval_s=0.0001,
        maximum_consecutive_no_stage_hold_s=maximum_hold_s,
        monotonic=clock,
        realtime=lambda: 1000.0 + clock(),
        sleep=time.sleep,
    )
    created = []
    orchestrator = BoundedC2Orchestrator(
        admission,
        runtime_factory=lambda value: created.append(runtime_factory(value))
        or created[-1],
        monotonic=clock,
    )

    with pytest.raises(
        BoundedC2RunError,
        match=(
            r"maximum consecutive no-stage observation hold exceeded: "
            r"duration=.*sequence=1.*reason=object_pointcloud_transient_invalid"
            r".*pointcloud_status=stale_palm.*latest_camera_frame=109"
        ),
    ):
        orchestrator.run()

    runtime = created[0]
    diagnostics = runtime.runtime_diagnostics_snapshot
    assert diagnostics["maximum_consecutive_no_stage_hold_duration_s"] >= (
        maximum_hold_s
    )
    assert diagnostics["last_no_stage_policy_hold"]["sequence"] == 1
    assert diagnostics["observation_source"] == {
        "last_pointcloud_status": "stale_palm",
        "last_camera_frame_id": 109,
        "last_pointcloud_frame_id": 100,
    }
    assert source.prepare_sequences
    assert set(source.prepare_sequences) == {1}
    assert source.prepared == []
    assert source.committed == []
    assert runtime.action_ledger.last_committed_sequence == 0
    assert runtime.action_ledger.pending_sequence is None
    assert owners[0].stopped is True
    assert "rh56_owner_stop_requested" in events
    assert "rh56_owner_stop_closed" in events
    assert backend.stop_calls == backend.close_calls == 1
    assert source_factory.closed is True
    assert runtime._stop_proof is not None
    assert runtime._stop_proof.franka_stop_verified is True
    assert runtime._stop_proof.rh56_disabled_verified is True
    assert gate.state is SafetyState.FAULT_LATCHED


def test_runtime_rechecks_expiry_before_transport_factory_or_open(tmp_path):
    admission, gate, _shadow = _admission(tmp_path, steps=1)
    clock = _Clock()
    deps = _runtime_factory(admission, clock)
    runtime = deps.factory(admission)
    clock.set(admission.hard_deadline_monotonic_s)

    with pytest.raises(BoundedC2AdmissionError, match="deadline|expired"):
        runtime.run(
            maximum_policy_steps=1,
            hard_deadline_monotonic_s=admission.hard_deadline_monotonic_s,
            stop_requested=threading.Event(),
        )
    assert deps.transport_factory.calls == 0
    assert deps.transport.open_calls == 0
    assert deps.backend.factory_calls == 0
    assert gate.state is SafetyState.FAULT_LATCHED


def test_authorization_loss_during_inert_construction_prevents_open(tmp_path):
    admission, gate, _shadow = _admission(tmp_path, steps=1)
    clock = _Clock()
    transport = _ManagedFakeTransport(clock)
    transport_factory = _TransportFactory(transport, gate_to_fault=gate)
    deps = _runtime_factory(
        admission,
        clock,
        transport=transport,
        transport_factory=transport_factory,
    )
    runtime = deps.factory(admission)

    with pytest.raises(BoundedC2AdmissionError, match="not armed"):
        runtime.run(
            maximum_policy_steps=1,
            hard_deadline_monotonic_s=admission.hard_deadline_monotonic_s,
            stop_requested=threading.Event(),
        )
    assert transport_factory.calls == 1
    assert transport.open_calls == 0
    assert deps.actuator_factory.calls == []
    assert deps.backend.factory_calls == 0


def test_orchestrator_rejects_incomplete_live_safety_before_transport_open(
    tmp_path,
):
    admission, gate, _shadow = _admission(tmp_path, steps=1)
    clock = _Clock()
    supervisor = _Supervisor()
    supervisor.independent_command_watchdogs_configured = False
    deps = _runtime_factory(
        admission,
        clock,
        supervisor=supervisor,
    )
    orchestrator = BoundedC2Orchestrator(
        admission,
        runtime_factory=deps.factory,
        monotonic=clock,
    )

    with pytest.raises(BoundedC2RunError, match="watchdogs"):
        orchestrator.run()
    assert deps.transport_factory.calls == 0
    assert deps.transport.open_calls == 0
    assert deps.actuator_factory.calls == []
    assert deps.backend.factory_calls == 0
    assert gate.state is SafetyState.FAULT_LATCHED


def test_franka_first_partial_ack_then_rh56_failure_disables_and_stops(tmp_path):
    admission, gate, _shadow = _admission(tmp_path, steps=1)
    clock = _Clock()
    backend = _Backend(clock)
    transport = _ManagedFakeTransport(
        clock,
        fail_numeric_after_franka_ack=backend.first_write,
    )
    deps = _runtime_factory(
        admission,
        clock,
        backend=backend,
        transport=transport,
    )
    created = []

    def create(value):
        runtime = deps.factory(value)
        created.append(runtime)
        return runtime

    orchestrator = BoundedC2Orchestrator(
        admission,
        runtime_factory=create,
        monotonic=clock,
    )
    with pytest.raises(BoundedC2RunError, match="numeric-write failure"):
        orchestrator.run()

    runtime = created[0]
    assert runtime.action_ledger.last_committed_sequence == 0
    assert runtime.action_ledger.fault_reason is not None
    failed_command = runtime.failure_pending_command
    assert failed_command is not None
    assert failed_command.sequence == 1
    np.testing.assert_array_equal(
        failed_command.raw_policy_action13,
        deps.source.prepared[0].raw_policy_action13,
    )
    np.testing.assert_array_equal(
        failed_command.executed_policy_action13,
        deps.source.prepared[0].executed_policy_action13,
    )
    np.testing.assert_array_equal(
        failed_command.rh56_angle_set_register_order,
        np.full(6, 901, dtype=np.int32),
    )
    assert backend.control.write_count >= 1
    assert transport.writes[0] == (901,) * 6
    assert transport.writes[-2:] == [(-1,) * 6, (-1,) * 6]
    assert transport.close_calls == 1
    assert backend.stop_calls == backend.close_calls == 1
    assert gate.state is SafetyState.FAULT_LATCHED


def test_rh56_first_partial_ack_then_franka_failure_disables_and_stops(tmp_path):
    admission, gate, _shadow = _admission(tmp_path, steps=1)
    clock = _Clock()
    backend = _Backend(clock, fail_first_write=True)
    transport = _ManagedFakeTransport(clock)
    deps = _runtime_factory(
        admission,
        clock,
        backend=backend,
        transport=transport,
    )
    created = []

    def create(value):
        runtime = deps.factory(value)
        created.append(runtime)
        return runtime

    orchestrator = BoundedC2Orchestrator(
        admission,
        runtime_factory=create,
        monotonic=clock,
    )
    with pytest.raises(BoundedC2RunError, match="Franka"):
        orchestrator.run()

    runtime = created[0]
    assert runtime.action_ledger.last_committed_sequence == 0
    assert runtime.action_ledger.fault_reason is not None
    assert "franka failed sequence 1" in runtime.action_ledger.fault_reason
    assert transport.numeric_write_completed.is_set()
    assert transport.writes[0] == (901,) * 6
    assert transport.writes[-2:] == [(-1,) * 6, (-1,) * 6]
    assert transport.close_calls == 1
    assert backend.stop_calls == backend.close_calls == 1
    assert gate.state is SafetyState.FAULT_LATCHED


def test_preexisting_stop_returns_zero_committed_ticks_not_prepared_ticks(tmp_path):
    admission, _gate, _shadow = _admission(tmp_path, steps=2)
    clock = _Clock()
    deps = _runtime_factory(admission, clock)
    runtime = deps.factory(admission)
    external_stop = threading.Event()
    external_stop.set()
    result = runtime.run(
        maximum_policy_steps=2,
        hard_deadline_monotonic_s=admission.hard_deadline_monotonic_s,
        stop_requested=external_stop,
    )
    proof = runtime.stop_and_verify()

    assert result["completed_policy_steps"] == 0
    assert result["last_dual_ack_sequence"] == 0
    assert result["stopped_early"] is True
    assert deps.source.prepared == []
    assert deps.backend.factory_calls == 0
    assert deps.transport.writes[-2:] == [(-1,) * 6, (-1,) * 6]
    assert proof.franka_stop_verified is True
    assert proof.rh56_disabled_verified is True


def test_policy_release_does_not_mask_async_rh56_fault_as_clean_stop(tmp_path):
    admission, _gate, _shadow = _admission(tmp_path, steps=1)
    clock = _Clock()
    deps = _runtime_factory(admission, clock)
    runtime = deps.factory(admission)
    fault = RH56OwnerFault(
        reason="synthetic asynchronous RH56 watchdog expiry",
        detected_monotonic_s=clock(),
        pending_sequence=None,
        stop_confirmed=False,
        stop_error=None,
    )

    # Mirror the production callback ordering: publish the owner fault, set
    # the runtime stop edge and request the Franka stop.  Even if SIGINT also
    # arrives during cleanup, the already-published hardware root cause must
    # win over a normal early-return result.
    runtime._rh56_fault_callback(fault)
    external_stop = threading.Event()
    external_stop.set()

    with pytest.raises(
        BoundedC2HardwareRuntimeError,
        match="synthetic asynchronous RH56 watchdog expiry",
    ):
        runtime._wait_for_policy_release(
            clock() + 0.1,
            hard_deadline_monotonic_s=admission.hard_deadline_monotonic_s,
            external_stop_requested=external_stop,
        )


def test_policy_release_does_not_mask_async_franka_fault_as_clean_stop(tmp_path):
    admission, _gate, _shadow = _admission(tmp_path, steps=1)
    clock = _Clock()
    deps = _runtime_factory(admission, clock)
    runtime = deps.factory(admission)
    franka_fault = RuntimeError("synthetic asynchronous Franka servo fault")
    with runtime._state_lock:
        runtime._franka_error = franka_fault
    runtime._stop_requested.set()
    external_stop = threading.Event()
    external_stop.set()

    with pytest.raises(
        BoundedC2HardwareRuntimeError,
        match="synthetic asynchronous Franka servo fault",
    ) as raised:
        runtime._wait_for_policy_release(
            clock() + 0.1,
            hard_deadline_monotonic_s=admission.hard_deadline_monotonic_s,
            external_stop_requested=external_stop,
        )
    assert raised.value.__cause__ is franka_fault


@pytest.mark.parametrize(
    ("owner_verified", "expected_verified"),
    ((True, True), (False, False)),
)
def test_supervised_owner_fault_does_not_overwrite_physical_stop_proof(
    tmp_path,
    owner_verified,
    expected_verified,
):
    admission, _gate, _shadow = _admission(tmp_path, steps=1)
    clock = _Clock()
    deps = _runtime_factory(admission, clock)
    runtime = deps.factory(admission)
    runtime._supervised_non_c2 = True

    owner_fault = RH56OwnerFault(
        reason="synthetic policy transaction fault",
        detected_monotonic_s=clock(),
        pending_sequence=1,
        stop_confirmed=owner_verified,
        stop_error=None if owner_verified else "synthetic stop failure",
    )
    stop_result = RH56OwnerStopResult(
        worker_stopped=True,
        rh56_disabled_verified=owner_verified,
        disable_attempted=True,
        close_attempted=True,
        serial_closed=True,
        stop_report=None,
        fault=owner_fault,
    )

    class FaultedOwner:
        def request_stop(self):
            pass

        def stop_and_close(self, *, timeout_s):
            assert timeout_s > 0.0
            return stop_result

    runtime._rh56_watchdog_owner = FaultedOwner()
    proof = runtime.stop_and_verify()

    assert proof.rh56_disabled_verified is expected_verified
    assert any(
        "RH56 watchdog owner fault" in error
        and "synthetic policy transaction fault" in error
        for error in runtime.stop_errors
    )
    assert (
        "RH56 watchdog owner did not verify disabled stop" in runtime.stop_errors
    ) is (not expected_verified)


def test_module_exposes_no_hardware_execution_cli():
    import sim2real.runtime.bounded_c2_runtime as runtime_module

    assert not hasattr(runtime_module, "main")
    assert not hasattr(runtime_module, "build_parser")
