from __future__ import annotations

import gc
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from robot_control.franka.backend import (
    PYLIBFRANKA_STATE_FIELDS,
    PylibfrankaBackendError,
    PylibfrankaBackendFactory,
    PylibfrankaPersistentBackend,
)


class _Duration:
    def __init__(self, seconds):
        self.seconds = seconds

    def to_sec(self):
        return self.seconds


class _JointPositions:
    def __init__(self, q):
        self.q = list(q)
        self.motion_finished = False


class _RobotState:
    def __init__(self, robot_time_s):
        self.time = _Duration(robot_time_s)


class _Control:
    def __init__(self, events, states=None):
        self.events = events
        self.states = list(
            states or [_RobotState(index * 0.001) for index in range(1, 11)]
        )
        self.read_count = 0
        self.commands = []
        self.fail_write = False

    def readOnce(self):
        self.events.append("active_read")
        state = self.states[min(self.read_count, len(self.states) - 1)]
        self.read_count += 1
        return state, _Duration(0.001)

    def writeOnce(self, command):
        self.events.append(
            "active_finish" if command.motion_finished else "active_write"
        )
        if self.fail_write:
            raise RuntimeError("synthetic write failure")
        self.commands.append((tuple(command.q), bool(command.motion_finished)))


class _Robot:
    def __init__(self, events, post_stop_states=None):
        self.events = events
        self.control = _Control(events)
        self.post_stop_states = list(post_stop_states or [_RobotState(1.0)])
        self.post_stop_reads = 0
        self.stop_failure = None
        self.recovery_calls = 0

    def start_joint_position_control(self, mode):
        self.events.append(("start_control", mode))
        return self.control

    def stop(self):
        self.events.append("robot_stop")
        if self.stop_failure is not None:
            raise self.stop_failure

    def read_once(self):
        self.events.append("post_stop_read")
        value = self.post_stop_states[
            min(self.post_stop_reads, len(self.post_stop_states) - 1)
        ]
        self.post_stop_reads += 1
        return value

    def automatic_error_recovery(self):
        self.recovery_calls += 1


def _fake_module(events, *, post_stop_states=None):
    holder = SimpleNamespace(robot=None, connections=[])

    def robot_constructor(ip, realtime):
        events.append(("robot", ip, realtime))
        holder.connections.append((ip, realtime))
        holder.robot = _Robot(events, post_stop_states=post_stop_states)
        return holder.robot

    module = SimpleNamespace(
        __version__="0.21.2",
        Robot=robot_constructor,
        JointPositions=_JointPositions,
        RealtimeConfig=SimpleNamespace(kEnforce="rt-enforced"),
        ControllerMode=SimpleNamespace(JointImpedance="joint-impedance"),
    )
    return module, holder


def _factory(events, *, post_stop_states=None):
    module, holder = _fake_module(events, post_stop_states=post_stop_states)
    loads = []

    def loader(name):
        loads.append(name)
        events.append(("load", name))
        return module

    return PylibfrankaBackendFactory("172.16.0.2", module_loader=loader), holder, loads


def test_module_import_does_not_import_pylibfranka_or_construct_hardware():
    code = (
        "import sys; "
        "assert 'pylibfranka' not in sys.modules; "
        "import robot_control.franka.backend; "
        "assert 'pylibfranka' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_factory_is_lazy_rt_enforced_and_single_use():
    events = []
    factory, holder, loads = _factory(events)
    assert factory.creation_attempted is False
    assert events == []
    assert loads == []

    backend = factory()
    assert isinstance(backend, PylibfrankaPersistentBackend)
    assert factory.creation_attempted is True
    assert loads == ["pylibfranka"]
    assert holder.connections == [("172.16.0.2", "rt-enforced")]
    with pytest.raises(PylibfrankaBackendError, match="single-use"):
        factory()


def test_factory_rejects_unreviewed_pylibfranka_before_robot_construction():
    events = []
    module, holder = _fake_module(events)
    module.__version__ = "99.0"
    factory = PylibfrankaBackendFactory(
        "172.16.0.2", module_loader=lambda _name: module
    )
    with pytest.raises(PylibfrankaBackendError, match="version differs"):
        factory()
    assert holder.connections == []
    assert factory.creation_attempted is True


def test_backend_constructor_cannot_bypass_delayed_factory():
    with pytest.raises(TypeError, match="delayed factory"):
        PylibfrankaPersistentBackend(
            _seal=object(),
            robot=object(),
            pylibfranka=object(),
            owner_ident=threading.get_ident(),
        )


def test_persistent_handle_preserves_one_read_one_write_and_sample_hold():
    events = []
    first_stop_state = _RobotState(1.0)
    second_stop_state = _RobotState(1.001)
    factory, holder, _loads = _factory(
        events, post_stop_states=[first_stop_state, second_stop_state]
    )
    backend = factory()
    control = backend.start_joint_position_session()

    state1, dt1 = control.read_once()
    assert dt1 == 0.001
    with pytest.raises(PylibfrankaBackendError, match="second read"):
        control.read_once()
    control.write_once([0.0] * 7, sequence=1)

    state2, dt2 = control.read_once()
    assert state2 is not state1
    assert dt2 == 0.001
    # The same 60 Hz target sequence is valid on successive 1 kHz cycles.
    control.write_once([0.001] * 7, sequence=1)
    control.read_once()
    control.write_once([0.002] * 7, sequence=2)
    control.finish([0.002] * 7)

    assert holder.robot.control.commands == [
        ((0.0,) * 7, False),
        ((0.001,) * 7, False),
        ((0.002,) * 7, False),
        ((0.002,) * 7, True),
    ]
    backend.request_stop()
    assert backend.read_post_stop_state() is first_stop_state
    assert backend.read_post_stop_state() is second_stop_state
    backend.close()

    assert events[-4:] == [
        "active_finish",
        "robot_stop",
        "post_stop_read",
        "post_stop_read",
    ]


def test_measured_q_bootstrap_is_one_read_one_write_and_does_not_consume_sequence():
    events = []
    factory, holder, _loads = _factory(events)
    backend = factory()
    control = backend.start_joint_position_session()

    control.read_once()
    control.write_bootstrap_hold([0.125] * 7)
    with pytest.raises(PylibfrankaBackendError, match="no preceding"):
        control.write_bootstrap_hold([0.125] * 7)

    # The measured hold may be repeated at the FCI cadence while observation
    # bootstrap and first inference complete; neither write owns sequence 1.
    control.read_once()
    control.write_bootstrap_hold([0.125] * 7)

    # Bootstrap did not consume sequence one.  The first policy write uses the
    # ordinary non-terminal JointPositions semantics and starts at sequence 1.
    control.read_once()
    control.write_once([0.126] * 7, sequence=1)

    control.read_once()
    with pytest.raises(PylibfrankaBackendError, match="forbidden after policy"):
        control.write_bootstrap_hold([0.126] * 7)

    assert holder.robot.control.commands == [
        ((0.125,) * 7, False),
        ((0.125,) * 7, False),
        ((0.126,) * 7, False),
    ]
    backend.request_stop()
    backend.close()


def test_write_requires_read_and_sequence_never_regresses_or_skips():
    events = []
    factory, _holder, _loads = _factory(events)
    backend = factory()
    control = backend.start_joint_position_session()
    with pytest.raises(PylibfrankaBackendError, match="no preceding"):
        control.write_once([0.0] * 7, sequence=1)
    control.read_once()
    control.write_once([0.0] * 7, sequence=2)
    control.read_once()
    with pytest.raises(PylibfrankaBackendError, match="skipped"):
        control.write_once([0.0] * 7, sequence=4)
    # Finish is permitted as the response to the still-pending read.
    control.finish([0.0] * 7)
    backend.request_stop()
    backend.read_post_stop_state()
    backend.close()


def test_stop_exception_remains_unconfirmed_but_does_not_block_fresh_reads():
    events = []
    stop_state = _RobotState(1.0)
    factory, holder, _loads = _factory(events, post_stop_states=[stop_state])
    backend = factory()
    backend.start_joint_position_session()
    holder.robot.stop_failure = RuntimeError("synthetic stop failure")
    with pytest.raises(PylibfrankaBackendError) as caught:
        backend.request_stop()
    assert caught.value.operation == "robot_stop"
    assert isinstance(caught.value.__cause__, RuntimeError)
    # Session cleanup can still collect evidence; it must not infer rest from
    # either a successful or a failed stop return.
    assert backend.read_post_stop_state() is stop_state
    backend.close()


def test_control_exception_is_wrapped_once_and_never_recovers_or_retries():
    events = []
    factory, holder, _loads = _factory(events)
    backend = factory()
    control = backend.start_joint_position_session()
    holder.robot.control.fail_write = True
    control.read_once()
    with pytest.raises(PylibfrankaBackendError) as caught:
        control.write_once([0.0] * 7, sequence=1)
    assert caught.value.operation == "active_write"
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert holder.robot.recovery_calls == 0
    assert events.count("active_write") == 1
    holder.robot.control.fail_write = False
    control.finish([0.0] * 7)
    backend.request_stop()
    backend.read_post_stop_state()
    backend.close()


def test_owner_thread_is_enforced_without_touching_the_handle():
    events = []
    factory, holder, _loads = _factory(events)
    backend = factory()
    control = backend.start_joint_position_session()
    failures = []

    def foreign_read():
        try:
            control.read_once()
        except BaseException as exc:  # captured for assertion on owner thread
            failures.append(exc)

    thread = threading.Thread(target=foreign_read)
    thread.start()
    thread.join()
    assert len(failures) == 1
    assert isinstance(failures[0], PylibfrankaBackendError)
    assert failures[0].operation == "thread_ownership"
    assert holder.robot.control.read_count == 0
    backend.request_stop()
    backend.read_post_stop_state()
    backend.close()


def test_cyclic_gc_is_disabled_only_while_active_handle_is_owned():
    initially_enabled = gc.isenabled()
    if not initially_enabled:
        gc.enable()
    try:
        events = []
        factory, _holder, _loads = _factory(events)
        backend = factory()
        assert gc.isenabled()
        backend.start_joint_position_session()
        assert not gc.isenabled()
        backend.request_stop()
        backend.read_post_stop_state()
        backend.close()
        assert gc.isenabled()
    finally:
        if not initially_enabled:
            gc.disable()


def test_reviewed_state_mapping_covers_every_persistent_session_field():
    assert set(PYLIBFRANKA_STATE_FIELDS.values()) == {
        "q",
        "dq",
        "O_T_EE",
        "F_T_EE",
        "m_ee",
        "F_x_Cee",
        "I_ee",
        "m_load",
        "F_x_Cload",
        "I_load",
        "m_total",
        "robot_mode",
        "current_errors",
        "joint_contact",
        "joint_collision",
        "cartesian_contact",
        "cartesian_collision",
        "control_command_success_rate",
        "time",
    }


def test_repeated_robot_timestamp_is_rejected_as_nonfresh_state():
    events = []
    factory, holder, _loads = _factory(events)
    repeated = _RobotState(0.001)
    backend = factory()
    holder.robot.control.states = [repeated, repeated]
    control = backend.start_joint_position_session()
    control.read_once()
    control.write_once([0.0] * 7, sequence=1)
    with pytest.raises(PylibfrankaBackendError, match="did not increase"):
        control.read_once()
    control.finish([0.0] * 7)
    backend.request_stop()
    backend.read_post_stop_state()
    backend.close()
