from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest


MODULE_NAME = "anydex_pipeline.franka_sequence_driver"


def _franka_values(transform):
    return np.asarray(transform, dtype=np.float64).reshape(16, order="F").tolist()


def _pose(xyz=(0.50, 0.0, 0.30), yaw=0.0):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transform[:3, 3] = xyz
    return transform


class _Mode:
    def __init__(self, name):
        self.name = name


class _Period:
    def __init__(self, seconds):
        self.seconds = seconds

    def to_sec(self):
        return self.seconds


class _Command:
    def __init__(self, values):
        self.values = list(values)
        self.motion_finished = False


class _FakePylibfranka:
    class ControllerMode:
        JointImpedance = "joint"
        CartesianImpedance = "cartesian"

    class RealtimeConfig:
        kEnforce = "enforce"
        kIgnore = "ignore"

    JointPositions = _Command
    CartesianPose = _Command


class _State:
    def __init__(self, *, F_T_EE=None, O_T_EE=None, q=None):
        self.robot_mode = _Mode("Idle")
        self.current_errors = SimpleNamespace()
        self.cartesian_contact = np.zeros(6)
        self.cartesian_collision = np.zeros(6)
        self.joint_contact = np.zeros(7)
        self.joint_collision = np.zeros(7)
        self.control_command_success_rate = 1.0
        self.q = np.asarray(
            [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0] if q is None else q,
            dtype=np.float64,
        )
        self.q_d = self.q.copy()
        self.dq = np.zeros(7)
        self.O_T_EE = _franka_values(_pose() if O_T_EE is None else O_T_EE)
        self.O_T_EE_c = list(self.O_T_EE)
        self.F_T_EE = _franka_values(np.eye(4) if F_T_EE is None else F_T_EE)
        self.m_ee = 0.75
        self.m_load = 0.0
        self.m_total = 0.75
        self.F_x_Cload = np.zeros(3)
        self.I_load = np.zeros(9)
        self.F_x_Cee = np.asarray([0.0, 0.0, 0.08])
        self.I_ee = np.diag([0.01, 0.01, 0.002]).reshape(9, order="F")


class _Control:
    def __init__(self, robot, kind):
        self.robot = robot
        self.kind = kind
        self.commands = []
        self.period_index = 0
        self.startup_index = 0
        self._post_startup_success_rate = float(
            robot.state.control_command_success_rate
        )

    def readOnce(self):
        if self.robot.auto_qualify_startup and self.startup_index < 101:
            self.startup_index += 1
            self.robot.state.control_command_success_rate = 1.0
            self.robot.state.robot_mode = _Mode("Move")
            return self.robot.state, _Period(0.00005)
        periods = self.robot.periods
        index = self.period_index
        period = periods[min(index, len(periods) - 1)]
        self.period_index += 1
        if self.robot.control_success_rates is not None:
            rates = self.robot.control_success_rates
            self.robot.state.control_command_success_rate = rates[
                min(index, len(rates) - 1)
            ]
        else:
            self.robot.state.control_command_success_rate = (
                self._post_startup_success_rate
            )
        self.robot.state.robot_mode = _Mode("Move")
        return self.robot.state, _Period(period)

    def writeOnce(self, command):
        self.commands.append(command)
        self.robot.all_commands.append((self.kind, command))
        if self.robot.apply_commands:
            if self.kind == "joint":
                self.robot.state.q = np.asarray(command.values, dtype=np.float64)
                self.robot.state.q_d = self.robot.state.q.copy()
            else:
                self.robot.state.O_T_EE = list(command.values)
        if self.kind == "cartesian":
            self.robot.state.O_T_EE_c = list(command.values)
        if command.motion_finished:
            self.robot.state.robot_mode = _Mode("Idle")


class _Robot:
    def __init__(
        self,
        state=None,
        *,
        periods=None,
        apply_commands=True,
        control_success_rates=None,
        auto_qualify_startup=True,
    ):
        self.state = _State() if state is None else state
        self.periods = [0.01] if periods is None else list(periods)
        self.apply_commands = apply_commands
        self.control_success_rates = (
            None
            if control_success_rates is None
            else list(control_success_rates)
        )
        self.auto_qualify_startup = bool(auto_qualify_startup)
        self.controls = []
        self.all_commands = []
        self.stop_count = 0
        self.set_load_calls = []
        self.fail_set_load = False

    def read_once(self):
        return self.state

    def start_joint_position_control(self, mode):
        assert mode == _FakePylibfranka.ControllerMode.JointImpedance
        control = _Control(self, "joint")
        self.controls.append(control)
        return control

    def start_cartesian_pose_control(self, mode):
        assert mode == _FakePylibfranka.ControllerMode.CartesianImpedance
        control = _Control(self, "cartesian")
        self.controls.append(control)
        return control

    def stop(self):
        self.stop_count += 1
        self.state.robot_mode = _Mode("Idle")

    def set_load(self, mass, center, inertia):
        values = (
            float(mass),
            tuple(float(value) for value in center),
            tuple(float(value) for value in inertia),
        )
        self.set_load_calls.append(values)
        if self.fail_set_load:
            raise RuntimeError("set_load failed")
        self.state.m_load = values[0]
        self.state.F_x_Cload = np.asarray(values[1], dtype=np.float64)
        self.state.I_load = np.asarray(values[2], dtype=np.float64)
        self.state.m_total = self.state.m_ee + values[0]


class _FakeClock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


@pytest.fixture
def driver_module():
    return importlib.import_module(MODULE_NAME)


def _limits(module, **overrides):
    values = dict(
        expected_F_T_EE=np.eye(4),
        workspace_min_m=np.asarray([0.30, -0.30, 0.10]),
        workspace_max_m=np.asarray([0.80, 0.30, 0.70]),
        max_joint_speed_rad_s=10.0,
        max_joint_segment_rad=0.20,
        min_joint_duration_s=0.02,
        max_cartesian_speed_m_s=10.0,
        max_angular_speed_rad_s=10.0,
        max_segment_translation_m=0.020,
        max_segment_rotation_rad=0.080,
        min_cartesian_duration_s=0.02,
        settle_time_s=0.03,
        settle_timeout_s=0.06,
        settle_poll_s=0.01,
        stop_verify_timeout_s=0.01,
        stop_verify_poll_s=0.001,
        stop_verify_max_dq_rad_s=0.02,
        stop_verify_consecutive_samples=3,
        control_success_warmup_s=0.0,
        control_success_evaluation_window_s=0.01,
    )
    values.update(overrides)
    return module.FrankaMotionLimits(**values)


def test_end_effector_dynamics_must_be_supplied_as_one_complete_set(driver_module):
    with pytest.raises(ValueError, match="provided together"):
        driver_module.FrankaMotionLimits(
            expected_F_T_EE=np.eye(4), expected_m_ee_kg=0.75
        )


def test_bare_flange_zero_end_effector_dynamics_are_valid(driver_module):
    limits = driver_module.FrankaMotionLimits(
        expected_F_T_EE=np.eye(4),
        expected_m_ee_kg=0.0,
        expected_F_x_Cee_m=[0.0, 0.0, 0.0],
        expected_I_ee_kg_m2=np.zeros((3, 3)),
    )
    assert limits.expected_m_ee_kg == 0.0


def test_import_has_no_pylibfranka_import_or_connection(monkeypatch):
    original_import = importlib.import_module
    calls = []

    def guarded_import(name, package=None):
        calls.append(name)
        if name == "pylibfranka":
            raise AssertionError("module import attempted to load pylibfranka")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded_import)
    sys.modules.pop(MODULE_NAME, None)
    module = original_import(MODULE_NAME)

    assert module.FrankaSequenceDriver
    assert "pylibfranka" not in calls


@pytest.mark.parametrize("failure", ["tool", "contact"])
def test_bad_tool_transform_or_contact_stops(driver_module, failure):
    module = driver_module
    state = _State()
    if failure == "tool":
        wrong = np.eye(4)
        wrong[2, 3] = 0.01
        state.F_T_EE = _franka_values(wrong)
    else:
        state.joint_contact[2] = 1.0
    robot = _Robot(state)
    driver = module.FrankaSequenceDriver(
        robot, _FakePylibfranka, _limits(module)
    )

    with pytest.raises(RuntimeError):
        driver.move_joints([0.1, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])

    assert robot.stop_count == 1
    assert not robot.controls


def test_audited_continuous_joint_tracking_bound_is_checked_each_cycle(
    driver_module,
):
    module = driver_module
    robot = _Robot(apply_commands=False, periods=[0.01])
    limits = _limits(
        module,
        min_joint_duration_s=0.04,
        joint_arrival_tolerance_rad=0.002,
        max_continuous_joint_tracking_error_rad=0.002,
    )
    driver = module.FrankaSequenceDriver(robot, _FakePylibfranka, limits)

    with pytest.raises(RuntimeError, match="continuous joint tracking error"):
        driver.move_joints([0.1, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])

    assert robot.stop_count == 1
    assert robot.controls


def test_joint_hot_loop_binds_static_provenance_once_per_control_handle(
    driver_module,
):
    module = driver_module

    class CountingState(_State):
        _STATIC_FIELDS = {
            "O_T_EE",
            "F_T_EE",
            "m_ee",
            "F_x_Cee",
            "I_ee",
            "m_load",
            "m_total",
        }

        def __init__(self):
            object.__setattr__(self, "reads", {})
            super().__init__()
            self.reads.clear()

        def __getattribute__(self, name):
            if name in object.__getattribute__(self, "_STATIC_FIELDS"):
                reads = object.__getattribute__(self, "reads")
                reads[name] = reads.get(name, 0) + 1
            return super().__getattribute__(name)

    state = CountingState()
    robot = _Robot(state, periods=[0.01])
    limits = _limits(
        module,
        expected_m_ee_kg=0.75,
        expected_F_x_Cee_m=[0.0, 0.0, 0.08],
        expected_I_ee_kg_m2=np.diag([0.01, 0.01, 0.002]),
    )
    driver = module.FrankaSequenceDriver(robot, _FakePylibfranka, limits)
    start = state.q.copy()
    target = start.copy()
    target[0] += 0.02

    driver._run_joint_segment(start, target, 0.05)

    # The 101 exact-start qualification holds plus five 10 ms trajectory
    # samples still validate every dynamic gate; static provenance remains a
    # one-time control-handle boundary check.
    assert len(robot.controls[0].commands) == 107
    assert state.reads == {
        "O_T_EE": 1,
        "F_T_EE": 1,
        "m_ee": 1,
        "F_x_Cee": 1,
        "I_ee": 1,
        "m_load": 1,
        "m_total": 1,
    }


def test_static_provenance_is_rechecked_after_each_completed_segment(
    driver_module,
):
    module = driver_module

    class MutatingControl(_Control):
        def writeOnce(self, command):
            super().writeOnce(command)
            if len(self.commands) == 1:
                changed = np.eye(4)
                changed[2, 3] = 0.01
                self.robot.state.F_T_EE = _franka_values(changed)

    class MutatingRobot(_Robot):
        def start_joint_position_control(self, mode):
            assert mode == _FakePylibfranka.ControllerMode.JointImpedance
            control = MutatingControl(self, "joint")
            self.controls.append(control)
            return control

    robot = MutatingRobot(periods=[0.01])
    driver = module.FrankaSequenceDriver(
        robot, _FakePylibfranka, _limits(module)
    )

    with pytest.raises(RuntimeError, match="commissioned transform"):
        driver.move_joints([0.10, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])

    assert robot.stop_count == 1
    assert len(robot.controls) == 1


def test_control_loop_timing_telemetry_counts_completed_samples_and_overruns(
    driver_module,
):
    module = driver_module
    robot = _Robot(periods=[0.01])
    driver = module.FrankaSequenceDriver(
        robot, _FakePylibfranka, _limits(module)
    )
    trajectory_durations = [100_000, 600_000, 499_999, 500_001, 50_000]
    durations = [50_000] * 101 + trajectory_durations
    ticks = []
    cursor = 10_000_000
    for duration in durations:
        ticks.extend((cursor, cursor + duration))
        cursor += 1_000_000
    tick_iterator = iter(ticks)
    driver._control_clock_ns = tick_iterator.__next__
    gc_initially_enabled = module.gc.isenabled()
    start = robot.state.q.copy()
    target = start.copy()
    target[0] += 0.02

    driver._run_joint_segment(start, target, 0.05)

    timing = driver.last_control_loop_telemetry
    assert timing is not None
    assert timing.kind == "joint"
    assert timing.samples == 106
    assert timing.max_read_to_write_ns == 600_000
    assert timing.max_read_to_write_us == pytest.approx(600.0)
    assert timing.read_to_write_overruns == 2
    assert module.gc.isenabled() is gc_initially_enabled


def test_native_telemetry_tap_replaces_active_read_and_reuses_validated_preflight(
    driver_module,
):
    module = driver_module
    robot = _Robot(periods=[0.01])
    driver = module.FrankaSequenceDriver(
        robot, _FakePylibfranka, _limits(module)
    )

    class Tap:
        def __init__(self):
            self.synchronized = []
            self.active_reads = 0

        def synchronize_and_publish_arm(self, state, unix_ns, monotonic_ns):
            self.synchronized.append((state, unix_ns, monotonic_ns))
            return len(self.synchronized)

        def read_once_tapped(self, control):
            self.active_reads += 1
            return control.readOnce()

    tap = Tap()
    driver.install_native_telemetry_tap(tap)
    start = robot.state.q.copy()
    target = start.copy()
    target[0] += 0.02

    driver._run_joint_segment(start, target, 0.05)

    assert tap.active_reads == 106
    assert len(tap.synchronized) == 1
    assert tap.synchronized[0][0] is robot.state
    assert tap.synchronized[0][1] > 0
    assert tap.synchronized[0][2] > 0
    assert len(robot.controls) == 1
    assert robot.controls[0].startup_index == 101
    assert robot.controls[0].period_index == 5
    driver.remove_native_telemetry_tap(tap)


def test_native_telemetry_tap_installation_is_identity_bound_and_idle_only(
    driver_module,
):
    module = driver_module
    driver = module.FrankaSequenceDriver(
        _Robot(), _FakePylibfranka, _limits(module)
    )

    class Tap:
        def synchronize_and_publish_arm(self, *_args):
            return 1

        def read_once_tapped(self, control):
            return control.readOnce()

    first = Tap()
    second = Tap()
    driver.install_native_telemetry_tap(first)
    with pytest.raises(RuntimeError, match="already installed"):
        driver.install_native_telemetry_tap(second)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        driver.remove_native_telemetry_tap(second)
    driver._control_handle_active = True
    with pytest.raises(RuntimeError, match="forbidden during a control handle"):
        driver.remove_native_telemetry_tap(first)
    driver._control_handle_active = False
    driver.remove_native_telemetry_tap(first)


def test_native_telemetry_publish_failure_prevents_control_handle_creation(
    driver_module,
):
    module = driver_module
    robot = _Robot(periods=[0.01])
    driver = module.FrankaSequenceDriver(
        robot, _FakePylibfranka, _limits(module)
    )

    class FailingTap:
        def synchronize_and_publish_arm(self, *_args):
            raise RuntimeError("telemetry publish failed")

        def read_once_tapped(self, _control):
            raise AssertionError("control handle must not be read")

    driver.install_native_telemetry_tap(FailingTap())
    start = robot.state.q.copy()
    target = start.copy()
    target[0] += 0.02

    with pytest.raises(RuntimeError, match="telemetry publish failed"):
        driver._run_joint_segment(start, target, 0.05)

    assert robot.controls == []


def test_native_telemetry_reuses_all_settle_reads_without_changing_read_count(
    driver_module,
):
    module = driver_module

    class CountingRobot(_Robot):
        def __init__(self):
            super().__init__()
            self.read_count = 0

        def read_once(self):
            self.read_count += 1
            return self.state

    class Tap:
        def __init__(self, stage):
            self.stage = stage
            self.samples = []

        def synchronize_and_publish_arm(self, state, unix_ns, monotonic_ns):
            self.samples.append((self.stage, state, unix_ns, monotonic_ns))
            return len(self.samples)

        def read_once_tapped(self, _control):
            raise AssertionError("settle verification has no active control handle")

    tolerances = SimpleNamespace(
        position_m=0.005,
        orientation_rad=0.05,
        stable_seconds=0.03,
    )
    baseline_clock = _FakeClock()
    baseline_robot = CountingRobot()
    baseline = module.FrankaSequenceDriver(
        baseline_robot,
        _FakePylibfranka,
        _limits(module),
        monotonic=baseline_clock.monotonic,
        sleep=baseline_clock.sleep,
    )
    assert baseline.verify_settled(_pose(), tolerances)

    tapped_clock = _FakeClock()
    tapped_robot = CountingRobot()
    tapped = module.FrankaSequenceDriver(
        tapped_robot,
        _FakePylibfranka,
        _limits(module),
        monotonic=tapped_clock.monotonic,
        sleep=tapped_clock.sleep,
    )
    tap = Tap("eef_settling")
    tapped.install_native_telemetry_tap(tap)
    assert tapped.verify_settled(_pose(), tolerances)

    assert tapped_robot.read_count == baseline_robot.read_count
    assert len(tap.samples) == tapped_robot.read_count
    assert len(tap.samples) >= 4
    assert {sample[0] for sample in tap.samples} == {"eef_settling"}
    assert all(sample[1] is tapped_robot.state for sample in tap.samples)
    assert all(sample[2] > 0 and sample[3] > 0 for sample in tap.samples)


def test_native_telemetry_reuses_validated_segment_boundaries_without_extra_reads(
    driver_module,
):
    module = driver_module

    class CountingRobot(_Robot):
        def __init__(self):
            super().__init__(periods=[0.01])
            self.read_count = 0

        def read_once(self):
            self.read_count += 1
            return self.state

    class Tap:
        def __init__(self):
            self.boundary_samples = []
            self.active_reads = 0

        def synchronize_and_publish_arm(self, state, unix_ns, monotonic_ns):
            self.boundary_samples.append((state, unix_ns, monotonic_ns))
            return len(self.boundary_samples)

        def read_once_tapped(self, control):
            self.active_reads += 1
            return control.readOnce()

    target = np.asarray([0.05, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])
    baseline_robot = CountingRobot()
    baseline = module.FrankaSequenceDriver(
        baseline_robot, _FakePylibfranka, _limits(module)
    )
    baseline.move_joints(target)

    tapped_robot = CountingRobot()
    tapped = module.FrankaSequenceDriver(
        tapped_robot, _FakePylibfranka, _limits(module)
    )
    tap = Tap()
    tapped.install_native_telemetry_tap(tap)
    tapped.move_joints(target)

    assert tapped_robot.read_count == baseline_robot.read_count == 4
    # Start boundary, validated preflight, verified segment final, and final
    # arrival boundary all reuse existing samples.
    assert len(tap.boundary_samples) == 4
    assert tap.active_reads == (
        baseline_robot.controls[0].startup_index
        + baseline_robot.controls[0].period_index
    )
    assert tap.active_reads == (
        tapped_robot.controls[0].startup_index
        + tapped_robot.controls[0].period_index
    )


def test_native_telemetry_publishes_only_validated_post_stop_samples_without_extra_reads(
    driver_module,
):
    module = driver_module

    def stopped_state():
        return _State()

    unsafe = _State()
    unsafe.robot_mode = _Mode("Move")
    unsafe.dq = np.full(7, 0.05)

    class ScriptedRobot(_Robot):
        def __init__(self, samples):
            super().__init__()
            self.samples = iter(samples)
            self.read_count = 0

        def stop(self):
            self.stop_count += 1

        def read_once(self):
            self.read_count += 1
            return next(self.samples)

    class Tap:
        def __init__(self):
            self.samples = []

        def synchronize_and_publish_arm(self, state, *_timestamps):
            self.samples.append(state)
            return len(self.samples)

        def read_once_tapped(self, _control):
            raise AssertionError("post-stop verification has no active handle")

    def sample_script():
        return [
            stopped_state(),
            unsafe,
            stopped_state(),
            stopped_state(),
            stopped_state(),
        ]

    baseline_clock = _FakeClock()
    baseline_robot = ScriptedRobot(sample_script())
    baseline = module.FrankaSequenceDriver(
        baseline_robot,
        _FakePylibfranka,
        _limits(module),
        monotonic=baseline_clock.monotonic,
        sleep=baseline_clock.sleep,
    )
    baseline.stop()

    tapped_clock = _FakeClock()
    tapped_robot = ScriptedRobot(sample_script())
    tapped = module.FrankaSequenceDriver(
        tapped_robot,
        _FakePylibfranka,
        _limits(module),
        monotonic=tapped_clock.monotonic,
        sleep=tapped_clock.sleep,
    )
    tap = Tap()
    tapped.install_native_telemetry_tap(tap)
    tapped.stop()

    assert tapped_robot.read_count == baseline_robot.read_count == 5
    assert len(tap.samples) == 4
    assert unsafe not in tap.samples
    assert all(
        str(getattr(sample.robot_mode, "name", "")).lower() == "idle"
        for sample in tap.samples
    )


def test_primary_hot_loop_fault_survives_timing_publish_and_gc_is_restored(
    driver_module,
):
    module = driver_module
    robot = _Robot(periods=[0.01])
    robot.state.control_command_success_rate = 0.79
    driver = module.FrankaSequenceDriver(
        robot,
        _FakePylibfranka,
        _limits(module, control_success_warmup_s=0.0),
    )
    driver._control_clock_ns = lambda: 123
    gc_initially_enabled = module.gc.isenabled()
    start = robot.state.q.copy()
    target = start.copy()
    target[0] += 0.02

    with pytest.raises(RuntimeError, match="crossed hard floor"):
        driver._run_joint_segment(start, target, 0.05)

    timing = driver.last_control_loop_telemetry
    assert timing is not None
    assert timing.samples == 101
    assert timing.max_read_to_write_ns == 0
    assert timing.read_to_write_overruns == 0
    assert timing.success_qualified
    assert module.gc.isenabled() is gc_initially_enabled


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda state: setattr(state, "robot_mode", _Mode("Reflex")), "mode became unsafe"),
        (
            lambda state: setattr(
                state.current_errors, "joint_position_limits_violation", True
            ),
            "current_errors are active",
        ),
        (
            lambda state: state.cartesian_contact.__setitem__(1, 1.0),
            "cartesian_contact",
        ),
        (
            lambda state: state.joint_collision.__setitem__(5, 1.0),
            "joint_collision",
        ),
        (
            lambda state: state.joint_contact.__setitem__(0, -1.0),
            r"outside \[0, 1\]",
        ),
        (lambda state: state.q.__setitem__(2, np.nan), "state.q contains"),
        (lambda state: state.q.__setitem__(0, -2.86), "limit margin"),
        (
            lambda state: setattr(state, "control_command_success_rate", np.nan),
            "success rate is non-finite",
        ),
    ],
)
def test_realtime_validator_keeps_each_dynamic_safety_gate(
    driver_module, mutate, message
):
    module = driver_module
    state = _State()
    driver = module.FrankaSequenceDriver(
        _Robot(state), _FakePylibfranka, _limits(module)
    )
    validator, _ = driver._prepare_realtime_validation(
        state, require_idle=True
    )
    state.robot_mode = _Mode("Move")
    mutate(state)

    with pytest.raises(RuntimeError, match=message):
        validator.validate(state, require_idle=False)


def test_passive_idle_read_can_omit_motion_planning_margin_only(driver_module):
    module = driver_module
    state = _State(q=[0.0, 0.0, 0.0, -3.0435088, 0.0, 1.0, 0.0])
    historical_profile_limits = np.asarray(
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
    driver = module.FrankaSequenceDriver(
        _Robot(state),
        _FakePylibfranka,
        _limits(module, joint_limits_rad=historical_profile_limits),
    )

    with pytest.raises(RuntimeError, match="limit margin"):
        driver._validate_state(state, require_idle=True)

    assert driver._validate_state(
        state,
        require_idle=True,
        enforce_success=False,
        enforce_joint_limit_margin=False,
    ) == pytest.approx(1.0)

    state.current_errors.joint_position_limits_violation = True
    with pytest.raises(RuntimeError, match="current_errors are active"):
        driver._validate_state(
            state,
            require_idle=True,
            enforce_success=False,
            enforce_joint_limit_margin=False,
        )


def test_active_validation_cannot_omit_joint_margin(driver_module):
    module = driver_module
    state = _State()
    driver = module.FrankaSequenceDriver(
        _Robot(state), _FakePylibfranka, _limits(module)
    )

    with pytest.raises(RuntimeError, match="only for a passive Idle read"):
        driver._prepare_realtime_validation(
            state,
            require_idle=False,
            enforce_joint_limit_margin=False,
        )


def test_realtime_validator_uses_native_pylibfranka_error_bool_each_sample(
    driver_module,
):
    module = driver_module

    class Errors:
        __module__ = "pylibfranka._pylibfranka"

        def __init__(self):
            self.joint_reflex = False
            self.bool_calls = 0

        def __bool__(self):
            self.bool_calls += 1
            return self.joint_reflex

    state = _State()
    errors = Errors()
    state.current_errors = errors
    validator = module._RealtimeStateValidator(_limits(module), state)
    state.robot_mode = _Mode("Move")

    assert validator.validate(state, require_idle=False) == 1.0
    assert errors.bool_calls == 1
    errors.joint_reflex = True
    with pytest.raises(RuntimeError, match="joint_reflex"):
        validator.validate(state, require_idle=False)
    assert errors.bool_calls == 2


@pytest.mark.parametrize("failure", ["mass", "com", "inertia"])
def test_bad_commissioned_end_effector_dynamics_stops_before_control(
    driver_module, failure
):
    module = driver_module
    state = _State()
    if failure == "mass":
        state.m_ee += 0.1
    elif failure == "com":
        state.F_x_Cee = state.F_x_Cee + [0.0, 0.01, 0.0]
    else:
        state.I_ee = np.diag([0.02, 0.01, 0.002]).reshape(9, order="F")
    limits = _limits(
        module,
        expected_m_ee_kg=0.75,
        expected_F_x_Cee_m=[0.0, 0.0, 0.08],
        expected_I_ee_kg_m2=np.diag([0.01, 0.01, 0.002]),
    )
    robot = _Robot(state)
    driver = module.FrankaSequenceDriver(robot, _FakePylibfranka, limits)

    with pytest.raises(RuntimeError):
        driver.move_joints([0.1, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])

    assert robot.stop_count == 1
    assert not robot.controls


@pytest.mark.parametrize("failure", ["external_load", "total_mass"])
def test_unexpected_external_or_total_mass_stops_before_control(
    driver_module, failure
):
    module = driver_module
    state = _State()
    if failure == "external_load":
        state.m_load = 0.1
        state.m_total = 0.85
    else:
        state.m_total = 0.85
    limits = _limits(
        module,
        expected_m_ee_kg=0.75,
        expected_F_x_Cee_m=[0.0, 0.0, 0.08],
        expected_I_ee_kg_m2=np.diag([0.01, 0.01, 0.002]),
    )
    robot = _Robot(state)
    driver = module.FrankaSequenceDriver(robot, _FakePylibfranka, limits)

    with pytest.raises(RuntimeError):
        driver.move_joints([0.1, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])

    assert robot.stop_count == 1
    assert not robot.controls


def _loaded_payload():
    return SimpleNamespace(
        mass_kg=0.25,
        F_x_Cload_m=np.asarray([0.01, -0.02, 0.09]),
        I_load_kg_m2=np.asarray(
            [
                [0.0020, 0.0001, 0.0],
                [0.0001, 0.0030, 0.0],
                [0.0, 0.0, 0.0040],
            ]
        ),
        binding_sha256="a" * 64,
    )


def _loaded_driver(module, robot=None):
    active_robot = _Robot() if robot is None else robot
    limits = _limits(
        module,
        expected_m_ee_kg=0.75,
        expected_F_x_Cee_m=[0.0, 0.0, 0.08],
        expected_I_ee_kg_m2=np.diag([0.01, 0.01, 0.002]),
    )
    return module.FrankaSequenceDriver(
        active_robot, _FakePylibfranka, limits
    ), active_robot


def test_external_load_is_applied_column_major_verified_and_cleared(driver_module):
    driver, robot = _loaded_driver(driver_module)
    payload = _loaded_payload()

    driver.apply_external_load_and_verify(payload)

    assert driver.external_load_binding_sha256 == "a" * 64
    assert robot.set_load_calls == [
        (
            0.25,
            (0.01, -0.02, 0.09),
            tuple(payload.I_load_kg_m2.reshape(9, order="F")),
        )
    ]
    driver._validate_static_state(robot.state)

    driver.clear_external_load_and_verify()

    assert driver.external_load_binding_sha256 is None
    assert robot.set_load_calls[-1] == (0.0, (0.0, 0.0, 0.0), (0.0,) * 9)
    assert robot.state.m_load == 0.0
    assert robot.state.m_total == 0.75


def test_active_payload_provenance_is_rechecked_before_motion(driver_module):
    driver, robot = _loaded_driver(driver_module)
    driver.apply_external_load_and_verify(_loaded_payload())
    robot.state.F_x_Cload = np.asarray([0.02, -0.02, 0.09])

    with pytest.raises(RuntimeError, match="F_x_Cload"):
        driver.move_joints([0.1, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])

    assert robot.stop_count == 1
    assert not robot.controls


def test_loaded_idle_state_refresh_is_read_only_and_rechecks_payload(driver_module):
    driver, robot = _loaded_driver(driver_module)
    driver.apply_external_load_and_verify(_loaded_payload())
    calls_before = list(robot.set_load_calls)

    driver.verify_idle_state()

    assert robot.set_load_calls == calls_before
    assert not robot.controls
    robot.state.I_load = np.diag([0.003, 0.003, 0.003])
    with pytest.raises(RuntimeError, match="I_load"):
        driver.verify_idle_state()


def test_loaded_joint_motion_uses_bound_velocity_and_acceleration_time_law(
    driver_module,
):
    module = driver_module
    robot = _Robot()
    limits = _limits(
        module,
        expected_m_ee_kg=0.75,
        expected_F_x_Cee_m=[0.0, 0.0, 0.08],
        expected_I_ee_kg_m2=np.diag([0.01, 0.01, 0.002]),
        max_joint_speed_rad_s=0.20,
        max_joint_segment_rad=0.20,
        min_joint_duration_s=0.60,
    )
    driver = module.FrankaSequenceDriver(robot, _FakePylibfranka, limits)
    driver.apply_external_load_and_verify(_loaded_payload())
    segments = []

    def run_segment(start, target, duration):
        segments.append((start.copy(), target.copy(), float(duration)))
        robot.state.q = target.copy()

    driver._run_joint_segment = run_segment
    target = robot.state.q.copy()
    target[0] += 0.20

    driver.move_loaded_joints(
        target,
        max_joint_velocity_rad_s=0.20,
        max_joint_acceleration_rad_s2=0.05,
        max_dynamic_segment_rad=0.20,
        min_segment_duration_s=0.60,
    )

    assert len(segments) == 1
    expected_acceleration_duration = np.sqrt(
        module.COSINE_PEAK_ACCELERATION_FACTOR * 0.20 / 0.05
    )
    assert segments[0][2] == pytest.approx(expected_acceleration_duration)
    peak_velocity = module.COSINE_PEAK_VELOCITY_FACTOR * 0.20 / segments[0][2]
    peak_acceleration = (
        module.COSINE_PEAK_ACCELERATION_FACTOR * 0.20 / segments[0][2] ** 2
    )
    assert peak_velocity <= 0.20
    assert peak_acceleration == pytest.approx(0.05)


def test_loaded_joint_motion_requires_verified_load_and_driver_bounds(driver_module):
    driver, _robot = _loaded_driver(driver_module)
    target = np.asarray([0.1, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])

    with pytest.raises(RuntimeError, match="verified active external load"):
        driver.move_loaded_joints(
            target,
            max_joint_velocity_rad_s=driver.limits.max_joint_speed_rad_s,
            max_joint_acceleration_rad_s2=0.10,
            max_dynamic_segment_rad=driver.limits.max_joint_segment_rad,
            min_segment_duration_s=driver.limits.min_joint_duration_s,
        )

    driver.apply_external_load_and_verify(_loaded_payload())
    with pytest.raises(ValueError, match="velocity exceeds"):
        driver.move_loaded_joints(
            target,
            max_joint_velocity_rad_s=driver.limits.max_joint_speed_rad_s + 0.01,
            max_joint_acceleration_rad_s2=0.10,
            max_dynamic_segment_rad=driver.limits.max_joint_segment_rad,
            min_segment_duration_s=driver.limits.min_joint_duration_s,
        )


def test_failed_payload_verification_rolls_back_to_verified_zero(driver_module):
    class WrongLoadRobot(_Robot):
        def set_load(self, mass, center, inertia):
            super().set_load(mass, center, inertia)
            if float(mass) > 0.0:
                self.state.m_total += 0.1

    driver, robot = _loaded_driver(driver_module, WrongLoadRobot())

    with pytest.raises(RuntimeError, match="zero-load rollback was verified"):
        driver.apply_external_load_and_verify(_loaded_payload())

    assert len(robot.set_load_calls) == 2
    assert robot.set_load_calls[-1][0] == 0.0
    assert driver.external_load_binding_sha256 is None
    driver._validate_static_state(robot.state)


def test_payload_rollback_failure_latches_configuration_uncertain(driver_module):
    class UnrecoverableLoadRobot(_Robot):
        def set_load(self, mass, center, inertia):
            if float(mass) == 0.0 and self.set_load_calls:
                self.set_load_calls.append(
                    (
                        float(mass),
                        tuple(float(value) for value in center),
                        tuple(float(value) for value in inertia),
                    )
                )
                raise RuntimeError("zero rollback failed")
            super().set_load(mass, center, inertia)
            self.state.m_total += 0.1

    driver, robot = _loaded_driver(driver_module, UnrecoverableLoadRobot())

    with pytest.raises(RuntimeError, match="LOAD CONFIGURATION UNCONFIRMED"):
        driver.apply_external_load_and_verify(_loaded_payload())

    with pytest.raises(RuntimeError, match="configuration is uncertain"):
        driver.apply_external_load_and_verify(_loaded_payload())
    assert len(robot.set_load_calls) == 2


def test_payload_configuration_is_forbidden_during_control_handle(driver_module):
    driver, robot = _loaded_driver(driver_module)
    driver._control_handle_active = True

    with pytest.raises(RuntimeError, match="during a control handle"):
        driver.apply_external_load_and_verify(_loaded_payload())

    assert robot.set_load_calls == []


def test_repeated_zero_period_and_tracking_failure_stop(driver_module):
    module = driver_module
    q_target = [0.1, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]

    bad_dt_robot = _Robot(periods=[0.0, 0.0])
    bad_dt_driver = module.FrankaSequenceDriver(
        bad_dt_robot, _FakePylibfranka, _limits(module)
    )
    with pytest.raises(RuntimeError, match="repeated zero"):
        bad_dt_driver.move_joints(q_target)
    assert bad_dt_robot.stop_count == 1

    tracking_robot = _Robot(apply_commands=False)
    tracking_driver = module.FrankaSequenceDriver(
        tracking_robot, _FakePylibfranka, _limits(module)
    )
    with pytest.raises(RuntimeError, match="tracking error"):
        tracking_driver.move_joints(q_target)
    assert tracking_robot.stop_count == 1


def test_low_control_success_after_warmup_stops(driver_module):
    module = driver_module
    robot = _Robot()
    robot.state.control_command_success_rate = 0.2
    limits = _limits(
        module,
        min_joint_duration_s=0.15,
        min_cartesian_duration_s=0.06,
        control_success_warmup_s=0.05,
        stop_verify_timeout_s=0.05,
    )
    driver = module.FrankaSequenceDriver(robot, _FakePylibfranka, limits)

    with pytest.raises(RuntimeError, match="success rate"):
        driver.move_joints([0.1, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])

    assert robot.stop_count == 1


def test_joint_startup_qualification_holds_active_q_d_before_motion(
    driver_module,
):
    module = driver_module
    rates = [0.79] * 100 + [1.0] * 20
    robot = _Robot(
        periods=[0.001],
        control_success_rates=rates,
        auto_qualify_startup=False,
    )
    active_start = robot.state.q.copy()
    active_start[0] += 0.001
    robot.state.q_d = active_start.copy()
    limits = _limits(
        module,
        min_joint_duration_s=0.20,
        min_cartesian_duration_s=0.20,
        control_success_warmup_s=0.0,
        control_success_evaluation_window_s=0.20,
        joint_arrival_tolerance_rad=0.003,
    )
    driver = module.FrankaSequenceDriver(robot, _FakePylibfranka, limits)
    target = active_start.copy()
    target[0] += 0.005

    driver._run_joint_segment(robot.state.q.copy(), target, 0.005)

    commands = robot.controls[0].commands
    assert len(commands) == 107
    for command in commands[:101]:
        np.testing.assert_allclose(command.values, active_start, atol=0.0, rtol=0.0)
        assert not command.motion_finished
    assert commands[101].values[0] > active_start[0]
    timing = driver.last_control_loop_telemetry
    assert timing is not None
    assert timing.success_qualified
    assert timing.success_qualification_positive_writes == 101
    assert timing.success_qualification_control_time_s == pytest.approx(0.101)
    assert timing.success_qualification_rate == pytest.approx(1.0)


def test_startup_qualification_times_out_without_moving(driver_module):
    module = driver_module
    robot = _Robot(
        periods=[0.001],
        control_success_rates=[0.79] * 200,
        auto_qualify_startup=False,
    )
    limits = _limits(
        module,
        min_joint_duration_s=0.105,
        min_cartesian_duration_s=0.105,
        control_success_warmup_s=0.0,
        control_success_evaluation_window_s=0.105,
    )
    driver = module.FrankaSequenceDriver(robot, _FakePylibfranka, limits)
    start = robot.state.q.copy()
    target = start.copy()
    target[0] += 0.005

    with pytest.raises(RuntimeError, match="startup qualification timed out"):
        driver._run_joint_segment(start, target, 0.005)

    assert robot.controls[0].commands
    assert all(
        np.array_equal(command.values, start)
        for command in robot.controls[0].commands
    )
    timing = driver.last_control_loop_telemetry
    assert timing is not None
    assert not timing.success_qualified
    assert timing.success_qualification_positive_writes >= 100


def test_hard_floor_is_enforced_immediately_after_startup_qualification(
    driver_module,
):
    module = driver_module
    rates = [0.79] * 100 + [1.0, 0.79]
    robot = _Robot(
        periods=[0.001],
        control_success_rates=rates,
        auto_qualify_startup=False,
    )
    limits = _limits(
        module,
        min_joint_duration_s=0.20,
        min_cartesian_duration_s=0.20,
        control_success_warmup_s=0.0,
        control_success_evaluation_window_s=0.20,
    )
    driver = module.FrankaSequenceDriver(robot, _FakePylibfranka, limits)
    start = robot.state.q.copy()
    target = start.copy()
    target[0] += 0.005

    with pytest.raises(RuntimeError, match="crossed hard floor"):
        driver._run_joint_segment(start, target, 0.005)

    assert len(robot.controls[0].commands) == 101
    assert all(
        np.array_equal(command.values, start)
        for command in robot.controls[0].commands
    )
    timing = driver.last_control_loop_telemetry
    assert timing is not None and timing.success_qualified


def test_cartesian_startup_qualification_holds_active_commanded_pose(
    driver_module,
):
    module = driver_module
    state = _State()
    active_start = _pose([0.502, 0.0, 0.30])
    state.O_T_EE_c = _franka_values(active_start)
    robot = _Robot(state, periods=[0.001], auto_qualify_startup=False)
    limits = _limits(
        module,
        min_joint_duration_s=0.20,
        min_cartesian_duration_s=0.20,
        control_success_warmup_s=0.0,
        control_success_evaluation_window_s=0.20,
        translation_arrival_tolerance_m=0.003,
    )
    driver = module.FrankaSequenceDriver(robot, _FakePylibfranka, limits)

    driver._run_pose_segment(_pose(), _pose([0.507, 0.0, 0.30]), 0.005)

    commands = robot.controls[0].commands
    active_values = np.asarray(_franka_values(active_start))
    for command in commands[:101]:
        np.testing.assert_allclose(
            command.values, active_values, atol=0.0, rtol=0.0
        )
    assert commands[101].values[12] > active_values[12]


def test_joint_startup_q_d_mismatch_fails_before_write(driver_module):
    module = driver_module
    robot = _Robot(periods=[0.001])
    robot.state.q_d[0] += 0.01
    limits = _limits(
        module,
        min_joint_duration_s=0.20,
        min_cartesian_duration_s=0.20,
        control_success_warmup_s=0.0,
        control_success_evaluation_window_s=0.20,
        joint_arrival_tolerance_rad=0.003,
    )
    driver = module.FrankaSequenceDriver(robot, _FakePylibfranka, limits)
    start = robot.state.q.copy()
    target = start.copy()
    target[0] += 0.005

    with pytest.raises(RuntimeError, match="desired joint start versus planned"):
        driver._run_joint_segment(start, target, 0.005)

    assert robot.controls[0].commands == []


def test_post_qualification_samples_trip_hard_floor_without_second_warmup(
    driver_module,
):
    module = driver_module
    robot = _Robot(control_success_rates=[0.2] * 5 + [1.0] * 20)
    limits = _limits(
        module,
        min_joint_duration_s=0.10,
        min_cartesian_duration_s=0.10,
        control_success_warmup_s=0.05,
        control_success_evaluation_window_s=0.01,
    )
    driver = module.FrankaSequenceDriver(robot, _FakePylibfranka, limits)

    with pytest.raises(RuntimeError, match="crossed hard floor"):
        driver.move_joints([0.1, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])

    assert robot.stop_count == 1


def test_control_success_rate_accepts_sub_tolerance_roundoff(driver_module):
    module = driver_module
    robot = _Robot()
    threshold = 0.95
    robot.state.control_command_success_rate = threshold - 0.1e-6
    limits = _limits(
        module,
        min_joint_duration_s=0.70,
        min_cartesian_duration_s=0.06,
        control_success_warmup_s=0.05,
        min_control_success_rate=threshold,
    )
    driver = module.FrankaSequenceDriver(robot, _FakePylibfranka, limits)

    actual = driver.move_joints([0.1, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])

    np.testing.assert_allclose(actual, [0.1, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])
    assert robot.stop_count == 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "min_control_success_rate": 0.95,
                "control_success_hard_floor": 0.96,
            },
            "control_success_hard_floor",
        ),
        (
            {"control_success_evaluation_window_s": 0.0},
            "control_success_evaluation_window_s",
        ),
        (
            {"min_joint_duration_s": 0.005},
            "minimum Franka segment durations",
        ),
        (
            {"min_cartesian_duration_s": 0.005},
            "minimum Franka segment durations",
        ),
    ],
)
def test_control_success_watchdog_configuration_fails_closed(
    driver_module, overrides, message
):
    with pytest.raises(ValueError, match=message):
        _limits(driver_module, **overrides)


@pytest.mark.parametrize("invalid_rate", [-0.01, 1.01, np.nan])
def test_invalid_control_success_sample_stops_before_control(
    driver_module, invalid_rate
):
    module = driver_module
    robot = _Robot()
    robot.state.control_command_success_rate = invalid_rate
    driver = module.FrankaSequenceDriver(
        robot, _FakePylibfranka, _limits(module)
    )

    with pytest.raises(RuntimeError, match="control command success rate"):
        driver.move_joints([0.1, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])

    assert robot.stop_count == 1
    assert not robot.controls


def test_control_success_rate_rejects_sustained_shortfall_with_window_evidence(
    driver_module,
):
    module = driver_module
    robot = _Robot()
    robot.state.control_command_success_rate = 0.949
    limits = _limits(
        module,
        min_joint_duration_s=0.70,
        min_cartesian_duration_s=0.70,
        control_success_warmup_s=0.05,
        min_control_success_rate=0.95,
        control_success_evaluation_window_s=0.50,
    )
    driver = module.FrankaSequenceDriver(robot, _FakePylibfranka, limits)

    with pytest.raises(RuntimeError) as exc_info:
        driver.move_joints([0.1, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])

    assert str(exc_info.value) == (
        "Franka control command success rate window average is below threshold: "
        "average=0.949000000, minimum=0.949000000, latest=0.949000000, "
        "threshold=0.950000000, window=0.500000000s"
    )
    assert robot.stop_count == 1


def test_control_success_window_is_time_weighted(driver_module):
    module = driver_module
    limits = _limits(
        module,
        min_joint_duration_s=0.05,
        min_cartesian_duration_s=0.05,
        min_control_success_rate=0.955,
        control_success_evaluation_window_s=0.05,
    )
    watchdog = module._ControlSuccessWatchdog(limits)

    watchdog.observe(0.94, 0.02)
    watchdog.observe(0.94, 0.02)
    # A sample-count mean would be 0.96 and pass.  Weighting by the validated
    # control periods gives 0.952 and therefore correctly fails the 0.955
    # quality gate once (and only once) the complete 0.05 s window exists.
    with pytest.raises(RuntimeError, match="average=0.952000000"):
        watchdog.observe(1.0, 0.01)


def test_one_short_rolling_window_dip_does_not_stop_joint_motion(driver_module):
    module = driver_module
    # The source metric already covers the last 100 FCI commands (~0.1 s).
    # Simulate one such 0.94 dip, then recovery, inside the independent 0.5 s
    # time-weighted quality window.
    rates = [1.0] * 10 + [0.94] * 10 + [1.0] * 100
    robot = _Robot(control_success_rates=rates)
    limits = _limits(
        module,
        min_joint_duration_s=0.80,
        min_cartesian_duration_s=0.80,
        control_success_warmup_s=0.10,
        min_control_success_rate=0.95,
        control_success_evaluation_window_s=0.50,
    )
    driver = module.FrankaSequenceDriver(robot, _FakePylibfranka, limits)

    actual = driver.move_joints([0.1, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])

    np.testing.assert_allclose(actual, [0.1, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])
    assert robot.stop_count == 0


def test_control_success_hard_floor_stops_before_window_fills(driver_module):
    module = driver_module
    robot = _Robot()
    robot.state.control_command_success_rate = 0.79
    limits = _limits(
        module,
        min_joint_duration_s=0.60,
        min_cartesian_duration_s=0.60,
        control_success_warmup_s=0.05,
        min_control_success_rate=0.95,
        control_success_hard_floor=0.80,
        control_success_evaluation_window_s=0.50,
    )
    driver = module.FrankaSequenceDriver(robot, _FakePylibfranka, limits)

    with pytest.raises(RuntimeError, match="crossed hard floor"):
        driver.move_joints([0.1, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])

    assert robot.stop_count == 1


def test_sustained_low_control_success_stops_cartesian_loop(driver_module):
    module = driver_module
    robot = _Robot()
    robot.state.control_command_success_rate = 0.94
    limits = _limits(
        module,
        min_joint_duration_s=0.70,
        min_cartesian_duration_s=0.70,
        control_success_warmup_s=0.05,
        control_success_evaluation_window_s=0.50,
    )
    driver = module.FrankaSequenceDriver(robot, _FakePylibfranka, limits)

    with pytest.raises(RuntimeError, match="window average is below threshold"):
        driver.move_pose(_pose([0.51, 0.0, 0.30]))

    assert robot.stop_count == 1


def test_cartesian_motion_recomputes_bounded_translation_rotation_segments(
    driver_module,
):
    module = driver_module
    robot = _Robot()
    limits = _limits(module)
    driver = module.FrankaSequenceDriver(robot, _FakePylibfranka, limits)
    target = _pose([0.555, 0.0, 0.30], yaw=0.19)

    actual = driver.move_pose(target)

    np.testing.assert_allclose(actual, target, atol=1e-8)
    assert len(robot.controls) >= 3
    previous = _pose()
    for control in robot.controls:
        final = np.asarray(control.commands[-1].values).reshape((4, 4), order="F")
        translation, rotation = module.pose_error(previous, final)
        assert translation <= limits.max_segment_translation_m + 1e-10
        assert rotation <= limits.max_segment_rotation_rad + 1e-10
        assert control.commands[-1].motion_finished
        previous = final


def test_cartesian_hold_control_opens_handle_without_commanded_displacement(
    driver_module,
):
    module = driver_module
    robot = _Robot()
    limits = _limits(module)
    driver = module.FrankaSequenceDriver(robot, _FakePylibfranka, limits)
    reviewed = _pose()

    final_pose = driver.hold_pose_control(reviewed, 1.0)

    np.testing.assert_allclose(final_pose, reviewed, atol=1e-8)
    assert len(robot.controls) == 1
    commands = robot.controls[0].commands
    assert commands
    expected = _franka_values(reviewed)
    assert all(command.values == expected for command in commands)
    assert commands[-1].motion_finished is True
    timing = driver.last_control_loop_telemetry
    assert timing is not None
    assert timing.success_qualified is True
    assert timing.success_qualification_positive_writes == 101


def test_bounded_pose_step_backs_off_from_float32_rotation_boundary(driver_module):
    module = driver_module
    # Representative Franka O_T_EE values have float32-sized residual
    # orthonormality error while remaining valid robot-state rotations.
    anchor = np.asarray(
        [
            [0.9023956656455994, 0.14086401462554932, 0.4072338938713074, 0.0],
            [0.14178113639354706, -0.9894992113113403, 0.02809724770486355, 0.0],
            [0.40691548585891724, 0.03238324820995331, -0.9128916263580322, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    start = anchor.copy()
    start[:3, :3] = anchor[:3, :3] @ module.so3_exp(np.deg2rad([16.0, 0.0, 0.0]))
    start[:3, 3] = [0.605, -0.080, 0.683]
    target = anchor.copy()
    target[:3, :3] = anchor[:3, :3] @ module.so3_exp(np.deg2rad([7.5, 0.0, 0.0]))
    target[:3, 3] = [0.590, -0.055, 0.683]
    limit = np.deg2rad(5.0)
    limits = _limits(
        module,
        max_segment_translation_m=0.020,
        max_segment_rotation_rad=limit,
    )

    _, full_rotation = module.pose_error(start, target)
    nominal = module.interpolate_pose_linear_se3(
        start, target, limit / full_rotation
    )
    # This is the field failure: aiming exactly at 5 degrees re-measures just
    # outside the hard bound because the input rotation is not bit-perfect SO(3).
    assert module.pose_error(start, nominal)[1] > limit

    bounded = module.bounded_pose_step(start, target, limits)
    translation, rotation = module.pose_error(start, bounded)

    assert translation < limits.max_segment_translation_m
    assert rotation < limits.max_segment_rotation_rad


@pytest.mark.parametrize("invalid_hold", [-0.01, np.nan, np.inf])
def test_cartesian_endpoint_hold_must_be_finite_and_non_negative(
    driver_module, invalid_hold
):
    with pytest.raises(ValueError, match="cartesian_endpoint_hold_s"):
        _limits(driver_module, cartesian_endpoint_hold_s=invalid_hold)


def test_cartesian_endpoint_timeout_must_cover_stable_dwell(driver_module):
    with pytest.raises(ValueError, match="at least settle_time_s"):
        _limits(
            driver_module,
            cartesian_endpoint_hold_s=0.02,
            settle_time_s=0.03,
        )


@pytest.mark.parametrize(
    "invalid_mode", ["", "CartesianImpedance", "position", None, 1]
)
def test_cartesian_pose_controller_mode_rejects_unknown_values(
    driver_module, invalid_mode
):
    with pytest.raises(ValueError, match="cartesian_pose_controller_mode"):
        _limits(driver_module, cartesian_pose_controller_mode=invalid_mode)


def test_cartesian_pose_can_use_joint_impedance_controller_mode(driver_module):
    module = driver_module

    class JointImpedanceCartesianRobot(_Robot):
        def start_cartesian_pose_control(self, mode):
            assert mode == _FakePylibfranka.ControllerMode.JointImpedance
            control = _Control(self, "cartesian")
            self.controls.append(control)
            return control

    robot = JointImpedanceCartesianRobot()
    limits = _limits(
        module, cartesian_pose_controller_mode="joint_impedance"
    )
    driver = module.FrankaSequenceDriver(robot, _FakePylibfranka, limits)
    target = _pose([0.51, 0.0, 0.30])

    actual = driver.move_pose(target)

    np.testing.assert_allclose(actual, target, atol=1.0e-12, rtol=0.0)
    assert robot.controls
    assert robot.controls[0].commands[-1].motion_finished


def test_cartesian_interpolation_starts_from_active_commanded_pose(driver_module):
    module = driver_module
    state = _State()
    state.O_T_EE_c = _franka_values(_pose([0.502, 0.0, 0.30]))
    robot = _Robot(state, periods=[0.01])
    limits = _limits(
        module,
        min_cartesian_duration_s=0.02,
        translation_arrival_tolerance_m=0.003,
    )
    driver = module.FrankaSequenceDriver(robot, _FakePylibfranka, limits)

    driver.move_pose(_pose([0.51, 0.0, 0.30]))

    first = np.asarray(robot.controls[0].commands[101].values).reshape(
        (4, 4), order="F"
    )
    # At alpha=0.5 minimum jerk has blend=0.5: interpolate from the active
    # commanded 0.502 m, not the outer measured/planned 0.500 m.
    assert first[0, 3] == pytest.approx(0.506)


def test_cartesian_active_commanded_start_mismatch_fails_before_write(
    driver_module,
):
    module = driver_module
    state = _State()
    state.O_T_EE_c = _franka_values(_pose([0.52, 0.0, 0.30]))
    robot = _Robot(state, periods=[0.01])
    limits = _limits(module, translation_arrival_tolerance_m=0.003)
    driver = module.FrankaSequenceDriver(robot, _FakePylibfranka, limits)

    with pytest.raises(RuntimeError, match="commanded start versus planned"):
        driver.move_pose(_pose([0.51, 0.0, 0.30]))

    assert robot.stop_count == 1
    assert robot.controls
    assert robot.controls[0].commands == []


def test_cartesian_endpoint_hold_repeats_target_and_allows_lag_to_converge(
    driver_module,
):
    module = driver_module

    class DelayedCartesianControl(_Control):
        def __init__(self, robot, kind):
            super().__init__(robot, kind)
            self.pending_values = []

        def writeOnce(self, command):
            self.commands.append(command)
            self.robot.all_commands.append((self.kind, command))
            self.pending_values.append(list(command.values))
            self.robot.state.O_T_EE_c = list(command.values)
            # Model a two-cycle Cartesian impedance/measurement lag.  A
            # command affects O_T_EE only after two newer commands arrive.
            if len(self.pending_values) > 2:
                self.robot.state.O_T_EE = self.pending_values.pop(0)
            if command.motion_finished:
                self.robot.state.robot_mode = _Mode("Idle")

    class DelayedCartesianRobot(_Robot):
        def start_cartesian_pose_control(self, mode):
            assert mode == _FakePylibfranka.ControllerMode.CartesianImpedance
            control = DelayedCartesianControl(self, "cartesian")
            self.controls.append(control)
            return control

    target = _pose([0.51, 0.0, 0.30])
    common_limits = dict(
        min_cartesian_duration_s=0.02,
        translation_arrival_tolerance_m=0.001,
    )

    no_hold_robot = DelayedCartesianRobot(periods=[0.01])
    no_hold_driver = module.FrankaSequenceDriver(
        no_hold_robot,
        _FakePylibfranka,
        _limits(module, **common_limits),
    )
    with pytest.raises(RuntimeError, match="tracking error"):
        no_hold_driver.move_pose(target)
    assert no_hold_robot.stop_count == 1

    hold_robot = DelayedCartesianRobot(periods=[0.01])
    hold_driver = module.FrankaSequenceDriver(
        hold_robot,
        _FakePylibfranka,
        _limits(
            module,
            cartesian_endpoint_hold_s=0.08,
            settle_time_s=0.02,
            **common_limits,
        ),
    )

    actual = hold_driver.move_pose(target)

    np.testing.assert_allclose(actual, target, atol=1.0e-12, rtol=0.0)
    assert hold_robot.stop_count == 0
    commands = hold_robot.controls[0].commands
    target_values = np.asarray(_franka_values(target))
    endpoint_commands = [
        command
        for command in commands[:-1]
        if np.allclose(command.values, target_values, atol=1.0e-12, rtol=0.0)
    ]
    assert len(endpoint_commands) >= 3
    assert all(not command.motion_finished for command in commands[:-1])
    assert commands[-1].motion_finished


def test_cartesian_endpoint_stuck_times_out_without_motion_finished(driver_module):
    module = driver_module
    robot = _Robot(apply_commands=False, periods=[0.01])
    limits = _limits(
        module,
        min_cartesian_duration_s=0.02,
        cartesian_endpoint_hold_s=0.04,
        settle_time_s=0.02,
        translation_arrival_tolerance_m=0.001,
    )
    driver = module.FrankaSequenceDriver(robot, _FakePylibfranka, limits)

    with pytest.raises(RuntimeError, match="endpoint convergence timed out"):
        driver.move_pose(_pose([0.51, 0.0, 0.30]))

    assert robot.stop_count == 1
    assert robot.controls
    assert robot.controls[0].commands
    assert not any(command.motion_finished for command in robot.controls[0].commands)


@pytest.mark.parametrize(
    ("inject_fault", "message"),
    [
        (
            lambda state: setattr(
                state.current_errors, "cartesian_motion_generator_velocity_discontinuity", True
            ),
            "current_errors are active",
        ),
        (
            lambda state: state.cartesian_contact.__setitem__(1, 1.0),
            "cartesian_contact",
        ),
        (
            lambda state: state.joint_collision.__setitem__(4, 1.0),
            "joint_collision",
        ),
        (
            lambda state: setattr(
                state, "O_T_EE", _franka_values(_pose([0.90, 0.0, 0.30]))
            ),
            "outside.*workspace",
        ),
        (
            lambda state: setattr(state, "control_command_success_rate", 0.79),
            "crossed hard floor",
        ),
        (
            lambda state: state.O_T_EE.__setitem__(3, 0.01),
            "homogeneous bottom row",
        ),
        (
            lambda state: state.O_T_EE.__setitem__(0, 1.01),
            "not orthonormal",
        ),
        (
            lambda state: state.O_T_EE.__setitem__(0, -1.0),
            r"determinant must be \+1",
        ),
    ],
)
def test_cartesian_endpoint_hold_faults_stop_before_motion_finished(
    driver_module, inject_fault, message
):
    module = driver_module

    class HoldFaultControl(_Control):
        def readOnce(self):
            result = super().readOnce()
            # With a 20 ms trajectory and 10 ms periods, read three is the
            # first endpoint-hold-only cycle.
            if self.period_index == 3:
                inject_fault(self.robot.state)
            return result

    class HoldFaultRobot(_Robot):
        def start_cartesian_pose_control(self, mode):
            assert mode == _FakePylibfranka.ControllerMode.CartesianImpedance
            control = HoldFaultControl(self, "cartesian")
            self.controls.append(control)
            return control

        def stop(self):
            # The injected fault caused the stop.  Model a successful stop
            # acknowledgement that clears transient controller indications so
            # the original fail-closed cause remains observable to the test.
            self.state.current_errors = SimpleNamespace()
            self.state.cartesian_contact[:] = 0.0
            self.state.cartesian_collision[:] = 0.0
            self.state.joint_contact[:] = 0.0
            self.state.joint_collision[:] = 0.0
            self.state.control_command_success_rate = 1.0
            super().stop()

    robot = HoldFaultRobot(periods=[0.01])
    limits = _limits(
        module,
        min_cartesian_duration_s=0.02,
        cartesian_endpoint_hold_s=0.03,
    )
    driver = module.FrankaSequenceDriver(robot, _FakePylibfranka, limits)

    with pytest.raises(RuntimeError, match=message):
        driver.move_pose(_pose([0.51, 0.0, 0.30]))

    assert robot.stop_count == 1
    assert len(robot.controls[0].commands) == 103
    assert not any(command.motion_finished for command in robot.controls[0].commands)


def test_cartesian_endpoint_hold_is_covered_by_wall_deadline(driver_module):
    module = driver_module
    clock = _FakeClock()

    class AdvancingCartesianControl(_Control):
        def readOnce(self):
            if self.startup_index >= 101:
                clock.value += 0.01
            return super().readOnce()

    class AdvancingCartesianRobot(_Robot):
        def start_cartesian_pose_control(self, mode):
            assert mode == _FakePylibfranka.ControllerMode.CartesianImpedance
            control = AdvancingCartesianControl(self, "cartesian")
            self.controls.append(control)
            return control

    robot = AdvancingCartesianRobot(periods=[0.01])
    limits = _limits(
        module,
        min_cartesian_duration_s=0.02,
        cartesian_endpoint_hold_s=0.03,
        wall_deadline_slack_s=0.015,
        wall_deadline_fraction=0.01,
    )
    driver = module.FrankaSequenceDriver(
        robot,
        _FakePylibfranka,
        limits,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    actual = driver.move_pose(_pose([0.51, 0.0, 0.30]))

    np.testing.assert_allclose(actual, _pose([0.51, 0.0, 0.30]))
    assert clock.value >= 0.05


def test_minimum_jerk_orientation_interpolation_is_on_so3(driver_module):
    module = driver_module
    start = _pose(yaw=0.0)
    target = _pose([0.54, 0.0, 0.30], yaw=np.pi / 2.0)

    halfway = module.interpolate_pose_minimum_jerk(start, target, 0.5)

    # Minimum jerk is exactly 0.5 at alpha=0.5.
    np.testing.assert_allclose(halfway[:3, 3], [0.52, 0.0, 0.30], atol=1e-10)
    assert module.pose_error(start, halfway)[1] == pytest.approx(np.pi / 4.0)
    np.testing.assert_allclose(
        halfway[:3, :3].T @ halfway[:3, :3], np.eye(3), atol=1e-12
    )
    assert np.linalg.det(halfway[:3, :3]) == pytest.approx(1.0)


def test_prepared_cartesian_hot_sampler_matches_reviewed_se3_interpolator(
    driver_module,
):
    module = driver_module
    start = _pose([0.48, -0.02, 0.31], yaw=0.17)
    target = np.eye(4)
    target[:3, :3] = module.so3_exp([0.23, -0.11, 0.31]) @ start[:3, :3]
    target[:3, 3] = [0.51, 0.01, 0.29]
    prepared = module._PreparedCartesianInterpolation(start, target)

    for alpha in (0.0, 0.01, 0.25, 0.5, 0.87, 1.0):
        sampled = np.asarray(prepared.command_values(alpha)).reshape(
            (4, 4), order="F"
        )
        reviewed = module.interpolate_pose_minimum_jerk(start, target, alpha)
        np.testing.assert_allclose(sampled, reviewed, atol=2.0e-12, rtol=0.0)


def test_realtime_cartesian_translation_returns_same_validated_buffer(
    driver_module,
):
    module = driver_module
    robot = _Robot()
    driver = module.FrankaSequenceDriver(
        robot, _FakePylibfranka, _limits(module)
    )
    original = robot.state.O_T_EE

    validated = driver._validate_realtime_cartesian_translation(robot.state)

    assert validated is original


def test_realtime_endpoint_scalar_pose_error_matches_reviewed_se3_error(
    driver_module,
):
    module = driver_module
    first = _pose([0.48, -0.02, 0.31], yaw=0.17)
    second = _pose([0.51, 0.01, 0.29], yaw=-0.24)
    tilted = _pose([0.49, 0.03, 0.33])
    tilted[:3, :3] = module.so3_exp([0.23, -0.11, 0.31])
    target = _pose([0.52, -0.01, 0.30])
    target[:3, :3] = module.so3_exp([-0.14, 0.19, -0.27])

    for actual, expected_target in ((first, second), (tilted, target)):
        reviewed = module.pose_error(actual, expected_target)
        scalar = module._realtime_endpoint_pose_error(
            _franka_values(actual),
            _franka_values(expected_target),
        )
        assert scalar[0] == pytest.approx(reviewed[0], abs=2.0e-12)
        assert scalar[1] == pytest.approx(reviewed[1], abs=2.0e-12)


@pytest.mark.parametrize(
    ("index", "value", "message"),
    [
        (3, 0.01, "homogeneous bottom row"),
        (0, 1.01, "not orthonormal"),
        (0, -1.0, r"determinant must be \+1"),
    ],
)
def test_realtime_endpoint_scalar_pose_error_retains_rigid_pose_gates(
    driver_module, index, value, message
):
    module = driver_module
    actual = _franka_values(_pose())
    actual[index] = value

    with pytest.raises(RuntimeError, match=message):
        module._realtime_endpoint_pose_error(actual, _franka_values(_pose()))


def test_realtime_endpoint_scalar_dq_validation_retains_shape_and_finite_gates(
    driver_module,
):
    module = driver_module

    assert module._validated_realtime_max_abs_dq(
        [0.0, -0.01, 0.02, -0.03, 0.0, 0.01, 0.0]
    ) == pytest.approx(0.03)
    with pytest.raises(RuntimeError, match="missing or malformed"):
        module._validated_realtime_max_abs_dq(np.zeros(6))
    invalid = np.zeros(7)
    invalid[4] = np.nan
    with pytest.raises(RuntimeError, match="NaN or infinity"):
        module._validated_realtime_max_abs_dq(invalid)


def test_cartesian_hot_loop_checks_live_eef_translation_workspace(
    driver_module,
):
    module = driver_module

    class WorkspaceFaultControl(_Control):
        def readOnce(self):
            self.robot.state.O_T_EE = _franka_values(_pose([0.90, 0.0, 0.30]))
            return super().readOnce()

    class WorkspaceFaultRobot(_Robot):
        def start_cartesian_pose_control(self, mode):
            assert mode == _FakePylibfranka.ControllerMode.CartesianImpedance
            control = WorkspaceFaultControl(self, "cartesian")
            self.controls.append(control)
            return control

    robot = WorkspaceFaultRobot()
    driver = module.FrankaSequenceDriver(
        robot, _FakePylibfranka, _limits(module)
    )

    with pytest.raises(RuntimeError, match="outside.*workspace"):
        driver.move_pose(_pose([0.51, 0.0, 0.30]))

    assert robot.stop_count == 1


def test_verify_settled_requires_consecutive_good_samples_and_accepts_tolerances(
    driver_module,
):
    module = driver_module
    clock = _FakeClock()
    robot = _Robot()
    limits = _limits(module)
    driver = module.FrankaSequenceDriver(
        robot,
        _FakePylibfranka,
        limits,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    tolerances = SimpleNamespace(
        position_m=0.005,
        orientation_rad=0.05,
        stable_seconds=0.03,
    )

    assert driver.verify_settled(_pose(), tolerances)
    assert clock.value >= 0.03

    bad_clock = _FakeClock()
    bad_robot = _Robot()
    bad_robot.state.O_T_EE = _franka_values(_pose([0.52, 0.0, 0.30]))
    bad_driver = module.FrankaSequenceDriver(
        bad_robot,
        _FakePylibfranka,
        limits,
        monotonic=bad_clock.monotonic,
        sleep=bad_clock.sleep,
    )
    assert not bad_driver.verify_settled(_pose(), tolerances)
    assert bad_robot.stop_count == 0


def test_audited_joint_settle_checks_bound_q_and_eef_without_workspace(
    driver_module,
):
    module = driver_module
    target_q = np.asarray([0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])
    clock = _FakeClock()
    robot = _Robot(_State(q=target_q, O_T_EE=_pose()))
    limits = _limits(
        module,
        workspace_min_m=None,
        workspace_max_m=None,
        max_continuous_joint_tracking_error_rad=0.002,
    )
    driver = module.FrankaSequenceDriver(
        robot,
        _FakePylibfranka,
        limits,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    tolerances = SimpleNamespace(
        position_m=0.005,
        orientation_rad=0.05,
        stable_seconds=0.03,
    )

    assert driver.verify_audited_joint_settled(
        target_q, _pose(), 0.002, tolerances
    )
    assert clock.value >= 0.03
    assert robot.stop_count == 0

    robot.state.q = target_q + np.asarray([0.01, 0, 0, 0, 0, 0, 0])
    clock.value = 0.0
    assert not driver.verify_audited_joint_settled(
        target_q, _pose(), 0.002, tolerances
    )
    assert robot.stop_count == 0


def test_audited_joint_settle_requires_exact_audit_tracking_tolerance(
    driver_module,
):
    module = driver_module
    robot = _Robot()
    limits = _limits(
        module,
        workspace_min_m=None,
        workspace_max_m=None,
        max_continuous_joint_tracking_error_rad=0.002,
    )
    driver = module.FrankaSequenceDriver(robot, _FakePylibfranka, limits)

    with pytest.raises(ValueError, match="exactly equal"):
        driver.verify_audited_joint_settled(
            robot.state.q, _pose(), 0.003, SimpleNamespace(stable_seconds=0.03)
        )
    assert robot.stop_count == 1


def test_cartesian_move_requires_explicit_workspace_and_connect_is_explicit(
    driver_module,
):
    module = driver_module
    robot = _Robot()
    no_workspace = module.FrankaMotionLimits(expected_F_T_EE=np.eye(4))
    driver = module.FrankaSequenceDriver(robot, _FakePylibfranka, no_workspace)
    with pytest.raises(RuntimeError, match="requires commissioned workspace"):
        driver.move_pose(_pose())
    assert robot.stop_count == 1
    with pytest.raises(RuntimeError, match="requires commissioned workspace"):
        driver.verify_settled(_pose())
    assert robot.stop_count == 2

    created = []

    class ConnectModule(_FakePylibfranka):
        @staticmethod
        def Robot(ip, realtime):
            created.append((ip, realtime))
            return _Robot()

    connected = module.FrankaSequenceDriver.connect(
        "172.16.0.2", no_workspace, pylibfranka=ConnectModule
    )
    assert created == [("172.16.0.2", ConnectModule.RealtimeConfig.kIgnore)]
    assert isinstance(connected.robot, _Robot)

    enforced = module.FrankaSequenceDriver.connect(
        "172.16.0.3",
        no_workspace,
        pylibfranka=ConnectModule,
        enforce_realtime=True,
    )
    assert created[-1] == (
        "172.16.0.3",
        ConnectModule.RealtimeConfig.kEnforce,
    )
    assert isinstance(enforced.robot, _Robot)


def test_stop_is_commanded_before_consecutive_physical_rest_verification(
    driver_module,
):
    module = driver_module
    clock = _FakeClock()
    log = []

    def state(*, mode="Idle", dq=0.0):
        value = _State()
        value.robot_mode = _Mode(mode)
        value.dq = np.full(7, float(dq))
        return value

    # Two initially good samples do not suffice: a later moving sample resets
    # the run, after which all three required samples must be observed again.
    samples = iter(
        [
            state(),
            state(),
            state(mode="Move", dq=0.05),
            state(),
            state(),
            state(),
        ]
    )

    class ScriptedRobot(_Robot):
        def stop(self):
            self.stop_count += 1
            log.append("stop")

        def read_once(self):
            log.append("read")
            return next(samples)

    robot = ScriptedRobot()
    driver = module.FrankaSequenceDriver(
        robot,
        _FakePylibfranka,
        _limits(module),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    driver.stop()

    assert robot.stop_count == 1
    assert log == ["stop"] + ["read"] * 6
    assert clock.value == pytest.approx(0.005)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda state: setattr(state, "robot_mode", _Mode("Move")), "expected idle"),
        (
            lambda state: setattr(
                state.current_errors, "joint_position_limits_violation", True
            ),
            "current_errors are active",
        ),
        (
            lambda state: state.cartesian_contact.__setitem__(2, 1.0),
            "cartesian_contact",
        ),
        (
            lambda state: state.joint_collision.__setitem__(4, 1.0),
            "joint_collision",
        ),
        (lambda state: state.dq.__setitem__(3, 0.021), "max |dq|"),
        (lambda state: setattr(state, "dq", np.zeros(6)), "state.dq"),
    ],
)
def test_stop_fails_closed_when_physical_rest_cannot_be_proved(
    driver_module, mutate, message
):
    module = driver_module
    clock = _FakeClock()
    unsafe = _State()
    mutate(unsafe)

    class UnsafeRobot(_Robot):
        def __init__(self):
            super().__init__(unsafe)
            self.events = []

        def stop(self):
            self.stop_count += 1
            self.events.append("stop")

        def read_once(self):
            self.events.append("read")
            return self.state

    robot = UnsafeRobot()
    driver = module.FrankaSequenceDriver(
        robot,
        _FakePylibfranka,
        _limits(module, stop_verify_timeout_s=0.003),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    with pytest.raises(module.FrankaStopUnconfirmed) as exc_info:
        driver.stop()

    assert "STOP UNCONFIRMED" in str(exc_info.value)
    assert message in str(exc_info.value)
    assert robot.stop_count == 1
    assert robot.events[0] == "stop"
    assert robot.events.count("read") >= 2


def test_stop_readback_failure_is_unconfirmed_but_never_blocks_stop_command(
    driver_module,
):
    module = driver_module
    clock = _FakeClock()

    class UnreadableRobot(_Robot):
        def read_once(self):
            raise RuntimeError("FCI read timeout")

    robot = UnreadableRobot()
    driver = module.FrankaSequenceDriver(
        robot,
        _FakePylibfranka,
        _limits(module, stop_verify_timeout_s=0.003),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    with pytest.raises(module.FrankaStopUnconfirmed, match="FCI read timeout"):
        driver.stop()

    assert robot.stop_count == 1


def test_stop_command_exception_is_unconfirmed_and_does_not_attempt_readback(
    driver_module,
):
    module = driver_module

    class StopFailureRobot(_Robot):
        def __init__(self):
            super().__init__()
            self.read_count = 0

        def stop(self):
            self.stop_count += 1
            raise RuntimeError("stop transport lost")

        def read_once(self):
            self.read_count += 1
            return self.state

    robot = StopFailureRobot()
    driver = module.FrankaSequenceDriver(
        robot, _FakePylibfranka, _limits(module)
    )

    with pytest.raises(module.FrankaStopUnconfirmed, match="Robot.stop.*transport"):
        driver.stop()

    assert robot.stop_count == 1
    assert robot.read_count == 0


def test_motion_exception_cleanup_reports_unconfirmed_physical_stop(driver_module):
    module = driver_module
    clock = _FakeClock()
    state = _State()
    state.joint_contact[0] = 1.0
    robot = _Robot(state)
    driver = module.FrankaSequenceDriver(
        robot,
        _FakePylibfranka,
        _limits(module, stop_verify_timeout_s=0.003),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    with pytest.raises(module.FrankaStopUnconfirmed, match="STOP UNCONFIRMED"):
        driver.move_joints([0.1, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])

    assert robot.stop_count == 1


@pytest.mark.parametrize("value", [True, 1, 1.5, np.nan])
def test_stop_sample_count_configuration_rejects_non_integer_or_too_small(
    driver_module, value
):
    with pytest.raises(ValueError, match="stop_verify_consecutive_samples"):
        _limits(driver_module, stop_verify_consecutive_samples=value)
