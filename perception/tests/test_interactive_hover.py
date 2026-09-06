import signal

import pytest

from dynamic_pcd.apps.interactive_hover import InteractiveHoverController


CALIBRATION_ID = "eye-to-hand-test"


class FakeProcess:
    def __init__(self, command, **kwargs):
        self.command = list(command)
        self.kwargs = kwargs
        self.returncode = None
        self.signals = []

    def poll(self):
        return self.returncode

    def send_signal(self, sig):
        self.signals.append(sig)

    def wait(self, timeout=None):
        self.returncode = 130
        return self.returncode


class FakePopenFactory:
    def __init__(self):
        self.processes = []

    def __call__(self, command, **kwargs):
        process = FakeProcess(command, **kwargs)
        self.processes.append(process)
        return process


def _controller(factory=None, clock=None, **overrides):
    values = {
        "config_path": "configs/d435_default.yaml",
        "zmq_addr": "tcp://127.0.0.1:5556",
        "robot_ip": "172.16.0.2",
        "clearance_m": 0.10,
        "loaded_calibration_id": CALIBRATION_ID,
        "enable_robot_motion": True,
        "confirm_calibration_id": CALIBRATION_ID,
        "confirm_workspace_clear": True,
        "confirm_eef_clear": True,
        "confirm_descent_clear": True,
        "python_executable": "/test/python",
        "popen_factory": factory or FakePopenFactory(),
        "monotonic": clock or (lambda: 100.0),
    }
    values.update(overrides)
    return InteractiveHoverController(**values)


def test_plan_command_is_non_executing_but_checks_guarded_descent_geometry():
    controller = _controller(enable_robot_motion=False)

    command = controller.build_command(execute=False)

    assert "--allow-descent" in command
    assert command[command.index("--clearance-m") + 1] == "0.1"
    assert "--execute" not in command
    assert not any(item.startswith("--confirm-") for item in command)


def test_execute_command_contains_every_fail_closed_confirmation():
    controller = _controller()

    command = controller.build_command(execute=True)

    assert "--execute" in command
    assert command[command.index("--confirm-calibration-id") + 1] == CALIBRATION_ID
    assert "--confirm-workspace-clear" in command
    assert "--confirm-eef-clear" in command
    assert "--confirm-descent-clear" in command


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"enable_robot_motion": False}, "disabled"),
        ({"confirm_calibration_id": "wrong"}, "does not match"),
        ({"confirm_workspace_clear": False}, "workspace"),
        ({"confirm_eef_clear": False}, "eef"),
        ({"confirm_descent_clear": False}, "descent"),
    ],
)
def test_motion_arm_refuses_missing_opt_in_or_confirmation(override, message, capsys):
    controller = _controller(**override)

    assert not controller.arm_motion(packet_valid=True, roi_locked=True)
    assert message in capsys.readouterr().out.lower()


def test_motion_requires_distinct_arm_then_confirm_and_locked_valid_target():
    factory = FakePopenFactory()
    controller = _controller(factory=factory)

    assert not controller.confirm_motion(packet_valid=True, roi_locked=True)
    assert not controller.arm_motion(packet_valid=False, roi_locked=True)
    assert not controller.arm_motion(packet_valid=True, roi_locked=False)
    assert controller.arm_motion(packet_valid=True, roi_locked=True)
    assert controller.armed
    assert controller.confirm_motion(packet_valid=True, roi_locked=True)

    assert len(factory.processes) == 1
    assert "--execute" in factory.processes[0].command
    assert factory.processes[0].kwargs["start_new_session"] is True
    assert not controller.armed


def test_arm_expires_and_cannot_be_confirmed():
    now = [100.0]
    controller = _controller(clock=lambda: now[0], arm_timeout_s=5.0)
    assert controller.arm_motion(packet_valid=True, roi_locked=True)

    now[0] = 106.0

    assert not controller.confirm_motion(packet_valid=True, roi_locked=True)
    assert controller.status == "ARM EXPIRED"


def test_only_one_child_job_can_run_and_completion_restores_idle_availability():
    factory = FakePopenFactory()
    controller = _controller(factory=factory)

    assert controller.start_plan(packet_valid=True)
    assert not controller.start_plan(packet_valid=True)
    assert not controller.arm_motion(packet_valid=True, roi_locked=True)
    factory.processes[0].returncode = 0
    assert controller.poll() == 0
    assert controller.process is None
    assert controller.status == "PLAN DONE"
    assert controller.start_plan(packet_valid=True)


def test_cancel_and_shutdown_use_sigint_not_sigterm():
    factory = FakePopenFactory()
    controller = _controller(factory=factory)
    assert controller.start_plan(packet_valid=True)
    process = factory.processes[0]

    assert controller.request_cancel()
    assert process.signals == [signal.SIGINT]

    controller.shutdown()

    assert process.signals == [signal.SIGINT, signal.SIGINT]
    assert signal.SIGTERM not in process.signals

