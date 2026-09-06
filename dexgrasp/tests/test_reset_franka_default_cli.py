from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"


def _load(name="test_reset_franka_default_app"):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "apps/reset_franka_default.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tokens(app):
    return [
        "--confirm-installed", app.INSTALLED_TOKEN,
        "--confirm-hand-open", app.HAND_OPEN_TOKEN,
        "--confirm-workspace-clear", app.SWEEP_CLEAR_TOKEN,
        "--confirm-stop-ready", app.STOP_TOKEN,
        "--confirm-pla-low-speed", app.PLA_TOKEN,
    ]


def _snapshot(**changes):
    value = {
        "hand_id": 1,
        "angles": (1000, 1000, 997, 1000, 1000, 985),
        "angle_targets": (-1,) * 6,
        "currents": (0,) * 6,
        "errors": (0,) * 6,
        "statuses": (2,) * 6,
        "speeds": (1000,) * 6,
        "force_limits": (500,) * 6,
    }
    value.update(changes)
    return value


def test_dry_run_imports_no_hardware(monkeypatch, capsys):
    app = _load("test_reset_franka_default_dry")

    def forbidden():
        raise AssertionError("dry-run imported hardware")

    monkeypatch.setattr(app, "_load_hardware_types", forbidden)
    assert app.main(["--config", str(CONFIG), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "No hardware driver was imported" in output
    assert "-0.1118436" in output


def test_missing_confirmation_cannot_reach_hardware(monkeypatch, capsys):
    app = _load("test_reset_franka_default_tokens")
    calls = []
    monkeypatch.setattr(app, "run_reset", lambda *_args, **_kwargs: calls.append(1))

    assert app.main(["--config", str(CONFIG)]) == 1
    assert calls == []
    assert "exact confirmations required before hardware import" in capsys.readouterr().err


def test_hand_gate_rejects_non_disabled_or_noncanonical_settings():
    app = _load("test_reset_franka_default_hand_gate")
    for broken in (
        _snapshot(angle_targets=(-1, -1, -1, -1, -1, 1000)),
        _snapshot(angles=(1000, 1000, 900, 1000, 1000, 985)),
        _snapshot(speeds=(1000, 1000, 1000, 1000, 1000, 40)),
        _snapshot(force_limits=(500, 500, 500, 500, 500, 80)),
        _snapshot(errors=(0, 0, 0, 0, 0, 1)),
    ):
        try:
            app._verify_open_disabled_snapshot(
                broken, expected_hand_id=1, label="test"
            )
        except RuntimeError as exc:
            assert "reset_installed_rh56_open.sh" in str(exc)
        else:
            raise AssertionError("unsafe RH56 state was accepted")


def test_disabled_franka_gate_uses_commissioned_q6_range_not_open_floor():
    app = _load("test_reset_franka_default_q6_release_drift")
    config, _ = app.load_control_config(CONFIG)
    minimums = app._profile_open_min_angles(config)
    assert minimums == (980, 980, 980, 980, 980, 0)

    accepted = _snapshot(
        angles=(1000, 1000, 1000, 1000, 1000, 0)
    )
    assert app._verify_open_disabled_snapshot(
        accepted,
        expected_hand_id=1,
        label="test",
        open_min_angles=minimums,
    ) == accepted["angles"]



def test_confirmed_run_keeps_hand_read_only_and_moves_only_franka(monkeypatch):
    app = _load("test_reset_franka_default_run")
    snapshots = [_snapshot(), _snapshot(), _snapshot(), _snapshot()]
    calls = []

    class Serial:
        def __init__(self, *args):
            calls.append(("serial-init", args))

        def __enter__(self):
            calls.append(("serial-enter",))
            return self

        def __exit__(self, *_args):
            calls.append(("serial-exit",))

    class Hand:
        def __init__(self, serial, hand_id):
            calls.append(("hand", serial, hand_id))

        def snapshot(self):
            calls.append(("snapshot",))
            return snapshots.pop(0)

    class Limits:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    target = np.asarray(
        [-0.1118436, -0.1207545, 0.0739457, -1.7431009, 0.046354,
         1.6809169, 0.8117281]
    )

    class Arm:
        def __init__(self, limits):
            self.limits = limits
            self.robot = self
            self.q = target.copy()
            self.last_control_loop_telemetry = SimpleNamespace(
                kind="joint", samples=100, max_read_to_write_us=20.0,
                read_to_write_overruns=0,
            )

        def read_once(self):
            return SimpleNamespace(q=self.q.copy())

        def _validate_state(self, _state, **kwargs):
            calls.append(("validate-franka", kwargs))
            return 1.0

        def move_joints(self, wanted):
            calls.append(("move", np.asarray(wanted).copy()))
            self.q = np.asarray(wanted).copy()

        def stop(self):
            calls.append(("stop",))

    class ArmType:
        @classmethod
        def connect(cls, ip, limits, enforce_realtime):
            calls.append(("connect-franka", ip, enforce_realtime))
            return Arm(limits)

    types = {
        "rh56_api": SimpleNamespace(LinuxSerial=Serial, RH56Hand=Hand),
        "FrankaMotionLimits": Limits,
        "FrankaSequenceDriver": ArmType,
    }
    monkeypatch.setattr(app, "_load_hardware_types", lambda: types)
    monkeypatch.setattr(
        app,
        "require_uncontended_franka_https_link",
        lambda ip: calls.append(("network", ip)),
    )

    assert app.main(["--config", str(CONFIG), *_tokens(app)]) == 0
    assert len([item for item in calls if item[0] == "snapshot"]) == 4
    assert len([item for item in calls if item[0] == "move"]) == 1
    assert len([item for item in calls if item[0] == "stop"]) == 1
    assert not snapshots
    connect_index = next(i for i, item in enumerate(calls) if item[0] == "connect-franka")
    assert sum(item[0] == "snapshot" for item in calls[:connect_index]) == 2


def test_keyboard_interrupt_still_stops_and_returns_130(monkeypatch):
    app = _load("test_reset_franka_default_interrupt")
    snapshots = [_snapshot(), _snapshot(), _snapshot(), _snapshot()]
    stops = []

    class Serial:
        def __init__(self, *_args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    class Hand:
        def __init__(self, *_args):
            pass

        def snapshot(self):
            return snapshots.pop(0)

    class Limits:
        def __init__(self, **_kwargs):
            pass

    class Arm:
        last_control_loop_telemetry = None

        def __init__(self):
            self.robot = self

        def read_once(self):
            return SimpleNamespace(q=np.zeros(7))

        def _validate_state(self, *_args, **_kwargs):
            return 1.0

        def move_joints(self, _wanted):
            raise KeyboardInterrupt()

        def stop(self):
            stops.append(1)

    class ArmType:
        @classmethod
        def connect(cls, *_args, **_kwargs):
            return Arm()

    monkeypatch.setattr(
        app,
        "_load_hardware_types",
        lambda: {
            "rh56_api": SimpleNamespace(LinuxSerial=Serial, RH56Hand=Hand),
            "FrankaMotionLimits": Limits,
            "FrankaSequenceDriver": ArmType,
        },
    )
    monkeypatch.setattr(app, "require_uncontended_franka_https_link", lambda _ip: None)

    assert app.main(["--config", str(CONFIG), *_tokens(app)]) == 130
    assert stops == [1]


def test_formal_start_envelope_is_checked_before_move_joints():
    app = _load("test_reset_franka_default_start_envelope")
    base, _ = app.load_control_config(CONFIG)

    def run(initial_delta):
        config = copy.deepcopy(base)
        config["franka"]["default_q_rad"] = [0.0] * 7
        snapshots = [_snapshot(), _snapshot(), _snapshot(), _snapshot()]
        calls = []

        class Serial:
            def __init__(self, *_args):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

        class Hand:
            def __init__(self, *_args):
                pass

            def snapshot(self):
                return snapshots.pop(0)

        class Limits:
            def __init__(self, **_kwargs):
                pass

        class Arm:
            last_control_loop_telemetry = None

            def __init__(self):
                self.robot = self
                self.q = np.asarray([initial_delta] + [0.0] * 6)

            def read_once(self):
                return SimpleNamespace(q=self.q.copy())

            def _validate_state(self, *_args, **_kwargs):
                return 1.0

            def move_joints(self, wanted):
                calls.append("move")
                self.q = np.asarray(wanted, dtype=np.float64)

            def stop(self):
                calls.append("stop")

        class ArmType:
            @classmethod
            def connect(cls, *_args, **_kwargs):
                return Arm()

        types = {
            "rh56_api": SimpleNamespace(LinuxSerial=Serial, RH56Hand=Hand),
            "FrankaMotionLimits": Limits,
            "FrankaSequenceDriver": ArmType,
        }
        return config, types, calls

    config, types, calls = run(1.21)
    proof = app.run_reset(
        config,
        hardware_types=types,
        network_preflight=lambda _ip: None,
        sleep=lambda _seconds: None,
        maximum_start_delta_rad=1.21,
    )
    assert calls == ["move", "stop"]
    assert proof.initial_linf_delta_rad == 1.21
    assert proof.maximum_start_delta_rad == 1.21

    config, types, calls = run(1.210001)
    try:
        app.run_reset(
            config,
            hardware_types=types,
            network_preflight=lambda _ip: None,
            sleep=lambda _seconds: None,
            maximum_start_delta_rad=1.21,
        )
    except RuntimeError as exc:
        assert "exceeding formal envelope" in str(exc)
    else:
        raise AssertionError("out-of-envelope Franka reset was accepted")
    assert calls == ["stop"]


def test_franka_only_reset_never_imports_or_accesses_rh56(monkeypatch):
    app = _load("test_reset_franka_default_arm_only")
    config, _ = app.load_control_config(CONFIG)
    target = np.asarray(config["franka"]["default_q_rad"], dtype=np.float64)
    calls = []

    class Limits:
        def __init__(self, **_kwargs):
            calls.append("limits")

    class Arm:
        last_control_loop_telemetry = None

        def __init__(self):
            self.robot = self
            self.q = target.copy()

        def read_once(self):
            calls.append("read")
            return SimpleNamespace(q=self.q.copy())

        def _validate_state(self, _state, **_kwargs):
            calls.append("validate")
            return 1.0

        def move_joints(self, wanted):
            calls.append("move")
            self.q = np.asarray(wanted, dtype=np.float64)

        def stop(self):
            calls.append("stop")

    class ArmType:
        @classmethod
        def connect(cls, _ip, _limits, enforce_realtime):
            assert enforce_realtime is True
            calls.append("connect")
            return Arm()

    franka_types = {
            "FrankaMotionLimits": Limits,
            "FrankaSequenceDriver": ArmType,
    }
    monkeypatch.setattr(
        app,
        "_load_hardware_types",
        lambda: (_ for _ in ()).throw(
            AssertionError("arm-only reset imported RH56 hardware types")
        ),
    )
    monkeypatch.setattr(
        app, "_load_franka_hardware_types", lambda: franka_types
    )

    proof = app.run_franka_only_reset(
        config,
        network_preflight=lambda _ip: calls.append("network"),
    )

    assert calls == [
        "limits",
        "network",
        "connect",
        "read",
        "validate",
        "move",
        "read",
        "validate",
        "stop",
    ]
    assert proof.franka_stop_verified is True
    assert not hasattr(proof, "rh56_open_disabled_verified")
