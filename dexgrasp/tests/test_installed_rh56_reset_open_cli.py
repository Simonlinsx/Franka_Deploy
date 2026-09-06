from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"
V94_CONFIG = ROOT / "configs/fr3_rh56_v94_commissioning.json"


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "apps/reset_installed_rh56_open.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tokens(app):
    return [
        "--confirm-installed", app.INSTALLED_TOKEN,
        "--confirm-24v-cutoff", app.POWER_TOKEN,
        "--confirm-franka-stop", app.STOP_TOKEN,
        "--confirm-workspace-clear", app.CLEAR_TOKEN,
        "--confirm-no-contact", app.NO_CONTACT_TOKEN,
        "--confirm-reset-open", app.RESET_TOKEN,
    ]


def test_reset_open_requires_exact_tokens_before_hardware(monkeypatch, capsys):
    app = _load("test_reset_open_tokens")
    calls = []
    monkeypatch.setattr(
        app,
        "_run_hardware_session",
        lambda *_args, **_kwargs: calls.append("hardware"),
    )

    result = app.main(["run"])

    assert result == 1
    assert calls == []
    assert "before hardware import" in capsys.readouterr().err


def test_confirmed_reset_open_dispatches_once(monkeypatch):
    app = _load("test_reset_open_confirmed")
    calls = []
    monkeypatch.setattr(
        app,
        "_run_hardware_session",
        lambda args, config: calls.append((args, config)) or (694, 719, 1000),
    )

    result = app.main(["run", *_tokens(app)])

    assert result == 0
    assert len(calls) == 1


def test_programmatic_reset_accepts_supervised_1000ma_host_limit():
    app = _load("test_reset_open_1000ma")
    config, _ = app.load_control_config(CONFIG)

    app._validate_reset_parameters(
        config,
        max_axis_current_ma=1000,
        max_inactive_drift_units=8,
        endpoint_stable_samples=3,
    )

    with pytest.raises(ValueError, match="50..1000"):
        app._validate_reset_parameters(
            config,
            max_axis_current_ma=1001,
            max_inactive_drift_units=8,
            endpoint_stable_samples=3,
        )


def test_hardware_session_passes_only_profile_commissioned_q6_range(monkeypatch):
    app = _load("test_reset_open_configured_q6_range")
    config, _ = app.load_control_config(CONFIG)
    captured = {}

    snapshot = {
        "angles": (1000, 1000, 997, 1000, 1000, 978),
        "angle_targets": (-1,) * 6,
        "currents": (0,) * 6,
        "statuses": (2,) * 6,
        "speeds": (1000,) * 6,
        "force_limits": (500,) * 6,
    }

    class FakeResetDriver:
        hand = SimpleNamespace(snapshot=lambda: dict(snapshot))

        @classmethod
        def connect(cls, **kwargs):
            captured.update(kwargs)
            return cls()

        def adopt_disabled_state_and_verify(self):
            return None

        def install_external_safety_check(self, callback):
            self.external_safety_check = callback

        def reset_to_open(self, **kwargs):
            captured["reset_kwargs"] = dict(kwargs)
            return (920, 1000)

        def disable_and_verify(self):
            return None

        def close(self):
            return None

    class FakeRobot:
        def read_once(self):
            return SimpleNamespace()

    class FakeArm:
        def __init__(self, robot, _library, _limits):
            self.robot = robot

        def _validate_state(self, _state, **_kwargs):
            return None

    class FakeLimits:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_pylibfranka = SimpleNamespace(
        Robot=lambda *_args, **_kwargs: FakeRobot(),
        RealtimeConfig=SimpleNamespace(kIgnore=object()),
    )
    monkeypatch.setitem(sys.modules, "pylibfranka", fake_pylibfranka)
    import anydex_pipeline.franka_sequence_driver as franka_driver

    monkeypatch.setattr(franka_driver, "FrankaMotionLimits", FakeLimits)
    monkeypatch.setattr(franka_driver, "FrankaSequenceDriver", FakeArm)
    monkeypatch.setattr(app, "RH56ResetOpenDriver", FakeResetDriver)
    args = app.build_parser().parse_args(["run", *_tokens(app)])

    assert app._run_hardware_session(args, config) == (920, 1000)
    assert captured["thumb_rotate_range"] == (0, 1000)
    assert captured["open_min_angle"] == 980
    assert captured["q6_open_min_angle"] == 975
    assert captured["q6_feedback_recovery_min_angle"] == 0
    assert captured["reset_kwargs"]["simultaneous_bend_open"] is True
    assert captured["reset_kwargs"]["direct_q6_endpoint_open"] is True


def test_v7_and_v94_profiles_share_installed_q6_feedback_endpoint_band():
    app = _load("test_reset_open_q6_feedback_endpoint")

    v7_config, _ = app.load_control_config(CONFIG)
    v94_config = json.loads(V94_CONFIG.read_text(encoding="utf-8"))
    for config in (v7_config, v94_config):
        assert tuple(config["inspire"]["open_targets"]) == (1000,) * 6
        assert app._profile_q6_open_min_angle(config) == 975


def test_final_disabled_snapshot_does_not_repeat_q6_endpoint_gate():
    app = _load("test_reset_open_final_q6_release_drift")
    baseline = {
        "angles": (1000, 1000, 1000, 1000, 1000, 900),
        "angle_targets": (-1,) * 6,
        "currents": (0,) * 6,
        "statuses": (2,) * 6,
        "speeds": (1000,) * 6,
        "force_limits": (500,) * 6,
    }

    values = app._validate_final_reset_open_snapshot(baseline)
    assert values[0] == baseline["angles"]

    for field, replacement in (
        ("angles", (1000, 1000, 979, 1000, 1000, 900)),
        ("angle_targets", (-1, -1, -1, -1, -1, 1000)),
        ("currents", (0, 0, 0, 0, 0, 101)),
        ("statuses", (2, 2, 2, 2, 2, 1)),
        ("speeds", (1000, 1000, 1000, 1000, 1000, 40)),
        ("force_limits", (500, 500, 500, 500, 500, 80)),
    ):
        broken = dict(baseline)
        broken[field] = replacement
        with pytest.raises(
            RuntimeError,
            match="final reset-open verification failed",
        ):
            app._validate_final_reset_open_snapshot(broken)


def _interrupting_hardware_types(app, events, *, disable_error=None):
    snapshot = {
        "angles": (1000, 1000, 1000, 1000, 1000, 980),
        "angle_targets": (-1,) * 6,
        "currents": (0,) * 6,
        "statuses": (2,) * 6,
        "speeds": (1000,) * 6,
        "force_limits": (500,) * 6,
    }

    class FakeHand:
        def snapshot(self):
            events.append("snapshot")
            return dict(snapshot)

    class FakeResetDriver:
        def __init__(self):
            self.hand = FakeHand()

        @classmethod
        def connect(cls, **_kwargs):
            events.append("connect")
            return cls()

        def adopt_disabled_state_and_verify(self):
            events.append("adopt_disabled")

        def install_external_safety_check(self, _callback):
            events.append("install_franka_gate")

        def reset_to_open(self, **_kwargs):
            events.append("reset")
            raise KeyboardInterrupt("operator Ctrl+C")

        def disable_and_verify(self):
            events.append("disable_and_verify")
            if disable_error is not None:
                raise disable_error

        def close(self):
            events.append("close")

    class FakeRobot:
        def read_once(self):
            events.append("franka_read")
            return SimpleNamespace()

    class FakeArm:
        def __init__(self, robot, _library, _limits):
            self.robot = robot

        def _validate_state(self, _state, **_kwargs):
            events.append("franka_validate")

    class FakeLimits:
        def __init__(self, **_kwargs):
            pass

    return {
        "pylibfranka": SimpleNamespace(
            Robot=lambda *_args, **_kwargs: FakeRobot(),
            RealtimeConfig=SimpleNamespace(kIgnore=object()),
        ),
        "FrankaMotionLimits": FakeLimits,
        "FrankaSequenceDriver": FakeArm,
        "RH56ResetOpenDriver": FakeResetDriver,
    }


def test_ctrl_c_is_reraised_after_verified_disable_and_cleanup():
    app = _load("test_reset_open_ctrl_c")
    config, _ = app.load_control_config(CONFIG)
    events = []

    with pytest.raises(KeyboardInterrupt, match="operator Ctrl\\+C"):
        app.run_installed_rh56_reset_open(
            config,
            hardware_types=_interrupting_hardware_types(app, events),
        )

    assert events[-3:] == ["disable_and_verify", "snapshot", "close"]


def test_ctrl_c_stop_unconfirmed_remains_higher_priority():
    app = _load("test_reset_open_ctrl_c_stop_unconfirmed")
    config, _ = app.load_control_config(CONFIG)
    events = []

    with pytest.raises(RuntimeError, match="STOP UNCONFIRMED"):
        app.run_installed_rh56_reset_open(
            config,
            hardware_types=_interrupting_hardware_types(
                app,
                events,
                disable_error=app.RH56StopUnconfirmed("no idle proof"),
            ),
        )

    assert "close" in events


def test_ctrl_c_cleanup_failure_remains_higher_priority():
    app = _load("test_reset_open_ctrl_c_cleanup_failure")
    config, _ = app.load_control_config(CONFIG)
    events = []

    with pytest.raises(
        RuntimeError,
        match="interrupted and motion is stopped, but cleanup/default-setting",
    ):
        app.run_installed_rh56_reset_open(
            config,
            hardware_types=_interrupting_hardware_types(
                app,
                events,
                disable_error=RuntimeError("restore failed"),
            ),
        )

    assert events[-2:] == ["snapshot", "close"]
