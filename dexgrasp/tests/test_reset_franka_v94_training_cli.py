from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"
BUNDLE = ROOT.parent / "data/test_fixtures/sim2real/deploy.zip"


def _load(name="test_reset_franka_v94_training_app"):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "apps/reset_franka_v94_training.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tokens(app):
    return [
        "--confirm-installed",
        app.INSTALLED_TOKEN,
        "--confirm-workspace-clear",
        app.SWEEP_CLEAR_TOKEN,
        "--confirm-stop-ready",
        app.STOP_TOKEN,
        "--confirm-pla-low-speed",
        app.PLA_TOKEN,
        "--confirm-training-target",
        app.TARGET_TOKEN,
    ]


def test_dry_run_is_hardware_free_and_uses_bundle_q_home(monkeypatch, capsys):
    app = _load("test_reset_franka_v94_training_dry")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run reached hardware reset")

    monkeypatch.setattr(app.DEFAULT_RESET, "run_franka_only_reset", forbidden)
    assert app.main(["--config", str(CONFIG), "--bundle", str(BUNDLE), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "No hardware driver was imported" in output
    assert "RH56 would not be imported or accessed" in output
    assert "-0.569000" in output
    assert "-2.809999" in output
    assert "3.036999" in output


def test_missing_confirmations_cannot_reach_reset(monkeypatch, capsys):
    app = _load("test_reset_franka_v94_training_tokens")
    calls = []
    monkeypatch.setattr(
        app.DEFAULT_RESET,
        "run_franka_only_reset",
        lambda *_args, **_kwargs: calls.append(1),
    )
    assert app.main(["--config", str(CONFIG), "--bundle", str(BUNDLE)]) == 1
    assert calls == []
    assert "exact confirmations required before hardware import" in capsys.readouterr().err


def test_confirmed_run_passes_only_hash_bound_target(monkeypatch):
    app = _load("test_reset_franka_v94_training_run")
    calls = []

    def record(config):
        calls.append(np.asarray(config["franka"]["default_q_rad"]))

    monkeypatch.setattr(app.DEFAULT_RESET, "run_franka_only_reset", record)
    assert app.main(
        [
            "--config",
            str(CONFIG),
            "--bundle",
            str(BUNDLE),
            *_tokens(app),
        ]
    ) == 0
    assert len(calls) == 1
    np.testing.assert_allclose(
        calls[0],
        [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741],
        atol=2.0e-6,
        rtol=0.0,
    )


def test_training_target_does_not_mutate_commissioning_default():
    app = _load("test_reset_franka_v94_training_copy")
    config, _ = app.load_control_config(CONFIG)
    original = np.asarray(config["franka"]["default_q_rad"], dtype=np.float64).copy()
    target, _ = app._training_target(BUNDLE)
    runtime = app._reset_config(config, target)
    np.testing.assert_array_equal(config["franka"]["default_q_rad"], original)
    assert not np.array_equal(runtime["franka"]["default_q_rad"], original)
