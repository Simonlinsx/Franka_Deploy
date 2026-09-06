from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "apps" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_rh56_missing_confirmations_cannot_reach_motion(monkeypatch, capsys):
    app = _load("test_rh56_bench_app", "run_rh56_bench_demo.py")
    calls = []
    monkeypatch.setattr(app, "run_demo", lambda *args: calls.append(args))

    assert app.main([]) == 1
    assert calls == []
    assert "exact confirmations required" in capsys.readouterr().err


def test_rh56_exact_confirmations_select_reviewed_partial_target(monkeypatch):
    app = _load("test_rh56_bench_app_confirmed", "run_rh56_bench_demo.py")
    calls = []
    monkeypatch.setattr(app, "run_demo", lambda *args: calls.append(args))
    result = app.main(
        [
            "--confirm-clamped",
            app.CLAMP_TOKEN,
            "--confirm-24v-cutoff",
            app.POWER_TOKEN,
            "--confirm-workspace-clear",
            app.CLEAR_TOKEN,
        ]
    )
    assert result == 0
    assert calls == [(app.DEFAULT_PORT, 1.5, app.CLOSE_TARGET)]
    assert app.CLOSE_TARGET == (700, 700, 700, 700, 800, 900)


def test_franka_missing_confirmations_cannot_reach_motion(monkeypatch, capsys):
    app = _load("test_franka_bare_app", "run_franka_unmounted_default.py")
    calls = []
    monkeypatch.setattr(app, "run_default", lambda *args: calls.append(args))

    assert app.main([]) == 1
    assert calls == []
    assert "exact confirmations required" in capsys.readouterr().err


def test_franka_exact_confirmations_use_reviewed_default(monkeypatch):
    app = _load("test_franka_bare_app_confirmed", "run_franka_unmounted_default.py")
    calls = []
    monkeypatch.setattr(app, "run_default", lambda *args: calls.append(args))
    result = app.main(
        [
            "--confirm-flange-empty",
            app.FLANGE_TOKEN,
            "--confirm-workspace-clear",
            app.CLEAR_TOKEN,
            "--confirm-stop-ready",
            app.STOP_TOKEN,
        ]
    )
    assert result == 0
    assert calls == [("172.16.0.2",)]
    np.testing.assert_allclose(
        app.DEFAULT_Q, [0.0, 0.0, 0.0, -np.pi / 2.0, 0.0, np.pi / 2.0, 0.0]
    )
