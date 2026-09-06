from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps/run_anydex_once.py"
SNAPSHOT = ROOT / "runs/d435_20260722_124733_official.npz"


def _load_app():
    spec = importlib.util.spec_from_file_location("test_run_anydex_once_app", APP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_one_shot_dry_run_opens_no_stage_and_creates_no_output(tmp_path, monkeypatch):
    app = _load_app()
    output = tmp_path / "one-shot"
    args = app.build_parser().parse_args(
        [
            "--dry-run",
            "--snapshot", str(SNAPSHOT),
            "--output-dir", str(output),
        ]
    )

    monkeypatch.setattr(
        app,
        "_run_stage",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run opened a child stage")
        ),
    )

    assert app.run_once(args) == output.resolve()
    assert not output.exists()


def test_lift_mode_rejects_missing_payload_profile_before_confirmation(
    tmp_path, monkeypatch
):
    app = _load_app()
    args = app.build_parser().parse_args(
        [
            "--execution-mode", "lift",
            "--snapshot", str(SNAPSHOT),
            "--output-dir", str(tmp_path / "lift"),
        ]
    )
    monkeypatch.setattr(
        app,
        "_confirm",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("lift blocker was evaluated after confirmation")
        ),
    )

    with pytest.raises(ValueError, match="D435 geometry cannot infer mass"):
        app.run_once(args)


def test_one_shot_command_builder_keeps_internal_session_out_of_user_interface(tmp_path):
    app = _load_app()
    args = app.build_parser().parse_args([])
    output = tmp_path / "run"
    commands = app._commands(
        args,
        output=output,
        snapshot=output / "official.npz",
        raw=output / "raw.npz",
        selection_plan=output / "selection.json",
        selected_index=32,
    )

    assert "prepare" in commands and "execute" in commands
    assert str(output / "execution/installed_air_grasp_session.json") in commands[
        "execute"
    ]
    assert "session" not in {action.dest for action in app.build_parser()._actions}


def test_host_preflight_rejects_missing_rh56_before_motion():
    app = _load_app()
    args = app.build_parser().parse_args(["--device", "cpu"])

    with pytest.raises(RuntimeError, match="RH56 serial port is missing"):
        app._host_preflight(
            {"inspire": {"port": "/dev/definitely-missing-rh56"}}, args
        )


def test_host_preflight_reports_broken_nvidia_driver(monkeypatch):
    app = _load_app()
    args = app.build_parser().parse_args(["--device", "cuda:0"])
    monkeypatch.setattr(app.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="NVIDIA-SMI has failed because it couldn't communicate",
        )

    with pytest.raises(RuntimeError, match="NVIDIA driver is unavailable"):
        app._host_preflight(
            {"inspire": {"port": "/dev/null"}}, args, runner=runner
        )
    assert calls == [["/usr/bin/nvidia-smi", "-L"]]
