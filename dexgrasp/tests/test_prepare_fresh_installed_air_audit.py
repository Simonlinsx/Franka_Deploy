from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from anydex_pipeline.air_audit_readiness import (
    validate_air_candidate_commissioning,
)


ROOT = Path(__file__).resolve().parents[1]
PREPARE_APP = ROOT / "apps/prepare_fresh_installed_air_audit.py"
FILTER_APP = ROOT / "apps/filter_installed_scene.py"
SUPERVISED_CONFIG = ROOT / "configs/fr3_rh56_v7_sim2real_supervised.json"
SNAPSHOT = ROOT / "runs/d435_current_pink_cylinder_sam2_official_dedup16_20260718.npz"
XLRD_SITE = Path(
    "/home/qiaoguanren/anaconda3/pkgs/xlrd-2.0.1-pyhd3eb1b0_0/site-packages"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_commissioning_check_requires_integer_exact_six_axis_evidence():
    targets = [0, 358, 799, 911, 922, 646]
    config = {
        "inspire": {
            "six_axis_coupled_closure_commissioned": True,
            "thumb_rotate_validated_realtime_range": [646, 1000],
            "commissioned_air_closure_targets": [targets],
        }
    }
    assert validate_air_candidate_commissioning(config, targets) == tuple(targets)
    with pytest.raises(ValueError, match="integer registers"):
        validate_air_candidate_commissioning(config, [0, 358, 799, 911, 922, 646.5])
    config["inspire"]["commissioned_air_closure_targets"] = []
    with pytest.raises(ValueError, match="no exact commissioned"):
        validate_air_candidate_commissioning(config, targets)


def test_sim2real_supervised_ranges_accept_q6_450_without_exact_vector():
    config = json.loads(SUPERVISED_CONFIG.read_text(encoding="utf-8"))
    candidate = [798, 798, 798, 798, 978, 450]

    assert candidate not in config["inspire"]["commissioned_air_closure_targets"]
    assert validate_air_candidate_commissioning(config, candidate) == tuple(candidate)

    with pytest.raises(ValueError, match="outside validated range"):
        validate_air_candidate_commissioning(
            config, [798, 798, 798, 798, 978, 415]
        )
    with pytest.raises(ValueError, match="integer registers"):
        validate_air_candidate_commissioning(
            config, [1001, 798, 798, 798, 978, 450]
        )


def test_prepare_parser_requires_explicit_stationary_q_token_and_new_output_dir():
    app = _load(PREPARE_APP, "test_prepare_fresh_air_parser")
    parser = app.build_parser()
    destinations = {action.dest: action for action in parser._actions}
    assert destinations["capture_q_rad"].required is True
    assert destinations["confirm_stationary_q"].required is True
    assert destinations["output_dir"].required is True
    assert "overwrite" not in destinations
    assert destinations["dry_run"].default is False
    assert app.STATIONARY_Q_TOKEN == "CURRENT_Q_READ_ONLY_AND_STATIONARY"


@pytest.mark.skipif(not SNAPSHOT.is_file(), reason="official candidate snapshot unavailable")
def test_dry_run_never_invokes_a_child_or_creates_output(monkeypatch, tmp_path):
    app = _load(PREPARE_APP, "test_prepare_fresh_air_dry_run")
    output = tmp_path / "dry-run"
    args = app.build_parser().parse_args(
        [
            "--snapshot", str(SNAPSHOT),
            "--capture-q-rad",
            "-0.114031", "-0.1182116", "0.0736167", "-1.7442204",
            "0.042868", "1.6763846", "0.8131912",
            "--confirm-stationary-q", app.STATIONARY_Q_TOKEN,
            "--output-dir", str(output),
            "--dry-run",
        ]
    )
    context = dict(app._preflight(args))
    context["commissioning_blocker"] = "not commissioned"
    monkeypatch.setattr(app, "_preflight", lambda namespace: context)

    def forbidden_runner(*args, **kwargs):
        raise AssertionError("dry-run invoked a child process")

    assert app.prepare(args, runner=forbidden_runner) == 3
    assert not output.exists()


@pytest.mark.skipif(not SNAPSHOT.is_file(), reason="official candidate snapshot unavailable")
def test_locked_commissioning_still_runs_only_plan_capture_filter(monkeypatch, tmp_path):
    app = _load(PREPARE_APP, "test_prepare_fresh_air_locked")
    args = app.build_parser().parse_args(
        [
            "--snapshot", str(SNAPSHOT),
            "--capture-q-rad",
            "-0.114031", "-0.1182116", "0.0736167", "-1.7442204",
            "0.042868", "1.6763846", "0.8131912",
            "--confirm-stationary-q", app.STATIONARY_Q_TOKEN,
            "--output-dir", str(tmp_path / "fresh-run"),
        ]
    )
    original_preflight = app._preflight

    def locked_preflight(namespace):
        context = dict(original_preflight(namespace))
        context["commissioning_blocker"] = "test q6 target is not commissioned"
        return context

    monkeypatch.setattr(app, "_preflight", locked_preflight)
    calls = []

    def fake_runner(command, **kwargs):
        calls.append((Path(command[0]).name, tuple(command), kwargs))
        output = Path(command[command.index("--output") + 1])
        output.write_bytes(b"test-output")
        if Path(command[0]).name == "filter_installed_scene.sh":
            evidence = Path(command[command.index("--evidence") + 1])
            evidence.write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    assert app.prepare(args, runner=fake_runner) == 3
    assert [item[0] for item in calls] == [
        "plan_installed_air_candidate.sh",
        "capture_live_scene.sh",
        "filter_installed_scene.sh",
    ]
    assert all("execute_control_sequence" not in " ".join(item[1]) for item in calls)
    receipt_path = tmp_path / "fresh-run/preparation_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "audit_locked"
    assert receipt["completed_steps"] == ["plan", "capture", "filter"]
    assert receipt["robot_or_hand_transport_opened"] is False
    assert receipt["motion_authorized"] is False
    assert receipt["point_cloud_roles"]["old_grasp_pose_icp_adjustment_allowed"] is False


@pytest.mark.skipif(not SNAPSHOT.is_file(), reason="official candidate snapshot unavailable")
def test_scene_that_expires_during_audit_is_not_reported_as_evidence_pass(
    monkeypatch, tmp_path
):
    app = _load(PREPARE_APP, "test_prepare_fresh_air_expired")
    args = app.build_parser().parse_args(
        [
            "--snapshot", str(SNAPSHOT),
            "--capture-q-rad",
            "-0.114031", "-0.1182116", "0.0736167", "-1.7442204",
            "0.042868", "1.6763846", "0.8131912",
            "--confirm-stationary-q", app.STATIONARY_Q_TOKEN,
            "--output-dir", str(tmp_path / "expired-run"),
        ]
    )
    original_preflight = app._preflight

    def ready_preflight(namespace):
        context = dict(original_preflight(namespace))
        context["commissioning_blocker"] = ""
        return context

    monkeypatch.setattr(app, "_preflight", ready_preflight)

    calls = []

    def fake_runner(command, **kwargs):
        calls.append(tuple(command))
        output = Path(command[command.index("--output") + 1])
        output.write_bytes(b"test-output")
        if Path(command[0]).name == "filter_installed_scene.sh":
            Path(command[command.index("--evidence") + 1]).write_text(
                "{}", encoding="utf-8"
            )
        return subprocess.CompletedProcess(command, 0)

    audit = {
        "decision": {"passed": True, "reasons": []},
        "bindings": {"scene": {"captured_at_s": 1.0}},
        "policies": {"max_scene_age_s": 2.0},
    }
    monkeypatch.setattr(app, "load_installed_tool_audit", lambda *a, **k: audit)
    monkeypatch.setattr(app.time, "time", lambda: 10.0)
    assert app.prepare(args, runner=fake_runner) == 3
    assert [Path(item[0]).name for item in calls] == [
        "plan_installed_air_candidate.sh",
        "capture_live_scene.sh",
        "filter_installed_scene.sh",
        "generate_installed_air_audit.sh",
        "capture_live_scene.sh",
        "filter_installed_scene.sh",
        "generate_installed_air_audit.sh",
    ]
    assert "--build-static-cache" in calls[3]
    assert "--static-cache" in calls[6]
    assert calls[1][calls[1].index("--output") + 1].endswith(
        "bootstrap_live_scene.npz"
    )
    assert calls[4][calls[4].index("--output") + 1].endswith(
        "live_scene.npz"
    )
    receipt = json.loads(
        (tmp_path / "expired-run/preparation_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["status"] == "audit_locked"
    assert "expired during offline audit" in receipt["blockers"][0]


def _live_scene(path: Path, config_path: Path, calibration_path: Path, **overrides):
    points = np.asarray([[0.1, 0.2, 0.3], [0.2, 0.2, 0.3]], dtype=np.float32)
    values = {
        "artifact_type": np.asarray("dexgrasp_live_scene"),
        "schema_name": np.asarray("dexgrasp_live_scene"),
        "schema_version": np.asarray(1),
        "scene_points": points,
        "scene_colors": np.full_like(points, 0.5),
        "reference_frame": np.asarray("robot_base"),
        "T_reference_camera": np.eye(4),
        "calibration_id": np.asarray("calibration-id"),
        "calibration_source": np.asarray(str(calibration_path.resolve())),
        "calibration_sha256": np.asarray(_sha256(calibration_path)),
        "camera_serial": np.asarray("camera-serial"),
        "camera_name": np.asarray("Intel RealSense D435"),
        "capture_q_rad": np.zeros(7),
        "capture_q_source": np.asarray("cli_asserted"),
        "frame_ids": np.asarray([10]),
        "frame_timestamps_s": np.asarray([1.0]),
        "frame_count": np.asarray(1),
        "capture_started_at_unix_s": np.asarray(100.0),
        "capture_completed_at_unix_s": np.asarray(101.0),
        "captured_at_unix_s": np.asarray(101.0),
        "points_per_frame": np.asarray([2]),
        "config_path": np.asarray(str(config_path.resolve())),
        "config_sha256": np.asarray(_sha256(config_path)),
    }
    values.update(overrides)
    np.savez_compressed(path, **values)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_filter_accepts_only_capture_app_provenance(tmp_path):
    app = _load(FILTER_APP, "test_filter_live_scene_provenance")
    config = tmp_path / "camera.yaml"
    calibration = tmp_path / "calibration.yaml"
    config.write_text("camera: fixed\n", encoding="utf-8")
    calibration.write_text("transform: fixed\n", encoding="utf-8")
    valid = tmp_path / "live.npz"
    _live_scene(valid, config, calibration)
    _, points, colors, q, metadata = app._load_live_scene(valid)
    assert points.shape == colors.shape == (2, 3)
    assert q.shape == (7,)
    assert metadata["capture_q_source"] == "cli_asserted"

    wrong_source = tmp_path / "wrong-source.npz"
    _live_scene(
        wrong_source,
        config,
        calibration,
        capture_q_source=np.asarray("copied_from_old_log"),
    )
    with pytest.raises(ValueError, match="capture_q_source"):
        app._load_live_scene(wrong_source)

    config.write_text("camera: changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="camera config provenance hash differs"):
        app._load_live_scene(valid)


@pytest.mark.skipif(
    not (XLRD_SITE / "xlrd/__init__.py").is_file(),
    reason="reviewed isolated xlrd cache is unavailable",
)
def test_xlrd_shell_resolver_adds_only_selected_pure_python_site():
    helper = ROOT / "scripts/lib/resolve_xlrd_site.sh"
    env = dict(os.environ)
    env["PYTHONPATH"] = "{}:/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages".format(
        ROOT / "src"
    )
    env["DEXGRASP_XLRD_SITE"] = str(XLRD_SITE)
    code = (
        "import pathlib,xlrd; "
        "from anydex_pipeline.rh56_actuator_mapping import "
        "DRIVER_WORKBOOK_SHA256,OfficialRH56ActuatorMapper; "
        "m=OfficialRH56ActuatorMapper.from_anydex_root({!r}); "
        "print(pathlib.Path(xlrd.__file__).resolve()); "
        "print(m.provenance()['driver_workbook_sha256']); "
        "assert m.provenance()['driver_workbook_sha256']==DRIVER_WORKBOOK_SHA256"
    ).format(str(ROOT / "third_party/AnyDexGrasp"))
    result = subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; source "$1"; resolve_dexgrasp_xlrd_site; '
            '/usr/bin/python3 -c "$2"',
            "bash",
            str(helper),
            code,
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines[0].startswith(str(XLRD_SITE / "xlrd"))
    assert lines[1] == "23ca934b1092ce1a42e46f7bd7edc1ac0e3cacb98efe397ab9d9db1c938e7bdb"
    assert str(XLRD_SITE) in result.stderr
