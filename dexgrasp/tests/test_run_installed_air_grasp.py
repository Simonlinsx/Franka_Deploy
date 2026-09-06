from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace
import sys
import uuid

import pytest

from anydex_pipeline.rh56_commissioning import (
    build_stage1_prerequisite_binding,
    build_snapshot_candidate_binding,
    seal_evidence,
    verify_evidence,
)
from anydex_pipeline.telemetry_session_manifest import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"
RUN_UUID = uuid.UUID("12345678-1234-4678-9234-567812345678")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "apps/run_installed_air_grasp.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_commissioning_fixture_module(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "tests/test_rh56_commissioning.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _identity_binding(name: str, path: Path):
    source = path.resolve()
    metadata = source.stat()
    return {
        "name": name,
        "path": str(source),
        "sha256": sha256_file(source),
        "identity": {
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "size": metadata.st_size,
            "mode": metadata.st_mode & 0o7777,
            "mtime_ns": metadata.st_mtime_ns,
            "ctime_ns": metadata.st_ctime_ns,
        },
    }


def _output_binding(name: str, path: Path):
    source = path.resolve()
    metadata = source.stat()
    return {
        "name": name,
        "path": str(source),
        "exists": True,
        "safe_regular_file": True,
        "sha256": sha256_file(source),
        "read_only": not bool(source.stat().st_mode & 0o222),
        "identity": {
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "size": metadata.st_size,
            "mode": metadata.st_mode & 0o7777,
            "mtime_ns": metadata.st_mtime_ns,
            "ctime_ns": metadata.st_ctime_ns,
        },
    }


def _write_sealed_receipt(app, path: Path, unsigned):
    payload = copy.deepcopy(unsigned)
    payload["integrity"] = {
        "algorithm": app.COMMISSION_RECEIPT_INTEGRITY_ALGORITHM,
        "payload_sha256": app.json_sha256(payload),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o444)
    return payload


def _current_commission_source_paths(app, assets):
    return {
        "commission_cli": (ROOT / "apps/commission_installed_rh56.py").resolve(),
        "commission_evidence_module": (
            ROOT / "src/anydex_pipeline/rh56_commissioning.py"
        ).resolve(),
        "rh56_hand_path": (ROOT / "src/anydex_pipeline/rh56_hand_path.py").resolve(),
        "rh56_reset_open": (
            ROOT / "src/anydex_pipeline/rh56_reset_open.py"
        ).resolve(),
        "rh56_sequence_driver": (
            ROOT / "src/anydex_pipeline/inspire_sequence_driver.py"
        ).resolve(),
        "rh56_register_api": (ROOT.parent / "examples/inspire_rh56_test.py").resolve(),
        "franka_sequence_driver": (
            ROOT / "src/anydex_pipeline/franka_sequence_driver.py"
        ).resolve(),
        "control_config_module": (
            ROOT / "src/anydex_pipeline/control_config.py"
        ).resolve(),
        "adapter_mesh": Path(assets.mesh_path).resolve(),
        "adapter_provenance": Path(assets.provenance_path).resolve(),
        "actuator_to_joint_xlsx": (
            ROOT
            / "third_party/AnyDexGrasp/generate_mesh_and_pointcloud/inspire_urdf"
            / "inspire_hand_routine_to_angle-use.xlsx"
        ).resolve(),
        "driver_to_angle_xls": (
            ROOT
            / "third_party/AnyDexGrasp/generate_mesh_and_pointcloud/inspire_urdf"
            / "driver_routine_to_angle.xls"
        ).resolve(),
        "actuator_to_urdf_generator": (
            ROOT
            / "third_party/AnyDexGrasp/generate_mesh_and_pointcloud"
            / "recover_inspire_hand_to_stl.py"
        ).resolve(),
    }


def _valid_commission_receipt(app, tmp_path: Path):
    fixture = _load_commissioning_fixture_module(
        "air_grasp_receipt_commission_fixture_{}".format(tmp_path.name)
    )
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["tool"]["adapter_asset"] = str(
        (ROOT / "assets/adapter/V7_FR3_RH56_M3_CAPTIVE_NUT_ROT45.stl").resolve()
    )
    config["tool"]["adapter_provenance"] = str(
        (
            ROOT
            / "assets/adapter/V7_FR3_RH56_M3_CAPTIVE_NUT_ROT45.provenance.json"
        ).resolve()
    )
    config_path = tmp_path / "receipt-base.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    snapshot_path = tmp_path / "receipt-official.npz"
    fixture._official_snapshot(snapshot_path, config, target_q6=850, score=0.9)
    snapshot_binding = build_snapshot_candidate_binding(
        snapshot_path,
        config=config,
        candidate_index=0,
    )
    evidence, stage2_path, _config, _config_path, _source = fixture._passing_evidence(
        tmp_path,
        target_q6=850,
        coupled=True,
        config_path=config_path,
        evidence_name="receipt-stage2.json",
        snapshot_binding=snapshot_binding,
    )
    assets = app.verify_adapter_assets(config, config_path)
    commission_source_paths = _current_commission_source_paths(app, assets)
    commission_sources = [
        {"name": name, "path": str(path), "sha256": sha256_file(path)}
        for name, path in commission_source_paths.items()
    ]
    stage1_path = Path(evidence["stage1_prerequisite"]["path"])
    stage1_document = json.loads(stage1_path.read_text(encoding="utf-8"))
    stage1_document["source_bindings"] = copy.deepcopy(commission_sources)
    stage1_document = seal_evidence(
        {key: value for key, value in stage1_document.items() if key != "integrity"}
    )
    stage1_path.chmod(0o644)
    stage1_path.write_text(
        json.dumps(stage1_document, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    stage1_path.chmod(0o444)
    evidence["source_bindings"] = copy.deepcopy(commission_sources)
    evidence["stage1_prerequisite"] = build_stage1_prerequisite_binding(
        stage1_path,
        expected_config=config,
        expected_config_path=config_path,
    )
    evidence = seal_evidence(
        {key: value for key, value in evidence.items() if key != "integrity"}
    )
    stage2_path.chmod(0o644)
    stage2_path.write_text(
        json.dumps(evidence, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    stage2_path.chmod(0o444)
    verification = verify_evidence(
        stage2_path,
        config_path=config_path,
        require_coupled=True,
    )
    assert verification.passed, verification.blockers
    applied = copy.deepcopy(config)
    applied["inspire"]["thumb_rotate_validated_realtime_range"] = list(
        verification.proposal["inspire.thumb_rotate_validated_realtime_range"]
    )
    applied["inspire"]["six_axis_coupled_closure_commissioned"] = True
    applied["inspire"]["commissioned_air_closure_targets"] = copy.deepcopy(
        verification.proposal["inspire.commissioned_air_closure_targets"]
    )
    profile_path = tmp_path / "receipt-derived.json"
    profile_path.write_text(json.dumps(applied, indent=2), encoding="utf-8")
    profile_path.chmod(0o444)
    stage1 = evidence["stage1_prerequisite"]
    historical_stage1_path = tmp_path / "receipt-historical-stage1.json"
    historical_stage1_path.write_bytes(stage1_path.read_bytes())
    historical_stage1_path.chmod(0o444)

    historical_sources = [
        copy.deepcopy(item)
        for item in evidence["source_bindings"]
        if item["name"] != "rh56_reset_open"
    ]
    assert {item["name"] for item in historical_sources} == set(
        app.COMMISSION_RECEIPT_HISTORICAL_SOURCE_NAMES
    )
    historical_stage1 = copy.deepcopy(stage1)
    historical_stage1["path"] = str(historical_stage1_path.resolve())
    historical_stage1["file_sha256"] = sha256_file(historical_stage1_path)
    failed_unsigned = {
        "run_id": "historical-failed-stage2-run",
        "source_bindings": historical_sources,
        "stage1_prerequisite": historical_stage1,
    }
    failed_payload = copy.deepcopy(failed_unsigned)
    failed_payload["integrity"] = {
        "algorithm": "sha256-canonical-json-without-integrity",
        "payload_sha256": app.json_sha256(failed_unsigned),
    }
    failed_path = tmp_path / "receipt-failed-stage2.json"
    failed_path.write_text(
        json.dumps(failed_payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    failed_path.chmod(0o444)

    recovery_source_paths = app._expected_recovery_runtime_source_paths(assets)
    recovery_sources = [
        {
            "name": name,
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for name, path in recovery_source_paths.items()
    ]
    candidate = evidence["snapshot_candidate"]["candidate"]
    targets = list(candidate["hand_targets"])
    recovery_plan = {
        "path": str(failed_path.resolve()),
        "file_sha256": sha256_file(failed_path),
        "payload_sha256": failed_payload["integrity"]["payload_sha256"],
        "run_id": failed_payload["run_id"],
        "target_q6": targets[5],
        "step_units": 25,
        "coupled_targets": targets,
        "recovery_mode": "interrupted_stage2_q6_return_v1",
        "original_speeds": [1000] * 6,
        "original_forces": [500] * 6,
        "allowed_recovery_routes": [
            "sealed_stage2_q6_return_v1",
            "profile_near_open_reset_v1",
        ],
        "profile_near_open_q6_range": [900, 1000],
        "q6_open_min_angle": 975,
        "historical_source_provenance": {
            "policy": "offline-test-historical-provenance",
            "source_bindings_sha256": app.json_sha256(historical_sources),
            "execution_authority": False,
            "live_migrations": [],
        },
    }
    open_snapshot = {
        "angle_targets": [-1] * 6,
        "angles": [1000, 1000, 1000, 1000, 1000, 975],
        "currents": [0] * 6,
        "errors": [0] * 6,
        "statuses": [2] * 6,
        "speeds": [1000] * 6,
        "force_limits": [500] * 6,
    }
    recovery_unsigned = {
        "schema_version": 1,
        "kind": app.RECOVERY_EVIDENCE_KIND,
        "motion_authorized": False,
        "commissioning_unlock_claimed": False,
        "failed_commissioning_binding": recovery_plan,
        "source_bindings": recovery_sources,
        "request": {
            "allowed_recovery_routes": recovery_plan["allowed_recovery_routes"],
            "selected_recovery_route": "profile_near_open_reset_v1",
            "route_dispatch_initial_q6": 968,
            "profile_near_open_q6_range": [900, 1000],
            "q6_open_min_angle": 975,
        },
        "result": {
            "status": "pass",
            "adopted_disabled_verified": True,
            "recovered_open_verified": True,
            "disabled_verified": True,
            "operation_error": None,
            "stop_error": None,
            "cleanup_error": None,
            "recovery_route": "profile_near_open_reset_v1",
            "recovery_route_initial_q6": 968,
            "evidence_bound_path_executed": False,
        },
        "route_dispatch": {
            "phase": "boundary_state_snapshot",
            "angle_targets": [-1] * 6,
            "angles": [1000, 1000, 1000, 1000, 1000, 968],
            "currents": [0] * 6,
            "errors": [0] * 6,
            "statuses": [2] * 6,
            "temperatures": [30] * 6,
        },
        "telemetry": [
            {
                "phase": "boundary_state_snapshot",
                "angle_targets": [-1] * 6,
                "angles": [1000, 1000, 1000, 1000, 1000, 968],
                "currents": [0] * 6,
                "errors": [0] * 6,
                "statuses": [2] * 6,
                "temperatures": [30] * 6,
            },
            {
                "phase": "reset_open_preflight",
                "angle_targets": [-1] * 6,
                "angles": [1000, 1000, 1000, 1000, 1000, 968],
                "currents": [0] * 6,
                "errors": [0] * 6,
                "statuses": [2] * 6,
                "temperatures": [30] * 6,
            },
        ],
        "executed_q6_return_waypoints": [],
        "executed_reset_q6_waypoints": [918, 1000],
        "final": {
            "angle_targets": [-1] * 6,
            "angles": [1000, 1000, 1000, 1000, 1000, 975],
            "currents": [0] * 6,
            "errors": [0] * 6,
            "statuses": [2] * 6,
            "original_settings_restored": True,
            "settings_restore_target": "reset_defaults",
            "snapshot_after_disable": open_snapshot,
        },
    }
    recovery_path = tmp_path / "receipt-recovery.json"
    recovery_path.write_text(
        json.dumps(seal_evidence(recovery_unsigned), sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    recovery_path.chmod(0o444)

    materialized_path = tmp_path / "receipt-materialized.json"
    materialized_path.write_bytes(profile_path.read_bytes())
    materialized_path.chmod(0o444)

    source_paths = app._expected_commission_workflow_sources()
    authority_paths = {
        "failed_stage2": failed_path,
        "base_config": config_path,
        "official_snapshot": snapshot_path,
        "historical_stage1_evidence": historical_stage1_path,
        **source_paths,
        **recovery_source_paths,
    }
    inputs = [_identity_binding(name, authority_paths[name]) for name in sorted(authority_paths)]
    workflow_sources = [
        _identity_binding(name, source_paths[name]) for name in sorted(source_paths)
    ]
    candidate = evidence["snapshot_candidate"]["candidate"]
    targets = list(candidate["hand_targets"])
    stages = [
        {
            "name": name,
            "argv": ["offline", name],
            "started_at_utc": "2026-07-22T00:00:00Z",
            "completed_at_utc": "2026-07-22T00:00:01Z",
            "returncode": 0,
            "interrupted": False,
            "error": None,
        }
        for name in app.COMMISSION_RECEIPT_STAGE_NAMES
    ]
    unsigned = {
        "schema_version": app.COMMISSION_RECEIPT_SCHEMA_VERSION,
        "kind": app.COMMISSION_RECEIPT_KIND,
        "workflow_id": "12345678-1234-4678-9234-567812345678",
        "started_at_utc": "2026-07-22T00:00:00Z",
        "completed_at_utc": "2026-07-22T00:01:00Z",
        "motion_authorized": False,
        "receipt_is_motion_authority": False,
        "inputs": inputs,
        "historical_commissioning_sources": historical_sources,
        "recovery_runtime_sources": [
            _identity_binding(name, recovery_source_paths[name])
            for name in sorted(recovery_source_paths)
        ],
        "selection": {
            "source": "failed_stage2_snapshot_candidate_binding",
            "candidate_index": 0,
            "official_score": candidate["official_score"],
            "hand_targets": targets,
            "q6_step": 25,
            "manual_target_or_candidate_override_allowed": False,
        },
        "recovery_plan_binding": recovery_plan,
        "recovery_execution": {
            "route": "profile_near_open_reset_v1",
            "evidence_bound_path_executed": False,
            "allowed_recovery_routes": recovery_plan["allowed_recovery_routes"],
            "route_dispatch_initial_q6": 968,
            "reset_preflight_initial_q6": 968,
            "executed_q6_return_waypoints": [],
            "executed_reset_q6_waypoints": [918, 1000],
        },
        "operator_confirmations": {
            "installed_on_fr3": True,
            "24v_cutoff_ready": True,
            "franka_stop_ready": True,
            "workspace_clear": True,
            "no_contact_PLA_scope": True,
            "interrupted_recovery": True,
            "wide_q6": True,
            "coupled_closure": True,
            "exact_air_target": True,
        },
        "stages": stages,
        "outputs": [
            _output_binding("recovery_evidence", recovery_path),
            _output_binding("stage1_evidence", stage1_path),
            _output_binding("stage2_evidence", stage2_path),
            _output_binding("materialized_staging_profile", materialized_path),
            _output_binding("derived_profile", profile_path),
        ],
        "workflow_sources": workflow_sources,
        "result": {
            "status": "pass",
            "failed_stage": None,
            "error": None,
            "final_profile_verified": True,
            "profile_path": str(profile_path.resolve()),
        },
    }
    receipt_path = tmp_path / "workflow_receipt.json"
    _write_sealed_receipt(app, receipt_path, unsigned)
    return receipt_path, unsigned, {
        "config": profile_path,
        "snapshot": snapshot_path,
        "stage2": stage2_path,
        "recovery": recovery_path,
        "candidate": candidate,
    }


def _prepare_argv(app, tmp_path: Path):
    config = tmp_path / "control.json"
    config.write_bytes(CONFIG.read_bytes())
    camera = tmp_path / "camera.yaml"
    camera.write_text("camera: offline-test\n", encoding="utf-8")
    snapshot = tmp_path / "snapshot.npz"
    snapshot.write_bytes(b"offline snapshot fixture")
    anydex = tmp_path / "AnyDexGrasp"
    anydex.mkdir()
    dynamic = tmp_path / "dynamic-python"
    dynamic.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    dynamic.chmod(0o755)
    reader = tmp_path / "reader-build/python"
    producer = tmp_path / "producer-build/python"
    output = tmp_path / "session-output"
    values = [
        "prepare",
        "--config", str(config),
        "--camera-config", str(camera),
        "--snapshot", str(snapshot),
        "--selected-index", "0",
        "--output-dir", str(output),
        "--anydex-root", str(anydex),
        "--dynamic-python", str(dynamic),
        "--reader-python-dir", str(reader),
        "--producer-python-dir", str(producer),
        "--confirm-installed", app.RH56_INSTALLED_TOKEN,
        "--confirm-24v-cutoff", app.RH56_POWER_TOKEN,
        "--confirm-franka-stop", app.FRANKA_STOP_TOKEN,
        "--confirm-rh56-workspace-clear", app.RH56_CLEAR_TOKEN,
        "--confirm-no-contact", app.NO_CONTACT_TOKEN,
        "--confirm-rh56-reset-open", app.RH56_RESET_TOKEN,
        "--confirm-hand-open", app.HAND_OPEN_TOKEN,
        "--confirm-default-sweep-clear", app.DEFAULT_SWEEP_TOKEN,
        "--confirm-pla-low-speed", app.PLA_LOW_SPEED_TOKEN,
        "--confirm-stationary-q", app.STATIONARY_Q_TOKEN,
    ]
    return values, {
        "config": config,
        "camera": camera,
        "snapshot": snapshot,
        "dynamic": dynamic,
        "reader": reader,
        "producer": producer,
        "output": output,
    }


def _run_argv(app, session: Path, dynamic: Path):
    return [
        "run",
        "--session", str(session),
        "--dynamic-python", str(dynamic),
        "--viewer-ready-timeout-seconds", "1",
        "--confirm-workspace-clear", app.EXECUTOR_WORKSPACE_TOKEN,
        "--confirm-immediate-stop", app.EXECUTOR_STOP_TOKEN,
        "--confirm-air-grasp", app.AIR_GRASP_TOKEN,
        "--confirm-q6-preshape", app.Q6_PRESHAPE_TOKEN,
        "--confirm-installed-collision-model", app.COLLISION_MODEL_TOKEN,
    ]


def _fake_audit_loader(_path, **_kwargs):
    return {
        "mode": "air_grasp",
        "decision": {"passed": True, "motion_authorized": False},
        "bindings": {"scene": {"captured_at_s": 100.0}},
        "policies": {"max_scene_age_s": 120.0},
    }


def _manifest_loader_factory(run_uuid, paths):
    def load(path, **_kwargs):
        manifest_path = Path(path).resolve()
        identity = SimpleNamespace(
            run_uuid=str(run_uuid),
            source_snapshot_sha256=sha256_file(paths["snapshot"]),
            control_config_sha256=sha256_file(paths["config"]),
            producer_build_sha256=sha256_file(
                paths["producer"] / "_anydex_franka_telemetry.test.so"
            ),
        )
        payload = {
            "execution": {"command": "air-grasp", "selected_index": 0},
            "sources": {
                "audit_artifact": {
                    "file_sha256": sha256_file(
                        paths["output"]
                        / "fresh_air_audit/installed_air_audit_v2.json"
                    )
                }
            },
        }
        return SimpleNamespace(
            path=manifest_path,
            identity=identity,
            payload=payload,
            manifest_file_sha256=sha256_file(manifest_path),
        )

    return load


def _prepare_fake_session(app, tmp_path: Path):
    argv, paths = _prepare_argv(app, tmp_path)
    args = app.build_parser().parse_args(argv)
    events = []
    commands = []

    def runner(command, **_kwargs):
        command = list(command)
        commands.append(command)
        script = Path(command[0]).name
        events.append(script)
        if script == "build_continuous_telemetry.sh":
            paths["reader"].mkdir(parents=True)
            paths["producer"].mkdir(parents=True)
            (paths["reader"] / "_anydex_telemetry.test.so").write_bytes(b"reader")
            (paths["producer"] / "_anydex_franka_telemetry.test.so").write_bytes(
                b"producer"
            )
        elif script == "prepare_fresh_installed_air_audit.sh":
            audit_dir = paths["output"] / "fresh_air_audit"
            audit_dir.mkdir()
            (audit_dir / "installed_air_audit_v2.json").write_text(
                '{"offline":"audit"}\n', encoding="utf-8"
            )
        elif script == "telemetry_session_manifest.sh":
            manifest = paths["output"] / "continuous_telemetry.manifest.json"
            manifest.write_text('{"offline":"manifest"}\n', encoding="utf-8")
        return SimpleNamespace(returncode=0)

    def q_reader(config):
        events.append("fresh_read_once")
        return tuple(config["franka"]["default_q_rad"])

    manifest_loader = _manifest_loader_factory(RUN_UUID, paths)
    session = app.prepare_workflow(
        args,
        runner=runner,
        q_reader=q_reader,
        audit_loader=_fake_audit_loader,
        manifest_loader=manifest_loader,
        clock=lambda: 110.0,
        uuid_factory=lambda: RUN_UUID,
    )
    return session, paths, events, commands, manifest_loader


def test_prepare_orders_existing_clis_and_writes_one_immutable_session(tmp_path):
    app = _load("test_air_grasp_prepare_order")
    session, paths, events, commands, _manifest_loader = _prepare_fake_session(
        app, tmp_path
    )

    assert events == [
        "reset_installed_rh56_open.sh",
        "reset_franka_default.sh",
        "build_continuous_telemetry.sh",
        "fresh_read_once",
        "prepare_fresh_installed_air_audit.sh",
        "execute_control_sequence.sh",
        "telemetry_session_manifest.sh",
    ]
    assert commands[0][1] == "run"
    assert ["--confirm-reset-open", app.RH56_RESET_TOKEN] == commands[0][
        commands[0].index("--confirm-reset-open") : commands[0].index("--confirm-reset-open") + 2
    ]
    assert ["--confirm-hand-open", app.HAND_OPEN_TOKEN] == commands[1][
        commands[1].index("--confirm-hand-open") : commands[1].index("--confirm-hand-open") + 2
    ]
    dry_run = commands[4]
    assert dry_run[1] == "air-grasp"
    assert "--dry-run" in dry_run
    assert "--confirm-air-grasp" not in dry_run
    assert session == paths["output"] / app.SESSION_FILENAME
    assert session.is_file()
    assert session.stat().st_mode & 0o222 == 0
    payload = json.loads(session.read_text(encoding="utf-8"))
    assert payload["motion_authorized"] is False
    assert payload["scope"] == {
        "mode": "air_grasp",
        "contact_allowed": False,
        "lift_allowed": False,
        "loaded_grasp_allowed": False,
    }
    assert payload["capture"]["q_rad"]
    assert payload["telemetry"]["mapping_path"].startswith("/tmp/fr3_rh56_air_")


def test_prepare_stops_on_first_failed_child_before_fresh_read(tmp_path):
    app = _load("test_air_grasp_prepare_failure")
    argv, paths = _prepare_argv(app, tmp_path)
    args = app.build_parser().parse_args(argv)
    calls = []

    def runner(command, **_kwargs):
        calls.append(Path(command[0]).name)
        return SimpleNamespace(returncode=7 if len(calls) == 2 else 0)

    with pytest.raises(app.StageError, match="reset_franka") as captured:
        app.prepare_workflow(
            args,
            runner=runner,
            q_reader=lambda _config: pytest.fail("fresh q read must not run"),
            clock=lambda: 110.0,
            uuid_factory=lambda: RUN_UUID,
        )

    assert captured.value.returncode == 7
    assert calls == ["reset_installed_rh56_open.sh", "reset_franka_default.sh"]
    assert not (paths["output"] / app.SESSION_FILENAME).exists()


def test_prepare_wrong_confirmation_is_rejected_before_any_child(tmp_path):
    app = _load("test_air_grasp_prepare_token_gate")
    argv, _paths = _prepare_argv(app, tmp_path)
    argv[argv.index(app.RH56_RESET_TOKEN)] = "WRONG"
    args = app.build_parser().parse_args(argv)
    with pytest.raises(ValueError, match="RH56_RESET_OPEN"):
        app.prepare_workflow(
            args,
            runner=lambda *_args, **_kwargs: pytest.fail("child must not start"),
            q_reader=lambda _config: pytest.fail("fresh q must not be read"),
        )


def test_commission_receipt_derives_profile_snapshot_candidate_and_is_not_authority(
    tmp_path,
):
    app = _load("test_air_grasp_valid_commission_receipt")
    receipt, _unsigned, paths = _valid_commission_receipt(app, tmp_path)

    authority = app.load_commission_receipt_authority(receipt)

    assert authority.receipt_path == receipt.resolve()
    assert authority.receipt_sha256 == sha256_file(receipt)
    assert authority.config_path == paths["config"].resolve()
    assert authority.snapshot_path == paths["snapshot"].resolve()
    assert authority.selected_index == 0
    assert list(authority.hand_targets) == paths["candidate"]["hand_targets"]


def test_prepare_receipt_mode_is_mutually_exclusive_before_receipt_or_child(tmp_path):
    app = _load("test_air_grasp_receipt_no_override")
    argv, paths = _prepare_argv(app, tmp_path)
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    receipt.chmod(0o444)
    argv.extend(("--commission-receipt", str(receipt)))
    args = app.build_parser().parse_args(argv)

    with pytest.raises(ValueError, match="cannot be mixed"):
        app.prepare_workflow(
            args,
            receipt_loader=lambda _path: pytest.fail("receipt must not load"),
            runner=lambda *_args, **_kwargs: pytest.fail("child must not start"),
            q_reader=lambda _config: pytest.fail("fresh q must not be read"),
        )

    assert not paths["output"].exists()


def test_prepare_receipt_mode_records_binding_and_uses_only_derived_inputs(tmp_path):
    app = _load("test_air_grasp_receipt_prepare_binding")
    argv, paths = _prepare_argv(app, tmp_path)
    for flag in ("--config", "--snapshot", "--selected-index"):
        index = argv.index(flag)
        del argv[index : index + 2]
    receipt = tmp_path / "verified-receipt.json"
    receipt.write_text('{"offline":"receipt"}\n', encoding="utf-8")
    receipt.chmod(0o444)
    authority = app.CommissionReceiptAuthority(
        receipt_path=receipt.resolve(),
        receipt_sha256=sha256_file(receipt),
        config_path=paths["config"].resolve(),
        snapshot_path=paths["snapshot"].resolve(),
        selected_index=0,
        hand_targets=(900, 900, 900, 900, 900, 850),
        stage2_evidence_path=(tmp_path / "stage2.json").resolve(),
        derived_profile_path=paths["config"].resolve(),
    )
    argv.extend(("--commission-receipt", str(receipt)))
    args = app.build_parser().parse_args(argv)
    commands = []

    def runner(command, **_kwargs):
        command = list(command)
        commands.append(command)
        script = Path(command[0]).name
        if script == "build_continuous_telemetry.sh":
            paths["reader"].mkdir(parents=True)
            paths["producer"].mkdir(parents=True)
            (paths["reader"] / "_anydex_telemetry.test.so").write_bytes(b"reader")
            (paths["producer"] / "_anydex_franka_telemetry.test.so").write_bytes(
                b"producer"
            )
        elif script == "prepare_fresh_installed_air_audit.sh":
            audit_dir = paths["output"] / "fresh_air_audit"
            audit_dir.mkdir()
            (audit_dir / "installed_air_audit_v2.json").write_text(
                '{"offline":"audit"}\n', encoding="utf-8"
            )
        elif script == "telemetry_session_manifest.sh":
            (paths["output"] / "continuous_telemetry.manifest.json").write_text(
                '{"offline":"manifest"}\n', encoding="utf-8"
            )
        return SimpleNamespace(returncode=0)

    session = app.prepare_workflow(
        args,
        runner=runner,
        q_reader=lambda config: tuple(config["franka"]["default_q_rad"]),
        audit_loader=_fake_audit_loader,
        manifest_loader=_manifest_loader_factory(RUN_UUID, paths),
        receipt_loader=lambda value: authority
        if Path(value).resolve() == receipt.resolve()
        else pytest.fail("unexpected receipt path"),
        clock=lambda: 110.0,
        uuid_factory=lambda: RUN_UUID,
    )

    payload = json.loads(session.read_text(encoding="utf-8"))
    assert payload["commissioning"] == {
        "source": "verified_recover_and_commission_receipt",
        "receipt_is_motion_authority": False,
        "receipt": {
            "path": str(receipt.resolve()),
            "sha256": sha256_file(receipt),
        },
    }
    for command in commands:
        if "--config" in command:
            assert command[command.index("--config") + 1] == str(paths["config"].resolve())
        if "--snapshot" in command:
            assert command[command.index("--snapshot") + 1] == str(paths["snapshot"].resolve())
        if "--selected-index" in command:
            assert command[command.index("--selected-index") + 1] == "0"


def test_commission_receipt_rejects_nonpass_tamper_and_final_symlink(tmp_path):
    app = _load("test_air_grasp_receipt_fail_closed")
    receipt, unsigned, _paths = _valid_commission_receipt(app, tmp_path)

    receipt.chmod(0o644)
    with pytest.raises(ValueError, match="mode must be exactly 0444"):
        app.load_commission_receipt_authority(receipt)

    unsigned["result"]["status"] = "fail"
    receipt.unlink()
    _write_sealed_receipt(app, receipt, unsigned)
    with pytest.raises(ValueError, match="unqualified PASS"):
        app.load_commission_receipt_authority(receipt)

    alias = tmp_path / "receipt-alias.json"
    os.symlink(receipt, alias)
    with pytest.raises(ValueError, match="without following a symlink"):
        app.load_commission_receipt_authority(alias)


def test_commission_receipt_rejects_historical_sources_as_execution_authority(
    tmp_path,
):
    app = _load("test_air_grasp_receipt_historical_not_authority")
    receipt, unsigned, _paths = _valid_commission_receipt(app, tmp_path)
    unsigned["recovery_plan_binding"]["historical_source_provenance"][
        "execution_authority"
    ] = True
    receipt.unlink()
    _write_sealed_receipt(app, receipt, unsigned)

    with pytest.raises(ValueError, match="cannot be execution authority"):
        app.load_commission_receipt_authority(receipt)


def test_commission_receipt_rejects_recovery_route_claim_mismatch(tmp_path):
    app = _load("test_air_grasp_receipt_route_mismatch")
    receipt, unsigned, _paths = _valid_commission_receipt(app, tmp_path)
    unsigned["recovery_execution"]["route"] = "sealed_stage2_q6_return_v1"
    receipt.chmod(0o644)
    receipt.unlink()
    _write_sealed_receipt(app, receipt, unsigned)

    with pytest.raises(ValueError, match="recovery route differs"):
        app.load_commission_receipt_authority(receipt)


def test_commission_receipt_rejects_q6_below_profile_open_floor(tmp_path):
    app = _load("test_air_grasp_receipt_q6_floor")
    receipt, unsigned, paths = _valid_commission_receipt(app, tmp_path)
    recovery = paths["recovery"]
    document = json.loads(recovery.read_text(encoding="utf-8"))
    recovery_unsigned = {
        key: value for key, value in document.items() if key != "integrity"
    }
    recovery_unsigned["final"]["angles"][5] = 974
    recovery_unsigned["final"]["snapshot_after_disable"]["angles"][5] = 974
    recovery.chmod(0o644)
    recovery.write_text(
        json.dumps(
            seal_evidence(recovery_unsigned), sort_keys=True, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    recovery.chmod(0o444)
    for index, output in enumerate(unsigned["outputs"]):
        if output["name"] == "recovery_evidence":
            unsigned["outputs"][index] = _output_binding(
                "recovery_evidence", recovery
            )
            break
    receipt.chmod(0o644)
    receipt.unlink()
    _write_sealed_receipt(app, receipt, unsigned)

    with pytest.raises(ValueError, match="not profile-open"):
        app.load_commission_receipt_authority(receipt)


def test_commission_receipt_rejects_noncanonical_near_open_reset_path(tmp_path):
    app = _load("test_air_grasp_receipt_reset_path")
    receipt, unsigned, paths = _valid_commission_receipt(app, tmp_path)
    recovery = paths["recovery"]
    document = json.loads(recovery.read_text(encoding="utf-8"))
    recovery_unsigned = {
        key: value for key, value in document.items() if key != "integrity"
    }
    recovery_unsigned["executed_reset_q6_waypoints"] = [100, 1000]
    recovery.chmod(0o644)
    recovery.write_text(
        json.dumps(
            seal_evidence(recovery_unsigned), sort_keys=True, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    recovery.chmod(0o444)
    unsigned["recovery_execution"]["executed_reset_q6_waypoints"] = [100, 1000]
    for index, output in enumerate(unsigned["outputs"]):
        if output["name"] == "recovery_evidence":
            unsigned["outputs"][index] = _output_binding(
                "recovery_evidence", recovery
            )
            break
    receipt.chmod(0o644)
    receipt.unlink()
    _write_sealed_receipt(app, receipt, unsigned)

    with pytest.raises(ValueError, match="canonical profile path"):
        app.load_commission_receipt_authority(receipt)


def test_commission_receipt_rejects_rebound_recovery_runtime_source(tmp_path):
    app = _load("test_air_grasp_receipt_rebound_recovery_source")
    receipt, unsigned, paths = _valid_commission_receipt(app, tmp_path)
    recovery = paths["recovery"]
    document = json.loads(recovery.read_text(encoding="utf-8"))
    recovery_unsigned = {
        key: value for key, value in document.items() if key != "integrity"
    }
    original = Path(
        next(
            item["path"]
            for item in unsigned["recovery_runtime_sources"]
            if item["name"] == "rh56_reset_open"
        )
    )
    alias = tmp_path / "same-bytes-reset-open.py"
    alias.write_bytes(original.read_bytes())
    alias.chmod(original.stat().st_mode & 0o7777)
    for item in recovery_unsigned["source_bindings"]:
        if item["name"] == "rh56_reset_open":
            item["path"] = str(alias.resolve())
            item["sha256"] = sha256_file(alias)
            break
    recovery.chmod(0o644)
    recovery.write_text(
        json.dumps(
            seal_evidence(recovery_unsigned), sort_keys=True, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    recovery.chmod(0o444)
    rebound_identity = _identity_binding("rh56_reset_open", alias)
    for collection_name in ("inputs", "recovery_runtime_sources"):
        for index, item in enumerate(unsigned[collection_name]):
            if item["name"] == "rh56_reset_open":
                unsigned[collection_name][index] = rebound_identity
                break
    for index, output in enumerate(unsigned["outputs"]):
        if output["name"] == "recovery_evidence":
            unsigned["outputs"][index] = _output_binding(
                "recovery_evidence", recovery
            )
            break
    receipt.chmod(0o644)
    receipt.unlink()
    _write_sealed_receipt(app, receipt, unsigned)

    with pytest.raises(ValueError, match="runtime source path changed"):
        app.load_commission_receipt_authority(receipt)


def test_commission_receipt_rejects_static_input_leaf_symlink_even_if_resealed(tmp_path):
    app = _load("test_air_grasp_receipt_static_symlink")
    receipt, unsigned, paths = _valid_commission_receipt(app, tmp_path)
    alias = tmp_path / "official-alias.npz"
    os.symlink(paths["snapshot"], alias)
    target = next(item for item in unsigned["inputs"] if item["name"] == "official_snapshot")
    target["path"] = str(alias)
    receipt.unlink()
    _write_sealed_receipt(app, receipt, unsigned)

    with pytest.raises(ValueError, match="without following a symlink"):
        app.load_commission_receipt_authority(receipt)


def test_commission_receipt_calls_coupled_and_applied_verifiers_fail_closed(
    tmp_path, monkeypatch
):
    app = _load("test_air_grasp_receipt_coupled_applied")
    receipt, _unsigned, _paths = _valid_commission_receipt(app, tmp_path)
    calls = []

    def reject_stage2(evidence, *, config_path, require_coupled):
        calls.append(("stage2", Path(evidence), Path(config_path), require_coupled))
        return SimpleNamespace(passed=False, blockers=("forced coupled failure",))

    monkeypatch.setattr(app, "verify_evidence", reject_stage2)
    with pytest.raises(ValueError, match="forced coupled failure"):
        app.load_commission_receipt_authority(receipt)
    assert calls[0][3] is True

    monkeypatch.setattr(
        app,
        "verify_evidence",
        lambda evidence, *, config_path, require_coupled: SimpleNamespace(
            passed=True, blockers=()
        ),
    )

    def reject_applied(evidence, profile, *, require_coupled):
        calls.append(("applied", Path(evidence), Path(profile), require_coupled))
        return SimpleNamespace(passed=False, blockers=("forced applied failure",))

    monkeypatch.setattr(app, "verify_applied_config", reject_applied)
    with pytest.raises(ValueError, match="forced applied failure"):
        app.load_commission_receipt_authority(receipt)
    assert calls[-1][0] == "applied"
    assert calls[-1][3] is True


def test_prepare_rejects_nonfinite_telemetry_wait_before_any_child(tmp_path):
    app = _load("test_air_grasp_prepare_nonfinite_wait")
    argv, _paths = _prepare_argv(app, tmp_path)
    argv.extend(("--telemetry-wait-seconds", "nan"))
    args = app.build_parser().parse_args(argv)
    with pytest.raises(ValueError, match="finite and positive"):
        app.prepare_workflow(
            args,
            runner=lambda *_args, **_kwargs: pytest.fail("child must not start"),
            q_reader=lambda _config: pytest.fail("fresh q must not be read"),
        )


def test_session_integrity_and_freshness_fail_before_any_process(tmp_path):
    app = _load("test_air_grasp_session_validation")
    session, paths, _events, _commands, manifest_loader = _prepare_fake_session(
        app, tmp_path
    )

    with pytest.raises(ValueError, match="expired"):
        app.load_validated_session(
            session,
            now_s=220.0,
            audit_loader=_fake_audit_loader,
            manifest_loader=manifest_loader,
        )

    session.chmod(0o644)
    with pytest.raises(ValueError, match="mode must be exactly 0444"):
        app.load_validated_session(
            session,
            now_s=110.0,
            audit_loader=_fake_audit_loader,
            manifest_loader=manifest_loader,
        )
    payload = json.loads(session.read_text(encoding="utf-8"))
    payload["selected_index"] = 1
    session.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    session.chmod(0o444)
    with pytest.raises(ValueError, match="integrity"):
        app.load_validated_session(
            session,
            now_s=110.0,
            audit_loader=_fake_audit_loader,
            manifest_loader=manifest_loader,
        )


def test_session_and_exclusive_writer_reject_final_symlinks(tmp_path):
    app = _load("test_air_grasp_session_symlink_gate")
    session, paths, _events, _commands, manifest_loader = _prepare_fake_session(
        app, tmp_path
    )

    session_alias = tmp_path / "session-alias.json"
    os.symlink(session, session_alias)
    with pytest.raises(ValueError, match="cannot load air-grasp session"):
        app.load_validated_session(
            session_alias,
            now_s=110.0,
            audit_loader=_fake_audit_loader,
            manifest_loader=manifest_loader,
        )

    missing_target = tmp_path / "must-not-be-created.json"
    dangling_output = tmp_path / "dangling-session.json"
    os.symlink(missing_target, dangling_output)
    with pytest.raises(FileExistsError):
        app._exclusive_json_write(dangling_output, {"offline": True})
    assert not missing_target.exists()


def test_prepare_rejects_dangling_output_dir_before_any_child(tmp_path):
    app = _load("test_air_grasp_prepare_dangling_output")
    argv, paths = _prepare_argv(app, tmp_path)
    missing_target = tmp_path / "unexpected-created-output"
    os.symlink(missing_target, paths["output"])
    args = app.build_parser().parse_args(argv)

    with pytest.raises(FileExistsError, match="including a symlink"):
        app.prepare_workflow(
            args,
            runner=lambda *_args, **_kwargs: pytest.fail("child must not start"),
            q_reader=lambda _config: pytest.fail("fresh q must not be read"),
        )
    assert not missing_target.exists()


def test_session_rejects_dangling_telemetry_mapping_leaf(tmp_path):
    app = _load("test_air_grasp_mapping_symlink_gate")
    session, paths, _events, _commands, manifest_loader = _prepare_fake_session(
        app, tmp_path
    )
    session.chmod(0o644)
    payload = json.loads(session.read_text(encoding="utf-8"))
    mapping = Path("/tmp/air-grasp-test-{}.map".format(uuid.uuid4()))
    payload["telemetry"]["mapping_path"] = str(mapping)
    unsigned = dict(payload)
    del unsigned["integrity"]
    session.write_text(
        json.dumps(app._seal_session(unsigned), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    session.chmod(0o444)
    os.symlink(tmp_path / "missing-map-target", mapping)
    try:
        with pytest.raises(ValueError, match="including a symlink"):
            app.load_validated_session(
                session,
                now_s=110.0,
                audit_loader=_fake_audit_loader,
                manifest_loader=manifest_loader,
            )
    finally:
        mapping.unlink()


class _FakeProcess:
    def __init__(self, *, immediate_code=None):
        self.code = immediate_code
        self.signals = []
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.code

    def send_signal(self, value):
        self.signals.append(value)
        self.code = 130

    def wait(self, timeout=None):
        if self.code is None:
            raise AssertionError("fake wait called before a stop signal")
        return self.code

    def terminate(self):
        self.terminated = True
        self.code = -15

    def kill(self):
        self.killed = True
        self.code = -9


def test_run_starts_viewer_then_foreground_executor_and_stops_viewer(tmp_path):
    app = _load("test_air_grasp_run_order")
    session, paths, _events, _commands, manifest_loader = _prepare_fake_session(
        app, tmp_path
    )
    args = app.build_parser().parse_args(_run_argv(app, session, paths["dynamic"]))
    viewer = _FakeProcess()
    executor = _FakeProcess(immediate_code=0)
    starts = []
    ready_checks = []

    def popen(command, **kwargs):
        starts.append((list(command), kwargs))
        return viewer if len(starts) == 1 else executor

    loader = lambda path, now_s: app.load_validated_session(
        path,
        now_s=now_s,
        audit_loader=_fake_audit_loader,
        manifest_loader=manifest_loader,
    )
    result = app.run_workflow(
        args,
        popen_factory=popen,
        session_loader=loader,
        clock=lambda: 110.0,
        sleep=lambda _seconds: None,
        ready_checker=lambda path: ready_checks.append((path, len(starts))) or True,
    )

    assert result == 0
    assert Path(starts[0][0][0]).name == "run_live_pipeline_preview.sh"
    assert Path(starts[1][0][0]).name == "execute_control_sequence.sh"
    assert starts[0][1]["start_new_session"] is True
    assert starts[1][1]["start_new_session"] is True
    assert "--show-current-hand-mesh" in starts[0][0]
    assert "--ready-file" in starts[0][0]
    ready_path = Path(starts[0][0][starts[0][0].index("--ready-file") + 1])
    assert ready_path.suffix == ".viewer-ready"
    assert ready_checks == [(ready_path, 1)]
    assert "--confirm-air-grasp" in starts[1][0]
    assert app.AIR_GRASP_TOKEN in starts[1][0]
    assert viewer.signals == [signal.SIGINT]


def test_run_wrong_confirmation_is_rejected_before_session_or_viewer(tmp_path):
    app = _load("test_air_grasp_run_token_gate")
    dynamic = tmp_path / "dynamic-python"
    dynamic.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    dynamic.chmod(0o755)
    argv = _run_argv(app, tmp_path / "missing.json", dynamic)
    argv[argv.index(app.AIR_GRASP_TOKEN)] = "WRONG"
    args = app.build_parser().parse_args(argv)
    with pytest.raises(ValueError, match="AIR_GRASP_NO_CONTACT_NO_LIFT"):
        app.run_workflow(
            args,
            popen_factory=lambda *_args, **_kwargs: pytest.fail("viewer must not start"),
            session_loader=lambda *_args, **_kwargs: pytest.fail("session must not load"),
        )


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    (
        ("--minimum-freshness-seconds", "nan", "finite and nonnegative"),
        ("--cleanup-timeout-seconds", "inf", "finite and positive"),
    ),
)
def test_run_rejects_nonfinite_supervisor_bounds_before_session(
    tmp_path, flag, value, message
):
    app = _load("test_air_grasp_run_nonfinite_{}_{}".format(flag, value))
    dynamic = tmp_path / "dynamic-python"
    dynamic.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    dynamic.chmod(0o755)
    argv = _run_argv(app, tmp_path / "missing.json", dynamic)
    argv.extend((flag, value))
    args = app.build_parser().parse_args(argv)
    with pytest.raises(ValueError, match=message):
        app.run_workflow(
            args,
            popen_factory=lambda *_args, **_kwargs: pytest.fail("viewer must not start"),
            session_loader=lambda *_args, **_kwargs: pytest.fail("session must not load"),
        )


def test_ctrl_c_forwards_sigint_to_executor_cleanup_then_viewer(tmp_path):
    app = _load("test_air_grasp_ctrl_c")
    session, paths, _events, _commands, manifest_loader = _prepare_fake_session(
        app, tmp_path
    )
    args = app.build_parser().parse_args(_run_argv(app, session, paths["dynamic"]))
    viewer = _FakeProcess()
    executor = _FakeProcess()
    starts = []

    def popen(command, **kwargs):
        starts.append(list(command))
        return viewer if len(starts) == 1 else executor

    def interrupt(_seconds):
        if len(starts) == 2:
            raise KeyboardInterrupt

    loader = lambda path, now_s: app.load_validated_session(
        path,
        now_s=now_s,
        audit_loader=_fake_audit_loader,
        manifest_loader=manifest_loader,
    )
    result = app.run_workflow(
        args,
        popen_factory=popen,
        session_loader=loader,
        clock=lambda: 110.0,
        sleep=interrupt,
        ready_checker=lambda _path: True,
    )

    assert result == 130
    assert executor.signals == [signal.SIGINT]
    assert viewer.signals == [signal.SIGINT]


def test_viewer_timeout_never_starts_executor(tmp_path):
    app = _load("test_air_grasp_viewer_ready_timeout")
    session, paths, _events, _commands, manifest_loader = _prepare_fake_session(
        app, tmp_path
    )
    args = app.build_parser().parse_args(_run_argv(app, session, paths["dynamic"]))
    viewer = _FakeProcess()
    starts = []

    def popen(command, **kwargs):
        starts.append((list(command), kwargs))
        assert len(starts) == 1, "executor must not start without viewer readiness"
        return viewer

    ticks = iter((0.0, 2.0))
    loader = lambda path, now_s: app.load_validated_session(
        path,
        now_s=now_s,
        audit_loader=_fake_audit_loader,
        manifest_loader=manifest_loader,
    )
    with pytest.raises(app.WorkflowError, match="executor was not started"):
        app.run_workflow(
            args,
            popen_factory=popen,
            session_loader=loader,
            clock=lambda: 110.0,
            monotonic=lambda: next(ticks),
            sleep=lambda _seconds: None,
            ready_checker=lambda _path: False,
        )

    assert len(starts) == 1
    assert viewer.signals == [signal.SIGINT]


def test_viewer_exit_before_ready_never_starts_executor(tmp_path):
    app = _load("test_air_grasp_viewer_exit_before_ready")
    session, paths, _events, _commands, manifest_loader = _prepare_fake_session(
        app, tmp_path
    )
    args = app.build_parser().parse_args(_run_argv(app, session, paths["dynamic"]))
    viewer = _FakeProcess(immediate_code=9)
    starts = []

    def popen(command, **kwargs):
        starts.append((list(command), kwargs))
        assert len(starts) == 1, "executor must not start after viewer failure"
        return viewer

    loader = lambda path, now_s: app.load_validated_session(
        path,
        now_s=now_s,
        audit_loader=_fake_audit_loader,
        manifest_loader=manifest_loader,
    )
    with pytest.raises(app.WorkflowError, match="executor was not started"):
        app.run_workflow(
            args,
            popen_factory=popen,
            session_loader=loader,
            clock=lambda: 110.0,
            ready_checker=lambda _path: False,
        )

    assert len(starts) == 1


class _HangingExecutor(_FakeProcess):
    def send_signal(self, value):
        self.signals.append(value)

    def wait(self, timeout=None):
        raise subprocess.TimeoutExpired("executor", timeout)


def test_ctrl_c_cleanup_timeout_never_terminates_or_kills_executor(
    tmp_path, capsys
):
    app = _load("test_air_grasp_ctrl_c_stop_unconfirmed")
    session, paths, _events, _commands, manifest_loader = _prepare_fake_session(
        app, tmp_path
    )
    args = app.build_parser().parse_args(_run_argv(app, session, paths["dynamic"]))
    viewer = _FakeProcess()
    executor = _HangingExecutor()
    starts = []

    def popen(command, **kwargs):
        starts.append(list(command))
        return viewer if len(starts) == 1 else executor

    def interrupt(_seconds):
        if len(starts) == 2:
            raise KeyboardInterrupt

    loader = lambda path, now_s: app.load_validated_session(
        path,
        now_s=now_s,
        audit_loader=_fake_audit_loader,
        manifest_loader=manifest_loader,
    )
    with pytest.raises(app.StopUnconfirmedError, match="STOP UNCONFIRMED"):
        app.run_workflow(
            args,
            popen_factory=popen,
            session_loader=loader,
            clock=lambda: 110.0,
            sleep=interrupt,
            ready_checker=lambda _path: True,
        )

    assert executor.signals == [signal.SIGINT]
    assert executor.terminated is False
    assert executor.killed is False
    assert "[hardware][STOP UNCONFIRMED]" in capsys.readouterr().err


def test_viewer_failure_cleanup_timeout_never_terminates_or_kills_executor(
    tmp_path, capsys
):
    app = _load("test_air_grasp_viewer_failure_stop_unconfirmed")
    session, paths, _events, _commands, manifest_loader = _prepare_fake_session(
        app, tmp_path
    )
    args = app.build_parser().parse_args(_run_argv(app, session, paths["dynamic"]))
    viewer = _FakeProcess()
    executor = _HangingExecutor()
    starts = []

    def popen(command, **kwargs):
        starts.append(list(command))
        return viewer if len(starts) == 1 else executor

    def fail_viewer(_seconds):
        if len(starts) == 2:
            viewer.code = 7

    loader = lambda path, now_s: app.load_validated_session(
        path,
        now_s=now_s,
        audit_loader=_fake_audit_loader,
        manifest_loader=manifest_loader,
    )
    with pytest.raises(app.StopUnconfirmedError, match="STOP UNCONFIRMED"):
        app.run_workflow(
            args,
            popen_factory=popen,
            session_loader=loader,
            clock=lambda: 110.0,
            sleep=fail_viewer,
            ready_checker=lambda _path: True,
        )

    assert executor.signals == [signal.SIGINT]
    assert executor.terminated is False
    assert executor.killed is False
    assert "[hardware][STOP UNCONFIRMED]" in capsys.readouterr().err


def test_import_is_hardware_free(tmp_path):
    script = r'''
import builtins
import importlib.util
import pathlib
import sys
original = builtins.__import__
forbidden = {"pylibfranka", "serial", "pyrealsense2"}
def guarded(name, *args, **kwargs):
    if name.split(".", 1)[0] in forbidden:
        raise RuntimeError("forbidden hardware import: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("air_grasp_import_guard", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
print("offline-import-ok")
'''
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = __import__("subprocess").run(
        [sys.executable, "-c", script, str(ROOT / "apps/run_installed_air_grasp.py")],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "offline-import-ok"
