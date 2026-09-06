from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from anydex_pipeline.snapshot import (
    GraspCandidates,
    VisualizationSnapshot,
    save_snapshot_npz,
)
from anydex_pipeline.telemetry_session_manifest import (
    ARTIFACT_TYPE,
    build_telemetry_session_manifest,
    calibration_fingerprint,
    canonical_json_sha256,
    load_telemetry_session_manifest,
    sha256_file,
    validate_telemetry_session_manifest,
    write_telemetry_session_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"
RUN_UUID = "12345678-1234-5678-9234-567812345678"


def _pose(xyz=(0.0, 0.0, 0.0), yaw_degrees=0.0):
    yaw = np.deg2rad(yaw_degrees)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    result[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return result


def _snapshot():
    canonical = np.stack(
        (_pose((0.50, 0.10, 0.25)), _pose((0.60, 0.20, 0.30), 90.0))
    )
    hands = np.stack(
        (_pose((0.52, 0.10, 0.28)), _pose((0.62, 0.20, 0.33), 90.0))
    )
    grasps = GraspCandidates(
        canonical_poses=canonical,
        scores=np.asarray([0.7, 0.9], dtype=np.float32),
        type_ids=np.asarray([1, 2], dtype=np.int32),
        collision_free=np.asarray([True, True]),
        collision_checked=np.asarray([True, True]),
        selected_index=-1,
        approach_axis_local=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        hand_poses=hands,
        hand_angles=np.asarray(
            [[900, 800, 700, 600, 500, 950], [850, 750, 650, 550, 450, 925]],
            dtype=np.float32,
        ),
        widths_m=np.asarray([0.05, 0.06], dtype=np.float32),
        depths_m=np.asarray([0.02, 0.03], dtype=np.float32),
        source_indices=np.asarray([10, 11], dtype=np.int64),
    )
    return VisualizationSnapshot(
        scene_points=np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32),
        scene_colors=np.asarray([[0.2, 0.3, 0.4]], dtype=np.float32),
        object_points=np.asarray([[0.5, 0.1, 0.2]], dtype=np.float32),
        object_colors=np.asarray([[0.9, 0.1, 0.2]], dtype=np.float32),
        grasps=grasps,
        reference_frame="robot_base",
        T_reference_camera=_pose((1.20, 0.35, 0.64), 28.0),
        frame_id=12,
        timestamp_s=1234.5,
        calibration_id="eye-to-hand-b722bce10485c8a3",
        camera_serial="337322072188",
        model_name="AnyDexGrasp official",
        representation_checkpoint_sha256="a" * 64,
        decision_checkpoint_sha256s=tuple(
            "{:064x}".format(index + 1) for index in range(8)
        ),
        official_source_commit="b" * 40,
    )


def _seal_audit(unsigned):
    result = deepcopy(unsigned)
    result["artifact_sha256"] = canonical_json_sha256(result)
    return result


def _reseal_manifest(payload):
    payload["integrity"]["payload_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "integrity"}
    )
    return payload


@pytest.fixture
def inputs(tmp_path):
    snapshot_path = save_snapshot_npz(tmp_path / "snapshot.npz", _snapshot())
    config_path = tmp_path / "control.json"
    config_path.write_bytes(CONFIG.read_bytes())
    audit_path = tmp_path / "pregrasp-audit.json"
    audit = _seal_audit(
        {
            "schema_version": 2,
            "artifact_type": "fr3_rh56_pregrasp_only_collision_audit",
            "execution_scope": "open_hand_current_to_pregrasp_only",
            "motion_authorized": False,
            "test_evidence": {"passed": True},
        }
    )
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    producer_path = tmp_path / "native-producer.so"
    producer_path.write_bytes(b"offline native producer fixture\x00\x01")
    return {
        "snapshot_path": snapshot_path,
        "control_config_path": config_path,
        "audit_artifact_path": audit_path,
        "producer_build_path": producer_path,
        "command": "pregrasp",
        "selected_index": 1,
        "run_uuid": RUN_UUID,
    }


def test_build_binds_shared_identity_calibration_audit_and_air_contract(inputs):
    payload = build_telemetry_session_manifest(**inputs)

    assert payload["artifact_type"] == ARTIFACT_TYPE
    assert payload["motion_authorized"] is False
    assert payload["identity"]["run_uuid"] == RUN_UUID
    assert payload["identity"]["source_snapshot_sha256"] == sha256_file(
        inputs["snapshot_path"]
    )
    assert payload["identity"]["control_config_sha256"] == sha256_file(
        inputs["control_config_path"]
    )
    assert payload["identity"]["producer_build_sha256"] == sha256_file(
        inputs["producer_build_path"]
    )
    assert (
        payload["identity"]["calibration_sha256"]
        == payload["calibration"]["sha256"]
    )
    execution = payload["execution"]
    assert execution["command"] == "pregrasp"
    assert execution["selected_index"] == 1
    assert (
        payload["identity"]["execution_contract_sha256"]
        == canonical_json_sha256(execution)
    )
    audit = payload["sources"]["audit_artifact"]
    assert audit["file_sha256"] == sha256_file(inputs["audit_artifact_path"])
    assert audit["self_seal_sha256"]
    air = execution["air_target_contract"]
    assert air["reference_frame"] == "robot_base"
    assert air["retreat_distance_m"] == pytest.approx(0.08)
    assert air["pregrasp_extra_distance_m"] == pytest.approx(0.01)
    validate_telemetry_session_manifest(payload, verify_files=True)


def test_atomic_write_and_default_loader_replay_every_file(inputs, tmp_path):
    payload = build_telemetry_session_manifest(**inputs)
    output = tmp_path / "nested" / "session.json"

    assert write_telemetry_session_manifest(output, payload) == output.resolve()
    assert not list(output.parent.glob(".session.json.*.tmp"))
    loaded = load_telemetry_session_manifest(output)

    assert loaded.path == output.resolve()
    assert loaded.identity.run_uuid == RUN_UUID
    assert loaded.manifest_file_sha256 == sha256_file(output)
    assert loaded.payload == payload


@pytest.mark.parametrize(
    "location",
    ("root", "identity", "sources", "air_contract", "calibration", "integrity"),
)
def test_unknown_manifest_fields_are_rejected_even_after_resealing(inputs, location):
    payload = build_telemetry_session_manifest(**inputs)
    if location == "root":
        payload["unknown"] = 1
    elif location == "identity":
        payload["identity"]["unknown"] = 1
    elif location == "sources":
        payload["sources"]["snapshot"]["unknown"] = 1
    elif location == "air_contract":
        payload["execution"]["air_target_contract"]["unknown"] = 1
    elif location == "calibration":
        payload["calibration"]["unknown"] = 1
    else:
        payload["integrity"]["unknown"] = 1
    _reseal_manifest(payload)

    with pytest.raises(ValueError, match="unknown|keys differ"):
        validate_telemetry_session_manifest(payload)


def test_manifest_content_tamper_fails_internal_integrity(inputs):
    payload = build_telemetry_session_manifest(**inputs)
    payload["execution"]["selected_index"] = 0

    with pytest.raises(ValueError, match="execution_contract_sha256|integrity"):
        validate_telemetry_session_manifest(payload)


@pytest.mark.parametrize("source_name", ("control_config_path", "producer_build_path"))
def test_bound_file_byte_tamper_is_rejected(inputs, tmp_path, source_name):
    payload = build_telemetry_session_manifest(**inputs)
    output = write_telemetry_session_manifest(tmp_path / "session.json", payload)
    source = Path(inputs[source_name])
    source.write_bytes(source.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="replay differs|cannot load control config"):
        load_telemetry_session_manifest(output)


def test_audit_semantic_content_and_file_hash_are_both_replayed(inputs, tmp_path):
    payload = build_telemetry_session_manifest(**inputs)
    output = write_telemetry_session_manifest(tmp_path / "session.json", payload)
    audit_path = Path(inputs["audit_artifact_path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["test_evidence"]["passed"] = False
    del audit["artifact_sha256"]
    audit = _seal_audit(audit)
    audit_path.write_text(json.dumps(audit) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="replay differs"):
        load_telemetry_session_manifest(output)


def test_duplicate_keys_and_nonfinite_json_are_rejected(inputs, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version":1,"schema_version":1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_telemetry_session_manifest(path, verify_files=False)

    path.write_text('{"schema_version":NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        load_telemetry_session_manifest(path, verify_files=False)


def test_noncanonical_uuid_and_wrong_audit_scope_fail_closed(inputs):
    bad_uuid = dict(
        inputs, run_uuid="ABCDEFAB-1234-5678-9234-567812345678"
    )
    with pytest.raises(ValueError, match="canonical lowercase"):
        build_telemetry_session_manifest(**bad_uuid)

    audit_path = Path(inputs["audit_artifact_path"])
    installed = _seal_audit(
        {
            "schema_version": 2,
            "artifact_type": "fr3_rh56_installed_tool_collision_audit",
            "mode": "air_grasp",
            "decision": {"motion_authorized": False},
        }
    )
    audit_path.write_text(json.dumps(installed) + "\n", encoding="utf-8")
    wrong_scope = dict(inputs, command="grasp")
    with pytest.raises(ValueError, match="requires audit execution scope"):
        build_telemetry_session_manifest(**wrong_scope)


def test_calibration_fingerprint_uses_snapshot_fields_and_transform_only():
    snapshot = _snapshot()
    baseline = calibration_fingerprint(snapshot)
    changed_id = calibration_fingerprint(
        replace(snapshot, calibration_id="eye-to-hand-other")
    )
    changed_serial = calibration_fingerprint(replace(snapshot, camera_serial="42"))
    transform = np.asarray(snapshot.T_reference_camera).copy()
    transform[0, 3] += 1.0e-6
    changed_transform = calibration_fingerprint(
        replace(snapshot, T_reference_camera=transform)
    )

    assert len({
        baseline["sha256"],
        changed_id["sha256"],
        changed_serial["sha256"],
        changed_transform["sha256"],
    }) == 4


def test_cli_create_and_verify_are_hardware_free(inputs, tmp_path):
    output = tmp_path / "cli-session.json"
    command = [
        sys.executable,
        str(ROOT / "apps/telemetry_session_manifest.py"),
        "create",
        "--snapshot",
        str(inputs["snapshot_path"]),
        "--config",
        str(inputs["control_config_path"]),
        "--audit-artifact",
        str(inputs["audit_artifact_path"]),
        "--producer-build",
        str(inputs["producer_build_path"]),
        "--command",
        "pregrasp",
        "--selected-index",
        "1",
        "--run-uuid",
        RUN_UUID,
        "--output",
        str(output),
    ]
    created = subprocess.run(command, text=True, capture_output=True, check=False)
    assert created.returncode == 0, created.stderr
    assert "motion_authorized=false" in created.stdout

    verified = subprocess.run(
        [
            sys.executable,
            str(ROOT / "apps/telemetry_session_manifest.py"),
            "verify",
            "--manifest",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stderr
    assert "verified=" in verified.stdout


def test_import_does_not_touch_hardware_modules():
    script = r'''
import builtins
original = builtins.__import__
forbidden = {"pylibfranka", "franka", "serial", "pyrealsense2"}
def guarded(name, *args, **kwargs):
    if name.split(".", 1)[0] in forbidden:
        raise RuntimeError("forbidden hardware import: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import anydex_pipeline.telemetry_session_manifest
print("offline-import-ok")
'''
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "offline-import-ok"
