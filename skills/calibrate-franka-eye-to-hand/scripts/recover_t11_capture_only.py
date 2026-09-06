#!/usr/bin/env python3
"""One-use, no-motion recovery for the remount5 T11 ENOSPC interruption.

The original T10->T11 motion succeeded and the process stopped while creating
camera telemetry.  This program can only recover the missing T11 *capture*.
It never calls a robot motion API.  Before opening FCI it atomically consumes a
fixed recovery claim, then performs one passive Idle/error/contact/pose read at
T11.  Fresh image inspection, 120-frame telemetry, and the formal 120-frame
same-batch gates must all pass before exactly sample 11 is appended.

After a verified recovery, ``prepare-resume-master`` creates a new immutable
master for T11->T12..T20.  It never overwrites or reuses the original master or
artifact directory.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_training_sequence as training  # noqa: E402


ROOT = Path("/home/qiaoguanren/code/franka")
CALIBRATION_ROOT = ROOT / "beta/dynamic_object_pcd"
RUNS = CALIBRATION_ROOT / "calibration_runs"
ORIGINAL_MASTER = RUNS / "fr3-d435-342222071785-20260812-002810-remount5-training-plan.yaml"
ORIGINAL_MASTER_SHA256 = "5caf5dc41cbc7e6b5e74a9205642bf8a045782e80ad6ca2c6f4d9974cb93c782"
DATASET = RUNS / "fr3-d435-342222071785-20260812-002810-remount5-training.yaml"
DATASET_10_SHA256 = "2c71e8413325324a4c85838b57b7810423ab3d1e466b43e33f1e14325c2a2433"
ORIGINAL_ARTIFACTS = RUNS / "fr3-d435-342222071785-20260812-002810-remount5-training-sequence-from-T01"
EDGE_STEM = "T10-to-T11-072b8fa59627d7db"
EDGE_PLAN = ORIGINAL_ARTIFACTS / "edges" / (EDGE_STEM + ".plan.yaml")
EDGE_PLAN_SHA256 = "072b8fa59627d7db0562aac11fb03651e9939ca574ebcfbc8284b12da11ce352"
ORIGINAL_CLAIM = RUNS / ".training-edge-claims" / (
    ORIGINAL_MASTER_SHA256 + "-T10-to-T11.claim.json"
)
ORIGINAL_CLAIM_SHA256 = "31ffa56db55d65072e1aefc23245a463ecdd194fca3981b83198305b1e247ec2"
ORIGINAL_MOTION_LOG = ORIGINAL_ARTIFACTS / "edges" / (EDGE_STEM + ".motion.log")
ORIGINAL_MOTION_LOG_SHA256 = "a0035810cbbdcb7aa155e6007b2dac8ab23f55d6c4ced25b18235be5e9136b6f"
ORIGINAL_INSPECT_LOG = ORIGINAL_ARTIFACTS / "edges" / (EDGE_STEM + ".inspect.log")
ORIGINAL_INSPECT_LOG_SHA256 = "3b678874b07677e53784f21b3f4d9d7f67f88d6d713990b013c613789e7d4b1c"
ORIGINAL_INSPECTION_RAW = ORIGINAL_ARTIFACTS / "edges" / (EDGE_STEM + ".inspection-raw.png")
ORIGINAL_INSPECTION_ANNOTATED = ORIGINAL_ARTIFACTS / "edges" / (EDGE_STEM + ".inspection-annotated.png")
ORIGINAL_INSPECTION_RAW_SHA256 = "3fcc532c92143b8837dbdb60780876d40be6a182f9520823bf6698af5f982f86"
ORIGINAL_INSPECTION_ANNOTATED_SHA256 = "891a5cea2bd7777b0ee243f22de9cf1b431a6107f5087510fe1ff19d8a1ea209"
ORIGINAL_PREVIEW_LOG = ORIGINAL_ARTIFACTS / "edges" / (EDGE_STEM + ".preview.log")
ORIGINAL_PREVIEW_LOG_SHA256 = "9f26281a653e35ea528557e517b9837effa693537ac7a8ffb11f6e6b1c7ac925"
ORIGINAL_TELEMETRY = ORIGINAL_ARTIFACTS / "edges" / (EDGE_STEM + ".frame-telemetry.json")
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
ORIGINAL_IMPORT_LOG = ORIGINAL_ARTIFACTS / "python-import-preflight.log"
ORIGINAL_IMPORT_LOG_SHA256 = "3b2d5313674aa7e231754522c95d17650f85cb80d2fc4609df75bf2d36032b0b"

RECOVERY_PLAN = RUNS / "fr3-d435-342222071785-20260812-002810-remount5-T11-enospc-capture-only-v1.plan.yaml"
RECOVERY_ARTIFACTS = RUNS / "fr3-d435-342222071785-20260812-002810-remount5-T11-enospc-capture-only-v1"
RECOVERY_CLAIM = RUNS / ".training-capture-recovery-claims" / (
    "fr3-d435-342222071785-remount5-T11-enospc-capture-only-v1.claim.json"
)
RESUME_MASTER = RUNS / "fr3-d435-342222071785-20260812-002810-remount5-training-resume-from-T11-v1-plan.yaml"
RESUME_ARTIFACTS = RUNS / "fr3-d435-342222071785-20260812-002810-remount5-training-resume-from-T11-v1"

PLAN_KIND = "franka_eye_to_hand_capture_only_recovery_plan"
PLAN_STATUS = "explicit_capture_only_recovery_reviewed"
CLAIM_KIND = "franka_eye_to_hand_capture_only_recovery_claim"
RECEIPT_KIND = "franka_eye_to_hand_capture_only_recovery_receipt"
RECOVERY_TOKEN = "FR3_T11_CAPTURE_ONLY_RECOVERY_NO_MOTION"
MIN_FREE_BYTES = 1_073_741_824
MAX_FILE_BYTES = 20_000_000
T11_TRANSLATION_TOLERANCE_M = 0.005
T11_ROTATION_TOLERANCE_DEG = 1.0


@dataclass(frozen=True)
class RecoveryConfig:
    plan: Path
    expected_plan_sha256: str
    python_executable: str


@dataclass(frozen=True)
class RecoveryResult:
    dataset_sha256: str
    sample_count: int
    receipt: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path, maximum_bytes: Optional[int] = None) -> str:
    return training._sha256_file(path, maximum_bytes=maximum_bytes)


def _file_record(path: Path, maximum_bytes: Optional[int] = MAX_FILE_BYTES) -> Mapping[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": _sha(resolved, maximum_bytes),
        "bytes": resolved.stat().st_size,
    }


def _assert_record(record: Mapping[str, Any], name: str) -> Tuple[Path, str]:
    path = training._lexical_absolute_path(record.get("path"), name + ".path").resolve(strict=True)
    digest = training._require_text(record.get("sha256"), name + ".sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("{}.sha256 is malformed".format(name))
    size = record.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("{}.bytes is invalid".format(name))
    if path.stat().st_size != size or _sha(path) != digest:
        raise ValueError("{} changed".format(name))
    return path, digest


def _load_master() -> Tuple[Mapping[str, Any], bytes]:
    path, data, master = training._read_yaml(
        ORIGINAL_MASTER, maximum_bytes=training.MAX_PLAN_BYTES
    )
    if path != ORIGINAL_MASTER.resolve() or training._sha256_bytes(data) != ORIGINAL_MASTER_SHA256:
        raise ValueError("original reviewed master changed")
    return master, data


def _load_edge_plan() -> Tuple[Mapping[str, Any], bytes]:
    path, data, plan = training._read_yaml(EDGE_PLAN, maximum_bytes=training.MAX_PLAN_BYTES)
    if path != EDGE_PLAN.resolve() or training._sha256_bytes(data) != EDGE_PLAN_SHA256:
        raise ValueError("original T10->T11 plan changed")
    return plan, data


def _planned_t11(master: Mapping[str, Any]) -> np.ndarray:
    poses = training._pose_map(master)
    if "T11" not in poses:
        raise ValueError("original master has no T11")
    return training._planned_target_transform(master, poses["T11"], "T11")


def _validate_dataset_10(master: Mapping[str, Any]) -> training.DatasetSnapshot:
    snapshot = training._dataset_snapshot(DATASET)
    if snapshot.sha256 != DATASET_10_SHA256 or len(snapshot.samples) != 10:
        raise ValueError("dataset is not the exact reviewed 10-sample prefix")
    expected = training._expected_dataset_metadata(master)
    training._validate_dataset_metadata(snapshot, expected)
    training._validate_existing_sample_prefix(snapshot, master=master, poses=training._pose_map(master))
    return snapshot


def _original_result_paths() -> Tuple[Path, ...]:
    edges = ORIGINAL_ARTIFACTS / "edges"
    return (
        edges / (EDGE_STEM + ".complete.json"),
        edges / (EDGE_STEM + ".failed.json"),
        edges / (EDGE_STEM + ".capture.log"),
        edges / (EDGE_STEM + ".capture.png"),
    )


def _validate_original_evidence(
    master: Mapping[str, Any], snapshot: training.DatasetSnapshot
) -> Mapping[str, Any]:
    plan, _ = _load_edge_plan()
    if plan.get("status") != training.RUN_STATUS:
        raise ValueError("original edge plan is not single-pose authorized")
    authorization = training._require_mapping(plan.get("motion_authorization"), "edge authorization")
    if authorization.get("start_pose_id") != "T10" or authorization.get("pose_id") != "T11":
        raise ValueError("original edge authorization is not T10->T11")
    claim_path, claim_bytes, claim = training._read_json_mapping(
        ORIGINAL_CLAIM, maximum_bytes=MAX_FILE_BYTES
    )
    if claim_path != ORIGINAL_CLAIM.resolve() or training._sha256_bytes(claim_bytes) != ORIGINAL_CLAIM_SHA256:
        raise ValueError("original canonical edge claim changed")
    exact_claim = {
        "kind": "franka_eye_to_hand_training_edge_claim",
        "state": "claimed_before_motion",
        "edge": "T10_to_T11",
        "start_pose_id": "T10",
        "target_pose_id": "T11",
        "reviewed_master_plan_sha256": ORIGINAL_MASTER_SHA256,
        "single_edge_plan_sha256": EDGE_PLAN_SHA256,
        "dataset_pre_motion_sha256": DATASET_10_SHA256,
        "dataset_pre_motion_sample_count": 10,
        "expected_post_capture_sample_count": 11,
        "holdout_capture_permitted": False,
    }
    for field, expected in exact_claim.items():
        if claim.get(field) != expected:
            raise ValueError("original claim field {} mismatch".format(field))
    if Path(str(claim.get("dataset"))).resolve() != snapshot.path:
        raise ValueError("original claim dataset path mismatch")
    if Path(str(claim.get("single_edge_plan"))).resolve() != EDGE_PLAN.resolve():
        raise ValueError("original claim edge-plan path mismatch")

    motion_text = ORIGINAL_MOTION_LOG.read_text(encoding="utf-8")
    if _sha(ORIGINAL_MOTION_LOG) != ORIGINAL_MOTION_LOG_SHA256:
        raise ValueError("original motion log changed")
    motion_result = training.CommandResult(("original-motion-evidence",), 0, motion_text)
    motion_telemetry = training._verify_motion_ready(
        motion_result, target_pose_id="T11", plan_sha256=EDGE_PLAN_SHA256
    )

    if _sha(ORIGINAL_PREVIEW_LOG) != ORIGINAL_PREVIEW_LOG_SHA256:
        raise ValueError("original preview log changed")
    preview_result = training.CommandResult(
        ("original-preview-evidence",),
        0,
        ORIGINAL_PREVIEW_LOG.read_text(encoding="utf-8"),
    )
    preview_target = training._parse_preview(
        preview_result,
        plan_path=EDGE_PLAN.resolve(),
        plan_sha256=EDGE_PLAN_SHA256,
        start_pose_id="T10",
        target_pose_id="T11",
    )
    preview_translation, preview_rotation = training._transform_error(
        preview_target, _planned_t11(master)
    )
    if preview_translation > 1.0e-9 or preview_rotation > 1.0e-6:
        raise ValueError("original preview target differs from reviewed T11")

    if _sha(ORIGINAL_INSPECT_LOG) != ORIGINAL_INSPECT_LOG_SHA256:
        raise ValueError("original inspection log changed")
    dataset_camera = training._require_mapping(snapshot.document.get("camera"), "dataset.camera")
    target = training._require_mapping(training._expected_dataset_metadata(master)["target"], "target")
    inspection_result = training.CommandResult(
        ("original-inspection-evidence",),
        0,
        ORIGINAL_INSPECT_LOG.read_text(encoding="utf-8"),
    )
    inspection_metrics = training._parse_inspection(
        inspection_result,
        expected_serial=str(training._expected_dataset_metadata(master)["camera_serial"]),
        target=target,
        dataset_camera=dataset_camera,
        raw_image=ORIGINAL_INSPECTION_RAW.resolve(),
        annotated_image=ORIGINAL_INSPECTION_ANNOTATED.resolve(),
    )
    if _sha(ORIGINAL_INSPECTION_RAW) != ORIGINAL_INSPECTION_RAW_SHA256:
        raise ValueError("original raw inspection image changed")
    if _sha(ORIGINAL_INSPECTION_ANNOTATED) != ORIGINAL_INSPECTION_ANNOTATED_SHA256:
        raise ValueError("original annotated inspection image changed")
    if ORIGINAL_TELEMETRY.stat().st_size != 0 or _sha(ORIGINAL_TELEMETRY) != EMPTY_SHA256:
        raise ValueError("original failed telemetry artifact is no longer exact zero bytes")
    if _sha(ORIGINAL_IMPORT_LOG) != ORIGINAL_IMPORT_LOG_SHA256:
        raise ValueError("original import preflight log changed")
    for path in _original_result_paths():
        if path.exists():
            raise ValueError("original T11 result path must remain absent: {}".format(path))
    return {
        "claim": _file_record(ORIGINAL_CLAIM),
        "edge_plan": _file_record(EDGE_PLAN),
        "motion_log": _file_record(ORIGINAL_MOTION_LOG),
        "motion_control_loop_telemetry": motion_telemetry,
        "preview_log": _file_record(ORIGINAL_PREVIEW_LOG),
        "preview_target_T_base_ee": preview_target.tolist(),
        "inspection_log": _file_record(ORIGINAL_INSPECT_LOG),
        "inspection_raw": _file_record(ORIGINAL_INSPECTION_RAW),
        "inspection_annotated": _file_record(ORIGINAL_INSPECTION_ANNOTATED),
        "inspection_metrics": inspection_metrics,
        "failed_zero_byte_telemetry": _file_record(ORIGINAL_TELEMETRY),
        "import_probe_log": _file_record(ORIGINAL_IMPORT_LOG),
        "verified_absent_result_paths": [str(path.resolve()) for path in _original_result_paths()],
    }


def _canonical_code_paths() -> Mapping[str, Path]:
    return {
        "recovery_script": Path(__file__).resolve(strict=True),
        "training_orchestrator": (SCRIPT_DIR / "run_training_sequence.py").resolve(strict=True),
        "motion_wrapper": (SCRIPT_DIR / "move_calibration_pose.py").resolve(strict=True),
        "inspection_script": (SCRIPT_DIR / "inspect_aruco_frame.py").resolve(strict=True),
        "frame_telemetry_script": (SCRIPT_DIR / "capture_aruco_frame_telemetry.py").resolve(strict=True),
        "capture_config": (CALIBRATION_ROOT / "configs/d435_default.yaml").resolve(strict=True),
        "capture_cli": (CALIBRATION_ROOT / "dynamic_pcd/apps/calibrate_eye_to_hand.py").resolve(strict=True),
        "detector": (CALIBRATION_ROOT / "dynamic_pcd/calibration/aruco.py").resolve(strict=True),
        "stationary": (CALIBRATION_ROOT / "dynamic_pcd/calibration/stationary.py").resolve(strict=True),
        "transforms": (CALIBRATION_ROOT / "dynamic_pcd/calibration/transforms.py").resolve(strict=True),
        "motion_driver": (ROOT / "dexgrasp/src/anydex_pipeline/franka_sequence_driver.py").resolve(strict=True),
        "link_preflight": (ROOT / "dexgrasp/src/anydex_pipeline/host_network_preflight.py").resolve(strict=True),
    }


def prepare_recovery_plan(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.confirm_capture_only_no_motion != RECOVERY_TOKEN:
        raise ValueError("exact capture-only/no-motion recovery token is required")
    plan_output = args.output_plan.expanduser().resolve()
    artifacts = args.artifacts_dir.expanduser().resolve()
    claim = args.claim_path.expanduser().resolve()
    if plan_output != RECOVERY_PLAN.resolve():
        raise ValueError("recovery plan path must be the fixed reviewed v1 path")
    if artifacts != RECOVERY_ARTIFACTS.resolve():
        raise ValueError("recovery artifacts path must be the fixed reviewed v1 path")
    if claim != RECOVERY_CLAIM.resolve():
        raise ValueError("recovery claim path must be the fixed one-use v1 ledger")
    if (
        isinstance(args.minimum_free_bytes, bool)
        or not isinstance(args.minimum_free_bytes, int)
        or args.minimum_free_bytes != MIN_FREE_BYTES
    ):
        raise ValueError(
            "minimum-free-bytes must be exactly {} (1 GiB)".format(MIN_FREE_BYTES)
        )
    if plan_output.exists() or artifacts.exists() or claim.exists():
        raise FileExistsError("recovery plan, artifacts, and claim paths must all be absent")
    master, _ = _load_master()
    snapshot = _validate_dataset_10(master)
    evidence = _validate_original_evidence(master, snapshot)
    code = _canonical_code_paths()
    python_launch = training._lexical_absolute_path(args.python_executable, "python executable")
    python_resolved = python_launch.resolve(strict=True)
    pins = {name: _file_record(path) for name, path in code.items()}
    pins["python_executable"] = {
        "path": str(python_launch),
        "resolved_path": str(python_resolved),
        "sha256": _sha(python_resolved),
        "bytes": python_resolved.stat().st_size,
    }
    plan = {
        "schema_version": 1,
        "kind": PLAN_KIND,
        "status": PLAN_STATUS,
        "session_slug": "fr3-d435-342222071785-remount5-T11-enospc-capture-only-v1",
        "scope": {
            "robot_motion_permitted": False,
            "robot_state_read_only": True,
            "camera_capture_permitted": True,
            "target_pose_id": "T11",
            "expected_new_sample_index": 11,
            "automatic_retry_permitted": False,
        },
        "topology": copy.deepcopy(master["topology"]),
        "hardware": copy.deepcopy(master["hardware"]),
        "target": copy.deepcopy(master["target"]),
        "reference_pose": copy.deepcopy(master["reference_pose"]),
        "safety": {
            "emergency_stop_reachable": True,
            "target_rigid_confirmed": True,
            "camera_fixed_confirmed": True,
            "minimum_free_bytes_before_hardware": int(args.minimum_free_bytes),
            "required_live_robot_mode": "Idle",
            "require_no_current_errors": True,
            "require_no_cartesian_or_joint_contact_or_collision": True,
            "maximum_live_T11_translation_error_m": T11_TRANSLATION_TOLERANCE_M,
            "maximum_live_T11_rotation_error_deg": T11_ROTATION_TOLERANCE_DEG,
        },
        "provenance": {
            "failure_class": "ENOSPC_during_original_frame_telemetry_write",
            "original_master": _file_record(ORIGINAL_MASTER),
            "original_dataset": {
                "path": str(snapshot.path),
                "sha256": snapshot.sha256,
                "sample_count": 10,
            },
            "original_edge_evidence": evidence,
            "pins": pins,
        },
        "recovery_transaction": {
            "plan_path": str(plan_output),
            "artifacts_dir": str(artifacts),
            "canonical_one_use_claim": str(claim),
            "dataset": str(snapshot.path),
            "dataset_pre_capture_sha256": snapshot.sha256,
            "dataset_pre_capture_sample_count": 10,
            "expected_dataset_post_capture_sample_count": 11,
            "target_T_base_ee": _planned_t11(master).tolist(),
            "fresh_inspection_required": True,
            "fresh_frame_telemetry_frames": 120,
            "formal_capture_frames": 120,
            "formal_same_batch_untrimmed_gates_required": True,
        },
    }
    payload = yaml.safe_dump(plan, sort_keys=False, allow_unicode=True).encode("utf-8")
    if len(payload) > training.MAX_PLAN_BYTES:
        raise ValueError("recovery plan is too large")
    training._publish_exclusive(plan_output, payload)
    result = {
        "plan": str(plan_output),
        "plan_sha256": training._sha256_bytes(payload),
        "dataset_sha256": snapshot.sha256,
        "dataset_sample_count": len(snapshot.samples),
        "robot_motion_permitted": False,
        "hardware_opened": False,
        "claim": str(claim),
        "artifacts_dir": str(artifacts),
    }
    print("T11_RECOVERY_PLAN_JSON=" + json.dumps(result, sort_keys=True, separators=(",", ":")))
    return result


def _load_recovery_plan(
    path: Path, expected_sha256: str
) -> Tuple[Path, str, Mapping[str, Any], Mapping[str, Path], training.DatasetSnapshot]:
    resolved, data, plan = training._read_yaml(path, maximum_bytes=training.MAX_PLAN_BYTES)
    digest = training._sha256_bytes(data)
    if digest != expected_sha256:
        raise ValueError(
            "recovery plan SHA mismatch: expected={} actual={}".format(expected_sha256, digest)
        )
    if plan.get("schema_version") != 1 or plan.get("kind") != PLAN_KIND:
        raise ValueError("not a schema-1 T11 capture-only recovery plan")
    if plan.get("status") != PLAN_STATUS:
        raise ValueError("recovery plan status mismatch")
    scope = training._require_mapping(plan.get("scope"), "scope")
    if scope != {
        "robot_motion_permitted": False,
        "robot_state_read_only": True,
        "camera_capture_permitted": True,
        "target_pose_id": "T11",
        "expected_new_sample_index": 11,
        "automatic_retry_permitted": False,
    }:
        raise ValueError("capture-only recovery scope changed")
    transaction = training._require_mapping(
        plan.get("recovery_transaction"), "recovery_transaction"
    )
    if Path(str(transaction.get("plan_path"))).resolve() != resolved:
        raise ValueError("recovery transaction plan path mismatch")
    if resolved != RECOVERY_PLAN.resolve():
        raise ValueError("recovery plan is not the fixed reviewed v1 path")
    if Path(str(transaction.get("dataset"))).resolve() != DATASET.resolve():
        raise ValueError("recovery transaction dataset path mismatch")
    if (
        transaction.get("dataset_pre_capture_sha256") != DATASET_10_SHA256
        or transaction.get("dataset_pre_capture_sample_count") != 10
        or transaction.get("expected_dataset_post_capture_sample_count") != 11
        or transaction.get("fresh_inspection_required") is not True
        or transaction.get("fresh_frame_telemetry_frames") != 120
        or transaction.get("formal_capture_frames") != 120
        or transaction.get("formal_same_batch_untrimmed_gates_required") is not True
    ):
        raise ValueError("recovery transaction gates changed")
    target_T = training._rigid_transform(
        transaction.get("target_T_base_ee"), "recovery target_T_base_ee"
    )
    artifacts = Path(str(transaction.get("artifacts_dir"))).resolve()
    claim = Path(str(transaction.get("canonical_one_use_claim"))).resolve()
    if artifacts != RECOVERY_ARTIFACTS.resolve():
        raise ValueError("recovery artifacts path is not the fixed reviewed v1 path")
    if claim != RECOVERY_CLAIM.resolve():
        raise ValueError("recovery claim path is not the fixed one-use v1 ledger")
    safety = training._require_mapping(plan.get("safety"), "safety")
    free_gate = safety.get("minimum_free_bytes_before_hardware")
    if isinstance(free_gate, bool) or not isinstance(free_gate, int) or free_gate != MIN_FREE_BYTES:
        raise ValueError("recovery plan must commit the exact 1 GiB free-space gate")

    master, _ = _load_master()
    expected_T = _planned_t11(master)
    translation, rotation = training._transform_error(target_T, expected_T)
    if translation > 1.0e-9 or rotation > 1.0e-6:
        raise ValueError("recovery target no longer exactly matches reviewed T11")
    snapshot = _validate_dataset_10(master)
    provenance = training._require_mapping(plan.get("provenance"), "provenance")
    if provenance.get("failure_class") != "ENOSPC_during_original_frame_telemetry_write":
        raise ValueError("recovery failure class mismatch")
    original_master_record = training._require_mapping(
        provenance.get("original_master"), "provenance.original_master"
    )
    original_master_path, original_master_sha = _assert_record(
        original_master_record, "provenance.original_master"
    )
    if original_master_path != ORIGINAL_MASTER.resolve() or original_master_sha != ORIGINAL_MASTER_SHA256:
        raise ValueError("recovery provenance original master mismatch")
    original_dataset = training._require_mapping(
        provenance.get("original_dataset"), "provenance.original_dataset"
    )
    if (
        Path(str(original_dataset.get("path"))).resolve() != snapshot.path
        or original_dataset.get("sha256") != snapshot.sha256
        or original_dataset.get("sample_count") != 10
    ):
        raise ValueError("recovery provenance dataset mismatch")
    current_evidence = _validate_original_evidence(master, snapshot)
    if training._canonical(provenance.get("original_edge_evidence")) != training._canonical(current_evidence):
        raise ValueError("original edge evidence changed since plan review")

    code = _canonical_code_paths()
    pins = training._require_mapping(provenance.get("pins"), "provenance.pins")
    pinned_paths: Dict[str, Path] = {}
    for name, expected_path in code.items():
        record = training._require_mapping(pins.get(name), "pins." + name)
        pinned_path, _ = _assert_record(record, "pins." + name)
        if pinned_path != expected_path:
            raise ValueError("pin {} is not the canonical path".format(name))
        pinned_paths[name] = pinned_path
    python_pin = training._require_mapping(pins.get("python_executable"), "pins.python_executable")
    python_launch = training._lexical_absolute_path(
        python_pin.get("path"), "pins.python_executable.path"
    )
    python_resolved = Path(
        training._require_text(
            python_pin.get("resolved_path"), "pins.python_executable.resolved_path"
        )
    ).resolve(strict=True)
    if python_launch.resolve(strict=True) != python_resolved:
        raise ValueError("pinned Python symlink target changed")
    if _sha(python_resolved) != python_pin.get("sha256"):
        raise ValueError("pinned Python executable changed")
    pinned_paths["python_launch"] = python_launch
    pinned_paths["python_resolved"] = python_resolved
    return resolved, digest, plan, pinned_paths, snapshot


def _recheck_immutable_plan_inputs(
    plan_path: Path,
    plan_sha256: str,
    plan: Mapping[str, Any],
) -> None:
    training._ensure_unchanged(plan_path, plan_sha256, "integrity")
    provenance = training._require_mapping(plan.get("provenance"), "provenance")
    _assert_record(
        training._require_mapping(provenance.get("original_master"), "original_master"),
        "original_master",
    )
    evidence = training._require_mapping(
        provenance.get("original_edge_evidence"), "original_edge_evidence"
    )
    for name in (
        "claim",
        "edge_plan",
        "motion_log",
        "preview_log",
        "inspection_log",
        "inspection_raw",
        "inspection_annotated",
        "failed_zero_byte_telemetry",
        "import_probe_log",
    ):
        _assert_record(training._require_mapping(evidence.get(name), name), name)
    for raw in evidence.get("verified_absent_result_paths", []):
        if Path(str(raw)).exists():
            raise ValueError("original result path unexpectedly appeared: {}".format(raw))
    pins = training._require_mapping(provenance.get("pins"), "pins")
    for name in _canonical_code_paths():
        _assert_record(training._require_mapping(pins.get(name), "pins." + name), "pins." + name)
    python_pin = training._require_mapping(pins.get("python_executable"), "pins.python_executable")
    python_resolved = Path(str(python_pin.get("resolved_path"))).resolve(strict=True)
    if _sha(python_resolved) != python_pin.get("sha256"):
        raise ValueError("Python executable changed")


def _recovery_paths(plan: Mapping[str, Any]) -> Tuple[Path, Path]:
    transaction = training._require_mapping(plan.get("recovery_transaction"), "recovery_transaction")
    artifacts = Path(str(transaction.get("artifacts_dir"))).resolve()
    claim = Path(str(transaction.get("canonical_one_use_claim"))).resolve()
    return artifacts, claim


def _claim_recovery(
    *,
    plan_path: Path,
    plan_sha256: str,
    plan: Mapping[str, Any],
    snapshot: training.DatasetSnapshot,
    import_probe_log: Mapping[str, Any],
    clock: Callable[[], str],
) -> Tuple[Path, str]:
    artifacts, claim_path = _recovery_paths(plan)
    payload = {
        "schema_version": 1,
        "kind": CLAIM_KIND,
        "state": "claimed_before_any_hardware_open",
        "claimed_at": clock(),
        "recovery_plan": str(plan_path),
        "recovery_plan_sha256": plan_sha256,
        "fixed_recovery_id": "remount5-T11-enospc-capture-only-v1",
        "robot_motion_permitted": False,
        "robot_state_read_only": True,
        "camera_capture_permitted": True,
        "dataset": str(snapshot.path),
        "dataset_pre_capture_sha256": snapshot.sha256,
        "dataset_pre_capture_sample_count": len(snapshot.samples),
        "expected_post_capture_sample_count": 11,
        "original_master_sha256": ORIGINAL_MASTER_SHA256,
        "original_edge_plan_sha256": EDGE_PLAN_SHA256,
        "original_edge_claim_sha256": ORIGINAL_CLAIM_SHA256,
        "original_motion_log_sha256": ORIGINAL_MOTION_LOG_SHA256,
        "original_inspection_log_sha256": ORIGINAL_INSPECT_LOG_SHA256,
        "original_failed_telemetry_sha256": EMPTY_SHA256,
        "recovery_artifacts_dir": str(artifacts),
        "import_probe_log": import_probe_log,
    }
    data = training._json_bytes(payload)
    training._publish_exclusive(claim_path, data)
    return claim_path, training._sha256_bytes(data)


def _parse_state_probe(
    result: training.CommandResult,
    *,
    plan_path: Path,
    plan_sha256: str,
    claim_path: Path,
    claim_sha256: str,
) -> Mapping[str, Any]:
    training._require_success(result, "robot_state_probe")
    sentinel = "FRANKA_T11_PASSIVE_STATE_JSON="
    lines = [line for line in result.output.splitlines() if line.startswith(sentinel)]
    if len(lines) != 1:
        raise training.SequenceFailure("robot_state_probe", "missing unique state sentinel")
    try:
        payload = json.loads(lines[0][len(sentinel) :])
    except json.JSONDecodeError as exc:
        raise training.SequenceFailure("robot_state_probe", "invalid state JSON") from exc
    expected = {
        "recovery_plan": str(plan_path),
        "recovery_plan_sha256": plan_sha256,
        "recovery_claim": str(claim_path),
        "recovery_claim_sha256": claim_sha256,
        "robot_motion_commanded": False,
        "robot_state_read_only": True,
        "robot_mode": "Idle",
        "current_errors_active": False,
        "contacts_or_collisions_active": False,
        "pose_id": "T11",
        "pose_gate_pass": True,
    }
    if not isinstance(payload, dict):
        raise training.SequenceFailure("robot_state_probe", "state payload must be a mapping")
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            raise training.SequenceFailure(
                "robot_state_probe", "state field {} mismatch".format(key)
            )
    training._rigid_transform(payload.get("T_base_ee"), "live T_base_ee")
    translation = training._finite_float(
        payload.get("translation_error_m"), "state translation error"
    )
    rotation = training._finite_float(
        payload.get("rotation_error_deg"), "state rotation error"
    )
    max_dq = training._finite_float(payload.get("max_abs_dq_rad_s"), "state max dq")
    if translation > T11_TRANSLATION_TOLERANCE_M or rotation > T11_ROTATION_TOLERANCE_DEG:
        raise training.SequenceFailure("robot_state_probe", "live robot is not at T11")
    if max_dq > 0.005:
        raise training.SequenceFailure("robot_state_probe", "live robot is not stationary")
    return payload


def passive_state_probe(
    plan_path: Path,
    expected_plan_sha256: str,
    claim_path: Path,
    expected_claim_sha256: str,
) -> Mapping[str, Any]:
    """Open FCI only for one validated read; never create a control handle."""

    resolved, digest, plan, pins, _ = _load_recovery_plan(
        plan_path, expected_plan_sha256
    )
    actual_claim = _sha(claim_path.resolve(strict=True))
    if actual_claim != expected_claim_sha256:
        raise ValueError("recovery claim SHA mismatch")
    _recheck_immutable_plan_inputs(resolved, digest, plan)

    motion_path = pins["motion_wrapper"]
    spec = importlib.util.spec_from_file_location("t11_capture_recovery_motion", motion_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned motion wrapper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    preview = module.load_pose_preview(
        EDGE_PLAN, "T11", expected_sha256=EDGE_PLAN_SHA256, for_run=False
    )
    if preview.pose_id != "T11" or preview.dynamics is None:
        raise RuntimeError("pinned edge plan cannot define strict T11 dynamics")
    hardware = training._require_mapping(plan.get("hardware"), "hardware")
    robot = training._require_mapping(hardware.get("robot"), "hardware.robot")
    robot_ip = training._require_text(robot.get("ip"), "hardware.robot.ip")

    # _connect_franka only opens/configures the FCI client.  No move/hold/control
    # method is called below; exactly one state read is used for all checks.
    arm = module._connect_franka(preview, robot_ip)
    state = arm.robot.read_once()
    success_rate = arm._validate_state(state, require_idle=True, enforce_success=False)
    max_dq = arm._validate_stopped_state(state)
    live = module._state_pose(state)
    module._validate_live_workspace(preview, live)
    expected_T = training._rigid_transform(
        training._require_mapping(plan.get("recovery_transaction"), "transaction").get(
            "target_T_base_ee"
        ),
        "target T11",
    )
    translation, rotation = training._transform_error(live, expected_T)
    if translation > T11_TRANSLATION_TOLERANCE_M or rotation > T11_ROTATION_TOLERANCE_DEG:
        raise RuntimeError(
            "live pose is not T11: {:.3f}mm/{:.3f}deg".format(
                translation * 1000.0, rotation
            )
        )
    del state
    del arm
    if max_dq > 0.005:
        raise RuntimeError(
            "live max |dq| {:.9f}rad/s exceeds 0.005rad/s".format(max_dq)
        )
    payload = {
        "recovery_plan": str(resolved),
        "recovery_plan_sha256": digest,
        "recovery_claim": str(claim_path.resolve()),
        "recovery_claim_sha256": actual_claim,
        "robot_motion_commanded": False,
        "robot_state_read_only": True,
        "robot_mode": "Idle",
        "current_errors_active": False,
        "contacts_or_collisions_active": False,
        "pose_id": "T11",
        "pose_gate_pass": True,
        "T_base_ee": live.tolist(),
        "translation_error_m": translation,
        "rotation_error_deg": rotation,
        "max_abs_dq_rad_s": max_dq,
        "control_command_success_rate": float(success_rate),
    }
    print(
        "FRANKA_T11_PASSIVE_STATE_JSON="
        + json.dumps(payload, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return payload


def _clean_environment() -> Mapping[str, str]:
    return training._clean_subprocess_environment(CALIBRATION_ROOT.resolve())


def _log(
    artifacts: Path, name: str, result: training.CommandResult
) -> Mapping[str, Any]:
    return training._write_log(artifacts / name, result)


def run_recovery(
    config: RecoveryConfig,
    *,
    runner: Optional[Any] = None,
    clock: Callable[[], str] = _utc_now,
) -> RecoveryResult:
    command_runner = runner if runner is not None else training.StreamingSubprocessRunner()
    plan_path, plan_sha256, plan, pins, snapshot = _load_recovery_plan(
        config.plan, config.expected_plan_sha256
    )
    transaction = training._require_mapping(
        plan.get("recovery_transaction"), "recovery_transaction"
    )
    artifacts, claim_path = _recovery_paths(plan)
    python_launch = training._lexical_absolute_path(
        config.python_executable, "python executable"
    )
    if python_launch != pins["python_launch"]:
        raise training.SequenceFailure(
            "preflight", "requested Python launch path differs from reviewed plan"
        )
    if artifacts.exists() or claim_path.exists():
        raise training.SequenceFailure(
            "preflight", "recovery artifacts/claim already exist; automatic retry forbidden"
        )
    if shutil.disk_usage(CALIBRATION_ROOT).free < int(
        training._require_mapping(plan.get("safety"), "safety").get(
            "minimum_free_bytes_before_hardware"
        )
    ):
        raise training.SequenceFailure("preflight", "insufficient free disk space")
    _recheck_immutable_plan_inputs(plan_path, plan_sha256, plan)
    training._ensure_unchanged(DATASET, snapshot.sha256, "preflight")

    target = training._require_mapping(
        training._expected_dataset_metadata(plan).get("target"), "target"
    )
    expected_metadata = training._expected_dataset_metadata(plan)
    dataset_camera = training._require_mapping(snapshot.document.get("camera"), "dataset.camera")
    environment = _clean_environment()
    phase = "import_preflight"
    import_command = (
        str(python_launch),
        "-I",
        "-c",
        training._IMPORT_PROBE_CODE,
        str(CALIBRATION_ROOT.resolve()),
    )
    import_result = command_runner.run(
        import_command,
        cwd=CALIBRATION_ROOT.resolve(),
        phase="T11-recovery:import-preflight",
        env=environment,
    )
    import_resolution = training._parse_import_probe(
        import_result,
        capture_cli=pins["capture_cli"],
        detector=pins["detector"],
        stationary=pins["stationary"],
        transforms=pins["transforms"],
    )
    artifacts.mkdir(parents=True, exist_ok=False)
    logs: Dict[str, Mapping[str, Any]] = {}
    logs["import_preflight"] = _log(artifacts, "python-import-preflight.log", import_result)
    claim_sha256: Optional[str] = None
    fresh_metrics: Dict[str, Any] = {}
    receipt_path = artifacts / "capture-only-recovery.complete.json"
    failure_path = artifacts / "capture-only-recovery.failed.json"
    try:
        phase = "claim"
        _recheck_immutable_plan_inputs(plan_path, plan_sha256, plan)
        training._ensure_unchanged(DATASET, snapshot.sha256, phase)
        claim_path, claim_sha256 = _claim_recovery(
            plan_path=plan_path,
            plan_sha256=plan_sha256,
            plan=plan,
            snapshot=snapshot,
            import_probe_log=logs["import_preflight"],
            clock=clock,
        )
        print(
            "[T11 recovery] CLAIMED no-motion transaction claim={} sha256={}".format(
                claim_path, claim_sha256
            ),
            flush=True,
        )

        phase = "robot_state_probe"
        _recheck_immutable_plan_inputs(plan_path, plan_sha256, plan)
        training._ensure_unchanged(DATASET, snapshot.sha256, phase)
        training._ensure_unchanged(claim_path, claim_sha256, phase)
        state_command = (
            str(python_launch),
            str(pins["recovery_script"]),
            "state-probe",
            "--plan",
            str(plan_path),
            "--expect-plan-sha256",
            plan_sha256,
            "--claim",
            str(claim_path),
            "--expect-claim-sha256",
            claim_sha256,
        )
        state_result = command_runner.run(
            state_command,
            cwd=CALIBRATION_ROOT.resolve(),
            phase="T11-recovery:passive-state",
            env=environment,
        )
        logs["robot_state_probe"] = _log(
            artifacts, "passive-robot-state.log", state_result
        )
        state_metrics = _parse_state_probe(
            state_result,
            plan_path=plan_path,
            plan_sha256=plan_sha256,
            claim_path=claim_path,
            claim_sha256=claim_sha256,
        )
        fresh_metrics["passive_robot_state"] = state_metrics
        training._ensure_unchanged(DATASET, snapshot.sha256, phase)

        phase = "fresh_inspection"
        _recheck_immutable_plan_inputs(plan_path, plan_sha256, plan)
        training._ensure_unchanged(DATASET, snapshot.sha256, phase)
        training._ensure_unchanged(claim_path, claim_sha256, phase)
        inspection_raw = artifacts / "fresh-inspection-raw.png"
        inspection_annotated = artifacts / "fresh-inspection-annotated.png"
        inspection_command = (
            str(python_launch),
            str(pins["inspection_script"]),
            "--config",
            str(pins["capture_config"]),
            "--camera-serial",
            str(expected_metadata["camera_serial"]),
            "--raw-output",
            str(inspection_raw),
            "--annotated-output",
            str(inspection_annotated),
            "--frames",
            str(training.INSPECTION_FRAMES),
            "--dictionary",
            str(target["dictionary"]),
        )
        inspection_result = command_runner.run(
            inspection_command,
            cwd=CALIBRATION_ROOT.resolve(),
            phase="T11-recovery:fresh-inspection",
            env=environment,
        )
        logs["fresh_inspection"] = _log(
            artifacts, "fresh-inspection.log", inspection_result
        )
        if inspection_result.returncode == 0:
            for image_path in (inspection_raw, inspection_annotated):
                if image_path.exists():
                    os.chmod(str(image_path), 0o444)
        inspection_metrics = training._parse_inspection(
            inspection_result,
            expected_serial=str(expected_metadata["camera_serial"]),
            target=target,
            dataset_camera=dataset_camera,
            raw_image=inspection_raw,
            annotated_image=inspection_annotated,
        )
        fresh_metrics["inspection"] = inspection_metrics
        training._ensure_unchanged(DATASET, snapshot.sha256, phase)

        phase = "fresh_telemetry"
        _recheck_immutable_plan_inputs(plan_path, plan_sha256, plan)
        training._ensure_unchanged(DATASET, snapshot.sha256, phase)
        training._ensure_unchanged(claim_path, claim_sha256, phase)
        telemetry_path = artifacts / "fresh-frame-telemetry.json"
        telemetry_command = (
            str(python_launch),
            str(pins["frame_telemetry_script"]),
            "--config",
            str(pins["capture_config"]),
            "--camera-serial",
            str(expected_metadata["camera_serial"]),
            "--dictionary",
            str(target["dictionary"]),
            "--marker-id",
            str(target["marker_id"]),
            "--marker-length-m",
            "{:.12g}".format(float(target["marker_length_m"])),
            "--frames",
            str(training.TELEMETRY_FRAMES),
            "--output",
            str(telemetry_path),
            "--max-reprojection-error-px",
            "{:.12g}".format(training.MAX_CAPTURE_REPROJECTION_P95_PX),
            "--max-translation-deviation-m",
            "{:.12g}".format(training.MAX_CAPTURE_TARGET_TRANSLATION_JITTER_M),
            "--max-rotation-deviation-deg",
            "{:.12g}".format(training.MAX_CAPTURE_TARGET_ROTATION_JITTER_DEG),
        )
        telemetry_result = command_runner.run(
            telemetry_command,
            cwd=CALIBRATION_ROOT.resolve(),
            phase="T11-recovery:fresh-telemetry",
            env=environment,
        )
        logs["fresh_telemetry"] = _log(
            artifacts, "fresh-telemetry.log", telemetry_result
        )
        telemetry_metrics = training._parse_frame_telemetry(
            telemetry_result,
            telemetry_path=telemetry_path,
            expected_serial=str(expected_metadata["camera_serial"]),
            target=target,
            dataset_camera=dataset_camera,
            capture_config=pins["capture_config"],
        )
        os.chmod(str(telemetry_path), 0o444)
        fresh_metrics["frame_telemetry"] = telemetry_metrics
        training._ensure_unchanged(DATASET, snapshot.sha256, phase)
        training._ensure_unchanged(
            telemetry_path, str(telemetry_metrics["sha256"]), phase
        )

        phase = "formal_capture"
        _recheck_immutable_plan_inputs(plan_path, plan_sha256, plan)
        training._ensure_unchanged(DATASET, snapshot.sha256, phase)
        training._ensure_unchanged(claim_path, claim_sha256, phase)
        training._ensure_unchanged(
            telemetry_path, str(telemetry_metrics["sha256"]), phase
        )
        debug_image = artifacts / "formal-capture-sample-011.png"
        capture_command = (
            str(python_launch),
            "-m",
            "dynamic_pcd.apps.calibrate_eye_to_hand",
            "capture",
            "--config",
            str(pins["capture_config"]),
            "--robot-ip",
            str(expected_metadata["robot_ip"]),
            "--camera-serial",
            str(expected_metadata["camera_serial"]),
            "--target-type",
            str(target["type"]),
            "--dictionary",
            str(target["dictionary"]),
            "--marker-id",
            str(target["marker_id"]),
            "--marker-length",
            "{:.12g}".format(float(target["marker_length_m"])),
            "--frames",
            str(training.CAPTURE_FRAMES),
            "--min-valid-frame-fraction",
            "0.95",
            "--min-valid-frames",
            str(training.MIN_CAPTURE_COHERENT_FRAMES),
            "--max-reprojection-error",
            "{:.12g}".format(training.MAX_CAPTURE_REPROJECTION_P95_PX),
            "--max-target-translation-jitter",
            "{:.12g}".format(training.MAX_CAPTURE_TARGET_TRANSLATION_JITTER_M),
            "--max-target-rotation-jitter",
            "{:.12g}".format(training.MAX_CAPTURE_TARGET_ROTATION_JITTER_DEG),
            "--min-all-reprojection-pass-frames",
            str(training.MIN_CAPTURE_ALL_REPROJECTION_PASS_FRAMES),
            "--max-all-reprojection-pass-translation-jitter",
            "{:.12g}".format(training.MAX_CAPTURE_TARGET_TRANSLATION_JITTER_M),
            "--max-all-reprojection-pass-rotation-jitter",
            "{:.12g}".format(training.MAX_CAPTURE_TARGET_ROTATION_JITTER_DEG),
            "--max-all-reprojection-pass-reprojection-error",
            "{:.12g}".format(training.MAX_CAPTURE_REPROJECTION_P95_PX),
            "--max-stationary-translation",
            "0.0005",
            "--max-stationary-rotation",
            "0.1",
            "--min-pose-translation",
            "0.01",
            "--min-pose-rotation",
            "3.0",
            "--debug-image",
            str(debug_image),
            "--output",
            str(DATASET.resolve()),
        )
        # Adjacent checks: no operation between these checks and collector spawn.
        _recheck_immutable_plan_inputs(plan_path, plan_sha256, plan)
        training._ensure_unchanged(DATASET, snapshot.sha256, phase)
        training._ensure_unchanged(claim_path, claim_sha256, phase)
        training._ensure_unchanged(
            telemetry_path, str(telemetry_metrics["sha256"]), phase
        )
        capture_result = command_runner.run(
            capture_command,
            cwd=CALIBRATION_ROOT.resolve(),
            phase="T11-recovery:formal-capture",
            env=environment,
        )
        logs["formal_capture"] = _log(
            artifacts, "formal-capture.log", capture_result
        )
        capture_metrics = training._parse_capture_output(capture_result, expected_count=11)
        if debug_image.exists():
            os.chmod(str(debug_image), 0o444)
        debug = training._decode_image(
            debug_image, "formal capture debug image", phase="formal_capture"
        )
        after = training._dataset_snapshot(DATASET)
        target_T = training._rigid_transform(
            transaction.get("target_T_base_ee"), "recovery target_T_base_ee"
        )
        training._verify_dataset_increment(
            snapshot,
            after,
            expected_metadata=expected_metadata,
            expected_count=11,
            target_T_base_ee=target_T,
        )
        fresh_metrics["formal_capture"] = capture_metrics

        phase = "receipt"
        _recheck_immutable_plan_inputs(plan_path, plan_sha256, plan)
        training._ensure_unchanged(DATASET, after.sha256, phase)
        training._ensure_unchanged(claim_path, claim_sha256, phase)
        for record in logs.values():
            training._ensure_unchanged(
                Path(str(record["path"])), str(record["sha256"]), phase
            )
        for image_record in (
            inspection_metrics["raw_image"],
            inspection_metrics["annotated_image"],
        ):
            training._ensure_unchanged(
                Path(str(image_record["path"])), str(image_record["sha256"]), phase
            )
        training._ensure_unchanged(
            telemetry_path, str(telemetry_metrics["sha256"]), phase
        )
        debug_sha256 = _sha(debug_image)
        training._ensure_unchanged(debug_image, debug_sha256, phase)
        receipt = {
            "schema_version": 1,
            "kind": RECEIPT_KIND,
            "state": "T11_capture_verified_complete_no_motion",
            "completed_at": clock(),
            "recovery_plan": str(plan_path),
            "recovery_plan_sha256": plan_sha256,
            "recovery_claim": str(claim_path),
            "recovery_claim_sha256": claim_sha256,
            "robot_motion_commanded": False,
            "robot_state_read_only": True,
            "target_pose_id": "T11",
            "dataset": str(after.path),
            "dataset_before": {"sha256": snapshot.sha256, "sample_count": 10},
            "dataset_after": {"sha256": after.sha256, "sample_count": 11},
            "dataset_append_invariants": {
                "exactly_one_sample_appended": True,
                "preexisting_10_sample_prefix_byte_semantics_preserved": True,
                "stable_metadata_preserved": True,
                "stable_metadata_fields": [
                    "camera",
                    "target",
                    "robot",
                    "transform_convention",
                    "units",
                ],
            },
            "original_motion_reused_as_motion_evidence_only": True,
            "fresh_gates": fresh_metrics,
            "formal_capture_debug_image": {
                "path": str(debug_image),
                "sha256": debug_sha256,
                "width": int(debug.shape[1]),
                "height": int(debug.shape[0]),
            },
            "import_resolution": import_resolution,
            "logs": logs,
            "subprocess_contract": {
                "robot_motion_wrapper_imported_for_passive_state": True,
                "robot_motion_command_invoked": False,
                "move_api_invoked": False,
                "passive_state_probe_invoked": True,
                "camera_inspection_invoked": True,
                "frame_telemetry_invoked": True,
                "formal_capture_invoked": True,
            },
        }
        training._publish_exclusive(receipt_path, training._json_bytes(receipt))
        print(
            "[T11 recovery] COMPLETE sample=11 dataset_sha256={} receipt={}".format(
                after.sha256, receipt_path
            ),
            flush=True,
        )
        return RecoveryResult(after.sha256, 11, receipt_path)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        failure = exc if isinstance(exc, training.SequenceFailure) else training.SequenceFailure(phase, str(exc))
        if claim_sha256 is not None:
            payload = {
                "schema_version": 1,
                "kind": "franka_eye_to_hand_capture_only_recovery_failure",
                "state": "failed_closed_no_automatic_retry",
                "failed_at": clock(),
                "phase": failure.phase,
                "error": str(failure),
                "recovery_plan": str(plan_path),
                "recovery_plan_sha256": plan_sha256,
                "recovery_claim": str(claim_path),
                "recovery_claim_sha256": claim_sha256,
                "robot_motion_commanded": False,
                "logs": logs,
                **training._safe_failure_state(DATASET),
            }
            try:
                training._publish_exclusive(failure_path, training._json_bytes(payload))
            except (OSError, ValueError) as receipt_error:
                print("[T11 recovery] failure receipt error: {}".format(receipt_error), file=sys.stderr)
        raise failure


def _validated_recovery_receipt(
    receipt_path: Path,
    expected_receipt_sha256: str,
    plan_path: Path,
    plan_sha256: str,
) -> Tuple[Mapping[str, Any], training.DatasetSnapshot]:
    resolved, data, receipt = training._read_json_mapping(
        receipt_path, maximum_bytes=MAX_FILE_BYTES
    )
    expected_receipt_path = (
        RECOVERY_ARTIFACTS / "capture-only-recovery.complete.json"
    ).resolve()
    if resolved != expected_receipt_path:
        raise ValueError("T11 recovery receipt is not the fixed success receipt")
    if (RECOVERY_ARTIFACTS / "capture-only-recovery.failed.json").exists():
        raise ValueError("T11 recovery failure receipt exists")
    digest = training._sha256_bytes(data)
    if digest != expected_receipt_sha256:
        raise ValueError("T11 recovery receipt SHA mismatch")
    if (
        receipt.get("kind") != RECEIPT_KIND
        or receipt.get("state") != "T11_capture_verified_complete_no_motion"
        or receipt.get("robot_motion_commanded") is not False
        or receipt.get("robot_state_read_only") is not True
        or receipt.get("target_pose_id") != "T11"
    ):
        raise ValueError("T11 recovery receipt contract mismatch")
    if (
        Path(str(receipt.get("recovery_plan"))).resolve() != plan_path.resolve()
        or receipt.get("recovery_plan_sha256") != plan_sha256
    ):
        raise ValueError("T11 recovery receipt plan provenance mismatch")
    before = training._require_mapping(receipt.get("dataset_before"), "dataset_before")
    after_record = training._require_mapping(receipt.get("dataset_after"), "dataset_after")
    if before != {"sha256": DATASET_10_SHA256, "sample_count": 10}:
        raise ValueError("T11 recovery receipt before-state mismatch")
    if after_record.get("sample_count") != 11:
        raise ValueError("T11 recovery receipt must report 11 samples")
    snapshot = training._dataset_snapshot(DATASET)
    if (
        len(snapshot.samples) != 11
        or snapshot.sha256 != after_record.get("sha256")
        or Path(str(receipt.get("dataset"))).resolve() != snapshot.path
    ):
        raise ValueError("live dataset does not match T11 recovery receipt")
    claim = Path(str(receipt.get("recovery_claim"))).resolve(strict=True)
    if claim != RECOVERY_CLAIM.resolve():
        raise ValueError("T11 recovery receipt references a noncanonical claim")
    if _sha(claim) != receipt.get("recovery_claim_sha256"):
        raise ValueError("T11 recovery claim changed")
    fresh = training._require_mapping(receipt.get("fresh_gates"), "fresh_gates")
    for field in ("passive_robot_state", "inspection", "frame_telemetry", "formal_capture"):
        if field not in fresh:
            raise ValueError("T11 recovery receipt lacks {}".format(field))
    append_invariants = training._require_mapping(
        receipt.get("dataset_append_invariants"), "dataset_append_invariants"
    )
    expected_append = {
        "exactly_one_sample_appended": True,
        "preexisting_10_sample_prefix_byte_semantics_preserved": True,
        "stable_metadata_preserved": True,
        "stable_metadata_fields": [
            "camera",
            "target",
            "robot",
            "transform_convention",
            "units",
        ],
    }
    if append_invariants != expected_append:
        raise ValueError("T11 recovery append invariants mismatch")
    state = training._require_mapping(fresh.get("passive_robot_state"), "passive_robot_state")
    for field, expected in (
        ("robot_motion_commanded", False),
        ("robot_state_read_only", True),
        ("robot_mode", "Idle"),
        ("current_errors_active", False),
        ("contacts_or_collisions_active", False),
        ("pose_id", "T11"),
        ("pose_gate_pass", True),
    ):
        if state.get(field) != expected:
            raise ValueError("T11 recovery passive-state field {} mismatch".format(field))
    if training._finite_float(state.get("translation_error_m"), "translation_error_m") > T11_TRANSLATION_TOLERANCE_M:
        raise ValueError("T11 recovery passive translation gate failed")
    if training._finite_float(state.get("rotation_error_deg"), "rotation_error_deg") > T11_ROTATION_TOLERANCE_DEG:
        raise ValueError("T11 recovery passive rotation gate failed")
    if training._finite_float(state.get("max_abs_dq_rad_s"), "max_abs_dq_rad_s") > 0.005:
        raise ValueError("T11 recovery passive stationary gate failed")
    training._rigid_transform(state.get("T_base_ee"), "receipt passive T_base_ee")

    inspection = training._require_mapping(fresh.get("inspection"), "inspection")
    if (
        inspection.get("camera_serial") != "342222071785"
        or training._finite_float(inspection.get("minimum_image_margin_px"), "margin")
        < training.MIN_LIVE_MARKER_MARGIN_PX
        or training._finite_float(inspection.get("minimum_marker_edge_px"), "edge")
        < training.MIN_LIVE_MARKER_EDGE_PX
    ):
        raise ValueError("T11 recovery fresh inspection gates failed")
    for name, expected_name in (
        ("raw_image", "fresh-inspection-raw.png"),
        ("annotated_image", "fresh-inspection-annotated.png"),
    ):
        image = training._require_mapping(inspection.get(name), "inspection." + name)
        image_path = Path(str(image.get("path"))).resolve(strict=True)
        if image_path != (RECOVERY_ARTIFACTS / expected_name).resolve() or _sha(image_path) != image.get("sha256"):
            raise ValueError("T11 recovery inspection image provenance failed")

    telemetry = training._require_mapping(fresh.get("frame_telemetry"), "frame_telemetry")
    telemetry_path = Path(str(telemetry.get("path"))).resolve(strict=True)
    if (
        telemetry_path != (RECOVERY_ARTIFACTS / "fresh-frame-telemetry.json").resolve()
        or _sha(telemetry_path) != telemetry.get("sha256")
        or telemetry.get("camera_serial") != "342222071785"
        or telemetry.get("requested_frame_count") != 120
        or telemetry.get("captured_frame_count") != 120
        or int(telemetry.get("detected_valid_count", -1)) < 118
        or int(telemetry.get("reprojection_gate_pass_count", -1)) < 118
        or int(telemetry.get("coherent_pose_count", -1)) < 114
    ):
        raise ValueError("T11 recovery fresh telemetry gates/provenance failed")
    capture = training._require_mapping(fresh.get("formal_capture"), "formal_capture")
    untrimmed = training._require_mapping(
        capture.get("formal_same_batch_untrimmed"), "formal_same_batch_untrimmed"
    )
    if (
        capture.get("sample_count") != 11
        or capture.get("requested_frames") != 120
        or int(capture.get("coherent_frames", -1)) < 114
        or untrimmed.get("requested_frame_count") != 120
        or int(untrimmed.get("all_reprojection_pass_count", -1)) < 118
    ):
        raise ValueError("T11 recovery formal capture gates failed")
    debug = training._require_mapping(
        receipt.get("formal_capture_debug_image"), "formal_capture_debug_image"
    )
    debug_path = Path(str(debug.get("path"))).resolve(strict=True)
    if (
        debug_path != (RECOVERY_ARTIFACTS / "formal-capture-sample-011.png").resolve()
        or _sha(debug_path) != debug.get("sha256")
    ):
        raise ValueError("T11 recovery debug-image provenance failed")
    contract = training._require_mapping(receipt.get("subprocess_contract"), "subprocess_contract")
    if contract != {
        "robot_motion_wrapper_imported_for_passive_state": True,
        "robot_motion_command_invoked": False,
        "move_api_invoked": False,
        "passive_state_probe_invoked": True,
        "camera_inspection_invoked": True,
        "frame_telemetry_invoked": True,
        "formal_capture_invoked": True,
    }:
        raise ValueError("T11 recovery subprocess contract mismatch")
    logs = training._require_mapping(receipt.get("logs"), "logs")
    if set(logs) != {
        "import_preflight",
        "robot_state_probe",
        "fresh_inspection",
        "fresh_telemetry",
        "formal_capture",
    }:
        raise ValueError("T11 recovery log set mismatch")
    for name, record_value in logs.items():
        record = training._require_mapping(record_value, "logs." + name)
        log_path = Path(str(record.get("path"))).resolve(strict=True)
        try:
            log_path.relative_to(RECOVERY_ARTIFACTS.resolve())
        except ValueError as exc:
            raise ValueError("T11 recovery log path escapes artifacts") from exc
        if _sha(log_path) != record.get("sha256") or record.get("returncode") != 0:
            raise ValueError("T11 recovery log provenance failed")
    return receipt, snapshot


def prepare_resume_master(args: argparse.Namespace) -> Mapping[str, Any]:
    plan_path, plan_data, recovery_plan = training._read_yaml(
        args.recovery_plan, maximum_bytes=training.MAX_PLAN_BYTES
    )
    plan_sha256 = training._sha256_bytes(plan_data)
    if plan_sha256 != args.expect_recovery_plan_sha256:
        raise ValueError("recovery plan SHA mismatch")
    if recovery_plan.get("kind") != PLAN_KIND:
        raise ValueError("recovery plan kind mismatch")
    _recheck_immutable_plan_inputs(plan_path, plan_sha256, recovery_plan)
    receipt, snapshot = _validated_recovery_receipt(
        args.recovery_receipt,
        args.expect_recovery_receipt_sha256,
        plan_path,
        plan_sha256,
    )
    original_master, _ = _load_master()
    expected_metadata = training._expected_dataset_metadata(original_master)
    training._validate_dataset_metadata(snapshot, expected_metadata)
    poses = training._pose_map(original_master)
    # Validate the unchanged 10-sample prefix and the newly appended T11 sample.
    prefix = training.DatasetSnapshot(
        snapshot.path,
        DATASET_10_SHA256,
        snapshot.document,
        snapshot.samples[:10],
    )
    training._validate_existing_sample_prefix(prefix, master=original_master, poses=poses)
    training._validate_new_sample(snapshot.samples[10], _planned_t11(original_master))

    output = args.output_master.expanduser().resolve()
    artifacts = args.artifacts_dir.expanduser().resolve()
    if output != RESUME_MASTER.resolve():
        raise ValueError("resume master path must be the fixed versioned v1 path")
    if artifacts != RESUME_ARTIFACTS.resolve():
        raise ValueError("resume artifacts path must be the fixed versioned v1 path")
    if output.exists() or artifacts.exists():
        raise FileExistsError("new resume master and artifacts directory must be absent")
    if output == ORIGINAL_MASTER.resolve() or artifacts == ORIGINAL_ARTIFACTS.resolve():
        raise ValueError("resume master/artifacts must be new versioned paths")
    output.relative_to(CALIBRATION_ROOT.resolve())
    artifacts.relative_to(CALIBRATION_ROOT.resolve())

    resumed = copy.deepcopy(dict(original_master))
    resumed["session_slug"] = "fr3-d435-342222071785-20260812-002810-remount5-training-resume-from-T11-v1"
    provenance = copy.deepcopy(
        dict(training._require_mapping(resumed.get("provenance"), "provenance"))
    )
    provenance.update(
        {
            "purpose": "formal_remount5_training_pose_collection_resumed_after_T11_capture_only_ENOSPC_recovery",
            "master_plan_path": str(output),
            "source_original_training_master": str(ORIGINAL_MASTER.resolve()),
            "source_original_training_master_sha256": ORIGINAL_MASTER_SHA256,
            "T11_capture_only_recovery_plan": str(plan_path),
            "T11_capture_only_recovery_plan_sha256": plan_sha256,
            "T11_capture_only_recovery_receipt": str(args.recovery_receipt.resolve()),
            "T11_capture_only_recovery_receipt_sha256": args.expect_recovery_receipt_sha256,
            "T11_recovered_dataset_sha256": snapshot.sha256,
            "T11_recovery_robot_motion_commanded": False,
            "prior_master_or_artifacts_overwritten": False,
        }
    )
    resumed["provenance"] = provenance
    resumed["status"] = training.MASTER_RUN_STATUS
    remaining = tuple("T{:02d}".format(index) for index in range(12, 21))
    edges = tuple(
        "{}_to_{}".format(start, target)
        for start, target in zip(("T11",) + remaining[:-1], remaining)
    )
    old_authorization = training._require_mapping(
        original_master.get("motion_authorization"), "motion_authorization"
    )
    resumed["motion_authorization"] = {
        "explicit_user_authorization_recorded": True,
        "scope": training.MASTER_AUTHORIZATION_SCOPE,
        "start_pose_id": "T11",
        "pose_id": "T12",
        "purpose": "formal_training_sample_012_through_020_after_capture_only_recovery",
        "source_text": training._require_text(
            old_authorization.get("source_text"), "source authorization text"
        ),
        "authorization_interpretation": (
            "continue_the_original_reviewed_bounded_remount5_training_suffix_"
            "T12_through_T20_after_verified_no_motion_T11_capture_recovery;_"
            "every_derived_wrapper_run_remains_SHA_bound_single_pose_and_fail_closed"
        ),
        "authorized_training_suffix": list(remaining),
        "authorized_edges": list(edges),
        "authorized_at_local": old_authorization.get("authorized_at_local"),
        "consumed": False,
    }
    resumed["collection_state"] = {
        "current_pose_id": "T11",
        "current_training_sample_count": 11,
        "next_pose_id": "T12",
        "next_capture_index": 12,
        "formal_training_dataset": str(DATASET.resolve().relative_to(CALIBRATION_ROOT.resolve())),
        "initial_dataset_sha256": snapshot.sha256,
        "training_sequence_artifacts_dir": str(
            artifacts.relative_to(CALIBRATION_ROOT.resolve())
        ),
    }
    training_poses = resumed.get("training_poses")
    if not isinstance(training_poses, list) or len(training_poses) != 20:
        raise ValueError("original training pose table changed")
    for index, raw in enumerate(training_poses, start=1):
        pose = training._require_mapping(raw, "training_poses[{}]".format(index - 1))
        if pose.get("id") != "T{:02d}".format(index):
            raise ValueError("training pose order changed")
        if index <= 11:
            pose["state"] = (
                "captured_via_T11_no_motion_ENOSPC_recovery"
                if index == 11
                else "captured_original_sequence"
            )
            pose["sample_index"] = index
        else:
            pose["state"] = "planned_fresh_remount5_resume_v1"
            pose.pop("sample_index", None)
    history = resumed.get("motion_history")
    if not isinstance(history, list):
        raise ValueError("motion_history must be a list")
    history.append(
        {
            "event": "T11_capture_only_ENOSPC_recovery",
            "robot_motion_commanded": False,
            "recovery_receipt": str(args.recovery_receipt.resolve()),
            "recovery_receipt_sha256": args.expect_recovery_receipt_sha256,
            "dataset_sample_count_after": 11,
            "dataset_sha256_after": snapshot.sha256,
        }
    )

    payload = yaml.safe_dump(resumed, sort_keys=False, allow_unicode=True).encode("utf-8")
    if len(payload) > training.MAX_PLAN_BYTES:
        raise ValueError("resume master is too large")
    master_sha256 = training._sha256_bytes(payload)
    code = _canonical_code_paths()
    python_launch = training._lexical_absolute_path(args.python_executable, "python executable")
    python_resolved = python_launch.resolve(strict=True)
    claims_dir = (RUNS / ".training-edge-claims").resolve()
    training._validate_master(
        resumed,
        master_sha256=master_sha256,
        master_path=output,
        calibration_root=CALIBRATION_ROOT.resolve(),
        claims_dir=claims_dir,
        dataset=DATASET.resolve(),
        artifacts_dir=artifacts,
        orchestrator_sha256=_sha(code["training_orchestrator"]),
        python_executable=python_launch,
        python_executable_resolved=python_resolved,
        python_executable_sha256=_sha(python_resolved),
        motion_script_sha256=_sha(code["motion_wrapper"]),
        motion_driver_sha256=_sha(code["motion_driver"]),
        link_preflight_sha256=_sha(code["link_preflight"]),
        inspection_script_sha256=_sha(code["inspection_script"]),
        detector_sha256=_sha(code["detector"]),
        stationary_sha256=_sha(code["stationary"]),
        transforms_sha256=_sha(code["transforms"]),
        capture_config_sha256=_sha(code["capture_config"]),
        capture_cli_sha256=_sha(code["capture_cli"]),
        frame_telemetry_sha256=_sha(code["frame_telemetry_script"]),
    )
    training._validate_existing_sample_prefix(snapshot, master=resumed, poses=training._pose_map(resumed))
    first_claim = claims_dir / (
        training._claim_stem("T11", "T12", master_sha256) + ".claim.json"
    )
    if first_claim.exists():
        raise FileExistsError("new-master first edge claim unexpectedly exists")
    training._publish_exclusive(output, payload)
    result = {
        "master_plan": str(output),
        "master_plan_sha256": master_sha256,
        "current_pose_id": "T11",
        "current_sample_count": 11,
        "remaining_suffix": list(remaining),
        "authorized_edges": list(edges),
        "dataset": str(snapshot.path),
        "dataset_sha256": snapshot.sha256,
        "artifacts_dir": str(artifacts),
        "hardware_opened": False,
        "robot_motion_commanded": False,
    }
    print("T11_RESUME_MASTER_JSON=" + json.dumps(result, sort_keys=True, separators=(",", ":")))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="offline immutable recovery-plan creation")
    prepare.add_argument("--output-plan", type=Path, default=RECOVERY_PLAN)
    prepare.add_argument("--artifacts-dir", type=Path, default=RECOVERY_ARTIFACTS)
    prepare.add_argument("--claim-path", type=Path, default=RECOVERY_CLAIM)
    prepare.add_argument("--python", dest="python_executable", default=str(ROOT / ".venv/bin/python"))
    prepare.add_argument("--minimum-free-bytes", type=int, default=MIN_FREE_BYTES)
    prepare.add_argument(
        "--confirm-capture-only-no-motion", metavar=RECOVERY_TOKEN, required=True
    )

    run = subparsers.add_parser("run", help="consume one claim and recover only sample T11")
    run.add_argument("--plan", type=Path, default=RECOVERY_PLAN)
    run.add_argument("--expect-plan-sha256", required=True)
    run.add_argument("--python", dest="python_executable", default=str(ROOT / ".venv/bin/python"))

    state = subparsers.add_parser("state-probe", help=argparse.SUPPRESS)
    state.add_argument("--plan", type=Path, required=True)
    state.add_argument("--expect-plan-sha256", required=True)
    state.add_argument("--claim", type=Path, required=True)
    state.add_argument("--expect-claim-sha256", required=True)

    resume = subparsers.add_parser(
        "prepare-resume-master", help="offline new T12..T20 master after verified T11 recovery"
    )
    resume.add_argument("--recovery-plan", type=Path, default=RECOVERY_PLAN)
    resume.add_argument("--expect-recovery-plan-sha256", required=True)
    resume.add_argument("--recovery-receipt", type=Path, required=True)
    resume.add_argument("--expect-recovery-receipt-sha256", required=True)
    resume.add_argument("--output-master", type=Path, default=RESUME_MASTER)
    resume.add_argument("--artifacts-dir", type=Path, default=RESUME_ARTIFACTS)
    resume.add_argument("--python", dest="python_executable", default=str(ROOT / ".venv/bin/python"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            prepare_recovery_plan(args)
        elif args.command == "run":
            run_recovery(
                RecoveryConfig(
                    plan=args.plan,
                    expected_plan_sha256=args.expect_plan_sha256,
                    python_executable=args.python_executable,
                )
            )
        elif args.command == "state-probe":
            passive_state_probe(
                args.plan,
                args.expect_plan_sha256,
                args.claim,
                args.expect_claim_sha256,
            )
        elif args.command == "prepare-resume-master":
            prepare_resume_master(args)
        else:  # pragma: no cover
            raise ValueError("unsupported command")
    except KeyboardInterrupt:
        print("[T11 recovery] interrupted; automatic retry forbidden", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        phase = getattr(exc, "phase", "preflight")
        print(
            "[T11 recovery] FAILED_CLOSED phase={} error={}".format(phase, exc),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
