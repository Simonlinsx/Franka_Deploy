#!/usr/bin/env python3
"""Offline-only new T15..T20 master after a Desk pre-motion block."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path("/home/qiaoguanren/code/franka")
SCRIPTS = ROOT / "skills/calibrate-franka-eye-to-hand/scripts"
sys.path.insert(0, str(SCRIPTS))
import run_training_sequence as training  # noqa: E402
import recover_t11_capture_only as recovery  # noqa: E402

CAL_ROOT = ROOT / "beta/dynamic_object_pcd"
RUNS = CAL_ROOT / "calibration_runs"
SOURCE = RUNS / "fr3-d435-342222071785-20260812-002810-remount5-training-resume-from-T14-v2-plan.yaml"
SOURCE_SHA = "b9b241de783e67acb0132a2c1db235c33eadf9d37611c0283b696bbf256fbbca"
DATASET = RUNS / "fr3-d435-342222071785-20260812-002810-remount5-training.yaml"
DATASET_SHA = "c8108d4e27e702938494340024c8c5e9027f0c470bc35f9c95634fdc8c2b40d9"
OLD_DIR = RUNS / "fr3-d435-342222071785-20260812-002810-remount5-training-resume-from-T14-v2"
OLD_PLAN = OLD_DIR / "edges/T14-to-T15-110fea27c7c931f3.plan.yaml"
OLD_CLAIM = RUNS / ".training-edge-claims/b9b241de783e67acb0132a2c1db235c33eadf9d37611c0283b696bbf256fbbca-T14-to-T15.claim.json"
OLD_PREVIEW = OLD_DIR / "edges/T14-to-T15-110fea27c7c931f3.preview.log"
OLD_MOTION = OLD_DIR / "edges/T14-to-T15-110fea27c7c931f3.motion.log"
OLD_FAILED = OLD_DIR / "edges/T14-to-T15-110fea27c7c931f3.failed.json"
PINS = {
    OLD_PLAN: "110fea27c7c931f33bd085117e6bb4e42d85928dcecaa7c6e6850a04e1f772d8",
    OLD_CLAIM: "c5c023f985f51eb96550c99fa751e9c129144cd4569ff48ed899f1a1aa94aaf1",
    OLD_PREVIEW: "66f420b58e6b3e3e8021de6dec4b367aabafa7d6cf94e4ab1b67df03f8a2769d",
    OLD_MOTION: "b5fa3c0de90c98c5b4d5b5679329b9479a1b0436e6840f515ffcecc365e24c68",
    OLD_FAILED: "f5aa91a8d4ffde970aff8593b1009873f6fde3a29d099e23822e479cdfc16d5e",
}
OUTPUT = RUNS / "fr3-d435-342222071785-20260812-002810-remount5-training-resume-from-T14-v3-plan.yaml"
ARTIFACTS = RUNS / "fr3-d435-342222071785-20260812-002810-remount5-training-resume-from-T14-v3"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if OUTPUT.exists() or ARTIFACTS.exists():
        raise FileExistsError("fixed v3 paths already exist")
    if sha(SOURCE) != SOURCE_SHA or sha(DATASET) != DATASET_SHA:
        raise ValueError("source master or 14-sample dataset changed")
    for path, digest in PINS.items():
        if not path.is_file() or sha(path) != digest:
            raise ValueError("Desk-block evidence changed: {}".format(path))
    if any((OLD_DIR / ("edges/T14-to-T15-110fea27c7c931f3" + suffix)).exists()
           for suffix in (".inspect.log", ".telemetry.log", ".capture.log", ".complete.json")):
        raise ValueError("Desk block was not strictly pre-inspection/pre-capture")
    _, _, failed = training._read_json_mapping(OLD_FAILED, maximum_bytes=1_000_000)
    exact = {
        "kind": "franka_eye_to_hand_training_edge_failure",
        "state": "failed_closed_no_automatic_retry",
        "phase": "motion",
        "edge": "T14_to_T15",
        "dataset_sha256": DATASET_SHA,
        "dataset_sample_count": 14,
        "single_edge_plan_sha256": PINS[OLD_PLAN],
    }
    for key, value in exact.items():
        if failed.get(key) != value:
            raise ValueError("failure receipt mismatch: {}".format(key))
    text = OLD_MOTION.read_text()
    if "motion blocked: found 37 ESTABLISHED TCP connection(s)" not in text or "CAPTURE_READY" in text:
        raise ValueError("motion log does not prove Desk pre-motion rejection")

    _, source_bytes, master = training._read_yaml(SOURCE, maximum_bytes=training.MAX_PLAN_BYTES)
    if training._sha256_bytes(source_bytes) != SOURCE_SHA:
        raise ValueError("source master bytes changed")
    snapshot = training._dataset_snapshot(DATASET)
    if snapshot.sha256 != DATASET_SHA or len(snapshot.samples) != 14:
        raise ValueError("dataset is not exact 14-sample state")
    training._validate_dataset_metadata(snapshot, training._expected_dataset_metadata(master))
    training._validate_existing_sample_prefix(snapshot, master=master, poses=training._pose_map(master))

    resumed = copy.deepcopy(dict(master))
    resumed["session_slug"] = "fr3-d435-342222071785-20260812-002810-remount5-training-resume-from-T14-v3"
    provenance = dict(training._require_mapping(resumed.get("provenance"), "provenance"))
    provenance.update({
        "purpose": "formal_remount5_training_resumed_after_verified_Franka_Desk_pre_motion_block",
        "master_plan_path": str(OUTPUT.resolve()),
        "source_T14_v2_master": str(SOURCE.resolve()),
        "source_T14_v2_master_sha256": SOURCE_SHA,
        "Desk_block_failure_receipt": str(OLD_FAILED.resolve()),
        "Desk_block_failure_receipt_sha256": PINS[OLD_FAILED],
        "Desk_block_occurred_before_motion": True,
        "dataset_unchanged_after_Desk_block": True,
        "prior_files_overwritten": False,
    })
    resumed["provenance"] = provenance
    resumed["status"] = training.MASTER_RUN_STATUS
    remaining = tuple("T{:02d}".format(i) for i in range(15, 21))
    edges = tuple("{}_to_{}".format(a, b) for a, b in zip(("T14",) + remaining[:-1], remaining))
    old = training._require_mapping(master.get("motion_authorization"), "motion_authorization")
    resumed["motion_authorization"] = {
        "explicit_user_authorization_recorded": True,
        "scope": training.MASTER_AUTHORIZATION_SCOPE,
        "start_pose_id": "T14", "pose_id": "T15",
        "purpose": "formal_training_sample_015_through_020_after_Desk_closed",
        "source_text": training._require_text(old.get("source_text"), "authorization source"),
        "authorization_interpretation": "continue_reviewed_T15_through_T20_after_operator_closed_all_Franka_Desk_connections;_every_edge_remains_SHA_bound_and_fail_closed",
        "authorized_training_suffix": list(remaining),
        "authorized_edges": list(edges),
        "authorized_at_local": old.get("authorized_at_local"),
        "consumed": False,
    }
    resumed["collection_state"] = {
        "current_pose_id": "T14", "current_training_sample_count": 14,
        "next_pose_id": "T15", "next_capture_index": 15,
        "formal_training_dataset": str(DATASET.relative_to(CAL_ROOT)),
        "initial_dataset_sha256": snapshot.sha256,
        "training_sequence_artifacts_dir": str(ARTIFACTS.relative_to(CAL_ROOT)),
    }
    history = resumed.get("motion_history")
    if not isinstance(history, list):
        raise ValueError("motion history changed")
    history.append({
        "event": "T14_to_T15_blocked_before_motion_by_Franka_Desk_connections",
        "robot_motion_commanded": False,
        "dataset_sample_count_after": 14,
        "dataset_sha256_after": snapshot.sha256,
        "failure_receipt_sha256": PINS[OLD_FAILED],
    })
    payload = yaml.safe_dump(resumed, sort_keys=False, allow_unicode=True).encode()
    digest = training._sha256_bytes(payload)
    code = recovery._canonical_code_paths()
    py_launch = ROOT / ".venv/bin/python"
    py_resolved = py_launch.resolve(strict=True)
    claims = (RUNS / ".training-edge-claims").resolve()
    training._validate_master(
        resumed, master_sha256=digest, master_path=OUTPUT.resolve(),
        calibration_root=CAL_ROOT.resolve(), claims_dir=claims,
        dataset=DATASET.resolve(), artifacts_dir=ARTIFACTS.resolve(),
        orchestrator_sha256=sha(code["training_orchestrator"]),
        python_executable=py_launch, python_executable_resolved=py_resolved,
        python_executable_sha256=sha(py_resolved),
        motion_script_sha256=sha(code["motion_wrapper"]),
        motion_driver_sha256=sha(code["motion_driver"]),
        link_preflight_sha256=sha(code["link_preflight"]),
        inspection_script_sha256=sha(code["inspection_script"]),
        detector_sha256=sha(code["detector"]),
        stationary_sha256=sha(code["stationary"]),
        transforms_sha256=sha(code["transforms"]),
        capture_config_sha256=sha(code["capture_config"]),
        capture_cli_sha256=sha(code["capture_cli"]),
        frame_telemetry_sha256=sha(code["frame_telemetry_script"]),
    )
    first_claim = claims / (training._claim_stem("T14", "T15", digest) + ".claim.json")
    if first_claim.exists():
        raise FileExistsError("new first edge claim already exists")
    training._publish_exclusive(OUTPUT, payload)
    print("DESK_BLOCK_RESUME_MASTER_JSON=" + json.dumps({
        "master": str(OUTPUT), "master_sha256": digest,
        "dataset_sha256": snapshot.sha256, "samples": 14,
        "artifacts": str(ARTIFACTS), "hardware_opened": False,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
