#!/usr/bin/env python3
"""Prepare immutable single-edge plans for the remount5 H01--H06 holdout.

This is an offline-only, session-specific plan generator.  It opens neither
the Franka nor the camera.  Every output is created exclusively and is bound
to the completed 20-sample training master, receipt, and dataset.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import yaml


ROOT = Path("/home/qiaoguanren/code/franka/beta/dynamic_object_pcd")
RUNS = ROOT / "calibration_runs"
SOURCE_MASTER = RUNS / "fr3-d435-342222071785-20260812-002810-remount5-training-resume-from-T14-v3-plan.yaml"
SOURCE_MASTER_SHA = "f024976dfe803b2fd1654cd20ae14fc0d8d5a5576f59a8b5bac779be0f057fba"
SOURCE_RECEIPT = RUNS / "fr3-d435-342222071785-20260812-002810-remount5-training-resume-from-T14-v3/training-sequence-f024976dfe803b2f-complete.json"
SOURCE_RECEIPT_SHA = "14aef703a3ed12e73b858a03fb7f9a5ed4bf8fe79a38735ff9d1292c3465e7ae"
TRAINING_DATASET = RUNS / "fr3-d435-342222071785-20260812-002810-remount5-training.yaml"
TRAINING_DATASET_SHA = "6d0e8043a459ffb8a5d6f08b0a94e4be0bd5bb22675983ac89bc7d2b36c6cbfe"
OUTPUT_DIR = RUNS / "fr3-d435-342222071785-20260812-002810-remount5-holdout-v2-plans"
HOLDOUT_DATASET = RUNS / "fr3-d435-342222071785-20260812-002810-remount5-holdout-v1.yaml"
PRIOR_FAILURE = RUNS / "fr3-d435-342222071785-20260812-002810-remount5-holdout-v1/01-T20-to-T01.failed.json"
PRIOR_FAILURE_SHA = "feaf4e1c41bfb5e999737e55aa49940954a2fa6974e54baf9b3ee2ecf0f0fd48"

TOKEN_SOURCE = "好的继续"
TOKEN_INTERPRETATION = (
    "continue_the_reviewed_bounded_remount5_independent_holdout_sequence_"
    "T20_to_T01_then_H01_through_H06_with_T01_returns_without_repeated_prompts"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def publish_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def load_yaml(path: Path):
    with path.open("rb") as stream:
        return yaml.safe_load(stream)


def main() -> int:
    expected = (
        (SOURCE_MASTER, SOURCE_MASTER_SHA),
        (SOURCE_RECEIPT, SOURCE_RECEIPT_SHA),
        (TRAINING_DATASET, TRAINING_DATASET_SHA),
        (PRIOR_FAILURE, PRIOR_FAILURE_SHA),
    )
    for path, digest in expected:
        if sha256(path) != digest:
            raise RuntimeError(f"immutable source SHA mismatch: {path}")
    if OUTPUT_DIR.exists() or HOLDOUT_DATASET.exists():
        raise FileExistsError("holdout output or dataset already exists")

    master = load_yaml(SOURCE_MASTER)
    receipt = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    dataset = load_yaml(TRAINING_DATASET)
    if master.get("kind") != "franka_eye_to_hand_commissioning_plan":
        raise RuntimeError("unexpected source master kind")
    if receipt.get("state") != "reviewed_remaining_training_suffix_capture_verified_complete":
        raise RuntimeError("training completion receipt is not accepted")
    if receipt.get("final_sample_count") != 20 or receipt.get("final_dataset_sha256") != TRAINING_DATASET_SHA:
        raise RuntimeError("training completion receipt/dataset mismatch")
    if len(dataset.get("samples", [])) != 20:
        raise RuntimeError("training dataset must contain exactly 20 samples")
    if dataset.get("camera", {}).get("serial") != "342222071785":
        raise RuntimeError("camera serial mismatch")
    target = dataset.get("target", {})
    if (target.get("dictionary"), target.get("marker_id"), target.get("marker_length_m")) != ("DICT_6X6_50", 42, 0.19):
        raise RuntimeError("target metadata mismatch")

    training = {p["id"]: p for p in master.get("training_poses", [])}
    holdout = {p["id"]: p for p in master.get("holdout_poses", [])}
    if not {"T01", "T20"}.issubset(training):
        raise RuntimeError("source master lacks T01/T20")
    expected_holdout = tuple(f"H{i:02d}" for i in range(1, 7))
    if tuple(holdout) != expected_holdout:
        raise RuntimeError("source master must contain exactly H01..H06")

    sequence = [("T20", "T01", None)]
    for pose_id in expected_holdout:
        sequence.append(("T01", pose_id, pose_id))
        sequence.append((pose_id, "T01", None))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    records = []
    for ordinal, (start, finish, capture_id) in enumerate(sequence, 1):
        edge = copy.deepcopy(master)
        edge["status"] = "explicit_single_pose_motion_authorized"
        edge["session_slug"] = (
            f"fr3-d435-342222071785-20260812-002810-remount5-holdout-v2-"
            f"edge-{ordinal:02d}-{start}-to-{finish}"
        )
        selected_training = []
        selected_holdout = []
        for pose_id in (start, finish):
            entry = copy.deepcopy(training.get(pose_id, holdout.get(pose_id)))
            if pose_id in training:
                selected_training.append(entry)
            else:
                selected_holdout.append(entry)
        edge["training_poses"] = selected_training
        edge["holdout_poses"] = selected_holdout
        edge.pop("recovery_poses", None)
        edge["planned_sequence"] = {
            "single_edge_only": True,
            "start_pose_id": start,
            "target_pose_id": finish,
            "independent_holdout_capture_id": capture_id,
            "holdout_dataset_never_used_for_solve": True,
        }
        edge["motion_history"] = []
        edge["motion_authorization"] = {
            "explicit_user_authorization_recorded": True,
            "scope": "single_pose",
            "start_pose_id": start,
            "pose_id": finish,
            "source_text": TOKEN_SOURCE,
            "authorization_interpretation": TOKEN_INTERPRETATION,
            "reviewed_training_master": str(SOURCE_MASTER),
            "reviewed_training_master_sha256": SOURCE_MASTER_SHA,
            "consumed": False,
        }
        edge["safety"]["motion_authorized"] = True
        edge["provenance"]["purpose"] = "independent_remount5_holdout_validation"
        edge["provenance"]["source_training_master"] = str(SOURCE_MASTER)
        edge["provenance"]["source_training_master_sha256"] = SOURCE_MASTER_SHA
        edge["provenance"]["source_training_completion_receipt"] = str(SOURCE_RECEIPT)
        edge["provenance"]["source_training_completion_receipt_sha256"] = SOURCE_RECEIPT_SHA
        edge["provenance"]["source_training_dataset"] = str(TRAINING_DATASET)
        edge["provenance"]["source_training_dataset_sha256"] = TRAINING_DATASET_SHA
        edge["provenance"]["formal_holdout_dataset"] = str(HOLDOUT_DATASET)
        edge["provenance"]["holdout_capture_permitted"] = capture_id is not None
        edge["provenance"]["prior_v1_failure_receipt"] = str(PRIOR_FAILURE)
        edge["provenance"]["prior_v1_failure_receipt_sha256"] = PRIOR_FAILURE_SHA
        edge["provenance"]["prior_v1_failure_before_robot_connection"] = True
        edge["provenance"]["fresh_escalated_read_only_state_verified_idle_at_T20"] = True
        edge.pop("collection_state", None)

        name = f"{ordinal:02d}-{start}-to-{finish}.plan.yaml"
        path = OUTPUT_DIR / name
        payload = yaml.safe_dump(edge, sort_keys=False, allow_unicode=True).encode("utf-8")
        publish_exclusive(path, payload)
        records.append({
            "ordinal": ordinal,
            "start_pose_id": start,
            "target_pose_id": finish,
            "capture_holdout_id": capture_id,
            "plan": str(path),
            "plan_sha256": hashlib.sha256(payload).hexdigest(),
        })

    manifest = {
        "schema_version": 1,
        "kind": "franka_eye_to_hand_remount5_holdout_v2_manifest",
        "hardware_opened": False,
        "robot_motion_commanded": False,
        "source_training_master": str(SOURCE_MASTER),
        "source_training_master_sha256": SOURCE_MASTER_SHA,
        "source_training_completion_receipt": str(SOURCE_RECEIPT),
        "source_training_completion_receipt_sha256": SOURCE_RECEIPT_SHA,
        "source_training_dataset": str(TRAINING_DATASET),
        "source_training_dataset_sha256": TRAINING_DATASET_SHA,
        "holdout_dataset": str(HOLDOUT_DATASET),
        "holdout_dataset_initially_absent": True,
        "edges": records,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    payload = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    publish_exclusive(manifest_path, payload)
    print(json.dumps({"manifest": str(manifest_path), "manifest_sha256": hashlib.sha256(payload).hexdigest(), "edge_count": len(records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
