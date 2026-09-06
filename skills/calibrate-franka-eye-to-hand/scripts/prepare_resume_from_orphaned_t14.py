#!/usr/bin/env python3
"""Offline-only recovery of the verified T14 append after parent termination.

The formal collector appended sample 14 and wrote its debug image, but the
supervising sequence process was interrupted before it could persist the
capture log/completion receipt.  This script validates every surviving input,
the camera-only telemetry, the exact dataset append and planned T14 pose, then
creates one fixed new T15..T20 master.  It never imports or opens hardware.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import yaml


ROOT = Path("/home/qiaoguanren/code/franka")
SCRIPT_DIR = ROOT / "skills/calibrate-franka-eye-to-hand/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_training_sequence as training  # noqa: E402
import recover_t11_capture_only as recovery  # noqa: E402


CAL_ROOT = ROOT / "beta/dynamic_object_pcd"
RUNS = CAL_ROOT / "calibration_runs"
SOURCE_MASTER = RUNS / "fr3-d435-342222071785-20260812-002810-remount5-training-resume-from-T11-v1-plan.yaml"
SOURCE_MASTER_SHA = "29b165b201714263f0c959131cff20c3a22e462d7f8d6d8c8b483c869758bbea"
DATASET = RUNS / "fr3-d435-342222071785-20260812-002810-remount5-training.yaml"
DATASET_SHA = "c8108d4e27e702938494340024c8c5e9027f0c470bc35f9c95634fdc8c2b40d9"
PRE_CAPTURE_DATASET_SHA = "b8322a6e4fb6cfd933de71a84bce61c8af247fe18c99c6b32615d0f5c80f7f90"
SOURCE_ARTIFACTS = RUNS / "fr3-d435-342222071785-20260812-002810-remount5-training-resume-from-T11-v1"
STEM = "T13-to-T14-eadd47d926735006"
EDGE = SOURCE_ARTIFACTS / "edges"
PLAN = EDGE / (STEM + ".plan.yaml")
MOTION_LOG = EDGE / (STEM + ".motion.log")
INSPECT_LOG = EDGE / (STEM + ".inspect.log")
RAW = EDGE / (STEM + ".inspection-raw.png")
ANNOTATED = EDGE / (STEM + ".inspection-annotated.png")
TELEMETRY = EDGE / (STEM + ".frame-telemetry.json")
TELEMETRY_LOG = EDGE / (STEM + ".telemetry.log")
DEBUG_IMAGE = EDGE / (STEM + ".capture.png")
CLAIM = RUNS / ".training-edge-claims/29b165b201714263f0c959131cff20c3a22e462d7f8d6d8c8b483c869758bbea-T13-to-T14.claim.json"
OUTPUT_MASTER = RUNS / "fr3-d435-342222071785-20260812-002810-remount5-training-resume-from-T14-v2-plan.yaml"
OUTPUT_ARTIFACTS = RUNS / "fr3-d435-342222071785-20260812-002810-remount5-training-resume-from-T14-v2"

EXPECTED = {
    PLAN: "eadd47d92673500686982fbc5ebb071465094de6dbeff93682fdc5727c4f6e0c",
    CLAIM: "a4da0662cf6cdff6745e2b2c81f42954c4d02b2d9af63df306fdb9d5e8aa8289",
    MOTION_LOG: "200fd7333d100ef850c340d4acbb71f4bd88688765bb944967e56343661c9488",
    INSPECT_LOG: "18d46749f845447a9547ae3c5241cdc690cf280c17a98edbb6ad04747f4e842c",
    RAW: "07ed4e3e2b41c7c443b8c2fc87612b7481b1fa247d353fd554dcbde7104515f7",
    ANNOTATED: "cd4b2e25c9c8b426cb0642d153a6b765894145c12c2fe86afea1474974e5c70e",
    TELEMETRY: "412cecb3c734c151f669110e1b1a1a216040d8c20336923c749eb5afc35a095d",
    TELEMETRY_LOG: "20d7a00aa06e8f8d6afc69d8bd8d232d96fa7441a5265a501cb292b418089ad4",
    DEBUG_IMAGE: "a347c8b32d9184be8f1abb4cd0d3f9e73ce31e26625dd52d5289d619c379fa66",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}


def main() -> int:
    if OUTPUT_MASTER.exists() or OUTPUT_ARTIFACTS.exists():
        raise FileExistsError("fixed v2 master/artifacts path already exists")
    if sha(SOURCE_MASTER) != SOURCE_MASTER_SHA:
        raise ValueError("source T11 resume master changed")
    if sha(DATASET) != DATASET_SHA:
        raise ValueError("training dataset changed")
    for path, expected in EXPECTED.items():
        if not path.is_file() or sha(path) != expected:
            raise ValueError("orphaned T14 evidence changed: {}".format(path))
    for forbidden in (
        EDGE / (STEM + ".capture.log"),
        EDGE / (STEM + ".complete.json"),
        EDGE / (STEM + ".failed.json"),
        SOURCE_ARTIFACTS / "sequence.complete.json",
    ):
        if forbidden.exists():
            raise ValueError("unexpected post-T14 result exists: {}".format(forbidden))

    _, source_bytes, source = training._read_yaml(
        SOURCE_MASTER, maximum_bytes=training.MAX_PLAN_BYTES
    )
    if training._sha256_bytes(source_bytes) != SOURCE_MASTER_SHA:
        raise ValueError("source master byte hash mismatch")
    snapshot = training._dataset_snapshot(DATASET)
    if snapshot.sha256 != DATASET_SHA or len(snapshot.samples) != 14:
        raise ValueError("dataset is not the exact 14-sample orphaned state")
    training._validate_dataset_metadata(snapshot, training._expected_dataset_metadata(source))

    _, _, claim = training._read_json_mapping(CLAIM, maximum_bytes=1_000_000)
    expected_claim = {
        "kind": "franka_eye_to_hand_training_edge_claim",
        "state": "claimed_before_motion",
        "edge": "T13_to_T14",
        "start_pose_id": "T13",
        "target_pose_id": "T14",
        "reviewed_master_plan_sha256": SOURCE_MASTER_SHA,
        "single_edge_plan_sha256": EXPECTED[PLAN],
        "dataset_pre_motion_sha256": PRE_CAPTURE_DATASET_SHA,
        "dataset_pre_motion_sample_count": 13,
        "expected_post_capture_sample_count": 14,
        "holdout_capture_permitted": False,
    }
    for key, value in expected_claim.items():
        if claim.get(key) != value:
            raise ValueError("T14 claim mismatch: {}".format(key))

    motion = training.CommandResult(("orphaned-T14-motion",), 0, MOTION_LOG.read_text())
    control = training._verify_motion_ready(motion, target_pose_id="T14", plan_sha256=EXPECTED[PLAN])
    camera = training._require_mapping(snapshot.document.get("camera"), "dataset.camera")
    target = training._require_mapping(training._expected_dataset_metadata(source)["target"], "target")
    inspection = training._parse_inspection(
        training.CommandResult(("orphaned-T14-inspection",), 0, INSPECT_LOG.read_text()),
        expected_serial="342222071785",
        target=target,
        dataset_camera=camera,
        raw_image=RAW.resolve(),
        annotated_image=ANNOTATED.resolve(),
    )
    telemetry = training._parse_frame_telemetry(
        training.CommandResult(("orphaned-T14-telemetry",), 0, TELEMETRY_LOG.read_text()),
        telemetry_path=TELEMETRY.resolve(),
        expected_serial="342222071785",
        target=target,
        dataset_camera=camera,
        capture_config=(CAL_ROOT / "configs/d435_default.yaml").resolve(),
    )
    training._decode_image(DEBUG_IMAGE, "orphaned T14 formal debug image", phase="capture")

    resumed = copy.deepcopy(dict(source))
    resumed["session_slug"] = "fr3-d435-342222071785-20260812-002810-remount5-training-resume-from-T14-v2"
    provenance = dict(training._require_mapping(resumed.get("provenance"), "provenance"))
    provenance.update({
        "purpose": "formal_remount5_training_resumed_after_verified_orphaned_T14_append",
        "master_plan_path": str(OUTPUT_MASTER.resolve()),
        "source_T11_resume_master": str(SOURCE_MASTER.resolve()),
        "source_T11_resume_master_sha256": SOURCE_MASTER_SHA,
        "orphaned_T14_dataset_sha256": snapshot.sha256,
        "orphaned_T14_append_verified_by_collector_commit_semantics": True,
        "orphaned_T14_formal_untrimmed_exact_values_recoverable": False,
        "orphaned_T14_formal_untrimmed_committed_limits": {"translation_m": 0.001, "rotation_deg": 0.3, "reprojection_px": 0.5},
        "orphaned_T14_control_telemetry": control,
        "orphaned_T14_inspection": inspection,
        "orphaned_T14_camera_telemetry": telemetry,
        "orphaned_T14_evidence": [record(path) for path in EXPECTED],
        "prior_files_overwritten": False,
    })
    resumed["provenance"] = provenance
    resumed["status"] = training.MASTER_RUN_STATUS
    remaining = tuple("T{:02d}".format(i) for i in range(15, 21))
    edges = tuple("{}_to_{}".format(a, b) for a, b in zip(("T14",) + remaining[:-1], remaining))
    old_auth = training._require_mapping(source.get("motion_authorization"), "motion_authorization")
    resumed["motion_authorization"] = {
        "explicit_user_authorization_recorded": True,
        "scope": training.MASTER_AUTHORIZATION_SCOPE,
        "start_pose_id": "T14",
        "pose_id": "T15",
        "purpose": "formal_training_sample_015_through_020_after_verified_T14_append",
        "source_text": training._require_text(old_auth.get("source_text"), "authorization source"),
        "authorization_interpretation": "continue_the_reviewed_bounded_remount5_suffix_T15_through_T20_after_verified_T14_append;_each_edge_remains_SHA_bound_and_fail_closed",
        "authorized_training_suffix": list(remaining),
        "authorized_edges": list(edges),
        "authorized_at_local": old_auth.get("authorized_at_local"),
        "consumed": False,
    }
    resumed["collection_state"] = {
        "current_pose_id": "T14",
        "current_training_sample_count": 14,
        "next_pose_id": "T15",
        "next_capture_index": 15,
        "formal_training_dataset": str(DATASET.relative_to(CAL_ROOT)),
        "initial_dataset_sha256": snapshot.sha256,
        "training_sequence_artifacts_dir": str(OUTPUT_ARTIFACTS.relative_to(CAL_ROOT)),
    }
    poses = resumed.get("training_poses")
    if not isinstance(poses, list) or len(poses) != 20:
        raise ValueError("training pose table changed")
    for index, pose in enumerate(poses, 1):
        if pose.get("id") != "T{:02d}".format(index):
            raise ValueError("pose ordering changed")
        if index <= 14:
            pose["state"] = "captured_verified_orphaned_T14" if index == 14 else "captured_prior_sequence"
            pose["sample_index"] = index
        else:
            pose["state"] = "planned_fresh_remount5_resume_v2"
            pose.pop("sample_index", None)
    history = resumed.get("motion_history")
    if not isinstance(history, list):
        raise ValueError("motion_history changed")
    history.append({
        "event": "verified_orphaned_T14_append_after_supervisor_termination",
        "edge": "T13_to_T14",
        "robot_motion_commanded": True,
        "control_success_qualified": True,
        "formal_capture_committed": True,
        "dataset_sample_count_after": 14,
        "dataset_sha256_after": snapshot.sha256,
    })

    training._validate_existing_sample_prefix(snapshot, master=resumed, poses=training._pose_map(resumed))
    payload = yaml.safe_dump(resumed, sort_keys=False, allow_unicode=True).encode()
    digest = training._sha256_bytes(payload)
    code = recovery._canonical_code_paths()
    py_launch = ROOT / ".venv/bin/python"
    py_resolved = py_launch.resolve(strict=True)
    claims_dir = (RUNS / ".training-edge-claims").resolve()
    training._validate_master(
        resumed,
        master_sha256=digest,
        master_path=OUTPUT_MASTER.resolve(),
        calibration_root=CAL_ROOT.resolve(),
        claims_dir=claims_dir,
        dataset=DATASET.resolve(),
        artifacts_dir=OUTPUT_ARTIFACTS.resolve(),
        orchestrator_sha256=sha(code["training_orchestrator"]),
        python_executable=py_launch,
        python_executable_resolved=py_resolved,
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
    first_claim = claims_dir / (training._claim_stem("T14", "T15", digest) + ".claim.json")
    if first_claim.exists():
        raise FileExistsError("new first edge claim already exists")
    training._publish_exclusive(OUTPUT_MASTER, payload)
    print("T14_RESUME_MASTER_JSON=" + json.dumps({
        "master": str(OUTPUT_MASTER), "master_sha256": digest,
        "dataset": str(DATASET), "dataset_sha256": snapshot.sha256,
        "samples": 14, "remaining": list(remaining),
        "artifacts": str(OUTPUT_ARTIFACTS), "hardware_opened": False,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
