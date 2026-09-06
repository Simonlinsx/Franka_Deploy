#!/usr/bin/env python3
"""Prepare and run an independent, fail-closed H01--H06 holdout sequence.

This module never imports a robot or camera SDK.  ``prepare`` creates one
immutable reviewed master after a completed T01--T20 training run.  ``run``
executes exactly T20->T01, then T01->Hi->T01 for H01 through H06.  Only the
outbound H edges capture samples, into a brand-new dataset that is never
supplied to the hand-eye solver by this program.

Every motion edge is represented by an immutable SHA-bound single-edge plan
and atomically claimed before motion.  Claims are never removed or reused, so
an interruption or failure requires an audit and a new master rather than an
automatic retry from an uncertain physical state.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

# Reuse the frozen, tested parsers and atomic artifact helpers without changing
# the pinned training orchestrator that may still be collecting T01--T20.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import run_training_sequence as _training  # noqa: E402


MASTER_KIND = "franka_eye_to_hand_holdout_collection_plan"
MASTER_STATUS = "explicit_bounded_holdout_sequence_authorized"
MASTER_AUTHORIZATION_SCOPE = "reviewed_bounded_holdout_sequence"
HOLDOUT_IDS = tuple("H{:02d}".format(index) for index in range(1, 7))
EXPECTED_TRAINING_IDS = tuple("T{:02d}".format(index) for index in range(1, 21))
MIN_FREE_BYTES_BEFORE_MOTION = 100_000_000
MIN_ALLOWED_FREE_BYTES_GATE = 64_000_000
MAX_RECEIPT_BYTES = 20_000_000

WORKFLOW_AUTHORIZATION_TOKEN = "FR3_REVIEWED_HOLDOUT_SEQUENCE_AUTHORIZED"


@dataclass(frozen=True)
class HoldoutEdge:
    ordinal: int
    start_pose_id: str
    target_pose_id: str
    capture_holdout_id: Optional[str]

    @property
    def label(self) -> str:
        return "{}_to_{}".format(self.start_pose_id, self.target_pose_id)

    @property
    def capture(self) -> bool:
        return self.capture_holdout_id is not None


@dataclass(frozen=True)
class RuntimeConfig:
    master_plan: Path
    expected_master_sha256: str
    holdout_dataset: Path
    artifacts_dir: Path
    calibration_root: Path
    capture_config: Path
    motion_script: Path
    motion_driver: Path
    link_preflight: Path
    inspection_script: Path
    frame_telemetry_script: Path
    python_executable: str


@dataclass(frozen=True)
class RuntimeInputs:
    master_path: Path
    master_sha256: str
    calibration_root: Path
    claims_dir: Path
    holdout_dataset: Path
    artifacts_dir: Path
    source_master: Path
    source_master_sha256: str
    source_completion: Path
    source_completion_sha256: str
    source_dataset: Path
    source_dataset_sha256: str
    source_dataset_snapshot: _training.DatasetSnapshot
    python_launch: Path
    python_resolved: Path
    python_sha256: str
    orchestrator: Path
    orchestrator_sha256: str
    training_helper: Path
    training_helper_sha256: str
    motion_script: Path
    motion_script_sha256: str
    motion_driver: Path
    motion_driver_sha256: str
    link_preflight: Path
    link_preflight_sha256: str
    inspection_script: Path
    inspection_script_sha256: str
    frame_telemetry_script: Path
    frame_telemetry_script_sha256: str
    capture_config: Path
    capture_config_sha256: str
    capture_cli: Path
    capture_cli_sha256: str
    detector: Path
    detector_sha256: str
    stationary: Path
    stationary_sha256: str
    transforms: Path
    transforms_sha256: str


@dataclass(frozen=True)
class HoldoutResult:
    completed_edges: Tuple[str, ...]
    captured_holdout_ids: Tuple[str, ...]
    final_dataset_sha256: str
    completion_receipt: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _exact_edges() -> Tuple[HoldoutEdge, ...]:
    edges: List[HoldoutEdge] = [HoldoutEdge(1, "T20", "T01", None)]
    ordinal = 2
    for holdout_id in HOLDOUT_IDS:
        edges.append(HoldoutEdge(ordinal, "T01", holdout_id, holdout_id))
        ordinal += 1
        edges.append(HoldoutEdge(ordinal, holdout_id, "T01", None))
        ordinal += 1
    return tuple(edges)


def _edge_record(edge: HoldoutEdge) -> Mapping[str, Any]:
    return {
        "ordinal": edge.ordinal,
        "edge": edge.label,
        "start_pose_id": edge.start_pose_id,
        "target_pose_id": edge.target_pose_id,
        "capture_holdout_sample": edge.capture,
        "capture_holdout_id": edge.capture_holdout_id,
    }


def _pose_collections(
    master: Mapping[str, Any],
) -> Tuple[Mapping[str, Mapping[str, Any]], Mapping[str, Mapping[str, Any]]]:
    training: Dict[str, Mapping[str, Any]] = {}
    holdout: Dict[str, Mapping[str, Any]] = {}
    for field, destination in (("training_poses", training), ("holdout_poses", holdout)):
        entries = master.get(field)
        if not isinstance(entries, list):
            raise ValueError("{} must be a list".format(field))
        for index, raw in enumerate(entries):
            entry = _training._require_mapping(raw, "{}[{}]".format(field, index))
            pose_id = _training._require_text(
                entry.get("id"), "{}[{}].id".format(field, index)
            )
            if not _training._POSE_ID.fullmatch(pose_id):
                raise ValueError("malformed pose id {!r}".format(pose_id))
            if pose_id in training or pose_id in holdout:
                raise ValueError("duplicate pose id {!r}".format(pose_id))
            destination[pose_id] = entry
    return training, holdout


def _assert_hash(value: Any, name: str) -> str:
    digest = _training._require_text(value, name)
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("{} is not a lowercase SHA-256".format(name))
    return digest


def _assert_under(path: Path, root: Path, name: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("{} escapes calibration root: {}".format(name, resolved)) from exc
    return resolved


def _stable_dataset_metadata_equal(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    fields = ("camera", "target", "robot", "transform_convention", "units")
    return all(
        _training._canonical(left.get(field)) == _training._canonical(right.get(field))
        for field in fields
    )


def _validate_source_training(
    *,
    source_master_path: Path,
    expected_source_master_sha256: str,
    source_completion_path: Path,
    expected_source_completion_sha256: str,
    source_dataset_path: Path,
    expected_source_dataset_sha256: str,
) -> Tuple[
    Mapping[str, Any], Mapping[str, Any], _training.DatasetSnapshot
]:
    master_path, master_bytes, source_master = _training._read_yaml(
        source_master_path, maximum_bytes=_training.MAX_PLAN_BYTES
    )
    actual_master_sha = _training._sha256_bytes(master_bytes)
    if actual_master_sha != expected_source_master_sha256:
        raise ValueError(
            "source training master SHA mismatch: expected={} actual={}".format(
                expected_source_master_sha256, actual_master_sha
            )
        )
    if (
        source_master.get("schema_version") != 1
        or source_master.get("kind") != _training.PLAN_KIND
    ):
        raise ValueError("source training master is not a schema-1 commissioning plan")
    topology = _training._require_mapping(source_master.get("topology"), "topology")
    if topology.get("type") != "eye_to_hand" or topology.get("camera_mount") != "fixed_external":
        raise ValueError("source training master is not fixed-external eye-to-hand")
    target = _training._require_mapping(source_master.get("target"), "target")
    if target.get("mount_must_not_change_until_training_and_holdout_finish") is not True:
        raise ValueError("source master does not preserve the target mount through holdout")

    completion_path, completion_bytes, completion = _training._read_json_mapping(
        source_completion_path, maximum_bytes=MAX_RECEIPT_BYTES
    )
    actual_completion_sha = _training._sha256_bytes(completion_bytes)
    if actual_completion_sha != expected_source_completion_sha256:
        raise ValueError(
            "source training completion SHA mismatch: expected={} actual={}".format(
                expected_source_completion_sha256, actual_completion_sha
            )
        )
    if completion.get("kind") != "franka_eye_to_hand_training_sequence_receipt":
        raise ValueError("source completion receipt has the wrong kind")
    if completion.get("state") != "reviewed_remaining_training_suffix_capture_verified_complete":
        raise ValueError("source training sequence is not capture-verified complete")
    if Path(str(completion.get("reviewed_master_plan"))).resolve() != master_path:
        raise ValueError("source completion references a different training master")
    if completion.get("reviewed_master_plan_sha256") != actual_master_sha:
        raise ValueError("source completion training-master SHA mismatch")
    if completion.get("final_sample_count") != 20:
        raise ValueError("source training completion must report exactly 20 samples")
    if tuple(completion.get("completed_pose_ids", ())) != EXPECTED_TRAINING_IDS[1:]:
        raise ValueError("source completion does not cover exactly T02 through T20")

    snapshot = _training._dataset_snapshot(source_dataset_path)
    if snapshot.sha256 != expected_source_dataset_sha256:
        raise ValueError(
            "source training dataset SHA mismatch: expected={} actual={}".format(
                expected_source_dataset_sha256, snapshot.sha256
            )
        )
    if len(snapshot.samples) != 20:
        raise ValueError("source training dataset must contain exactly 20 samples")
    if Path(str(completion.get("dataset"))).resolve() != snapshot.path:
        raise ValueError("source completion references a different training dataset")
    if completion.get("final_dataset_sha256") != snapshot.sha256:
        raise ValueError("source completion final dataset SHA mismatch")
    expected_metadata = _training._expected_dataset_metadata(source_master)
    _training._validate_dataset_metadata(snapshot, expected_metadata)

    training, holdout = _pose_collections(source_master)
    if tuple(training) != EXPECTED_TRAINING_IDS:
        raise ValueError("source training poses must be exactly T01 through T20")
    if tuple(holdout) != HOLDOUT_IDS:
        raise ValueError("source holdout poses must be exactly H01 through H06")
    for sample_index, pose_id in ((0, "T01"), (19, "T20")):
        observed = _training._rigid_transform(
            snapshot.samples[sample_index].get("T_base_ee"),
            "source sample {} T_base_ee".format(sample_index + 1),
        )
        expected = _training._planned_target_transform(
            source_master, training[pose_id], pose_id
        )
        translation, rotation = _training._transform_error(observed, expected)
        if (
            translation > _training.MAX_CAPTURE_EEF_TRANSLATION_ERROR_M
            or rotation > _training.MAX_CAPTURE_EEF_ROTATION_ERROR_DEG
        ):
            raise ValueError(
                "source sample {} does not match {}: {:.3f}mm/{:.3f}deg".format(
                    sample_index + 1, pose_id, translation * 1000.0, rotation
                )
            )
    return source_master, completion, snapshot


def _canonical_paths(calibration_root: Path) -> Mapping[str, Path]:
    root = calibration_root.expanduser().resolve(strict=True)
    workspace_root = root.parents[1]
    return {
        "calibration_root": root,
        "capture_cli": (root / "dynamic_pcd/apps/calibrate_eye_to_hand.py").resolve(strict=True),
        "detector": (root / "dynamic_pcd/calibration/aruco.py").resolve(strict=True),
        "stationary": (root / "dynamic_pcd/calibration/stationary.py").resolve(strict=True),
        "transforms": (root / "dynamic_pcd/calibration/transforms.py").resolve(strict=True),
        "motion_driver": (
            workspace_root / "dexgrasp/src/anydex_pipeline/franka_sequence_driver.py"
        ).resolve(strict=True),
        "link_preflight": (
            workspace_root / "dexgrasp/src/anydex_pipeline/host_network_preflight.py"
        ).resolve(strict=True),
    }


def _pin(path: Path, maximum_bytes: Optional[int] = _training.MAX_PLAN_BYTES) -> Mapping[str, str]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": _training._sha256_file(resolved, maximum_bytes=maximum_bytes),
    }


def prepare_master(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.confirm_workflow_authorization != WORKFLOW_AUTHORIZATION_TOKEN:
        raise ValueError("missing exact existing-workflow authorization token")
    confirmations = {
        "confirm_e_stop": (args.confirm_e_stop, _training.E_STOP_TOKEN),
        "confirm_swept_volume": (args.confirm_swept_volume, _training.SWEPT_VOLUME_TOKEN),
        "confirm_target_rigid": (args.confirm_target_rigid, _training.TARGET_RIGID_TOKEN),
        "confirm_camera_fixed": (args.confirm_camera_fixed, _training.CAMERA_FIXED_TOKEN),
    }
    for name, (actual, expected) in confirmations.items():
        if actual != expected:
            raise ValueError("{} must equal {!r}".format(name, expected))

    calibration_root = args.calibration_root.expanduser().resolve(strict=True)
    canonical = _canonical_paths(calibration_root)
    source_master_path = args.training_master.expanduser().resolve(strict=True)
    source_completion_path = args.training_completion_receipt.expanduser().resolve(strict=True)
    source_dataset_path = args.training_dataset.expanduser().resolve(strict=True)
    source_master, completion, source_snapshot = _validate_source_training(
        source_master_path=source_master_path,
        expected_source_master_sha256=args.expect_training_master_sha256,
        source_completion_path=source_completion_path,
        expected_source_completion_sha256=args.expect_training_completion_sha256,
        source_dataset_path=source_dataset_path,
        expected_source_dataset_sha256=args.expect_training_dataset_sha256,
    )

    plan_path = _assert_under(args.output_plan, calibration_root, "output plan")
    holdout_dataset = _assert_under(args.holdout_dataset, calibration_root, "holdout dataset")
    artifacts_dir = _assert_under(args.artifacts_dir, calibration_root, "artifacts directory")
    claims_dir = (calibration_root / "calibration_runs/.holdout-edge-claims").resolve()
    calibration_runs = (calibration_root / "calibration_runs").resolve()
    for path, name in (
        (plan_path, "output plan"),
        (holdout_dataset, "holdout dataset"),
        (artifacts_dir, "artifacts directory"),
    ):
        try:
            path.relative_to(calibration_runs)
        except ValueError as exc:
            raise ValueError("{} must be under calibration_runs".format(name)) from exc
        if path.exists():
            raise FileExistsError("{} must be absent: {}".format(name, path))
    if len({plan_path, holdout_dataset, artifacts_dir, source_dataset_path}) != 4:
        raise ValueError("plan, source dataset, holdout dataset, and artifacts paths must differ")

    minimum_free_bytes = int(args.minimum_free_bytes_before_motion)
    if minimum_free_bytes < MIN_ALLOWED_FREE_BYTES_GATE:
        raise ValueError(
            "minimum-free-bytes gate may not be below {}".format(
                MIN_ALLOWED_FREE_BYTES_GATE
            )
        )
    python_launch = _training._lexical_absolute_path(args.python_executable, "python executable")
    python_resolved = python_launch.resolve(strict=True)
    motion_script = args.motion_script.expanduser().resolve(strict=True)
    inspection_script = args.inspection_script.expanduser().resolve(strict=True)
    telemetry_script = args.frame_telemetry_script.expanduser().resolve(strict=True)
    capture_config = args.capture_config.expanduser().resolve(strict=True)
    orchestrator = Path(__file__).resolve(strict=True)
    training_helper = _SCRIPT_DIR / "run_training_sequence.py"

    training_poses, holdout_poses = _pose_collections(source_master)
    selected_training = []
    for pose_id, sample_index in (("T01", 1), ("T20", 20)):
        entry = copy.deepcopy(dict(training_poses[pose_id]))
        entry["state"] = "source_training_pose_capture_verified"
        entry["sample_index"] = sample_index
        selected_training.append(entry)
    selected_holdout = [copy.deepcopy(dict(holdout_poses[pose_id])) for pose_id in HOLDOUT_IDS]
    edges = _exact_edges()

    safety = copy.deepcopy(dict(_training._require_mapping(source_master.get("safety"), "safety")))
    for field in (
        "emergency_stop_reachable",
        "full_large_board_bracket_and_cable_sequence_swept_volume_clear",
        "target_rigid_confirmed",
        "camera_fixed_confirmed",
    ):
        if safety.get(field) is not True:
            raise ValueError("source safety.{} must be exactly true".format(field))
    safety["motion_authorized"] = True
    safety["minimum_free_bytes_before_motion"] = minimum_free_bytes

    pins = {
        "holdout_orchestrator": _pin(orchestrator),
        "training_orchestrator_helper": _pin(training_helper),
        "python_executable": {
            "path": str(python_launch),
            "resolved_path": str(python_resolved),
            "sha256": _training._sha256_file(python_resolved),
        },
        "motion_wrapper": _pin(motion_script),
        "motion_driver": _pin(canonical["motion_driver"]),
        "link_preflight": _pin(canonical["link_preflight"]),
        "inspection_script": _pin(inspection_script),
        "frame_telemetry": _pin(telemetry_script),
        "capture_config": _pin(capture_config),
        "capture_cli": _pin(canonical["capture_cli"]),
        "aruco_detector": _pin(canonical["detector"]),
        "stationary_aggregator": _pin(canonical["stationary"]),
        "calibration_transforms": _pin(canonical["transforms"]),
    }

    collection_state = {
        "start_pose_id": "T20",
        "final_pose_id": "T01",
        "source_training_sample_count": 20,
        "expected_holdout_sample_count": 6,
        "formal_holdout_dataset": str(holdout_dataset.relative_to(calibration_root)),
        "holdout_dataset_initial_state": "must_be_absent",
        "holdout_sequence_artifacts_dir": str(artifacts_dir.relative_to(calibration_root)),
        "holdout_dataset_role": "independent_validation_only_never_solver_input",
    }
    master = {
        "schema_version": 1,
        "kind": MASTER_KIND,
        "status": MASTER_STATUS,
        "session_slug": args.session_slug,
        "provenance": {
            "purpose": "independent_remount5_holdout_collection",
            "calibration_root_path": str(calibration_root),
            "master_plan_path": str(plan_path),
            "canonical_claims_dir": str(claims_dir),
            "source_training_master": str(source_master_path),
            "source_training_master_sha256": args.expect_training_master_sha256,
            "source_training_completion_receipt": str(source_completion_path),
            "source_training_completion_receipt_sha256": args.expect_training_completion_sha256,
            "source_training_dataset": str(source_dataset_path),
            "source_training_dataset_sha256": source_snapshot.sha256,
            "source_training_calibration_not_solved_by_this_workflow": True,
            "pins": pins,
        },
        "topology": copy.deepcopy(source_master["topology"]),
        "hardware": copy.deepcopy(source_master["hardware"]),
        "target": copy.deepcopy(source_master["target"]),
        "reference_pose": copy.deepcopy(source_master["reference_pose"]),
        "target_observability": copy.deepcopy(source_master.get("target_observability", {})),
        "safety": safety,
        "planned_coverage": copy.deepcopy(source_master.get("planned_coverage", {})),
        "planned_sequence": {
            "start_pose_id": "T20",
            "staging": ["T20", "T01"],
            "holdout_order": list(HOLDOUT_IDS),
            "return_to_T01_after_every_holdout": True,
            "final_pose_id": "T01",
            "edges": [_edge_record(edge) for edge in edges],
            "holdout_dataset_is_separate": True,
            "holdout_dataset_never_passed_to_solve": True,
        },
        "motion_history": [],
        "motion_authorization": {
            "explicit_user_authorization_recorded": True,
            "scope": MASTER_AUTHORIZATION_SCOPE,
            "source_text": args.authorization_source_text,
            "authorization_interpretation": args.authorization_interpretation,
            "authorized_edges": [edge.label for edge in edges],
            "authorized_edge_records": [_edge_record(edge) for edge in edges],
            "authorized_holdout_captures": list(HOLDOUT_IDS),
            "authorization_recorded_at_local": args.authorization_recorded_at_local,
            "consumed": False,
        },
        "collection_state": collection_state,
        "training_poses": selected_training,
        "holdout_poses": selected_holdout,
    }
    _validate_master_semantics(
        master,
        master_path=plan_path,
        calibration_root=calibration_root,
        claims_dir=claims_dir,
        holdout_dataset=holdout_dataset,
        artifacts_dir=artifacts_dir,
    )
    payload = yaml.safe_dump(master, sort_keys=False, allow_unicode=True).encode("utf-8")
    if len(payload) > _training.MAX_PLAN_BYTES:
        raise ValueError("holdout master is too large")
    _training._publish_exclusive(plan_path, payload)
    result = {
        "schema_version": 1,
        "kind": "franka_eye_to_hand_holdout_master_preparation_receipt",
        "hardware_opened": False,
        "robot_motion_commanded": False,
        "master_plan": str(plan_path),
        "master_plan_sha256": _training._sha256_bytes(payload),
        "source_training_completion_receipt": str(source_completion_path),
        "source_training_completion_receipt_sha256": _training._sha256_bytes(
            source_completion_path.read_bytes()
        ),
        "source_training_dataset": str(source_dataset_path),
        "source_training_dataset_sha256": source_snapshot.sha256,
        "holdout_dataset": str(holdout_dataset),
        "holdout_dataset_initially_absent": True,
        "artifacts_dir": str(artifacts_dir),
        "edge_count": len(edges),
        "capture_count": len(HOLDOUT_IDS),
    }
    print(
        "HOLDOUT_MASTER_PREPARED_JSON="
        + json.dumps(result, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return result


def _validate_master_semantics(
    master: Mapping[str, Any],
    *,
    master_path: Path,
    calibration_root: Path,
    claims_dir: Path,
    holdout_dataset: Path,
    artifacts_dir: Path,
) -> None:
    if master.get("schema_version") != 1 or master.get("kind") != MASTER_KIND:
        raise ValueError("holdout master kind/schema mismatch")
    if master.get("status") != MASTER_STATUS:
        raise ValueError("holdout master status mismatch")
    topology = _training._require_mapping(master.get("topology"), "topology")
    if topology.get("type") != "eye_to_hand" or topology.get("camera_mount") != "fixed_external":
        raise ValueError("holdout master must be fixed-external eye-to-hand")

    provenance = _training._require_mapping(master.get("provenance"), "provenance")
    path_checks = (
        ("calibration_root_path", calibration_root, "calibration root"),
        ("master_plan_path", master_path, "master plan"),
        ("canonical_claims_dir", claims_dir, "canonical claims directory"),
    )
    for field, expected, label in path_checks:
        declared = _training._lexical_absolute_path(
            provenance.get(field), "provenance." + field
        )
        if declared != expected.resolve():
            raise ValueError("{} path does not match the runner".format(label))

    safety = _training._require_mapping(master.get("safety"), "safety")
    for field in (
        "motion_authorized",
        "emergency_stop_reachable",
        "full_large_board_bracket_and_cable_sequence_swept_volume_clear",
        "target_rigid_confirmed",
        "camera_fixed_confirmed",
    ):
        if safety.get(field) is not True:
            raise ValueError("safety.{} must be exactly true".format(field))
    free_gate = safety.get("minimum_free_bytes_before_motion")
    if (
        isinstance(free_gate, bool)
        or not isinstance(free_gate, int)
        or free_gate < MIN_ALLOWED_FREE_BYTES_GATE
    ):
        raise ValueError("safety.minimum_free_bytes_before_motion is too small")

    sequence = _training._require_mapping(master.get("planned_sequence"), "planned_sequence")
    if sequence.get("start_pose_id") != "T20" or sequence.get("staging") != ["T20", "T01"]:
        raise ValueError("holdout staging must be exactly T20->T01")
    if tuple(sequence.get("holdout_order", ())) != HOLDOUT_IDS:
        raise ValueError("holdout order must be exactly H01 through H06")
    if sequence.get("return_to_T01_after_every_holdout") is not True:
        raise ValueError("every holdout must return to T01")
    if sequence.get("final_pose_id") != "T01":
        raise ValueError("holdout sequence must finish at T01")
    if sequence.get("holdout_dataset_is_separate") is not True:
        raise ValueError("holdout dataset must be declared separate")
    if sequence.get("holdout_dataset_never_passed_to_solve") is not True:
        raise ValueError("holdout dataset must be excluded from solve")
    expected_records = [_edge_record(edge) for edge in _exact_edges()]
    if sequence.get("edges") != expected_records:
        raise ValueError("planned holdout edges do not exactly match the bounded sequence")

    authorization = _training._require_mapping(
        master.get("motion_authorization"), "motion_authorization"
    )
    if authorization.get("explicit_user_authorization_recorded") is not True:
        raise ValueError("holdout master lacks explicit workflow authorization")
    if authorization.get("scope") != MASTER_AUTHORIZATION_SCOPE:
        raise ValueError("holdout authorization scope mismatch")
    if authorization.get("consumed") is not False:
        raise ValueError("holdout master authorization consumed must be exactly false")
    _training._require_text(authorization.get("source_text"), "authorization.source_text")
    _training._require_text(
        authorization.get("authorization_interpretation"),
        "authorization.authorization_interpretation",
    )
    expected_labels = [edge.label for edge in _exact_edges()]
    if authorization.get("authorized_edges") != expected_labels:
        raise ValueError("authorized_edges must exactly match all 13 bounded edges")
    if authorization.get("authorized_edge_records") != expected_records:
        raise ValueError("authorized_edge_records mismatch")
    if tuple(authorization.get("authorized_holdout_captures", ())) != HOLDOUT_IDS:
        raise ValueError("authorized holdout captures must be exactly H01 through H06")

    collection = _training._require_mapping(master.get("collection_state"), "collection_state")
    if (
        collection.get("start_pose_id") != "T20"
        or collection.get("final_pose_id") != "T01"
        or collection.get("source_training_sample_count") != 20
        or collection.get("expected_holdout_sample_count") != 6
        or collection.get("holdout_dataset_initial_state") != "must_be_absent"
        or collection.get("holdout_dataset_role")
        != "independent_validation_only_never_solver_input"
    ):
        raise ValueError("holdout collection_state contract mismatch")
    declared_dataset = (
        calibration_root
        / _training._require_text(
            collection.get("formal_holdout_dataset"),
            "collection_state.formal_holdout_dataset",
        )
    ).resolve()
    if declared_dataset != holdout_dataset.resolve():
        raise ValueError("holdout dataset path does not match the master")
    relative_artifacts = Path(
        _training._require_text(
            collection.get("holdout_sequence_artifacts_dir"),
            "collection_state.holdout_sequence_artifacts_dir",
        )
    )
    if relative_artifacts.is_absolute():
        raise ValueError("holdout artifacts path must be relative")
    declared_artifacts = (calibration_root / relative_artifacts).resolve()
    _assert_under(declared_artifacts, calibration_root, "holdout artifacts")
    if declared_artifacts != artifacts_dir.resolve():
        raise ValueError("holdout artifacts path does not match the master")

    training, holdout = _pose_collections(master)
    if tuple(training) != ("T01", "T20") or tuple(holdout) != HOLDOUT_IDS:
        raise ValueError("holdout master must contain only T01,T20 and H01..H06")
    for pose_id in ("T01", "T20") + HOLDOUT_IDS:
        pose = training.get(pose_id, holdout.get(pose_id))
        assert pose is not None
        _training._planned_target_transform(master, pose, pose_id)


def _path_pin(
    pins: Mapping[str, Any], name: str, expected_path: Path
) -> str:
    record = _training._require_mapping(pins.get(name), "pins." + name)
    declared = _training._lexical_absolute_path(record.get("path"), "pins.{}.path".format(name))
    if declared != expected_path.resolve():
        raise ValueError("pin {} path mismatch".format(name))
    expected_hash = _assert_hash(record.get("sha256"), "pins.{}.sha256".format(name))
    actual_hash = _training._sha256_file(
        expected_path.resolve(strict=True), maximum_bytes=_training.MAX_PLAN_BYTES
    )
    if actual_hash != expected_hash:
        raise ValueError("pin {} SHA mismatch".format(name))
    return actual_hash


def _load_runtime_inputs(
    config: RuntimeConfig,
    master_path: Path,
    master_sha256: str,
    master: Mapping[str, Any],
) -> RuntimeInputs:
    calibration_root = config.calibration_root.expanduser().resolve(strict=True)
    canonical = _canonical_paths(calibration_root)
    holdout_dataset = config.holdout_dataset.expanduser().resolve()
    artifacts_dir = config.artifacts_dir.expanduser().resolve()
    claims_dir = (calibration_root / "calibration_runs/.holdout-edge-claims").resolve()
    _validate_master_semantics(
        master,
        master_path=master_path,
        calibration_root=calibration_root,
        claims_dir=claims_dir,
        holdout_dataset=holdout_dataset,
        artifacts_dir=artifacts_dir,
    )
    provenance = _training._require_mapping(master.get("provenance"), "provenance")
    pins = _training._require_mapping(provenance.get("pins"), "provenance.pins")

    orchestrator = Path(__file__).resolve(strict=True)
    training_helper = (_SCRIPT_DIR / "run_training_sequence.py").resolve(strict=True)
    python_launch = _training._lexical_absolute_path(
        config.python_executable, "python executable"
    )
    python_resolved = python_launch.resolve(strict=True)
    python_pin = _training._require_mapping(
        pins.get("python_executable"), "pins.python_executable"
    )
    if _training._lexical_absolute_path(
        python_pin.get("path"), "pins.python_executable.path"
    ) != python_launch:
        raise ValueError("Python launch path does not match the master")
    if Path(
        _training._require_text(
            python_pin.get("resolved_path"), "pins.python_executable.resolved_path"
        )
    ).resolve(strict=True) != python_resolved:
        raise ValueError("resolved Python path does not match the master")
    python_sha = _assert_hash(
        python_pin.get("sha256"), "pins.python_executable.sha256"
    )
    if _training._sha256_file(python_resolved) != python_sha:
        raise ValueError("Python executable SHA mismatch")

    motion_script = config.motion_script.expanduser().resolve(strict=True)
    motion_driver = config.motion_driver.expanduser().resolve(strict=True)
    link_preflight = config.link_preflight.expanduser().resolve(strict=True)
    if motion_driver != canonical["motion_driver"] or link_preflight != canonical["link_preflight"]:
        raise ValueError("motion dependencies are not canonical workspace modules")
    inspection_script = config.inspection_script.expanduser().resolve(strict=True)
    telemetry_script = config.frame_telemetry_script.expanduser().resolve(strict=True)
    capture_config = config.capture_config.expanduser().resolve(strict=True)
    capture_cli = canonical["capture_cli"]
    detector = canonical["detector"]
    stationary = canonical["stationary"]
    transforms = canonical["transforms"]

    calculated = {
        "holdout_orchestrator": (orchestrator, _path_pin(pins, "holdout_orchestrator", orchestrator)),
        "training_orchestrator_helper": (
            training_helper,
            _path_pin(pins, "training_orchestrator_helper", training_helper),
        ),
        "motion_wrapper": (motion_script, _path_pin(pins, "motion_wrapper", motion_script)),
        "motion_driver": (motion_driver, _path_pin(pins, "motion_driver", motion_driver)),
        "link_preflight": (link_preflight, _path_pin(pins, "link_preflight", link_preflight)),
        "inspection_script": (
            inspection_script,
            _path_pin(pins, "inspection_script", inspection_script),
        ),
        "frame_telemetry": (
            telemetry_script,
            _path_pin(pins, "frame_telemetry", telemetry_script),
        ),
        "capture_config": (capture_config, _path_pin(pins, "capture_config", capture_config)),
        "capture_cli": (capture_cli, _path_pin(pins, "capture_cli", capture_cli)),
        "aruco_detector": (detector, _path_pin(pins, "aruco_detector", detector)),
        "stationary_aggregator": (
            stationary,
            _path_pin(pins, "stationary_aggregator", stationary),
        ),
        "calibration_transforms": (
            transforms,
            _path_pin(pins, "calibration_transforms", transforms),
        ),
    }

    source_master = _training._lexical_absolute_path(
        provenance.get("source_training_master"), "provenance.source_training_master"
    ).resolve(strict=True)
    source_completion = _training._lexical_absolute_path(
        provenance.get("source_training_completion_receipt"),
        "provenance.source_training_completion_receipt",
    ).resolve(strict=True)
    source_dataset = _training._lexical_absolute_path(
        provenance.get("source_training_dataset"), "provenance.source_training_dataset"
    ).resolve(strict=True)
    source_master_sha = _assert_hash(
        provenance.get("source_training_master_sha256"),
        "provenance.source_training_master_sha256",
    )
    source_completion_sha = _assert_hash(
        provenance.get("source_training_completion_receipt_sha256"),
        "provenance.source_training_completion_receipt_sha256",
    )
    source_dataset_sha = _assert_hash(
        provenance.get("source_training_dataset_sha256"),
        "provenance.source_training_dataset_sha256",
    )
    source_master_document, _, source_snapshot = _validate_source_training(
        source_master_path=source_master,
        expected_source_master_sha256=source_master_sha,
        source_completion_path=source_completion,
        expected_source_completion_sha256=source_completion_sha,
        source_dataset_path=source_dataset,
        expected_source_dataset_sha256=source_dataset_sha,
    )
    del source_master_document

    _, _, capture_config_document = _training._read_yaml(
        capture_config, maximum_bytes=_training.MAX_PLAN_BYTES
    )
    _training._validate_capture_config(capture_config_document, master)
    if holdout_dataset == source_dataset:
        raise ValueError("holdout dataset may not be the training dataset")
    if holdout_dataset.exists():
        raise ValueError("holdout dataset must be absent at first launch")
    if artifacts_dir.exists():
        raise ValueError("holdout artifacts directory must be absent at first launch")
    disk_free = shutil.disk_usage(calibration_root).free
    required_free = int(
        _training._require_mapping(master.get("safety"), "safety").get(
            "minimum_free_bytes_before_motion"
        )
    )
    if disk_free < required_free:
        raise ValueError(
            "insufficient free storage before motion: {} < {} bytes".format(
                disk_free, required_free
            )
        )
    for edge in _exact_edges():
        claim = claims_dir / (_claim_stem(edge, master_sha256) + ".claim.json")
        if claim.exists():
            raise ValueError(
                "canonical holdout edge claim already exists; retry forbidden: {}".format(
                    claim
                )
            )

    return RuntimeInputs(
        master_path=master_path,
        master_sha256=master_sha256,
        calibration_root=calibration_root,
        claims_dir=claims_dir,
        holdout_dataset=holdout_dataset,
        artifacts_dir=artifacts_dir,
        source_master=source_master,
        source_master_sha256=source_master_sha,
        source_completion=source_completion,
        source_completion_sha256=source_completion_sha,
        source_dataset=source_dataset,
        source_dataset_sha256=source_dataset_sha,
        source_dataset_snapshot=source_snapshot,
        python_launch=python_launch,
        python_resolved=python_resolved,
        python_sha256=python_sha,
        orchestrator=calculated["holdout_orchestrator"][0],
        orchestrator_sha256=calculated["holdout_orchestrator"][1],
        training_helper=calculated["training_orchestrator_helper"][0],
        training_helper_sha256=calculated["training_orchestrator_helper"][1],
        motion_script=calculated["motion_wrapper"][0],
        motion_script_sha256=calculated["motion_wrapper"][1],
        motion_driver=calculated["motion_driver"][0],
        motion_driver_sha256=calculated["motion_driver"][1],
        link_preflight=calculated["link_preflight"][0],
        link_preflight_sha256=calculated["link_preflight"][1],
        inspection_script=calculated["inspection_script"][0],
        inspection_script_sha256=calculated["inspection_script"][1],
        frame_telemetry_script=calculated["frame_telemetry"][0],
        frame_telemetry_script_sha256=calculated["frame_telemetry"][1],
        capture_config=calculated["capture_config"][0],
        capture_config_sha256=calculated["capture_config"][1],
        capture_cli=calculated["capture_cli"][0],
        capture_cli_sha256=calculated["capture_cli"][1],
        detector=calculated["aruco_detector"][0],
        detector_sha256=calculated["aruco_detector"][1],
        stationary=calculated["stationary_aggregator"][0],
        stationary_sha256=calculated["stationary_aggregator"][1],
        transforms=calculated["calibration_transforms"][0],
        transforms_sha256=calculated["calibration_transforms"][1],
    )


def _claim_stem(edge: HoldoutEdge, master_sha256: str) -> str:
    _assert_hash(master_sha256, "master SHA-256")
    return "{}-{:02d}-{}-to-{}".format(
        master_sha256, edge.ordinal, edge.start_pose_id, edge.target_pose_id
    )


def _edge_stem(edge: HoldoutEdge, plan_sha256: str) -> str:
    return "{:02d}-{}-to-{}-{}".format(
        edge.ordinal, edge.start_pose_id, edge.target_pose_id, plan_sha256[:16]
    )


def _ensure_holdout_state(
    path: Path,
    snapshot: Optional[_training.DatasetSnapshot],
    phase: str,
) -> None:
    if snapshot is None:
        if path.exists():
            raise _training.SequenceFailure(
                phase, "holdout dataset appeared before its first capture"
            )
    else:
        _training._ensure_unchanged(path, snapshot.sha256, phase)


def _safe_holdout_state(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {"holdout_dataset_state": "absent", "holdout_sample_count": 0}
    state = _training._safe_failure_state(path)
    return {"holdout_dataset_state": "present", **state}


def _recheck_inputs(inputs: RuntimeInputs, phase: str) -> None:
    checks = (
        (inputs.master_path, inputs.master_sha256),
        (inputs.source_master, inputs.source_master_sha256),
        (inputs.source_completion, inputs.source_completion_sha256),
        (inputs.source_dataset, inputs.source_dataset_sha256),
        (inputs.orchestrator, inputs.orchestrator_sha256),
        (inputs.training_helper, inputs.training_helper_sha256),
        (inputs.motion_script, inputs.motion_script_sha256),
        (inputs.motion_driver, inputs.motion_driver_sha256),
        (inputs.link_preflight, inputs.link_preflight_sha256),
        (inputs.inspection_script, inputs.inspection_script_sha256),
        (inputs.frame_telemetry_script, inputs.frame_telemetry_script_sha256),
        (inputs.capture_config, inputs.capture_config_sha256),
        (inputs.capture_cli, inputs.capture_cli_sha256),
        (inputs.detector, inputs.detector_sha256),
        (inputs.stationary, inputs.stationary_sha256),
        (inputs.transforms, inputs.transforms_sha256),
    )
    for path, digest in checks:
        _training._ensure_unchanged(path, digest, phase)
    _training._ensure_python_unchanged(
        inputs.python_launch, inputs.python_resolved, inputs.python_sha256, phase
    )

