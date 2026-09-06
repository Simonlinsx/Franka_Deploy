#!/usr/bin/env python3
"""Fail-closed headless collection of the reviewed remaining training suffix.

The orchestrator deliberately does not import a Franka or RealSense package.
It derives a new immutable, single-edge motion plan from an exact-SHA reviewed
master, asks ``move_calibration_pose.py`` to preview and execute that one edge,
and only then invokes the existing calibration CLI for one stationary sample.

Each edge is atomically claimed before the motion subprocess starts.  A claim
is never removed or reused, including after a failure.  Consequently, a
failed or interrupted edge requires an explicit audit/new plan instead of an
automatic retry from an uncertain physical state.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
import uuid

import cv2
import numpy as np
import yaml

PLAN_KIND = "franka_eye_to_hand_commissioning_plan"
PLAN_SCHEMA_VERSION = 1
RUN_STATUS = "explicit_single_pose_motion_authorized"
MASTER_RUN_STATUS = "explicit_remaining_training_suffix_authorized"
MASTER_AUTHORIZATION_SCOPE = "reviewed_remaining_training_suffix"
EDGE_AUTHORIZATION_SCOPE = "single_pose"

LAST_REMAINING_POSE_ID = "T20"
CAPTURE_FRAMES = 120
MIN_CAPTURE_COHERENT_FRAMES = 114
MIN_CAPTURE_ALL_REPROJECTION_PASS_FRAMES = 118
TELEMETRY_FRAMES = 120
MIN_TELEMETRY_DETECTED_VALID_FRAMES = 118
MIN_TELEMETRY_REPROJECTION_PASS_FRAMES = 118
MIN_TELEMETRY_COHERENT_FRAMES = 114
INSPECTION_FRAMES = 30

E_STOP_TOKEN = "FR3_ESTOP_REACHABLE"
SWEPT_VOLUME_TOKEN = "FR3_CALIBRATION_SWEPT_VOLUME_CLEAR"
TARGET_RIGID_TOKEN = "FR3_CALIBRATION_TARGET_RIGID"
CAMERA_FIXED_TOKEN = "EYE_TO_HAND_CAMERA_FIXED"

MAX_DATASET_BYTES = 20_000_000
MAX_PLAN_BYTES = 2_000_000
MAX_TELEMETRY_BYTES = 20_000_000
MAX_CAPTURE_REPROJECTION_P95_PX = 0.5
MAX_CAPTURE_TARGET_TRANSLATION_JITTER_M = 0.001
MAX_CAPTURE_TARGET_ROTATION_JITTER_DEG = 0.3
MAX_CAPTURE_EEF_TRANSLATION_ERROR_M = 0.005
MAX_CAPTURE_EEF_ROTATION_ERROR_DEG = 1.0
MIN_FORMAL_SAMPLE_TRANSLATION_M = 0.010
MIN_FORMAL_SAMPLE_ROTATION_DEG = 3.0
MIN_LIVE_MARKER_MARGIN_PX = 20.0
MIN_LIVE_MARKER_EDGE_PX = 80.0
INTRINSICS_ABSOLUTE_TOLERANCE = 1.0e-6

_POSE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_FLOAT = r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
_CAPTURE_LINE = re.compile(
    r"Captured sample (?P<sample>[0-9]+):\s*"
    r"coherent=(?P<coherent>[0-9]+)/(?P<frames>[0-9]+),\s*"
    r"target jitter p95=(?P<translation>" + _FLOAT + r")\s*mm/"
    r"(?P<rotation>" + _FLOAT + r")\s*deg,\s*"
    r"reprojection p95=(?P<reprojection>" + _FLOAT + r")px;"
)
_CAPTURE_UNTRIMMED_SENTINEL = "CAPTURE_UNTRIMMED_P95_JSON="
_IMPORT_PROBE_SENTINEL = "CALIBRATION_IMPORT_PATHS_JSON="
_IMPORT_PROBE_CODE = "\n".join(
    (
        "import importlib.util, json, pathlib, sys",
        "root = pathlib.Path(sys.argv[1]).resolve()",
        "sys.path.insert(0, str(root))",
        "names = ('dynamic_pcd.apps.calibrate_eye_to_hand', "
        "'dynamic_pcd.calibration.aruco', "
        "'dynamic_pcd.calibration.stationary', "
        "'dynamic_pcd.calibration.transforms')",
        "paths = {name: str(pathlib.Path(importlib.util.find_spec(name).origin).resolve()) "
        "for name in names}",
        "print('CALIBRATION_IMPORT_PATHS_JSON=' + "
        "json.dumps(paths, sort_keys=True, separators=(',', ':')))",
    )
)


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> Mapping[str, Any]:
    result: Dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found duplicate key {!r}".format(key),
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


class SequenceFailure(RuntimeError):
    """Failure annotated with the phase that must stop the sequence."""

    def __init__(self, phase: str, message: str):
        super().__init__(message)
        self.phase = phase


@dataclass(frozen=True)
class CommandResult:
    argv: Tuple[str, ...]
    returncode: int
    output: str


@dataclass(frozen=True)
class DatasetSnapshot:
    path: Path
    sha256: str
    document: Mapping[str, Any]
    samples: Tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class OrchestratorConfig:
    master_plan: Path
    expected_master_sha256: str
    dataset: Path
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
class SequenceResult:
    completed_pose_ids: Tuple[str, ...]
    final_sample_count: int
    final_dataset_sha256: str
    completion_receipt: Path


class StreamingSubprocessRunner:
    """Run argv without a shell while teeing merged output immediately."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        phase: str,
        env: Optional[Mapping[str, str]] = None,
    ) -> CommandResult:
        command = tuple(str(value) for value in argv)
        print("[sequence:{}] $ {}".format(phase, shlex.join(command)), flush=True)
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=None if env is None else dict(env),
        )
        chunks: List[str] = []
        assert process.stdout is not None
        try:
            for line in process.stdout:
                chunks.append(line)
                print("[sequence:{}] {}".format(phase, line.rstrip("\n")), flush=True)
            returncode = process.wait()
        except KeyboardInterrupt:
            try:
                process.send_signal(signal.SIGINT)
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        return CommandResult(command, int(returncode), "".join(chunks))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, *, maximum_bytes: Optional[int] = None) -> str:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("not a regular file: {}".format(resolved))
    if maximum_bytes is not None and resolved.stat().st_size > maximum_bytes:
        raise ValueError("file is too large: {}".format(resolved))
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _read_yaml(
    path: Path, *, maximum_bytes: int
) -> Tuple[Path, bytes, Mapping[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("not a regular YAML file: {}".format(resolved))
    data = resolved.read_bytes()
    if len(data) > maximum_bytes:
        raise ValueError("YAML file is too large: {}".format(resolved))
    try:
        payload = yaml.load(data.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("invalid YAML in {}: {}".format(resolved, exc)) from exc
    if not isinstance(payload, dict):
        raise ValueError("{} must contain a YAML mapping".format(resolved))
    return resolved, data, payload


def _reject_duplicate_json_keys(pairs: Sequence[Tuple[str, Any]]) -> Mapping[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key {!r}".format(key))
        result[key] = value
    return result


def _read_json_mapping(
    path: Path, *, maximum_bytes: int
) -> Tuple[Path, bytes, Mapping[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("not a regular JSON file: {}".format(resolved))
    data = resolved.read_bytes()
    if len(data) > maximum_bytes:
        raise ValueError("JSON file is too large: {}".format(resolved))
    try:
        payload = json.loads(
            data.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("invalid JSON in {}: {}".format(resolved, exc)) from exc
    if not isinstance(payload, dict):
        raise ValueError("{} must contain a JSON mapping".format(resolved))
    return resolved, data, payload


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("{} must be a mapping".format(name))
    return value


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string".format(name))
    return value.strip()


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError("{} must be a finite number".format(name))
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be a finite number".format(name)) from exc
    if not math.isfinite(result):
        raise ValueError("{} must be a finite number".format(name))
    return result


def _lexical_absolute_path(value: Any, name: str) -> Path:
    text_value = _require_text(value, name)
    expanded = Path(text_value).expanduser()
    if not expanded.is_absolute():
        raise ValueError("{} must be an absolute path".format(name))
    return Path(os.path.normpath(str(expanded)))


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _publish_exclusive(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    """Atomically publish complete bytes and refuse any existing destination."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (
        ".{}.tmp-{}-{}".format(destination.name, os.getpid(), uuid.uuid4().hex)
    )
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(
            str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(str(temporary), mode)
        try:
            os.link(str(temporary), str(destination))
        except FileExistsError as exc:
            raise FileExistsError(
                "refusing to reuse existing immutable artifact: {}".format(destination)
            ) from exc
        directory_fd = os.open(str(destination.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_log(path: Path, result: CommandResult) -> Mapping[str, Any]:
    payload = result.output.encode("utf-8", errors="replace")
    _publish_exclusive(path, payload)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_bytes(payload),
        "returncode": result.returncode,
        "argv": list(result.argv),
    }


def _pose_map(master: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    entries = master.get("training_poses")
    if not isinstance(entries, list):
        raise ValueError("training_poses must be a list")
    result: Dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(entries):
        entry = _require_mapping(value, "training_poses[{}]".format(index))
        pose_id = _require_text(entry.get("id"), "training_poses[{}].id".format(index))
        if not _POSE_ID.fullmatch(pose_id):
            raise ValueError("malformed training pose id {!r}".format(pose_id))
        if pose_id in result:
            raise ValueError("duplicate training pose id {!r}".format(pose_id))
        result[pose_id] = entry
    return result


def _validate_master(
    master: Mapping[str, Any],
    *,
    master_sha256: str,
    master_path: Path,
    calibration_root: Path,
    claims_dir: Path,
    dataset: Path,
    artifacts_dir: Path,
    orchestrator_sha256: str,
    python_executable: Path,
    python_executable_resolved: Path,
    python_executable_sha256: str,
    motion_script_sha256: str,
    motion_driver_sha256: str,
    link_preflight_sha256: str,
    inspection_script_sha256: str,
    detector_sha256: str,
    stationary_sha256: str,
    transforms_sha256: str,
    capture_config_sha256: str,
    capture_cli_sha256: str,
    frame_telemetry_sha256: str,
) -> Tuple[
    Tuple[str, ...],
    Mapping[str, Mapping[str, Any]],
    str,
    int,
    str,
]:
    if master.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("master schema_version must be exactly 1")
    if master.get("kind") != PLAN_KIND:
        raise ValueError("master kind must be {!r}".format(PLAN_KIND))
    if master.get("status") != MASTER_RUN_STATUS:
        raise ValueError("master status must be {!r}".format(MASTER_RUN_STATUS))
    topology = _require_mapping(master.get("topology"), "topology")
    if (
        topology.get("type") != "eye_to_hand"
        or topology.get("camera_mount") != "fixed_external"
    ):
        raise ValueError("master must describe a fixed-external eye-to-hand camera")

    authorization = _require_mapping(
        master.get("motion_authorization"), "motion_authorization"
    )
    if authorization.get("explicit_user_authorization_recorded") is not True:
        raise ValueError("master lacks explicit continuous motion authorization")
    if authorization.get("scope") != MASTER_AUTHORIZATION_SCOPE:
        raise ValueError(
            "master motion authorization scope must be {!r}".format(
                MASTER_AUTHORIZATION_SCOPE
            )
        )
    if authorization.get("consumed") is not False:
        raise ValueError("master motion authorization consumed must be exactly false")
    _require_text(authorization.get("source_text"), "motion_authorization.source_text")
    _require_text(
        authorization.get("authorization_interpretation"),
        "motion_authorization.authorization_interpretation",
    )

    safety = _require_mapping(master.get("safety"), "safety")
    required_safety = (
        "emergency_stop_reachable",
        "full_large_board_bracket_and_cable_sequence_swept_volume_clear",
        "target_rigid_confirmed",
        "camera_fixed_confirmed",
    )
    for field in required_safety:
        if safety.get(field) is not True:
            raise ValueError("safety.{} must be exactly true".format(field))
    if safety.get("motion_authorized") is not True:
        raise ValueError("safety.motion_authorized must be exactly true")

    provenance = _require_mapping(master.get("provenance"), "provenance")
    declared_calibration_root = _lexical_absolute_path(
        provenance.get("calibration_root_path"),
        "provenance.calibration_root_path",
    )
    if declared_calibration_root != calibration_root:
        raise ValueError("master calibration-root path does not match the runner")
    declared_master_path = _lexical_absolute_path(
        provenance.get("master_plan_path"), "provenance.master_plan_path"
    )
    if declared_master_path != master_path:
        raise ValueError("master plan path does not match the reviewed file")
    declared_claims_dir = _lexical_absolute_path(
        provenance.get("canonical_claims_dir"),
        "provenance.canonical_claims_dir",
    )
    if declared_claims_dir != claims_dir:
        raise ValueError("master canonical-claims path does not match the runner")
    if provenance.get("training_orchestrator_sha256") != orchestrator_sha256:
        raise ValueError(
            "master training-orchestrator SHA does not match the executable"
        )
    declared_python_path = _lexical_absolute_path(
        provenance.get("python_executable_path"),
        "provenance.python_executable_path",
    )
    if declared_python_path != python_executable:
        raise ValueError("master Python launch path does not match the runner")
    declared_python_resolved = _require_text(
        provenance.get("python_executable_resolved_path"),
        "provenance.python_executable_resolved_path",
    )
    try:
        declared_python_resolved_path = (
            Path(declared_python_resolved).expanduser().resolve(strict=True)
        )
    except OSError as exc:
        raise ValueError("master Python executable is unavailable: {}".format(exc)) from exc
    if declared_python_resolved_path != python_executable_resolved:
        raise ValueError("master resolved Python executable does not match the runner")
    if provenance.get("python_executable_sha256") != python_executable_sha256:
        raise ValueError("master Python executable SHA does not match the runner")
    if provenance.get("motion_wrapper_sha256") != motion_script_sha256:
        raise ValueError("master motion-wrapper SHA does not match the executable")
    if provenance.get("motion_driver_sha256") != motion_driver_sha256:
        raise ValueError("master motion-driver SHA does not match the executable")
    if provenance.get("link_preflight_sha256") != link_preflight_sha256:
        raise ValueError("master link-preflight SHA does not match the executable")
    if provenance.get("inspection_script_sha256") != inspection_script_sha256:
        raise ValueError("master inspection-script SHA does not match the executable")
    if provenance.get("aruco_detector_sha256") != detector_sha256:
        raise ValueError("master ArUco-detector SHA does not match the capture code")
    if provenance.get("stationary_aggregator_sha256") != stationary_sha256:
        raise ValueError(
            "master stationary-aggregator SHA does not match the capture code"
        )
    if provenance.get("calibration_transforms_sha256") != transforms_sha256:
        raise ValueError(
            "master calibration-transforms SHA does not match the capture code"
        )
    if provenance.get("capture_config_sha256") != capture_config_sha256:
        raise ValueError("master capture-config SHA does not match the reviewed config")
    if provenance.get("frame_telemetry_sha256") != frame_telemetry_sha256:
        raise ValueError(
            "master frame-telemetry SHA does not match the telemetry executable"
        )
    if provenance.get("capture_cli_sha256") != capture_cli_sha256:
        raise ValueError("master capture-CLI SHA does not match the executable")

    sequence = _require_mapping(master.get("planned_sequence"), "planned_sequence")
    training = sequence.get("training")
    if not isinstance(training, list) or not all(
        isinstance(value, str) for value in training
    ):
        raise ValueError("planned_sequence.training must be a list of pose ids")
    if len(training) != len(set(training)):
        raise ValueError("planned_sequence.training contains duplicate pose ids")
    expected_training = tuple("T{:02d}".format(index) for index in range(1, 21))
    if tuple(training) != expected_training:
        raise ValueError("master training ids must be exactly T01,T02,...,T20")

    poses = _pose_map(master)
    for pose_id in expected_training:
        if pose_id not in poses:
            raise ValueError("planned pose {!r} is missing".format(pose_id))

    collection = _require_mapping(master.get("collection_state"), "collection_state")
    sample_count_raw = collection.get("current_training_sample_count")
    if (
        isinstance(sample_count_raw, bool)
        or not isinstance(sample_count_raw, int)
        or sample_count_raw < 1
        or sample_count_raw >= len(training)
    ):
        raise ValueError(
            "collection_state.current_training_sample_count must be an integer in [1, 19]"
        )
    sample_count = int(sample_count_raw)
    current_pose_id = _require_text(
        collection.get("current_pose_id"), "collection_state.current_pose_id"
    )
    expected_current_pose_id = str(training[sample_count - 1])
    if current_pose_id != expected_current_pose_id:
        raise ValueError(
            "collection_state.current_pose_id must be {!r} for sample count {}".format(
                expected_current_pose_id, sample_count
            )
        )
    next_pose_id = _require_text(
        collection.get("next_pose_id"), "collection_state.next_pose_id"
    )
    expected_next_pose_id = str(training[sample_count])
    if next_pose_id != expected_next_pose_id:
        raise ValueError(
            "collection_state.next_pose_id must be {!r}".format(expected_next_pose_id)
        )
    next_capture_index = collection.get("next_capture_index")
    if next_capture_index != sample_count + 1:
        raise ValueError(
            "collection_state.next_capture_index must be {}".format(sample_count + 1)
        )

    for index, pose_id in enumerate(training, start=1):
        pose = poses[str(pose_id)]
        state = _require_text(pose.get("state"), "pose[{}].state".format(pose_id))
        if index <= sample_count:
            if not state.startswith("captured") or pose.get("sample_index") != index:
                raise ValueError(
                    "pose {} must be captured with sample_index {}".format(
                        pose_id, index
                    )
                )
        elif state.startswith("captured") or "sample_index" in pose:
            raise ValueError(
                "uncaptured suffix pose {} may not claim a sample".format(pose_id)
            )

    if authorization.get("start_pose_id") != current_pose_id:
        raise ValueError(
            "motion authorization start_pose_id does not match collection state"
        )
    if authorization.get("pose_id") != next_pose_id:
        raise ValueError("motion authorization pose_id does not match collection state")

    remaining = tuple(str(value) for value in training[sample_count:])
    if (
        not remaining
        or remaining[0] != next_pose_id
        or remaining[-1] != LAST_REMAINING_POSE_ID
    ):
        raise ValueError(
            "collection state does not identify a strict remaining suffix through T20"
        )
    authorized_suffix = authorization.get("authorized_training_suffix")
    if not isinstance(authorized_suffix, list) or not all(
        isinstance(value, str) for value in authorized_suffix
    ):
        raise ValueError(
            "motion_authorization.authorized_training_suffix must be a list of pose ids"
        )
    if tuple(authorized_suffix) != remaining:
        raise ValueError(
            "motion_authorization.authorized_training_suffix must exactly match the "
            "remaining training suffix"
        )
    expected_authorized_edges = tuple(
        "{}_to_{}".format(start, target)
        for start, target in zip((current_pose_id,) + remaining[:-1], remaining)
    )
    authorized_edges = authorization.get("authorized_edges")
    if not isinstance(authorized_edges, list) or not all(
        isinstance(value, str) for value in authorized_edges
    ):
        raise ValueError("motion_authorization.authorized_edges must be a list")
    if tuple(authorized_edges) != expected_authorized_edges:
        raise ValueError(
            "motion_authorization.authorized_edges must exactly match every remaining "
            "training edge"
        )

    declared_dataset = _require_text(
        collection.get("formal_training_dataset"),
        "collection_state.formal_training_dataset",
    )
    declared_path = (calibration_root / declared_dataset).resolve()
    if declared_path != dataset.resolve():
        raise ValueError(
            "dataset does not match master collection_state: {} != {}".format(
                dataset.resolve(), declared_path
            )
        )

    declared_artifacts_text = _require_text(
        collection.get("training_sequence_artifacts_dir"),
        "collection_state.training_sequence_artifacts_dir",
    )
    declared_artifacts_relative = Path(declared_artifacts_text)
    if declared_artifacts_relative.is_absolute():
        raise ValueError(
            "collection_state.training_sequence_artifacts_dir must be relative to the "
            "calibration root"
        )
    declared_artifacts_path = (
        calibration_root / declared_artifacts_relative
    ).resolve()
    try:
        declared_artifacts_path.relative_to(calibration_root.resolve())
    except ValueError as exc:
        raise ValueError(
            "collection_state.training_sequence_artifacts_dir escapes the calibration root"
        ) from exc
    if declared_artifacts_path != artifacts_dir.resolve():
        raise ValueError(
            "artifacts directory does not match the master ledger path: {} != {}".format(
                artifacts_dir.resolve(), declared_artifacts_path
            )
        )

    initial_dataset_sha256 = _require_text(
        collection.get("initial_dataset_sha256"),
        "collection_state.initial_dataset_sha256",
    )
    if not re.fullmatch(r"[0-9a-f]{64}", initial_dataset_sha256):
        raise ValueError("collection_state.initial_dataset_sha256 is malformed")

    # The caller-supplied hash is intentionally copied into every edge plan.
    if not re.fullmatch(r"[0-9a-f]{64}", master_sha256):
        raise ValueError("master SHA-256 is malformed")
    return remaining, poses, current_pose_id, sample_count, initial_dataset_sha256


def _validate_capture_config(
    config: Mapping[str, Any], master: Mapping[str, Any]
) -> None:
    camera_cfg = _require_mapping(config.get("camera"), "capture config camera")
    hardware = _require_mapping(master.get("hardware"), "hardware")
    camera = _require_mapping(hardware.get("camera"), "hardware.camera")
    profile = _require_mapping(
        camera.get("color_profile"), "hardware.camera.color_profile"
    )
    for config_field, profile_field in (
        ("width", "width"),
        ("height", "height"),
        ("fps", "fps"),
    ):
        if camera_cfg.get(config_field) != profile.get(profile_field):
            raise ValueError(
                "capture config camera.{} does not match master color profile".format(
                    config_field
                )
            )


def _dataset_snapshot(path: Path) -> DatasetSnapshot:
    resolved, data, document = _read_yaml(path, maximum_bytes=MAX_DATASET_BYTES)
    if (
        document.get("schema_version") != 1
        or document.get("kind") != "eye_to_hand_dataset"
    ):
        raise ValueError("{} is not a schema-1 eye-to-hand dataset".format(resolved))
    samples = document.get("samples")
    if not isinstance(samples, list):
        raise ValueError("dataset samples must be a list")
    checked: List[Mapping[str, Any]] = []
    for index, sample in enumerate(samples):
        checked.append(_require_mapping(sample, "samples[{}]".format(index)))
    return DatasetSnapshot(resolved, _sha256_bytes(data), document, tuple(checked))


def _expected_dataset_metadata(master: Mapping[str, Any]) -> Mapping[str, Any]:
    hardware = _require_mapping(master.get("hardware"), "hardware")
    camera = _require_mapping(hardware.get("camera"), "hardware.camera")
    robot = _require_mapping(hardware.get("robot"), "hardware.robot")
    target = _require_mapping(master.get("target"), "target")
    if target.get("type") != "aruco":
        raise ValueError(
            "this training orchestrator supports the reviewed ArUco target only"
        )
    marker_id = target.get("marker_id")
    if isinstance(marker_id, bool) or not isinstance(marker_id, int) or marker_id < 0:
        raise ValueError("target.marker_id must be a non-negative integer")
    return {
        "camera_serial": _require_text(camera.get("serial"), "hardware.camera.serial"),
        "robot_ip": _require_text(robot.get("ip"), "hardware.robot.ip"),
        "target": {
            "type": _require_text(target.get("type"), "target.type"),
            "dictionary": _require_text(target.get("dictionary"), "target.dictionary"),
            "marker_id": marker_id,
            "marker_length_m": _finite_float(
                target.get("marker_length_m"), "target.marker_length_m"
            ),
        },
    }


def _validate_dataset_metadata(
    snapshot: DatasetSnapshot, expected: Mapping[str, Any]
) -> None:
    document = snapshot.document
    camera = _require_mapping(document.get("camera"), "dataset.camera")
    if str(camera.get("serial")) != expected["camera_serial"]:
        raise ValueError("dataset camera serial does not match the reviewed master")
    robot = _require_mapping(document.get("robot"), "dataset.robot")
    if str(robot.get("ip")) != expected["robot_ip"] or robot.get("pose") != "O_T_EE":
        raise ValueError("dataset robot metadata does not match the reviewed master")
    target = _require_mapping(document.get("target"), "dataset.target")
    expected_target = _require_mapping(expected.get("target"), "expected target")
    if target.get("type") != expected_target.get("type"):
        raise ValueError("dataset target type does not match the reviewed master")
    if target.get("dictionary") != expected_target.get("dictionary"):
        raise ValueError("dataset target dictionary does not match the reviewed master")
    if target.get("marker_id") != expected_target.get("marker_id"):
        raise ValueError("dataset marker id does not match the reviewed master")
    if not math.isclose(
        _finite_float(target.get("marker_length_m"), "dataset.target.marker_length_m"),
        float(expected_target["marker_length_m"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("dataset marker length does not match the reviewed master")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _rigid_transform(value: Any, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "{} must be a finite rigid 4x4 transform".format(name)
        ) from exc
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("{} must be a finite rigid 4x4 transform".format(name))
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-8, rtol=0.0):
        raise ValueError("{} has an invalid homogeneous row".format(name))
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-4, rtol=0.0):
        raise ValueError("{} rotation is not orthonormal".format(name))
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=2.0e-4):
        raise ValueError("{} rotation determinant is not +1".format(name))
    return matrix


def _transform_error(left: np.ndarray, right: np.ndarray) -> Tuple[float, float]:
    translation = float(np.linalg.norm(left[:3, 3] - right[:3, 3]))
    relative = left[:3, :3].T @ right[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return translation, math.degrees(math.acos(cosine))


def _so3_exp(rotation_vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotation_vector, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("rotation vector must contain three finite values")
    angle = float(np.linalg.norm(vector))
    if angle <= 1.0e-12:
        return np.eye(3)
    axis = vector / angle
    skew = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _planned_target_transform(
    master: Mapping[str, Any], pose: Mapping[str, Any], pose_id: str
) -> np.ndarray:
    reference = _require_mapping(master.get("reference_pose"), "reference_pose")
    anchor = _rigid_transform(reference.get("T_base_ee"), "reference_pose.T_base_ee")
    try:
        offset = np.asarray(pose.get("xyz_offset_base_m"), dtype=np.float64)
        rotation_deg = np.asarray(pose.get("rotation_vector_eef_deg"), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("pose {} has invalid offset/rotation".format(pose_id)) from exc
    if (
        offset.shape != (3,)
        or rotation_deg.shape != (3,)
        or not np.all(np.isfinite(offset))
        or not np.all(np.isfinite(rotation_deg))
    ):
        raise ValueError("pose {} has invalid offset/rotation".format(pose_id))
    target = anchor.copy()
    target[:3, 3] += offset
    target[:3, :3] = anchor[:3, :3] @ _so3_exp(np.radians(rotation_deg))
    return _rigid_transform(target, "pose {} target".format(pose_id))


def _validate_existing_sample_prefix(
    snapshot: DatasetSnapshot,
    *,
    master: Mapping[str, Any],
    poses: Mapping[str, Mapping[str, Any]],
) -> None:
    for index, sample in enumerate(snapshot.samples, start=1):
        pose_id = "T{:02d}".format(index)
        if pose_id not in poses:
            raise ValueError(
                "dataset sample {} has no matching training pose".format(index)
            )
        observed = _rigid_transform(
            sample.get("T_base_ee"), "sample {} T_base_ee".format(index)
        )
        expected = _planned_target_transform(master, poses[pose_id], pose_id)
        translation, rotation = _transform_error(observed, expected)
        if (
            translation > MAX_CAPTURE_EEF_TRANSLATION_ERROR_M
            or rotation > MAX_CAPTURE_EEF_ROTATION_ERROR_DEG
        ):
            raise ValueError(
                "dataset sample {} does not match {}: {:.3f}mm/{:.3f}deg".format(
                    index, pose_id, translation * 1000.0, rotation
                )
            )


def _validate_new_sample(
    sample: Mapping[str, Any], target_T_base_ee: np.ndarray
) -> None:
    timestamp = _finite_float(sample.get("timestamp"), "new sample timestamp")
    if timestamp <= 0.0:
        raise ValueError("new sample timestamp must be positive")
    frame_id = sample.get("frame_id")
    if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
        raise ValueError("new sample frame_id must be a non-negative integer")
    reprojection = _finite_float(
        sample.get("reprojection_error_px"), "new sample reprojection_error_px"
    )
    if reprojection > MAX_CAPTURE_REPROJECTION_P95_PX:
        raise ValueError(
            "new sample reprojection {:.6f}px exceeds {:.6f}px".format(
                reprojection, MAX_CAPTURE_REPROJECTION_P95_PX
            )
        )
    observed = _rigid_transform(sample.get("T_base_ee"), "new sample T_base_ee")
    _rigid_transform(sample.get("T_camera_target"), "new sample T_camera_target")
    translation, rotation = _transform_error(observed, target_T_base_ee)
    if (
        translation > MAX_CAPTURE_EEF_TRANSLATION_ERROR_M
        or rotation > MAX_CAPTURE_EEF_ROTATION_ERROR_DEG
    ):
        raise ValueError(
            "new sample EEF does not match edge target: {:.3f}mm/{:.3f}deg".format(
                translation * 1000.0, rotation
            )
        )


def _verify_dataset_increment(
    before: DatasetSnapshot,
    after: DatasetSnapshot,
    *,
    expected_metadata: Mapping[str, Any],
    expected_count: int,
    target_T_base_ee: np.ndarray,
) -> None:
    _validate_dataset_metadata(after, expected_metadata)
    if len(after.samples) != expected_count:
        raise ValueError(
            "dataset must contain exactly {} samples after capture; found {}".format(
                expected_count, len(after.samples)
            )
        )
    if len(after.samples) != len(before.samples) + 1:
        raise ValueError("capture did not append exactly one dataset sample")
    for index, old_sample in enumerate(before.samples):
        if _canonical(old_sample) != _canonical(after.samples[index]):
            raise ValueError(
                "capture changed pre-existing dataset sample {}".format(index + 1)
            )
    for key in ("camera", "target", "robot", "transform_convention", "units"):
        if _canonical(before.document.get(key)) != _canonical(after.document.get(key)):
            raise ValueError("capture changed stable dataset metadata {!r}".format(key))
    _validate_new_sample(after.samples[-1], target_T_base_ee)
    previous_T = _rigid_transform(
        before.samples[-1].get("T_base_ee"), "previous formal sample T_base_ee"
    )
    current_T = _rigid_transform(
        after.samples[-1].get("T_base_ee"), "new formal sample T_base_ee"
    )
    translation, rotation = _transform_error(previous_T, current_T)
    if (
        translation < MIN_FORMAL_SAMPLE_TRANSLATION_M
        and rotation < MIN_FORMAL_SAMPLE_ROTATION_DEG
    ):
        raise ValueError(
            "new formal sample is too close to previous sample: {:.3f}mm/{:.3f}deg".format(
                translation * 1000.0, rotation
            )
        )


def _derive_edge_plan(
    master: Mapping[str, Any],
    *,
    master_path: Path,
    master_sha256: str,
    start_pose_id: str,
    target_pose_id: str,
    poses: Mapping[str, Mapping[str, Any]],
    orchestrator_sha256: str,
    python_executable: Path,
    python_executable_resolved: Path,
    python_executable_sha256: str,
    capture_config: Path,
    capture_config_sha256: str,
    capture_cli_sha256: str,
    stationary: Path,
    stationary_sha256: str,
    transforms: Path,
    transforms_sha256: str,
    motion_driver: Path,
    motion_driver_sha256: str,
    link_preflight: Path,
    link_preflight_sha256: str,
    inspection_script: Path,
    inspection_script_sha256: str,
    frame_telemetry_script: Path,
    frame_telemetry_sha256: str,
) -> Mapping[str, Any]:
    edge = copy.deepcopy(dict(master))
    edge["status"] = RUN_STATUS
    edge["session_slug"] = "{}-edge-{}-to-{}".format(
        _require_text(master.get("session_slug"), "session_slug"),
        start_pose_id,
        target_pose_id,
    )
    edge["training_poses"] = [
        copy.deepcopy(dict(poses[start_pose_id])),
        copy.deepcopy(dict(poses[target_pose_id])),
    ]
    # A training transaction must be incapable of selecting a holdout pose.
    edge["holdout_poses"] = []
    edge.pop("recovery_poses", None)
    edge["planned_sequence"] = {
        "training": [start_pose_id, target_pose_id],
        "holdout_poses_included": False,
        "single_edge_only": True,
    }
    safety = copy.deepcopy(dict(_require_mapping(edge.get("safety"), "safety")))
    safety["motion_authorized"] = True
    edge["safety"] = safety

    source_authorization = _require_mapping(
        master.get("motion_authorization"), "motion_authorization"
    )
    edge["motion_authorization"] = {
        "explicit_user_authorization_recorded": True,
        "scope": EDGE_AUTHORIZATION_SCOPE,
        "start_pose_id": start_pose_id,
        "pose_id": target_pose_id,
        "source_text": _require_text(
            source_authorization.get("source_text"), "motion_authorization.source_text"
        ),
        "authorization_interpretation": _require_text(
            source_authorization.get("authorization_interpretation"),
            "motion_authorization.authorization_interpretation",
        ),
        "source_authorized_training_suffix": list(
            source_authorization.get("authorized_training_suffix", [])
        ),
        "source_authorized_edges": list(
            source_authorization.get("authorized_edges", [])
        ),
        "reviewed_master_plan": str(master_path),
        "reviewed_master_plan_sha256": master_sha256,
        "consumed": False,
    }
    provenance = copy.deepcopy(
        dict(_require_mapping(edge.get("provenance"), "provenance"))
    )
    provenance["headless_training_orchestrator"] = {
        "orchestrator_sha256": orchestrator_sha256,
        "python_executable": str(python_executable),
        "python_executable_resolved": str(python_executable_resolved),
        "python_executable_sha256": python_executable_sha256,
        "reviewed_master_plan": str(master_path),
        "reviewed_master_plan_sha256": master_sha256,
        "edge": "{}_to_{}".format(start_pose_id, target_pose_id),
        "capture_config": str(capture_config),
        "capture_config_sha256": capture_config_sha256,
        "capture_cli_sha256": capture_cli_sha256,
        "stationary_aggregator": str(stationary),
        "stationary_aggregator_sha256": stationary_sha256,
        "calibration_transforms": str(transforms),
        "calibration_transforms_sha256": transforms_sha256,
        "motion_driver": str(motion_driver),
        "motion_driver_sha256": motion_driver_sha256,
        "link_preflight": str(link_preflight),
        "link_preflight_sha256": link_preflight_sha256,
        "inspection_script": str(inspection_script),
        "inspection_script_sha256": inspection_script_sha256,
        "frame_telemetry_script": str(frame_telemetry_script),
        "frame_telemetry_sha256": frame_telemetry_sha256,
        "capture_frames": CAPTURE_FRAMES,
        "minimum_capture_coherent_frames": MIN_CAPTURE_COHERENT_FRAMES,
        "minimum_capture_all_reprojection_pass_frames": (
            MIN_CAPTURE_ALL_REPROJECTION_PASS_FRAMES
        ),
        "telemetry_frames": TELEMETRY_FRAMES,
        "minimum_telemetry_detected_valid_frames": (
            MIN_TELEMETRY_DETECTED_VALID_FRAMES
        ),
        "minimum_telemetry_reprojection_pass_frames": (
            MIN_TELEMETRY_REPROJECTION_PASS_FRAMES
        ),
        "minimum_telemetry_coherent_frames": MIN_TELEMETRY_COHERENT_FRAMES,
        "maximum_untrimmed_translation_jitter_p95_m": (
            MAX_CAPTURE_TARGET_TRANSLATION_JITTER_M
        ),
        "maximum_untrimmed_rotation_jitter_p95_deg": (
            MAX_CAPTURE_TARGET_ROTATION_JITTER_DEG
        ),
        "maximum_untrimmed_reprojection_error_p95_px": (
            MAX_CAPTURE_REPROJECTION_P95_PX
        ),
        "inspection_frames": INSPECTION_FRAMES,
        "minimum_live_marker_margin_px": MIN_LIVE_MARKER_MARGIN_PX,
        "minimum_live_marker_edge_px": MIN_LIVE_MARKER_EDGE_PX,
        "holdout_capture_permitted": False,
    }
    edge["provenance"] = provenance
    edge["collection_state"] = {
        "current_pose_id": start_pose_id,
        "next_pose_id": target_pose_id,
        "formal_training_dataset": _require_text(
            _require_mapping(master.get("collection_state"), "collection_state").get(
                "formal_training_dataset"
            ),
            "collection_state.formal_training_dataset",
        ),
        "initial_dataset_sha256": _require_text(
            _require_mapping(master.get("collection_state"), "collection_state").get(
                "initial_dataset_sha256"
            ),
            "collection_state.initial_dataset_sha256",
        ),
        "training_sequence_artifacts_dir": _require_text(
            _require_mapping(master.get("collection_state"), "collection_state").get(
                "training_sequence_artifacts_dir"
            ),
            "collection_state.training_sequence_artifacts_dir",
        ),
        "single_edge_transaction": True,
    }
    return edge


def _edge_stem(start_pose_id: str, target_pose_id: str, plan_sha256: str) -> str:
    return "{}-to-{}-{}".format(start_pose_id, target_pose_id, plan_sha256[:16])


def _claim_stem(
    start_pose_id: str, target_pose_id: str, master_sha256: str
) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", master_sha256):
        raise ValueError("master SHA-256 is malformed")
    return "{}-{}-to-{}".format(master_sha256, start_pose_id, target_pose_id)


def _require_success(result: CommandResult, phase: str) -> None:
    if result.returncode != 0:
        raise SequenceFailure(
            phase,
            "subprocess exited {} (output tail: {!r})".format(
                result.returncode, result.output[-1000:]
            ),
        )


def _parse_preview(
    result: CommandResult,
    *,
    plan_path: Path,
    plan_sha256: str,
    start_pose_id: str,
    target_pose_id: str,
) -> np.ndarray:
    _require_success(result, "preview")
    try:
        payload = json.loads(result.output)
    except json.JSONDecodeError as exc:
        raise SequenceFailure(
            "preview", "preview output is not one JSON object"
        ) from exc
    if not isinstance(payload, dict):
        raise SequenceFailure("preview", "preview JSON must be a mapping")
    checks = {
        "plan": str(plan_path.resolve()),
        "plan_sha256": plan_sha256,
        "pose_id": target_pose_id,
        "start_pose_id": start_pose_id,
        "status": RUN_STATUS,
        "run_authorized": True,
        "dynamics_ready": True,
        "camera_capture_performed": False,
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise SequenceFailure(
                "preview",
                "preview field {!r} mismatch: expected {!r}, got {!r}".format(
                    key, expected, payload.get(key)
                ),
            )
    try:
        return _rigid_transform(
            payload.get("target_T_base_ee"), "preview target_T_base_ee"
        )
    except ValueError as exc:
        raise SequenceFailure("preview", str(exc)) from exc


def _verify_motion_ready(
    result: CommandResult, *, target_pose_id: str, plan_sha256: str
) -> Mapping[str, Any]:
    _require_success(result, "motion")
    expected = "[calibration pose] CAPTURE_READY pose={} plan_sha256={}".format(
        target_pose_id, plan_sha256
    )
    if (
        expected not in result.output
        or "camera_capture_performed=false" not in result.output
    ):
        raise SequenceFailure(
            "motion", "motion did not emit the exact CAPTURE_READY receipt"
        )
    telemetry_prefix = "control_loop_telemetry="
    telemetry_index = result.output.rfind(telemetry_prefix)
    if telemetry_index < 0:
        raise SequenceFailure(
            "motion", "motion receipt is missing control-loop telemetry"
        )
    telemetry_text = (
        result.output[telemetry_index + len(telemetry_prefix) :].splitlines()[0].strip()
    )
    try:
        telemetry = json.loads(telemetry_text)
    except json.JSONDecodeError as exc:
        raise SequenceFailure(
            "motion", "control-loop telemetry is not valid JSON"
        ) from exc
    if (
        not isinstance(telemetry, dict)
        or telemetry.get("success_qualified") is not True
    ):
        raise SequenceFailure("motion", "control-loop success was not qualified")
    for field in (
        "samples",
        "max_read_to_write_ns",
        "read_to_write_overruns",
        "success_qualification_positive_writes",
    ):
        value = telemetry.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SequenceFailure(
                "motion", "invalid control-loop telemetry field {!r}".format(field)
            )
    if (
        telemetry["samples"] <= 0
        or telemetry["success_qualification_positive_writes"] <= 0
    ):
        raise SequenceFailure(
            "motion", "control-loop telemetry has no positive control evidence"
        )
    return telemetry


def _decode_image(path: Path, name: str, *, phase: str) -> np.ndarray:
    if not path.is_file() or path.stat().st_size <= 0:
        raise SequenceFailure(
            phase, "{} was not created as a non-empty file".format(name)
        )
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0 or image.ndim not in (2, 3):
        raise SequenceFailure(phase, "{} is not a decodable image".format(name))
    return image


def _validate_live_intrinsics(
    live: Mapping[str, Any],
    dataset_camera: Mapping[str, Any],
    *,
    phase: str = "inspect",
) -> Mapping[str, Any]:
    recorded = _require_mapping(
        dataset_camera.get("intrinsics"), "dataset.camera.intrinsics"
    )
    for field in ("width", "height"):
        if live.get(field) != recorded.get(field):
            raise SequenceFailure(phase, "live intrinsic {} changed".format(field))
    for field in ("fx", "fy", "ppx", "ppy"):
        live_value = _finite_float(live.get(field), "live intrinsics." + field)
        recorded_value = _finite_float(
            recorded.get(field), "dataset intrinsics." + field
        )
        if not math.isclose(
            live_value,
            recorded_value,
            rel_tol=0.0,
            abs_tol=INTRINSICS_ABSOLUTE_TOLERANCE,
        ):
            raise SequenceFailure(phase, "live intrinsic {} changed".format(field))
    if live.get("model") != recorded.get("model"):
        raise SequenceFailure(phase, "live distortion model changed")
    live_distortion = live.get("distortion")
    recorded_distortion = recorded.get("distortion")
    try:
        live_array = np.asarray(live_distortion, dtype=np.float64)
        recorded_array = np.asarray(recorded_distortion, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SequenceFailure(
            phase, "invalid live distortion coefficients"
        ) from exc
    if (
        live_array.shape != recorded_array.shape
        or not np.all(np.isfinite(live_array))
        or not np.allclose(
            live_array,
            recorded_array,
            atol=INTRINSICS_ABSOLUTE_TOLERANCE,
            rtol=0.0,
        )
    ):
        raise SequenceFailure(phase, "live distortion coefficients changed")
    return {
        "width": int(live["width"]),
        "height": int(live["height"]),
        "fx": float(live["fx"]),
        "fy": float(live["fy"]),
        "ppx": float(live["ppx"]),
        "ppy": float(live["ppy"]),
        "model": str(live["model"]),
        "distortion": live_array.tolist(),
    }


def _parse_inspection(
    result: CommandResult,
    *,
    expected_serial: str,
    target: Mapping[str, Any],
    dataset_camera: Mapping[str, Any],
    raw_image: Path,
    annotated_image: Path,
) -> Mapping[str, Any]:
    _require_success(result, "inspect")
    sentinel = "ARUCO_FRAME_INSPECTION_JSON="
    lines = [line for line in result.output.splitlines() if line.startswith(sentinel)]
    if len(lines) != 1:
        raise SequenceFailure(
            "inspect", "inspection output must contain one JSON sentinel"
        )
    try:
        payload = json.loads(lines[0][len(sentinel) :])
    except json.JSONDecodeError as exc:
        raise SequenceFailure(
            "inspect", "inspection sentinel is not valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise SequenceFailure("inspect", "inspection JSON must be a mapping")
    if str(payload.get("camera_serial")) != expected_serial:
        raise SequenceFailure("inspect", "inspection opened the wrong camera serial")
    if Path(str(payload.get("raw_output"))).resolve() != raw_image.resolve():
        raise SequenceFailure("inspect", "inspection raw-output path mismatch")
    if (
        Path(str(payload.get("annotated_output"))).resolve()
        != annotated_image.resolve()
    ):
        raise SequenceFailure("inspect", "inspection annotated-output path mismatch")
    raw = _decode_image(raw_image, "inspection raw image", phase="inspect")
    annotated = _decode_image(
        annotated_image, "inspection annotated image", phase="inspect"
    )
    if raw.shape[:2] != annotated.shape[:2]:
        raise SequenceFailure("inspect", "inspection image dimensions disagree")
    live_intrinsics = _require_mapping(
        payload.get("intrinsics"), "inspection.intrinsics"
    )
    intrinsics = _validate_live_intrinsics(live_intrinsics, dataset_camera)
    live_depth_scale = _finite_float(
        payload.get("depth_scale"), "inspection.depth_scale"
    )
    recorded_depth_scale = _finite_float(
        dataset_camera.get("depth_scale"), "dataset.camera.depth_scale"
    )
    if not math.isclose(
        live_depth_scale,
        recorded_depth_scale,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise SequenceFailure("inspect", "live depth scale changed")
    if raw.shape[1] != intrinsics["width"] or raw.shape[0] != intrinsics["height"]:
        raise SequenceFailure(
            "inspect", "inspection image dimensions disagree with intrinsics"
        )
    detections = payload.get("detections")
    if not isinstance(detections, list):
        raise SequenceFailure("inspect", "inspection detections must be a list")
    matches = [
        detection
        for detection in detections
        if isinstance(detection, dict)
        and detection.get("dictionary") == target.get("dictionary")
        and detection.get("marker_id") == target.get("marker_id")
    ]
    if len(matches) != 1:
        raise SequenceFailure(
            "inspect", "inspection must find exactly one reviewed target"
        )
    detection = matches[0]
    margin = _finite_float(
        detection.get("minimum_image_margin_px"), "target minimum image margin"
    )
    edge = _finite_float(detection.get("edge_min_px"), "target minimum edge")
    if detection.get("complete_in_image") is not True:
        raise SequenceFailure(
            "inspect", "reviewed target is incomplete in the live image"
        )
    if margin < MIN_LIVE_MARKER_MARGIN_PX:
        raise SequenceFailure(
            "inspect",
            "live marker margin {:.3f}px is below {:.3f}px".format(
                margin, MIN_LIVE_MARKER_MARGIN_PX
            ),
        )
    if edge < MIN_LIVE_MARKER_EDGE_PX:
        raise SequenceFailure(
            "inspect",
            "live marker edge {:.3f}px is below {:.3f}px".format(
                edge, MIN_LIVE_MARKER_EDGE_PX
            ),
        )
    return {
        "camera_serial": expected_serial,
        "intrinsics": intrinsics,
        "depth_scale": live_depth_scale,
        "minimum_image_margin_px": margin,
        "minimum_marker_edge_px": edge,
        "focus_laplacian_variance": _finite_float(
            payload.get("focus_laplacian_variance"), "inspection focus"
        ),
        "raw_image": {
            "path": str(raw_image),
            "sha256": _sha256_file(raw_image),
        },
        "annotated_image": {
            "path": str(annotated_image),
            "sha256": _sha256_file(annotated_image),
        },
    }


def _telemetry_count(value: Any, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > TELEMETRY_FRAMES
    ):
        raise SequenceFailure(
            "telemetry",
            "{} must be an integer in [0, {}]".format(name, TELEMETRY_FRAMES),
        )
    return int(value)


def _telemetry_indices(value: Any, name: str, expected_count: int) -> Tuple[int, ...]:
    if not isinstance(value, list) or len(value) != expected_count:
        raise SequenceFailure(
            "telemetry", "{} count does not match telemetry summary".format(name)
        )
    result: List[int] = []
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
            or item >= TELEMETRY_FRAMES
        ):
            raise SequenceFailure("telemetry", "{} contains an invalid index".format(name))
        result.append(int(item))
    if len(result) != len(set(result)):
        raise SequenceFailure("telemetry", "{} contains duplicate indices".format(name))
    return tuple(result)


def _parse_frame_telemetry(
    result: CommandResult,
    *,
    telemetry_path: Path,
    expected_serial: str,
    target: Mapping[str, Any],
    dataset_camera: Mapping[str, Any],
    capture_config: Path,
) -> Mapping[str, Any]:
    _require_success(result, "telemetry")
    sentinel = "ARUCO_FRAME_TELEMETRY_JSON="
    lines = [line for line in result.output.splitlines() if line.startswith(sentinel)]
    if len(lines) != 1:
        raise SequenceFailure(
            "telemetry", "telemetry output must contain one JSON sentinel"
        )
    try:
        summary = json.loads(
            lines[0][len(sentinel) :],
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise SequenceFailure("telemetry", "telemetry sentinel is invalid JSON") from exc
    if not isinstance(summary, dict):
        raise SequenceFailure("telemetry", "telemetry sentinel must be a mapping")

    try:
        resolved, data, payload = _read_json_mapping(
            telemetry_path, maximum_bytes=MAX_TELEMETRY_BYTES
        )
    except (OSError, ValueError) as exc:
        raise SequenceFailure("telemetry", str(exc)) from exc
    if resolved != telemetry_path.resolve():
        raise SequenceFailure("telemetry", "telemetry output path changed")
    if payload.get("schema_version") != 1:
        raise SequenceFailure("telemetry", "telemetry schema_version must be exactly 1")
    if payload.get("kind") != "realsense_aruco_frame_telemetry":
        raise SequenceFailure("telemetry", "telemetry kind is invalid")
    if payload.get("scope") != {
        "camera_only": True,
        "read_only": True,
        "franka_fci_opened": False,
        "robot_motion_commanded": False,
    }:
        raise SequenceFailure("telemetry", "telemetry scope is not camera-only/read-only")

    camera = _require_mapping(payload.get("camera"), "telemetry.camera")
    if (
        str(camera.get("requested_serial")) != expected_serial
        or str(camera.get("opened_serial")) != expected_serial
    ):
        raise SequenceFailure("telemetry", "telemetry opened the wrong camera serial")
    if camera.get("intrinsics_consistent_across_frames") is not True:
        raise SequenceFailure("telemetry", "telemetry intrinsics changed across frames")
    if Path(str(camera.get("config_path"))).resolve() != capture_config.resolve():
        raise SequenceFailure("telemetry", "telemetry capture-config path mismatch")
    live_intrinsics = _require_mapping(
        camera.get("intrinsics"), "telemetry.camera.intrinsics"
    )
    intrinsics = _validate_live_intrinsics(
        live_intrinsics, dataset_camera, phase="telemetry"
    )
    depth_scale = _finite_float(
        camera.get("depth_scale"), "telemetry.camera.depth_scale"
    )
    recorded_depth_scale = _finite_float(
        dataset_camera.get("depth_scale"), "dataset.camera.depth_scale"
    )
    if not math.isclose(
        depth_scale, recorded_depth_scale, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise SequenceFailure("telemetry", "telemetry depth scale changed")

    observed_target = _require_mapping(payload.get("target"), "telemetry.target")
    if (
        observed_target.get("dictionary") != target.get("dictionary")
        or observed_target.get("marker_id") != target.get("marker_id")
        or not math.isclose(
            _finite_float(
                observed_target.get("marker_length_m"),
                "telemetry.target.marker_length_m",
            ),
            float(target["marker_length_m"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise SequenceFailure("telemetry", "telemetry target does not match the master")

    capture = _require_mapping(payload.get("capture"), "telemetry.capture")
    requested = _telemetry_count(
        capture.get("requested_frame_count"), "requested_frame_count"
    )
    captured = _telemetry_count(
        capture.get("captured_frame_count"), "captured_frame_count"
    )
    if requested != TELEMETRY_FRAMES or captured != TELEMETRY_FRAMES:
        raise SequenceFailure(
            "telemetry", "telemetry did not capture exactly 120/120 frames"
        )
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) != TELEMETRY_FRAMES:
        raise SequenceFailure("telemetry", "telemetry frame-record count is not 120")
    capture_indices: List[int] = []
    for index, frame in enumerate(frames):
        record = _require_mapping(frame, "telemetry.frames[{}]".format(index))
        capture_index = record.get("capture_index")
        if capture_index != index:
            raise SequenceFailure(
                "telemetry", "telemetry frame indices are not contiguous"
            )
        capture_indices.append(int(capture_index))

    thresholds = _require_mapping(payload.get("thresholds"), "telemetry.thresholds")
    for field, expected in (
        ("max_reprojection_error_px", MAX_CAPTURE_REPROJECTION_P95_PX),
        (
            "max_translation_deviation_m",
            MAX_CAPTURE_TARGET_TRANSLATION_JITTER_M,
        ),
        (
            "max_rotation_deviation_deg",
            MAX_CAPTURE_TARGET_ROTATION_JITTER_DEG,
        ),
    ):
        if not math.isclose(
            _finite_float(thresholds.get(field), "telemetry.thresholds." + field),
            float(expected),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise SequenceFailure(
                "telemetry", "telemetry threshold {!r} changed".format(field)
            )

    coherence = _require_mapping(payload.get("coherence"), "telemetry.coherence")
    detected = _telemetry_count(
        coherence.get("detected_valid_count"), "detected_valid_count"
    )
    reprojection_pass = _telemetry_count(
        coherence.get("reprojection_gate_pass_count"),
        "reprojection_gate_pass_count",
    )
    coherent = _telemetry_count(
        coherence.get("coherent_pose_count"), "coherent_pose_count"
    )
    if not (coherent <= reprojection_pass <= detected <= captured):
        raise SequenceFailure("telemetry", "telemetry counts are internally inconsistent")
    if detected < MIN_TELEMETRY_DETECTED_VALID_FRAMES:
        raise SequenceFailure(
            "telemetry",
            "telemetry valid detections {}/120 are below 118".format(detected),
        )
    if reprojection_pass < MIN_TELEMETRY_REPROJECTION_PASS_FRAMES:
        raise SequenceFailure(
            "telemetry",
            "telemetry reprojection-pass frames {}/120 are below 118".format(
                reprojection_pass
            ),
        )
    if coherent < MIN_TELEMETRY_COHERENT_FRAMES:
        raise SequenceFailure(
            "telemetry",
            "telemetry coherent frames {}/120 are below 114".format(coherent),
        )
    coherent_fraction = _finite_float(
        coherence.get("coherent_pose_fraction"), "telemetry coherent fraction"
    )
    if not math.isclose(
        coherent_fraction,
        coherent / float(TELEMETRY_FRAMES),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise SequenceFailure("telemetry", "telemetry coherent fraction is inconsistent")

    invalid = _telemetry_indices(
        coherence.get("invalid_detection_capture_indices"),
        "invalid_detection_capture_indices",
        captured - detected,
    )
    reprojection_outliers = _telemetry_indices(
        coherence.get("reprojection_outlier_capture_indices"),
        "reprojection_outlier_capture_indices",
        detected - reprojection_pass,
    )
    coherent_indices = _telemetry_indices(
        coherence.get("coherent_capture_indices"),
        "coherent_capture_indices",
        coherent,
    )
    coherence_outliers = _telemetry_indices(
        coherence.get("coherence_outlier_capture_indices"),
        "coherence_outlier_capture_indices",
        reprojection_pass - coherent,
    )
    partition = invalid + reprojection_outliers + coherent_indices + coherence_outliers
    if len(partition) != TELEMETRY_FRAMES or set(partition) != set(capture_indices):
        raise SequenceFailure("telemetry", "telemetry index classes do not partition frames")
    medoid_index = coherence.get("medoid_capture_index")
    if medoid_index not in coherent_indices:
        raise SequenceFailure("telemetry", "telemetry medoid is not a coherent frame")
    try:
        _rigid_transform(
            coherence.get("aggregate_T_camera_target"),
            "telemetry aggregate_T_camera_target",
        )
    except ValueError as exc:
        raise SequenceFailure("telemetry", str(exc)) from exc

    p95_fields = {
        "all_reprojection_pass_translation_jitter_p95_m": (
            MAX_CAPTURE_TARGET_TRANSLATION_JITTER_M
        ),
        "all_reprojection_pass_rotation_jitter_p95_deg": (
            MAX_CAPTURE_TARGET_ROTATION_JITTER_DEG
        ),
        "all_reprojection_pass_reprojection_error_p95_px": (
            MAX_CAPTURE_REPROJECTION_P95_PX
        ),
    }
    p95_values: Dict[str, float] = {}
    for field, maximum in p95_fields.items():
        value = _finite_float(coherence.get(field), "telemetry.coherence." + field)
        if value < 0.0 or value > float(maximum):
            raise SequenceFailure(
                "telemetry",
                "telemetry {} {:.9g} exceeds {:.9g}".format(field, value, maximum),
            )
        p95_values[field] = value

    abnormal_images = payload.get("abnormal_images")
    if abnormal_images != []:
        raise SequenceFailure(
            "telemetry", "unexpected abnormal-image side effects in telemetry phase"
        )
    sentinel_checks = {
        "output": str(telemetry_path.resolve()),
        "camera_serial": expected_serial,
        "captured": captured,
        "valid": detected,
        "reprojection_gate_pass": reprojection_pass,
        "coherent": coherent,
        "abnormal_images": 0,
    }
    for key, expected in sentinel_checks.items():
        if summary.get(key) != expected:
            raise SequenceFailure(
                "telemetry", "telemetry sentinel field {!r} mismatch".format(key)
            )
    return {
        "path": str(telemetry_path.resolve()),
        "sha256": _sha256_bytes(data),
        "camera_serial": expected_serial,
        "intrinsics": intrinsics,
        "depth_scale": depth_scale,
        "requested_frame_count": requested,
        "captured_frame_count": captured,
        "detected_valid_count": detected,
        "reprojection_gate_pass_count": reprojection_pass,
        "coherent_pose_count": coherent,
        **p95_values,
    }


def _parse_capture_output(
    result: CommandResult, *, expected_count: int
) -> Mapping[str, Any]:
    _require_success(result, "capture")
    matches = list(_CAPTURE_LINE.finditer(result.output))
    if len(matches) != 1:
        raise SequenceFailure(
            "capture", "capture output must contain exactly one gate summary"
        )
    values = matches[0].groupdict()
    sample_count = int(values["sample"])
    coherent = int(values["coherent"])
    frames = int(values["frames"])
    translation_mm = float(values["translation"])
    rotation_deg = float(values["rotation"])
    reprojection_px = float(values["reprojection"])
    if sample_count != expected_count:
        raise SequenceFailure("capture", "capture reported an unexpected sample count")
    if (
        frames != CAPTURE_FRAMES
        or coherent < MIN_CAPTURE_COHERENT_FRAMES
        or coherent > CAPTURE_FRAMES
    ):
        raise SequenceFailure(
            "capture",
            "capture did not pass the 114/120 coherent-frame gate",
        )
    if translation_mm > MAX_CAPTURE_TARGET_TRANSLATION_JITTER_M * 1000.0:
        raise SequenceFailure(
            "capture", "capture target translation jitter exceeded its gate"
        )
    if rotation_deg > MAX_CAPTURE_TARGET_ROTATION_JITTER_DEG:
        raise SequenceFailure(
            "capture", "capture target rotation jitter exceeded its gate"
        )
    if reprojection_px > MAX_CAPTURE_REPROJECTION_P95_PX:
        raise SequenceFailure("capture", "capture reprojection p95 exceeded its gate")

    sentinel_lines = [
        line
        for line in result.output.splitlines()
        if line.startswith(_CAPTURE_UNTRIMMED_SENTINEL)
    ]
    if len(sentinel_lines) != 1:
        raise SequenceFailure(
            "capture",
            "capture output must contain exactly one same-batch untrimmed p95 sentinel",
        )
    try:
        untrimmed = json.loads(
            sentinel_lines[0][len(_CAPTURE_UNTRIMMED_SENTINEL) :],
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise SequenceFailure(
            "capture", "capture same-batch untrimmed p95 sentinel is invalid JSON"
        ) from exc
    expected_keys = {
        "requested_frame_count",
        "all_reprojection_pass_count",
        "all_reprojection_pass_translation_jitter_p95_m",
        "all_reprojection_pass_rotation_jitter_p95_deg",
        "all_reprojection_pass_reprojection_error_p95_px",
    }
    if not isinstance(untrimmed, dict) or set(untrimmed) != expected_keys:
        raise SequenceFailure(
            "capture",
            "capture same-batch untrimmed p95 sentinel has an unexpected schema",
        )
    requested = untrimmed["requested_frame_count"]
    all_pass = untrimmed["all_reprojection_pass_count"]
    if (
        isinstance(requested, bool)
        or not isinstance(requested, int)
        or requested != CAPTURE_FRAMES
    ):
        raise SequenceFailure(
            "capture", "capture same-batch requested frame count must be 120"
        )
    if (
        isinstance(all_pass, bool)
        or not isinstance(all_pass, int)
        or all_pass < MIN_CAPTURE_ALL_REPROJECTION_PASS_FRAMES
        or all_pass > CAPTURE_FRAMES
    ):
        raise SequenceFailure(
            "capture",
            "capture same-batch reprojection-pass frames must be at least 118/120",
        )
    p95_specs = (
        (
            "all_reprojection_pass_translation_jitter_p95_m",
            MAX_CAPTURE_TARGET_TRANSLATION_JITTER_M,
        ),
        (
            "all_reprojection_pass_rotation_jitter_p95_deg",
            MAX_CAPTURE_TARGET_ROTATION_JITTER_DEG,
        ),
        (
            "all_reprojection_pass_reprojection_error_p95_px",
            MAX_CAPTURE_REPROJECTION_P95_PX,
        ),
    )
    checked_untrimmed: Dict[str, Any] = {
        "requested_frame_count": requested,
        "all_reprojection_pass_count": all_pass,
    }
    for key, limit in p95_specs:
        try:
            value = _finite_float(untrimmed[key], "capture." + key)
        except ValueError as exc:
            raise SequenceFailure("capture", str(exc)) from exc
        if value < 0.0 or value > limit:
            raise SequenceFailure(
                "capture",
                "capture same-batch {} exceeded its gate".format(key),
            )
        checked_untrimmed[key] = value
    return {
        "sample_count": sample_count,
        "coherent_frames": coherent,
        "requested_frames": frames,
        "translation_jitter_p95_mm": translation_mm,
        "rotation_jitter_p95_deg": rotation_deg,
        "reprojection_error_p95_px": reprojection_px,
        "formal_same_batch_untrimmed": checked_untrimmed,
    }


def _ensure_unchanged(path: Path, expected_sha256: str, phase: str) -> None:
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise SequenceFailure(
            phase,
            "immutable input changed: {} expected={} actual={}".format(
                path.resolve(), expected_sha256, actual
            ),
        )


def _ensure_python_unchanged(
    launch_path: Path,
    expected_resolved_path: Path,
    expected_sha256: str,
    phase: str,
) -> None:
    try:
        actual_resolved = launch_path.resolve(strict=True)
    except OSError as exc:
        raise SequenceFailure(
            phase, "Python launch path became unavailable: {}".format(exc)
        ) from exc
    if actual_resolved != expected_resolved_path:
        raise SequenceFailure(
            phase,
            "Python launch path target changed: expected={} actual={}".format(
                expected_resolved_path, actual_resolved
            ),
        )
    _ensure_unchanged(actual_resolved, expected_sha256, phase)


def _safe_failure_state(dataset: Path) -> Mapping[str, Any]:
    try:
        snapshot = _dataset_snapshot(dataset)
    except (OSError, ValueError) as exc:
        return {"dataset_read_error": str(exc)}
    return {
        "dataset_sha256": snapshot.sha256,
        "dataset_sample_count": len(snapshot.samples),
    }


def _clean_subprocess_environment(calibration_root: Path) -> Mapping[str, str]:
    environment = dict(os.environ)
    for field in ("PYTHONHOME", "PYTHONUSERBASE", "PYTHONSTARTUP"):
        environment.pop(field, None)
    environment["PYTHONPATH"] = str(calibration_root.resolve())
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _parse_import_probe(
    result: CommandResult,
    *,
    capture_cli: Path,
    detector: Path,
    stationary: Path,
    transforms: Path,
) -> Mapping[str, str]:
    _require_success(result, "import_preflight")
    lines = [
        line
        for line in result.output.splitlines()
        if line.startswith(_IMPORT_PROBE_SENTINEL)
    ]
    if len(lines) != 1:
        raise SequenceFailure(
            "import_preflight", "Python import probe must emit one path sentinel"
        )
    try:
        payload = json.loads(
            lines[0][len(_IMPORT_PROBE_SENTINEL) :],
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise SequenceFailure(
            "import_preflight", "Python import probe emitted invalid JSON"
        ) from exc
    expected = {
        "dynamic_pcd.apps.calibrate_eye_to_hand": str(capture_cli.resolve()),
        "dynamic_pcd.calibration.aruco": str(detector.resolve()),
        "dynamic_pcd.calibration.stationary": str(stationary.resolve()),
        "dynamic_pcd.calibration.transforms": str(transforms.resolve()),
    }
    if payload != expected:
        raise SequenceFailure(
            "import_preflight",
            "Python import resolution does not match the reviewed calibration sources",
        )
    return expected


def _canonical_motion_dependencies(motion_script: Path) -> Tuple[Path, Path]:
    resolved = motion_script.resolve(strict=True)
    try:
        workspace_root = resolved.parents[3]
    except IndexError as exc:
        raise ValueError(
            "motion wrapper path cannot identify its canonical workspace root"
        ) from exc
    package_root = workspace_root / "dexgrasp" / "src" / "anydex_pipeline"
    return (
        (package_root / "franka_sequence_driver.py").resolve(strict=True),
        (package_root / "host_network_preflight.py").resolve(strict=True),
    )


def run_sequence(
    config: OrchestratorConfig,
    *,
    runner: Optional[Any] = None,
    clock: Callable[[], str] = _utc_now,
) -> SequenceResult:
    """Execute only the strict uncaptured suffix declared by the master."""

    command_runner = runner if runner is not None else StreamingSubprocessRunner()
    master_path, master_bytes, master = _read_yaml(
        config.master_plan, maximum_bytes=MAX_PLAN_BYTES
    )
    master_sha256 = _sha256_bytes(master_bytes)
    if config.expected_master_sha256 != master_sha256:
        raise SequenceFailure(
            "preflight",
            "reviewed master SHA mismatch: expected={} actual={}".format(
                config.expected_master_sha256, master_sha256
            ),
        )

    calibration_root = config.calibration_root.expanduser().resolve(strict=True)
    if not calibration_root.is_dir():
        raise SequenceFailure("preflight", "calibration root is not a directory")
    try:
        python_executable = _lexical_absolute_path(
            config.python_executable, "python executable"
        )
    except ValueError as exc:
        raise SequenceFailure("preflight", str(exc)) from exc
    if not python_executable.is_file():
        raise SequenceFailure("preflight", "Python launch path is not a file")
    python_executable_resolved = python_executable.resolve(strict=True)
    motion_script = config.motion_script.expanduser().resolve(strict=True)
    motion_driver = config.motion_driver.expanduser().resolve(strict=True)
    link_preflight = config.link_preflight.expanduser().resolve(strict=True)
    inspection_script = config.inspection_script.expanduser().resolve(strict=True)
    frame_telemetry_script = (
        config.frame_telemetry_script.expanduser().resolve(strict=True)
    )
    capture_config = config.capture_config.expanduser().resolve(strict=True)
    capture_cli = (
        calibration_root / "dynamic_pcd" / "apps" / "calibrate_eye_to_hand.py"
    ).resolve(strict=True)
    detector = (calibration_root / "dynamic_pcd" / "calibration" / "aruco.py").resolve(
        strict=True
    )
    stationary = (
        calibration_root / "dynamic_pcd" / "calibration" / "stationary.py"
    ).resolve(strict=True)
    transforms = (
        calibration_root / "dynamic_pcd" / "calibration" / "transforms.py"
    ).resolve(strict=True)
    try:
        canonical_motion_driver, canonical_link_preflight = (
            _canonical_motion_dependencies(motion_script)
        )
    except (OSError, ValueError) as exc:
        raise SequenceFailure("preflight", str(exc)) from exc
    if motion_driver != canonical_motion_driver:
        raise SequenceFailure(
            "preflight",
            "motion-driver path is not the canonical module imported by the wrapper",
        )
    if link_preflight != canonical_link_preflight:
        raise SequenceFailure(
            "preflight",
            "link-preflight path is not the canonical module imported by the wrapper",
        )
    orchestrator_path = Path(__file__).resolve(strict=True)
    motion_script_sha256 = _sha256_file(motion_script, maximum_bytes=MAX_PLAN_BYTES)
    motion_driver_sha256 = _sha256_file(motion_driver, maximum_bytes=MAX_PLAN_BYTES)
    link_preflight_sha256 = _sha256_file(link_preflight, maximum_bytes=MAX_PLAN_BYTES)
    inspection_script_sha256 = _sha256_file(
        inspection_script, maximum_bytes=MAX_PLAN_BYTES
    )
    frame_telemetry_sha256 = _sha256_file(
        frame_telemetry_script, maximum_bytes=MAX_PLAN_BYTES
    )
    capture_config_sha256 = _sha256_file(capture_config, maximum_bytes=MAX_PLAN_BYTES)
    capture_cli_sha256 = _sha256_file(capture_cli, maximum_bytes=MAX_PLAN_BYTES)
    detector_sha256 = _sha256_file(detector, maximum_bytes=MAX_PLAN_BYTES)
    stationary_sha256 = _sha256_file(stationary, maximum_bytes=MAX_PLAN_BYTES)
    transforms_sha256 = _sha256_file(transforms, maximum_bytes=MAX_PLAN_BYTES)
    orchestrator_sha256 = _sha256_file(orchestrator_path, maximum_bytes=MAX_PLAN_BYTES)
    python_executable_sha256 = _sha256_file(python_executable_resolved)

    dataset = config.dataset.expanduser().resolve(strict=True)
    artifacts_dir = config.artifacts_dir.expanduser().resolve()
    claims_dir = (
        calibration_root / "calibration_runs" / ".training-edge-claims"
    ).resolve()
    try:
        (
            remaining,
            poses,
            current_pose_id,
            initial_sample_count,
            initial_dataset_sha256,
        ) = _validate_master(
            master,
            master_sha256=master_sha256,
            master_path=master_path,
            calibration_root=calibration_root,
            claims_dir=claims_dir,
            dataset=dataset,
            artifacts_dir=artifacts_dir,
            orchestrator_sha256=orchestrator_sha256,
            python_executable=python_executable,
            python_executable_resolved=python_executable_resolved,
            python_executable_sha256=python_executable_sha256,
            motion_script_sha256=motion_script_sha256,
            motion_driver_sha256=motion_driver_sha256,
            link_preflight_sha256=link_preflight_sha256,
            inspection_script_sha256=inspection_script_sha256,
            detector_sha256=detector_sha256,
            stationary_sha256=stationary_sha256,
            transforms_sha256=transforms_sha256,
            capture_config_sha256=capture_config_sha256,
            capture_cli_sha256=capture_cli_sha256,
            frame_telemetry_sha256=frame_telemetry_sha256,
        )
        _, _, capture_config_document = _read_yaml(
            capture_config, maximum_bytes=MAX_PLAN_BYTES
        )
        _validate_capture_config(capture_config_document, master)
        expected_metadata = _expected_dataset_metadata(master)
        snapshot = _dataset_snapshot(dataset)
        _validate_dataset_metadata(snapshot, expected_metadata)
    except ValueError as exc:
        raise SequenceFailure("preflight", str(exc)) from exc
    if len(snapshot.samples) != initial_sample_count:
        raise SequenceFailure(
            "preflight",
            "training dataset must match master count {}; found {}".format(
                initial_sample_count, len(snapshot.samples)
            ),
        )
    if snapshot.sha256 != initial_dataset_sha256:
        raise SequenceFailure(
            "preflight",
            "training dataset SHA does not match collection_state.initial_dataset_sha256: "
            "expected={} actual={}".format(initial_dataset_sha256, snapshot.sha256),
        )
    try:
        _validate_existing_sample_prefix(snapshot, master=master, poses=poses)
    except ValueError as exc:
        raise SequenceFailure("preflight", str(exc)) from exc

    first_claim_path = claims_dir / (
        _claim_stem(current_pose_id, remaining[0], master_sha256) + ".claim.json"
    )
    if first_claim_path.exists():
        raise SequenceFailure(
            "preflight",
            "canonical edge claim already exists; automatic retry is forbidden: {}".format(
                first_claim_path
            ),
        )

    subprocess_environment = _clean_subprocess_environment(calibration_root)
    import_probe_command = (
        str(python_executable),
        "-I",
        "-c",
        _IMPORT_PROBE_CODE,
        str(calibration_root),
    )
    import_probe_result = command_runner.run(
        import_probe_command,
        cwd=calibration_root,
        phase="import_preflight",
        env=subprocess_environment,
    )
    import_resolution = _parse_import_probe(
        import_probe_result,
        capture_cli=capture_cli,
        detector=detector,
        stationary=stationary,
        transforms=transforms,
    )

    edges_dir = artifacts_dir / "edges"
    edges_dir.mkdir(parents=True, exist_ok=True)
    claims_dir.mkdir(parents=True, exist_ok=True)
    import_probe_log = _write_log(
        artifacts_dir / "python-import-preflight.log", import_probe_result
    )
    print(
        "[sequence] BEGIN master_sha256={} dataset_samples={} range={}..{}".format(
            master_sha256,
            len(snapshot.samples),
            remaining[0],
            remaining[-1],
        ),
        flush=True,
    )

    initial_start_pose_id = current_pose_id
    completed: List[str] = []
    completion_records: List[Mapping[str, Any]] = []

    for target_pose_id in remaining:
        edge_label = "{}_to_{}".format(current_pose_id, target_pose_id)
        phase = "derive"
        plan_path: Optional[Path] = None
        plan_sha256: Optional[str] = None
        claim_path: Optional[Path] = None
        claim_sha256: Optional[str] = None
        logs: Dict[str, Mapping[str, Any]] = {}
        try:
            print(
                "[sequence] EDGE_BEGIN {} sample_before={} dataset_sha256={}".format(
                    edge_label, len(snapshot.samples), snapshot.sha256
                ),
                flush=True,
            )
            _ensure_unchanged(master_path, master_sha256, phase)
            _ensure_unchanged(dataset, snapshot.sha256, phase)
            _ensure_python_unchanged(
                python_executable,
                python_executable_resolved,
                python_executable_sha256,
                phase,
            )
            _ensure_unchanged(motion_script, motion_script_sha256, phase)
            _ensure_unchanged(motion_driver, motion_driver_sha256, phase)
            _ensure_unchanged(link_preflight, link_preflight_sha256, phase)
            _ensure_unchanged(inspection_script, inspection_script_sha256, phase)
            _ensure_unchanged(
                frame_telemetry_script, frame_telemetry_sha256, phase
            )
            _ensure_unchanged(capture_config, capture_config_sha256, phase)
            _ensure_unchanged(capture_cli, capture_cli_sha256, phase)
            _ensure_unchanged(detector, detector_sha256, phase)
            _ensure_unchanged(stationary, stationary_sha256, phase)
            _ensure_unchanged(transforms, transforms_sha256, phase)

            edge_plan = _derive_edge_plan(
                master,
                master_path=master_path,
                master_sha256=master_sha256,
                start_pose_id=current_pose_id,
                target_pose_id=target_pose_id,
                poses=poses,
                orchestrator_sha256=orchestrator_sha256,
                python_executable=python_executable,
                python_executable_resolved=python_executable_resolved,
                python_executable_sha256=python_executable_sha256,
                capture_config=capture_config,
                capture_config_sha256=capture_config_sha256,
                capture_cli_sha256=capture_cli_sha256,
                stationary=stationary,
                stationary_sha256=stationary_sha256,
                transforms=transforms,
                transforms_sha256=transforms_sha256,
                motion_driver=motion_driver,
                motion_driver_sha256=motion_driver_sha256,
                link_preflight=link_preflight,
                link_preflight_sha256=link_preflight_sha256,
                inspection_script=inspection_script,
                inspection_script_sha256=inspection_script_sha256,
                frame_telemetry_script=frame_telemetry_script,
                frame_telemetry_sha256=frame_telemetry_sha256,
            )
            plan_bytes = yaml.safe_dump(
                dict(edge_plan), sort_keys=False, allow_unicode=True
            ).encode("utf-8")
            if len(plan_bytes) > MAX_PLAN_BYTES:
                raise SequenceFailure(phase, "derived edge plan is too large")
            plan_sha256 = _sha256_bytes(plan_bytes)
            stem = _edge_stem(current_pose_id, target_pose_id, plan_sha256)
            plan_path = edges_dir / (stem + ".plan.yaml")
            claim_path = claims_dir / (
                _claim_stem(current_pose_id, target_pose_id, master_sha256)
                + ".claim.json"
            )
            success_path = edges_dir / (stem + ".complete.json")
            failure_path = edges_dir / (stem + ".failed.json")
            _publish_exclusive(plan_path, plan_bytes)

            phase = "preview"
            preview_command = (
                str(python_executable),
                str(motion_script),
                "preview",
                "--plan",
                str(plan_path),
                "--pose-id",
                target_pose_id,
                "--expect-plan-sha256",
                plan_sha256,
                "--json",
            )
            preview_result = command_runner.run(
                preview_command,
                cwd=calibration_root,
                phase=edge_label + ":preview",
                env=subprocess_environment,
            )
            logs["preview"] = _write_log(
                edges_dir / (stem + ".preview.log"), preview_result
            )
            target_T_base_ee = _parse_preview(
                preview_result,
                plan_path=plan_path,
                plan_sha256=plan_sha256,
                start_pose_id=current_pose_id,
                target_pose_id=target_pose_id,
            )
            _ensure_unchanged(dataset, snapshot.sha256, phase)
            _ensure_unchanged(plan_path, plan_sha256, phase)
            _ensure_unchanged(motion_driver, motion_driver_sha256, phase)
            _ensure_unchanged(link_preflight, link_preflight_sha256, phase)
            _ensure_unchanged(capture_config, capture_config_sha256, phase)
            _ensure_unchanged(
                frame_telemetry_script, frame_telemetry_sha256, phase
            )

            phase = "claim"
            claim = {
                "schema_version": 1,
                "kind": "franka_eye_to_hand_training_edge_claim",
                "state": "claimed_before_motion",
                "claimed_at": clock(),
                "edge": edge_label,
                "start_pose_id": current_pose_id,
                "target_pose_id": target_pose_id,
                "reviewed_master_plan": str(master_path),
                "reviewed_master_plan_sha256": master_sha256,
                "calibration_root": str(calibration_root),
                "canonical_claims_dir": str(claims_dir),
                "single_edge_plan": str(plan_path),
                "single_edge_plan_sha256": plan_sha256,
                "training_sequence_artifacts_dir": str(artifacts_dir),
                "dataset": str(dataset),
                "dataset_pre_motion_sha256": snapshot.sha256,
                "dataset_pre_motion_sample_count": len(snapshot.samples),
                "expected_post_capture_sample_count": len(snapshot.samples) + 1,
                "training_orchestrator_sha256": orchestrator_sha256,
                "python_executable_path": str(python_executable),
                "python_executable_resolved_path": str(python_executable_resolved),
                "python_executable_sha256": python_executable_sha256,
                "motion_driver_sha256": motion_driver_sha256,
                "link_preflight_sha256": link_preflight_sha256,
                "inspection_script_sha256": inspection_script_sha256,
                "frame_telemetry_script": str(frame_telemetry_script),
                "frame_telemetry_sha256": frame_telemetry_sha256,
                "capture_config": str(capture_config),
                "capture_config_sha256": capture_config_sha256,
                "capture_cli_sha256": capture_cli_sha256,
                "detector_sha256": detector_sha256,
                "stationary_aggregator": str(stationary),
                "stationary_aggregator_sha256": stationary_sha256,
                "calibration_transforms": str(transforms),
                "calibration_transforms_sha256": transforms_sha256,
                "import_resolution": import_resolution,
                "import_probe_log": import_probe_log,
                "holdout_capture_permitted": False,
            }
            claim_bytes = _json_bytes(claim)
            claim_sha256 = _sha256_bytes(claim_bytes)
            _publish_exclusive(claim_path, claim_bytes)
            print(
                "[sequence] EDGE_CLAIMED {} plan_sha256={} claim={}".format(
                    edge_label, plan_sha256, claim_path
                ),
                flush=True,
            )

            # The claim must already be durably visible before this call.
            phase = "motion"
            _ensure_unchanged(master_path, master_sha256, phase)
            _ensure_unchanged(dataset, snapshot.sha256, phase)
            _ensure_python_unchanged(
                python_executable,
                python_executable_resolved,
                python_executable_sha256,
                phase,
            )
            _ensure_unchanged(plan_path, plan_sha256, phase)
            _ensure_unchanged(motion_script, motion_script_sha256, phase)
            _ensure_unchanged(motion_driver, motion_driver_sha256, phase)
            _ensure_unchanged(link_preflight, link_preflight_sha256, phase)
            _ensure_unchanged(capture_config, capture_config_sha256, phase)
            _ensure_unchanged(
                frame_telemetry_script, frame_telemetry_sha256, phase
            )
            _ensure_unchanged(claim_path, claim_sha256, phase)
            motion_command = (
                str(python_executable),
                str(motion_script),
                "run",
                "--plan",
                str(plan_path),
                "--pose-id",
                target_pose_id,
                "--robot-ip",
                str(expected_metadata["robot_ip"]),
                "--confirm-plan-sha256",
                plan_sha256,
                "--confirm-pose-id",
                target_pose_id,
                "--confirm-e-stop",
                E_STOP_TOKEN,
                "--confirm-swept-volume",
                SWEPT_VOLUME_TOKEN,
                "--confirm-target-rigid",
                TARGET_RIGID_TOKEN,
                "--confirm-camera-fixed",
                CAMERA_FIXED_TOKEN,
            )
            motion_result = command_runner.run(
                motion_command,
                cwd=calibration_root,
                phase=edge_label + ":motion",
                env=subprocess_environment,
            )
            logs["motion"] = _write_log(
                edges_dir / (stem + ".motion.log"), motion_result
            )
            motion_telemetry = _verify_motion_ready(
                motion_result,
                target_pose_id=target_pose_id,
                plan_sha256=plan_sha256,
            )
            _ensure_unchanged(dataset, snapshot.sha256, phase)

            phase = "inspect"
            _ensure_unchanged(master_path, master_sha256, phase)
            _ensure_unchanged(plan_path, plan_sha256, phase)
            _ensure_unchanged(claim_path, claim_sha256, phase)
            _ensure_unchanged(inspection_script, inspection_script_sha256, phase)
            _ensure_unchanged(capture_config, capture_config_sha256, phase)
            target = _require_mapping(expected_metadata.get("target"), "target")
            inspection_raw = edges_dir / (stem + ".inspection-raw.png")
            inspection_annotated = edges_dir / (stem + ".inspection-annotated.png")
            if inspection_raw.exists() or inspection_annotated.exists():
                raise SequenceFailure(phase, "inspection image path already exists")
            inspection_command = (
                str(python_executable),
                str(inspection_script),
                "--config",
                str(capture_config),
                "--camera-serial",
                str(expected_metadata["camera_serial"]),
                "--raw-output",
                str(inspection_raw),
                "--annotated-output",
                str(inspection_annotated),
                "--frames",
                str(INSPECTION_FRAMES),
                "--dictionary",
                str(target["dictionary"]),
            )
            inspection_result = command_runner.run(
                inspection_command,
                cwd=calibration_root,
                phase=edge_label + ":inspect",
                env=subprocess_environment,
            )
            logs["inspect"] = _write_log(
                edges_dir / (stem + ".inspect.log"), inspection_result
            )
            if inspection_result.returncode == 0:
                for inspection_image in (inspection_raw, inspection_annotated):
                    if inspection_image.exists():
                        os.chmod(str(inspection_image), 0o444)
            dataset_camera = _require_mapping(
                snapshot.document.get("camera"), "dataset.camera"
            )
            inspection_metrics = _parse_inspection(
                inspection_result,
                expected_serial=str(expected_metadata["camera_serial"]),
                target=target,
                dataset_camera=dataset_camera,
                raw_image=inspection_raw,
                annotated_image=inspection_annotated,
            )
            _ensure_unchanged(dataset, snapshot.sha256, phase)

            phase = "telemetry"
            _ensure_unchanged(master_path, master_sha256, phase)
            _ensure_unchanged(plan_path, plan_sha256, phase)
            _ensure_unchanged(capture_config, capture_config_sha256, phase)
            _ensure_unchanged(detector, detector_sha256, phase)
            _ensure_unchanged(
                frame_telemetry_script, frame_telemetry_sha256, phase
            )
            telemetry_path = edges_dir / (stem + ".frame-telemetry.json")
            if telemetry_path.exists():
                raise SequenceFailure(phase, "frame telemetry path already exists")
            telemetry_command = (
                str(python_executable),
                str(frame_telemetry_script),
                "--config",
                str(capture_config),
                "--camera-serial",
                str(expected_metadata["camera_serial"]),
                "--dictionary",
                str(target["dictionary"]),
                "--marker-id",
                str(target["marker_id"]),
                "--marker-length-m",
                "{:.12g}".format(float(target["marker_length_m"])),
                "--frames",
                str(TELEMETRY_FRAMES),
                "--output",
                str(telemetry_path),
                "--max-reprojection-error-px",
                "{:.12g}".format(MAX_CAPTURE_REPROJECTION_P95_PX),
                "--max-translation-deviation-m",
                "{:.12g}".format(MAX_CAPTURE_TARGET_TRANSLATION_JITTER_M),
                "--max-rotation-deviation-deg",
                "{:.12g}".format(MAX_CAPTURE_TARGET_ROTATION_JITTER_DEG),
            )
            telemetry_result = command_runner.run(
                telemetry_command,
                cwd=calibration_root,
                phase=edge_label + ":telemetry",
                env=subprocess_environment,
            )
            logs["telemetry"] = _write_log(
                edges_dir / (stem + ".telemetry.log"), telemetry_result
            )
            telemetry_metrics = _parse_frame_telemetry(
                telemetry_result,
                telemetry_path=telemetry_path,
                expected_serial=str(expected_metadata["camera_serial"]),
                target=target,
                dataset_camera=dataset_camera,
                capture_config=capture_config,
            )
            os.chmod(str(telemetry_path), 0o444)
            _ensure_unchanged(dataset, snapshot.sha256, phase)
            _ensure_unchanged(
                telemetry_path, str(telemetry_metrics["sha256"]), phase
            )

            phase = "capture"
            _ensure_unchanged(capture_cli, capture_cli_sha256, phase)
            _ensure_unchanged(detector, detector_sha256, phase)
            _ensure_unchanged(stationary, stationary_sha256, phase)
            _ensure_unchanged(transforms, transforms_sha256, phase)
            _ensure_unchanged(capture_config, capture_config_sha256, phase)
            _ensure_unchanged(
                frame_telemetry_script, frame_telemetry_sha256, phase
            )
            _ensure_unchanged(
                telemetry_path, str(telemetry_metrics["sha256"]), phase
            )
            debug_image = edges_dir / (stem + ".capture.png")
            if debug_image.exists():
                raise SequenceFailure(phase, "debug image path already exists")
            expected_count = len(snapshot.samples) + 1
            capture_command = (
                str(python_executable),
                "-m",
                "dynamic_pcd.apps.calibrate_eye_to_hand",
                "capture",
                "--config",
                str(capture_config),
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
                str(CAPTURE_FRAMES),
                "--min-valid-frame-fraction",
                "0.95",
                "--min-valid-frames",
                str(MIN_CAPTURE_COHERENT_FRAMES),
                "--max-reprojection-error",
                "{:.12g}".format(MAX_CAPTURE_REPROJECTION_P95_PX),
                "--max-target-translation-jitter",
                "{:.12g}".format(MAX_CAPTURE_TARGET_TRANSLATION_JITTER_M),
                "--max-target-rotation-jitter",
                "{:.12g}".format(MAX_CAPTURE_TARGET_ROTATION_JITTER_DEG),
                "--min-all-reprojection-pass-frames",
                str(MIN_CAPTURE_ALL_REPROJECTION_PASS_FRAMES),
                "--max-all-reprojection-pass-translation-jitter",
                "{:.12g}".format(MAX_CAPTURE_TARGET_TRANSLATION_JITTER_M),
                "--max-all-reprojection-pass-rotation-jitter",
                "{:.12g}".format(MAX_CAPTURE_TARGET_ROTATION_JITTER_DEG),
                "--max-all-reprojection-pass-reprojection-error",
                "{:.12g}".format(MAX_CAPTURE_REPROJECTION_P95_PX),
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
                str(dataset),
            )
            # This is the last user-space work before the collector subprocess.
            # Recheck every reviewed input, the append-only dataset, and the
            # single-edge plan here so telemetry/log processing cannot create a
            # stale-code or stale-state window before capture.
            _ensure_unchanged(master_path, master_sha256, phase)
            _ensure_unchanged(dataset, snapshot.sha256, phase)
            _ensure_unchanged(plan_path, plan_sha256, phase)
            _ensure_unchanged(claim_path, claim_sha256, phase)
            _ensure_python_unchanged(
                python_executable,
                python_executable_resolved,
                python_executable_sha256,
                phase,
            )
            _ensure_unchanged(orchestrator_path, orchestrator_sha256, phase)
            _ensure_unchanged(motion_script, motion_script_sha256, phase)
            _ensure_unchanged(motion_driver, motion_driver_sha256, phase)
            _ensure_unchanged(link_preflight, link_preflight_sha256, phase)
            _ensure_unchanged(inspection_script, inspection_script_sha256, phase)
            _ensure_unchanged(
                frame_telemetry_script, frame_telemetry_sha256, phase
            )
            _ensure_unchanged(capture_config, capture_config_sha256, phase)
            _ensure_unchanged(capture_cli, capture_cli_sha256, phase)
            _ensure_unchanged(detector, detector_sha256, phase)
            _ensure_unchanged(stationary, stationary_sha256, phase)
            _ensure_unchanged(transforms, transforms_sha256, phase)
            _ensure_unchanged(
                telemetry_path, str(telemetry_metrics["sha256"]), phase
            )
            capture_adjacent_integrity_recheck = {
                "master_plan_sha256": master_sha256,
                "dataset_sha256": snapshot.sha256,
                "single_edge_plan_sha256": plan_sha256,
                "claim_sha256": claim_sha256,
                "orchestrator_sha256": orchestrator_sha256,
                "python_executable_path": str(python_executable),
                "python_executable_resolved_path": str(python_executable_resolved),
                "python_executable_sha256": python_executable_sha256,
                "motion_wrapper_sha256": motion_script_sha256,
                "motion_driver_sha256": motion_driver_sha256,
                "link_preflight_sha256": link_preflight_sha256,
                "inspection_script_sha256": inspection_script_sha256,
                "frame_telemetry_sha256": frame_telemetry_sha256,
                "capture_config_sha256": capture_config_sha256,
                "capture_cli_sha256": capture_cli_sha256,
                "aruco_detector_sha256": detector_sha256,
                "stationary_aggregator_sha256": stationary_sha256,
                "calibration_transforms_sha256": transforms_sha256,
                "frame_telemetry_artifact_sha256": str(
                    telemetry_metrics["sha256"]
                ),
            }
            capture_result = command_runner.run(
                capture_command,
                cwd=calibration_root,
                phase=edge_label + ":capture",
                env=subprocess_environment,
            )
            logs["capture"] = _write_log(
                edges_dir / (stem + ".capture.log"), capture_result
            )
            capture_metrics = _parse_capture_output(
                capture_result, expected_count=expected_count
            )
            if debug_image.exists():
                os.chmod(str(debug_image), 0o444)
            debug = _decode_image(debug_image, "capture debug image", phase="capture")
            after = _dataset_snapshot(dataset)
            try:
                _verify_dataset_increment(
                    snapshot,
                    after,
                    expected_metadata=expected_metadata,
                    expected_count=expected_count,
                    target_T_base_ee=target_T_base_ee,
                )
            except ValueError as exc:
                raise SequenceFailure(phase, str(exc)) from exc

            phase = "receipt"
            _ensure_unchanged(master_path, master_sha256, phase)
            _ensure_unchanged(dataset, after.sha256, phase)
            _ensure_unchanged(plan_path, plan_sha256, phase)
            _ensure_unchanged(claim_path, claim_sha256, phase)
            _ensure_python_unchanged(
                python_executable,
                python_executable_resolved,
                python_executable_sha256,
                phase,
            )
            _ensure_unchanged(orchestrator_path, orchestrator_sha256, phase)
            _ensure_unchanged(motion_script, motion_script_sha256, phase)
            _ensure_unchanged(motion_driver, motion_driver_sha256, phase)
            _ensure_unchanged(link_preflight, link_preflight_sha256, phase)
            _ensure_unchanged(inspection_script, inspection_script_sha256, phase)
            _ensure_unchanged(
                telemetry_path, str(telemetry_metrics["sha256"]), phase
            )
            _ensure_unchanged(
                frame_telemetry_script, frame_telemetry_sha256, phase
            )
            _ensure_unchanged(capture_config, capture_config_sha256, phase)
            _ensure_unchanged(capture_cli, capture_cli_sha256, phase)
            _ensure_unchanged(detector, detector_sha256, phase)
            _ensure_unchanged(stationary, stationary_sha256, phase)
            _ensure_unchanged(transforms, transforms_sha256, phase)
            _ensure_unchanged(
                Path(str(import_probe_log["path"])),
                str(import_probe_log["sha256"]),
                phase,
            )
            for log_record in logs.values():
                _ensure_unchanged(
                    Path(str(log_record["path"])),
                    str(log_record["sha256"]),
                    phase,
                )
            for image_record in (
                inspection_metrics["raw_image"],
                inspection_metrics["annotated_image"],
            ):
                _ensure_unchanged(
                    Path(str(image_record["path"])),
                    str(image_record["sha256"]),
                    phase,
                )
            debug_sha256 = _sha256_file(debug_image)
            completion = {
                "schema_version": 1,
                "kind": "franka_eye_to_hand_training_edge_receipt",
                "state": "capture_verified_complete",
                "completed_at": clock(),
                "edge": edge_label,
                "start_pose_id": current_pose_id,
                "target_pose_id": target_pose_id,
                "reviewed_master_plan_sha256": master_sha256,
                "calibration_root": str(calibration_root),
                "canonical_claims_dir": str(claims_dir),
                "single_edge_plan": str(plan_path),
                "single_edge_plan_sha256": plan_sha256,
                "claim": str(claim_path),
                "claim_sha256": claim_sha256,
                "training_sequence_artifacts_dir": str(artifacts_dir),
                "dataset": str(dataset),
                "dataset_before": {
                    "sha256": snapshot.sha256,
                    "sample_count": len(snapshot.samples),
                },
                "dataset_after": {
                    "sha256": after.sha256,
                    "sample_count": len(after.samples),
                },
                "capture_metrics": capture_metrics,
                "live_endpoint_inspection": inspection_metrics,
                "independent_frame_telemetry": telemetry_metrics,
                "frame_telemetry_executable": {
                    "path": str(frame_telemetry_script),
                    "sha256": frame_telemetry_sha256,
                },
                "formal_capture_executables": {
                    "capture_cli": {
                        "path": str(capture_cli),
                        "sha256": capture_cli_sha256,
                    },
                    "aruco_detector": {
                        "path": str(detector),
                        "sha256": detector_sha256,
                    },
                    "stationary_aggregator": {
                        "path": str(stationary),
                        "sha256": stationary_sha256,
                    },
                    "calibration_transforms": {
                        "path": str(transforms),
                        "sha256": transforms_sha256,
                    },
                },
                "import_resolution": import_resolution,
                "import_probe_log": import_probe_log,
                "capture_adjacent_integrity_recheck": (
                    capture_adjacent_integrity_recheck
                ),
                "motion_control_loop_telemetry": motion_telemetry,
                "debug_image": {
                    "path": str(debug_image),
                    "sha256": debug_sha256,
                    "width": int(debug.shape[1]),
                    "height": int(debug.shape[0]),
                },
                "logs": logs,
                "holdout_capture_performed": False,
            }
            _publish_exclusive(success_path, _json_bytes(completion))
            completion_records.append(completion)
            completed.append(target_pose_id)
            snapshot = after
            current_pose_id = target_pose_id
            print(
                "[sequence] EDGE_COMPLETE {} sample={} dataset_sha256={}".format(
                    edge_label, len(snapshot.samples), snapshot.sha256
                ),
                flush=True,
            )
        except KeyboardInterrupt:
            failure = SequenceFailure(phase, "operator interrupt")
            if plan_path is not None and plan_sha256 is not None:
                failure_payload = {
                    "schema_version": 1,
                    "kind": "franka_eye_to_hand_training_edge_failure",
                    "state": "failed_closed_no_automatic_retry",
                    "failed_at": clock(),
                    "phase": phase,
                    "error": str(failure),
                    "edge": edge_label,
                    "single_edge_plan": str(plan_path),
                    "single_edge_plan_sha256": plan_sha256,
                    "claim": None if claim_path is None else str(claim_path),
                    "logs": logs,
                    **_safe_failure_state(dataset),
                }
                try:
                    _publish_exclusive(failure_path, _json_bytes(failure_payload))
                except (OSError, ValueError) as receipt_error:
                    print(
                        "[sequence] failure receipt error: {}".format(receipt_error),
                        file=sys.stderr,
                    )
            raise failure
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            failure = (
                exc
                if isinstance(exc, SequenceFailure)
                else SequenceFailure(phase, str(exc))
            )
            if plan_path is not None and plan_sha256 is not None:
                failure_payload = {
                    "schema_version": 1,
                    "kind": "franka_eye_to_hand_training_edge_failure",
                    "state": "failed_closed_no_automatic_retry",
                    "failed_at": clock(),
                    "phase": failure.phase,
                    "error": str(failure),
                    "edge": edge_label,
                    "single_edge_plan": str(plan_path),
                    "single_edge_plan_sha256": plan_sha256,
                    "claim": None if claim_path is None else str(claim_path),
                    "logs": logs,
                    **_safe_failure_state(dataset),
                }
                try:
                    _publish_exclusive(failure_path, _json_bytes(failure_payload))
                except (OSError, ValueError) as receipt_error:
                    print(
                        "[sequence] failure receipt error: {}".format(receipt_error),
                        file=sys.stderr,
                    )
            print(
                "[sequence] STOP edge={} phase={} error={}".format(
                    edge_label, failure.phase, failure
                ),
                file=sys.stderr,
                flush=True,
            )
            raise failure

    if len(snapshot.samples) != 20 or tuple(completed) != remaining:
        raise SequenceFailure(
            "completion", "internal final-count/range invariant failed"
        )
    completion_path = artifacts_dir / (
        "training-sequence-{}-complete.json".format(master_sha256[:16])
    )
    sequence_receipt = {
        "schema_version": 1,
        "kind": "franka_eye_to_hand_training_sequence_receipt",
        "state": "reviewed_remaining_training_suffix_capture_verified_complete",
        "completed_at": clock(),
        "reviewed_master_plan": str(master_path),
        "reviewed_master_plan_sha256": master_sha256,
        "calibration_root": str(calibration_root),
        "canonical_claims_dir": str(claims_dir),
        "dataset": str(dataset),
        "training_sequence_artifacts_dir": str(artifacts_dir),
        "initial_start_pose_id": initial_start_pose_id,
        "initial_sample_count": initial_sample_count,
        "final_sample_count": len(snapshot.samples),
        "final_dataset_sha256": snapshot.sha256,
        "frame_telemetry_executable": {
            "path": str(frame_telemetry_script),
            "sha256": frame_telemetry_sha256,
        },
        "formal_capture_executables": {
            "capture_cli": {
                "path": str(capture_cli),
                "sha256": capture_cli_sha256,
            },
            "aruco_detector": {
                "path": str(detector),
                "sha256": detector_sha256,
            },
            "stationary_aggregator": {
                "path": str(stationary),
                "sha256": stationary_sha256,
            },
            "calibration_transforms": {
                "path": str(transforms),
                "sha256": transforms_sha256,
            },
        },
        "import_resolution": import_resolution,
        "import_probe_log": import_probe_log,
        "completed_pose_ids": completed,
        "edge_receipts": [
            str(
                edges_dir
                / (
                    _edge_stem(
                        str(record["start_pose_id"]),
                        str(record["target_pose_id"]),
                        str(record["single_edge_plan_sha256"]),
                    )
                    + ".complete.json"
                )
            )
            for record in completion_records
        ],
        "holdout_capture_performed": False,
    }
    _publish_exclusive(completion_path, _json_bytes(sequence_receipt))
    print(
        "[sequence] COMPLETE samples=20 dataset_sha256={} receipt={}".format(
            snapshot.sha256, completion_path
        ),
        flush=True,
    )
    return SequenceResult(tuple(completed), 20, snapshot.sha256, completion_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed remaining-training orchestrator.  Every edge receives "
            "an immutable SHA-bound plan and one-use claim before motion."
        )
    )
    parser.add_argument("--master-plan", type=Path, required=True)
    parser.add_argument("--expect-master-sha256", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=Path("/home/qiaoguanren/code/franka/beta/dynamic_object_pcd"),
    )
    parser.add_argument("--capture-config", type=Path)
    parser.add_argument(
        "--motion-script",
        type=Path,
        default=Path(__file__).with_name("move_calibration_pose.py"),
    )
    parser.add_argument("--motion-driver", type=Path)
    parser.add_argument("--link-preflight", type=Path)
    parser.add_argument(
        "--inspection-script",
        type=Path,
        default=Path(__file__).with_name("inspect_aruco_frame.py"),
    )
    parser.add_argument(
        "--frame-telemetry-script",
        type=Path,
        default=Path(__file__).with_name("capture_aruco_frame_telemetry.py"),
    )
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    capture_config = (
        args.capture_config
        if args.capture_config is not None
        else args.calibration_root / "configs" / "d435_default.yaml"
    )
    calibration_root = args.calibration_root.expanduser().resolve()
    try:
        workspace_root = calibration_root.parents[1]
    except IndexError:
        print("[sequence] invalid calibration root", file=sys.stderr)
        return 1
    motion_driver = (
        args.motion_driver
        if args.motion_driver is not None
        else workspace_root
        / "dexgrasp"
        / "src"
        / "anydex_pipeline"
        / "franka_sequence_driver.py"
    )
    link_preflight = (
        args.link_preflight
        if args.link_preflight is not None
        else workspace_root
        / "dexgrasp"
        / "src"
        / "anydex_pipeline"
        / "host_network_preflight.py"
    )
    config = OrchestratorConfig(
        master_plan=args.master_plan,
        expected_master_sha256=str(args.expect_master_sha256),
        dataset=args.dataset,
        artifacts_dir=args.artifacts_dir,
        calibration_root=args.calibration_root,
        capture_config=capture_config,
        motion_script=args.motion_script,
        motion_driver=motion_driver,
        link_preflight=link_preflight,
        inspection_script=args.inspection_script,
        frame_telemetry_script=args.frame_telemetry_script,
        python_executable=str(args.python_executable),
    )
    try:
        run_sequence(config)
    except KeyboardInterrupt:
        print("[sequence] interrupted; automatic retry is forbidden", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        phase = getattr(exc, "phase", "preflight")
        print(
            "[sequence] FAILED_CLOSED phase={} error={}".format(phase, exc),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
