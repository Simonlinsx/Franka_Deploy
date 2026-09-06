"""Offline session manifest shared by a telemetry producer and viewer.

The manifest binds one continuous-telemetry run to immutable perception and
execution inputs.  It is deliberately *not* a motion authorization artifact:
``motion_authorized`` is always false, and loading it never imports or opens a
robot, serial port, camera, GUI, or native telemetry transport.

The execution-specific geometry is derived through the existing
``derive_air_target_poses`` source of truth.  This module does not duplicate
the executor's collision-audit pass/fail policy.  It instead binds an exact,
self-sealed audit file by both byte hash and canonical JSON content hash; the
executor remains responsible for replaying that artifact with its existing
strict audit API before motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union
import uuid

import numpy as np

from .air_target_poses import derive_air_target_poses
from .continuous_telemetry import TelemetryIdentity
from .control_config import load_control_config
from .control_plan import validate_rigid_transform
from .snapshot import VisualizationSnapshot, load_snapshot_npz


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "fr3_rh56_continuous_telemetry_session_manifest"
INTEGRITY_ALGORITHM = "sha256-canonical-json-without-integrity"
CALIBRATION_FINGERPRINT_ALGORITHM = (
    "sha256-canonical-snapshot-calibration-v1"
)
AIR_TARGET_CONTRACT_KIND = "fr3_rh56_installed_air_target_poses_v1"
EXECUTION_CONTRACT_KIND = "fr3_rh56_telemetry_execution_contract_v1"

EXECUTION_COMMANDS: Tuple[str, ...] = (
    "pregrasp",
    "grasp",
    "air-grasp",
    "grasp-lift",
)

_PREGRASP_AUDIT = "fr3_rh56_pregrasp_only_collision_audit"
_INSTALLED_AUDIT = "fr3_rh56_installed_tool_collision_audit"
_LOADED_LIFT_AUDIT = "fr3_rh56_loaded_lift_collision_audit"
_EXPECTED_AUDIT: Mapping[str, Tuple[str, str, int]] = {
    "pregrasp": (
        _PREGRASP_AUDIT,
        "open_hand_current_to_pregrasp_only",
        2,
    ),
    "air-grasp": (_INSTALLED_AUDIT, "air_grasp", 2),
    "grasp": (_INSTALLED_AUDIT, "loaded_grasp", 2),
    "grasp-lift": (_LOADED_LIFT_AUDIT, "loaded_lift_round_trip", 1),
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
PathLike = Union[str, Path]


@dataclass(frozen=True)
class LoadedTelemetrySessionManifest:
    """One fully validated manifest plus its immutable telemetry identity."""

    path: Optional[Path]
    payload: Mapping[str, Any]
    identity: TelemetryIdentity
    manifest_file_sha256: Optional[str]


def sha256_file(path: PathLike) -> str:
    """Return the SHA-256 of one ordinary file without interpreting it."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError("bound file does not exist: {}".format(source))
    digest = hashlib.sha256()
    try:
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError("cannot hash bound file {}: {}".format(source, exc)) from exc
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash the repository-wide finite, sorted, compact JSON representation."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not finite canonical JSON: {}".format(exc)) from exc
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: Any, *, shape: Tuple[int, ...], name: str) -> str:
    try:
        array = np.asarray(value, dtype="<f8")
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be a finite {} array".format(name, shape)) from exc
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError("{} must be a finite {} array".format(name, shape))
    array = np.ascontiguousarray(array, dtype="<f8")
    dimensions = ",".join(str(item) for item in shape)
    digest = hashlib.sha256()
    digest.update("dtype=<f8;shape={};".format(dimensions).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def calibration_fingerprint(snapshot: VisualizationSnapshot) -> Dict[str, Any]:
    """Build the canonical snapshot-only eye-to-hand calibration fingerprint."""

    calibration_id = str(snapshot.calibration_id)
    camera_serial = str(snapshot.camera_serial)
    if not calibration_id.strip():
        raise ValueError("snapshot calibration_id must be non-empty")
    if not camera_serial.strip():
        raise ValueError("snapshot camera_serial must be non-empty")
    transform = validate_rigid_transform(
        np.asarray(snapshot.T_reference_camera, dtype=np.float64),
        "snapshot T_reference_camera",
    )
    unsigned: Dict[str, Any] = {
        "algorithm": CALIBRATION_FINGERPRINT_ALGORITHM,
        "calibration_id": calibration_id,
        "camera_serial": camera_serial,
        "T_reference_camera": transform.tolist(),
        "T_reference_camera_sha256": _array_sha256(
            transform,
            shape=(4, 4),
            name="snapshot T_reference_camera",
        ),
    }
    return dict(unsigned, sha256=canonical_json_sha256(unsigned))


def _air_target_contract(
    snapshot: VisualizationSnapshot,
    config: Mapping[str, Any],
    selected_index: int,
) -> Dict[str, Any]:
    hand_poses = snapshot.grasps.hand_poses
    if hand_poses is None:
        raise ValueError("selected snapshot has no official hand poses")
    tool = _object(config.get("tool"), "control config tool")
    grasp = _object(config.get("grasp"), "control config grasp")
    if tool.get("T_EE_hand") is None:
        raise ValueError("control config tool.T_EE_hand is missing")
    targets = derive_air_target_poses(
        np.asarray(
            snapshot.grasps.canonical_poses[selected_index], dtype=np.float64
        ),
        np.asarray(hand_poses[selected_index], dtype=np.float64),
        np.asarray(snapshot.grasps.approach_axis_local, dtype=np.float64),
        np.asarray(tool["T_EE_hand"], dtype=np.float64),
        retreat_distance_m=float(grasp["air_retreat_distance_m"]),
        pregrasp_extra_distance_m=float(grasp["air_pregrasp_distance_m"]),
    )
    arrays = {
        "approach_reference": np.asarray(
            targets.approach_reference, dtype=np.float64
        ),
        "T_reference_EE_nominal": np.asarray(
            targets.T_reference_EE_nominal, dtype=np.float64
        ),
        "T_reference_EE_pregrasp": np.asarray(
            targets.T_reference_EE_pregrasp, dtype=np.float64
        ),
        "T_reference_EE_final_air": np.asarray(
            targets.T_reference_EE_final_air, dtype=np.float64
        ),
        "T_reference_hand_final_air": np.asarray(
            targets.T_reference_hand_final_air, dtype=np.float64
        ),
    }
    array_hashes = {
        name: _array_sha256(
            value,
            shape=(3,) if name == "approach_reference" else (4, 4),
            name="air target {}".format(name),
        )
        for name, value in arrays.items()
    }
    unsigned: Dict[str, Any] = {
        "kind": AIR_TARGET_CONTRACT_KIND,
        "reference_frame": str(snapshot.reference_frame),
        "retreat_distance_m": float(grasp["air_retreat_distance_m"]),
        "pregrasp_extra_distance_m": float(
            grasp["air_pregrasp_distance_m"]
        ),
        "approach_reference": arrays["approach_reference"].tolist(),
        "T_reference_EE_nominal": arrays["T_reference_EE_nominal"].tolist(),
        "T_reference_EE_pregrasp": arrays[
            "T_reference_EE_pregrasp"
        ].tolist(),
        "T_reference_EE_final_air": arrays[
            "T_reference_EE_final_air"
        ].tolist(),
        "T_reference_hand_final_air": arrays[
            "T_reference_hand_final_air"
        ].tolist(),
        "array_sha256": array_hashes,
    }
    return dict(unsigned, sha256=canonical_json_sha256(unsigned))


def _strict_json_file(path: PathLike, name: str) -> Tuple[Mapping[str, Any], Path]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError("{} does not exist: {}".format(name, source))
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("cannot load {} {}: {}".format(name, source, exc)) from exc
    if not isinstance(payload, dict):
        raise ValueError("{} must contain a JSON object".format(name))
    return payload, source


def _audit_binding(path: PathLike, command: str) -> Dict[str, Any]:
    payload, source = _strict_json_file(path, "audit artifact")
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version <= 0:
        raise ValueError("audit artifact schema_version must be a positive integer")
    artifact_type = payload.get("artifact_type")
    if not isinstance(artifact_type, str) or not artifact_type.strip():
        raise ValueError("audit artifact_type must be a non-empty string")
    expected_type, expected_scope, expected_schema = _EXPECTED_AUDIT[command]
    if artifact_type != expected_type:
        raise ValueError(
            "{} requires audit artifact_type {!r}, got {!r}".format(
                command, expected_type, artifact_type
            )
        )
    if schema_version != expected_schema:
        raise ValueError(
            "{} requires audit schema_version {}, got {}".format(
                command, expected_schema, schema_version
            )
        )
    if artifact_type == _PREGRASP_AUDIT:
        motion_authorized = payload.get("motion_authorized")
        actual_scope = payload.get("execution_scope")
    else:
        decision = payload.get("decision")
        motion_authorized = (
            decision.get("motion_authorized")
            if isinstance(decision, Mapping)
            else None
        )
        actual_scope = payload.get("mode")
    if motion_authorized is not False:
        raise ValueError(
            "audit artifact must explicitly keep motion_authorized=false"
        )
    if actual_scope != expected_scope:
        raise ValueError(
            "{} requires audit execution scope {!r}".format(
                command, expected_scope
            )
        )

    # Existing audit formats use one of these two repository-wide seals.  We
    # verify the seal here only to ensure the exact opaque artifact is sound;
    # the executor's existing validator owns all geometric/pass semantics.
    if "artifact_sha256" in payload and "integrity" not in payload:
        supplied = _sha256(payload["artifact_sha256"], "audit artifact_sha256")
        unsigned = dict(payload)
        del unsigned["artifact_sha256"]
        if canonical_json_sha256(unsigned) != supplied:
            raise ValueError("audit artifact_sha256 does not match its content")
        seal_algorithm = "sha256-canonical-json-without-artifact_sha256"
        seal_sha256 = supplied
    elif "integrity" in payload and "artifact_sha256" not in payload:
        integrity = _expect_exact_keys(
            payload["integrity"],
            ("algorithm", "payload_sha256"),
            "audit integrity",
        )
        if integrity["algorithm"] != INTEGRITY_ALGORITHM:
            raise ValueError("audit integrity algorithm is unsupported")
        supplied = _sha256(
            integrity["payload_sha256"], "audit integrity payload_sha256"
        )
        unsigned = dict(payload)
        del unsigned["integrity"]
        if canonical_json_sha256(unsigned) != supplied:
            raise ValueError("audit integrity payload_sha256 does not match content")
        seal_algorithm = INTEGRITY_ALGORITHM
        seal_sha256 = supplied
    else:
        raise ValueError("audit artifact must contain exactly one supported content seal")

    return {
        "path": str(source),
        "file_sha256": sha256_file(source),
        "canonical_json_sha256": canonical_json_sha256(payload),
        "artifact_type": artifact_type,
        "schema_version": schema_version,
        "execution_scope": actual_scope,
        "self_seal_algorithm": seal_algorithm,
        "self_seal_sha256": seal_sha256,
    }


def build_telemetry_session_manifest(
    *,
    snapshot_path: PathLike,
    control_config_path: PathLike,
    audit_artifact_path: PathLike,
    producer_build_path: PathLike,
    command: str,
    selected_index: int,
    run_uuid: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one deterministic, hardware-free manifest mapping.

    If ``run_uuid`` is omitted a UUID4 is generated.  Supplying it is useful
    for deterministic replay tests; supplied UUIDs must already use canonical
    lowercase, hyphenated spelling.
    """

    command = _execution_command(command)
    selected_index = _selected_index(selected_index)
    canonical_run_uuid = _run_uuid(run_uuid)

    snapshot_source = Path(snapshot_path).expanduser().resolve()
    config, config_source = load_control_config(control_config_path)
    snapshot = load_snapshot_npz(snapshot_source)
    if selected_index >= snapshot.grasps.count:
        raise ValueError(
            "selected_index={} is outside snapshot candidates 0..{}".format(
                selected_index, snapshot.grasps.count - 1
            )
        )
    if snapshot.reference_frame != config["reference_frame"]:
        raise ValueError("snapshot/control reference_frame mismatch")
    calibration_config = _object(config["calibration"], "control calibration")
    if str(snapshot.calibration_id) != str(calibration_config["id"]):
        raise ValueError("snapshot/control calibration id mismatch")
    if str(snapshot.camera_serial) != str(calibration_config["camera_serial"]):
        raise ValueError("snapshot/control camera serial mismatch")

    producer_source = Path(producer_build_path).expanduser().resolve()
    audit = _audit_binding(audit_artifact_path, command)
    sources: Dict[str, Any] = {
        "snapshot": {
            "path": str(snapshot_source),
            "file_sha256": sha256_file(snapshot_source),
        },
        "control_config": {
            "path": str(config_source),
            "file_sha256": sha256_file(config_source),
            "canonical_json_sha256": canonical_json_sha256(config),
        },
        "audit_artifact": audit,
        "producer_build": {
            "path": str(producer_source),
            "file_sha256": sha256_file(producer_source),
        },
    }
    calibration = calibration_fingerprint(snapshot)
    air_contract = _air_target_contract(snapshot, config, selected_index)
    execution: Dict[str, Any] = {
        "kind": EXECUTION_CONTRACT_KIND,
        "command": command,
        "selected_index": selected_index,
        "audit_artifact_file_sha256": audit["file_sha256"],
        "audit_artifact_content_sha256": audit["canonical_json_sha256"],
        "air_target_contract": air_contract,
    }
    execution_sha256 = canonical_json_sha256(execution)
    identity_mapping = {
        "run_uuid": canonical_run_uuid,
        "execution_contract_sha256": execution_sha256,
        "source_snapshot_sha256": sources["snapshot"]["file_sha256"],
        "control_config_sha256": sources["control_config"]["file_sha256"],
        "calibration_sha256": calibration["sha256"],
        "producer_build_sha256": sources["producer_build"]["file_sha256"],
    }
    # Reuse the exact continuous-reader identity validator rather than
    # maintaining a second UUID/SHA spelling contract here.
    TelemetryIdentity(**identity_mapping)

    unsigned: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "motion_authorized": False,
        "identity": identity_mapping,
        "execution": execution,
        "sources": sources,
        "calibration": calibration,
    }
    payload = dict(
        unsigned,
        integrity={
            "algorithm": INTEGRITY_ALGORITHM,
            "payload_sha256": canonical_json_sha256(unsigned),
        },
    )
    validate_telemetry_session_manifest(payload, verify_files=False)
    return payload


def validate_telemetry_session_manifest(
    payload: Mapping[str, Any],
    *,
    verify_files: bool = False,
) -> LoadedTelemetrySessionManifest:
    """Strictly validate a manifest and optionally replay every bound file."""

    root = _expect_exact_keys(
        payload,
        (
            "schema_version",
            "artifact_type",
            "motion_authorized",
            "identity",
            "execution",
            "sources",
            "calibration",
            "integrity",
        ),
        "telemetry session manifest",
    )
    if type(root["schema_version"]) is not int or root["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported telemetry session manifest schema_version")
    if root["artifact_type"] != ARTIFACT_TYPE:
        raise ValueError("telemetry session manifest artifact_type is invalid")
    if root["motion_authorized"] is not False:
        raise ValueError("telemetry session manifest must not authorize motion")

    identity_mapping = _expect_exact_keys(
        root["identity"],
        (
            "run_uuid",
            "execution_contract_sha256",
            "source_snapshot_sha256",
            "control_config_sha256",
            "calibration_sha256",
            "producer_build_sha256",
        ),
        "telemetry identity",
    )
    identity = TelemetryIdentity(**identity_mapping)

    sources = _expect_exact_keys(
        root["sources"],
        ("snapshot", "control_config", "audit_artifact", "producer_build"),
        "manifest sources",
    )
    snapshot_binding = _file_binding(
        sources["snapshot"], "snapshot source", extra=()
    )
    config_binding = _file_binding(
        sources["control_config"],
        "control config source",
        extra=("canonical_json_sha256",),
    )
    _sha256(
        config_binding["canonical_json_sha256"],
        "control config canonical_json_sha256",
    )
    producer_binding = _file_binding(
        sources["producer_build"], "producer build source", extra=()
    )
    audit_binding = _file_binding(
        sources["audit_artifact"],
        "audit artifact source",
        extra=(
            "canonical_json_sha256",
            "artifact_type",
            "schema_version",
            "execution_scope",
            "self_seal_algorithm",
            "self_seal_sha256",
        ),
    )
    _sha256(audit_binding["canonical_json_sha256"], "audit canonical_json_sha256")
    _sha256(audit_binding["self_seal_sha256"], "audit self_seal_sha256")
    if not isinstance(audit_binding["artifact_type"], str) or not audit_binding[
        "artifact_type"
    ].strip():
        raise ValueError("audit binding artifact_type must be non-empty")
    if type(audit_binding["schema_version"]) is not int or audit_binding[
        "schema_version"
    ] <= 0:
        raise ValueError("audit binding schema_version must be positive")
    if (
        not isinstance(audit_binding["execution_scope"], str)
        or not audit_binding["execution_scope"].strip()
    ):
        raise ValueError("audit binding execution_scope must be non-empty")
    if audit_binding["self_seal_algorithm"] not in (
        "sha256-canonical-json-without-artifact_sha256",
        INTEGRITY_ALGORITHM,
    ):
        raise ValueError("audit binding self_seal_algorithm is unsupported")

    calibration = _validate_calibration(root["calibration"])
    execution = _validate_execution(root["execution"], audit_binding)
    command = str(execution["command"])
    expected_type, expected_scope, expected_schema = _EXPECTED_AUDIT[command]
    if audit_binding["artifact_type"] != expected_type:
        raise ValueError("execution command and audit artifact_type differ")
    if audit_binding["execution_scope"] != expected_scope:
        raise ValueError("execution command and audit execution_scope differ")
    if audit_binding["schema_version"] != expected_schema:
        raise ValueError("execution command and audit schema_version differ")

    if identity.execution_contract_sha256 != canonical_json_sha256(execution):
        raise ValueError("execution_contract_sha256 does not match execution content")
    if identity.source_snapshot_sha256 != snapshot_binding["file_sha256"]:
        raise ValueError("source_snapshot_sha256 differs from snapshot file binding")
    if identity.control_config_sha256 != config_binding["file_sha256"]:
        raise ValueError("control_config_sha256 differs from config file binding")
    if identity.calibration_sha256 != calibration["sha256"]:
        raise ValueError("calibration_sha256 differs from calibration fingerprint")
    if identity.producer_build_sha256 != producer_binding["file_sha256"]:
        raise ValueError("producer_build_sha256 differs from producer file binding")

    integrity = _expect_exact_keys(
        root["integrity"],
        ("algorithm", "payload_sha256"),
        "manifest integrity",
    )
    if integrity["algorithm"] != INTEGRITY_ALGORITHM:
        raise ValueError("manifest integrity algorithm is unsupported")
    supplied_integrity = _sha256(
        integrity["payload_sha256"], "manifest integrity payload_sha256"
    )
    unsigned = dict(root)
    del unsigned["integrity"]
    if canonical_json_sha256(unsigned) != supplied_integrity:
        raise ValueError("manifest integrity payload_sha256 does not match content")

    if verify_files:
        rebuilt = build_telemetry_session_manifest(
            snapshot_path=snapshot_binding["path"],
            control_config_path=config_binding["path"],
            audit_artifact_path=audit_binding["path"],
            producer_build_path=producer_binding["path"],
            command=command,
            selected_index=execution["selected_index"],
            run_uuid=identity.run_uuid,
        )
        if canonical_json_sha256(rebuilt) != canonical_json_sha256(root):
            raise ValueError(
                "manifest replay differs from its currently bound file contents"
            )
    return LoadedTelemetrySessionManifest(
        path=None,
        payload=root,
        identity=identity,
        manifest_file_sha256=None,
    )


def write_telemetry_session_manifest(
    path: PathLike, payload: Mapping[str, Any]
) -> Path:
    """Atomically write one validated finite manifest in its destination dir."""

    validate_telemetry_session_manifest(payload, verify_files=False)
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(output.parent),
            prefix=".{}.".format(output.name),
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, str(output))
        temporary_name = None
        try:
            directory_fd = os.open(str(output.parent), os.O_RDONLY | os.O_DIRECTORY)
        except (AttributeError, OSError):
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
    return output


def load_telemetry_session_manifest(
    path: PathLike,
    *,
    verify_files: bool = True,
) -> LoadedTelemetrySessionManifest:
    """Load strict JSON, validate content, and by default replay bound files."""

    payload, source = _strict_json_file(path, "telemetry session manifest")
    validated = validate_telemetry_session_manifest(
        payload, verify_files=verify_files
    )
    return LoadedTelemetrySessionManifest(
        path=source,
        payload=validated.payload,
        identity=validated.identity,
        manifest_file_sha256=sha256_file(source),
    )


def _validate_calibration(value: Any) -> Mapping[str, Any]:
    calibration = _expect_exact_keys(
        value,
        (
            "algorithm",
            "calibration_id",
            "camera_serial",
            "T_reference_camera",
            "T_reference_camera_sha256",
            "sha256",
        ),
        "calibration fingerprint",
    )
    if calibration["algorithm"] != CALIBRATION_FINGERPRINT_ALGORITHM:
        raise ValueError("calibration fingerprint algorithm is unsupported")
    for name in ("calibration_id", "camera_serial"):
        if not isinstance(calibration[name], str) or not calibration[name].strip():
            raise ValueError("calibration {} must be non-empty".format(name))
    transform = validate_rigid_transform(
        np.asarray(calibration["T_reference_camera"], dtype=np.float64),
        "calibration T_reference_camera",
    )
    transform_sha = _array_sha256(
        transform, shape=(4, 4), name="calibration T_reference_camera"
    )
    if _sha256(
        calibration["T_reference_camera_sha256"],
        "calibration T_reference_camera_sha256",
    ) != transform_sha:
        raise ValueError("calibration transform SHA-256 does not match transform")
    supplied = _sha256(calibration["sha256"], "calibration sha256")
    unsigned = dict(calibration)
    del unsigned["sha256"]
    if canonical_json_sha256(unsigned) != supplied:
        raise ValueError("calibration sha256 does not match fingerprint content")
    return calibration


def _validate_execution(
    value: Any, audit_binding: Mapping[str, Any]
) -> Mapping[str, Any]:
    execution = _expect_exact_keys(
        value,
        (
            "kind",
            "command",
            "selected_index",
            "audit_artifact_file_sha256",
            "audit_artifact_content_sha256",
            "air_target_contract",
        ),
        "execution contract",
    )
    if execution["kind"] != EXECUTION_CONTRACT_KIND:
        raise ValueError("execution contract kind is unsupported")
    _execution_command(execution["command"])
    _selected_index(execution["selected_index"])
    if _sha256(
        execution["audit_artifact_file_sha256"],
        "execution audit_artifact_file_sha256",
    ) != audit_binding["file_sha256"]:
        raise ValueError("execution audit file hash differs from source binding")
    if _sha256(
        execution["audit_artifact_content_sha256"],
        "execution audit_artifact_content_sha256",
    ) != audit_binding["canonical_json_sha256"]:
        raise ValueError("execution audit content hash differs from source binding")
    _validate_air_target_contract(execution["air_target_contract"])
    return execution


def _validate_air_target_contract(value: Any) -> Mapping[str, Any]:
    contract = _expect_exact_keys(
        value,
        (
            "kind",
            "reference_frame",
            "retreat_distance_m",
            "pregrasp_extra_distance_m",
            "approach_reference",
            "T_reference_EE_nominal",
            "T_reference_EE_pregrasp",
            "T_reference_EE_final_air",
            "T_reference_hand_final_air",
            "array_sha256",
            "sha256",
        ),
        "air target contract",
    )
    if contract["kind"] != AIR_TARGET_CONTRACT_KIND:
        raise ValueError("air target contract kind is unsupported")
    if contract["reference_frame"] != "robot_base":
        raise ValueError("air target contract reference_frame must be robot_base")
    for name in ("retreat_distance_m", "pregrasp_extra_distance_m"):
        value_number = contract[name]
        if isinstance(value_number, bool) or not isinstance(value_number, (int, float)):
            raise ValueError("air target {} must be numeric".format(name))
        if not np.isfinite(float(value_number)) or float(value_number) <= 0.0:
            raise ValueError("air target {} must be finite and positive".format(name))
    approach = np.asarray(contract["approach_reference"], dtype=np.float64)
    if approach.shape != (3,) or not np.all(np.isfinite(approach)):
        raise ValueError("air target approach_reference must be a finite 3-vector")
    if not np.isclose(np.linalg.norm(approach), 1.0, atol=1.0e-10, rtol=0.0):
        raise ValueError("air target approach_reference must be a unit vector")
    arrays: Dict[str, np.ndarray] = {"approach_reference": approach}
    for name in (
        "T_reference_EE_nominal",
        "T_reference_EE_pregrasp",
        "T_reference_EE_final_air",
        "T_reference_hand_final_air",
    ):
        arrays[name] = validate_rigid_transform(
            np.asarray(contract[name], dtype=np.float64),
            "air target {}".format(name),
        )
    hashes = _expect_exact_keys(
        contract["array_sha256"], tuple(arrays), "air target array_sha256"
    )
    for name, array in arrays.items():
        expected = _array_sha256(
            array,
            shape=(3,) if name == "approach_reference" else (4, 4),
            name="air target {}".format(name),
        )
        if _sha256(hashes[name], "air target {} sha256".format(name)) != expected:
            raise ValueError("air target {} SHA-256 mismatch".format(name))
    supplied = _sha256(contract["sha256"], "air target contract sha256")
    unsigned = dict(contract)
    del unsigned["sha256"]
    if canonical_json_sha256(unsigned) != supplied:
        raise ValueError("air target contract sha256 does not match content")
    return contract


def _file_binding(
    value: Any,
    name: str,
    *,
    extra: Sequence[str],
) -> Mapping[str, Any]:
    binding = _expect_exact_keys(
        value, ("path", "file_sha256") + tuple(extra), name
    )
    path = binding["path"]
    if not isinstance(path, str) or not path:
        raise ValueError("{} path must be a non-empty string".format(name))
    candidate = Path(path)
    if not candidate.is_absolute() or str(candidate) != os.path.normpath(path):
        raise ValueError("{} path must be canonical and absolute".format(name))
    _sha256(binding["file_sha256"], "{} file_sha256".format(name))
    return binding


def _execution_command(value: Any) -> str:
    if not isinstance(value, str) or value not in EXECUTION_COMMANDS:
        raise ValueError(
            "command must be one of {}".format(", ".join(EXECUTION_COMMANDS))
        )
    return value


def _selected_index(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("selected_index must be a non-negative integer")
    return value


def _run_uuid(value: Optional[str]) -> str:
    if value is None:
        return str(uuid.uuid4())
    # TelemetryIdentity performs the exact shared canonical-spelling check.
    identity = TelemetryIdentity(
        run_uuid=value,
        execution_contract_sha256="0" * 64,
        source_snapshot_sha256="0" * 64,
        control_config_sha256="0" * 64,
        calibration_sha256="0" * 64,
        producer_build_sha256="0" * 64,
    )
    return identity.run_uuid


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("{} must be a lowercase SHA-256 digest".format(name))
    return value


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("{} must be a JSON object".format(name))
    return value


def _expect_exact_keys(
    value: Any, expected: Sequence[str], name: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("{} must be a JSON object".format(name))
    expected_set = set(expected)
    actual = set(value)
    if actual != expected_set:
        raise ValueError(
            "{} keys differ: missing={} unknown={}".format(
                name,
                sorted(expected_set - actual),
                sorted(actual - expected_set),
            )
        )
    return value


def _reject_duplicate_pairs(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key {!r}".format(key))
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant {!r} is forbidden".format(value))


__all__ = [
    "AIR_TARGET_CONTRACT_KIND",
    "ARTIFACT_TYPE",
    "CALIBRATION_FINGERPRINT_ALGORITHM",
    "EXECUTION_COMMANDS",
    "EXECUTION_CONTRACT_KIND",
    "INTEGRITY_ALGORITHM",
    "LoadedTelemetrySessionManifest",
    "SCHEMA_VERSION",
    "build_telemetry_session_manifest",
    "calibration_fingerprint",
    "canonical_json_sha256",
    "load_telemetry_session_manifest",
    "sha256_file",
    "validate_telemetry_session_manifest",
    "write_telemetry_session_manifest",
]
