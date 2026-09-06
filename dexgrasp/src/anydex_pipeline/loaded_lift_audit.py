"""Strict, hardware-free evidence contract for a loaded lift round trip.

An installed-tool ``loaded_grasp`` audit stops at the supported grasp pose.  It
does *not* prove that the grasp may be lifted.  This module deliberately uses a
different artifact type and schema so that an ordinary installed-tool JSON can
never be mistaken for loaded-lift permission.

The loaded-lift artifact binds all of the additional facts needed after the
hand has closed: a passing base loaded-grasp artifact, a load-rated adapter
approval, payload geometry/dynamics/retention evidence, the closed 13-link hand
model, an exact grasp-to-lift joint polyline and its reverse setdown path, dense
FK, and authoritative continuous collision checks.  The artifact remains
offline evidence only; ``motion_authorized`` is required to be false.

Importing this module opens no camera, serial port, FCI connection, or GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .control_plan import validate_rigid_transform
from .installed_tool_audit import (
    ARTIFACT_TYPE as INSTALLED_TOOL_ARTIFACT_TYPE,
    SCHEMA_VERSION as INSTALLED_TOOL_SCHEMA_VERSION,
    load_installed_tool_audit,
)
from .joint_path_sampling import (
    CANONICAL_JOINT_SAMPLING_ALGORITHM,
    canonical_joint_path_samples,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "fr3_rh56_loaded_lift_collision_audit"
MODE = "loaded_lift_round_trip"
MATERIAL_APPROVAL_ARTIFACT_TYPE = (
    "fr3_rh56_adapter_loaded_lift_material_approval"
)
RETENTION_APPROVAL_ARTIFACT_TYPE = "fr3_rh56_payload_retention_approval"
MASS_PROPERTIES_ARTIFACT_TYPE = "fr3_payload_mass_properties_approval"
COLLISION_REPORT_ARTIFACT_TYPE = "fr3_rh56_loaded_lift_collision_report"
LOADED_RECOVERY_PROCEDURE_ARTIFACT_TYPE = (
    "fr3_rh56_loaded_lift_recovery_procedure"
)

COSINE_PEAK_VELOCITY_FACTOR = math.pi / 2.0
COSINE_PEAK_ACCELERATION_FACTOR = math.pi * math.pi / 2.0

FR3_JOINT_LIMITS = np.asarray(
    [
        [-2.7437, 2.7437],
        [-1.7837, 1.7837],
        [-2.9007, 2.9007],
        [-3.0421, -0.1518],
        [-2.8065, 2.8065],
        [0.5445, 4.5169],
        [-3.0159, 3.0159],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class LoadedLiftCheckSpec:
    check_id: str
    scope: str
    expectation: str
    margin_policy: str


LOADED_LIFT_CHECK_SPECS: Tuple[LoadedLiftCheckSpec, ...] = (
    LoadedLiftCheckSpec(
        "fr3_self_loaded_round_trip", "fr3_vs_fr3", "clear", "robot"
    ),
    LoadedLiftCheckSpec(
        "fr3_scene_loaded_round_trip", "fr3_vs_scene", "clear", "scene"
    ),
    LoadedLiftCheckSpec(
        "adapter_fr3_loaded_round_trip", "adapter_vs_fr3", "clear", "robot"
    ),
    LoadedLiftCheckSpec(
        "adapter_scene_loaded_round_trip", "adapter_vs_scene", "clear", "scene"
    ),
    LoadedLiftCheckSpec(
        "closed_rh56_fr3_loaded_round_trip",
        "closed_rh56_vs_fr3",
        "clear",
        "robot",
    ),
    LoadedLiftCheckSpec(
        "closed_rh56_adapter_loaded_round_trip",
        "closed_rh56_vs_adapter",
        "clear",
        "robot",
    ),
    LoadedLiftCheckSpec(
        "closed_rh56_self_loaded_round_trip",
        "closed_rh56_self",
        "clear",
        "robot",
    ),
    LoadedLiftCheckSpec(
        "closed_rh56_scene_nonobject_loaded_round_trip",
        "closed_rh56_vs_scene_excluding_payload",
        "clear",
        "scene",
    ),
    LoadedLiftCheckSpec(
        "payload_fr3_loaded_round_trip", "payload_vs_fr3", "clear", "robot"
    ),
    LoadedLiftCheckSpec(
        "payload_adapter_loaded_round_trip",
        "payload_vs_adapter",
        "clear",
        "robot",
    ),
    LoadedLiftCheckSpec(
        "payload_scene_non_support_loaded_round_trip",
        "payload_vs_scene_excluding_support_contact",
        "clear",
        "scene",
    ),
    LoadedLiftCheckSpec(
        "closed_rh56_payload_retention_loaded_round_trip",
        "closed_rh56_vs_payload",
        "retained_contact",
        "retention",
    ),
)


def _expect_keys(
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


def _sha256_text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError("{} must be a lowercase SHA-256 hex string".format(name))
    text = value
    if (
        len(text) != 64
        or text != text.lower()
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise ValueError("{} must be a lowercase SHA-256 hex digest".format(name))
    return text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    shape = ",".join(str(item) for item in array.shape)
    digest = hashlib.sha256()
    digest.update(("dtype=<f8;shape={};".format(shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_number(
    value: Any, name: str, *, positive: bool = False, allow_zero: bool = False
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float)
    ):
        raise ValueError("{} must be a JSON number".format(name))
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("{} must be finite".format(name))
    if positive and (number < 0.0 if allow_zero else number <= 0.0):
        raise ValueError(
            "{} must be {}".format(
                name, "non-negative" if allow_zero else "positive"
            )
        )
    return number


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    if not _numeric_tree(value):
        raise ValueError("{} must contain only JSON numbers".format(name))
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be a finite {}-vector".format(name, size)) from exc
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError("{} must be a finite {}-vector".format(name, size))
    return result.copy()


def _numeric_tree(value: Any) -> bool:
    """Reject JSON booleans/strings before NumPy can silently coerce them.

    ``np.asarray([True, 1])`` and ``np.asarray(["1", "2"])`` both produce
    arrays that can otherwise pass finite/range checks.  Evidence arrays are
    numeric JSON, so accepting those coercions would make the exact schema
    claim false and creates boolean-equals-integer bypasses.
    """

    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "iuf":
            return False
        return all(_numeric_tree(item) for item in value.reshape(-1))
    if isinstance(value, (list, tuple)):
        return all(_numeric_tree(item) for item in value)
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.integer, np.floating))
    )


def _finite_array(value: Any, shape: Tuple[int, ...], name: str) -> np.ndarray:
    if not _numeric_tree(value):
        raise ValueError("{} must contain only JSON numbers".format(name))
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be a finite {} array".format(name, shape)) from exc
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError("{} must be a finite {} array".format(name, shape))
    return result.copy()


def _fr3_q(value: Any, name: str) -> np.ndarray:
    q = _finite_vector(value, 7, name)
    if np.any(q < FR3_JOINT_LIMITS[:, 0]) or np.any(
        q > FR3_JOINT_LIMITS[:, 1]
    ):
        raise ValueError("{} lies outside FR3 joint limits".format(name))
    return q


def _hand_targets(value: Any, name: str) -> np.ndarray:
    targets = _finite_vector(value, 6, name)
    if (
        np.any(targets < 0.0)
        or np.any(targets > 1000.0)
        or not np.array_equal(targets, np.rint(targets))
    ):
        raise ValueError("{} must contain six integer registers in 0..1000".format(name))
    return targets


def _inertia(value: Any, name: str) -> np.ndarray:
    inertia = _finite_array(value, (3, 3), name)
    if not np.allclose(inertia, inertia.T, atol=1e-12, rtol=0.0):
        raise ValueError("{} must be symmetric".format(name))
    moments = np.linalg.eigvalsh(inertia)
    if np.any(moments <= 0.0):
        raise ValueError("{} must be positive definite".format(name))
    if float(moments[-1]) > float(moments[0] + moments[1]) + 1e-12:
        raise ValueError("{} violates the principal-moment triangle inequality".format(name))
    return inertia.copy()


def _resolved_config_asset(config_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("configured asset path must be a non-empty string")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path(config_path).parent / candidate
    return candidate.resolve()


def _verify_bound_file(
    binding: Mapping[str, Any], name: str, *, verify_files: bool
) -> Path:
    path_value = binding["path"]
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("{}.path must be a non-empty string".format(name))
    source = Path(path_value).expanduser().resolve()
    _sha256_text(binding["sha256"], name + ".sha256")
    if verify_files and (
        not source.is_file() or _sha256_file(source) != binding["sha256"]
    ):
        raise ValueError("{} file no longer matches the artifact".format(name))
    return source


def _reject_duplicate_pairs(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate JSON key {!r}".format(key))
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant {!r} is forbidden".format(value))


def _load_strict_json(path: Path, name: str) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("cannot load {} {}: {}".format(name, source, exc)) from exc
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object".format(name))
    return value


def _load_bound_strict_json(
    binding: Mapping[str, Any], name: str
) -> Mapping[str, Any]:
    """Hash and parse the exact same bytes, avoiding a check/load race."""

    path_value = binding["path"]
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("{}.path must be a non-empty string".format(name))
    source = Path(path_value).expanduser().resolve()
    expected = _sha256_text(binding["sha256"], name + ".sha256")
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise ValueError("cannot load {} {}: {}".format(name, source, exc)) from exc
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("{} file no longer matches the artifact".format(name))
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("cannot load {} {}: {}".format(name, source, exc)) from exc
    if not isinstance(value, dict):
        raise ValueError("{} must contain a JSON object".format(name))
    return value


def _validate_loaded_recovery_procedure(
    binding: Mapping[str, Any], *, config_path: Path
) -> Mapping[str, Any]:
    """Replay the separately reviewed recovery contract without executing it."""

    procedure_path = _resolved_config_asset(config_path, binding["path"])
    normalized_binding = {
        "path": str(procedure_path),
        "sha256": binding["sha256"],
    }
    procedure = _expect_keys(
        _load_bound_strict_json(
            normalized_binding, "loaded-lift recovery procedure"
        ),
        (
            "schema_version",
            "artifact_type",
            "procedure_id",
            "mode",
            "executable",
            "required_stage_order",
            "arm_fault_action",
            "hand_fault_action",
            "setdown_path",
            "release_gate",
            "operator_immediate_stop_required",
            "procedure_reviewed",
            "motion_authorized",
        ),
        "loaded-lift recovery procedure",
    )
    executable = _expect_keys(
        procedure["executable"],
        ("path", "sha256", "interface"),
        "loaded-lift recovery executable",
    )
    executable_path = _resolved_config_asset(
        procedure_path, executable["path"]
    )
    _verify_bound_file(
        {"path": str(executable_path), "sha256": executable["sha256"]},
        "loaded-lift recovery executable",
        verify_files=True,
    )
    if not os.access(str(executable_path), os.X_OK):
        raise ValueError("loaded-lift recovery executable is not executable")
    expected_stages = [
        "read_and_verify_loaded_state",
        "preserve_numeric_closed_hand_hold",
        "stop_franka_in_place",
        "reverse_exact_audited_setdown_path",
        "verify_joint_and_eef_settle",
        "release_payload",
        "clear_payload_dynamics",
    ]
    if (
        type(procedure["schema_version"]) is not int
        or procedure["schema_version"] != 1
        or procedure["artifact_type"]
        != LOADED_RECOVERY_PROCEDURE_ARTIFACT_TYPE
        or not isinstance(procedure["procedure_id"], str)
        or not procedure["procedure_id"].strip()
        or procedure["mode"] != MODE
        or executable["interface"] != "loaded_lift_recovery_cli_v1"
        or procedure["required_stage_order"] != expected_stages
        or procedure["arm_fault_action"]
        != "stop_in_place_without_hand_release"
        or procedure["hand_fault_action"]
        != "preserve_last_verified_numeric_closed_hold"
        or procedure["setdown_path"]
        != "exact_audited_reverse_loaded_lift_path"
        or procedure["release_gate"]
        != "joint_and_eef_settle_verified_after_setdown"
        or procedure["operator_immediate_stop_required"] is not True
        or procedure["procedure_reviewed"] is not True
        or procedure["motion_authorized"] is not False
    ):
        raise ValueError(
            "loaded-lift recovery procedure is incomplete or unsafe"
        )
    return procedure


def _validate_material_approval(
    value: Mapping[str, Any], *, adapter_sha256: str, material: str,
    payload_mass_kg: float
) -> Mapping[str, Any]:
    approval = _expect_keys(
        value,
        (
            "schema_version",
            "artifact_type",
            "approval_id",
            "adapter_sha256",
            "material",
            "approved_payload_mass_kg",
            "approved_motion_scope",
            "max_joint_velocity_rad_s",
            "max_joint_acceleration_rad_s2",
            "load_rated",
            "approved",
            "motion_authorized",
        ),
        "material approval",
    )
    if approval["schema_version"] != 1 or isinstance(
        approval["schema_version"], bool
    ):
        raise ValueError("material approval schema_version must be 1")
    if approval["artifact_type"] != MATERIAL_APPROVAL_ARTIFACT_TYPE:
        raise ValueError("material approval artifact_type is unsupported")
    if not isinstance(approval["approval_id"], str) or not approval[
        "approval_id"
    ].strip():
        raise ValueError("material approval_id must be non-empty")
    if _sha256_text(
        approval["adapter_sha256"], "material approval adapter_sha256"
    ) != adapter_sha256:
        raise ValueError("material approval binds a different adapter")
    if approval["material"] != material or not str(material).strip():
        raise ValueError("material approval binds a different material")
    capacity = _finite_number(
        approval["approved_payload_mass_kg"],
        "material approval payload capacity",
        positive=True,
    )
    _finite_number(
        approval["max_joint_velocity_rad_s"],
        "material approval max joint velocity",
        positive=True,
    )
    _finite_number(
        approval["max_joint_acceleration_rad_s2"],
        "material approval max joint acceleration",
        positive=True,
    )
    if capacity + 1e-12 < payload_mass_kg:
        raise ValueError("payload mass exceeds material approval")
    if (
        approval["approved_motion_scope"] != MODE
        or approval["load_rated"] is not True
        or approval["approved"] is not True
        or approval["motion_authorized"] is not False
    ):
        raise ValueError("material approval does not approve loaded lift round trip")
    return approval


def _validate_retention_approval(
    value: Mapping[str, Any], *, payload_sha256: str,
    hand_targets_sha256: str, minimum_contact_axes: int,
    payload_mass_kg: float
) -> Mapping[str, Any]:
    approval = _expect_keys(
        value,
        (
            "schema_version",
            "artifact_type",
            "approval_id",
            "payload_geometry_sha256",
            "closed_hand_targets_sha256",
            "minimum_contact_axes",
            "validated_payload_mass_kg",
            "retention_verified",
            "loaded_lift_round_trip_approved",
            "motion_authorized",
        ),
        "retention approval",
    )
    if approval["schema_version"] != 1 or isinstance(
        approval["schema_version"], bool
    ):
        raise ValueError("retention approval schema_version must be 1")
    if approval["artifact_type"] != RETENTION_APPROVAL_ARTIFACT_TYPE:
        raise ValueError("retention approval artifact_type is unsupported")
    if not isinstance(approval["approval_id"], str) or not approval[
        "approval_id"
    ].strip():
        raise ValueError("retention approval_id must be non-empty")
    if _sha256_text(
        approval["payload_geometry_sha256"],
        "retention payload_geometry_sha256",
    ) != payload_sha256:
        raise ValueError("retention approval binds a different payload geometry")
    if _sha256_text(
        approval["closed_hand_targets_sha256"],
        "retention closed_hand_targets_sha256",
    ) != hand_targets_sha256:
        raise ValueError("retention approval binds different hand targets")
    if (
        type(approval["minimum_contact_axes"]) is not int
        or approval["minimum_contact_axes"] != minimum_contact_axes
    ):
        raise ValueError("retention approval contact-axis count differs")
    mass = _finite_number(
        approval["validated_payload_mass_kg"],
        "retention validated payload mass",
        positive=True,
    )
    if mass + 1e-12 < payload_mass_kg:
        raise ValueError("payload mass exceeds retention approval")
    if (
        approval["retention_verified"] is not True
        or approval["loaded_lift_round_trip_approved"] is not True
        or approval["motion_authorized"] is not False
    ):
        raise ValueError("payload retention is not approved for the round trip")
    return approval


def _validate_mass_properties_approval(
    value: Mapping[str, Any], *, payload_sha256: str,
    T_F_payload_sha256: str, mass_kg: float,
    F_x_Cload_m: np.ndarray, inertia_kg_m2: np.ndarray,
) -> Mapping[str, Any]:
    """Replay one independently generated mesh/dynamics calculation record."""

    approval = _expect_keys(
        value,
        (
            "schema_version",
            "artifact_type",
            "computation_id",
            "payload_geometry_sha256",
            "T_F_payload_sha256",
            "mass_kg",
            "F_x_Cload_m",
            "inertia_at_com_F_kg_m2",
            "backend_name",
            "backend_version",
            "implementation_sha256",
            "watertight_verified",
            "mass_properties_verified",
            "authoritative",
            "motion_authorized",
        ),
        "payload mass-properties approval",
    )
    if type(approval["schema_version"]) is not int or approval["schema_version"] != 1:
        raise ValueError("mass-properties approval schema_version must be 1")
    if approval["artifact_type"] != MASS_PROPERTIES_ARTIFACT_TYPE:
        raise ValueError("mass-properties approval artifact_type is unsupported")
    if not isinstance(approval["computation_id"], str) or not approval[
        "computation_id"
    ].strip():
        raise ValueError("mass-properties computation_id must be non-empty")
    if _sha256_text(
        approval["payload_geometry_sha256"],
        "mass-properties payload_geometry_sha256",
    ) != payload_sha256:
        raise ValueError("mass-properties approval binds a different payload mesh")
    if _sha256_text(
        approval["T_F_payload_sha256"],
        "mass-properties T_F_payload_sha256",
    ) != T_F_payload_sha256:
        raise ValueError("mass-properties approval binds a different payload transform")
    approved_mass = _finite_number(
        approval["mass_kg"], "mass-properties mass_kg", positive=True
    )
    approved_center = _finite_vector(
        approval["F_x_Cload_m"], 3, "mass-properties F_x_Cload_m"
    )
    approved_inertia = _inertia(
        approval["inertia_at_com_F_kg_m2"],
        "mass-properties inertia_at_com_F_kg_m2",
    )
    if (
        not np.isclose(approved_mass, mass_kg, atol=1e-15, rtol=0.0)
        or not np.array_equal(approved_center, F_x_Cload_m)
        or not np.array_equal(approved_inertia, inertia_kg_m2)
    ):
        raise ValueError("payload dynamics differ from mass-properties approval")
    for key in ("backend_name", "backend_version"):
        if not isinstance(approval[key], str) or not approval[key].strip():
            raise ValueError("mass-properties {} must be non-empty".format(key))
    _sha256_text(
        approval["implementation_sha256"],
        "mass-properties implementation_sha256",
    )
    if (
        approval["watertight_verified"] is not True
        or approval["mass_properties_verified"] is not True
        or approval["authoritative"] is not True
        or approval["motion_authorized"] is not False
    ):
        raise ValueError("payload mass properties are not authoritative and verified")
    return approval


def _segments_from_interval_counts(
    names: Sequence[str], interval_counts: Sequence[int]
) -> list[dict[str, Any]]:
    segments = []
    cursor = 0
    for start_name, end_name, count in zip(
        names[:-1], names[1:], interval_counts
    ):
        start = cursor
        cursor += int(count)
        segments.append(
            {
                "name": "{}_to_{}".format(start_name, end_name),
                "start_index": int(start),
                "end_index": int(cursor),
            }
        )
    return segments


def _official_endpoint_feedback_envelope(
    targets: Sequence[int], arrival_tolerance_units: int, mapper: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recompute the exact endpoint q12 box from the pinned official XLS."""

    registers = _hand_targets(targets, "feedback-envelope targets").astype(np.int64)
    if (
        isinstance(arrival_tolerance_units, (bool, np.bool_))
        or type(arrival_tolerance_units) is not int
        or not 0 <= arrival_tolerance_units <= 50
    ):
        raise ValueError("feedback-envelope tolerance must be an integer in 0..50")
    converter = getattr(mapper, "to_joint_positions_rad", None)
    if not callable(converter):
        raise TypeError("official mapper must provide to_joint_positions_rad")
    nominal = _finite_vector(
        converter(registers.tolist()), 12, "official nominal closed-hand q12"
    )
    lower = nominal.copy()
    upper = nominal.copy()
    claimed_support = set()
    for axis in range(6):
        samples = []
        low = max(0, int(registers[axis]) - arrival_tolerance_units)
        high = min(1000, int(registers[axis]) + arrival_tolerance_units)
        for value in range(low, high + 1):
            perturbed = registers.copy()
            perturbed[axis] = value
            samples.append(
                _finite_vector(
                    converter(perturbed.tolist()),
                    12,
                    "official feedback-envelope q12",
                )
            )
        stack = np.stack(samples, axis=0)
        affected = set(
            int(index)
            for index in np.flatnonzero(
                np.any(np.abs(stack - nominal[None, :]) > 1e-14, axis=0)
            )
        )
        if claimed_support.intersection(affected):
            raise ValueError("official RH56 actuator mapping is not joint-disjoint")
        claimed_support.update(affected)
        if affected:
            indices = np.asarray(sorted(affected), dtype=np.int64)
            lower[indices] = np.min(stack[:, indices], axis=0)
            upper[indices] = np.max(stack[:, indices], axis=0)
    return nominal, lower, upper


def _validate_collision_check(
    raw: Any,
    spec: LoadedLiftCheckSpec,
    *,
    sample_count: int,
    policies: Mapping[str, Any],
    feedback_envelope_sha256: str,
) -> Mapping[str, Any]:
    item = _expect_keys(
        raw,
        (
            "check_id",
            "scope",
            "expectation",
            "coverage",
            "authoritative",
            "tested_sample_indices",
            "minimum_signed_distance_m",
            "observed_pairs",
            "details",
            "passed",
            "failures",
        ),
        "collision check {}".format(spec.check_id),
    )
    if (
        item["check_id"] != spec.check_id
        or item["scope"] != spec.scope
        or item["expectation"] != spec.expectation
        or item["coverage"] != "loaded_round_trip_dense_path"
    ):
        raise ValueError("collision check {} contract differs".format(spec.check_id))
    failures = []
    if item["authoritative"] is not True:
        failures.append("observation is not authoritative")
    expected_indices = list(range(sample_count))
    if (
        not isinstance(item["tested_sample_indices"], list)
        or any(type(index) is not int for index in item["tested_sample_indices"])
        or item["tested_sample_indices"] != expected_indices
    ):
        failures.append("dense round-trip sample coverage is incomplete or out of order")
    distance = _finite_number(
        item["minimum_signed_distance_m"],
        "{}.minimum_signed_distance_m".format(spec.check_id),
    )
    if not isinstance(item["observed_pairs"], list) or any(
        not isinstance(pair, str) or not pair.strip()
        for pair in item["observed_pairs"]
    ):
        raise ValueError("collision observed_pairs must be a list of non-empty strings")
    if spec.expectation == "clear":
        margin = float(policies[spec.margin_policy + "_clearance_margin_m"])
        if distance <= margin:
            failures.append(
                "minimum signed distance does not exceed the {} margin".format(
                    spec.margin_policy
                )
            )
        if item["observed_pairs"]:
            failures.append("collision pairs were observed in a clear-required check")
    else:
        if distance > float(policies["retention_contact_max_separation_m"]):
            failures.append("hand/payload contact separation exceeds retention policy")
        if distance < -float(policies["retention_max_penetration_m"]):
            failures.append("hand/payload penetration exceeds retention policy")
        if not item["observed_pairs"]:
            failures.append("no retained hand/payload contact pair was observed")

    details = _expect_keys(
        item["details"],
        (
            "max_q_tracking_error_rad",
            "joint_tracking_uncertainty_applied",
            "continuous_segment_envelope_verified",
            "conservative_motion_bound_m",
            "minimum_distance_is_after_motion_bound",
            "closed_hand_feedback_envelope_applied",
            "closed_hand_feedback_envelope_sha256",
            "payload_pose_uncertainty_applied",
            "payload_geometry_included",
            "adapter_geometry_included",
            "fr3_geometry_included",
            "closed_hand_geometry_included",
            "unknown_space_policy",
            "unknown_space_policy_applied",
        ),
        "collision check {} details".format(spec.check_id),
    )
    reported_tracking = _finite_number(
        details["max_q_tracking_error_rad"],
        "collision check tracking error",
        positive=True,
    )
    if not np.isclose(
        reported_tracking,
        float(policies["max_q_tracking_error_rad"]),
        atol=1e-15,
        rtol=0.0,
    ):
        failures.append("joint tracking uncertainty differs from policy")
    _finite_number(
        details["conservative_motion_bound_m"],
        "collision conservative motion bound",
        positive=True,
        allow_zero=True,
    )
    required_true = (
        "joint_tracking_uncertainty_applied",
        "continuous_segment_envelope_verified",
        "minimum_distance_is_after_motion_bound",
        "closed_hand_feedback_envelope_applied",
        "payload_pose_uncertainty_applied",
        "payload_geometry_included",
        "adapter_geometry_included",
        "fr3_geometry_included",
        "closed_hand_geometry_included",
        "unknown_space_policy_applied",
    )
    for key in required_true:
        if details[key] is not True:
            failures.append("{} is not true".format(key))
    if details["unknown_space_policy"] != "occupied":
        failures.append("unknown-space policy is not occupied")
    if details["closed_hand_feedback_envelope_sha256"] != (
        feedback_envelope_sha256
    ):
        failures.append("closed-hand feedback envelope hash differs")
    if not isinstance(item["failures"], list) or any(
        not isinstance(value, str) for value in item["failures"]
    ):
        raise ValueError("collision check failures must be an array of strings")
    if item["failures"] != failures:
        raise ValueError(
            "collision check {} failures are inconsistent".format(spec.check_id)
        )
    if type(item["passed"]) is not bool or item["passed"] != (not failures):
        raise ValueError(
            "collision check {} passed flag is inconsistent".format(spec.check_id)
        )
    return item


def validate_loaded_lift_audit(
    artifact: Mapping[str, Any],
    *,
    verify_files: bool = False,
    require_pass: bool = False,
) -> Mapping[str, Any]:
    """Validate the exact schema, hashes, path, and recomputed decision."""

    root = _expect_keys(
        artifact,
        (
            "schema_version",
            "artifact_type",
            "mode",
            "created_at_s",
            "audit_generator",
            "bindings",
            "policies",
            "collision_backend",
            "checks",
            "decision",
            "artifact_sha256",
        ),
        "loaded-lift artifact",
    )
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise ValueError("unsupported loaded-lift audit schema_version")
    if root["artifact_type"] != ARTIFACT_TYPE:
        raise ValueError("artifact_type is not a loaded-lift collision audit")
    if root["mode"] != MODE:
        raise ValueError("loaded-lift audit mode must be {}".format(MODE))
    _finite_number(root["created_at_s"], "created_at_s", positive=True, allow_zero=True)
    supplied_digest = _sha256_text(root["artifact_sha256"], "artifact_sha256")
    unsigned = dict(root)
    del unsigned["artifact_sha256"]
    if _json_sha256(unsigned) != supplied_digest:
        raise ValueError("artifact_sha256 does not match artifact content")

    generator = _expect_keys(
        root["audit_generator"],
        ("name", "version", "implementation_path", "implementation_sha256"),
        "audit_generator",
    )
    if not isinstance(generator["name"], str) or not generator["name"].strip():
        raise ValueError("audit generator name must be non-empty")
    if not isinstance(generator["version"], str) or not generator["version"].strip():
        raise ValueError("audit generator version must be non-empty")
    generator_binding = {
        "path": generator["implementation_path"],
        "sha256": generator["implementation_sha256"],
    }
    _verify_bound_file(
        generator_binding, "loaded-lift audit generator", verify_files=verify_files
    )

    bindings = _expect_keys(
        root["bindings"],
        (
            "base_loaded_grasp",
            "control_profile",
            "adapter_material_approval",
            "payload",
            "closed_hand",
            "lift_endpoint",
            "joint_path",
            "fk_manifest",
            "collision_geometry",
            "collision_report",
        ),
        "bindings",
    )
    base = _expect_keys(
        bindings["base_loaded_grasp"],
        (
            "path",
            "file_sha256",
            "artifact_sha256",
            "schema_version",
            "artifact_type",
            "mode",
            "decision_passed",
            "selected_candidate_index",
            "grasp_q_sha256",
            "grasp_pose_sha256",
            "closed_hand_targets_sha256",
            "closed_hand_q12_sha256",
            "closure_fk_sha256",
            "adapter_sha256",
            "hand_model_binding_sha256",
        ),
        "bindings.base_loaded_grasp",
    )
    for key in (
        "file_sha256",
        "artifact_sha256",
        "grasp_q_sha256",
        "grasp_pose_sha256",
        "closed_hand_targets_sha256",
        "closed_hand_q12_sha256",
        "closure_fk_sha256",
        "adapter_sha256",
        "hand_model_binding_sha256",
    ):
        _sha256_text(base[key], "bindings.base_loaded_grasp." + key)
    if (
        base["schema_version"] != INSTALLED_TOOL_SCHEMA_VERSION
        or type(base["schema_version"]) is not int
        or base["artifact_type"] != INSTALLED_TOOL_ARTIFACT_TYPE
        or base["mode"] != "loaded_grasp"
        or base["decision_passed"] is not True
        or type(base["selected_candidate_index"]) is not int
        or base["selected_candidate_index"] < 0
    ):
        raise ValueError(
            "base evidence must explicitly bind one passing schema-v2 loaded_grasp audit"
        )
    if not isinstance(base["path"], str) or not base["path"].strip():
        raise ValueError("bindings.base_loaded_grasp.path must be a non-empty string")
    base_path = Path(base["path"]).expanduser().resolve()
    if verify_files and (
        not base_path.is_file() or _sha256_file(base_path) != base["file_sha256"]
    ):
        raise ValueError("base loaded-grasp file no longer matches the artifact")

    profile = _expect_keys(
        bindings["control_profile"],
        ("path", "sha256"),
        "bindings.control_profile",
    )
    _verify_bound_file(profile, "control profile", verify_files=verify_files)

    material = _expect_keys(
        bindings["adapter_material_approval"],
        (
            "path",
            "sha256",
            "approval_id",
            "adapter_sha256",
            "material",
            "approved_payload_mass_kg",
            "max_joint_velocity_rad_s",
            "max_joint_acceleration_rad_s2",
            "load_rated",
            "approved_motion_scope",
        ),
        "bindings.adapter_material_approval",
    )
    material_path = _verify_bound_file(
        material, "adapter material approval", verify_files=verify_files
    )
    if not isinstance(material["approval_id"], str) or not material[
        "approval_id"
    ].strip():
        raise ValueError("material approval_id must be non-empty")
    if (
        _sha256_text(material["adapter_sha256"], "material adapter_sha256")
        != base["adapter_sha256"]
        or not isinstance(material["material"], str)
        or not material["material"].strip()
        or material["load_rated"] is not True
        or material["approved_motion_scope"] != MODE
    ):
        raise ValueError("adapter material binding is not load-rated for this mode")
    approved_mass = _finite_number(
        material["approved_payload_mass_kg"],
        "approved payload mass",
        positive=True,
    )
    approved_velocity = _finite_number(
        material["max_joint_velocity_rad_s"],
        "approved maximum joint velocity",
        positive=True,
    )
    approved_acceleration = _finite_number(
        material["max_joint_acceleration_rad_s2"],
        "approved maximum joint acceleration",
        positive=True,
    )

    payload = _expect_keys(
        bindings["payload"],
        (
            "geometry",
            "dynamics",
            "attachment",
            "mass_properties",
            "retention_approval",
            "binding_sha256",
        ),
        "bindings.payload",
    )
    geometry = _expect_keys(
        payload["geometry"],
        ("path", "sha256", "representation", "frame", "watertight"),
        "bindings.payload.geometry",
    )
    _verify_bound_file(geometry, "payload geometry", verify_files=verify_files)
    if (
        geometry["representation"] != "watertight_triangle_mesh"
        or geometry["frame"] != "payload"
        or geometry["watertight"] is not True
    ):
        raise ValueError("payload geometry must be a watertight triangle mesh")
    dynamics = _expect_keys(
        payload["dynamics"],
        ("mass_kg", "F_x_Cload_m", "inertia_kg_m2"),
        "bindings.payload.dynamics",
    )
    mass_kg = _finite_number(
        dynamics["mass_kg"], "payload mass_kg", positive=True
    )
    F_x_Cload_m = _finite_vector(
        dynamics["F_x_Cload_m"], 3, "payload F_x_Cload_m"
    )
    inertia_kg_m2 = _inertia(
        dynamics["inertia_kg_m2"], "payload inertia_kg_m2"
    )
    if mass_kg > approved_mass + 1e-12:
        raise ValueError("payload mass exceeds bound adapter approval")
    attachment = _expect_keys(
        payload["attachment"],
        (
            "convention",
            "F_T_EE",
            "F_T_EE_sha256",
            "T_EE_hand",
            "T_EE_hand_sha256",
            "T_hand_payload",
            "T_hand_payload_sha256",
            "T_F_payload",
            "T_F_payload_sha256",
            "rigid_attachment_verified",
        ),
        "bindings.payload.attachment",
    )
    if (
        attachment["convention"]
        != "T_F_payload=F_T_EE@T_EE_hand@T_hand_payload"
    ):
        raise ValueError("unsupported payload attachment convention")
    F_T_EE = validate_rigid_transform(
        _finite_array(attachment["F_T_EE"], (4, 4), "F_T_EE"),
        "F_T_EE",
    )
    T_EE_hand = validate_rigid_transform(
        _finite_array(attachment["T_EE_hand"], (4, 4), "T_EE_hand"),
        "T_EE_hand",
    )
    T_hand_payload = validate_rigid_transform(
        _finite_array(
            attachment["T_hand_payload"], (4, 4), "T_hand_payload"
        ),
        "T_hand_payload",
    )
    T_F_payload = validate_rigid_transform(
        _finite_array(attachment["T_F_payload"], (4, 4), "T_F_payload"),
        "T_F_payload",
    )
    expected_T_F_payload = F_T_EE @ T_EE_hand @ T_hand_payload
    if (
        _array_sha256(F_T_EE)
        != _sha256_text(attachment["F_T_EE_sha256"], "F_T_EE sha256")
        or _array_sha256(T_EE_hand)
        != _sha256_text(attachment["T_EE_hand_sha256"], "T_EE_hand sha256")
        or _array_sha256(T_hand_payload)
        != _sha256_text(
            attachment["T_hand_payload_sha256"], "T_hand_payload sha256"
        )
        or _array_sha256(T_F_payload)
        != _sha256_text(
            attachment["T_F_payload_sha256"], "T_F_payload sha256"
        )
        or not np.allclose(
            T_F_payload, expected_T_F_payload, atol=1e-12, rtol=0.0
        )
        or attachment["rigid_attachment_verified"] is not True
    ):
        raise ValueError("payload attachment transform binding is invalid")
    mass_properties = _expect_keys(
        payload["mass_properties"],
        (
            "path",
            "sha256",
            "computation_id",
            "payload_geometry_sha256",
            "T_F_payload_sha256",
            "mass_kg",
            "F_x_Cload_m",
            "inertia_at_com_F_kg_m2",
            "backend_name",
            "backend_version",
            "implementation_sha256",
            "watertight_verified",
            "mass_properties_verified",
            "authoritative",
        ),
        "bindings.payload.mass_properties",
    )
    mass_properties_path = _verify_bound_file(
        mass_properties, "payload mass-properties approval", verify_files=verify_files
    )
    if (
        not isinstance(mass_properties["computation_id"], str)
        or not mass_properties["computation_id"].strip()
        or mass_properties["payload_geometry_sha256"] != geometry["sha256"]
        or mass_properties["T_F_payload_sha256"]
        != attachment["T_F_payload_sha256"]
        or not np.isclose(
            _finite_number(
                mass_properties["mass_kg"],
                "bound mass-properties mass_kg",
                positive=True,
            ),
            mass_kg,
            atol=1e-15,
            rtol=0.0,
        )
        or not np.array_equal(
            _finite_vector(
                mass_properties["F_x_Cload_m"],
                3,
                "bound mass-properties F_x_Cload_m",
            ),
            F_x_Cload_m,
        )
        or not np.array_equal(
            _inertia(
                mass_properties["inertia_at_com_F_kg_m2"],
                "bound mass-properties inertia_at_com_F_kg_m2",
            ),
            inertia_kg_m2,
        )
        or not isinstance(mass_properties["backend_name"], str)
        or not mass_properties["backend_name"].strip()
        or not isinstance(mass_properties["backend_version"], str)
        or not mass_properties["backend_version"].strip()
        or mass_properties["watertight_verified"] is not True
        or mass_properties["mass_properties_verified"] is not True
        or mass_properties["authoritative"] is not True
    ):
        raise ValueError("payload mass-properties binding is invalid")
    _sha256_text(
        mass_properties["implementation_sha256"],
        "bound mass-properties implementation_sha256",
    )
    retention = _expect_keys(
        payload["retention_approval"],
        (
            "path",
            "sha256",
            "approval_id",
            "payload_geometry_sha256",
            "closed_hand_targets_sha256",
            "minimum_contact_axes",
            "validated_payload_mass_kg",
            "retention_verified",
        ),
        "bindings.payload.retention_approval",
    )
    retention_path = _verify_bound_file(
        retention, "payload retention approval", verify_files=verify_files
    )
    if not isinstance(retention["approval_id"], str) or not retention[
        "approval_id"
    ].strip():
        raise ValueError("retention approval_id must be non-empty")
    if (
        _sha256_text(
            retention["payload_geometry_sha256"],
            "retention payload geometry sha256",
        )
        != geometry["sha256"]
        or _sha256_text(
            retention["closed_hand_targets_sha256"],
            "retention hand target sha256",
        )
        != base["closed_hand_targets_sha256"]
        or retention["retention_verified"] is not True
    ):
        raise ValueError("payload retention binding is invalid")
    minimum_contacts = retention["minimum_contact_axes"]
    if type(minimum_contacts) is not int or not 1 <= minimum_contacts <= 5:
        raise ValueError("minimum_contact_axes must be an integer in 1..5")
    retained_mass = _finite_number(
        retention["validated_payload_mass_kg"],
        "retention validated payload mass",
        positive=True,
    )
    if retained_mass + 1e-12 < mass_kg:
        raise ValueError("payload mass exceeds retention approval")
    payload_unsigned = {
        "geometry": dict(geometry),
        "dynamics": dict(dynamics),
        "attachment": dict(attachment),
        "mass_properties": dict(mass_properties),
        "retention_approval": dict(retention),
    }
    if _json_sha256(payload_unsigned) != _sha256_text(
        payload["binding_sha256"], "payload binding_sha256"
    ):
        raise ValueError("payload binding_sha256 is invalid")
    if verify_files:
        replayed_material = _validate_material_approval(
            _load_bound_strict_json(material, "adapter material approval"),
            adapter_sha256=base["adapter_sha256"],
            material=material["material"],
            payload_mass_kg=mass_kg,
        )
        if (
            replayed_material["approval_id"] != material["approval_id"]
            or not np.isclose(
                float(replayed_material["approved_payload_mass_kg"]),
                approved_mass,
                atol=1e-15,
                rtol=0.0,
            )
        ):
            raise ValueError("material approval metadata differs from its bound file")
        if (
            not np.isclose(
                float(replayed_material["max_joint_velocity_rad_s"]),
                approved_velocity,
                atol=1e-15,
                rtol=0.0,
            )
            or not np.isclose(
                float(replayed_material["max_joint_acceleration_rad_s2"]),
                approved_acceleration,
                atol=1e-15,
                rtol=0.0,
            )
        ):
            raise ValueError("material approval motion limits differ from its bound file")
        replayed_mass_properties = _validate_mass_properties_approval(
            _load_bound_strict_json(
                mass_properties, "payload mass-properties approval"
            ),
            payload_sha256=geometry["sha256"],
            T_F_payload_sha256=attachment["T_F_payload_sha256"],
            mass_kg=mass_kg,
            F_x_Cload_m=F_x_Cload_m,
            inertia_kg_m2=inertia_kg_m2,
        )
        if (
            replayed_mass_properties["computation_id"]
            != mass_properties["computation_id"]
            or replayed_mass_properties["backend_name"]
            != mass_properties["backend_name"]
            or replayed_mass_properties["backend_version"]
            != mass_properties["backend_version"]
            or replayed_mass_properties["implementation_sha256"]
            != mass_properties["implementation_sha256"]
        ):
            raise ValueError(
                "mass-properties metadata differs from its bound file"
            )
        replayed_retention = _validate_retention_approval(
            _load_bound_strict_json(retention, "payload retention approval"),
            payload_sha256=geometry["sha256"],
            hand_targets_sha256=base["closed_hand_targets_sha256"],
            minimum_contact_axes=minimum_contacts,
            payload_mass_kg=mass_kg,
        )
        if (
            replayed_retention["approval_id"] != retention["approval_id"]
            or not np.isclose(
                float(replayed_retention["validated_payload_mass_kg"]),
                retained_mass,
                atol=1e-15,
                rtol=0.0,
            )
        ):
            raise ValueError("retention approval metadata differs from its bound file")

    closed_hand = _expect_keys(
        bindings["closed_hand"],
        (
            "actuator_targets",
            "actuator_targets_sha256",
            "joint_positions_rad",
            "joint_positions_sha256",
            "closure_fk_sha256",
            "link_count",
            "link_mesh_manifest_sha256",
            "feedback_envelope_method",
            "feedback_q12_lower_rad",
            "feedback_q12_upper_rad",
            "feedback_envelope_sha256",
            "feedback_envelope_applied",
        ),
        "bindings.closed_hand",
    )
    targets = _hand_targets(closed_hand["actuator_targets"], "closed hand targets")
    q12 = _finite_vector(
        closed_hand["joint_positions_rad"], 12, "closed hand q12"
    )
    feedback_lower = _finite_vector(
        closed_hand["feedback_q12_lower_rad"],
        12,
        "closed hand feedback lower q12",
    )
    feedback_upper = _finite_vector(
        closed_hand["feedback_q12_upper_rad"],
        12,
        "closed hand feedback upper q12",
    )
    feedback_envelope = np.stack((feedback_lower, feedback_upper), axis=0)
    if (
        _array_sha256(targets)
        != _sha256_text(
            closed_hand["actuator_targets_sha256"],
            "closed hand targets sha256",
        )
        or closed_hand["actuator_targets_sha256"]
        != base["closed_hand_targets_sha256"]
        or _array_sha256(q12)
        != _sha256_text(
            closed_hand["joint_positions_sha256"], "closed hand q12 sha256"
        )
        or closed_hand["joint_positions_sha256"] != base["closed_hand_q12_sha256"]
        or _sha256_text(closed_hand["closure_fk_sha256"], "closed hand FK sha256")
        != base["closure_fk_sha256"]
        or type(closed_hand["link_count"]) is not int
        or closed_hand["link_count"] != 13
        or _sha256_text(
            closed_hand["link_mesh_manifest_sha256"],
            "closed hand mesh manifest sha256",
        )
        != base["hand_model_binding_sha256"]
        or closed_hand["feedback_envelope_method"]
        != "official_xls_all_six_arrival_box_v1"
        or np.any(feedback_lower > q12)
        or np.any(feedback_upper < q12)
        or np.any(feedback_lower > feedback_upper)
        or _array_sha256(feedback_envelope)
        != _sha256_text(
            closed_hand["feedback_envelope_sha256"],
            "closed hand feedback envelope sha256",
        )
        or closed_hand["feedback_envelope_applied"] is not True
    ):
        raise ValueError("closed-hand binding differs from the base loaded grasp")

    endpoint = _expect_keys(
        bindings["lift_endpoint"],
        ("q_rad", "q_sha256", "T_base_EE", "pose_sha256"),
        "bindings.lift_endpoint",
    )
    lift_q = _fr3_q(endpoint["q_rad"], "lift endpoint q")
    lift_pose = validate_rigid_transform(
        _finite_array(endpoint["T_base_EE"], (4, 4), "lift endpoint T_base_EE"),
        "lift endpoint T_base_EE",
    )
    if (
        _array_sha256(lift_q)
        != _sha256_text(endpoint["q_sha256"], "lift endpoint q sha256")
        or _array_sha256(lift_pose)
        != _sha256_text(endpoint["pose_sha256"], "lift endpoint pose sha256")
    ):
        raise ValueError("lift endpoint hashes are invalid")

    policies = _expect_keys(
        root["policies"],
        (
            "max_joint_step_rad",
            "max_q_tracking_error_rad",
            "robot_clearance_margin_m",
            "scene_clearance_margin_m",
            "retention_contact_max_separation_m",
            "retention_max_penetration_m",
            "payload_translation_uncertainty_m",
            "payload_rotation_uncertainty_rad",
            "hand_arrival_tolerance_units",
            "settle",
            "unknown_space_policy",
            "round_trip_setdown_required",
            "load_applied_after_grasp_settle",
            "load_cleared_only_after_setdown_settle_and_release",
        ),
        "policies",
    )
    numeric_policy_keys = (
        "max_joint_step_rad",
        "max_q_tracking_error_rad",
        "robot_clearance_margin_m",
        "scene_clearance_margin_m",
        "retention_contact_max_separation_m",
        "retention_max_penetration_m",
        "payload_translation_uncertainty_m",
        "payload_rotation_uncertainty_rad",
    )
    for key in numeric_policy_keys:
        _finite_number(
            policies[key], key, positive=True,
            allow_zero=key not in ("max_joint_step_rad", "max_q_tracking_error_rad")
        )
    if not 0.0 < float(policies["max_joint_step_rad"]) <= 0.05:
        raise ValueError("loaded lift max_joint_step_rad must be in (0,0.05]")
    if not 0.0 < float(policies["max_q_tracking_error_rad"]) <= 0.005:
        raise ValueError("loaded lift max_q_tracking_error_rad must be in (0,0.005]")
    if (
        type(policies["hand_arrival_tolerance_units"]) is not int
        or not 0 <= policies["hand_arrival_tolerance_units"] <= 50
    ):
        raise ValueError("hand arrival tolerance must be an integer in 0..50")
    settle = _expect_keys(
        policies["settle"],
        (
            "position_m",
            "orientation_rad",
            "linear_speed_m_s",
            "angular_speed_rad_s",
            "stable_seconds",
        ),
        "policies.settle",
    )
    for key in settle:
        _finite_number(
            settle[key], "settle." + key, positive=True,
            allow_zero=False,
        )
    settle_limits = {
        "position_m": (0.0, 0.010),
        "orientation_rad": (0.0, 0.100),
        "linear_speed_m_s": (0.0, 0.020),
        "angular_speed_rad_s": (0.0, 0.100),
        "stable_seconds": (0.200, 2.000),
    }
    for key, (minimum, maximum) in settle_limits.items():
        value = float(settle[key])
        if value < minimum or value > maximum:
            raise ValueError(
                "settle.{} must be in [{},{}]".format(key, minimum, maximum)
            )
    if (
        policies["unknown_space_policy"] != "occupied"
        or policies["round_trip_setdown_required"] is not True
        or policies["load_applied_after_grasp_settle"] is not True
        or policies[
            "load_cleared_only_after_setdown_settle_and_release"
        ]
        is not True
    ):
        raise ValueError("loaded-lift fail-closed round-trip policy was altered")

    path = _expect_keys(
        bindings["joint_path"],
        (
            "trajectory_contract",
            "sampling_algorithm",
            "waypoints",
            "segments",
            "outbound_sample_count",
            "outbound_samples_rad",
            "outbound_sha256",
            "round_trip_sample_count",
            "round_trip_samples_rad",
            "round_trip_sha256",
            "maximum_observed_joint_step_rad",
            "round_trip_setdown_required",
            "execution_time_law",
        ),
        "bindings.joint_path",
    )
    if (
        path["trajectory_contract"] != "loaded_lift_joint_waypoint_round_trip_v1"
        or path["sampling_algorithm"] != CANONICAL_JOINT_SAMPLING_ALGORITHM
        or path["round_trip_setdown_required"] is not True
    ):
        raise ValueError("unsupported loaded-lift trajectory contract")
    raw_waypoints = path["waypoints"]
    if not isinstance(raw_waypoints, list) or len(raw_waypoints) < 2:
        raise ValueError("loaded-lift waypoints must contain grasp and lift")
    names = []
    values = []
    for index, raw_waypoint in enumerate(raw_waypoints):
        waypoint = _expect_keys(
            raw_waypoint,
            ("name", "q_rad", "q_sha256"),
            "loaded-lift waypoint {}".format(index),
        )
        if not isinstance(waypoint["name"], str) or not waypoint["name"]:
            raise ValueError("loaded-lift waypoint name must be non-empty")
        q = _fr3_q(waypoint["q_rad"], "loaded-lift waypoint q")
        if _array_sha256(q) != _sha256_text(
            waypoint["q_sha256"], "loaded-lift waypoint q sha256"
        ):
            raise ValueError("loaded-lift waypoint q SHA-256 mismatch")
        names.append(waypoint["name"])
        values.append(q)
    expected_names = ["grasp"] + [
        "lift_transit_{}".format(index) for index in range(len(names) - 2)
    ] + ["lift"]
    if names != expected_names or len(set(names)) != len(names):
        raise ValueError(
            "lift waypoints must be grasp, optional lift_transit_N, lift"
        )
    if base["grasp_q_sha256"] != _array_sha256(values[0]):
        raise ValueError("lift path does not start at the base audited grasp q")
    if not np.array_equal(values[-1], lift_q):
        raise ValueError("lift endpoint q differs from final lift waypoint")
    if float(np.max(np.abs(values[-1] - values[0]))) <= float(
        policies["max_q_tracking_error_rad"]
    ):
        raise ValueError("loaded-lift path is a no-op within tracking uncertainty")
    outbound, interval_counts = canonical_joint_path_samples(
        values,
        float(policies["max_joint_step_rad"]),
        joint_count=7,
    )
    expected_segments = _segments_from_interval_counts(names, interval_counts)
    if not isinstance(path["segments"], list):
        raise ValueError("loaded-lift path segments must be an array")
    for index, raw_segment in enumerate(path["segments"]):
        segment_item = _expect_keys(
            raw_segment,
            ("name", "start_index", "end_index"),
            "loaded-lift path segment {}".format(index),
        )
        if (
            not isinstance(segment_item["name"], str)
            or not segment_item["name"]
            or type(segment_item["start_index"]) is not int
            or type(segment_item["end_index"]) is not int
        ):
            raise ValueError("loaded-lift path segment fields have invalid types")
    if path["segments"] != expected_segments:
        raise ValueError("loaded-lift path segments are inconsistent")
    stored_outbound = _finite_array(
        path["outbound_samples_rad"],
        outbound.shape,
        "outbound lift path samples",
    )
    if stored_outbound.shape != outbound.shape or not np.array_equal(
        stored_outbound, outbound
    ):
        raise ValueError("outbound lift path does not reconstruct canonically")
    if (
        type(path["outbound_sample_count"]) is not int
        or path["outbound_sample_count"] != len(outbound)
        or _array_sha256(outbound)
        != _sha256_text(path["outbound_sha256"], "outbound path sha256")
    ):
        raise ValueError("outbound lift path count/hash is invalid")
    round_trip = np.concatenate((outbound, outbound[-2::-1]), axis=0)
    stored_round_trip = _finite_array(
        path["round_trip_samples_rad"],
        round_trip.shape,
        "round-trip lift path samples",
    )
    if stored_round_trip.shape != round_trip.shape or not np.array_equal(
        stored_round_trip, round_trip
    ):
        raise ValueError("round-trip samples are not the exact audited reverse path")
    if (
        type(path["round_trip_sample_count"]) is not int
        or path["round_trip_sample_count"] != len(round_trip)
        or _array_sha256(round_trip)
        != _sha256_text(path["round_trip_sha256"], "round-trip path sha256")
    ):
        raise ValueError("round-trip path count/hash is invalid")
    maximum_step = float(np.max(np.abs(np.diff(round_trip, axis=0))))
    if not np.isclose(
        _finite_number(
            path["maximum_observed_joint_step_rad"],
            "maximum observed joint step",
            positive=True,
            allow_zero=True,
        ),
        maximum_step,
        atol=1e-15,
        rtol=0.0,
    ):
        raise ValueError("maximum observed joint step is inconsistent")

    time_law = _expect_keys(
        path["execution_time_law"],
        (
            "profile",
            "max_joint_velocity_rad_s",
            "max_joint_acceleration_rad_s2",
            "max_dynamic_segment_rad",
            "min_segment_duration_s",
            "velocity_peak_factor",
            "acceleration_peak_factor",
            "material_approval_limits_applied",
            "stop_at_every_waypoint",
            "reverse_reuses_same_limits",
        ),
        "bindings.joint_path.execution_time_law",
    )
    if time_law["profile"] != "cosine_stop_to_stop_dynamic_segments_v1":
        raise ValueError("unsupported loaded-lift execution time law")
    time_law_velocity = _finite_number(
        time_law["max_joint_velocity_rad_s"],
        "time-law maximum joint velocity",
        positive=True,
    )
    time_law_acceleration = _finite_number(
        time_law["max_joint_acceleration_rad_s2"],
        "time-law maximum joint acceleration",
        positive=True,
    )
    time_law_segment = _finite_number(
        time_law["max_dynamic_segment_rad"],
        "time-law maximum dynamic segment",
        positive=True,
    )
    time_law_duration = _finite_number(
        time_law["min_segment_duration_s"],
        "time-law minimum segment duration",
        positive=True,
    )
    velocity_factor = _finite_number(
        time_law["velocity_peak_factor"],
        "time-law velocity peak factor",
        positive=True,
    )
    acceleration_factor = _finite_number(
        time_law["acceleration_peak_factor"],
        "time-law acceleration peak factor",
        positive=True,
    )
    if (
        not np.isclose(
            velocity_factor,
            COSINE_PEAK_VELOCITY_FACTOR,
            atol=1e-15,
            rtol=0.0,
        )
        or not np.isclose(
            acceleration_factor,
            COSINE_PEAK_ACCELERATION_FACTOR,
            atol=1e-15,
            rtol=0.0,
        )
        or time_law_velocity > approved_velocity + 1e-15
        or time_law_acceleration > approved_acceleration + 1e-15
        or time_law_segment > 0.20
        or time_law["material_approval_limits_applied"] is not True
        or time_law["stop_at_every_waypoint"] is not True
        or time_law["reverse_reuses_same_limits"] is not True
    ):
        raise ValueError("loaded-lift time law exceeds or differs from its approval")
    worst_peak_velocity = (
        COSINE_PEAK_VELOCITY_FACTOR * time_law_segment / time_law_duration
    )
    worst_peak_acceleration = (
        COSINE_PEAK_ACCELERATION_FACTOR
        * time_law_segment
        / (time_law_duration * time_law_duration)
    )
    if (
        worst_peak_velocity > time_law_velocity + 1e-15
        or worst_peak_acceleration > time_law_acceleration + 1e-15
    ):
        raise ValueError(
            "loaded-lift segment/duration does not satisfy velocity/acceleration caps"
        )

    fk = _expect_keys(
        bindings["fk_manifest"],
        (
            "path",
            "sha256",
            "backend_name",
            "backend_version",
            "implementation_sha256",
            "joint_samples_sha256",
            "sample_count",
            "outbound_ee_poses_base",
            "outbound_ee_poses_sha256",
            "grasp_pose_sha256",
            "lift_pose_sha256",
            "all_samples_verified",
            "motion_authorized",
        ),
        "bindings.fk_manifest",
    )
    _verify_bound_file(fk, "FK manifest", verify_files=verify_files)
    if not isinstance(fk["backend_name"], str) or not fk["backend_name"].strip():
        raise ValueError("FK backend_name must be non-empty")
    if not isinstance(fk["backend_version"], str) or not fk[
        "backend_version"
    ].strip():
        raise ValueError("FK backend_version must be non-empty")
    _sha256_text(fk["implementation_sha256"], "FK implementation sha256")
    poses = _finite_array(
        fk["outbound_ee_poses_base"],
        (len(outbound), 4, 4),
        "FK manifest outbound EEF poses",
    )
    for index, pose in enumerate(poses):
        validate_rigid_transform(pose, "FK pose {}".format(index))
    if (
        fk["joint_samples_sha256"] != path["outbound_sha256"]
        or type(fk["sample_count"]) is not int
        or fk["sample_count"] != len(outbound)
        or _array_sha256(poses)
        != _sha256_text(fk["outbound_ee_poses_sha256"], "FK poses sha256")
        or fk["grasp_pose_sha256"] != base["grasp_pose_sha256"]
        or fk["lift_pose_sha256"] != endpoint["pose_sha256"]
        or _array_sha256(poses[0]) != base["grasp_pose_sha256"]
        or not np.array_equal(poses[-1], lift_pose)
        or fk["all_samples_verified"] is not True
        or fk["motion_authorized"] is not False
    ):
        raise ValueError("FK manifest path/endpoint binding is invalid")
    minimum_vertical_lift = max(
        0.01, 2.0 * float(settle["position_m"])
    )
    if float(poses[-1, 2, 3] - poses[0, 2, 3]) < minimum_vertical_lift:
        raise ValueError(
            "lift endpoint must rise in robot-base z beyond settle uncertainty"
        )

    geometry_manifest = _expect_keys(
        bindings["collision_geometry"],
        (
            "fr3_model",
            "adapter_mesh_sha256",
            "closed_hand_link_count",
            "closed_hand_link_manifest_sha256",
            "payload_mesh_sha256",
            "scene",
            "required_components",
            "all_geometry_included",
        ),
        "bindings.collision_geometry",
    )
    fr3_model = _expect_keys(
        geometry_manifest["fr3_model"],
        ("path", "sha256"),
        "bindings.collision_geometry.fr3_model",
    )
    _verify_bound_file(fr3_model, "FR3 collision model", verify_files=verify_files)
    scene = _expect_keys(
        geometry_manifest["scene"],
        (
            "path",
            "sha256",
            "reference_frame",
            "payload_removed",
            "support_contact_region_excluded",
        ),
        "bindings.collision_geometry.scene",
    )
    _verify_bound_file(scene, "collision scene", verify_files=verify_files)
    expected_components = [
        "fr3",
        "adapter",
        "closed_rh56_13_links",
        "payload",
        "scene",
    ]
    if (
        _sha256_text(
            geometry_manifest["adapter_mesh_sha256"],
            "collision adapter mesh sha256",
        )
        != base["adapter_sha256"]
        or geometry_manifest["closed_hand_link_count"] != 13
        or type(geometry_manifest["closed_hand_link_count"]) is not int
        or _sha256_text(
            geometry_manifest["closed_hand_link_manifest_sha256"],
            "collision hand mesh manifest sha256",
        )
        != base["hand_model_binding_sha256"]
        or _sha256_text(
            geometry_manifest["payload_mesh_sha256"],
            "collision payload mesh sha256",
        )
        != geometry["sha256"]
        or scene["reference_frame"] != "robot_base"
        or scene["payload_removed"] is not True
        or scene["support_contact_region_excluded"] is not True
        or geometry_manifest["required_components"] != expected_components
        or geometry_manifest["all_geometry_included"] is not True
    ):
        raise ValueError("loaded-lift collision geometry coverage is incomplete")

    collision_report = _expect_keys(
        bindings["collision_report"],
        ("path", "sha256", "report_id"),
        "bindings.collision_report",
    )
    _verify_bound_file(
        collision_report, "loaded-lift collision report", verify_files=verify_files
    )
    if not isinstance(collision_report["report_id"], str) or not collision_report[
        "report_id"
    ].strip():
        raise ValueError("collision report_id must be non-empty")

    backend = _expect_keys(
        root["collision_backend"],
        (
            "name",
            "version",
            "implementation_sha256",
            "configuration_sha256",
            "error",
        ),
        "collision_backend",
    )
    if not isinstance(backend["name"], str) or not backend["name"].strip():
        raise ValueError("collision backend name must be non-empty")
    if not isinstance(backend["version"], str) or not backend["version"].strip():
        raise ValueError("collision backend version must be non-empty")
    _sha256_text(backend["implementation_sha256"], "backend implementation sha256")
    _sha256_text(backend["configuration_sha256"], "backend configuration sha256")
    if not isinstance(backend["error"], str):
        raise ValueError("collision backend error must be a string")

    checks = root["checks"]
    if not isinstance(checks, list) or len(checks) != len(
        LOADED_LIFT_CHECK_SPECS
    ):
        raise ValueError("loaded-lift audit must contain every exact required check")
    rebuilt_checks = [
        _validate_collision_check(
            item,
            spec,
            sample_count=len(round_trip),
            policies=policies,
            feedback_envelope_sha256=closed_hand[
                "feedback_envelope_sha256"
            ],
        )
        for item, spec in zip(checks, LOADED_LIFT_CHECK_SPECS)
    ]
    expected_preconditions = (
        []
        if not backend["error"]
        else ["collision backend failed: {}".format(backend["error"])]
    )
    checks_pass = all(item["passed"] for item in rebuilt_checks)
    expected_pass = not expected_preconditions and checks_pass
    expected_reasons = list(expected_preconditions)
    expected_reasons.extend(
        "{}: {}".format(item["check_id"], failure)
        for item in rebuilt_checks
        for failure in item["failures"]
    )
    if verify_files:
        report = _expect_keys(
            _load_bound_strict_json(
                collision_report, "loaded-lift collision report"
            ),
            (
                "schema_version",
                "artifact_type",
                "report_id",
                "audit_generator_implementation_sha256",
                "collision_backend",
                "round_trip_samples_sha256",
                "joint_path_binding_sha256",
                "fk_manifest_binding_sha256",
                "payload_binding_sha256",
                "collision_geometry_binding_sha256",
                "closed_hand_feedback_envelope_sha256",
                "policies_sha256",
                "checks",
                "authoritative",
                "motion_authorized",
            ),
            "loaded-lift collision report",
        )
        if (
            type(report["schema_version"]) is not int
            or report["schema_version"] != 1
            or report["artifact_type"] != COLLISION_REPORT_ARTIFACT_TYPE
            or report["report_id"] != collision_report["report_id"]
            or report["audit_generator_implementation_sha256"]
            != generator["implementation_sha256"]
            or report["collision_backend"] != dict(backend)
            or report["round_trip_samples_sha256"] != path["round_trip_sha256"]
            or report["joint_path_binding_sha256"] != _json_sha256(path)
            or report["fk_manifest_binding_sha256"] != _json_sha256(fk)
            or report["payload_binding_sha256"] != payload["binding_sha256"]
            or report["collision_geometry_binding_sha256"]
            != _json_sha256(geometry_manifest)
            or report["closed_hand_feedback_envelope_sha256"]
            != closed_hand["feedback_envelope_sha256"]
            or report["policies_sha256"] != _json_sha256(policies)
            or report["checks"] != checks
            or report["authoritative"] is not True
            or report["motion_authorized"] is not False
        ):
            raise ValueError(
                "collision report differs from the exact loaded-lift evidence"
            )
    decision = _expect_keys(
        root["decision"],
        (
            "passed",
            "all_required_checks_passed",
            "precondition_failures",
            "reasons",
            "round_trip_loaded_lift_eligible",
            "runtime_conditions_satisfied_in_artifact",
            "motion_authorized",
            "meaning",
        ),
        "decision",
    )
    if (
        type(decision["passed"]) is not bool
        or decision["passed"] != expected_pass
        or type(decision["all_required_checks_passed"]) is not bool
        or decision["all_required_checks_passed"] != checks_pass
        or decision["precondition_failures"] != expected_preconditions
        or decision["reasons"] != expected_reasons
        or decision["round_trip_loaded_lift_eligible"] is not expected_pass
        or decision["runtime_conditions_satisfied_in_artifact"] is not False
        or decision["motion_authorized"] is not False
        or decision["meaning"]
        != "offline loaded-lift round-trip evidence only; never a motion command"
    ):
        raise ValueError("loaded-lift decision is inconsistent with evidence")
    if require_pass and not expected_pass:
        raise ValueError("loaded-lift audit did not pass: {}".format(expected_reasons))
    return artifact


def load_loaded_lift_audit(
    path: Path, *, verify_files: bool = False, require_pass: bool = False
) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve()
    artifact = _load_strict_json(source, "loaded-lift audit")
    return validate_loaded_lift_audit(
        artifact, verify_files=verify_files, require_pass=require_pass
    )


def write_loaded_lift_audit(path: Path, artifact: Mapping[str, Any]) -> Path:
    """Atomically write one already produced and strictly validated artifact."""

    validate_loaded_lift_audit(artifact)
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(".{}.{}.tmp".format(output.name, os.getpid()))
    temporary.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(output))
    return output


def loaded_lift_profile_blockers(
    config: Mapping[str, Any], config_path: Path
) -> Tuple[str, ...]:
    """Return explicit profile blockers before any audit is considered."""

    blockers = []
    tool = config.get("tool")
    inspire = config.get("inspire")
    lift = config.get("loaded_lift")
    if not isinstance(tool, Mapping):
        return ("control profile has no tool section",)
    if tool.get("low_speed_unloaded_commissioning_only") is not False:
        blockers.append(
            "installed adapter is approved only for low-speed unloaded checks"
        )
    if tool.get("installed_collision_model_verified") is not True:
        blockers.append("installed adapter + hand collision model is unverified")
    if not isinstance(inspire, Mapping) or inspire.get(
        "six_axis_coupled_closure_commissioned"
    ) is not True:
        blockers.append("six-axis coupled Inspire loaded closure is not commissioned")
    if not isinstance(lift, Mapping):
        blockers.append("control profile has no loaded_lift commissioning section")
        return tuple(blockers)
    required_true = (
        ("commissioned", "loaded lift is not commissioned"),
        (
            "round_trip_setdown_commissioned",
            "loaded reverse setdown path is not commissioned",
        ),
        (
            "load_rated_adapter_approved",
            "adapter has no load-rated loaded-lift approval",
        ),
    )
    for key, message in required_true:
        if lift.get(key) is not True:
            blockers.append(message)
    pinned_files = (
        ("material_approval", "material approval"),
        ("retention_approval", "retention approval"),
        ("mass_properties_approval", "mass-properties approval"),
        ("fk_manifest", "FK manifest"),
        ("collision_report", "authoritative collision report"),
        ("loaded_recovery_procedure", "loaded recovery procedure"),
    )
    for prefix, label in pinned_files:
        approval_path = lift.get(prefix + "_path")
        approval_sha = lift.get(prefix + "_sha256")
        if not isinstance(approval_path, str) or not approval_path.strip():
            blockers.append("loaded-lift {} path is missing".format(label))
        if not isinstance(approval_sha, str):
            blockers.append("loaded-lift {} SHA-256 is missing".format(label))
        else:
            try:
                _sha256_text(
                    approval_sha, "loaded-lift {} SHA-256".format(label)
                )
            except ValueError:
                blockers.append(
                    "loaded-lift {} SHA-256 is invalid".format(label)
                )
    recovery_path = lift.get("loaded_recovery_procedure_path")
    recovery_sha = lift.get("loaded_recovery_procedure_sha256")
    if (
        isinstance(recovery_path, str)
        and recovery_path.strip()
        and isinstance(recovery_sha, str)
    ):
        try:
            _validate_loaded_recovery_procedure(
                {"path": recovery_path, "sha256": recovery_sha},
                config_path=Path(config_path).expanduser().resolve(),
            )
        except (OSError, TypeError, ValueError) as exc:
            blockers.append(
                "loaded-lift recovery procedure replay failed: {}".format(exc)
            )
    identity_specs = (
        (
            "audit_generator",
            "audit generator",
            ("name", "version", "implementation_path", "implementation_sha256"),
        ),
        (
            "fk_backend",
            "FK backend",
            ("name", "version", "implementation_sha256"),
        ),
        (
            "collision_backend",
            "collision backend",
            (
                "name",
                "version",
                "implementation_sha256",
                "configuration_sha256",
            ),
        ),
        (
            "mass_properties_backend",
            "mass-properties backend",
            ("name", "version", "implementation_sha256"),
        ),
    )
    for key, label, expected_keys in identity_specs:
        identity = lift.get(key)
        if not isinstance(identity, Mapping) or set(identity) != set(expected_keys):
            blockers.append(
                "loaded-lift commissioned {} identity is missing or malformed".format(
                    label
                )
            )
            continue
        for text_key in ("name", "version"):
            if text_key in identity and (
                not isinstance(identity[text_key], str)
                or not identity[text_key].strip()
            ):
                blockers.append(
                    "loaded-lift commissioned {} {} is invalid".format(
                        label, text_key
                    )
                )
        if "implementation_path" in identity and (
            not isinstance(identity["implementation_path"], str)
            or not identity["implementation_path"].strip()
        ):
            blockers.append(
                "loaded-lift commissioned audit generator path is invalid"
            )
        for hash_key in (
            "implementation_sha256",
            "configuration_sha256",
        ):
            if hash_key not in identity:
                continue
            try:
                _sha256_text(
                    identity[hash_key],
                    "loaded-lift commissioned {} {}".format(label, hash_key),
                )
            except ValueError:
                blockers.append(
                    "loaded-lift commissioned {} {} is invalid".format(
                        label, hash_key
                    )
                )
    limit = lift.get("approved_payload_mass_limit_kg")
    try:
        _finite_number(limit, "approved payload mass limit", positive=True)
    except ValueError:
        blockers.append("loaded-lift approved payload mass limit is missing or invalid")
    commissioned_targets = lift.get("commissioned_closed_hand_targets")
    if not isinstance(commissioned_targets, list) or not commissioned_targets:
        blockers.append("no exact loaded closed-hand target is commissioned")
    else:
        for target in commissioned_targets:
            try:
                _hand_targets(target, "commissioned loaded hand target")
            except ValueError:
                blockers.append("loaded closed-hand commissioning targets are invalid")
                break
    return tuple(blockers)


@dataclass(frozen=True)
class LoadedLiftAuditBinding:
    path: Optional[Path]
    artifact: Optional[Mapping[str, Any]]
    base_artifact: Optional[Mapping[str, Any]]
    blockers: Tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.blockers and self.artifact is not None and self.base_artifact is not None

    @property
    def artifact_sha256(self) -> Optional[str]:
        return None if self.artifact is None else str(self.artifact["artifact_sha256"])

    @property
    def lift_waypoints(self) -> Tuple[Tuple[str, np.ndarray], ...]:
        if self.artifact is None:
            return ()
        return tuple(
            (item["name"], np.asarray(item["q_rad"], dtype=np.float64).copy())
            for item in self.artifact["bindings"]["joint_path"]["waypoints"]
        )

    @property
    def lift_pose(self) -> Optional[np.ndarray]:
        if self.artifact is None:
            return None
        return np.asarray(
            self.artifact["bindings"]["lift_endpoint"]["T_base_EE"],
            dtype=np.float64,
        ).copy()

    @property
    def payload_mass_kg(self) -> Optional[float]:
        if self.artifact is None:
            return None
        return float(
            self.artifact["bindings"]["payload"]["dynamics"]["mass_kg"]
        )

    @property
    def payload_F_x_Cload_m(self) -> Optional[np.ndarray]:
        if self.artifact is None:
            return None
        return np.asarray(
            self.artifact["bindings"]["payload"]["dynamics"]["F_x_Cload_m"],
            dtype=np.float64,
        ).copy()

    @property
    def payload_inertia_kg_m2(self) -> Optional[np.ndarray]:
        if self.artifact is None:
            return None
        return np.asarray(
            self.artifact["bindings"]["payload"]["dynamics"]["inertia_kg_m2"],
            dtype=np.float64,
        ).copy()

    @property
    def payload_binding_sha256(self) -> Optional[str]:
        if self.artifact is None:
            return None
        return str(self.artifact["bindings"]["payload"]["binding_sha256"])

    @property
    def minimum_contact_axes(self) -> Optional[int]:
        if self.artifact is None:
            return None
        return int(
            self.artifact["bindings"]["payload"]["retention_approval"][
                "minimum_contact_axes"
            ]
        )

    @property
    def max_q_tracking_error_rad(self) -> Optional[float]:
        if self.artifact is None:
            return None
        return float(self.artifact["policies"]["max_q_tracking_error_rad"])

    @property
    def settle_tolerances(self) -> Optional[Mapping[str, float]]:
        if self.artifact is None:
            return None
        return dict(self.artifact["policies"]["settle"])

    @property
    def approved_max_joint_velocity_rad_s(self) -> Optional[float]:
        if self.artifact is None:
            return None
        return float(
            self.artifact["bindings"]["adapter_material_approval"][
                "max_joint_velocity_rad_s"
            ]
        )

    @property
    def approved_max_joint_acceleration_rad_s2(self) -> Optional[float]:
        if self.artifact is None:
            return None
        return float(
            self.artifact["bindings"]["adapter_material_approval"][
                "max_joint_acceleration_rad_s2"
            ]
        )

    @property
    def execution_time_law(self) -> Optional[Mapping[str, Any]]:
        if self.artifact is None:
            return None
        return dict(
            self.artifact["bindings"]["joint_path"]["execution_time_law"]
        )

    @property
    def time_law_max_joint_velocity_rad_s(self) -> Optional[float]:
        law = self.execution_time_law
        return None if law is None else float(law["max_joint_velocity_rad_s"])

    @property
    def time_law_max_joint_acceleration_rad_s2(self) -> Optional[float]:
        law = self.execution_time_law
        return None if law is None else float(law["max_joint_acceleration_rad_s2"])

    @property
    def time_law_max_dynamic_segment_rad(self) -> Optional[float]:
        law = self.execution_time_law
        return None if law is None else float(law["max_dynamic_segment_rad"])

    @property
    def time_law_min_segment_duration_s(self) -> Optional[float]:
        law = self.execution_time_law
        return None if law is None else float(law["min_segment_duration_s"])


def _base_hand_model_binding(base: Mapping[str, Any]) -> Mapping[str, Any]:
    hand = base["bindings"]["hand_model"]
    return {
        "urdf_sha256": hand["urdf_sha256"],
        "mapping_sha256": hand["mapping_sha256"],
        "mesh_resolution": hand["mesh_resolution"],
        "link_count": hand["link_count"],
        "links": hand["links"],
        "closure_joint_positions_rad": hand["closure_joint_positions_rad"],
        "closure_fk_sha256": hand["closure_fk_sha256"],
    }


def bind_loaded_lift_audit(
    audit_path: Optional[Path],
    *,
    base_loaded_grasp_audit_path: Optional[Path],
    config: Mapping[str, Any],
    config_path: Path,
    expected_selected_candidate_index: Optional[int] = None,
    expected_grasp_q_rad: Optional[Sequence[float]] = None,
    expected_grasp_pose_base_EE: Optional[np.ndarray] = None,
    expected_closed_hand_targets: Optional[Sequence[int]] = None,
    verify_files: bool = True,
) -> LoadedLiftAuditBinding:
    """Bind one new loaded-lift artifact to one exact passing base audit.

    ``audit_path`` must point to :data:`ARTIFACT_TYPE`; passing an ordinary
    installed-tool artifact always fails at the distinct artifact-type check.
    """

    profile_blockers = list(
        loaded_lift_profile_blockers(config, Path(config_path).resolve())
    )
    if audit_path is None:
        return LoadedLiftAuditBinding(
            None,
            None,
            None,
            tuple(profile_blockers + ["a passing loaded-lift round-trip audit is required"]),
        )
    source = Path(audit_path).expanduser().resolve()
    try:
        artifact = load_loaded_lift_audit(
            source, verify_files=verify_files, require_pass=True
        )
    except (OSError, TypeError, ValueError) as exc:
        return LoadedLiftAuditBinding(
            source,
            None,
            None,
            tuple(profile_blockers + ["loaded-lift audit load/replay failed: {}".format(exc)]),
        )
    if base_loaded_grasp_audit_path is None:
        return LoadedLiftAuditBinding(
            source,
            artifact,
            None,
            tuple(profile_blockers + ["the exact base loaded-grasp audit is required"]),
        )
    base_path = Path(base_loaded_grasp_audit_path).expanduser().resolve()
    try:
        base_artifact = load_installed_tool_audit(
            base_path, verify_files=verify_files, require_pass=True
        )
    except (OSError, TypeError, ValueError) as exc:
        return LoadedLiftAuditBinding(
            source,
            artifact,
            None,
            tuple(profile_blockers + ["base loaded-grasp audit load/replay failed: {}".format(exc)]),
        )
    blockers = list(profile_blockers)
    if base_artifact.get("mode") != "loaded_grasp":
        blockers.append("base audit is not loaded_grasp evidence")
    base_binding = artifact["bindings"]["base_loaded_grasp"]
    if Path(base_binding["path"]).expanduser().resolve() != base_path:
        blockers.append("loaded-lift audit binds a different base audit path")
    if _sha256_file(base_path) != base_binding["file_sha256"]:
        blockers.append("base loaded-grasp file SHA-256 differs from loaded-lift audit")
    if base_artifact.get("artifact_sha256") != base_binding["artifact_sha256"]:
        blockers.append("base loaded-grasp artifact SHA-256 differs")
    if base_artifact.get("schema_version") != base_binding["schema_version"]:
        blockers.append("base loaded-grasp schema version differs")
    if base_artifact.get("artifact_type") != base_binding["artifact_type"]:
        blockers.append("base loaded-grasp artifact type differs")
    if base_artifact.get("decision", {}).get("passed") is not True:
        blockers.append("base loaded-grasp decision is not passing")

    base_bindings = base_artifact.get("bindings", {})
    base_snapshot = base_bindings.get("snapshot", {})
    base_candidate = base_snapshot.get("selected_candidate", {})
    base_joint_path = base_bindings.get("joint_path", {})
    base_waypoints = base_joint_path.get("waypoints", [])
    base_plan = base_bindings.get("execution_plan", {})
    base_hand = base_bindings.get("hand_model", {})
    base_adapter = base_bindings.get("adapter", {})
    if not base_waypoints or base_waypoints[-1].get("name") != "grasp":
        blockers.append("base loaded-grasp audit has no final grasp joint waypoint")
    else:
        actual_grasp_q = np.asarray(base_waypoints[-1].get("q_rad"), dtype=np.float64)
        if _array_sha256(actual_grasp_q) != base_binding["grasp_q_sha256"]:
            blockers.append("base grasp q differs from loaded-lift binding")
    try:
        actual_grasp_pose = np.asarray(base_plan["grasp_pose_base_EE"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        blockers.append("base loaded-grasp audit has no valid grasp pose")
        actual_grasp_pose = None
    if actual_grasp_pose is not None and _array_sha256(actual_grasp_pose) != base_binding[
        "grasp_pose_sha256"
    ]:
        blockers.append("base grasp pose differs from loaded-lift binding")
    try:
        actual_targets = _hand_targets(
            base_hand["closure_actuator_targets"], "base closure targets"
        )
        actual_q12 = _finite_vector(
            base_hand["closure_joint_positions_rad"], 12, "base closure q12"
        )
    except (KeyError, TypeError, ValueError) as exc:
        blockers.append("base closed-hand binding is invalid: {}".format(exc))
        actual_targets = None
        actual_q12 = None
    if actual_targets is not None and _array_sha256(actual_targets) != base_binding[
        "closed_hand_targets_sha256"
    ]:
        blockers.append("base closed-hand targets differ from loaded-lift binding")
    if actual_q12 is not None and _array_sha256(actual_q12) != base_binding[
        "closed_hand_q12_sha256"
    ]:
        blockers.append("base closed-hand q12 differs from loaded-lift binding")
    if base_hand.get("closure_fk_sha256") != base_binding["closure_fk_sha256"]:
        blockers.append("base closed-hand FK differs from loaded-lift binding")
    try:
        hand_manifest_sha = _json_sha256(_base_hand_model_binding(base_artifact))
    except (KeyError, TypeError, ValueError) as exc:
        blockers.append("base hand-model binding is incomplete: {}".format(exc))
    else:
        if hand_manifest_sha != base_binding["hand_model_binding_sha256"]:
            blockers.append("base 13-link hand-model binding differs")
    if base_adapter.get("sha256") != base_binding["adapter_sha256"]:
        blockers.append("base adapter binding differs")
    if int(base_candidate.get("index", -1)) != base_binding[
        "selected_candidate_index"
    ]:
        blockers.append("base selected candidate differs from loaded-lift binding")

    attachment = artifact["bindings"]["payload"]["attachment"]
    try:
        configured_F_T_EE = validate_rigid_transform(
            _finite_array(
                config["franka"]["expected_F_T_EE"],
                (4, 4),
                "configured expected_F_T_EE",
            ),
            "configured expected_F_T_EE",
        )
        base_T_EE_hand = validate_rigid_transform(
            _finite_array(
                base_bindings["mount_transform"]["T_EE_hand"],
                (4, 4),
                "base T_EE_hand",
            ),
            "base T_EE_hand",
        )
    except (KeyError, TypeError, ValueError) as exc:
        blockers.append("payload flange/hand transform provenance is invalid: {}".format(exc))
    else:
        if _array_sha256(configured_F_T_EE) != attachment["F_T_EE_sha256"]:
            blockers.append("payload dynamics bind a different configured F_T_EE")
        if _array_sha256(base_T_EE_hand) != attachment["T_EE_hand_sha256"]:
            blockers.append("payload dynamics bind a different base T_EE_hand")

    if actual_targets is not None and actual_q12 is not None:
        try:
            from .rh56_actuator_mapping import OfficialRH56ActuatorMapper

            provenance = base_hand["actuator_mapping_provenance"]
            mapper = OfficialRH56ActuatorMapper(provenance["driver_workbook_path"])
            tolerance = artifact["policies"]["hand_arrival_tolerance_units"]
            nominal, expected_lower, expected_upper = (
                _official_endpoint_feedback_envelope(
                    actual_targets.astype(np.int64).tolist(), tolerance, mapper
                )
            )
            artifact_hand = artifact["bindings"]["closed_hand"]
            actual_lower = _finite_vector(
                artifact_hand["feedback_q12_lower_rad"],
                12,
                "artifact feedback lower q12",
            )
            actual_upper = _finite_vector(
                artifact_hand["feedback_q12_upper_rad"],
                12,
                "artifact feedback upper q12",
            )
            if not np.array_equal(nominal, actual_q12):
                blockers.append(
                    "base closed-hand q12 differs from the official XLS mapping"
                )
            if not np.array_equal(actual_lower, expected_lower) or not np.array_equal(
                actual_upper, expected_upper
            ):
                blockers.append(
                    "closed-hand feedback envelope differs from the official XLS endpoint box"
                )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            blockers.append(
                "official closed-hand feedback-envelope replay failed: {}".format(exc)
            )

    config_resolved = Path(config_path).expanduser().resolve()
    profile = artifact["bindings"]["control_profile"]
    if Path(profile["path"]).expanduser().resolve() != config_resolved:
        blockers.append("loaded-lift audit binds a different control profile")
    elif _sha256_file(config_resolved) != profile["sha256"]:
        blockers.append("control profile SHA-256 differs from loaded-lift audit")

    lift_config = config.get("loaded_lift")
    if isinstance(lift_config, Mapping):
        approval = artifact["bindings"]["adapter_material_approval"]
        pinned_bindings = (
            ("material_approval", "material approval", approval),
            (
                "retention_approval",
                "retention approval",
                artifact["bindings"]["payload"]["retention_approval"],
            ),
            (
                "mass_properties_approval",
                "mass-properties approval",
                artifact["bindings"]["payload"]["mass_properties"],
            ),
            (
                "collision_report",
                "collision report",
                artifact["bindings"]["collision_report"],
            ),
            (
                "fk_manifest",
                "FK manifest",
                artifact["bindings"]["fk_manifest"],
            ),
        )
        for prefix, label, bound in pinned_bindings:
            configured_path = lift_config.get(prefix + "_path")
            if not isinstance(configured_path, str) or not configured_path.strip():
                blockers.append("{} path is not pinned by control profile".format(label))
            elif _resolved_config_asset(
                config_resolved, configured_path
            ) != Path(bound["path"]).expanduser().resolve():
                blockers.append("{} path differs from control profile".format(label))
            if lift_config.get(prefix + "_sha256") != bound["sha256"]:
                blockers.append(
                    "{} SHA-256 differs from control profile".format(label)
                )

        expected_generator = lift_config.get("audit_generator")
        actual_generator = {
            "name": artifact["audit_generator"]["name"],
            "version": artifact["audit_generator"]["version"],
            "implementation_path": artifact["audit_generator"][
                "implementation_path"
            ],
            "implementation_sha256": artifact["audit_generator"][
                "implementation_sha256"
            ],
        }
        if isinstance(expected_generator, Mapping):
            normalized_generator = dict(expected_generator)
            configured_generator_path = normalized_generator.get(
                "implementation_path"
            )
            if isinstance(configured_generator_path, str):
                normalized_generator["implementation_path"] = str(
                    _resolved_config_asset(
                        config_resolved, configured_generator_path
                    )
                )
            actual_generator["implementation_path"] = str(
                Path(actual_generator["implementation_path"])
                .expanduser()
                .resolve()
            )
            if normalized_generator != actual_generator:
                blockers.append("audit generator identity differs from control profile")
        else:
            blockers.append("audit generator identity is not pinned by control profile")

        actual_identities = {
            "fk_backend": {
                "name": artifact["bindings"]["fk_manifest"]["backend_name"],
                "version": artifact["bindings"]["fk_manifest"]["backend_version"],
                "implementation_sha256": artifact["bindings"]["fk_manifest"][
                    "implementation_sha256"
                ],
            },
            "collision_backend": {
                "name": artifact["collision_backend"]["name"],
                "version": artifact["collision_backend"]["version"],
                "implementation_sha256": artifact["collision_backend"][
                    "implementation_sha256"
                ],
                "configuration_sha256": artifact["collision_backend"][
                    "configuration_sha256"
                ],
            },
            "mass_properties_backend": {
                "name": artifact["bindings"]["payload"]["mass_properties"][
                    "backend_name"
                ],
                "version": artifact["bindings"]["payload"]["mass_properties"][
                    "backend_version"
                ],
                "implementation_sha256": artifact["bindings"]["payload"][
                    "mass_properties"
                ]["implementation_sha256"],
            },
        }
        for key, actual_identity in actual_identities.items():
            configured_identity = lift_config.get(key)
            if not isinstance(configured_identity, Mapping) or dict(
                configured_identity
            ) != actual_identity:
                blockers.append(
                    "{} identity differs from control profile".format(
                        key.replace("_", " ")
                    )
                )
        limit = lift_config.get("approved_payload_mass_limit_kg")
        if isinstance(limit, (int, float)) and not isinstance(limit, bool):
            if artifact["bindings"]["payload"]["dynamics"]["mass_kg"] > float(limit) + 1e-12:
                blockers.append("payload mass exceeds the control-profile limit")
        commissioned = lift_config.get("commissioned_closed_hand_targets", [])
        if actual_targets is not None and not any(
            np.array_equal(actual_targets, np.asarray(item, dtype=np.float64))
            for item in commissioned
            if isinstance(item, list) and len(item) == 6
        ):
            blockers.append("exact loaded closed-hand target is not commissioned")
    if config.get("tool", {}).get("material") != artifact["bindings"][
        "adapter_material_approval"
    ]["material"]:
        blockers.append("adapter material differs from the load-rated approval")

    if expected_selected_candidate_index is not None and int(
        expected_selected_candidate_index
    ) != base_binding["selected_candidate_index"]:
        blockers.append("runtime selected candidate differs from loaded-lift audit")
    if expected_grasp_q_rad is not None and _array_sha256(
        _fr3_q(expected_grasp_q_rad, "expected grasp q")
    ) != base_binding["grasp_q_sha256"]:
        blockers.append("runtime grasp q differs from loaded-lift audit")
    if expected_grasp_pose_base_EE is not None and _array_sha256(
        validate_rigid_transform(expected_grasp_pose_base_EE, "expected grasp pose")
    ) != base_binding["grasp_pose_sha256"]:
        blockers.append("runtime grasp pose differs from loaded-lift audit")
    if expected_closed_hand_targets is not None and _array_sha256(
        _hand_targets(expected_closed_hand_targets, "expected closed-hand targets")
    ) != base_binding["closed_hand_targets_sha256"]:
        blockers.append("runtime closed-hand targets differ from loaded-lift audit")

    return LoadedLiftAuditBinding(
        source,
        artifact,
        base_artifact,
        tuple(blockers),
    )


__all__ = [
    "ARTIFACT_TYPE",
    "COLLISION_REPORT_ARTIFACT_TYPE",
    "LOADED_LIFT_CHECK_SPECS",
    "LOADED_RECOVERY_PROCEDURE_ARTIFACT_TYPE",
    "LoadedLiftAuditBinding",
    "LoadedLiftCheckSpec",
    "MASS_PROPERTIES_ARTIFACT_TYPE",
    "MATERIAL_APPROVAL_ARTIFACT_TYPE",
    "MODE",
    "RETENTION_APPROVAL_ARTIFACT_TYPE",
    "SCHEMA_VERSION",
    "bind_loaded_lift_audit",
    "load_loaded_lift_audit",
    "loaded_lift_profile_blockers",
    "validate_loaded_lift_audit",
    "write_loaded_lift_audit",
]
