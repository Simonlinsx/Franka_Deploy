"""Strict, hardware-free audit contract for open-hand motion to pregrasp.

This artifact is intentionally different from the full grasp audit.  It binds
only the joint-space prefix

``current -> [default_transit_*] -> default -> [approach_transit_*] -> pregrasp``

with the RH56 in its verified official open configuration.  A failed full
grasp audit is never interpreted as prefix evidence.  The operator token is
the authority for the advisory scene/cable cloud only; it can never override
an authoritative robot/tool/object collision or clearance failure.

The module imports no camera or hardware driver.  ``motion_authorized`` is
always false: execution still needs the live-q, freshness, robot-state, and
operator gates in :mod:`apps.execute_control_sequence`.

The single-view *scene* cloud is deliberately advisory here.  The exact fresh
operator-clearance token is the environmental/cable gate.  In contrast, the
FR3/adapter/open-RH56 mesh checks and both checks against the exact bound
AnyDex object cloud are mandatory and cannot be overridden by that token.
The scene capture q is retained and hashed only as capture provenance; the
execution prefix current q is independently bound to the joint-plan manifest
and must match fresh live Franka feedback within the audited tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .control_plan import validate_rigid_transform
from .installed_tool_audit import (
    V7_ADAPTER_SHA256,
    V7_T_EE_HAND,
    V7_T_EE_HAND_SHA256,
)
from .joint_path_sampling import (
    CANONICAL_JOINT_SAMPLING_ALGORITHM,
    canonical_joint_path_samples,
)
from .snapshot import VisualizationSnapshot, load_snapshot_npz, validate_snapshot


SCHEMA_VERSION = 2
ARTIFACT_TYPE = "fr3_rh56_pregrasp_only_collision_audit"
EXECUTION_SCOPE = "open_hand_current_to_pregrasp_only"
RUNTIME_WORKSPACE_CLEAR_CONDITION_ID = "runtime_operator_workspace_clear_required"

REQUIRED_CHECK_IDS: Tuple[str, ...] = (
    "fr3_self_path",
    "fr3_scene_path",
    "adapter_fr3_path",
    "rh56_open_adapter_path",
    "adapter_scene_path",
    "adapter_object_path",
    "rh56_open_fr3_path",
    "rh56_open_scene_path",
    "rh56_open_object_path",
)
ROBOT_CHECK_IDS = frozenset(
    (
        "fr3_self_path",
        "adapter_fr3_path",
        "rh56_open_fr3_path",
        "rh56_open_adapter_path",
    )
)
SCENE_CHECK_IDS = frozenset(
    ("fr3_scene_path", "adapter_scene_path", "rh56_open_scene_path")
)
OBJECT_CHECK_IDS = frozenset(
    ("adapter_object_path", "rh56_open_object_path")
)


@dataclass(frozen=True)
class PregraspOnlyAuditBinding:
    path: Optional[Path]
    artifact: Optional[Mapping[str, Any]]
    blockers: Tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.blockers and self.artifact is not None


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
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError("{} must be a lowercase SHA-256 digest".format(name))
    return text


def _finite_q(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (7,) or not np.all(np.isfinite(result)):
        raise ValueError("{} must be a finite seven-axis vector".format(name))
    return result.copy()


def _finite_pose(value: Any, name: str) -> np.ndarray:
    return validate_rigid_transform(np.asarray(value, dtype=np.float64), name)


def _expect_exact_keys(value: Any, expected: Sequence[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("{} must be an object".format(name))
    missing = sorted(set(expected) - set(value))
    extra = sorted(set(value) - set(expected))
    if missing or extra:
        raise ValueError(
            "{} keys differ (missing={}, extra={})".format(name, missing, extra)
        )
    return value


def normalize_pregrasp_waypoints(
    waypoints: Sequence[Tuple[str, Sequence[float]]],
) -> Tuple[Tuple[str, np.ndarray], ...]:
    """Validate the only named waypoint grammar accepted by arm-only mode."""

    normalized = []
    for index, item in enumerate(tuple(waypoints)):
        try:
            name, q = item
        except (TypeError, ValueError) as exc:
            raise ValueError("waypoint {} must be a (name, q) pair".format(index)) from exc
        if not isinstance(name, str) or not name:
            raise ValueError("waypoint names must be non-empty strings")
        normalized.append((name, _finite_q(q, "waypoint {}".format(name))))
    names = [item[0] for item in normalized]
    if len(names) < 3 or names[0] != "current" or names[-1] != "pregrasp":
        raise ValueError(
            "pregrasp prefix must begin at current and end exactly at pregrasp"
        )
    if names.count("default") != 1 or len(set(names)) != len(names):
        raise ValueError("pregrasp prefix needs one default and unique waypoint names")
    default_index = names.index("default")
    expected_default = [
        "default_transit_{}".format(index) for index in range(default_index - 1)
    ]
    if names[1:default_index] != expected_default:
        raise ValueError("default transit names/order are invalid")
    expected_approach = [
        "approach_transit_{}".format(index)
        for index in range(len(names) - default_index - 2)
    ]
    if names[default_index + 1 : -1] != expected_approach:
        raise ValueError("approach transit names/order are invalid")
    forbidden = {"grasp", "thumb_preshape", "close", "lift"}
    if forbidden.intersection(names):
        raise ValueError("pregrasp-only prefix contains a forbidden post-pregrasp stage")
    return tuple(normalized)


def build_pregrasp_joint_path(
    waypoints: Sequence[Tuple[str, Sequence[float]]],
    max_joint_step_rad: float,
) -> Tuple[np.ndarray, Tuple[Mapping[str, Any], ...]]:
    normalized = normalize_pregrasp_waypoints(waypoints)
    maximum = float(max_joint_step_rad)
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("max_joint_step_rad must be finite and positive")
    samples, interval_counts = canonical_joint_path_samples(
        tuple(item[1] for item in normalized), maximum
    )
    segments = []
    cursor = 0
    for (start_name, _), (end_name, _), count in zip(
        normalized[:-1], normalized[1:], interval_counts
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
    return np.asarray(samples, dtype=np.float64), tuple(segments)


def pregrasp_prefix_contract_sha256(
    *,
    waypoints: Sequence[Tuple[str, Sequence[float]]],
    pregrasp_pose: Any,
    max_joint_step_rad: float,
    max_q_tracking_error_rad: float,
    samples_sha256: str,
) -> str:
    normalized = normalize_pregrasp_waypoints(waypoints)
    pose = _finite_pose(pregrasp_pose, "pregrasp_pose")
    maximum = float(max_joint_step_rad)
    tracking = float(max_q_tracking_error_rad)
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("max_joint_step_rad must be finite and positive")
    if not np.isfinite(tracking) or not 0.0 < tracking <= 0.01:
        raise ValueError("max_q_tracking_error_rad must be in (0,0.01]")
    payload = {
        "contract": "pregrasp_only_named_joint_prefix_v1",
        "sampling_algorithm": CANONICAL_JOINT_SAMPLING_ALGORITHM,
        "waypoints": [
            {"name": name, "q_rad": q.tolist()} for name, q in normalized
        ],
        "pregrasp_pose_base_EE": pose.tolist(),
        "max_joint_step_rad": maximum,
        "max_q_tracking_error_rad": tracking,
        "samples_sha256": _sha256_text(samples_sha256, "samples_sha256"),
    }
    return _json_sha256(payload)


def _check_margin(check_id: str, policies: Mapping[str, Any]) -> float:
    if check_id in ROBOT_CHECK_IDS:
        return float(policies["robot_clearance_margin_m"])
    if check_id in SCENE_CHECK_IDS:
        return float(policies["scene_clearance_margin_m"])
    if check_id in OBJECT_CHECK_IDS:
        return float(policies["object_clearance_margin_m"])
    raise ValueError("unknown pregrasp check {}".format(check_id))


def _check_geometry_failure(
    check: Mapping[str, Any], policies: Mapping[str, Any], sample_count: int
) -> Optional[str]:
    check_id = str(check.get("check_id", ""))
    if check_id not in REQUIRED_CHECK_IDS:
        return "unknown check"
    if check.get("coverage") != "full_path":
        return "coverage is not full_path"
    if check.get("expectation") != "clear":
        return "expectation is not clear"
    tested = check.get("tested_sample_indices")
    if tested != list(range(sample_count)):
        return "sample coverage is incomplete"
    if check.get("expected_sample_count") != sample_count:
        return "expected sample count differs from prefix"
    details = check.get("details")
    if not isinstance(details, dict):
        return "details are absent"
    tracking = details.get("max_q_tracking_error_rad")
    if (
        isinstance(tracking, bool)
        or not isinstance(tracking, (int, float))
        or not np.isclose(
            float(tracking),
            float(policies["max_q_tracking_error_rad"]),
            atol=1e-15,
            rtol=0.0,
        )
    ):
        return "joint tracking uncertainty differs from policy"
    if details.get("joint_tracking_uncertainty_applied") is not True:
        return "joint tracking uncertainty was not applied"
    if details.get("continuous_segment_envelope_verified") is not True:
        return "continuous joint segment envelope is unverified"
    if details.get("minimum_distance_is_after_motion_bound") is not True:
        return "minimum clearance omits the motion bound"
    bound = details.get("conservative_motion_bound_m")
    if (
        isinstance(bound, bool)
        or not isinstance(bound, (int, float))
        or not np.isfinite(float(bound))
        or float(bound) < 0.0
    ):
        return "conservative motion bound is malformed"
    if check_id in ROBOT_CHECK_IDS and check.get("authoritative") is not True:
        return "robot geometry check is non-authoritative"
    if check_id == "rh56_open_adapter_path":
        # This is a rigid installed interface, not a configurable collision
        # allow-list.  Link111 is the one designed seating/spigot pair; every
        # other full-resolution RH56 link must be tested at every prefix
        # sample.  Arm interpolation/tracking moves both bodies by the same
        # transform, so (and only so) the relative-motion bound is exactly 0.
        if details.get("geometry_model") != "exact triangle meshes":
            return "open-hand/adapter check did not use exact triangle meshes"
        if details.get("fixed_mount_pair_exclusions") != ["Link111"]:
            return "open-hand/adapter fixed-mount exclusion differs from Link111"
        checked_count = details.get("checked_non_mount_link_count")
        if isinstance(checked_count, bool) or checked_count != 12:
            return "open-hand/adapter check did not cover twelve non-mount links"
        evaluated_count = details.get("evaluated_pair_count")
        if (
            isinstance(evaluated_count, bool)
            or evaluated_count != 12 * sample_count
        ):
            return "open-hand/adapter pair/sample coverage is incomplete"
        if details.get("rigid_relative_transform_invariant") is not True:
            return "open-hand/adapter rigid relative transform is unverified"
        if details.get("relative_transform") != (
            "inv(T_base_EE@T_EE_adapter)@(T_base_EE@T_EE_hand)"
        ):
            return "open-hand/adapter relative-transform contract was altered"
        if float(bound) != 0.0:
            return "open-hand/adapter rigid-relative motion bound is not zero"
        if details.get("motion_bound_method") != (
            "zero: V7 and open RH56 share the same rigid EE transform, "
            "including interpolation and tracking error"
        ):
            return "open-hand/adapter tracking semantics were altered"
        minimum_pair = str(details.get("minimum_pair", ""))
        if not minimum_pair.startswith("adapter / ") or minimum_pair == (
            "adapter / Link111"
        ):
            return "open-hand/adapter minimum pair violates mount-exclusion scope"
    if check_id not in ROBOT_CHECK_IDS:
        if details.get("point_cloud_observations_authoritative") is not False:
            return "captured-point authority semantics were altered"
        if details.get("unknown_space_policy") != "occupied":
            return "unknown-space policy is not occupied"
    distance = check.get("minimum_signed_distance_m")
    if (
        isinstance(distance, bool)
        or not isinstance(distance, (int, float))
        or not np.isfinite(float(distance))
    ):
        return "minimum distance is malformed"
    # Full-scene returns contain the installed robot, hand and cable.  They are
    # displayed and hashed, but do not decide this arm-only prefix.  The fresh
    # workspace token is the environmental authority.  The object checks are
    # still mandatory: they are exact-mesh versus the exact bound AnyDex
    # object-point cube union and cannot be bypassed by that token.
    if check_id in SCENE_CHECK_IDS:
        return None
    if float(distance) <= _check_margin(check_id, policies):
        return "observed clearance does not exceed the required margin"
    pairs = check.get("observed_pairs")
    if not isinstance(pairs, list) or pairs:
        return "an observed collision pair exists"
    return None


def create_pregrasp_only_audit(
    *,
    config_path: Path,
    snapshot_path: Path,
    filtered_scene_path: Path,
    filter_evidence_path: Path,
    adapter_path: Path,
    joint_plan_path: Path,
    candidate_index: int,
    T_EE_hand: Any,
    scene_points: Any,
    object_points: Any,
    scene_capture_q: Any,
    scene_captured_at_s: float,
    waypoints: Sequence[Tuple[str, Sequence[float]]],
    pregrasp_pose: Any,
    max_joint_step_rad: float,
    max_q_tracking_error_rad: float,
    scene_clearance_margin_m: float,
    robot_clearance_margin_m: float,
    object_clearance_margin_m: float,
    max_scene_age_s: float,
    q_path: Any,
    path_segments: Sequence[Mapping[str, Any]],
    checks: Sequence[Mapping[str, Any]],
    collision_backend: Mapping[str, Any],
    open_hand_configuration: Mapping[str, Any],
    created_at_s: float,
) -> Dict[str, Any]:
    """Create and immediately self-validate one immutable prefix artifact."""

    config_path = Path(config_path).expanduser().resolve()
    snapshot_path = Path(snapshot_path).expanduser().resolve()
    filtered_scene_path = Path(filtered_scene_path).expanduser().resolve()
    filter_evidence_path = Path(filter_evidence_path).expanduser().resolve()
    adapter_path = Path(adapter_path).expanduser().resolve()
    joint_plan_path = Path(joint_plan_path).expanduser().resolve()
    for path in (
        config_path,
        snapshot_path,
        filtered_scene_path,
        filter_evidence_path,
        adapter_path,
        joint_plan_path,
    ):
        if not path.is_file():
            raise FileNotFoundError("pregrasp audit input not found: {}".format(path))

    normalized = normalize_pregrasp_waypoints(waypoints)
    samples = np.asarray(q_path, dtype=np.float64)
    rebuilt, rebuilt_segments = build_pregrasp_joint_path(
        normalized, max_joint_step_rad
    )
    if samples.shape != rebuilt.shape or not np.array_equal(samples, rebuilt):
        raise ValueError("provided q_path is not the canonical named pregrasp prefix")
    if [dict(item) for item in path_segments] != [dict(item) for item in rebuilt_segments]:
        raise ValueError("provided path segments differ from canonical prefix segments")
    pose = _finite_pose(pregrasp_pose, "pregrasp_pose")
    capture_q = _finite_q(scene_capture_q, "scene_capture_q")
    prefix_current_q = normalized[0][1]
    capture_to_prefix_linf = float(
        np.max(np.abs(capture_q - prefix_current_q))
    )
    capture_matches_prefix = bool(
        np.array_equal(capture_q, prefix_current_q)
    )
    points = np.asarray(scene_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0:
        raise ValueError("scene_points must be non-empty (N,3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("scene_points contain NaN or infinity")
    object_cloud = np.asarray(object_points, dtype=np.float64)
    if object_cloud.ndim != 2 or object_cloud.shape[1:] != (3,) or len(object_cloud) == 0:
        raise ValueError("object_points must be non-empty (N,3)")
    if not np.all(np.isfinite(object_cloud)):
        raise ValueError("object_points contain NaN or infinity")
    created = float(created_at_s)
    captured = float(scene_captured_at_s)
    maximum_age = float(max_scene_age_s)
    if not all(np.isfinite(value) for value in (created, captured, maximum_age)):
        raise ValueError("audit/scene times must be finite")
    if maximum_age <= 0.0:
        raise ValueError("max_scene_age_s must be positive")
    scene_age_at_creation = created - captured
    scene_age_warning = scene_age_at_creation > maximum_age

    policies = {
        "scene_clearance_margin_m": float(scene_clearance_margin_m),
        "robot_clearance_margin_m": float(robot_clearance_margin_m),
        "object_clearance_margin_m": float(object_clearance_margin_m),
        "max_scene_age_s": maximum_age,
        "max_joint_step_rad": float(max_joint_step_rad),
        "max_q_tracking_error_rad": float(max_q_tracking_error_rad),
        "unknown_space_policy": "occupied",
    }
    for name in (
        "scene_clearance_margin_m",
        "robot_clearance_margin_m",
        "object_clearance_margin_m",
        "max_joint_step_rad",
        "max_q_tracking_error_rad",
    ):
        value = float(policies[name])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("{} must be finite and positive".format(name))
    if float(policies["max_q_tracking_error_rad"]) > 0.01:
        raise ValueError("max_q_tracking_error_rad exceeds 0.01")

    check_by_id = {str(item.get("check_id", "")): dict(item) for item in checks}
    if tuple(check_by_id) != REQUIRED_CHECK_IDS:
        raise ValueError("pregrasp checks must be supplied once in canonical order")
    normalized_checks = []
    failures = []
    for check_id in REQUIRED_CHECK_IDS:
        item = check_by_id[check_id]
        failure = _check_geometry_failure(item, policies, len(samples))
        item["required_for_decision"] = check_id not in SCENE_CHECK_IDS
        item["advisory_only"] = check_id in SCENE_CHECK_IDS
        item["authority_basis"] = (
            "authoritative_robot_and_installed_tool_mesh_geometry"
            if check_id in ROBOT_CHECK_IDS
            else (
                "authoritative_for_exact_bound_anydex_object_point_cube_union"
                if check_id in OBJECT_CHECK_IDS
                else "advisory_filtered_scene_return_only"
            )
        )
        item["authoritative_for_bound_object_cloud"] = (
            check_id in OBJECT_CHECK_IDS
        )
        item["required_geometry_passed"] = failure is None
        item["geometry_failure"] = "" if failure is None else failure
        normalized_checks.append(item)
        if failure is not None:
            failures.append("{}: {}".format(check_id, failure))

    sample_hash = _array_sha256(samples)
    prefix_hash = pregrasp_prefix_contract_sha256(
        waypoints=normalized,
        pregrasp_pose=pose,
        max_joint_step_rad=float(max_joint_step_rad),
        max_q_tracking_error_rad=float(max_q_tracking_error_rad),
        samples_sha256=sample_hash,
    )
    config_hash = _sha256_file(config_path)
    snapshot_hash = _sha256_file(snapshot_path)
    scene_hash = _sha256_file(filtered_scene_path)
    filter_hash = _sha256_file(filter_evidence_path)
    adapter_hash = _sha256_file(adapter_path)
    joint_plan_hash = _sha256_file(joint_plan_path)
    transform = _finite_pose(T_EE_hand, "T_EE_hand")
    artifact: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "execution_scope": EXECUTION_SCOPE,
        "created_at_s": created,
        "motion_authorized": False,
        "bindings": {
            "control_profile": {"path": str(config_path), "sha256": config_hash},
            "snapshot": {
                "path": str(snapshot_path),
                "sha256": snapshot_hash,
                "selected_candidate_index": int(candidate_index),
            },
            "filtered_scene": {
                "path": str(filtered_scene_path),
                "sha256": scene_hash,
                "points_sha256": _array_sha256(points),
                "point_count": int(len(points)),
                "capture_q_rad": capture_q.tolist(),
                "capture_q_sha256": _array_sha256(capture_q),
                "capture_q_role": "advisory_scene_provenance_only",
                "capture_to_prefix_current_linf_rad": capture_to_prefix_linf,
                "capture_matches_prefix_current": capture_matches_prefix,
                "captured_at_s": captured,
                "age_at_artifact_creation_s": scene_age_at_creation,
                "configured_recommended_max_scene_age_s": maximum_age,
                "age_exceeded_recommended_max": scene_age_warning,
                "freshness_role": "advisory_provenance_only",
                "freshness_warning": (
                    "filtered scene exceeded the configured recommended age at "
                    "artifact creation; scene checks remain advisory and every "
                    "execution requires a fresh exact workspace-clear token"
                    if scene_age_warning
                    else ""
                ),
            },
            "object_cloud": {
                "source": "bound official AnyDex snapshot object_points",
                "points_sha256": _array_sha256(object_cloud),
                "point_count": int(len(object_cloud)),
            },
            "filter_evidence": {
                "path": str(filter_evidence_path),
                "sha256": filter_hash,
            },
            "adapter": {
                "path": str(adapter_path),
                "sha256": adapter_hash,
                "required_v7_sha256": V7_ADAPTER_SHA256,
            },
            "mount_transform": {
                "T_EE_hand": transform.tolist(),
                "sha256": _array_sha256(transform),
                "required_v7_sha256": V7_T_EE_HAND_SHA256,
            },
            "joint_plan_manifest": {
                "path": str(joint_plan_path),
                "sha256": joint_plan_hash,
            },
            "joint_prefix": {
                "contract": "pregrasp_only_named_joint_prefix_v1",
                "prefix_contract_sha256": prefix_hash,
                "sampling_algorithm": CANONICAL_JOINT_SAMPLING_ALGORITHM,
                "waypoints": [
                    {"name": name, "q_rad": q.tolist()} for name, q in normalized
                ],
                "current_q_sha256": _array_sha256(prefix_current_q),
                "current_q_source": "joint_plan_manifest.q_start_rad",
                "runtime_live_q_gate_required": True,
                "pregrasp_pose_base_EE": pose.tolist(),
                "max_joint_step_rad": float(max_joint_step_rad),
                "max_q_tracking_error_rad": float(max_q_tracking_error_rad),
                "segments": [dict(item) for item in rebuilt_segments],
                "sample_count": int(len(samples)),
                "samples_rad": samples.tolist(),
                "samples_sha256": sample_hash,
            },
            "open_hand_configuration": dict(open_hand_configuration),
        },
        "policies": policies,
        "collision_backend": dict(collision_backend),
        "checks": normalized_checks,
        "decision": {
            "passed": not failures,
            "pass_kind": (
                "authoritative_geometry_and_bound_object_with_runtime_workspace_confirmation"
                if not failures
                else "failed"
            ),
            "motion_authorized": False,
            "required_geometry_passed": not failures,
            "runtime_workspace_clear_required": True,
            "runtime_condition_id": RUNTIME_WORKSPACE_CLEAR_CONDITION_ID,
            "operator_token_can_override_required_geometry": False,
            "reasons": failures,
            "meaning": "offline pregrasp-prefix collision evidence only; never a motion command",
        },
    }
    artifact["artifact_sha256"] = _json_sha256(artifact)
    validate_pregrasp_only_audit(artifact, verify_files=True, require_pass=False)
    return artifact


def _load_scene_points(path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    with np.load(str(path), allow_pickle=False) as archive:
        key = "filtered_scene_points" if "filtered_scene_points" in archive.files else "scene_points"
        points = np.asarray(archive[key], dtype=np.float64)
        capture_q = np.asarray(archive["capture_q_rad"], dtype=np.float64)
        captured = float(np.asarray(archive["captured_at_unix_s"]).item())
    return points, capture_q, captured


def _validate_joint_plan_binding(
    binding: Mapping[str, Any],
    *,
    config_sha256: str,
    snapshot_sha256: str,
    current_q: np.ndarray,
    default_q: np.ndarray,
    pregrasp_q: np.ndarray,
    pregrasp_pose: np.ndarray,
) -> None:
    path = Path(str(binding["path"])).expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_type") != "installed_air_candidate_joint_plan"
        or manifest.get("motion_authorized") is not False
    ):
        raise ValueError("joint-plan manifest schema/type is invalid")
    integrity = manifest.get("integrity")
    unsigned = dict(manifest)
    unsigned.pop("integrity", None)
    if (
        not isinstance(integrity, dict)
        or integrity.get("algorithm") != "sha256-canonical-json-without-integrity"
        or integrity.get("payload_sha256") != _json_sha256(unsigned)
    ):
        raise ValueError("joint-plan manifest integrity failed")
    inputs = manifest["inputs"]
    if inputs["config"]["sha256"] != config_sha256:
        raise ValueError("joint-plan config hash differs from prefix audit")
    if inputs["snapshot"]["sha256"] != snapshot_sha256:
        raise ValueError("joint-plan snapshot hash differs from prefix audit")
    plan = manifest["joint_plan"]
    if not np.array_equal(np.asarray(plan["q_start_rad"], dtype=np.float64), current_q):
        raise ValueError("joint-plan start q differs from prefix current q")
    if not np.array_equal(np.asarray(plan["q_default_rad"], dtype=np.float64), default_q):
        raise ValueError("joint-plan default q differs from prefix")
    if not np.array_equal(np.asarray(plan["q_pregrasp_rad"], dtype=np.float64), pregrasp_q):
        raise ValueError("joint-plan pregrasp q differs from prefix")
    if not np.allclose(
        np.asarray(manifest["air_geometry"]["pregrasp_pose_base_EE"], dtype=np.float64),
        pregrasp_pose,
        atol=1e-10,
        rtol=0.0,
    ):
        raise ValueError("joint-plan pregrasp FK pose differs from prefix")
    fk = manifest["fk_residual"]
    ik = manifest["planner"]["ik"]
    if (
        fk.get("passed") is not True
        or float(fk["pregrasp_position_m"]) > float(ik["position_tolerance_m"])
        or float(fk["pregrasp_rotation_rad"]) > float(ik["rotation_tolerance_rad"])
        or not 0.0 < float(ik["position_tolerance_m"]) <= 1e-9
        or not 0.0 < float(ik["rotation_tolerance_rad"]) <= 1e-9
    ):
        raise ValueError("joint-plan pregrasp FK residual is not certified")


def validate_pregrasp_only_audit(
    artifact: Mapping[str, Any],
    *,
    verify_files: bool = False,
    require_pass: bool = False,
) -> Mapping[str, Any]:
    root = _expect_exact_keys(
        artifact,
        (
            "schema_version",
            "artifact_type",
            "execution_scope",
            "created_at_s",
            "motion_authorized",
            "bindings",
            "policies",
            "collision_backend",
            "checks",
            "decision",
            "artifact_sha256",
        ),
        "pregrasp audit",
    )
    if root["schema_version"] != SCHEMA_VERSION or root["artifact_type"] != ARTIFACT_TYPE:
        raise ValueError("unsupported pregrasp-only audit schema/type")
    if root["execution_scope"] != EXECUTION_SCOPE or root["motion_authorized"] is not False:
        raise ValueError("pregrasp audit execution scope/authorization was altered")
    supplied = _sha256_text(root["artifact_sha256"], "artifact_sha256")
    unsigned = dict(root)
    del unsigned["artifact_sha256"]
    if _json_sha256(unsigned) != supplied:
        raise ValueError("pregrasp audit artifact_sha256 does not match content")

    bindings = _expect_exact_keys(
        root["bindings"],
        (
            "control_profile",
            "snapshot",
            "filtered_scene",
            "object_cloud",
            "filter_evidence",
            "adapter",
            "mount_transform",
            "joint_plan_manifest",
            "joint_prefix",
            "open_hand_configuration",
        ),
        "bindings",
    )
    policies = _expect_exact_keys(
        root["policies"],
        (
            "scene_clearance_margin_m",
            "robot_clearance_margin_m",
            "object_clearance_margin_m",
            "max_scene_age_s",
            "max_joint_step_rad",
            "max_q_tracking_error_rad",
            "unknown_space_policy",
        ),
        "policies",
    )
    if policies["unknown_space_policy"] != "occupied":
        raise ValueError("pregrasp unknown-space policy must remain occupied")
    for name in (
        "scene_clearance_margin_m",
        "robot_clearance_margin_m",
        "object_clearance_margin_m",
        "max_scene_age_s",
        "max_joint_step_rad",
        "max_q_tracking_error_rad",
    ):
        value = policies[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError("policy {} must be finite and positive".format(name))
    if float(policies["max_q_tracking_error_rad"]) > 0.01:
        raise ValueError("pregrasp tracking bound exceeds 0.01rad")

    prefix = _expect_exact_keys(
        bindings["joint_prefix"],
        (
            "contract",
            "prefix_contract_sha256",
            "sampling_algorithm",
            "waypoints",
            "current_q_sha256",
            "current_q_source",
            "runtime_live_q_gate_required",
            "pregrasp_pose_base_EE",
            "max_joint_step_rad",
            "max_q_tracking_error_rad",
            "segments",
            "sample_count",
            "samples_rad",
            "samples_sha256",
        ),
        "bindings.joint_prefix",
    )
    if prefix["contract"] != "pregrasp_only_named_joint_prefix_v1":
        raise ValueError("pregrasp prefix contract is unsupported")
    if prefix["sampling_algorithm"] != CANONICAL_JOINT_SAMPLING_ALGORITHM:
        raise ValueError("pregrasp prefix sampler is not canonical")
    waypoint_values = tuple(
        (str(item["name"]), item["q_rad"]) for item in prefix["waypoints"]
    )
    waypoints = normalize_pregrasp_waypoints(waypoint_values)
    current_q = waypoints[0][1]
    if prefix["current_q_sha256"] != _array_sha256(current_q):
        raise ValueError("pregrasp prefix current-q hash is wrong")
    if prefix["current_q_source"] != "joint_plan_manifest.q_start_rad":
        raise ValueError("pregrasp prefix current-q source is not the joint plan")
    if prefix["runtime_live_q_gate_required"] is not True:
        raise ValueError("pregrasp prefix must require the runtime live-q gate")
    pose = _finite_pose(prefix["pregrasp_pose_base_EE"], "pregrasp pose")
    maximum = float(prefix["max_joint_step_rad"])
    tracking = float(prefix["max_q_tracking_error_rad"])
    if not np.isclose(maximum, float(policies["max_joint_step_rad"]), atol=1e-15, rtol=0.0):
        raise ValueError("prefix max step differs from policy")
    if not np.isclose(tracking, float(policies["max_q_tracking_error_rad"]), atol=1e-15, rtol=0.0):
        raise ValueError("prefix tracking bound differs from policy")
    samples = np.asarray(prefix["samples_rad"], dtype=np.float64)
    rebuilt, segments = build_pregrasp_joint_path(waypoints, maximum)
    if samples.shape != rebuilt.shape or not np.array_equal(samples, rebuilt):
        raise ValueError("stored prefix samples differ from canonical reconstruction")
    if prefix["sample_count"] != len(samples):
        raise ValueError("stored prefix sample_count is wrong")
    sample_hash = _array_sha256(samples)
    if prefix["samples_sha256"] != sample_hash:
        raise ValueError("stored prefix sample hash is wrong")
    if prefix["segments"] != [dict(item) for item in segments]:
        raise ValueError("stored prefix segments differ from reconstruction")
    expected_prefix_hash = pregrasp_prefix_contract_sha256(
        waypoints=waypoints,
        pregrasp_pose=pose,
        max_joint_step_rad=maximum,
        max_q_tracking_error_rad=tracking,
        samples_sha256=sample_hash,
    )
    if prefix["prefix_contract_sha256"] != expected_prefix_hash:
        raise ValueError("pregrasp prefix contract hash is wrong")

    adapter = bindings["adapter"]
    if adapter["sha256"] != V7_ADAPTER_SHA256 or adapter["required_v7_sha256"] != V7_ADAPTER_SHA256:
        raise ValueError("pregrasp audit is not bound to the commissioned V7 adapter")
    mount = bindings["mount_transform"]
    transform = _finite_pose(mount["T_EE_hand"], "bound T_EE_hand")
    # The artifact is bound bit-for-bit to the control profile by its own
    # transform digest (and, during file replay, by an exact array equality).
    # The commissioned V7 reference is independently a numerical geometry
    # constraint.  Do not require those two JSON/NumPy encodings to share the
    # same digest: ``sqrt(0.5)`` and the commissioned decimal literal differ
    # by one float64 ULP even though they describe the same rigid transform.
    if (
        mount["sha256"] != _array_sha256(transform)
        or mount["required_v7_sha256"] != V7_T_EE_HAND_SHA256
        or not np.allclose(transform, V7_T_EE_HAND, atol=1.0e-15, rtol=0.0)
    ):
        raise ValueError("pregrasp audit is not bound to commissioned T_EE_hand")

    open_config = bindings["open_hand_configuration"]
    targets = np.asarray(open_config.get("actuator_targets"), dtype=np.float64)
    if targets.shape != (6,) or not np.array_equal(targets, np.full(6, 1000.0)):
        raise ValueError("pregrasp audit requires official all-six open targets")

    checks = root["checks"]
    if not isinstance(checks, list) or [item.get("check_id") for item in checks] != list(REQUIRED_CHECK_IDS):
        raise ValueError("pregrasp check set/order is incomplete")
    reasons = []
    for item in checks:
        failure = _check_geometry_failure(item, policies, len(samples))
        expected_pass = failure is None
        if item.get("required_for_decision") is not (item["check_id"] not in SCENE_CHECK_IDS):
            raise ValueError("check required_for_decision is inconsistent")
        if item.get("advisory_only") is not (item["check_id"] in SCENE_CHECK_IDS):
            raise ValueError("check advisory_only is inconsistent")
        expected_authority = (
            "authoritative_robot_and_installed_tool_mesh_geometry"
            if item["check_id"] in ROBOT_CHECK_IDS
            else (
                "authoritative_for_exact_bound_anydex_object_point_cube_union"
                if item["check_id"] in OBJECT_CHECK_IDS
                else "advisory_filtered_scene_return_only"
            )
        )
        if item.get("authority_basis") != expected_authority:
            raise ValueError("check authority_basis is inconsistent")
        if item.get("authoritative_for_bound_object_cloud") is not (
            item["check_id"] in OBJECT_CHECK_IDS
        ):
            raise ValueError(
                "check authoritative_for_bound_object_cloud is inconsistent"
            )
        if item.get("required_geometry_passed") is not expected_pass:
            raise ValueError("check required_geometry_passed is inconsistent")
        if item.get("geometry_failure") != ("" if failure is None else failure):
            raise ValueError("check geometry_failure is inconsistent")
        if failure is not None:
            reasons.append("{}: {}".format(item["check_id"], failure))

    decision = _expect_exact_keys(
        root["decision"],
        (
            "passed",
            "pass_kind",
            "motion_authorized",
            "required_geometry_passed",
            "runtime_workspace_clear_required",
            "runtime_condition_id",
            "operator_token_can_override_required_geometry",
            "reasons",
            "meaning",
        ),
        "decision",
    )
    expected_pass = not reasons
    if (
        decision["passed"] is not expected_pass
        or decision["required_geometry_passed"] is not expected_pass
        or decision["reasons"] != reasons
        or decision["pass_kind"]
        != (
            "authoritative_geometry_and_bound_object_with_runtime_workspace_confirmation"
            if expected_pass
            else "failed"
        )
    ):
        raise ValueError("pregrasp decision is inconsistent with geometric checks")
    if (
        decision["motion_authorized"] is not False
        or decision["runtime_workspace_clear_required"] is not True
        or decision["runtime_condition_id"] != RUNTIME_WORKSPACE_CLEAR_CONDITION_ID
        or decision["operator_token_can_override_required_geometry"] is not False
        or decision["meaning"] != "offline pregrasp-prefix collision evidence only; never a motion command"
    ):
        raise ValueError("pregrasp decision safety semantics were altered")

    scene = _expect_exact_keys(
        bindings["filtered_scene"],
        (
            "path",
            "sha256",
            "points_sha256",
            "point_count",
            "capture_q_rad",
            "capture_q_sha256",
            "capture_q_role",
            "capture_to_prefix_current_linf_rad",
            "capture_matches_prefix_current",
            "captured_at_s",
            "age_at_artifact_creation_s",
            "configured_recommended_max_scene_age_s",
            "age_exceeded_recommended_max",
            "freshness_role",
            "freshness_warning",
        ),
        "bindings.filtered_scene",
    )
    captured = float(scene["captured_at_s"])
    created = float(root["created_at_s"])
    # The scene return is advisory by contract.  Its age is retained for
    # provenance but is not an execution gate; every execution needs a fresh
    # exact physical workspace-clear token and a live-q match instead.
    recorded_age = scene.get("age_at_artifact_creation_s")
    if (
        isinstance(recorded_age, bool)
        or not isinstance(recorded_age, (int, float))
        or not np.isfinite(float(recorded_age))
        or not np.isclose(float(recorded_age), created - captured, atol=1e-9, rtol=0.0)
    ):
        raise ValueError("scene age_at_artifact_creation_s is inconsistent")
    recommended = scene.get("configured_recommended_max_scene_age_s")
    if (
        isinstance(recommended, bool)
        or not isinstance(recommended, (int, float))
        or not np.isclose(
            float(recommended),
            float(policies["max_scene_age_s"]),
            atol=1e-15,
            rtol=0.0,
        )
    ):
        raise ValueError("scene recommended age differs from policy provenance")
    exceeded = float(recorded_age) > float(recommended)
    if scene.get("age_exceeded_recommended_max") is not exceeded:
        raise ValueError("scene age warning boolean is inconsistent")
    if scene.get("freshness_role") != "advisory_provenance_only":
        raise ValueError("scene freshness role is not advisory-only")
    expected_warning = (
        "filtered scene exceeded the configured recommended age at artifact "
        "creation; scene checks remain advisory and every execution requires "
        "a fresh exact workspace-clear token"
        if exceeded
        else ""
    )
    if scene.get("freshness_warning") != expected_warning:
        raise ValueError("scene freshness warning text is inconsistent")
    capture_q = _finite_q(scene["capture_q_rad"], "scene capture q")
    if scene["capture_q_sha256"] != _array_sha256(capture_q):
        raise ValueError("scene capture q hash is wrong")
    if scene["capture_q_role"] != "advisory_scene_provenance_only":
        raise ValueError("scene capture q role is not advisory-only")
    expected_linf = float(np.max(np.abs(capture_q - current_q)))
    recorded_linf = scene["capture_to_prefix_current_linf_rad"]
    if (
        isinstance(recorded_linf, bool)
        or not isinstance(recorded_linf, (int, float))
        or not np.isfinite(float(recorded_linf))
        or not np.isclose(
            float(recorded_linf), expected_linf, atol=1e-15, rtol=0.0
        )
    ):
        raise ValueError("scene-to-prefix current-q distance is inconsistent")
    expected_match = bool(np.array_equal(capture_q, current_q))
    if scene["capture_matches_prefix_current"] is not expected_match:
        raise ValueError("scene/prefix current-q match flag is inconsistent")

    if verify_files:
        for name in (
            "control_profile",
            "snapshot",
            "filtered_scene",
            "filter_evidence",
            "adapter",
            "joint_plan_manifest",
        ):
            item = bindings[name]
            source = Path(str(item["path"])).expanduser().resolve()
            if not source.is_file() or _sha256_file(source) != item["sha256"]:
                raise ValueError("{} file/hash replay failed".format(name))
        config = json.loads(Path(bindings["control_profile"]["path"]).read_text(encoding="utf-8"))
        if not np.array_equal(np.asarray(config["tool"]["T_EE_hand"], dtype=np.float64), transform):
            raise ValueError("control profile T_EE_hand differs from prefix audit")
        default_q = next(q for name, q in waypoints if name == "default")
        if not np.array_equal(np.asarray(config["franka"]["default_q_rad"], dtype=np.float64), default_q):
            raise ValueError("control profile default q differs from prefix audit")
        snapshot = load_snapshot_npz(Path(bindings["snapshot"]["path"]))
        index = int(bindings["snapshot"]["selected_candidate_index"])
        if index < 0 or index >= snapshot.grasps.count:
            raise ValueError("bound candidate index is outside snapshot")
        object_binding = bindings["object_cloud"]
        if (
            object_binding.get("source")
            != "bound official AnyDex snapshot object_points"
            or object_binding.get("points_sha256")
            != _array_sha256(snapshot.object_points)
            or object_binding.get("point_count") != len(snapshot.object_points)
        ):
            raise ValueError("bound AnyDex object cloud replay failed")
        points, replay_q, replay_time = _load_scene_points(Path(scene["path"]))
        if _array_sha256(points) != scene["points_sha256"] or len(points) != scene["point_count"]:
            raise ValueError("filtered-scene point binding replay failed")
        if not np.array_equal(replay_q, capture_q) or not np.isclose(replay_time, captured, atol=1e-9, rtol=0.0):
            raise ValueError("filtered-scene capture binding replay failed")
        evidence = json.loads(Path(bindings["filter_evidence"]["path"]).read_text(encoding="utf-8"))
        if evidence.get("motion_authorized") is not False:
            raise ValueError("filter evidence improperly authorizes motion")
        _validate_joint_plan_binding(
            bindings["joint_plan_manifest"],
            config_sha256=bindings["control_profile"]["sha256"],
            snapshot_sha256=bindings["snapshot"]["sha256"],
            current_q=waypoints[0][1],
            default_q=default_q,
            pregrasp_q=waypoints[-1][1],
            pregrasp_pose=pose,
        )
    if require_pass and not decision["passed"]:
        raise ValueError("pregrasp-only audit did not pass: {}".format(reasons))
    return root


def load_pregrasp_only_audit(
    path: Path, *, verify_files: bool = False, require_pass: bool = False
) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve()
    artifact = json.loads(source.read_text(encoding="utf-8"))
    return validate_pregrasp_only_audit(
        artifact, verify_files=verify_files, require_pass=require_pass
    )


def bind_pregrasp_only_audit(
    path: Optional[Path],
    *,
    config: Mapping[str, Any],
    config_path: Path,
    snapshot: VisualizationSnapshot,
    snapshot_path: Path,
) -> PregraspOnlyAuditBinding:
    if path is None:
        return PregraspOnlyAuditBinding(
            None, None, ("a dedicated passing pregrasp-only audit is required",)
        )
    source = Path(path).expanduser().resolve()
    try:
        artifact = load_pregrasp_only_audit(
            source, verify_files=True, require_pass=True
        )
        validate_snapshot(snapshot)
        bindings = artifact["bindings"]
        if Path(bindings["control_profile"]["path"]).resolve() != Path(config_path).resolve():
            raise ValueError("pregrasp audit control-profile path differs")
        if bindings["control_profile"]["sha256"] != _sha256_file(Path(config_path)):
            raise ValueError("pregrasp audit control-profile hash differs")
        if Path(bindings["snapshot"]["path"]).resolve() != Path(snapshot_path).resolve():
            raise ValueError("pregrasp audit snapshot path differs")
        if bindings["snapshot"]["sha256"] != _sha256_file(Path(snapshot_path)):
            raise ValueError("pregrasp audit snapshot hash differs")
        selected = int(bindings["snapshot"]["selected_candidate_index"])
        if int(snapshot.grasps.selected_index) != selected:
            raise ValueError("pregrasp audit candidate differs from selected snapshot view")
        if not np.array_equal(
            np.asarray(config["tool"]["T_EE_hand"], dtype=np.float64),
            np.asarray(bindings["mount_transform"]["T_EE_hand"], dtype=np.float64),
        ):
            raise ValueError("runtime T_EE_hand differs from pregrasp audit")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return PregraspOnlyAuditBinding(
            source, None, ("pregrasp-only audit load/replay failed: {}".format(exc),)
        )
    return PregraspOnlyAuditBinding(source, artifact, ())


__all__ = [
    "ARTIFACT_TYPE",
    "EXECUTION_SCOPE",
    "PregraspOnlyAuditBinding",
    "REQUIRED_CHECK_IDS",
    "RUNTIME_WORKSPACE_CLEAR_CONDITION_ID",
    "SCHEMA_VERSION",
    "bind_pregrasp_only_audit",
    "build_pregrasp_joint_path",
    "create_pregrasp_only_audit",
    "load_pregrasp_only_audit",
    "normalize_pregrasp_waypoints",
    "pregrasp_prefix_contract_sha256",
    "validate_pregrasp_only_audit",
]
