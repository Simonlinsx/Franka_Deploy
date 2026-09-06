#!/usr/bin/env python3
"""Execute one fresh, collision-audited, empty-flange FR3 preview plan.

This utility is intentionally arm-only.  It has no hand transport, hand
driver, serial-port option, or hand-module import.  Motion is available only
for a planner artifact linked byte-for-byte to a recent live-scene NPZ.  The
live robot must still be at the bare-flange default/capture configuration.

The successful sequence is deliberately reversible::

    default -> pregrasp -> grasp -> pregrasp -> default

It never leaves the empty flange parked in the diagnostic grasp pose.  A
failure stops control in place; it does not attempt an unaudited recovery
motion.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


FLANGE_TOKEN = "FR3_FLANGE_EMPTY"
CLEAR_TOKEN = "FR3_WORKSPACE_CLEAR"
STOP_TOKEN = "FR3_STOP_READY"
SCENE_TOKEN = "FR3_LIVE_SCENE_UNCHANGED"
PREVIEW_TOKEN = "FR3_EMPTY_FLANGE_DIAGNOSTIC_PREVIEW"
SAVED_POSE_TOKEN = "FR3_SAVED_POSE_ONLY"

ARTIFACT_TYPE = "empty_flange_diagnostic_grasp_preview"
LIVE_SCENE_SCHEMA = "dexgrasp_live_scene"
LIVE_SCENE_SCHEMA_VERSION = 1
MAX_CAPTURE_AGE_S = 120.0
MAX_CLOCK_SKEW_S = 5.0
MAX_CAPTURE_TO_AUDIT_S = 120.0
MAX_Q_FROM_CAPTURE_RAD = 0.015
MAX_CAPTURE_Q_FROM_DEFAULT_RAD = 0.015
MIN_SCENE_POINTS = 100
MIN_SCENE_MARGIN_M = 0.005
MIN_PATH_SAMPLES = 41
MAX_JSON_BYTES = 2_000_000

DEFAULT_Q = np.asarray(
    [0.0, 0.0, 0.0, -np.pi / 2.0, 0.0, np.pi / 2.0, 0.0],
    dtype=np.float64,
)
COMMISSIONING_JOINT_LIMITS = np.asarray(
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
class PreviewPlan:
    """Validated motion targets and immutable live-scene provenance."""

    artifact_path: Path
    scene_path: Path
    scene_sha256: str
    camera_serial: str
    calibration_id: str
    capture_completed_s: float
    audited_at_s: float
    q_start_rad: np.ndarray
    q_capture_rad: np.ndarray
    q_pregrasp_rad: np.ndarray
    q_grasp_rad: np.ndarray
    scene_margin_m: float
    minimum_clearance_m: float
    saved_pose_only: bool


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("{} must be a JSON object".format(name))
    return value


def _require_true(value: Any, name: str) -> None:
    if value is not True:
        raise ValueError("{} must be exactly true".format(name))


def _require_false(value: Any, name: str) -> None:
    if value is not False:
        raise ValueError("{} must be exactly false".format(name))


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("{} must be a finite number".format(name))
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be a finite number".format(name)) from exc
    if not np.isfinite(number):
        raise ValueError("{} must be a finite number".format(name))
    return number


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("{} must be a positive integer".format(name))
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be a positive integer".format(name)) from exc
    if integer <= 0 or float(value) != float(integer):
        raise ValueError("{} must be a positive integer".format(name))
    return integer


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string".format(name))
    return value.strip()


def _joint_vector(value: Any, name: str) -> np.ndarray:
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        raise ValueError("{} must be a finite seven-vector".format(name))
    lower = COMMISSIONING_JOINT_LIMITS[:, 0] + 0.06
    upper = COMMISSIONING_JOINT_LIMITS[:, 1] - 0.06
    if np.any(q < lower) or np.any(q > upper):
        raise ValueError("{} violates the commissioned FR3 joint margin".format(name))
    result = q.copy()
    result.setflags(write=False)
    return result


def _rigid_transform(value: Any, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("{} must be a finite 4x4 transform".format(name))
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8, rtol=0.0):
        raise ValueError("{} has an invalid homogeneous row".format(name))
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0):
        raise ValueError("{} rotation is not orthonormal".format(name))
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("{} rotation is not right-handed".format(name))
    return transform.copy()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _npz_scalar(values: Mapping[str, np.ndarray], key: str) -> Any:
    if key not in values:
        raise ValueError("live scene is missing {}".format(key))
    raw = np.asarray(values[key])
    if raw.shape != ():
        raise ValueError("live scene {} must be scalar".format(key))
    value = raw.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
    return value


def _load_live_scene(path: Path) -> Mapping[str, np.ndarray]:
    try:
        with np.load(str(path), allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise ValueError("cannot load live-scene NPZ {}: {}".format(path, exc)) from exc


def _resolve_scene_path(artifact_path: Path, raw_path: Any) -> Path:
    text = _nonempty_text(raw_path, "collision_scene_input.scene_path")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = artifact_path.parent / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError("live-scene NPZ is unavailable: {}".format(candidate)) from exc
    if not resolved.is_file():
        raise ValueError("live-scene path is not a regular file: {}".format(resolved))
    return resolved


def _require_matching_text(
    plan_value: Any, scene_value: Any, name: str
) -> str:
    planned = _nonempty_text(plan_value, "collision_scene_input." + name)
    captured = _nonempty_text(scene_value, "live scene " + name)
    if planned != captured:
        raise ValueError("{} mismatch between plan and live scene".format(name))
    return planned


def _validate_freshness(
    capture_completed_s: float, audited_at_s: float, now_s: float
) -> None:
    if capture_completed_s > now_s + MAX_CLOCK_SKEW_S:
        raise ValueError("live-scene capture timestamp is in the future")
    if audited_at_s > now_s + MAX_CLOCK_SKEW_S:
        raise ValueError("collision-audit timestamp is in the future")
    capture_age = now_s - capture_completed_s
    audit_age = now_s - audited_at_s
    if capture_age > MAX_CAPTURE_AGE_S:
        raise ValueError(
            "live-scene capture is stale ({:.1f}s > {:.1f}s)".format(
                capture_age, MAX_CAPTURE_AGE_S
            )
        )
    if audit_age > MAX_CAPTURE_AGE_S:
        raise ValueError(
            "collision audit is stale ({:.1f}s > {:.1f}s)".format(
                audit_age, MAX_CAPTURE_AGE_S
            )
        )
    if audited_at_s + MAX_CLOCK_SKEW_S < capture_completed_s:
        raise ValueError("collision audit predates the live-scene capture")
    if audited_at_s - capture_completed_s > MAX_CAPTURE_TO_AUDIT_S:
        raise ValueError("collision audit was not produced promptly after capture")


def load_preview_plan(
    artifact_path: Path, *, now_s: Optional[float] = None
) -> PreviewPlan:
    """Load and independently validate one executable diagnostic artifact."""

    path = artifact_path.expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("preview artifact is not a regular file: {}".format(path))
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("preview artifact is unexpectedly large")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot load preview artifact {}: {}".format(path, exc)) from exc
    root = _require_mapping(document, "preview artifact")

    if root.get("schema_version") != 2:
        raise ValueError("preview artifact schema_version must be 2")
    if root.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("unexpected preview artifact_type")
    _require_true(root.get("diagnostic_only"), "diagnostic_only")
    _require_false(
        root.get("installed_mount_calibration"), "installed_mount_calibration"
    )
    saved_pose_only = root.get("saved_pose_only") is True

    virtual_tcp = _require_mapping(root.get("virtual_tcp"), "virtual_tcp")
    never_copy_to_installed = virtual_tcp.get("must_not_copy_to_installed_profile")
    if never_copy_to_installed is None:
        # Backward compatibility for already-saved schema-v2 diagnostic artifacts.
        never_copy_to_installed = virtual_tcp.get(
            "must_not_copy_to_installed_V2_profile"
        )
    _require_true(
        never_copy_to_installed,
        "virtual_tcp.must_not_copy_to_installed_profile",
    )
    tcp_offset = _finite_float(virtual_tcp.get("F_T_TCP_z_m"), "virtual TCP offset")
    if not 0.0 <= tcp_offset <= 0.10:
        raise ValueError("virtual TCP offset is outside the diagnostic range")

    q_start = _joint_vector(root.get("q_start_rad"), "q_start_rad")
    if not np.allclose(q_start, DEFAULT_Q, atol=1e-8, rtol=0.0):
        raise ValueError("q_start_rad is not the reviewed bare-flange default")
    q_pregrasp = _joint_vector(root.get("q_pregrasp_rad"), "q_pregrasp_rad")
    q_grasp = _joint_vector(root.get("q_grasp_rad"), "q_grasp_rad")
    _rigid_transform(
        root.get("T_robot_base_flange_pregrasp"),
        "T_robot_base_flange_pregrasp",
    )
    _rigid_transform(
        root.get("T_robot_base_flange_grasp"), "T_robot_base_flange_grasp"
    )

    scene_input = _require_mapping(
        root.get("collision_scene_input"), "collision_scene_input"
    )
    if scene_input.get("source_kind") != "live_scene_npz":
        raise ValueError("collision scene must come from a live_scene_npz")
    if scene_input.get("reference_frame") != "robot_base":
        raise ValueError("collision scene reference_frame must be robot_base")
    if scene_input.get("source_schema_name") != LIVE_SCENE_SCHEMA:
        raise ValueError("collision scene source_schema_name is invalid")
    if scene_input.get("source_schema_version") != LIVE_SCENE_SCHEMA_VERSION:
        raise ValueError("collision scene source_schema_version is invalid")
    _require_true(
        scene_input.get("saved_grasp_pose_retained"),
        "collision_scene_input.saved_grasp_pose_retained",
    )
    _require_true(
        scene_input.get("snapshot_scene_ignored"),
        "collision_scene_input.snapshot_scene_ignored",
    )
    if bool(scene_input.get("saved_pose_only")) != saved_pose_only:
        raise ValueError("saved_pose_only mismatch in collision-scene metadata")

    scene_path = _resolve_scene_path(path, scene_input.get("scene_path"))
    expected_sha = _nonempty_text(
        scene_input.get("scene_sha256"), "collision_scene_input.scene_sha256"
    ).lower()
    if len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
        raise ValueError("collision_scene_input.scene_sha256 is malformed")
    actual_sha = _sha256_file(scene_path)
    if actual_sha != expected_sha:
        raise ValueError("live-scene NPZ changed after collision planning")

    scene_values = _load_live_scene(scene_path)
    if _npz_scalar(scene_values, "schema_name") != LIVE_SCENE_SCHEMA:
        raise ValueError("live-scene schema_name is not dexgrasp_live_scene")
    if int(_npz_scalar(scene_values, "schema_version")) != LIVE_SCENE_SCHEMA_VERSION:
        raise ValueError("unsupported live-scene schema_version")
    if _npz_scalar(scene_values, "reference_frame") != "robot_base":
        raise ValueError("live-scene reference_frame must be robot_base")
    camera_serial = _require_matching_text(
        scene_input.get("camera_serial"),
        _npz_scalar(scene_values, "camera_serial"),
        "camera_serial",
    )
    calibration_id = _require_matching_text(
        scene_input.get("calibration_id"),
        _npz_scalar(scene_values, "calibration_id"),
        "calibration_id",
    )
    scene_points = np.asarray(scene_values.get("scene_points"), dtype=np.float64)
    if (
        scene_points.ndim != 2
        or scene_points.shape[1:] != (3,)
        or len(scene_points) < MIN_SCENE_POINTS
        or not np.all(np.isfinite(scene_points))
    ):
        raise ValueError(
            "live scene must contain at least {} finite 3D points".format(
                MIN_SCENE_POINTS
            )
        )
    planned_count = _positive_int(
        scene_input.get("scene_points_raw"),
        "collision_scene_input.scene_points_raw",
    )
    if planned_count != len(scene_points):
        raise ValueError("live-scene point count differs from collision plan")

    capture_q = _joint_vector(scene_values.get("capture_q_rad"), "capture_q_rad")
    capture_q_source = _nonempty_text(
        _npz_scalar(scene_values, "capture_q_source"), "capture_q_source"
    )
    if capture_q_source != "cli_asserted":
        raise ValueError("live scene capture_q_source must be cli_asserted")
    planned_capture_q = _joint_vector(
        scene_input.get("capture_q_rad"), "collision_scene_input.capture_q_rad"
    )
    if not np.allclose(capture_q, planned_capture_q, atol=1e-8, rtol=0.0):
        raise ValueError("capture_q_rad mismatch between plan and live scene")
    if scene_input.get("capture_q_source") != capture_q_source:
        raise ValueError("capture_q_source mismatch between plan and live scene")
    if scene_input.get("capture_q_value_source") != "live_scene_npz:capture_q_rad":
        raise ValueError("planner did not source capture q from the live-scene NPZ")
    filter_q = _joint_vector(
        scene_input.get("robot_return_filter_q_rad"),
        "collision_scene_input.robot_return_filter_q_rad",
    )
    if not np.allclose(filter_q, capture_q, atol=1e-8, rtol=0.0):
        raise ValueError("robot-return filter q does not equal live capture q")
    if scene_input.get("robot_return_filter_q_source") != "live_scene_npz:capture_q_rad":
        raise ValueError("planner did not use live-scene capture_q_rad for filtering")
    capture_default_error = float(np.max(np.abs(capture_q - q_start)))
    if capture_default_error > MAX_CAPTURE_Q_FROM_DEFAULT_RAD:
        raise ValueError(
            "live scene was not captured at the reviewed default q: {:.6f}rad".format(
                capture_default_error
            )
        )

    capture_completed_s = _finite_float(
        _npz_scalar(scene_values, "capture_completed_s"), "capture_completed_s"
    )
    planned_capture_completed_s = _finite_float(
        scene_input.get("capture_completed_s"),
        "collision_scene_input.capture_completed_s",
    )
    if not np.isclose(
        capture_completed_s, planned_capture_completed_s, atol=1e-6, rtol=0.0
    ):
        raise ValueError("capture_completed_s mismatch between plan and live scene")

    audit = _require_mapping(root.get("collision_audit"), "collision_audit")
    _require_true(audit.get("self_collision_free"), "collision_audit.self_collision_free")
    _require_true(audit.get("scene_capsule_clear"), "collision_audit.scene_capsule_clear")
    _require_true(
        audit.get("live_scene_revalidated"),
        "collision_audit.live_scene_revalidated",
    )
    alignment_valid = audit.get("object_alignment_valid") is True
    stale_pose_preview = audit.get("stale_object_pose_preview") is True
    scene_stale_override = scene_input.get("stale_object_pose_override") is True
    alignment = _require_mapping(
        audit.get("live_object_alignment"),
        "collision_audit.live_object_alignment",
    )
    median_limit = _finite_float(
        alignment.get("median_max_m"), "live-object median limit"
    )
    p95_limit = _finite_float(
        alignment.get("p95_max_m"), "live-object p95 limit"
    )
    coverage_distance = _finite_float(
        alignment.get("coverage_distance_m"), "live-object coverage distance"
    )
    minimum_coverage = _finite_float(
        alignment.get("minimum_coverage"), "live-object minimum coverage"
    )
    if (
        median_limit > 0.008
        or p95_limit > 0.015
        or coverage_distance > 0.015
        or minimum_coverage < 0.80
    ):
        raise ValueError("live-object alignment thresholds are weaker than reviewed")
    if saved_pose_only:
        if alignment_valid or not stale_pose_preview or not scene_stale_override:
            raise ValueError("saved-pose-only artifact lacks the stale-object preview gate")
        if alignment.get("passed") is not False:
            raise ValueError("saved-pose-only artifact must record failed object alignment")
    else:
        if not alignment_valid or stale_pose_preview or scene_stale_override:
            raise ValueError("object alignment is not valid for a grasp preview")
        if alignment.get("passed") is not True:
            raise ValueError("live object alignment did not pass")
    _require_false(
        audit.get("authoritative_for_installed_tool"),
        "collision_audit.authoritative_for_installed_tool",
    )
    audit_sha = _nonempty_text(
        audit.get("scene_sha256"), "collision_audit.scene_sha256"
    ).lower()
    if audit_sha != expected_sha:
        raise ValueError("collision audit is not linked to the live-scene NPZ")
    audit_q = _joint_vector(
        audit.get("robot_return_filter_q_rad"),
        "collision_audit.robot_return_filter_q_rad",
    )
    if not np.allclose(audit_q, capture_q, atol=1e-8, rtol=0.0):
        raise ValueError("collision-audit filter q differs from capture q")
    if audit.get("robot_return_filter_q_source") != "live_scene_npz:capture_q_rad":
        raise ValueError("collision audit did not use live-scene capture_q_rad")
    margin = _finite_float(audit.get("scene_margin_m"), "scene margin")
    minimum = _finite_float(
        audit.get("minimum_sampled_signed_distance_m"), "minimum scene clearance"
    )
    if margin < MIN_SCENE_MARGIN_M:
        raise ValueError(
            "collision-audit margin {:.6f}m is below {:.6f}m".format(
                margin, MIN_SCENE_MARGIN_M
            )
        )
    if minimum <= margin:
        raise ValueError("minimum sampled scene clearance does not exceed margin")
    if _positive_int(audit.get("path_samples"), "collision_audit.path_samples") < MIN_PATH_SAMPLES:
        raise ValueError("collision audit used too few path samples")
    audited_at_s = _finite_float(audit.get("audited_at_s"), "audited_at_s")

    current_time = float(time.time() if now_s is None else now_s)
    if not np.isfinite(current_time):
        raise ValueError("current wall-clock time is non-finite")
    _validate_freshness(capture_completed_s, audited_at_s, current_time)

    return PreviewPlan(
        artifact_path=path,
        scene_path=scene_path,
        scene_sha256=expected_sha,
        camera_serial=camera_serial,
        calibration_id=calibration_id,
        capture_completed_s=capture_completed_s,
        audited_at_s=audited_at_s,
        q_start_rad=q_start,
        q_capture_rad=capture_q,
        q_pregrasp_rad=q_pregrasp,
        q_grasp_rad=q_grasp,
        scene_margin_m=margin,
        minimum_clearance_m=minimum,
        saved_pose_only=saved_pose_only,
    )


def _require_confirmations(args: argparse.Namespace, plan: PreviewPlan) -> None:
    expected = [
        ("--confirm-flange-empty", args.confirm_flange_empty, FLANGE_TOKEN),
        ("--confirm-workspace-clear", args.confirm_workspace_clear, CLEAR_TOKEN),
        ("--confirm-stop-ready", args.confirm_stop_ready, STOP_TOKEN),
        ("--confirm-live-scene-unchanged", args.confirm_live_scene_unchanged, SCENE_TOKEN),
        ("--confirm-diagnostic-preview", args.confirm_diagnostic_preview, PREVIEW_TOKEN),
    ]
    if plan.saved_pose_only:
        expected.append(
            ("--confirm-saved-pose-only", args.confirm_saved_pose_only, SAVED_POSE_TOKEN)
        )
    missing = [
        "{} {}".format(flag, token)
        for flag, actual, token in expected
        if actual != token
    ]
    if missing:
        raise ValueError("exact confirmations required: " + "; ".join(missing))


def _connect_bare_franka(robot_ip: str):
    # Lazy import: offline validation and missing confirmations cannot import
    # pylibfranka or open an FCI connection.
    from anydex_pipeline.franka_sequence_driver import (
        FrankaMotionLimits,
        FrankaSequenceDriver,
    )

    limits = FrankaMotionLimits(
        expected_F_T_EE=np.eye(4),
        expected_m_ee_kg=0.0,
        expected_F_x_Cee_m=np.zeros(3),
        expected_I_ee_kg_m2=np.zeros((3, 3)),
        joint_limits_rad=COMMISSIONING_JOINT_LIMITS,
        joint_limit_margin_rad=0.06,
        max_joint_speed_rad_s=0.04,
        max_joint_segment_rad=0.12,
        min_joint_duration_s=5.0,
        joint_arrival_tolerance_rad=0.015,
    )
    return FrankaSequenceDriver.connect(
        str(robot_ip), limits, enforce_realtime=False
    )


def _state_q(state: Any) -> np.ndarray:
    q = np.asarray(getattr(state, "q", None), dtype=np.float64)
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        raise RuntimeError("Franka state.q is missing or malformed")
    return q.copy()


def execute_preview(
    plan: PreviewPlan,
    robot_ip: str,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Run the reversible arm-only sequence after a final live-state gate."""

    _validate_freshness(
        plan.capture_completed_s, plan.audited_at_s, float(time.time())
    )
    if _sha256_file(plan.scene_path) != plan.scene_sha256:
        raise RuntimeError("live-scene NPZ changed after offline validation")
    arm = _connect_bare_franka(str(robot_ip))
    failure: Optional[BaseException] = None
    try:
        initial = arm.robot.read_once()
        # This gate verifies Idle/no-errors/no-contact, F_T_EE=identity and all
        # three configured EE dynamics values are exactly the bare zero model.
        arm._validate_state(initial, require_idle=True, enforce_success=False)
        initial_q = _state_q(initial)
        capture_error = float(np.max(np.abs(initial_q - plan.q_capture_rad)))
        default_error = float(np.max(np.abs(initial_q - plan.q_start_rad)))
        if capture_error > MAX_Q_FROM_CAPTURE_RAD:
            raise RuntimeError(
                "live q moved away from the collision-scene capture/default q: "
                "max_error={:.6f}rad limit={:.6f}rad".format(
                    capture_error, MAX_Q_FROM_CAPTURE_RAD
                )
            )
        if default_error > MAX_Q_FROM_CAPTURE_RAD:
            raise RuntimeError(
                "live q is not at the reviewed bare-flange default: "
                "max_error={:.6f}rad limit={:.6f}rad".format(
                    default_error, MAX_Q_FROM_CAPTURE_RAD
                )
            )

        print(
            "[Franka empty-flange preview] LIVE AUDIT ACCEPTED "
            "camera={} calibration={} age={:.1f}s clearance={:.4f}m".format(
                plan.camera_serial,
                plan.calibration_id,
                max(0.0, time.time() - plan.capture_completed_s),
                plan.minimum_clearance_m,
            )
        )
        print(
            "[Franka empty-flange preview] max_speed=0.04rad/s; "
            "sequence=default->pregrasp->grasp->pregrasp->default"
        )

        stages = (
            ("default", plan.q_start_rad),
            ("pregrasp", plan.q_pregrasp_rad),
            ("grasp", plan.q_grasp_rad),
        )
        for name, target in stages:
            print("[Franka empty-flange preview] moving to {}".format(name))
            arm.move_joints(target)
        print("[Franka empty-flange preview] grasp reached; pausing 1.0s")
        sleep(1.0)
        for name, target in (
            ("pregrasp (retreat)", plan.q_pregrasp_rad),
            ("default (return)", plan.q_start_rad),
        ):
            print("[Franka empty-flange preview] moving to {}".format(name))
            arm.move_joints(target)
        final = arm.robot.read_once()
        arm._validate_state(final, require_idle=True, enforce_success=False)
        final_error = float(np.max(np.abs(_state_q(final) - plan.q_start_rad)))
        if final_error > arm.limits.joint_arrival_tolerance_rad:
            raise RuntimeError(
                "return-to-default error {:.6f}rad exceeds {:.6f}rad".format(
                    final_error, arm.limits.joint_arrival_tolerance_rad
                )
            )
        print(
            "[Franka empty-flange preview] complete; returned to default "
            "(max_error={:.6f}rad)".format(final_error)
        )
    except BaseException as exc:
        failure = exc
    finally:
        try:
            arm.stop()
        except BaseException as stop_exc:
            if failure is None:
                failure = RuntimeError("Franka stop failed: {}".format(stop_exc))
            else:
                failure = RuntimeError(
                    "{}; STOP UNCONFIRMED: {}".format(failure, stop_exc)
                )
    if failure is not None:
        raise failure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute a fresh live-scene-audited grasp pose on a bare FR3 flange. "
            "The arm always retreats and returns to default; no hand is connected."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate artifact/provenance/freshness without importing pylibfranka",
    )
    parser.add_argument("--confirm-flange-empty", metavar=FLANGE_TOKEN)
    parser.add_argument("--confirm-workspace-clear", metavar=CLEAR_TOKEN)
    parser.add_argument("--confirm-stop-ready", metavar=STOP_TOKEN)
    parser.add_argument("--confirm-live-scene-unchanged", metavar=SCENE_TOKEN)
    parser.add_argument("--confirm-diagnostic-preview", metavar=PREVIEW_TOKEN)
    parser.add_argument("--confirm-saved-pose-only", metavar=SAVED_POSE_TOKEN)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = load_preview_plan(args.plan)
        if args.validate_only:
            print(
                "[Franka empty-flange preview] VALIDATED ONLY; no hardware connected; "
                "scene={} sha256={}".format(plan.scene_path, plan.scene_sha256)
            )
            return 0
        _require_confirmations(args, plan)
        execute_preview(plan, str(args.robot_ip))
    except KeyboardInterrupt:
        print(
            "[Franka empty-flange preview] interrupted; use the physical stop if "
            "motion remains",
            file=sys.stderr,
        )
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print("[Franka empty-flange preview] failed: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
