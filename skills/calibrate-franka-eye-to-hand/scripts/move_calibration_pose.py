#!/usr/bin/env python3
"""Preview or execute exactly one reviewed eye-to-hand calibration pose.

``preview`` is deliberately hardware-free: this module does not import
``pylibfranka`` or the Franka sequence driver at import time.  ``run`` checks
the immutable plan bytes, explicit single-pose authorization, and six exact
operator confirmations before the lazy driver import and FCI connection.

This program never opens a camera and never captures a calibration sample.
Its only successful motion result is a stationary ``capture_ready`` receipt
for a separate collector process.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from types import TracebackType
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml


PLAN_KIND = "franka_eye_to_hand_commissioning_plan"
PLAN_SCHEMA_VERSION = 1
PREVIEW_STATUSES = frozenset(
    {
        "awaiting_explicit_motion_authorization",
        "explicit_single_pose_motion_authorized",
    }
)
RUN_STATUS = "explicit_single_pose_motion_authorized"
AUTHORIZATION_SCOPE = "single_pose"

MAX_PLAN_BYTES = 2_000_000
MAX_TRANSLATION_NORM_M = 0.050
MAX_ROTATION_ANGLE_DEG = 16.0
START_TRANSLATION_TOLERANCE_M = 0.005
START_ROTATION_TOLERANCE_RAD = math.radians(1.0)
NUMERICAL_TOLERANCE = 1.0e-9

E_STOP_TOKEN = "FR3_ESTOP_REACHABLE"
SWEPT_VOLUME_TOKEN = "FR3_CALIBRATION_SWEPT_VOLUME_CLEAR"
TARGET_RIGID_TOKEN = "FR3_CALIBRATION_TARGET_RIGID"
CAMERA_FIXED_TOKEN = "EYE_TO_HAND_CAMERA_FIXED"

MAX_CARTESIAN_SPEED_M_S = 0.005
MAX_ANGULAR_SPEED_RAD_S = 0.020
MAX_SEGMENT_TRANSLATION_M = 0.020
MAX_SEGMENT_ROTATION_RAD = math.radians(5.0)
MIN_CARTESIAN_DURATION_S = 4.0
CARTESIAN_ENDPOINT_HOLD_S = 6.0
CARTESIAN_POSE_CONTROLLER_MODE = "joint_impedance"
DIAGNOSTIC_HOLD_DURATION_S = 10.0
MINIMUM_JERK_PEAK_VELOCITY_FACTOR = 1.875
MINIMUM_JERK_PEAK_ACCELERATION_FACTOR = 10.0 * math.sqrt(3.0) / 3.0

_POSE_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")

# A Cartesian commissioning plan must never be repurposed for a singularity
# escape.  Joint recovery has its own immutable schema and deliberately tighter
# envelope.  These are wrapper caps, not defaults that a plan may enlarge.
JOINT_RECOVERY_PLAN_KIND = "franka_eye_to_hand_joint_recovery_plan"
JOINT_RECOVERY_RUN_STATUS = "explicit_bounded_joint_recovery_authorized"
JOINT_RECOVERY_AUTHORIZATION_SCOPE = "bounded_joint_recovery"
JOINT_RECOVERY_PREVIEW_STATUSES = frozenset(
    {
        "awaiting_explicit_motion_authorization",
        JOINT_RECOVERY_RUN_STATUS,
    }
)
MAX_SINGLE_JOINT_DELTA_RAD = math.radians(5.0)
MAX_JOINT_DELTA_NORM_RAD = math.radians(9.0)
MIN_JOINT_DELTA_NORM_RAD = math.radians(0.1)
MAX_JOINT_RECOVERY_SPEED_RAD_S = 0.020
MAX_JOINT_RECOVERY_ACCELERATION_RAD_S2 = 0.010
MIN_JOINT_RECOVERY_DURATION_S = 8.0
START_Q_TOLERANCE_RAD = 0.001
MAX_START_DQ_RAD_S = 0.005
JOINT_ENDPOINT_TOLERANCE_RAD = 0.003
START_POSE_TRANSLATION_TOLERANCE_M = 0.003
START_POSE_ROTATION_TOLERANCE_RAD = math.radians(0.5)
ENDPOINT_POSE_TRANSLATION_TOLERANCE_M = 0.010
ENDPOINT_POSE_ROTATION_TOLERANCE_RAD = math.radians(1.0)
ENDPOINT_MAX_DQ_RAD_S = 0.005
ENDPOINT_CONSECUTIVE_SAMPLES = 10
ENDPOINT_POLL_S = 0.050
MAX_PREDICTED_EEF_TRANSLATION_M = 0.050
MAX_CONSERVATIVE_BOARD_SWEEP_M = 0.070
JOINT_RECOVERY_MAX_DYNAMIC_SEGMENTS = 1
JOINT_RECOVERY_TIME_LAW = "cosine_zero_endpoint_velocity"
JOINT_RECOVERY_SWEEP_TOKEN = "FR3_BOUNDED_JOINT_RECOVERY_SWEEP_CLEAR"

_FR3_JOINT_LIMITS_RAD = np.asarray(
    [
        [-2.9007, 2.9007],
        [-1.8361, 1.8361],
        [-2.9007, 2.9007],
        [-3.0770, -0.1169],
        [-2.8763, 2.8763],
        [0.4398, 4.6216],
        [-3.0508, 3.0508],
    ],
    dtype=np.float64,
)
JOINT_LIMIT_MARGIN_RAD = 0.05


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> Mapping[str, Any]:
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found duplicate key {!r}".format(key),
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


@dataclass(frozen=True)
class EndEffectorDynamics:
    expected_F_T_EE: np.ndarray
    expected_m_ee_kg: float
    expected_F_x_Cee_m: np.ndarray
    expected_I_ee_kg_m2: np.ndarray


@dataclass(frozen=True)
class PosePreview:
    plan_path: Path
    plan_sha256: str
    plan_status: str
    session_slug: str
    pose_set: str
    pose_id: str
    start_pose_id: Optional[str]
    T_anchor: np.ndarray
    xyz_offset_base_m: np.ndarray
    rotation_vector_eef_rad: np.ndarray
    T_target: np.ndarray
    T_start: Optional[np.ndarray]
    workspace_min_m: np.ndarray
    workspace_max_m: np.ndarray
    minimum_eef_z_m: float
    translation_norm_m: float
    rotation_angle_rad: float
    settle_time_s: float
    motion_authorized: bool
    run_authorized: bool
    authorization_error: Optional[str]
    dynamics: Optional[EndEffectorDynamics]
    dynamics_error: Optional[str]


@dataclass(frozen=True)
class MotionResult:
    plan_sha256: str
    pose_id: str
    T_target: np.ndarray
    capture_ready: bool
    camera_capture_performed: bool = False
    control_loop_telemetry: Optional[Mapping[str, Any]] = None
    diagnostic_hold_completed: bool = False


@dataclass(frozen=True)
class _ValidatedPlanPose:
    xyz_offset_base_m: np.ndarray
    rotation_vector_eef_rad: np.ndarray
    T_target: np.ndarray
    translation_norm_m: float
    rotation_angle_rad: float


@dataclass(frozen=True)
class JointRecoveryPreview:
    """Fully offline, immutable preview of one bounded joint recovery."""

    plan_path: Path
    plan_sha256: str
    plan_status: str
    session_slug: str
    recovery_id: str
    robot_ip: str
    expected_start_q_rad: np.ndarray
    planned_target_q_rad: np.ndarray
    commanded_delta_rad: np.ndarray
    maximum_live_to_target_delta_rad: float
    expected_start_T_base_ee: np.ndarray
    expected_target_T_base_ee: np.ndarray
    workspace_min_m: np.ndarray
    workspace_max_m: np.ndarray
    minimum_eef_z_m: float
    board_max_extent_from_eef_m: float
    conservative_board_sweep_displacement_m: float
    predicted_eef_translation_m: float
    predicted_eef_rotation_rad: float
    maximum_joint_velocity_rad_s: float
    maximum_joint_acceleration_rad_s2: float
    minimum_joint_duration_s: float
    expected_peak_joint_velocity_rad_s: float
    expected_peak_joint_acceleration_rad_s2: float
    motion_authorized: bool
    run_authorized: bool
    authorization_error: Optional[str]
    authorization_receipt_path: Path
    authorization_receipt_exists: bool
    dynamics: Optional[EndEffectorDynamics]
    dynamics_error: Optional[str]


@dataclass(frozen=True)
class JointRecoveryResult:
    plan_sha256: str
    recovery_id: str
    effective_target_q_rad: np.ndarray
    endpoint_q_rad: np.ndarray
    endpoint_T_base_ee: np.ndarray
    authorization_receipt_path: Path
    control_loop_telemetry: Optional[Mapping[str, Any]] = None


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("{} must be a mapping".format(name))
    return value


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string".format(name))
    return value.strip()


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


def _finite_vector(value: Any, length: int, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be a finite {}-vector".format(name, length)) from exc
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError("{} must be a finite {}-vector".format(name, length))
    result = vector.copy()
    result.setflags(write=False)
    return result


def _rigid_transform(value: Any, name: str) -> np.ndarray:
    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be a finite rigid 4x4 transform".format(name)) from exc
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("{} must be a finite rigid 4x4 transform".format(name))
    if not np.allclose(
        transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-8, rtol=0.0
    ):
        raise ValueError("{} has an invalid homogeneous row".format(name))
    rotation = transform[:3, :3]
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=1.0e-6, rtol=0.0
    ):
        raise ValueError("{} rotation is not orthonormal".format(name))
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6, rtol=0.0):
        raise ValueError("{} rotation determinant is not +1".format(name))
    result = transform.copy()
    result.setflags(write=False)
    return result


def _so3_exp(rotation_vector_rad: np.ndarray) -> np.ndarray:
    vector = _finite_vector(rotation_vector_rad, 3, "rotation vector")
    theta = float(np.linalg.norm(vector))
    if theta < 1.0e-12:
        skew = np.asarray(
            [
                [0.0, -vector[2], vector[1]],
                [vector[2], 0.0, -vector[0]],
                [-vector[1], vector[0], 0.0],
            ],
            dtype=np.float64,
        )
        return np.eye(3) + skew
    axis = vector / theta
    skew = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (
        skew @ skew
    )


def _rotation_error_rad(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.acos(cosine))


def build_target_transform(
    T_anchor: np.ndarray,
    xyz_offset_base_m: np.ndarray,
    rotation_vector_eef_rad: np.ndarray,
) -> np.ndarray:
    """Apply base-frame translation and right-multiplied EEF-frame rotation."""

    anchor = _rigid_transform(T_anchor, "T_anchor")
    offset = _finite_vector(xyz_offset_base_m, 3, "xyz_offset_base_m")
    rotation_vector = _finite_vector(
        rotation_vector_eef_rad, 3, "rotation_vector_eef_rad"
    )
    target = anchor.copy()
    target[:3, 3] = anchor[:3, 3] + offset
    target[:3, :3] = anchor[:3, :3] @ _so3_exp(rotation_vector)
    return _rigid_transform(target, "T_target")


def _read_plan(path: Path) -> Tuple[Path, str, Mapping[str, Any]]:
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError("session plan is unavailable: {}".format(candidate)) from exc
    if not resolved.is_file():
        raise ValueError("session plan is not a regular file: {}".format(resolved))
    try:
        raw = resolved.read_bytes()
    except OSError as exc:
        raise ValueError("cannot read session plan {}: {}".format(resolved, exc)) from exc
    if not raw or len(raw) > MAX_PLAN_BYTES:
        raise ValueError(
            "session plan size must be in [1, {}] bytes".format(MAX_PLAN_BYTES)
        )
    digest = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("session plan must be UTF-8") from exc
    try:
        payload = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError("invalid session plan YAML: {}".format(exc)) from exc
    return resolved, digest, _require_mapping(payload, "session plan")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise RuntimeError("cannot re-read session plan: {}".format(exc)) from exc
    return digest.hexdigest()


def _require_bound_artifact(
    plan_path: Path,
    raw_path: Any,
    expected_sha256: Any,
    label: str,
) -> Path:
    """Resolve and hash-check one offline provenance artifact."""

    text = _require_text(raw_path, label + "_path")
    expected = _require_text(expected_sha256, label + "_sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("{} must be a lowercase SHA-256".format(label + "_sha256"))
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = plan_path.parent / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError("{} artifact is unavailable: {}".format(label, exc)) from exc
    if not resolved.is_file():
        raise ValueError("{} artifact is not a regular file: {}".format(label, resolved))
    actual = _sha256_file(resolved)
    if actual != expected:
        raise ValueError(
            "{} SHA-256 mismatch: expected={} actual={}".format(
                label, expected, actual
            )
        )
    return resolved


def _workspace(safety: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    bounds = _require_mapping(
        safety.get("workspace_bounds_base_m"), "safety.workspace_bounds_base_m"
    )
    lower = []
    upper = []
    for axis in ("x", "y", "z"):
        interval = _finite_vector(
            bounds.get(axis), 2, "safety.workspace_bounds_base_m.{}".format(axis)
        )
        if interval[0] >= interval[1]:
            raise ValueError("workspace {} lower bound must be below upper bound".format(axis))
        lower.append(float(interval[0]))
        upper.append(float(interval[1]))
    minimum = np.asarray(lower, dtype=np.float64)
    maximum = np.asarray(upper, dtype=np.float64)
    minimum.setflags(write=False)
    maximum.setflags(write=False)
    return minimum, maximum


def _implicit_minimum_jerk_acceleration_bound(
    max_distance: float, max_speed: float
) -> float:
    crossover_distance = (
        max_speed
        * MIN_CARTESIAN_DURATION_S
        / MINIMUM_JERK_PEAK_VELOCITY_FACTOR
    )
    worst_distance = min(max_distance, crossover_distance)
    duration = max(
        MIN_CARTESIAN_DURATION_S,
        MINIMUM_JERK_PEAK_VELOCITY_FACTOR * worst_distance / max_speed,
    )
    return (
        MINIMUM_JERK_PEAK_ACCELERATION_FACTOR
        * worst_distance
        / (duration * duration)
    )


def _validate_motion_contract(safety: Mapping[str, Any]) -> None:
    checks = (
        (
            "maximum_translation_step_m",
            MAX_SEGMENT_TRANSLATION_M,
            "m",
        ),
        (
            "maximum_rotation_step_deg",
            math.degrees(MAX_SEGMENT_ROTATION_RAD),
            "deg",
        ),
        (
            "maximum_translation_velocity_m_s",
            MAX_CARTESIAN_SPEED_M_S,
            "m/s",
        ),
        (
            "maximum_angular_velocity_rad_s",
            MAX_ANGULAR_SPEED_RAD_S,
            "rad/s",
        ),
        (
            "maximum_translation_acceleration_m_s2",
            _implicit_minimum_jerk_acceleration_bound(
                MAX_SEGMENT_TRANSLATION_M, MAX_CARTESIAN_SPEED_M_S
            ),
            "m/s^2",
        ),
        (
            "maximum_angular_acceleration_rad_s2",
            _implicit_minimum_jerk_acceleration_bound(
                MAX_SEGMENT_ROTATION_RAD, MAX_ANGULAR_SPEED_RAD_S
            ),
            "rad/s^2",
        ),
    )
    for field, wrapper_value, unit in checks:
        plan_limit = _finite_float(safety.get(field), "safety." + field)
        if plan_limit <= 0.0:
            raise ValueError("safety.{} must be positive".format(field))
        if wrapper_value > plan_limit + NUMERICAL_TOLERANCE:
            raise ValueError(
                "wrapper {} {:.9f}{} exceeds plan limit {:.9f}{}".format(
                    field, wrapper_value, unit, plan_limit, unit
                )
            )
    endpoint_hold_s = _finite_float(
        safety.get("cartesian_endpoint_hold_s"),
        "safety.cartesian_endpoint_hold_s",
    )
    if not np.isclose(
        endpoint_hold_s,
        CARTESIAN_ENDPOINT_HOLD_S,
        atol=NUMERICAL_TOLERANCE,
        rtol=0.0,
    ):
        raise ValueError(
            "safety.cartesian_endpoint_hold_s must exactly match wrapper "
            "value {:.3f}s".format(CARTESIAN_ENDPOINT_HOLD_S)
        )
    if safety.get("cartesian_pose_controller_mode") != CARTESIAN_POSE_CONTROLLER_MODE:
        raise ValueError(
            "safety.cartesian_pose_controller_mode must be {!r}".format(
                CARTESIAN_POSE_CONTROLLER_MODE
            )
        )


def _load_dynamics(robot: Mapping[str, Any]) -> EndEffectorDynamics:
    transform = _rigid_transform(
        robot.get("expected_F_T_EE"), "hardware.robot.expected_F_T_EE"
    )
    mass = _finite_float(
        robot.get("expected_m_ee_kg"), "hardware.robot.expected_m_ee_kg"
    )
    if mass < 0.0:
        raise ValueError("hardware.robot.expected_m_ee_kg must be non-negative")
    center = _finite_vector(
        robot.get("expected_F_x_Cee_m"), 3, "hardware.robot.expected_F_x_Cee_m"
    )
    try:
        inertia = np.asarray(robot.get("expected_I_ee_kg_m2"), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "hardware.robot.expected_I_ee_kg_m2 must be a finite 3x3 matrix"
        ) from exc
    if inertia.shape != (3, 3) or not np.all(np.isfinite(inertia)):
        raise ValueError(
            "hardware.robot.expected_I_ee_kg_m2 must be a finite 3x3 matrix"
        )
    if not np.allclose(inertia, inertia.T, atol=1.0e-10, rtol=0.0):
        raise ValueError("hardware.robot.expected_I_ee_kg_m2 must be symmetric")
    if float(np.min(np.linalg.eigvalsh(inertia))) < -1.0e-10:
        raise ValueError(
            "hardware.robot.expected_I_ee_kg_m2 must be positive semidefinite"
        )

    external = robot.get("expected_external_load")
    if external is not None:
        load = _require_mapping(external, "hardware.robot.expected_external_load")
        load_mass = _finite_float(
            load.get("m_load_kg"), "hardware.robot.expected_external_load.m_load_kg"
        )
        load_center = _finite_vector(
            load.get("F_x_Cload_m"),
            3,
            "hardware.robot.expected_external_load.F_x_Cload_m",
        )
        try:
            load_inertia = np.asarray(load.get("I_load_kg_m2"), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "hardware.robot.expected_external_load.I_load_kg_m2 must be 3x3"
            ) from exc
        if load_inertia.shape != (3, 3) or not np.all(np.isfinite(load_inertia)):
            raise ValueError(
                "hardware.robot.expected_external_load.I_load_kg_m2 must be 3x3"
            )
        if (
            abs(load_mass) > NUMERICAL_TOLERANCE
            or np.max(np.abs(load_center)) > NUMERICAL_TOLERANCE
            or np.max(np.abs(load_inertia)) > NUMERICAL_TOLERANCE
        ):
            raise ValueError(
                "non-zero expected_external_load is unsupported by this wrapper"
            )

    inertia = inertia.copy()
    inertia.setflags(write=False)
    return EndEffectorDynamics(transform, mass, center, inertia)


def _pose_entries(payload: Mapping[str, Any]) -> Mapping[str, Tuple[str, Mapping[str, Any]]]:
    by_id = {}
    collections = (
        ("training_poses", "training", True),
        ("holdout_poses", "holdout", True),
        ("recovery_poses", "recovery", False),
    )
    for collection_name, pose_set, required in collections:
        entries = payload.get(collection_name)
        if entries is None and not required:
            continue
        if not isinstance(entries, list):
            raise ValueError("{} must be a list".format(collection_name))
        for index, raw in enumerate(entries):
            entry = _require_mapping(raw, "{}[{}]".format(collection_name, index))
            pose_id = _require_text(
                entry.get("id"), "{}[{}].id".format(collection_name, index)
            )
            if not _POSE_ID_PATTERN.fullmatch(pose_id):
                raise ValueError("pose id {!r} is malformed".format(pose_id))
            if pose_id in by_id:
                raise ValueError("duplicate pose id {!r}".format(pose_id))
            by_id[pose_id] = (pose_set, entry)
    return by_id


def _validated_plan_pose(
    pose_id: str,
    entry: Mapping[str, Any],
    anchor: np.ndarray,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
    minimum_z: float,
    translation_limit_m: float,
    rotation_limit_deg: float,
) -> _ValidatedPlanPose:
    offset = _finite_vector(
        entry.get("xyz_offset_base_m"),
        3,
        "pose[{}].xyz_offset_base_m".format(pose_id),
    )
    rotation_deg = _finite_vector(
        entry.get("rotation_vector_eef_deg"),
        3,
        "pose[{}].rotation_vector_eef_deg".format(pose_id),
    )
    translation_norm = float(np.linalg.norm(offset))
    rotation_angle_deg = float(np.linalg.norm(rotation_deg))
    if translation_norm > translation_limit_m + NUMERICAL_TOLERANCE:
        raise ValueError(
            "pose {} translation norm {:.9f}m exceeds {:.9f}m".format(
                pose_id, translation_norm, translation_limit_m
            )
        )
    if rotation_angle_deg > rotation_limit_deg + NUMERICAL_TOLERANCE:
        raise ValueError(
            "pose {} rotation angle {:.9f}deg exceeds {:.9f}deg".format(
                pose_id, rotation_angle_deg, rotation_limit_deg
            )
        )
    if pose_id == "T01" and (
        translation_norm > NUMERICAL_TOLERANCE
        or rotation_angle_deg > NUMERICAL_TOLERANCE
    ):
        raise ValueError("T01 must be the zero-offset anchor pose")

    rotation_rad = np.deg2rad(rotation_deg)
    rotation_rad.setflags(write=False)
    target = build_target_transform(anchor, offset, rotation_rad)
    if np.any(target[:3, 3] < workspace_min - NUMERICAL_TOLERANCE) or np.any(
        target[:3, 3] > workspace_max + NUMERICAL_TOLERANCE
    ):
        raise ValueError(
            "pose {} target {} is outside hard workspace [{}, {}]".format(
                pose_id,
                target[:3, 3].tolist(),
                workspace_min.tolist(),
                workspace_max.tolist(),
            )
        )
    if target[2, 3] < minimum_z - NUMERICAL_TOLERANCE:
        raise ValueError(
            "pose {} target z {:.9f}m is below minimum {:.9f}m".format(
                pose_id, target[2, 3], minimum_z
            )
        )
    return _ValidatedPlanPose(
        xyz_offset_base_m=offset,
        rotation_vector_eef_rad=rotation_rad,
        T_target=target,
        translation_norm_m=translation_norm,
        rotation_angle_rad=math.radians(rotation_angle_deg),
    )


def _validate_authorization(
    payload: Mapping[str, Any],
    safety: Mapping[str, Any],
    pose_id: str,
    poses: Mapping[str, Tuple[str, Mapping[str, Any]]],
) -> str:
    status = _require_text(payload.get("status"), "status")
    if status != RUN_STATUS:
        raise ValueError(
            "run requires status={!r}; current status={!r}".format(RUN_STATUS, status)
        )
    if safety.get("motion_authorized") is not True:
        raise ValueError("run requires safety.motion_authorized to be exactly true")
    authorization = _require_mapping(
        payload.get("motion_authorization"), "motion_authorization"
    )
    if authorization.get("explicit_user_authorization_recorded") is not True:
        raise ValueError(
            "motion_authorization.explicit_user_authorization_recorded must be true"
        )
    if authorization.get("consumed") is True:
        raise ValueError("motion_authorization has already been consumed")
    if authorization.get("scope") != AUTHORIZATION_SCOPE:
        raise ValueError(
            "motion_authorization.scope must be {!r}".format(AUTHORIZATION_SCOPE)
        )
    if authorization.get("pose_id") != pose_id:
        raise ValueError(
            "motion authorization is for pose {!r}, not {!r}".format(
                authorization.get("pose_id"), pose_id
            )
        )
    start_pose_id = _require_text(
        authorization.get("start_pose_id"),
        "motion_authorization.start_pose_id",
    )
    if start_pose_id not in poses:
        raise ValueError(
            "motion_authorization.start_pose_id {!r} is not a planned pose".format(
                start_pose_id
            )
        )
    return start_pose_id


def load_pose_preview(
    plan_path: Path,
    pose_id: str,
    *,
    expected_sha256: Optional[str] = None,
    for_run: bool = False,
) -> PosePreview:
    """Load and validate one pose without importing any hardware package."""

    requested_id = _require_text(pose_id, "pose_id")
    resolved, digest, payload = _read_plan(plan_path)
    if expected_sha256 is not None and expected_sha256 != digest:
        raise ValueError(
            "session plan SHA-256 mismatch: expected={} actual={}".format(
                expected_sha256, digest
            )
        )
    if payload.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError(
            "schema_version must be exactly {}".format(PLAN_SCHEMA_VERSION)
        )
    if payload.get("kind") != PLAN_KIND:
        raise ValueError("kind must be {!r}".format(PLAN_KIND))
    status = _require_text(payload.get("status"), "status")
    if status not in PREVIEW_STATUSES:
        raise ValueError("plan status {!r} is not previewable".format(status))

    topology = _require_mapping(payload.get("topology"), "topology")
    if topology.get("type") != "eye_to_hand" or topology.get("camera_mount") != "fixed_external":
        raise ValueError("plan must describe a fixed-external eye-to-hand camera")
    session_slug = _require_text(payload.get("session_slug"), "session_slug")
    safety = _require_mapping(payload.get("safety"), "safety")
    _validate_motion_contract(safety)
    workspace_min, workspace_max = _workspace(safety)
    minimum_z = _finite_float(
        safety.get("minimum_eef_z_m"), "safety.minimum_eef_z_m"
    )
    if minimum_z < workspace_min[2] - NUMERICAL_TOLERANCE:
        raise ValueError("minimum_eef_z_m may not be below the workspace z minimum")
    if minimum_z > workspace_max[2] + NUMERICAL_TOLERANCE:
        raise ValueError("minimum_eef_z_m lies above the workspace")

    plan_translation_limit = _finite_float(
        safety.get("maximum_translation_norm_from_anchor_m"),
        "safety.maximum_translation_norm_from_anchor_m",
    )
    if (
        plan_translation_limit <= 0.0
        or plan_translation_limit > MAX_TRANSLATION_NORM_M + NUMERICAL_TOLERANCE
    ):
        raise ValueError(
            "plan translation norm limit exceeds wrapper cap {:.3f}m".format(
                MAX_TRANSLATION_NORM_M
            )
        )
    plan_rotation_limit_deg = _finite_float(
        safety.get("maximum_rotation_angle_from_anchor_deg"),
        "safety.maximum_rotation_angle_from_anchor_deg",
    )
    if (
        plan_rotation_limit_deg <= 0.0
        or plan_rotation_limit_deg > MAX_ROTATION_ANGLE_DEG + NUMERICAL_TOLERANCE
    ):
        raise ValueError(
            "plan rotation limit exceeds wrapper cap {:.1f}deg".format(
                MAX_ROTATION_ANGLE_DEG
            )
        )

    reference = _require_mapping(payload.get("reference_pose"), "reference_pose")
    anchor = _rigid_transform(reference.get("T_base_ee"), "reference_pose.T_base_ee")
    anchor_xyz = _finite_vector(
        reference.get("xyz_base_m"), 3, "reference_pose.xyz_base_m"
    )
    if not np.allclose(anchor[:3, 3], anchor_xyz, atol=1.0e-8, rtol=0.0):
        raise ValueError("reference_pose.xyz_base_m does not match T_base_ee")

    poses = _pose_entries(payload)
    if requested_id not in poses:
        raise ValueError(
            "pose id {!r} is not in training_poses or holdout_poses".format(
                requested_id
            )
        )
    pose_set, entry = poses[requested_id]
    selected_pose = _validated_plan_pose(
        requested_id,
        entry,
        anchor,
        workspace_min,
        workspace_max,
        minimum_z,
        plan_translation_limit,
        plan_rotation_limit_deg,
    )

    settle_time = _finite_float(safety.get("settle_time_s"), "safety.settle_time_s")
    if settle_time <= 0.0:
        raise ValueError("safety.settle_time_s must be positive")

    hardware = _require_mapping(payload.get("hardware"), "hardware")
    robot = _require_mapping(hardware.get("robot"), "hardware.robot")
    dynamics = None
    dynamics_error = None
    try:
        dynamics = _load_dynamics(robot)
    except ValueError as exc:
        dynamics_error = str(exc)

    run_authorized = False
    authorization_error = None
    start_pose_id = None
    start_target = None
    try:
        start_pose_id = _validate_authorization(
            payload, safety, requested_id, poses
        )
        start_entry = poses[start_pose_id][1]
        start_target = _validated_plan_pose(
            start_pose_id,
            start_entry,
            anchor,
            workspace_min,
            workspace_max,
            minimum_z,
            plan_translation_limit,
            plan_rotation_limit_deg,
        ).T_target
        maximum_edge_translation_m = _finite_float(
            safety.get("maximum_translation_step_m"),
            "safety.maximum_translation_step_m",
        )
        maximum_edge_rotation_deg = _finite_float(
            safety.get("maximum_rotation_step_deg"),
            "safety.maximum_rotation_step_deg",
        )
        edge_translation_m = float(
            np.linalg.norm(
                selected_pose.T_target[:3, 3] - start_target[:3, 3]
            )
        )
        edge_rotation_deg = math.degrees(
            _rotation_error_rad(start_target, selected_pose.T_target)
        )
        if edge_translation_m > maximum_edge_translation_m + NUMERICAL_TOLERANCE:
            raise ValueError(
                "authorized edge {}->{} translation {:.9f}m exceeds {:.9f}m".format(
                    start_pose_id,
                    requested_id,
                    edge_translation_m,
                    maximum_edge_translation_m,
                )
            )
        if edge_rotation_deg > maximum_edge_rotation_deg + NUMERICAL_TOLERANCE:
            raise ValueError(
                "authorized edge {}->{} rotation {:.9f}deg exceeds {:.9f}deg".format(
                    start_pose_id,
                    requested_id,
                    edge_rotation_deg,
                    maximum_edge_rotation_deg,
                )
            )
    except ValueError as exc:
        authorization_error = str(exc)
    else:
        run_authorized = True

    if for_run:
        if dynamics is None:
            raise ValueError("run blocked by missing/invalid strict dynamics: {}".format(dynamics_error))
        if not run_authorized:
            raise ValueError("run blocked by plan authorization: {}".format(authorization_error))

    return PosePreview(
        plan_path=resolved,
        plan_sha256=digest,
        plan_status=status,
        session_slug=session_slug,
        pose_set=pose_set,
        pose_id=requested_id,
        start_pose_id=start_pose_id,
        T_anchor=anchor,
        xyz_offset_base_m=selected_pose.xyz_offset_base_m,
        rotation_vector_eef_rad=selected_pose.rotation_vector_eef_rad,
        T_target=selected_pose.T_target,
        T_start=start_target,
        workspace_min_m=workspace_min,
        workspace_max_m=workspace_max,
        minimum_eef_z_m=minimum_z,
        translation_norm_m=selected_pose.translation_norm_m,
        rotation_angle_rad=selected_pose.rotation_angle_rad,
        settle_time_s=settle_time,
        motion_authorized=safety.get("motion_authorized") is True,
        run_authorized=run_authorized,
        authorization_error=authorization_error,
        dynamics=dynamics,
        dynamics_error=dynamics_error,
    )


def _require_exact_float(
    mapping: Mapping[str, Any], field: str, expected: float, prefix: str = "safety"
) -> float:
    value = _finite_float(mapping.get(field), "{}.{}".format(prefix, field))
    if not np.isclose(value, expected, atol=NUMERICAL_TOLERANCE, rtol=0.0):
        raise ValueError(
            "{}.{} must exactly match wrapper value {:.9f}".format(
                prefix, field, expected
            )
        )
    return value


def _joint_recovery_receipt_path(plan_path: Path, digest: str) -> Path:
    """Return the SHA-bound, sibling one-use authorization claim path."""

    return plan_path.parent / (
        ".franka-joint-recovery-consumed-{}.json".format(digest)
    )


def _validate_joint_margin(q: np.ndarray, name: str) -> None:
    lower = _FR3_JOINT_LIMITS_RAD[:, 0] + JOINT_LIMIT_MARGIN_RAD
    upper = _FR3_JOINT_LIMITS_RAD[:, 1] - JOINT_LIMIT_MARGIN_RAD
    outside = np.flatnonzero((q < lower) | (q > upper))
    if len(outside):
        index = int(outside[0])
        raise ValueError(
            "{} joint {} value {:.9f}rad is outside [{:.9f}, {:.9f}]".format(
                name, index + 1, q[index], lower[index], upper[index]
            )
        )


def _joint_recovery_authorization_error(
    payload: Mapping[str, Any],
    safety: Mapping[str, Any],
    recovery_id: str,
    receipt_path: Path,
) -> Optional[str]:
    try:
        status = _require_text(payload.get("status"), "status")
        if status != JOINT_RECOVERY_RUN_STATUS:
            raise ValueError(
                "run requires status={!r}; current status={!r}".format(
                    JOINT_RECOVERY_RUN_STATUS, status
                )
            )
        if safety.get("motion_authorized") is not True:
            raise ValueError("run requires safety.motion_authorized to be exactly true")
        for field in (
            "emergency_stop_reachable",
            "full_board_bracket_and_cable_swept_volume_clear",
            "target_rigid_confirmed",
            "camera_fixed_confirmed",
        ):
            if safety.get(field) is not True:
                raise ValueError("safety.{} must be exactly true".format(field))
        authorization = _require_mapping(
            payload.get("motion_authorization"), "motion_authorization"
        )
        if authorization.get("explicit_user_authorization_recorded") is not True:
            raise ValueError(
                "motion_authorization.explicit_user_authorization_recorded must be true"
            )
        if authorization.get("scope") != JOINT_RECOVERY_AUTHORIZATION_SCOPE:
            raise ValueError(
                "motion_authorization.scope must be {!r}".format(
                    JOINT_RECOVERY_AUTHORIZATION_SCOPE
                )
            )
        if authorization.get("recovery_id") != recovery_id:
            raise ValueError(
                "motion authorization is for recovery {!r}, not {!r}".format(
                    authorization.get("recovery_id"), recovery_id
                )
            )
        if authorization.get("consumed") is not False:
            raise ValueError("motion_authorization.consumed must be exactly false")
        _require_text(authorization.get("source_text"), "motion_authorization.source_text")
        if receipt_path.exists():
            raise ValueError(
                "bounded joint recovery authorization already consumed by {}".format(
                    receipt_path
                )
            )
    except ValueError as exc:
        return str(exc)
    return None


def load_joint_recovery_preview(
    plan_path: Path,
    *,
    expected_sha256: Optional[str] = None,
    for_run: bool = False,
) -> JointRecoveryPreview:
    """Load one dedicated bounded joint recovery plan without hardware imports."""

    resolved, digest, payload = _read_plan(plan_path)
    if expected_sha256 is not None and expected_sha256 != digest:
        raise ValueError(
            "session plan SHA-256 mismatch: expected={} actual={}".format(
                expected_sha256, digest
            )
        )
    if payload.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("schema_version must be exactly {}".format(PLAN_SCHEMA_VERSION))
    if payload.get("kind") != JOINT_RECOVERY_PLAN_KIND:
        raise ValueError("kind must be {!r}".format(JOINT_RECOVERY_PLAN_KIND))
    status = _require_text(payload.get("status"), "status")
    if status not in JOINT_RECOVERY_PREVIEW_STATUSES:
        raise ValueError("joint recovery plan status {!r} is not previewable".format(status))
    topology = _require_mapping(payload.get("topology"), "topology")
    if (
        topology.get("type") != "eye_to_hand"
        or topology.get("camera_mount") != "fixed_external"
    ):
        raise ValueError("plan must describe a fixed-external eye-to-hand camera")
    session_slug = _require_text(payload.get("session_slug"), "session_slug")

    hardware = _require_mapping(payload.get("hardware"), "hardware")
    robot = _require_mapping(hardware.get("robot"), "hardware.robot")
    robot_ip = _require_text(robot.get("ip"), "hardware.robot.ip")
    dynamics = None
    dynamics_error = None
    try:
        dynamics = _load_dynamics(robot)
    except ValueError as exc:
        dynamics_error = str(exc)

    safety = _require_mapping(payload.get("safety"), "safety")
    workspace_min, workspace_max = _workspace(safety)
    minimum_z = _finite_float(
        safety.get("minimum_eef_z_m"), "safety.minimum_eef_z_m"
    )
    if minimum_z < workspace_min[2] - NUMERICAL_TOLERANCE:
        raise ValueError("minimum_eef_z_m may not be below workspace z minimum")
    if minimum_z > workspace_max[2] + NUMERICAL_TOLERANCE:
        raise ValueError("minimum_eef_z_m lies above the workspace")
    _require_exact_float(safety, "expected_start_q_tolerance_rad", START_Q_TOLERANCE_RAD)
    _require_exact_float(safety, "maximum_start_dq_rad_s", MAX_START_DQ_RAD_S)
    _require_exact_float(
        safety, "joint_arrival_tolerance_rad", JOINT_ENDPOINT_TOLERANCE_RAD
    )
    _require_exact_float(
        safety,
        "start_pose_translation_tolerance_m",
        START_POSE_TRANSLATION_TOLERANCE_M,
    )
    _require_exact_float(
        safety,
        "start_pose_rotation_tolerance_deg",
        math.degrees(START_POSE_ROTATION_TOLERANCE_RAD),
    )
    _require_exact_float(
        safety,
        "endpoint_pose_translation_tolerance_m",
        ENDPOINT_POSE_TRANSLATION_TOLERANCE_M,
    )
    _require_exact_float(
        safety,
        "endpoint_pose_rotation_tolerance_deg",
        math.degrees(ENDPOINT_POSE_ROTATION_TOLERANCE_RAD),
    )
    _require_exact_float(safety, "endpoint_max_dq_rad_s", ENDPOINT_MAX_DQ_RAD_S)
    _require_exact_float(safety, "endpoint_poll_s", ENDPOINT_POLL_S)
    if safety.get("endpoint_consecutive_samples") != ENDPOINT_CONSECUTIVE_SAMPLES:
        raise ValueError(
            "safety.endpoint_consecutive_samples must be exactly {}".format(
                ENDPOINT_CONSECUTIVE_SAMPLES
            )
        )
    if safety.get("maximum_dynamic_segments") != JOINT_RECOVERY_MAX_DYNAMIC_SEGMENTS:
        raise ValueError(
            "safety.maximum_dynamic_segments must be exactly {}".format(
                JOINT_RECOVERY_MAX_DYNAMIC_SEGMENTS
            )
        )
    if safety.get("joint_time_law") != JOINT_RECOVERY_TIME_LAW:
        raise ValueError(
            "safety.joint_time_law must be {!r}".format(JOINT_RECOVERY_TIME_LAW)
        )

    per_joint_delta_limit = _finite_float(
        safety.get("maximum_per_joint_delta_rad"),
        "safety.maximum_per_joint_delta_rad",
    )
    if not 0.0 < per_joint_delta_limit <= MAX_SINGLE_JOINT_DELTA_RAD + NUMERICAL_TOLERANCE:
        raise ValueError("per-joint recovery delta limit exceeds wrapper cap of 5deg")
    delta_norm_limit = _finite_float(
        safety.get("maximum_joint_delta_norm_rad"),
        "safety.maximum_joint_delta_norm_rad",
    )
    if not 0.0 < delta_norm_limit <= MAX_JOINT_DELTA_NORM_RAD + NUMERICAL_TOLERANCE:
        raise ValueError("joint recovery delta norm limit exceeds wrapper cap of 9deg")
    maximum_velocity = _finite_float(
        safety.get("maximum_joint_velocity_rad_s"),
        "safety.maximum_joint_velocity_rad_s",
    )
    if not 0.0 < maximum_velocity <= MAX_JOINT_RECOVERY_SPEED_RAD_S + NUMERICAL_TOLERANCE:
        raise ValueError("joint recovery velocity exceeds wrapper cap")
    maximum_acceleration = _finite_float(
        safety.get("maximum_joint_acceleration_rad_s2"),
        "safety.maximum_joint_acceleration_rad_s2",
    )
    if not 0.0 < maximum_acceleration <= MAX_JOINT_RECOVERY_ACCELERATION_RAD_S2 + NUMERICAL_TOLERANCE:
        raise ValueError("joint recovery acceleration exceeds wrapper cap")
    minimum_duration = _finite_float(
        safety.get("minimum_joint_duration_s"), "safety.minimum_joint_duration_s"
    )
    if minimum_duration < MIN_JOINT_RECOVERY_DURATION_S - NUMERICAL_TOLERANCE:
        raise ValueError("joint recovery minimum duration is below wrapper floor")

    recovery = _require_mapping(payload.get("recovery"), "recovery")
    recovery_id = _require_text(recovery.get("id"), "recovery.id")
    if not _POSE_ID_PATTERN.fullmatch(recovery_id):
        raise ValueError("recovery id {!r} is malformed".format(recovery_id))
    start_q = _finite_vector(
        recovery.get("expected_start_q_rad"), 7, "recovery.expected_start_q_rad"
    )
    target_q = _finite_vector(
        recovery.get("target_q_rad"), 7, "recovery.target_q_rad"
    )
    delta = _finite_vector(
        recovery.get("commanded_delta_rad"), 7, "recovery.commanded_delta_rad"
    )
    maximum_joint_delta = float(np.max(np.abs(delta)))
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm < MIN_JOINT_DELTA_NORM_RAD - NUMERICAL_TOLERANCE:
        raise ValueError("joint recovery delta norm must be at least 0.1deg")
    if maximum_joint_delta > per_joint_delta_limit + NUMERICAL_TOLERANCE:
        raise ValueError("joint recovery per-joint delta exceeds plan limit")
    if delta_norm > delta_norm_limit + NUMERICAL_TOLERANCE:
        raise ValueError("joint recovery delta norm exceeds plan limit")
    expected_target_q = start_q + delta
    if not np.allclose(target_q, expected_target_q, atol=1.0e-12, rtol=0.0):
        raise ValueError(
            "recovery.target_q_rad must exactly equal expected_start_q_rad + commanded_delta_rad"
        )
    _validate_joint_margin(start_q, "recovery expected start")
    _validate_joint_margin(target_q, "recovery planned target")

    start_T = _rigid_transform(
        recovery.get("expected_start_T_base_ee"),
        "recovery.expected_start_T_base_ee",
    )
    target_T = _rigid_transform(
        recovery.get("expected_target_T_base_ee"),
        "recovery.expected_target_T_base_ee",
    )
    for label, transform in (("start", start_T), ("target", target_T)):
        position = transform[:3, 3]
        if np.any(position < workspace_min - NUMERICAL_TOLERANCE) or np.any(
            position > workspace_max + NUMERICAL_TOLERANCE
        ):
            raise ValueError(
                "predicted {} EEF position {} is outside hard workspace".format(
                    label, position.tolist()
                )
            )
        if position[2] < minimum_z - NUMERICAL_TOLERANCE:
            raise ValueError(
                "predicted {} EEF z {:.9f}m is below minimum {:.9f}m".format(
                    label, position[2], minimum_z
                )
            )
    endpoint_translation = float(
        np.linalg.norm(target_T[:3, 3] - start_T[:3, 3])
    )
    endpoint_rotation = _rotation_error_rad(start_T, target_T)

    kinematics = _require_mapping(
        recovery.get("kinematics_provenance"), "recovery.kinematics_provenance"
    )
    _require_bound_artifact(
        resolved,
        kinematics.get("urdf_path"),
        kinematics.get("urdf_sha256"),
        "recovery.kinematics_provenance.urdf",
    )
    _require_bound_artifact(
        resolved,
        kinematics.get("fk_implementation"),
        kinematics.get("fk_implementation_sha256"),
        "recovery.kinematics_provenance.fk_implementation",
    )
    _require_bound_artifact(
        resolved,
        kinematics.get("audit_artifact_path"),
        kinematics.get("audit_artifact_sha256"),
        "recovery.kinematics_provenance.audit_artifact",
    )
    if kinematics.get("expected_start_fk_matches_plan") is not True:
        raise ValueError("kinematics expected_start_fk_matches_plan must be true")
    if kinematics.get("expected_target_fk_matches_plan") is not True:
        raise ValueError("kinematics expected_target_fk_matches_plan must be true")
    raw_sample_count = kinematics.get("interpolation_sample_count")
    if (
        isinstance(raw_sample_count, (bool, np.bool_))
        or not isinstance(raw_sample_count, int)
        or raw_sample_count < 21
    ):
        raise ValueError("kinematics interpolation_sample_count must be integer >= 21")
    if kinematics.get("all_interpolation_samples_within_workspace") is not True:
        raise ValueError("kinematics interpolation samples must all be within workspace")
    path_minimum_z = _finite_float(
        kinematics.get("minimum_interpolation_eef_z_m"),
        "recovery.kinematics_provenance.minimum_interpolation_eef_z_m",
    )
    if path_minimum_z < minimum_z - NUMERICAL_TOLERANCE:
        raise ValueError("predicted joint interpolation path descends below minimum EEF z")
    predicted_translation = _finite_float(
        kinematics.get("maximum_interpolation_eef_center_displacement_m"),
        "recovery.kinematics_provenance.maximum_interpolation_eef_center_displacement_m",
    )
    predicted_rotation = math.radians(
        _finite_float(
            kinematics.get("maximum_interpolation_eef_rotation_deg"),
            "recovery.kinematics_provenance.maximum_interpolation_eef_rotation_deg",
        )
    )
    if predicted_translation + NUMERICAL_TOLERANCE < endpoint_translation:
        raise ValueError("predicted path displacement is below endpoint displacement")
    if predicted_rotation + NUMERICAL_TOLERANCE < endpoint_rotation:
        raise ValueError("predicted path rotation is below endpoint rotation")
    if predicted_translation > MAX_PREDICTED_EEF_TRANSLATION_M + NUMERICAL_TOLERANCE:
        raise ValueError("predicted EEF path translation exceeds wrapper cap")
    if predicted_rotation > math.radians(10.0) + NUMERICAL_TOLERANCE:
        raise ValueError("predicted EEF path rotation exceeds wrapper cap of 10deg")
    board_extent = _finite_float(
        safety.get("board_max_extent_from_eef_m"),
        "safety.board_max_extent_from_eef_m",
    )
    if not 0.0 < board_extent <= 0.30 + NUMERICAL_TOLERANCE:
        raise ValueError("board extent must be positive and at most 0.30m")
    computed_board_sweep = predicted_translation + 2.0 * board_extent * math.sin(
        0.5 * predicted_rotation
    )
    declared_board_sweep = _finite_float(
        safety.get("conservative_board_sweep_displacement_m"),
        "safety.conservative_board_sweep_displacement_m",
    )
    if declared_board_sweep + NUMERICAL_TOLERANCE < computed_board_sweep:
        raise ValueError(
            "declared conservative board sweep {:.9f}m is below computed {:.9f}m".format(
                declared_board_sweep, computed_board_sweep
            )
        )
    if declared_board_sweep > MAX_CONSERVATIVE_BOARD_SWEEP_M + NUMERICAL_TOLERANCE:
        raise ValueError("conservative board sweep exceeds wrapper cap")

    maximum_live_to_target_delta = maximum_joint_delta + START_Q_TOLERANCE_RAD
    duration = max(
        minimum_duration,
        (math.pi / 2.0) * maximum_live_to_target_delta / maximum_velocity,
    )
    expected_peak_velocity = (
        (math.pi / 2.0) * maximum_live_to_target_delta / duration
    )
    expected_peak_acceleration = (
        (math.pi * math.pi / 2.0)
        * maximum_live_to_target_delta
        / (duration * duration)
    )
    if expected_peak_acceleration > maximum_acceleration + NUMERICAL_TOLERANCE:
        raise ValueError(
            "implicit cosine peak acceleration {:.9f}rad/s^2 exceeds plan limit {:.9f}".format(
                expected_peak_acceleration, maximum_acceleration
            )
        )

    receipt_path = _joint_recovery_receipt_path(resolved, digest)
    authorization_error = _joint_recovery_authorization_error(
        payload, safety, recovery_id, receipt_path
    )
    run_authorized = authorization_error is None
    if for_run:
        if dynamics is None:
            raise ValueError(
                "run blocked by missing/invalid strict dynamics: {}".format(
                    dynamics_error
                )
            )
        if not run_authorized:
            raise ValueError(
                "run blocked by plan authorization: {}".format(authorization_error)
            )

    return JointRecoveryPreview(
        plan_path=resolved,
        plan_sha256=digest,
        plan_status=status,
        session_slug=session_slug,
        recovery_id=recovery_id,
        robot_ip=robot_ip,
        expected_start_q_rad=start_q,
        planned_target_q_rad=target_q,
        commanded_delta_rad=delta,
        maximum_live_to_target_delta_rad=maximum_live_to_target_delta,
        expected_start_T_base_ee=start_T,
        expected_target_T_base_ee=target_T,
        workspace_min_m=workspace_min,
        workspace_max_m=workspace_max,
        minimum_eef_z_m=minimum_z,
        board_max_extent_from_eef_m=board_extent,
        conservative_board_sweep_displacement_m=declared_board_sweep,
        predicted_eef_translation_m=predicted_translation,
        predicted_eef_rotation_rad=predicted_rotation,
        maximum_joint_velocity_rad_s=maximum_velocity,
        maximum_joint_acceleration_rad_s2=maximum_acceleration,
        minimum_joint_duration_s=minimum_duration,
        expected_peak_joint_velocity_rad_s=expected_peak_velocity,
        expected_peak_joint_acceleration_rad_s2=expected_peak_acceleration,
        motion_authorized=safety.get("motion_authorized") is True,
        run_authorized=run_authorized,
        authorization_error=authorization_error,
        authorization_receipt_path=receipt_path,
        authorization_receipt_exists=receipt_path.exists(),
        dynamics=dynamics,
        dynamics_error=dynamics_error,
    )


def _require_confirmations(args: argparse.Namespace, preview: PosePreview) -> None:
    expected = (
        ("--confirm-plan-sha256", args.confirm_plan_sha256, preview.plan_sha256),
        ("--confirm-pose-id", args.confirm_pose_id, preview.pose_id),
        ("--confirm-e-stop", args.confirm_e_stop, E_STOP_TOKEN),
        ("--confirm-swept-volume", args.confirm_swept_volume, SWEPT_VOLUME_TOKEN),
        ("--confirm-target-rigid", args.confirm_target_rigid, TARGET_RIGID_TOKEN),
        ("--confirm-camera-fixed", args.confirm_camera_fixed, CAMERA_FIXED_TOKEN),
    )
    missing = [
        "{} {}".format(flag, token)
        for flag, actual, token in expected
        if actual != token
    ]
    if missing:
        raise ValueError("exact confirmations required: " + "; ".join(missing))


def _require_joint_recovery_confirmations(
    args: argparse.Namespace, preview: JointRecoveryPreview
) -> None:
    expected = (
        ("--confirm-plan-sha256", args.confirm_plan_sha256, preview.plan_sha256),
        ("--confirm-recovery-id", args.confirm_recovery_id, preview.recovery_id),
        ("--confirm-e-stop", args.confirm_e_stop, E_STOP_TOKEN),
        (
            "--confirm-swept-volume",
            args.confirm_swept_volume,
            JOINT_RECOVERY_SWEEP_TOKEN,
        ),
        ("--confirm-target-rigid", args.confirm_target_rigid, TARGET_RIGID_TOKEN),
        ("--confirm-camera-fixed", args.confirm_camera_fixed, CAMERA_FIXED_TOKEN),
    )
    missing = [
        "{} {}".format(flag, token)
        for flag, actual, token in expected
        if actual != token
    ]
    if missing:
        raise ValueError("exact confirmations required: " + "; ".join(missing))


def _require_uncontended_franka_link(robot_ip: str) -> None:
    """Run the read-only Desk/HTTPS contention gate on the dedicated link."""

    workspace_root = Path(__file__).resolve().parents[3]
    dexgrasp_src = workspace_root / "dexgrasp" / "src"
    if str(dexgrasp_src) not in sys.path:
        sys.path.insert(0, str(dexgrasp_src))
    from anydex_pipeline.host_network_preflight import (  # noqa: PLC0415
        require_uncontended_franka_https_link,
    )

    require_uncontended_franka_https_link(str(robot_ip))


def _connect_franka(preview: PosePreview, robot_ip: str) -> Any:
    """Lazy hardware import; callers must complete every offline gate first."""

    if preview.dynamics is None:
        raise RuntimeError("strict end-effector dynamics are unavailable")
    workspace_root = Path(__file__).resolve().parents[3]
    dexgrasp_src = workspace_root / "dexgrasp" / "src"
    if str(dexgrasp_src) not in sys.path:
        sys.path.insert(0, str(dexgrasp_src))

    # Run before importing the motion driver and opening FCI.  A second check
    # immediately before move_pose closes the connection-to-motion race.
    _require_uncontended_franka_link(str(robot_ip))

    from anydex_pipeline.franka_sequence_driver import (  # noqa: PLC0415
        FrankaMotionLimits,
        FrankaSequenceDriver,
    )

    dynamics = preview.dynamics
    limits = FrankaMotionLimits(
        expected_F_T_EE=dynamics.expected_F_T_EE,
        expected_m_ee_kg=dynamics.expected_m_ee_kg,
        expected_F_x_Cee_m=dynamics.expected_F_x_Cee_m,
        expected_I_ee_kg_m2=dynamics.expected_I_ee_kg_m2,
        joint_limit_margin_rad=0.05,
        workspace_min_m=preview.workspace_min_m,
        workspace_max_m=preview.workspace_max_m,
        max_cartesian_speed_m_s=MAX_CARTESIAN_SPEED_M_S,
        max_angular_speed_rad_s=MAX_ANGULAR_SPEED_RAD_S,
        max_segment_translation_m=MAX_SEGMENT_TRANSLATION_M,
        max_segment_rotation_rad=MAX_SEGMENT_ROTATION_RAD,
        min_cartesian_duration_s=MIN_CARTESIAN_DURATION_S,
        cartesian_endpoint_hold_s=CARTESIAN_ENDPOINT_HOLD_S,
        cartesian_pose_controller_mode=CARTESIAN_POSE_CONTROLLER_MODE,
        translation_arrival_tolerance_m=0.003,
        rotation_arrival_tolerance_rad=0.010,
        settle_time_s=max(1.0, preview.settle_time_s),
        settle_timeout_s=max(4.0, preview.settle_time_s + 2.0),
        settle_poll_s=0.010,
        settle_max_dq_rad_s=0.015,
        stop_verify_timeout_s=2.0,
        stop_verify_poll_s=0.020,
        stop_verify_max_dq_rad_s=0.010,
        stop_verify_consecutive_samples=5,
        max_dynamic_segments=32,
    )
    return FrankaSequenceDriver.connect(
        str(robot_ip), limits, enforce_realtime=True
    )


def _connect_joint_recovery(
    preview: JointRecoveryPreview, robot_ip: str
) -> Any:
    """Lazy FCI connection with a joint-only recovery envelope."""

    if preview.dynamics is None:
        raise RuntimeError("strict end-effector dynamics are unavailable")
    workspace_root = Path(__file__).resolve().parents[3]
    dexgrasp_src = workspace_root / "dexgrasp" / "src"
    if str(dexgrasp_src) not in sys.path:
        sys.path.insert(0, str(dexgrasp_src))
    _require_uncontended_franka_link(str(robot_ip))
    from anydex_pipeline.franka_sequence_driver import (  # noqa: PLC0415
        FrankaMotionLimits,
        FrankaSequenceDriver,
    )

    dynamics = preview.dynamics
    limits = FrankaMotionLimits(
        expected_F_T_EE=dynamics.expected_F_T_EE,
        expected_m_ee_kg=dynamics.expected_m_ee_kg,
        expected_F_x_Cee_m=dynamics.expected_F_x_Cee_m,
        expected_I_ee_kg_m2=dynamics.expected_I_ee_kg_m2,
        joint_limit_margin_rad=JOINT_LIMIT_MARGIN_RAD,
        max_joint_speed_rad_s=preview.maximum_joint_velocity_rad_s,
        max_joint_segment_rad=preview.maximum_live_to_target_delta_rad,
        min_joint_duration_s=preview.minimum_joint_duration_s,
        workspace_min_m=preview.workspace_min_m,
        workspace_max_m=preview.workspace_max_m,
        joint_arrival_tolerance_rad=JOINT_ENDPOINT_TOLERANCE_RAD,
        max_continuous_joint_tracking_error_rad=0.010,
        settle_time_s=0.5,
        settle_timeout_s=2.0,
        settle_poll_s=0.010,
        settle_max_dq_rad_s=ENDPOINT_MAX_DQ_RAD_S,
        stop_verify_timeout_s=2.0,
        stop_verify_poll_s=0.020,
        stop_verify_max_dq_rad_s=ENDPOINT_MAX_DQ_RAD_S,
        stop_verify_consecutive_samples=5,
        max_dynamic_segments=JOINT_RECOVERY_MAX_DYNAMIC_SEGMENTS,
    )
    return FrankaSequenceDriver.connect(
        str(robot_ip), limits, enforce_realtime=True
    )


def _state_pose(state: Any) -> np.ndarray:
    raw = np.asarray(getattr(state, "O_T_EE", None), dtype=np.float64)
    if raw.shape == (16,):
        raw = raw.reshape((4, 4), order="F")
    return _rigid_transform(raw, "live state.O_T_EE")


_CONTROL_LOOP_TELEMETRY_FIELDS = (
    "kind",
    "samples",
    "max_read_to_write_ns",
    "read_to_write_overruns",
    "success_qualified",
    "success_qualification_positive_writes",
    "success_qualification_control_time_s",
    "success_qualification_wall_time_s",
    "success_qualification_rate",
)


def _control_loop_telemetry_payload(arm: Any) -> Optional[Mapping[str, Any]]:
    """Copy the driver's post-handle evidence outside the realtime loop."""

    telemetry = getattr(arm, "last_control_loop_telemetry", None)
    if telemetry is None:
        return None
    payload = {}
    for field_name in _CONTROL_LOOP_TELEMETRY_FIELDS:
        if not hasattr(telemetry, field_name):
            return None
        payload[field_name] = getattr(telemetry, field_name)
    return payload


def _validate_live_workspace(preview: PosePreview, live_pose: np.ndarray) -> None:
    position = live_pose[:3, 3]
    if np.any(position < preview.workspace_min_m - NUMERICAL_TOLERANCE) or np.any(
        position > preview.workspace_max_m + NUMERICAL_TOLERANCE
    ):
        raise RuntimeError(
            "live EEF position {} is outside hard workspace [{}, {}]".format(
                position.tolist(),
                preview.workspace_min_m.tolist(),
                preview.workspace_max_m.tolist(),
            )
        )
    if position[2] < preview.minimum_eef_z_m - NUMERICAL_TOLERANCE:
        raise RuntimeError(
            "live EEF z {:.9f}m is below minimum {:.9f}m".format(
                position[2], preview.minimum_eef_z_m
            )
        )


def _validate_live_start(preview: PosePreview, arm: Any) -> None:
    if preview.start_pose_id is None or preview.T_start is None:
        raise RuntimeError("authorized planned start pose is unavailable")
    state = arm.robot.read_once()
    arm._validate_state(state, require_idle=True, enforce_success=False)
    live_pose = _state_pose(state)
    _validate_live_workspace(preview, live_pose)
    translation_error = float(
        np.linalg.norm(live_pose[:3, 3] - preview.T_start[:3, 3])
    )
    rotation_error = _rotation_error_rad(preview.T_start, live_pose)
    if (
        translation_error > START_TRANSLATION_TOLERANCE_M
        or rotation_error > START_ROTATION_TOLERANCE_RAD
    ):
        raise RuntimeError(
            "live pose does not match authorized planned start {}: "
            "translation={:.6f}m limit={:.6f}m, "
            "rotation={:.6f}deg limit={:.6f}deg".format(
                preview.start_pose_id,
                translation_error,
                START_TRANSLATION_TOLERANCE_M,
                math.degrees(rotation_error),
                math.degrees(START_ROTATION_TOLERANCE_RAD),
            )
        )


def execute_one_pose(
    preview: PosePreview,
    robot_ip: str,
    *,
    connector: Callable[[PosePreview, str], Any] = _connect_franka,
    link_preflight: Callable[[str], None] = _require_uncontended_franka_link,
    hold_probe_duration_s: Optional[float] = None,
) -> MotionResult:
    """Execute one target or exact-start hold, prove settling, then stop."""

    if not preview.run_authorized:
        raise ValueError(
            "run requires an explicit single-pose authorization bound to {}".format(
                preview.pose_id
            )
        )
    if preview.start_pose_id is None or preview.T_start is None:
        raise ValueError("run requires an authorization-bound planned start pose")
    if preview.dynamics is None:
        raise ValueError("run requires complete strict end-effector dynamics")
    hold_duration = None
    if hold_probe_duration_s is not None:
        hold_duration = float(hold_probe_duration_s)
        if not np.isfinite(hold_duration) or not 1.0 <= hold_duration <= 10.0:
            raise ValueError("hold probe duration must be in [1, 10] seconds")
    if _sha256_file(preview.plan_path) != preview.plan_sha256:
        raise RuntimeError("session plan changed before hardware import")

    arm = connector(preview, str(robot_ip))
    failure: Optional[BaseException] = None
    failure_tb: Optional[TracebackType] = None
    result: Optional[MotionResult] = None
    control_loop_telemetry: Optional[Mapping[str, Any]] = None
    try:
        if _sha256_file(preview.plan_path) != preview.plan_sha256:
            raise RuntimeError("session plan changed after connection")
        _validate_live_start(preview, arm)
        link_preflight(str(robot_ip))
        if hold_duration is None:
            arm.move_pose(preview.T_target)
        else:
            arm.hold_pose_control(preview.T_target, hold_duration)
        control_loop_telemetry = _control_loop_telemetry_payload(arm)
        settled = arm.verify_settled(preview.T_target)
        if settled is not True:
            raise RuntimeError(
                "pose {} did not satisfy the consecutive settle gate".format(
                    preview.pose_id
                )
            )
        result = MotionResult(
            plan_sha256=preview.plan_sha256,
            pose_id=preview.pose_id,
            T_target=preview.T_target,
            capture_ready=hold_duration is None,
            camera_capture_performed=False,
            control_loop_telemetry=control_loop_telemetry,
            diagnostic_hold_completed=hold_duration is not None,
        )
    except BaseException as exc:
        failure = exc
        failure_tb = sys.exc_info()[2]
    finally:
        try:
            arm.stop()
        except BaseException as stop_exc:
            if failure is None:
                failure = RuntimeError("STOP UNCONFIRMED: {}".format(stop_exc))
                failure_tb = sys.exc_info()[2]
            else:
                combined = RuntimeError(
                    "{}: {}; STOP UNCONFIRMED: {}".format(
                        type(failure).__name__, failure, stop_exc
                    )
                )
                setattr(combined, "motion_error", failure)
                setattr(combined, "stop_error", stop_exc)
                failure = combined
                failure_tb = None
    if control_loop_telemetry is None:
        control_loop_telemetry = _control_loop_telemetry_payload(arm)
    if failure is not None:
        if control_loop_telemetry is not None:
            try:
                setattr(failure, "control_loop_telemetry", control_loop_telemetry)
            except Exception:
                pass
        raise failure.with_traceback(failure_tb)
    if result is None:
        raise RuntimeError("internal error: no motion result")
    return result


def _write_json_fd(fd: int, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    offset = 0
    while offset < len(data):
        written = os.write(fd, data[offset:])
        if written <= 0:
            raise OSError("short write while recording authorization consumption")
        offset += written
    os.fsync(fd)


def _claim_joint_recovery_authorization(
    preview: JointRecoveryPreview,
) -> Mapping[str, Any]:
    """Atomically consume an authorization before any hardware import."""

    if _sha256_file(preview.plan_path) != preview.plan_sha256:
        raise RuntimeError("joint recovery plan changed before authorization claim")
    receipt = preview.authorization_receipt_path
    payload = {
        "kind": "franka_bounded_joint_recovery_authorization_consumption",
        "schema_version": 1,
        "plan": str(preview.plan_path),
        "plan_sha256": preview.plan_sha256,
        "recovery_id": preview.recovery_id,
        "commanded_delta_rad": preview.commanded_delta_rad.tolist(),
        "status": "claimed_before_hardware_import",
        "claimed_unix_time_ns": time.time_ns(),
        "authorization_consumed": True,
        "camera_capture_performed": False,
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(str(receipt), flags, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            "bounded joint recovery authorization already consumed by {}".format(
                receipt
            )
        ) from exc
    try:
        _write_json_fd(fd, payload)
    except BaseException:
        try:
            os.close(fd)
        finally:
            # The exclusive path intentionally remains present on a partial
            # write: fail closed rather than making the authorization replayable.
            pass
        raise
    os.close(fd)
    return payload


def _finalize_joint_recovery_claim(
    preview: JointRecoveryPreview,
    claim: Mapping[str, Any],
    *,
    status: str,
    error: Optional[BaseException],
    effective_target_q: Optional[np.ndarray],
    endpoint_q: Optional[np.ndarray],
) -> None:
    payload = dict(claim)
    payload.update(
        {
            "status": status,
            "finalized_unix_time_ns": time.time_ns(),
            "error": None
            if error is None
            else "{}: {}".format(type(error).__name__, error)[:2000],
            "effective_target_q_rad": None
            if effective_target_q is None
            else np.asarray(effective_target_q, dtype=np.float64).tolist(),
            "endpoint_q_rad": None
            if endpoint_q is None
            else np.asarray(endpoint_q, dtype=np.float64).tolist(),
        }
    )
    receipt = preview.authorization_receipt_path
    temporary = receipt.with_name(
        receipt.name + ".tmp-{}-{}".format(os.getpid(), time.time_ns())
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(str(temporary), flags, 0o600)
    try:
        _write_json_fd(fd, payload)
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    os.replace(str(temporary), str(receipt))


def _joint_state_values(state: Any) -> Tuple[np.ndarray, np.ndarray]:
    q = _finite_vector(getattr(state, "q", None), 7, "live state.q")
    dq = _finite_vector(getattr(state, "dq", None), 7, "live state.dq")
    return q, dq


def _validate_joint_recovery_pose_workspace(
    preview: JointRecoveryPreview, pose: np.ndarray, label: str
) -> None:
    position = pose[:3, 3]
    if np.any(position < preview.workspace_min_m - NUMERICAL_TOLERANCE) or np.any(
        position > preview.workspace_max_m + NUMERICAL_TOLERANCE
    ):
        raise RuntimeError(
            "{} EEF position {} is outside hard workspace [{}, {}]".format(
                label,
                position.tolist(),
                preview.workspace_min_m.tolist(),
                preview.workspace_max_m.tolist(),
            )
        )
    if position[2] < preview.minimum_eef_z_m - NUMERICAL_TOLERANCE:
        raise RuntimeError(
            "{} EEF z {:.9f}m is below minimum {:.9f}m".format(
                label, position[2], preview.minimum_eef_z_m
            )
        )


def _validate_joint_recovery_live_start(
    preview: JointRecoveryPreview, arm: Any
) -> Tuple[np.ndarray, np.ndarray]:
    state = arm.robot.read_once()
    arm._validate_state(state, require_idle=True, enforce_success=False)
    live_q, live_dq = _joint_state_values(state)
    maximum_q_error = float(
        np.max(np.abs(live_q - preview.expected_start_q_rad))
    )
    if maximum_q_error > START_Q_TOLERANCE_RAD + NUMERICAL_TOLERANCE:
        raise RuntimeError(
            "live q does not match exact authorized start: max_error={:.9f}rad "
            "limit={:.9f}rad".format(maximum_q_error, START_Q_TOLERANCE_RAD)
        )
    maximum_dq = float(np.max(np.abs(live_dq)))
    if maximum_dq > MAX_START_DQ_RAD_S + NUMERICAL_TOLERANCE:
        raise RuntimeError(
            "live start is not stationary: max|dq|={:.9f}rad/s limit={:.9f}rad/s".format(
                maximum_dq, MAX_START_DQ_RAD_S
            )
        )
    live_pose = _state_pose(state)
    _validate_joint_recovery_pose_workspace(preview, live_pose, "live start")
    translation_error = float(
        np.linalg.norm(
            live_pose[:3, 3] - preview.expected_start_T_base_ee[:3, 3]
        )
    )
    rotation_error = _rotation_error_rad(
        preview.expected_start_T_base_ee, live_pose
    )
    if (
        translation_error > START_POSE_TRANSLATION_TOLERANCE_M
        or rotation_error > START_POSE_ROTATION_TOLERANCE_RAD
    ):
        raise RuntimeError(
            "live EEF pose does not match exact authorized start: "
            "translation={:.6f}m limit={:.6f}m, rotation={:.6f}deg "
            "limit={:.6f}deg".format(
                translation_error,
                START_POSE_TRANSLATION_TOLERANCE_M,
                math.degrees(rotation_error),
                math.degrees(START_POSE_ROTATION_TOLERANCE_RAD),
            )
        )
    return live_q, live_pose


def _verify_joint_recovery_endpoint(
    preview: JointRecoveryPreview,
    arm: Any,
    effective_target_q: np.ndarray,
    *,
    sleep: Callable[[float], None],
) -> Tuple[np.ndarray, np.ndarray]:
    endpoint_q = None
    endpoint_pose = None
    for sample_index in range(ENDPOINT_CONSECUTIVE_SAMPLES):
        state = arm.robot.read_once()
        arm._validate_state(state, require_idle=True, enforce_success=False)
        q, dq = _joint_state_values(state)
        q_error = float(np.max(np.abs(q - effective_target_q)))
        if q_error > JOINT_ENDPOINT_TOLERANCE_RAD + NUMERICAL_TOLERANCE:
            raise RuntimeError(
                "joint recovery endpoint q error {:.9f}rad exceeds {:.9f}rad".format(
                    q_error, JOINT_ENDPOINT_TOLERANCE_RAD
                )
            )
        maximum_dq = float(np.max(np.abs(dq)))
        if maximum_dq > ENDPOINT_MAX_DQ_RAD_S + NUMERICAL_TOLERANCE:
            raise RuntimeError(
                "joint recovery endpoint max|dq| {:.9f}rad/s exceeds {:.9f}rad/s".format(
                    maximum_dq, ENDPOINT_MAX_DQ_RAD_S
                )
            )
        pose = _state_pose(state)
        _validate_joint_recovery_pose_workspace(preview, pose, "live endpoint")
        translation_error = float(
            np.linalg.norm(
                pose[:3, 3] - preview.expected_target_T_base_ee[:3, 3]
            )
        )
        rotation_error = _rotation_error_rad(
            preview.expected_target_T_base_ee, pose
        )
        if (
            translation_error > ENDPOINT_POSE_TRANSLATION_TOLERANCE_M
            or rotation_error > ENDPOINT_POSE_ROTATION_TOLERANCE_RAD
        ):
            raise RuntimeError(
                "joint recovery endpoint EEF mismatch: translation={:.6f}m, "
                "rotation={:.6f}deg".format(
                    translation_error, math.degrees(rotation_error)
                )
            )
        endpoint_q = q
        endpoint_pose = pose
        if sample_index + 1 < ENDPOINT_CONSECUTIVE_SAMPLES:
            sleep(ENDPOINT_POLL_S)
    if endpoint_q is None or endpoint_pose is None:
        raise RuntimeError("internal error: no endpoint samples")
    return endpoint_q, endpoint_pose


def execute_joint_recovery(
    preview: JointRecoveryPreview,
    robot_ip: str,
    *,
    connector: Callable[[JointRecoveryPreview, str], Any] = _connect_joint_recovery,
    link_preflight: Callable[[str], None] = _require_uncontended_franka_link,
    sleep: Callable[[float], None] = time.sleep,
) -> JointRecoveryResult:
    """Consume and execute one SHA-bound, bounded joint-vector recovery."""

    if not preview.run_authorized:
        raise ValueError("run requires an unconsumed bounded joint recovery authorization")
    if preview.dynamics is None:
        raise ValueError("run requires complete strict end-effector dynamics")
    claim = _claim_joint_recovery_authorization(preview)
    arm = None
    failure: Optional[BaseException] = None
    failure_tb: Optional[TracebackType] = None
    result: Optional[JointRecoveryResult] = None
    effective_target_q = None
    endpoint_q = None
    try:
        if _sha256_file(preview.plan_path) != preview.plan_sha256:
            raise RuntimeError("joint recovery plan changed after authorization claim")
        arm = connector(preview, str(robot_ip))
        if _sha256_file(preview.plan_path) != preview.plan_sha256:
            raise RuntimeError("joint recovery plan changed after FCI connection")
        live_q, _ = _validate_joint_recovery_live_start(preview, arm)
        link_preflight(str(robot_ip))
        effective_target_q = preview.planned_target_q_rad.copy()
        _validate_joint_margin(effective_target_q, "effective recovery target")
        # Command the immutable absolute target, never a live-relative target.
        # The start-q tolerance is included in the one-segment bound.
        arm.move_joints(effective_target_q)
        telemetry = _control_loop_telemetry_payload(arm)
        endpoint_q, endpoint_pose = _verify_joint_recovery_endpoint(
            preview, arm, effective_target_q, sleep=sleep
        )
        result = JointRecoveryResult(
            plan_sha256=preview.plan_sha256,
            recovery_id=preview.recovery_id,
            effective_target_q_rad=effective_target_q.copy(),
            endpoint_q_rad=endpoint_q.copy(),
            endpoint_T_base_ee=endpoint_pose.copy(),
            authorization_receipt_path=preview.authorization_receipt_path,
            control_loop_telemetry=telemetry,
        )
    except BaseException as exc:
        failure = exc
        failure_tb = sys.exc_info()[2]
    finally:
        if arm is not None:
            try:
                arm.stop()
            except BaseException as stop_exc:
                if failure is None:
                    failure = RuntimeError("STOP UNCONFIRMED: {}".format(stop_exc))
                    failure_tb = sys.exc_info()[2]
                else:
                    combined = RuntimeError(
                        "{}: {}; STOP UNCONFIRMED: {}".format(
                            type(failure).__name__, failure, stop_exc
                        )
                    )
                    setattr(combined, "motion_error", failure)
                    setattr(combined, "stop_error", stop_exc)
                    failure = combined
                    failure_tb = None
        try:
            _finalize_joint_recovery_claim(
                preview,
                claim,
                status=("succeeded" if failure is None else "failed"),
                error=failure,
                effective_target_q=effective_target_q,
                endpoint_q=endpoint_q,
            )
        except BaseException as receipt_exc:
            if failure is None:
                failure = RuntimeError(
                    "AUTHORIZATION RECEIPT FINALIZATION FAILED: {}".format(
                        receipt_exc
                    )
                )
                failure_tb = sys.exc_info()[2]
            else:
                failure = RuntimeError(
                    "{}: {}; AUTHORIZATION RECEIPT FINALIZATION FAILED: {}".format(
                        type(failure).__name__, failure, receipt_exc
                    )
                )
                failure_tb = None
    if failure is not None:
        raise failure.with_traceback(failure_tb)
    if result is None:
        raise RuntimeError("internal error: no joint recovery result")
    return result


def _preview_dict(preview: PosePreview) -> Mapping[str, Any]:
    return {
        "camera_capture_performed": False,
        "dynamics_error": preview.dynamics_error,
        "dynamics_ready": preview.dynamics is not None,
        "minimum_eef_z_m": preview.minimum_eef_z_m,
        "motion_authorized": preview.motion_authorized,
        "run_authorized": preview.run_authorized,
        "authorization_error": preview.authorization_error,
        "plan": str(preview.plan_path),
        "plan_sha256": preview.plan_sha256,
        "pose_id": preview.pose_id,
        "pose_set": preview.pose_set,
        "start_pose_id": preview.start_pose_id,
        "start_T_base_ee": (
            None if preview.T_start is None else preview.T_start.tolist()
        ),
        "rotation_angle_deg": math.degrees(preview.rotation_angle_rad),
        "rotation_vector_eef_deg": np.degrees(
            preview.rotation_vector_eef_rad
        ).tolist(),
        "session_slug": preview.session_slug,
        "status": preview.plan_status,
        "target_T_base_ee": preview.T_target.tolist(),
        "target_xyz_base_m": preview.T_target[:3, 3].tolist(),
        "translation_norm_m": preview.translation_norm_m,
        "workspace_max_m": preview.workspace_max_m.tolist(),
        "workspace_min_m": preview.workspace_min_m.tolist(),
        "xyz_offset_base_m": preview.xyz_offset_base_m.tolist(),
    }


def _joint_recovery_preview_dict(
    preview: JointRecoveryPreview,
) -> Mapping[str, Any]:
    return {
        "authorization_error": preview.authorization_error,
        "authorization_receipt_exists": preview.authorization_receipt_exists,
        "authorization_receipt_path": str(preview.authorization_receipt_path),
        "camera_capture_performed": False,
        "commanded_delta_deg": np.degrees(preview.commanded_delta_rad).tolist(),
        "commanded_delta_rad": preview.commanded_delta_rad.tolist(),
        "dynamics_error": preview.dynamics_error,
        "dynamics_ready": preview.dynamics is not None,
        "expected_peak_joint_acceleration_rad_s2": (
            preview.expected_peak_joint_acceleration_rad_s2
        ),
        "expected_peak_joint_velocity_rad_s": (
            preview.expected_peak_joint_velocity_rad_s
        ),
        "expected_start_T_base_ee": preview.expected_start_T_base_ee.tolist(),
        "expected_start_q_rad": preview.expected_start_q_rad.tolist(),
        "expected_target_T_base_ee": preview.expected_target_T_base_ee.tolist(),
        "minimum_eef_z_m": preview.minimum_eef_z_m,
        "motion_authorized": preview.motion_authorized,
        "plan": str(preview.plan_path),
        "plan_sha256": preview.plan_sha256,
        "planned_target_q_rad": preview.planned_target_q_rad.tolist(),
        "predicted_eef_rotation_deg": math.degrees(
            preview.predicted_eef_rotation_rad
        ),
        "predicted_eef_translation_m": preview.predicted_eef_translation_m,
        "recovery_id": preview.recovery_id,
        "run_authorized": preview.run_authorized,
        "session_slug": preview.session_slug,
        "status": preview.plan_status,
        "sweep": {
            "board_max_extent_from_eef_m": preview.board_max_extent_from_eef_m,
            "conservative_board_sweep_displacement_m": (
                preview.conservative_board_sweep_displacement_m
            ),
        },
        "workspace_max_m": preview.workspace_max_m.tolist(),
        "workspace_min_m": preview.workspace_min_m.tolist(),
    }


def _print_preview(preview: PosePreview, as_json: bool) -> None:
    payload = _preview_dict(preview)
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print("[calibration pose preview] OFFLINE ONLY; no hardware imported or connected")
    print("  plan={} sha256={}".format(preview.plan_path, preview.plan_sha256))
    print(
        "  status={} motion_authorized={} run_authorized={} session={}".format(
            preview.plan_status,
            preview.motion_authorized,
            preview.run_authorized,
            preview.session_slug,
        )
    )
    print(
        "  pose={} set={} offset_base_m={} norm={:.6f}m".format(
            preview.pose_id,
            preview.pose_set,
            np.round(preview.xyz_offset_base_m, 9).tolist(),
            preview.translation_norm_m,
        )
    )
    print("  authorized_start_pose={}".format(preview.start_pose_id))
    print(
        "  rotvec_eef_deg={} angle={:.6f}deg".format(
            np.round(np.degrees(preview.rotation_vector_eef_rad), 9).tolist(),
            math.degrees(preview.rotation_angle_rad),
        )
    )
    print(
        "  target_xyz_base_m={} min_z={:.6f}m workspace=[{}, {}]".format(
            np.round(preview.T_target[:3, 3], 9).tolist(),
            preview.minimum_eef_z_m,
            preview.workspace_min_m.tolist(),
            preview.workspace_max_m.tolist(),
        )
    )
    print("  target_T_base_ee={}".format(np.round(preview.T_target, 9).tolist()))
    if preview.dynamics is None:
        print("  RUN BLOCKED: missing/invalid strict dynamics: {}".format(preview.dynamics_error))
    else:
        print("  strict_dynamics=READY")
    print("  camera_capture_performed=false")


def _print_joint_recovery_preview(
    preview: JointRecoveryPreview, as_json: bool
) -> None:
    payload = _joint_recovery_preview_dict(preview)
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print("[joint recovery preview] OFFLINE ONLY; no hardware imported or connected")
    print("  plan={} sha256={}".format(preview.plan_path, preview.plan_sha256))
    print(
        "  recovery={} delta_deg={} status={} run_authorized={}".format(
            preview.recovery_id,
            np.round(np.degrees(preview.commanded_delta_rad), 6).tolist(),
            preview.plan_status,
            preview.run_authorized,
        )
    )
    print(
        "  predicted_eef_translation={:.6f}m rotation={:.6f}deg "
        "target_z={:.6f}m min_z={:.6f}m".format(
            preview.predicted_eef_translation_m,
            math.degrees(preview.predicted_eef_rotation_rad),
            preview.expected_target_T_base_ee[2, 3],
            preview.minimum_eef_z_m,
        )
    )
    print(
        "  board_sweep_bound={:.6f}m peak_velocity={:.6f}rad/s "
        "peak_acceleration={:.6f}rad/s^2".format(
            preview.conservative_board_sweep_displacement_m,
            preview.expected_peak_joint_velocity_rad_s,
            preview.expected_peak_joint_acceleration_rad_s2,
        )
    )
    print(
        "  one_use_receipt={} exists={}".format(
            preview.authorization_receipt_path,
            preview.authorization_receipt_exists,
        )
    )
    if preview.authorization_error is not None:
        print("  RUN BLOCKED: {}".format(preview.authorization_error))
    if preview.dynamics is None:
        print("  RUN BLOCKED: missing/invalid strict dynamics: {}".format(preview.dynamics_error))
    print("  camera_capture_performed=false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline-preview or execute exactly one session-plan calibration pose; "
            "this utility never captures a camera frame."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preview_parser = subparsers.add_parser("preview", help="offline validation only")
    preview_parser.add_argument("--plan", type=Path, required=True)
    preview_parser.add_argument("--pose-id", required=True)
    preview_parser.add_argument("--expect-plan-sha256")
    preview_parser.add_argument("--json", action="store_true", dest="as_json")

    run_parser = subparsers.add_parser("run", help="execute one explicitly authorized pose")
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--pose-id", required=True)
    run_parser.add_argument("--robot-ip")
    run_parser.add_argument("--confirm-plan-sha256")
    run_parser.add_argument("--confirm-pose-id")
    run_parser.add_argument("--confirm-e-stop", metavar=E_STOP_TOKEN)
    run_parser.add_argument("--confirm-swept-volume", metavar=SWEPT_VOLUME_TOKEN)
    run_parser.add_argument("--confirm-target-rigid", metavar=TARGET_RIGID_TOKEN)
    run_parser.add_argument("--confirm-camera-fixed", metavar=CAMERA_FIXED_TOKEN)

    probe_parser = subparsers.add_parser(
        "probe-hold",
        help="run one explicitly authorized exact-start active-control hold",
    )
    probe_parser.add_argument("--plan", type=Path, required=True)
    probe_parser.add_argument("--pose-id", required=True)
    probe_parser.add_argument("--robot-ip")
    probe_parser.add_argument("--confirm-plan-sha256")
    probe_parser.add_argument("--confirm-pose-id")
    probe_parser.add_argument("--confirm-e-stop", metavar=E_STOP_TOKEN)
    probe_parser.add_argument("--confirm-swept-volume", metavar=SWEPT_VOLUME_TOKEN)
    probe_parser.add_argument("--confirm-target-rigid", metavar=TARGET_RIGID_TOKEN)
    probe_parser.add_argument("--confirm-camera-fixed", metavar=CAMERA_FIXED_TOKEN)

    joint_preview_parser = subparsers.add_parser(
        "preview-joint-recovery",
        help="offline validation of one dedicated bounded joint recovery plan",
    )
    joint_preview_parser.add_argument("--plan", type=Path, required=True)
    joint_preview_parser.add_argument("--expect-plan-sha256")
    joint_preview_parser.add_argument("--json", action="store_true", dest="as_json")

    joint_run_parser = subparsers.add_parser(
        "run-joint-recovery",
        help="consume and execute one explicitly authorized bounded joint recovery",
    )
    joint_run_parser.add_argument("--plan", type=Path, required=True)
    joint_run_parser.add_argument("--robot-ip")
    joint_run_parser.add_argument("--confirm-plan-sha256")
    joint_run_parser.add_argument("--confirm-recovery-id")
    joint_run_parser.add_argument("--confirm-e-stop", metavar=E_STOP_TOKEN)
    joint_run_parser.add_argument(
        "--confirm-swept-volume", metavar=JOINT_RECOVERY_SWEEP_TOKEN
    )
    joint_run_parser.add_argument(
        "--confirm-target-rigid", metavar=TARGET_RIGID_TOKEN
    )
    joint_run_parser.add_argument(
        "--confirm-camera-fixed", metavar=CAMERA_FIXED_TOKEN
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command in ("preview-joint-recovery", "run-joint-recovery"):
            preview = load_joint_recovery_preview(
                args.plan,
                expected_sha256=(
                    args.expect_plan_sha256
                    if args.command == "preview-joint-recovery"
                    else None
                ),
                for_run=args.command == "run-joint-recovery",
            )
            if args.command == "preview-joint-recovery":
                _print_joint_recovery_preview(preview, bool(args.as_json))
                return 0 if preview.dynamics is not None else 2
            _require_joint_recovery_confirmations(args, preview)
            robot_ip = preview.robot_ip if args.robot_ip is None else str(args.robot_ip)
            if robot_ip != preview.robot_ip:
                raise ValueError(
                    "--robot-ip {!r} does not match plan robot IP {!r}".format(
                        robot_ip, preview.robot_ip
                    )
                )
            result = execute_joint_recovery(preview, robot_ip)
            telemetry_text = (
                "none"
                if result.control_loop_telemetry is None
                else json.dumps(
                    result.control_loop_telemetry,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            print(
                "[joint recovery] COMPLETE recovery={} plan_sha256={} "
                "receipt={} camera_capture_performed=false "
                "control_loop_telemetry={}".format(
                    result.recovery_id,
                    result.plan_sha256,
                    result.authorization_receipt_path,
                    telemetry_text,
                )
            )
            return 0

        preview = load_pose_preview(
            args.plan,
            args.pose_id,
            expected_sha256=(
                args.expect_plan_sha256 if args.command == "preview" else None
            ),
            for_run=args.command in ("run", "probe-hold"),
        )
        if args.command == "preview":
            _print_preview(preview, bool(args.as_json))
            return 0 if preview.dynamics is not None else 2

        _require_confirmations(args, preview)
        hardware = _require_mapping(
            _read_plan(preview.plan_path)[2].get("hardware"), "hardware"
        )
        robot = _require_mapping(hardware.get("robot"), "hardware.robot")
        plan_robot_ip = _require_text(robot.get("ip"), "hardware.robot.ip")
        robot_ip = plan_robot_ip if args.robot_ip is None else str(args.robot_ip)
        if robot_ip != plan_robot_ip:
            raise ValueError(
                "--robot-ip {!r} does not match plan robot IP {!r}".format(
                    robot_ip, plan_robot_ip
                )
            )
        result = execute_one_pose(
            preview,
            robot_ip,
            hold_probe_duration_s=(
                DIAGNOSTIC_HOLD_DURATION_S
                if args.command == "probe-hold"
                else None
            ),
        )
        telemetry_text = (
            "none"
            if result.control_loop_telemetry is None
            else json.dumps(
                result.control_loop_telemetry,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        if result.diagnostic_hold_completed:
            print(
                "[calibration pose] HOLD_PROBE_COMPLETE pose={} plan_sha256={} "
                "camera_capture_performed=false control_loop_telemetry={}".format(
                    result.pose_id, result.plan_sha256, telemetry_text
                )
            )
        else:
            print(
                "[calibration pose] CAPTURE_READY pose={} plan_sha256={} "
                "camera_capture_performed=false control_loop_telemetry={}".format(
                    result.pose_id, result.plan_sha256, telemetry_text
                )
            )
    except KeyboardInterrupt:
        print(
            "[calibration pose] interrupted; use the physical emergency stop if "
            "motion remains",
            file=sys.stderr,
        )
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        telemetry = getattr(exc, "control_loop_telemetry", None)
        telemetry_suffix = (
            ""
            if telemetry is None
            else " control_loop_telemetry={}".format(
                json.dumps(telemetry, sort_keys=True, separators=(",", ":"))
            )
        )
        print(
            "[calibration pose] failed: {}{}".format(exc, telemetry_suffix),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
