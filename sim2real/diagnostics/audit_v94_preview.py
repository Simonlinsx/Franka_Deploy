#!/usr/bin/env python3
"""Offline audit for a read-only V94 live-preview NPZ.

This module only opens local files.  It does not import the hardware adapters,
open a camera or serial port, connect to Franka, or contain a command path.
The audit establishes tensor/schema and deterministic policy-replay alignment;
it deliberately does not claim that a policy action is semantically correct or
safe to execute on hardware.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import yaml

if __package__ in (None, ""):
    workspace = Path(__file__).resolve().parents[2]
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    from sim2real.deployment.bundle import DeployBundle, load_checkpoint_safely
    from sim2real.policy import RollingStudentPolicy
    from sim2real.contracts.actions import V94ActionMapper
    from sim2real.contracts.v94 import V94Contract
    from sim2real.observation.model import pose_from_position_quaternion_wxyz
else:
    from sim2real.deployment.bundle import DeployBundle, load_checkpoint_safely
    from sim2real.policy import RollingStudentPolicy
    from sim2real.contracts.actions import V94ActionMapper
    from sim2real.contracts.v94 import V94Contract
    from sim2real.observation.model import pose_from_position_quaternion_wxyz


from sim2real.workspace_paths import DEFAULT_V94_DEPLOY_BUNDLE

DEFAULT_BUNDLE = DEFAULT_V94_DEPLOY_BUNDLE
AUDIT_SCHEMA_VERSION = 1
MAX_AUDIT_FILE_BYTES = 512 * 1024 * 1024
DEFAULT_REPLAY_ATOL = 2.0e-6
INITIAL_ALIGNMENT_NPZ = (
    "alignment/reset_idle_open/initial_observations_and_student_response.npz"
)
CLOSED_ALIGNMENT_NPZ = (
    "alignment/student_closed_loop/closed_loop_observation_action_pairs.npz"
)
RESET_ARM_TOLERANCE_RAD = 0.02
RESET_HAND_OPEN_MIN_UNITS = 980
MAX_TRAINING_PROFILE_BYTES = 1024 * 1024

REQUIRED_FIELDS = {
    "audit_schema_version",
    "pointcloud_history_xyzrgb_palm",
    "pointcloud_valid_history",
    "proprio_history67",
    "pointcloud_current_xyzrgb_palm",
    "pointcloud_current_valid",
    "proprio_current67",
    "raw_policy_action13",
    "predicted_privileged32",
    "predicted_future_motion24",
    "predicted_hold6",
    "predicted_hold_logit",
    "franka_timestamp_s",
    "pointcloud_timestamp_s",
    "rh56_timestamp_s",
    "timestamp_s",
}

ACTION_MAPPING_FIELDS = {
    "executed_policy_action13",
    "franka_target_q_rad",
    "rh56_target_q_policy_order_rad",
    "rh56_proposed_angle_set_register_order",
    "franka_q_rad",
    "rh56_virtual_q_policy_order_rad",
    "proposal_mode",
}
ACTION_MAPPING_TRIGGER_FIELDS = {
    "executed_policy_action13",
    "franka_target_q_rad",
    "rh56_target_q_policy_order_rad",
    "rh56_proposed_angle_set_register_order",
    "proposal_mode",
}


def _scalar_integer(value: np.ndarray, name: str) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be one scalar integer")
    return int(array)


def _float_array(
    values: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
    *,
    exact_float32: bool = False,
) -> np.ndarray:
    value = np.asarray(values[name])
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype.kind != "f":
        raise ValueError(f"{name} must have a floating dtype")
    if exact_float32 and value.dtype != np.dtype(np.float32):
        raise ValueError(f"{name} must use float32, got {value.dtype}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values")
    return value


def _load_audit_npz(path: str | Path) -> dict[str, np.ndarray]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"audit NPZ not found: {source}")
    size = source.stat().st_size
    if size <= 0 or size > MAX_AUDIT_FILE_BYTES:
        raise ValueError(f"audit NPZ has an unsafe file size: {size} bytes")
    try:
        with np.load(source, allow_pickle=False) as archive:
            values = {name: archive[name].copy() for name in archive.files}
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid no-pickle audit NPZ {source}: {exc}") from exc
    missing = sorted(REQUIRED_FIELDS - set(values))
    if missing:
        raise ValueError(f"audit NPZ is missing required fields: {missing}")
    for name, value in values.items():
        if np.asarray(value).dtype.kind == "O":
            raise ValueError(f"audit field {name} has forbidden object dtype")
    version = _scalar_integer(values["audit_schema_version"], "audit_schema_version")
    if version != AUDIT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported audit_schema_version {version}; "
            f"expected {AUDIT_SCHEMA_VERSION}"
        )
    if "hardware_writes" in values:
        hardware_writes = np.asarray(values["hardware_writes"])
        if hardware_writes.shape != () or hardware_writes.dtype.kind != "b":
            raise ValueError("hardware_writes must be one scalar boolean")
        if bool(hardware_writes):
            raise ValueError("audit claims that hardware writes occurred")
    if "robot_command_writes" in values:
        robot_writes = np.asarray(values["robot_command_writes"])
        if robot_writes.shape != () or robot_writes.dtype.kind != "b":
            raise ValueError("robot_command_writes must be one scalar boolean")
        if bool(robot_writes):
            raise ValueError("audit claims that robot command writes occurred")
    if "camera_configuration_writes" in values:
        camera_writes = np.asarray(values["camera_configuration_writes"])
        if camera_writes.shape != () or camera_writes.dtype.kind != "b":
            raise ValueError(
                "camera_configuration_writes must be one scalar boolean"
            )
    if "hardware_writes_semantics" in values:
        semantics = np.asarray(values["hardware_writes_semantics"])
        if semantics.shape != () or semantics.dtype.kind not in "US":
            raise ValueError("hardware_writes_semantics must be one scalar string")
        if str(semantics) != "robot_actuator_or_register_commands_only":
            raise ValueError("hardware_writes_semantics is not recognized")
    if "write" in values and np.any(np.asarray(values["write"], dtype=bool)):
        raise ValueError("audit contains a step with write=True")
    return values


def _strictly_increasing(value: np.ndarray, name: str) -> None:
    if value.size > 1 and np.any(np.diff(value) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")


def _nondecreasing(value: np.ndarray, name: str) -> None:
    if value.size > 1 and np.any(np.diff(value) < 0.0):
        raise ValueError(f"{name} must be nondecreasing")


def _series_stats(value: np.ndarray) -> dict[str, float]:
    array = np.asarray(value, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def _feature_zscore_stats(value: np.ndarray) -> dict[str, list[float]]:
    array = np.asarray(value, dtype=np.float64).reshape(-1, value.shape[-1])
    absolute = np.abs(array)
    return {
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
        "abs_p95": np.percentile(absolute, 95.0, axis=0).tolist(),
        "abs_max": np.max(absolute, axis=0).tolist(),
    }


def _maximum_error(actual: np.ndarray, expected: np.ndarray, name: str) -> float:
    first = np.asarray(actual)
    second = np.asarray(expected)
    if first.shape != second.shape:
        raise ValueError(f"{name} replay shape mismatch: {first.shape}!={second.shape}")
    error = float(np.max(np.abs(first.astype(np.float64) - second.astype(np.float64))))
    if not np.isfinite(error):
        raise ValueError(f"{name} replay error is non-finite")
    return error


def _nearest_row_linf(
    query: np.ndarray, reference: np.ndarray, name: str
) -> np.ndarray:
    first = np.asarray(query, dtype=np.float64)
    second = np.asarray(reference, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError(
            f"{name} requires [N,D] and [M,D] arrays, got "
            f"{first.shape} and {second.shape}"
        )
    if first.shape[0] < 1 or second.shape[0] < 1:
        raise ValueError(f"{name} requires non-empty arrays")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError(f"{name} contains non-finite values")
    pairwise = np.max(np.abs(first[:, None, :] - second[None, :, :]), axis=2)
    return np.min(pairwise, axis=1)


def _symmetric_chamfer_xyz(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape[1] != 3
        or right.shape[1] != 3
        or left.shape[0] < 1
        or right.shape[0] < 1
    ):
        raise ValueError("Chamfer inputs must be non-empty [N,3] arrays")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("Chamfer inputs contain non-finite values")
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    return 0.5 * float(
        np.mean(np.min(distances, axis=1))
        + np.mean(np.min(distances, axis=0))
    )


def _nearest_pointcloud_reference_metrics(
    points: np.ndarray,
    valid: np.ndarray,
    reference_points: np.ndarray,
    reference_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    query_points = np.asarray(points, dtype=np.float64)
    query_valid = np.asarray(valid, dtype=bool)
    refs = np.asarray(reference_points, dtype=np.float64)
    refs_valid = np.asarray(reference_valid, dtype=bool)
    if query_points.ndim != 3 or query_points.shape[1:] != (128, 6):
        raise ValueError("query point clouds must have shape [N,128,6]")
    if query_valid.shape != query_points.shape[:2]:
        raise ValueError("query point validity shape mismatch")
    if refs.ndim != 3 or refs.shape[1:] != (128, 6):
        raise ValueError("reference point clouds must have shape [M,128,6]")
    if refs_valid.shape != refs.shape[:2]:
        raise ValueError("reference point validity shape mismatch")
    reference_xyz = [
        refs[index, refs_valid[index], :3] for index in range(refs.shape[0])
    ]
    if any(item.shape[0] < 1 for item in reference_xyz):
        raise ValueError("reference point cloud contains no valid XYZ")
    chamfer = []
    centroid = []
    for index in range(query_points.shape[0]):
        xyz = query_points[index, query_valid[index], :3]
        if xyz.shape[0] < 1:
            raise ValueError("query point cloud contains no valid XYZ")
        chamfer_candidates = [
            _symmetric_chamfer_xyz(xyz, reference) for reference in reference_xyz
        ]
        centroid_candidates = [
            float(np.linalg.norm(np.mean(xyz, axis=0) - np.mean(reference, axis=0)))
            for reference in reference_xyz
        ]
        chamfer.append(min(chamfer_candidates))
        centroid.append(min(centroid_candidates))
    return np.asarray(chamfer), np.asarray(centroid)


def _quaternion_angular_distance_rad(
    query_wxyz: np.ndarray, reference_wxyz: np.ndarray
) -> np.ndarray:
    query = np.asarray(query_wxyz, dtype=np.float64)
    reference = np.asarray(reference_wxyz, dtype=np.float64)
    if (
        query.ndim != 2
        or reference.ndim != 2
        or query.shape[1:] != (4,)
        or reference.shape[1:] != (4,)
    ):
        raise ValueError("quaternion arrays must have shape [N,4] and [M,4]")
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    reference_norm = np.linalg.norm(reference, axis=1, keepdims=True)
    if np.any(query_norm <= 0.0) or np.any(reference_norm <= 0.0):
        raise ValueError("quaternion arrays contain a zero quaternion")
    dots = np.abs((query / query_norm) @ (reference / reference_norm).T)
    return np.min(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)), axis=1)


def _validate_rigid_history(value: np.ndarray, steps: int) -> np.ndarray:
    poses = _float_array(
        {"poses": value}, "poses", (steps, 4, 4), exact_float32=False
    ).astype(np.float64)
    for index, pose in enumerate(poses):
        if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-7, rtol=0.0):
            raise ValueError(
                f"capture palm pose {index} has an invalid homogeneous row"
            )
        rotation = pose[:3, :3]
        if not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=1.0e-5, rtol=0.0
        ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-5, rtol=0.0):
            raise ValueError(f"capture palm pose {index} is not rigid")
    return poses


def _finite_vector(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be one finite {shape} array")
    return result


def _bundle_training_profile(bundle: DeployBundle) -> tuple[str, Mapping[str, Any]]:
    hardware = bundle.manifest.get("hardware_alignment")
    if not isinstance(hardware, Mapping):
        raise ValueError("bundle manifest hardware_alignment is invalid")
    entry = hardware.get("training_profile")
    if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
        raise ValueError("bundle manifest training_profile is invalid")
    path = str(entry["path"])
    payload = bundle.read_bytes(path, max_bytes=MAX_TRAINING_PROFILE_BYTES)
    try:
        profile = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid bundle training profile {path}: {exc}") from exc
    if not isinstance(profile, Mapping):
        raise ValueError("bundle training profile must contain one mapping")
    return path, profile


def _plane_height_at_xy(plane_abcd: np.ndarray, xy: np.ndarray) -> float:
    plane = _finite_vector(plane_abcd, (4,), "table plane")
    point = _finite_vector(xy, (2,), "table XY")
    if plane[2] == 0.0:
        raise ValueError("bundle table plane cannot have a zero z coefficient")
    return float(-(plane[0] * point[0] + plane[1] * point[1] + plane[3]) / plane[2])


def _object_reset_evidence(bundle: DeployBundle) -> dict[str, Any]:
    """Derive a finite sphere/reset envelope only from serialized evidence.

    ``object_init_pos`` in the validation config is a simulator spawn pose.  It
    is retained for provenance but is not used as the settled sphere centre or
    as a visible-cloud height target.  Settled surface bounds instead come from
    the measured table plane, sphere geometry, reset XY support, scale support,
    and tabletop-height randomization serialized in the training profile.
    """

    validation_path = (
        "validation_video/rolling_student_dynamic_20trial.config.json"
    )
    validation = bundle.read_json(validation_path)
    environment = validation.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("bundle validation.environment must be an object")
    profile_path, profile = _bundle_training_profile(bundle)

    training = profile.get("training_distribution")
    scene = profile.get("scene_alignment")
    dr = profile.get("domain_randomization")
    observation_dr = profile.get("student_observation_domain_randomization")
    if not all(isinstance(item, Mapping) for item in (training, scene, dr, observation_dr)):
        raise ValueError("bundle training profile lacks object/reset DR mappings")
    tabletop = scene.get("tabletop")
    sampling = training.get("object_sampling")
    observation_points = observation_dr.get("palm_frame_pointcloud_at_alpha_1")
    if not all(
        isinstance(item, Mapping)
        for item in (tabletop, sampling, observation_points)
    ):
        raise ValueError("bundle training profile lacks tabletop/object observation evidence")

    validation_shape = str(environment.get("object_shape", ""))
    validation_radius = float(environment.get("object_radius", np.nan))
    validation_size = _finite_vector(
        environment.get("object_size"), (3,), "validation object_size"
    )
    validation_spawn = _finite_vector(
        environment.get("object_init_pos"), (3,), "validation object_init_pos"
    )
    validation_reset_noise = _finite_vector(
        environment.get("reset_object_pos_noise"),
        (3,),
        "validation reset_object_pos_noise",
    )
    if (
        validation_shape != "sphere"
        or not np.isfinite(validation_radius)
        or validation_radius <= 0.0
        or not np.array_equal(
            validation_size,
            np.full(3, 2.0 * validation_radius, dtype=np.float64),
        )
    ):
        raise ValueError("bundle validation object is not one consistent sphere")

    reset_center_xy = _finite_vector(
        training.get("reset_center_robot_base_m"),
        (2,),
        "training reset center",
    )
    reset_half_xy = _finite_vector(
        training.get("reset_xy_half_range_m"),
        (2,),
        "training reset XY half range",
    )
    if np.any(reset_half_xy < 0.0):
        raise ValueError("training reset XY half range cannot be negative")
    if not np.array_equal(reset_center_xy, validation_spawn[:2]) or not np.array_equal(
        reset_half_xy, validation_reset_noise[:2]
    ) or validation_reset_noise[2] != 0.0:
        raise ValueError("training and validation reset supports disagree")

    scale_factors = np.asarray(sampling.get("scale_factors"), dtype=np.float64)
    if (
        scale_factors.ndim != 1
        or scale_factors.size < 1
        or not np.all(np.isfinite(scale_factors))
        or np.any(scale_factors <= 0.0)
    ):
        raise ValueError("training object scale_factors are invalid")
    radius_support = np.asarray(
        [
            validation_radius * float(np.min(scale_factors)),
            validation_radius * float(np.max(scale_factors)),
        ],
        dtype=np.float64,
    )

    plane = _finite_vector(
        tabletop.get("plane_abcd_robot_base"), (4,), "training table plane"
    )
    tabletop_height_offset = _finite_vector(
        dr.get("tabletop_height_offset_m"),
        (2,),
        "training tabletop height offset",
    )
    if tabletop_height_offset[0] > tabletop_height_offset[1]:
        raise ValueError("training tabletop height offset is reversed")

    center_xy_min = reset_center_xy - reset_half_xy
    center_xy_max = reset_center_xy + reset_half_xy
    corner_heights = np.asarray(
        [
            _plane_height_at_xy(plane, np.asarray([x, y]))
            for x in (center_xy_min[0], center_xy_max[0])
            for y in (center_xy_min[1], center_xy_max[1])
        ],
        dtype=np.float64,
    )
    table_height_support = np.asarray(
        [
            float(np.min(corner_heights) + tabletop_height_offset[0]),
            float(np.max(corner_heights) + tabletop_height_offset[1]),
        ],
        dtype=np.float64,
    )
    maximum_radius = float(radius_support[1])
    # This is the union AABB of every surface point on a supported sphere at a
    # supported reset centre.  It is deliberately not a fitted real sphere.
    surface_aabb_min = np.asarray(
        [
            center_xy_min[0] - maximum_radius,
            center_xy_min[1] - maximum_radius,
            table_height_support[0],
        ],
        dtype=np.float64,
    )
    surface_aabb_max = np.asarray(
        [
            center_xy_max[0] + maximum_radius,
            center_xy_max[1] + maximum_radius,
            table_height_support[1] + 2.0 * maximum_radius,
        ],
        dtype=np.float64,
    )
    center_z_support = np.asarray(
        [
            table_height_support[0] + radius_support[0],
            table_height_support[1] + radius_support[1],
        ],
        dtype=np.float64,
    )
    validation_surface_aabb_min = np.asarray(
        [
            center_xy_min[0] - validation_radius,
            center_xy_min[1] - validation_radius,
            float(np.min(corner_heights)),
        ],
        dtype=np.float64,
    )
    validation_surface_aabb_max = np.asarray(
        [
            center_xy_max[0] + validation_radius,
            center_xy_max[1] + validation_radius,
            float(np.max(corner_heights) + 2.0 * validation_radius),
        ],
        dtype=np.float64,
    )

    outlier_probability = float(observation_points.get("outlier_probability", np.nan))
    if (
        not np.isfinite(outlier_probability)
        or outlier_probability < 0.0
        or outlier_probability > 1.0
    ):
        raise ValueError("training point-cloud outlier probability is invalid")

    initial_pose_path = "alignment/reset_idle_open/initial_pose.json"
    initial_pose = bundle.read_json(initial_pose_path)
    privileged = initial_pose.get("object_reference_first_raw_step")
    if not isinstance(privileged, Mapping):
        raise ValueError("bundle initial pose lacks privileged object reference")
    privileged_position = _finite_vector(
        privileged.get("position_base_m"),
        (3,),
        "packaged initial privileged object position",
    )
    privileged_size = _finite_vector(
        privileged.get("size_m"),
        (3,),
        "packaged initial privileged object size",
    )
    if not np.array_equal(
        privileged_size.astype(np.float32), validation_size.astype(np.float32)
    ):
        raise ValueError("packaged initial and validation object geometries disagree")
    privileged_table_height = _plane_height_at_xy(
        plane, privileged_position[:2]
    )
    privileged_contact_residual = float(
        privileged_position[2] - privileged_table_height - validation_radius
    )

    return {
        "training_profile_path": profile_path,
        "validation_config_path": validation_path,
        "packaged_initial_pose_path": initial_pose_path,
        "shape": validation_shape,
        "validation_radius_m": validation_radius,
        "validation_size_m": validation_size,
        "validation_spawn_position_base_m": validation_spawn,
        "reset_center_xy_m": reset_center_xy,
        "reset_xy_half_range_m": reset_half_xy,
        "reset_center_xy_min_m": center_xy_min,
        "reset_center_xy_max_m": center_xy_max,
        "scale_factors": scale_factors,
        "radius_support_m": radius_support,
        "table_plane_abcd_robot_base": plane,
        "tabletop_height_offset_support_m": tabletop_height_offset,
        "table_height_over_reset_xy_support_m": table_height_support,
        "possible_center_z_support_m": center_z_support,
        "possible_surface_aabb_min_m": surface_aabb_min,
        "possible_surface_aabb_max_m": surface_aabb_max,
        "validation_surface_aabb_min_m": validation_surface_aabb_min,
        "validation_surface_aabb_max_m": validation_surface_aabb_max,
        "configured_point_outlier_probability": outlier_probability,
        "packaged_initial_privileged_center_base_m": privileged_position,
        "packaged_initial_privileged_surface_contact_residual_m": (
            privileged_contact_residual
        ),
    }


def _pointclouds_in_base(
    points_xyzrgb_palm: np.ndarray,
    valid: np.ndarray,
    T_base_palm: np.ndarray,
) -> list[np.ndarray]:
    points = np.asarray(points_xyzrgb_palm)
    masks = np.asarray(valid, dtype=bool)
    poses = np.asarray(T_base_palm, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (128, 6):
        raise ValueError("base conversion points must have shape [N,128,6]")
    if masks.shape != points.shape[:2] or poses.shape != (points.shape[0], 4, 4):
        raise ValueError("base conversion point validity/pose shape mismatch")
    result = []
    for index in range(points.shape[0]):
        xyz_palm = points[index, masks[index], :3].astype(np.float64)
        if xyz_palm.shape[0] < 1:
            raise ValueError("base conversion point cloud is empty")
        pose = poses[index]
        result.append(xyz_palm @ pose[:3, :3].T + pose[:3, 3])
    return result


def _optional_series_stats(value: np.ndarray) -> Optional[dict[str, float]]:
    finite = np.asarray(value, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if finite.size == 0 else _series_stats(finite)


def _axis_series_stats(value: np.ndarray) -> dict[str, dict[str, float]]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError("axis statistics require one finite [N,3] array")
    return {
        axis: _series_stats(array[:, index])
        for index, axis in enumerate(("x", "y", "z"))
    }


def _cosine_to_nearest_linf_reference(
    query: np.ndarray, reference: np.ndarray, *, dimensions: slice
) -> np.ndarray:
    first = np.asarray(query, dtype=np.float64)
    second = np.asarray(reference, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("action cosine inputs must be [N,D] and [M,D]")
    pairwise_linf = np.max(
        np.abs(first[:, None, :] - second[None, :, :]), axis=2
    )
    selected = second[np.argmin(pairwise_linf, axis=1), dimensions]
    selected_query = first[:, dimensions]
    denominator = np.linalg.norm(selected_query, axis=1) * np.linalg.norm(
        selected, axis=1
    )
    result = np.full(first.shape[0], np.nan, dtype=np.float64)
    nonzero = denominator > 0.0
    result[nonzero] = np.sum(
        selected_query[nonzero] * selected[nonzero], axis=1
    ) / denominator[nonzero]
    return np.clip(result, -1.0, 1.0)


def audit_v94_preview(
    audit_path: str | Path,
    *,
    bundle_path: str | Path = DEFAULT_BUNDLE,
    replay_atol: float = DEFAULT_REPLAY_ATOL,
) -> dict[str, Any]:
    """Validate and deterministically replay one read-only preview capture."""

    tolerance = float(replay_atol)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("replay_atol must be finite and non-negative")
    values = _load_audit_npz(audit_path)
    points = np.asarray(values["pointcloud_history_xyzrgb_palm"])
    if points.ndim != 4 or points.shape[1:] != (4, 128, 6):
        raise ValueError("pointcloud_history_xyzrgb_palm must have shape [S,4,128,6]")
    steps = int(points.shape[0])
    if steps < 1:
        raise ValueError("audit must contain at least one policy step")
    points = _float_array(
        values,
        "pointcloud_history_xyzrgb_palm",
        (steps, 4, 128, 6),
        exact_float32=True,
    )
    valid = _float_array(
        values,
        "pointcloud_valid_history",
        (steps, 4, 128),
        exact_float32=True,
    )
    proprio = _float_array(
        values, "proprio_history67", (steps, 4, 67), exact_float32=True
    )
    current_points = _float_array(
        values,
        "pointcloud_current_xyzrgb_palm",
        (steps, 128, 6),
        exact_float32=True,
    )
    current_valid = _float_array(
        values,
        "pointcloud_current_valid",
        (steps, 128),
        exact_float32=True,
    )
    current_proprio = _float_array(
        values, "proprio_current67", (steps, 67), exact_float32=True
    )
    logged_action = _float_array(
        values, "raw_policy_action13", (steps, 13), exact_float32=True
    )
    logged_privileged = _float_array(
        values, "predicted_privileged32", (steps, 32), exact_float32=True
    )
    logged_future = _float_array(
        values, "predicted_future_motion24", (steps, 24), exact_float32=True
    )
    logged_hold = _float_array(
        values, "predicted_hold6", (steps, 6), exact_float32=True
    )
    logged_hold_logit = _float_array(
        values, "predicted_hold_logit", (steps,), exact_float32=False
    )

    if not np.array_equal(current_points, points[:, -1]):
        raise ValueError("pointcloud_current_xyzrgb_palm != history[:, -1]")
    if not np.array_equal(current_valid, valid[:, -1]):
        raise ValueError("pointcloud_current_valid != history[:, -1]")
    if not np.array_equal(current_proprio, proprio[:, -1]):
        raise ValueError("proprio_current67 != history[:, -1]")
    if np.any((valid != 0.0) & (valid != 1.0)):
        raise ValueError("pointcloud validity must be exactly binary")
    valid_counts = np.sum(valid, axis=2)
    if np.any(valid_counts < 1.0):
        raise ValueError("every history frame must contain at least one valid point")
    if np.any(points[valid == 0.0] != 0.0):
        raise ValueError("invalid point-cloud slots must be zero padded")
    valid_rgb = points[..., 3:][valid.astype(bool)]
    if np.any(valid_rgb < 0.0) or np.any(valid_rgb > 1.0):
        raise ValueError("valid RGB point features must lie in [0,1]")
    if np.any(logged_action < -1.0) or np.any(logged_action > 1.0):
        raise ValueError("raw_policy_action13 must lie in [-1,1]")

    timestamps = {}
    for name in (
        "franka_timestamp_s",
        "pointcloud_timestamp_s",
        "rh56_timestamp_s",
        "timestamp_s",
    ):
        timestamps[name] = _float_array(values, name, (steps,))
    _strictly_increasing(timestamps["timestamp_s"], "timestamp_s")
    _strictly_increasing(timestamps["franka_timestamp_s"], "franka_timestamp_s")
    _nondecreasing(timestamps["pointcloud_timestamp_s"], "pointcloud_timestamp_s")
    _nondecreasing(timestamps["rh56_timestamp_s"], "rh56_timestamp_s")
    if not np.allclose(
        timestamps["timestamp_s"],
        timestamps["franka_timestamp_s"],
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise ValueError("timestamp_s must identify the Franka observation time")

    bundle = DeployBundle(bundle_path)
    bundle.verify()
    contract = V94Contract.from_bundle(bundle)
    initial_alignment = bundle.load_npz(INITIAL_ALIGNMENT_NPZ)
    closed_alignment = bundle.load_npz(CLOSED_ALIGNMENT_NPZ)
    checkpoint = load_checkpoint_safely(bundle.checkpoint_bytes())
    policy = RollingStudentPolicy(checkpoint)
    replay_action = []
    replay_privileged = []
    replay_future = []
    replay_hold = []
    replay_hold_logit = []
    for index in range(steps):
        output = policy.act(points[index], valid[index], proprio[index])
        replay_action.append(output.action13)
        replay_privileged.append(output.predicted_privileged32)
        replay_future.append(output.future_motion24)
        replay_hold.append(output.predicted_hold6)
        replay_hold_logit.append(output.predicted_hold_logit)
    replay = {
        "raw_policy_action13": np.stack(replay_action),
        "predicted_privileged32": np.stack(replay_privileged),
        "predicted_future_motion24": np.stack(replay_future),
        "predicted_hold6": np.stack(replay_hold),
        "predicted_hold_logit": np.asarray(replay_hold_logit),
    }
    errors = {
        "raw_policy_action13": _maximum_error(
            replay["raw_policy_action13"], logged_action, "raw policy action"
        ),
        "predicted_privileged32": _maximum_error(
            replay["predicted_privileged32"],
            logged_privileged,
            "predicted privileged state",
        ),
        "predicted_future_motion24": _maximum_error(
            replay["predicted_future_motion24"],
            logged_future,
            "predicted future motion",
        ),
        "predicted_hold6": _maximum_error(
            replay["predicted_hold6"], logged_hold, "predicted hold target"
        ),
        "predicted_hold_logit": _maximum_error(
            replay["predicted_hold_logit"],
            logged_hold_logit,
            "predicted hold logit",
        ),
    }
    worst_replay_error = max(errors.values())
    if worst_replay_error > tolerance:
        raise ValueError(
            f"policy replay error {worst_replay_error:.9g} exceeds {tolerance:.9g}"
        )

    mapping_evidence_present = bool(ACTION_MAPPING_TRIGGER_FIELDS.intersection(values))
    if mapping_evidence_present and not ACTION_MAPPING_FIELDS.issubset(values):
        missing_mapping = sorted(ACTION_MAPPING_FIELDS - set(values))
        raise ValueError(
            "incomplete action-target mapping evidence; missing fields: "
            f"{missing_mapping}"
        )
    action_target_mapping: dict[str, Any] = {
        "available": False,
        "passed": None,
        "note": (
            "This audit did not contain the complete one-step target mapping "
            "evidence."
        ),
    }
    if mapping_evidence_present:
        executed_action = _float_array(
            values,
            "executed_policy_action13",
            (steps, 13),
            exact_float32=True,
        )
        franka_q_for_mapping = _float_array(
            values, "franka_q_rad", (steps, 7)
        )
        hand_q_for_mapping = _float_array(
            values,
            "rh56_virtual_q_policy_order_rad",
            (steps, 6),
        )
        logged_arm_target = _float_array(
            values,
            "franka_target_q_rad",
            (steps, 7),
            exact_float32=True,
        )
        logged_hand_target = _float_array(
            values,
            "rh56_target_q_policy_order_rad",
            (steps, 6),
            exact_float32=True,
        )
        logged_register_target = np.asarray(
            values["rh56_proposed_angle_set_register_order"]
        )
        if (
            logged_register_target.shape != (steps, 6)
            or logged_register_target.dtype.kind not in "iu"
        ):
            raise ValueError(
                "rh56_proposed_angle_set_register_order must be integer "
                "[steps,6]"
            )
        proposal_mode = np.asarray(values["proposal_mode"])
        if proposal_mode.shape != (steps,) or proposal_mode.dtype.kind not in "US":
            raise ValueError("proposal_mode must be a string [steps]")
        if np.any(proposal_mode != "one_step_from_measured_idle"):
            raise ValueError(
                "offline mapping audit only accepts one_step_from_measured_idle"
            )

        recomputed = []
        for index in range(steps):
            mapper = V94ActionMapper(
                initial_arm_target_q_rad=franka_q_for_mapping[index],
                initial_hand_target_q_policy_order_rad=hand_q_for_mapping[index],
                joint_limits_rad=contract.joint_limits_rad,
                q_hand_close_rad=contract.q_hand_close_rad,
            )
            recomputed.append(
                mapper.map(
                    logged_action[index],
                    measured_q_rad=franka_q_for_mapping[index],
                )
            )
        expected_executed = np.stack(
            [item.executed_policy_action13 for item in recomputed]
        )
        expected_arm_target = np.stack(
            [item.franka_target_q_rad for item in recomputed]
        )
        expected_hand_target = np.stack(
            [item.rh56_target_q_policy_order_rad for item in recomputed]
        )
        expected_register_target = np.stack(
            [item.rh56_angle_set_register_order for item in recomputed]
        )
        mapping_errors = {
            "executed_policy_action13": _maximum_error(
                expected_executed,
                executed_action,
                "executed policy action mapping",
            ),
            "franka_target_q_rad": _maximum_error(
                expected_arm_target,
                logged_arm_target,
                "Franka target mapping",
            ),
            "rh56_target_q_policy_order_rad": _maximum_error(
                expected_hand_target,
                logged_hand_target,
                "RH56 virtual target mapping",
            ),
        }
        worst_mapping_error = max(mapping_errors.values())
        if worst_mapping_error > tolerance:
            raise ValueError(
                "action-target mapping error "
                f"{worst_mapping_error:.9g} exceeds {tolerance:.9g}"
            )
        if not np.array_equal(expected_register_target, logged_register_target):
            maximum_register_error = int(
                np.max(
                    np.abs(
                        expected_register_target.astype(np.int64)
                        - logged_register_target.astype(np.int64)
                    )
                )
            )
            raise ValueError(
                "RH56 register target mapping mismatch: "
                f"maximum error {maximum_register_error} units"
            )

        arm_step = logged_arm_target.astype(np.float64) - franka_q_for_mapping
        hand_step = logged_hand_target.astype(np.float64) - hand_q_for_mapping
        measured_envelope = np.max(np.abs(arm_step), axis=1)
        if np.any(measured_envelope > 0.050001):
            raise ValueError("Franka target exceeds measured-q safety envelope")
        if np.any(logged_arm_target < contract.joint_limits_rad[:, 0]) or np.any(
            logged_arm_target > contract.joint_limits_rad[:, 1]
        ):
            raise ValueError("Franka target exceeds commissioned joint limits")
        if np.any(np.abs(arm_step) > 0.003001):
            raise ValueError("one-step Franka nominal target delta exceeds 0.003 rad")
        if np.any(np.abs(hand_step) > 0.050001):
            raise ValueError("one-step RH56 virtual target delta exceeds 0.05 rad")
        action_target_mapping = {
            "available": True,
            "passed": True,
            "proposal_mode": "one_step_from_measured_idle",
            "atol": tolerance,
            "max_abs_error_by_float_output": mapping_errors,
            "worst_float_max_abs_error": worst_mapping_error,
            "rh56_register_targets_exact": True,
            "franka_target_delta_abs_max_rad": _series_stats(
                np.max(np.abs(arm_step), axis=1)
            ),
            "rh56_virtual_target_delta_abs_max_rad": _series_stats(
                np.max(np.abs(hand_step), axis=1)
            ),
            "note": (
                "Targets were recomputed independently from each measured "
                "idle state; this is not a hypothetical accumulated closed loop."
            ),
        }

    normalized_points = (points - policy.point_mean) / policy.point_std
    normalized_proprio = (proprio - policy.proprio_mean) / policy.proprio_std
    valid_normalized_points = normalized_points[valid.astype(bool)]
    if valid_normalized_points.shape[0] < 1:
        raise RuntimeError("normalized point audit unexpectedly has no valid points")

    point_age = timestamps["timestamp_s"] - timestamps["pointcloud_timestamp_s"]
    hand_age = timestamps["timestamp_s"] - timestamps["rh56_timestamp_s"]
    sensor_stack = np.stack(
        [
            timestamps["franka_timestamp_s"],
            timestamps["pointcloud_timestamp_s"],
            timestamps["rh56_timestamp_s"],
        ],
        axis=1,
    )
    capture_span = np.max(sensor_stack, axis=1) - np.min(sensor_stack, axis=1)

    pose_name = "T_base_palm_at_pointcloud_capture"
    capture_poses: Optional[np.ndarray] = None
    if pose_name in values:
        capture_poses = _validate_rigid_history(values[pose_name], steps)

    if steps > 1:
        action_delta = np.diff(logged_action.astype(np.float64), axis=0)
        action_stability: dict[str, Any] = {
            "transitions": steps - 1,
            "mean_abs_step_delta": float(np.mean(np.abs(action_delta))),
            "max_abs_step_delta": float(np.max(np.abs(action_delta))),
            "mean_l2_step_delta": float(np.mean(np.linalg.norm(action_delta, axis=1))),
            "max_l2_step_delta": float(np.max(np.linalg.norm(action_delta, axis=1))),
        }
    else:
        action_stability = {
            "transitions": 0,
            "mean_abs_step_delta": None,
            "max_abs_step_delta": None,
            "mean_l2_step_delta": None,
            "max_l2_step_delta": None,
        }
    action_stability["saturation_fraction"] = float(
        np.mean(np.abs(logged_action) >= 1.0)
    )
    action_stability["saturation_fraction_per_axis"] = np.mean(
        np.abs(logged_action) >= 1.0, axis=0
    ).tolist()

    initial_actions = np.asarray(
        initial_alignment["policy_action13"], dtype=np.float64
    )
    closed_actions = np.asarray(
        closed_alignment["policy_action13"], dtype=np.float64
    )
    initial_action_distance = _nearest_row_linf(
        logged_action, initial_actions, "initial action comparison"
    )
    accepted_action_distance = _nearest_row_linf(
        logged_action,
        np.concatenate([initial_actions, closed_actions], axis=0),
        "packaged action comparison",
    )
    initial_raw_action_cosine = _cosine_to_nearest_linf_reference(
        logged_action, initial_actions, dimensions=slice(0, 13)
    )
    initial_arm_action_cosine = _cosine_to_nearest_linf_reference(
        logged_action, initial_actions, dimensions=slice(0, 7)
    )

    initial_points = np.asarray(
        initial_alignment["pointcloud_xyzrgb_palm"], dtype=np.float32
    )
    initial_valid = np.asarray(
        initial_alignment["pointcloud_valid"], dtype=np.float32
    )
    point_chamfer, point_centroid = _nearest_pointcloud_reference_metrics(
        current_points, current_valid, initial_points, initial_valid
    )

    base_point_chamfer: Optional[np.ndarray] = None
    base_point_coordinate_mean: Optional[np.ndarray] = None
    if capture_poses is not None:
        initial_pose_values = np.stack(
            [
                pose_from_position_quaternion_wxyz(
                    position,
                    quaternion,
                )
                for position, quaternion in zip(
                    np.asarray(initial_alignment["proprio67"])[:, 26:29],
                    np.asarray(initial_alignment["proprio67"])[:, 29:33],
                )
            ]
        )
        initial_pose_values = _validate_rigid_history(
            initial_pose_values, initial_points.shape[0]
        )
        current_points_base = current_points.copy()
        initial_points_base = initial_points.copy()
        for index, xyz_base in enumerate(
            _pointclouds_in_base(current_points, current_valid, capture_poses)
        ):
            current_points_base[index, current_valid[index].astype(bool), :3] = (
                xyz_base.astype(np.float32)
            )
        for index, xyz_base in enumerate(
            _pointclouds_in_base(initial_points, initial_valid, initial_pose_values)
        ):
            initial_points_base[index, initial_valid[index].astype(bool), :3] = (
                xyz_base.astype(np.float32)
            )
        base_point_chamfer, base_point_coordinate_mean = (
            _nearest_pointcloud_reference_metrics(
                current_points_base,
                current_valid,
                initial_points_base,
                initial_valid,
            )
        )

    initial_proprio = np.asarray(initial_alignment["proprio67"], dtype=np.float64)
    all_reference_proprio = np.concatenate(
        [
            initial_proprio,
            np.asarray(closed_alignment["current_proprio67"], dtype=np.float64),
        ],
        axis=0,
    )
    proprio_mean67 = np.asarray(policy.proprio_mean, dtype=np.float64).reshape(67)
    proprio_std67 = np.asarray(policy.proprio_std, dtype=np.float64).reshape(67)
    normalized_current_proprio = (
        current_proprio.astype(np.float64) - proprio_mean67
    ) / proprio_std67
    normalized_initial_proprio = (
        initial_proprio - proprio_mean67
    ) / proprio_std67
    normalized_reference_proprio = (
        all_reference_proprio - proprio_mean67
    ) / proprio_std67
    initial_proprio_distance = _nearest_row_linf(
        normalized_current_proprio,
        normalized_initial_proprio,
        "initial normalized proprio comparison",
    )
    accepted_proprio_distance = _nearest_row_linf(
        normalized_current_proprio,
        normalized_reference_proprio,
        "packaged normalized proprio comparison",
    )
    palm_position_distance = np.min(
        np.linalg.norm(
            current_proprio[:, None, 26:29].astype(np.float64)
            - initial_proprio[None, :, 26:29],
            axis=2,
        ),
        axis=1,
    )
    palm_orientation_distance = _quaternion_angular_distance_rad(
        current_proprio[:, 29:33], initial_proprio[:, 29:33]
    )
    packaged_reference_comparison = {
        "diagnostic_only": True,
        "initial_action_linf_nearest": _series_stats(initial_action_distance),
        "initial_or_closed_action_linf_nearest": _series_stats(
            accepted_action_distance
        ),
        "initial_raw_action13_cosine_at_linf_nearest": (
            _optional_series_stats(initial_raw_action_cosine)
        ),
        "initial_arm_increment_action7_cosine_at_linf_nearest": (
            _optional_series_stats(initial_arm_action_cosine)
        ),
        "initial_pointcloud_xyz_symmetric_chamfer_m_nearest": _series_stats(
            point_chamfer
        ),
        "initial_pointcloud_centroid_l2_m_nearest": _series_stats(
            point_centroid
        ),
        "initial_partial_cloud_coordinate_mean_l2_m_nearest": _series_stats(
            point_centroid
        ),
        "initial_base_frame_pointcloud_xyz_symmetric_chamfer_m_nearest": (
            None if base_point_chamfer is None else _series_stats(base_point_chamfer)
        ),
        "initial_base_frame_partial_cloud_coordinate_mean_l2_m_nearest": (
            None
            if base_point_coordinate_mean is None
            else _series_stats(base_point_coordinate_mean)
        ),
        "initial_normalized_proprio_linf_nearest": _series_stats(
            initial_proprio_distance
        ),
        "initial_or_closed_normalized_proprio_linf_nearest": _series_stats(
            accepted_proprio_distance
        ),
        "initial_palm_position_l2_m_nearest": _series_stats(
            palm_position_distance
        ),
        "initial_palm_orientation_rad_nearest": _series_stats(
            palm_orientation_distance
        ),
        "note": (
            "Nearest packaged samples are diagnostics, not calibrated OOD or "
            "semantic-action acceptance thresholds. Point-cloud coordinate "
            "means describe visible partial samples and are not object/sphere "
            "centres. RH56 outputs are absolute targets, so only the first "
            "seven incremental Franka axes are labelled an action direction."
        ),
    }

    robot_reset_alignment: dict[str, Any] = {
        "available": False,
        "aligned": None,
        "note": "Raw Franka/RH56 reset fields were not present in this audit.",
    }
    reset_fields = {
        "franka_q_rad",
        "rh56_angle_act_register_order",
        "rh56_angle_set_register_order",
        "rh56_virtual_q_policy_order_rad",
    }
    if reset_fields.issubset(values):
        franka_q = _float_array(values, "franka_q_rad", (steps, 7))
        hand_angles = np.asarray(values["rh56_angle_act_register_order"])
        hand_targets = np.asarray(values["rh56_angle_set_register_order"])
        hand_virtual_q = _float_array(
            values, "rh56_virtual_q_policy_order_rad", (steps, 6)
        )
        if hand_angles.shape != (steps, 6) or hand_angles.dtype.kind not in "iu":
            raise ValueError(
                "rh56_angle_act_register_order must be integer [steps,6]"
            )
        if hand_targets.shape != (steps, 6) or hand_targets.dtype.kind not in "iu":
            raise ValueError(
                "rh56_angle_set_register_order must be integer [steps,6]"
            )
        reference_q_home = np.median(
            np.asarray(initial_alignment["franka_measured_q_rad"], dtype=np.float64),
            axis=0,
        )
        q_error = franka_q - reference_q_home
        q_abs_max_per_step = np.max(np.abs(q_error), axis=1)
        hand_virtual_abs_max = np.max(np.abs(hand_virtual_q), axis=1)
        arm_aligned = bool(
            np.all(q_abs_max_per_step <= RESET_ARM_TOLERANCE_RAD)
        )
        hand_open = bool(np.all(hand_angles >= RESET_HAND_OPEN_MIN_UNITS))
        hand_disabled = bool(np.all(hand_targets == -1))
        hand_virtual_open = bool(np.all(hand_virtual_abs_max <= 0.05))
        previous_action = current_proprio[:, 54:67]
        expected_previous_action = np.asarray(
            [0.0] * 7 + [-1.0] * 6, dtype=np.float64
        )
        previous_action_error = np.max(
            np.abs(previous_action - expected_previous_action), axis=1
        )
        previous_action_reset = bool(np.all(previous_action_error <= 1.0e-6))
        robot_reset_alignment = {
            "available": True,
            "aligned": bool(
                arm_aligned
                and hand_open
                and hand_disabled
                and hand_virtual_open
                and previous_action_reset
            ),
            "franka_q_home_reference_rad": reference_q_home.tolist(),
            "franka_q_error_rad_per_joint_median": np.median(
                q_error, axis=0
            ).tolist(),
            "franka_q_abs_max_rad": _series_stats(q_abs_max_per_step),
            "franka_tolerance_rad": RESET_ARM_TOLERANCE_RAD,
            "franka_aligned": arm_aligned,
            "rh56_angle_min_units": int(np.min(hand_angles)),
            "rh56_open_min_units": RESET_HAND_OPEN_MIN_UNITS,
            "rh56_open": hand_open,
            "rh56_targets_all_disabled": hand_disabled,
            "rh56_virtual_q_abs_max_rad": _series_stats(hand_virtual_abs_max),
            "rh56_virtual_open": hand_virtual_open,
            "previous_action_reset_linf": _series_stats(previous_action_error),
            "previous_action_reset": previous_action_reset,
            "note": (
                "This verifies the recorded reset state only; it does not "
                "authorize policy action execution."
            ),
        }

    reset_alignment: dict[str, Any] = {
        "available": False,
        "status": "unknown",
        "reset_reference_aligned": None,
        "semantic_action_correctness_claimed": False,
        "note": (
            "No capture-time palm pose was recorded, so the partial cloud "
            "cannot be transformed to the bundle's robot-base reset frame."
        ),
    }
    if capture_poses is not None:
        try:
            object_evidence = _object_reset_evidence(bundle)
            base_clouds = _pointclouds_in_base(
                current_points, current_valid, capture_poses
            )
            surface_min = np.asarray(
                object_evidence["possible_surface_aabb_min_m"],
                dtype=np.float64,
            )
            surface_max = np.asarray(
                object_evidence["possible_surface_aabb_max_m"],
                dtype=np.float64,
            )
            outside_fractions = []
            all_inside_per_step = []
            per_step_coordinate_median = []
            per_step_coordinate_min = []
            per_step_coordinate_max = []
            outside_count = 0
            total_count = 0
            maximum_below = np.zeros(3, dtype=np.float64)
            maximum_above = np.zeros(3, dtype=np.float64)
            for xyz_base in base_clouds:
                inside = np.all(
                    (xyz_base >= surface_min[None, :])
                    & (xyz_base <= surface_max[None, :]),
                    axis=1,
                )
                count = int(xyz_base.shape[0])
                rejected = int(np.count_nonzero(~inside))
                total_count += count
                outside_count += rejected
                outside_fractions.append(float(rejected / count))
                all_inside_per_step.append(bool(rejected == 0))
                per_step_coordinate_median.append(
                    np.median(xyz_base, axis=0).tolist()
                )
                per_step_coordinate_min.append(np.min(xyz_base, axis=0).tolist())
                per_step_coordinate_max.append(np.max(xyz_base, axis=0).tolist())
                maximum_below = np.maximum(
                    maximum_below,
                    np.max(np.maximum(surface_min[None, :] - xyz_base, 0.0), axis=0),
                )
                maximum_above = np.maximum(
                    maximum_above,
                    np.max(np.maximum(xyz_base - surface_max[None, :], 0.0), axis=0),
                )

            allowed_outlier_fraction = float(
                object_evidence["configured_point_outlier_probability"]
            )
            outside_fractions_array = np.asarray(
                outside_fractions, dtype=np.float64
            )
            aggregate_outside_fraction = float(outside_count / total_count)
            maximum_step_outside_fraction = float(
                np.max(outside_fractions_array)
            )
            # This is an observation-compatibility check, not a deterministic
            # bound on a Bernoulli/Gaussian random process.  The only allowance
            # is the point-outlier probability explicitly serialized in the
            # training observation DR; no new percentile or metric threshold
            # is introduced here.
            surface_compatible = bool(
                aggregate_outside_fraction <= allowed_outlier_fraction
                and maximum_step_outside_fraction <= allowed_outlier_fraction
            )
            exact_initial_match = []
            for index in range(steps):
                exact_initial_match.append(
                    any(
                        np.array_equal(current_points[index], initial_points[ref])
                        and np.array_equal(current_valid[index], initial_valid[ref])
                        for ref in range(initial_points.shape[0])
                    )
                )
            median_array = np.asarray(
                per_step_coordinate_median, dtype=np.float64
            )
            minimum_array = np.asarray(
                per_step_coordinate_min, dtype=np.float64
            )
            maximum_array = np.asarray(
                per_step_coordinate_max, dtype=np.float64
            )

            reset_alignment = {
                "available": True,
                "status": "compatible" if surface_compatible else "incompatible",
                "reset_reference_aligned": surface_compatible,
                "alignment_semantics": (
                    "bundle_derived_possible_surface_observation_compatibility"
                ),
                "bundle_evidence": {
                    key: (value.tolist() if isinstance(value, np.ndarray) else value)
                    for key, value in object_evidence.items()
                },
                "possible_surface_union_aabb": {
                    "minimum_base_m": surface_min.tolist(),
                    "maximum_base_m": surface_max.tolist(),
                    "all_points_contained": bool(all(all_inside_per_step)),
                    "steps_with_all_points_contained": int(
                        np.count_nonzero(all_inside_per_step)
                    ),
                    "steps_with_outside_points": int(
                        steps - np.count_nonzero(all_inside_per_step)
                    ),
                    "outside_point_count": outside_count,
                    "valid_point_count": total_count,
                    "aggregate_outside_fraction": aggregate_outside_fraction,
                    "outside_fraction_per_step_stats": _series_stats(
                        outside_fractions_array
                    ),
                    "maximum_step_outside_fraction": (
                        maximum_step_outside_fraction
                    ),
                    "allowed_outside_fraction": allowed_outlier_fraction,
                    "allowed_outside_fraction_source": (
                        "training_profile.student_observation_domain_"
                        "randomization.palm_frame_pointcloud_at_alpha_1."
                        "outlier_probability"
                    ),
                    "maximum_below_minimum_m_by_axis": maximum_below.tolist(),
                    "maximum_above_maximum_m_by_axis": maximum_above.tolist(),
                    "compatibility_passed": surface_compatible,
                    "threshold_note": (
                        "The configured outlier probability is reused directly "
                        "as an empirical observation-compatibility threshold. "
                        "It is not a hard support or confidence bound."
                    ),
                },
                "visible_partial_cloud_base": {
                    "coordinatewise_median_per_step_m": _axis_series_stats(
                        median_array
                    ),
                    "coordinatewise_minimum_per_step_m": _axis_series_stats(
                        minimum_array
                    ),
                    "coordinatewise_maximum_per_step_m": _axis_series_stats(
                        maximum_array
                    ),
                    "coordinatewise_median_is_object_center": False,
                },
                "packaged_initial_observation_exact_match_steps": int(
                    np.count_nonzero(exact_initial_match)
                ),
                "packaged_initial_observation_exact_match_fraction": float(
                    np.mean(exact_initial_match)
                ),
                "sphere_center_alignment": {
                    "status": "unknown",
                    "estimated_center_base_m": None,
                    "reason": (
                        "A visible partial-cloud coordinate mean/median is not "
                        "the sphere centre. The bundle supplies Gaussian depth, "
                        "XYZ jitter, and outlier priors but no finite residual "
                        "bound for a strict real sphere fit."
                    ),
                },
                "semantic_action_correctness_claimed": False,
                "note": (
                    "The reset gate is a necessary observation-compatibility "
                    "check against a bundle-derived union of possible sphere "
                    "surfaces. It does not claim an exact real object centre, "
                    "object identity, grasp correctness, or action safety."
                ),
            }
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            reset_alignment = {
                "available": False,
                "status": "unknown",
                "reset_reference_aligned": None,
                "evidence_error": f"{type(exc).__name__}: {exc}",
                "semantic_action_correctness_claimed": False,
                "note": (
                    "The bundle did not support a strict object reset "
                    "compatibility result; no fallback tolerance was invented."
                ),
            }

    robot_command_writes = bool(
        np.asarray(values.get("robot_command_writes", np.asarray(False)))
    )
    camera_configuration_writes = (
        None
        if "camera_configuration_writes" not in values
        else bool(np.asarray(values["camera_configuration_writes"]))
    )
    return {
        "result": "PASS",
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "steps": steps,
        # Backward-compatible alias.  In schema v1 this means robot actuator or
        # register commands, not D435 option configuration.
        "hardware_writes": False,
        "hardware_writes_semantics": (
            "robot_actuator_or_register_commands_only"
        ),
        "robot_command_writes": robot_command_writes,
        "camera_configuration_writes": camera_configuration_writes,
        "schema_shapes_finite_and_current_history_consistent": True,
        "policy_replay": {
            "passed": True,
            "atol": tolerance,
            "max_abs_error_by_output": errors,
            "worst_max_abs_error": worst_replay_error,
        },
        "action_target_mapping": action_target_mapping,
        "pointcloud_valid": {
            "history_counts": _series_stats(valid_counts),
            "current_counts": _series_stats(valid_counts[:, -1]),
        },
        "timing": {
            "policy_period_s": (
                None
                if steps == 1
                else _series_stats(np.diff(timestamps["timestamp_s"]))
            ),
            "pointcloud_age_at_policy_s": _series_stats(point_age),
            "rh56_age_at_policy_s": _series_stats(hand_age),
            "sensor_capture_span_s": _series_stats(capture_span),
        },
        "action_stability": action_stability,
        "packaged_reference_comparison": packaged_reference_comparison,
        "robot_reset_alignment": robot_reset_alignment,
        "normalized_zscore": {
            "pointcloud_valid_features": _feature_zscore_stats(valid_normalized_points),
            "proprio": _feature_zscore_stats(normalized_proprio),
        },
        "reset_reference_alignment": reset_alignment,
        "reset_reference_aligned": reset_alignment["reset_reference_aligned"],
        "semantic_action_correctness_claimed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline-only schema, timing, normalization, and deterministic "
            "policy replay audit for a live_v94_preview NPZ."
        )
    )
    parser.add_argument("audit_npz", type=Path)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--replay-atol", type=float, default=DEFAULT_REPLAY_ATOL)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit_v94_preview(
            args.audit_npz,
            bundle_path=args.bundle,
            replay_atol=args.replay_atol,
        )
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            print("V94 read-only preview audit: PASS")
            print(json.dumps(report, indent=2, sort_keys=True))
            print("hardware: untouched (offline audit only)")
        return 0
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(
            f"V94 read-only preview audit: FAIL: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
