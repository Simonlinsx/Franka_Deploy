#!/usr/bin/python3
"""Offline IK/collision plan for an empty-flange grasp-pose preview.

This is deliberately separate from installed-tool execution.  It reproduces
the upstream Inspire convention with a *virtual* TCP offset (44 mm by default)
so an empty FR3 flange can preview the saved hand pose.  The value is never
written into any installed-tool adapter configuration.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


P_MOUNT_SOURCE = np.asarray(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
DEFAULT_Q = np.asarray(
    [0.0, 0.0, 0.0, -np.pi / 2.0, 0.0, np.pi / 2.0, 0.0],
    dtype=np.float64,
)
JOINT_LIMITS = np.asarray(
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
FR3_XACRO = Path(
    "/opt/ros/humble/share/franka_description/robots/fr3/fr3.urdf.xacro"
)
FR3_SRDF_XACRO = Path(
    "/opt/ros/humble/share/franka_description/robots/fr3/fr3.srdf.xacro"
)
SAVED_POSE_ONLY_TOKEN = "FR3_SAVED_POSE_ONLY"


@dataclass(frozen=True)
class CollisionSceneInput:
    """Collision points plus provenance for one planner invocation.

    A live scene replaces only the stale scene portion of the grasp snapshot.
    The selected grasp, canonical pose, hand target, and segmented object cloud
    continue to come from the saved snapshot.
    """

    points: np.ndarray
    robot_return_filter_q: np.ndarray
    metadata: Mapping[str, Any]


def _scalar_text(
    values: Mapping[str, np.ndarray], key: str, *, required: bool = False
) -> str:
    if key not in values:
        if required:
            raise ValueError("{} is missing".format(key))
        return ""
    raw = np.asarray(values[key])
    if raw.shape != ():
        raise ValueError("{} must be a scalar string".format(key))
    value = raw.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value).strip()
    if required and not text:
        raise ValueError("{} must be non-empty".format(key))
    return text


def _point_array(value: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("{} must have shape (N, 3)".format(name))
    if len(points) == 0:
        raise ValueError("{} must not be empty".format(name))
    if not np.all(np.isfinite(points)):
        raise ValueError("{} contains NaN or infinity".format(name))
    return points.copy()


def _joint_vector(value: np.ndarray, name: str) -> np.ndarray:
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        raise ValueError("{} must be a finite 7-vector".format(name))
    if np.any(q < JOINT_LIMITS[:, 0]) or np.any(q > JOINT_LIMITS[:, 1]):
        raise ValueError("{} is outside the FR3 joint limits".format(name))
    return q.copy()


def _optional_scalar(values: Mapping[str, np.ndarray], key: str) -> Optional[Any]:
    if key not in values:
        return None
    raw = np.asarray(values[key])
    if raw.shape != ():
        raise ValueError("{} must be scalar".format(key))
    value = raw.item()
    if isinstance(value, np.generic):
        value = value.item()
    return value


def _resolve_robot_return_filter_q(
    live_values: Optional[Mapping[str, np.ndarray]],
    explicit_q: Optional[Sequence[float]],
    *,
    require_live_q: bool = False,
) -> Tuple[np.ndarray, str]:
    if explicit_q is not None:
        explicit = _joint_vector(
            np.asarray(explicit_q, dtype=np.float64), "--scene-capture-q-rad"
        )
        if live_values is not None:
            aliases = (
                "capture_q_rad",
                "franka_q_rad",
                "franka_q",
                "q_capture_rad",
            )
            for alias in aliases:
                if alias not in live_values:
                    continue
                embedded = _joint_vector(live_values[alias], alias)
                if not np.allclose(explicit, embedded, atol=1e-6, rtol=0.0):
                    raise ValueError(
                        "--scene-capture-q-rad conflicts with live-scene {}".format(
                            alias
                        )
                    )
        return explicit, "cli:--scene-capture-q-rad"

    if live_values is not None:
        aliases = ("capture_q_rad", "franka_q_rad", "franka_q", "q_capture_rad")
        available = [name for name in aliases if name in live_values]
        if available:
            first_name = available[0]
            first = _joint_vector(live_values[first_name], first_name)
            for alias in available[1:]:
                other = _joint_vector(live_values[alias], alias)
                if not np.allclose(first, other, atol=1e-6, rtol=0.0):
                    raise ValueError(
                        "live-scene joint metadata conflicts: {} vs {}".format(
                            first_name, alias
                        )
                    )
            return first, "live_scene_npz:{}".format(first_name)

        if require_live_q:
            raise ValueError(
                "live scene must contain capture_q_rad (or pass "
                "--scene-capture-q-rad)"
            )

    return DEFAULT_Q.copy(), "planner_default_q"


def _load_npz_values(path: Path) -> Mapping[str, np.ndarray]:
    expanded = path.expanduser()
    with np.load(str(expanded), allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _optional_array_list(
    values: Mapping[str, np.ndarray], key: str
) -> Optional[list]:
    if key not in values:
        return None
    array = np.asarray(values[key])
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("{} must be a finite vector".format(key))
    return array.tolist()


def _nearest_scene_distances(query_points: np.ndarray, scene_points: np.ndarray) -> np.ndarray:
    """Return exact nearest-scene distances, with a dependency-free fallback."""

    query = _point_array(query_points, "saved object points")
    scene = _point_array(scene_points, "live scene points")
    try:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(scene).query(query, k=1, workers=1)
        return np.asarray(distances, dtype=np.float64)
    except ImportError:
        # Keep the planner usable in a minimal NumPy environment.  Both axes
        # are chunked so the fallback cannot allocate an N_object x N_scene
        # matrix for a full RealSense frame.
        result = np.full(len(query), np.inf, dtype=np.float64)
        for query_start in range(0, len(query), 256):
            query_chunk = query[query_start : query_start + 256]
            best_squared = np.full(len(query_chunk), np.inf, dtype=np.float64)
            for scene_start in range(0, len(scene), 4096):
                scene_chunk = scene[scene_start : scene_start + 4096]
                delta = query_chunk[:, None, :] - scene_chunk[None, :, :]
                squared = np.einsum("ijk,ijk->ij", delta, delta)
                best_squared = np.minimum(best_squared, np.min(squared, axis=1))
            result[query_start : query_start + len(query_chunk)] = np.sqrt(
                best_squared
            )
        return result


def _live_object_alignment_stats(
    saved_object_points: np.ndarray,
    live_scene_points: np.ndarray,
    coverage_distance_m: float,
) -> Mapping[str, Any]:
    distances = _nearest_scene_distances(saved_object_points, live_scene_points)
    return {
        "method": "saved_object_to_live_scene_nearest_neighbor",
        "saved_object_point_count": int(len(distances)),
        "median_distance_m": float(np.median(distances)),
        "p95_distance_m": float(np.percentile(distances, 95.0)),
        "maximum_distance_m": float(np.max(distances)),
        "coverage_distance_m": float(coverage_distance_m),
        "coverage_fraction": float(np.mean(distances <= float(coverage_distance_m))),
    }


def _validate_live_object_alignment(
    saved_object_points: np.ndarray,
    live_scene_points: np.ndarray,
    *,
    median_max_m: float,
    p95_max_m: float,
    coverage_distance_m: float,
    minimum_coverage: float,
    allow_failure: bool = False,
) -> Mapping[str, Any]:
    stats = dict(
        _live_object_alignment_stats(
            saved_object_points, live_scene_points, coverage_distance_m
        )
    )
    stats.update(
        {
            "median_max_m": float(median_max_m),
            "p95_max_m": float(p95_max_m),
            "minimum_coverage": float(minimum_coverage),
        }
    )
    failures = []
    if stats["median_distance_m"] > float(median_max_m):
        failures.append(
            "median {:.6f}m > {:.6f}m".format(
                stats["median_distance_m"], median_max_m
            )
        )
    if stats["p95_distance_m"] > float(p95_max_m):
        failures.append(
            "p95 {:.6f}m > {:.6f}m".format(stats["p95_distance_m"], p95_max_m)
        )
    if stats["coverage_fraction"] < float(minimum_coverage):
        failures.append(
            "coverage {:.3f} < {:.3f}".format(
                stats["coverage_fraction"], minimum_coverage
            )
        )
    stats["passed"] = not failures
    stats["failures"] = list(failures)
    if failures and not allow_failure:
        raise ValueError(
            "saved object no longer aligns with live scene: {}".format(
                "; ".join(failures)
            )
        )
    return stats


def _select_collision_scene(
    snapshot_values: Mapping[str, np.ndarray],
    snapshot_path: Path,
    live_scene_path: Optional[Path],
    explicit_capture_q: Optional[Sequence[float]],
    *,
    object_median_max_m: float = 0.008,
    object_p95_max_m: float = 0.015,
    object_coverage_distance_m: float = 0.015,
    object_minimum_coverage: float = 0.80,
    capture_q_default_tolerance_rad: float = 0.02,
    saved_pose_only: bool = False,
) -> CollisionSceneInput:
    """Select fresh scene geometry without replacing saved grasp/object data."""

    snapshot_frame = _scalar_text(snapshot_values, "reference_frame", required=True)
    if snapshot_frame != "robot_base":
        raise ValueError("snapshot reference_frame must be robot_base")
    snapshot_scene = _point_array(snapshot_values["scene_points"], "snapshot scene_points")
    snapshot_object = _point_array(
        snapshot_values["object_points"], "snapshot object_points"
    )
    snapshot_calibration = _scalar_text(snapshot_values, "calibration_id")
    snapshot_camera = _scalar_text(snapshot_values, "camera_serial")

    live_values: Optional[Mapping[str, np.ndarray]] = None
    if live_scene_path is None:
        scene = snapshot_scene
        source_kind = "snapshot"
        source_path = snapshot_path.expanduser().resolve()
        calibration_id = snapshot_calibration
        camera_serial = snapshot_camera
        frame_id = _optional_scalar(snapshot_values, "frame_id")
        timestamp_s = _optional_scalar(snapshot_values, "timestamp_s")
        captured_at_utc = ""
        captured_at_unix_s = None
        capture_completed_at_unix_s = None
        capture_started_s = None
        capture_completed_s = None
        frame_ids = None
        frame_timestamps_s = None
        source_schema_name = "dexgrasp_grasp_snapshot"
        source_schema_version = _optional_scalar(snapshot_values, "schema_version")
        embedded_capture_q_source = ""
        object_alignment = None
    else:
        resolved_live_path = live_scene_path.expanduser().resolve()
        live_hash_before_load = _sha256_file(resolved_live_path)
        live_values = _load_npz_values(resolved_live_path)
        live_hash_after_load = _sha256_file(resolved_live_path)
        if live_hash_after_load != live_hash_before_load:
            raise ValueError("live-scene NPZ changed while it was being loaded")
        live_frame = _scalar_text(live_values, "reference_frame", required=True)
        if live_frame != "robot_base":
            raise ValueError("live-scene reference_frame must be robot_base")
        scene = _point_array(live_values["scene_points"], "live scene_points")
        source_kind = "live_scene_npz"
        source_path = resolved_live_path
        calibration_id = _scalar_text(live_values, "calibration_id", required=True)
        camera_serial = _scalar_text(live_values, "camera_serial", required=True)
        frame_id = _optional_scalar(live_values, "frame_id")
        timestamp_s = _optional_scalar(live_values, "timestamp_s")
        captured_at_utc = _scalar_text(live_values, "captured_at_utc")
        captured_at_unix_s = _optional_scalar(live_values, "captured_at_unix_s")
        capture_completed_at_unix_s = _optional_scalar(
            live_values, "capture_completed_at_unix_s"
        )
        capture_started_s = _optional_scalar(live_values, "capture_started_s")
        capture_completed_s = _optional_scalar(live_values, "capture_completed_s")
        frame_ids = _optional_array_list(live_values, "frame_ids")
        frame_timestamps_s = _optional_array_list(
            live_values, "frame_timestamps_s"
        )
        source_schema_name = _scalar_text(live_values, "schema_name")
        source_schema_version = _optional_scalar(live_values, "schema_version")
        embedded_capture_q_source = _scalar_text(live_values, "capture_q_source")
        if snapshot_calibration and calibration_id and calibration_id != snapshot_calibration:
            raise ValueError(
                "live-scene calibration_id does not match the grasp snapshot"
            )
        if snapshot_camera and camera_serial and camera_serial != snapshot_camera:
            raise ValueError("live-scene camera_serial does not match the grasp snapshot")
        object_alignment = _validate_live_object_alignment(
            snapshot_object,
            scene,
            median_max_m=object_median_max_m,
            p95_max_m=object_p95_max_m,
            coverage_distance_m=object_coverage_distance_m,
            minimum_coverage=object_minimum_coverage,
            allow_failure=bool(saved_pose_only),
        )

    filter_q, filter_q_source = _resolve_robot_return_filter_q(
        live_values,
        explicit_capture_q,
        require_live_q=live_scene_path is not None,
    )
    capture_q_error = float(np.max(np.abs(filter_q - DEFAULT_Q)))
    if (
        live_scene_path is not None
        and capture_q_error > float(capture_q_default_tolerance_rad)
    ):
        raise ValueError(
            "live scene was not captured at planner DEFAULT_Q: max error "
            "{:.6f}rad > {:.6f}rad".format(
                capture_q_error, capture_q_default_tolerance_rad
            )
        )
    combined = np.concatenate([scene, snapshot_object], axis=0)
    source_stat = source_path.stat()
    source_sha256 = (
        live_hash_after_load
        if live_scene_path is not None
        else _sha256_file(source_path)
    )
    metadata = {
        "source_kind": source_kind,
        "scene_path": str(source_path),
        "scene_sha256": source_sha256,
        "scene_file_mtime_s": float(source_stat.st_mtime),
        "scene_file_size_bytes": int(source_stat.st_size),
        "source_schema_name": source_schema_name,
        "source_schema_version": source_schema_version,
        "saved_grasp_snapshot": str(snapshot_path.expanduser().resolve()),
        "scene_geometry_source": (
            "live_scene_npz:scene_points"
            if live_scene_path is not None
            else "grasp_snapshot:scene_points"
        ),
        "object_geometry_source": "grasp_snapshot:object_points",
        "saved_grasp_pose_retained": True,
        "saved_object_cloud_retained": True,
        "snapshot_scene_ignored": bool(live_scene_path is not None),
        "reference_frame": "robot_base",
        "calibration_id": calibration_id,
        "camera_serial": camera_serial,
        "frame_id": frame_id,
        "timestamp_s": timestamp_s,
        "frame_ids": frame_ids,
        "frame_timestamps_s": frame_timestamps_s,
        "captured_at_utc": captured_at_utc,
        "captured_at_unix_s": captured_at_unix_s,
        "capture_completed_at_unix_s": capture_completed_at_unix_s,
        "capture_started_s": capture_started_s,
        "capture_completed_s": capture_completed_s,
        "scene_points_raw": int(len(scene)),
        "saved_object_points_raw": int(len(snapshot_object)),
        "combined_points_raw": int(len(combined)),
        "robot_return_filter_q_rad": filter_q.tolist(),
        "robot_return_filter_q_source": filter_q_source,
        "capture_q_rad": filter_q.tolist(),
        "capture_q_source": embedded_capture_q_source or filter_q_source,
        "capture_q_value_source": filter_q_source,
        "capture_q_default_tolerance_rad": float(capture_q_default_tolerance_rad),
        "robot_return_filter_q_max_abs_from_default_rad": capture_q_error,
        "capture_q_max_error_to_default_rad": capture_q_error,
        "live_object_alignment": object_alignment,
        "live_scene_revalidated": object_alignment is not None,
        "saved_pose_only": bool(saved_pose_only),
        "stale_object_pose_override": bool(
            saved_pose_only
            and object_alignment is not None
            and not bool(object_alignment["passed"])
        ),
    }
    return CollisionSceneInput(
        points=combined,
        robot_return_filter_q=filter_q,
        metadata=metadata,
    )


def _rigid(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("{} must be a finite 4x4 transform".format(name))
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("{} has an invalid homogeneous row".format(name))
    if not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6):
        raise ValueError("{} rotation is not orthonormal".format(name))
    return matrix.copy()


def _inverse(T: np.ndarray) -> np.ndarray:
    transform = _rigid(T, "transform")
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def _write_robot_descriptions(directory: Path) -> Tuple[Path, Path, Path]:
    urdf = directory / "fr3_bare.urdf"
    urdf_sc = directory / "fr3_bare_sc.urdf"
    srdf = directory / "fr3_bare.srdf"
    commands = (
        ["xacro", str(FR3_XACRO), "hand:=false", "with_sc:=false", "-o", str(urdf)],
        ["xacro", str(FR3_XACRO), "hand:=false", "with_sc:=true", "-o", str(urdf_sc)],
        ["xacro", str(FR3_SRDF_XACRO), "hand:=false", "-o", str(srdf)],
    )
    for command in commands:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return urdf, urdf_sc, srdf


def _pose_error(pin, current, target) -> Tuple[float, float]:
    relative = current.actInv(target)
    vector = pin.log6(relative).vector
    return float(np.linalg.norm(vector[:3])), float(np.linalg.norm(vector[3:]))


def _solve_ik(pin, model, frame_id: int, target_matrix: np.ndarray, starts: Iterable[np.ndarray]):
    target = pin.SE3(target_matrix[:3, :3], target_matrix[:3, 3])
    lower = JOINT_LIMITS[:, 0] + 0.06
    upper = JOINT_LIMITS[:, 1] - 0.06
    best = None
    for initial in starts:
        q = np.clip(np.asarray(initial, dtype=np.float64), lower, upper)
        data = model.createData()
        for iteration in range(2500):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            current = data.oMf[frame_id]
            iMd = current.actInv(target)
            error = pin.log6(iMd).vector
            translation_error, rotation_error = _pose_error(pin, current, target)
            metric = translation_error + 0.15 * rotation_error
            if best is None or metric < best[0]:
                best = (metric, q.copy(), translation_error, rotation_error, iteration)
            if translation_error <= 0.0015 and rotation_error <= 0.015:
                return q.copy(), translation_error, rotation_error, iteration
            jacobian = pin.computeFrameJacobian(
                model, data, q, frame_id, pin.ReferenceFrame.LOCAL
            )
            jacobian = -pin.Jlog6(iMd.inverse()) @ jacobian
            damping = 1e-5
            velocity = -jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(6), error
            )
            norm = float(np.linalg.norm(velocity))
            if norm > 1.0:
                velocity *= 1.0 / norm
            q = pin.integrate(model, q, velocity * 0.10)
            q = np.clip(q, lower, upper)
    assert best is not None
    raise RuntimeError(
        "IK did not converge; best translation={:.6f}m rotation={:.6f}rad q={}".format(
            best[2], best[3], np.round(best[1], 6).tolist()
        )
    )


def _joint_path(start: np.ndarray, end: np.ndarray, samples: int) -> np.ndarray:
    return np.asarray(
        [(1.0 - alpha) * start + alpha * end for alpha in np.linspace(0.0, 1.0, samples)]
    )


def _check_self_collision(pin, model, collision_model, srdf: Path, path: np.ndarray):
    collision_model.addAllCollisionPairs()
    pin.removeCollisionPairs(model, collision_model, str(srdf))
    data = model.createData()
    geometry_data = pin.GeometryData(collision_model)
    for sample_index, q in enumerate(path):
        if pin.computeCollisions(
            model, data, collision_model, geometry_data, q, False
        ):
            pairs = []
            for pair_index, result in enumerate(geometry_data.collisionResults):
                if result.isCollision():
                    pair = collision_model.collisionPairs[pair_index]
                    first = collision_model.geometryObjects[pair.first].name
                    second = collision_model.geometryObjects[pair.second].name
                    pairs.append("{} / {}".format(first, second))
            raise RuntimeError(
                "self collision at path sample {}: {}".format(sample_index, pairs)
            )


def _voxel_downsample(points: np.ndarray, voxel_m: float) -> np.ndarray:
    finite = np.asarray(points, dtype=np.float64)
    finite = finite[np.all(np.isfinite(finite), axis=1)]
    keys = np.floor(finite / float(voxel_m)).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return finite[np.sort(indices)]


def _primitive_signed_distance(local: np.ndarray, geometry) -> np.ndarray:
    name = type(geometry).__name__
    if name == "Sphere":
        return np.linalg.norm(local, axis=1) - float(geometry.radius)
    if name == "Cylinder":
        radial = np.linalg.norm(local[:, :2], axis=1) - float(geometry.radius)
        axial = np.abs(local[:, 2]) - float(geometry.halfLength)
        pair = np.column_stack([radial, axial])
        outside = np.linalg.norm(np.maximum(pair, 0.0), axis=1)
        inside = np.minimum(np.maximum(radial, axial), 0.0)
        return outside + inside
    raise TypeError("unsupported safety primitive {}".format(name))


def _check_scene_clearance(pin, model, geometry_model, path, points, margin_m):
    data = model.createData()
    geometry_data = pin.GeometryData(geometry_model)
    minimum = np.inf
    minimum_detail = None
    primitive_indices = [
        index
        for index, item in enumerate(geometry_model.geometryObjects)
        if "_sc_" in item.name and type(item.geometry).__name__ in ("Sphere", "Cylinder")
    ]
    if not primitive_indices:
        raise RuntimeError("FR3 safety-collision primitives are unavailable")
    for sample_index, q in enumerate(path):
        pin.forwardKinematics(model, data, q)
        pin.updateGeometryPlacements(model, data, geometry_model, geometry_data, q)
        for geometry_index in primitive_indices:
            item = geometry_model.geometryObjects[geometry_index]
            placement = geometry_data.oMg[geometry_index]
            local = (placement.rotation.T @ (points - placement.translation).T).T
            distances = _primitive_signed_distance(local, item.geometry)
            point_index = int(np.argmin(distances))
            distance = float(distances[point_index])
            if distance < minimum:
                minimum = distance
                minimum_detail = (sample_index, item.name, point_index)
            if distance <= margin_m:
                raise RuntimeError(
                    "scene clearance failed: sample={} primitive={} distance={:.6f}m "
                    "margin={:.6f}m point={}".format(
                        sample_index,
                        item.name,
                        distance,
                        margin_m,
                        np.round(points[point_index], 6).tolist(),
                    )
                )
    return minimum, minimum_detail


def _exclude_initial_robot_returns(
    pin, model, geometry_model, q_initial, points, margin_m
):
    """Remove depth returns already on the robot at the captured default pose."""

    data = model.createData()
    geometry_data = pin.GeometryData(geometry_model)
    pin.forwardKinematics(model, data, q_initial)
    pin.updateGeometryPlacements(
        model, data, geometry_model, geometry_data, q_initial
    )
    minimum = np.full(len(points), np.inf, dtype=np.float64)
    for geometry_index, item in enumerate(geometry_model.geometryObjects):
        if "_sc_" not in item.name or type(item.geometry).__name__ not in (
            "Sphere",
            "Cylinder",
        ):
            continue
        placement = geometry_data.oMg[geometry_index]
        local = (placement.rotation.T @ (points - placement.translation).T).T
        minimum = np.minimum(
            minimum, _primitive_signed_distance(local, item.geometry)
        )
    keep = minimum > float(margin_m)
    excluded = int(np.count_nonzero(~keep))
    if excluded > max(1000, int(0.25 * len(points))):
        raise RuntimeError(
            "too many scene points overlap the robot at capture: {}/{}".format(
                excluded, len(points)
            )
        )
    return points[keep], excluded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline FR3 IK/collision plan for a saved hand-pose preview"
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--live-scene",
        type=Path,
        default=None,
        help=(
            "Fresh calibrated robot_base scene NPZ. It replaces the snapshot's "
            "scene_points only; the saved grasp pose and object_points are retained."
        ),
    )
    parser.add_argument(
        "--scene-capture-q-rad",
        type=float,
        nargs=7,
        default=None,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"),
        help=(
            "FR3 q at scene capture for robot-return filtering. Resolution order: "
            "this option, live NPZ capture_q_rad/franka_q_rad, planner DEFAULT_Q."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--virtual-tcp-offset-m", type=float, default=0.044)
    parser.add_argument("--pregrasp-distance-m", type=float, default=0.10)
    parser.add_argument("--path-samples", type=int, default=101)
    parser.add_argument("--scene-voxel-m", type=float, default=0.01)
    parser.add_argument("--scene-margin-m", type=float, default=0.005)
    parser.add_argument("--live-object-median-max-m", type=float, default=0.008)
    parser.add_argument("--live-object-p95-max-m", type=float, default=0.015)
    parser.add_argument("--live-object-coverage-distance-m", type=float, default=0.015)
    parser.add_argument("--live-object-minimum-coverage", type=float, default=0.80)
    parser.add_argument("--capture-q-default-tolerance-rad", type=float, default=0.02)
    parser.add_argument(
        "--confirm-saved-pose-only",
        metavar=SAVED_POSE_ONLY_TOKEN,
        default=None,
        help=(
            "Permit an empty-flange fixed-pose preview when the saved object no "
            "longer aligns. This does not validate a grasp; live collision checks "
            "remain mandatory."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.confirm_saved_pose_only not in (None, SAVED_POSE_ONLY_TOKEN):
        raise SystemExit(
            "--confirm-saved-pose-only must be exactly {}".format(
                SAVED_POSE_ONLY_TOKEN
            )
        )
    saved_pose_only = args.confirm_saved_pose_only == SAVED_POSE_ONLY_TOKEN
    if saved_pose_only and args.live_scene is None:
        raise SystemExit("saved-pose-only preview requires --live-scene")
    if not 0.0 <= args.virtual_tcp_offset_m <= 0.10:
        raise SystemExit("virtual TCP offset must be in 0..0.10 m")
    if not 0.05 <= args.pregrasp_distance_m <= 0.20:
        raise SystemExit("pregrasp distance must be in 0.05..0.20 m")
    if args.path_samples < 21:
        raise SystemExit("path samples must be at least 21")
    if not 0.002 <= args.scene_voxel_m <= 0.03:
        raise SystemExit("scene voxel must be in 0.002..0.03 m")
    if not 0.0 <= args.scene_margin_m <= 0.03:
        raise SystemExit("scene margin must be in 0..0.03 m")
    if not 0.001 <= args.live_object_median_max_m <= 0.05:
        raise SystemExit("live object median threshold must be in 0.001..0.05 m")
    if not args.live_object_median_max_m <= args.live_object_p95_max_m <= 0.08:
        raise SystemExit("live object p95 threshold must be >= median and <= 0.08 m")
    if not 0.001 <= args.live_object_coverage_distance_m <= 0.08:
        raise SystemExit("live object coverage distance must be in 0.001..0.08 m")
    if not 0.5 <= args.live_object_minimum_coverage <= 1.0:
        raise SystemExit("live object minimum coverage must be in 0.5..1.0")
    if not 0.001 <= args.capture_q_default_tolerance_rad <= 0.10:
        raise SystemExit("capture-q default tolerance must be in 0.001..0.10 rad")

    try:
        import pinocchio as pin
    except ImportError as exc:
        raise SystemExit("run with /usr/bin/python3; Pinocchio is unavailable: {}".format(exc))

    values = _load_npz_values(args.snapshot)
    try:
        collision_input = _select_collision_scene(
            values,
            args.snapshot,
            args.live_scene,
            args.scene_capture_q_rad,
            object_median_max_m=float(args.live_object_median_max_m),
            object_p95_max_m=float(args.live_object_p95_max_m),
            object_coverage_distance_m=float(
                args.live_object_coverage_distance_m
            ),
            object_minimum_coverage=float(args.live_object_minimum_coverage),
            capture_q_default_tolerance_rad=float(
                args.capture_q_default_tolerance_rad
            ),
            saved_pose_only=saved_pose_only,
        )
    except (KeyError, OSError, ValueError) as exc:
        raise SystemExit("invalid collision-scene input: {}".format(exc))
    selected = int(values["selected_grasp_index"])
    hand = _rigid(values["hand_poses"][selected], "T_robot_base_hand_source")
    canonical = _rigid(values["canonical_grasp_poses"][selected], "canonical pose")
    approach = canonical[:3, :3] @ np.asarray(values["approach_axis_local"], dtype=np.float64)
    approach /= np.linalg.norm(approach)

    T_F_TCP = np.eye(4)
    T_F_TCP[2, 3] = float(args.virtual_tcp_offset_m)
    T_base_TCP = hand @ _inverse(P_MOUNT_SOURCE)
    T_base_F_grasp = _rigid(T_base_TCP @ _inverse(T_F_TCP), "T_base_F_grasp")
    T_base_F_pregrasp = T_base_F_grasp.copy()
    T_base_F_pregrasp[:3, 3] -= float(args.pregrasp_distance_m) * approach

    points = _voxel_downsample(collision_input.points, float(args.scene_voxel_m))

    with tempfile.TemporaryDirectory(prefix="fr3-preview-") as temporary:
        urdf, urdf_sc, srdf = _write_robot_descriptions(Path(temporary))
        model, collision_model, _ = pin.buildModelsFromUrdf(
            str(urdf), package_dirs=["/opt/ros/humble/share"]
        )
        model_sc, collision_model_sc, _ = pin.buildModelsFromUrdf(
            str(urdf_sc), package_dirs=["/opt/ros/humble/share"]
        )
        frame_id = model.getFrameId("fr3_link8")
        starts = [DEFAULT_Q]
        rng = np.random.default_rng(7)
        for _ in range(12):
            starts.append(
                np.clip(
                    DEFAULT_Q + rng.normal(0.0, 0.35, 7),
                    JOINT_LIMITS[:, 0] + 0.08,
                    JOINT_LIMITS[:, 1] - 0.08,
                )
            )
        q_pre, pre_t, pre_r, pre_iterations = _solve_ik(
            pin, model, frame_id, T_base_F_pregrasp, starts
        )
        q_grasp, grasp_t, grasp_r, grasp_iterations = _solve_ik(
            pin, model, frame_id, T_base_F_grasp, [q_pre] + starts
        )
        first_path = _joint_path(DEFAULT_Q, q_pre, int(args.path_samples))
        second_path = _joint_path(q_pre, q_grasp, int(args.path_samples))
        path = np.concatenate([first_path, second_path[1:]], axis=0)
        _check_self_collision(pin, model, collision_model, srdf, path)
        initial_robot_filter_margin_m = max(float(args.scene_margin_m), 0.020)
        points_checked, excluded_initial_robot_points = _exclude_initial_robot_returns(
            pin,
            model_sc,
            collision_model_sc,
            collision_input.robot_return_filter_q,
            points,
            initial_robot_filter_margin_m,
        )
        minimum, detail = _check_scene_clearance(
            pin,
            model_sc,
            collision_model_sc,
            path,
            points_checked,
            float(args.scene_margin_m),
        )

    audited_scene_path = Path(str(collision_input.metadata["scene_path"]))
    if _sha256_file(audited_scene_path) != collision_input.metadata["scene_sha256"]:
        raise RuntimeError(
            "collision-scene NPZ changed during planning; discard this plan and recapture"
        )

    target_angles = np.rint(values["hand_angles"][selected]).astype(int)
    artifact = {
        "schema_version": 2,
        "artifact_type": "empty_flange_diagnostic_grasp_preview",
        "snapshot": str(args.snapshot.expanduser().resolve()),
        "selected_index": selected,
        "model_name": str(values["model_name"]),
        "diagnostic_only": True,
        "installed_mount_calibration": False,
        "saved_pose_only": bool(saved_pose_only),
        "q_start_rad": DEFAULT_Q.tolist(),
        "virtual_tcp": {
            "source": "upstream UR preview convention only",
            "F_T_TCP_z_m": float(args.virtual_tcp_offset_m),
            "must_not_copy_to_installed_profile": True,
        },
        "collision_scene_input": dict(collision_input.metadata),
        "T_robot_base_flange_pregrasp": T_base_F_pregrasp.tolist(),
        "T_robot_base_flange_grasp": T_base_F_grasp.tolist(),
        "q_pregrasp_rad": q_pre.tolist(),
        "q_grasp_rad": q_grasp.tolist(),
        "ik_error": {
            "pregrasp_translation_m": pre_t,
            "pregrasp_rotation_rad": pre_r,
            "pregrasp_iterations": pre_iterations,
            "grasp_translation_m": grasp_t,
            "grasp_rotation_rad": grasp_r,
            "grasp_iterations": grasp_iterations,
        },
        "collision_audit": {
            "self_collision_free": True,
            "scene_capsule_clear": True,
            "scene_margin_m": float(args.scene_margin_m),
            "minimum_sampled_signed_distance_m": float(minimum),
            "minimum_detail": list(detail) if detail is not None else None,
            "path_samples": int(len(path)),
            "scene_points_after_voxel": int(len(points)),
            "live_scene_revalidated": bool(
                collision_input.metadata["live_scene_revalidated"]
            ),
            "object_alignment_valid": bool(
                collision_input.metadata["live_object_alignment"] is not None
                and collision_input.metadata["live_object_alignment"]["passed"]
            ),
            "stale_object_pose_preview": bool(
                collision_input.metadata["stale_object_pose_override"]
            ),
            "live_object_alignment": collision_input.metadata[
                "live_object_alignment"
            ],
            "scene_sha256": collision_input.metadata["scene_sha256"],
            "audited_at_s": float(time.time()),
            "initial_robot_returns_excluded": excluded_initial_robot_points,
            "initial_robot_filter_margin_m": initial_robot_filter_margin_m,
            "robot_return_filter_q_rad": (
                collision_input.robot_return_filter_q.tolist()
            ),
            "robot_return_filter_q_source": collision_input.metadata[
                "robot_return_filter_q_source"
            ],
            "scene_points_checked": int(len(points_checked)),
            "authoritative_for_installed_tool": False,
        },
        "inspire_snapshot_target": target_angles.tolist(),
        "inspire_safe_preview_target": (
            target_angles[:5].tolist() + [900]
        ),
    }
    args.output.expanduser().write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
