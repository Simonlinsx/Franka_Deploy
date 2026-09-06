from __future__ import annotations

"""Plan or execute one conservative move above a stable visible object cloud."""

import argparse
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from dynamic_pcd.calibration.io import ResolvedExtrinsics, resolve_extrinsics
from dynamic_pcd.calibration.transforms import (
    franka_pose_to_matrix,
    validate_rigid_transform,
)
from dynamic_pcd.config import load_config
from dynamic_pcd.ipc.zmq_pubsub import ZMQObjectPCDSubscriber
from dynamic_pcd.types import ObjectPCDPacket


FR3_JOINT_LIMITS = np.asarray(
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

# This app is intentionally limited to the region exercised during this first
# commissioning.  A YAML edit may tighten these bounds, but may not silently
# expand them for physical motion.
COMMISSIONED_OBJECT_MIN = np.asarray([0.505, 0.008, -0.05], dtype=np.float64)
COMMISSIONED_OBJECT_MAX = np.asarray([0.650, 0.104, 0.30], dtype=np.float64)
COMMISSIONED_EEF_MIN = np.asarray([0.480, -0.020, 0.300], dtype=np.float64)
COMMISSIONED_EEF_MAX = np.asarray([0.660, 0.130, 0.360], dtype=np.float64)
COMMISSIONED_DESCENT_EEF_MIN = np.asarray(
    [0.480, -0.020, 0.190], dtype=np.float64
)
MAX_COMMISSIONED_SEGMENT_M = 0.030
MAX_COMMISSIONED_VELOCITY_MPS = 0.010
MIN_COMMISSIONED_CLEARANCE_M = 0.150
MIN_COMMISSIONED_DESCENT_CLEARANCE_M = 0.100
MIN_CONTROL_PERIOD_S = 1e-6
MAX_CONTROL_PERIOD_S = 0.020
# Pose reads before execution can differ from the planning read by a few
# micrometres.  Treat sub-millimetre negative Z deltas as numerical/servo
# settling noise and clamp them to the current height; real descent segments in
# this commissioned path are centimetres long.
DESCENT_Z_EPS_M = 0.0005
# libfranka reports a zero command-success rate before the first command has
# populated its rolling window.  Other safety fields remain enforced during
# this short startup interval; only the rate threshold is deferred.
CONTROL_SUCCESS_WARMUP_S = 0.10


@dataclass(frozen=True)
class StableTarget:
    center: np.ndarray
    z_p95: float
    sample_count: int
    duration_s: float
    center_p95_m: float
    max_step_m: float
    fitted_speed_mps: float


class MotionCancelled(RuntimeError):
    """Raised after a cooperative request stops a one-shot robot motion."""


@dataclass(frozen=True)
class OneShotHoverResult:
    """Result returned by :func:`execute_hover_one_shot`."""

    current_xyz: np.ndarray
    goal_xyz: np.ndarray
    final_xyz: np.ndarray
    waypoint_count: int


def validate_hover_config(
    values: Dict[str, Any], *, allow_descent: bool = False
) -> Dict[str, Any]:
    """Validate every safety-relevant hover setting before touching the robot."""

    cfg = dict(values)
    vector_pairs = (
        ("object_workspace_min", "object_workspace_max"),
        ("eef_workspace_min", "eef_workspace_max"),
    )
    parsed: Dict[str, np.ndarray] = {}
    for lower_key, upper_key in vector_pairs:
        parsed[lower_key] = _vec3(cfg.get(lower_key), name=lower_key)
        parsed[upper_key] = _vec3(cfg.get(upper_key), name=upper_key)
        if not np.all(parsed[lower_key] < parsed[upper_key]):
            raise ValueError(f"{lower_key} must be strictly below {upper_key}")

    object_min = parsed["object_workspace_min"]
    object_max = parsed["object_workspace_max"]
    eef_min = parsed["eef_workspace_min"]
    eef_max = parsed["eef_workspace_max"]
    if np.any(object_min < COMMISSIONED_OBJECT_MIN) or np.any(
        object_max > COMMISSIONED_OBJECT_MAX
    ):
        raise ValueError("object workspace expands beyond the commissioned envelope")
    commissioned_eef_min = (
        COMMISSIONED_DESCENT_EEF_MIN if allow_descent else COMMISSIONED_EEF_MIN
    )
    if np.any(eef_min < commissioned_eef_min) or np.any(
        eef_max > COMMISSIONED_EEF_MAX
    ):
        raise ValueError("EEF workspace expands beyond the commissioned envelope")

    positive_keys = (
        "stable_seconds",
        "max_packet_age",
        "motion_max_packet_age",
        "max_center_p95",
        "max_center_step",
        "max_fitted_speed",
        "max_target_drift",
        "max_segment_distance",
        "max_velocity",
        "min_segment_duration",
        "min_joint_margin",
        "max_final_error",
    )
    scalars: Dict[str, float] = {}
    for key in positive_keys:
        try:
            value = float(cfg[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"hover.{key} must be a finite positive number") from exc
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"hover.{key} must be a finite positive number")
        scalars[key] = value

    for key in ("target_z_min", "target_z_max", "hover_clearance"):
        try:
            value = float(cfg[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"hover.{key} must be finite") from exc
        if not np.isfinite(value):
            raise ValueError(f"hover.{key} must be finite")
        scalars[key] = value

    try:
        stable_frames = int(cfg["stable_frames"])
        min_raw_points = int(cfg["min_raw_points"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("stable_frames/min_raw_points must be integers") from exc
    if stable_frames < 2 or min_raw_points < 10:
        raise ValueError("stable_frames must be >=2 and min_raw_points must be >=10")

    try:
        success_rate = float(cfg["min_control_success_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("min_control_success_rate must be finite in [0, 1]") from exc
    if not np.isfinite(success_rate) or not 0.0 <= success_rate <= 1.0:
        raise ValueError("min_control_success_rate must be finite in [0, 1]")

    if not scalars["target_z_min"] <= scalars["target_z_max"]:
        raise ValueError("target_z_min must not exceed target_z_max")
    if (
        scalars["target_z_min"] < eef_min[2]
        or scalars["target_z_max"] > min(eef_max[2], 0.345)
    ):
        raise ValueError("hover target z is outside the commissioned EEF envelope")
    minimum_clearance = (
        MIN_COMMISSIONED_DESCENT_CLEARANCE_M
        if allow_descent
        else MIN_COMMISSIONED_CLEARANCE_M
    )
    if scalars["hover_clearance"] < minimum_clearance:
        raise ValueError(
            f"hover_clearance must be at least {minimum_clearance:.3f}m"
        )
    if scalars["max_segment_distance"] > MAX_COMMISSIONED_SEGMENT_M:
        raise ValueError("max_segment_distance exceeds the commissioned limit")
    if scalars["max_velocity"] > MAX_COMMISSIONED_VELOCITY_MPS:
        raise ValueError("max_velocity exceeds the commissioned limit")
    if scalars["max_final_error"] > scalars["max_segment_distance"]:
        raise ValueError("max_final_error must not exceed max_segment_distance")
    if stable_frames < 30 or scalars["stable_seconds"] < 1.0:
        raise ValueError("stability window is weaker than the commissioned limits")
    commissioned_maxima = {
        "max_packet_age": 0.15,
        "motion_max_packet_age": 0.25,
        "max_center_p95": 0.005,
        "max_center_step": 0.010,
        "max_fitted_speed": 0.010,
        "max_target_drift": 0.020,
        "max_final_error": 0.015,
    }
    for key, maximum in commissioned_maxima.items():
        if scalars[key] > maximum:
            raise ValueError(f"hover.{key} weakens the commissioned limit")
    if min_raw_points < 200:
        raise ValueError("min_raw_points weakens the commissioned limit")
    if scalars["min_segment_duration"] < 3.0:
        raise ValueError("min_segment_duration weakens the commissioned limit")
    if scalars["min_joint_margin"] < 0.05:
        raise ValueError("min_joint_margin weakens the commissioned limit")
    if success_rate < 0.95:
        raise ValueError("min_control_success_rate weakens the commissioned limit")
    minimum_target_z = 0.200 if allow_descent else 0.320
    if scalars["target_z_min"] < minimum_target_z:
        raise ValueError("target_z_min weakens the commissioned limit")
    if allow_descent:
        for key in (
            "descent_clearance_guard",
            "descent_velocity",
            "max_descent_xy_error",
            "max_top_drift",
        ):
            try:
                value = float(cfg[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"hover.{key} must be finite and positive") from exc
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"hover.{key} must be finite and positive")
        if float(cfg["descent_clearance_guard"]) < 0.010:
            raise ValueError("descent_clearance_guard must be at least 0.010m")
        if float(cfg["descent_velocity"]) > 0.005:
            raise ValueError("descent_velocity exceeds 0.005m/s")
        if float(cfg["max_descent_xy_error"]) > 0.005:
            raise ValueError("max_descent_xy_error exceeds 0.005m")
        if float(cfg["max_top_drift"]) > 0.010:
            raise ValueError("max_top_drift exceeds 0.010m")
    return cfg


def _vec3(values: Iterable[float], *, name: str) -> np.ndarray:
    try:
        vector = np.asarray(list(values), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain three finite values") from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite values")
    return vector


def _packet_points(packet: ObjectPCDPacket) -> np.ndarray:
    points = np.asarray(packet.pcd_reference)
    if points.ndim != 2 or points.shape[0] < 10 or points.shape[1] < 3:
        raise ValueError("packet has no usable absolute pcd_reference")
    xyz = np.asarray(points[:, :3], dtype=np.float64)
    if not np.all(np.isfinite(xyz)):
        raise ValueError("packet pcd_reference contains NaN or infinity")
    return xyz


def validate_perception_packet(
    packet: ObjectPCDPacket,
    *,
    expected_calibration_id: str,
    expected_camera_serial: Optional[str],
    expected_T_base_camera: Optional[np.ndarray] = None,
    max_age_s: float,
    min_raw_points: int,
    max_center_reference_error_m: float = 0.020,
    now_s: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Fail closed unless a packet is fresh, calibrated, valid, and absolute."""

    if not np.isfinite(max_age_s) or max_age_s <= 0.0:
        raise ValueError("max_age_s must be finite and positive")
    if int(min_raw_points) < 10:
        raise ValueError("min_raw_points must be at least 10")
    if (
        not np.isfinite(max_center_reference_error_m)
        or max_center_reference_error_m <= 0.0
    ):
        raise ValueError("max_center_reference_error_m must be finite and positive")
    if not packet.valid:
        raise ValueError("packet.valid is false")
    if packet.reference_frame != "robot_base":
        raise ValueError(
            f"reference_frame must be robot_base, got {packet.reference_frame!r}"
        )
    if packet.calibration_id != expected_calibration_id:
        raise ValueError(
            f"calibration_id mismatch: {packet.calibration_id!r} != "
            f"{expected_calibration_id!r}"
        )
    if expected_camera_serial and packet.camera_serial != expected_camera_serial:
        raise ValueError(
            f"camera serial mismatch: {packet.camera_serial!r} != "
            f"{expected_camera_serial!r}"
        )
    if expected_T_base_camera is not None:
        packet_transform = validate_rigid_transform(
            packet.T_base_camera, name="packet.T_base_camera"
        )
        expected_transform = validate_rigid_transform(
            expected_T_base_camera, name="expected T_base_camera"
        )
        if not np.allclose(
            packet_transform, expected_transform, atol=1e-6, rtol=0.0
        ):
            raise ValueError("packet T_base_camera does not match calibration")

    try:
        now = time.time() if now_s is None else float(now_s)
        timestamp = float(packet.timestamp)
    except (TypeError, ValueError) as exc:
        raise ValueError("packet timestamp is not finite") from exc
    if not np.isfinite(timestamp):
        raise ValueError("packet timestamp is not finite")
    age = now - timestamp
    if age < -0.10 or age > max_age_s:
        raise ValueError(f"packet age {age:.3f}s exceeds limit {max_age_s:.3f}s")

    center = _vec3(packet.center, name="packet.center")
    points = _packet_points(packet)
    try:
        raw_points = int((packet.debug or {}).get("raw_points", 0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("raw point count is invalid") from exc
    if raw_points < int(min_raw_points):
        raise ValueError(
            f"raw point count {raw_points} is below {int(min_raw_points)}"
        )
    reference_center = np.median(points, axis=0)
    center_error = float(np.linalg.norm(reference_center - center))
    if (
        not np.isfinite(center_error)
        or center_error > float(max_center_reference_error_m)
    ):
        raise ValueError(
            f"packet center/reference mismatch is {center_error:.4f}m"
        )
    return center, float(np.percentile(points[:, 2], 95))


def analyze_stability(
    centers: np.ndarray,
    z_p95_values: np.ndarray,
    timestamps: np.ndarray,
) -> StableTarget:
    centers = np.asarray(centers, dtype=np.float64)
    z_values = np.asarray(z_p95_values, dtype=np.float64).reshape(-1)
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) < 2:
        raise ValueError("at least two 3D centers are required")
    if len(z_values) != len(centers) or len(times) != len(centers):
        raise ValueError("stability arrays must have equal length")
    if not (
        np.all(np.isfinite(centers))
        and np.all(np.isfinite(z_values))
        and np.all(np.isfinite(times))
    ):
        raise ValueError("stability samples contain NaN or infinity")

    median = np.median(centers, axis=0)
    radial = np.linalg.norm(centers - median[None, :], axis=1)
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    relative_time = times - times[0]
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("packet timestamps are not strictly increasing")
    design = np.column_stack([relative_time, np.ones(len(relative_time))])
    slopes = np.linalg.lstsq(design, centers, rcond=None)[0][0]
    return StableTarget(
        center=median.astype(np.float64),
        # A high quantile of the per-frame cloud tops is safer for vertical
        # clearance than their median while remaining robust to one bad frame.
        z_p95=float(np.percentile(z_values, 95)),
        sample_count=len(centers),
        duration_s=float(relative_time[-1]),
        center_p95_m=float(np.percentile(radial, 95)),
        max_step_m=float(np.max(steps)),
        fitted_speed_mps=float(np.linalg.norm(slopes)),
    )


def stable_enough(target: StableTarget, hover_cfg: Dict[str, Any]) -> bool:
    return (
        target.sample_count >= int(hover_cfg["stable_frames"])
        and target.duration_s >= float(hover_cfg["stable_seconds"])
        and target.center_p95_m <= float(hover_cfg["max_center_p95"])
        and target.max_step_m <= float(hover_cfg["max_center_step"])
        and target.fitted_speed_mps <= float(hover_cfg["max_fitted_speed"])
    )


def collect_stable_target(
    subscriber: ZMQObjectPCDSubscriber,
    extrinsics: ResolvedExtrinsics,
    hover_cfg: Dict[str, Any],
    *,
    timeout_s: float,
) -> StableTarget:
    max_window = max(60, int(hover_cfg["stable_frames"]) * 2)
    samples: Deque[Tuple[np.ndarray, float, float]] = deque(maxlen=max_window)
    deadline = time.monotonic() + float(timeout_s)
    last_report = 0.0
    last_rejection = "waiting for packets"

    while time.monotonic() < deadline:
        packet = subscriber.recv_latest(timeout_ms=250)
        if packet is None:
            last_rejection = "publisher timeout"
            samples.clear()
            continue
        try:
            center, z_p95 = validate_perception_packet(
                packet,
                expected_calibration_id=str(extrinsics.calibration_id),
                expected_camera_serial=extrinsics.camera_serial,
                expected_T_base_camera=extrinsics.T_base_camera,
                max_age_s=float(hover_cfg["max_packet_age"]),
                min_raw_points=int(hover_cfg["min_raw_points"]),
            )
        except ValueError as exc:
            last_rejection = str(exc)
            samples.clear()
            continue

        samples.append((center, z_p95, float(packet.timestamp)))
        if len(samples) < max(2, int(hover_cfg["stable_frames"])):
            continue
        try:
            target = analyze_stability(
                np.stack([sample[0] for sample in samples]),
                np.asarray([sample[1] for sample in samples]),
                np.asarray([sample[2] for sample in samples]),
            )
        except ValueError as exc:
            last_rejection = str(exc)
            samples.clear()
            continue
        last_rejection = (
            f"p95={target.center_p95_m * 1000:.1f}mm, "
            f"step={target.max_step_m * 1000:.1f}mm, "
            f"speed={target.fitted_speed_mps * 1000:.1f}mm/s, "
            f"duration={target.duration_s:.2f}s"
        )
        now = time.monotonic()
        if now - last_report >= 0.5:
            print(
                f"[Observe] n={target.sample_count} center="
                f"{np.array2string(target.center, precision=4)} "
                f"z95={target.z_p95:.4f}m | {last_rejection}"
            )
            last_report = now
        if stable_enough(target, hover_cfg):
            return target
    raise TimeoutError(
        f"no stable calibrated object target within {timeout_s:.1f}s: "
        f"{last_rejection}"
    )


def _inside(point: np.ndarray, lower: Sequence[float], upper: Sequence[float]) -> bool:
    minimum = _vec3(lower, name="workspace minimum")
    maximum = _vec3(upper, name="workspace maximum")
    return bool(np.all(point >= minimum) and np.all(point <= maximum))


def _line_endpoints(
    start: np.ndarray, end: np.ndarray, max_distance: float
) -> List[np.ndarray]:
    delta = end - start
    distance = float(np.linalg.norm(delta))
    if distance < 1e-6:
        return []
    count = max(1, int(math.ceil(distance / float(max_distance))))
    return [start + delta * (index / count) for index in range(1, count + 1)]


def plan_hover_waypoints(
    current_xyz: Sequence[float],
    target: StableTarget,
    hover_cfg: Dict[str, Any],
    *,
    allow_descent: bool = False,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Plan a high transit, then optionally a vertical guarded descent."""

    current = _vec3(current_xyz, name="current EEF xyz")
    center = _vec3(target.center, name="stable target center")
    if not _inside(
        center,
        hover_cfg["object_workspace_min"],
        hover_cfg["object_workspace_max"],
    ):
        raise ValueError(
            "visible-cloud center is outside the commissioned object workspace: "
            f"{np.array2string(center, precision=4)}"
        )

    clearance_guard = (
        float(hover_cfg["descent_clearance_guard"]) if allow_descent else 0.0
    )
    clearance_z = (
        float(target.z_p95)
        + float(hover_cfg["hover_clearance"])
        + clearance_guard
    )
    if allow_descent:
        target_z = max(clearance_z, float(hover_cfg["target_z_min"]))
    else:
        target_z = max(
            float(current[2]),
            clearance_z,
            float(hover_cfg["target_z_min"]),
        )
    if target_z > float(hover_cfg["target_z_max"]) + 1e-9:
        raise ValueError(
            f"required hover z={target_z:.4f}m exceeds commissioned maximum "
            f"{float(hover_cfg['target_z_max']):.4f}m"
        )
    goal = np.array([center[0], center[1], target_z], dtype=np.float64)
    if not _inside(
        current,
        hover_cfg["eef_workspace_min"],
        hover_cfg["eef_workspace_max"],
    ):
        raise ValueError(
            "current EEF is outside the commissioned motion workspace: "
            f"{np.array2string(current, precision=4)}"
        )
    if not _inside(
        goal,
        hover_cfg["eef_workspace_min"],
        hover_cfg["eef_workspace_max"],
    ):
        raise ValueError(
            "planned hover goal is outside the commissioned EEF workspace: "
            f"{np.array2string(goal, precision=4)}"
        )

    max_segment = float(hover_cfg["max_segment_distance"])
    transit = current.copy()
    transit[2] = (
        max(float(current[2]), target_z, 0.320)
        if allow_descent
        else target_z
    )
    if not _inside(
        transit,
        hover_cfg["eef_workspace_min"],
        hover_cfg["eef_workspace_max"],
    ):
        raise ValueError("high transit point is outside the commissioned workspace")
    waypoints = _line_endpoints(current, transit, max_segment)
    horizontal_start = transit if waypoints else current
    horizontal_goal = np.asarray([center[0], center[1], transit[2]])
    waypoints.extend(
        _line_endpoints(horizontal_start, horizontal_goal, max_segment)
    )
    if allow_descent:
        waypoints.extend(_line_endpoints(horizontal_goal, goal, max_segment))
    return goal, waypoints


_ERROR_FIELD_CACHE: Dict[type, Tuple[str, ...]] = {}


def _active_error_names(errors: Any) -> List[str]:
    if errors is None:
        return []
    error_type = type(errors)
    names = _ERROR_FIELD_CACHE.get(error_type)
    if names is None:
        names = tuple(name for name in dir(errors) if not name.startswith("_"))
        _ERROR_FIELD_CACHE[error_type] = names
    active: List[str] = []
    for name in names:
        try:
            value = getattr(errors, name)
        except Exception:
            continue
        if isinstance(value, (bool, np.bool_)) and bool(value):
            active.append(name)
    return active


def validate_robot_state(
    state: Any,
    *,
    require_idle: bool,
    min_joint_margin: float,
    min_success_rate: float,
) -> None:
    robot_mode = getattr(state, "robot_mode", None)
    mode = str(getattr(robot_mode, "name", robot_mode or "unknown")).lower()
    mode = mode.rsplit(".", 1)[-1]
    if require_idle and mode != "idle":
        raise RuntimeError(f"robot must be Idle before motion, got {mode}")
    if not require_idle and mode not in ("move", "idle"):
        raise RuntimeError(f"robot mode became unsafe during motion: {mode}")

    current_errors = getattr(state, "current_errors", None)
    if current_errors is None:
        raise RuntimeError("Franka state has no current_errors field")
    errors = _active_error_names(current_errors)
    if errors:
        raise RuntimeError(f"Franka current_errors are active: {errors}")
    for field, expected_size in (
        ("cartesian_contact", 6),
        ("cartesian_collision", 6),
        ("joint_contact", 7),
        ("joint_collision", 7),
    ):
        values = getattr(state, field, None)
        if values is None:
            raise RuntimeError(f"Franka state has no {field} field")
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (expected_size,) or not np.all(np.isfinite(array)):
            raise RuntimeError(f"Franka {field} is malformed or non-finite")
        if np.any(array > 0.5):
            raise RuntimeError(f"Franka reports {field}: {array.tolist()}")

    success_value = getattr(state, "control_command_success_rate", None)
    if success_value is None:
        raise RuntimeError("Franka state has no control_command_success_rate")
    success_rate = float(success_value)
    if not np.isfinite(success_rate):
        raise RuntimeError("control command success rate is non-finite")
    # libfranka reports 0.0 while Idle because no control commands are being
    # exchanged.  The rate is meaningful and enforced only in RobotMode.Move.
    if mode == "move" and success_rate < float(min_success_rate):
        raise RuntimeError(
            f"control command success rate {success_rate:.3f} is below "
            f"{float(min_success_rate):.3f}"
        )
    joints = np.asarray(getattr(state, "q", None), dtype=np.float64)
    if joints.shape != (7,) or not np.all(np.isfinite(joints)):
        raise RuntimeError(f"unexpected or non-finite Franka q: shape={joints.shape}")
    margins = np.minimum(joints - FR3_JOINT_LIMITS[:, 0], FR3_JOINT_LIMITS[:, 1] - joints)
    if float(np.min(margins)) < float(min_joint_margin):
        index = int(np.argmin(margins))
        raise RuntimeError(
            f"joint {index + 1} has only {margins[index]:.4f}rad limit margin"
        )
    try:
        franka_pose_to_matrix(getattr(state, "O_T_EE", ()))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid Franka O_T_EE: {exc}") from exc


def _pose_with_translation(values: Sequence[float], xyz: Sequence[float]) -> List[float]:
    pose = list(values)
    pose[12], pose[13], pose[14] = [float(value) for value in xyz]
    return pose


def _cosine_interpolate(start: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    blend = 0.5 - 0.5 * math.cos(math.pi * min(1.0, max(0.0, alpha)))
    return start + blend * (target - start)


class PerceptionWatchdog:
    def __init__(
        self,
        subscriber_or_addr: Any,
        extrinsics: ResolvedExtrinsics,
        hover_cfg: Dict[str, Any],
        frozen_center: np.ndarray,
        frozen_z_p95: Optional[float] = None,
    ):
        self.subscriber = (
            None if isinstance(subscriber_or_addr, str) else subscriber_or_addr
        )
        self.addr = (
            str(subscriber_or_addr)
            if isinstance(subscriber_or_addr, str)
            else None
        )
        self.extrinsics = extrinsics
        self.cfg = hover_cfg
        self.frozen_center = _vec3(frozen_center, name="frozen center")
        self.frozen_z_p95 = (
            None if frozen_z_p95 is None else float(frozen_z_p95)
        )
        self.last_valid_monotonic: Optional[float] = None
        self.valid_until_monotonic: Optional[float] = None
        self.last_center: Optional[np.ndarray] = None
        self.last_rejection = "waiting for a valid motion-watchdog packet"
        self.fatal_error: Optional[str] = None
        self._lock = threading.Lock()
        self._updated = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _validate(self, packet: ObjectPCDPacket) -> Tuple[np.ndarray, float]:
        wall_now = time.time()
        monotonic_now = time.monotonic()
        center, z_p95 = validate_perception_packet(
            packet,
            expected_calibration_id=str(self.extrinsics.calibration_id),
            expected_camera_serial=self.extrinsics.camera_serial,
            expected_T_base_camera=self.extrinsics.T_base_camera,
            max_age_s=float(self.cfg["motion_max_packet_age"]),
            min_raw_points=int(self.cfg["min_raw_points"]),
        )
        drift = float(np.linalg.norm(center - self.frozen_center))
        if not np.isfinite(drift) or drift > float(self.cfg["max_target_drift"]):
            raise RuntimeError(
                f"object target drifted {drift * 1000:.1f}mm during motion"
            )
        if self.frozen_z_p95 is not None:
            top_drift = float(z_p95 - self.frozen_z_p95)
            if (
                not np.isfinite(top_drift)
                or top_drift > float(self.cfg["max_top_drift"])
            ):
                raise RuntimeError(
                    f"object cloud top rose {top_drift * 1000:.1f}mm during motion"
                )
        packet_age = max(0.0, wall_now - float(packet.timestamp))
        remaining_freshness = max(
            0.0, float(self.cfg["motion_max_packet_age"]) - packet_age
        )
        return center, monotonic_now + remaining_freshness

    def start(self) -> None:
        """Start a receiver thread when given an address.

        The 1 kHz robot loop only reads the resulting snapshot; ZMQ, pickle and
        point-cloud percentiles stay outside that timing-sensitive loop.
        """

        if self.addr is None or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._receiver_loop,
            name="hover-perception-watchdog",
            daemon=True,
        )
        self._thread.start()

    def _receiver_loop(self) -> None:
        subscriber: Optional[ZMQObjectPCDSubscriber] = None
        try:
            subscriber = ZMQObjectPCDSubscriber(self.addr, timeout_ms=50)
            while not self._stop.is_set():
                packet = subscriber.recv_latest(timeout_ms=50, max_drain=8)
                if packet is None:
                    continue
                try:
                    center, valid_until = self._validate(packet)
                except (RuntimeError, TypeError, ValueError) as exc:
                    with self._lock:
                        self.last_rejection = str(exc)
                        if self.last_valid_monotonic is not None:
                            self.fatal_error = str(exc)
                    self._updated.set()
                    continue
                with self._lock:
                    self.last_center = center.copy()
                    self.last_valid_monotonic = time.monotonic()
                    self.valid_until_monotonic = valid_until
                self._updated.set()
        except BaseException as exc:
            with self._lock:
                self.fatal_error = f"perception receiver failed: {exc}"
            self._updated.set()
        finally:
            if subscriber is not None:
                subscriber.close()

    def wait_until_valid(
        self,
        timeout_s: float = 2.0,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        self.start()
        deadline = time.monotonic() + float(timeout_s)
        rejection = self.last_rejection
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                raise MotionCancelled("robot motion cancelled while awaiting perception")
            if self.addr is None:
                self.check()
                return
            with self._lock:
                fatal_error = self.fatal_error
                valid = self.last_valid_monotonic is not None
                rejection = self.last_rejection
            if fatal_error:
                raise RuntimeError(fatal_error)
            if valid:
                self.check()
                return
            self._updated.wait(timeout=0.05)
            self._updated.clear()
        raise RuntimeError(
            f"no valid perception watchdog packet within {timeout_s:.2f}s: "
            f"{rejection}"
        )

    def check(self) -> None:
        if self.subscriber is not None:
            packet = self.subscriber.recv_latest(timeout_ms=0)
            if packet is not None:
                center, valid_until = self._validate(packet)
                self.last_center = center.copy()
                self.last_valid_monotonic = time.monotonic()
                self.valid_until_monotonic = valid_until
        with self._lock:
            fatal_error = self.fatal_error
            last_valid = self.last_valid_monotonic
            valid_until = self.valid_until_monotonic
        if fatal_error:
            raise RuntimeError(fatal_error)
        if last_valid is None:
            raise RuntimeError("perception watchdog has no valid current packet")
        now = time.monotonic()
        if valid_until is None or now > valid_until:
            total_age = float(self.cfg["motion_max_packet_age"])
            if valid_until is not None:
                total_age += now - valid_until
            raise RuntimeError(
                f"perception watchdog packet age is about {total_age:.3f}s"
            )

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)


def _raise_if_motion_cancelled(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise MotionCancelled("robot motion cancelled by operator")


def _report_motion_progress(
    callback: Optional[Callable[[str], None]], message: str
) -> None:
    if callback is not None:
        callback(message)


def execute_waypoints(
    robot: Any,
    pylibfranka: Any,
    waypoints: Sequence[np.ndarray],
    perception_source: Any,
    extrinsics: ResolvedExtrinsics,
    hover_cfg: Dict[str, Any],
    frozen_center: np.ndarray,
    *,
    allow_descent: bool = False,
    frozen_z_p95: Optional[float] = None,
    minimum_descent_z: Optional[float] = None,
    cancel_event: Optional[threading.Event] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    hover_cfg = validate_hover_config(hover_cfg, allow_descent=allow_descent)
    frozen_center_vec = _vec3(frozen_center, name="frozen center")
    if allow_descent:
        if minimum_descent_z is None or not np.isfinite(minimum_descent_z):
            raise ValueError("minimum_descent_z is required for guarded descent")
        hard_clearance_z = float(minimum_descent_z) - float(
            hover_cfg["descent_clearance_guard"]
        )
    else:
        hard_clearance_z = -math.inf
    watchdog = PerceptionWatchdog(
        perception_source,
        extrinsics,
        hover_cfg,
        frozen_center_vec,
        frozen_z_p95=frozen_z_p95 if allow_descent else None,
    )
    max_velocity = float(hover_cfg["max_velocity"])
    min_duration = float(hover_cfg["min_segment_duration"])
    max_segment = float(hover_cfg["max_segment_distance"])
    max_tracking_error = float(hover_cfg["max_final_error"])
    pending: Deque[np.ndarray] = deque(
        _vec3(point, name="planned waypoint").copy() for point in waypoints
    )
    command_index = 0

    try:
        _raise_if_motion_cancelled(cancel_event)
        _report_motion_progress(progress_callback, "waiting for fresh perception")
        watchdog.wait_until_valid(timeout_s=2.0, cancel_event=cancel_event)
        while pending:
            _raise_if_motion_cancelled(cancel_event)
            command_index += 1
            if command_index > 100:
                raise RuntimeError("dynamic waypoint subdivision exceeded 100 segments")
            watchdog.check()
            state = robot.read_once()
            validate_robot_state(
                state,
                require_idle=True,
                min_joint_margin=float(hover_cfg["min_joint_margin"]),
                min_success_rate=float(hover_cfg["min_control_success_rate"]),
            )
            start_pose = list(state.O_T_EE)
            start = franka_pose_to_matrix(start_pose)[:3, 3]
            if not _inside(
                start,
                hover_cfg["eef_workspace_min"],
                hover_cfg["eef_workspace_max"],
            ):
                raise RuntimeError(
                    "actual EEF start left the commissioned workspace: "
                    f"{np.array2string(start, precision=4)}"
                )

            desired = pending.popleft().copy()
            z_delta = float(desired[2] - start[2])
            is_descent = z_delta < -DESCENT_Z_EPS_M
            if z_delta < 0.0 and not is_descent:
                desired[2] = start[2]
            if is_descent:
                if not allow_descent:
                    print(
                        f"[Move] clamping waypoint z {desired[2]:.5f} -> "
                        f"{start[2]:.5f}m to prevent a downward command"
                    )
                    desired[2] = start[2]
                else:
                    desired_xy_error = float(
                        np.linalg.norm(desired[:2] - frozen_center_vec[:2])
                    )
                    if desired_xy_error > 1e-6:
                        # A high-transit waypoint can sit slightly below the
                        # latest readback because Cartesian tracking is not
                        # exact.  While XY is not yet over the frozen target,
                        # preserve the actual current height unconditionally.
                        desired[2] = start[2]
                        is_descent = False
                    else:
                        if desired[2] < float(minimum_descent_z) - 1e-9:
                            raise RuntimeError(
                                "descent waypoint violates the guarded z floor"
                            )
                        start_xy_error = float(
                            np.linalg.norm(start[:2] - frozen_center_vec[:2])
                        )
                        if start_xy_error > float(
                            hover_cfg["max_descent_xy_error"]
                        ):
                            # Correct XY at the current safe height before any z
                            # decrease.  Requeue the original descent waypoint.
                            pending.appendleft(desired)
                            desired = np.asarray(
                                [frozen_center_vec[0], frozen_center_vec[1], start[2]],
                                dtype=np.float64,
                            )
            if not _inside(
                desired,
                hover_cfg["eef_workspace_min"],
                hover_cfg["eef_workspace_max"],
            ):
                raise RuntimeError("waypoint left the commissioned EEF workspace")

            delta = desired - start
            distance = float(np.linalg.norm(delta))
            if not np.isfinite(distance):
                raise RuntimeError("waypoint distance is non-finite")
            if distance > max_segment:
                target = start + delta * (max_segment / distance)
                pending.appendleft(desired)
            else:
                target = desired
            if target[2] < start[2] - DESCENT_Z_EPS_M and not allow_descent:
                raise RuntimeError("internal error: planned a downward command")
            distance = float(np.linalg.norm(target - start))
            if distance > max_segment + 1e-9:
                raise RuntimeError("internal error: segment exceeds distance limit")
            if distance < 1e-6:
                continue
            segment_velocity = (
                float(hover_cfg["descent_velocity"])
                if target[2] < start[2] - DESCENT_Z_EPS_M
                else max_velocity
            )
            duration = max(
                min_duration, (math.pi / 2.0) * distance / segment_velocity
            )
            if not np.isfinite(duration) or duration <= 0.0:
                raise RuntimeError("computed motion duration is invalid")
            print(
                f"[Move {command_index}] {np.array2string(start, precision=4)} "
                f"-> {np.array2string(target, precision=4)}, "
                f"duration={duration:.2f}s"
            )
            _report_motion_progress(
                progress_callback,
                f"moving segment {command_index}: "
                f"{np.array2string(start, precision=4)} -> "
                f"{np.array2string(target, precision=4)}",
            )

            # Recheck perception immediately before opening each control loop.
            _raise_if_motion_cancelled(cancel_event)
            watchdog.check()
            try:
                # Cover the complete lifetime of an active control handle.  In
                # particular, SIGINT must not land in a gap after control has
                # started but before the stop-on-exception guard is active.
                control = robot.start_cartesian_pose_control(
                    pylibfranka.ControllerMode.CartesianImpedance
                )
                elapsed = 0.0
                monitor_elapsed = 0.0
                allowed_initial_zero_period = True
                segment_wall_start = time.monotonic()
                segment_wall_deadline = segment_wall_start + duration + max(
                    2.0, 0.25 * duration
                )
                last_monitor_wall = segment_wall_start - 0.020
                while elapsed < duration:
                    _raise_if_motion_cancelled(cancel_event)
                    active_state, period = control.readOnce()
                    wall_now = time.monotonic()
                    if wall_now > segment_wall_deadline:
                        raise RuntimeError("Franka segment exceeded wall-clock deadline")
                    dt = float(period.to_sec())
                    if not np.isfinite(dt) or dt < 0.0 or dt > MAX_CONTROL_PERIOD_S:
                        raise RuntimeError(f"unsafe Franka control period dt={dt!r}s")
                    if dt == 0.0:
                        if not allowed_initial_zero_period:
                            raise RuntimeError("repeated zero Franka control period")
                        allowed_initial_zero_period = False
                    else:
                        if dt < MIN_CONTROL_PERIOD_S:
                            raise RuntimeError(
                                f"unsafe tiny Franka control period dt={dt!r}s"
                            )
                        allowed_initial_zero_period = False
                        elapsed += dt
                        monitor_elapsed += dt
                    if (
                        wall_now - last_monitor_wall >= 0.020
                        or monitor_elapsed >= 0.020
                    ):
                        validate_robot_state(
                            active_state,
                            require_idle=False,
                            min_joint_margin=float(hover_cfg["min_joint_margin"]),
                            min_success_rate=(
                                0.0
                                if elapsed < CONTROL_SUCCESS_WARMUP_S
                                else float(hover_cfg["min_control_success_rate"])
                            ),
                        )
                        watchdog.check()
                        last_monitor_wall = wall_now
                        monitor_elapsed = 0.0
                    command = _cosine_interpolate(
                        start, target, elapsed / duration
                    )
                    control.writeOnce(
                        pylibfranka.CartesianPose(
                            _pose_with_translation(start_pose, command)
                        )
                    )
                _raise_if_motion_cancelled(cancel_event)
                final_command = pylibfranka.CartesianPose(
                    _pose_with_translation(start_pose, target)
                )
                final_command.motion_finished = True
                control.writeOnce(final_command)
            except MotionCancelled:
                raise
            except BaseException:
                try:
                    robot.stop()
                except Exception:
                    pass
                raise

            segment_state = robot.read_once()
            validate_robot_state(
                segment_state,
                require_idle=True,
                min_joint_margin=float(hover_cfg["min_joint_margin"]),
                min_success_rate=float(hover_cfg["min_control_success_rate"]),
            )
            actual = franka_pose_to_matrix(segment_state.O_T_EE)[:3, 3]
            if not _inside(
                actual,
                hover_cfg["eef_workspace_min"],
                hover_cfg["eef_workspace_max"],
            ):
                raise RuntimeError("actual EEF pose left the commissioned workspace")
            segment_error = float(np.linalg.norm(actual - target))
            if not np.isfinite(segment_error) or segment_error > max_tracking_error:
                raise RuntimeError(
                    f"waypoint tracking error {segment_error * 1000:.1f}mm "
                    f"exceeds {max_tracking_error * 1000:.1f}mm; not continuing"
                )
            commanded_descent = target[2] < start[2] - DESCENT_Z_EPS_M
            if not commanded_descent and actual[2] < start[2] - 0.003:
                raise RuntimeError("actual EEF z decreased unexpectedly; not continuing")
            if commanded_descent and actual[2] < hard_clearance_z:
                raise RuntimeError(
                    "actual EEF crossed the requested object-top clearance floor"
                )
            print(
                f"[Move {command_index}] actual="
                f"{np.array2string(actual, precision=4)}, "
                f"error={segment_error * 1000:.1f}mm"
            )

        _raise_if_motion_cancelled(cancel_event)
        final_state = robot.read_once()
        validate_robot_state(
            final_state,
            require_idle=True,
            min_joint_margin=float(hover_cfg["min_joint_margin"]),
            min_success_rate=float(hover_cfg["min_control_success_rate"]),
        )
        final_xyz = franka_pose_to_matrix(final_state.O_T_EE)[:3, 3]
        _report_motion_progress(progress_callback, "motion complete")
        return final_xyz
    except MotionCancelled:
        # Cancellation is cooperative so pylibfranka is only stopped from the
        # worker that owns the active control loop.  This avoids calling into a
        # Robot object concurrently from the display thread.
        try:
            robot.stop()
        except Exception:
            pass
        _report_motion_progress(progress_callback, "motion cancelled")
        raise
    finally:
        watchdog.close()


def execute_hover_one_shot(
    robot: Any,
    pylibfranka: Any,
    perception_source: Any,
    extrinsics: ResolvedExtrinsics,
    hover_cfg: Dict[str, Any],
    target: StableTarget,
    *,
    allow_descent: bool = False,
    cancel_event: Optional[threading.Event] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> OneShotHoverResult:
    """Plan and execute one frozen, watchdog-guarded hover request.

    This is the synchronous callable intended for display/UI integration.  It
    always reads the current EEF pose immediately before planning.  The planner
    rises vertically to the commissioned transit height, translates in XY at
    that height, and only then (when ``allow_descent`` is true) descends over
    the frozen target.  A live perception watchdog remains mandatory during
    every segment.

    ``cancel_event`` is cooperative: setting it causes the control-owning
    thread to call ``robot.stop()`` and raise :class:`MotionCancelled`.  An
    operator must still use the Franka emergency stop for an emergency.
    """

    if (
        not extrinsics.calibrated
        or extrinsics.reference_frame != "robot_base"
        or not extrinsics.calibration_id
    ):
        raise RuntimeError(
            "interactive hover requires calibrated robot_base extrinsics"
        )
    if extrinsics.quality_status != "pass":
        raise RuntimeError(
            "interactive hover requires calibration quality status pass"
        )
    validate_rigid_transform(
        extrinsics.T_base_camera, name="interactive T_base_camera"
    )
    cfg = validate_hover_config(hover_cfg, allow_descent=allow_descent)
    center = _vec3(target.center, name="stable target center")
    if not np.isfinite(float(target.z_p95)):
        raise ValueError("stable target z_p95 must be finite")
    if not stable_enough(target, cfg):
        raise ValueError("interactive motion requires a commissioned stable target")

    _raise_if_motion_cancelled(cancel_event)
    _report_motion_progress(progress_callback, "planning frozen one-shot target")
    state = robot.read_once()
    validate_robot_state(
        state,
        require_idle=True,
        min_joint_margin=float(cfg["min_joint_margin"]),
        min_success_rate=float(cfg["min_control_success_rate"]),
    )
    current_xyz = franka_pose_to_matrix(state.O_T_EE)[:3, 3]
    goal_xyz, waypoints = plan_hover_waypoints(
        current_xyz,
        target,
        cfg,
        allow_descent=allow_descent,
    )
    _report_motion_progress(
        progress_callback,
        f"planned {len(waypoints)} segments to "
        f"{np.array2string(goal_xyz, precision=4)}",
    )

    if waypoints:
        final_xyz = execute_waypoints(
            robot,
            pylibfranka,
            waypoints,
            perception_source,
            extrinsics,
            cfg,
            center,
            allow_descent=allow_descent,
            frozen_z_p95=float(target.z_p95),
            minimum_descent_z=float(goal_xyz[2]) if allow_descent else None,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )
    else:
        final_xyz = current_xyz.copy()
        _report_motion_progress(progress_callback, "EEF is already at the hover goal")

    final_error = float(np.linalg.norm(final_xyz - goal_xyz))
    if not np.isfinite(final_error) or final_error > float(cfg["max_final_error"]):
        raise RuntimeError(
            f"final EEF error {final_error * 1000:.1f}mm exceeds limit; "
            "motion will not be retried automatically"
        )
    return OneShotHoverResult(
        current_xyz=np.asarray(current_xyz, dtype=np.float64).copy(),
        goal_xyz=np.asarray(goal_xyz, dtype=np.float64).copy(),
        final_xyz=np.asarray(final_xyz, dtype=np.float64).copy(),
        waypoint_count=len(waypoints),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Observe a calibrated robot-base object center, then plan or execute "
            "one rise-and-translate move above it. Default is plan-only."
        )
    )
    parser.add_argument("--config", default="configs/d435_default.yaml")
    parser.add_argument("--addr", default="tcp://127.0.0.1:5556")
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument("--observe-timeout", type=float, default=30.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--allow-descent",
        action="store_true",
        help="After high XY alignment, allow a guarded vertical descent",
    )
    parser.add_argument(
        "--clearance-m",
        type=float,
        default=None,
        help="Requested EEF clearance above the observed cloud top",
    )
    parser.add_argument("--confirm-calibration-id")
    parser.add_argument("--confirm-workspace-clear", action="store_true")
    parser.add_argument("--confirm-eef-clear", action="store_true")
    parser.add_argument("--confirm-descent-clear", action="store_true")
    parser.add_argument("--enforce-realtime", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    cfg = load_config(args.config)
    hover_cfg = dict(cfg["hover"])
    if args.clearance_m is not None:
        hover_cfg["hover_clearance"] = float(args.clearance_m)
    if args.allow_descent:
        if args.clearance_m is None:
            raise ValueError("--allow-descent requires an explicit --clearance-m")
        eef_min = list(hover_cfg["eef_workspace_min"])
        eef_min[2] = 0.190
        hover_cfg["eef_workspace_min"] = eef_min
        hover_cfg["target_z_min"] = 0.200
        hover_cfg["max_target_drift"] = min(
            float(hover_cfg["max_target_drift"]), 0.010
        )
    elif args.clearance_m is not None:
        raise ValueError("--clearance-m is only supported with --allow-descent")
    hover_cfg = validate_hover_config(
        hover_cfg, allow_descent=args.allow_descent
    )
    extrinsics = resolve_extrinsics(cfg.get("extrinsics"))
    if not extrinsics.calibrated or extrinsics.reference_frame != "robot_base":
        raise RuntimeError("hover motion requires calibrated robot_base packets")
    if extrinsics.quality_status != "pass":
        raise RuntimeError(
            f"hover motion requires calibration quality pass, got "
            f"{extrinsics.quality_status or 'missing'}"
        )

    subscriber = ZMQObjectPCDSubscriber(args.addr, timeout_ms=1000)
    try:
        print(
            f"[Observe] expecting calibration_id={extrinsics.calibration_id}, "
            f"camera_serial={extrinsics.camera_serial}"
        )
        target = collect_stable_target(
            subscriber, extrinsics, hover_cfg, timeout_s=args.observe_timeout
        )
        print(
            f"[Stable] visible center={np.array2string(target.center, precision=5)}m "
            f"z95={target.z_p95:.5f}m, p95={target.center_p95_m * 1000:.2f}mm, "
            f"speed={target.fitted_speed_mps * 1000:.2f}mm/s"
        )

        try:
            import pylibfranka
        except ImportError as exc:
            raise RuntimeError("pylibfranka is required to read Franka state") from exc
        realtime_config = (
            pylibfranka.RealtimeConfig.kEnforce
            if args.enforce_realtime
            else pylibfranka.RealtimeConfig.kIgnore
        )
        robot = pylibfranka.Robot(args.robot_ip, realtime_config)
        state = robot.read_once()
        validate_robot_state(
            state,
            require_idle=True,
            min_joint_margin=float(hover_cfg["min_joint_margin"]),
            min_success_rate=float(hover_cfg["min_control_success_rate"]),
        )
        current_xyz = franka_pose_to_matrix(state.O_T_EE)[:3, 3]
        goal, waypoints = plan_hover_waypoints(
            current_xyz,
            target,
            hover_cfg,
            allow_descent=args.allow_descent,
        )
        print(f"[Plan] current EEF={np.array2string(current_xyz, precision=5)}m")
        print(f"[Plan] frozen goal={np.array2string(goal, precision=5)}m")
        if args.allow_descent:
            hard_clearance_z = goal[2] - float(
                hover_cfg["descent_clearance_guard"]
            )
            print(
                f"[Plan] guarded descent: observed_top={target.z_p95:.5f}m, "
                f"requested_clearance={float(hover_cfg['hover_clearance']):.3f}m, "
                f"commanded_z={goal[2]:.5f}m, "
                f"hard_clearance_floor_z={hard_clearance_z:.5f}m"
            )
        for index, waypoint in enumerate(waypoints, start=1):
            print(f"[Plan] waypoint {index}: {np.array2string(waypoint, precision=5)}m")

        if not args.execute:
            print("[Plan only] No robot motion. Add all execution confirmations after review.")
            return 0
        if args.confirm_calibration_id != extrinsics.calibration_id:
            raise RuntimeError(
                "--confirm-calibration-id must exactly match the loaded calibration"
            )
        if not args.confirm_workspace_clear:
            raise RuntimeError("--confirm-workspace-clear is required for motion")
        if not args.confirm_eef_clear:
            raise RuntimeError(
                "--confirm-eef-clear is required; remove the large calibration board "
                "and verify the EEF/TCP clearance first"
            )
        if args.allow_descent and not args.confirm_descent_clear:
            raise RuntimeError(
                "--confirm-descent-clear is required for downward motion"
            )
        if not waypoints:
            print("[Done] EEF is already at the planned hover position")
            return 0

        print("[Execute] Frozen one-shot target; motion starts in 3 seconds")
        for remaining in (3, 2, 1):
            print(f"[Execute] {remaining}...")
            time.sleep(1.0)
        final_xyz = execute_waypoints(
            robot,
            pylibfranka,
            waypoints,
            args.addr,
            extrinsics,
            hover_cfg,
            target.center,
            allow_descent=args.allow_descent,
            frozen_z_p95=target.z_p95,
            minimum_descent_z=goal[2] if args.allow_descent else None,
        )
        final_error = float(np.linalg.norm(final_xyz - goal))
        print(
            f"[Done] final EEF={np.array2string(final_xyz, precision=5)}m, "
            f"goal error={final_error * 1000:.1f}mm"
        )
        if final_error > float(hover_cfg["max_final_error"]):
            raise RuntimeError(
                f"final EEF error {final_error * 1000:.1f}mm exceeds limit; "
                "motion will not be retried automatically"
            )
        return 0
    finally:
        subscriber.close()


if __name__ == "__main__":
    raise SystemExit(main())
