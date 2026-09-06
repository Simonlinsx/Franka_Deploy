"""NumPy runtime for the frozen V61 thrown-object 17-D contracts.

This is a direct deploy-time port of ``thrown_ballistic_contracts.py`` from
the SHA-bound runtime-alignment supplement.  Inputs are raw metric histories;
checkpoint normalization must happen only in the Student core.

Authoritative packaged source SHA256:
547c570ba93593541e6e1db466463e759b155c683211a007c2de69bc7ed9b956
"""

from __future__ import annotations

import numpy as np


THROWN_V35_BALLISTIC_17D_V1 = "thrown_v35_ballistic_17d_v1"
THROWN_VISUAL_BALLISTIC_17D_20HZ_V2 = (
    "thrown_visual_ballistic_17d_20hz_v2"
)
SUPPORTED_17D_CONTRACTS = frozenset(
    (THROWN_V35_BALLISTIC_17D_V1, THROWN_VISUAL_BALLISTIC_17D_20HZ_V2)
)
CONTRACT_DIM = 17
PACKAGED_SOURCE_SHA256 = (
    "547c570ba93593541e6e1db466463e759b155c683211a007c2de69bc7ed9b956"
)


def _float32(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def quat_rotate(quat_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quaternion = _float32(quat_wxyz, "quaternion")
    value = _float32(vector, "vector")
    scalar = quaternion[..., :1]
    qvec = quaternion[..., 1:]
    first = np.cross(np.broadcast_to(qvec, value.shape), value, axis=-1)
    second = np.cross(np.broadcast_to(qvec, value.shape), first, axis=-1)
    return value + np.float32(2.0) * (scalar * first + second)


def quat_rotate_inverse(quat_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quaternion = _float32(quat_wxyz, "quaternion")
    conjugate = np.concatenate(
        (quaternion[..., :1], -quaternion[..., 1:]), axis=-1
    )
    return quat_rotate(conjugate, vector)


def align_pointcloud_history_to_current_palm(
    pointcloud_seq: np.ndarray,
    proprio_seq: np.ndarray,
) -> np.ndarray:
    points = _float32(pointcloud_seq, "pointcloud_seq")
    proprio = _float32(proprio_seq, "proprio_seq")
    if points.ndim != 4 or points.shape[-1] < 3:
        raise ValueError("pointcloud_seq must have shape [B,H,N,F] with F>=3")
    if proprio.ndim != 3 or proprio.shape[:2] != points.shape[:2]:
        raise ValueError("proprio_seq must match pointcloud_seq B/H dimensions")
    if proprio.shape[-1] < 33:
        raise ValueError("proprio_seq must contain the palm pose")
    source_position = proprio[..., 26:29]
    source_quaternion = proprio[..., 29:33]
    target_position = np.broadcast_to(
        source_position[:, -1:], source_position.shape
    )
    target_quaternion = np.broadcast_to(
        source_quaternion[:, -1:], source_quaternion.shape
    )
    world = quat_rotate(
        source_quaternion[..., None, :], points[..., :3]
    ) + source_position[..., None, :]
    result = points.copy()
    result[..., :3] = quat_rotate_inverse(
        target_quaternion[..., None, :], world - target_position[..., None, :]
    )
    return result


def deployable_visual_ballistic_future_contract(
    pointcloud_seq: np.ndarray,
    valid_seq: np.ndarray,
    proprio_seq: np.ndarray,
    *,
    control_dt_s: float = 0.05,
    history_frames: int = 4,
    min_valid_points: int = 16,
    min_displacement_m: float = 0.005,
    max_speed_m_s: float = 6.0,
    gravity_m_s2: float = 9.81,
    minimum_world_height_m: float = 0.36,
    palm_speed_m_s: float = 0.65,
) -> np.ndarray:
    points = _float32(pointcloud_seq, "pointcloud_seq")
    valid = _float32(valid_seq, "valid_seq")
    proprio = _float32(proprio_seq, "proprio_seq")
    if points.ndim != 4 or points.shape[-1] < 3:
        raise ValueError("pointcloud_seq must have shape [B,H,N,F] with F>=3")
    if valid.shape != points.shape[:-1]:
        raise ValueError("valid_seq must match pointcloud_seq [B,H,N]")
    if proprio.ndim != 3 or proprio.shape[:2] != points.shape[:2]:
        raise ValueError("proprio_seq must match pointcloud_seq B/H dimensions")
    if proprio.shape[-1] < 54:
        raise ValueError("visual ballistic contract requires 54D proprio")
    if control_dt_s <= 0.0 or history_frames < 2 or min_valid_points <= 0:
        raise ValueError("invalid visual ballistic temporal configuration")
    if min_displacement_m < 0.0 or max_speed_m_s <= 0.0:
        raise ValueError("invalid visual ballistic motion bounds")

    aligned = align_pointcloud_history_to_current_palm(points, proprio)
    history_valid = (valid > 0.0) & np.all(
        np.isfinite(aligned[..., :3]), axis=-1
    )
    counts = np.sum(history_valid, axis=-1)
    xyz = np.where(history_valid[..., None], aligned[..., :3], np.float32(0.0))
    centers = np.sum(xyz, axis=2) / np.maximum(counts, 1).astype(
        np.float32
    )[..., None]
    batch_size, history = centers.shape[:2]
    if history < 2:
        raise ValueError("pointcloud history is too short for visual velocity")
    frame_valid = counts >= int(min_valid_points)
    window_start = max(history - int(history_frames), 0)
    previous_index = np.full(batch_size, history, dtype=np.int64)
    for index in range(window_start, history - 1):
        select = (previous_index == history) & frame_valid[:, index]
        previous_index[select] = index
    has_previous = previous_index < history
    safe_previous = np.minimum(previous_index, history - 1)
    batch = np.arange(batch_size)
    previous_center = centers[batch, safe_previous]
    current_center = centers[:, -1]
    elapsed_steps = np.maximum(history - 1 - safe_previous, 1)
    displacement = current_center - previous_center
    velocity_palm = displacement / (
        elapsed_steps.astype(np.float32)[:, None] * np.float32(control_dt_s)
    )
    displacement_norm = np.linalg.norm(displacement, axis=-1)
    speed = np.linalg.norm(velocity_palm, axis=-1)
    motion_valid = (
        frame_valid[:, -1]
        & has_previous
        & (displacement_norm >= np.float32(min_displacement_m))
        & (speed <= np.float32(max_speed_m_s))
    )
    velocity_palm = np.where(
        motion_valid[:, None], velocity_palm, np.float32(0.0)
    )

    palm_position = proprio[:, -1, 26:29]
    palm_quaternion = proprio[:, -1, 29:33]
    gravity_world = np.broadcast_to(
        np.asarray((0.0, 0.0, -gravity_m_s2), dtype=np.float32),
        (batch_size, 3),
    )
    gravity_palm = quat_rotate_inverse(palm_quaternion, gravity_world)
    fingertips = proprio[:, -1, 39:54].reshape(-1, 5, 3)
    catch_reference = np.float32(0.5) * (
        fingertips[:, 0] + np.mean(fingertips[:, 1:], axis=1)
    )

    trajectory_times = np.asarray((0.05, 0.10, 0.20, 0.40), dtype=np.float32)
    future = (
        current_center[:, None, :]
        + velocity_palm[:, None, :] * trajectory_times[None, :, None]
        + np.float32(0.5)
        * gravity_palm[:, None, :]
        * np.square(trajectory_times[None, :, None])
    )
    trajectory = (
        (future - catch_reference[:, None, :]) / np.float32(0.50)
    ) * motion_valid.astype(np.float32)[:, None, None]

    candidate_times = np.asarray(
        (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.36, 0.42, 0.50, 0.60, 0.72, 0.84),
        dtype=np.float32,
    )
    candidates = (
        current_center[:, None, :]
        + velocity_palm[:, None, :] * candidate_times[None, :, None]
        + np.float32(0.5)
        * gravity_palm[:, None, :]
        * np.square(candidate_times[None, :, None])
    )
    distance = np.linalg.norm(
        candidates - catch_reference[:, None, :], axis=-1
    )
    required_time = np.float32(0.05) + distance / np.float32(palm_speed_m_s)
    candidates_world = quat_rotate(
        palm_quaternion[:, None, :], candidates
    ) + palm_position[:, None, :]
    height_valid = candidates_world[..., 2] >= np.float32(minimum_world_height_m)
    feasible = (
        height_valid
        & (candidate_times[None, :] >= required_time)
        & motion_valid[:, None]
    )
    any_feasible = np.any(feasible, axis=-1)
    first_feasible = np.argmax(feasible.astype(np.float32), axis=-1)
    reach_slack = candidate_times[None, :] - required_time
    fallback = np.argmax(
        np.where(height_valid, reach_slack, np.float32(-np.inf)), axis=-1
    )
    selected = np.where(any_feasible, first_feasible, fallback)
    intercept_valid = motion_valid & any_feasible
    intercept_position = candidates[batch, selected]
    intercept_time = candidate_times[selected]
    intercept = np.concatenate(
        (
            (intercept_position - catch_reference)
            / np.float32(0.50)
            * intercept_valid.astype(np.float32)[:, None],
            (
                intercept_time
                / np.float32(0.84)
                * intercept_valid.astype(np.float32)
            )[:, None],
            intercept_valid.astype(np.float32)[:, None],
        ),
        axis=-1,
    )
    return np.concatenate((intercept, trajectory.reshape(batch_size, 12)), axis=-1)


def deployable_thrown_v35_future_contract(
    compact_privileged: np.ndarray,
    metric_proprio: np.ndarray,
) -> np.ndarray:
    compact = _float32(compact_privileged, "compact_privileged")
    proprio = _float32(metric_proprio, "metric_proprio")
    if compact.ndim != 2 or compact.shape[-1] < 6:
        raise ValueError("compact_privileged must have shape [B,C] with C>=6")
    if proprio.ndim != 2 or proprio.shape[0] != compact.shape[0]:
        raise ValueError("metric_proprio must have shape [B,P] with matching B")
    if proprio.shape[-1] < 54:
        raise ValueError("thrown future contract requires 54D proprio")

    center_palm = compact[:, 0:3]
    center_velocity_palm = compact[:, 3:6]
    palm_position = proprio[:, 26:29]
    palm_quaternion = proprio[:, 29:33]
    palm_linear_velocity = proprio[:, 33:36] / np.float32(0.1)
    palm_angular_velocity = proprio[:, 36:39] / np.float32(0.1)
    inverse_quaternion = np.concatenate(
        (palm_quaternion[:, :1], -palm_quaternion[:, 1:]), axis=-1
    )
    angular_palm = quat_rotate(inverse_quaternion, palm_angular_velocity)
    object_velocity = palm_linear_velocity + quat_rotate(
        palm_quaternion,
        center_velocity_palm + np.cross(angular_palm, center_palm, axis=-1),
    )
    fingertips = proprio[:, 39:54].reshape(-1, 5, 3)
    catch_reference_palm = np.float32(0.5) * (
        fingertips[:, 0] + np.mean(fingertips[:, 1:], axis=1)
    )
    catch_reference = palm_position + quat_rotate(
        palm_quaternion, catch_reference_palm
    )
    object_position = palm_position + quat_rotate(palm_quaternion, center_palm)
    gravity = np.asarray((0.0, 0.0, -9.81), dtype=np.float32)
    trajectory_times = np.asarray((0.05, 0.10, 0.20, 0.40), dtype=np.float32)
    future = (
        object_position[:, None, :]
        + object_velocity[:, None, :] * trajectory_times[None, :, None]
        + np.float32(0.5)
        * gravity[None, None, :]
        * np.square(trajectory_times[None, :, None])
    )
    trajectory = ((future - catch_reference[:, None, :]) / np.float32(0.50)).reshape(
        compact.shape[0], 12
    )
    candidate_times = np.asarray(
        (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.36, 0.42, 0.50, 0.60, 0.72, 0.84),
        dtype=np.float32,
    )
    candidates = (
        object_position[:, None, :]
        + object_velocity[:, None, :] * candidate_times[None, :, None]
        + np.float32(0.5)
        * gravity[None, None, :]
        * np.square(candidate_times[None, :, None])
    )
    distance = np.linalg.norm(candidates - catch_reference[:, None, :], axis=-1)
    required_time = np.float32(0.05) + distance / np.float32(0.65)
    height_valid = candidates[..., 2] >= np.float32(0.36)
    feasible = height_valid & (candidate_times[None, :] >= required_time)
    any_feasible = np.any(feasible, axis=-1)
    first_feasible = np.argmax(feasible.astype(np.float32), axis=-1)
    reach_slack = candidate_times[None, :] - required_time
    fallback = np.argmax(
        np.where(height_valid, reach_slack, np.float32(-np.inf)), axis=-1
    )
    selected = np.where(any_feasible, first_feasible, fallback)
    batch = np.arange(compact.shape[0])
    intercept_valid = any_feasible.astype(np.float32)
    intercept_position = candidates[batch, selected]
    intercept_time = candidate_times[selected]
    intercept = np.concatenate(
        (
            (intercept_position - catch_reference)
            / np.float32(0.50)
            * intercept_valid[:, None],
            (intercept_time / np.float32(0.84) * intercept_valid)[:, None],
            intercept_valid[:, None],
        ),
        axis=-1,
    )
    return np.concatenate((intercept, trajectory), axis=-1)


def build_17d_future_contract(
    contract_id: str,
    *,
    metric_proprio_seq: np.ndarray,
    metric_pointcloud_seq: np.ndarray | None = None,
    valid_seq: np.ndarray | None = None,
    predicted_compact_privileged: np.ndarray | None = None,
) -> np.ndarray:
    if contract_id == THROWN_VISUAL_BALLISTIC_17D_20HZ_V2:
        if metric_pointcloud_seq is None or valid_seq is None:
            raise ValueError("visual-ballistic v2 requires pointcloud and valid histories")
        return deployable_visual_ballistic_future_contract(
            metric_pointcloud_seq,
            valid_seq,
            metric_proprio_seq,
            control_dt_s=0.05,
            history_frames=4,
            min_valid_points=16,
        )
    if contract_id == THROWN_V35_BALLISTIC_17D_V1:
        if predicted_compact_privileged is None:
            raise ValueError("ballistic v1 requires the predicted compact state")
        return deployable_thrown_v35_future_contract(
            predicted_compact_privileged, metric_proprio_seq[:, -1]
        )
    raise ValueError(
        f"unsupported 17D contract {contract_id!r}; expected "
        f"{sorted(SUPPORTED_17D_CONTRACTS)}"
    )


__all__ = [
    "CONTRACT_DIM",
    "PACKAGED_SOURCE_SHA256",
    "SUPPORTED_17D_CONTRACTS",
    "THROWN_V35_BALLISTIC_17D_V1",
    "THROWN_VISUAL_BALLISTIC_17D_20HZ_V2",
    "align_pointcloud_history_to_current_palm",
    "build_17d_future_contract",
    "deployable_thrown_v35_future_contract",
    "deployable_visual_ballistic_future_contract",
    "quat_rotate",
    "quat_rotate_inverse",
]
