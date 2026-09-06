from __future__ import annotations

import numpy as np

from .types import rigid_transform


def transform_points(T_target_source: np.ndarray, points_source: np.ndarray) -> np.ndarray:
    """Apply ``p_target = R @ p_source + t`` to an ``[N, 3]`` cloud."""

    transform = rigid_transform(T_target_source, "T_target_source")
    points = np.asarray(points_source, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_source must have shape [N, 3], got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("points_source contains NaN or infinity")
    result = points @ transform[:3, :3].T + transform[:3, 3]
    return result.astype(np.float32, copy=False)


def change_pose_reference(
    T_target_source: np.ndarray, T_source_local: np.ndarray
) -> np.ndarray:
    """Return ``T_target_local = T_target_source @ T_source_local``."""

    left = rigid_transform(T_target_source, "T_target_source")
    right = rigid_transform(T_source_local, "T_source_local")
    return rigid_transform(left @ right, "T_target_local")


def inverse_transform(T_target_source: np.ndarray) -> np.ndarray:
    transform = rigid_transform(T_target_source, "T_target_source")
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -(inverse[:3, :3] @ transform[:3, 3])
    return inverse


def pose_from_axes(
    origin: np.ndarray,
    approach_x: np.ndarray,
    closing_y_hint: np.ndarray,
) -> np.ndarray:
    """Construct a right-handed canonical grasp pose.

    Local +X is the approach direction.  ``closing_y_hint`` is projected onto
    the plane normal to +X and becomes local +Y; local +Z completes the frame.
    """

    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    x_axis = np.asarray(approach_x, dtype=np.float64).reshape(3)
    hint = np.asarray(closing_y_hint, dtype=np.float64).reshape(3)
    if not np.isfinite(np.r_[origin, x_axis, hint]).all():
        raise ValueError("origin and axes must be finite")
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-9:
        raise ValueError("approach_x must be non-zero")
    x_axis /= x_norm
    y_axis = hint - x_axis * np.dot(hint, x_axis)
    if np.linalg.norm(y_axis) < 1e-8:
        fallback = np.array((0.0, 0.0, 1.0))
        if abs(np.dot(fallback, x_axis)) > 0.9:
            fallback = np.array((0.0, 1.0, 0.0))
        y_axis = fallback - x_axis * np.dot(fallback, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    pose[:3, 3] = origin
    return rigid_transform(pose, "pose")

