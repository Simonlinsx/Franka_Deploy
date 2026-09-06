from __future__ import annotations

from typing import Iterable

import numpy as np


def validate_rigid_transform(
    matrix: np.ndarray,
    *,
    name: str = "transform",
    atol: float = 1e-5,
) -> np.ndarray:
    """Return a validated SE(3) matrix as float64."""

    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {transform.shape}")
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} contains NaN or infinity")
    if not np.allclose(
        transform[3], [0.0, 0.0, 0.0, 1.0], atol=atol, rtol=0.0
    ):
        raise ValueError(f"{name} last row must be [0, 0, 0, 1]")

    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol, rtol=0.0):
        raise ValueError(f"{name} rotation is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=atol, rtol=0.0):
        raise ValueError(
            f"{name} rotation determinant must be +1, got {determinant:.8f}"
        )
    return transform


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map an ``[..., 3]`` point array through an SE(3) transform."""

    transform = validate_rigid_transform(matrix)
    values = np.asarray(points)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError(f"points must have shape (..., 3), got {values.shape}")
    result = values.astype(np.float64) @ transform[:3, :3].T + transform[:3, 3]
    return result.astype(np.result_type(values.dtype, np.float32), copy=False)


def franka_pose_to_matrix(values: Iterable[float]) -> np.ndarray:
    """Convert libfranka's column-major ``O_T_EE`` into a 4x4 matrix."""

    pose = np.asarray(list(values), dtype=np.float64)
    if pose.shape != (16,):
        raise ValueError(f"Franka O_T_EE must contain 16 values, got {pose.shape}")
    return validate_rigid_transform(
        pose.reshape((4, 4), order="F"), name="O_T_EE"
    )
