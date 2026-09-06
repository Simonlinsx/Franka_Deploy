"""Conservative point-cloud collision audit for the FR3/RH56 adapter sweep.

The check uses the adapter's full Ø70 x 17.8 mm enclosing cylinder and ignores
all holes.  A clear report only means that no supplied point sample entered
that inflated volume; it is intentionally not an authoritative replacement
for robot, hand, table, or continuous mesh collision planning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .control_plan import AdapterGeometry, inverse_rigid_transform, validate_rigid_transform


@dataclass(frozen=True)
class AdapterSweepCollisionReport:
    trajectory_samples: int
    point_count: int
    margin_m: float
    colliding_point_indices: Tuple[int, ...]
    colliding_sample_indices: Tuple[int, ...]
    authoritative: bool = False

    @property
    def collision_free(self) -> bool:
        return not self.colliding_point_indices


def _points(value: Sequence[Sequence[float]]) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("collision points must have shape (N, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("collision points contain NaN or infinity")
    return points


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1e-12:
        return np.zeros(3, dtype=np.float64)
    if np.pi - angle < 1e-6:
        # Stable extraction near pi from the symmetric part.
        diagonal = np.maximum((np.diag(rotation) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(diagonal)
        index = int(np.argmax(axis))
        if axis[index] < 1e-8:
            raise ValueError("cannot interpolate a near-pi rotation")
        for other in range(3):
            if other != index:
                axis[other] = (
                    rotation[index, other] + rotation[other, index]
                ) / (4.0 * axis[index])
        axis /= np.linalg.norm(axis)
        return axis * angle
    axis = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * np.sin(angle))
    return axis * angle


def _rotation_matrix(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = vector / angle
    x, y, z = axis
    skew = np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def interpolate_pose(T_start: np.ndarray, T_end: np.ndarray, alpha: float) -> np.ndarray:
    """Geodesic SO(3) plus linear translation interpolation."""

    start = validate_rigid_transform(T_start, "T_start")
    end = validate_rigid_transform(T_end, "T_end")
    fraction = float(alpha)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("alpha must be finite in [0, 1]")
    relative = end[:3, :3] @ start[:3, :3].T
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = _rotation_matrix(fraction * _rotation_vector(relative)) @ start[:3, :3]
    pose[:3, 3] = (1.0 - fraction) * start[:3, 3] + fraction * end[:3, 3]
    return validate_rigid_transform(pose, "interpolated pose")


def audit_adapter_sweep_against_points(
    points_reference: Sequence[Sequence[float]],
    *,
    T_reference_EE_start: np.ndarray,
    T_reference_EE_end: np.ndarray,
    T_EE_adapter: np.ndarray,
    adapter: AdapterGeometry = AdapterGeometry(),
    margin_m: float = 0.005,
    trajectory_samples: int = 21,
) -> AdapterSweepCollisionReport:
    """Check supplied points against an inflated full adapter envelope sweep."""

    points = _points(points_reference)
    ee_to_adapter = validate_rigid_transform(T_EE_adapter, "T_EE_adapter")
    margin = float(margin_m)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("margin_m must be finite and non-negative")
    if isinstance(trajectory_samples, bool) or int(trajectory_samples) < 2:
        raise ValueError("trajectory_samples must be an integer >= 2")
    samples = int(trajectory_samples)
    radius = 0.5 * adapter.disk_diameter_m + margin
    lower_z = -margin
    upper_z = adapter.total_height_m + margin

    colliding_points = set()
    colliding_samples = set()
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    for sample_index, alpha in enumerate(np.linspace(0.0, 1.0, samples)):
        T_reference_EE = interpolate_pose(
            T_reference_EE_start, T_reference_EE_end, float(alpha)
        )
        T_reference_adapter = validate_rigid_transform(
            T_reference_EE @ ee_to_adapter, "T_reference_adapter"
        )
        T_adapter_reference = inverse_rigid_transform(T_reference_adapter)
        local = (T_adapter_reference @ homogeneous.T).T[:, :3]
        radial_squared = np.sum(local[:, :2] ** 2, axis=1)
        inside = (
            (radial_squared <= radius * radius)
            & (local[:, 2] >= lower_z)
            & (local[:, 2] <= upper_z)
        )
        indices = np.flatnonzero(inside)
        if len(indices):
            colliding_samples.add(sample_index)
            colliding_points.update(int(index) for index in indices)

    return AdapterSweepCollisionReport(
        trajectory_samples=samples,
        point_count=len(points),
        margin_m=margin,
        colliding_point_indices=tuple(sorted(colliding_points)),
        colliding_sample_indices=tuple(sorted(colliding_samples)),
        authoritative=False,
    )


__all__ = [
    "AdapterSweepCollisionReport",
    "audit_adapter_sweep_against_points",
    "interpolate_pose",
]

