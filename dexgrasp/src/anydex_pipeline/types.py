from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np


def _points(value: np.ndarray, name: str, *, allow_empty: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3], got {array.shape}")
    if not allow_empty and len(array) == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return np.ascontiguousarray(array)


def _colors(value: Optional[np.ndarray], count: int, name: str) -> np.ndarray:
    if value is None:
        return np.full((count, 3), 0.55, dtype=np.float32)
    array = _points(value, name, allow_empty=count == 0)
    if len(array) != count:
        raise ValueError(f"{name} has {len(array)} rows, expected {count}")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must be RGB values in [0, 1]")
    return array


def rigid_transform(value: np.ndarray, name: str = "transform") -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape [4, 4], got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinity")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-7):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation determinant must be +1")
    return matrix.copy()


@dataclass(frozen=True)
class PointCloudObservation:
    """Scene and segmented object points in one named reference frame."""

    scene_points: np.ndarray
    object_points: np.ndarray
    reference_frame: str
    T_reference_camera: np.ndarray
    scene_colors: Optional[np.ndarray] = None
    object_colors: Optional[np.ndarray] = None
    frame_id: int = -1
    timestamp_s: float = 0.0
    calibration_id: str = ""
    camera_serial: str = ""

    def __post_init__(self) -> None:
        scene = _points(self.scene_points, "scene_points", allow_empty=True)
        obj = _points(self.object_points, "object_points")
        if not str(self.reference_frame).strip():
            raise ValueError("reference_frame must be a non-empty name")
        transform = rigid_transform(self.T_reference_camera, "T_reference_camera")
        scene_rgb = _colors(self.scene_colors, len(scene), "scene_colors")
        object_rgb = _colors(self.object_colors, len(obj), "object_colors")
        if not np.isfinite(float(self.timestamp_s)):
            raise ValueError("timestamp_s must be finite")
        object.__setattr__(self, "scene_points", scene)
        object.__setattr__(self, "object_points", obj)
        object.__setattr__(self, "scene_colors", scene_rgb)
        object.__setattr__(self, "object_colors", object_rgb)
        object.__setattr__(self, "T_reference_camera", transform)
        object.__setattr__(self, "reference_frame", str(self.reference_frame))
        object.__setattr__(self, "calibration_id", str(self.calibration_id or ""))
        object.__setattr__(self, "camera_serial", str(self.camera_serial or ""))


@dataclass(frozen=True)
class GraspCandidate:
    """One canonical two-finger/contact grasp and optional Inspire palm pose.

    ``T_reference_grasp`` maps canonical grasp-local coordinates into the
    observation reference frame.  The canonical local +X axis is the AnyDex /
    GraspNet approach (insertion) direction.

    ``T_reference_hand`` is a separate, optional palm/hand transform.  Its axes
    must not be assumed to match the canonical grasp axes.
    """

    T_reference_grasp: np.ndarray
    score: float
    width_m: float = np.nan
    depth_m: float = np.nan
    collision_free: bool = True
    grasp_type_id: int = -1
    T_reference_hand: Optional[np.ndarray] = None
    hand_angles: Optional[np.ndarray] = None
    source_index: int = -1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pose = rigid_transform(self.T_reference_grasp, "T_reference_grasp")
        hand_pose = None
        if self.T_reference_hand is not None:
            hand_pose = rigid_transform(self.T_reference_hand, "T_reference_hand")
        angles = None
        if self.hand_angles is not None:
            angles = np.asarray(self.hand_angles, dtype=np.float64)
            if angles.shape != (6,) or not np.isfinite(angles).all():
                raise ValueError("hand_angles must be a finite [6] array")
            angles = angles.copy()
        if not np.isfinite(float(self.score)):
            raise ValueError("score must be finite")
        for name, value in (("width_m", self.width_m), ("depth_m", self.depth_m)):
            if not (np.isnan(value) or np.isfinite(float(value))):
                raise ValueError(f"{name} must be finite or NaN")
        object.__setattr__(self, "T_reference_grasp", pose)
        object.__setattr__(self, "T_reference_hand", hand_pose)
        object.__setattr__(self, "hand_angles", angles)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class GraspResult:
    candidates: Sequence[GraspCandidate]
    backend_name: str
    reference_frame: str
    inference_time_s: float = 0.0
    selected_index: int = 0
    model_name: str = ""
    checkpoint_sha256: str = ""
    inference_points: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        selected = int(self.selected_index)
        if candidates and not 0 <= selected < len(candidates):
            raise ValueError("selected_index is outside candidates")
        if not candidates and selected not in (-1, 0):
            raise ValueError("selected_index must be -1 or 0 when there are no candidates")
        if not str(self.backend_name).strip():
            raise ValueError("backend_name must not be empty")
        if not str(self.reference_frame).strip():
            raise ValueError("reference_frame must not be empty")
        points = None
        if self.inference_points is not None:
            points = _points(self.inference_points, "inference_points", allow_empty=True)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "selected_index", selected if candidates else -1)
        object.__setattr__(self, "inference_points", points)

