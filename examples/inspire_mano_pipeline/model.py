from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


HARDWARE_JOINTS: Tuple[str, ...] = (
    "pinky",
    "ring",
    "middle",
    "index",
    "thumb_bend",
    "thumb_rotate",
)

DEX_JOINTS: Tuple[str, ...] = (
    "pinky_proximal_joint",
    "ring_proximal_joint",
    "middle_proximal_joint",
    "index_proximal_joint",
    "thumb_proximal_pitch_joint",
    "thumb_proximal_yaw_joint",
)


@dataclass(frozen=True)
class CameraFrame:
    color_bgr: np.ndarray
    depth_m: Optional[np.ndarray]
    captured_at_monotonic: float
    frame_number: int
    depth_frame_number: Optional[int] = None
    color_sensor_timestamp_ms: Optional[float] = None
    depth_sensor_timestamp_ms: Optional[float] = None
    sensor_timestamp_domain: Optional[str] = None
    ready_at_monotonic: Optional[float] = None


@dataclass(frozen=True)
class ManoDetection:
    """One WiLoR detection normalized for retargeting."""

    is_right: bool
    bbox_xyxy: np.ndarray
    keypoints_3d_raw: np.ndarray
    keypoints_3d_canonical: np.ndarray
    keypoints_2d: np.ndarray
    global_orient: np.ndarray
    hand_pose: np.ndarray
    betas: np.ndarray
    vertices: np.ndarray
    captured_at_monotonic: float
    frame_number: int
    palm_depth_m: Optional[float] = None
    control_palm_depth_m: Optional[float] = None
    palm_depth_source: str = "missing"
    palm_depth_reason: str = "not_estimated"
    palm_depth_evidence_at_monotonic: Optional[float] = None
    palm_depth_age_seconds: Optional[float] = None
    palm_depth_roi_sample_count: int = 0
    palm_depth_inlier_count: int = 0
    palm_depth_valid_pixel_count: int = 0
    palm_depth_radius_px: int = 0
    # Optional image-space MANO surface data for visualization.  Retargeting
    # continues to depend only on the validated 21-joint representation.
    vertices_2d: Optional[np.ndarray] = None
    mesh_faces: Optional[np.ndarray] = None


@dataclass(frozen=True)
class RetargetOutput:
    qpos: np.ndarray
    hardware_targets: Tuple[int, ...]
    backend: str
    raw_qpos: Optional[np.ndarray] = None
    raw_hardware_targets: Optional[Tuple[int, ...]] = None
