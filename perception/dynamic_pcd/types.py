from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    model: str = ""
    distortion: Tuple[float, ...] = ()

    def as_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.ppx], [0.0, self.fy, self.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def as_dist_coeffs(self) -> np.ndarray:
        if not self.distortion:
            return np.zeros((5, 1), dtype=np.float64)
        return np.asarray(self.distortion, dtype=np.float64).reshape(-1, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "fx": float(self.fx),
            "fy": float(self.fy),
            "ppx": float(self.ppx),
            "ppy": float(self.ppy),
            "model": str(self.model),
            "distortion": [float(value) for value in self.distortion],
        }


@dataclass
class RGBDFrame:
    color_bgr: np.ndarray
    depth_raw: np.ndarray
    depth_scale: float
    intrinsics: CameraIntrinsics
    timestamp: float
    frame_id: int
    retrieved_at_s: Optional[float] = None
    timestamp_domain: str = ""
    depth_timestamp_s: Optional[float] = None
    color_depth_timestamp_skew_s: Optional[float] = None
    color_depth_epoch_timestamp_skew_s: Optional[float] = None
    rejected_timestamp_skew_frames: int = 0
    last_rejected_color_depth_skew_s: Optional[float] = None
    dropped_queued_framesets: int = 0
    sensor_frame_number: int = 0
    depth_sensor_frame_number: int = 0
    rejected_transport_stale_frames: int = 0
    last_rejected_transport_age_s: Optional[float] = None
    retrieved_monotonic_s: Optional[float] = None
    host_clock_pair_span_s: Optional[float] = None
    capture_diagnostic: Optional[Dict[str, Any]] = None

    @property
    def depth_m(self) -> np.ndarray:
        return self.depth_raw.astype(np.float32) * float(self.depth_scale)


@dataclass
class MaskResult:
    mask: np.ndarray
    bbox_xyxy: np.ndarray
    score: float
    valid: bool
    message: str = ""
    # Structured provenance for initialization/recovery safety checks.  Empty
    # retains compatibility with third-party/test producers that predate it.
    source: str = ""
    # Optional exact-current RGB semantic evidence for task-specific policy
    # point-cloud fallback.  This is deliberately separate from ``valid``:
    # guarded publication may reject robot-facing RGB-D authority while SAM2
    # still owns a fresh, non-empty current-frame silhouette.  Consumers must
    # opt in explicitly and may never use this field to advance provider
    # tracker/full-target/recovery authority.
    policy_semantic_mask: Optional[np.ndarray] = None
    policy_semantic_valid: bool = False
    policy_semantic_source: str = ""


@dataclass
class ObjectPCDPacket:
    pcd_current: Optional[np.ndarray]
    pcd_history: Optional[np.ndarray]
    center: Optional[np.ndarray]
    velocity: Optional[np.ndarray]
    bbox_xyxy: Optional[np.ndarray]
    timestamp: float
    frame_id: int
    valid: bool
    age: float = 0.0
    message: str = ""
    debug: Optional[Dict[str, Any]] = None
    # Fixed-N points before optional object centering. They are expressed in
    # ``reference_frame``. ``pcd_current`` retains the configured policy form.
    pcd_reference: Optional[np.ndarray] = None
    reference_frame: str = "camera_color_optical_frame"
    point_frame: str = "camera_color_optical_frame"
    calibration_id: Optional[str] = None
    T_base_camera: Optional[np.ndarray] = None
    camera_serial: Optional[str] = None

    def to_policy_obs(self) -> Dict[str, Any]:
        return {
            "object_pcd": self.pcd_current,
            "object_pcd_history": self.pcd_history,
            "object_pcd_reference": self.pcd_reference,
            "object_center": self.center,
            "object_velocity": self.velocity,
            "bbox_xyxy": self.bbox_xyxy,
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "valid": self.valid,
            "reference_frame": self.reference_frame,
            "point_frame": self.point_frame,
            "calibration_id": self.calibration_id,
            "T_base_camera": self.T_base_camera,
            "camera_serial": self.camera_serial,
        }
