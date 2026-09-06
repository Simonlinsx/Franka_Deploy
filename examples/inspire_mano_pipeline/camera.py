from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Optional

import numpy as np

from .model import CameraFrame


PALM_KEYPOINT_INDICES = (0, 5, 9, 13, 17)


@dataclass(frozen=True)
class PalmDepthSpatialEstimate:
    """One-frame palm-depth estimate with enough evidence for safety logging."""

    depth_m: Optional[float]
    reason: str
    roi_sample_count: int = 0
    inlier_count: int = 0
    valid_pixel_count: int = 0
    radius_px: int = 0
    palm_center_xy: Optional[tuple[float, float]] = None
    palm_width_px: Optional[float] = None


@dataclass(frozen=True)
class PalmDepthObservation:
    """Measured or conservatively held palm depth for the hardware gate."""

    depth_m: Optional[float]
    raw_depth_m: Optional[float]
    source: str
    reason: str
    evidence_at_monotonic: Optional[float]
    age_seconds: Optional[float]
    roi_sample_count: int
    inlier_count: int
    valid_pixel_count: int
    radius_px: int


def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != (4,) or b.shape != (4,) or not (
        np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    ):
        return 0.0
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


class PalmDepthStabilizer:
    """Bridge at most a couple of depth-hole frames on one continuous hand track.

    Held observations retain the timestamp of the last *measured* depth.  The
    serial watchdog therefore cannot be kept alive by repeatedly held values.
    A hand at the image boundary is never eligible for holding.
    """

    def __init__(
        self,
        max_hold_seconds: float = 0.12,
        min_bbox_iou: float = 0.50,
        max_center_shift_palm_widths: float = 0.50,
    ) -> None:
        if not 0.0 <= max_hold_seconds <= 0.20:
            raise ValueError("max_hold_seconds must be in 0..0.20")
        if not 0.0 <= min_bbox_iou <= 1.0:
            raise ValueError("min_bbox_iou must be in 0..1")
        if max_center_shift_palm_widths <= 0.0:
            raise ValueError("max_center_shift_palm_widths must be positive")
        self.max_hold_seconds = float(max_hold_seconds)
        self.min_bbox_iou = float(min_bbox_iou)
        self.max_center_shift_palm_widths = float(max_center_shift_palm_widths)
        self.reset()

    def reset(self) -> None:
        self._last_depth_m: Optional[float] = None
        self._last_evidence_at: Optional[float] = None
        self._last_bbox: Optional[np.ndarray] = None
        self._last_center: Optional[np.ndarray] = None
        self._last_palm_width_px: Optional[float] = None

    def apply(
        self,
        estimate: PalmDepthSpatialEstimate,
        bbox_xyxy: np.ndarray,
        captured_at_monotonic: float,
    ) -> PalmDepthObservation:
        timestamp = float(captured_at_monotonic)
        if not np.isfinite(timestamp):
            raise ValueError("depth observation timestamp must be finite")
        bbox = np.asarray(bbox_xyxy, dtype=np.float32)
        if bbox.shape != (4,) or not np.all(np.isfinite(bbox)):
            raise ValueError("depth observation bbox must contain four finite values")

        if estimate.depth_m is not None:
            self._last_depth_m = float(estimate.depth_m)
            self._last_evidence_at = timestamp
            self._last_bbox = bbox.copy()
            self._last_center = (
                None
                if estimate.palm_center_xy is None
                else np.asarray(estimate.palm_center_xy, dtype=np.float32)
            )
            self._last_palm_width_px = estimate.palm_width_px
            return self._observation(
                estimate,
                depth_m=self._last_depth_m,
                source="measured",
                reason=estimate.reason,
                evidence_at=timestamp,
                age=0.0,
            )

        # Image-boundary failures are a real loss of the palm surface, not a
        # stereo hole.  Do not synthesize evidence while the operator leaves
        # the safe field of view.
        hold_forbidden = estimate.reason in {
            "palm_outside_safe_roi",
            "invalid_depth_frame",
            "invalid_keypoints",
            "degenerate_palm_geometry",
        }
        if (
            not hold_forbidden
            and self.max_hold_seconds > 0.0
            and self._last_depth_m is not None
            and self._last_evidence_at is not None
            and self._last_bbox is not None
        ):
            age = timestamp - self._last_evidence_at
            iou = _bbox_iou(bbox, self._last_bbox)
            center_ok = True
            if (
                estimate.palm_center_xy is not None
                and estimate.palm_width_px is not None
                and self._last_center is not None
                and self._last_palm_width_px is not None
            ):
                center = np.asarray(estimate.palm_center_xy, dtype=np.float32)
                width = max(
                    1.0,
                    min(float(estimate.palm_width_px), self._last_palm_width_px),
                )
                center_ok = float(np.linalg.norm(center - self._last_center)) <= (
                    self.max_center_shift_palm_widths * width
                )
            if 0.0 <= age <= self.max_hold_seconds and iou >= self.min_bbox_iou and center_ok:
                return self._observation(
                    estimate,
                    depth_m=self._last_depth_m,
                    source="held",
                    reason=f"held_after_{estimate.reason}",
                    evidence_at=self._last_evidence_at,
                    age=age,
                )

        return self._observation(
            estimate,
            depth_m=None,
            source="missing",
            reason=estimate.reason,
            evidence_at=None,
            age=None,
        )

    @staticmethod
    def _observation(
        estimate: PalmDepthSpatialEstimate,
        *,
        depth_m: Optional[float],
        source: str,
        reason: str,
        evidence_at: Optional[float],
        age: Optional[float],
    ) -> PalmDepthObservation:
        return PalmDepthObservation(
            depth_m=depth_m,
            raw_depth_m=estimate.depth_m,
            source=source,
            reason=reason,
            evidence_at_monotonic=evidence_at,
            age_seconds=age,
            roi_sample_count=estimate.roi_sample_count,
            inlier_count=estimate.inlier_count,
            valid_pixel_count=estimate.valid_pixel_count,
            radius_px=estimate.radius_px,
        )


class RealSenseSource:
    """Small aligned RGB-D source with no dependency on the larger project stack."""

    def __init__(
        self,
        width: int = 848,
        height: int = 480,
        fps: int = 30,
        serial: Optional[str] = None,
        warmup_frames: int = 20,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self.warmup_frames = warmup_frames
        self._rs = None
        self._pipeline = None
        self._align = None
        self._depth_scale = 0.001
        self._started = False
        self.device_name = "unknown"
        self.device_serial = "unknown"

    def start(self) -> None:
        if self._started:
            return
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(
            rs.stream.depth,
            self.width,
            self.height,
            rs.format.z16,
            self.fps,
        )
        config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps,
        )
        profile = pipeline.start(config)
        device = profile.get_device()
        self.device_name = device.get_info(rs.camera_info.name)
        self.device_serial = device.get_info(rs.camera_info.serial_number)
        depth_sensor = device.first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())
        self._rs = rs
        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color)
        for _ in range(self.warmup_frames):
            pipeline.wait_for_frames(timeout_ms=2000)
        self._started = True

    def read(self, timeout_ms: int = 2000) -> CameraFrame:
        if not self._started:
            self.start()
        assert self._pipeline is not None
        assert self._align is not None
        raw_frames = self._pipeline.wait_for_frames(timeout_ms=timeout_ms)
        # This host timestamp is taken immediately after the SDK releases the
        # frameset.  Keep the hardware timestamps as separate fields: D4xx
        # timestamps are not guaranteed to share the host monotonic clock.
        received_at = time.monotonic()
        frames = self._align.process(raw_frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense returned an incomplete aligned RGB-D frame")
        color = np.asanyarray(color_frame.get_data()).copy()
        depth = (
            np.asanyarray(depth_frame.get_data()).astype(np.float32)
            * self._depth_scale
        )
        ready_at = time.monotonic()
        try:
            timestamp_domain = str(color_frame.get_frame_timestamp_domain())
        except (AttributeError, RuntimeError):
            timestamp_domain = None
        return CameraFrame(
            color_bgr=color,
            depth_m=depth,
            captured_at_monotonic=received_at,
            frame_number=int(color_frame.get_frame_number()),
            depth_frame_number=int(depth_frame.get_frame_number()),
            color_sensor_timestamp_ms=float(color_frame.get_timestamp()),
            depth_sensor_timestamp_ms=float(depth_frame.get_timestamp()),
            sensor_timestamp_domain=timestamp_domain,
            ready_at_monotonic=ready_at,
        )

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        self._started = False

    def __enter__(self) -> "RealSenseSource":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()


def estimate_palm_depth(
    depth_m: Optional[np.ndarray],
    keypoints_2d: np.ndarray,
    radius: Optional[int] = None,
    consensus_tolerance_m: float = 0.04,
) -> PalmDepthSpatialEstimate:
    """Estimate palm depth from inward, spatially independent palm patches.

    The old estimator rejected a whole frame when any one wrist/MCP patch hit
    background.  This version samples slightly inside the palm, requires a
    meaningful number of valid pixels per patch, and accepts only a strong
    spatial consensus.  It still refuses edge-clipped palms and ambiguous
    foreground/background splits.
    """

    if depth_m is None or depth_m.ndim != 2:
        return PalmDepthSpatialEstimate(None, "invalid_depth_frame")
    points = np.asarray(keypoints_2d, dtype=np.float32)
    if points.shape != (21, 2) or not np.all(np.isfinite(points)):
        return PalmDepthSpatialEstimate(None, "invalid_keypoints")
    palm = points[np.asarray(PALM_KEYPOINT_INDICES)]
    palm_center = np.mean(palm, axis=0)
    palm_width_px = float(np.linalg.norm(points[5] - points[17]))
    if not np.isfinite(palm_width_px) or palm_width_px < 4.0:
        return PalmDepthSpatialEstimate(
            None,
            "degenerate_palm_geometry",
            palm_center_xy=(float(palm_center[0]), float(palm_center[1])),
            palm_width_px=palm_width_px,
        )
    if radius is None:
        radius_px = int(np.clip(round(0.06 * palm_width_px), 6, 14))
    else:
        radius_px = int(radius)
        if radius_px < 1:
            raise ValueError("radius must be positive")
    if not 0.01 <= consensus_tolerance_m <= 0.08:
        raise ValueError("consensus_tolerance_m must be in 0.01..0.08")

    # Pull anatomical landmarks 35% toward the palm centroid so their patches
    # stay on skin when a predicted joint lies on a silhouette or is occluded.
    anchors = palm + 0.35 * (palm_center - palm)
    anchors = np.concatenate((anchors, palm_center[None, :]), axis=0)
    height, width = depth_m.shape
    edge_margin = radius_px + 4
    safe_anchor_mask = (
        (anchors[:, 0] >= edge_margin)
        & (anchors[:, 0] < width - edge_margin)
        & (anchors[:, 1] >= edge_margin)
        & (anchors[:, 1] < height - edge_margin)
    )
    if int(np.count_nonzero(safe_anchor_mask)) < 3 or not bool(safe_anchor_mask[-1]):
        return PalmDepthSpatialEstimate(
            None,
            "palm_outside_safe_roi",
            radius_px=radius_px,
            palm_center_xy=(float(palm_center[0]), float(palm_center[1])),
            palm_width_px=palm_width_px,
        )

    samples: list[float] = []
    total_valid_pixels = 0
    for (x_f, y_f), safe in zip(anchors, safe_anchor_mask):
        if not safe:
            continue
        x = int(round(float(x_f)))
        y = int(round(float(y_f)))
        x0, x1 = x - radius_px, x + radius_px + 1
        y0, y1 = y - radius_px, y + radius_px + 1
        if x0 >= x1 or y0 >= y1:
            continue
        patch = depth_m[y0:y1, x0:x1]
        valid = patch[np.isfinite(patch) & (patch > 0.10) & (patch < 2.0)]
        minimum_valid_pixels = max(16, int(math.ceil(0.15 * patch.size)))
        if valid.size >= minimum_valid_pixels:
            samples.append(float(np.median(valid)))
            total_valid_pixels += int(valid.size)
    if len(samples) < 3:
        return PalmDepthSpatialEstimate(
            None,
            "insufficient_valid_palm_rois",
            roi_sample_count=len(samples),
            valid_pixel_count=total_valid_pixels,
            radius_px=radius_px,
            palm_center_xy=(float(palm_center[0]), float(palm_center[1])),
            palm_width_px=palm_width_px,
        )
    samples_array = np.asarray(samples, dtype=np.float32)
    best_inliers = np.zeros(len(samples_array), dtype=bool)
    best_spread = float("inf")
    for candidate in samples_array:
        inliers = np.abs(samples_array - candidate) <= consensus_tolerance_m
        spread = (
            float(np.ptp(samples_array[inliers]))
            if int(np.count_nonzero(inliers)) > 1
            else 0.0
        )
        if int(np.count_nonzero(inliers)) > int(np.count_nonzero(best_inliers)) or (
            int(np.count_nonzero(inliers)) == int(np.count_nonzero(best_inliers))
            and spread < best_spread
        ):
            best_inliers = inliers
            best_spread = spread
    inlier_count = int(np.count_nonzero(best_inliers))
    required_inliers = max(3, int(math.ceil(0.70 * len(samples_array))))
    if inlier_count < required_inliers:
        return PalmDepthSpatialEstimate(
            None,
            "inconsistent_palm_depth_rois",
            roi_sample_count=len(samples),
            inlier_count=inlier_count,
            valid_pixel_count=total_valid_pixels,
            radius_px=radius_px,
            palm_center_xy=(float(palm_center[0]), float(palm_center[1])),
            palm_width_px=palm_width_px,
        )
    depth = float(np.median(samples_array[best_inliers]))
    return PalmDepthSpatialEstimate(
        depth,
        "measured_consensus",
        roi_sample_count=len(samples),
        inlier_count=inlier_count,
        valid_pixel_count=total_valid_pixels,
        radius_px=radius_px,
        palm_center_xy=(float(palm_center[0]), float(palm_center[1])),
        palm_width_px=palm_width_px,
    )


def robust_palm_depth(
    depth_m: Optional[np.ndarray],
    keypoints_2d: np.ndarray,
    radius: int = 8,
) -> Optional[float]:
    """Backwards-compatible scalar wrapper around :func:`estimate_palm_depth`."""

    return estimate_palm_depth(depth_m, keypoints_2d, radius=radius).depth_m
