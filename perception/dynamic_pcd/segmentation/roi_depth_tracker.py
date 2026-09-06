from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from dynamic_pcd.types import MaskResult, RGBDFrame
from dynamic_pcd.utils.geometry import (
    bbox_area,
    bbox_from_mask,
    bbox_to_mask,
    clip_bbox,
    depth_stats,
    enlarge_bbox,
    largest_component,
    morph_mask,
)


@dataclass
class ROITrackerState:
    bbox_xyxy: np.ndarray
    depth_median: float
    center_uv: Tuple[float, float]
    area: float
    initial_area: float
    initial_bbox_area: float
    last_mask: Optional[np.ndarray] = None
    lost_count: int = 0
    valid: bool = True


class ROIDepthTracker:
    """Fast ROI + depth tracker for a single target object.

    This is meant to run every frame. It does not try to be a semantic detector.
    It assumes that an initial target ROI/mask is known and then tracks pixels near
    the previous image ROI and previous median depth.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.state: Optional[ROITrackerState] = None
        self.fixed_bbox_xyxy: Optional[np.ndarray] = None

    @property
    def initialized(self) -> bool:
        return self.state is not None

    @property
    def roi_locked(self) -> bool:
        return bool(self.cfg.get("lock_roi", False))

    def set_roi_locked(self, locked: bool, anchor_current: bool = True) -> bool:
        """Enable/disable a fixed image ROI and return the resulting state.

        When locking an active tracker, anchor the fixed ROI at its *current*
        bbox.  Otherwise an interactive lock command could unexpectedly jump
        back to the bbox used at process startup.
        """

        locked = bool(locked)
        self.cfg["lock_roi"] = locked
        if locked and anchor_current and self.state is not None:
            self.fixed_bbox_xyxy = self.state.bbox_xyxy.astype(np.int32).copy()
        return locked

    def toggle_roi_lock(self) -> bool:
        return self.set_roi_locked(not self.roi_locked, anchor_current=True)

    def _configured_depth_stats(
        self, frame: RGBDFrame, mask: np.ndarray
    ) -> Tuple[float, float, float]:
        """Depth statistics inside the commissioned point-cloud interval."""

        depth_m = frame.depth_m
        z_min = float(self.cfg.get("z_min", 0.0))
        z_max = float(self.cfg.get("z_max", 10.0))
        valid = (
            np.isfinite(depth_m)
            & (depth_m > z_min)
            & (depth_m < z_max)
        )
        bounded_depth = np.where(valid, depth_m, 0.0)
        return depth_stats(bounded_depth, mask)

    def initialize(self, frame: RGBDFrame, mask: Optional[np.ndarray] = None, bbox_xyxy: Optional[np.ndarray] = None) -> MaskResult:
        h, w = frame.depth_raw.shape
        if mask is None:
            if bbox_xyxy is None:
                raise ValueError("initialize requires mask or bbox")
            bbox = clip_bbox(bbox_xyxy, w, h)
            mask = bbox_to_mask(bbox, (h, w))
        else:
            mask = (mask > 0).astype(np.uint8)
            bbox = bbox_from_mask(mask, min_area=1)
            if bbox is None:
                if bbox_xyxy is None:
                    raise ValueError("empty mask and no bbox")
                bbox = clip_bbox(bbox_xyxy, w, h)
                mask = bbox_to_mask(bbox, (h, w))
            else:
                bbox = clip_bbox(bbox, w, h)

        z5, zmed, z95 = self._configured_depth_stats(frame, mask)
        if zmed <= 0:
            # Fall back to bbox valid depth.
            box_mask = bbox_to_mask(bbox, (h, w))
            z5, zmed, z95 = self._configured_depth_stats(frame, box_mask)
        if zmed <= 0:
            raise RuntimeError("cannot initialize tracker: no valid depth in target ROI")

        x1, y1, x2, y2 = bbox.astype(float)
        center_uv = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        area = float(mask.sum())
        self.state = ROITrackerState(
            bbox_xyxy=bbox,
            depth_median=float(zmed),
            center_uv=center_uv,
            area=area,
            initial_area=area,
            initial_bbox_area=float(bbox_area(bbox)),
            last_mask=mask.copy(),
            lost_count=0,
            valid=True,
        )
        self.fixed_bbox_xyxy = bbox.astype(np.int32).copy()
        return MaskResult(mask=mask, bbox_xyxy=bbox, score=1.0, valid=True, message=f"init z={zmed:.3f}m area={int(mask.sum())}")

    def update(self, frame: RGBDFrame) -> MaskResult:
        if self.state is None:
            raise RuntimeError("tracker not initialized")

        h, w = frame.depth_raw.shape
        roi_scale = float(self.cfg.get("roi_scale", 1.8))
        depth_tol = float(self.cfg.get("depth_tolerance", 0.08))
        min_area = int(self.cfg.get("min_area", 80))
        morph_kernel = int(self.cfg.get("morph_kernel", 5))
        lost_after = int(self.cfg.get("lost_after", 8))

        lock_roi = bool(self.cfg.get("lock_roi", False))
        if lock_roi and self.fixed_bbox_xyxy is not None:
            roi = clip_bbox(self.fixed_bbox_xyxy, w, h)
        else:
            roi = enlarge_bbox(self.state.bbox_xyxy, roi_scale, w, h)
        x1, y1, x2, y2 = roi.astype(int)

        depth_m = frame.depth_m
        roi_depth = depth_m[y1:y2, x1:x2]
        z0 = self.state.depth_median
        valid_roi = np.isfinite(roi_depth) & (roi_depth > 0) & (np.abs(roi_depth - z0) <= depth_tol)

        # Prefer area close to previous 2D center. Connected-component keeps only one blob.
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = valid_roi.astype(np.uint8)
        full_mask = morph_mask(full_mask, morph_kernel)
        if lock_roi:
            full_mask &= bbox_to_mask(roi, (h, w))
        full_mask = largest_component(full_mask, min_area=min_area, prefer_center=self.state.center_uv)

        bbox = bbox_from_mask(full_mask, min_area=min_area)
        if bbox is None:
            self.state.lost_count += 1
            if self.state.lost_count > lost_after:
                self.state.valid = False
            msg = f"lost {self.state.lost_count}/{lost_after}"
            # Return previous mask to keep visualization stable, but mark invalid if too long.
            fallback = self.state.last_mask if self.state.last_mask is not None else np.zeros((h, w), dtype=np.uint8)
            return MaskResult(mask=fallback, bbox_xyxy=self.state.bbox_xyxy, score=0.0, valid=False, message=msg)

        z5, zmed, z95 = depth_stats(depth_m, full_mask)
        if zmed <= 0:
            self.state.lost_count += 1
            return MaskResult(mask=full_mask, bbox_xyxy=bbox, score=0.0, valid=False, message="no valid depth in mask")

        cx = 0.5 * (bbox[0] + bbox[2])
        cy = 0.5 * (bbox[1] + bbox[3])
        area = float(full_mask.sum())
        reject_reason = self._size_reject_reason(area, bbox, check_last=True)
        if reject_reason:
            self.state.lost_count += 1
            fallback = self.state.last_mask if self.state.last_mask is not None else np.zeros((h, w), dtype=np.uint8)
            msg = f"reject {reject_reason}"
            return MaskResult(mask=fallback, bbox_xyxy=self.state.bbox_xyxy, score=0.0, valid=False, message=msg)

        score = min(1.0, area / max(1.0, float(min_area) * 10.0))

        # Smooth depth median and bbox center slightly to reduce jitter.
        alpha = 0.65
        self.state.depth_median = alpha * float(zmed) + (1 - alpha) * self.state.depth_median
        self.state.center_uv = (alpha * cx + (1 - alpha) * self.state.center_uv[0],
                                alpha * cy + (1 - alpha) * self.state.center_uv[1])
        self.state.bbox_xyxy = bbox.astype(np.int32)
        self.state.area = area
        self.state.last_mask = full_mask.copy()
        self.state.lost_count = 0
        self.state.valid = True

        return MaskResult(mask=full_mask, bbox_xyxy=bbox, score=score, valid=True,
                          message=f"track z={self.state.depth_median:.3f}m area={int(area)}")

    def snapshot_state(self) -> Optional[ROITrackerState]:
        return copy.deepcopy(self.state)

    def restore_state(self, state: Optional[ROITrackerState]) -> None:
        self.state = copy.deepcopy(state)

    def reinitialize_with_mask(self, frame: RGBDFrame, mask: np.ndarray) -> MaskResult:
        prev_state = self.state
        prev_initial_area = prev_state.initial_area if prev_state is not None else None
        prev_initial_bbox_area = prev_state.initial_bbox_area if prev_state is not None else None
        res = self.initialize(frame, mask=mask)
        if self.state is not None and prev_initial_area is not None and prev_initial_bbox_area is not None:
            self.state.initial_area = prev_initial_area
            self.state.initial_bbox_area = prev_initial_bbox_area
            reject_reason = self._size_reject_reason(self.state.area, self.state.bbox_xyxy, check_last=False)
            if reject_reason and prev_state is not None:
                self.state = prev_state
                fallback = prev_state.last_mask if prev_state.last_mask is not None else np.zeros_like(mask, dtype=np.uint8)
                msg = f"reject reinit {reject_reason}"
                return MaskResult(mask=fallback, bbox_xyxy=prev_state.bbox_xyxy, score=0.0, valid=False, message=msg)
        return res

    def _size_reject_reason(self, area: float, bbox: np.ndarray, check_last: bool) -> str:
        if self.state is None:
            return ""

        max_area_growth = float(self.cfg.get("max_area_growth", 0.0))
        max_area_vs_initial = float(self.cfg.get("max_area_vs_initial", 0.0))
        max_bbox_growth = float(self.cfg.get("max_bbox_growth", 0.0))
        min_area_vs_initial = float(self.cfg.get("min_area_vs_initial", 0.0))
        min_bbox_area_vs_initial = float(
            self.cfg.get("min_bbox_area_vs_initial", 0.0)
        )

        if check_last and max_area_growth > 0.0 and self.state.area > 0.0 and area > self.state.area * max_area_growth:
            return f"area jump {int(self.state.area)}->{int(area)} max_x={max_area_growth:.1f}"
        if max_area_vs_initial > 0.0 and self.state.initial_area > 0.0 and area > self.state.initial_area * max_area_vs_initial:
            return f"area vs init {int(self.state.initial_area)}->{int(area)} max_x={max_area_vs_initial:.1f}"
        if (
            min_area_vs_initial > 0.0
            and self.state.initial_area > 0.0
            and area < self.state.initial_area * min_area_vs_initial
        ):
            return (
                f"area vs init {int(self.state.initial_area)}->{int(area)} "
                f"min_x={min_area_vs_initial:.2f}"
            )
        if max_bbox_growth > 0.0 and self.state.initial_bbox_area > 0.0:
            current_bbox_area = float(bbox_area(bbox))
            if current_bbox_area > self.state.initial_bbox_area * max_bbox_growth:
                return f"bbox vs init {int(self.state.initial_bbox_area)}->{int(current_bbox_area)} max_x={max_bbox_growth:.1f}"
        if min_bbox_area_vs_initial > 0.0 and self.state.initial_bbox_area > 0.0:
            current_bbox_area = float(bbox_area(bbox))
            if current_bbox_area < self.state.initial_bbox_area * min_bbox_area_vs_initial:
                return (
                    f"bbox vs init {int(self.state.initial_bbox_area)}->"
                    f"{int(current_bbox_area)} min_x={min_bbox_area_vs_initial:.2f}"
                )
        return ""
