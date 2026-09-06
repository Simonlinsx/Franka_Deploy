from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from dynamic_pcd.types import RGBDFrame, MaskResult
from dynamic_pcd.utils.geometry import bbox_from_mask, bbox_to_mask, clip_bbox, morph_mask


def select_roi_bbox(frame: RGBDFrame, window_name: str = "Select target ROI") -> Optional[np.ndarray]:
    img = frame.color_bgr.copy()
    instruction = (
        "Draw tight object box (padding is automatic); ENTER/SPACE, c cancels."
    )
    cv2.putText(
        img,
        instruction,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        instruction,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    try:
        x, y, w, h = cv2.selectROI(
            window_name,
            img,
            fromCenter=False,
            showCrosshair=True,
        )
    finally:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            # The window may already be gone after GUI cancellation or a
            # backend exception.  Cleanup must not replace the original error.
            pass
    if w <= 1 or h <= 1:
        return None
    return np.array([x, y, x + w, y + h], dtype=np.int32)


class BoxMaskInitializer:
    """Create a rectangular initial mask from a user-selected box."""

    def initialize(self, frame: RGBDFrame, bbox_xyxy: np.ndarray) -> MaskResult:
        h, w = frame.depth_raw.shape
        bbox = clip_bbox(bbox_xyxy, w, h)
        mask = bbox_to_mask(bbox, (h, w))
        return MaskResult(
            mask=mask,
            bbox_xyxy=bbox,
            score=1.0,
            valid=True,
            message="box mask",
            source="box",
        )


class GrabCutMaskInitializer:
    """OpenCV GrabCut initializer from a box prompt.

    This is a lightweight fallback when SAM2 is not installed. It is not semantic,
    but often provides a cleaner object mask than a raw rectangular box.
    """

    def __init__(self, iters: int = 3, morph_kernel: int = 3):
        self.iters = int(iters)
        self.morph_kernel = int(morph_kernel)

    @staticmethod
    def _failed(
        shape: tuple[int, int],
        bbox_xyxy: np.ndarray,
        message: str,
    ) -> MaskResult:
        """Return an explicit fail-closed result; never promote the prompt rectangle."""
        return MaskResult(
            mask=np.zeros(shape, dtype=np.uint8),
            bbox_xyxy=np.asarray(bbox_xyxy, dtype=np.int32).copy(),
            score=0.0,
            valid=False,
            message=message,
            source="grabcut_failed",
        )

    def initialize(self, frame: RGBDFrame, bbox_xyxy: np.ndarray) -> MaskResult:
        h, w = frame.depth_raw.shape
        bbox = clip_bbox(bbox_xyxy, w, h)
        x1, y1, x2, y2 = bbox.astype(int)
        rect = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
        if rect[2] < 5 or rect[3] < 5:
            return self._failed(
                (h, w),
                bbox,
                "grabcut prompt rectangle is smaller than 5x5",
            )

        gc_mask = np.zeros((h, w), dtype=np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(frame.color_bgr, gc_mask, rect, bgd, fgd, self.iters, cv2.GC_INIT_WITH_RECT)
            mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
            mask = morph_mask(mask, self.morph_kernel)
            bbox2 = bbox_from_mask(mask, min_area=20)
            if bbox2 is None or mask.sum() < 20:
                return self._failed(
                    (h, w),
                    bbox,
                    "grabcut produced fewer than 20 foreground pixels",
                )
            return MaskResult(
                mask=mask,
                bbox_xyxy=bbox2,
                score=1.0,
                valid=True,
                message="grabcut mask",
                source="grabcut",
            )
        except Exception as e:
            return self._failed(
                (h, w),
                bbox,
                f"grabcut failed: {type(e).__name__}: {e}",
            )
