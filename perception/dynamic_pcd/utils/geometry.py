from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def bbox_from_mask(mask: np.ndarray, min_area: int = 1) -> Optional[np.ndarray]:
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask_array.shape}")

    # ``np.where`` materializes two full coordinate arrays before reducing
    # them.  At 848x480 that costs about a millisecond even for a small object
    # mask.  OpenCV performs the same non-zero count/bounds reduction in C and
    # does not allocate per-pixel coordinates.  Keep a zero-copy fast path for
    # the uint8/bool masks produced by the bundled trackers; unusual numeric
    # dtypes still retain NumPy's non-zero/NaN truth semantics via bool first.
    if mask_array.dtype == np.uint8:
        mask_u8 = np.ascontiguousarray(mask_array)
    elif mask_array.dtype == np.bool_:
        mask_u8 = np.ascontiguousarray(mask_array).view(np.uint8)
    else:
        mask_u8 = np.ascontiguousarray(mask_array.astype(bool), dtype=np.uint8)

    foreground = int(cv2.countNonZero(mask_u8))
    if foreground < max(1, int(min_area)):
        return None
    x, y, width, height = cv2.boundingRect(mask_u8)
    return np.array([x, y, x + width, y + height], dtype=np.int32)


def bbox_area(bbox: np.ndarray) -> int:
    x1, y1, x2, y2 = bbox.astype(int)
    return max(0, x2 - x1) * max(0, y2 - y1)


def clip_bbox(bbox: np.ndarray, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(int)
    x1 = int(np.clip(x1, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    x2 = int(np.clip(x2, x1 + 1, width))
    y2 = int(np.clip(y2, y1 + 1, height))
    return np.array([x1, y1, x2, y2], dtype=np.int32)


def enlarge_bbox(bbox: np.ndarray, scale: float, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(float)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max(1.0, (x2 - x1) * scale)
    bh = max(1.0, (y2 - y1) * scale)
    out = np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dtype=np.float32)
    return clip_bbox(out, width, height)


def bbox_to_mask(bbox: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    x1, y1, x2, y2 = clip_bbox(bbox, w, h)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def erode_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 0:
        return mask.astype(np.uint8)
    k = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.erode(mask.astype(np.uint8), k, iterations=1)


def morph_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask.astype(np.uint8)
    k = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    return m


def largest_component(mask: np.ndarray, min_area: int = 1, prefer_center: Optional[Tuple[float, float]] = None) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return mask_u8

    best_idx = -1
    best_score = -1e18
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        if prefer_center is None:
            score = float(area)
        else:
            cx, cy = centroids[i]
            dx = cx - prefer_center[0]
            dy = cy - prefer_center[1]
            score = float(area) - 0.1 * float(dx * dx + dy * dy)
        if score > best_score:
            best_score = score
            best_idx = i

    out = np.zeros_like(mask_u8)
    if best_idx >= 0:
        out[labels == best_idx] = 1
    return out


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool)
    bb = b.astype(bool)
    inter = np.logical_and(aa, bb).sum()
    union = np.logical_or(aa, bb).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def draw_mask_overlay(
    image_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    bbox: Optional[np.ndarray] = None,
    color=(0, 255, 0),
    alpha: float = 0.35,
    text: Optional[str] = None,
) -> np.ndarray:
    out = image_bgr.copy()
    if mask is not None:
        m = mask.astype(bool)
        overlay = out.copy()
        overlay[m] = color
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    if bbox is not None:
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    if text:
        cv2.putText(out, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def depth_stats(depth_m: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float]:
    vals = depth_m[mask.astype(bool)]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.percentile(vals, 5)), float(np.median(vals)), float(np.percentile(vals, 95))
