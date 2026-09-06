"""Pure binary-mask geometry used by the guarded point-cloud provider.

These helpers have no provider, camera, tracker, or publication state. Keeping
them here makes their deterministic geometry contract independently testable.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from dynamic_pcd.utils.geometry import bbox_from_mask


def _binary_mask_centroid(
    mask: np.ndarray, bbox_xyxy: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """Return the binary centroid using only the foreground-local ROI."""

    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    bbox = (
        bbox_from_mask(mask_u8, min_area=1)
        if bbox_xyxy is None
        else np.asarray(bbox_xyxy, dtype=np.int32).reshape(4)
    )
    if bbox is None:
        return None
    x1, y1, x2, y2 = (int(value) for value in bbox.tolist())
    if x2 <= x1 or y2 <= y1:
        return None
    moments = cv2.moments(mask_u8[y1:y2, x1:x2], binaryImage=True)
    area = float(moments["m00"])
    if area <= 0.0:
        return None
    return np.asarray(
        [
            float(x1) + moments["m10"] / area,
            float(y1) + moments["m01"] / area,
        ],
        dtype=np.float64,
    )


def _integer_translation_overlap_counts(
    reference_mask: np.ndarray,
    candidate_mask: np.ndarray,
    shift_xy: np.ndarray,
    reference_bbox_xyxy: Optional[np.ndarray] = None,
) -> Tuple[int, int]:
    """Count an integer-translated mask and its candidate overlap locally.

    This is exactly the count-only equivalent of a full-frame
    ``cv2.warpAffine(..., INTER_NEAREST, BORDER_CONSTANT=0)`` for an
    integer translation.  Publication needs only the two counts, so do
    not allocate and scan another camera-sized image.
    """

    reference = np.asarray(reference_mask)
    candidate = np.asarray(candidate_mask)
    if reference.ndim != 2 or candidate.shape != reference.shape:
        return 0, 0
    shift = np.asarray(shift_xy, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(shift)):
        return 0, 0
    rounded = np.rint(shift)
    if not np.array_equal(shift, rounded):
        raise ValueError("translation overlap requires an integer shift")
    dx, dy = (int(value) for value in rounded.tolist())
    bbox = (
        bbox_from_mask((reference > 0).astype(np.uint8), min_area=1)
        if reference_bbox_xyxy is None
        else np.asarray(reference_bbox_xyxy, dtype=np.int32).reshape(4)
    )
    if bbox is None:
        return 0, 0
    height, width = reference.shape
    x1, y1, x2, y2 = (int(value) for value in bbox.tolist())
    source_x1 = max(0, x1, -dx)
    source_y1 = max(0, y1, -dy)
    source_x2 = min(width, x2, width - dx)
    source_y2 = min(height, y2, height - dy)
    if source_x2 <= source_x1 or source_y2 <= source_y1:
        return 0, 0
    destination_x1 = source_x1 + dx
    destination_y1 = source_y1 + dy
    reference_roi = reference[source_y1:source_y2, source_x1:source_x2] > 0
    candidate_roi = candidate[
        destination_y1 : destination_y1 + (source_y2 - source_y1),
        destination_x1 : destination_x1 + (source_x2 - source_x1),
    ] > 0
    translated_pixels = int(np.count_nonzero(reference_roi))
    retained_pixels = int(np.count_nonzero(reference_roi & candidate_roi))
    return translated_pixels, retained_pixels


def _mask_overlap_registration_shift(
    clean_mask: np.ndarray,
    candidate_mask: np.ndarray,
    predicted_shift: np.ndarray,
    correction_limit: float,
    clean_bbox_xyxy: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Find the bounded translation with maximum clean/candidate overlap.

    Centroid-only registration is biased toward a newly attached finger.
    A tight clean-mask template instead locks onto the still-visible
    object core.  Ties prefer the motion prediction, which preserves a
    deterministic result under severe partial occlusion.
    """

    clean = (np.asarray(clean_mask) > 0).astype(np.uint8)
    candidate = (np.asarray(candidate_mask) > 0).astype(np.uint8)
    clean_bbox = (
        bbox_from_mask(clean, min_area=1)
        if clean_bbox_xyxy is None
        else np.asarray(clean_bbox_xyxy, dtype=np.int32).reshape(4)
    )
    if clean_bbox is None or correction_limit <= 0.0:
        return None
    x1, y1, x2, y2 = clean_bbox.astype(int)
    template = clean[y1:y2, x1:x2].astype(np.float32)
    if template.size == 0:
        return None

    radius = max(1, int(np.ceil(correction_limit)))
    predicted_left = float(x1) + float(predicted_shift[0])
    predicted_top = float(y1) + float(predicted_shift[1])
    search_origin_x = int(np.floor(predicted_left - radius))
    search_origin_y = int(np.floor(predicted_top - radius))
    search_width = template.shape[1] + 2 * radius + 2
    search_height = template.shape[0] + 2 * radius + 2
    # Build only the local search window. Copying/padding the full 848x480
    # frame added several milliseconds to every otherwise small-object
    # guard evaluation.
    search = np.zeros((search_height, search_width), dtype=np.float32)
    image_x1 = max(0, search_origin_x)
    image_y1 = max(0, search_origin_y)
    image_x2 = min(candidate.shape[1], search_origin_x + search_width)
    image_y2 = min(candidate.shape[0], search_origin_y + search_height)
    if image_x2 > image_x1 and image_y2 > image_y1:
        local_x1 = image_x1 - search_origin_x
        local_y1 = image_y1 - search_origin_y
        search[
            local_y1 : local_y1 + (image_y2 - image_y1),
            local_x1 : local_x1 + (image_x2 - image_x1),
        ] = candidate[image_y1:image_y2, image_x1:image_x2]
    response = cv2.matchTemplate(search, template, cv2.TM_CCORR)
    if response.size == 0 or not np.all(np.isfinite(response)):
        return None
    maximum = float(response.max())
    if maximum <= 0.0:
        return None
    ys, xs = np.nonzero(response >= maximum - 1.0e-6)
    if xs.size == 0:
        return None
    global_left = search_origin_x + xs.astype(np.float64)
    global_top = search_origin_y + ys.astype(np.float64)
    shifts = np.stack(
        [
            global_left - float(x1),
            global_top - float(y1),
        ],
        axis=1,
    )
    residuals = shifts - np.asarray(predicted_shift, dtype=np.float64).reshape(1, 2)
    best = int(np.argmin(np.sum(residuals * residuals, axis=1)))
    return shifts[best]


def _scale_mask_about_bbox_center(
    mask: np.ndarray, bbox_xyxy: np.ndarray, scale: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale one clean silhouette without moving its bbox centre."""

    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    bbox = np.asarray(bbox_xyxy, dtype=np.int32).reshape(4)
    if not np.isfinite(scale) or scale <= 0.0:
        return mask_u8.copy(), bbox.copy()
    x1, y1, x2, y2 = bbox.astype(int)
    crop = mask_u8[y1:y2, x1:x2]
    if crop.size == 0:
        return mask_u8.copy(), bbox.copy()
    new_width = max(1, int(np.rint(crop.shape[1] * scale)))
    new_height = max(1, int(np.rint(crop.shape[0] * scale)))
    resized = cv2.resize(
        crop,
        (new_width, new_height),
        interpolation=cv2.INTER_NEAREST,
    )
    center_x = 0.5 * (float(x1) + float(x2))
    center_y = 0.5 * (float(y1) + float(y2))
    dst_x1 = int(np.rint(center_x - 0.5 * new_width))
    dst_y1 = int(np.rint(center_y - 0.5 * new_height))
    dst_x2 = dst_x1 + new_width
    dst_y2 = dst_y1 + new_height
    height, width = mask_u8.shape
    image_x1 = max(0, dst_x1)
    image_y1 = max(0, dst_y1)
    image_x2 = min(width, dst_x2)
    image_y2 = min(height, dst_y2)
    scaled = np.zeros_like(mask_u8)
    if image_x2 > image_x1 and image_y2 > image_y1:
        src_x1 = image_x1 - dst_x1
        src_y1 = image_y1 - dst_y1
        scaled[image_y1:image_y2, image_x1:image_x2] = resized[
            src_y1 : src_y1 + (image_y2 - image_y1),
            src_x1 : src_x1 + (image_x2 - image_x1),
        ]
    scaled_bbox = bbox_from_mask(scaled, min_area=1)
    return (
        scaled,
        (
            bbox.copy()
            if scaled_bbox is None
            else np.asarray(scaled_bbox, dtype=np.int32)
        ),
    )


__all__ = [
    "_binary_mask_centroid",
    "_integer_translation_overlap_counts",
    "_mask_overlap_registration_shift",
    "_scale_mask_about_bbox_center",
]
