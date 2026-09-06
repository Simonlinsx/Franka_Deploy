from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from dynamic_pcd.types import MaskResult, RGBDFrame
from dynamic_pcd.utils.geometry import bbox_area, bbox_from_mask, clip_bbox, largest_component


def _normalize_predict_output(masks: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    masks_arr = np.asarray(masks)
    scores_arr = np.asarray(scores, dtype=np.float32).reshape(-1)

    if masks_arr.ndim == 2:
        masks_arr = masks_arr[None, :, :]
    elif masks_arr.ndim == 4 and masks_arr.shape[0] == 1:
        masks_arr = masks_arr[0]
    elif masks_arr.ndim == 4:
        masks_arr = masks_arr.reshape((-1,) + masks_arr.shape[-2:])
    if masks_arr.ndim != 3:
        raise ValueError(f"unexpected SAM2 mask shape {masks_arr.shape}")

    if scores_arr.size == 0:
        scores_arr = np.zeros((masks_arr.shape[0],), dtype=np.float32)
    elif scores_arr.size != masks_arr.shape[0]:
        scores_arr = np.resize(scores_arr, masks_arr.shape[0]).astype(np.float32)

    return masks_arr, scores_arr


def _select_best_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    prompt_box: np.ndarray,
    image_shape_hw: Tuple[int, int],
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float, bool, str]:
    h, w = image_shape_hw
    masks_arr, scores_arr = _normalize_predict_output(masks, scores)
    box = clip_bbox(prompt_box, w, h)
    prompt_area = max(1, bbox_area(box))
    score_threshold = float(cfg.get("score_threshold", 0.0))
    min_area = int(cfg.get("min_area", 1))
    max_mask_area_ratio = float(cfg.get("max_mask_area_ratio", 0.0))
    max_bbox_area_ratio = float(cfg.get("max_bbox_area_ratio", 0.0))
    keep_largest = bool(cfg.get("keep_largest_component", True))
    center_uv = (0.5 * float(box[0] + box[2]), 0.5 * float(box[1] + box[3]))

    candidates = []
    for i, raw_mask in enumerate(masks_arr):
        mask = (raw_mask > 0).astype(np.uint8)
        if keep_largest:
            mask = largest_component(mask, min_area=min_area, prefer_center=center_uv)

        bbox = bbox_from_mask(mask, min_area=min_area)
        area = int(mask.sum())
        score = float(scores_arr[i])
        reasons = []
        mask_ratio = float(area) / float(prompt_area)
        bbox_ratio = 0.0
        if bbox is None:
            bbox = box.copy()
            reasons.append("empty")
        else:
            bbox = clip_bbox(bbox, w, h)
            bbox_ratio = float(bbox_area(bbox)) / float(prompt_area)

        if score < score_threshold:
            reasons.append("low_score")
        if area < min_area:
            reasons.append("small")
        if max_mask_area_ratio > 0.0 and mask_ratio > max_mask_area_ratio:
            reasons.append("mask_too_large")
        if max_bbox_area_ratio > 0.0 and bbox_ratio > max_bbox_area_ratio:
            reasons.append("bbox_too_large")

        valid = len(reasons) == 0
        candidates.append(
            {
                "index": i,
                "mask": mask,
                "bbox": bbox,
                "score": score,
                "area": area,
                "mask_ratio": mask_ratio,
                "bbox_ratio": bbox_ratio,
                "valid": valid,
                "reasons": reasons,
            }
        )

    valid_candidates = [c for c in candidates if c["valid"]]
    pool = valid_candidates if valid_candidates else candidates
    best = max(pool, key=lambda c: c["score"])
    parts = []
    for c in candidates:
        status = "ok" if c["valid"] else "+".join(c["reasons"])
        parts.append(
            f"{c['index']}:{status},s={c['score']:.3f},a={c['area']},"
            f"mr={c['mask_ratio']:.1f},br={c['bbox_ratio']:.1f}"
        )
    message = (
        f"SAM2 score={best['score']:.3f} area={best['area']} "
        f"bbox={best['bbox'].astype(int).tolist()} prompt_area={prompt_area} "
        f"candidates=[{'; '.join(parts)}]"
    )
    return best["mask"], best["bbox"], best["score"], bool(best["valid"]), message


class SAM2ImageSegmenter:
    """Optional SAM2 image predictor wrapper.

    This wrapper is designed for initialization/re-initialization on a single RGB
    frame with a box prompt. For real-time control, use it at low frequency and
    let ROIDepthTracker handle high-frequency updates.

    Expected official SAM2-style imports:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", False))
        self.predictor = None
        self.device = cfg.get("device", "cuda")
        if self.enabled:
            self._load()

    def _load(self):
        checkpoint = self.cfg.get("checkpoint")
        model_cfg = self.cfg.get("model_cfg")
        if not checkpoint or not model_cfg:
            raise ValueError("SAM2 enabled but sam2.checkpoint or sam2.model_cfg is missing")

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as e:
            raise ImportError(
                "Could not import SAM2. Install official facebookresearch/sam2 first. "
                "See requirements_sam2.txt and README.md."
            ) from e

        model = build_sam2(model_cfg, checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(model)
        print(f"[SAM2] loaded checkpoint={checkpoint}, cfg={model_cfg}, device={self.device}")

    def available(self) -> bool:
        return self.predictor is not None

    def segment_with_box(self, frame: RGBDFrame, bbox_xyxy: np.ndarray) -> MaskResult:
        if self.predictor is None:
            raise RuntimeError("SAM2 predictor not loaded")

        # img_rgb = frame.color_bgr[:, :, ::-1]
        img_rgb = np.ascontiguousarray(frame.color_bgr[:, :, ::-1])
        h, w = img_rgb.shape[:2]
        box = clip_bbox(bbox_xyxy, w, h).astype(np.float32)

        multimask_output = bool(self.cfg.get("multimask_output", False))

        self.predictor.set_image(img_rgb)
        masks, scores, _logits = self.predictor.predict(
            box=box[None, :],
            multimask_output=multimask_output,
        )
        if masks is None or len(masks) == 0:
            return MaskResult(mask=np.zeros((h, w), dtype=np.uint8), bbox_xyxy=box.astype(np.int32), score=0.0, valid=False, message="SAM2 no mask")

        mask, bbox, score, valid, message = _select_best_mask(masks, scores, box, (h, w), self.cfg)
        return MaskResult(mask=mask, bbox_xyxy=bbox, score=score, valid=valid, message=message)
