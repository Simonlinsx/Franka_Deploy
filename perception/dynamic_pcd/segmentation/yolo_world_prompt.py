from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from dynamic_pcd.segmentation.prompt_protocol import (
    PromptCandidate,
    PromptSegmentationResult,
    validate_bgr_image,
    validate_prompt,
)
from dynamic_pcd.utils.geometry import bbox_area


class YOLOWorldPromptBackend:
    """Fast text-conditioned bbox detector behind the prompt RPC protocol.

    The returned rectangle mask is protocol padding only.  Callers must use the
    returned ``bbox_xyxy`` to initialize the production SAM2 box-prompt path;
    they must never publish this rectangle as an object mask or point cloud.
    """

    detector_backend = "yolo_world"
    mask_backend = "bbox_only"

    def __init__(
        self,
        *,
        weights: str,
        device: str = "cuda",
        confidence: float = 0.03,
        image_size: int = 640,
        preload_prompt: Optional[str] = None,
        model: Optional[Any] = None,
    ):
        self.weights = str(Path(weights).expanduser())
        self.device = str(device)
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.preload_prompt = (
            None
            if preload_prompt is None
            else validate_prompt(str(preload_prompt))
        )
        if not np.isfinite(self.confidence) or not 0.001 <= self.confidence <= 1.0:
            raise ValueError("YOLO-World confidence must be in [0.001, 1]")
        if self.image_size < 320 or self.image_size > 1280:
            raise ValueError("YOLO-World image_size must be in 320..1280")
        self.model = model
        self._model_injected = model is not None
        self._active_prompt: Optional[str] = None
        self._warmed_up = False

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if not self.loaded:
            weights = Path(self.weights)
            if not weights.is_file():
                raise FileNotFoundError(
                    f"YOLO-World weights do not exist: {weights}"
                )
            from ultralytics import YOLOWorld

            self.model = YOLOWorld(str(weights.resolve()))
        if (
            not self._model_injected
            and not self._warmed_up
        ):
            # Construct Ultralytics' predictor before installing the custom
            # vocabulary.  On the first predict(), Ultralytics may restore the
            # checkpoint's default class list; setting the text classes before
            # that call therefore leaves `_active_prompt` claiming a vocabulary
            # that the actual predictor no longer owns (balls happened to work
            # through the COCO `sports ball` class, while `red cube` did not).
            self.model.predict(
                np.zeros((480, 848, 3), dtype=np.uint8),
                imgsz=self.image_size,
                conf=self.confidence,
                iou=0.70,
                device=self.device,
                verbose=False,
            )
            self._warmed_up = True
        if self.preload_prompt is not None:
            changed = self._set_prompt(self.preload_prompt)
            if not self._model_injected and changed:
                # Warm the final one-class prediction head as well.  The
                # service is not READY until the first real camera frame pays
                # only hot inference latency.
                self.model.predict(
                    np.zeros((480, 848, 3), dtype=np.uint8),
                    imgsz=self.image_size,
                    conf=self.confidence,
                    iou=0.70,
                    device=self.device,
                    verbose=False,
                )

    def _set_prompt(self, prompt: str) -> bool:
        prompt = validate_prompt(prompt)
        if prompt == self._active_prompt:
            return False
        self.model.set_classes([prompt])
        self._active_prompt = prompt
        return True

    @staticmethod
    def _prompt_color(prompt: str) -> Optional[str]:
        tokens = {
            token.strip(".,;:()[]{}")
            for token in str(prompt).lower().split()
        }
        for name in (
            "red",
            "green",
            "blue",
            "yellow",
            "orange",
            "pink",
            "purple",
            "cyan",
            "black",
            "white",
        ):
            if name in tokens:
                return name
        return None

    @staticmethod
    def _color_fraction(image_bgr: np.ndarray, box: np.ndarray, color: str) -> float:
        height, width = image_bgr.shape[:2]
        x1, y1, x2, y2 = np.rint(box).astype(np.int32)
        x1 = int(np.clip(x1, 0, width))
        x2 = int(np.clip(x2, 0, width))
        y1 = int(np.clip(y1, 0, height))
        y2 = int(np.clip(y2, 0, height))
        if x2 <= x1 or y2 <= y1:
            return 0.0
        pixels = image_bgr[y1:y2, x1:x2].astype(np.float32)
        blue, green, red = (pixels[..., index] for index in range(3))
        maximum = np.maximum(np.maximum(red, green), blue)
        minimum = np.minimum(np.minimum(red, green), blue)
        chroma = maximum - minimum
        saturated = chroma >= 35.0
        if color == "red":
            keep = saturated & (red >= 65.0) & (red >= 1.18 * green) & (red >= 1.12 * blue)
        elif color == "green":
            keep = saturated & (green >= 55.0) & (green >= 1.12 * red) & (green >= 1.12 * blue)
        elif color == "blue":
            keep = saturated & (blue >= 55.0) & (blue >= 1.12 * red) & (blue >= 1.12 * green)
        elif color == "yellow":
            keep = saturated & (red >= 80.0) & (green >= 70.0) & (blue <= 0.72 * np.minimum(red, green))
        elif color == "orange":
            keep = saturated & (red >= 85.0) & (green >= 0.25 * red) & (green <= 0.85 * red) & (blue <= 0.65 * red)
        elif color == "pink":
            keep = (red >= 90.0) & (red >= 1.12 * green) & (blue >= 0.35 * red)
        elif color == "purple":
            keep = saturated & (red >= 55.0) & (blue >= 55.0) & (green <= 0.82 * np.maximum(red, blue))
        elif color == "cyan":
            keep = saturated & (green >= 55.0) & (blue >= 55.0) & (red <= 0.75 * np.minimum(green, blue))
        elif color == "black":
            keep = maximum <= 55.0
        elif color == "white":
            keep = (minimum >= 150.0) & (chroma <= 35.0)
        else:
            return 0.0
        return float(np.mean(keep))

    @staticmethod
    def _iou(left: np.ndarray, right: np.ndarray) -> float:
        x1 = max(float(left[0]), float(right[0]))
        y1 = max(float(left[1]), float(right[1]))
        x2 = min(float(left[2]), float(right[2]))
        y2 = min(float(left[3]), float(right[3]))
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = float(bbox_area(left)) + float(bbox_area(right)) - intersection
        return 0.0 if union <= 0.0 else intersection / union

    @staticmethod
    def _center_distance(left: np.ndarray, right: np.ndarray) -> float:
        left_center = 0.5 * (left[:2] + left[2:])
        right_center = 0.5 * (right[:2] + right[2:])
        return float(np.linalg.norm(left_center - right_center))

    def segment(
        self,
        *,
        image_bgr: np.ndarray,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
        mask_threshold: float,
        reference_bbox_xyxy: Optional[Sequence[float]],
        top_k: int,
        request_id: str,
        frame_id: Optional[int],
        frame_timestamp: Optional[float],
        frame_metadata: Optional[dict],
    ) -> PromptSegmentationResult:
        del text_threshold, mask_threshold
        image = validate_bgr_image(image_bgr)
        prompt_value = validate_prompt(prompt)
        self.load()
        text_started = time.perf_counter()
        self._set_prompt(prompt_value)
        text_ms = (time.perf_counter() - text_started) * 1000.0

        confidence = max(self.confidence, float(box_threshold))
        inference_started = time.perf_counter()
        prediction = self.model.predict(
            image,
            imgsz=self.image_size,
            conf=confidence,
            iou=0.70,
            device=self.device,
            verbose=False,
        )[0]
        inference_ms = (time.perf_counter() - inference_started) * 1000.0
        height, width = image.shape[:2]
        boxes = np.asarray(
            prediction.boxes.xyxy.detach().cpu().numpy(), dtype=np.float32
        ).reshape(-1, 4)
        scores = np.asarray(
            prediction.boxes.conf.detach().cpu().numpy(), dtype=np.float32
        ).reshape(-1)
        reference = (
            None
            if reference_bbox_xyxy is None
            else np.asarray(reference_bbox_xyxy, dtype=np.float32).reshape(4)
        )
        prompt_color = self._prompt_color(prompt_value)

        ranked: list[tuple[float, int, PromptCandidate]] = []
        for index, (raw_box, raw_score) in enumerate(zip(boxes, scores)):
            box = raw_box.copy()
            box[[0, 2]] = np.clip(box[[0, 2]], 0.0, float(width))
            box[[1, 3]] = np.clip(box[[1, 3]], 0.0, float(height))
            if bbox_area(box) <= 0.0:
                continue
            reference_iou = None
            reference_distance = None
            rank_score = float(raw_score)
            if prompt_color is not None:
                color_fraction = self._color_fraction(image, box, prompt_color)
                # A color word is a useful object prior, not a replacement for
                # the detector.  The boost selects the tight color-consistent
                # proposal among overlapping open-vocabulary candidates; a
                # nearly color-free region is strongly demoted.
                rank_score += 1.5 * color_fraction
                if color_fraction < 0.05:
                    rank_score -= 1.0
            if reference is not None:
                reference_iou = self._iou(box, reference)
                reference_distance = self._center_distance(box, reference)
                diagonal = max(1.0, float(np.hypot(width, height)))
                center = 0.5 * (box[:2] + box[2:])
                center_inside = bool(
                    reference[0] <= center[0] <= reference[2]
                    and reference[1] <= center[1] <= reference[3]
                )
                # The calibrated manipulation-region hint is deliberately
                # decisive: detections elsewhere in the lab are not actionable
                # objects even if their raw open-vocabulary score is higher.
                rank_score += (2.0 if center_inside else -2.0)
                rank_score += reference_iou - 0.25 * reference_distance / diagonal
            candidate = PromptCandidate(
                bbox_xyxy=box,
                detector_score=float(raw_score),
                label=prompt_value,
                rank_score=rank_score,
                detector_index=index,
                reference_iou=reference_iou,
                reference_center_distance=reference_distance,
            )
            ranked.append((rank_score, index, candidate))
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected_candidates = [item[2] for item in ranked[: max(1, int(top_k))]]

        mask = np.zeros((height, width), dtype=np.uint8)
        selected_bbox = None
        selected_score = 0.0
        selected_index = None
        if selected_candidates:
            selected = selected_candidates[0]
            selected_bbox = np.rint(selected.bbox_xyxy).astype(np.int32)
            x1, y1, x2, y2 = (int(value) for value in selected_bbox)
            mask[y1:y2, x1:x2] = 1
            selected_score = float(selected.detector_score)
            selected_index = int(selected.detector_index)

        metadata = dict(frame_metadata or {})
        metadata.update(
            {
                "prompt_detector_backend": "yolo_world",
                "prompt_output_kind": "bbox_only",
                "must_initialize_sam2_from_bbox": True,
            }
        )
        return PromptSegmentationResult(
            mask=mask,
            bbox_xyxy=selected_bbox,
            score=selected_score,
            valid=selected_bbox is not None,
            prompt=prompt_value,
            message=(
                "YOLO-World bbox ready; initialize SAM2 from bbox"
                if selected_bbox is not None
                else "YOLO-World found no matching object"
            ),
            selected_index=selected_index,
            candidates=selected_candidates,
            timings_ms={"text_embedding": text_ms, "detector": inference_ms},
            request_id=str(request_id),
            frame_id=None if frame_id is None else int(frame_id),
            frame_timestamp=(
                None if frame_timestamp is None else float(frame_timestamp)
            ),
            frame_metadata=metadata,
        )
