from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from dynamic_pcd.segmentation.prompt_protocol import (
    PromptCandidate,
    PromptSegmentationResult,
    validate_bgr_image,
    validate_prompt,
)


class PromptModelDependencyError(RuntimeError):
    """A model dependency or weight file needed by the service is unavailable."""


@dataclass
class TextDetection:
    bbox_xyxy: np.ndarray
    score: float
    label: str
    index: int = 0


class NativeGroundingDINODetector:
    """Offline adapter for the original GroundingDINO Python package."""

    def __init__(self, config_path: str, checkpoint: str, device: str = "cpu"):
        self.config_path = str(config_path)
        self.checkpoint = str(checkpoint)
        self.device = str(device)
        self.model = None
        self._torch = None
        self._transform = None
        self._image_type = None
        self._get_phrases_from_posmap = None

    def load(self) -> None:
        if self.model is not None:
            return
        if not os.path.isfile(self.config_path):
            raise PromptModelDependencyError(
                f"Grounding DINO config does not exist: {self.config_path}"
            )
        if not os.path.isfile(self.checkpoint):
            raise PromptModelDependencyError(
                f"Grounding DINO checkpoint does not exist: {self.checkpoint}"
            )
        try:
            import torch
            from PIL import Image
            import groundingdino.datasets.transforms as transforms
            from groundingdino.models import build_model
            from groundingdino.util.misc import clean_state_dict
            from groundingdino.util.slconfig import SLConfig
            from groundingdino.util.utils import get_phrases_from_posmap
        except ImportError as exc:
            raise PromptModelDependencyError(
                "The native Grounding DINO backend requires the original "
                "GroundingDINO package, Torch and Pillow. Run the service with "
                "the FoundationPose environment or configure its Python path."
            ) from exc

        try:
            args = SLConfig.fromfile(self.config_path)
            args.device = self.device
            model = build_model(args)
            checkpoint = torch.load(self.checkpoint, map_location="cpu")
            state = checkpoint.get("model", checkpoint)
            model.load_state_dict(clean_state_dict(state), strict=False)
            model.to(self.device)
            model.eval()
            self.model = model
            self._torch = torch
            self._image_type = Image
            self._get_phrases_from_posmap = get_phrases_from_posmap
            self._transform = transforms.Compose(
                [
                    transforms.RandomResize([800], max_size=1333),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            )
        except Exception as exc:
            raise PromptModelDependencyError(
                f"Could not load native Grounding DINO config={self.config_path!r}, "
                f"checkpoint={self.checkpoint!r}, device={self.device!r}. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

    def detect(
        self,
        image_rgb: np.ndarray,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> List[TextDetection]:
        self.load()
        assert self.model is not None
        assert self._torch is not None
        assert self._image_type is not None
        assert self._transform is not None
        assert self._get_phrases_from_posmap is not None

        caption = prompt.lower().strip()
        if not caption.endswith("."):
            caption += "."
        image_pil = self._image_type.fromarray(np.ascontiguousarray(image_rgb))
        image_tensor, _ = self._transform(image_pil, None)
        image_tensor = image_tensor.to(self.device)
        with self._torch.inference_mode():
            outputs = self.model(image_tensor[None], captions=[caption])

        logits = outputs["pred_logits"].detach().cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].detach().cpu()[0]
        confidence = logits.max(dim=1)[0]
        keep = confidence > float(box_threshold)
        logits = logits[keep]
        boxes = boxes[keep]
        confidence = confidence[keep]
        tokenized = self.model.tokenizer(caption)
        h, w = image_rgb.shape[:2]

        detections: List[TextDetection] = []
        for index, (box_cxcywh, score, logit) in enumerate(
            zip(boxes, confidence, logits)
        ):
            cx, cy, box_w, box_h = [float(value) for value in box_cxcywh]
            box = np.array(
                [
                    (cx - 0.5 * box_w) * w,
                    (cy - 0.5 * box_h) * h,
                    (cx + 0.5 * box_w) * w,
                    (cy + 0.5 * box_h) * h,
                ],
                dtype=np.float32,
            )
            phrase = self._get_phrases_from_posmap(
                logit > float(text_threshold), tokenized, self.model.tokenizer
            ).replace(".", "").strip()
            detections.append(
                TextDetection(
                    bbox_xyxy=box,
                    score=float(score),
                    label=phrase if phrase else prompt,
                    index=index,
                )
            )
        return detections


class TransformersGroundingDINODetector:
    """Lazy Transformers adapter for Grounding DINO zero-shot detection."""

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        local_files_only: bool = False,
    ):
        self.model_id = str(model_id)
        self.device = str(device)
        self.local_files_only = bool(local_files_only)
        self.processor = None
        self.model = None
        self._torch = None

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )
        except ImportError as exc:
            raise PromptModelDependencyError(
                "Grounding DINO requires the optional 'transformers' package in "
                "the prompt-service environment. Install the dependencies listed "
                "in requirements_prompt_segmentation.txt into the dynamic Python "
                "environment; the camera/client environment does not need them."
            ) from exc

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, local_files_only=self.local_files_only
            )
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_id, local_files_only=self.local_files_only
            )
            self.model.to(self.device)
            self.model.eval()
            self._torch = torch
        except Exception as exc:
            mode = "local cache" if self.local_files_only else "model source/cache"
            raise PromptModelDependencyError(
                f"Could not load Grounding DINO model {self.model_id!r} from the "
                f"{mode}. Verify the model id, cached weights, and network access. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

    def detect(
        self,
        image_rgb: np.ndarray,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> List[TextDetection]:
        self.load()
        assert self.processor is not None
        assert self.model is not None
        assert self._torch is not None

        # Grounding DINO's tokenizer performs best when a category phrase ends
        # in a period.  This is syntax normalization, not a fixed class list.
        model_prompt = prompt.strip()
        if not model_prompt.endswith("."):
            model_prompt += "."

        inputs = self.processor(
            images=np.ascontiguousarray(image_rgb),
            text=model_prompt,
            return_tensors="pt",
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        else:
            inputs = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

        with self._torch.inference_mode():
            outputs = self.model(**inputs)

        h, w = image_rgb.shape[:2]
        input_ids = inputs.get("input_ids")
        try:
            processed = self.processor.post_process_grounded_object_detection(
                outputs,
                input_ids=input_ids,
                box_threshold=float(box_threshold),
                text_threshold=float(text_threshold),
                target_sizes=[(h, w)],
            )
        except TypeError:
            # Compatibility with Transformers releases which used ``threshold``
            # rather than separate box/text thresholds.
            processed = self.processor.post_process_grounded_object_detection(
                outputs,
                input_ids,
                threshold=float(box_threshold),
                target_sizes=[(h, w)],
            )

        if not processed:
            return []
        output = processed[0]
        boxes = _to_numpy(output.get("boxes", [])).reshape(-1, 4)
        scores = _to_numpy(output.get("scores", [])).reshape(-1)
        labels = output.get("text_labels", output.get("labels", []))
        labels_list = _labels_to_strings(labels, len(boxes), fallback=prompt)

        detections: List[TextDetection] = []
        for index, (box, score) in enumerate(zip(boxes, scores)):
            detections.append(
                TextDetection(
                    bbox_xyxy=np.asarray(box, dtype=np.float32),
                    score=float(score),
                    label=labels_list[index],
                    index=index,
                )
            )
        return detections


class SAM1BoxMaskPredictor:
    """Offline adapter for Meta's original Segment Anything predictor."""

    def __init__(
        self,
        checkpoint: str,
        model_type: str = "vit_b",
        device: str = "cpu",
        multimask_output: bool = True,
    ):
        self.checkpoint = str(checkpoint)
        self.model_type = str(model_type)
        self.device = str(device)
        self.multimask_output = bool(multimask_output)
        self.predictor = None

    def load(self) -> None:
        if self.predictor is not None:
            return
        if not os.path.isfile(self.checkpoint):
            raise PromptModelDependencyError(
                f"SAM checkpoint does not exist: {self.checkpoint}"
            )
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise PromptModelDependencyError(
                "The SAM1 backend requires the original segment_anything package. "
                "Run the service with the FoundationPose environment or configure "
                "its Python path."
            ) from exc
        if self.model_type not in sam_model_registry:
            available = ", ".join(sorted(sam_model_registry.keys()))
            raise PromptModelDependencyError(
                f"Unknown SAM model type {self.model_type!r}; available: {available}"
            )
        try:
            model = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
            model.to(device=self.device)
            model.eval()
            self.predictor = SamPredictor(model)
        except Exception as exc:
            raise PromptModelDependencyError(
                f"Could not load SAM model_type={self.model_type!r}, "
                f"checkpoint={self.checkpoint!r}, device={self.device!r}. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

    def set_image(self, image_rgb: np.ndarray) -> None:
        self.load()
        assert self.predictor is not None
        self.predictor.set_image(np.ascontiguousarray(image_rgb))

    def predict(self, bbox_xyxy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.predictor is None:
            raise RuntimeError("SAM predictor has no image embedding")
        masks, scores, _logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.asarray(bbox_xyxy, dtype=np.float32).reshape(4),
            multimask_output=self.multimask_output,
        )
        return _normalize_masks_and_scores(masks, scores)


class SAM2BoxMaskPredictor:
    """Lazy adapter around the local official SAM2 image predictor."""

    def __init__(
        self,
        checkpoint: str,
        model_cfg: str,
        device: str = "cpu",
        multimask_output: bool = True,
    ):
        self.checkpoint = str(checkpoint)
        self.model_cfg = str(model_cfg)
        self.device = str(device)
        self.multimask_output = bool(multimask_output)
        self.predictor = None

    def load(self) -> None:
        if self.predictor is not None:
            return
        if not os.path.isfile(self.checkpoint):
            raise PromptModelDependencyError(
                f"SAM2 checkpoint does not exist: {self.checkpoint}"
            )
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise PromptModelDependencyError(
                "Could not import the local official SAM2 package in the prompt-"
                "service environment. Install it editable or expose the local "
                "facebookresearch/sam2 checkout on PYTHONPATH."
            ) from exc

        try:
            model = build_sam2(
                self.model_cfg,
                self.checkpoint,
                device=self.device,
            )
            self.predictor = SAM2ImagePredictor(model)
        except Exception as exc:
            raise PromptModelDependencyError(
                f"Could not load SAM2 checkpoint={self.checkpoint!r}, "
                f"model_cfg={self.model_cfg!r}, device={self.device!r}. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

    def set_image(self, image_rgb: np.ndarray) -> None:
        self.load()
        assert self.predictor is not None
        self.predictor.set_image(np.ascontiguousarray(image_rgb))

    def predict(self, bbox_xyxy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.predictor is None:
            raise RuntimeError("SAM2 predictor has no image embedding")
        masks, scores, _logits = self.predictor.predict(
            box=np.asarray(bbox_xyxy, dtype=np.float32).reshape(1, 4),
            multimask_output=self.multimask_output,
        )
        return _normalize_masks_and_scores(masks, scores)


class GroundedSAMBackend:
    """Text prompt -> Grounding DINO boxes -> SAM/SAM2 mask.

    Grounding DINO and the SAM image encoder are intentionally used only for
    acquisition/re-acquisition.  A high-rate tracker can consume the resulting
    mask/box without placing these CPU-heavy models in its 20 Hz frame loop.

    ``detector`` and ``mask_predictor`` can be supplied by tests or alternate
    implementations.  The production adapters are loaded lazily and then kept
    resident for the lifetime of the service process.
    """

    def __init__(
        self,
        detector_backend: str = "native",
        mask_backend: str = "sam1",
        dino_config: Optional[str] = None,
        dino_checkpoint: Optional[str] = None,
        dino_model_id: str = "IDEA-Research/grounding-dino-tiny",
        sam_checkpoint: Optional[str] = None,
        sam_model_type: str = "vit_b",
        sam2_checkpoint: Optional[str] = None,
        sam2_model_cfg: Optional[str] = None,
        device: str = "cpu",
        local_files_only: bool = False,
        detector: Optional[Any] = None,
        mask_predictor: Optional[Any] = None,
        min_mask_area: int = 20,
        max_mask_area_ratio: float = 6.0,
        max_sam_candidates: int = 3,
        nms_iou_threshold: float = 0.80,
        reference_iou_weight: float = 1.0,
        reference_center_weight: float = 0.25,
    ):
        self.detector_backend = str(detector_backend).lower()
        self.mask_backend = str(mask_backend).lower()
        self.dino_config = dino_config
        self.dino_checkpoint = dino_checkpoint
        self.dino_model_id = str(dino_model_id)
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = str(sam_model_type)
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_model_cfg = sam2_model_cfg
        self.device = str(device)
        self.local_files_only = bool(local_files_only)
        self.detector = detector
        self.mask_predictor = mask_predictor
        self.min_mask_area = max(1, int(min_mask_area))
        self.max_mask_area_ratio = max(0.0, float(max_mask_area_ratio))
        self.max_sam_candidates = max(1, int(max_sam_candidates))
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.reference_iou_weight = float(reference_iou_weight)
        self.reference_center_weight = float(reference_center_weight)

    def load(self) -> None:
        if self.detector is None:
            if self.detector_backend == "native":
                if not self.dino_config or not self.dino_checkpoint:
                    raise PromptModelDependencyError(
                        "native Grounding DINO requires --dino-config and "
                        "--dino-checkpoint"
                    )
                self.detector = NativeGroundingDINODetector(
                    config_path=self.dino_config,
                    checkpoint=self.dino_checkpoint,
                    device=self.device,
                )
            elif self.detector_backend == "transformers":
                self.detector = TransformersGroundingDINODetector(
                    model_id=self.dino_model_id,
                    device=self.device,
                    local_files_only=self.local_files_only,
                )
            else:
                raise ValueError(
                    "detector_backend must be 'native' or 'transformers', "
                    f"received {self.detector_backend!r}"
                )
        if self.mask_predictor is None:
            if self.mask_backend == "sam1":
                if not self.sam_checkpoint:
                    raise PromptModelDependencyError(
                        "SAM1 requires --sam-checkpoint"
                    )
                self.mask_predictor = SAM1BoxMaskPredictor(
                    checkpoint=self.sam_checkpoint,
                    model_type=self.sam_model_type,
                    device=self.device,
                )
            elif self.mask_backend == "sam2":
                if not self.sam2_checkpoint or not self.sam2_model_cfg:
                    raise PromptModelDependencyError(
                        "SAM2 requires both --sam2-checkpoint and --sam2-model-cfg"
                    )
                self.mask_predictor = SAM2BoxMaskPredictor(
                    checkpoint=self.sam2_checkpoint,
                    model_cfg=self.sam2_model_cfg,
                    device=self.device,
                )
            else:
                raise ValueError(
                    "mask_backend must be 'sam1' or 'sam2', "
                    f"received {self.mask_backend!r}"
                )
        if hasattr(self.detector, "load"):
            self.detector.load()
        if hasattr(self.mask_predictor, "load"):
            self.mask_predictor.load()

    def segment(
        self,
        image_bgr: np.ndarray,
        prompt: str,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        mask_threshold: float = 0.50,
        reference_bbox_xyxy: Optional[Sequence[float]] = None,
        top_k: int = 5,
        request_id: str = "",
        frame_id: Optional[int] = None,
        frame_timestamp: Optional[float] = None,
        frame_metadata: Optional[Dict[str, Any]] = None,
    ) -> PromptSegmentationResult:
        image = validate_bgr_image(image_bgr)
        prompt_value = validate_prompt(prompt)
        h, w = image.shape[:2]
        image_rgb = np.ascontiguousarray(image[:, :, ::-1])
        total_start = time.perf_counter()

        self.load()
        detector_start = time.perf_counter()
        raw_detections = self.detector.detect(
            image_rgb=image_rgb,
            prompt=prompt_value,
            box_threshold=float(box_threshold),
            text_threshold=float(text_threshold),
        )
        detection_ms = (time.perf_counter() - detector_start) * 1000.0
        detections = _normalize_detections(raw_detections, width=w, height=h)
        detections = non_maximum_suppression(
            detections, iou_threshold=self.nms_iou_threshold
        )
        candidates = rank_detections(
            detections,
            image_shape_hw=(h, w),
            reference_bbox_xyxy=reference_bbox_xyxy,
            reference_iou_weight=self.reference_iou_weight,
            reference_center_weight=self.reference_center_weight,
        )[: max(1, int(top_k))]

        if not candidates:
            return _empty_result(
                shape_hw=(h, w),
                prompt=prompt_value,
                message="Grounding DINO found no box above the requested thresholds",
                request_id=request_id,
                frame_id=frame_id,
                frame_timestamp=frame_timestamp,
                frame_metadata=frame_metadata,
                timings_ms={
                    "detector": detection_ms,
                    "total": (time.perf_counter() - total_start) * 1000.0,
                },
            )

        sam_start = time.perf_counter()
        self.mask_predictor.set_image(image_rgb)
        selected_index: Optional[int] = None
        selected_mask: Optional[np.ndarray] = None
        selected_bbox: Optional[np.ndarray] = None
        selected_score = 0.0

        for candidate_index, candidate in enumerate(
            candidates[: self.max_sam_candidates]
        ):
            masks, mask_scores = self.mask_predictor.predict(candidate.bbox_xyxy)
            best_mask, best_mask_score = _best_mask(masks, mask_scores)
            center = (
                0.5 * float(candidate.bbox_xyxy[0] + candidate.bbox_xyxy[2]),
                0.5 * float(candidate.bbox_xyxy[1] + candidate.bbox_xyxy[3]),
            )
            mask = _largest_component(
                (best_mask > 0).astype(np.uint8),
                min_area=self.min_mask_area,
                prefer_center=center,
            )
            mask_area = int(mask.sum())
            detector_area = max(1.0, _bbox_area(candidate.bbox_xyxy))
            mask_area_ratio = float(mask_area) / detector_area
            reasons: List[str] = []
            if best_mask_score < float(mask_threshold):
                reasons.append("mask_score_below_threshold")
            if mask_area < self.min_mask_area:
                reasons.append("mask_too_small")
            if (
                self.max_mask_area_ratio > 0.0
                and mask_area_ratio > self.max_mask_area_ratio
            ):
                reasons.append("mask_too_large_for_detector_box")
            mask_bbox = _mask_bbox(mask, min_area=self.min_mask_area)
            if mask_bbox is None:
                reasons.append("empty_mask")

            candidate.mask_score = float(best_mask_score)
            candidate.mask_area = mask_area
            candidate.mask_valid = not reasons
            candidate.rejection_reason = ",".join(reasons)
            if reasons:
                continue

            selected_index = candidate_index
            selected_mask = mask
            selected_bbox = np.asarray(mask_bbox, dtype=np.int32)
            selected_score = float(candidate.detector_score * best_mask_score)
            break

        sam_ms = (time.perf_counter() - sam_start) * 1000.0
        timings = {
            "detector": detection_ms,
            "segmenter": sam_ms,
            "total": (time.perf_counter() - total_start) * 1000.0,
        }
        if selected_mask is None:
            result = _empty_result(
                shape_hw=(h, w),
                prompt=prompt_value,
                message="segmenter rejected every evaluated Grounding DINO candidate",
                request_id=request_id,
                frame_id=frame_id,
                frame_timestamp=frame_timestamp,
                frame_metadata=frame_metadata,
                candidates=candidates,
                timings_ms=timings,
            )
            return result

        message = (
            f"selected candidate {selected_index} of {len(candidates)}; "
            f"detector={candidates[selected_index].detector_score:.3f}, "
            f"mask={candidates[selected_index].mask_score:.3f}"
        )
        result = PromptSegmentationResult(
            mask=selected_mask,
            bbox_xyxy=selected_bbox,
            score=selected_score,
            valid=True,
            prompt=prompt_value,
            message=message,
            selected_index=selected_index,
            candidates=candidates,
            timings_ms=timings,
            request_id=str(request_id),
            frame_id=frame_id,
            frame_timestamp=frame_timestamp,
            frame_metadata={} if frame_metadata is None else dict(frame_metadata),
        )
        return result


def rank_detections(
    detections: Iterable[TextDetection],
    image_shape_hw: Tuple[int, int],
    reference_bbox_xyxy: Optional[Sequence[float]] = None,
    reference_iou_weight: float = 1.0,
    reference_center_weight: float = 0.25,
) -> List[PromptCandidate]:
    """Rank detections by text confidence and optional previous/reference box."""

    h, w = (int(image_shape_hw[0]), int(image_shape_hw[1]))
    reference = None
    if reference_bbox_xyxy is not None:
        reference = _clip_bbox(reference_bbox_xyxy, width=w, height=h)
    diagonal = max(1.0, float(np.hypot(w, h)))

    candidates: List[PromptCandidate] = []
    for detection in detections:
        box = _clip_bbox(detection.bbox_xyxy, width=w, height=h)
        iou: Optional[float] = None
        center_distance: Optional[float] = None
        rank_score = float(detection.score)
        if reference is not None:
            iou = bbox_iou(box, reference)
            center_distance = _bbox_center_distance(box, reference) / diagonal
            rank_score += float(reference_iou_weight) * iou
            rank_score -= float(reference_center_weight) * center_distance
        candidates.append(
            PromptCandidate(
                bbox_xyxy=box,
                detector_score=float(detection.score),
                label=str(detection.label),
                rank_score=rank_score,
                detector_index=int(detection.index),
                reference_iou=iou,
                reference_center_distance=center_distance,
            )
        )
    candidates.sort(key=lambda item: item.rank_score, reverse=True)
    return candidates


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    a = np.asarray(first, dtype=np.float32).reshape(4)
    b = np.asarray(second, dtype=np.float32).reshape(4)
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _bbox_area(a) + _bbox_area(b) - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def non_maximum_suppression(
    detections: Sequence[TextDetection], iou_threshold: float = 0.80
) -> List[TextDetection]:
    """Small NumPy NMS used by both native and Transformers detectors."""

    threshold = float(iou_threshold)
    if threshold <= 0.0 or threshold >= 1.0:
        return sorted(detections, key=lambda item: item.score, reverse=True)
    remaining = sorted(detections, key=lambda item: item.score, reverse=True)
    kept: List[TextDetection] = []
    while remaining:
        selected = remaining.pop(0)
        kept.append(selected)
        remaining = [
            candidate
            for candidate in remaining
            if bbox_iou(selected.bbox_xyxy, candidate.bbox_xyxy) <= threshold
        ]
    return kept


def _empty_result(
    shape_hw: Tuple[int, int],
    prompt: str,
    message: str,
    request_id: str,
    frame_id: Optional[int],
    frame_timestamp: Optional[float],
    frame_metadata: Optional[Dict[str, Any]],
    candidates: Optional[List[PromptCandidate]] = None,
    timings_ms: Optional[Dict[str, float]] = None,
) -> PromptSegmentationResult:
    result = PromptSegmentationResult(
        mask=np.zeros(shape_hw, dtype=np.uint8),
        bbox_xyxy=None,
        score=0.0,
        valid=False,
        prompt=prompt,
        message=message,
        candidates=[] if candidates is None else candidates,
        timings_ms={} if timings_ms is None else timings_ms,
        request_id=str(request_id),
        frame_id=frame_id,
        frame_timestamp=frame_timestamp,
        frame_metadata={} if frame_metadata is None else dict(frame_metadata),
    )
    return result


def _normalize_detections(
    raw_detections: Iterable[Any], width: int, height: int
) -> List[TextDetection]:
    normalized: List[TextDetection] = []
    for fallback_index, value in enumerate(raw_detections):
        if isinstance(value, TextDetection):
            detection = value
        elif isinstance(value, dict):
            detection = TextDetection(
                bbox_xyxy=np.asarray(value["bbox_xyxy"], dtype=np.float32),
                score=float(value.get("score", value.get("detector_score", 0.0))),
                label=str(value.get("label", "")),
                index=int(value.get("index", fallback_index)),
            )
        else:
            raise TypeError(
                "detector results must be TextDetection instances or dictionaries"
            )
        box = _clip_bbox(detection.bbox_xyxy, width=width, height=height)
        if _bbox_area(box) <= 0.0 or not np.isfinite(detection.score):
            continue
        normalized.append(
            TextDetection(
                bbox_xyxy=box,
                score=float(detection.score),
                label=str(detection.label),
                index=int(detection.index),
            )
        )
    return normalized


def _clip_bbox(value: Sequence[float], width: int, height: int) -> np.ndarray:
    box = np.asarray(value, dtype=np.float32).reshape(4).copy()
    box[0] = np.clip(box[0], 0.0, float(width))
    box[2] = np.clip(box[2], 0.0, float(width))
    box[1] = np.clip(box[1], 0.0, float(height))
    box[3] = np.clip(box[3], 0.0, float(height))
    return box


def _bbox_area(value: Sequence[float]) -> float:
    box = np.asarray(value, dtype=np.float32).reshape(4)
    return max(0.0, float(box[2] - box[0])) * max(
        0.0, float(box[3] - box[1])
    )


def _bbox_center_distance(first: Sequence[float], second: Sequence[float]) -> float:
    a = np.asarray(first, dtype=np.float32).reshape(4)
    b = np.asarray(second, dtype=np.float32).reshape(4)
    a_center = 0.5 * (a[:2] + a[2:])
    b_center = 0.5 * (b[:2] + b[2:])
    return float(np.linalg.norm(a_center - b_center))


def _mask_bbox(mask: np.ndarray, min_area: int = 1) -> Optional[np.ndarray]:
    binary = np.asarray(mask) > 0
    if int(binary.sum()) < int(min_area):
        return None
    ys, xs = np.nonzero(binary)
    return np.array(
        [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        dtype=np.int32,
    )


def _largest_component(
    mask: np.ndarray,
    min_area: int,
    prefer_center: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Keep one connected component without adding an import-time OpenCV dep."""

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    try:
        import cv2

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        if count <= 1:
            return binary if int(binary.sum()) >= min_area else np.zeros_like(binary)
        eligible = [
            index
            for index in range(1, count)
            if int(stats[index, cv2.CC_STAT_AREA]) >= int(min_area)
        ]
        if not eligible:
            return np.zeros_like(binary)
        if prefer_center is None:
            selected = max(eligible, key=lambda index: stats[index, cv2.CC_STAT_AREA])
        else:
            cx, cy = prefer_center
            selected = min(
                eligible,
                key=lambda index: (
                    float(centroids[index, 0] - cx) ** 2
                    + float(centroids[index, 1] - cy) ** 2,
                    -int(stats[index, cv2.CC_STAT_AREA]),
                ),
            )
        return (labels == selected).astype(np.uint8)
    except ImportError:
        # Unit-test/minimal-client fallback. Production model environments have
        # OpenCV, so this deliberately simple flood fill is rarely exercised.
        return _largest_component_flood_fill(binary, min_area, prefer_center)


def _largest_component_flood_fill(
    binary: np.ndarray,
    min_area: int,
    prefer_center: Optional[Tuple[float, float]],
) -> np.ndarray:
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    components: List[List[Tuple[int, int]]] = []
    for y in range(h):
        for x in range(w):
            if not binary[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            component: List[Tuple[int, int]] = []
            while stack:
                current_y, current_x = stack.pop()
                component.append((current_y, current_x))
                for next_y in range(max(0, current_y - 1), min(h, current_y + 2)):
                    for next_x in range(max(0, current_x - 1), min(w, current_x + 2)):
                        if binary[next_y, next_x] and not visited[next_y, next_x]:
                            visited[next_y, next_x] = True
                            stack.append((next_y, next_x))
            if len(component) >= int(min_area):
                components.append(component)
    if not components:
        return np.zeros_like(binary)
    if prefer_center is None:
        selected = max(components, key=len)
    else:
        cx, cy = prefer_center
        selected = min(
            components,
            key=lambda component: (
                (sum(point[1] for point in component) / len(component) - cx) ** 2
                + (sum(point[0] for point in component) / len(component) - cy) ** 2,
                -len(component),
            ),
        )
    output = np.zeros_like(binary)
    for y, x in selected:
        output[y, x] = 1
    return output


def _best_mask(masks: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, float]:
    masks_array, scores_array = _normalize_masks_and_scores(masks, scores)
    if masks_array.shape[0] == 0:
        raise ValueError("segmenter returned no masks")
    index = int(np.argmax(scores_array))
    return masks_array[index], float(scores_array[index])


def _normalize_masks_and_scores(
    masks: Any, scores: Any
) -> Tuple[np.ndarray, np.ndarray]:
    masks_array = np.asarray(masks)
    scores_array = np.asarray(scores, dtype=np.float32).reshape(-1)
    if masks_array.ndim == 2:
        masks_array = masks_array[None, :, :]
    elif masks_array.ndim == 4 and masks_array.shape[0] == 1:
        masks_array = masks_array[0]
    elif masks_array.ndim == 4:
        masks_array = masks_array.reshape((-1,) + masks_array.shape[-2:])
    if masks_array.ndim != 3:
        raise ValueError(f"unexpected segmenter mask shape {masks_array.shape}")
    if scores_array.size == 0:
        scores_array = np.zeros((masks_array.shape[0],), dtype=np.float32)
    elif scores_array.size != masks_array.shape[0]:
        raise ValueError(
            f"segmenter returned {masks_array.shape[0]} masks and "
            f"{scores_array.size} scores"
        )
    return masks_array, scores_array


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _labels_to_strings(labels: Any, count: int, fallback: str) -> List[str]:
    if labels is None:
        values: List[Any] = []
    elif isinstance(labels, str):
        values = [labels]
    elif hasattr(labels, "tolist"):
        values = list(labels.tolist())
    else:
        values = list(labels)
    result: List[str] = []
    for index in range(count):
        value = values[index] if index < len(values) else fallback
        # Some older Transformers versions return token/category indices rather
        # than grounded text.  The caller's dynamic prompt is more informative.
        result.append(str(value) if isinstance(value, str) else str(fallback))
    return result


# Compatibility for callers which used the first, SAM2-only class name.
GroundedSAM2Backend = GroundedSAMBackend
