from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


PROTOCOL_VERSION = 1


class PromptServiceError(RuntimeError):
    """Raised by the lightweight client when the prompt service fails."""


@dataclass
class PromptCandidate:
    """Metadata for one text-detector candidate.

    ``bbox_xyxy`` is the detector box.  The selected result's mask-derived box
    is stored separately on :class:`PromptSegmentationResult`.
    """

    bbox_xyxy: np.ndarray
    detector_score: float
    label: str
    rank_score: float
    detector_index: int
    reference_iou: Optional[float] = None
    reference_center_distance: Optional[float] = None
    mask_score: Optional[float] = None
    mask_area: Optional[int] = None
    mask_valid: Optional[bool] = None
    rejection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox_xyxy": np.asarray(self.bbox_xyxy, dtype=np.float32).tolist(),
            "detector_score": float(self.detector_score),
            "label": str(self.label),
            "rank_score": float(self.rank_score),
            "detector_index": int(self.detector_index),
            "reference_iou": (
                None if self.reference_iou is None else float(self.reference_iou)
            ),
            "reference_center_distance": (
                None
                if self.reference_center_distance is None
                else float(self.reference_center_distance)
            ),
            "mask_score": None if self.mask_score is None else float(self.mask_score),
            "mask_area": None if self.mask_area is None else int(self.mask_area),
            "mask_valid": None if self.mask_valid is None else bool(self.mask_valid),
            "rejection_reason": str(self.rejection_reason),
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "PromptCandidate":
        return cls(
            bbox_xyxy=np.asarray(value["bbox_xyxy"], dtype=np.float32),
            detector_score=float(value["detector_score"]),
            label=str(value.get("label", "")),
            rank_score=float(value.get("rank_score", value["detector_score"])),
            detector_index=int(value.get("detector_index", 0)),
            reference_iou=_optional_float(value.get("reference_iou")),
            reference_center_distance=_optional_float(
                value.get("reference_center_distance")
            ),
            mask_score=_optional_float(value.get("mask_score")),
            mask_area=(
                None if value.get("mask_area") is None else int(value["mask_area"])
            ),
            mask_valid=(
                None if value.get("mask_valid") is None else bool(value["mask_valid"])
            ),
            rejection_reason=str(value.get("rejection_reason", "")),
        )


@dataclass
class PromptSegmentationResult:
    mask: np.ndarray
    bbox_xyxy: Optional[np.ndarray]
    score: float
    valid: bool
    prompt: str
    message: str = ""
    selected_index: Optional[int] = None
    candidates: List[PromptCandidate] = field(default_factory=list)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    request_id: str = ""
    frame_id: Optional[int] = None
    frame_timestamp: Optional[float] = None
    frame_metadata: Dict[str, Any] = field(default_factory=dict)

    def metadata_dict(self) -> Dict[str, Any]:
        return {
            "bbox_xyxy": (
                None
                if self.bbox_xyxy is None
                else np.asarray(self.bbox_xyxy, dtype=np.int32).tolist()
            ),
            "score": float(self.score),
            "valid": bool(self.valid),
            "prompt": str(self.prompt),
            "message": str(self.message),
            "selected_index": (
                None if self.selected_index is None else int(self.selected_index)
            ),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "timings_ms": {
                str(key): float(value) for key, value in self.timings_ms.items()
            },
            "request_id": str(self.request_id),
            "frame_id": None if self.frame_id is None else int(self.frame_id),
            "frame_timestamp": (
                None
                if self.frame_timestamp is None
                else float(self.frame_timestamp)
            ),
            "frame_metadata": dict(self.frame_metadata),
        }

    @classmethod
    def from_wire(
        cls, metadata: Dict[str, Any], mask_payload: bytes
    ) -> "PromptSegmentationResult":
        mask_info = metadata.get("mask") or {}
        shape = tuple(int(value) for value in mask_info.get("shape", ()))
        if len(shape) != 2:
            raise PromptServiceError(f"invalid response mask shape: {shape!r}")
        expected_size = int(shape[0]) * int(shape[1])
        if len(mask_payload) != expected_size:
            raise PromptServiceError(
                "invalid response mask payload: "
                f"expected {expected_size} bytes, received {len(mask_payload)}"
            )
        mask = np.frombuffer(mask_payload, dtype=np.uint8).reshape(shape).copy()
        bbox_value = metadata.get("bbox_xyxy")
        return cls(
            mask=mask,
            bbox_xyxy=(
                None if bbox_value is None else np.asarray(bbox_value, dtype=np.int32)
            ),
            score=float(metadata.get("score", 0.0)),
            valid=bool(metadata.get("valid", False)),
            prompt=str(metadata.get("prompt", "")),
            message=str(metadata.get("message", "")),
            selected_index=(
                None
                if metadata.get("selected_index") is None
                else int(metadata["selected_index"])
            ),
            candidates=[
                PromptCandidate.from_dict(candidate)
                for candidate in metadata.get("candidates", [])
            ],
            timings_ms={
                str(key): float(value)
                for key, value in metadata.get("timings_ms", {}).items()
            },
            request_id=str(metadata.get("request_id", "")),
            frame_id=(
                None if metadata.get("frame_id") is None else int(metadata["frame_id"])
            ),
            frame_timestamp=_optional_float(metadata.get("frame_timestamp")),
            frame_metadata=dict(metadata.get("frame_metadata") or {}),
        )


def make_segment_request(
    image_bgr: np.ndarray,
    prompt: str,
    request_id: str,
    box_threshold: float,
    text_threshold: float,
    mask_threshold: float,
    reference_bbox_xyxy: Optional[Sequence[float]],
    top_k: int,
    frame_id: Optional[int] = None,
    frame_timestamp: Optional[float] = None,
    frame_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[bytes, bytes]:
    image = validate_bgr_image(image_bgr)
    prompt_value = validate_prompt(prompt)
    header = {
        "version": PROTOCOL_VERSION,
        "op": "segment",
        "request_id": str(request_id),
        "prompt": prompt_value,
        "image": {
            "shape": [int(value) for value in image.shape],
            "dtype": "uint8",
            "color_space": "BGR",
        },
        "box_threshold": _unit_interval("box_threshold", box_threshold),
        "text_threshold": _unit_interval("text_threshold", text_threshold),
        "mask_threshold": _unit_interval("mask_threshold", mask_threshold),
        "reference_bbox_xyxy": _optional_bbox(reference_bbox_xyxy),
        "top_k": max(1, int(top_k)),
        "frame_id": None if frame_id is None else int(frame_id),
        "frame_timestamp": (
            None if frame_timestamp is None else float(frame_timestamp)
        ),
        "frame_metadata": {} if frame_metadata is None else dict(frame_metadata),
    }
    return encode_json(header), image.tobytes(order="C")


def decode_segment_request(
    parts: Sequence[bytes], max_image_bytes: int = 16 * 1024 * 1024
) -> Tuple[Dict[str, Any], np.ndarray]:
    if len(parts) != 2:
        raise ValueError(f"segment request needs 2 parts, received {len(parts)}")
    header = decode_json(parts[0])
    if int(header.get("version", -1)) != PROTOCOL_VERSION:
        raise ValueError(
            f"unsupported protocol version {header.get('version')!r}; "
            f"expected {PROTOCOL_VERSION}"
        )
    if header.get("op") != "segment":
        raise ValueError(f"unsupported request op {header.get('op')!r}")
    image_info = header.get("image") or {}
    shape = tuple(int(value) for value in image_info.get("shape", ()))
    if len(shape) != 3 or shape[2] != 3 or min(shape) <= 0:
        raise ValueError(f"invalid BGR image shape {shape!r}")
    if image_info.get("dtype") != "uint8":
        raise ValueError("only uint8 images are supported")
    expected_size = int(np.prod(shape))
    if expected_size > int(max_image_bytes):
        raise ValueError(
            f"image payload is {expected_size} bytes; limit is {max_image_bytes}"
        )
    if len(parts[1]) != expected_size:
        raise ValueError(
            f"image payload mismatch: expected {expected_size}, received {len(parts[1])}"
        )
    image = np.frombuffer(parts[1], dtype=np.uint8).reshape(shape).copy()
    header["prompt"] = validate_prompt(header.get("prompt", ""))
    header["box_threshold"] = _unit_interval(
        "box_threshold", header.get("box_threshold", 0.3)
    )
    header["text_threshold"] = _unit_interval(
        "text_threshold", header.get("text_threshold", 0.25)
    )
    header["mask_threshold"] = _unit_interval(
        "mask_threshold", header.get("mask_threshold", 0.0)
    )
    header["reference_bbox_xyxy"] = _optional_bbox(
        header.get("reference_bbox_xyxy")
    )
    header["top_k"] = max(1, int(header.get("top_k", 5)))
    header["frame_id"] = (
        None if header.get("frame_id") is None else int(header["frame_id"])
    )
    header["frame_timestamp"] = _optional_float(header.get("frame_timestamp"))
    if (
        header["frame_timestamp"] is not None
        and not np.isfinite(header["frame_timestamp"])
    ):
        raise ValueError("frame_timestamp must be finite")
    frame_metadata = header.get("frame_metadata") or {}
    if not isinstance(frame_metadata, dict):
        raise ValueError("frame_metadata must be a JSON object")
    header["frame_metadata"] = frame_metadata
    return header, image


def make_result_parts(result: PromptSegmentationResult) -> Tuple[bytes, bytes]:
    mask = np.ascontiguousarray(result.mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError(f"result mask must be HxW, received {mask.shape}")
    metadata = result.metadata_dict()
    metadata.update(
        {
            "version": PROTOCOL_VERSION,
            "status": "ok",
            "mask": {"shape": [int(value) for value in mask.shape], "dtype": "uint8"},
        }
    )
    return encode_json(metadata), mask.tobytes(order="C")


def make_error_parts(message: str, request_id: str = "") -> Tuple[bytes, bytes]:
    return (
        encode_json(
            {
                "version": PROTOCOL_VERSION,
                "status": "error",
                "request_id": str(request_id),
                "message": str(message),
            }
        ),
        b"",
    )


def encode_json(value: Dict[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def decode_json(payload: bytes) -> Dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise ValueError("invalid UTF-8 JSON header") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON header must be an object")
    return value


def validate_bgr_image(image_bgr: np.ndarray) -> np.ndarray:
    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "image_bgr must be a uint8 HxWx3 array; "
            f"received dtype={image.dtype}, shape={image.shape}"
        )
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("image_bgr cannot be empty")
    return np.ascontiguousarray(image)


def validate_prompt(prompt: Any) -> str:
    value = str(prompt).strip()
    if not value:
        raise ValueError("text prompt cannot be empty")
    if len(value) > 512:
        raise ValueError("text prompt is longer than 512 characters")
    return value


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _unit_interval(name: str, value: Any) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], received {result}")
    return result


def _optional_bbox(value: Optional[Sequence[float]]) -> Optional[List[float]]:
    if value is None:
        return None
    bbox = np.asarray(value, dtype=np.float32).reshape(-1)
    if bbox.size != 4 or not np.all(np.isfinite(bbox)):
        raise ValueError("reference_bbox_xyxy must contain four finite numbers")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError("reference_bbox_xyxy must have positive width and height")
    return [float(item) for item in bbox]
