from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


PROTOCOL_VERSION = 1
DEFAULT_MAX_IMAGE_BYTES = 16 * 1024 * 1024


class SAM2VideoServiceError(RuntimeError):
    """Raised when the online SAM2 service violates the RPC contract."""


@dataclass
class SAM2VideoResult:
    """One mask produced by the stateful SAM2 video predictor."""

    mask: np.ndarray
    valid: bool
    frame_id: int
    message: str = ""
    timings_ms: Dict[str, float] = field(default_factory=dict)
    request_id: str = ""
    internal_frame_idx: int = 0
    object_score: Optional[float] = None
    mask_area: int = 0
    state_reset: bool = False

    def metadata_dict(self) -> Dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "frame_id": validate_frame_id(self.frame_id),
            "message": str(self.message),
            "timings_ms": {
                str(key): _finite_float(f"timings_ms[{key!r}]", value)
                for key, value in self.timings_ms.items()
            },
            "request_id": validate_request_id(self.request_id),
            "internal_frame_idx": int(self.internal_frame_idx),
            "object_score": (
                None
                if self.object_score is None
                else _finite_float("object_score", self.object_score)
            ),
            "mask_area": int(self.mask_area),
            "state_reset": bool(self.state_reset),
        }


def make_initialize_request(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    frame_id: int,
    request_id: str,
) -> Tuple[bytes, bytes, bytes]:
    image = validate_bgr_image(image_bgr)
    mask_u8 = validate_mask(mask, expected_shape=image.shape[:2], require_nonempty=True)
    header = _frame_header("initialize", image, frame_id, request_id)
    header["mask"] = {
        "shape": [int(value) for value in mask_u8.shape],
        "dtype": "uint8",
    }
    return (
        encode_json(header),
        image.tobytes(order="C"),
        mask_u8.tobytes(order="C"),
    )


def make_initialize_box_request(
    image_bgr: np.ndarray,
    bbox_xyxy: np.ndarray,
    frame_id: int,
    request_id: str,
) -> Tuple[bytes, bytes]:
    """Build a first-frame SAM2 box-prompt request.

    The box is expressed in original image pixels as ``[x1, y1, x2, y2]``
    with an exclusive lower-right corner, matching the provider ROI contract.
    """

    image = validate_bgr_image(image_bgr)
    bbox = validate_bbox_xyxy(bbox_xyxy, expected_shape=image.shape[:2])
    header = _frame_header("initialize_box", image, frame_id, request_id)
    header["bbox_xyxy"] = [int(value) for value in bbox]
    return encode_json(header), image.tobytes(order="C")


def make_track_request(
    image_bgr: np.ndarray,
    frame_id: int,
    request_id: str,
) -> Tuple[bytes, bytes]:
    image = validate_bgr_image(image_bgr)
    return (
        encode_json(_frame_header("track", image, frame_id, request_id)),
        image.tobytes(order="C"),
    )


def make_control_request(op: str, request_id: str) -> Tuple[bytes]:
    operation = str(op).strip().lower()
    if operation not in ("health", "reset"):
        raise ValueError(f"unsupported control operation {op!r}")
    return (
        encode_json(
            {
                "version": PROTOCOL_VERSION,
                "op": operation,
                "request_id": validate_request_id(request_id),
            }
        ),
    )


def decode_request(
    parts: Sequence[bytes],
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
    if not parts:
        raise ValueError("empty SAM2 video request")
    header = decode_json(parts[0])
    if int(header.get("version", -1)) != PROTOCOL_VERSION:
        raise ValueError(
            f"unsupported protocol version {header.get('version')!r}; "
            f"expected {PROTOCOL_VERSION}"
        )
    header["request_id"] = validate_request_id(header.get("request_id", ""))
    operation = str(header.get("op", "")).strip().lower()
    header["op"] = operation
    if operation in ("health", "reset"):
        if len(parts) != 1:
            raise ValueError(f"{operation} request must contain exactly one part")
        return header, None, None
    if operation not in ("initialize", "initialize_box", "track"):
        raise ValueError(f"unsupported SAM2 video operation {operation!r}")

    expected_parts = 3 if operation == "initialize" else 2
    if len(parts) != expected_parts:
        raise ValueError(
            f"{operation} request needs {expected_parts} parts, received {len(parts)}"
        )
    header["frame_id"] = validate_frame_id(header.get("frame_id"))
    image_info = header.get("image") or {}
    shape = _shape_tuple(image_info.get("shape"), dimensions=3, name="image")
    if shape[2] != 3:
        raise ValueError(f"BGR image must have three channels, received {shape!r}")
    if image_info.get("dtype") != "uint8":
        raise ValueError("only uint8 BGR images are supported")
    if image_info.get("color_space") != "BGR":
        raise ValueError("image color_space must be BGR")
    expected_image_bytes = math.prod(shape)
    limit = max(1, int(max_image_bytes))
    if expected_image_bytes > limit:
        raise ValueError(
            f"image payload is {expected_image_bytes} bytes; limit is {limit}"
        )
    if len(parts[1]) != expected_image_bytes:
        raise ValueError(
            "image payload mismatch: expected "
            f"{expected_image_bytes}, received {len(parts[1])}"
        )
    image = np.frombuffer(parts[1], dtype=np.uint8).reshape(shape).copy()

    if operation == "initialize_box":
        bbox = validate_bbox_xyxy(
            header.get("bbox_xyxy"), expected_shape=shape[:2]
        )
        header["bbox_xyxy"] = [int(value) for value in bbox]

    mask = None
    if operation == "initialize":
        mask_info = header.get("mask") or {}
        mask_shape = _shape_tuple(mask_info.get("shape"), dimensions=2, name="mask")
        if mask_shape != shape[:2]:
            raise ValueError(
                f"mask shape {mask_shape!r} does not match image shape {shape[:2]!r}"
            )
        if mask_info.get("dtype") != "uint8":
            raise ValueError("only uint8 initialization masks are supported")
        expected_mask_bytes = math.prod(mask_shape)
        if len(parts[2]) != expected_mask_bytes:
            raise ValueError(
                "mask payload mismatch: expected "
                f"{expected_mask_bytes}, received {len(parts[2])}"
            )
        mask = np.frombuffer(parts[2], dtype=np.uint8).reshape(mask_shape).copy()
        mask = validate_mask(mask, expected_shape=shape[:2], require_nonempty=True)
    return header, image, mask


def validate_bbox_xyxy(
    bbox_xyxy: Any,
    *,
    expected_shape: Sequence[int],
) -> np.ndarray:
    """Validate an integer pixel box against an ``(H, W)`` image shape."""

    shape = _shape_tuple(expected_shape, dimensions=2, name="image")
    raw = np.asarray(bbox_xyxy)
    if raw.shape != (4,):
        raise ValueError(
            f"bbox_xyxy must contain exactly four values, received {raw.shape!r}"
        )
    if np.issubdtype(raw.dtype, np.bool_):
        raise TypeError("bbox_xyxy values must be integer pixels, not booleans")
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("bbox_xyxy values must be numeric") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError("bbox_xyxy values must be finite")
    if not np.all(numeric == np.floor(numeric)):
        raise ValueError("bbox_xyxy values must be integer pixels")
    bbox = numeric.astype(np.int64)
    x1, y1, x2, y2 = (int(value) for value in bbox)
    height, width = shape
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(
            f"bbox_xyxy {bbox.tolist()} is outside image bounds "
            f"width={width}, height={height}"
        )
    return bbox


def make_result_parts(result: SAM2VideoResult) -> Tuple[bytes, bytes]:
    mask = validate_mask(result.mask, require_nonempty=False)
    metadata = result.metadata_dict()
    metadata["mask_area"] = int(mask.sum())
    metadata.update(
        {
            "version": PROTOCOL_VERSION,
            "status": "ok",
            "mask": {
                "shape": [int(value) for value in mask.shape],
                "dtype": "uint8",
            },
        }
    )
    return encode_json(metadata), mask.tobytes(order="C")


def make_ok_parts(op: str, request_id: str, **metadata: Any) -> Tuple[bytes]:
    value = {
        "version": PROTOCOL_VERSION,
        "status": "ok",
        "op": str(op),
        "request_id": validate_request_id(request_id),
    }
    value.update(metadata)
    return (encode_json(value),)


def make_error_parts(message: str, request_id: str = "") -> Tuple[bytes]:
    return (
        encode_json(
            {
                "version": PROTOCOL_VERSION,
                "status": "error",
                "request_id": str(request_id),
                "message": str(message),
            }
        ),
    )


def decode_result_parts(
    parts: Sequence[bytes],
    *,
    expected_request_id: Optional[str] = None,
    expected_frame_id: Optional[int] = None,
    expected_shape: Optional[Sequence[int]] = None,
) -> SAM2VideoResult:
    if len(parts) == 1:
        metadata = decode_json(parts[0])
        _validate_response_header(
            metadata, expected_request_id=expected_request_id
        )
        raise SAM2VideoServiceError("SAM2 video service returned no mask")
    if len(parts) != 2:
        raise SAM2VideoServiceError(
            f"SAM2 video result needs 2 parts, received {len(parts)}"
        )
    metadata = decode_json(parts[0])
    _validate_response_header(metadata, expected_request_id=expected_request_id)
    mask_info = metadata.get("mask") or {}
    shape = _shape_tuple(mask_info.get("shape"), dimensions=2, name="result mask")
    if mask_info.get("dtype") != "uint8":
        raise SAM2VideoServiceError("result mask dtype must be uint8")
    if expected_shape is not None and shape != tuple(int(v) for v in expected_shape):
        raise SAM2VideoServiceError(
            f"result mask shape {shape!r} does not match request {tuple(expected_shape)!r}"
        )
    expected_bytes = math.prod(shape)
    if len(parts[1]) != expected_bytes:
        raise SAM2VideoServiceError(
            "result mask payload mismatch: expected "
            f"{expected_bytes}, received {len(parts[1])}"
        )
    frame_id = validate_frame_id(metadata.get("frame_id"))
    if expected_frame_id is not None and frame_id != validate_frame_id(expected_frame_id):
        raise SAM2VideoServiceError(
            f"response frame_id {frame_id} does not match request {expected_frame_id}"
        )
    mask = np.frombuffer(parts[1], dtype=np.uint8).reshape(shape).copy()
    mask = (mask > 0).astype(np.uint8)
    actual_mask_area = int(mask.sum())
    mask_area = int(metadata.get("mask_area", actual_mask_area))
    if mask_area != actual_mask_area:
        raise SAM2VideoServiceError(
            f"result mask_area {mask_area} does not match payload {actual_mask_area}"
        )
    if bool(metadata.get("valid", False)) and actual_mask_area == 0:
        raise SAM2VideoServiceError("service marked an empty result mask as valid")
    timings = metadata.get("timings_ms") or {}
    if not isinstance(timings, dict):
        raise SAM2VideoServiceError("timings_ms must be a JSON object")
    return SAM2VideoResult(
        mask=mask,
        valid=bool(metadata.get("valid", False)),
        frame_id=frame_id,
        message=str(metadata.get("message", "")),
        timings_ms={
            str(key): _finite_float(f"timings_ms[{key!r}]", value)
            for key, value in timings.items()
        },
        request_id=str(metadata.get("request_id", "")),
        internal_frame_idx=int(metadata.get("internal_frame_idx", 0)),
        object_score=(
            None
            if metadata.get("object_score") is None
            else _finite_float("object_score", metadata["object_score"])
        ),
        mask_area=mask_area,
        state_reset=bool(metadata.get("state_reset", False)),
    )


def decode_control_response(
    parts: Sequence[bytes],
    *,
    expected_request_id: str,
) -> Dict[str, Any]:
    if len(parts) != 1:
        raise SAM2VideoServiceError(
            f"control response needs one part, received {len(parts)}"
        )
    metadata = decode_json(parts[0])
    _validate_response_header(metadata, expected_request_id=expected_request_id)
    return metadata


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


def validate_mask(
    mask: np.ndarray,
    *,
    expected_shape: Optional[Sequence[int]] = None,
    require_nonempty: bool = False,
) -> np.ndarray:
    value = np.asarray(mask)
    if value.dtype not in (np.dtype(np.uint8), np.dtype(np.bool_)) or value.ndim != 2:
        raise ValueError(
            "mask must be a uint8 or bool HxW array; "
            f"received dtype={value.dtype}, shape={value.shape}"
        )
    if value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError("mask cannot be empty")
    if expected_shape is not None and value.shape != tuple(int(v) for v in expected_shape):
        raise ValueError(
            f"mask shape {value.shape!r} does not match image shape "
            f"{tuple(int(v) for v in expected_shape)!r}"
        )
    result = np.ascontiguousarray(value > 0, dtype=np.uint8)
    if require_nonempty and not bool(result.any()):
        raise ValueError("initialization mask cannot be empty")
    return result


def validate_frame_id(frame_id: Any) -> int:
    if isinstance(frame_id, (bool, np.bool_)) or not isinstance(
        frame_id, (int, np.integer)
    ):
        raise ValueError("frame_id must be a non-negative integer")
    result = int(frame_id)
    if result < 0:
        raise ValueError("frame_id must be a non-negative integer")
    return result


def validate_request_id(request_id: Any) -> str:
    value = str(request_id).strip()
    if not value:
        raise ValueError("request_id cannot be empty")
    if len(value) > 128:
        raise ValueError("request_id is longer than 128 characters")
    return value


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


def _frame_header(
    operation: str,
    image: np.ndarray,
    frame_id: int,
    request_id: str,
) -> Dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "op": operation,
        "request_id": validate_request_id(request_id),
        "frame_id": validate_frame_id(frame_id),
        "image": {
            "shape": [int(value) for value in image.shape],
            "dtype": "uint8",
            "color_space": "BGR",
        },
    }


def _shape_tuple(value: Any, *, dimensions: int, name: str) -> Tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"invalid {name} shape {value!r}")
    if any(
        isinstance(item, (bool, np.bool_))
        or not isinstance(item, (int, np.integer))
        for item in value
    ):
        raise ValueError(f"invalid {name} shape {value!r}")
    shape = tuple(int(item) for item in value)
    if len(shape) != dimensions or any(item <= 0 for item in shape):
        raise ValueError(f"invalid {name} shape {shape!r}")
    return shape


def _finite_float(name: str, value: Any) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_response_header(
    metadata: Dict[str, Any],
    *,
    expected_request_id: Optional[str],
) -> None:
    if int(metadata.get("version", -1)) != PROTOCOL_VERSION:
        raise SAM2VideoServiceError(
            f"SAM2 video protocol mismatch: {metadata.get('version')!r}"
        )
    if metadata.get("status") != "ok":
        raise SAM2VideoServiceError(
            str(metadata.get("message", "SAM2 video service failed"))
        )
    if expected_request_id is not None and metadata.get("request_id") != expected_request_id:
        raise SAM2VideoServiceError("SAM2 video response request_id mismatch")
