from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, Optional, Sequence

import numpy as np
import zmq

from dynamic_pcd.segmentation.prompt_protocol import (
    PROTOCOL_VERSION,
    PromptSegmentationResult,
    PromptServiceError,
    decode_json,
    encode_json,
    make_segment_request,
)


class ZMQPromptSegmentationClient:
    """Model-free Python 3.9 client for the external prompt service.

    The service is intentionally a separate process because Grounding DINO and
    SAM2 use a newer, heavier Python environment than the camera loop.  Images
    and masks are sent as raw contiguous uint8 arrays, so the client has no
    Torch, Transformers, Pillow, or SAM2 dependency.
    """

    def __init__(
        self,
        addr: str = "tcp://127.0.0.1:5557",
        timeout_ms: int = 15000,
        context: Optional[zmq.Context] = None,
    ):
        self.addr = str(addr)
        self.timeout_ms = max(1, int(timeout_ms))
        self.context = context if context is not None else zmq.Context.instance()
        self._socket: Optional[zmq.Socket] = None
        self._closed = False
        self._lock = threading.Lock()
        self._connect()

    def _connect(self) -> None:
        if self._closed:
            raise RuntimeError("prompt segmentation client is closed")
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDHWM, 1)
        socket.setsockopt(zmq.RCVHWM, 1)
        socket.connect(self.addr)
        self._socket = socket

    def _reset_socket(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = None
        if not self._closed:
            self._connect()

    def segment(
        self,
        image_bgr: np.ndarray,
        prompt: str,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        mask_threshold: float = 0.50,
        reference_bbox_xyxy: Optional[Sequence[float]] = None,
        top_k: int = 5,
        timeout_ms: Optional[int] = None,
        request_id: Optional[str] = None,
        frame_id: Optional[int] = None,
        frame_timestamp: Optional[float] = None,
        frame_metadata: Optional[Dict[str, Any]] = None,
        previous_bbox_xyxy: Optional[Sequence[float]] = None,
    ) -> PromptSegmentationResult:
        if reference_bbox_xyxy is not None and previous_bbox_xyxy is not None:
            raise ValueError(
                "provide reference_bbox_xyxy or previous_bbox_xyxy, not both"
            )
        reference_bbox = (
            reference_bbox_xyxy
            if reference_bbox_xyxy is not None
            else previous_bbox_xyxy
        )
        request_id_value = uuid.uuid4().hex if request_id is None else str(request_id)
        if not request_id_value:
            raise ValueError("request_id cannot be empty")
        request_parts = make_segment_request(
            image_bgr=image_bgr,
            prompt=prompt,
            request_id=request_id_value,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            mask_threshold=mask_threshold,
            reference_bbox_xyxy=reference_bbox,
            top_k=top_k,
            frame_id=frame_id,
            frame_timestamp=frame_timestamp,
            frame_metadata=frame_metadata,
        )
        response_parts = self._round_trip(request_parts, timeout_ms=timeout_ms)
        if len(response_parts) != 2:
            raise PromptServiceError(
                f"prompt service returned {len(response_parts)} parts; expected 2"
            )
        metadata = decode_json(response_parts[0])
        if int(metadata.get("version", -1)) != PROTOCOL_VERSION:
            raise PromptServiceError(
                f"prompt service protocol mismatch: {metadata.get('version')!r}"
            )
        if metadata.get("status") != "ok":
            raise PromptServiceError(
                str(metadata.get("message", "prompt segmentation service failed"))
            )
        if metadata.get("request_id") != request_id_value:
            raise PromptServiceError(
                "prompt service response request_id does not match the request"
            )
        return PromptSegmentationResult.from_wire(metadata, response_parts[1])

    def health(self, timeout_ms: Optional[int] = None) -> Dict[str, Any]:
        request_id = uuid.uuid4().hex
        request = encode_json(
            {
                "version": PROTOCOL_VERSION,
                "op": "health",
                "request_id": request_id,
            }
        )
        response_parts = self._round_trip((request,), timeout_ms=timeout_ms)
        if len(response_parts) != 1:
            raise PromptServiceError("invalid health response")
        metadata = decode_json(response_parts[0])
        if metadata.get("status") != "ok":
            raise PromptServiceError(str(metadata.get("message", "health check failed")))
        if metadata.get("request_id") != request_id:
            raise PromptServiceError("health response request_id mismatch")
        return metadata

    def _round_trip(
        self, parts: Sequence[bytes], timeout_ms: Optional[int]
    ) -> Sequence[bytes]:
        timeout = self.timeout_ms if timeout_ms is None else max(1, int(timeout_ms))
        with self._lock:
            if self._closed or self._socket is None:
                raise RuntimeError("prompt segmentation client is closed")
            try:
                self._socket.send_multipart(list(parts))
                if not self._socket.poll(timeout=timeout, flags=zmq.POLLIN):
                    self._reset_socket()
                    raise TimeoutError(
                        f"prompt segmentation service at {self.addr} did not respond "
                        f"within {timeout} ms"
                    )
                return self._socket.recv_multipart()
            except TimeoutError:
                raise
            except zmq.ZMQError as exc:
                self._reset_socket()
                raise PromptServiceError(
                    f"prompt segmentation ZMQ request failed: {exc}"
                ) from exc

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._socket is not None:
                self._socket.close(linger=0)
                self._socket = None

    def __enter__(self) -> "ZMQPromptSegmentationClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
