from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, Optional, Sequence

import numpy as np
import zmq

from dynamic_pcd.segmentation.sam2_video_protocol import (
    SAM2VideoResult,
    SAM2VideoServiceError,
    decode_control_response,
    decode_result_parts,
    make_control_request,
    make_initialize_box_request,
    make_initialize_request,
    make_track_request,
    validate_bgr_image,
    validate_frame_id,
    validate_mask,
    validate_request_id,
)


class ZMQSAM2VideoClient:
    """Synchronous, model-free client for the stateful SAM2 worker.

    A frame RPC timeout or malformed response invalidates the local session
    and returns a fresh all-zero result with ``valid=False``.  The caller never
    receives an old mask as if it belonged to the current frame.
    """

    def __init__(
        self,
        addr: str = "tcp://127.0.0.1:5558",
        timeout_ms: int = 2000,
        context: Optional[zmq.Context] = None,
    ):
        self.addr = str(addr)
        self.timeout_ms = max(1, int(timeout_ms))
        self.context = context if context is not None else zmq.Context.instance()
        self._socket: Optional[zmq.Socket] = None
        self._closed = False
        self._lock = threading.Lock()
        self._shape_hw: Optional[tuple] = None
        self._last_frame_id: Optional[int] = None
        self._connect()

    def _connect(self) -> None:
        if self._closed:
            raise RuntimeError("SAM2 video client is closed")
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

    def health(self, timeout_ms: Optional[int] = None) -> Dict[str, Any]:
        request_id = uuid.uuid4().hex
        parts = self._round_trip(
            make_control_request("health", request_id), timeout_ms=timeout_ms
        )
        metadata = decode_control_response(parts, expected_request_id=request_id)
        if metadata.get("service") != "dynamic-pcd-online-sam2":
            raise SAM2VideoServiceError(
                "unexpected service identity at SAM2 video address: "
                f"{metadata.get('service')!r}"
            )
        if int(metadata.get("protocol_version", -1)) != 1:
            raise SAM2VideoServiceError(
                "SAM2 video service protocol identity is missing or invalid"
            )
        return metadata

    def initialize(
        self,
        image_bgr: np.ndarray,
        mask: np.ndarray,
        frame_id: int,
        *,
        timeout_ms: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> SAM2VideoResult:
        image = validate_bgr_image(image_bgr)
        mask_u8 = validate_mask(
            mask, expected_shape=image.shape[:2], require_nonempty=True
        )
        frame = validate_frame_id(frame_id)
        request = self._request_id(request_id)
        try:
            response = self._round_trip(
                make_initialize_request(image, mask_u8, frame, request),
                timeout_ms=timeout_ms,
            )
            result = decode_result_parts(
                response,
                expected_request_id=request,
                expected_frame_id=frame,
                expected_shape=image.shape[:2],
            )
        except (TimeoutError, SAM2VideoServiceError, ValueError) as exc:
            self._shape_hw = None
            self._last_frame_id = None
            return self._invalid_result(image.shape[:2], frame, request, exc)
        self._shape_hw = tuple(image.shape[:2])
        self._last_frame_id = frame
        return result

    def initialize_box(
        self,
        image_bgr: np.ndarray,
        bbox_xyxy: np.ndarray,
        frame_id: int,
        *,
        timeout_ms: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> SAM2VideoResult:
        """Initialize SAM2 temporal memory from a pixel-space box prompt."""

        image = validate_bgr_image(image_bgr)
        frame = validate_frame_id(frame_id)
        request = self._request_id(request_id)
        try:
            response = self._round_trip(
                make_initialize_box_request(
                    image, bbox_xyxy, frame, request
                ),
                timeout_ms=timeout_ms,
            )
            result = decode_result_parts(
                response,
                expected_request_id=request,
                expected_frame_id=frame,
                expected_shape=image.shape[:2],
            )
        except (TimeoutError, SAM2VideoServiceError, ValueError) as exc:
            self._shape_hw = None
            self._last_frame_id = None
            return self._invalid_result(image.shape[:2], frame, request, exc)
        self._shape_hw = tuple(image.shape[:2])
        self._last_frame_id = frame
        return result

    def track(
        self,
        image_bgr: np.ndarray,
        frame_id: int,
        *,
        timeout_ms: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> SAM2VideoResult:
        image = validate_bgr_image(image_bgr)
        frame = validate_frame_id(frame_id)
        if self._shape_hw is None or self._last_frame_id is None:
            raise RuntimeError("SAM2 video client must be initialized before track")
        if tuple(image.shape[:2]) != self._shape_hw:
            raise ValueError(
                f"track image shape {image.shape[:2]} does not match initialized "
                f"shape {self._shape_hw}"
            )
        if frame <= self._last_frame_id:
            raise ValueError(
                f"track frame_id {frame} must be greater than {self._last_frame_id}"
            )
        request = self._request_id(request_id)
        try:
            response = self._round_trip(
                make_track_request(image, frame, request), timeout_ms=timeout_ms
            )
            result = decode_result_parts(
                response,
                expected_request_id=request,
                expected_frame_id=frame,
                expected_shape=self._shape_hw,
            )
        except (TimeoutError, SAM2VideoServiceError, ValueError) as exc:
            shape_hw = self._shape_hw
            self._shape_hw = None
            self._last_frame_id = None
            return self._invalid_result(shape_hw, frame, request, exc)
        self._last_frame_id = frame
        return result

    def reset(
        self,
        *,
        timeout_ms: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        request = self._request_id(request_id)
        try:
            response = self._round_trip(
                make_control_request("reset", request), timeout_ms=timeout_ms
            )
            return decode_control_response(
                response, expected_request_id=request
            )
        finally:
            # A lost reset response is ambiguous: the service may already have
            # cleared its state.  Never keep tracking the previous session.
            self._shape_hw = None
            self._last_frame_id = None

    def _round_trip(
        self,
        parts: Sequence[bytes],
        *,
        timeout_ms: Optional[int],
    ) -> Sequence[bytes]:
        timeout = self.timeout_ms if timeout_ms is None else max(1, int(timeout_ms))
        with self._lock:
            if self._closed or self._socket is None:
                raise RuntimeError("SAM2 video client is closed")
            try:
                self._socket.send_multipart(list(parts))
                if not self._socket.poll(timeout=timeout, flags=zmq.POLLIN):
                    self._reset_socket()
                    raise TimeoutError(
                        f"SAM2 video service at {self.addr} did not respond "
                        f"within {timeout} ms"
                    )
                return self._socket.recv_multipart()
            except TimeoutError:
                raise
            except (zmq.ZMQError, ValueError, SAM2VideoServiceError) as exc:
                self._reset_socket()
                if isinstance(exc, SAM2VideoServiceError):
                    raise
                raise SAM2VideoServiceError(
                    f"SAM2 video ZMQ request failed: {exc}"
                ) from exc

    @staticmethod
    def _request_id(value: Optional[str]) -> str:
        request_id = uuid.uuid4().hex if value is None else str(value).strip()
        return validate_request_id(request_id)

    @staticmethod
    def _invalid_result(
        shape_hw,
        frame_id: int,
        request_id: str,
        error: BaseException,
    ) -> SAM2VideoResult:
        return SAM2VideoResult(
            mask=np.zeros(tuple(int(value) for value in shape_hw), dtype=np.uint8),
            valid=False,
            frame_id=int(frame_id),
            message=f"SAM2 video RPC failed closed: {type(error).__name__}: {error}",
            request_id=str(request_id),
            internal_frame_idx=-1,
            mask_area=0,
        )

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._shape_hw = None
            self._last_frame_id = None
            if self._socket is not None:
                self._socket.close(linger=0)
                self._socket = None

    def __enter__(self) -> "ZMQSAM2VideoClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
