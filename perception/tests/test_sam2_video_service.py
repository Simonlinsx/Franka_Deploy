from __future__ import annotations

import contextlib
import json
import sys
import threading
import types
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import zmq

import dynamic_pcd.apps.sam2_video_service as service_module
from dynamic_pcd.apps.sam2_video_service import SAM2VideoService
from dynamic_pcd.segmentation.sam2_video_backend import (
    OfficialSAM2VideoBackend,
    VOS_COMPILE_PREWARM_CONTRACT,
)
from dynamic_pcd.segmentation.sam2_video_client import ZMQSAM2VideoClient
from dynamic_pcd.segmentation.sam2_video_protocol import (
    PROTOCOL_VERSION,
    SAM2VideoResult,
    decode_request,
    decode_result_parts,
    make_control_request,
    make_initialize_box_request,
    make_error_parts,
    make_initialize_request,
    make_result_parts,
    make_track_request,
)
from dynamic_pcd.segmentation.sam2_video_runtime import (
    SAM2VideoServiceManager,
    sam2_video_service_manager_kwargs_from_config,
)


def _decode_header(payload: bytes) -> Dict[str, Any]:
    value = json.loads(payload.decode("utf-8"))
    assert isinstance(value, dict)
    return value


def _replace_header(
    parts: Sequence[bytes], update: Dict[str, Any]
) -> Tuple[bytes, ...]:
    header = _decode_header(parts[0])
    header.update(update)
    return (json.dumps(header).encode("utf-8"), *parts[1:])


def _wire_result(parts: Sequence[bytes]) -> SAM2VideoResult:
    return decode_result_parts(parts)


class FakeVideoBackend:
    """Stateful CPU-only stand-in for the persistent SAM2 video predictor."""

    def __init__(self) -> None:
        self.calls: List[Tuple[Any, ...]] = []
        self.mask: Optional[np.ndarray] = None
        self.internal_frame_idx = -1
        self.closed = False

    def health(self) -> Dict[str, Any]:
        self.calls.append(("health",))
        return {
            "backend": "fake",
            "device": "cpu",
            "initialized": self.mask is not None,
        }

    def initialize(
        self, image_bgr: np.ndarray, mask: np.ndarray, frame_id: int
    ) -> SAM2VideoResult:
        self.calls.append(
            ("initialize", image_bgr.copy(), mask.copy(), int(frame_id))
        )
        self.mask = (np.asarray(mask) != 0).astype(np.uint8)
        self.internal_frame_idx = 0
        return SAM2VideoResult(
            mask=self.mask.copy(),
            valid=bool(self.mask.any()),
            frame_id=int(frame_id),
            message="fake initialized",
            timings_ms={"fake_backend": 0.25},
            internal_frame_idx=self.internal_frame_idx,
        )

    def initialize_box(
        self, image_bgr: np.ndarray, bbox_xyxy: np.ndarray, frame_id: int
    ) -> SAM2VideoResult:
        self.calls.append(
            ("initialize_box", image_bgr.copy(), tuple(bbox_xyxy), int(frame_id))
        )
        x1, y1, x2, y2 = (int(value) for value in bbox_xyxy)
        self.mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        self.mask[y1:y2, x1:x2] = 1
        self.internal_frame_idx = 0
        return SAM2VideoResult(
            mask=self.mask.copy(),
            valid=True,
            frame_id=int(frame_id),
            message="fake box initialized",
            timings_ms={"fake_backend": 0.30},
            internal_frame_idx=0,
        )

    def track(self, image_bgr: np.ndarray, frame_id: int) -> SAM2VideoResult:
        self.calls.append(("track", image_bgr.copy(), int(frame_id)))
        if self.mask is None:
            raise RuntimeError("fake backend is not initialized")
        self.mask = np.roll(self.mask, shift=1, axis=1)
        self.internal_frame_idx += 1
        return SAM2VideoResult(
            mask=self.mask.copy(),
            valid=True,
            frame_id=int(frame_id),
            message="fake tracked",
            timings_ms={"fake_backend": 0.5},
            internal_frame_idx=self.internal_frame_idx,
        )

    def reset(self) -> None:
        self.calls.append(("reset",))
        self.mask = None
        self.internal_frame_idx = -1

    def close(self) -> None:
        self.calls.append(("close",))
        self.closed = True


def test_manager_config_contract_is_shared_with_parent_prewarm():
    kwargs = sam2_video_service_manager_kwargs_from_config(
        {
            "camera": {"width": 424, "height": 240},
            "online_sam2": {
                "service_addr": "tcp://127.0.0.1:15558",
                "service_autostart": False,
                "startup_timeout_s": 75.0,
                "request_timeout_s": 0.125,
                "checkpoint": "/tmp/sam2.pt",
                "vos_optimized": True,
                "vos_compile_mode": "max-autotune-no-cudagraphs",
            },
        }
    )

    assert kwargs["addr"] == "tcp://127.0.0.1:15558"
    assert kwargs["autostart"] is False
    assert kwargs["startup_timeout_s"] == pytest.approx(75.0)
    assert kwargs["request_timeout_ms"] == 125
    launcher_args = kwargs["launcher_args"]
    assert launcher_args[launcher_args.index("--checkpoint") + 1] == (
        "/tmp/sam2.pt"
    )
    assert launcher_args[launcher_args.index("--prewarm-input-width") + 1] == (
        "424"
    )
    assert launcher_args[launcher_args.index("--prewarm-input-height") + 1] == (
        "240"
    )


def test_initialize_wire_round_trip_preserves_arrays_and_frame_metadata():
    base = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    image = base[:, ::-1, :]  # Exercise C-order serialization of a view.
    mask = np.zeros((5, 7), dtype=bool)
    mask[1:4, 2:6] = True

    parts = make_initialize_request(
        image_bgr=image,
        mask=mask[:, ::-1],
        frame_id=np.int64(17),
        request_id="init-17",
    )

    assert len(parts) == 3
    wire_header = _decode_header(parts[0])
    assert wire_header == {
        "version": PROTOCOL_VERSION,
        "op": "initialize",
        "request_id": "init-17",
        "frame_id": 17,
        "image": {
            "shape": [5, 7, 3],
            "dtype": "uint8",
            "color_space": "BGR",
        },
        "mask": {"shape": [5, 7], "dtype": "uint8"},
    }

    header, decoded_image, decoded_mask = decode_request(parts)
    assert header["frame_id"] == 17
    assert decoded_image.dtype == np.uint8
    assert decoded_image.flags.c_contiguous
    np.testing.assert_array_equal(decoded_image, image)
    assert decoded_mask.dtype == np.uint8
    assert decoded_mask.flags.c_contiguous
    np.testing.assert_array_equal(decoded_mask, mask[:, ::-1].astype(np.uint8))


def test_initialize_box_wire_round_trip_preserves_pixel_prompt():
    image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    parts = make_initialize_box_request(
        image, np.asarray([1, 2, 7, 6]), frame_id=18, request_id="box-18"
    )

    assert len(parts) == 2
    header, decoded_image, decoded_mask = decode_request(parts)
    assert header["op"] == "initialize_box"
    assert header["bbox_xyxy"] == [1, 2, 7, 6]
    assert header["frame_id"] == 18
    np.testing.assert_array_equal(decoded_image, image)
    assert decoded_mask is None

    for invalid in (
        [-1, 0, 3, 3],
        [0, 0, 9, 3],
        [2, 2, 2, 4],
        [0.5, 0, 2, 2],
    ):
        with pytest.raises((TypeError, ValueError)):
            make_initialize_box_request(
                image, np.asarray(invalid), frame_id=1, request_id="bad-box"
            )


def test_service_dispatches_native_box_initialization():
    backend = FakeVideoBackend()
    service = SAM2VideoService(backend)
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    response = service.handle_request(
        make_initialize_box_request(image, [1, 2, 7, 5], 40, "box-direct")
    )
    result = _wire_result(response)
    assert result.valid
    assert result.frame_id == 40
    assert result.mask.sum() == 18
    assert backend.calls[0][0] == "initialize_box"
    service.close()


def test_track_and_control_wire_shapes_are_unambiguous():
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    track_parts = make_track_request(
        image_bgr=image, frame_id=21, request_id="track-21"
    )
    assert len(track_parts) == 2
    header, decoded_image, decoded_mask = decode_request(track_parts)
    assert header["op"] == "track"
    assert header["request_id"] == "track-21"
    assert header["frame_id"] == 21
    np.testing.assert_array_equal(decoded_image, image)
    assert decoded_mask is None

    for operation in ("health", "reset"):
        parts = make_control_request(operation, request_id=operation + "-1")
        assert len(parts) == 1
        header, decoded_image, decoded_mask = decode_request(parts)
        assert header["op"] == operation
        assert header["request_id"] == operation + "-1"
        assert decoded_image is None
        assert decoded_mask is None

    with pytest.raises(ValueError):
        make_control_request("initialize", request_id="wrong-control-op")


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((3, 4, 3), dtype=np.float32),
        np.zeros((3, 4), dtype=np.uint8),
        np.zeros((3, 4, 4), dtype=np.uint8),
        np.zeros((0, 4, 3), dtype=np.uint8),
    ],
)
def test_protocol_rejects_non_uint8_hwc3_images(image: np.ndarray):
    with pytest.raises(ValueError):
        make_track_request(image, frame_id=0, request_id="bad-image")


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((3, 4), dtype=np.float32),
        np.zeros((3, 4, 1), dtype=np.uint8),
        np.zeros((2, 4), dtype=np.uint8),
        np.zeros((3, 5), dtype=bool),
    ],
)
def test_initialize_rejects_invalid_or_image_mismatched_masks(mask: np.ndarray):
    with pytest.raises(ValueError):
        make_initialize_request(
            np.zeros((3, 4, 3), dtype=np.uint8),
            mask,
            frame_id=0,
            request_id="bad-mask",
        )


@pytest.mark.parametrize("frame_id", [-1, True, 1.0, 1.5, "1", None])
def test_protocol_requires_a_non_negative_integer_frame_id(frame_id: Any):
    with pytest.raises((TypeError, ValueError)):
        make_track_request(
            np.zeros((3, 4, 3), dtype=np.uint8),
            frame_id=frame_id,
            request_id="bad-frame",
        )


def test_decoder_rejects_tampered_dtype_shape_frame_and_payload():
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    mask = np.ones((3, 4), dtype=np.uint8)
    good = make_initialize_request(image, mask, frame_id=3, request_id="init-3")

    header = _decode_header(good[0])
    header["image"]["dtype"] = "float32"
    with pytest.raises(ValueError):
        decode_request((json.dumps(header).encode(), *good[1:]))

    header = _decode_header(good[0])
    header["mask"]["shape"] = [2, 4]
    with pytest.raises(ValueError):
        decode_request((json.dumps(header).encode(), *good[1:]))

    with pytest.raises(ValueError):
        decode_request((good[0], good[1][:-1], good[2]))
    with pytest.raises(ValueError):
        decode_request((good[0], good[1], good[2][:-1]))
    with pytest.raises(ValueError):
        decode_request((*good, b"unexpected"))

    for bad_frame_id in (-1, True, 3.0, "3"):
        with pytest.raises((TypeError, ValueError)):
            decode_request(_replace_header(good, {"frame_id": bad_frame_id}))

    with pytest.raises(ValueError):
        decode_request(_replace_header(good, {"version": PROTOCOL_VERSION + 1}))
    with pytest.raises(ValueError):
        decode_request(good, max_image_bytes=image.nbytes - 1)


def test_result_wire_round_trip_validates_mask_dtype_shape_and_frame():
    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[1:3, 2:5] = 1
    result = SAM2VideoResult(
        mask=mask,
        valid=True,
        frame_id=9,
        message="tracked",
        timings_ms={"inference": 1.25, "total": 1.5},
        request_id="track-9",
        internal_frame_idx=2,
    )

    parts = make_result_parts(result)
    metadata = _decode_header(parts[0])
    assert metadata["status"] == "ok"
    assert metadata["mask"] == {"shape": [4, 6], "dtype": "uint8"}
    decoded = decode_result_parts(parts)
    np.testing.assert_array_equal(decoded.mask, mask)
    assert decoded.valid
    assert decoded.frame_id == 9
    assert decoded.internal_frame_idx == 2
    assert decoded.request_id == "track-9"
    assert decoded.timings_ms == {"inference": 1.25, "total": 1.5}

    bad_metadata = dict(metadata)
    bad_metadata["mask"] = {"shape": [4, 6], "dtype": "float32"}
    with pytest.raises((ValueError, RuntimeError)):
        decode_result_parts((json.dumps(bad_metadata).encode(), parts[1]))

    bad_metadata = dict(metadata)
    bad_metadata["mask"] = {"shape": [4, 5], "dtype": "uint8"}
    with pytest.raises((ValueError, RuntimeError)):
        decode_result_parts((json.dumps(bad_metadata).encode(), parts[1]))

    bad_metadata = dict(metadata)
    bad_metadata["frame_id"] = -1
    with pytest.raises((ValueError, RuntimeError)):
        decode_result_parts((json.dumps(bad_metadata).encode(), parts[1]))


def test_service_handler_round_trip_uses_only_the_injected_fake_backend():
    backend = FakeVideoBackend()
    service = SAM2VideoService(backend)
    image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    seed = np.zeros((6, 8), dtype=bool)
    seed[2:5, 1:4] = True

    health_response = service.handle_request(
        make_control_request("health", request_id="health-direct")
    )
    health = _decode_header(health_response[0])
    assert len(health_response) == 1
    assert health["status"] == "ok"
    assert health["request_id"] == "health-direct"
    assert health["backend"] == "fake"
    assert health["device"] == "cpu"

    init_response = service.handle_request(
        make_initialize_request(
            image, seed, frame_id=40, request_id="init-direct"
        )
    )
    initialized = _wire_result(init_response)
    assert initialized.valid
    assert initialized.request_id == "init-direct"
    assert initialized.frame_id == 40
    assert initialized.internal_frame_idx == 0
    assert initialized.timings_ms["fake_backend"] == 0.25
    np.testing.assert_array_equal(initialized.mask, seed.astype(np.uint8))

    tracked_response = service.handle_request(
        make_track_request(image, frame_id=43, request_id="track-direct")
    )
    tracked = _wire_result(tracked_response)
    assert tracked.valid
    assert tracked.request_id == "track-direct"
    assert tracked.frame_id == 43
    assert tracked.internal_frame_idx == 1
    assert tracked.timings_ms["fake_backend"] == 0.5
    np.testing.assert_array_equal(
        tracked.mask, np.roll(seed.astype(np.uint8), shift=1, axis=1)
    )

    reset_response = service.handle_request(
        make_control_request("reset", request_id="reset-direct")
    )
    reset = _decode_header(reset_response[0])
    assert len(reset_response) == 1
    assert reset["status"] == "ok"
    assert reset["request_id"] == "reset-direct"
    assert backend.mask is None

    service.close()
    assert backend.closed
    assert [call[0] for call in backend.calls] == [
        "health",
        "initialize",
        "track",
        "reset",
        "close",
    ]
    init_call = backend.calls[1]
    np.testing.assert_array_equal(init_call[1], image)
    np.testing.assert_array_equal(init_call[2], seed.astype(np.uint8))


def test_service_enforces_initialization_shape_and_strict_frame_order():
    backend = FakeVideoBackend()
    service = SAM2VideoService(backend)
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    seed = np.ones((5, 7), dtype=np.uint8)

    before_init = service.handle_request(
        make_track_request(image, frame_id=1, request_id="too-early")
    )
    assert _decode_header(before_init[0])["status"] == "error"

    assert _decode_header(
        service.handle_request(
            make_initialize_request(image, seed, 10, "init-order")
        )[0]
    )["status"] == "ok"

    duplicate = service.handle_request(
        make_track_request(image, frame_id=10, request_id="duplicate")
    )
    duplicate_meta = _decode_header(duplicate[0])
    assert duplicate_meta["status"] == "error"
    assert duplicate_meta["request_id"] == "duplicate"

    older = service.handle_request(
        make_track_request(image, frame_id=9, request_id="older")
    )
    assert _decode_header(older[0])["status"] == "error"

    wrong_shape = service.handle_request(
        make_track_request(
            np.zeros((4, 7, 3), dtype=np.uint8),
            frame_id=11,
            request_id="wrong-shape",
        )
    )
    assert _decode_header(wrong_shape[0])["status"] == "error"

    valid_skip = service.handle_request(
        make_track_request(image, frame_id=12, request_id="valid-skip")
    )
    assert _decode_header(valid_skip[0])["status"] == "ok"

    service.handle_request(make_control_request("reset", "reset-order"))
    after_reset = service.handle_request(
        make_track_request(image, frame_id=13, request_id="after-reset")
    )
    assert _decode_header(after_reset[0])["status"] == "error"

    assert [call[0] for call in backend.calls].count("track") == 1
    service.close()


def test_client_and_service_round_trip_over_inproc_zmq_with_fake_backend():
    backend = FakeVideoBackend()
    service = SAM2VideoService(backend)
    context = zmq.Context()
    address = "inproc://sam2-video-" + uuid.uuid4().hex
    ready = threading.Event()
    server_errors: List[BaseException] = []

    def serve_four_requests() -> None:
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.bind(address)
            ready.set()
            for _ in range(4):
                if not socket.poll(timeout=2000, flags=zmq.POLLIN):
                    raise TimeoutError("test server did not receive four requests")
                response = service.handle_request(socket.recv_multipart())
                socket.send_multipart(list(response))
        except BaseException as exc:  # Propagate worker failures in the test.
            server_errors.append(exc)
        finally:
            socket.close(linger=0)
            service.close()

    server = threading.Thread(target=serve_four_requests, daemon=True)
    server.start()
    assert ready.wait(timeout=2.0)
    client = ZMQSAM2VideoClient(
        addr=address, timeout_ms=1000, context=context
    )
    image = np.zeros((5, 9, 3), dtype=np.uint8)
    seed = np.zeros((5, 9), dtype=np.uint8)
    seed[1:4, 2:5] = 1

    try:
        health = client.health()
        assert health["status"] == "ok"
        assert health["backend"] == "fake"

        initialized = client.initialize(image, seed, frame_id=100)
        assert initialized.valid
        assert initialized.frame_id == 100
        np.testing.assert_array_equal(initialized.mask, seed)

        tracked = client.track(image, frame_id=104)
        assert tracked.valid
        assert tracked.frame_id == 104
        np.testing.assert_array_equal(
            tracked.mask, np.roll(seed, shift=1, axis=1)
        )

        reset = client.reset()
        assert reset["status"] == "ok"
    finally:
        client.close()
        client.close()  # Closing an IPC resource should be idempotent.
        server.join(timeout=3.0)
        context.term()

    assert not server.is_alive()
    assert server_errors == []
    assert backend.closed


@pytest.mark.parametrize(
    "failure_mode",
    ["timeout", "service_error", "bad_json", "request_id", "frame_id", "shape"],
)
def test_client_tracking_failures_return_a_fresh_empty_invalid_mask(
    monkeypatch: pytest.MonkeyPatch, failure_mode: str
):
    """No transport failure may expose or retain the previous valid mask."""

    context = zmq.Context()
    client = ZMQSAM2VideoClient(
        addr="inproc://unused-" + uuid.uuid4().hex,
        timeout_ms=5,
        context=context,
    )
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    seed = np.ones((4, 6), dtype=np.uint8)
    call_count = 0

    def scripted_round_trip(
        parts: Sequence[bytes], timeout_ms: Optional[int] = None
    ) -> Sequence[bytes]:
        nonlocal call_count
        call_count += 1
        header, request_image, request_mask = decode_request(parts)
        if call_count == 1:
            assert header["op"] == "initialize"
            return make_result_parts(
                SAM2VideoResult(
                    mask=request_mask.copy(),
                    valid=True,
                    frame_id=header["frame_id"],
                    message="valid seed",
                    timings_ms={},
                    request_id=header["request_id"],
                    internal_frame_idx=0,
                )
            )

        assert header["op"] == "track"
        if failure_mode == "timeout":
            raise TimeoutError("scripted timeout")
        if failure_mode == "service_error":
            return make_error_parts("scripted backend failure", header["request_id"])
        if failure_mode == "bad_json":
            return (b"not-json", b"")

        response_frame_id = header["frame_id"]
        response_request_id = header["request_id"]
        response_mask = np.ones(request_image.shape[:2], dtype=np.uint8)
        if failure_mode == "request_id":
            response_request_id = "somebody-elses-request"
        elif failure_mode == "frame_id":
            response_frame_id -= 1
        elif failure_mode == "shape":
            response_mask = np.ones((3, 6), dtype=np.uint8)
        return make_result_parts(
            SAM2VideoResult(
                mask=response_mask,
                valid=True,
                frame_id=response_frame_id,
                message="malformed success",
                timings_ms={},
                request_id=response_request_id,
                internal_frame_idx=1,
            )
        )

    monkeypatch.setattr(client, "_round_trip", scripted_round_trip)
    try:
        initialized = client.initialize(image, seed, frame_id=1)
        assert initialized.valid
        assert initialized.mask.sum() == seed.size

        failed = client.track(image, frame_id=2)
        assert not failed.valid
        assert failed.frame_id == 2
        assert failed.mask.dtype == np.uint8
        assert failed.mask.shape == image.shape[:2]
        assert failed.mask.sum() == 0
        assert failed.mask is not initialized.mask
        assert failed.message
        with pytest.raises(RuntimeError, match="initialized"):
            client.track(image, frame_id=3)
        assert call_count == 2
    finally:
        client.close()
        context.term()


def test_client_initialize_is_also_fail_closed_on_transport_error(
    monkeypatch: pytest.MonkeyPatch,
):
    context = zmq.Context()
    client = ZMQSAM2VideoClient(
        addr="inproc://unused-" + uuid.uuid4().hex,
        timeout_ms=5,
        context=context,
    )

    def timeout(
        _parts: Sequence[bytes], timeout_ms: Optional[int] = None
    ) -> Sequence[bytes]:
        raise TimeoutError("offline")

    monkeypatch.setattr(client, "_round_trip", timeout)
    image = np.zeros((3, 5, 3), dtype=np.uint8)
    mask = np.ones((3, 5), dtype=np.uint8)
    try:
        result = client.initialize(image, mask, frame_id=0)
        assert not result.valid
        assert result.frame_id == 0
        assert result.mask.shape == (3, 5)
        assert result.mask.dtype == np.uint8
        assert result.mask.sum() == 0
        assert "offline" in result.message
        with pytest.raises(RuntimeError, match="initialized"):
            client.track(image, frame_id=1)
    finally:
        client.close()
        context.term()


def test_backend_history_window_and_pruning_keep_only_safe_recent_outputs():
    predictor = type(
        "FakePredictor",
        (),
        {
            "memory_temporal_stride_for_eval": 1,
            "num_maskmem": 7,
            "max_obj_ptrs_in_encoder": 16,
        },
    )()
    backend = OfficialSAM2VideoBackend(reset_every_frames=0)
    assert backend._history_window(predictor) == 16
    backend._safe_history_frames = 16
    sentinel = object()
    backend.state = {
        "images": [None] * 24 + [sentinel],
        "output_dict_per_obj": {
            0: {
                "cond_frame_outputs": {0: {"condition": True}},
                "non_cond_frame_outputs": {
                    frame: {"frame": frame} for frame in range(1, 25)
                },
            }
        },
        "frames_tracked_per_obj": {
            0: {frame: {"reverse": False} for frame in range(25)}
        },
    }

    backend._prune_state(current_idx=24)

    output = backend.state["output_dict_per_obj"][0]
    assert list(output["cond_frame_outputs"]) == [0]
    assert list(output["non_cond_frame_outputs"]) == list(range(9, 25))
    assert list(backend.state["frames_tracked_per_obj"][0]) == list(range(9, 25))
    assert backend.state["images"][23] is None
    assert backend.state["images"][24] is sentinel


def test_backend_health_reports_hashes_of_loaded_model_assets_without_path_trust():
    backend = OfficialSAM2VideoBackend()
    backend._checkpoint_sha256 = "a" * 64
    backend._model_config_sha256 = "b" * 64
    backend._resolved_model_config_path = "/resolved/model.yaml"

    health = backend.health()

    assert health["checkpoint_sha256"] == "a" * 64
    assert health["model_config_sha256"] == "b" * 64
    assert health["resolved_model_config_path"] == "/resolved/model.yaml"
    assert health["vos_optimized"] is True


@pytest.mark.parametrize("value", [True, False])
def test_backend_health_exposes_explicit_vos_optimized_mode(value: bool):
    backend = OfficialSAM2VideoBackend(vos_optimized=value)

    assert backend.vos_optimized is value
    assert backend.health()["vos_optimized"] is value


def test_diagnostic_cudagraph_mode_is_explicit_in_component_provenance():
    backend = OfficialSAM2VideoBackend(
        vos_optimized=True, vos_compile_mode="max-autotune"
    )

    health = backend.health()

    assert health["vos_compile_cuda_graphs"] is True
    assert set(health["vos_component_compile_modes"].values()) == {
        "max-autotune"
    }


@pytest.mark.parametrize("value", [None, 0, 1, "true", np.bool_(True)])
def test_backend_rejects_non_boolean_vos_optimized(value):
    with pytest.raises(ValueError, match="vos_optimized must be true or false"):
        OfficialSAM2VideoBackend(vos_optimized=value)


@pytest.mark.parametrize("value", [None, "", "reduce-overhead", 1])
def test_backend_rejects_unknown_vos_compile_mode(value):
    with pytest.raises(ValueError, match="vos_compile_mode must be one of"):
        OfficialSAM2VideoBackend(vos_compile_mode=value)


@pytest.mark.parametrize("value", [True, False])
def test_backend_load_forwards_vos_optimized_to_official_builder(
    monkeypatch, tmp_path, value: bool
):
    checkpoint = tmp_path / "sam2.pt"
    checkpoint.write_bytes(b"checkpoint")
    package_root = tmp_path / "sam2"
    model_config = package_root / "configs" / "model.yaml"
    model_config.parent.mkdir(parents=True)
    model_config.write_text("model: {}\n", encoding="utf-8")
    package_init = package_root / "__init__.py"
    package_init.write_text("", encoding="utf-8")

    calls = []
    compile_calls = []

    def compiled_forward_stub(*_args, **_kwargs):
        return None

    def original_forward_stub(*_args, **_kwargs):
        return None

    compiled_forward_stub._torchdynamo_orig_callable = original_forward_stub

    def fake_component():
        return types.SimpleNamespace(forward=compiled_forward_stub)

    memory_attention = fake_component()
    memory_attention.num_layers = 2
    memory_attention.layers = [
        types.SimpleNamespace(
            self_attn=types.SimpleNamespace(
                freqs_cis=np.empty((32 * 32, 128), dtype=np.complex64)
            ),
            cross_attn_image=types.SimpleNamespace(
                freqs_cis=np.empty((32 * 32, 128), dtype=np.complex64)
            ),
        )
        for _ in range(memory_attention.num_layers)
    ]

    predictor = types.SimpleNamespace(
        fill_hole_area=-1,
        memory_temporal_stride_for_eval=1,
        num_maskmem=7,
        max_obj_ptrs_in_encoder=16,
        image_encoder=fake_component(),
        memory_encoder=fake_component(),
        memory_attention=memory_attention,
        sam_prompt_encoder=fake_component(),
        sam_mask_decoder=fake_component(),
    )

    def fake_build(model_cfg, checkpoint_path, **kwargs):
        calls.append((model_cfg, checkpoint_path, dict(kwargs)))
        return predictor

    torch_module = types.ModuleType("torch")
    torch_module.__path__ = []
    torch_module.bfloat16 = object()
    torch_module.float16 = object()
    torch_module.backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            matmul=types.SimpleNamespace(allow_tf32=False)
        ),
        cudnn=types.SimpleNamespace(allow_tf32=False),
    )
    torch_module.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        synchronize=lambda _device: None,
    )
    torch_module.device = lambda _name: types.SimpleNamespace(
        type="cuda", index=None
    )

    def fake_compile(original, **kwargs):
        compile_calls.append((original, dict(kwargs)))
        wrapper = lambda *_args, **_kwargs: None
        wrapper._torchdynamo_orig_callable = original
        return wrapper

    torch_module.compile = fake_compile
    torch_nn = types.ModuleType("torch.nn")
    torch_nn.__path__ = []
    torch_functional = types.ModuleType("torch.nn.functional")
    torch_nn.functional = torch_functional
    torch_module.nn = torch_nn

    sam2_module = types.ModuleType("sam2")
    sam2_module.__path__ = [str(package_root)]
    sam2_module.__file__ = str(package_init)
    build_module = types.ModuleType("sam2.build_sam")
    build_module.build_sam2_video_predictor = fake_build
    predictor_module = types.ModuleType("sam2.sam2_video_predictor")
    sam2_module.build_sam = build_module
    sam2_module.sam2_video_predictor = predictor_module

    for name, module in (
        ("torch", torch_module),
        ("torch.nn", torch_nn),
        ("torch.nn.functional", torch_functional),
        ("sam2", sam2_module),
        ("sam2.build_sam", build_module),
        ("sam2.sam2_video_predictor", predictor_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    backend = OfficialSAM2VideoBackend(
        checkpoint=str(checkpoint),
        model_config="configs/model.yaml",
        vos_optimized=value,
    )
    backend.load()

    assert len(calls) == 1
    assert calls[0][2]["vos_optimized"] is value
    assert calls[0][2]["apply_postprocessing"] is False
    assert predictor.fill_hole_area == 0
    assert len(compile_calls) == (5 if value else 0)
    if value:
        assert [call[1]["mode"] for call in compile_calls] == [
            "max-autotune-no-cudagraphs",
            "max-autotune-no-cudagraphs",
            "max-autotune-no-cudagraphs",
            "max-autotune-no-cudagraphs",
            "max-autotune-no-cudagraphs",
        ]
        assert [call[1]["dynamic"] for call in compile_calls] == [
            False,
            False,
            True,
            False,
            False,
        ]
        assert backend.health()["vos_compile_cuda_graphs"] is False
        assert backend.health()["vos_component_compile_modes"] == {
            "image_encoder": "max-autotune-no-cudagraphs",
            "memory_encoder": "max-autotune-no-cudagraphs",
            "memory_attention": "max-autotune-no-cudagraphs",
            "sam_prompt_encoder": "max-autotune-no-cudagraphs",
            "sam_mask_decoder": "max-autotune-no-cudagraphs",
        }
        assert backend.health()["vos_component_compile_dynamic"] == {
            "image_encoder": False,
            "memory_encoder": False,
            "memory_attention": True,
            "sam_prompt_encoder": False,
            "sam_mask_decoder": False,
        }
        assert calls[0][2]["hydra_overrides_extra"] == [
            "++model.image_size=512",
            "++model.memory_attention.layer.self_attention.feat_sizes=[32,32]",
            "++model.memory_attention.layer.cross_attention.feat_sizes=[32,32]",
        ]
        assert backend.health()["vos_memory_attention_rope_grid_hw"] == [32, 32]
        assert (
            backend.health()["vos_memory_attention_rope_expected_tokens"] == 1024
        )
        assert backend.health()["vos_memory_attention_rope_cache_count"] == 4
        assert backend.health()["vos_memory_attention_rope_cache_token_counts"] == [
            1024,
            1024,
            1024,
            1024,
        ]
        assert backend.health()["vos_memory_attention_rope_caches_verified"] is True
        assert backend.health()["vos_rewrapped_components"] == [
            "image_encoder",
            "memory_encoder",
            "memory_attention",
            "sam_prompt_encoder",
            "sam_mask_decoder",
        ]


def test_vos_rope_cache_verification_fails_closed_before_compile_rewrap():
    backend = OfficialSAM2VideoBackend(image_size=512, vos_optimized=True)
    predictor = types.SimpleNamespace(
        memory_attention=types.SimpleNamespace(
            num_layers=1,
            layers=[
                types.SimpleNamespace(
                    self_attn=types.SimpleNamespace(
                        freqs_cis=np.empty((64 * 64, 128), dtype=np.complex64)
                    ),
                    cross_attn_image=types.SimpleNamespace(
                        freqs_cis=np.empty((32 * 32, 128), dtype=np.complex64)
                    ),
                )
            ],
        )
    )

    with pytest.raises(
        RuntimeError,
        match="RoPE cache token grid differs from the compiled image-token grid",
    ):
        backend._verify_vos_memory_attention_rope_caches(predictor)

    health = backend.health()
    assert health["vos_memory_attention_rope_grid_hw"] == [32, 32]
    assert health["vos_memory_attention_rope_expected_tokens"] == 1024
    assert health["vos_memory_attention_rope_cache_count"] == 0
    assert health["vos_memory_attention_rope_caches_verified"] is False


def test_vos_compile_prewarm_is_deterministic_complete_and_state_reset(
    monkeypatch,
):
    backend = OfficialSAM2VideoBackend(
        image_size=64,
        vos_optimized=True,
        prewarm_input_size_wh=(80, 48),
    )
    calls = []

    def fake_load():
        calls.append(("load",))

    def fake_initialize_box(image, bbox, frame_id):
        calls.append(
            ("initialize_box", image.copy(), np.asarray(bbox).copy(), frame_id)
        )
        return SAM2VideoResult(
            mask=np.ones(image.shape[:2], dtype=np.uint8),
            valid=True,
            frame_id=frame_id,
        )

    def fake_track(image, frame_id):
        calls.append(("track", image.copy(), frame_id))
        return SAM2VideoResult(
            mask=np.ones(image.shape[:2], dtype=np.uint8),
            valid=True,
            frame_id=frame_id,
        )

    def fake_initialize(image, mask, frame_id):
        calls.append(
            ("initialize", image.copy(), np.asarray(mask).copy(), frame_id)
        )
        return SAM2VideoResult(
            mask=np.asarray(mask, dtype=np.uint8).copy(),
            valid=True,
            frame_id=frame_id,
        )

    monkeypatch.setattr(backend, "load", fake_load)
    monkeypatch.setattr(backend, "initialize_box", fake_initialize_box)
    monkeypatch.setattr(backend, "initialize", fake_initialize)
    monkeypatch.setattr(backend, "track", fake_track)
    monkeypatch.setattr(backend, "reset", lambda: calls.append(("reset",)))

    backend.compile_prewarm()
    first_call_count = len(calls)
    backend.compile_prewarm()  # Idempotent after the successful proof.

    assert len(calls) == first_call_count
    assert [call[0] for call in calls] == [
        "load",
        "initialize_box",
        "track",
        "reset",
        "initialize",
        "track",
        "reset",
    ]
    initialize = calls[1]
    track = calls[2]
    assert initialize[1].shape == (48, 80, 3)
    assert initialize[1].dtype == np.uint8
    assert initialize[3] == 0
    np.testing.assert_array_equal(initialize[2], [20, 12, 60, 36])
    np.testing.assert_array_equal(track[1], initialize[1])
    assert track[2] == 1
    explicit_initialize = calls[4]
    explicit_track = calls[5]
    np.testing.assert_array_equal(explicit_initialize[1], initialize[1])
    assert explicit_initialize[3] == 0
    np.testing.assert_array_equal(
        explicit_initialize[2][12:36, 20:60], 1
    )
    assert int(explicit_initialize[2].sum()) == 24 * 40
    np.testing.assert_array_equal(explicit_track[1], initialize[1])
    assert explicit_track[2] == 1

    health = backend.health()
    assert health["compile_prewarm_required"] is True
    assert health["compile_prewarm_completed"] is True
    assert health["compile_prewarm_input_size_wh"] == [80, 48]
    assert health["compile_prewarm_shape_hw"] == [48, 80]
    assert health["compile_prewarm_contract"] == VOS_COMPILE_PREWARM_CONTRACT
    assert health["compile_prewarm_ms"] >= 0.0
    assert health["compile_prewarm_initialize_box_ms"] >= 0.0
    assert health["compile_prewarm_track_ms"] >= 0.0
    assert health["compile_prewarm_initialize_mask_ms"] >= 0.0
    assert health["compile_prewarm_mask_track_ms"] >= 0.0


def test_official_explicit_mask_initialize_acks_exact_accepted_seed(monkeypatch):
    """The initialize result is an ACK, not a same-frame re-prediction."""

    backend = OfficialSAM2VideoBackend(image_size=64, vos_optimized=False)
    image = np.zeros((48, 80, 3), dtype=np.uint8)
    seed = np.zeros((48, 80), dtype=np.uint8)
    seed[9:36, 17:61] = 1
    consumed = []
    acknowledged = []

    monkeypatch.setattr(backend, "load", lambda: None)
    monkeypatch.setattr(backend, "_drop_state", lambda: None)

    def fake_start_state(_image, mask):
        consumed.append(np.asarray(mask).copy())
        # Deliberately opaque/model-derived logits: the ACK must not be made
        # from their thresholded shape.
        return {"initialize": 1.0}, object()

    def fake_make_result(_logits, frame_id, timings, *, mask=None):
        acknowledged.append(np.asarray(mask).copy())
        return SAM2VideoResult(
            mask=np.asarray(mask, dtype=np.uint8).copy(),
            valid=True,
            frame_id=frame_id,
            timings_ms=dict(timings),
            internal_frame_idx=0,
            object_score=3.0,
            mask_area=int(np.count_nonzero(mask)),
        )

    monkeypatch.setattr(backend, "_start_state", fake_start_state)
    monkeypatch.setattr(backend, "_make_result", fake_make_result)

    result = backend.initialize(image, seed, frame_id=75)

    np.testing.assert_array_equal(consumed[0], seed)
    np.testing.assert_array_equal(acknowledged[0], seed)
    np.testing.assert_array_equal(result.mask, seed)
    assert result.mask_area == int(seed.sum())
    assert result.frame_id == 75
    assert result.internal_frame_idx == 0
    assert result.message == "initialized"


def test_vos_compile_prewarm_failure_resets_and_never_advertises_complete(
    monkeypatch,
):
    backend = OfficialSAM2VideoBackend(image_size=64, vos_optimized=True)
    calls = []
    monkeypatch.setattr(backend, "load", lambda: calls.append("load"))
    monkeypatch.setattr(
        backend,
        "initialize_box",
        lambda image, bbox, frame_id: SAM2VideoResult(
            np.ones(image.shape[:2], dtype=np.uint8), True, frame_id
        ),
    )

    def fail_track(_image, frame_id):
        raise RuntimeError("scripted compile failure")

    monkeypatch.setattr(backend, "track", fail_track)
    monkeypatch.setattr(backend, "reset", lambda: calls.append("reset"))

    with pytest.raises(RuntimeError, match="scripted compile failure"):
        backend.compile_prewarm()

    assert calls == ["load", "reset"]
    assert backend.health()["compile_prewarm_completed"] is False
    assert backend.health()["compile_prewarm_shape_hw"] is None


@pytest.mark.parametrize("invalid_stage", ["initialize", "track"])
def test_vos_compile_prewarm_explicit_mask_session_failure_is_fail_closed(
    monkeypatch, invalid_stage,
):
    backend = OfficialSAM2VideoBackend(
        image_size=64,
        vos_optimized=True,
        prewarm_input_size_wh=(80, 48),
    )
    calls = []
    track_count = 0

    monkeypatch.setattr(backend, "load", lambda: calls.append("load"))
    monkeypatch.setattr(
        backend,
        "initialize_box",
        lambda image, bbox, frame_id: SAM2VideoResult(
            np.ones(image.shape[:2], dtype=np.uint8), True, frame_id
        ),
    )

    def fake_initialize(image, mask, frame_id):
        valid = invalid_stage != "initialize"
        return SAM2VideoResult(
            np.asarray(mask, dtype=np.uint8).copy(), valid, frame_id
        )

    def fake_track(image, frame_id):
        nonlocal track_count
        track_count += 1
        valid = not (invalid_stage == "track" and track_count == 2)
        return SAM2VideoResult(
            np.ones(image.shape[:2], dtype=np.uint8), valid, frame_id
        )

    monkeypatch.setattr(backend, "initialize", fake_initialize)
    monkeypatch.setattr(backend, "track", fake_track)
    monkeypatch.setattr(backend, "reset", lambda: calls.append("reset"))

    with pytest.raises(RuntimeError, match="explicit-mask"):
        backend.compile_prewarm()

    # One phase-boundary reset plus the unconditional final reset; no
    # ambiguous temporal identity may survive either failure stage.
    assert calls.count("reset") == 2
    health = backend.health()
    assert health["compile_prewarm_completed"] is False
    assert health["compile_prewarm_shape_hw"] is None


def test_nonoptimized_backend_has_no_compile_prewarm_obligation(monkeypatch):
    backend = OfficialSAM2VideoBackend(vos_optimized=False)
    monkeypatch.setattr(
        backend,
        "load",
        lambda: pytest.fail("nonoptimized prewarm must not load the model"),
    )

    backend.compile_prewarm()

    health = backend.health()
    assert health["compile_prewarm_required"] is False
    assert health["compile_prewarm_completed"] is False
    assert health["compile_prewarm_shape_hw"] is None


@pytest.mark.parametrize(
    "value",
    [None, (), (848,), (848, 480, 3), (0, 480), (848, -1), (848.5, 480), "84"],
)
def test_backend_rejects_invalid_compile_prewarm_input_size(value):
    with pytest.raises(ValueError, match="prewarm_input_size_wh"):
        OfficialSAM2VideoBackend(prewarm_input_size_wh=value)


def test_run_server_completes_compile_prewarm_before_socket_bind(monkeypatch):
    events = []

    class FakeBackend:
        device_name = "cuda"
        image_size = 64
        prewarm_input_size_wh = (80, 48)
        amp_dtype_name = "bfloat16"
        vos_optimized = True
        vos_compile_mode = "max-autotune-no-cudagraphs"

        def load(self):
            events.append("load")

        def compile_prewarm(self):
            events.append("compile_prewarm")

        def health(self):
            events.append("health")
            return {
                "compile_prewarm_completed": True,
                "compile_prewarm_ms": 12.5,
            }

        def close(self):
            events.append("close_backend")

    class FakeSocket:
        def setsockopt(self, *_args):
            pass

        def bind(self, addr):
            events.append(("bind", addr))

        def poll(self, **_kwargs):
            raise KeyboardInterrupt

        def close(self, **_kwargs):
            events.append("close_socket")

    socket = FakeSocket()
    fake_context = types.SimpleNamespace(socket=lambda _kind: socket)
    monkeypatch.setattr(
        service_module.zmq,
        "Context",
        types.SimpleNamespace(instance=lambda: fake_context),
    )
    monkeypatch.setattr(
        service_module.signal,
        "signal",
        lambda *_args: None,
    )

    service_module.run_server(
        FakeBackend(),
        addr="inproc://compile-prewarm-order",
        max_image_bytes=1024,
        preload=True,
    )

    assert events.index("load") < events.index("compile_prewarm")
    assert events.index("compile_prewarm") < events.index(
        ("bind", "inproc://compile-prewarm-order")
    )
    assert "close_backend" in events


@pytest.mark.parametrize(
    ("flag", "expected"),
    (("--vos-optimized", True), ("--no-vos-optimized", False)),
)
def test_service_cli_threads_vos_optimized_to_backend(
    monkeypatch, flag: str, expected: bool
):
    constructed = []

    class FakeBackend:
        def __init__(self, **kwargs):
            constructed.append(dict(kwargs))

    served = []
    monkeypatch.setattr(service_module, "OfficialSAM2VideoBackend", FakeBackend)
    monkeypatch.setattr(
        service_module,
        "run_server",
        lambda backend, **kwargs: served.append((backend, dict(kwargs))),
    )

    service_module.main([flag, "--lazy-load"])

    assert len(constructed) == 1
    assert constructed[0]["vos_optimized"] is expected
    assert constructed[0]["vos_compile_mode"] == (
        "max-autotune-no-cudagraphs"
    )
    assert constructed[0]["prewarm_input_size_wh"] == (848, 480)
    assert len(served) == 1


def test_service_cli_exposes_diagnostic_cudagraph_ab_mode(monkeypatch):
    constructed = []

    class FakeBackend:
        def __init__(self, **kwargs):
            constructed.append(dict(kwargs))

    monkeypatch.setattr(service_module, "OfficialSAM2VideoBackend", FakeBackend)
    monkeypatch.setattr(service_module, "run_server", lambda *_args, **_kwargs: None)

    service_module.main(
        ["--vos-compile-mode", "max-autotune", "--lazy-load"]
    )

    assert constructed[0]["vos_compile_mode"] == "max-autotune"


def test_backend_box_start_uses_native_sam2_box_prompt(monkeypatch):
    backend = OfficialSAM2VideoBackend()
    state = {"state": "sentinel"}
    logits = object()
    calls = []

    class FakePredictor:
        def add_new_points_or_box(self, value, **kwargs):
            calls.append((value, kwargs))

    backend.predictor = FakePredictor()
    monkeypatch.setattr(
        backend, "_create_initial_state", lambda _image: (state, 1.25)
    )
    monkeypatch.setattr(backend, "_model_context", contextlib.nullcontext)
    monkeypatch.setattr(
        backend, "_propagate_one", lambda frame_idx: (frame_idx, [1], logits)
    )
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    bbox = np.asarray([1, 2, 7, 5], dtype=np.int64)

    timings, observed_logits = backend._start_state_box(image, bbox)

    assert observed_logits is logits
    assert timings["preprocess"] == 1.25
    assert "initialize_box" in timings
    assert calls[0][0] is state
    assert calls[0][1]["frame_idx"] == 0
    assert calls[0][1]["obj_id"] == 1
    np.testing.assert_array_equal(
        calls[0][1]["box"], bbox.astype(np.float32)
    )


@pytest.mark.parametrize("reset_every", [-1, 1])
def test_backend_rejects_unsafe_periodic_reset_values(reset_every: int):
    with pytest.raises(ValueError, match="reset_every_frames"):
        OfficialSAM2VideoBackend(reset_every_frames=reset_every)


def test_service_manager_validates_identity_and_forwards_to_started_client():
    image = np.zeros((3, 5, 3), dtype=np.uint8)
    mask = np.ones((3, 5), dtype=np.uint8)

    class FakeClient:
        def health(self, timeout_ms=None):
            return {
                "service": "dynamic-pcd-online-sam2",
                "protocol_version": 1,
                "timeout_ms": timeout_ms,
            }

        def initialize(self, image_bgr, seed, frame_id, timeout_ms=None):
            return SAM2VideoResult(seed.copy(), True, frame_id)

        def initialize_box(self, image_bgr, bbox, frame_id, timeout_ms=None):
            result = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
            x1, y1, x2, y2 = bbox
            result[y1:y2, x1:x2] = 1
            return SAM2VideoResult(result, True, frame_id)

        def track(self, image_bgr, frame_id, timeout_ms=None):
            return SAM2VideoResult(mask.copy(), True, frame_id)

        def reset(self, timeout_ms=None):
            return {"status": "ok", "timeout_ms": timeout_ms}

        def close(self):
            pass

    manager = SAM2VideoServiceManager(autostart=False, request_timeout_ms=321)
    manager.client = FakeClient()
    assert manager.health()["service"] == "dynamic-pcd-online-sam2"
    assert manager.initialize(image, mask, 7).frame_id == 7
    assert manager.initialize_box(image, [1, 0, 4, 3], 7).mask.sum() == 9
    assert manager.track(image, 8).frame_id == 8
    assert manager.reset()["timeout_ms"] == 321
    manager.close()

    with pytest.raises(RuntimeError, match="unexpected service identity"):
        manager._validate_health(
            {"service": "wrong", "protocol_version": 1}
        )


def test_service_manager_accepts_only_completed_shape_bound_vos_prewarm():
    base = {
        "service": "dynamic-pcd-online-sam2",
        "protocol_version": 1,
        "vos_optimized": True,
        "vos_compile_mode": "max-autotune-no-cudagraphs",
        "vos_compile_cuda_graphs": False,
        "vos_component_compile_modes": {
            "image_encoder": "max-autotune-no-cudagraphs",
            "memory_encoder": "max-autotune-no-cudagraphs",
            "memory_attention": "max-autotune-no-cudagraphs",
            "sam_prompt_encoder": "max-autotune-no-cudagraphs",
            "sam_mask_decoder": "max-autotune-no-cudagraphs",
        },
        "vos_component_compile_dynamic": {
            "image_encoder": False,
            "memory_encoder": False,
            "memory_attention": True,
            "sam_prompt_encoder": False,
            "sam_mask_decoder": False,
        },
        "vos_memory_attention_rope_grid_hw": [32, 32],
        "vos_memory_attention_rope_expected_tokens": 1024,
        "vos_memory_attention_rope_cache_count": 8,
        "vos_memory_attention_rope_cache_token_counts": [1024] * 8,
        "vos_memory_attention_rope_caches_verified": True,
        "image_size": 512,
        "initialized": False,
        "compile_prewarm_required": True,
        "compile_prewarm_completed": True,
        "compile_prewarm_contract": VOS_COMPILE_PREWARM_CONTRACT,
        "compile_prewarm_input_size_wh": [848, 480],
        "compile_prewarm_shape_hw": [480, 848],
        "compile_prewarm_ms": 123.0,
        "compile_prewarm_initialize_box_ms": 10.0,
        "compile_prewarm_track_ms": 11.0,
        "compile_prewarm_initialize_mask_ms": 12.0,
        "compile_prewarm_mask_track_ms": 13.0,
    }

    accepted = SAM2VideoServiceManager._validate_health(dict(base))
    assert accepted["compile_prewarm_completed"] is True

    runtime_health = dict(base)
    runtime_health["initialized"] = True
    runtime_accepted = SAM2VideoServiceManager._validate_health(
        runtime_health,
        require_uninitialized_after_prewarm=False,
    )
    assert runtime_accepted["initialized"] is True

    mutations = (
        ("compile_prewarm_required", False, "mandatory compile prewarm"),
        ("compile_prewarm_completed", False, "not compile-prewarmed"),
        ("compile_prewarm_contract", "box_only", "does not cover both"),
        ("compile_prewarm_shape_hw", None, "shape is missing"),
        ("compile_prewarm_shape_hw", [512, 512], "does not match configured input"),
        ("initialized", True, "retained temporal state"),
    )
    for key, value, message in mutations:
        health = dict(base)
        health[key] = value
        with pytest.raises(RuntimeError, match=message):
            SAM2VideoServiceManager._validate_health(health)


def test_service_manager_runtime_health_accepts_an_initialized_temporal_session():
    runtime_health = {
        "service": "dynamic-pcd-online-sam2",
        "protocol_version": 1,
        "vos_optimized": False,
        "initialized": True,
    }

    class FakeClient:
        def health(self, timeout_ms=None):
            assert timeout_ms == 250
            return dict(runtime_health)

    manager = SAM2VideoServiceManager(autostart=False)
    manager.client = FakeClient()
    assert manager.health(timeout_ms=250) == runtime_health


def test_service_manager_keeps_nonoptimized_ab_and_legacy_fake_health_compatible():
    for health in (
        {
            "service": "dynamic-pcd-online-sam2",
            "protocol_version": 1,
            "vos_optimized": False,
        },
        {
            "service": "dynamic-pcd-online-sam2",
            "protocol_version": 1,
        },
    ):
        assert SAM2VideoServiceManager._validate_health(dict(health)) == health
