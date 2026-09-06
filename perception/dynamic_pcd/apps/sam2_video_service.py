from __future__ import annotations

import argparse
import signal
import time
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import zmq

from dynamic_pcd.segmentation.sam2_video_backend import (
    ALLOWED_VOS_COMPILE_MODES,
    DEFAULT_CHECKPOINT,
    DEFAULT_MODEL_CONFIG,
    OfficialSAM2VideoBackend,
    PRODUCTION_VOS_COMPILE_MODE,
)
from dynamic_pcd.segmentation.sam2_video_protocol import (
    PROTOCOL_VERSION,
    SAM2VideoResult,
    decode_json,
    decode_request,
    make_error_parts,
    make_ok_parts,
    make_result_parts,
    validate_mask,
)


SERVICE_NAME = "dynamic-pcd-online-sam2"


class SAM2VideoService:
    """Synchronous RPC handler around one stateful video backend."""

    def __init__(
        self,
        backend: Any,
        *,
        max_image_bytes: int = 16 * 1024 * 1024,
        model_load_ms: float = 0.0,
    ):
        self.backend = backend
        self.max_image_bytes = max(1, int(max_image_bytes))
        self.model_load_ms = float(model_load_ms)
        self.started_at = time.time()
        self.request_count = 0
        self._shape_hw = None
        self._last_frame_id = None

    def handle_request(self, parts: Sequence[bytes]) -> Tuple[bytes, ...]:
        request_id = ""
        operation = ""
        started = time.perf_counter()
        try:
            if not parts:
                raise ValueError("empty request")
            initial_header = decode_json(parts[0])
            request_id = str(initial_header.get("request_id", ""))
            header, image, mask = decode_request(
                parts, max_image_bytes=self.max_image_bytes
            )
            operation = header["op"]
            request_id = header["request_id"]

            if operation == "health":
                health = dict(self.backend.health())
                health.update(
                    {
                        "service": SERVICE_NAME,
                        "protocol_version": PROTOCOL_VERSION,
                        "request_count": self.request_count,
                        "uptime_s": max(0.0, time.time() - self.started_at),
                        "model_load_ms": self.model_load_ms,
                    }
                )
                return make_ok_parts("health", request_id, **health)

            if operation == "reset":
                try:
                    self.backend.reset()
                finally:
                    # Even a backend cleanup error leaves its temporal state
                    # ambiguous; the RPC state machine must stay fail-closed.
                    self._shape_hw = None
                    self._last_frame_id = None
                self.request_count += 1
                return make_ok_parts(
                    "reset",
                    request_id,
                    service=SERVICE_NAME,
                    initialized=False,
                )

            frame_id = header["frame_id"]
            if operation == "initialize":
                try:
                    result = self.backend.initialize(image, mask, frame_id)
                except Exception:
                    self._reset_after_model_error()
                    raise
            elif operation == "initialize_box":
                try:
                    result = self.backend.initialize_box(
                        image, header["bbox_xyxy"], frame_id
                    )
                except Exception:
                    self._reset_after_model_error()
                    raise
            elif operation == "track":
                if self._shape_hw is None or self._last_frame_id is None:
                    raise RuntimeError("SAM2 video service is not initialized")
                if tuple(image.shape[:2]) != self._shape_hw:
                    raise ValueError(
                        f"track image shape {image.shape[:2]} does not match "
                        f"initialized shape {self._shape_hw}"
                    )
                if frame_id <= self._last_frame_id:
                    raise ValueError(
                        f"track frame_id {frame_id} must be greater than "
                        f"{self._last_frame_id}"
                    )
                try:
                    result = self.backend.track(image, frame_id)
                except Exception:
                    self._reset_after_model_error()
                    raise
            else:  # decode_request rejects this; keep the dispatch fail-closed.
                raise ValueError(f"unsupported operation {operation!r}")
            if not isinstance(result, SAM2VideoResult):
                self._reset_after_model_error()
                raise TypeError(
                    "SAM2 backend must return SAM2VideoResult, received "
                    f"{type(result).__name__}"
                )
            if result.frame_id != frame_id:
                self._reset_after_model_error()
                raise RuntimeError(
                    f"backend frame_id {result.frame_id} does not match request {frame_id}"
                )
            try:
                result.mask = validate_mask(
                    result.mask,
                    expected_shape=image.shape[:2],
                    require_nonempty=False,
                )
            except Exception:
                self._reset_after_model_error()
                raise
            if result.mask_area != int(result.mask.sum()):
                result.mask_area = int(result.mask.sum())
            if result.valid and result.mask_area == 0:
                self._reset_after_model_error()
                raise RuntimeError("backend marked an empty mask as valid")
            if operation in ("initialize", "initialize_box"):
                self._shape_hw = tuple(image.shape[:2])
            self._last_frame_id = frame_id
            result.request_id = request_id
            result.timings_ms["service_total"] = (
                time.perf_counter() - started
            ) * 1000.0
            self.request_count += 1
            try:
                return make_result_parts(result)
            except Exception:
                self._reset_after_model_error()
                raise
        except Exception as exc:
            error_code = self._error_code(exc, operation)
            message = f"{type(exc).__name__}: {exc}"
            return make_error_parts(
                f"{error_code}: {message}", request_id=request_id
            )

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()

    def _reset_after_model_error(self) -> None:
        self._shape_hw = None
        self._last_frame_id = None
        try:
            self.backend.reset()
        except Exception:
            pass

    @staticmethod
    def _error_code(exc: BaseException, operation: str) -> str:
        text = str(exc).lower()
        if "not initialized" in text:
            return "not_initialized"
        if "frame_id" in text and ("greater" in text or "out of" in text):
            return "out_of_order"
        if isinstance(exc, (ValueError, TypeError)):
            return "bad_request"
        if "cuda" in text or "out of memory" in text:
            return "cuda_error"
        return "model_error" if operation in ("initialize", "track") else "internal"


def run_server(
    backend: Any,
    *,
    addr: str,
    max_image_bytes: int,
    preload: bool = True,
) -> None:
    model_load_ms = 0.0
    if preload:
        print("[SAM2VideoService] loading persistent SAM2 model ...", flush=True)
        load_started = time.perf_counter()
        backend.load()
        model_load_ms = (time.perf_counter() - load_started) * 1000.0
        print(
            f"[SAM2VideoService] model loaded in {model_load_ms:.1f} ms",
            flush=True,
        )

    # The official VOS-optimized builder defers torch.compile work until the
    # first prompted frame and first temporal propagation.  Complete both
    # forwards locally before constructing/binding the RPC socket.  Therefore
    # neither health nor the manager's READY message can race the compile tail,
    # while ordinary runtime initialize/track RPC deadlines remain unchanged.
    if bool(getattr(backend, "vos_optimized", False)):
        print(
            "[SAM2VideoService] compiling VOS first-forward graphs: "
            "initialize_box + track at "
            f"{int(backend.prewarm_input_size_wh[0])}x"
            f"{int(backend.prewarm_input_size_wh[1])} "
            f"(model image_size={int(backend.image_size)}) ...",
            flush=True,
        )
        backend.compile_prewarm()
        prewarm_health = dict(backend.health())
        if not preload:
            # ``--lazy-load`` remains useful for the non-optimized debug path.
            # Optimized mode must load during mandatory prewarm, so preserve
            # the backend's measured model-load provenance in service health.
            model_load_ms = float(prewarm_health.get("model_load_ms", 0.0))
        if prewarm_health.get("compile_prewarm_completed") is not True:
            raise RuntimeError(
                "VOS-optimized SAM2 compile prewarm did not complete"
            )
        print(
            "[SAM2VideoService] compile prewarm complete in "
            f"{float(prewarm_health.get('compile_prewarm_ms', 0.0)):.1f} ms; "
            "temporal state reset",
            flush=True,
        )

    service = SAM2VideoService(
        backend,
        max_image_bytes=max_image_bytes,
        model_load_ms=model_load_ms,
    )
    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDHWM, 2)
    socket.setsockopt(zmq.RCVHWM, 2)
    try:
        socket.bind(addr)
    except zmq.ZMQError as exc:
        socket.close(linger=0)
        service.close()
        raise RuntimeError(
            f"cannot bind SAM2 video service to {addr}: {exc}; "
            "check for an old service process or choose another --addr"
        ) from exc
    print(
        f"[SAM2VideoService] listening on {addr}; "
        f"device={backend.device_name}, image_size={backend.image_size}, "
        f"amp={backend.amp_dtype_name}, vos_optimized={backend.vos_optimized}, "
        f"vos_compile_mode={backend.vos_compile_mode}",
        flush=True,
    )

    stopping = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stopping
        stopping = True

    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        while not stopping:
            if not socket.poll(timeout=250, flags=zmq.POLLIN):
                continue
            request = socket.recv_multipart()
            response = service.handle_request(request)
            socket.send_multipart(list(response))
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        socket.close(linger=0)
        service.close()
        print("[SAM2VideoService] stopped", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Persistent in-memory official SAM2 video tracking service"
    )
    parser.add_argument("--addr", default="tcp://127.0.0.1:5558")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--prewarm-input-width",
        type=int,
        default=848,
        help="camera input width used by mandatory optimized compile prewarm",
    )
    parser.add_argument(
        "--prewarm-input-height",
        type=int,
        default=480,
        help="camera input height used by mandatory optimized compile prewarm",
    )
    parser.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--min-mask-area", type=int, default=20)
    parser.add_argument("--object-score-threshold", type=float, default=0.0)
    parser.add_argument(
        "--vos-optimized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "use the official SAM2 VOS-optimized predictor path; "
            "--no-vos-optimized keeps an explicit A/B baseline"
        ),
    )
    parser.add_argument(
        "--vos-compile-mode",
        choices=ALLOWED_VOS_COMPILE_MODES,
        default=PRODUCTION_VOS_COMPILE_MODE,
        help=(
            "Inductor mode for official VOS components; production uses "
            "max-autotune-no-cudagraphs, while max-autotune is diagnostic "
            "only because CUDA Graph storage reuse conflicts with SAM2 caches"
        ),
    )
    parser.add_argument(
        "--reset-every-frames",
        type=int,
        default=0,
        help=(
            "re-seed temporal state from the latest valid mask after this many "
            "frames; 0 disables periodic reset"
        ),
    )
    parser.add_argument("--max-image-mb", type=float, default=16.0)
    parser.add_argument(
        "--lazy-load",
        action="store_true",
        help="bind before model load; intended only for dependency debugging",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    backend = OfficialSAM2VideoBackend(
        checkpoint=args.checkpoint,
        model_config=args.model_config,
        device=args.device,
        image_size=args.image_size,
        amp_dtype=args.amp_dtype,
        min_mask_area=args.min_mask_area,
        object_score_threshold=args.object_score_threshold,
        reset_every_frames=args.reset_every_frames,
        vos_optimized=args.vos_optimized,
        vos_compile_mode=args.vos_compile_mode,
        prewarm_input_size_wh=(
            args.prewarm_input_width,
            args.prewarm_input_height,
        ),
    )
    run_server(
        backend,
        addr=str(args.addr),
        max_image_bytes=max(1, int(float(args.max_image_mb) * 1024 * 1024)),
        preload=not bool(args.lazy_load),
    )


if __name__ == "__main__":
    main()
