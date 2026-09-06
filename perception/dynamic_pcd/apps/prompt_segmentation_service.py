from __future__ import annotations

import argparse
import importlib
import os
import re
import signal
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import zmq

from dynamic_pcd.segmentation.grounded_sam import (
    GroundedSAMBackend,
    PromptModelDependencyError,
)
from dynamic_pcd.segmentation.prompt_protocol import (
    PROTOCOL_VERSION,
    decode_json,
    decode_segment_request,
    encode_json,
    make_error_parts,
    make_result_parts,
)


DEFAULT_GROUNDED_SAM_ROOT = (
    "/home/qiaoguanren/下载/founpose/FoundationPose-main/"
    "Grounded-Segment-Anything"
)
DEFAULT_DINO_CONFIG = (
    DEFAULT_GROUNDED_SAM_ROOT
    + "/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
DEFAULT_DINO_CHECKPOINT = (
    DEFAULT_GROUNDED_SAM_ROOT + "/groundingdino_swint_ogc.pth"
)
DEFAULT_SAM_CHECKPOINT = "/home/qiaoguanren/下载/sam_vit_b_01ec64.pth"
DEFAULT_SAM2_CHECKPOINT = (
    "/home/qiaoguanren/code/franka/third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt"
)
DEFAULT_SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"


_CUDA_DEVICE_RE = re.compile(r"cuda(?::([0-9]+))?\Z")


def resolve_service_device(
    requested_device: str,
    *,
    torch_module: Optional[Any] = None,
) -> str:
    """Resolve ``auto`` and fail closed for an unavailable explicit CUDA.

    ``auto`` deliberately falls back to CPU when PyTorch is absent, CUDA is
    unavailable, or the CUDA runtime cannot be queried.  An explicit
    ``cuda``/``cuda:N`` request is different: silently running a multi-second
    CPU model would violate the operator's latency expectation, so startup
    fails with an actionable error instead.

    ``torch_module`` is injectable so this policy can be tested without a GPU.
    """

    if not isinstance(requested_device, str) or not requested_device.strip():
        raise ValueError("device must be one of: auto, cpu, cuda, cuda:N")
    requested = requested_device.strip().lower()
    if requested == "cpu":
        return "cpu"
    cuda_match = _CUDA_DEVICE_RE.fullmatch(requested)
    if requested != "auto" and cuda_match is None:
        raise ValueError(
            f"unsupported device {requested_device!r}; expected auto, cpu, "
            "cuda, or cuda:N"
        )

    try:
        torch = (
            torch_module
            if torch_module is not None
            else importlib.import_module("torch")
        )
        cuda_available = bool(torch.cuda.is_available())
    except (ImportError, AttributeError, RuntimeError) as exc:
        if requested == "auto":
            return "cpu"
        raise PromptModelDependencyError(
            f"explicit device {requested!r} requires a working CUDA-enabled "
            f"PyTorch installation ({type(exc).__name__}: {exc})"
        ) from exc

    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if not cuda_available:
        raise PromptModelDependencyError(
            f"explicit device {requested!r} was requested, but "
            "torch.cuda.is_available() is false; use --device auto or "
            "--device cpu, or restore the NVIDIA driver/CUDA runtime"
        )

    index_text = cuda_match.group(1)
    if index_text is not None:
        index = int(index_text)
        try:
            device_count = int(torch.cuda.device_count())
        except (AttributeError, RuntimeError) as exc:
            raise PromptModelDependencyError(
                f"could not validate explicit device {requested!r}: {exc}"
            ) from exc
        if index >= device_count:
            raise PromptModelDependencyError(
                f"explicit device {requested!r} does not exist; PyTorch "
                f"reports {device_count} CUDA device(s)"
            )
    return requested


class PromptSegmentationService:
    """Synchronous request handler around a persistent grounded-SAM backend."""

    def __init__(
        self,
        backend: GroundedSAMBackend,
        max_image_bytes: int = 16 * 1024 * 1024,
        model_load_ms: float = 0.0,
    ):
        self.backend = backend
        self.max_image_bytes = max(1, int(max_image_bytes))
        self.started_at = time.time()
        self.request_count = 0
        self.model_load_ms = float(model_load_ms)

    def handle_request(self, parts: Sequence[bytes]) -> Tuple[bytes, ...]:
        request_id = ""
        try:
            if not parts:
                raise ValueError("empty request")
            initial_header = decode_json(parts[0])
            request_id = str(initial_header.get("request_id", ""))
            if int(initial_header.get("version", -1)) != PROTOCOL_VERSION:
                raise ValueError(
                    f"unsupported protocol version {initial_header.get('version')!r}"
                )
            operation = initial_header.get("op")
            if operation == "health":
                if len(parts) != 1:
                    raise ValueError("health request must contain one part")
                return (
                    encode_json(
                        {
                            "version": PROTOCOL_VERSION,
                            "status": "ok",
                            "request_id": request_id,
                            "service": "grounded_sam_prompt_segmentation",
                            "uptime_s": max(0.0, time.time() - self.started_at),
                            "request_count": int(self.request_count),
                            "detector_backend": self.backend.detector_backend,
                            "mask_backend": self.backend.mask_backend,
                            "device": self.backend.device,
                            "model_load_ms": self.model_load_ms,
                        }
                    ),
                )
            if operation != "segment":
                raise ValueError(f"unsupported request op {operation!r}")

            header, image_bgr = decode_segment_request(
                parts, max_image_bytes=self.max_image_bytes
            )
            self.request_count += 1
            inference_start = time.perf_counter()
            result = self.backend.segment(
                image_bgr=image_bgr,
                prompt=header["prompt"],
                box_threshold=header["box_threshold"],
                text_threshold=header["text_threshold"],
                mask_threshold=header["mask_threshold"],
                reference_bbox_xyxy=header.get("reference_bbox_xyxy"),
                top_k=header["top_k"],
                request_id=request_id,
                frame_id=header.get("frame_id"),
                frame_timestamp=header.get("frame_timestamp"),
                frame_metadata=header.get("frame_metadata"),
            )
            result.timings_ms["service_inference"] = (
                time.perf_counter() - inference_start
            ) * 1000.0
            return make_result_parts(result)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            return make_error_parts(message, request_id=request_id)


def run_server(
    backend: GroundedSAMBackend,
    addr: str,
    max_image_bytes: int,
    preload: bool = True,
) -> None:
    model_load_ms = 0.0
    if preload:
        print(
            "[PromptService] loading persistent "
            f"{backend.detector_backend}/{backend.mask_backend} models ...",
            flush=True,
        )
        load_start = time.perf_counter()
        backend.load()
        model_load_ms = (time.perf_counter() - load_start) * 1000.0
        print(
            f"[PromptService] models loaded in {model_load_ms:.1f} ms", flush=True
        )

    service = PromptSegmentationService(
        backend=backend,
        max_image_bytes=max_image_bytes,
        model_load_ms=model_load_ms,
    )
    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDHWM, 2)
    socket.setsockopt(zmq.RCVHWM, 2)
    socket.bind(addr)
    print(
        f"[PromptService] listening on {addr}; "
        f"detector={backend.detector_backend}, mask={backend.mask_backend}, "
        f"device={backend.device}",
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
            parts = socket.recv_multipart()
            response = service.handle_request(parts)
            socket.send_multipart(list(response))
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        socket.close(linger=0)
        print("[PromptService] stopped", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Persistent Grounding-DINO + SAM prompt segmentation ZMQ service"
        )
    )
    parser.add_argument("--addr", default="tcp://127.0.0.1:5557")
    parser.add_argument(
        "--detector-backend",
        choices=("native", "transformers"),
        default="native",
    )
    parser.add_argument("--dino-config", default=DEFAULT_DINO_CONFIG)
    parser.add_argument("--dino-checkpoint", default=DEFAULT_DINO_CHECKPOINT)
    parser.add_argument(
        "--dino-model-id", default="IDEA-Research/grounding-dino-tiny"
    )
    parser.add_argument("--local-files-only", action="store_true")
    network_group = parser.add_mutually_exclusive_group()
    network_group.add_argument(
        "--offline",
        dest="offline",
        action="store_true",
        help="force Hugging Face/Transformers offline mode (default)",
    )
    network_group.add_argument(
        "--allow-network",
        dest="offline",
        action="store_false",
        help="allow model/tokenizer downloads when a cache entry is missing",
    )
    parser.set_defaults(offline=True)
    parser.add_argument(
        "--mask-backend", choices=("sam1", "sam2"), default="sam1"
    )
    parser.add_argument("--sam-checkpoint", default=DEFAULT_SAM_CHECKPOINT)
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--sam2-checkpoint", default=DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument("--sam2-model-cfg", default=DEFAULT_SAM2_MODEL_CFG)
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "inference device: auto selects CUDA when available and otherwise "
            "CPU; explicit cuda/cuda:N fails if unavailable"
        ),
    )
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--min-mask-area", type=int, default=20)
    parser.add_argument("--max-mask-area-ratio", type=float, default=6.0)
    parser.add_argument("--max-sam-candidates", type=int, default=3)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.80)
    parser.add_argument("--reference-iou-weight", type=float, default=1.0)
    parser.add_argument("--reference-center-weight", type=float, default=0.25)
    parser.add_argument("--max-image-mb", type=float, default=16.0)
    parser.add_argument(
        "--lazy-load",
        action="store_true",
        help="bind before loading models; intended only for dependency debugging",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
    if args.torch_threads > 0:
        try:
            import torch

            torch.set_num_threads(int(args.torch_threads))
        except ImportError:
            pass

    try:
        resolved_device = resolve_service_device(args.device)
    except (ValueError, PromptModelDependencyError) as exc:
        raise SystemExit(f"[PromptService] device error: {exc}") from exc
    print(
        f"[PromptService] device requested={args.device!r}, "
        f"resolved={resolved_device!r}",
        flush=True,
    )

    backend = GroundedSAMBackend(
        detector_backend=args.detector_backend,
        mask_backend=args.mask_backend,
        dino_config=args.dino_config,
        dino_checkpoint=args.dino_checkpoint,
        dino_model_id=args.dino_model_id,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_model_cfg=args.sam2_model_cfg,
        device=resolved_device,
        local_files_only=args.local_files_only,
        min_mask_area=args.min_mask_area,
        max_mask_area_ratio=args.max_mask_area_ratio,
        max_sam_candidates=args.max_sam_candidates,
        nms_iou_threshold=args.nms_iou_threshold,
        reference_iou_weight=args.reference_iou_weight,
        reference_center_weight=args.reference_center_weight,
    )
    try:
        run_server(
            backend=backend,
            addr=args.addr,
            max_image_bytes=int(float(args.max_image_mb) * 1024 * 1024),
            preload=not args.lazy_load,
        )
    except PromptModelDependencyError as exc:
        raise SystemExit(f"[PromptService] dependency/model error: {exc}") from exc


if __name__ == "__main__":
    main()
