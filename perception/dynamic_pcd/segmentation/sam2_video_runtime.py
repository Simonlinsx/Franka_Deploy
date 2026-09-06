from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from dynamic_pcd.segmentation.sam2_video_backend import (
    ALLOWED_VOS_COMPILE_MODES,
    PRODUCTION_VOS_COMPONENT_COMPILE_MODES,
    PRODUCTION_VOS_COMPONENT_DYNAMIC,
    PRODUCTION_VOS_COMPILE_MODE,
    SAM2_MEMORY_ATTENTION_STRIDE,
    VOS_COMPILE_PREWARM_CONTRACT,
)
from dynamic_pcd.segmentation.sam2_video_client import ZMQSAM2VideoClient
from dynamic_pcd.segmentation.sam2_video_protocol import SAM2VideoResult


SERVICE_NAME = "dynamic-pcd-online-sam2"


def sam2_video_launcher_args_from_config(
    config: Mapping[str, Any],
) -> Tuple[str, ...]:
    """Build the exact service command shared by providers and prewarm."""

    online_cfg = dict(config.get("online_sam2", {}))
    camera_cfg = dict(config.get("camera", {}))
    vos_optimized = online_cfg.get("vos_optimized", True)
    if not isinstance(vos_optimized, bool):
        raise ValueError("online_sam2.vos_optimized must be true or false")
    vos_compile_mode = online_cfg.get(
        "vos_compile_mode", PRODUCTION_VOS_COMPILE_MODE
    )
    if vos_compile_mode not in ALLOWED_VOS_COMPILE_MODES:
        raise ValueError(
            "online_sam2.vos_compile_mode must be one of "
            + ", ".join(ALLOWED_VOS_COMPILE_MODES)
        )
    if vos_optimized and vos_compile_mode != PRODUCTION_VOS_COMPILE_MODE:
        raise ValueError(
            "online_sam2 optimized production runtime requires "
            f"vos_compile_mode={PRODUCTION_VOS_COMPILE_MODE}"
        )
    return (
        "--checkpoint",
        str(online_cfg.get("checkpoint", "")),
        "--model-config",
        str(
            online_cfg.get(
                "model_config", "configs/sam2.1/sam2.1_hiera_t.yaml"
            )
        ),
        "--device",
        str(online_cfg.get("device", "cuda")),
        "--image-size",
        str(int(online_cfg.get("image_size", 512))),
        "--prewarm-input-width",
        str(int(camera_cfg.get("width", 848))),
        "--prewarm-input-height",
        str(int(camera_cfg.get("height", 480))),
        "--amp-dtype",
        str(online_cfg.get("amp_dtype", "bfloat16")),
        "--reset-every-frames",
        str(int(online_cfg.get("reset_every_frames", 0))),
        "--min-mask-area",
        str(int(online_cfg.get("min_mask_area", 20))),
        "--object-score-threshold",
        str(float(online_cfg.get("object_score_threshold", 0.0))),
        "--vos-compile-mode",
        str(vos_compile_mode),
        "--vos-optimized" if vos_optimized else "--no-vos-optimized",
    )


def sam2_video_service_manager_kwargs_from_config(
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return one canonical manager contract for every runtime phase."""

    online_cfg = dict(config.get("online_sam2", {}))
    return {
        "addr": str(
            online_cfg.get("service_addr", "tcp://127.0.0.1:5558")
        ),
        "autostart": bool(online_cfg.get("service_autostart", True)),
        "startup_timeout_s": float(
            online_cfg.get("startup_timeout_s", 60.0)
        ),
        "request_timeout_ms": max(
            1,
            int(1000.0 * float(online_cfg.get("request_timeout_s", 0.50))),
        ),
        "launcher_args": sam2_video_launcher_args_from_config(config),
    }


class SAM2VideoServiceManager:
    """Connect to an existing online SAM2 service or launch one on demand."""

    def __init__(
        self,
        addr: str = "tcp://127.0.0.1:5558",
        *,
        autostart: bool = True,
        startup_timeout_s: float = 60.0,
        request_timeout_ms: int = 2000,
        launcher_path: Optional[str] = None,
        launcher_args: Optional[Sequence[str]] = None,
    ):
        self.addr = str(addr)
        self.autostart = bool(autostart)
        self.startup_timeout_s = max(1.0, float(startup_timeout_s))
        self.request_timeout_ms = max(1, int(request_timeout_ms))
        repo_root = Path(__file__).resolve().parents[2]
        self.launcher_path = Path(
            launcher_path
            if launcher_path is not None
            else repo_root / "scripts" / "run_sam2_video_service.sh"
        ).expanduser()
        self.launcher_args = [str(value) for value in (launcher_args or ())]
        self.client: Optional[ZMQSAM2VideoClient] = None
        self.process: Optional[subprocess.Popen] = None
        self.owns_process = False

    def start(self) -> dict:
        if self.client is not None:
            return self._validate_health(self.client.health(timeout_ms=1000))
        self.client = ZMQSAM2VideoClient(
            self.addr, timeout_ms=self.request_timeout_ms
        )
        try:
            health = self._validate_health(self.client.health(timeout_ms=300))
            print(
                f"[SAM2Video] using existing service at {self.addr}: "
                f"device={health.get('device')} image_size={health.get('image_size')}",
                flush=True,
            )
            return health
        except Exception as first_error:
            if not self.autostart:
                self.client.close()
                self.client = None
                raise RuntimeError(
                    f"SAM2 video service at {self.addr} is unavailable and "
                    "autostart is disabled"
                ) from first_error

        if not self.launcher_path.is_file():
            self.close()
            raise RuntimeError(
                f"SAM2 video service launcher does not exist: {self.launcher_path}"
            )
        command = [str(self.launcher_path), "--addr", self.addr]
        command.extend(self.launcher_args)
        self.process = subprocess.Popen(
            command,
            cwd=str(self.launcher_path.parent.parent),
            # The isolated ROI parent intentionally suppresses its own Qt
            # stderr. Merge the service traceback into inherited stdout so a
            # model/dependency failure remains visible and diagnosable.
            stderr=subprocess.STDOUT,
        )
        self.owns_process = True
        print(
            f"[SAM2Video] starting service pid={self.process.pid}; waiting up to "
            f"{self.startup_timeout_s:.0f}s",
            flush=True,
        )

        deadline = time.monotonic() + self.startup_timeout_s
        last_error: Optional[BaseException] = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                return_code = self.process.returncode
                self.close()
                raise RuntimeError(
                    "SAM2 video service exited during startup with code "
                    f"{return_code}"
                )
            try:
                health = self._validate_health(
                    self.client.health(timeout_ms=500)
                )
                print(
                    "[SAM2Video] service ready: "
                    f"device={health.get('device')} "
                    f"image_size={health.get('image_size')} "
                    f"load={float(health.get('model_load_ms', 0.0)):.1f}ms "
                    "compile_prewarm="
                    f"{float(health.get('compile_prewarm_ms', 0.0)):.1f}ms",
                    flush=True,
                )
                return health
            except Exception as exc:
                last_error = exc
                time.sleep(0.10)
        self.close()
        raise TimeoutError(
            f"SAM2 video service at {self.addr} did not become ready within "
            f"{self.startup_timeout_s:.1f}s"
        ) from last_error

    def health(self, timeout_ms: Optional[int] = None) -> dict:
        return self._validate_health(
            self._client().health(timeout_ms=timeout_ms),
            require_uninitialized_after_prewarm=False,
        )

    def initialize(
        self,
        image_bgr: np.ndarray,
        mask: np.ndarray,
        frame_id: int,
        *,
        timeout_ms: Optional[int] = None,
    ) -> SAM2VideoResult:
        return self._client().initialize(
            image_bgr,
            mask,
            frame_id,
            timeout_ms=(
                self.request_timeout_ms if timeout_ms is None else timeout_ms
            ),
        )

    def initialize_box(
        self,
        image_bgr: np.ndarray,
        bbox_xyxy: np.ndarray,
        frame_id: int,
        *,
        timeout_ms: Optional[int] = None,
    ) -> SAM2VideoResult:
        return self._client().initialize_box(
            image_bgr,
            bbox_xyxy,
            frame_id,
            timeout_ms=(
                self.request_timeout_ms if timeout_ms is None else timeout_ms
            ),
        )

    def track(
        self,
        image_bgr: np.ndarray,
        frame_id: int,
        *,
        timeout_ms: Optional[int] = None,
    ) -> SAM2VideoResult:
        return self._client().track(
            image_bgr,
            frame_id,
            timeout_ms=(
                self.request_timeout_ms if timeout_ms is None else timeout_ms
            ),
        )

    def reset(self, timeout_ms: Optional[int] = None) -> dict:
        return self._client().reset(
            timeout_ms=(
                self.request_timeout_ms if timeout_ms is None else timeout_ms
            )
        )

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
        if self.owns_process and self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2.0)
            print("[SAM2Video] owned service stopped", flush=True)
        self.process = None
        self.owns_process = False

    def _client(self) -> ZMQSAM2VideoClient:
        if self.client is None:
            raise RuntimeError("SAM2 video service manager is not started")
        return self.client

    @staticmethod
    def _validate_health(
        health: dict,
        *,
        require_uninitialized_after_prewarm: bool = True,
    ) -> dict:
        if health.get("service") != SERVICE_NAME:
            raise RuntimeError(
                "unexpected service identity at SAM2 video address: "
                f"{health.get('service')!r}"
            )
        if int(health.get("protocol_version", -1)) != 1:
            raise RuntimeError(
                "SAM2 video service protocol mismatch: "
                f"{health.get('protocol_version')!r}"
            )
        if health.get("vos_optimized") is True:
            image_size = int(health.get("image_size", 0))
            rope_side = image_size // SAM2_MEMORY_ATTENTION_STRIDE
            expected_rope_grid = [rope_side, rope_side]
            expected_rope_tokens = rope_side * rope_side
            rope_token_counts = health.get(
                "vos_memory_attention_rope_cache_token_counts"
            )
            if (
                image_size <= 0
                or image_size % SAM2_MEMORY_ATTENTION_STRIDE != 0
                or health.get("vos_compile_mode")
                != PRODUCTION_VOS_COMPILE_MODE
                or health.get("vos_compile_cuda_graphs") is not False
                or health.get("vos_component_compile_modes")
                != dict(PRODUCTION_VOS_COMPONENT_COMPILE_MODES)
                or health.get("vos_component_compile_dynamic")
                != dict(PRODUCTION_VOS_COMPONENT_DYNAMIC)
                or health.get("vos_memory_attention_rope_grid_hw")
                != expected_rope_grid
                or health.get("vos_memory_attention_rope_expected_tokens")
                != expected_rope_tokens
                or health.get("vos_memory_attention_rope_caches_verified") is not True
                or not isinstance(rope_token_counts, list)
                or not rope_token_counts
                or any(
                    int(token_count) != expected_rope_tokens
                    for token_count in rope_token_counts
                )
                or health.get("vos_memory_attention_rope_cache_count")
                != len(rope_token_counts)
            ):
                raise RuntimeError(
                    "VOS-optimized SAM2 production service must use "
                    f"{PRODUCTION_VOS_COMPILE_MODE!r} with CUDA Graphs disabled "
                    "and an image-aligned verified RoPE cache; "
                    f"health mode={health.get('vos_compile_mode')!r}, "
                    "cuda_graphs="
                    f"{health.get('vos_compile_cuda_graphs')!r}, components="
                    f"{health.get('vos_component_compile_modes')!r}, dynamic="
                    f"{health.get('vos_component_compile_dynamic')!r}, "
                    "rope_grid="
                    f"{health.get('vos_memory_attention_rope_grid_hw')!r}, "
                    "rope_tokens="
                    f"{health.get('vos_memory_attention_rope_cache_token_counts')!r}, "
                    "rope_verified="
                    f"{health.get('vos_memory_attention_rope_caches_verified')!r}"
                )
            if health.get("compile_prewarm_required") is not True:
                raise RuntimeError(
                    "VOS-optimized SAM2 service did not declare mandatory "
                    "compile prewarm"
                )
            if health.get("compile_prewarm_completed") is not True:
                raise RuntimeError(
                    "VOS-optimized SAM2 service is not compile-prewarmed"
                )
            if (
                health.get("compile_prewarm_contract")
                != VOS_COMPILE_PREWARM_CONTRACT
            ):
                raise RuntimeError(
                    "VOS-optimized SAM2 service compile prewarm contract "
                    "does not cover both bbox and explicit-mask sessions: "
                    f"{health.get('compile_prewarm_contract')!r}"
                )
            shape = health.get("compile_prewarm_shape_hw")
            input_size_wh = health.get("compile_prewarm_input_size_wh")
            try:
                shape_hw = [int(shape[0]), int(shape[1])]
                expected_hw = [int(input_size_wh[1]), int(input_size_wh[0])]
            except (TypeError, ValueError, IndexError) as exc:
                raise RuntimeError(
                    "VOS-optimized SAM2 service compile prewarm shape is "
                    "missing or malformed"
                ) from exc
            if (
                len(shape) != 2
                or len(input_size_wh) != 2
                or min(expected_hw) <= 0
                or shape_hw != expected_hw
            ):
                raise RuntimeError(
                    "VOS-optimized SAM2 service compile prewarm shape does "
                    f"not match configured input: shape={shape!r}, "
                    f"input_size_wh={input_size_wh!r}"
                )
            initialized = health.get("initialized")
            if not isinstance(initialized, bool):
                raise RuntimeError(
                    "VOS-optimized SAM2 service initialized state is malformed"
                )
            if require_uninitialized_after_prewarm and initialized is not False:
                raise RuntimeError(
                    "VOS-optimized SAM2 service retained temporal state after "
                    "compile prewarm"
                )
            timing_fields = (
                "compile_prewarm_initialize_box_ms",
                "compile_prewarm_track_ms",
                "compile_prewarm_initialize_mask_ms",
                "compile_prewarm_mask_track_ms",
            )
            try:
                timings = [float(health[name]) for name in timing_fields]
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(
                    "VOS-optimized SAM2 service compile prewarm stage "
                    "timings are missing or malformed"
                ) from exc
            if any(not np.isfinite(value) or value < 0.0 for value in timings):
                raise RuntimeError(
                    "VOS-optimized SAM2 service compile prewarm stage "
                    f"timings are invalid: {dict(zip(timing_fields, timings))!r}"
                )
        return health

    def __enter__(self) -> "SAM2VideoServiceManager":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
