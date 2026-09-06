from __future__ import annotations

import contextlib
import hashlib
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from dynamic_pcd.segmentation.sam2_video_protocol import (
    SAM2VideoResult,
    validate_bgr_image,
    validate_bbox_xyxy,
    validate_frame_id,
    validate_mask,
)


DEFAULT_CHECKPOINT = (
    "/home/qiaoguanren/code/franka/third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt"
)
DEFAULT_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
PRODUCTION_VOS_COMPILE_MODE = "max-autotune-no-cudagraphs"
DIAGNOSTIC_VOS_COMPILE_MODE = "max-autotune"
ALLOWED_VOS_COMPILE_MODES = (
    PRODUCTION_VOS_COMPILE_MODE,
    DIAGNOSTIC_VOS_COMPILE_MODE,
)
PRODUCTION_VOS_COMPONENT_COMPILE_MODES = (
    ("image_encoder", PRODUCTION_VOS_COMPILE_MODE),
    ("memory_encoder", PRODUCTION_VOS_COMPILE_MODE),
    ("memory_attention", PRODUCTION_VOS_COMPILE_MODE),
    ("sam_prompt_encoder", PRODUCTION_VOS_COMPILE_MODE),
    ("sam_mask_decoder", PRODUCTION_VOS_COMPILE_MODE),
)
PRODUCTION_VOS_COMPONENT_DYNAMIC = (
    ("image_encoder", False),
    ("memory_encoder", False),
    # The number of temporal memories varies while the image-token grid is
    # fixed by ``image_size`` and the SAM2 stride below.
    ("memory_attention", True),
    ("sam_prompt_encoder", False),
    ("sam_mask_decoder", False),
)
SAM2_MEMORY_ATTENTION_STRIDE = 16
VOS_COMPILE_PREWARM_CONTRACT = (
    "initialize_box_track_reset_explicit_mask_track_reset_v1"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OfficialSAM2VideoBackend:
    """Append-only adapter around the official ``SAM2VideoPredictor``.

    Imports of Torch and SAM2 are deliberately delayed until :meth:`load`, so
    the Python 3.9 camera/client environment can import the protocol and
    service manager without installing the model stack.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        model_config: str = DEFAULT_MODEL_CONFIG,
        *,
        device: str = "cuda",
        image_size: int = 512,
        amp_dtype: str = "bfloat16",
        min_mask_area: int = 20,
        object_score_threshold: float = 0.0,
        reset_every_frames: int = 0,
        vos_optimized: bool = True,
        vos_compile_mode: str = PRODUCTION_VOS_COMPILE_MODE,
        prewarm_input_size_wh: Tuple[int, int] = (848, 480),
    ):
        self.checkpoint = str(checkpoint)
        self.model_config = str(model_config)
        self.device_name = str(device)
        self.image_size = int(image_size)
        if self.image_size <= 0 or self.image_size % 16 != 0:
            raise ValueError("image_size must be a positive multiple of 16")
        self.amp_dtype_name = str(amp_dtype).lower()
        if self.amp_dtype_name not in ("bfloat16", "float16"):
            raise ValueError("amp_dtype must be bfloat16 or float16")
        self.min_mask_area = max(1, int(min_mask_area))
        self.object_score_threshold = float(object_score_threshold)
        if not np.isfinite(self.object_score_threshold):
            raise ValueError("object_score_threshold must be finite")
        self.reset_every_frames = int(reset_every_frames)
        if self.reset_every_frames < 0 or self.reset_every_frames == 1:
            raise ValueError("reset_every_frames must be 0 or at least 2")
        if not isinstance(vos_optimized, bool):
            raise ValueError("vos_optimized must be true or false")
        self.vos_optimized = vos_optimized
        if (
            not isinstance(vos_compile_mode, str)
            or vos_compile_mode not in ALLOWED_VOS_COMPILE_MODES
        ):
            raise ValueError(
                "vos_compile_mode must be one of "
                + ", ".join(ALLOWED_VOS_COMPILE_MODES)
            )
        self.vos_compile_mode = vos_compile_mode
        if isinstance(prewarm_input_size_wh, (str, bytes)):
            raise ValueError(
                "prewarm_input_size_wh must contain positive integer width,height"
            )
        try:
            if len(prewarm_input_size_wh) != 2:
                raise ValueError
            prewarm_width = int(prewarm_input_size_wh[0])
            prewarm_height = int(prewarm_input_size_wh[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(
                "prewarm_input_size_wh must contain positive integer width,height"
            ) from exc
        if (
            isinstance(prewarm_input_size_wh[0], bool)
            or isinstance(prewarm_input_size_wh[1], bool)
            or prewarm_width <= 0
            or prewarm_height <= 0
            or float(prewarm_width) != float(prewarm_input_size_wh[0])
            or float(prewarm_height) != float(prewarm_input_size_wh[1])
        ):
            raise ValueError(
                "prewarm_input_size_wh must contain positive integer width,height"
            )
        self.prewarm_input_size_wh = (prewarm_width, prewarm_height)

        self.predictor = None
        self.state: Optional[Dict[str, Any]] = None
        self._torch = None
        self._functional = None
        self._device = None
        self._amp_dtype = None
        self._shape_hw: Optional[Tuple[int, int]] = None
        self._last_frame_id: Optional[int] = None
        self._internal_frame_idx = -1
        self._generation = 0
        self._model_load_ms = 0.0
        self._safe_history_frames = 16
        self._checkpoint_sha256: Optional[str] = None
        self._model_config_sha256: Optional[str] = None
        self._resolved_model_config_path: Optional[str] = None
        # ``vos_optimized=True`` uses torch.compile in the official SAM2
        # predictor.  Model construction alone does not compile the temporal
        # first-forward graphs, so a service must not advertise readiness just
        # because ``load`` returned.  ``compile_prewarm`` records the exact
        # deterministic initialize_box -> track -> reset proof performed before
        # the RPC socket is bound.
        self._compile_prewarm_required = bool(self.vos_optimized)
        self._compile_prewarm_completed = False
        self._compile_prewarm_ms = 0.0
        self._compile_prewarm_shape_hw: Optional[Tuple[int, int]] = None
        self._compile_prewarm_initialize_box_ms = 0.0
        self._compile_prewarm_track_ms = 0.0
        self._compile_prewarm_initialize_mask_ms = 0.0
        self._compile_prewarm_mask_track_ms = 0.0
        self._vos_rewrapped_components: Tuple[str, ...] = ()
        rope_side = self.image_size // SAM2_MEMORY_ATTENTION_STRIDE
        self._vos_memory_attention_rope_grid_hw = (rope_side, rope_side)
        self._vos_memory_attention_rope_cache_labels: Tuple[str, ...] = ()
        self._vos_memory_attention_rope_cache_token_counts: Tuple[int, ...] = ()
        self._vos_memory_attention_rope_caches_verified = False

    @property
    def loaded(self) -> bool:
        return self.predictor is not None

    @property
    def initialized(self) -> bool:
        return self.state is not None

    def load(self) -> None:
        if self.loaded:
            return
        checkpoint_path = Path(self.checkpoint).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"SAM2 checkpoint does not exist: {checkpoint_path}")

        load_start = time.perf_counter()
        import torch
        import torch.nn.functional as functional

        from sam2.build_sam import build_sam2_video_predictor
        import sam2
        import sam2.sam2_video_predictor as predictor_module

        model_config_path = (
            Path(sam2.__file__).resolve().parent / self.model_config
        ).resolve()
        if not model_config_path.is_file():
            raise FileNotFoundError(
                "SAM2 model config does not exist in the loaded package: "
                f"{model_config_path}"
            )
        # Hash the exact bytes immediately before model construction. The
        # validator compares these values with its independently pinned
        # acceptance contract; a service cannot pass by reporting paths alone.
        self._checkpoint_sha256 = _sha256_file(checkpoint_path.resolve())
        self._model_config_sha256 = _sha256_file(model_config_path)
        self._resolved_model_config_path = str(model_config_path)

        device = torch.device(self.device_name)
        if device.type != "cuda":
            raise RuntimeError(
                "online SAM2 service requires CUDA; use the lightweight tracker "
                "when CUDA is unavailable"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device.index} is unavailable; "
                f"device_count={torch.cuda.device_count()}"
            )

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        amp_dtype = (
            torch.bfloat16
            if self.amp_dtype_name == "bfloat16"
            else torch.float16
        )
        # The public generator wraps every range in tqdm.  A live service would
        # otherwise print a progress bar for every frame.
        predictor_module.tqdm = lambda iterable, **_kwargs: iterable
        predictor = build_sam2_video_predictor(
            self.model_config,
            str(checkpoint_path),
            device=device,
            apply_postprocessing=False,
            vos_optimized=self.vos_optimized,
            hydra_overrides_extra=[
                f"++model.image_size={self.image_size}",
                "++model.memory_attention.layer.self_attention.feat_sizes="
                f"[{self._vos_memory_attention_rope_grid_hw[1]},"
                f"{self._vos_memory_attention_rope_grid_hw[0]}]",
                "++model.memory_attention.layer.cross_attention.feat_sizes="
                f"[{self._vos_memory_attention_rope_grid_hw[1]},"
                f"{self._vos_memory_attention_rope_grid_hw[0]}]",
            ],
        )
        if self.vos_optimized:
            self._verify_vos_memory_attention_rope_caches(predictor)
            self._vos_rewrapped_components = self._rewrap_vos_components(
                predictor, torch
            )
        # ``apply_postprocessing=False`` leaves the predictor default at zero;
        # make this invariant explicit because the local CUDA extension is not
        # required by the service.
        predictor.fill_hole_area = 0

        self._torch = torch
        self._functional = functional
        self._device = device
        self._amp_dtype = amp_dtype
        self.predictor = predictor
        self._safe_history_frames = self._history_window(predictor)
        torch.cuda.synchronize(device)
        self._model_load_ms = (time.perf_counter() - load_start) * 1000.0

    def health(self) -> Dict[str, Any]:
        gpu_name = None
        if self.loaded:
            try:
                gpu_name = self._torch.cuda.get_device_name(self._device)
            except Exception:
                gpu_name = None
        return {
            "backend": "official-sam2-video-predictor",
            "loaded": self.loaded,
            "initialized": self.initialized,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self._checkpoint_sha256,
            "model_config": self.model_config,
            "model_config_sha256": self._model_config_sha256,
            "resolved_model_config_path": self._resolved_model_config_path,
            "device": self.device_name,
            "gpu_name": gpu_name,
            "image_size": self.image_size,
            "amp_dtype": self.amp_dtype_name,
            "tf32": True,
            "fill_hole_area": 0,
            "last_frame_id": self._last_frame_id,
            "internal_frame_idx": self._internal_frame_idx,
            "generation": self._generation,
            "safe_history_frames": self._safe_history_frames,
            "reset_every_frames": self.reset_every_frames,
            "vos_optimized": self.vos_optimized,
            "vos_compile_mode": self.vos_compile_mode,
            "vos_compile_cuda_graphs": bool(
                self.vos_optimized
                and self.vos_compile_mode == DIAGNOSTIC_VOS_COMPILE_MODE
            ),
            "vos_component_compile_modes": self._vos_component_compile_modes(),
            "vos_component_compile_dynamic": self._vos_component_compile_dynamic(),
            "vos_rewrapped_components": list(
                self._vos_rewrapped_components
            ),
            "vos_memory_attention_rope_grid_hw": list(
                self._vos_memory_attention_rope_grid_hw
            ),
            "vos_memory_attention_rope_expected_tokens": int(
                self._vos_memory_attention_rope_grid_hw[0]
                * self._vos_memory_attention_rope_grid_hw[1]
            ),
            "vos_memory_attention_rope_cache_count": len(
                self._vos_memory_attention_rope_cache_labels
            ),
            "vos_memory_attention_rope_cache_labels": list(
                self._vos_memory_attention_rope_cache_labels
            ),
            "vos_memory_attention_rope_cache_token_counts": list(
                self._vos_memory_attention_rope_cache_token_counts
            ),
            "vos_memory_attention_rope_caches_verified": (
                self._vos_memory_attention_rope_caches_verified
            ),
            "model_load_ms": self._model_load_ms,
            "compile_prewarm_required": self._compile_prewarm_required,
            "compile_prewarm_contract": VOS_COMPILE_PREWARM_CONTRACT,
            "compile_prewarm_input_size_wh": list(self.prewarm_input_size_wh),
            "compile_prewarm_completed": self._compile_prewarm_completed,
            "compile_prewarm_ms": self._compile_prewarm_ms,
            "compile_prewarm_shape_hw": (
                None
                if self._compile_prewarm_shape_hw is None
                else list(self._compile_prewarm_shape_hw)
            ),
            "compile_prewarm_initialize_box_ms": (
                self._compile_prewarm_initialize_box_ms
            ),
            "compile_prewarm_track_ms": self._compile_prewarm_track_ms,
            "compile_prewarm_initialize_mask_ms": (
                self._compile_prewarm_initialize_mask_ms
            ),
            "compile_prewarm_mask_track_ms": (
                self._compile_prewarm_mask_track_ms
            ),
        }

    def compile_prewarm(self) -> None:
        """Materialize VOS-optimized first-forward graphs before RPC READY.

        The official optimized predictor defers torch.compile work until the
        first prompted frame and temporal propagation.  Use a deterministic
        local frame at the explicitly configured camera input size. Exercise
        ``initialize_box -> track -> reset`` and the distinct explicit-mask
        graph ``initialize -> track -> reset`` before READY.  The second path
        is mandatory because bbox initialization later restores its accepted
        semantic mask with ``initialize``; leaving that branch lazy can spend
        its first RPC compiling and falsely appear as an empty timed-out seed.
        No camera frame or external input is consumed.

        The method is idempotent.  The non-optimized A/B backend has no compile
        obligation and returns without loading or mutating model state.
        """

        if not self._compile_prewarm_required or self._compile_prewarm_completed:
            return

        self.load()
        width, height = self.prewarm_input_size_wh
        shape_hw = (height, width)
        # A non-uniform but deterministic image avoids a degenerate all-zero
        # embedding while remaining independent of any camera, file or RNG.
        image = np.zeros((height, width, 3), dtype=np.uint8)
        y_grid = np.arange(height, dtype=np.uint16)[:, None]
        x_grid = np.arange(width, dtype=np.uint16)[None, :]
        image[..., 0] = ((x_grid + y_grid) % 251).astype(np.uint8)
        image[..., 1] = ((2 * x_grid + y_grid) % 253).astype(np.uint8)
        image[..., 2] = ((x_grid + 2 * y_grid) % 255).astype(np.uint8)
        margin_x = max(1, width // 4)
        margin_y = max(1, height // 4)
        bbox = np.asarray(
            [
                margin_x,
                margin_y,
                width - margin_x,
                height - margin_y,
            ],
            dtype=np.int32,
        )
        explicit_mask = np.zeros((height, width), dtype=np.uint8)
        explicit_mask[
            int(bbox[1]) : int(bbox[3]),
            int(bbox[0]) : int(bbox[2]),
        ] = 1

        started = time.perf_counter()
        initialize_ms = 0.0
        track_ms = 0.0
        initialize_mask_ms = 0.0
        mask_track_ms = 0.0
        succeeded = False
        try:
            initialize_started = time.perf_counter()
            initialized = self.initialize_box(image, bbox, frame_id=0)
            initialize_ms = (time.perf_counter() - initialize_started) * 1000.0
            if not isinstance(initialized, SAM2VideoResult):
                raise TypeError("compile prewarm initialize_box returned no result")

            track_started = time.perf_counter()
            tracked = self.track(image, frame_id=1)
            track_ms = (time.perf_counter() - track_started) * 1000.0
            if not isinstance(tracked, SAM2VideoResult):
                raise TypeError("compile prewarm track returned no result")

            # The explicit-mask prompt uses a different optimized control-flow
            # branch from a box prompt. Reset between the two sessions so the
            # same frame ids and a clean temporal state exactly mirror the
            # deployment bbox-reseed sequence.
            self.reset()
            initialize_mask_started = time.perf_counter()
            initialized_mask = self.initialize(
                image, explicit_mask, frame_id=0
            )
            initialize_mask_ms = (
                time.perf_counter() - initialize_mask_started
            ) * 1000.0
            if not isinstance(initialized_mask, SAM2VideoResult):
                raise TypeError(
                    "compile prewarm explicit-mask initialize returned no result"
                )
            if (
                initialized_mask.frame_id != 0
                or initialized_mask.internal_frame_idx < 0
                or not initialized_mask.valid
                or int(np.count_nonzero(initialized_mask.mask)) <= 0
            ):
                raise RuntimeError(
                    "compile prewarm explicit-mask initialize returned an "
                    "invalid/empty temporal session"
                )

            mask_track_started = time.perf_counter()
            mask_tracked = self.track(image, frame_id=1)
            mask_track_ms = (
                time.perf_counter() - mask_track_started
            ) * 1000.0
            if not isinstance(mask_tracked, SAM2VideoResult):
                raise TypeError(
                    "compile prewarm explicit-mask track returned no result"
                )
            if (
                mask_tracked.frame_id != 1
                or mask_tracked.internal_frame_idx < 0
                or not mask_tracked.valid
                or int(np.count_nonzero(mask_tracked.mask)) <= 0
            ):
                raise RuntimeError(
                    "compile prewarm explicit-mask track returned an "
                    "invalid/empty temporal session"
                )
            succeeded = True
        finally:
            # A compiled service still begins with no temporal identity.  This
            # reset is part of the readiness proof, not best-effort cleanup.
            self.reset()

        if succeeded:
            self._compile_prewarm_shape_hw = shape_hw
            self._compile_prewarm_initialize_box_ms = float(initialize_ms)
            self._compile_prewarm_track_ms = float(track_ms)
            self._compile_prewarm_initialize_mask_ms = float(
                initialize_mask_ms
            )
            self._compile_prewarm_mask_track_ms = float(mask_track_ms)
            self._compile_prewarm_ms = (
                time.perf_counter() - started
            ) * 1000.0
            self._compile_prewarm_completed = True

    def initialize(
        self, image_bgr: np.ndarray, mask: np.ndarray, frame_id: int
    ) -> SAM2VideoResult:
        self.load()
        image = validate_bgr_image(image_bgr)
        mask_u8 = validate_mask(
            mask, expected_shape=image.shape[:2], require_nonempty=True
        )
        frame = validate_frame_id(frame_id)
        total_start = time.perf_counter()
        self._drop_state()
        self._generation += 1
        timings, logits = self._start_state(image, mask_u8)
        # ``initialize(mask=...)`` is an explicit seed transaction, not a
        # same-frame tracking publication.  The predictor has already
        # consumed ``mask_u8`` via ``add_new_mask`` above; return those exact
        # accepted bytes as the ACK while retaining the model-derived object
        # score and temporal state.  Re-thresholding the initialization logits
        # can legitimately erode/dilate the seed, which made an ACK look like
        # a different target and broke the canonical local/remote seed
        # contract.  Subsequent ``track`` calls still publish model masks.
        result = self._make_result(logits, frame, timings, mask=mask_u8)
        self._shape_hw = tuple(image.shape[:2])
        self._last_frame_id = frame
        result.timings_ms["total"] = (time.perf_counter() - total_start) * 1000.0
        result.message = "initialized"
        return result

    def initialize_box(
        self,
        image_bgr: np.ndarray,
        bbox_xyxy: np.ndarray,
        frame_id: int,
    ) -> SAM2VideoResult:
        """Initialize the temporal predictor with SAM2's native box prompt."""

        self.load()
        image = validate_bgr_image(image_bgr)
        bbox = validate_bbox_xyxy(
            bbox_xyxy, expected_shape=image.shape[:2]
        )
        frame = validate_frame_id(frame_id)
        total_start = time.perf_counter()
        self._drop_state()
        self._generation += 1
        timings, logits = self._start_state_box(image, bbox)
        result = self._make_result(logits, frame, timings)
        self._shape_hw = tuple(image.shape[:2])
        self._last_frame_id = frame
        result.timings_ms["total"] = (
            time.perf_counter() - total_start
        ) * 1000.0
        result.message = (
            "initialized from box prompt"
            if result.valid
            else "SAM2 box prompt produced no valid object mask"
        )
        return result

    def track(self, image_bgr: np.ndarray, frame_id: int) -> SAM2VideoResult:
        if not self.initialized:
            raise RuntimeError("SAM2 video backend is not initialized")
        image = validate_bgr_image(image_bgr)
        frame = validate_frame_id(frame_id)
        if tuple(image.shape[:2]) != self._shape_hw:
            raise ValueError(
                f"track image shape {image.shape[:2]} does not match "
                f"initialized shape {self._shape_hw}"
            )
        if frame <= self._last_frame_id:
            raise ValueError(
                f"track frame_id {frame} must be greater than {self._last_frame_id}"
            )

        total_start = time.perf_counter()
        try:
            preprocess_start = time.perf_counter()
            with self._model_context():
                internal_idx = self._append_image(image)
            preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

            inference_start = time.perf_counter()
            with self._model_context():
                _, _, logits = self._propagate_one(internal_idx)
                mask = self._mask_from_logits(logits)
            inference_ms = (time.perf_counter() - inference_start) * 1000.0
            self._prune_state(internal_idx)
            self._internal_frame_idx = internal_idx
            self._last_frame_id = frame
            timings = {
                "preprocess": preprocess_ms,
                "propagate": inference_ms,
            }
            result = self._make_result(logits, frame, timings, mask=mask)

            if (
                self.reset_every_frames > 0
                and internal_idx + 1 >= self.reset_every_frames
                and result.valid
            ):
                reset_start = time.perf_counter()
                current_mask = result.mask.copy()
                self._drop_state()
                self._generation += 1
                self._start_state(image, current_mask)
                self._last_frame_id = frame
                result.state_reset = True
                result.internal_frame_idx = 0
                result.timings_ms["periodic_reset"] = (
                    time.perf_counter() - reset_start
                ) * 1000.0
            result.timings_ms["total"] = (
                time.perf_counter() - total_start
            ) * 1000.0
            return result
        except Exception:
            # Once an append/propagation fails, the state may have advanced only
            # partially.  Drop it so a retry cannot be paired with the wrong
            # temporal memory.
            self._drop_state()
            raise

    def reset(self) -> None:
        self._drop_state()
        self._generation += 1

    def close(self) -> None:
        self._drop_state()
        self.predictor = None
        self._torch = None
        self._functional = None
        self._device = None
        self._amp_dtype = None

    def _start_state(
        self, image_bgr: np.ndarray, mask: np.ndarray
    ) -> Tuple[Dict[str, float], Any]:
        state, preprocess_ms = self._create_initial_state(image_bgr)

        inference_start = time.perf_counter()
        with self._model_context():
            self.predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=mask)
            _, _, logits = self._propagate_one(0)
        inference_ms = (time.perf_counter() - inference_start) * 1000.0
        self._shape_hw = tuple(image_bgr.shape[:2])
        self._internal_frame_idx = 0
        return {"preprocess": preprocess_ms, "initialize": inference_ms}, logits

    def _start_state_box(
        self, image_bgr: np.ndarray, bbox_xyxy: np.ndarray
    ) -> Tuple[Dict[str, float], Any]:
        state, preprocess_ms = self._create_initial_state(image_bgr)

        inference_start = time.perf_counter()
        with self._model_context():
            self.predictor.add_new_points_or_box(
                state,
                frame_idx=0,
                obj_id=1,
                box=np.asarray(bbox_xyxy, dtype=np.float32),
            )
            _, _, logits = self._propagate_one(0)
        inference_ms = (time.perf_counter() - inference_start) * 1000.0
        self._shape_hw = tuple(image_bgr.shape[:2])
        self._internal_frame_idx = 0
        return {
            "preprocess": preprocess_ms,
            "initialize_box": inference_ms,
        }, logits

    def _create_initial_state(
        self, image_bgr: np.ndarray
    ) -> Tuple[Dict[str, Any], float]:
        preprocess_start = time.perf_counter()
        with self._model_context():
            first = self._preprocess(image_bgr)
            height, width = image_bgr.shape[:2]
            device = self._device
            state = {
                "images": [first],
                "num_frames": 1,
                "offload_video_to_cpu": False,
                "offload_state_to_cpu": False,
                "video_height": height,
                "video_width": width,
                "device": device,
                "storage_device": device,
                "point_inputs_per_obj": {},
                "mask_inputs_per_obj": {},
                "cached_features": {},
                "constants": {},
                "obj_id_to_idx": OrderedDict(),
                "obj_idx_to_id": OrderedDict(),
                "obj_ids": [],
                "output_dict_per_obj": {},
                "temp_output_dict_per_obj": {},
                "frames_tracked_per_obj": {},
            }
            self.state = state
            self.predictor._get_image_feature(state, frame_idx=0, batch_size=1)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
        return state, preprocess_ms

    def _append_image(self, image_bgr: np.ndarray) -> int:
        image = self._preprocess(image_bgr)
        self.state["images"].append(image)
        self.state["num_frames"] += 1
        return int(self.state["num_frames"] - 1)

    def _preprocess(self, image_bgr: np.ndarray):
        rgb = np.ascontiguousarray(image_bgr[..., ::-1])
        image = self._torch.from_numpy(rgb).permute(2, 0, 1)
        image = image.to(device=self._device, dtype=self._torch.float32).div_(255.0)
        image = self._functional.interpolate(
            image[None],
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0]
        mean = image.new_tensor((0.485, 0.456, 0.406))[:, None, None]
        std = image.new_tensor((0.229, 0.224, 0.225))[:, None, None]
        return image.sub_(mean).div_(std)

    def _propagate_one(self, frame_idx: int):
        generator = self.predictor.propagate_in_video(
            self.state,
            start_frame_idx=int(frame_idx),
            max_frame_num_to_track=1,
            reverse=False,
        )
        try:
            return next(generator)
        finally:
            generator.close()

    def _mask_from_logits(self, logits) -> np.ndarray:
        mask = (logits[0, 0] > 0.0).to(dtype=self._torch.uint8)
        return np.ascontiguousarray(mask.cpu().numpy())

    def _make_result(
        self,
        logits,
        frame_id: int,
        timings: Dict[str, float],
        *,
        mask: Optional[np.ndarray] = None,
    ) -> SAM2VideoResult:
        if mask is None:
            mask = self._mask_from_logits(logits)
        output = self._current_output(self._internal_frame_idx)
        score_tensor = None if output is None else output.get("object_score_logits")
        score = None if score_tensor is None else float(score_tensor.item())
        area = int(mask.sum())
        valid = area >= self.min_mask_area and (
            score is None or score >= self.object_score_threshold
        )
        return SAM2VideoResult(
            mask=mask,
            valid=valid,
            frame_id=frame_id,
            message="tracked" if valid else "SAM2 reported no valid object mask",
            timings_ms=dict(timings),
            internal_frame_idx=self._internal_frame_idx,
            object_score=score,
            mask_area=area,
        )

    def _current_output(self, frame_idx: int) -> Optional[Dict[str, Any]]:
        if self.state is None:
            return None
        output = self.state["output_dict_per_obj"].get(0)
        if output is None:
            return None
        return output["non_cond_frame_outputs"].get(
            frame_idx, output["cond_frame_outputs"].get(frame_idx)
        )

    def _prune_state(self, current_idx: int) -> None:
        if current_idx > 0:
            self.state["images"][current_idx - 1] = None
        oldest = max(0, current_idx - self._safe_history_frames + 1)
        for output in self.state["output_dict_per_obj"].values():
            non_conditioning = output["non_cond_frame_outputs"]
            for frame_idx in list(non_conditioning):
                if frame_idx < oldest:
                    del non_conditioning[frame_idx]
        for tracked in self.state["frames_tracked_per_obj"].values():
            for frame_idx in list(tracked):
                if frame_idx < oldest:
                    del tracked[frame_idx]

    def _drop_state(self) -> None:
        self.state = None
        self._shape_hw = None
        self._last_frame_id = None
        self._internal_frame_idx = -1

    def _model_context(self):
        stack = contextlib.ExitStack()
        stack.enter_context(self._torch.inference_mode())
        stack.enter_context(
            self._torch.autocast(
                device_type="cuda", dtype=self._amp_dtype, enabled=True
            )
        )
        return stack

    @staticmethod
    def _history_window(predictor) -> int:
        stride = max(1, int(predictor.memory_temporal_stride_for_eval))
        memory_horizon = max(1, (int(predictor.num_maskmem) - 1) * stride)
        pointer_horizon = max(1, int(predictor.max_obj_ptrs_in_encoder) - 1)
        return max(memory_horizon, pointer_horizon) + 1

    def _rewrap_vos_components(self, predictor: Any, torch: Any) -> Tuple[str, ...]:
        """Select the explicit Inductor mode before any lazy graph executes.

        The official VOS predictor constructs five ``torch.compile`` wrappers
        with ``mode='max-autotune'``.  In PyTorch 2.5 that mode enables CUDA
        Graph Trees.  SAM2's ``PositionEmbeddingSine._pe`` retains tensors in a
        Python cache across video frames, which is incompatible with graph
        storage reuse and can raise the documented "overwritten by a
        subsequent run" error.

        Rewrap the original callables before their first forward. Production
        keeps max autotuning without CUDA Graphs for all five components.
        Memory attention retains the official ``dynamic=True`` contract; the
        separately verified image-sized RoPE caches keep its fixed query-token
        grid out of symbolic ``sqrt`` guards. No component falls back to eager
        and compiler errors remain fatal. ``max-autotune`` remains an explicit
        diagnostic A/B only; production admission rejects it.
        """

        modules = {
            "image_encoder": predictor.image_encoder,
            "memory_encoder": predictor.memory_encoder,
            "memory_attention": predictor.memory_attention,
            "sam_prompt_encoder": predictor.sam_prompt_encoder,
            "sam_mask_decoder": predictor.sam_mask_decoder,
        }
        rewrapped = []
        component_modes = self._vos_component_compile_modes()
        component_dynamic = self._vos_component_compile_dynamic()
        for name, _dynamic in PRODUCTION_VOS_COMPONENT_DYNAMIC:
            module = modules[name]
            compiled_forward = module.forward
            original = getattr(
                compiled_forward, "_torchdynamo_orig_callable", None
            )
            if not callable(original):
                raise RuntimeError(
                    "official optimized SAM2 component is not an unexecuted "
                    f"torch.compile wrapper: {name}"
                )
            module.forward = torch.compile(
                original,
                mode=component_modes[name],
                fullgraph=True,
                dynamic=component_dynamic[name],
            )
            rewrapped.append(name)
        return tuple(rewrapped)

    def _vos_component_compile_modes(self) -> Dict[str, str]:
        if not self.vos_optimized:
            return {}
        if self.vos_compile_mode == PRODUCTION_VOS_COMPILE_MODE:
            return dict(PRODUCTION_VOS_COMPONENT_COMPILE_MODES)
        return {
            name: self.vos_compile_mode
            for name, _mode in PRODUCTION_VOS_COMPONENT_COMPILE_MODES
        }

    def _vos_component_compile_dynamic(self) -> Dict[str, bool]:
        if not self.vos_optimized:
            return {}
        return dict(PRODUCTION_VOS_COMPONENT_DYNAMIC)

    def _verify_vos_memory_attention_rope_caches(self, predictor: Any) -> None:
        """Prove that dynamic memory attention never needs a symbolic RoPE resize.

        The upstream SAM2.1 configs pin both RoPE caches to 64x64 for the
        default 1024px model.  This service overrides the model to 512px.  If
        only ``model.image_size`` is changed, the first temporal frame enters
        ``math.sqrt(q.shape[-2])`` while memory attention is compiled with
        ``dynamic=True``.  PyTorch 2.5.1 then emits an invalid
        ``OpaqueUnaryFn_sqrt`` guard.  The Hydra overrides above construct the
        caches at the deployment grid; this check makes that compile contract
        fail closed before any wrapper is executed.
        """

        memory_attention = getattr(predictor, "memory_attention", None)
        layers = getattr(memory_attention, "layers", None)
        if layers is None:
            raise RuntimeError(
                "optimized SAM2 memory_attention exposes no layers for RoPE "
                "cache verification"
            )
        layers = list(layers)
        configured_num_layers = getattr(memory_attention, "num_layers", None)
        if (
            not layers
            or configured_num_layers is None
            or int(configured_num_layers) != len(layers)
        ):
            raise RuntimeError(
                "optimized SAM2 memory_attention layer count is inconsistent: "
                f"num_layers={configured_num_layers!r}, layers={len(layers)}"
            )

        expected_tokens = int(
            self._vos_memory_attention_rope_grid_hw[0]
            * self._vos_memory_attention_rope_grid_hw[1]
        )
        labels = []
        token_counts = []
        for layer_index, layer in enumerate(layers):
            for label, attribute in (
                ("self_attention", "self_attn"),
                ("cross_attention", "cross_attn_image"),
            ):
                attention = getattr(layer, attribute, None)
                freqs_cis = getattr(attention, "freqs_cis", None)
                shape = getattr(freqs_cis, "shape", None)
                try:
                    normalized_shape = tuple(int(value) for value in shape)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "optimized SAM2 RoPE cache has no concrete tensor "
                        f"shape: layer={layer_index} attention={label} "
                        f"shape={shape!r}"
                    ) from exc
                if len(normalized_shape) != 2:
                    raise RuntimeError(
                        "optimized SAM2 RoPE cache must be rank 2: "
                        f"layer={layer_index} attention={label} "
                        f"shape={normalized_shape!r}"
                    )
                token_count = normalized_shape[0]
                if token_count != expected_tokens:
                    raise RuntimeError(
                        "optimized SAM2 RoPE cache token grid differs from "
                        "the compiled image-token grid: "
                        f"layer={layer_index} attention={label} "
                        f"tokens={token_count}, expected={expected_tokens}, "
                        "grid_hw="
                        f"{self._vos_memory_attention_rope_grid_hw!r}"
                    )
                labels.append(f"layer{layer_index}.{label}")
                token_counts.append(token_count)

        self._vos_memory_attention_rope_cache_labels = tuple(labels)
        self._vos_memory_attention_rope_cache_token_counts = tuple(token_counts)
        self._vos_memory_attention_rope_caches_verified = True
