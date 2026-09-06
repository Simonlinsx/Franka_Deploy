"""Replay the production guarded_v2 mask path on old colour-only videos.

The historical test videos predate lossless RGB-D recording.  This module
therefore supplies a spatially uniform, neutral depth image solely to exercise
the *2-D* production tracker/SAM2/publication state machine.  Its artifacts are
not evidence about depth gating, point-cloud geometry, camera calibration, or
robot-frame alignment.

No RealSense, Franka, or RH56 interface is opened.  ``ObjectPCDProvider`` gets
an explicitly supplied in-memory camera, so its normal RealSense constructor
path is never used.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Optional

import cv2
import numpy as np

from dynamic_pcd.config import load_config
from dynamic_pcd.evaluation.guarded_v2_offline import (
    _bbox,
    _component_proxy,
    _sample_indices,
)
from dynamic_pcd.evaluation.guarded_v2_source_provenance import (
    GuardedV2SourceProvenanceError,
    validate_contract_source_provenance,
)
from dynamic_pcd.provider.object_pcd_provider import ObjectPCDProvider
from dynamic_pcd.types import CameraIntrinsics, RGBDFrame

REPLAY_SCHEMA = "guarded_v2_real_rgb_replay_v1"
PRODUCTION_REPLAY_SUMMARY_SCHEMA = "guarded_v2_real_rgb_replay_summary_v2"
PRODUCTION_CASE_NAMES = (
    "fast_green_ball_entry",
    "rolling_green_ball",
    "rolling_red_cylinder",
    "static_green_ball_rh56",
    "rh56_heavy_occlusion",
)
PRODUCTION_NEUTRAL_DEPTH_M = 1.0
PRODUCTION_SAM2_IMAGE_SIZE = 512
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


class RealRGBReplayError(RuntimeError):
    """Raised when a replay case or produced candidate is incomplete."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _video_metadata(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RealRGBReplayError(f"cannot open video metadata: {path}")
    try:
        metadata = {
            "fps_hz": float(capture.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "size_wh": [
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ],
        }
    finally:
        capture.release()
    if (
        not np.isfinite(metadata["fps_hz"])
        or metadata["fps_hz"] <= 0.0
        or metadata["frame_count"] <= 0
        or any(int(value) <= 0 for value in metadata["size_wh"])
    ):
        raise RealRGBReplayError(f"video has invalid metadata: {path}")
    metadata["sha256"] = _sha256(path)
    return metadata


def _validate_video_pin(
    *,
    case_name: str,
    description: str,
    metadata: dict[str, Any],
    expected_sha256: Any,
    expected_fps_hz: Any,
    expected_frame_count: Any,
    expected_size_wh: Any,
) -> bool:
    """Validate every supplied pin and report whether the full pin set exists."""

    supplied = (
        expected_sha256 is not None,
        expected_fps_hz is not None,
        expected_frame_count is not None,
        expected_size_wh is not None,
    )
    if (
        expected_sha256 is not None
        and str(expected_sha256).lower() != str(metadata["sha256"]).lower()
    ):
        raise RealRGBReplayError(f"case {case_name}: {description} SHA-256 mismatch")
    if expected_fps_hz is not None and not math.isclose(
        float(expected_fps_hz), float(metadata["fps_hz"]), abs_tol=1e-6
    ):
        raise RealRGBReplayError(
            f"case {case_name}: {description} FPS={metadata['fps_hz']} != "
            f"pinned {expected_fps_hz}"
        )
    if expected_frame_count is not None and int(expected_frame_count) != int(
        metadata["frame_count"]
    ):
        raise RealRGBReplayError(
            f"case {case_name}: {description} frame_count="
            f"{metadata['frame_count']} != pinned {expected_frame_count}"
        )
    if expected_size_wh is not None:
        if not isinstance(expected_size_wh, list) or len(expected_size_wh) != 2:
            raise RealRGBReplayError(
                f"case {case_name}: {description} size pin must be [width,height]"
            )
        if [int(value) for value in expected_size_wh] != list(metadata["size_wh"]):
            raise RealRGBReplayError(
                f"case {case_name}: {description} size={metadata['size_wh']} != "
                f"pinned {expected_size_wh}"
            )
    return all(supplied)


def _resolve_model_config_path(raw: Any, *, config_path: Path) -> Path:
    path = Path(str(raw)).expanduser()
    candidates = (
        [path]
        if path.is_absolute()
        else [
            config_path.parent / path,
            Path.cwd() / path,
            Path(__file__).resolve().parents[3]
            / "third_party"
            / "sam2"
            / "sam2"
            / path,
        ]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise RealRGBReplayError(
        f"cannot resolve online_sam2.model_config={raw!r}; tried "
        + ", ".join(str(item.resolve()) for item in candidates)
    )


def _effective_config_provenance(
    cfg: dict[str, Any], *, config_path: Path
) -> dict[str, Any]:
    online = dict(cfg.get("online_sam2") or {})
    checkpoint = Path(str(online.get("checkpoint", ""))).expanduser().resolve()
    if not checkpoint.is_file():
        raise RealRGBReplayError(f"missing online SAM2 checkpoint: {checkpoint}")
    model_config = _resolve_model_config_path(
        online.get("model_config"), config_path=config_path
    )
    return {
        "source_config": str(config_path),
        "source_config_sha256": _sha256(config_path),
        "canonical_effective_config_sha256": _canonical_json_sha256(cfg),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "model_config": str(model_config),
        "model_config_sha256": _sha256(model_config),
        "vos_optimized": online.get("vos_optimized"),
        "vos_compile_mode": online.get("vos_compile_mode"),
    }


def _candidate_artifact_manifest(
    *, case_output: Path, sampled_source_indices: list[int],
) -> dict[str, Any]:
    masks = []
    for frame_index, source_frame_index in enumerate(sampled_source_indices):
        relative = Path("masks") / f"{frame_index:06d}.png"
        path = case_output / relative
        if not path.is_file():
            raise RealRGBReplayError(f"missing produced mask: {path}")
        masks.append(
            {
                "frame_index": int(frame_index),
                "source_frame_index": int(source_frame_index),
                "path": relative.as_posix(),
                "sha256": _sha256(path),
            }
        )
    states_path = case_output / "states.jsonl"
    if not states_path.is_file():
        raise RealRGBReplayError(f"missing produced states: {states_path}")
    content = {
        "masks": masks,
        "states": {"path": "states.jsonl", "sha256": _sha256(states_path),},
    }
    return {
        **content,
        "aggregate_sha256": _canonical_json_sha256(content),
    }


def _bbox_prewarm_provenance(provider: Any) -> dict[str, Any] | None:
    evidence = getattr(provider, "last_online_sam2_bbox_prewarm_evidence", None)
    if evidence is None:
        return None
    return {
        "enabled": bool(getattr(evidence, "enabled", False)),
        "attempted": bool(getattr(evidence, "attempted", False)),
        "passed": bool(getattr(evidence, "passed", False)),
        "seed_frame_id": getattr(evidence, "seed_frame_id", None),
        "discarded_track_frame_id": getattr(evidence, "discarded_track_frame_id", None),
        "discarded_track_valid": bool(
            getattr(evidence, "discarded_track_valid", False)
        ),
        "discarded_track_rpc_ms": float(
            getattr(evidence, "discarded_track_rpc_ms", 0.0)
        ),
        "discarded_track_frame_ids": list(
            getattr(evidence, "discarded_track_frame_ids", ())
        ),
        "discarded_track_rpc_ms_values": list(
            getattr(evidence, "discarded_track_rpc_ms_values", ())
        ),
        "attempt_count": int(getattr(evidence, "attempt_count", 0)),
        "required_stable_tracks": int(
            getattr(evidence, "required_stable_tracks", 0)
        ),
        "achieved_stable_tracks": int(
            getattr(evidence, "achieved_stable_tracks", 0)
        ),
        "max_attempts": int(getattr(evidence, "max_attempts", 0)),
        "max_rpc_ms": float(getattr(evidence, "max_rpc_ms", 0.0)),
        "exact_reseed_rpc_ms": float(getattr(evidence, "exact_reseed_rpc_ms", 0.0)),
        "total_ms": float(getattr(evidence, "total_ms", 0.0)),
        "status": str(getattr(evidence, "status", "unavailable")),
    }


def _service_health_provenance(provider: Any) -> dict[str, Any] | None:
    """Persist the exact service identity that provider.start() admitted."""

    health = getattr(provider, "_online_sam2_service_health", None)
    if not isinstance(health, dict):
        return None
    fields = (
        "backend",
        "loaded",
        "initialized",
        "checkpoint",
        "checkpoint_sha256",
        "model_config",
        "model_config_sha256",
        "resolved_model_config_path",
        "device",
        "gpu_name",
        "image_size",
        "amp_dtype",
        "tf32",
        "fill_hole_area",
        "vos_optimized",
        "vos_compile_mode",
        "vos_compile_cuda_graphs",
        "vos_component_compile_modes",
        "vos_component_compile_dynamic",
        "vos_memory_attention_rope_grid_hw",
        "vos_memory_attention_rope_expected_tokens",
        "vos_memory_attention_rope_cache_count",
        "vos_memory_attention_rope_cache_token_counts",
        "vos_memory_attention_rope_caches_verified",
        "compile_prewarm_required",
        "compile_prewarm_contract",
        "compile_prewarm_input_size_wh",
        "compile_prewarm_completed",
        "compile_prewarm_ms",
        "compile_prewarm_shape_hw",
        "compile_prewarm_initialize_box_ms",
        "compile_prewarm_track_ms",
        "compile_prewarm_initialize_mask_ms",
        "compile_prewarm_mask_track_ms",
    )
    # JSON round-trip both proves serializability and detaches nested lists or
    # dictionaries from the live manager result.
    return json.loads(
        json.dumps(
            {name: health.get(name) for name in fields},
            sort_keys=True,
            allow_nan=False,
        )
    )


def _appearance_probability_cache_provenance(provider: Any) -> dict[str, int]:
    """Expose cumulative, decision-neutral cache evidence for replay audits."""

    return {
        "build_count": int(
            getattr(provider, "_appearance_probability_roi_build_count", 0)
        ),
        "query_count": int(
            getattr(provider, "_appearance_probability_roi_query_count", 0)
        ),
        "hit_count": int(
            getattr(provider, "_appearance_probability_roi_hit_count", 0)
        ),
    }


def _appearance_gate_cache_provenance(provider: Any) -> dict[str, int]:
    """Expose decision-neutral exact-mask gate-cache hit/miss evidence."""

    return {
        "stats_build_count": int(
            getattr(provider, "_appearance_stats_cache_build_count", 0)
        ),
        "stats_hit_count": int(
            getattr(provider, "_appearance_stats_cache_hit_count", 0)
        ),
        "stats_miss_count": int(
            getattr(provider, "_appearance_stats_cache_miss_count", 0)
        ),
        "stats_bypass_count": int(
            getattr(provider, "_appearance_stats_cache_bypass_count", 0)
        ),
        "supported_build_count": int(
            getattr(provider, "_appearance_supported_cache_build_count", 0)
        ),
        "supported_hit_count": int(
            getattr(provider, "_appearance_supported_cache_hit_count", 0)
        ),
        "supported_miss_count": int(
            getattr(provider, "_appearance_supported_cache_miss_count", 0)
        ),
        "supported_bypass_count": int(
            getattr(provider, "_appearance_supported_cache_bypass_count", 0)
        ),
    }


def _bootstrap_commissioning_provenance(provider: Any) -> dict[str, Any]:
    """Return JSON-safe provider bootstrap provenance for mask-only replay."""

    state = getattr(provider, "guarded_v2_bootstrap_state", None)
    if isinstance(state, dict):
        return dict(state)
    return {
        "phase": "unavailable",
        "status": "provider exposes no bootstrap commissioning state",
        "target_generation": 0,
        "seed_frame_id": None,
        "seed_area": 0.0,
        "seed_bbox_area": 0.0,
        "candidate_frame_id": None,
        "candidate_step_count": None,
        "commissioned_frame_id": None,
        "commission_count": 0,
        "provider_initial_area": 0.0,
        "provider_initial_bbox_area": 0.0,
        "tracker_initial_area": None,
        "tracker_initial_bbox_area": None,
    }


def _latency_evaluable_after_bootstrap(
    provider: Any,
    *,
    frame_id: int,
) -> bool:
    """Start formal 20 Hz timing only after camera-only commissioning.

    Seed/bootstrap work happens before any policy action may consume a mask.
    The frame which atomically commissions guarded-v2 is readiness work too;
    the next fresh frame is the first policy-facing timing sample.  Providers
    without bootstrap provenance retain the legacy evaluator behavior.
    """

    state = getattr(provider, "guarded_v2_bootstrap_state", None)
    if not isinstance(state, dict):
        return True
    commissioned_frame_id = state.get("commissioned_frame_id")
    if commissioned_frame_id is None:
        return False
    return int(frame_id) > int(commissioned_frame_id)


def _resolve_path(raw: Any, base: Path, *, description: str) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.exists():
        raise RealRGBReplayError(f"missing {description}: {path}")
    return path


def _read_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RealRGBReplayError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise RealRGBReplayError(f"video returned an invalid frame: {path}")
            frames.append(np.ascontiguousarray(frame))
    finally:
        capture.release()
    if not frames or not np.isfinite(fps) or fps <= 0.0:
        raise RealRGBReplayError(f"video has no valid frames/FPS: {path}")
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise RealRGBReplayError(f"video changes resolution: {path}")
    return frames, fps


def _read_binary_video_frame(
    path: Path, frame_index: int, threshold: int
) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RealRGBReplayError(f"cannot open seed mask video: {path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RealRGBReplayError(f"seed mask video has no frame {frame_index}: {path}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray > int(threshold)


def _expand_bbox(
    bbox: tuple[int, int, int, int], *, padding_px: int, width: int, height: int
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.asarray(
        [
            max(0, x1 - padding_px),
            max(0, y1 - padding_px),
            min(width, x2 + padding_px),
            min(height, y2 + padding_px),
        ],
        dtype=np.int32,
    )


def _proxy_masks(frames: list[np.ndarray], config: dict[str, Any]) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    previous_frame: Optional[np.ndarray] = None
    previous_center: Optional[tuple[float, float]] = None
    for frame in frames:
        mask, previous_center = _component_proxy(
            frame, previous_frame, config, previous_center
        )
        masks.append(np.asarray(mask, dtype=bool))
        previous_frame = frame
    return masks


def _conservative_proxy_diagnostics(
    case: dict[str, Any],
    frames: list[np.ndarray],
    candidate_masks: list[np.ndarray],
    *,
    seed_position: int,
) -> dict[str, Any] | None:
    """Return a conservative 2-D drift diagnostic, never a ground-truth IoU."""

    config = case.get("target_proxy")
    if not isinstance(config, dict):
        return None
    previous_frame = None if seed_position == 0 else frames[seed_position - 1]
    previous_center: Optional[tuple[float, float]] = None
    recalls: list[float] = []
    contaminations: list[float] = []
    covered = 0
    visible = 0
    dilation_px = int(case.get("contamination_proxy_dilation_px", 8))
    kernel_size = max(1, 2 * dilation_px + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    for position in range(seed_position, len(frames)):
        proxy, previous_center = _component_proxy(
            frames[position], previous_frame, config, previous_center
        )
        previous_frame = frames[position]
        proxy = np.asarray(proxy, dtype=bool)
        if int(proxy.sum()) < int(config.get("min_visible_area_px", 1)):
            continue
        candidate = np.asarray(candidate_masks[position], dtype=bool)
        intersection = int(np.count_nonzero(candidate & proxy))
        recall = intersection / float(max(1, int(proxy.sum())))
        visible += 1
        covered += int(recall >= 0.20)
        recalls.append(recall)
        allowed = cv2.dilate(proxy.astype(np.uint8), kernel) > 0
        contaminations.append(
            float(np.count_nonzero(candidate & ~allowed))
            / float(max(1, int(candidate.sum())))
        )
    if not recalls:
        return {
            "scope": "conservative_colour_motion_proxy_not_ground_truth",
            "visible_proxy_frames": 0,
            "target_coverage_at_20pct": None,
            "proxy_recall_p05": None,
            "proxy_recall_p50": None,
            "contamination_proxy_p95": None,
        }
    return {
        "scope": "conservative_colour_motion_proxy_not_ground_truth",
        "visible_proxy_frames": visible,
        "target_coverage_at_20pct": covered / float(visible),
        "proxy_recall_p05": float(np.percentile(recalls, 5)),
        "proxy_recall_p50": float(np.percentile(recalls, 50)),
        "contamination_proxy_p95": float(np.percentile(contaminations, 95)),
    }


def _boolean_run_lengths(values: Iterable[bool]) -> dict[str, int]:
    """Summarize valid/invalid continuity without hiding empty frames."""

    longest_valid = 0
    longest_invalid = 0
    current_value: Optional[bool] = None
    current_length = 0
    transitions = 0
    for raw_value in values:
        value = bool(raw_value)
        if current_value is None or value != current_value:
            if current_value is not None:
                transitions += 1
            current_value = value
            current_length = 1
        else:
            current_length += 1
        if value:
            longest_valid = max(longest_valid, current_length)
        else:
            longest_invalid = max(longest_invalid, current_length)
    return {
        "longest_consecutive_valid_frames": int(longest_valid),
        "longest_consecutive_invalid_frames": int(longest_invalid),
        "validity_transitions": int(transitions),
    }


class NeutralDepthVideoCamera:
    """RealSense-compatible in-memory source with explicitly synthetic depth."""

    def __init__(
        self,
        frames: list[np.ndarray],
        source_indices: list[int],
        *,
        next_position: int,
        rate_hz: float,
        source_rate_hz: float,
        depth_m: float,
        device_serial: str,
        realtime: bool,
    ) -> None:
        if len(frames) != len(source_indices) or not frames:
            raise ValueError("frames/source_indices must be non-empty and aligned")
        if not 0 <= next_position <= len(frames):
            raise ValueError("next_position is outside the sampled sequence")
        if not np.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("rate_hz must be finite and positive")
        if not np.isfinite(source_rate_hz) or source_rate_hz <= 0.0:
            raise ValueError("source_rate_hz must be finite and positive")
        if not np.isfinite(depth_m) or not 0.25 <= depth_m <= 1.20:
            raise ValueError("neutral depth must be in 0.25..1.20m")
        self.frames = frames
        self.source_indices = source_indices
        self.start_position = int(next_position)
        self.position = int(next_position)
        self.rate_hz = float(rate_hz)
        self.source_rate_hz = float(source_rate_hz)
        self.synthetic_depth_m = float(depth_m)
        self.device_serial = str(device_serial)
        self.device_name = "offline-neutral-depth-colour-video"
        self.device_firmware_version = "not-applicable"
        self.device_usb_type_descriptor = "not-applicable"
        self.sdk_version = "not-applicable"
        self.depth_scale = 0.001
        height, width = frames[0].shape[:2]
        self.intrinsics = CameraIntrinsics(
            width=width,
            height=height,
            fx=float(width),
            fy=float(width),
            ppx=0.5 * float(width - 1),
            ppy=0.5 * float(height - 1),
            model="offline-neutral-depth",
        )
        self._depth_raw = np.full(
            (height, width),
            int(round(self.synthetic_depth_m / self.depth_scale)),
            dtype=np.uint16,
        )
        # Pacing happens explicitly in ``wait_until_position`` immediately
        # before the timed provider call.  Keeping the wait outside
        # ``processing_ms`` makes that field comparable to historical model
        # compute latency instead of mechanically reporting the 50 ms policy
        # period itself.
        self.preserve_schedule = bool(realtime)
        self.started = False
        self._schedule_started_s = 0.0
        self._schedule_position = int(next_position)

    def start(self) -> None:
        self.position = self.start_position
        self.started = True
        self.rebase_timing()

    def rebase_timing(self) -> None:
        self._schedule_started_s = time.monotonic()
        self._schedule_position = self.position

    def stop(self) -> None:
        self.started = False

    def wait_until_position(self, position: int) -> float:
        if not self.preserve_schedule:
            return 0.0
        due = self._schedule_started_s + (
            float(position - self._schedule_position) / self.rate_hz
        )
        delay = due - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)
            return delay * 1000.0
        return 0.0

    def frame_at(self, position: int) -> RGBDFrame:
        source_index = int(self.source_indices[position])
        # Wall scheduling is the requested 20 Hz policy cadence, while the
        # device timestamp must retain the recorded camera's true 30 Hz
        # source time.  Sampling 30 Hz at 20 Hz alternates one- and two-frame
        # gaps; replacing both with a synthetic 50 ms step corrupts velocity
        # and acceleration gates that production evaluates from RealSense
        # sensor timestamps.
        timestamp = 1000.0 + float(source_index) / self.source_rate_hz
        now = time.monotonic()
        return RGBDFrame(
            color_bgr=self.frames[position],
            depth_raw=self._depth_raw,
            depth_scale=self.depth_scale,
            intrinsics=self.intrinsics,
            timestamp=timestamp,
            frame_id=source_index,
            retrieved_at_s=timestamp,
            retrieved_monotonic_s=now,
            sensor_frame_number=source_index,
            depth_sensor_frame_number=source_index,
            depth_timestamp_s=timestamp,
            color_depth_timestamp_skew_s=0.0,
            color_depth_epoch_timestamp_skew_s=0.0,
            timestamp_domain="offline_source_video",
        )

    def get_frame(self, timeout_ms: int = 1000) -> RGBDFrame:
        del timeout_ms
        if not self.started:
            self.start()
        if self.position >= len(self.frames):
            raise EOFError("end of sampled colour video")
        frame = self.frame_at(self.position)
        self.position += 1
        return frame


def _seed_mask(
    case: dict[str, Any],
    *,
    base: Path,
    source_frames: list[np.ndarray],
    sampled_frames: list[np.ndarray],
    sampled_source_indices: list[int],
) -> tuple[int, np.ndarray, str]:
    seed_source_frame = int(case["seed_source_frame"])
    try:
        seed_position = sampled_source_indices.index(seed_source_frame)
    except ValueError as exc:
        raise RealRGBReplayError(
            f"case {case.get('name')}: seed source frame {seed_source_frame} "
            "is absent from the exact evaluation sequence"
        ) from exc
    seed = case.get("seed")
    if not isinstance(seed, dict):
        raise RealRGBReplayError(f"case {case.get('name')}: seed must be an object")
    kind = str(seed.get("kind", "target_proxy"))
    if kind == "target_proxy":
        proxy_config = case.get("target_proxy")
        if not isinstance(proxy_config, dict):
            raise RealRGBReplayError(
                f"case {case.get('name')}: target_proxy seed needs target_proxy"
            )
        masks = _proxy_masks(sampled_frames[: seed_position + 1], proxy_config)
        mask = masks[-1]
        source = "conservative_target_colour_proxy"
    elif kind == "target_proxy_single_frame":
        proxy_config = case.get("target_proxy")
        if not isinstance(proxy_config, dict):
            raise RealRGBReplayError(
                f"case {case.get('name')}: target_proxy seed needs target_proxy"
            )
        previous = None if seed_position == 0 else sampled_frames[seed_position - 1]
        mask, _center = _component_proxy(
            sampled_frames[seed_position], previous, proxy_config, None
        )
        source = "single_frame_conservative_target_colour_motion_proxy"
    elif kind == "mask_video_frame":
        path = _resolve_path(
            seed.get("path"), base, description="seed binary mask video"
        )
        seed_video_frame = int(seed.get("frame_index", seed_source_frame))
        if seed_video_frame != seed_source_frame:
            raise RealRGBReplayError(
                f"case {case.get('name')}: seed mask frame {seed_video_frame} "
                f"must equal seed_source_frame {seed_source_frame}; future/past "
                "seed lookahead is forbidden"
            )
        mask = _read_binary_video_frame(
            path, seed_video_frame, int(seed.get("threshold", 96)),
        )
        source = f"saved_verified_mask_video:{path.name}"
    else:
        raise RealRGBReplayError(
            f"case {case.get('name')}: unsupported seed kind={kind!r}"
        )
    expected_shape = sampled_frames[seed_position].shape[:2]
    if mask.shape != expected_shape or int(np.count_nonzero(mask)) == 0:
        raise RealRGBReplayError(
            f"case {case.get('name')}: seed mask is empty or has shape "
            f"{mask.shape}, expected {expected_shape}"
        )
    # Explicitly ensure a saved source-frame seed refers to the same underlying
    # RGB frame.  ``source_frames`` is otherwise retained only to read mask
    # videos at their original (30 Hz) indexing.
    if seed_source_frame >= len(source_frames):
        raise RealRGBReplayError(
            f"case {case.get('name')}: seed frame exceeds source video"
        )
    return seed_position, np.asarray(mask, dtype=bool), source


def _load_cases(path: Path) -> tuple[dict[str, Any], Path]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != REPLAY_SCHEMA:
        raise RealRGBReplayError(f"replay manifest schema must be {REPLAY_SCHEMA!r}")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RealRGBReplayError("replay manifest cases must be non-empty")
    names = [str(case.get("name", "")) for case in cases if isinstance(case, dict)]
    if len(names) != len(cases) or any(not name for name in names):
        raise RealRGBReplayError("every replay case needs a non-empty name")
    if len(names) != len(set(names)):
        raise RealRGBReplayError("replay case names must be unique")
    return payload, resolved.parent


def replay_case(
    case: dict[str, Any],
    *,
    manifest_base: Path,
    output_root: Path,
    config_path: Path,
    neutral_depth_m: float,
    realtime: bool,
    diagnostic_semantic_sam2_direct: bool,
    production_contract: dict[str, Any] | None,
    implementation_source_provenance: dict[str, Any] | None,
    full_suite_requested: bool,
    sam2_image_size_override: Optional[int] = None,
) -> dict[str, Any]:
    name = str(case["name"])
    output = output_root / name
    if output.exists():
        raise FileExistsError(output)
    masks_dir = output / "masks"
    masks_dir.mkdir(parents=True)

    video_path = _resolve_path(case.get("video"), manifest_base, description="video")
    source_video_metadata = _video_metadata(video_path)
    source_video_pin_complete = _validate_video_pin(
        case_name=name,
        description="source video",
        metadata=source_video_metadata,
        expected_sha256=case.get("source_video_sha256"),
        expected_fps_hz=case.get("source_video_fps_hz"),
        expected_frame_count=case.get("source_video_frame_count"),
        expected_size_wh=case.get("source_video_size_wh"),
    )
    source_frames, source_fps = _read_video(video_path)
    if len(source_frames) != int(
        source_video_metadata["frame_count"]
    ) or not math.isclose(
        source_fps, float(source_video_metadata["fps_hz"]), abs_tol=1e-6
    ):
        raise RealRGBReplayError(
            f"case {name}: decoded source metadata changed while opening video"
        )
    evaluation_fps = float(case.get("evaluation_fps_hz", 20.0))
    if abs(evaluation_fps - 20.0) > 1e-9:
        raise RealRGBReplayError(
            f"case {name}: this acceptance runner requires exact 20 Hz"
        )
    sampled_indices = _sample_indices(
        len(source_frames),
        source_fps=source_fps,
        start_frame=int(case.get("source_start_frame", 0)),
        evaluation_fps=evaluation_fps,
    )
    sampled_frames = [source_frames[index] for index in sampled_indices]
    seed_position, seed_mask, seed_source = _seed_mask(
        case,
        base=manifest_base,
        source_frames=source_frames,
        sampled_frames=sampled_frames,
        sampled_source_indices=sampled_indices,
    )
    seed_spec = dict(case.get("seed") or {})
    seed_video_provenance: dict[str, Any] | None = None
    seed_video_pin_complete = True
    if str(seed_spec.get("kind", "target_proxy")) == "mask_video_frame":
        seed_video_path = _resolve_path(
            seed_spec.get("path"), manifest_base, description="seed binary mask video"
        )
        seed_video_provenance = _video_metadata(seed_video_path)
        seed_video_provenance["path"] = str(seed_video_path)
        seed_video_provenance["frame_index"] = int(seed_spec["frame_index"])
        seed_video_pin_complete = _validate_video_pin(
            case_name=name,
            description="seed mask video",
            metadata=seed_video_provenance,
            expected_sha256=seed_spec.get("sha256"),
            expected_fps_hz=seed_spec.get("fps_hz"),
            expected_frame_count=seed_spec.get("frame_count"),
            expected_size_wh=seed_spec.get("size_wh"),
        )
    seed_bbox_tuple = _bbox(seed_mask)
    if seed_bbox_tuple is None:
        raise RealRGBReplayError(f"case {name}: seed mask has no bbox")
    height, width = sampled_frames[0].shape[:2]
    seed_bbox = _expand_bbox(
        seed_bbox_tuple,
        padding_px=int(case.get("seed_bbox_padding_px", 3)),
        width=width,
        height=height,
    )

    cfg = load_config(str(config_path))
    cfg["online_sam2"]["enabled"] = True
    publication_mode = (
        "semantic_sam2" if diagnostic_semantic_sam2_direct else "guarded_sam2_primary"
    )
    cfg["online_sam2"]["mask_publication_mode"] = publication_mode
    cfg["online_sam2"]["require_for_bbox_init"] = True
    if sam2_image_size_override is not None:
        image_size = int(sam2_image_size_override)
        if image_size < 256 or image_size > 1024 or image_size % 32 != 0:
            raise RealRGBReplayError(
                "sam2_image_size_override must be a multiple of 32 in " "256..1024"
            )
        cfg["online_sam2"]["image_size"] = image_size
    cfg["tracker"]["mode"] = "adaptive_color_depth"
    cfg["tracker"]["recovery_publication_mode"] = "unified_three_evidence"
    cfg["runtime"]["save_debug_masks"] = False
    config_provenance = _effective_config_provenance(cfg, config_path=config_path)
    contract = dict(production_contract or {})
    production_ineligibility_reasons: list[str] = []
    if not full_suite_requested:
        production_ineligibility_reasons.append("partial_or_noncanonical_case_suite")
    if not source_video_pin_complete:
        production_ineligibility_reasons.append("incomplete_source_video_pin")
    if not seed_video_pin_complete:
        production_ineligibility_reasons.append("incomplete_seed_video_pin")
    if not realtime:
        production_ineligibility_reasons.append("realtime_schedule_disabled")
    if diagnostic_semantic_sam2_direct:
        production_ineligibility_reasons.append("semantic_sam2_direct_diagnostic")
    if sam2_image_size_override is not None:
        production_ineligibility_reasons.append("sam2_image_size_override")
    if not math.isclose(
        float(neutral_depth_m), PRODUCTION_NEUTRAL_DEPTH_M, abs_tol=1e-12
    ):
        production_ineligibility_reasons.append("neutral_depth_override")
    required_contract_fields = {
        "required_case_names",
        "default_config_sha256",
        "default_neutral_depth_m",
        "default_sam2_image_size",
        "checkpoint_sha256",
        "model_config_sha256",
    }
    if not required_contract_fields <= set(contract):
        production_ineligibility_reasons.append("incomplete_production_contract")
    else:
        if tuple(contract["required_case_names"]) != PRODUCTION_CASE_NAMES:
            production_ineligibility_reasons.append("noncanonical_required_case_names")
        expected_pairs = (
            ("default_config_sha256", config_provenance["source_config_sha256"],),
            ("checkpoint_sha256", config_provenance["checkpoint_sha256"],),
            ("model_config_sha256", config_provenance["model_config_sha256"],),
        )
        for label, actual in expected_pairs:
            if str(contract[label]).lower() != str(actual).lower():
                production_ineligibility_reasons.append(f"{label}_mismatch")
        if not math.isclose(
            float(contract["default_neutral_depth_m"]),
            float(neutral_depth_m),
            abs_tol=1e-12,
        ):
            production_ineligibility_reasons.append("nondefault_neutral_depth")
        if int(contract["default_sam2_image_size"]) != int(
            cfg["online_sam2"]["image_size"]
        ):
            production_ineligibility_reasons.append("nondefault_sam2_image_size")
    production_defaults_used = not production_ineligibility_reasons
    serial = str(cfg.get("camera", {}).get("serial", ""))
    camera = NeutralDepthVideoCamera(
        sampled_frames,
        sampled_indices,
        next_position=seed_position + 1,
        rate_hz=evaluation_fps,
        source_rate_hz=source_fps,
        depth_m=neutral_depth_m,
        device_serial=serial,
        realtime=realtime,
    )
    provider = ObjectPCDProvider(cfg, camera=camera)
    outputs: list[np.ndarray] = [
        np.zeros((height, width), dtype=bool) for _ in sampled_frames
    ]
    states: list[dict[str, Any]] = []
    for position in range(seed_position):
        states.append(
            {
                "frame_index": position,
                "source_frame_index": sampled_indices[position],
                "processing_ms": 0.0,
                "valid": False,
                "mask_area_px": 0,
                "mask_bbox_xyxy": None,
                "mask_source": "pre_seed_fail_closed",
                "online_sam2_status": "not_started",
                "message": "target not yet seeded; empty mask is intentional",
                "schedule_wait_ms": 0.0,
                "compute_processing_ms": 0.0,
                "latency_evaluable": False,
                "publication_guard_state": "pre_seed_fail_closed",
            }
        )
    provider_started = False
    bbox_prewarm_provenance = None
    service_health_provenance = None
    try:
        provider.start()
        provider_started = True
        service_health_provenance = _service_health_provenance(provider)
        seed_frame = camera.frame_at(seed_position)
        started = time.perf_counter()
        if not provider.initialize_from_bbox(seed_frame, seed_bbox):
            raise RealRGBReplayError(f"case {name}: production bbox init failed")
        initialization_ms = (time.perf_counter() - started) * 1000.0
        initialized = provider.last_mask_result
        if initialized is None or not initialized.valid:
            raise RealRGBReplayError(
                f"case {name}: provider has no valid initialized mask"
            )
        bbox_prewarm_provenance = _bbox_prewarm_provenance(provider)
        outputs[seed_position] = np.asarray(initialized.mask, dtype=bool)
        states.append(
            {
                "frame_index": seed_position,
                "source_frame_index": sampled_indices[seed_position],
                "processing_ms": initialization_ms,
                "valid": True,
                "mask_area_px": int(outputs[seed_position].sum()),
                "mask_bbox_xyxy": [int(value) for value in initialized.bbox_xyxy],
                "mask_source": "online_sam2_box_initialization",
                "online_sam2_status": str(provider.online_sam2_status),
                "message": str(initialized.message),
                "schedule_wait_ms": 0.0,
                "compute_processing_ms": initialization_ms,
                "latency_evaluable": False,
                "publication_guard_state": "new_target_initialization",
                "online_sam2_bbox_prewarm": bbox_prewarm_provenance,
                "appearance_probability_cache": (
                    _appearance_probability_cache_provenance(provider)
                ),
                "appearance_gate_cache": (
                    _appearance_gate_cache_provenance(provider)
                ),
                "mask_gate_breakdown_ms": dict(
                    getattr(provider, "_last_mask_gate_breakdown_ms", {})
                ),
                "guarded_v2_bootstrap": (_bootstrap_commissioning_provenance(provider)),
            }
        )
        camera.rebase_timing()
        for position in range(seed_position + 1, len(sampled_frames)):
            schedule_wait_ms = camera.wait_until_position(position)
            started = time.perf_counter()
            frame, result = provider.step_mask_only()
            processing_ms = (time.perf_counter() - started) * 1000.0
            if int(frame.frame_id) != int(sampled_indices[position]):
                raise RealRGBReplayError(
                    f"case {name}: provider frame {frame.frame_id} != expected "
                    f"{sampled_indices[position]}"
                )
            mask = (
                np.asarray(result.mask, dtype=bool)
                if bool(result.valid)
                else np.zeros((height, width), dtype=bool)
            )
            outputs[position] = mask
            published_bbox = _bbox(mask)
            source = str(provider.published_mask_source)
            sam2_status = str(provider.online_sam2_status)
            latency_evaluable = _latency_evaluable_after_bootstrap(
                provider,
                frame_id=int(frame.frame_id),
            )
            guard_state = (
                "published"
                if bool(result.valid)
                else (
                    "recovery_probation"
                    if "probation" in source or "pending" in str(result.message)
                    else "fail_closed_invalid"
                )
            )
            partial_owner = getattr(
                provider, "_visible_exact_partial_continuity", None
            )
            lost_seed = getattr(
                provider, "_lost_reappearance_visible_core_seed", None
            )
            pending_geometry = getattr(
                provider, "_pending_full_target_geometry_commit", None
            )
            states.append(
                {
                    "frame_index": position,
                    "source_frame_index": sampled_indices[position],
                    "processing_ms": processing_ms,
                    "compute_processing_ms": processing_ms,
                    "schedule_wait_ms": schedule_wait_ms,
                    "latency_evaluable": latency_evaluable,
                    "valid": bool(result.valid),
                    "mask_area_px": int(mask.sum()),
                    "mask_bbox_xyxy": (
                        None
                        if published_bbox is None
                        else [int(value) for value in published_bbox]
                    ),
                    "mask_source": source,
                    "online_sam2_status": sam2_status,
                    "publication_guard_state": guard_state,
                    "publication_guard_detail": str(
                        getattr(provider, "_last_publication_guard_status", "unknown",)
                    ),
                    "visible_exact_partial_failure_kind": str(
                        getattr(
                            provider,
                            "_last_visible_exact_partial_failure_kind",
                            "not_evaluated",
                        )
                    ),
                    "visible_exact_partial_reason": str(
                        getattr(
                            provider,
                            "_last_visible_exact_partial_reason",
                            "not evaluated",
                        )
                    ),
                    "visible_exact_partial_repair_reason": str(
                        getattr(
                            provider,
                            "_last_visible_exact_partial_repair_reason",
                            "not evaluated",
                        )
                    ),
                    "sanitized_partial_chain_reason": str(
                        getattr(
                            provider,
                            "_last_sanitized_partial_chain_reason",
                            "not evaluated",
                        )
                    ),
                    "partial_owner_frame_id": (
                        None
                        if partial_owner is None
                        else int(partial_owner.frame_id)
                    ),
                    "partial_owner_evidence_kind": (
                        "none"
                        if partial_owner is None
                        else str(partial_owner.evidence_kind)
                    ),
                    "partial_owner_hits": (
                        0 if partial_owner is None else int(partial_owner.hits)
                    ),
                    "partial_owner_core_area": (
                        0.0
                        if partial_owner is None
                        else float(partial_owner.core_area)
                    ),
                    "partial_owner_sanitized_anchor_area": (
                        None
                        if partial_owner is None
                        or getattr(
                            partial_owner, "sanitized_anchor_area", None
                        )
                        is None
                        else float(partial_owner.sanitized_anchor_area)
                    ),
                    "partial_owner_sanitized_anchor_frame_id": (
                        None
                        if partial_owner is None
                        or getattr(
                            partial_owner, "sanitized_anchor_frame_id", None
                        )
                        is None
                        else int(partial_owner.sanitized_anchor_frame_id)
                    ),
                    "partial_owner_sanitized_anchor_phase": (
                        "none"
                        if partial_owner is None
                        else str(
                            getattr(
                                partial_owner,
                                "sanitized_anchor_phase",
                                "legacy_fixed",
                            )
                        )
                    ),
                    "partial_owner_broad_visible_anchor_area": (
                        None
                        if partial_owner is None
                        or getattr(
                            partial_owner, "broad_visible_anchor_area", None
                        )
                        is None
                        else float(partial_owner.broad_visible_anchor_area)
                    ),
                    "partial_owner_broad_visible_anchor_frame_id": (
                        None
                        if partial_owner is None
                        or getattr(
                            partial_owner,
                            "broad_visible_anchor_frame_id",
                            None,
                        )
                        is None
                        else int(partial_owner.broad_visible_anchor_frame_id)
                    ),
                    "lost_reappearance_seed_frame_id": (
                        None if lost_seed is None else int(lost_seed.frame_id)
                    ),
                    "lost_reappearance_seed_candidate_area": (
                        0 if lost_seed is None else int(lost_seed.candidate_pixels)
                    ),
                    "pending_geometry_eligibility_kind": (
                        "none"
                        if pending_geometry is None
                        else str(pending_geometry.eligibility_kind)
                    ),
                    "lost_reappearance_proof_pending": bool(
                        pending_geometry is not None
                        and getattr(
                            pending_geometry,
                            "lost_reappearance_visible_core_proof",
                            None,
                        )
                        is not None
                    ),
                    "tracking_contraction_stage_reason": str(
                        getattr(
                            provider,
                            "_last_tracking_contraction_stage_reason",
                            "unavailable",
                        )
                    ),
                    "tracking_contraction_arm_reason": str(
                        getattr(
                            provider,
                            "_last_tracking_contraction_arm_reason",
                            "unavailable",
                        )
                    ),
                    "deep_occlusion_broad_reason": str(
                        getattr(
                            provider,
                            "_last_deep_occlusion_broad_reason",
                            "not attempted",
                        )
                    ),
                    "guarded_v2_bootstrap": (
                        _bootstrap_commissioning_provenance(provider)
                    ),
                    "appearance_probability_cache": (
                        _appearance_probability_cache_provenance(provider)
                    ),
                    "appearance_gate_cache": (
                        _appearance_gate_cache_provenance(provider)
                    ),
                    "mask_gate_breakdown_ms": {
                        str(key): float(value)
                        for key, value in getattr(
                            provider, "_last_mask_gate_breakdown_ms", {}
                        ).items()
                    },
                    "tracking_committed": bool(
                        getattr(provider, "tracking_committed", True)
                    ),
                    "recovery_probation_active": bool(
                        getattr(provider, "recovery_probation_active", False)
                    ),
                    "provider_pipeline_stage": str(
                        getattr(provider, "last_pipeline_stage", "unknown")
                    ),
                    "message": str(result.message),
                    "timings_ms": {
                        key: float(value)
                        for key, value in provider.last_timings_ms.items()
                    },
                }
            )
    finally:
        if provider_started:
            provider.stop()

    if len(states) != len(outputs):
        raise RealRGBReplayError(
            f"case {name}: states/masks are incomplete ({len(states)}/{len(outputs)})"
        )
    for position, mask in enumerate(outputs):
        path = masks_dir / f"{position:06d}.png"
        if not cv2.imwrite(str(path), mask.astype(np.uint8) * np.uint8(255)):
            raise RealRGBReplayError(f"cannot write mask: {path}")
    with (output / "states.jsonl").open("x", encoding="utf-8") as handle:
        for record in states:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    artifact_manifest = _candidate_artifact_manifest(
        case_output=output, sampled_source_indices=sampled_indices,
    )

    tracking_times = [
        float(record["processing_ms"])
        for record in states
        if int(record["frame_index"]) > seed_position
    ]
    schedule_wait_times = [
        float(record.get("schedule_wait_ms", 0.0))
        for record in states
        if int(record["frame_index"]) > seed_position
    ]
    valid_fraction = float(np.mean([bool(record["valid"]) for record in states]))
    valid_after_seed = float(
        np.mean([bool(record["valid"]) for record in states[seed_position:]])
    )
    continuity = _boolean_run_lengths(
        bool(record["valid"]) for record in states[seed_position:]
    )
    source_counts: dict[str, int] = {}
    sam2_status_counts: dict[str, int] = {}
    for record in states[seed_position:]:
        source = str(record["mask_source"])
        status = str(record["online_sam2_status"])
        source_counts[source] = source_counts.get(source, 0) + 1
        sam2_status_counts[status] = sam2_status_counts.get(status, 0) + 1
    initial_bbox_area = max(
        1, int((seed_bbox[2] - seed_bbox[0]) * (seed_bbox[3] - seed_bbox[1]))
    )
    bbox_growth: list[float] = []
    final_bbox: Optional[list[int]] = None
    for mask in outputs[seed_position:]:
        box = _bbox(mask)
        if box is None:
            continue
        area = int((box[2] - box[0]) * (box[3] - box[1]))
        bbox_growth.append(area / float(initial_bbox_area))
        final_bbox = [int(value) for value in box]
    proxy_diagnostics = _conservative_proxy_diagnostics(
        case, sampled_frames, outputs, seed_position=seed_position
    )
    summary = {
        "schema": PRODUCTION_REPLAY_SUMMARY_SCHEMA,
        "manifest_schema": REPLAY_SCHEMA,
        "case": name,
        "source_video": str(video_path),
        "source_video_sha256": source_video_metadata["sha256"],
        "source_video_metadata": source_video_metadata,
        "source_video_pin_complete": source_video_pin_complete,
        "source_fps_hz": source_fps,
        "evaluation_fps_hz": evaluation_fps,
        "sam2_image_size": int(cfg["online_sam2"]["image_size"]),
        "online_sam2_bbox_prewarm": bbox_prewarm_provenance,
        "online_sam2_service_health": service_health_provenance,
        "appearance_probability_cache_final": (
            _appearance_probability_cache_provenance(provider)
        ),
        "appearance_gate_cache_final": (
            _appearance_gate_cache_provenance(provider)
        ),
        "guarded_v2_bootstrap_final": (_bootstrap_commissioning_provenance(provider)),
        "sampled_frames": len(sampled_frames),
        "sampled_source_indices": sampled_indices,
        "seed_position": seed_position,
        "seed_source_frame": sampled_indices[seed_position],
        "seed_source": seed_source,
        "seed_video_provenance": seed_video_provenance,
        "seed_video_pin_complete": seed_video_pin_complete,
        "seed_prompt_bbox_xyxy": seed_bbox.tolist(),
        "neutral_synthetic_depth_m": neutral_depth_m,
        "realtime_schedule_preserved": bool(realtime),
        "effective_config_provenance": config_provenance,
        "implementation_source_provenance": implementation_source_provenance,
        "candidate_artifacts": artifact_manifest,
        "valid_fraction_including_preseed": valid_fraction,
        "valid_fraction_after_seed": valid_after_seed,
        "validity_continuity_after_seed": continuity,
        "published_mask_source_counts_after_seed": dict(sorted(source_counts.items())),
        "online_sam2_status_counts_after_seed": dict(
            sorted(sam2_status_counts.items())
        ),
        "final_nonempty_mask_bbox_xyxy": final_bbox,
        "bbox_growth_vs_seed_prompt_p95": (
            float(np.percentile(bbox_growth, 95)) if bbox_growth else None
        ),
        "conservative_proxy_diagnostics": proxy_diagnostics,
        "tracking_processing_ms_p50": (
            float(np.percentile(tracking_times, 50)) if tracking_times else None
        ),
        "tracking_processing_ms_p95": (
            float(np.percentile(tracking_times, 95)) if tracking_times else None
        ),
        "tracking_compute_processing_ms_p50": (
            float(np.percentile(tracking_times, 50)) if tracking_times else None
        ),
        "tracking_compute_processing_ms_p95": (
            float(np.percentile(tracking_times, 95)) if tracking_times else None
        ),
        "schedule_wait_ms_p50": (
            float(np.percentile(schedule_wait_times, 50))
            if schedule_wait_times
            else None
        ),
        "schedule_wait_ms_p95": (
            float(np.percentile(schedule_wait_times, 95))
            if schedule_wait_times
            else None
        ),
        "requested_object_mask_mode": (
            "diagnostic_semantic_sam2_direct"
            if diagnostic_semantic_sam2_direct
            else "guarded_v2"
        ),
        "effective_mask_publication_mode": publication_mode,
        "effective_recovery_publication_mode": (
            "semantic_sam2_direct"
            if diagnostic_semantic_sam2_direct
            else "unified_three_evidence"
        ),
        "production_defaults_used": bool(production_defaults_used),
        "production_acceptance_eligible": bool(production_defaults_used),
        "production_ineligibility_reasons": production_ineligibility_reasons,
        "diagnostic_only": not bool(production_defaults_used),
        "diagnostic_only_not_production_acceptance": not bool(production_defaults_used),
        "hardware_interfaces_opened": False,
        "evidence_scope": "2d_mask_identity_state_timing_only",
        "pipeline_coverage": (
            "object_pcd_provider_only_without_object_text_prompt_owner_or_"
            "category_redetection"
        ),
        "not_evidence_for": [
            "depth_alignment",
            "depth_discontinuities_or_holes",
            "3d_point_cloud_geometry",
            "camera_extrinsics",
            "workspace_or_support_plane_crop",
            "object_text_auto_reacquire_after_target_loss_or_scene_reentry",
            "robot_control",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def replay_manifest(
    manifest_path: Path,
    output_root: Path,
    *,
    config_path: Path,
    selected_cases: Optional[Iterable[str]] = None,
    neutral_depth_m: float = 1.0,
    realtime: bool = True,
    diagnostic_semantic_sam2_direct: bool = False,
    sam2_image_size_override: Optional[int] = None,
) -> dict[str, Any]:
    manifest, base = _load_cases(manifest_path)
    resolved_manifest_path = manifest_path.expanduser().resolve()
    output = output_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    requested = (
        None if selected_cases is None else {str(name) for name in selected_cases}
    )
    available = {str(case["name"]) for case in manifest["cases"]}
    if requested is not None and not requested <= available:
        raise RealRGBReplayError(
            f"unknown cases: {sorted(requested - available)}; available={sorted(available)}"
        )
    manifest_case_names = tuple(str(case["name"]) for case in manifest["cases"])
    selected_name_set = available if requested is None else requested
    full_suite_requested = bool(
        manifest_case_names == PRODUCTION_CASE_NAMES
        and selected_name_set == set(PRODUCTION_CASE_NAMES)
    )
    production_contract_raw = manifest.get("production_acceptance_contract")
    production_contract = (
        dict(production_contract_raw)
        if isinstance(production_contract_raw, dict)
        else None
    )
    try:
        implementation_source_provenance = validate_contract_source_provenance(
            production_contract or {},
            contract_base=base,
            workspace_root=WORKSPACE_ROOT,
            # The existing outer manifest is refreshed only after producer
            # behavior freezes.  Once its three source fields are present,
            # byte validation is mandatory and fail-closed.
            required=False,
        )
    except GuardedV2SourceProvenanceError as exc:
        raise RealRGBReplayError(str(exc)) from exc
    # No output directory is created before every content-addressed outer
    # contract has passed.  A refused source identity leaves no partial
    # candidate root that could be mistaken for replay evidence.
    output.mkdir(parents=True)
    summaries = []
    for case in manifest["cases"]:
        if requested is not None and str(case["name"]) not in requested:
            continue
        summaries.append(
            replay_case(
                case,
                manifest_base=base,
                output_root=output,
                config_path=config_path.expanduser().resolve(),
                neutral_depth_m=float(neutral_depth_m),
                realtime=bool(realtime),
                diagnostic_semantic_sam2_direct=bool(diagnostic_semantic_sam2_direct),
                production_contract=production_contract,
                implementation_source_provenance=implementation_source_provenance,
                full_suite_requested=full_suite_requested,
                sam2_image_size_override=sam2_image_size_override,
            )
        )
    case_summary_chain: list[dict[str, Any]] = []
    for summary in summaries:
        case_name = str(summary["case"])
        case_summary_path = output / case_name / "summary.json"
        case_summary_chain.append(
            {
                "case": case_name,
                "summary_path": f"{case_name}/summary.json",
                "summary_sha256": _sha256(case_summary_path),
                "candidate_artifacts_sha256": str(
                    summary["candidate_artifacts"]["aggregate_sha256"]
                ),
            }
        )
    root_artifact_digest = _canonical_json_sha256(case_summary_chain)
    production_ineligibility_reasons = sorted(
        {
            reason
            for summary in summaries
            for reason in summary["production_ineligibility_reasons"]
        }
    )
    if len(summaries) != len(PRODUCTION_CASE_NAMES):
        production_ineligibility_reasons.append("case_count_not_five")
    production_ineligibility_reasons = sorted(set(production_ineligibility_reasons))
    production_eligible = bool(
        not production_ineligibility_reasons
        and tuple(summary["case"] for summary in summaries) == PRODUCTION_CASE_NAMES
    )
    effective_config_provenance = (
        None if not summaries else summaries[0]["effective_config_provenance"]
    )
    if any(
        summary["effective_config_provenance"] != effective_config_provenance
        for summary in summaries
    ):
        raise RealRGBReplayError("effective config provenance changed between cases")
    result = {
        "schema": PRODUCTION_REPLAY_SUMMARY_SCHEMA,
        "manifest_schema": REPLAY_SCHEMA,
        "replay_manifest": str(resolved_manifest_path),
        "replay_manifest_sha256": _sha256(resolved_manifest_path),
        "case_count": len(summaries),
        "cases": summaries,
        "case_summary_chain": case_summary_chain,
        "candidate_root_artifacts_sha256": root_artifact_digest,
        "candidate_root": str(output),
        "effective_config_provenance": effective_config_provenance,
        "implementation_source_provenance": implementation_source_provenance,
        "hardware_interfaces_opened": False,
        "evidence_scope": "2d_mask_identity_state_timing_only",
        "pipeline_coverage": (
            "object_pcd_provider_only_without_object_text_prompt_owner_or_"
            "category_redetection"
        ),
        "realtime_schedule_preserved": bool(realtime),
        "production_acceptance_contract": production_contract,
        "production_defaults_used": bool(production_eligible),
        "production_acceptance_eligible": bool(production_eligible),
        "production_ineligibility_reasons": production_ineligibility_reasons,
        "diagnostic_only": not bool(production_eligible),
        "diagnostic_only_not_production_acceptance": not bool(production_eligible),
        "sam2_image_size_override": sam2_image_size_override,
        "neutral_synthetic_depth_m": float(neutral_depth_m),
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
