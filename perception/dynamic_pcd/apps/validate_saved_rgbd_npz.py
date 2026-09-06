#!/usr/bin/env python3
"""Validate saved real RGB-D through explicit offline perception modes.

The default mode is the historical-mask projector check.  The adaptive replay
keeps SAM2 disabled for deterministic regression coverage.  The explicit
guarded-v2 production replay instead uses frame zero's stored mask to seed the
current ``ObjectPCDProvider`` and its provider-owned local temporal SAM2
service, then sends every later final publication mask through the formal
128-point ``MaskedRGBDProjector``.

All modes supply an in-memory NPZ camera.  No RealSense, Franka, or RH56
interface is constructed.  The archives contain no capture-time palm pose, so
the returned XYZ coordinates are robot-base coordinates via an identity
``T_base_palm_at_capture`` and are not palm-frame replay evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = PACKAGE_ROOT.parent
for import_root in (PACKAGE_ROOT, WORKSPACE):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from sim2real.observation.model import MaskedRGBDProjector  # noqa: E402
from dynamic_pcd.evaluation.guarded_v2_source_provenance import (  # noqa: E402
    GuardedV2SourceProvenanceError,
    validate_contract_source_provenance,
)
from dynamic_pcd.segmentation.sam2_video_backend import (  # noqa: E402
    PRODUCTION_VOS_COMPONENT_COMPILE_MODES,
    PRODUCTION_VOS_COMPONENT_DYNAMIC,
    PRODUCTION_VOS_COMPILE_MODE,
    SAM2_MEMORY_ATTENTION_STRIDE,
    VOS_COMPILE_PREWARM_CONTRACT,
)


REQUIRED_FIELDS = (
    "rgb",
    "depth_raw",
    "object_mask",
    "camera_frame_id",
    "camera_timestamp_s",
    "frame_camera_K",
    "frame_depth_scale_m_per_unit",
    "T_base_camera_optical",
    "depth_range_m",
)
DEFAULT_CONFIG = (
    WORKSPACE / "perception" / "configs" / "d435_default.yaml"
)
ACCEPTANCE_CONTRACT_PATH = (
    WORKSPACE
    / "perception"
    / "configs"
    / "guarded_v2_real_rgb_replay_cases.json"
)
BENCHMARK_CONTRACT_PATH = (
    WORKSPACE
    / "perception"
    / "configs"
    / "guarded_v2_benchmark_long_vos_baseline.json"
)
PINNED_LEGACY_REAL_CAPTURE_SHA256 = (
    "769179e512615bda5fca49406790a2c23fc9dadde23201e038a1f208c63e542b"
)
PRODUCTION_MINIMUM_FRAMES = 60
PRODUCTION_MAX_RETRIEVED_GAP_P95_S = 0.050
PRODUCTION_MAX_RETRIEVED_GAP_S = 0.100
PRODUCTION_MAX_RETRIEVED_OVER_50MS_FRACTION = 0.020
PRODUCTION_MAX_PROVIDER_PROJECTOR_COMPUTE_P95_MS = 50.0
PRODUCTION_MAX_PROVIDER_PROJECTOR_COMPUTE_MS = 100.0
PRODUCTION_MAX_PROVIDER_PROJECTOR_OVER_50MS_FRACTION = 0.020
PRODUCTION_MAX_PROVIDER_PROJECTOR_OVER_50MS_LONGEST_RUN = 1
DEFAULT_MAXIMUM_MASK_DEPTH_DEVIATION_M = 0.055
DEFAULT_MINIMUM_FRESH_FRACTION = 1.0
DEFAULT_MINIMUM_PROVIDER_VALID_FRACTION = 0.95
DEFAULT_MINIMUM_SAM2_TRACK_CALL_FRACTION = 1.0
DEFAULT_MINIMUM_FULL_128_FRACTION = 1.0


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


def _load_fixed_acceptance_contract() -> Dict[str, Any]:
    path = ACCEPTANCE_CONTRACT_PATH.resolve()
    benchmark_path = BENCHMARK_CONTRACT_PATH.resolve()
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    expected_contract_sha256 = str(
        benchmark.get("expected_replay_manifest_sha256", "")
    ).lower()
    if len(expected_contract_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in expected_contract_sha256
    ):
        raise ValueError(
            "guarded-v2 benchmark does not pin a valid replay-manifest SHA-256"
        )
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_contract_sha256:
        raise ValueError(
            "guarded-v2 acceptance contract SHA-256 mismatch: "
            f"expected={expected_contract_sha256}, actual={actual_sha256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    contract = payload.get("production_acceptance_contract")
    if not isinstance(contract, dict):
        raise ValueError("guarded-v2 acceptance contract is missing")
    required = {
        "default_config_sha256",
        "checkpoint_sha256",
        "model_config_sha256",
        "default_neutral_depth_m",
        "default_sam2_image_size",
        "required_case_names",
    }
    if not required <= set(contract):
        raise ValueError("guarded-v2 acceptance contract is incomplete")
    try:
        implementation_source_provenance = validate_contract_source_provenance(
            contract,
            contract_base=path.parent,
            workspace_root=WORKSPACE,
            # The outer replay-manifest pin is refreshed only after the
            # implementation freezes.  If source fields are supplied they are
            # always validated; an incomplete/mismatched trio is refused.
            required=False,
        )
    except GuardedV2SourceProvenanceError as exc:
        raise ValueError(str(exc)) from exc
    return {
        **dict(contract),
        "path": str(path),
        "sha256": actual_sha256,
        "sha256_pin_source": {
            "path": str(benchmark_path),
            "sha256": _sha256(benchmark_path),
            "field": "expected_replay_manifest_sha256",
        },
        "implementation_source_provenance": implementation_source_provenance,
    }


def _write_json_exclusive_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Publish one JSON artifact atomically without replacing prior evidence."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{time.monotonic_ns()}"
    )
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and fails if destination exists.
        os.link(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _longest_true_run(values: Sequence[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _optional_archive_array(
    archive: Any,
    name: str,
    default: np.ndarray,
) -> np.ndarray:
    return np.asarray(archive[name]) if name in archive else np.asarray(default)


def _service_result_audit_fields(result: Any) -> Dict[str, Any]:
    """Return a compact JSON-safe description without retaining image arrays."""

    if isinstance(result, dict):
        fields: Dict[str, Any] = {}
        for name in (
            "service",
            "protocol_version",
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
            "safe_history_frames",
            "reset_every_frames",
            "vos_optimized",
            "vos_compile_mode",
            "vos_compile_cuda_graphs",
            "vos_memory_attention_rope_expected_tokens",
            "vos_memory_attention_rope_cache_count",
            "vos_memory_attention_rope_caches_verified",
            "compile_prewarm_required",
            "compile_prewarm_completed",
            "compile_prewarm_contract",
            "compile_prewarm_ms",
            "compile_prewarm_initialize_box_ms",
            "compile_prewarm_track_ms",
            "compile_prewarm_initialize_mask_ms",
            "compile_prewarm_mask_track_ms",
            "generation",
            "last_frame_id",
            "internal_frame_idx",
            "model_load_ms",
            "status",
        ):
            value = result.get(name)
            if isinstance(value, np.generic):
                value = value.item()
            if value is None or isinstance(value, (str, int, float, bool)):
                fields[name] = value
        for name in (
            "compile_prewarm_input_size_wh",
            "compile_prewarm_shape_hw",
            "vos_memory_attention_rope_grid_hw",
        ):
            value = result.get(name)
            if (
                isinstance(value, (list, tuple))
                and len(value) == 2
                and all(
                    isinstance(item, (int, np.integer))
                    and not isinstance(item, (bool, np.bool_))
                    for item in value
                )
            ):
                fields[name] = [int(item) for item in value]
        component_modes = result.get("vos_component_compile_modes")
        if isinstance(component_modes, dict):
            fields["vos_component_compile_modes"] = {
                str(name): str(mode)
                for name, mode in component_modes.items()
            }
        component_dynamic = result.get("vos_component_compile_dynamic")
        if isinstance(component_dynamic, dict):
            fields["vos_component_compile_dynamic"] = {
                str(name): bool(dynamic)
                for name, dynamic in component_dynamic.items()
                if isinstance(dynamic, (bool, np.bool_))
            }
        for name in (
            "vos_memory_attention_rope_cache_token_counts",
            "vos_memory_attention_rope_cache_labels",
        ):
            value = result.get(name)
            if not isinstance(value, (list, tuple)):
                continue
            if name.endswith("token_counts") and all(
                isinstance(item, (int, np.integer))
                and not isinstance(item, (bool, np.bool_))
                for item in value
            ):
                fields[name] = [int(item) for item in value]
            elif name.endswith("labels") and all(
                isinstance(item, str) for item in value
            ):
                fields[name] = list(value)
        return fields
    fields = {}
    for name in (
        "valid",
        "frame_id",
        "message",
        "internal_frame_idx",
        "mask_area",
    ):
        if not hasattr(result, name):
            continue
        value = getattr(result, name)
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (str, int, float, bool)):
            fields[name] = value
    timings = getattr(result, "timings_ms", None)
    if isinstance(timings, dict):
        fields["timings_ms"] = {
            str(name): float(value) for name, value in timings.items()
        }
    return fields


class _SAM2ServiceAudit:
    """Non-owning call audit around the provider's normal service manager."""

    def __init__(self, delegate_factory: Callable[..., Any]) -> None:
        self.delegate_factory = delegate_factory
        self.factory_calls: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.managers: List[Any] = []
        self._lock = threading.Lock()

    def manager_factory(self, *args: Any, **kwargs: Any) -> Any:
        factory_call = {
            "addr": str(kwargs.get("addr", "")),
            "autostart": bool(kwargs.get("autostart", False)),
            "startup_timeout_s": float(kwargs.get("startup_timeout_s", 0.0)),
            "request_timeout_ms": int(kwargs.get("request_timeout_ms", 0)),
        }
        manager = _AuditedSAM2Manager(
            self.delegate_factory(*args, **kwargs), audit=self
        )
        with self._lock:
            self.factory_calls.append(factory_call)
            self.managers.append(manager)
        return manager

    def record(self, event: Dict[str, Any]) -> None:
        with self._lock:
            self.events.append(dict(event))

    def count(self, operation: str, frame_id: Optional[int] = None) -> int:
        with self._lock:
            return sum(
                1
                for event in self.events
                if event["operation"] == str(operation)
                and (
                    frame_id is None
                    or event.get("frame_id") == int(frame_id)
                )
            )

    def snapshot_events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(event) for event in self.events]


class _AuditedSAM2Manager:
    """Delegate every service operation while recording only safe metadata."""

    def __init__(self, delegate: Any, *, audit: _SAM2ServiceAudit) -> None:
        self._delegate = delegate
        self._audit = audit

    def _call(
        self,
        operation: str,
        call: Callable[..., Any],
        *args: Any,
        frame_id: Optional[int] = None,
        mask_area_px: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        started = time.perf_counter()
        event: Dict[str, Any] = {"operation": str(operation)}
        if frame_id is not None:
            event["frame_id"] = int(frame_id)
        if mask_area_px is not None:
            event["input_mask_area_px"] = int(mask_area_px)
        try:
            result = call(*args, **kwargs)
        except BaseException as exc:
            event["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
            event["outcome"] = "error"
            event["error"] = f"{type(exc).__name__}: {exc}"
            self._audit.record(event)
            raise
        event["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        event["outcome"] = "ok"
        event["result"] = _service_result_audit_fields(result)
        if operation == "start":
            process = getattr(self._delegate, "process", None)
            child_pid = getattr(process, "pid", None)
            event["launcher_ownership"] = {
                "manager_owns_process": bool(
                    getattr(self._delegate, "owns_process", False)
                ),
                "child_pid": (
                    None if child_pid is None else int(child_pid)
                ),
                "validator_parent_pid": int(os.getpid()),
            }
        self._audit.record(event)
        return result

    def start(self) -> Any:
        return self._call("start", self._delegate.start)

    def health(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("health", self._delegate.health, *args, **kwargs)

    def initialize(
        self, image_bgr: np.ndarray, mask: np.ndarray, frame_id: int, **kwargs: Any
    ) -> Any:
        return self._call(
            "initialize",
            self._delegate.initialize,
            image_bgr,
            mask,
            frame_id,
            frame_id=int(frame_id),
            mask_area_px=int(np.count_nonzero(mask)),
            **kwargs,
        )

    def initialize_box(
        self,
        image_bgr: np.ndarray,
        bbox_xyxy: np.ndarray,
        frame_id: int,
        **kwargs: Any,
    ) -> Any:
        return self._call(
            "initialize_box",
            self._delegate.initialize_box,
            image_bgr,
            bbox_xyxy,
            frame_id,
            frame_id=int(frame_id),
            **kwargs,
        )

    def track(
        self, image_bgr: np.ndarray, frame_id: int, **kwargs: Any
    ) -> Any:
        return self._call(
            "track",
            self._delegate.track,
            image_bgr,
            frame_id,
            frame_id=int(frame_id),
            **kwargs,
        )

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("reset", self._delegate.reset, *args, **kwargs)

    def close(self) -> Any:
        return self._call("close", self._delegate.close)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


def _is_local_service_address(address: str) -> bool:
    value = str(address).strip().lower()
    if value.startswith(("ipc://", "inproc://")):
        return True
    parsed = urlsplit(value)
    if parsed.scheme != "tcp" or parsed.hostname is None:
        return False
    if parsed.hostname == "localhost":
        return True
    try:
        return bool(ipaddress.ip_address(parsed.hostname).is_loopback)
    except ValueError:
        return False


class _SavedNPZCamera:
    """Minimal RealSense-compatible camera over one in-memory NPZ sequence."""

    def __init__(self, payload: Dict[str, np.ndarray]) -> None:
        self.payload = payload
        self.device_serial = str(payload.get("camera_serial", ""))
        self.device_name = "saved camera-only RGB-D NPZ"
        self.device_firmware_version = ""
        self.device_usb_type_descriptor = "offline"
        self.sdk_version = "offline"
        self.depth_scale = float(payload["depth_scales"][0])
        self.started = False
        self._index = 1

    def start(self) -> None:
        # Frame zero is reserved for explicit initialize_from_mask.
        self._index = 1
        self.started = True

    def stop(self) -> None:
        self.started = False

    def frame(self, index: int):
        from dynamic_pcd.types import CameraIntrinsics, RGBDFrame

        position = int(index)
        camera_k = self.payload["camera_k"][position]
        distortion = self.payload["distortion"][position]
        rgb = self.payload["rgb"]
        return RGBDFrame(
            color_bgr=rgb[position],
            depth_raw=self.payload["depth_raw"][position],
            depth_scale=float(self.payload["depth_scales"][position]),
            intrinsics=CameraIntrinsics(
                width=int(rgb.shape[2]),
                height=int(rgb.shape[1]),
                fx=float(camera_k[0, 0]),
                fy=float(camera_k[1, 1]),
                ppx=float(camera_k[0, 2]),
                ppy=float(camera_k[1, 2]),
                model="none",
                distortion=tuple(float(value) for value in distortion),
            ),
            timestamp=float(self.payload["timestamps"][position]),
            frame_id=int(self.payload["frame_ids"][position]),
            retrieved_at_s=float(self.payload["retrieved_at_s"][position]),
            timestamp_domain=str(self.payload["timestamp_domain"][position]),
            sensor_frame_number=int(
                self.payload["sensor_frame_number"][position]
            ),
            depth_sensor_frame_number=int(
                self.payload["depth_sensor_frame_number"][position]
            ),
        )

    def get_frame(self, timeout_ms: int = 1000):
        del timeout_ms
        if not self.started:
            self.start()
        if self._index >= int(self.payload["rgb"].shape[0]):
            raise RuntimeError("saved RGB-D NPZ is exhausted")
        frame = self.frame(self._index)
        self._index += 1
        return frame


def _load_provider_replay_payload(path: Path) -> Dict[str, Any]:
    archive_path = Path(path).expanduser().resolve()
    with np.load(archive_path, allow_pickle=False) as archive:
        missing = [name for name in REQUIRED_FIELDS if name not in archive]
        if missing:
            raise ValueError(
                "saved RGB-D NPZ is missing required fields: "
                + ", ".join(missing)
            )
        rgb = np.asarray(archive["rgb"])
        frames = int(rgb.shape[0]) if rgb.ndim >= 1 else 0
        archive_fields = tuple(sorted(str(name) for name in archive.files))
        def scalar(name: str, default: Any) -> Any:
            return np.asarray(archive[name]).item() if name in archive else default

        payload: Dict[str, Any] = {
            "archive_fields": archive_fields,
            "rgb": rgb,
            "depth_raw": np.asarray(archive["depth_raw"]),
            "masks": np.asarray(archive["object_mask"], dtype=bool),
            "frame_ids": np.asarray(archive["camera_frame_id"]),
            "timestamps": np.asarray(archive["camera_timestamp_s"]),
            "camera_k": np.asarray(archive["frame_camera_K"]),
            "depth_scales": np.asarray(
                archive["frame_depth_scale_m_per_unit"]
            ),
            "T_base_camera": np.asarray(archive["T_base_camera_optical"]),
            "depth_range": np.asarray(archive["depth_range_m"]),
            "distortion": _optional_archive_array(
                archive,
                "frame_camera_distortion",
                np.zeros((frames, 5), dtype=np.float64),
            ),
            "retrieved_at_s": _optional_archive_array(
                archive,
                "camera_retrieved_at_s",
                np.asarray(archive["camera_timestamp_s"]),
            ),
            "timestamp_domain": _optional_archive_array(
                archive,
                "camera_timestamp_domain",
                np.full(frames, "offline_saved_npz"),
            ),
            "sensor_frame_number": _optional_archive_array(
                archive,
                "camera_sensor_frame_number",
                np.asarray(archive["camera_frame_id"]),
            ),
            "depth_sensor_frame_number": _optional_archive_array(
                archive,
                "camera_depth_sensor_frame_number",
                np.asarray(archive["camera_frame_id"]),
            ),
            "stored_bbox": _optional_archive_array(
                archive,
                "object_mask_bbox_xyxy",
                np.zeros((frames, 4), dtype=np.int32),
            ),
            "camera_serial": (
                str(np.asarray(archive["camera_serial"]).item())
                if "camera_serial" in archive
                else ""
            ),
            "calibration_id": (
                str(np.asarray(archive["calibration_id"]).item())
                if "calibration_id" in archive
                else ""
            ),
            "stored_object_text": (
                str(np.asarray(archive["object_text"]).item())
                if "object_text" in archive
                else ""
            ),
            "capture_schema_version": int(
                scalar("capture_schema_version", -1)
            ),
            "hardware_writes": bool(scalar("hardware_writes", True)),
            "hardware_writes_semantics": str(
                scalar("hardware_writes_semantics", "")
            ),
            "robot_command_writes": bool(
                scalar("robot_command_writes", True)
            ),
            "camera_configuration_writes": bool(
                scalar("camera_configuration_writes", False)
            ),
            "franka_interface_opened_at_capture": bool(
                scalar("franka_interface_opened", True)
            ),
            "rh56_interface_opened_at_capture": bool(
                scalar("rh56_interface_opened", True)
            ),
            "stored_requested_object_mask_mode": str(
                scalar("requested_object_mask_mode", "")
            ),
            "stored_effective_mask_publication_mode": str(
                scalar("effective_mask_publication_mode", "")
            ),
            "bundle_contract": str(scalar("bundle_contract", "")),
        }
    return payload


def _validate_provider_replay_payload(
    payload: Dict[str, Any],
) -> Tuple[int, int, int]:
    rgb = np.asarray(payload["rgb"])
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must have uint8 shape [N,H,W,3]")
    frame_count, height, width, _ = rgb.shape
    if frame_count < 2:
        raise ValueError("current-provider replay requires at least two frames")
    depth_raw = np.asarray(payload["depth_raw"])
    if (
        depth_raw.shape != (frame_count, height, width)
        or depth_raw.dtype != np.uint16
    ):
        raise ValueError("depth_raw must have uint16 shape [N,H,W]")
    masks = np.asarray(payload["masks"])
    if masks.shape != (frame_count, height, width):
        raise ValueError("object_mask must have shape [N,H,W]")
    if not np.any(masks[0]):
        raise ValueError("frame 0 stored initialization mask is empty")
    frame_ids = np.asarray(payload["frame_ids"])
    timestamps = np.asarray(payload["timestamps"])
    if frame_ids.shape != (frame_count,) or timestamps.shape != (frame_count,):
        raise ValueError("camera frame ids/timestamps must have shape [N]")
    if not np.all(np.isfinite(timestamps.astype(np.float64))):
        raise ValueError("camera_timestamp_s must be finite")
    if not np.all(np.diff(frame_ids.astype(np.int64)) > 0):
        raise ValueError("camera_frame_id must strictly increase")
    if not np.all(np.diff(timestamps.astype(np.float64)) > 0.0):
        raise ValueError("camera_timestamp_s must strictly increase")
    camera_k = np.asarray(payload["camera_k"], dtype=np.float64)
    if camera_k.shape != (frame_count, 3, 3):
        raise ValueError("frame_camera_K must have shape [N,3,3]")
    if not np.all(np.isfinite(camera_k)) or not np.all(camera_k[:, 0, 0] > 0.0) or not np.all(
        camera_k[:, 1, 1] > 0.0
    ):
        raise ValueError("frame_camera_K must be finite with positive focal lengths")
    if not np.allclose(camera_k, camera_k[0], atol=0.0, rtol=0.0):
        raise ValueError("camera intrinsics changed inside one saved archive")
    depth_scales = np.asarray(payload["depth_scales"], dtype=np.float64)
    if (
        depth_scales.shape != (frame_count,)
        or not np.all(np.isfinite(depth_scales))
        or not np.all(depth_scales > 0.0)
    ):
        raise ValueError(
            "frame_depth_scale_m_per_unit must be finite positive shape [N]"
        )
    transform = np.asarray(payload["T_base_camera"], dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("T_base_camera_optical must have shape [4,4]")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12):
        raise ValueError("T_base_camera_optical must be homogeneous")
    depth_range = np.asarray(payload["depth_range"], dtype=np.float64)
    if (
        depth_range.shape != (2,)
        or not np.all(np.isfinite(depth_range))
        or not 0.0 < float(depth_range[0]) < float(depth_range[1])
    ):
        raise ValueError("depth_range_m must contain two ordered positive values")
    if np.asarray(payload["stored_bbox"]).shape != (frame_count, 4):
        raise ValueError("object_mask_bbox_xyxy must have shape [N,4]")
    distortion = np.asarray(payload["distortion"], dtype=np.float64)
    if distortion.shape != (frame_count, 5) or not np.all(np.isfinite(distortion)):
        raise ValueError("frame_camera_distortion must have finite shape [N,5]")
    retrieved = np.asarray(payload["retrieved_at_s"], dtype=np.float64)
    if retrieved.shape != (frame_count,) or not np.all(np.isfinite(retrieved)):
        raise ValueError("camera_retrieved_at_s must have finite shape [N]")
    if not np.all(np.diff(retrieved) > 0.0):
        raise ValueError("camera_retrieved_at_s must strictly increase")
    timestamp_domain = np.asarray(payload["timestamp_domain"])
    if timestamp_domain.shape != (frame_count,) or any(
        not str(value).strip() for value in timestamp_domain.tolist()
    ):
        raise ValueError("camera_timestamp_domain must be non-empty shape [N]")
    for name in ("sensor_frame_number", "depth_sensor_frame_number"):
        values = np.asarray(payload[name])
        if values.shape != (frame_count,) or not np.all(
            np.diff(values.astype(np.int64)) > 0
        ):
            raise ValueError(f"{name} must strictly increase with shape [N]")
    stored_bbox = np.asarray(payload["stored_bbox"], dtype=np.int64)
    for index, mask in enumerate(masks):
        derived = _mask_bbox(mask)
        expected = [0, 0, 0, 0] if derived is None else derived
        if stored_bbox[index].tolist() != expected:
            raise ValueError(
                f"object_mask_bbox_xyxy[{index}] does not match object_mask"
            )
    return int(frame_count), int(height), int(width)


def _mask_bbox(mask: np.ndarray) -> Optional[List[int]]:
    rows, columns = np.nonzero(np.asarray(mask, dtype=bool))
    if columns.size == 0:
        return None
    return [
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    ]


def _project_frame_record(
    projector: MaskedRGBDProjector,
    *,
    frame: Any,
    mask: np.ndarray,
) -> Dict[str, Any]:
    started = time.perf_counter()
    point_frame = projector.project(
        color_bgr=frame.color_bgr,
        depth_raw=frame.depth_raw,
        depth_scale_m_per_unit=frame.depth_scale,
        object_mask=np.asarray(mask, dtype=bool),
        # Saved camera-only evidence contains no capture-time palm pose.
        T_base_palm_at_capture=np.eye(4, dtype=np.float64),
        captured_at_s=float(frame.timestamp),
        frame_id=int(frame.frame_id),
    )
    compute_ms = (time.perf_counter() - started) * 1000.0
    provenance = projector.last_effective_object_mask_provenance
    features = np.asarray(point_frame.xyzrgb_palm)
    validity = np.asarray(point_frame.valid)
    if features.ndim != 2 or features.shape[0] != 128:
        raise RuntimeError("formal projector did not return exactly 128 rows")
    if validity.shape != (128,):
        raise RuntimeError("formal projector validity must have shape [128]")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(validity)):
        raise RuntimeError("formal projector returned non-finite feature/validity data")
    valid_points = int(np.count_nonzero(validity > 0.5))
    current_frame_provenance = bool(
        provenance.source_frame_id == int(frame.frame_id)
        and provenance.source_captured_at_s is not None
        and np.isclose(
            float(provenance.source_captured_at_s),
            float(frame.timestamp),
            atol=1e-12,
            rtol=0.0,
        )
    )
    return {
        "pcd_status": str(point_frame.status),
        "pcd_fresh": bool(point_frame.status == "fresh"),
        "pcd_source_valid_points": int(point_frame.source_valid_points),
        "pcd_output_valid_points": valid_points,
        "pcd_output_rows": int(np.asarray(point_frame.xyzrgb_palm).shape[0]),
        "pcd_point_feature_dim": int(
            features.shape[1]
        ),
        "pcd_output_finite": True,
        "pcd_current_frame_provenance": current_frame_provenance,
        "pcd_compute_ms": float(compute_ms),
        "pcd_provenance_kind": str(provenance.kind),
        "pcd_provenance_source_frame_id": (
            None
            if provenance.source_frame_id is None
            else int(provenance.source_frame_id)
        ),
        "pcd_provenance_source_captured_at_s": (
            None
            if provenance.source_captured_at_s is None
            else float(provenance.source_captured_at_s)
        ),
        "pcd_provenance_area_px": int(provenance.area_px),
        "pcd_provenance_bbox_xyxy": (
            None
            if provenance.bbox_xyxy is None
            else [int(value) for value in provenance.bbox_xyxy]
        ),
    }


def _fraction_at_least_one_call(
    audit: _SAM2ServiceAudit, operation: str, frame_ids: Sequence[int]
) -> float:
    if not frame_ids:
        return 1.0
    covered = sum(
        int(audit.count(operation, frame_id=int(frame_id)) > 0)
        for frame_id in frame_ids
    )
    return float(covered / len(frame_ids))


def _resolve_model_config_file(raw: Any, *, config_path: Path) -> Optional[Path]:
    if raw is None or not str(raw).strip():
        return None
    value = Path(str(raw)).expanduser()
    candidates = [value] if value.is_absolute() else [
        config_path.parent / value,
        WORKSPACE / value,
        WORKSPACE / "third_party" / "sam2" / "sam2" / value,
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    return None


def _effective_config_provenance(
    cfg: Dict[str, Any], *, config_path: Path
) -> Dict[str, Any]:
    online = dict(cfg.get("online_sam2") or {})
    checkpoint_raw = online.get("checkpoint")
    checkpoint = (
        None
        if checkpoint_raw is None or not str(checkpoint_raw).strip()
        else Path(str(checkpoint_raw)).expanduser().resolve()
    )
    model_config = _resolve_model_config_file(
        online.get("model_config"), config_path=config_path
    )
    return {
        "source_config": str(config_path),
        "source_config_sha256": _sha256(config_path),
        "canonical_effective_config_sha256": _canonical_json_sha256(cfg),
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "checkpoint_sha256": (
            None if checkpoint is None or not checkpoint.is_file() else _sha256(checkpoint)
        ),
        "model_config": None if model_config is None else str(model_config),
        "model_config_sha256": (
            None if model_config is None else _sha256(model_config)
        ),
        "vos_optimized": online.get("vos_optimized"),
        "vos_compile_mode": online.get("vos_compile_mode"),
    }


def _config_matches_fixed_acceptance_contract(
    *,
    cfg: Dict[str, Any],
    config_provenance: Dict[str, Any],
    acceptance_contract: Dict[str, Any],
) -> bool:
    return bool(
        str(config_provenance.get("source_config_sha256", "")).lower()
        == str(acceptance_contract["default_config_sha256"]).lower()
        and str(config_provenance.get("checkpoint_sha256", "")).lower()
        == str(acceptance_contract["checkpoint_sha256"]).lower()
        and str(config_provenance.get("model_config_sha256", "")).lower()
        == str(acceptance_contract["model_config_sha256"]).lower()
        and int((cfg.get("online_sam2") or {}).get("image_size", -1))
        == int(acceptance_contract["default_sam2_image_size"])
        and (cfg.get("online_sam2") or {}).get("vos_compile_mode")
        == PRODUCTION_VOS_COMPILE_MODE
    )


def _service_health_identity(
    audit: _SAM2ServiceAudit,
    *,
    cfg: Dict[str, Any],
    config_provenance: Dict[str, Any],
    acceptance_contract: Dict[str, Any],
    injected_fixture: bool,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], bool]:
    candidates = [
        dict(event.get("result") or {})
        for event in audit.snapshot_events()
        if event.get("operation") in {"start", "health"}
        and isinstance(event.get("result"), dict)
    ]
    identity = next(
        (
            candidate
            for candidate in reversed(candidates)
            if candidate.get("backend") is not None
            or candidate.get("service") is not None
        ),
        None,
    )
    online = dict(cfg.get("online_sam2") or {})
    expected_checkpoint = str(online.get("checkpoint", ""))
    expected_model_config = str(online.get("model_config", ""))
    camera = dict(cfg.get("camera") or {})
    expected_prewarm_shape = [
        int(camera.get("height", 480)),
        int(camera.get("width", 848)),
    ]
    expected_prewarm_size_wh = [
        expected_prewarm_shape[1],
        expected_prewarm_shape[0],
    ]
    expected_image_size = int(online.get("image_size", -1))
    expected_rope_side = (
        expected_image_size // SAM2_MEMORY_ATTENTION_STRIDE
        if expected_image_size > 0
        else -1
    )
    expected_rope_grid = [expected_rope_side, expected_rope_side]
    expected_rope_tokens = expected_rope_side * expected_rope_side
    start_ownership = next(
        (
            dict(event.get("launcher_ownership") or {})
            for event in audit.snapshot_events()
            if event.get("operation") == "start"
            and isinstance(event.get("launcher_ownership"), dict)
        ),
        {},
    )
    launcher_owned_child = bool(
        not injected_fixture
        and start_ownership.get("manager_owns_process") is True
        and isinstance(start_ownership.get("child_pid"), int)
        and int(start_ownership["child_pid"]) > 0
        and int(start_ownership.get("validator_parent_pid", -1)) == os.getpid()
    )
    if identity is not None:
        identity["validator_launcher_ownership"] = start_ownership
    checks = [
        {
            "check": "real_noninjected_service",
            "passed": not injected_fixture,
        },
        {
            "check": "validator_launcher_owned_child_no_foreign_service_reuse",
            "passed": launcher_owned_child,
        },
        {
            "check": "health_identity_present",
            "passed": identity is not None,
        },
        {
            "check": "official_backend",
            "passed": bool(
                identity
                and identity.get("backend") == "official-sam2-video-predictor"
            ),
        },
        {
            "check": "model_loaded",
            "passed": bool(identity and identity.get("loaded") is True),
        },
        {
            "check": "checkpoint_identity",
            "passed": bool(
                identity
                and str(identity.get("checkpoint", "")) == expected_checkpoint
                and str(identity.get("checkpoint_sha256", "")).lower()
                == str(acceptance_contract["checkpoint_sha256"]).lower()
                and str(config_provenance.get("checkpoint_sha256", "")).lower()
                == str(acceptance_contract["checkpoint_sha256"]).lower()
            ),
        },
        {
            "check": "model_config_identity",
            "passed": bool(
                identity
                and str(identity.get("model_config", "")) == expected_model_config
                and str(identity.get("model_config_sha256", "")).lower()
                == str(acceptance_contract["model_config_sha256"]).lower()
                and str(config_provenance.get("model_config_sha256", "")).lower()
                == str(acceptance_contract["model_config_sha256"]).lower()
            ),
        },
        {
            "check": "cuda_device_identity",
            "passed": bool(
                identity
                and str(identity.get("device", "")).startswith("cuda")
                and str(identity.get("gpu_name", "")).strip()
            ),
        },
        {
            "check": "image_size_identity",
            "passed": bool(
                identity
                and int(identity.get("image_size", -1))
                == int(online.get("image_size", -2))
            ),
        },
        {
            "check": "amp_dtype_identity",
            "passed": bool(
                identity
                and str(identity.get("amp_dtype", ""))
                == str(online.get("amp_dtype", ""))
            ),
        },
        {
            "check": "vos_optimized_identity",
            "passed": bool(
                identity
                and isinstance(identity.get("vos_optimized"), bool)
                and identity.get("vos_optimized")
                is online.get("vos_optimized")
            ),
        },
        {
            "check": "optimized_compile_mode_identity",
            "passed": bool(
                identity
                and (
                    online.get("vos_optimized") is not True
                    or (
                        online.get("vos_compile_mode")
                        == PRODUCTION_VOS_COMPILE_MODE
                        and identity.get("vos_compile_mode")
                        == PRODUCTION_VOS_COMPILE_MODE
                        and identity.get("vos_compile_cuda_graphs") is False
                        and identity.get("vos_component_compile_modes")
                        == dict(PRODUCTION_VOS_COMPONENT_COMPILE_MODES)
                        and identity.get("vos_component_compile_dynamic")
                        == dict(PRODUCTION_VOS_COMPONENT_DYNAMIC)
                    )
                )
            ),
        },
        {
            "check": "optimized_memory_attention_rope_identity",
            "passed": bool(
                identity
                and (
                    online.get("vos_optimized") is not True
                    or (
                        expected_image_size > 0
                        and expected_image_size
                        % SAM2_MEMORY_ATTENTION_STRIDE
                        == 0
                        and identity.get(
                            "vos_memory_attention_rope_grid_hw"
                        )
                        == expected_rope_grid
                        and identity.get(
                            "vos_memory_attention_rope_expected_tokens"
                        )
                        == expected_rope_tokens
                        and identity.get(
                            "vos_memory_attention_rope_caches_verified"
                        )
                        is True
                        and isinstance(
                            identity.get(
                                "vos_memory_attention_rope_cache_token_counts"
                            ),
                            list,
                        )
                        and bool(
                            identity.get(
                                "vos_memory_attention_rope_cache_token_counts"
                            )
                        )
                        and all(
                            int(token_count) == expected_rope_tokens
                            for token_count in identity[
                                "vos_memory_attention_rope_cache_token_counts"
                            ]
                        )
                        and identity.get(
                            "vos_memory_attention_rope_cache_count"
                        )
                        == len(
                            identity[
                                "vos_memory_attention_rope_cache_token_counts"
                            ]
                        )
                    )
                )
            ),
        },
        {
            "check": "optimized_compile_prewarm_identity",
            "passed": bool(
                identity
                and (
                    online.get("vos_optimized") is not True
                    or (
                        identity.get("compile_prewarm_required") is True
                        and identity.get("compile_prewarm_completed") is True
                        and identity.get("compile_prewarm_contract")
                        == VOS_COMPILE_PREWARM_CONTRACT
                        and identity.get("compile_prewarm_input_size_wh")
                        == expected_prewarm_size_wh
                        and identity.get("compile_prewarm_shape_hw")
                        == expected_prewarm_shape
                        and identity.get("initialized") is False
                        and all(
                            isinstance(identity.get(name), (int, float))
                            and not isinstance(identity.get(name), bool)
                            and np.isfinite(float(identity[name]))
                            and float(identity[name]) >= 0.0
                            for name in (
                                "compile_prewarm_initialize_box_ms",
                                "compile_prewarm_track_ms",
                                "compile_prewarm_initialize_mask_ms",
                                "compile_prewarm_mask_track_ms",
                            )
                        )
                    )
                )
            ),
        },
        {
            "check": "runtime_numeric_identity",
            "passed": bool(
                identity
                and identity.get("tf32") is True
                and int(identity.get("fill_hole_area", -1)) == 0
            ),
        },
    ]
    return identity, checks, bool(all(item["passed"] for item in checks))


def evaluate_guarded_v2_provider_from_saved_rgbd_npz(
    path: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
    point_feature_mode: str = "xyz",
    maximum_mask_depth_deviation_m: Optional[float] = 0.055,
    fixed_sphere_radius_m: Optional[float] = None,
    online_sam2_manager_factory: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Replay the exact production mask owner over lossless saved RGB-D.

    Frame zero's stored provider mask is the sole trusted initialization input.
    It initializes both the current adaptive identity owner and the
    provider-owned temporal SAM2 session.  Every later RGB-D frame is processed
    by ``guarded_sam2_primary`` and its final (possibly invalid/empty) published
    mask is passed to the formal 128-point projector.

    ``online_sam2_manager_factory`` exists only for deterministic unit tests.
    The CLI never supplies it and therefore uses the real local ZMQ SAM2 service
    manager owned by ``ObjectPCDProvider``.
    """

    from dynamic_pcd.config import load_config
    from dynamic_pcd.provider.object_pcd_provider import ObjectPCDProvider
    from dynamic_pcd.segmentation.sam2_video_runtime import (
        SAM2VideoServiceManager,
    )

    archive_path = Path(path).expanduser().resolve()
    resolved_config_path = Path(config_path).expanduser().resolve()
    acceptance_contract = _load_fixed_acceptance_contract()
    payload = _load_provider_replay_payload(archive_path)
    frame_count, height, width = _validate_provider_replay_payload(payload)
    feature_mode = str(point_feature_mode).strip().lower()
    if feature_mode not in ("xyz", "xyzrgb"):
        raise ValueError("point_feature_mode must be xyz or xyzrgb")

    cfg = load_config(str(resolved_config_path))
    if (int(cfg["camera"]["width"]), int(cfg["camera"]["height"])) != (
        width,
        height,
    ):
        raise ValueError("active config image size differs from saved RGB-D")
    cfg["sam2"]["enabled"] = False
    cfg["sam2"]["require_for_bbox_init"] = False
    cfg["online_sam2"]["enabled"] = True
    cfg["online_sam2"]["guarded_v2_semantic_primary"] = True
    cfg["online_sam2"]["mask_publication_mode"] = "guarded_sam2_primary"
    # Initialization is a trusted stored mask, not a box-prompt replay.  The
    # temporal service is still synchronously initialized from that exact mask.
    cfg["online_sam2"]["require_for_bbox_init"] = False
    cfg["tracker"]["mode"] = "adaptive_color_depth"
    cfg["tracker"]["recovery_publication_mode"] = "unified_three_evidence"
    cfg["runtime"]["save_debug_masks"] = False
    config_provenance = _effective_config_provenance(
        cfg, config_path=resolved_config_path
    )
    archive_camera_serial = str(payload["camera_serial"]).strip()
    active_camera_serial = str(cfg.get("camera", {}).get("serial") or "").strip()
    camera_serial_matches_config = bool(
        archive_camera_serial
        and active_camera_serial
        and archive_camera_serial == active_camera_serial
    )
    service_address = str(
        cfg["online_sam2"].get("service_addr", "tcp://127.0.0.1:5558")
    )
    if not _is_local_service_address(service_address):
        raise ValueError(
            "guarded-v2 saved RGB-D acceptance requires a loopback/local "
            f"SAM2 service address, got {service_address!r}"
        )

    injected_service_fixture = online_sam2_manager_factory is not None
    delegate_factory = (
        SAM2VideoServiceManager
        if online_sam2_manager_factory is None
        else online_sam2_manager_factory
    )
    service_audit = _SAM2ServiceAudit(delegate_factory)
    camera = _SavedNPZCamera(payload)
    provider = ObjectPCDProvider(
        cfg,
        camera=camera,
        online_sam2_manager_factory=service_audit.manager_factory,
    )
    projector = MaskedRGBDProjector(
        camera_K=np.asarray(payload["camera_k"][0], dtype=np.float64),
        T_base_camera_optical=np.asarray(
            payload["T_base_camera"], dtype=np.float64
        ),
        image_size=(width, height),
        depth_range_m=(
            float(payload["depth_range"][0]),
            float(payload["depth_range"][1]),
        ),
        num_points=128,
        minimum_valid_points=16,
        point_feature_dim=3 if feature_mode == "xyz" else 6,
        maximum_mask_depth_deviation_m=maximum_mask_depth_deviation_m,
        fixed_sphere_completion_radius_m=fixed_sphere_radius_m,
    )

    active_transform = np.asarray(
        provider.extrinsics.T_base_camera, dtype=np.float64
    )
    archive_transform = np.asarray(
        payload["T_base_camera"], dtype=np.float64
    )
    transform_matches_archive = bool(
        np.allclose(active_transform, archive_transform, atol=1.0e-6, rtol=0.0)
    )
    active_calibration_id = str(provider.extrinsics.calibration_id or "")
    archive_calibration_id = str(payload["calibration_id"])
    calibration_id_matches_archive = bool(
        archive_calibration_id
        and active_calibration_id
        and archive_calibration_id == active_calibration_id
    )
    if not transform_matches_archive or (
        archive_calibration_id
        and active_calibration_id
        and not calibration_id_matches_archive
    ):
        # Do not let the provider's 3-D guard and the formal projector silently
        # reason in two different camera calibrations.
        provider.stop()
        raise ValueError(
            "active config calibration differs from the saved RGB-D archive"
        )

    records: List[Dict[str, Any]] = []
    temporal_seed_initialized = False
    try:
        provider.start()
        if not injected_service_fixture:
            _startup_identity, startup_identity_checks, startup_identity_valid = (
                _service_health_identity(
                    service_audit,
                    cfg=cfg,
                    config_provenance=config_provenance,
                    acceptance_contract=acceptance_contract,
                    injected_fixture=False,
                )
            )
            if not startup_identity_valid:
                failed = [
                    item["check"]
                    for item in startup_identity_checks
                    if not bool(item["passed"])
                ]
                raise RuntimeError(
                    "production saved-RGBD refuses a foreign/unpinned SAM2 "
                    f"service; failed identity checks={failed}"
                )
        initialization_frame = camera.frame(0)
        stored_initialization_bbox = np.asarray(
            payload["stored_bbox"][0], dtype=np.int32
        ).reshape(4)
        if (
            int(stored_initialization_bbox[2])
            <= int(stored_initialization_bbox[0])
            or int(stored_initialization_bbox[3])
            <= int(stored_initialization_bbox[1])
        ):
            derived_bbox = _mask_bbox(payload["masks"][0])
            if derived_bbox is None:
                raise ValueError("frame 0 stored initialization mask is empty")
            stored_initialization_bbox = np.asarray(
                derived_bbox, dtype=np.int32
            )
        initialization_started = time.perf_counter()
        initialized_ok = provider.initialize_from_mask(
            initialization_frame,
            payload["masks"][0],
            stored_initialization_bbox,
            source="saved_frame0_provider_mask",
            # This False is the production-critical distinction from the
            # deterministic adaptive replay: it invokes manager.initialize().
            online_sam2_already_initialized=False,
        )
        initialization_compute_ms = (
            time.perf_counter() - initialization_started
        ) * 1000.0
        initialized = provider.last_mask_result
        temporal_seed_initialized = bool(
            getattr(provider, "_online_sam2_initialized", False)
        )
        if (
            not initialized_ok
            or initialized is None
            or not bool(initialized.valid)
        ):
            raise RuntimeError(
                "stored frame 0 mask failed current-provider initialization"
            )
        if not temporal_seed_initialized:
            raise RuntimeError(
                "stored frame 0 mask did not initialize temporal SAM2 service"
            )
        initialization_mask = np.asarray(initialized.mask, dtype=bool)
        initialization_record: Dict[str, Any] = {
            "archive_index": 0,
            "camera_frame_id": int(initialization_frame.frame_id),
            "camera_timestamp_s": float(initialization_frame.timestamp),
            "phase": "frame0_stored_mask_initialization",
            "provider_valid": True,
            "provider_source": "saved_frame0_provider_mask_initialization",
            "provider_candidate_source": str(
                getattr(initialized, "source", "")
            ),
            "provider_message": str(initialized.message),
            "provider_mask_area_px": int(np.count_nonzero(initialization_mask)),
            "provider_mask_bbox_xyxy": _mask_bbox(initialization_mask),
            "provider_compute_ms": float(initialization_compute_ms),
            "provider_timings_ms": {
                "initialization_total": float(initialization_compute_ms)
            },
            "provider_online_sam2_status": "initialized_from_saved_frame0_mask",
            "provider_publication_guard_source": "new_target_initialization",
            "provider_publication_guard_status": "trusted_stored_mask",
            "provider_tracking_committed": bool(provider.tracking_committed),
            "provider_recovery_probation_active": bool(
                provider.recovery_probation_active
            ),
            "provider_pipeline_stage": str(provider.last_pipeline_stage),
            "historical_mask_iou": 1.0,
        }
        initialization_record.update(
            _project_frame_record(
                projector,
                frame=initialization_frame,
                mask=initialization_mask,
            )
        )
        records.append(initialization_record)

        for archive_index in range(1, frame_count):
            provider_started = time.perf_counter()
            frame, mask_result = provider.step_mask_only()
            provider_compute_ms = (
                time.perf_counter() - provider_started
            ) * 1000.0
            expected_frame_id = int(payload["frame_ids"][archive_index])
            if int(frame.frame_id) != expected_frame_id:
                raise RuntimeError(
                    f"provider frame {frame.frame_id} != saved frame "
                    f"{expected_frame_id} at archive index {archive_index}"
                )
            expected_timestamp = float(payload["timestamps"][archive_index])
            if float(frame.timestamp) != expected_timestamp:
                raise RuntimeError(
                    "provider timestamp differs from saved RGB-D at archive "
                    f"index {archive_index}"
                )
            valid = bool(mask_result.valid)
            mask = (
                np.asarray(mask_result.mask, dtype=bool)
                if valid
                else np.zeros((height, width), dtype=bool)
            )
            historical_iou: Optional[float] = None
            if valid:
                historical = np.asarray(
                    payload["masks"][archive_index], dtype=bool
                )
                union = int(np.count_nonzero(mask | historical))
                historical_iou = float(
                    np.count_nonzero(mask & historical) / max(1, union)
                )
            record = {
                "archive_index": int(archive_index),
                "camera_frame_id": int(frame.frame_id),
                "camera_timestamp_s": float(frame.timestamp),
                "phase": "guarded_v2_temporal_tracking",
                "provider_valid": valid,
                "provider_source": str(provider.published_mask_source),
                "provider_candidate_source": str(
                    getattr(mask_result, "source", "")
                ),
                "provider_message": str(mask_result.message),
                "provider_mask_area_px": int(np.count_nonzero(mask)),
                "provider_mask_bbox_xyxy": _mask_bbox(mask),
                "provider_compute_ms": float(provider_compute_ms),
                "provider_timings_ms": {
                    str(name): float(value)
                    for name, value in provider.last_timings_ms.items()
                },
                "provider_online_sam2_status": str(
                    provider.online_sam2_status
                ),
                "provider_publication_guard_source": str(
                    getattr(provider, "_last_publication_guard_source", "")
                ),
                "provider_publication_guard_status": str(
                    getattr(provider, "_last_publication_guard_status", "")
                ),
                "provider_tracking_committed": bool(
                    provider.tracking_committed
                ),
                "provider_recovery_probation_active": bool(
                    provider.recovery_probation_active
                ),
                "provider_pipeline_stage": str(provider.last_pipeline_stage),
                "historical_mask_iou": historical_iou,
            }
            # Invalid publication is represented by an exact-current empty mask.
            # The projector may report stale_palm; provenance makes that explicit.
            record.update(
                _project_frame_record(projector, frame=frame, mask=mask)
            )
            records.append(record)
    finally:
        provider.stop()

    for record in records:
        frame_id = int(record["camera_frame_id"])
        record["sam2_service_initialize_calls_for_frame"] = service_audit.count(
            "initialize", frame_id=frame_id
        )
        record["sam2_service_track_calls_for_frame"] = service_audit.count(
            "track", frame_id=frame_id
        )

    tracking_records = records[1:]
    tracking_frame_ids = [
        int(record["camera_frame_id"]) for record in tracking_records
    ]
    provider_valid_values = np.asarray(
        [bool(record["provider_valid"]) for record in tracking_records],
        dtype=bool,
    )
    provider_compute_values = np.asarray(
        [float(record["provider_compute_ms"]) for record in tracking_records],
        dtype=np.float64,
    )
    pcd_compute_values = np.asarray(
        [float(record["pcd_compute_ms"]) for record in records],
        dtype=np.float64,
    )
    if provider_compute_values.size != max(0, pcd_compute_values.size - 1):
        raise RuntimeError(
            "provider/projector timing vectors do not share the post-seed frames"
        )
    provider_projector_compute_values = (
        provider_compute_values + pcd_compute_values[1:]
    )
    provider_projector_over_50ms = provider_projector_compute_values > 50.0
    pcd_source_counts = np.asarray(
        [int(record["pcd_source_valid_points"]) for record in records],
        dtype=np.int64,
    )
    pcd_output_counts = np.asarray(
        [int(record["pcd_output_valid_points"]) for record in records],
        dtype=np.int64,
    )
    pcd_statuses = [str(record["pcd_status"]) for record in records]
    status_counts = {
        status: int(pcd_statuses.count(status))
        for status in sorted(set(pcd_statuses))
    }
    historical_ious = np.asarray(
        [
            float(record["historical_mask_iou"])
            for record in tracking_records
            if record["historical_mask_iou"] is not None
        ],
        dtype=np.float64,
    )
    track_call_fraction = _fraction_at_least_one_call(
        service_audit, "track", tracking_frame_ids
    )
    initialization_frame_id = int(records[0]["camera_frame_id"])
    initialization_service_calls = service_audit.count(
        "initialize", frame_id=initialization_frame_id
    )
    operation_counts = {
        operation: service_audit.count(operation)
        for operation in (
            "start",
            "health",
            "initialize",
            "initialize_box",
            "track",
            "reset",
            "close",
        )
    }
    frame_id_gaps = np.diff(
        np.asarray(payload["frame_ids"], dtype=np.int64)
    )
    timestamp_gaps_s = np.diff(
        np.asarray(payload["timestamps"], dtype=np.float64)
    )
    retrieved_gaps_s = np.diff(
        np.asarray(payload["retrieved_at_s"], dtype=np.float64)
    )
    sensor_frame_gaps = np.diff(
        np.asarray(payload["sensor_frame_number"], dtype=np.int64)
    )
    depth_sensor_frame_gaps = np.diff(
        np.asarray(payload["depth_sensor_frame_number"], dtype=np.int64)
    )
    def gap_distribution(values: np.ndarray) -> Dict[str, int]:
        unique, counts = np.unique(values, return_counts=True)
        return {
            str(int(value)): int(count)
            for value, count in zip(unique.tolist(), counts.tolist())
        }

    full_128_fraction = float(np.mean(pcd_output_counts == 128))
    fresh_fraction = float(status_counts.get("fresh", 0) / frame_count)
    current_provenance_fraction = float(
        np.mean(
            [bool(record["pcd_current_frame_provenance"]) for record in records]
        )
    )
    health_identity, health_identity_checks, health_identity_valid = (
        _service_health_identity(
            service_audit,
            cfg=cfg,
            config_provenance=config_provenance,
            acceptance_contract=acceptance_contract,
            injected_fixture=injected_service_fixture,
        )
    )
    archive_sha256 = _sha256(archive_path)
    required_capture_fields = {
        "capture_schema_version",
        "hardware_writes",
        "robot_command_writes",
        "franka_interface_opened",
        "rh56_interface_opened",
        "camera_serial",
        "calibration_id",
        "camera_sensor_frame_number",
        "camera_depth_sensor_frame_number",
        "frame_camera_distortion",
        "requested_object_mask_mode",
        "effective_mask_publication_mode",
    }
    archive_field_set = set(payload["archive_fields"])
    timestamp_gap_p95 = _percentile(timestamp_gaps_s, 95.0)
    timestamp_gap_max = float(np.max(timestamp_gaps_s))
    retrieved_gap_p95 = _percentile(retrieved_gaps_s, 95.0)
    retrieved_gap_max = float(np.max(retrieved_gaps_s))
    retrieved_over_50ms_fraction = float(
        np.mean(retrieved_gaps_s > 0.050)
    )
    cadence_checks = [
        {
            "check": "retrieved_wall_gap_p95_at_most_50ms",
            "passed": retrieved_gap_p95 <= PRODUCTION_MAX_RETRIEVED_GAP_P95_S,
            "actual": retrieved_gap_p95,
        },
        {
            "check": "retrieved_wall_gap_max_at_most_100ms",
            "passed": retrieved_gap_max <= PRODUCTION_MAX_RETRIEVED_GAP_S,
            "actual": retrieved_gap_max,
        },
        {
            "check": "retrieved_wall_gap_over_50ms_fraction_at_most_2pct",
            "passed": retrieved_over_50ms_fraction
            <= PRODUCTION_MAX_RETRIEVED_OVER_50MS_FRACTION,
            "actual": retrieved_over_50ms_fraction,
        },
    ]
    capture_cadence_20hz_eligible = bool(
        all(item["passed"] for item in cadence_checks)
    )
    fixed_contract_config_identity = _config_matches_fixed_acceptance_contract(
        cfg=cfg,
        config_provenance=config_provenance,
        acceptance_contract=acceptance_contract,
    )
    production_evidence_checks = [
        {
            "check": "pinned_legacy_archive_sha256",
            "passed": archive_sha256 == PINNED_LEGACY_REAL_CAPTURE_SHA256,
        },
        {
            "check": "legacy_capture_schema_v1",
            "passed": int(payload["capture_schema_version"]) == 1,
        },
        {
            "check": "at_least_60_frames",
            "passed": frame_count >= PRODUCTION_MINIMUM_FRAMES,
        },
        {
            "check": "required_real_capture_fields_present",
            "passed": required_capture_fields <= archive_field_set,
        },
        {
            "check": "camera_serial_nonempty_matches_config",
            "passed": camera_serial_matches_config,
        },
        {
            "check": "calibration_nonempty_matches_config",
            "passed": bool(
                transform_matches_archive and calibration_id_matches_archive
            ),
        },
        {
            "check": "no_robot_interfaces_or_writes_at_capture",
            "passed": bool(
                payload["hardware_writes"] is False
                and payload["robot_command_writes"] is False
                and payload["franka_interface_opened_at_capture"] is False
                and payload["rh56_interface_opened_at_capture"] is False
            ),
        },
        {
            "check": "fixed_contract_default_production_config_bytes",
            "passed": config_provenance["source_config_sha256"]
            == str(acceptance_contract["default_config_sha256"]).lower(),
        },
        {
            "check": "fixed_contract_checkpoint_and_model_config_content_addressed",
            "passed": fixed_contract_config_identity,
        },
        {
            "check": "real_service_health_identity",
            "passed": health_identity_valid,
        },
        {
            "check": "all_projector_outputs_current_frame_provenance",
            "passed": current_provenance_fraction == 1.0,
        },
    ]
    production_evidence_eligible = bool(
        all(item["passed"] for item in production_evidence_checks)
    )
    limitations = [
        "frame 0 is a trusted stored provider mask; prompt grounding is not replayed",
        (
            "historical masks on frames 1..N-1 are consistency references, "
            "not ground truth or provider inputs"
        ),
        (
            "the archive contains only historically valid publications, so "
            "original rejected/occluded RGB-D frames are absent"
        ),
        "no capture-time T_base_palm is stored, so XYZ is checked in robot_base",
        "offline compute timing excludes camera transport and wall-clock rollout scheduling",
    ]
    if injected_service_fixture:
        limitations.append(
            "an injected temporal-service fixture was used; this run is "
            "unit-test evidence, not a real local SAM2/GPU acceptance"
        )

    return {
        "schema": "saved_real_rgbd_guarded_v2_projector_acceptance_v1",
        "mode": "guarded_v2_production_saved_rgbd",
        "source": str(archive_path),
        "source_sha256": archive_sha256,
        "source_contract": (
            "trusted stored frame0 mask initializes current provider and local "
            "temporal SAM2; frames1..N-1 use guarded_sam2_primary final masks "
            "with same-frame saved Z16 in the formal 128-point projector"
        ),
        "active_config": str(resolved_config_path),
        "active_config_sha256": _sha256(resolved_config_path),
        "effective_config_provenance": config_provenance,
        "fixed_acceptance_contract": acceptance_contract,
        "frames": int(frame_count),
        "image_size_wh": [int(width), int(height)],
        "initialization_frames": 1,
        "guarded_v2_tracking_frames_evaluated": int(frame_count - 1),
        "source_frame_ids_consecutive": bool(np.all(frame_id_gaps == 1)),
        "source_frame_id_gap_max": int(np.max(frame_id_gaps)),
        "source_timestamp_gap_s_p95_max": [
            timestamp_gap_p95,
            timestamp_gap_max,
        ],
        "capture_cadence_evidence": {
            "abstract_camera_frame_id_gaps": {
                "distribution": gap_distribution(frame_id_gaps),
                "max": int(np.max(frame_id_gaps)),
                "all_consecutive": bool(np.all(frame_id_gaps == 1)),
            },
            "color_sensor_frame_number_gaps": {
                "distribution": gap_distribution(sensor_frame_gaps),
                "max": int(np.max(sensor_frame_gaps)),
            },
            "depth_sensor_frame_number_gaps": {
                "distribution": gap_distribution(depth_sensor_frame_gaps),
                "max": int(np.max(depth_sensor_frame_gaps)),
            },
            "camera_timestamp_gaps_s": {
                "p95": timestamp_gap_p95,
                "max": timestamp_gap_max,
                "role": "sensor_sampling_diagnostic_non_gating",
            },
            "camera_retrieved_wall_gaps_s": {
                "p50": _percentile(retrieved_gaps_s, 50.0),
                "p95": retrieved_gap_p95,
                "max": retrieved_gap_max,
                "over_50ms_fraction": retrieved_over_50ms_fraction,
                "over_50ms_count": int(np.count_nonzero(retrieved_gaps_s > 0.050)),
                "sample_count": int(retrieved_gaps_s.size),
                "role": "20hz_publication_cadence_gating_evidence",
            },
            "checks": cadence_checks,
            "accepted_end_to_end_cadence": capture_cadence_20hz_eligible,
        },
        "camera_serial": str(payload["camera_serial"]),
        "active_config_camera_serial": active_camera_serial,
        "camera_serial_matches_active_config": camera_serial_matches_config,
        "calibration_id": archive_calibration_id,
        "active_config_calibration_id": active_calibration_id,
        "active_config_calibration_matches_archive": bool(
            transform_matches_archive and calibration_id_matches_archive
        ),
        "stored_object_text": str(payload["stored_object_text"]),
        "stored_requested_object_mask_mode": str(
            payload["stored_requested_object_mask_mode"]
        ),
        "stored_effective_mask_publication_mode": str(
            payload["stored_effective_mask_publication_mode"]
        ),
        "stored_masks_are_historical_adaptive_not_guarded_v2_ground_truth": True,
        "requested_object_mask_mode": "guarded_v2",
        "effective_object_mask_mode": "guarded_v2",
        "online_sam2_enabled": True,
        "effective_mask_publication_mode": "guarded_sam2_primary",
        "effective_recovery_publication_mode": "unified_three_evidence",
        "frame0_temporal_sam2_initialized": bool(
            temporal_seed_initialized and initialization_service_calls >= 1
        ),
        "guarded_v2_provider_valid_frames": int(
            np.count_nonzero(provider_valid_values)
        ),
        "guarded_v2_provider_invalid_frames": int(
            provider_valid_values.size - np.count_nonzero(provider_valid_values)
        ),
        "current_provider_valid_fraction": float(
            np.mean(provider_valid_values)
        ),
        "provider_published_mask_sources": sorted(
            {str(record["provider_source"]) for record in tracking_records}
        ),
        "provider_online_sam2_statuses": sorted(
            {
                str(record["provider_online_sam2_status"])
                for record in tracking_records
            }
        ),
        "provider_publication_guard_statuses": sorted(
            {
                str(record["provider_publication_guard_status"])
                for record in tracking_records
            }
        ),
        "provider_invalid_messages": [
            str(record["provider_message"])
            for record in tracking_records
            if not bool(record["provider_valid"])
        ],
        "provider_compute_ms_p50_p95_max": [
            _percentile(provider_compute_values, 50.0),
            _percentile(provider_compute_values, 95.0),
            float(np.max(provider_compute_values)),
        ],
        "initialization_compute_ms": float(records[0]["provider_compute_ms"]),
        "historical_provider_mask_iou_p05_p50_min": (
            [
                _percentile(historical_ious, 5.0),
                _percentile(historical_ious, 50.0),
                float(np.min(historical_ious)),
            ]
            if historical_ious.size
            else None
        ),
        "historical_provider_mask_is_not_ground_truth": True,
        "sam2_temporal_service": {
            "owner": "ObjectPCDProvider_single_worker_service_manager",
            "address": service_address,
            "address_is_local": True,
            "autostart": bool(
                cfg["online_sam2"].get("service_autostart", True)
            ),
            "backend": (
                "injected_test_fixture"
                if injected_service_fixture
                else "local_zmq_sam2_video_service"
            ),
            "operation_counts": operation_counts,
            "frame0_initialize_call_count": int(initialization_service_calls),
            "expected_tracking_frame_ids": tracking_frame_ids,
            "track_call_fraction": float(track_call_fraction),
            "every_tracking_frame_called": bool(track_call_fraction == 1.0),
            "factory_calls": service_audit.factory_calls,
            "events": service_audit.snapshot_events(),
            "health_identity": health_identity,
            "health_identity_checks": health_identity_checks,
            "health_identity_valid_for_production": health_identity_valid,
        },
        "projector_status_counts_including_initialization": status_counts,
        "projector_fresh_fraction_including_initialization": fresh_fraction,
        "projector_full_128_fraction_including_initialization": (
            full_128_fraction
        ),
        "projector_source_valid_points_min_p50_max": [
            int(np.min(pcd_source_counts)),
            _percentile(pcd_source_counts, 50.0),
            int(np.max(pcd_source_counts)),
        ],
        "projector_output_valid_points_min_p50_max": [
            int(np.min(pcd_output_counts)),
            _percentile(pcd_output_counts, 50.0),
            int(np.max(pcd_output_counts)),
        ],
        "projector_compute_ms_p50_p95_max": [
            _percentile(pcd_compute_values, 50.0),
            _percentile(pcd_compute_values, 95.0),
            float(np.max(pcd_compute_values)),
        ],
        "provider_plus_projector_compute_ms": {
            "p50": _percentile(provider_projector_compute_values, 50.0),
            "p95": _percentile(provider_projector_compute_values, 95.0),
            "max": float(np.max(provider_projector_compute_values)),
            "over_50ms_fraction": float(
                np.mean(provider_projector_over_50ms)
            ),
            "over_50ms_count": int(
                np.count_nonzero(provider_projector_over_50ms)
            ),
            "over_50ms_longest_run": _longest_true_run(
                provider_projector_over_50ms.tolist()
            ),
            "sample_count": int(provider_projector_compute_values.size),
            "scope": "post_seed_provider_step_plus_same_frame_128_point_projector",
        },
        "projector_mask_provenance": sorted(
            {str(record["pcd_provenance_kind"]) for record in records}
        ),
        "projector_current_frame_provenance_fraction": current_provenance_fraction,
        "projector_parameters": {
            "image_size_wh": [int(width), int(height)],
            "num_points": 128,
            "minimum_valid_points": 16,
            "point_feature_dim": 3 if feature_mode == "xyz" else 6,
            "maximum_mask_depth_deviation_m": maximum_mask_depth_deviation_m,
            "fixed_sphere_completion_radius_m": fixed_sphere_radius_m,
            "depth_range_m": [
                float(payload["depth_range"][0]),
                float(payload["depth_range"][1]),
            ],
        },
        "formal_projector_class": "sim2real.observation.model.MaskedRGBDProjector",
        "formal_projector_num_points": 128,
        "formal_projector_minimum_valid_points": 16,
        "point_feature_mode": feature_mode,
        "coordinate_frame": "robot_base_via_identity_T_base_palm",
        "frame_records": records,
        "current_provider_tracking_evaluated": True,
        "online_sam2_evaluated": True,
        "actual_local_sam2_service_evaluated": bool(
            not injected_service_fixture
        ),
        "prompt_grounding_evaluated": False,
        "camera_interface_opened": False,
        "camera_hardware_interface_opened": False,
        "realsense_interface_opened": False,
        "franka_interface_opened": False,
        "rh56_interface_opened": False,
        "hardware_interfaces_opened": False,
        "hardware_writes": False,
        "capture_provenance": {
            "profile": "pinned_legacy_real_capture_schema_v1",
            "capture_schema_version": int(payload["capture_schema_version"]),
            "bundle_contract": str(payload["bundle_contract"]),
            "archive_fields": list(payload["archive_fields"]),
            "hardware_writes": bool(payload["hardware_writes"]),
            "hardware_writes_semantics": str(
                payload["hardware_writes_semantics"]
            ),
            "robot_command_writes": bool(payload["robot_command_writes"]),
            "camera_configuration_writes": bool(
                payload["camera_configuration_writes"]
            ),
            "franka_interface_opened": bool(
                payload["franka_interface_opened_at_capture"]
            ),
            "rh56_interface_opened": bool(
                payload["rh56_interface_opened_at_capture"]
            ),
        },
        "production_evidence_checks": production_evidence_checks,
        "production_evidence_eligible": production_evidence_eligible,
        "limitations": limitations,
    }


def evaluate_current_provider_from_saved_rgbd_npz(
    path: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
    point_feature_mode: str = "xyz",
    maximum_mask_depth_deviation_m: Optional[float] = 0.055,
    fixed_sphere_radius_m: Optional[float] = None,
) -> Dict[str, Any]:
    """Rerun the current adaptive provider on stored valid-publication RGB-D.

    Frame zero is an explicit trusted mask initialization.  Online/image SAM2
    is disabled so this regression is deterministic and does not claim to test
    current grounding or semantic recovery.  Frames 1..N-1 exercise the
    current adaptive tracker, final publication guard and formal projector.
    """

    from dynamic_pcd.config import load_config
    from dynamic_pcd.provider.object_pcd_provider import ObjectPCDProvider

    archive_path = Path(path).expanduser().resolve()
    payload = _load_provider_replay_payload(archive_path)
    rgb = payload["rgb"]
    frame_count, height, width, _ = rgb.shape
    if frame_count < 2:
        raise ValueError("current-provider replay requires at least two frames")
    feature_mode = str(point_feature_mode).strip().lower()
    if feature_mode not in ("xyz", "xyzrgb"):
        raise ValueError("point_feature_mode must be xyz or xyzrgb")

    cfg = load_config(str(Path(config_path).expanduser().resolve()))
    if (int(cfg["camera"]["width"]), int(cfg["camera"]["height"])) != (
        width,
        height,
    ):
        raise ValueError("active config image size differs from saved RGB-D")
    cfg["sam2"]["enabled"] = False
    cfg["sam2"]["require_for_bbox_init"] = False
    cfg["online_sam2"]["enabled"] = False
    cfg["online_sam2"]["require_for_bbox_init"] = False
    cfg["online_sam2"]["mask_publication_mode"] = "adaptive_fusion"
    cfg["tracker"]["recovery_publication_mode"] = "unified_three_evidence"
    cfg["runtime"]["save_debug_masks"] = False

    camera = _SavedNPZCamera(payload)
    provider = ObjectPCDProvider(cfg, camera=camera)
    projector = MaskedRGBDProjector(
        camera_K=np.asarray(payload["camera_k"][0], dtype=np.float64),
        T_base_camera_optical=np.asarray(
            payload["T_base_camera"], dtype=np.float64
        ),
        image_size=(width, height),
        depth_range_m=(
            float(payload["depth_range"][0]),
            float(payload["depth_range"][1]),
        ),
        num_points=128,
        minimum_valid_points=16,
        point_feature_dim=3 if feature_mode == "xyz" else 6,
        maximum_mask_depth_deviation_m=maximum_mask_depth_deviation_m,
        fixed_sphere_completion_radius_m=fixed_sphere_radius_m,
    )

    provider_valid = []
    provider_sources = []
    provider_messages = []
    provider_guard_status = []
    provider_step_ms = []
    projector_status = []
    projector_source_counts = []
    projector_provenance = []
    historical_mask_iou = []
    try:
        provider.start()
        initialization_frame = camera.frame(0)
        if not provider.initialize_from_mask(
            initialization_frame,
            payload["masks"][0],
            payload["stored_bbox"][0],
            source="saved_provider_mask",
            online_sam2_already_initialized=True,
        ):
            raise RuntimeError("stored provider mask failed explicit initialization")

        # Seed the projector with the same trusted first RGB-D/mask frame.
        initial_points = projector.project(
            color_bgr=initialization_frame.color_bgr,
            depth_raw=initialization_frame.depth_raw,
            depth_scale_m_per_unit=initialization_frame.depth_scale,
            object_mask=payload["masks"][0],
            T_base_palm_at_capture=np.eye(4, dtype=np.float64),
            captured_at_s=float(initialization_frame.timestamp),
            frame_id=int(initialization_frame.frame_id),
        )
        projector_status.append(str(initial_points.status))
        projector_source_counts.append(int(initial_points.source_valid_points))
        projector_provenance.append(
            str(projector.last_effective_object_mask_provenance.kind)
        )

        for archive_index in range(1, frame_count):
            started = time.perf_counter()
            frame, mask_result = provider.step_mask_only()
            provider_step_ms.append((time.perf_counter() - started) * 1000.0)
            is_valid = bool(mask_result.valid)
            provider_valid.append(is_valid)
            provider_sources.append(str(provider.published_mask_source))
            provider_messages.append(str(mask_result.message))
            provider_guard_status.append(
                str(getattr(provider, "_last_publication_guard_status", ""))
            )
            mask = (
                np.asarray(mask_result.mask, dtype=bool)
                if is_valid
                else np.zeros((height, width), dtype=bool)
            )
            if is_valid:
                historical = np.asarray(
                    payload["masks"][archive_index], dtype=bool
                )
                union = int(np.count_nonzero(mask | historical))
                historical_mask_iou.append(
                    float(np.count_nonzero(mask & historical) / max(1, union))
                )
            points = projector.project(
                color_bgr=frame.color_bgr,
                depth_raw=frame.depth_raw,
                depth_scale_m_per_unit=frame.depth_scale,
                object_mask=mask,
                T_base_palm_at_capture=np.eye(4, dtype=np.float64),
                captured_at_s=float(frame.timestamp),
                frame_id=int(frame.frame_id),
            )
            projector_status.append(str(points.status))
            projector_source_counts.append(int(points.source_valid_points))
            projector_provenance.append(
                str(projector.last_effective_object_mask_provenance.kind)
            )
    finally:
        provider.stop()

    evaluated = frame_count - 1
    provider_valid_count = int(np.count_nonzero(provider_valid))
    frame_id_gaps = np.diff(
        np.asarray(payload["frame_ids"], dtype=np.int64)
    )
    timestamp_gaps_s = np.diff(
        np.asarray(payload["timestamps"], dtype=np.float64)
    )
    projector_status_counts = {
        status: int(projector_status.count(status))
        for status in sorted(set(projector_status))
    }
    summary: Dict[str, Any] = {
        "schema": "saved_real_rgbd_current_provider_projector_replay_v1",
        "source": str(archive_path),
        "source_sha256": _sha256(archive_path),
        "source_contract": (
            "explicit stored-mask initialization then current adaptive provider "
            "over ordered stored historically-valid RGB-D frames"
        ),
        "frames": int(frame_count),
        "source_frame_ids_consecutive": bool(np.all(frame_id_gaps == 1)),
        "source_frame_id_gap_max": int(np.max(frame_id_gaps)),
        "source_timestamp_gap_s_p95_max": [
            _percentile(timestamp_gaps_s, 95.0),
            float(np.max(timestamp_gaps_s)),
        ],
        "initialization_frames": 1,
        "current_provider_frames_evaluated": int(evaluated),
        "current_provider_valid_frames": provider_valid_count,
        "current_provider_valid_fraction": float(
            provider_valid_count / evaluated
        ),
        "current_provider_invalid_frames": int(evaluated - provider_valid_count),
        "provider_published_mask_sources": sorted(set(provider_sources)),
        "provider_guard_statuses": sorted(set(provider_guard_status)),
        "provider_invalid_messages": [
            provider_messages[index]
            for index, valid in enumerate(provider_valid)
            if not valid
        ],
        "provider_compute_ms_p50_p95_max": [
            _percentile(np.asarray(provider_step_ms), 50.0),
            _percentile(np.asarray(provider_step_ms), 95.0),
            float(np.max(provider_step_ms)),
        ],
        "historical_provider_mask_iou_p05_p50_min": (
            [
                _percentile(np.asarray(historical_mask_iou), 5.0),
                _percentile(np.asarray(historical_mask_iou), 50.0),
                float(np.min(historical_mask_iou)),
            ]
            if historical_mask_iou
            else None
        ),
        "historical_provider_mask_is_not_ground_truth": True,
        "projector_status_counts_including_initialization": (
            projector_status_counts
        ),
        "projector_fresh_fraction_including_initialization": float(
            projector_status_counts.get("fresh", 0) / frame_count
        ),
        "projector_source_valid_points_min_p50_max": [
            int(np.min(projector_source_counts)),
            _percentile(np.asarray(projector_source_counts), 50.0),
            int(np.max(projector_source_counts)),
        ],
        "projector_mask_provenance": sorted(set(projector_provenance)),
        "point_feature_mode": feature_mode,
        "coordinate_frame": "robot_base_via_identity_T_base_palm",
        "stored_object_text": str(payload["stored_object_text"]),
        "current_provider_tracking_evaluated": True,
        "prompt_grounding_evaluated": False,
        "online_sam2_evaluated": False,
        "hardware_interfaces_opened": False,
        "limitations": [
            "archive contains only historically valid provider publications",
            "original rejected/occluded RGB-D frames are absent; any frame-gap response is not an exact recovery replay",
            "full occlusion, exact invalid-frame recovery and prompt grounding are not evaluated",
            "online/image SAM2 is disabled for this deterministic adaptive-provider replay",
            "historical masks are a consistency reference, not ground-truth labels",
            "no capture-time T_base_palm is stored, so XYZ is checked in robot_base",
        ],
    }
    return summary


def evaluate_saved_rgbd_npz(
    path: Path,
    *,
    point_feature_mode: str = "xyz",
    maximum_mask_depth_deviation_m: Optional[float] = 0.055,
    fixed_sphere_radius_m: Optional[float] = None,
) -> Dict[str, Any]:
    """Run stored provider masks through the current formal 128-point adapter."""

    archive_path = Path(path).expanduser().resolve()
    with np.load(archive_path, allow_pickle=False) as archive:
        missing = [name for name in REQUIRED_FIELDS if name not in archive]
        if missing:
            raise ValueError(
                "saved RGB-D NPZ is missing required fields: "
                + ", ".join(missing)
            )
        rgb = np.asarray(archive["rgb"])
        depth_raw = np.asarray(archive["depth_raw"])
        masks = np.asarray(archive["object_mask"])
        frame_ids = np.asarray(archive["camera_frame_id"])
        timestamps = np.asarray(archive["camera_timestamp_s"])
        camera_k = np.asarray(archive["frame_camera_K"], dtype=np.float64)
        depth_scales = np.asarray(
            archive["frame_depth_scale_m_per_unit"], dtype=np.float64
        )
        T_base_camera = np.asarray(
            archive["T_base_camera_optical"], dtype=np.float64
        )
        depth_range = np.asarray(archive["depth_range_m"], dtype=np.float64)
        camera_serial = (
            str(np.asarray(archive["camera_serial"]).item())
            if "camera_serial" in archive
            else ""
        )
        calibration_id = (
            str(np.asarray(archive["calibration_id"]).item())
            if "calibration_id" in archive
            else ""
        )
        object_text = (
            str(np.asarray(archive["object_text"]).item())
            if "object_text" in archive
            else ""
        )
        requested_object_mask_mode = (
            str(np.asarray(archive["requested_object_mask_mode"]).item())
            if "requested_object_mask_mode" in archive
            else ""
        )

    if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must have uint8 shape [N,H,W,3]")
    frame_count, height, width, _ = rgb.shape
    if frame_count <= 0:
        raise ValueError("saved RGB-D NPZ contains no frames")
    if depth_raw.shape != (frame_count, height, width) or depth_raw.dtype != np.uint16:
        raise ValueError("depth_raw must have uint16 shape [N,H,W]")
    if masks.shape != (frame_count, height, width):
        raise ValueError("object_mask must have shape [N,H,W]")
    if frame_ids.shape != (frame_count,) or timestamps.shape != (frame_count,):
        raise ValueError("camera frame ids/timestamps must have shape [N]")
    if camera_k.shape != (frame_count, 3, 3):
        raise ValueError("frame_camera_K must have shape [N,3,3]")
    if depth_scales.shape != (frame_count,):
        raise ValueError("frame_depth_scale_m_per_unit must have shape [N]")
    if depth_range.shape != (2,):
        raise ValueError("depth_range_m must have shape [2]")
    if not np.allclose(camera_k, camera_k[0], atol=0.0, rtol=0.0):
        raise ValueError("camera intrinsics changed inside one saved archive")
    if not np.all(np.diff(frame_ids.astype(np.int64)) > 0):
        raise ValueError("camera_frame_id must strictly increase")
    if not np.all(np.diff(timestamps.astype(np.float64)) > 0.0):
        raise ValueError("camera_timestamp_s must strictly increase")
    feature_mode = str(point_feature_mode).strip().lower()
    if feature_mode not in ("xyz", "xyzrgb"):
        raise ValueError("point_feature_mode must be xyz or xyzrgb")

    projector = MaskedRGBDProjector(
        camera_K=camera_k[0],
        T_base_camera_optical=T_base_camera,
        image_size=(width, height),
        depth_range_m=(float(depth_range[0]), float(depth_range[1])),
        num_points=128,
        minimum_valid_points=16,
        point_feature_dim=3 if feature_mode == "xyz" else 6,
        maximum_mask_depth_deviation_m=maximum_mask_depth_deviation_m,
        fixed_sphere_completion_radius_m=fixed_sphere_radius_m,
    )

    statuses = []
    source_counts = []
    valid_counts = []
    timings_ms = []
    provenance = []
    completion_count = 0
    centroids_base = []
    extents_base = []
    for index in range(frame_count):
        started = time.perf_counter()
        point_frame = projector.project(
            color_bgr=rgb[index],
            depth_raw=depth_raw[index],
            depth_scale_m_per_unit=float(depth_scales[index]),
            object_mask=np.asarray(masks[index], dtype=bool),
            # The camera-only archive has no capture-time palm transform.
            # Identity therefore makes the returned XYZ robot_base.
            T_base_palm_at_capture=np.eye(4, dtype=np.float64),
            captured_at_s=float(timestamps[index]),
            frame_id=int(frame_ids[index]),
        )
        timings_ms.append((time.perf_counter() - started) * 1000.0)
        statuses.append(str(point_frame.status))
        source_counts.append(int(point_frame.source_valid_points))
        valid = np.asarray(point_frame.valid) > 0.5
        valid_counts.append(int(np.count_nonzero(valid)))
        provenance.append(
            str(projector.last_effective_object_mask_provenance.kind)
        )
        completion_count += int(
            bool(
                projector.last_sphere_completion is not None
                and projector.last_sphere_completion.get("applied", False)
            )
        )
        if np.any(valid):
            xyz = np.asarray(point_frame.xyzrgb_palm)[valid, :3]
            centroids_base.append(np.mean(xyz, axis=0))
            extents_base.append(np.max(xyz, axis=0) - np.min(xyz, axis=0))

    status_counts = {
        status: int(statuses.count(status)) for status in sorted(set(statuses))
    }
    centroids = np.asarray(centroids_base, dtype=np.float64)
    extents = np.asarray(extents_base, dtype=np.float64)
    return {
        "schema": "saved_real_rgbd_provider_mask_projector_check_v1",
        "source": str(archive_path),
        "source_sha256": _sha256(archive_path),
        "source_contract": (
            "camera_only_npz_with_stored_provider_publications; "
            "not_full_provider_replay"
        ),
        "frames": int(frame_count),
        "image_size_wh": [int(width), int(height)],
        "camera_serial": camera_serial,
        "calibration_id": calibration_id,
        "stored_object_text": object_text,
        "stored_requested_object_mask_mode": requested_object_mask_mode,
        "prompt_grounding_evaluated": False,
        "current_provider_tracking_evaluated": False,
        "point_feature_mode": feature_mode,
        "coordinate_frame": "robot_base_via_identity_T_base_palm",
        "status_counts": status_counts,
        "fresh_fraction": float(status_counts.get("fresh", 0) / frame_count),
        "source_valid_points_min_p50_max": [
            int(np.min(source_counts)),
            _percentile(np.asarray(source_counts), 50.0),
            int(np.max(source_counts)),
        ],
        "output_valid_points_min_p50_max": [
            int(np.min(valid_counts)),
            _percentile(np.asarray(valid_counts), 50.0),
            int(np.max(valid_counts)),
        ],
        "projector_full_128_fraction_including_initialization": float(
            np.count_nonzero(np.asarray(valid_counts, dtype=np.int64) == 128)
            / frame_count
        ),
        "projector_compute_ms_p50_p95_max": [
            _percentile(np.asarray(timings_ms), 50.0),
            _percentile(np.asarray(timings_ms), 95.0),
            float(np.max(timings_ms)),
        ],
        "projector_mask_provenance": sorted(set(provenance)),
        "fixed_sphere_completion_frames": int(completion_count),
        "centroid_std_base_mm": (
            (np.std(centroids, axis=0) * 1000.0).tolist()
            if centroids.size
            else None
        ),
        "extent_base_m_p50": (
            np.median(extents, axis=0).tolist() if extents.size else None
        ),
        "hardware_interfaces_opened": False,
        "limitations": [
            "stored masks are historical provider outputs; current provider is not rerun",
            "a non-empty historical mask does not validate the stored text prompt or grounding",
            "only stored valid publications are present; rejected/occluded gaps are absent",
            "no capture-time T_base_palm is stored, so palm-frame coordinates are not checked",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hardware-free validation of saved real RGB-D/provider-mask NPZ "
            "data through an explicit replay mode and the formal 128-point "
            "projector."
        )
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--point-feature-mode", choices=("xyz", "xyzrgb"), default="xyz"
    )
    parser.add_argument(
        "--maximum-mask-depth-deviation-m",
        type=float,
        default=DEFAULT_MAXIMUM_MASK_DEPTH_DEVIATION_M,
    )
    parser.add_argument("--fixed-sphere-radius-m", type=float)
    parser.add_argument(
        "--minimum-fresh-fraction",
        type=float,
        default=DEFAULT_MINIMUM_FRESH_FRACTION,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional exclusive/atomic summary JSON artifact path",
    )
    replay_mode = parser.add_mutually_exclusive_group()
    replay_mode.add_argument(
        "--rerun-current-provider-adaptive",
        action="store_true",
        help=(
            "use frame 0 as an explicit stored-mask initialization, then "
            "rerun the current adaptive provider/guard on remaining frames; "
            "SAM2 and prompt grounding stay disabled"
        ),
    )
    replay_mode.add_argument(
        "--rerun-current-provider-guarded-v2",
        action="store_true",
        help=(
            "production acceptance: seed current ObjectPCDProvider and its "
            "real local temporal SAM2 service from frame 0's stored mask, "
            "then run guarded_sam2_primary and the formal 128-point projector "
            "on every later saved RGB-D frame"
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--minimum-provider-valid-fraction",
        type=float,
        default=DEFAULT_MINIMUM_PROVIDER_VALID_FRACTION,
    )
    parser.add_argument(
        "--minimum-sam2-track-call-fraction",
        type=float,
        default=DEFAULT_MINIMUM_SAM2_TRACK_CALL_FRACTION,
        help=(
            "guarded-v2 mode only: minimum fraction of post-seed frames with "
            "an audited temporal SAM2 track call"
        ),
    )
    parser.add_argument(
        "--minimum-full-128-fraction",
        type=float,
        default=DEFAULT_MINIMUM_FULL_128_FRACTION,
        help=(
            "guarded-v2 mode only: minimum fraction of frames whose formal "
            "projector output has all 128 validity entries set"
        ),
    )
    return parser


def _validated_fraction(value: Any, *, option: str) -> float:
    number = float(value)
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{option} must be finite and in 0..1")
    return number


def _finalize_acceptance_artifact(
    summary: Dict[str, Any],
    *,
    guarded_v2_mode: bool,
    adaptive_mode: bool,
    point_feature_mode: str,
    maximum_depth_deviation: Optional[float],
    fixed_sphere_radius_m: Optional[float],
    minimum_fresh_fraction: float,
    minimum_provider_valid_fraction: float,
    minimum_sam2_track_call_fraction: float,
    minimum_full_128_fraction: float,
) -> Dict[str, Any]:
    thresholds = {
        "minimum_fresh_fraction": minimum_fresh_fraction,
        "minimum_provider_valid_fraction": minimum_provider_valid_fraction,
        "minimum_sam2_track_call_fraction": minimum_sam2_track_call_fraction,
        "minimum_full_128_fraction": minimum_full_128_fraction,
        "maximum_capture_retrieved_wall_gap_p95_s": (
            PRODUCTION_MAX_RETRIEVED_GAP_P95_S
        ),
        "maximum_capture_retrieved_wall_gap_s": PRODUCTION_MAX_RETRIEVED_GAP_S,
        "maximum_capture_retrieved_over_50ms_fraction": (
            PRODUCTION_MAX_RETRIEVED_OVER_50MS_FRACTION
        ),
        "maximum_provider_plus_projector_compute_p95_ms": (
            PRODUCTION_MAX_PROVIDER_PROJECTOR_COMPUTE_P95_MS
        ),
        "maximum_provider_plus_projector_compute_ms": (
            PRODUCTION_MAX_PROVIDER_PROJECTOR_COMPUTE_MS
        ),
        "maximum_provider_plus_projector_over_50ms_fraction": (
            PRODUCTION_MAX_PROVIDER_PROJECTOR_OVER_50MS_FRACTION
        ),
        "maximum_provider_plus_projector_over_50ms_longest_run": (
            PRODUCTION_MAX_PROVIDER_PROJECTOR_OVER_50MS_LONGEST_RUN
        ),
    }
    defaults_reasons: List[str] = []
    if not guarded_v2_mode:
        defaults_reasons.append("not_guarded_v2_rerun_mode")
    if point_feature_mode != "xyz":
        defaults_reasons.append("point_feature_mode_override")
    if maximum_depth_deviation is None:
        defaults_reasons.append("mask_depth_deviation_disabled")
    elif not np.isclose(
        maximum_depth_deviation,
        DEFAULT_MAXIMUM_MASK_DEPTH_DEVIATION_M,
        atol=0.0,
        rtol=0.0,
    ):
        defaults_reasons.append("mask_depth_deviation_override")
    if fixed_sphere_radius_m is not None:
        defaults_reasons.append("fixed_sphere_shape_completion_enabled")
    default_threshold_pairs = (
        (
            minimum_fresh_fraction,
            DEFAULT_MINIMUM_FRESH_FRACTION,
            "minimum_fresh_fraction_override",
        ),
        (
            minimum_provider_valid_fraction,
            DEFAULT_MINIMUM_PROVIDER_VALID_FRACTION,
            "minimum_provider_valid_fraction_override",
        ),
        (
            minimum_sam2_track_call_fraction,
            DEFAULT_MINIMUM_SAM2_TRACK_CALL_FRACTION,
            "minimum_sam2_track_call_fraction_override",
        ),
        (
            minimum_full_128_fraction,
            DEFAULT_MINIMUM_FULL_128_FRACTION,
            "minimum_full_128_fraction_override",
        ),
    )
    for actual, expected, reason in default_threshold_pairs:
        if actual != expected:
            defaults_reasons.append(reason)
    production_defaults_used = not defaults_reasons

    fresh_fraction = float(
        summary.get(
            "fresh_fraction",
            summary.get("projector_fresh_fraction_including_initialization", 0.0),
        )
    )
    full_128_fraction = float(
        summary.get("projector_full_128_fraction_including_initialization", 0.0)
    )
    current_provenance_fraction = float(
        summary.get(
            "projector_current_frame_provenance_fraction",
            1.0 if summary.get("projector_mask_provenance") == [
                "current_frame_projector_input_from_provider_mask"
            ] else 0.0,
        )
    )
    geometry_checks = [
        {
            "check": "projector_fresh_fraction",
            "category": "geometry",
            "actual": fresh_fraction,
            "expected_min": minimum_fresh_fraction,
            "passed": fresh_fraction >= minimum_fresh_fraction,
        },
        {
            "check": "projector_full_128_fraction",
            "category": "geometry",
            "actual": full_128_fraction,
            "expected_min": minimum_full_128_fraction,
            "passed": full_128_fraction >= minimum_full_128_fraction,
        },
        {
            "check": "projector_current_frame_provenance_fraction",
            "category": "geometry",
            "actual": current_provenance_fraction,
            "expected_min": 1.0,
            "passed": current_provenance_fraction == 1.0,
        },
    ]
    provider_valid_fraction = float(
        summary.get("current_provider_valid_fraction", 0.0)
    )
    sam2_track_fraction = float(
        (summary.get("sam2_temporal_service") or {}).get(
            "track_call_fraction", 0.0
        )
    )
    pipeline_checks = [
        {
            "check": "guarded_v2_provider_valid_fraction",
            "category": "pipeline_identity",
            "actual": provider_valid_fraction,
            "expected_min": minimum_provider_valid_fraction,
            "passed": guarded_v2_mode
            and provider_valid_fraction >= minimum_provider_valid_fraction,
        },
        {
            "check": "temporal_sam2_track_call_fraction",
            "category": "pipeline_identity",
            "actual": sam2_track_fraction,
            "expected_min": minimum_sam2_track_call_fraction,
            "passed": guarded_v2_mode
            and sam2_track_fraction >= minimum_sam2_track_call_fraction,
        },
        {
            "check": "production_evidence_provenance",
            "category": "pipeline_identity",
            "passed": bool(summary.get("production_evidence_eligible", False)),
        },
    ]
    combined_timing = summary.get("provider_plus_projector_compute_ms")
    if not isinstance(combined_timing, dict):
        combined_timing = {}
    combined_p95 = combined_timing.get("p95")
    combined_max = combined_timing.get("max")
    combined_over_fraction = combined_timing.get("over_50ms_fraction")
    combined_over_longest = combined_timing.get("over_50ms_longest_run")
    combined_samples = combined_timing.get("sample_count")
    combined_timing_available = bool(
        all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and np.isfinite(float(value))
            for value in (
                combined_p95,
                combined_max,
                combined_over_fraction,
                combined_over_longest,
                combined_samples,
            )
        )
        and int(combined_samples) >= 1
    )
    pipeline_timing_checks = [
        {
            "check": "provider_plus_projector_timing_available",
            "category": "pipeline_timing",
            "passed": guarded_v2_mode and combined_timing_available,
        },
        {
            "check": "provider_plus_projector_compute_p95",
            "category": "pipeline_timing",
            "actual_ms": combined_p95,
            "expected_max_ms": PRODUCTION_MAX_PROVIDER_PROJECTOR_COMPUTE_P95_MS,
            "passed": combined_timing_available
            and float(combined_p95)
            <= PRODUCTION_MAX_PROVIDER_PROJECTOR_COMPUTE_P95_MS,
        },
        {
            "check": "provider_plus_projector_compute_max",
            "category": "pipeline_timing",
            "actual_ms": combined_max,
            "expected_max_ms": PRODUCTION_MAX_PROVIDER_PROJECTOR_COMPUTE_MS,
            "passed": combined_timing_available
            and float(combined_max) <= PRODUCTION_MAX_PROVIDER_PROJECTOR_COMPUTE_MS,
        },
        {
            "check": "provider_plus_projector_over_50ms_fraction",
            "category": "pipeline_timing",
            "actual": combined_over_fraction,
            "expected_max": PRODUCTION_MAX_PROVIDER_PROJECTOR_OVER_50MS_FRACTION,
            "passed": combined_timing_available
            and float(combined_over_fraction)
            <= PRODUCTION_MAX_PROVIDER_PROJECTOR_OVER_50MS_FRACTION,
        },
        {
            "check": "provider_plus_projector_over_50ms_longest_run",
            "category": "pipeline_timing",
            "actual": combined_over_longest,
            "expected_max": PRODUCTION_MAX_PROVIDER_PROJECTOR_OVER_50MS_LONGEST_RUN,
            "passed": combined_timing_available
            and int(combined_over_longest)
            <= PRODUCTION_MAX_PROVIDER_PROJECTOR_OVER_50MS_LONGEST_RUN,
        },
    ]
    cadence_accepted = bool(
        (summary.get("capture_cadence_evidence") or {}).get(
            "accepted_end_to_end_cadence", False
        )
    )
    cadence_check = {
        "check": "capture_cadence_20hz_evidence",
        "category": "capture_cadence",
        "passed": cadence_accepted,
    }
    accepted_geometry = bool(all(item["passed"] for item in geometry_checks))
    accepted_pipeline_identity = bool(
        all(item["passed"] for item in pipeline_checks)
    )
    accepted_pipeline_timing = bool(
        all(item["passed"] for item in pipeline_timing_checks)
    )
    accepted = bool(
        production_defaults_used
        and accepted_geometry
        and accepted_pipeline_identity
        and accepted_pipeline_timing
        and cadence_accepted
    )
    summary.update(
        {
            "acceptance_profile": (
                "production_guarded_v2_saved_rgbd_geometry_and_cadence_v2"
                if production_defaults_used
                else "diagnostic_saved_rgbd_v2"
            ),
            "thresholds": thresholds,
            "production_defaults_used": production_defaults_used,
            "diagnostic_only": not production_defaults_used,
            "diagnostic_reasons": defaults_reasons,
            "checks": (
                geometry_checks
                + pipeline_checks
                + pipeline_timing_checks
                + [cadence_check]
            ),
            "accepted_geometry": accepted_geometry,
            "accepted_pipeline_identity": accepted_pipeline_identity,
            "accepted_pipeline_timing": accepted_pipeline_timing,
            "accepted_end_to_end_cadence": cadence_accepted,
            "accepted": accepted,
            "acceptance_notice": (
                "Geometry, guarded-v2/service identity, and original capture "
                "cadence are independent gates. The pinned legacy archive may "
                "pass geometry while failing end-to-end 20 Hz cadence."
            ),
            "adaptive_mode_requested": bool(adaptive_mode),
        }
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        minimum_fresh_fraction = _validated_fraction(
            args.minimum_fresh_fraction, option="--minimum-fresh-fraction"
        )
        minimum_provider_valid_fraction = _validated_fraction(
            args.minimum_provider_valid_fraction,
            option="--minimum-provider-valid-fraction",
        )
        minimum_sam2_track_call_fraction = _validated_fraction(
            args.minimum_sam2_track_call_fraction,
            option="--minimum-sam2-track-call-fraction",
        )
        minimum_full_128_fraction = _validated_fraction(
            args.minimum_full_128_fraction,
            option="--minimum-full-128-fraction",
        )
        raw_maximum_depth_deviation = float(
            args.maximum_mask_depth_deviation_m
        )
        if not np.isfinite(raw_maximum_depth_deviation):
            raise ValueError(
                "--maximum-mask-depth-deviation-m must be finite"
            )
        maximum_depth_deviation = (
            None
            if raw_maximum_depth_deviation <= 0.0
            else raw_maximum_depth_deviation
        )
        fixed_sphere_radius = (
            None
            if args.fixed_sphere_radius_m is None
            else float(args.fixed_sphere_radius_m)
        )
        if fixed_sphere_radius is not None and (
            not np.isfinite(fixed_sphere_radius) or fixed_sphere_radius <= 0.0
        ):
            raise ValueError("--fixed-sphere-radius-m must be finite and positive")
        if args.rerun_current_provider_guarded_v2:
            summary = evaluate_guarded_v2_provider_from_saved_rgbd_npz(
                args.archive,
                config_path=args.config,
                point_feature_mode=args.point_feature_mode,
                maximum_mask_depth_deviation_m=maximum_depth_deviation,
                fixed_sphere_radius_m=fixed_sphere_radius,
            )
        elif args.rerun_current_provider_adaptive:
            summary = evaluate_current_provider_from_saved_rgbd_npz(
                args.archive,
                config_path=args.config,
                point_feature_mode=args.point_feature_mode,
                maximum_mask_depth_deviation_m=maximum_depth_deviation,
                fixed_sphere_radius_m=fixed_sphere_radius,
            )
        else:
            summary = evaluate_saved_rgbd_npz(
                args.archive,
                point_feature_mode=args.point_feature_mode,
                maximum_mask_depth_deviation_m=maximum_depth_deviation,
                fixed_sphere_radius_m=fixed_sphere_radius,
            )
        summary = _finalize_acceptance_artifact(
            summary,
            guarded_v2_mode=bool(args.rerun_current_provider_guarded_v2),
            adaptive_mode=bool(args.rerun_current_provider_adaptive),
            point_feature_mode=str(args.point_feature_mode),
            maximum_depth_deviation=maximum_depth_deviation,
            fixed_sphere_radius_m=fixed_sphere_radius,
            minimum_fresh_fraction=minimum_fresh_fraction,
            minimum_provider_valid_fraction=minimum_provider_valid_fraction,
            minimum_sam2_track_call_fraction=(
                minimum_sam2_track_call_fraction
            ),
            minimum_full_128_fraction=minimum_full_128_fraction,
        )
        if args.output is not None:
            _write_json_exclusive_atomic(args.output, summary)
            print(f"[SAVED] {args.output.expanduser().resolve()}")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if bool(summary["accepted"]) else 1
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[validation failed] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
