"""Single-source V94 observations for the bounded real runtime.

The provider in this module never opens Franka or RH56 independently.  Arm
state comes only from :class:`FrankaPersistentSession.pose_ring`; hand state
comes only from the immutable cache published by the dedicated RH56 owner.
The camera has its own producer because RGB-D/mask computation is independent
of both actuator ownership domains.

``snapshot`` is proposal-safe: it does not advance policy history, mapper
state, previous action, or a velocity tracker.  Velocities are derived from
immutable sensor histories supplied by their owners.  The projector retains
the previous valid camera cloud for provenance and explicit legacy A/B use,
but guarded deployment modes reject every ``stale_palm`` result as a
recoverable no-stage hold before policy inference.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from robot_control.franka.session import FrankaPoseRing, FrankaPoseSample
from sim2real.closed_loop_core import TransactionalV94ActionMapper
from sim2real.observation.live_preview import (
    PROVIDER_OUTPUT_MODES,
    _CameraSample,
    _Latest,
    _camera_worker,
    _initialize_provider,
    _validate_camera_contract,
)
from robot_control.rh56.actuator import RH56SafetyFeedback
from sim2real.observation.camera_profile import ThrownRolloutTriggerConfig
from sim2real.contracts.v94 import INITIAL_PREVIOUS_ACTION13, V94Contract
from sim2real.observation.kinematics import (
    RH56FeedbackMapper,
    RH56FingertipKinematics,
    T_base_policy_palm_from_franka,
)
from sim2real.observation.model import (
    MaskedRGBDProjector,
    PolicyPointFrame,
    PolicyRGBDInputs,
    PolicyRGBDResolutionAdapter,
    ProjectorEffectiveMaskProvenance,
    Proprio67Builder,
    resolve_policy_rgbd_resolution,
)
from sim2real.observation.visualization import (
    V94LiveVisualizationSample,
    V94LiveVisualizationSink,
)
from .v94_policy_tick_source import (
    OBJECT_POINTCLOUD_INVALID_HOLD_REASON,
    OBSERVATION_SNAPSHOT_CONTRACT,
    QD_G015_STARTUP_NON_ACTUATED_POLICY_STEPS,
    TransactionalBoundedV94PolicyTickSource,
    TransactionalStateV94Policy,
    V94FreshActuatorSnapshot,
    V94PolicyObservation,
    V94RecoverableObservationHold,
)
from sim2real.contracts.actions import (
    LEGACY_FRANKA_ACTION_CONTRACT_ID,
    QD_G015_FRANKA_ACTION_CONTRACT_ID,
    TransactionalRH56HardwareCommandShaper,
    V94ActionMapper,
    normalize_franka_action_contract_id,
)


class V94LiveObservationError(RuntimeError):
    """A stale, inconsistent, unavailable, or deadline-expired live snapshot."""


@dataclass(frozen=True)
class _ThrownRolloutTriggerDecision:
    armed_now: bool
    detected: bool
    reason: str
    frame_id: int
    timestamp_s: float
    mask_kind: str
    mask_area_px: int
    centroid_speed_px_s: float
    centroid_displacement_px: float


@dataclass
class _ThrownRolloutTriggerWaitState:
    """One trigger transaction spanning robot preparation and release."""

    trigger: "_ThrownObjectRolloutTrigger"
    started_monotonic_s: float
    configured_deadline_monotonic_s: float
    last_frame_id: Optional[int] = None
    last_decision: Optional[_ThrownRolloutTriggerDecision] = None
    observed_frames: int = 0
    prepared_monotonic_s: Optional[float] = None
    ready_monotonic_s: Optional[float] = None
    motion_detected_decision: Optional[_ThrownRolloutTriggerDecision] = None
    motion_detected_monotonic_s: Optional[float] = None
    tabletop_height_confirmed_frames: int = 0
    last_tabletop_center_base_z_m: Optional[float] = None
    last_tabletop_height_source_points: int = 0


class _ThrownObjectRolloutTrigger:
    """Pure exact-frame state machine for throw entry or release motion."""

    def __init__(self, config: ThrownRolloutTriggerConfig) -> None:
        self.config = config
        self._last_frame_id: Optional[int] = None
        self._previous_visible: Optional[tuple[float, float, float, int]] = None
        self._absent_frames = 0
        self._stable_frames = 0
        self._armed_reason: Optional[str] = None

    def observe(
        self,
        *,
        frame_id: int,
        timestamp_s: float,
        mask: Optional[np.ndarray],
        mask_kind: str,
    ) -> _ThrownRolloutTriggerDecision:
        frame = int(frame_id)
        timestamp = _finite(timestamp_s, "throw-trigger frame timestamp")
        if self._last_frame_id is not None and frame <= self._last_frame_id:
            raise V94LiveObservationError(
                "throw trigger requires strictly increasing camera frame IDs"
            )
        self._last_frame_id = frame
        valid_mask: Optional[np.ndarray] = None
        if mask is not None:
            candidate = np.asarray(mask, dtype=np.bool_)
            if candidate.ndim != 2:
                raise V94LiveObservationError(
                    "throw trigger mask must be a two-dimensional image"
                )
            if int(np.count_nonzero(candidate)) >= self.config.minimum_mask_area_px:
                valid_mask = candidate

        if valid_mask is None:
            self._absent_frames += 1
            self._stable_frames = 0
            self._previous_visible = None
            armed_now = False
            if (
                self._armed_reason is None
                and self.config.allow_absent_entry
                and self._absent_frames >= self.config.absent_arm_frames
            ):
                self._armed_reason = "empty_field"
                armed_now = True
            return _ThrownRolloutTriggerDecision(
                armed_now=armed_now,
                detected=False,
                reason="waiting_for_object_entry",
                frame_id=frame,
                timestamp_s=timestamp,
                mask_kind="none",
                mask_area_px=0,
                centroid_speed_px_s=0.0,
                centroid_displacement_px=0.0,
            )

        ys, xs = np.nonzero(valid_mask)
        center_x = float(np.mean(xs, dtype=np.float64))
        center_y = float(np.mean(ys, dtype=np.float64))
        area = int(xs.size)
        if self._armed_reason == "empty_field":
            return _ThrownRolloutTriggerDecision(
                armed_now=False,
                detected=True,
                reason="absent_to_visible_entry",
                frame_id=frame,
                timestamp_s=timestamp,
                mask_kind=str(mask_kind),
                mask_area_px=area,
                centroid_speed_px_s=0.0,
                centroid_displacement_px=0.0,
            )

        speed = 0.0
        displacement = 0.0
        area_ratio = 1.0
        previous = self._previous_visible
        if previous is None:
            self._stable_frames = 1
        else:
            previous_t, previous_x, previous_y, previous_area = previous
            dt = timestamp - previous_t
            if not math.isfinite(dt) or dt <= 0.0:
                raise V94LiveObservationError(
                    "throw trigger camera timestamps are not strictly increasing"
                )
            displacement = math.hypot(center_x - previous_x, center_y - previous_y)
            speed = displacement / dt
            area_ratio = float(area) / float(previous_area)
            locally_consistent = bool(
                self.config.minimum_area_ratio
                <= area_ratio
                <= self.config.maximum_area_ratio
            )
            if (
                self._armed_reason == "stable_object"
                and locally_consistent
                and speed >= self.config.trigger_min_centroid_speed_px_s
                and displacement >= self.config.trigger_min_displacement_px
            ):
                return _ThrownRolloutTriggerDecision(
                    armed_now=False,
                    detected=True,
                    reason="stable_to_fast_motion",
                    frame_id=frame,
                    timestamp_s=timestamp,
                    mask_kind=str(mask_kind),
                    mask_area_px=area,
                    centroid_speed_px_s=speed,
                    centroid_displacement_px=displacement,
                )
            if locally_consistent and speed <= (
                self.config.stable_max_centroid_speed_px_s
            ):
                self._stable_frames += 1
            else:
                self._stable_frames = 1
        self._previous_visible = (timestamp, center_x, center_y, area)
        self._absent_frames = 0
        armed_now = False
        if (
            self._armed_reason is None
            and self._stable_frames >= self.config.stable_arm_frames
        ):
            self._armed_reason = "stable_object"
            armed_now = True
        return _ThrownRolloutTriggerDecision(
            armed_now=armed_now,
            detected=False,
            reason=(
                "waiting_for_fast_motion"
                if self._armed_reason == "stable_object"
                else "waiting_for_stable_object"
            ),
            frame_id=frame,
            timestamp_s=timestamp,
            mask_kind=str(mask_kind),
            mask_area_px=area,
            centroid_speed_px_s=speed,
            centroid_displacement_px=displacement,
        )


_OBJECT_MASK_MODES = frozenset(
    ("guarded", "guarded_v2", "guarded_v1", "legacy")
)
GUARDED_STALE_PALM_HOLD_REASON = "guarded_stale_palm_fail_closed"


def _normalize_object_mask_modes(
    requested: object,
    effective: Optional[object],
) -> tuple[str, str]:
    requested_mode = str(requested).strip().lower()
    if requested_mode not in _OBJECT_MASK_MODES:
        raise ValueError(
            "requested_object_mask_mode must be guarded, guarded_v2, "
            "guarded_v1, or legacy"
        )
    effective_mode = (
        "guarded_v2"
        if effective is None and requested_mode == "guarded"
        else (
            requested_mode
            if effective is None
            else str(effective).strip().lower()
        )
    )
    if effective_mode not in _OBJECT_MASK_MODES:
        raise ValueError(
            "effective_object_mask_mode must be guarded, guarded_v2, "
            "guarded_v1, or legacy"
        )
    if requested_mode == "legacy" and effective_mode != "legacy":
        raise ValueError("legacy request must have effective legacy mask mode")
    if effective_mode == "legacy" and requested_mode != "legacy":
        raise ValueError(
            "legacy stale-palm compatibility requires an explicit legacy request"
        )
    if requested_mode == "guarded" and effective_mode != "guarded_v2":
        raise ValueError("guarded request must resolve to effective guarded_v2")
    return requested_mode, effective_mode


# Mirrors protocol.hpp.  These are the lower contact-band flags only; current
# errors and joint/Cartesian collision flags never reach this observation path
# because the native owner fails closed first.
_FRANKA_JOINT_CONTACT_STATUS = 1 << 1
_FRANKA_CARTESIAN_CONTACT_STATUS = 1 << 3
_FRANKA_CONTACT_STATUS_MASK = (
    _FRANKA_JOINT_CONTACT_STATUS | _FRANKA_CARTESIAN_CONTACT_STATUS
)


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: object, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


@runtime_checkable
class V94CameraSnapshotSource(Protocol):
    """Thread-safe latest-frame view published by the camera owner."""

    def snapshot(self) -> _CameraSample: ...

    def wait_for_frame_change(self, frame_id: int, *, timeout_s: float) -> bool: ...


@runtime_checkable
class RH56FeedbackHistorySource(Protocol):
    """Immutable history published by the dedicated RH56 serial owner."""

    def require_fresh_feedback_history(
        self,
        *,
        minimum_samples: int,
        maximum_age_s: float,
    ) -> Any: ...


@runtime_checkable
class RH56FingertipModel(Protocol):
    def positions_base(
        self,
        *,
        angle_act_register_order: np.ndarray,
        T_base_palm: np.ndarray,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class _RetainedPointcloudSource:
    """Fresh camera data that produced the retained native-palm cloud."""

    camera: _CameraSample
    T_base_palm_at_capture: np.ndarray
    source_valid_points: int


@dataclass(frozen=True)
class _AdaptedPolicyRGBDCache:
    """Private policy-resolution pixels for one exact camera publication."""

    # Identity, rather than frame_id alone, is intentional.  It prevents a
    # malformed/restarted producer that reuses a numeric frame id from pairing
    # new pixels with an old adaptation.  Holding the sample reference also
    # prevents Python object-id reuse while this one-entry cache is live.
    camera: _CameraSample
    policy_rgbd: PolicyRGBDInputs


@dataclass(frozen=True)
class _ProjectedPointcloudCache:
    """One exact-frame projection bound to its capture-time Franka pose."""

    camera: _CameraSample
    capture_pose_cycle: int
    T_base_palm_at_capture: np.ndarray
    point_frame: PolicyPointFrame
    visual_source: Optional[_RetainedPointcloudSource]
    mask_provenance: ProjectorEffectiveMaskProvenance


_PROVIDER_TIMING_NAMES_BY_MODE = {
    "mask_only": (
        "camera",
        "tracker",
        "sam2",
        "sam2_reinit",
        "mask_gate",
        "total",
    ),
    "full_packet": (
        "camera",
        "tracker",
        "sam2",
        "sam2_reinit",
        "pcd",
        "total",
    ),
}


def _provider_timing_names(camera: _CameraSample) -> tuple[str, ...]:
    mode = str(getattr(camera, "provider_output_mode", "mask_only"))
    return _PROVIDER_TIMING_NAMES_BY_MODE.get(
        mode,
        _PROVIDER_TIMING_NAMES_BY_MODE["mask_only"],
    )


def _private_readonly_array(
    value: np.ndarray,
    *,
    dtype: Any,
    possible_shared_sources: Sequence[np.ndarray] = (),
) -> np.ndarray:
    """Return C-contiguous storage that no producer or caller can mutate.

    Exact 2x policy decimation already creates owned arrays, so those buffers
    can be made read-only without another full image copy.  Native-resolution
    adaptation can alias the camera owner's arrays; explicit shares-memory
    checks detach that case before caching it.
    """

    result = np.asarray(value, dtype=dtype)
    shared = any(
        np.shares_memory(result, np.asarray(source))
        for source in possible_shared_sources
    )
    if shared or not result.flags.owndata or not result.flags.c_contiguous:
        result = np.array(result, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _private_policy_rgbd(
    value: PolicyRGBDInputs,
    *,
    camera: _CameraSample,
) -> PolicyRGBDInputs:
    """Detach one adapted RGB-D/mask tuple for the exact-frame cache."""

    source_depths = tuple(
        item
        for item in (camera.depth_m, camera.depth_raw)
        if item is not None
    )
    return PolicyRGBDInputs(
        color_bgr=_private_readonly_array(
            value.color_bgr,
            dtype=np.uint8,
            possible_shared_sources=(camera.color_bgr,),
        ),
        object_mask=_private_readonly_array(
            value.object_mask,
            dtype=np.bool_,
            possible_shared_sources=(camera.mask,),
        ),
        depth_m=(
            None
            if value.depth_m is None
            else _private_readonly_array(
                value.depth_m,
                dtype=np.float32,
                possible_shared_sources=source_depths,
            )
        ),
        depth_raw=(
            None
            if value.depth_raw is None
            else _private_readonly_array(
                value.depth_raw,
                dtype=np.uint16,
                possible_shared_sources=source_depths,
            )
        ),
    )


def _private_point_frame(value: PolicyPointFrame) -> PolicyPointFrame:
    """Copy the small final policy tensor into immutable cache storage."""

    return PolicyPointFrame(
        xyzrgb_palm=_private_readonly_array(
            value.xyzrgb_palm,
            dtype=np.float32,
        ),
        valid=_private_readonly_array(value.valid, dtype=np.float32),
        captured_at_s=float(value.captured_at_s),
        frame_id=int(value.frame_id),
        source_valid_points=int(value.source_valid_points),
        status=str(value.status),
        measured_at_s=(
            None if value.measured_at_s is None else float(value.measured_at_s)
        ),
        measured_frame_id=(
            None
            if value.measured_frame_id is None
            else int(value.measured_frame_id)
        ),
        fallback_age_s=float(value.fallback_age_s),
        fallback_steps=int(value.fallback_steps),
    )


@dataclass(frozen=True)
class LiveD435ProviderFactory:
    """Lazy, non-interactive constructor for the D435 object-mask provider.

    Calling the dataclass constructor is inert.  The camera/import/ROI path is
    reached only when :class:`D435ObjectCameraOwner.open` calls this object.
    A numeric ROI is mandatory here because ``None`` delegates to OpenCV's
    interactive ``selectROI`` window.  A Qt GUI must never be created from a
    deployment lifecycle or camera worker thread.
    """

    config_path: Path
    roi_xywh: Optional[Tuple[int, int, int, int]]
    disable_online_sam2: bool = False
    object_mask_mode: str = "guarded"
    required_runtime_frame_timeout_ms: Optional[int] = None

    construction_is_inert: bool = True

    def __post_init__(self) -> None:
        roi = self.roi_xywh
        if roi is None:
            raise ValueError(
                "live D435 deployment requires a fixed numeric ROI; "
                "interactive selectROI is forbidden"
            )
        try:
            raw = tuple(roi)
        except TypeError as exc:
            raise ValueError("live D435 ROI must contain x y width height") from exc
        if len(raw) != 4:
            raise ValueError("live D435 ROI must contain x y width height")
        values = []
        for index, value in enumerate(raw):
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"live D435 ROI[{index}] must be an integer")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"live D435 ROI[{index}] must be an integer") from exc
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise ValueError(f"live D435 ROI[{index}] must be an integer")
            values.append(int(numeric))
        x, y, width, height = values
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError(
                "live D435 ROI requires non-negative x/y and positive width/height"
            )
        object.__setattr__(self, "roi_xywh", (x, y, width, height))
        mask_mode = str(self.object_mask_mode).strip().lower()
        if mask_mode not in (
            "guarded",
            "guarded_v2",
            "guarded_v1",
            "legacy",
        ):
            raise ValueError(
                "object_mask_mode must be guarded, guarded_v2, guarded_v1, "
                "or legacy"
            )
        object.__setattr__(self, "object_mask_mode", mask_mode)
        timeout = self.required_runtime_frame_timeout_ms
        if timeout is not None:
            if (
                isinstance(timeout, (bool, np.bool_))
                or not isinstance(timeout, (int, np.integer))
                or int(timeout) <= 0
            ):
                raise ValueError(
                    "required_runtime_frame_timeout_ms must be a positive integer"
                )
            object.__setattr__(
                self,
                "required_runtime_frame_timeout_ms",
                int(timeout),
            )

    def __call__(self) -> Any:
        initialization_options: dict[str, object] = {
            "disable_online_sam2": bool(self.disable_online_sam2),
            "object_mask_mode": self.object_mask_mode,
        }
        if self.required_runtime_frame_timeout_ms is not None:
            initialization_options["required_runtime_frame_timeout_ms"] = (
                self.required_runtime_frame_timeout_ms
            )
        provider, _selection = _initialize_provider(
            Path(self.config_path),
            self.roi_xywh,
            **initialization_options,
        )
        return provider


class D435ObjectCameraOwner:
    """Own one initialized provider and publish immutable camera samples."""

    def __init__(
        self,
        *,
        provider_factory: Callable[[], Any],
        provider_output_mode: str = "mask_only",
        join_timeout_s: object = 5.0,
        maximum_publication_stall_s: object = 0.20,
        worker: Callable[[Any, threading.Event, _Latest, str], None] = _camera_worker,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(provider_factory):
            raise TypeError("provider_factory must be callable")
        if not callable(worker):
            raise TypeError("camera worker must be callable")
        if not callable(monotonic):
            raise TypeError("camera owner monotonic clock must be callable")
        mode = str(provider_output_mode)
        if mode not in PROVIDER_OUTPUT_MODES:
            raise ValueError(
                f"provider_output_mode must be one of {PROVIDER_OUTPUT_MODES}"
            )
        self._provider_factory = provider_factory
        self._provider_output_mode = mode
        self._join_timeout_s = _positive(join_timeout_s, "join_timeout_s")
        self._maximum_publication_stall_s = _positive(
            maximum_publication_stall_s,
            "maximum_publication_stall_s",
        )
        self._worker = worker
        self._monotonic = monotonic
        self._latest = _Latest()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._provider: Any = None
        self._thread: Optional[threading.Thread] = None
        self._opened = False
        self._closed = False
        # Production samples contain the RealSense retrieval monotonic time.
        # The first-seen fallback exists only for reviewed test/adapter samples
        # that predate that field; it still detects a cached frame that stops
        # changing instead of allowing an actuator watchdog to become the
        # misleading top-level fault.
        self._fallback_frame_id: Optional[int] = None
        self._fallback_frame_first_seen_monotonic_s: Optional[float] = None

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._opened and not self._closed

    @property
    def maximum_publication_stall_s(self) -> float:
        return self._maximum_publication_stall_s

    @property
    def support_plane_abcd(self) -> Optional[np.ndarray]:
        provider = self._provider
        extractor = getattr(provider, "extractor", None)
        plane = getattr(extractor, "support_plane_abcd", None)
        if plane is None:
            return None
        return np.asarray(plane, dtype=np.float64).copy()

    @property
    def support_plane_min_clearance_m(self) -> float:
        provider = self._provider
        extractor = getattr(provider, "extractor", None)
        return float(
            getattr(extractor, "support_plane_min_clearance_m", 0.0)
        )

    @property
    def pointcloud_temporal_fallback_config(self) -> dict[str, object]:
        provider = self._provider
        cfg = getattr(provider, "cfg", {})
        pointcloud = cfg.get("pointcloud", {}) if isinstance(cfg, dict) else {}
        if not isinstance(pointcloud, dict):
            return {}
        allowed = (
            "temporal_fallback",
            "temporal_fallback_max_stale_s",
            "temporal_fallback_max_stale_steps",
            "temporal_fallback_max_image_speed_px_s",
        )
        return {name: pointcloud[name] for name in allowed if name in pointcloud}

    def open(self) -> "D435ObjectCameraOwner":
        with self._lock:
            if self._opened or self._closed:
                raise V94LiveObservationError("D435 owner is single-use")
            # The factory is the first point permitted to touch the camera.
            provider = self._provider_factory()
            if provider is None:
                raise V94LiveObservationError("D435 provider factory returned None")
            self._provider = provider
            self._opened = True
            thread = threading.Thread(
                target=self._worker,
                args=(
                    provider,
                    self._stop,
                    self._latest,
                    self._provider_output_mode,
                ),
                name="v94-d435-object-owner",
                daemon=False,
            )
            self._thread = thread
            thread.start()
        return self

    def snapshot(self) -> _CameraSample:
        if not self.is_open:
            raise V94LiveObservationError("D435 owner is not open")
        try:
            value = self._latest.get()
        except BaseException as exc:
            raise V94LiveObservationError(
                "D435 background provider failed; "
                f"{self._provider_progress_text()}; {type(exc).__name__}: {exc}"
            ) from exc
        if value is None:
            raise V94LiveObservationError("D435 has not published a formal frame")
        if not isinstance(value, _CameraSample):
            raise V94LiveObservationError("D435 owner published an invalid sample")
        now = _finite(self._monotonic(), "camera owner monotonic clock")
        age_s = self._publication_age_s(value, now_monotonic_s=now)
        if age_s > self._maximum_publication_stall_s:
            message = " ".join(str(value.message).split())[:240]
            worker = self._latest.diagnostics_snapshot()
            raise V94LiveObservationError(
                "D435 formal publication stalled: "
                f"age={age_s:.6f}s, "
                f"limit={self._maximum_publication_stall_s:.6f}s, "
                f"last_frame={int(value.frame_id)}, "
                f"sensor_frame={int(value.sensor_frame_number)}, "
                f"mask_valid={bool(value.mask_valid)}, "
                f"mask_px={int(np.count_nonzero(value.mask))}, "
                f"mask_message={message!r}; "
                "retryable_capture_timeouts="
                f"{int(worker['consecutive_transient_error_count'])}/"
                f"{int(worker['transient_error_count'])}, "
                "last_retryable_capture_error="
                f"{str(worker['last_transient_error'])[:500]!r}; "
                f"{self._provider_progress_text(now_monotonic_s=now)}"
            )
        return value

    def _publication_age_s(
        self,
        sample: _CameraSample,
        *,
        now_monotonic_s: float,
    ) -> float:
        """Return age of the last formal producer progress in one clock."""

        retrieved = float(getattr(sample, "retrieved_monotonic_s", 0.0))
        if math.isfinite(retrieved) and retrieved > 0.0:
            age_s = now_monotonic_s - retrieved
            if age_s < -0.050:
                raise V94LiveObservationError(
                    "D435 retrieval monotonic timestamp is in the future"
                )
            return max(0.0, age_s)

        with self._lock:
            if self._fallback_frame_id != int(sample.frame_id):
                self._fallback_frame_id = int(sample.frame_id)
                self._fallback_frame_first_seen_monotonic_s = now_monotonic_s
            first_seen = self._fallback_frame_first_seen_monotonic_s
        if first_seen is None:
            raise V94LiveObservationError(
                "D435 fallback publication clock was not initialized"
            )
        return max(0.0, now_monotonic_s - first_seen)

    def _provider_progress_text(
        self,
        *,
        now_monotonic_s: Optional[float] = None,
    ) -> str:
        """Best-effort, read-only provider stage for actionable fault text."""

        provider = self._provider
        thread = self._thread
        stage = str(getattr(provider, "last_pipeline_stage", "unavailable"))
        stage_frame = getattr(provider, "last_pipeline_frame_id", None)
        started = getattr(
            provider,
            "last_pipeline_stage_started_monotonic_s",
            None,
        )
        if now_monotonic_s is None:
            try:
                now_monotonic_s = _finite(
                    self._monotonic(),
                    "camera owner monotonic clock",
                )
            except BaseException:
                now_monotonic_s = None
        stage_age_text = "unavailable"
        try:
            numeric_started = float(started)
            if (
                now_monotonic_s is not None
                and math.isfinite(numeric_started)
                and numeric_started > 0.0
            ):
                stage_age_text = f"{max(0.0, now_monotonic_s - numeric_started):.6f}s"
        except (TypeError, ValueError):
            pass
        return (
            f"provider_stage={stage!r}, "
            f"provider_stage_frame={stage_frame!r}, "
            f"provider_stage_age={stage_age_text}, "
            f"worker_alive={bool(thread is not None and thread.is_alive())}"
        )

    @property
    def diagnostics_snapshot(self) -> dict[str, object]:
        """Current camera-owner liveness without touching the camera SDK."""

        now = _finite(self._monotonic(), "camera owner monotonic clock")
        value: object = None
        snapshot_error: Optional[str] = None
        try:
            value = self._latest.get()
        except BaseException as exc:
            snapshot_error = " ".join(f"{type(exc).__name__}: {exc}".split())[:2000]
        age_s = (
            None
            if not isinstance(value, _CameraSample)
            else self._publication_age_s(value, now_monotonic_s=now)
        )
        provider = self._provider
        camera = getattr(provider, "camera", None)
        worker = self._latest.diagnostics_snapshot()
        last_wait_started = getattr(camera, "last_frame_wait_started_monotonic_s", None)
        last_wait_failed = getattr(camera, "last_frame_wait_failed_monotonic_s", None)

        def elapsed_since(value: object) -> Optional[float]:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(numeric) or numeric <= 0.0:
                return None
            return max(0.0, now - numeric)

        return {
            "last_frame_id": (
                None if not isinstance(value, _CameraSample) else int(value.frame_id)
            ),
            "last_sensor_frame_number": (
                None
                if not isinstance(value, _CameraSample)
                else int(value.sensor_frame_number)
            ),
            "last_depth_sensor_frame_number": (
                None
                if not isinstance(value, _CameraSample)
                else int(value.depth_sensor_frame_number)
            ),
            "last_mask_valid": (
                None if not isinstance(value, _CameraSample) else bool(value.mask_valid)
            ),
            "last_mask_area_px": (
                None
                if not isinstance(value, _CameraSample)
                else int(np.count_nonzero(value.mask))
            ),
            "last_capture_diagnostic_json": (
                None
                if not isinstance(value, _CameraSample)
                else str(value.capture_diagnostic_json)[:4000]
            ),
            "publication_age_s": age_s,
            "maximum_publication_stall_s": self._maximum_publication_stall_s,
            "provider_progress": self._provider_progress_text(now_monotonic_s=now),
            "online_sam2": {
                "enabled": bool(getattr(provider, "online_sam2_enabled", False)),
                "manager_created": (
                    getattr(provider, "online_sam2_manager", None) is not None
                ),
                "executor_created": (
                    getattr(provider, "_online_sam2_executor", None) is not None
                ),
                "last_status": str(
                    getattr(provider, "_last_online_sam2_status", "unavailable")
                ),
                "step_count": int(getattr(provider, "_online_sam2_step_count", 0)),
                "submit_count": int(getattr(provider, "_online_sam2_submit_count", 0)),
                "eligible_count": int(
                    getattr(provider, "_online_sam2_eligible_count", 0)
                ),
                "busy_skip_count": int(
                    getattr(provider, "_online_sam2_busy_skip_count", 0)
                ),
                "exact_count": int(getattr(provider, "_online_sam2_exact_count", 0)),
                "exact_coverage_fraction": (
                    None
                    if int(
                        getattr(provider, "_online_sam2_eligible_count", 0)
                    ) <= 0
                    else float(
                        getattr(provider, "_online_sam2_exact_count", 0)
                    )
                    / float(
                        getattr(provider, "_online_sam2_eligible_count", 0)
                    )
                ),
                "exact_valid_count": int(
                    getattr(provider, "_online_sam2_exact_valid_count", 0)
                ),
                "deadline_miss_count": int(
                    getattr(provider, "_online_sam2_deadline_miss_count", 0)
                ),
                "late_count": int(getattr(provider, "_online_sam2_late_count", 0)),
            },
            "worker": worker,
            "snapshot_error": snapshot_error,
            "transport": {
                "device_serial": getattr(camera, "device_serial", None),
                "device_name": getattr(camera, "device_name", None),
                "device_firmware_version": getattr(
                    camera, "device_firmware_version", None
                ),
                "device_usb_type_descriptor": getattr(
                    camera, "device_usb_type_descriptor", None
                ),
                "sdk_version": getattr(camera, "sdk_version", None),
                "runtime_frame_wait_timeouts": int(
                    getattr(camera, "runtime_frame_wait_timeouts", 0)
                ),
                "rejected_transport_stale_frames": int(
                    getattr(camera, "rejected_transport_stale_frames", 0)
                ),
                "rejected_timestamp_skew_frames": int(
                    getattr(camera, "rejected_timestamp_skew_frames", 0)
                ),
                "rejected_non_increasing_frames": int(
                    getattr(camera, "rejected_non_increasing_frames", 0)
                ),
                "dropped_queued_framesets": int(
                    getattr(camera, "dropped_queued_framesets", 0)
                ),
                "drained_pending_framesets": int(
                    getattr(camera, "drained_pending_framesets", 0)
                ),
                "last_accepted_sensor_frame_number": getattr(
                    camera, "last_accepted_sensor_frame_number", None
                ),
                "last_accepted_depth_sensor_frame_number": getattr(
                    camera, "last_accepted_depth_sensor_frame_number", None
                ),
                "last_frame_wait_started_age_s": elapsed_since(last_wait_started),
                "last_frame_wait_failed_age_s": elapsed_since(last_wait_failed),
            },
        }

    def wait_for_frame_change(self, frame_id: int, *, timeout_s: float) -> bool:
        if not self.is_open:
            raise V94LiveObservationError("D435 owner is not open")
        return self._latest.wait_for_frame_change(frame_id, timeout_s=timeout_s)

    def wait_for_next_fresh_publication(
        self,
        *,
        timeout_s: float,
    ) -> _CameraSample:
        """Require a newly published fresh frame at a pre-runtime boundary.

        A retained owner may keep running while checkpoint/native and fresh
        hardware preflights execute.  An old last value is never accepted at
        handoff: this waits for a strictly newer formal publication and then
        applies the unchanged runtime publication-age watchdog to that new
        sample.  It is used only before either actuator owner starts.
        """

        if not self.is_open:
            raise V94LiveObservationError("D435 owner is not open")
        timeout = _positive(timeout_s, "pre-runtime D435 resync timeout_s")
        deadline = self._monotonic() + timeout
        try:
            initial = self._latest.get()
        except BaseException as exc:
            raise V94LiveObservationError(
                "D435 background provider failed before pre-runtime resync; "
                f"{self._provider_progress_text()}; {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(initial, _CameraSample):
            raise V94LiveObservationError(
                "D435 has not published a frame before pre-runtime resync"
            )
        previous_frame_id = int(initial.frame_id)
        previous_timestamp_s = float(initial.timestamp_s)
        while True:
            remaining = deadline - self._monotonic()
            if remaining <= 0.0 or not self._latest.wait_for_frame_change(
                previous_frame_id,
                timeout_s=max(remaining, 1.0e-6),
            ):
                diagnostics = self._latest.diagnostics_snapshot()
                raise V94LiveObservationError(
                    "D435 did not publish a new formal frame during bounded "
                    f"pre-runtime resync: timeout={timeout:.6f}s, "
                    f"last_frame={previous_frame_id}, "
                    "retryable_capture_timeouts="
                    f"{int(diagnostics['consecutive_transient_error_count'])}/"
                    f"{int(diagnostics['transient_error_count'])}; "
                    f"{self._provider_progress_text()}"
                )
            try:
                candidate = self._latest.get()
            except BaseException as exc:
                raise V94LiveObservationError(
                    "D435 background provider failed during pre-runtime resync; "
                    f"{self._provider_progress_text()}; "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(candidate, _CameraSample):
                raise V94LiveObservationError(
                    "D435 pre-runtime resync published an invalid sample"
                )
            candidate_frame_id = int(candidate.frame_id)
            candidate_timestamp_s = float(candidate.timestamp_s)
            if (
                candidate_frame_id <= previous_frame_id
                or candidate_timestamp_s <= previous_timestamp_s
            ):
                raise V94LiveObservationError(
                    "D435 pre-runtime resync frame/timestamp did not strictly "
                    "advance"
                )
            age_s = self._publication_age_s(
                candidate,
                now_monotonic_s=self._monotonic(),
            )
            if age_s <= self._maximum_publication_stall_s:
                return candidate
            previous_frame_id = candidate_frame_id
            previous_timestamp_s = candidate_timestamp_s

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
            provider = self._provider
        self._stop.set()
        if thread is not None:
            thread.join(timeout=self._join_timeout_s)
            if thread.is_alive():
                # provider.step() and provider.stop() have no documented
                # concurrent-cancellation contract.  Do not race them.
                raise V94LiveObservationError(
                    "D435 worker did not stop; provider.stop was not called concurrently"
                )
        if provider is not None:
            stop = getattr(provider, "stop", None)
            if not callable(stop):
                raise V94LiveObservationError("D435 provider has no stop method")
            stop()


_PREWARMED_D435_HANDOFF_SEAL = object()


@dataclass(frozen=True)
class _PrewarmedD435Binding:
    roi_xywh: tuple[int, int, int, int]
    object_mask_mode: str
    pcd_config_sha256: str
    checkpoint_sha256: str
    preflight_sha256: str
    camera_serial: str
    calibration_id: str
    last_preflight_frame_id: int
    last_preflight_timestamp_s: float
    adopted_frame_id: int
    adopted_timestamp_s: float


class PrewarmedD435CameraHandoff:
    """Single-use, exact-preflight-bound ownership transfer into rollout.

    The owner is already publishing while CPU-only policy/native checks run,
    so the provider's tracker never sees an artificial close/reopen gap. A
    consumer must reproduce every immutable preflight binding before taking
    the owner. ``close`` is idempotent so both deployment layers may request
    cleanup without stopping the provider twice.
    """

    def __init__(
        self,
        *,
        _seal: object,
        owner: D435ObjectCameraOwner,
        binding: _PrewarmedD435Binding,
    ) -> None:
        if _seal is not _PREWARMED_D435_HANDOFF_SEAL:
            raise TypeError("prewarmed D435 handoff is internally constructed")
        if not isinstance(owner, D435ObjectCameraOwner) or not owner.is_open:
            raise ValueError("prewarmed D435 handoff requires an open owner")
        if (
            int(binding.adopted_frame_id) <= int(binding.last_preflight_frame_id)
            or float(binding.adopted_timestamp_s)
            <= float(binding.last_preflight_timestamp_s)
        ):
            raise ValueError(
                "prewarmed D435 handoff adoption did not advance beyond preflight"
            )
        self._owner = owner
        self._binding = binding
        self._lock = threading.Lock()
        self._consumed = False

    @property
    def is_open(self) -> bool:
        return self._owner.is_open

    def consume(
        self,
        *,
        contract: V94Contract,
        roi_xywh: Sequence[int],
        object_mask_mode: str,
        pcd_config_sha256: str,
        checkpoint_sha256: str,
        preflight_sha256: str,
    ) -> D435ObjectCameraOwner:
        requested, _effective = _normalize_object_mask_modes(
            object_mask_mode,
            None,
        )
        expected = _PrewarmedD435Binding(
            roi_xywh=tuple(int(value) for value in roi_xywh),
            object_mask_mode=requested,
            pcd_config_sha256=str(pcd_config_sha256).lower(),
            checkpoint_sha256=str(checkpoint_sha256).lower(),
            preflight_sha256=str(preflight_sha256).lower(),
            camera_serial=str(contract.camera_serial),
            calibration_id=str(contract.calibration_id),
            last_preflight_frame_id=self._binding.last_preflight_frame_id,
            last_preflight_timestamp_s=self._binding.last_preflight_timestamp_s,
            adopted_frame_id=self._binding.adopted_frame_id,
            adopted_timestamp_s=self._binding.adopted_timestamp_s,
        )
        with self._lock:
            if self._consumed:
                raise V94LiveObservationError(
                    "prewarmed D435 handoff was already consumed"
                )
            if self._binding != expected or not self._owner.is_open:
                raise V94LiveObservationError(
                    "prewarmed D435 handoff differs from the current ROI/artifacts"
                )
            self._consumed = True
            return self._owner

    def close(self) -> None:
        self._owner.close()


def open_prevalidated_d435_handoff(
    *,
    provider: Any,
    contract: V94Contract,
    roi_xywh: Sequence[int],
    object_mask_mode: str,
    pcd_config_sha256: str,
    checkpoint_sha256: str,
    preflight_sha256: str,
    last_preflight_frame_id: int,
    last_preflight_timestamp_s: float,
    maximum_publication_stall_s: float,
) -> PrewarmedD435CameraHandoff:
    """Adopt one validated provider and prove continuous newer publication."""

    requested, _effective = _normalize_object_mask_modes(object_mask_mode, None)
    provider_slot = [provider]

    def take_provider() -> Any:
        if not provider_slot:
            raise V94LiveObservationError(
                "prevalidated D435 provider may be adopted exactly once"
            )
        return provider_slot.pop()

    owner = D435ObjectCameraOwner(
        provider_factory=take_provider,
        maximum_publication_stall_s=maximum_publication_stall_s,
    )
    try:
        owner.open()
        if not owner.wait_for_frame_change(
            int(last_preflight_frame_id),
            timeout_s=float(maximum_publication_stall_s),
        ):
            raise V94LiveObservationError(
                "prewarmed D435 handoff did not publish a newer frame"
            )
        sample = owner.snapshot()
        _validate_camera_contract(sample, contract)
        if (
            int(sample.frame_id) <= int(last_preflight_frame_id)
            or float(sample.timestamp_s) <= float(last_preflight_timestamp_s)
        ):
            raise V94LiveObservationError(
                "prewarmed D435 handoff frame/timestamp did not advance"
            )
        binding = _PrewarmedD435Binding(
            roi_xywh=tuple(int(value) for value in roi_xywh),
            object_mask_mode=requested,
            pcd_config_sha256=str(pcd_config_sha256).lower(),
            checkpoint_sha256=str(checkpoint_sha256).lower(),
            preflight_sha256=str(preflight_sha256).lower(),
            camera_serial=str(contract.camera_serial),
            calibration_id=str(contract.calibration_id),
            last_preflight_frame_id=int(last_preflight_frame_id),
            last_preflight_timestamp_s=float(last_preflight_timestamp_s),
            adopted_frame_id=int(sample.frame_id),
            adopted_timestamp_s=float(sample.timestamp_s),
        )
        return PrewarmedD435CameraHandoff(
            _seal=_PREWARMED_D435_HANDOFF_SEAL,
            owner=owner,
            binding=binding,
        )
    except BaseException:
        try:
            owner.close()
        except BaseException:
            pass
        raise


def _nearest_pose(
    samples: Sequence[FrankaPoseSample],
    timestamp_s: float,
    maximum_skew_s: float,
) -> Tuple[FrankaPoseSample, float]:
    if not samples:
        raise V94LiveObservationError("Franka pose ring is empty")
    skew, sample = min(
        ((abs(item.realtime_s - timestamp_s), item) for item in samples),
        key=lambda pair: pair[0],
    )
    if skew > maximum_skew_s:
        raise V94LiveObservationError(
            f"camera/Franka capture pose skew {skew:.6f}s exceeds "
            f"{maximum_skew_s:.6f}s"
        )
    return sample, float(skew)


def _palm_twist(
    previous: FrankaPoseSample,
    latest: FrankaPoseSample,
    *,
    F_T_EE: np.ndarray,
    T_flange_policy_palm: np.ndarray,
    maximum_dt_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    dt = latest.realtime_s - previous.realtime_s
    if not math.isfinite(dt) or dt <= 0.0 or dt > maximum_dt_s:
        raise V94LiveObservationError(
            f"unsafe Franka pose finite-difference interval {dt!r}s"
        )
    prior_palm = T_base_policy_palm_from_franka(
        T_base_ee=previous.T_base_eef,
        F_T_EE=F_T_EE,
        T_flange_policy_palm=T_flange_policy_palm,
    )
    current_palm = T_base_policy_palm_from_franka(
        T_base_ee=latest.T_base_eef,
        F_T_EE=F_T_EE,
        T_flange_policy_palm=T_flange_policy_palm,
    )
    linear = (current_palm[:3, 3] - prior_palm[:3, 3]) / dt
    relative = current_palm[:3, :3] @ prior_palm[:3, :3].T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew = np.asarray(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1.0e-8:
        rotation_vector = 0.5 * skew
    else:
        sine = math.sin(angle)
        if abs(sine) < 1.0e-12:
            raise V94LiveObservationError("Franka palm angular velocity is singular")
        rotation_vector = skew * (angle / (2.0 * sine))
    return linear.astype(np.float32), (rotation_vector / dt).astype(np.float32)


class PersistentV94ObservationOwner:
    """Build V94 observations from the three production owner caches."""

    transactional_snapshot_contract = OBSERVATION_SNAPSHOT_CONTRACT

    def __init__(
        self,
        *,
        contract: V94Contract,
        franka_pose_ring: FrankaPoseRing,
        expected_F_T_EE: np.ndarray,
        rh56_feedback_source: RH56FeedbackHistorySource,
        camera_source: V94CameraSnapshotSource,
        maximum_pose_skew_s: object = 0.025,
        maximum_franka_action_age_s: object = 0.025,
        maximum_franka_age_s: object = 0.050,
        maximum_rh56_age_s: object = 0.025,
        maximum_future_skew_s: object = 0.010,
        maximum_velocity_dt_s: object = 0.10,
        point_feature_dim: object = 6,
        policy_rgbd_resolution: str = "848x480",
        requested_object_mask_mode: str = "guarded",
        effective_object_mask_mode: Optional[str] = None,
        fixed_sphere_completion_radius_m: Optional[float] = None,
        rollout_trigger_config: Optional[ThrownRolloutTriggerConfig] = None,
        support_plane_abcd: Optional[np.ndarray] = None,
        support_plane_min_clearance_m: object = 0.0,
        feedback_mapper: Optional[RH56FeedbackMapper] = None,
        fingertip_model: Optional[RH56FingertipModel] = None,
        proprio_builder: Optional[Proprio67Builder] = None,
        visualization_sink: Optional[V94LiveVisualizationSink] = None,
        monotonic: Callable[[], float] = time.monotonic,
        realtime: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(contract, V94Contract):
            raise TypeError("contract must be V94Contract")
        if not isinstance(franka_pose_ring, FrankaPoseRing):
            raise TypeError("franka_pose_ring must be FrankaPoseRing")
        if not isinstance(rh56_feedback_source, RH56FeedbackHistorySource):
            raise TypeError("RH56 feedback source has no immutable history contract")
        if not isinstance(camera_source, V94CameraSnapshotSource):
            raise TypeError("camera_source does not implement its snapshot contract")
        if not callable(monotonic) or not callable(realtime):
            raise TypeError("observation clocks must be callable")
        flange_ee = np.asarray(expected_F_T_EE, dtype=np.float64)
        if flange_ee.shape != (4, 4) or not np.all(np.isfinite(flange_ee)):
            raise ValueError("expected_F_T_EE must be a finite 4x4 transform")
        self.contract = contract
        self.franka_pose_ring = franka_pose_ring
        self.expected_F_T_EE = flange_ee.copy()
        self.rh56_feedback_source = rh56_feedback_source
        self.camera_source = camera_source
        self.maximum_pose_skew_s = _positive(maximum_pose_skew_s, "maximum_pose_skew_s")
        self.maximum_franka_action_age_s = _positive(
            maximum_franka_action_age_s,
            "maximum_franka_action_age_s",
        )
        self.maximum_franka_age_s = _positive(
            maximum_franka_age_s, "maximum_franka_age_s"
        )
        if self.maximum_franka_action_age_s >= self.maximum_franka_age_s:
            raise ValueError(
                "maximum_franka_action_age_s must be below maximum_franka_age_s"
            )
        self.maximum_rh56_age_s = _positive(maximum_rh56_age_s, "maximum_rh56_age_s")
        self.maximum_future_skew_s = _finite(
            maximum_future_skew_s, "maximum_future_skew_s"
        )
        if self.maximum_future_skew_s < 0.0:
            raise ValueError("maximum_future_skew_s cannot be negative")
        self.maximum_velocity_dt_s = _positive(
            maximum_velocity_dt_s, "maximum_velocity_dt_s"
        )
        if (
            isinstance(point_feature_dim, bool)
            or not isinstance(point_feature_dim, (int, np.integer))
            or int(point_feature_dim) not in (3, 6)
        ):
            raise ValueError("point_feature_dim must be 3 (XYZ) or 6 (XYZRGB)")
        self.point_feature_dim = int(point_feature_dim)
        self.policy_rgbd_resolution = str(policy_rgbd_resolution).strip().lower()
        (
            self.requested_object_mask_mode,
            self.effective_object_mask_mode,
        ) = _normalize_object_mask_modes(
            requested_object_mask_mode,
            effective_object_mask_mode,
        )
        self.legacy_stale_palm_compatibility_enabled = bool(
            self.requested_object_mask_mode == "legacy"
            and self.effective_object_mask_mode == "legacy"
        )
        policy_image_size = resolve_policy_rgbd_resolution(
            self.policy_rgbd_resolution
        )
        self._policy_rgbd_adapter = PolicyRGBDResolutionAdapter(
            camera_K=self.contract.camera_K,
            source_image_size=(
                self.contract.camera_width,
                self.contract.camera_height,
            ),
            target_image_size=policy_image_size,
        )
        self.feedback_mapper = feedback_mapper or RH56FeedbackMapper(
            q_hand_close_rad=contract.q_hand_close_rad
        )
        self.fingertip_model = fingertip_model or RH56FingertipKinematics(contract)
        self.proprio_builder = proprio_builder or Proprio67Builder(
            q_home_rad=contract.q_home_rad,
            q_hand_close_rad=contract.q_hand_close_rad,
        )
        if visualization_sink is not None and not isinstance(
            visualization_sink, V94LiveVisualizationSink
        ):
            raise TypeError("visualization_sink does not implement try_publish")
        self.visualization_sink = visualization_sink
        fallback_config_raw = getattr(
            camera_source, "pointcloud_temporal_fallback_config", {}
        )
        fallback_config = (
            fallback_config_raw if isinstance(fallback_config_raw, dict) else {}
        )
        # This projector must persist across camera snapshots.  V94 was
        # trained/deployed with a previous-masked-cloud fallback when a newer
        # frame has fewer than 16 valid object pixels.  Reconstructing this
        # object per snapshot erases that cache and turns normal short
        # occlusions into zero-point observation holds.
        self._pointcloud_projector = MaskedRGBDProjector(
            camera_K=self._policy_rgbd_adapter.camera_K,
            T_base_camera_optical=self.contract.T_base_camera_optical,
            image_size=self._policy_rgbd_adapter.target_image_size,
            depth_range_m=self.contract.depth_range_m,
            point_feature_dim=self.point_feature_dim,
            maximum_mask_depth_deviation_m=0.055,
            fixed_sphere_completion_radius_m=fixed_sphere_completion_radius_m,
            support_plane_abcd=support_plane_abcd,
            support_plane_min_clearance_m=float(
                support_plane_min_clearance_m
            ),
            temporal_fallback=str(
                fallback_config.get("temporal_fallback", "legacy_stale_palm")
            ),
            temporal_fallback_max_stale_s=float(
                fallback_config.get("temporal_fallback_max_stale_s", 0.25)
            ),
            temporal_fallback_max_stale_steps=int(
                fallback_config.get("temporal_fallback_max_stale_steps", 5)
            ),
            temporal_fallback_max_image_speed_px_s=float(
                fallback_config.get(
                    "temporal_fallback_max_image_speed_px_s", 2400.0
                )
            ),
        )
        # The live viewer promises that its mask/color/palm transform are from
        # the same source frame as the final policy cloud.  Retain those
        # immutable camera-owner arrays alongside the projector cache so a
        # stale-palm visualization never pairs an old cloud with a newer empty
        # mask or newer palm transform.
        self._retained_pointcloud_source: Optional[_RetainedPointcloudSource] = None
        self._pointcloud_projector_lock = threading.Lock()
        # Both caches are single-entry and live under the projector lock.  The
        # policy may sample at 20/60 Hz while D435 publishes at 30 Hz; repeated
        # snapshots of the exact same immutable camera object must not redo
        # decimation, support-plane filtering, sphere fitting, or projection.
        self._adapted_policy_rgbd_cache: Optional[_AdaptedPolicyRGBDCache] = None
        self._projected_pointcloud_cache: Optional[_ProjectedPointcloudCache] = None
        self._diagnostics_lock = threading.Lock()
        self._snapshot_attempt_count = 0
        self._fresh_pointcloud_count = 0
        self._motion_compensated_pointcloud_count = 0
        self._stale_palm_recovery_count = 0
        self._stale_palm_fail_closed_hold_count = 0
        self._invalid_pointcloud_hold_count = 0
        self._last_camera_frame_id: Optional[int] = None
        self._last_camera_sensor_frame_number: Optional[int] = None
        self._last_camera_mask_valid: Optional[bool] = None
        self._last_camera_mask_area_px: Optional[int] = None
        self._last_camera_message = ""
        self._last_requested_object_mask_mode = "unknown"
        self._last_effective_object_mask_mode = "unknown"
        self._last_effective_provider_mask_publication_mode = "unknown"
        self._last_effective_provider_recovery_publication_mode = "unknown"
        self._last_provider_published_mask_source = "unknown"
        self._last_provider_online_sam2_status = "unknown"
        self._last_projector_effective_policy_mask: dict[str, object] = {
            "provenance": "unavailable_before_first_projection",
            "source_frame_id": None,
            "source_captured_realtime_s": None,
            "coordinate_space": "policy_rgbd_pixels",
            "area_px": 0,
            "bbox_xyxy": None,
        }
        self._last_pointcloud_status = "none"
        self._last_pointcloud_source_valid_points = 0
        self._last_pointcloud_frame_id: Optional[int] = None
        self._last_pointcloud_captured_realtime_s: Optional[float] = None
        self._last_recoverable_hold_reason: Optional[str] = None
        self._stale_palm_episode_started_monotonic_s: Optional[float] = None
        self._stale_palm_episode_first_camera_frame_id: Optional[int] = None
        self._stale_palm_episode_last_camera_frame_id: Optional[int] = None
        self._stale_palm_episode_camera_frame_changes = 0
        self._maximum_stale_palm_episode_duration_s = 0.0
        self._rgbd_adaptation_cache_hits = 0
        self._rgbd_adaptation_cache_misses = 0
        self._pointcloud_projection_cache_hits = 0
        self._pointcloud_projection_cache_misses = 0
        self._latency_sample_count = 0
        self._latency_last_s: dict[str, float] = {}
        self._latency_sum_s: dict[str, float] = {}
        self._latency_count_by_stage: dict[str, int] = {}
        self._latency_max_s: dict[str, float] = {}
        self._last_provider_timings_ms: dict[str, float] = {}
        self._last_latency_camera_frame_id: Optional[int] = None
        self._last_latency_pointcloud_frame_id: Optional[int] = None
        self._last_rgbd_adaptation_cache_hit = False
        self._last_pointcloud_projection_cache_hit = False
        self._monotonic = monotonic
        self._realtime = realtime

    def _record_camera_snapshot(self, camera: _CameraSample) -> None:
        with self._diagnostics_lock:
            self._snapshot_attempt_count += 1
            self._last_camera_frame_id = int(camera.frame_id)
            self._last_camera_sensor_frame_number = int(camera.sensor_frame_number)
            self._last_camera_mask_valid = bool(camera.mask_valid)
            self._last_camera_mask_area_px = int(np.count_nonzero(camera.mask))
            self._last_camera_message = " ".join(str(camera.message).split())[:1000]
            self._last_requested_object_mask_mode = str(
                camera.requested_object_mask_mode
            )
            self._last_effective_object_mask_mode = str(
                camera.effective_object_mask_mode
            )
            self._last_effective_provider_mask_publication_mode = str(
                camera.effective_mask_publication_mode
            )
            self._last_effective_provider_recovery_publication_mode = str(
                camera.effective_recovery_publication_mode
            )
            self._last_provider_published_mask_source = str(
                camera.provider_published_mask_source
            )
            self._last_provider_online_sam2_status = str(
                camera.provider_online_sam2_status
            )

    def _record_pointcloud_result(
        self,
        *,
        camera: _CameraSample,
        point_frame: Any,
        mask_provenance: ProjectorEffectiveMaskProvenance,
        observed_monotonic_s: float,
    ) -> None:
        status = str(point_frame.status)
        camera_frame_id = int(camera.frame_id)
        observed = float(observed_monotonic_s)
        with self._diagnostics_lock:
            self._last_pointcloud_status = status
            self._last_pointcloud_source_valid_points = int(
                point_frame.source_valid_points
            )
            self._last_pointcloud_frame_id = int(point_frame.frame_id)
            self._last_pointcloud_captured_realtime_s = float(point_frame.captured_at_s)
            self._last_projector_effective_policy_mask = {
                "provenance": str(mask_provenance.kind),
                "source_frame_id": mask_provenance.source_frame_id,
                "source_captured_realtime_s": (
                    mask_provenance.source_captured_at_s
                ),
                "coordinate_space": "policy_rgbd_pixels",
                "area_px": int(mask_provenance.area_px),
                "bbox_xyxy": (
                    None
                    if mask_provenance.bbox_xyxy is None
                    else [int(value) for value in mask_provenance.bbox_xyxy]
                ),
            }
            if status in {"fresh", "motion_compensated"}:
                if status == "fresh":
                    self._fresh_pointcloud_count += 1
                else:
                    self._motion_compensated_pointcloud_count += 1
                started = self._stale_palm_episode_started_monotonic_s
                if started is not None:
                    self._maximum_stale_palm_episode_duration_s = max(
                        self._maximum_stale_palm_episode_duration_s,
                        max(0.0, observed - started),
                    )
                self._stale_palm_episode_started_monotonic_s = None
                self._stale_palm_episode_first_camera_frame_id = None
                self._stale_palm_episode_last_camera_frame_id = None
                self._stale_palm_episode_camera_frame_changes = 0
                self._last_recoverable_hold_reason = None
            elif status == "stale_palm":
                self._stale_palm_recovery_count += 1
                if self._stale_palm_episode_started_monotonic_s is None:
                    self._stale_palm_episode_started_monotonic_s = observed
                    self._stale_palm_episode_first_camera_frame_id = camera_frame_id
                    self._stale_palm_episode_last_camera_frame_id = camera_frame_id
                    self._stale_palm_episode_camera_frame_changes = 0
                elif self._stale_palm_episode_last_camera_frame_id != camera_frame_id:
                    self._stale_palm_episode_camera_frame_changes += 1
                    self._stale_palm_episode_last_camera_frame_id = camera_frame_id
                started = self._stale_palm_episode_started_monotonic_s
                assert started is not None
                self._maximum_stale_palm_episode_duration_s = max(
                    self._maximum_stale_palm_episode_duration_s,
                    max(0.0, observed - started),
                )
            else:
                self._invalid_pointcloud_hold_count += 1

    def _record_recoverable_hold(self, reason: str) -> None:
        with self._diagnostics_lock:
            self._last_recoverable_hold_reason = " ".join(str(reason).split())[:1500]

    @staticmethod
    def _realtime_delta(end_s: object, start_s: object) -> Optional[float]:
        """Best-effort stage delta for diagnostics, never for admission."""

        try:
            end = float(end_s)
            start = float(start_s)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(end) or not math.isfinite(start):
            return None
        return end - start

    def _record_perception_latency(
        self,
        *,
        camera: _CameraSample,
        point_frame: PolicyPointFrame,
        snapshot_started_monotonic_s: float,
        snapshot_started_realtime_s: float,
        camera_acquired_monotonic_s: float,
        camera_acquired_realtime_s: float,
        projection_finished_monotonic_s: float,
        rgbd_adaptation_s: float,
        pointcloud_projection_s: float,
        rgbd_cache_hit: bool,
        projection_cache_hit: bool,
    ) -> None:
        """Accumulate constant-space, read-only perception timing evidence."""

        projection_finished_realtime_s = float(snapshot_started_realtime_s) + (
            float(projection_finished_monotonic_s)
            - float(snapshot_started_monotonic_s)
        )
        stage_values: dict[str, Optional[float]] = {
            "capture_to_retrieval": self._realtime_delta(
                camera.retrieved_at_s, camera.timestamp_s
            ),
            "retrieval_to_provider_publication": self._realtime_delta(
                camera.published_at_s, camera.retrieved_at_s
            ),
            "provider_publication_to_snapshot_start": self._realtime_delta(
                snapshot_started_realtime_s, camera.published_at_s
            ),
            "provider_publication_to_camera_snapshot": self._realtime_delta(
                camera_acquired_realtime_s, camera.published_at_s
            ),
            "capture_to_snapshot_start": self._realtime_delta(
                snapshot_started_realtime_s, camera.timestamp_s
            ),
            "capture_to_camera_snapshot": self._realtime_delta(
                camera_acquired_realtime_s, camera.timestamp_s
            ),
            "provider_publication_to_projection_complete": self._realtime_delta(
                projection_finished_realtime_s, camera.published_at_s
            ),
            "capture_to_projection_complete": self._realtime_delta(
                projection_finished_realtime_s, camera.timestamp_s
            ),
            "snapshot_start_to_camera_snapshot": (
                float(camera_acquired_monotonic_s)
                - float(snapshot_started_monotonic_s)
            ),
            "rgbd_adaptation_compute": float(rgbd_adaptation_s),
            "pointcloud_projection_compute": float(pointcloud_projection_s),
            "snapshot_start_to_projection_complete": (
                float(projection_finished_monotonic_s)
                - float(snapshot_started_monotonic_s)
            ),
        }
        finite_stages = {
            name: float(value)
            for name, value in stage_values.items()
            if value is not None and math.isfinite(float(value))
        }
        timings = np.asarray(camera.provider_timings_ms, dtype=np.float64).reshape(-1)
        provider_timings = {
            name: float(timings[index])
            for index, name in enumerate(_provider_timing_names(camera))
            if index < timings.size and math.isfinite(float(timings[index]))
        }
        with self._diagnostics_lock:
            self._latency_sample_count += 1
            if rgbd_cache_hit:
                self._rgbd_adaptation_cache_hits += 1
            else:
                self._rgbd_adaptation_cache_misses += 1
            if projection_cache_hit:
                self._pointcloud_projection_cache_hits += 1
            else:
                self._pointcloud_projection_cache_misses += 1
            self._latency_last_s = finite_stages
            for name, value in finite_stages.items():
                self._latency_sum_s[name] = self._latency_sum_s.get(name, 0.0) + value
                self._latency_count_by_stage[name] = (
                    self._latency_count_by_stage.get(name, 0) + 1
                )
                self._latency_max_s[name] = max(
                    self._latency_max_s.get(name, -math.inf), value
                )
            self._last_provider_timings_ms = provider_timings
            self._last_latency_camera_frame_id = int(camera.frame_id)
            self._last_latency_pointcloud_frame_id = int(point_frame.frame_id)
            self._last_rgbd_adaptation_cache_hit = bool(rgbd_cache_hit)
            self._last_pointcloud_projection_cache_hit = bool(
                projection_cache_hit
            )

    @property
    def diagnostics_snapshot(self) -> dict[str, object]:
        """Constant-space perception provenance for PASS and FAIL audits."""

        now_monotonic = _finite(
            self._monotonic(), "observation diagnostics monotonic clock"
        )
        now_realtime = _finite(
            self._realtime(), "observation diagnostics realtime clock"
        )
        with self._diagnostics_lock:
            started = self._stale_palm_episode_started_monotonic_s
            stale_duration = (
                None if started is None else max(0.0, now_monotonic - started)
            )
            captured = self._last_pointcloud_captured_realtime_s
            pointcloud_age = (
                None if captured is None else max(0.0, now_realtime - captured)
            )
            result: dict[str, object] = {
                "snapshot_attempt_count": self._snapshot_attempt_count,
                "fresh_pointcloud_count": self._fresh_pointcloud_count,
                "motion_compensated_pointcloud_count": (
                    self._motion_compensated_pointcloud_count
                ),
                "stale_palm_recovery_count": (self._stale_palm_recovery_count),
                "stale_palm_fail_closed_hold_count": (
                    self._stale_palm_fail_closed_hold_count
                ),
                "invalid_pointcloud_hold_count": (self._invalid_pointcloud_hold_count),
                "configured_requested_object_mask_mode": (
                    self.requested_object_mask_mode
                ),
                "configured_effective_object_mask_mode": (
                    self.effective_object_mask_mode
                ),
                "stale_palm_policy": (
                    "legacy_explicit_compatibility"
                    if self.legacy_stale_palm_compatibility_enabled
                    else "recoverable_fail_closed_no_policy_stage"
                ),
                "last_camera_frame_id": self._last_camera_frame_id,
                "last_camera_sensor_frame_number": (
                    self._last_camera_sensor_frame_number
                ),
                "last_camera_mask_valid": self._last_camera_mask_valid,
                "last_camera_mask_area_px": self._last_camera_mask_area_px,
                "last_camera_message": self._last_camera_message,
                "requested_object_mask_mode": (
                    self._last_requested_object_mask_mode
                ),
                "effective_object_mask_mode": (
                    self._last_effective_object_mask_mode
                ),
                "effective_provider_mask_publication_mode": (
                    self._last_effective_provider_mask_publication_mode
                ),
                "effective_provider_recovery_publication_mode": (
                    self._last_effective_provider_recovery_publication_mode
                ),
                "provider_published_mask_source": (
                    self._last_provider_published_mask_source
                ),
                "provider_published_mask_message": self._last_camera_message,
                "provider_online_sam2_status": (
                    self._last_provider_online_sam2_status
                ),
                "projector_effective_policy_mask": dict(
                    self._last_projector_effective_policy_mask
                ),
                "last_pointcloud_status": self._last_pointcloud_status,
                "last_pointcloud_source_valid_points": (
                    self._last_pointcloud_source_valid_points
                ),
                "last_pointcloud_frame_id": self._last_pointcloud_frame_id,
                "last_pointcloud_age_s": pointcloud_age,
                "last_recoverable_hold_reason": (self._last_recoverable_hold_reason),
                "perception_latency": {
                    "sample_count": self._latency_sample_count,
                    "last_camera_frame_id": self._last_latency_camera_frame_id,
                    "last_pointcloud_frame_id": (
                        self._last_latency_pointcloud_frame_id
                    ),
                    "last_stage_s": dict(self._latency_last_s),
                    "mean_stage_s": {
                        name: total / max(
                            1, self._latency_count_by_stage.get(name, 0)
                        )
                        for name, total in self._latency_sum_s.items()
                    },
                    "maximum_stage_s": dict(self._latency_max_s),
                    "last_provider_stage_ms": dict(
                        self._last_provider_timings_ms
                    ),
                    "rgbd_adaptation_cache": {
                        "hits": self._rgbd_adaptation_cache_hits,
                        "misses": self._rgbd_adaptation_cache_misses,
                        "last_hit": self._last_rgbd_adaptation_cache_hit,
                    },
                    "pointcloud_projection_cache": {
                        "hits": self._pointcloud_projection_cache_hits,
                        "misses": self._pointcloud_projection_cache_misses,
                        "last_hit": self._last_pointcloud_projection_cache_hit,
                    },
                },
                "stale_palm_episode": {
                    "active": started is not None,
                    "duration_s": stale_duration,
                    "first_camera_frame_id": (
                        self._stale_palm_episode_first_camera_frame_id
                    ),
                    "last_camera_frame_id": (
                        self._stale_palm_episode_last_camera_frame_id
                    ),
                    "camera_frame_changes": (
                        self._stale_palm_episode_camera_frame_changes
                    ),
                    "maximum_duration_s": (self._maximum_stale_palm_episode_duration_s),
                },
            }
        camera_diagnostics = getattr(self.camera_source, "diagnostics_snapshot", None)
        if camera_diagnostics is not None:
            try:
                result["camera_owner"] = (
                    camera_diagnostics
                    if isinstance(camera_diagnostics, dict)
                    else dict(camera_diagnostics)
                )
            except BaseException as exc:
                result["camera_owner_diagnostics_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
        return result

    @property
    def diagnostics_summary(self) -> str:
        diagnostics = self.diagnostics_snapshot
        stale = diagnostics["stale_palm_episode"]
        assert isinstance(stale, dict)
        latency = diagnostics["perception_latency"]
        assert isinstance(latency, dict)
        rgbd_cache = latency["rgbd_adaptation_cache"]
        projection_cache = latency["pointcloud_projection_cache"]
        assert isinstance(rgbd_cache, dict)
        assert isinstance(projection_cache, dict)
        camera_owner = diagnostics.get("camera_owner")
        worker_timeouts = None
        if isinstance(camera_owner, dict):
            worker = camera_owner.get("worker")
            if isinstance(worker, dict):
                worker_timeouts = worker.get("consecutive_transient_error_count")
        return (
            "perception="
            f"pointcloud_status={diagnostics['last_pointcloud_status']},"
            f"pointcloud_frame={diagnostics['last_pointcloud_frame_id']},"
            f"pointcloud_age_s={diagnostics['last_pointcloud_age_s']},"
            f"latest_camera_frame={diagnostics['last_camera_frame_id']},"
            f"latest_sensor_frame="
            f"{diagnostics['last_camera_sensor_frame_number']},"
            f"mask_valid={diagnostics['last_camera_mask_valid']},"
            f"mask_px={diagnostics['last_camera_mask_area_px']},"
            f"stale_camera_frame_changes={stale['camera_frame_changes']},"
            f"retryable_capture_timeouts={worker_timeouts},"
            "rgbd_cache_hits="
            f"{rgbd_cache['hits']},"
            "projection_cache_hits="
            f"{projection_cache['hits']}"
        )

    def _clock_pair(self) -> Tuple[float, float]:
        before = _finite(self._monotonic(), "monotonic clock")
        realtime = _finite(self._realtime(), "realtime clock")
        after = _finite(self._monotonic(), "monotonic clock")
        if after < before:
            raise V94LiveObservationError("host monotonic clock regressed")
        return (0.5 * (before + after), realtime)

    def inputs_ready_for_camera(self, camera: _CameraSample) -> bool:
        """Read-only startup predicate; it opens or writes no device."""

        samples = self.franka_pose_ring.snapshot()
        try:
            feedback = self.rh56_feedback_source.require_fresh_feedback_history(
                minimum_samples=2,
                maximum_age_s=self.maximum_rh56_age_s,
            )
        except BaseException:
            return False
        if len(samples) < 2 or len(feedback.samples) < 2:
            return False
        return any(
            abs(sample.realtime_s - float(camera.timestamp_s))
            <= self.maximum_pose_skew_s
            for sample in samples
        )

    def snapshot(
        self,
        *,
        sequence: int,
        now_monotonic_s: float,
        hard_deadline_monotonic_s: float,
    ) -> V94PolicyObservation:
        policy_sequence = int(sequence)
        requested_now = _finite(now_monotonic_s, "now_monotonic_s")
        hard_deadline = _finite(hard_deadline_monotonic_s, "hard_deadline_monotonic_s")
        actual_mono, actual_realtime = self._clock_pair()
        if requested_now > actual_mono + self.maximum_future_skew_s:
            raise V94LiveObservationError("snapshot monotonic request is in the future")
        if actual_mono >= hard_deadline:
            raise V94LiveObservationError(
                "snapshot deadline expired before acquisition"
            )

        camera = self.camera_source.snapshot()
        camera_acquired_mono, camera_acquired_realtime = self._clock_pair()
        _validate_camera_contract(camera, self.contract)
        self._record_camera_snapshot(camera)
        pose_samples = self.franka_pose_ring.snapshot()
        if len(pose_samples) < 2:
            raise V94LiveObservationError("Franka pose ring has fewer than two samples")
        hand_history = self.rh56_feedback_source.require_fresh_feedback_history(
            minimum_samples=2,
            maximum_age_s=self.maximum_rh56_age_s,
        )
        timestamped_hand_samples = tuple(hand_history.samples)
        if not timestamped_hand_samples:
            raise V94LiveObservationError("RH56 owner has no safety feedback")
        try:
            hand_samples = tuple(item.feedback for item in timestamped_hand_samples)
        except AttributeError as exc:
            raise V94LiveObservationError(
                "RH56 owner cache has no timestamped feedback records"
            ) from exc
        if any(not isinstance(item, RH56SafetyFeedback) for item in hand_samples):
            raise V94LiveObservationError("RH56 owner cache contains invalid feedback")

        latest_pose = pose_samples[-1]
        latest_hand = hand_samples[-1]
        franka_age = actual_mono - latest_pose.monotonic_s
        hand_age = actual_mono - latest_hand.captured_monotonic_s
        for name, age, maximum in (
            ("Franka", franka_age, self.maximum_franka_age_s),
            ("RH56", hand_age, self.maximum_rh56_age_s),
        ):
            if age < -self.maximum_future_skew_s:
                raise V94LiveObservationError(
                    f"{name} sample timestamp is in the future"
                )
            if age > maximum and name != "Franka":
                raise V94LiveObservationError(
                    f"{name} sample is stale: age={age:.6f}s, limit={maximum:.6f}s"
                )
        if franka_age > self.maximum_franka_age_s:
            # The native owner continues holding the last admitted target.
            # Refuse to stage from this stale sample and retry the same policy
            # sequence; a sustained STATE-delivery outage is still bounded by
            # the runtime's consecutive no-stage timeout.
            reason = (
                "franka_state_hard_age_retry:"
                f"age={franka_age:.6f}s:"
                f"limit={self.maximum_franka_age_s:.6f}s"
            )
            self._record_recoverable_hold(reason)
            raise V94RecoverableObservationHold(
                camera_frame_id=int(camera.frame_id),
                reason=reason,
            )
        if franka_age > self.maximum_franka_action_age_s:
            reason = (
                "franka_state_action_age_limit:"
                f"age={franka_age:.6f}s:"
                f"limit={self.maximum_franka_action_age_s:.6f}s"
            )
            self._record_recoverable_hold(reason)
            raise V94RecoverableObservationHold(
                camera_frame_id=int(camera.frame_id),
                reason=reason,
            )

        capture_pose_sample, _capture_skew = _nearest_pose(
            pose_samples,
            float(camera.timestamp_s),
            self.maximum_pose_skew_s,
        )
        capture_palm = T_base_policy_palm_from_franka(
            T_base_ee=capture_pose_sample.T_base_eef,
            F_T_EE=self.expected_F_T_EE,
            T_flange_policy_palm=self.contract.T_flange_policy_palm,
        )
        current_palm = T_base_policy_palm_from_franka(
            T_base_ee=latest_pose.T_base_eef,
            F_T_EE=self.expected_F_T_EE,
            T_flange_policy_palm=self.contract.T_flange_policy_palm,
        )
        palm_linear, palm_angular = _palm_twist(
            pose_samples[-2],
            latest_pose,
            F_T_EE=self.expected_F_T_EE,
            T_flange_policy_palm=self.contract.T_flange_policy_palm,
            maximum_dt_s=self.maximum_velocity_dt_s,
        )

        mapped_hand = self.feedback_mapper.map(
            np.asarray(latest_hand.angles, dtype=np.int64)
        )
        if len(hand_samples) < 2:
            hand_dq = np.zeros(6, dtype=np.float32)
        else:
            previous_hand = hand_samples[-2]
            hand_dt = (
                latest_hand.captured_monotonic_s - previous_hand.captured_monotonic_s
            )
            if (
                not math.isfinite(hand_dt)
                or hand_dt <= 0.0
                or hand_dt > self.maximum_velocity_dt_s
            ):
                raise V94LiveObservationError(
                    f"unsafe RH56 finite-difference interval {hand_dt!r}s"
                )
            previous_mapped = self.feedback_mapper.map(
                np.asarray(previous_hand.angles, dtype=np.int64)
            )
            hand_dq = (
                (mapped_hand.q_policy_order_rad - previous_mapped.q_policy_order_rad)
                / hand_dt
            ).astype(np.float32)

        fingertips = self.fingertip_model.positions_base(
            angle_act_register_order=np.asarray(latest_hand.angles, dtype=np.int64),
            T_base_palm=current_palm,
        )
        proprio67 = self.proprio_builder.build(
            franka_q_rad=latest_pose.q_rad,
            franka_dq_rad_s=latest_pose.dq_rad_s,
            rh56_virtual_q_policy_order_rad=mapped_hand.q_policy_order_rad,
            rh56_virtual_dq_policy_order_rad_s=hand_dq,
            T_base_palm=current_palm,
            palm_linear_velocity_base_m_s=palm_linear,
            palm_angular_velocity_base_rad_s=palm_angular,
            fingertip_positions_base_m=fingertips,
            previous_executed_action13=INITIAL_PREVIOUS_ACTION13,
        )

        rgbd_adaptation_s = 0.0
        pointcloud_projection_s = 0.0
        rgbd_cache_hit = False
        projection_cache_hit = False
        with self._pointcloud_projector_lock:
            adapted_cache = self._adapted_policy_rgbd_cache
            if adapted_cache is not None and adapted_cache.camera is camera:
                policy_rgbd = adapted_cache.policy_rgbd
                rgbd_cache_hit = True
            else:
                adaptation_started = _finite(
                    self._monotonic(), "RGB-D adaptation monotonic clock"
                )
                projection_source_mask = camera.mask
                if (
                    not camera.mask_valid
                    and bool(getattr(camera, "policy_semantic_valid", False))
                    and getattr(camera, "policy_semantic_mask", None) is not None
                ):
                    projection_source_mask = np.asarray(
                        camera.policy_semantic_mask, dtype=np.bool_
                    )
                adapted = self._policy_rgbd_adapter.adapt(
                    color_bgr=camera.color_bgr,
                    depth_m=camera.depth_m if camera.depth_raw is None else None,
                    depth_raw=camera.depth_raw,
                    object_mask=(
                        projection_source_mask
                        if camera.mask_valid
                        or bool(getattr(camera, "policy_semantic_valid", False))
                        else np.zeros_like(camera.mask)
                    ),
                )
                policy_rgbd = _private_policy_rgbd(adapted, camera=camera)
                adaptation_finished = _finite(
                    self._monotonic(), "RGB-D adaptation monotonic clock"
                )
                rgbd_adaptation_s = max(
                    0.0, adaptation_finished - adaptation_started
                )
                self._adapted_policy_rgbd_cache = _AdaptedPolicyRGBDCache(
                    camera=camera,
                    policy_rgbd=policy_rgbd,
                )

            projected_cache = self._projected_pointcloud_cache
            if (
                projected_cache is not None
                and projected_cache.camera is camera
                and int(projected_cache.capture_pose_cycle)
                == int(capture_pose_sample.cycle)
                and np.array_equal(
                    projected_cache.T_base_palm_at_capture,
                    capture_palm,
                )
            ):
                point_frame = projected_cache.point_frame
                pointcloud_visual_source = projected_cache.visual_source
                projector_mask_provenance = projected_cache.mask_provenance
                projection_cache_hit = True
            else:
                projection_started = _finite(
                    self._monotonic(), "pointcloud projection monotonic clock"
                )
                projected = self._pointcloud_projector.project(
                    color_bgr=policy_rgbd.color_bgr,
                    depth_m=policy_rgbd.depth_m,
                    depth_raw=policy_rgbd.depth_raw,
                    depth_scale_m_per_unit=(
                        camera.depth_scale_m_per_unit
                        if camera.depth_raw is not None
                        else None
                    ),
                    object_mask=policy_rgbd.object_mask,
                    T_base_palm_at_capture=capture_palm,
                    captured_at_s=float(camera.timestamp_s),
                    frame_id=int(camera.frame_id),
                )
                pointcloud_projection_s = max(
                    0.0,
                    _finite(
                        self._monotonic(),
                        "pointcloud projection monotonic clock",
                    )
                    - projection_started,
                )
                point_frame = _private_point_frame(projected)
                projector_mask_provenance = (
                    self._pointcloud_projector.last_effective_object_mask_provenance
                )
                if point_frame.status in {"fresh", "motion_compensated"}:
                    effective_mask = (
                        self._pointcloud_projector.last_effective_object_mask
                    )
                    if effective_mask is None:
                        raise V94LiveObservationError(
                            "accepted policy cloud has no effective object mask"
                        )
                    source_mask = _private_readonly_array(
                        self._policy_rgbd_adapter.mask_to_source_resolution(
                            effective_mask
                        ),
                        dtype=np.bool_,
                    )
                    visual_camera = replace(camera, mask=source_mask)
                    visual_capture_palm = _private_readonly_array(
                        capture_palm,
                        dtype=np.float64,
                    )
                    pointcloud_visual_source = _RetainedPointcloudSource(
                        camera=visual_camera,
                        T_base_palm_at_capture=visual_capture_palm,
                        source_valid_points=int(point_frame.source_valid_points),
                    )
                    if point_frame.status == "fresh":
                        self._retained_pointcloud_source = pointcloud_visual_source
                elif point_frame.status == "stale_palm":
                    pointcloud_visual_source = self._retained_pointcloud_source
                    if (
                        pointcloud_visual_source is None
                        or int(pointcloud_visual_source.camera.frame_id)
                        != int(point_frame.frame_id)
                        or float(pointcloud_visual_source.camera.timestamp_s)
                        != float(point_frame.captured_at_s)
                    ):
                        raise V94LiveObservationError(
                            "stale-palm projector/source cache is inconsistent"
                        )
                else:
                    pointcloud_visual_source = None
                cached_capture_palm = _private_readonly_array(
                    capture_palm,
                    dtype=np.float64,
                )
                self._projected_pointcloud_cache = _ProjectedPointcloudCache(
                    camera=camera,
                    capture_pose_cycle=int(capture_pose_sample.cycle),
                    T_base_palm_at_capture=cached_capture_palm,
                    point_frame=point_frame,
                    visual_source=pointcloud_visual_source,
                    mask_provenance=projector_mask_provenance,
                )
        projection_finished_mono = _finite(
            self._monotonic(), "projection-finished monotonic clock"
        )
        self._record_perception_latency(
            camera=camera,
            point_frame=point_frame,
            snapshot_started_monotonic_s=actual_mono,
            snapshot_started_realtime_s=actual_realtime,
            camera_acquired_monotonic_s=camera_acquired_mono,
            camera_acquired_realtime_s=camera_acquired_realtime,
            projection_finished_monotonic_s=projection_finished_mono,
            rgbd_adaptation_s=rgbd_adaptation_s,
            pointcloud_projection_s=pointcloud_projection_s,
            rgbd_cache_hit=rgbd_cache_hit,
            projection_cache_hit=projection_cache_hit,
        )
        self._record_pointcloud_result(
            camera=camera,
            point_frame=point_frame,
            mask_provenance=projector_mask_provenance,
            observed_monotonic_s=actual_mono,
        )
        fresh_actuator_snapshot = V94FreshActuatorSnapshot(
            measured_franka_q_rad=latest_pose.q_rad,
            franka_state_captured_monotonic_s=latest_pose.monotonic_s,
            observation_realtime_s=actual_realtime,
            hold_arm_target=bool(
                int(latest_pose.status_flags) & _FRANKA_CONTACT_STATUS_MASK
            ),
            shaper_q_d_rad=latest_pose.shaper_q_d_rad,
        )
        # Guarded modes must never stage policy inference from a retained
        # capture-time palm cloud.  ``stale_palm`` remains representable in the
        # projector for provenance and explicit legacy A/B only.  Raising here
        # happens before V94PolicyObservation is returned, so the transactional
        # tick source cannot advance policy history or propose mapper state.
        if (
            point_frame.status == "stale_palm"
            and not self.legacy_stale_palm_compatibility_enabled
        ):
            reason = (
                f"{GUARDED_STALE_PALM_HOLD_REASON}:"
                f"requested_mode={self.requested_object_mask_mode}:"
                f"effective_mode={self.effective_object_mask_mode}:"
                f"camera_frame={int(camera.frame_id)}:"
                f"retained_pointcloud_frame={int(point_frame.frame_id)}:"
                f"source_points={int(point_frame.source_valid_points)}"
            )
            with self._diagnostics_lock:
                self._stale_palm_fail_closed_hold_count += 1
            self._record_recoverable_hold(reason)
            raise V94RecoverableObservationHold(
                camera_frame_id=int(camera.frame_id),
                reason=reason,
                fresh_actuator_snapshot=fresh_actuator_snapshot,
            )

        # Explicit legacy compatibility retains the previous cloud's original
        # timestamp, frame ID, and native palm coordinates.  Any invalid frame
        # before the first valid cloud remains a recoverable no-stage hold.
        if (
            point_frame.status
            not in {"fresh", "motion_compensated", "stale_palm"}
            or float(np.sum(point_frame.valid)) < 1.0
        ):
            reason = (
                f"{OBJECT_POINTCLOUD_INVALID_HOLD_REASON}:"
                f"status={point_frame.status}:"
                f"source_points={point_frame.source_valid_points}:"
                f"mask_valid={bool(camera.mask_valid)}:"
                f"mask_px={int(np.count_nonzero(camera.mask))}"
            )
            self._record_recoverable_hold(reason)
            raise V94RecoverableObservationHold(
                camera_frame_id=int(camera.frame_id),
                reason=reason,
                fresh_actuator_snapshot=fresh_actuator_snapshot,
            )
        finished_mono, finished_realtime = self._clock_pair()
        if finished_mono >= hard_deadline:
            raise V94LiveObservationError("snapshot deadline expired during projection")
        # Fresh point clouds retain the capture-age check before/after
        # inference. Explicit legacy ``stale_palm`` keeps its older timestamp
        # solely as provenance; current-camera publication/reuse guards
        # independently detect a frozen transport.
        observation = V94PolicyObservation(
            pointcloud_xyzrgb_palm=point_frame.xyzrgb_palm,
            pointcloud_valid=point_frame.valid,
            proprio_prefix54=proprio67[:54],
            measured_franka_q_rad=latest_pose.q_rad,
            franka_state_captured_monotonic_s=latest_pose.monotonic_s,
            pointcloud_captured_realtime_s=point_frame.captured_at_s,
            observation_realtime_s=finished_realtime,
            # Camera liveness follows the current formal RGB-D publication.
            # The retained cloud's older frame remains explicit provenance
            # below and in its original native-palm coordinates.
            camera_frame_id=int(camera.frame_id),
            hold_arm_target=bool(
                int(latest_pose.status_flags) & _FRANKA_CONTACT_STATUS_MASK
            ),
            controller_state29=latest_pose.controller_state29,
            shaper_q_d_rad=latest_pose.shaper_q_d_rad,
            pointcloud_status=point_frame.status,
            source_valid_points=point_frame.source_valid_points,
            pointcloud_source_frame_id=int(point_frame.frame_id),
        )
        # Visualization is an explicitly lossy side channel.  The sink's
        # production implementation only swaps one reference here; all image
        # copies, IPC, mask GUI, and Open3D work happen outside the policy path.
        # A viewer failure after startup must never change this observation or
        # the action/stop ledger.
        if self.visualization_sink is not None:
            if pointcloud_visual_source is None:
                raise V94LiveObservationError(
                    "accepted point cloud has no matching visualization source"
                )
            try:
                self.visualization_sink.try_publish(
                    V94LiveVisualizationSample(
                        sequence=policy_sequence,
                        frame_id=int(point_frame.frame_id),
                        captured_realtime_s=float(point_frame.captured_at_s),
                        color_bgr=pointcloud_visual_source.camera.color_bgr,
                        object_mask=pointcloud_visual_source.camera.mask,
                        pointcloud_xyzrgb_palm=observation.pointcloud_xyzrgb_palm,
                        pointcloud_valid=observation.pointcloud_valid,
                        T_base_palm_at_capture=(
                            pointcloud_visual_source.T_base_palm_at_capture
                        ),
                        source_valid_points=int(
                            pointcloud_visual_source.source_valid_points
                        ),
                        requested_object_mask_mode=str(
                            pointcloud_visual_source.camera.requested_object_mask_mode
                        ),
                        effective_object_mask_mode=str(
                            pointcloud_visual_source.camera.effective_object_mask_mode
                        ),
                        effective_provider_mask_publication_mode=str(
                            pointcloud_visual_source.camera.effective_mask_publication_mode
                        ),
                        effective_provider_recovery_publication_mode=str(
                            pointcloud_visual_source.camera.effective_recovery_publication_mode
                        ),
                        provider_published_mask_valid=bool(
                            pointcloud_visual_source.camera.mask_valid
                        ),
                        provider_published_mask_area_px=int(
                            pointcloud_visual_source.camera.object_mask_area_px
                        ),
                        provider_published_mask_bbox_xyxy=tuple(
                            int(value)
                            for value in np.asarray(
                                pointcloud_visual_source.camera.object_mask_bbox_xyxy,
                                dtype=np.int32,
                            ).reshape(4)
                        ),
                        provider_published_mask_source=str(
                            pointcloud_visual_source.camera.provider_published_mask_source
                        ),
                        provider_published_mask_message=str(
                            pointcloud_visual_source.camera.provider_published_mask_message
                        ),
                        provider_online_sam2_status=str(
                            pointcloud_visual_source.camera.provider_online_sam2_status
                        ),
                        projector_effective_policy_mask_provenance=str(
                            projector_mask_provenance.kind
                        ),
                        projector_effective_policy_mask_source_frame_id=(
                            projector_mask_provenance.source_frame_id
                        ),
                        projector_effective_policy_mask_source_captured_realtime_s=(
                            projector_mask_provenance.source_captured_at_s
                        ),
                        projector_effective_policy_mask_area_px=int(
                            projector_mask_provenance.area_px
                        ),
                        projector_effective_policy_mask_bbox_xyxy=(
                            projector_mask_provenance.bbox_xyxy
                        ),
                    )
                )
            except BaseException:
                pass
        return observation


class ProductionV94PolicyTickSourceFactory:
    """Camera-first, state-bound factory for the production runtime path.

    Construction is hardware-inert.  ``open_and_warm_camera`` is intentionally
    separate so the runtime can complete expensive tracker/FK initialization
    before starting either actuator owner.  Franka then establishes its
    measured-hold pose ring, a newer D435 frame is aligned to that ring, and
    only then may the RH56 owner start its strict heartbeat and provide the two
    feedback samples used to initialize the action mapper.
    """

    construction_is_inert = True

    def __init__(
        self,
        *,
        contract: V94Contract,
        policy: TransactionalStateV94Policy,
        camera_owner: D435ObjectCameraOwner,
        camera_owner_preopened: bool = False,
        maximum_pose_skew_s: object,
        maximum_franka_action_age_s: object,
        maximum_franka_age_s: object,
        maximum_rh56_age_s: object,
        maximum_future_skew_s: object,
        maximum_velocity_dt_s: object,
        camera_poll_interval_s: object,
        maximum_object_pointcloud_age_s: object = 0.20,
        maximum_actions_per_camera_frame: object = 2,
        policy_rgbd_resolution: str = "848x480",
        requested_object_mask_mode: str = "guarded",
        effective_object_mask_mode: Optional[str] = None,
        franka_arrival_gate_enabled: bool = False,
        franka_arrival_gate_tolerance_rad: object = 0.015,
        franka_arrival_gate_timeout_s: object = 0.350,
        feedback_mapper: Optional[RH56FeedbackMapper] = None,
        fingertip_model: Optional[RH56FingertipModel] = None,
        proprio_builder: Optional[Proprio67Builder] = None,
        live_visualizer: Optional[Any] = None,
        policy_io_recorder: Optional[Any] = None,
        fixed_sphere_completion_radius_m: Optional[float] = None,
        rollout_trigger_config: Optional[ThrownRolloutTriggerConfig] = None,
        rh56_hardware_rate_hz: object = 20.0,
        rh56_maximum_register_delta_per_update: Sequence[int] = (
            137,
            143,
            158,
            158,
            251,
            120,
        ),
        rh56_minimum_angle_set_register_order: Sequence[int] = (
            0,
            0,
            0,
            0,
            0,
            0,
        ),
        rh56_maximum_angle_set_register_order: Sequence[int] = (1000,) * 6,
        rh56_feedback_to_command_offset_units: Sequence[int] = (
            0,
            0,
            0,
            0,
            0,
            15,
        ),
        rh56_feedback_to_command_valid_min: Sequence[int] = (
            0,
            0,
            0,
            0,
            0,
            0,
        ),
        rh56_feedback_to_command_valid_max: Sequence[int] = (1000,) * 6,
        monotonic: Callable[[], float] = time.monotonic,
        realtime: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(contract, V94Contract):
            raise TypeError("contract must be V94Contract")
        if not isinstance(policy, TransactionalStateV94Policy):
            raise TypeError("policy does not implement the transactional contract")
        if not isinstance(camera_owner, D435ObjectCameraOwner):
            raise TypeError("camera_owner must be D435ObjectCameraOwner")
        if not callable(monotonic) or not callable(realtime) or not callable(sleep):
            raise TypeError("factory clocks and sleep must be callable")
        self.contract = contract
        self.policy = policy
        point_feature_dim = getattr(policy, "point_feature_dim", 6)
        if (
            isinstance(point_feature_dim, bool)
            or not isinstance(point_feature_dim, (int, np.integer))
            or int(point_feature_dim) not in (3, 6)
        ):
            raise ValueError("policy point_feature_dim must be 3 or 6")
        self.point_feature_dim = int(point_feature_dim)
        self.policy_rgbd_resolution = str(policy_rgbd_resolution).strip().lower()
        resolve_policy_rgbd_resolution(self.policy_rgbd_resolution)
        (
            self.requested_object_mask_mode,
            self.effective_object_mask_mode,
        ) = _normalize_object_mask_modes(
            requested_object_mask_mode,
            effective_object_mask_mode,
        )
        self.fixed_sphere_completion_radius_m = fixed_sphere_completion_radius_m
        action_controller = getattr(policy, "action_controller", None)
        self.franka_action_contract_id = normalize_franka_action_contract_id(
            getattr(
                action_controller,
                "contract_id",
                LEGACY_FRANKA_ACTION_CONTRACT_ID,
            )
        )
        self.arm_raw_gain_rad = float(
            getattr(action_controller, "arm_raw_gain_rad", 0.015)
        )
        self.arm_target_filter_alpha = float(
            getattr(action_controller, "arm_target_filter_alpha", 0.20)
        )
        self.maximum_arm_target_step_rad = float(
            getattr(action_controller, "maximum_arm_target_step_rad", 0.015)
        )
        self.hand_target_filter_alpha = float(
            getattr(action_controller, "hand_target_filter_alpha", 0.20)
        )
        self.maximum_hand_target_step_rad = float(
            getattr(action_controller, "maximum_hand_target_step_rad", 0.05)
        )
        self.camera_owner = camera_owner
        self.maximum_pose_skew_s = _positive(maximum_pose_skew_s, "maximum_pose_skew_s")
        self.maximum_franka_action_age_s = _positive(
            maximum_franka_action_age_s,
            "maximum_franka_action_age_s",
        )
        self.maximum_franka_age_s = _positive(
            maximum_franka_age_s, "maximum_franka_age_s"
        )
        if self.maximum_franka_action_age_s >= self.maximum_franka_age_s:
            raise ValueError(
                "maximum_franka_action_age_s must be below maximum_franka_age_s"
            )
        self.maximum_rh56_age_s = _positive(maximum_rh56_age_s, "maximum_rh56_age_s")
        self.maximum_future_skew_s = _finite(
            maximum_future_skew_s, "maximum_future_skew_s"
        )
        if self.maximum_future_skew_s < 0.0:
            raise ValueError("maximum_future_skew_s cannot be negative")
        self.maximum_velocity_dt_s = _positive(
            maximum_velocity_dt_s, "maximum_velocity_dt_s"
        )
        self.camera_poll_interval_s = _positive(
            camera_poll_interval_s, "camera_poll_interval_s"
        )
        self.maximum_object_pointcloud_age_s = _positive(
            maximum_object_pointcloud_age_s,
            "maximum_object_pointcloud_age_s",
        )
        if (
            isinstance(maximum_actions_per_camera_frame, bool)
            or not isinstance(maximum_actions_per_camera_frame, int)
            or not 2 <= maximum_actions_per_camera_frame <= 6
        ):
            raise ValueError(
                "maximum_actions_per_camera_frame must be an integer in 2..6"
            )
        self.maximum_actions_per_camera_frame = int(maximum_actions_per_camera_frame)
        if not isinstance(franka_arrival_gate_enabled, (bool, np.bool_)):
            raise ValueError("franka_arrival_gate_enabled must be boolean")
        self.franka_arrival_gate_enabled = bool(franka_arrival_gate_enabled)
        self.franka_arrival_gate_tolerance_rad = _positive(
            franka_arrival_gate_tolerance_rad,
            "franka_arrival_gate_tolerance_rad",
        )
        self.franka_arrival_gate_timeout_s = _positive(
            franka_arrival_gate_timeout_s,
            "franka_arrival_gate_timeout_s",
        )
        self.rh56_hardware_rate_hz = _positive(
            rh56_hardware_rate_hz, "rh56_hardware_rate_hz"
        )
        if self.rh56_hardware_rate_hz > self.contract.policy_rate_hz:
            raise ValueError("RH56 hardware rate exceeds the policy rate")

        def six_ints(value: Sequence[int], name: str) -> tuple[int, ...]:
            raw = np.asarray(value)
            if raw.shape != (6,) or not np.all(np.isfinite(raw)):
                raise ValueError(f"{name} must contain six finite integers")
            numeric = raw.astype(np.float64)
            if not np.all(numeric == np.rint(numeric)):
                raise ValueError(f"{name} must contain six integers")
            return tuple(int(item) for item in numeric)

        configured_rh56_delta = np.asarray(six_ints(
            rh56_maximum_register_delta_per_update,
            "rh56_maximum_register_delta_per_update",
        ), dtype=np.int32)
        policy_ticks_per_hardware_update = self.contract.policy_rate_hz / self.rh56_hardware_rate_hz
        rounded_ticks = int(round(policy_ticks_per_hardware_update))
        if rounded_ticks < 1 or not np.isclose(
            policy_ticks_per_hardware_update,
            float(rounded_ticks),
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise ValueError("RH56 hardware rate must divide the policy rate")
        policy_order_delta = np.ceil(
            rounded_ticks
            * 1000.0
            * self.maximum_hand_target_step_rad
            / self.contract.q_hand_close_rad.astype(np.float64)
        ).astype(np.int32)
        checkpoint_rh56_delta = V94ActionMapper.policy_to_register_order(
            policy_order_delta
        ).astype(np.int32)
        self.rh56_maximum_register_delta_per_update = tuple(
            int(item)
            for item in np.maximum(configured_rh56_delta, checkpoint_rh56_delta)
        )
        self.rh56_minimum_angle_set_register_order = six_ints(
            rh56_minimum_angle_set_register_order,
            "rh56_minimum_angle_set_register_order",
        )
        self.rh56_maximum_angle_set_register_order = six_ints(
            rh56_maximum_angle_set_register_order,
            "rh56_maximum_angle_set_register_order",
        )
        self.rh56_feedback_to_command_offset_units = six_ints(
            rh56_feedback_to_command_offset_units,
            "rh56_feedback_to_command_offset_units",
        )
        self.rh56_feedback_to_command_valid_min = six_ints(
            rh56_feedback_to_command_valid_min,
            "rh56_feedback_to_command_valid_min",
        )
        self.rh56_feedback_to_command_valid_max = six_ints(
            rh56_feedback_to_command_valid_max,
            "rh56_feedback_to_command_valid_max",
        )
        self._feedback_mapper = feedback_mapper
        self._fingertip_model = fingertip_model
        self._proprio_builder = proprio_builder
        if live_visualizer is not None and not all(
            callable(getattr(live_visualizer, name, None))
            for name in ("open", "try_publish", "close")
        ):
            raise TypeError("live_visualizer lifecycle is incomplete")
        self.live_visualizer = live_visualizer
        if policy_io_recorder is not None and not callable(
            getattr(policy_io_recorder, "record_tick", None)
        ):
            raise TypeError("policy_io_recorder must expose record_tick()")
        self.policy_io_recorder = policy_io_recorder
        if rollout_trigger_config is not None and not isinstance(
            rollout_trigger_config, ThrownRolloutTriggerConfig
        ):
            raise TypeError(
                "rollout_trigger_config must be ThrownRolloutTriggerConfig or None"
            )
        self.rollout_trigger_config = rollout_trigger_config
        release_gate = (
            None
            if rollout_trigger_config is None
            else rollout_trigger_config.tabletop_release_height_gate
        )
        planner_config = getattr(policy, "config", None)
        if release_gate is not None and planner_config is not None:
            planner_reference = np.asarray(
                getattr(planner_config, "reference_intercept_center_base_m", ()),
                dtype=np.float64,
            )
            if planner_reference.shape != (3,) or not np.all(
                np.isfinite(planner_reference)
            ):
                raise ValueError(
                    "tabletop release gate requires a finite planner reference center"
                )
            if not np.isclose(
                planner_reference[2],
                release_gate.reference_center_base_z_m,
                atol=1.0e-6,
                rtol=0.0,
            ):
                raise ValueError(
                    "tabletop release-height reference differs from the replay planner"
                )
            planner_z_limit = float(
                getattr(planner_config, "max_position_correction_z_m", -1.0)
            )
            if release_gate.tolerance_m > planner_z_limit + 1.0e-9:
                raise ValueError(
                    "tabletop release-height tolerance exceeds the planner Z envelope"
                )
        self._tabletop_release_rgbd_adapter = (
            None
            if release_gate is None
            else PolicyRGBDResolutionAdapter(
                camera_K=self.contract.camera_K,
                source_image_size=(
                    self.contract.camera_width,
                    self.contract.camera_height,
                ),
                target_image_size=resolve_policy_rgbd_resolution(
                    self.policy_rgbd_resolution
                ),
            )
        )
        self._tabletop_release_projector: Optional[MaskedRGBDProjector] = None
        self._monotonic = monotonic
        self._realtime = realtime
        self._sleep = sleep
        self._camera_warmed = False
        self._camera_owner_preopened = bool(camera_owner_preopened)
        self._warm_frame_id: Optional[int] = None
        self._aligned_frame_id: Optional[int] = None
        self._built = False
        self._closed = False
        self._rollout_trigger_result: Optional[dict[str, object]] = None
        self._rollout_trigger_wait_state: Optional[
            _ThrownRolloutTriggerWaitState
        ] = None
        self._trigger_visualization_sequence = 0
        trigger_points = np.zeros((128, self.point_feature_dim), dtype=np.float32)
        trigger_valid = np.zeros(128, dtype=np.float32)
        trigger_transform = np.eye(4, dtype=np.float64)
        for value in (trigger_points, trigger_valid, trigger_transform):
            value.setflags(write=False)
        self._trigger_visualization_points = trigger_points
        self._trigger_visualization_valid = trigger_valid
        self._trigger_visualization_transform = trigger_transform

    def _now(self) -> float:
        return _finite(self._monotonic(), "source-factory monotonic clock")

    def _require_before(self, deadline: float, operation: str) -> float:
        now = self._now()
        if now >= deadline:
            raise V94LiveObservationError(f"hard deadline expired during {operation}")
        return now

    def open_and_warm_camera(
        self,
        admission: Any,
        *,
        hard_deadline_monotonic_s: float,
    ) -> None:
        del admission  # Runtime validates the sealed admission around this call.
        if self._camera_warmed or self._closed:
            raise V94LiveObservationError("production source factory is single-use")
        deadline = _finite(hard_deadline_monotonic_s, "hard_deadline_monotonic_s")
        self._require_before(deadline, "live visualization/D435 open")
        try:
            if self.live_visualizer is not None:
                self.live_visualizer.open()
            if self._camera_owner_preopened:
                if not self.camera_owner.is_open:
                    raise V94LiveObservationError(
                        "prewarmed D435 owner was closed before rollout"
                    )
                remaining = deadline - self._now()
                if remaining <= 0.0:
                    raise V94LiveObservationError(
                        "hard deadline expired during prewarmed D435 resync"
                    )
                camera = self.camera_owner.wait_for_next_fresh_publication(
                    timeout_s=min(
                        remaining,
                        self.camera_owner.maximum_publication_stall_s,
                    )
                )
                _validate_camera_contract(camera, self.contract)
                self._warm_frame_id = int(camera.frame_id)
            else:
                if self.camera_owner.is_open:
                    raise V94LiveObservationError(
                        "ordinary D435 owner was unexpectedly already open"
                    )
                self.camera_owner.open()
                while True:
                    self._require_before(deadline, "D435 warm-up")
                    try:
                        camera = self.camera_owner.snapshot()
                    except V94LiveObservationError as exc:
                        if "has not published" not in str(exc):
                            raise
                        self._sleep(self.camera_poll_interval_s)
                        continue
                    _validate_camera_contract(camera, self.contract)
                    self._warm_frame_id = int(camera.frame_id)
                    break
        except BaseException:
            if self.camera_owner.is_open:
                try:
                    self.camera_owner.close()
                except BaseException:
                    pass
            if self.live_visualizer is not None:
                try:
                    self.live_visualizer.close()
                except BaseException:
                    pass
            raise

        # These are filesystem/CPU-only initializations.  Finish them before
        # either robot owner starts so they consume no actuator heartbeat.
        if self._feedback_mapper is None:
            self._feedback_mapper = RH56FeedbackMapper(
                q_hand_close_rad=self.contract.q_hand_close_rad
            )
        if self._fingertip_model is None:
            self._fingertip_model = RH56FingertipKinematics(self.contract)
        if self._proprio_builder is None:
            self._proprio_builder = Proprio67Builder(
                q_home_rad=self.contract.q_home_rad,
                q_hand_close_rad=self.contract.q_hand_close_rad,
            )
        self._camera_warmed = True

    @staticmethod
    def _throw_trigger_mask(
        camera: _CameraSample,
    ) -> tuple[Optional[np.ndarray], str]:
        provider_mask = np.asarray(camera.mask, dtype=np.bool_)
        if bool(camera.mask_valid) and int(np.count_nonzero(provider_mask)) > 0:
            return provider_mask, "guarded_provider_publication"
        semantic = getattr(camera, "policy_semantic_mask", None)
        if bool(getattr(camera, "policy_semantic_valid", False)) and semantic is not None:
            semantic_mask = np.asarray(semantic, dtype=np.bool_)
            if int(np.count_nonzero(semantic_mask)) > 0:
                return semantic_mask, "exact_current_semantic_non_authoritative"
        return None, "none"

    def _tabletop_release_center_base_z(
        self,
        *,
        camera: _CameraSample,
        mask: Optional[np.ndarray],
    ) -> Optional[tuple[float, int]]:
        """Return a fresh policy-projector object height for ramp release.

        This uses the same RGB-D decimation, depth outlier threshold, calibrated
        eye-to-hand transform, and support-plane filter as the policy path.  It
        deliberately refuses projector fallback: every confirmation frame must
        contain current measured object depth.
        """

        config = self.rollout_trigger_config
        gate = (
            None if config is None else config.tabletop_release_height_gate
        )
        adapter = self._tabletop_release_rgbd_adapter
        if gate is None or adapter is None or mask is None:
            return None
        if self._tabletop_release_projector is None:
            self._tabletop_release_projector = MaskedRGBDProjector(
                camera_K=adapter.camera_K,
                T_base_camera_optical=self.contract.T_base_camera_optical,
                image_size=adapter.target_image_size,
                depth_range_m=self.contract.depth_range_m,
                point_feature_dim=3,
                minimum_valid_points=gate.minimum_valid_points,
                maximum_mask_depth_deviation_m=0.055,
                fixed_sphere_completion_radius_m=(
                    self.fixed_sphere_completion_radius_m
                ),
                support_plane_abcd=self.camera_owner.support_plane_abcd,
                support_plane_min_clearance_m=(
                    self.camera_owner.support_plane_min_clearance_m
                ),
            )
        adapted = adapter.adapt(
            color_bgr=camera.color_bgr,
            depth_m=camera.depth_m if camera.depth_raw is None else None,
            depth_raw=camera.depth_raw,
            object_mask=np.asarray(mask, dtype=np.bool_),
        )
        projected = self._tabletop_release_projector.project(
            color_bgr=adapted.color_bgr,
            depth_m=adapted.depth_m,
            depth_raw=adapted.depth_raw,
            depth_scale_m_per_unit=(
                camera.depth_scale_m_per_unit
                if adapted.depth_raw is not None
                else None
            ),
            object_mask=adapted.object_mask,
            # Identity makes the projector's historical palm-coordinate output
            # exactly robot-base XYZ for this camera-only admission check.
            T_base_palm_at_capture=np.eye(4, dtype=np.float64),
            captured_at_s=float(camera.timestamp_s),
            frame_id=int(camera.frame_id),
        )
        valid = np.asarray(projected.valid, dtype=np.float32) >= 0.5
        count = int(np.count_nonzero(valid))
        if projected.status != "fresh" or count < gate.minimum_valid_points:
            return None
        points = np.asarray(projected.xyzrgb_palm, dtype=np.float64)[valid, :3]
        if points.shape[0] != count or not np.all(np.isfinite(points)):
            return None
        return float(np.mean(points[:, 2], dtype=np.float64)), int(
            projected.source_valid_points
        )

    def _publish_throw_trigger_visualization(
        self,
        *,
        camera: _CameraSample,
        mask: Optional[np.ndarray],
        mask_kind: str,
    ) -> None:
        """Publish the camera-only trigger phase to the lossy UI/video sink."""

        sink = self.live_visualizer
        if sink is None:
            return
        selected = (
            np.zeros_like(np.asarray(camera.mask, dtype=np.bool_))
            if mask is None
            else np.asarray(mask, dtype=np.bool_)
        )
        area = int(np.count_nonzero(selected))
        bbox: Optional[tuple[int, int, int, int]] = None
        if area > 0:
            ys, xs = np.nonzero(selected)
            bbox = (
                int(xs.min()),
                int(ys.min()),
                int(xs.max()) + 1,
                int(ys.max()) + 1,
            )
        self._trigger_visualization_sequence += 1
        try:
            sink.try_publish(
                V94LiveVisualizationSample(
                    sequence=self._trigger_visualization_sequence,
                    frame_id=int(camera.frame_id),
                    captured_realtime_s=float(camera.timestamp_s),
                    color_bgr=camera.color_bgr,
                    object_mask=selected,
                    pointcloud_xyzrgb_palm=self._trigger_visualization_points,
                    pointcloud_valid=self._trigger_visualization_valid,
                    T_base_palm_at_capture=self._trigger_visualization_transform,
                    source_valid_points=0,
                    point_coordinate_frame=(
                        "robot_base_via_identity_T_base_palm"
                    ),
                    requested_object_mask_mode=str(
                        camera.requested_object_mask_mode
                    ),
                    effective_object_mask_mode=str(
                        camera.effective_object_mask_mode
                    ),
                    effective_provider_mask_publication_mode=str(
                        camera.effective_mask_publication_mode
                    ),
                    effective_provider_recovery_publication_mode=str(
                        camera.effective_recovery_publication_mode
                    ),
                    provider_published_mask_valid=bool(camera.mask_valid),
                    provider_published_mask_area_px=int(
                        camera.object_mask_area_px
                    ),
                    provider_published_mask_bbox_xyxy=(
                        tuple(int(value) for value in camera.object_mask_bbox_xyxy)
                    ),
                    provider_published_mask_source=str(camera.mask_source),
                    provider_published_mask_message=str(camera.message),
                    provider_online_sam2_status=str(camera.online_sam2_status),
                    projector_effective_policy_mask_provenance=(
                        f"throw_trigger_camera_only:{mask_kind}"
                    ),
                    projector_effective_policy_mask_source_frame_id=(
                        int(camera.frame_id)
                    ),
                    projector_effective_policy_mask_source_captured_realtime_s=(
                        float(camera.timestamp_s)
                    ),
                    projector_effective_policy_mask_area_px=area,
                    projector_effective_policy_mask_bbox_xyxy=bbox,
                )
            )
        except BaseException:
            # Visualization is explicitly lossy and must not affect trigger
            # admission or actuator ownership.
            return

    def _rollout_trigger_state(self) -> _ThrownRolloutTriggerWaitState:
        state = getattr(self, "_rollout_trigger_wait_state", None)
        if state is not None:
            return state
        config = self.rollout_trigger_config
        if config is None:
            raise V94LiveObservationError("throw trigger is not configured")
        started = self._now()
        state = _ThrownRolloutTriggerWaitState(
            trigger=_ThrownObjectRolloutTrigger(config),
            started_monotonic_s=started,
            configured_deadline_monotonic_s=(
                started + config.maximum_wait_s
            ),
        )
        self._rollout_trigger_wait_state = state
        return state

    def _wait_for_rollout_trigger_phase(
        self,
        *,
        hard_deadline_monotonic_s: float,
        stop_requested: threading.Event,
        stop_when_armed: bool,
        owners_active: bool,
    ) -> Optional[Mapping[str, object]]:
        """Advance one exact-frame trigger transaction to arm or detection."""

        config = self.rollout_trigger_config
        if config is None:
            self._rollout_trigger_result = {"enabled": False}
            return self._rollout_trigger_result
        if not self._camera_warmed or not self.camera_owner.is_open:
            raise V94LiveObservationError(
                "throw trigger requires a warmed, camera-only owner"
            )
        if not isinstance(stop_requested, threading.Event):
            raise TypeError("stop_requested must be a threading.Event")
        hard_deadline = _finite(
            hard_deadline_monotonic_s, "hard_deadline_monotonic_s"
        )
        state = self._rollout_trigger_state()
        trigger_name = str(config.display_name)
        started = state.started_monotonic_s
        deadline = min(
            hard_deadline, state.configured_deadline_monotonic_s
        )
        deadline_kind = (
            "runtime_hard_deadline"
            if hard_deadline <= state.configured_deadline_monotonic_s
            else "configured_trigger_wait"
        )
        while True:
            if stop_requested.is_set():
                raise V94LiveObservationError(
                    "throw trigger cancelled before rollout"
                )
            now = self._now()
            if now >= deadline:
                elapsed = now - started
                self._rollout_trigger_result = {
                    "enabled": True,
                    "mode": config.mode,
                    "detected": False,
                    "timed_out": True,
                    "timeout_kind": deadline_kind,
                    "wait_s": elapsed,
                    "configured_maximum_wait_s": config.maximum_wait_s,
                    "runtime_budget_at_start_s": hard_deadline - started,
                    "observed_frames": state.observed_frames,
                    "armed": state.trigger._armed_reason is not None,
                    "armed_reason": state.trigger._armed_reason,
                    "last_frame_id": (
                        None
                        if state.last_decision is None
                        else state.last_decision.frame_id
                    ),
                    "last_reason": (
                        None
                        if state.last_decision is None
                        else state.last_decision.reason
                    ),
                    "last_mask_kind": (
                        None
                        if state.last_decision is None
                        else state.last_decision.mask_kind
                    ),
                    "last_mask_area_px": (
                        0
                        if state.last_decision is None
                        else state.last_decision.mask_area_px
                    ),
                    "last_centroid_speed_px_s": (
                        0.0
                        if state.last_decision is None
                        else state.last_decision.centroid_speed_px_s
                    ),
                    "motion_detected_frame_id": (
                        None
                        if state.motion_detected_decision is None
                        else state.motion_detected_decision.frame_id
                    ),
                    "tabletop_height_confirmed_frames": (
                        state.tabletop_height_confirmed_frames
                    ),
                    "last_tabletop_center_base_z_m": (
                        state.last_tabletop_center_base_z_m
                    ),
                    "last_tabletop_height_source_points": (
                        state.last_tabletop_height_source_points
                    ),
                }
                print(
                    f"[{trigger_name} TIMEOUT] "
                    f"kind={deadline_kind} waited={elapsed:.3f}s "
                    f"armed={state.trigger._armed_reason is not None} "
                    f"last_reason={self._rollout_trigger_result['last_reason']} "
                    f"last_mask_area={self._rollout_trigger_result['last_mask_area_px']}",
                    flush=True,
                )
                raise V94LiveObservationError(
                    "throw trigger timed out before a qualifying motion/entry "
                    f"frame: kind={deadline_kind} waited={elapsed:.3f}s "
                    f"armed={state.trigger._armed_reason is not None}"
                )
            camera = self.camera_owner.snapshot()
            if state.last_frame_id != int(camera.frame_id):
                mask, mask_kind = self._throw_trigger_mask(camera)
                decision = state.trigger.observe(
                    frame_id=int(camera.frame_id),
                    timestamp_s=float(camera.timestamp_s),
                    mask=mask,
                    mask_kind=mask_kind,
                )
                state.last_decision = decision
                state.observed_frames += 1
                state.last_frame_id = int(camera.frame_id)
                self._publish_throw_trigger_visualization(
                    camera=camera,
                    mask=mask,
                    mask_kind=mask_kind,
                )
                if decision.armed_now and stop_when_armed:
                    state.prepared_monotonic_s = self._now()
                    self._rollout_trigger_result = {
                        "enabled": True,
                        "mode": config.mode,
                        "detected": False,
                        "armed": True,
                        "armed_reason": state.trigger._armed_reason,
                        "armed_frame_id": decision.frame_id,
                        "armed_camera_timestamp_s": decision.timestamp_s,
                        "preparation_started_monotonic_s": (
                            state.prepared_monotonic_s
                        ),
                        "observed_frames": state.observed_frames,
                    }
                    print(
                        f"[{trigger_name} PREPARING] "
                        f"frame={decision.frame_id} basis={decision.reason}; "
                        "keep holding while robot owners and policy synchronize",
                        flush=True,
                    )
                    return dict(self._rollout_trigger_result)
                release_gate = config.tabletop_release_height_gate
                if (
                    decision.detected
                    and release_gate is not None
                    and state.motion_detected_decision is None
                ):
                    state.motion_detected_decision = decision
                    state.motion_detected_monotonic_s = self._now()
                    measured = self._tabletop_release_center_base_z(
                        camera=camera,
                        mask=mask,
                    )
                    measured_text = "depth unavailable"
                    if measured is not None:
                        center_z, source_points = measured
                        state.last_tabletop_center_base_z_m = center_z
                        state.last_tabletop_height_source_points = source_points
                        measured_text = (
                            f"center_z={center_z:.4f}m "
                            f"delta={center_z - release_gate.reference_center_base_z_m:+.4f}m"
                        )
                    print(
                        f"[{trigger_name} MOTION] frame={decision.frame_id} "
                        f"speed={decision.centroid_speed_px_s:.1f}px/s; "
                        f"{measured_text}; waiting for "
                        f"{release_gate.confirm_frames} fresh flat-table frames",
                        flush=True,
                    )

                if (
                    release_gate is not None
                    and state.motion_detected_decision is not None
                ):
                    measured = self._tabletop_release_center_base_z(
                        camera=camera,
                        mask=mask,
                    )
                    if measured is None:
                        state.tabletop_height_confirmed_frames = 0
                        state.last_tabletop_center_base_z_m = None
                        state.last_tabletop_height_source_points = 0
                    else:
                        center_z, source_points = measured
                        state.last_tabletop_center_base_z_m = center_z
                        state.last_tabletop_height_source_points = source_points
                        within_flat_height = bool(
                            abs(
                                center_z
                                - release_gate.reference_center_base_z_m
                            )
                            <= release_gate.tolerance_m + 1.0e-9
                        )
                        if within_flat_height:
                            state.tabletop_height_confirmed_frames += 1
                        else:
                            state.tabletop_height_confirmed_frames = 0
                    if (
                        state.tabletop_height_confirmed_frames
                        >= release_gate.confirm_frames
                    ):
                        detected = self._now()
                        motion = state.motion_detected_decision
                        assert motion is not None
                        assert state.last_tabletop_center_base_z_m is not None
                        self._rollout_trigger_result = {
                            "enabled": True,
                            "mode": config.mode,
                            "detected": True,
                            "reason": "motion_then_tabletop_height_confirmed",
                            "frame_id": int(camera.frame_id),
                            "camera_timestamp_s": float(camera.timestamp_s),
                            "detected_monotonic_s": detected,
                            "wait_s": detected - started,
                            "mask_kind": str(mask_kind),
                            "mask_area_px": int(decision.mask_area_px),
                            "centroid_speed_px_s": (
                                motion.centroid_speed_px_s
                            ),
                            "centroid_displacement_px": (
                                motion.centroid_displacement_px
                            ),
                            "motion_detected_frame_id": motion.frame_id,
                            "motion_detected_camera_timestamp_s": (
                                motion.timestamp_s
                            ),
                            "motion_detected_monotonic_s": (
                                state.motion_detected_monotonic_s
                            ),
                            "tabletop_reference_center_base_z_m": (
                                release_gate.reference_center_base_z_m
                            ),
                            "tabletop_center_base_z_m": (
                                state.last_tabletop_center_base_z_m
                            ),
                            "tabletop_height_delta_m": (
                                state.last_tabletop_center_base_z_m
                                - release_gate.reference_center_base_z_m
                            ),
                            "tabletop_height_tolerance_m": (
                                release_gate.tolerance_m
                            ),
                            "tabletop_height_confirmed_frames": (
                                state.tabletop_height_confirmed_frames
                            ),
                            "tabletop_height_source_points": (
                                state.last_tabletop_height_source_points
                            ),
                            "preparation_started_monotonic_s": (
                                state.prepared_monotonic_s
                            ),
                            "armed_ready_monotonic_s": state.ready_monotonic_s,
                            "owner_policy_preparation_s": (
                                None
                                if state.prepared_monotonic_s is None
                                or state.ready_monotonic_s is None
                                else state.ready_monotonic_s
                                - state.prepared_monotonic_s
                            ),
                            "armed_ready_to_detection_s": (
                                None
                                if state.ready_monotonic_s is None
                                else detected - state.ready_monotonic_s
                            ),
                            "active_franka_owner_at_detection": bool(
                                owners_active
                            ),
                            "active_rh56_owner_at_detection": bool(
                                owners_active
                            ),
                        }
                        print(
                            f"[{trigger_name} RELEASED] "
                            f"frame={camera.frame_id} "
                            f"center_z={state.last_tabletop_center_base_z_m:.4f}m "
                            f"delta={self._rollout_trigger_result['tabletop_height_delta_m']:+.4f}m "
                            f"confirmed={state.tabletop_height_confirmed_frames}/"
                            f"{release_gate.confirm_frames}; starting first policy tick",
                            flush=True,
                        )
                        return dict(self._rollout_trigger_result)
                elif decision.detected:
                    detected = self._now()
                    self._rollout_trigger_result = {
                        "enabled": True,
                        "mode": config.mode,
                        "detected": True,
                        "reason": decision.reason,
                        "frame_id": decision.frame_id,
                        "camera_timestamp_s": decision.timestamp_s,
                        "detected_monotonic_s": detected,
                        "wait_s": detected - started,
                        "mask_kind": decision.mask_kind,
                        "mask_area_px": decision.mask_area_px,
                        "centroid_speed_px_s": (
                            decision.centroid_speed_px_s
                        ),
                        "centroid_displacement_px": (
                            decision.centroid_displacement_px
                        ),
                        "preparation_started_monotonic_s": (
                            state.prepared_monotonic_s
                        ),
                        "armed_ready_monotonic_s": state.ready_monotonic_s,
                        "owner_policy_preparation_s": (
                            None
                            if state.prepared_monotonic_s is None
                            or state.ready_monotonic_s is None
                            else state.ready_monotonic_s
                            - state.prepared_monotonic_s
                        ),
                        "armed_ready_to_detection_s": (
                            None
                            if state.ready_monotonic_s is None
                            else detected - state.ready_monotonic_s
                        ),
                        "active_franka_owner_at_detection": bool(owners_active),
                        "active_rh56_owner_at_detection": bool(owners_active),
                    }
                    print(
                        f"[{trigger_name} DETECTED] "
                        f"frame={decision.frame_id} reason={decision.reason} "
                        f"speed={decision.centroid_speed_px_s:.1f}px/s; "
                        "starting first policy tick",
                        flush=True,
                    )
                    return dict(self._rollout_trigger_result)
            remaining = deadline - now
            if remaining <= 0.0:
                continue
            if state.last_frame_id is None:
                self._sleep(min(config.poll_interval_s, remaining))
            else:
                self.camera_owner.wait_for_frame_change(
                    state.last_frame_id,
                    timeout_s=min(config.poll_interval_s, remaining),
                )

    def prepare_rollout_trigger(
        self,
        admission: Any,
        *,
        hard_deadline_monotonic_s: float,
        stop_requested: threading.Event,
    ) -> Optional[Mapping[str, object]]:
        """Reach stable/empty pre-arm while both robot owners remain closed."""

        del admission
        if self.rollout_trigger_config is None:
            self._rollout_trigger_result = {"enabled": False}
            return self._rollout_trigger_result
        trigger_name = str(self.rollout_trigger_config.display_name)
        print(
            f"[{trigger_name} START] camera/model services ready; "
            "keep the object still until the trigger is armed",
            flush=True,
        )
        return self._wait_for_rollout_trigger_phase(
            hard_deadline_monotonic_s=hard_deadline_monotonic_s,
            stop_requested=stop_requested,
            stop_when_armed=True,
            owners_active=False,
        )

    def detect_rollout_trigger(
        self,
        admission: Any,
        *,
        hard_deadline_monotonic_s: float,
        stop_requested: threading.Event,
    ) -> Optional[Mapping[str, object]]:
        """Detect release only after static owners and policy are ready."""

        del admission
        state = self._rollout_trigger_wait_state
        if state is None or state.trigger._armed_reason is None:
            raise V94LiveObservationError(
                "throw trigger detection requires a completed pre-arm phase"
            )
        state.ready_monotonic_s = self._now()
        config = self.rollout_trigger_config
        assert config is not None
        print(
            f"[{config.display_name} ARMED] robot owners and policy ready; "
            f"{config.ready_instruction}",
            flush=True,
        )
        # The stable-object pre-arm frame may be hundreds of milliseconds old
        # after static owner startup and policy warmup.  Establish a fresh
        # post-ready centroid before velocity admission so accumulated natural
        # hand drift cannot masquerade as a release.  Empty-field entry needs
        # no visible baseline and deliberately retains its arm state.
        if state.trigger._armed_reason == "stable_object":
            state.trigger._previous_visible = None
        return self._wait_for_rollout_trigger_phase(
            hard_deadline_monotonic_s=hard_deadline_monotonic_s,
            stop_requested=stop_requested,
            stop_when_armed=False,
            owners_active=True,
        )

    def wait_for_rollout_trigger(
        self,
        admission: Any,
        *,
        hard_deadline_monotonic_s: float,
        stop_requested: threading.Event,
    ) -> Optional[Mapping[str, object]]:
        """Camera-only compatibility path used by perception tests."""

        prepared = self.prepare_rollout_trigger(
            admission,
            hard_deadline_monotonic_s=hard_deadline_monotonic_s,
            stop_requested=stop_requested,
        )
        if self.rollout_trigger_config is None:
            return prepared
        state = self._rollout_trigger_wait_state
        assert state is not None
        state.ready_monotonic_s = self._now()
        config = self.rollout_trigger_config
        assert config is not None
        print(
            f"[{config.display_name} ARMED] camera/model ready; "
            f"{config.ready_instruction}",
            flush=True,
        )
        return self._wait_for_rollout_trigger_phase(
            hard_deadline_monotonic_s=hard_deadline_monotonic_s,
            stop_requested=stop_requested,
            stop_when_armed=False,
            owners_active=False,
        )

    @property
    def rollout_trigger_diagnostics(self) -> Mapping[str, object]:
        if self._rollout_trigger_result is None:
            return {"enabled": self.rollout_trigger_config is not None}
        return dict(self._rollout_trigger_result)

    def wait_for_camera_pose_alignment(
        self,
        *,
        franka_session: Any,
        hard_deadline_monotonic_s: float,
    ) -> None:
        if not self._camera_warmed or self._warm_frame_id is None:
            raise V94LiveObservationError("D435 warm-up is incomplete")
        if not bool(getattr(franka_session, "c2_bootstrap_ready", False)):
            raise V94LiveObservationError(
                "Franka measured-hold bootstrap is incomplete"
            )
        pose_ring = getattr(franka_session, "pose_ring", None)
        if not isinstance(pose_ring, FrankaPoseRing):
            raise V94LiveObservationError("Franka session has no owned pose ring")
        deadline = _finite(hard_deadline_monotonic_s, "hard_deadline_monotonic_s")
        blocked_frame = self._warm_frame_id
        while True:
            now = self._require_before(deadline, "camera/Franka pose alignment")
            camera = self.camera_owner.snapshot()
            _validate_camera_contract(camera, self.contract)
            samples = pose_ring.snapshot()
            newer = int(camera.frame_id) != blocked_frame
            aligned = any(
                abs(sample.realtime_s - float(camera.timestamp_s))
                <= self.maximum_pose_skew_s
                for sample in samples
            )
            if newer and len(samples) >= 2 and aligned:
                self._aligned_frame_id = int(camera.frame_id)
                return
            wait_s = min(self.camera_poll_interval_s, deadline - now)
            if wait_s <= 0.0:
                continue
            self.camera_owner.wait_for_frame_change(
                int(camera.frame_id), timeout_s=wait_s
            )

    def __call__(
        self,
        *,
        franka_session: Any,
        rh56_feedback_source: RH56FeedbackHistorySource,
        hard_deadline_monotonic_s: float,
    ) -> TransactionalBoundedV94PolicyTickSource:
        if self._built or self._closed:
            raise V94LiveObservationError("production source factory is single-use")
        if self._aligned_frame_id is None:
            raise V94LiveObservationError("camera/Franka alignment is incomplete")
        self._require_before(
            _finite(hard_deadline_monotonic_s, "hard_deadline_monotonic_s"),
            "policy source construction",
        )
        if not isinstance(rh56_feedback_source, RH56FeedbackHistorySource):
            raise TypeError("RH56 source has no owner-cache feedback contract")
        history = rh56_feedback_source.require_fresh_feedback_history(
            minimum_samples=2,
            maximum_age_s=self.maximum_rh56_age_s,
        )
        latest_feedback = history.samples[-1].feedback
        pose_ring = getattr(franka_session, "pose_ring", None)
        envelope = getattr(franka_session, "envelope", None)
        if not isinstance(pose_ring, FrankaPoseRing) or envelope is None:
            raise V94LiveObservationError("Franka session binding is incomplete")
        poses = pose_ring.snapshot()
        if len(poses) < 2:
            raise V94LiveObservationError("Franka pose bootstrap was lost")
        camera = self.camera_owner.snapshot()
        if int(camera.frame_id) < self._aligned_frame_id:
            raise V94LiveObservationError("D435 frame regressed after alignment")
        if not np.allclose(
            np.asarray(envelope.joint_limits_rad, dtype=np.float64),
            self.contract.joint_limits_rad,
            atol=1.0e-7,
            rtol=0.0,
        ):
            raise V94LiveObservationError(
                "commissioned Franka joint limits differ from the V94 contract"
            )
        assert self._feedback_mapper is not None
        assert self._fingertip_model is not None
        assert self._proprio_builder is not None
        mapped_hand = self._feedback_mapper.map(
            np.asarray(latest_feedback.angles, dtype=np.int64)
        )
        initial_rh56_command = []
        for axis, (actual, offset, valid_min, valid_max, lower, upper) in enumerate(
            zip(
                latest_feedback.angles,
                self.rh56_feedback_to_command_offset_units,
                self.rh56_feedback_to_command_valid_min,
                self.rh56_feedback_to_command_valid_max,
                self.rh56_minimum_angle_set_register_order,
                self.rh56_maximum_angle_set_register_order,
            )
        ):
            if not valid_min <= int(actual) <= valid_max:
                raise V94LiveObservationError(
                    f"RH56 axis {axis} ANGLE_ACT {actual} is outside the "
                    f"validated feedback-to-command range {valid_min}..{valid_max}"
                )
            command = int(actual) + offset
            if command > upper and command - upper <= abs(offset):
                command = upper
            elif command < lower and lower - command <= abs(offset):
                command = lower
            if not lower <= command <= upper:
                raise V94LiveObservationError(
                    f"RH56 axis {axis} feedback-equivalent command {command} "
                    f"is outside {lower}..{upper}"
                )
            initial_rh56_command.append(command)
        observation_provider = PersistentV94ObservationOwner(
            contract=self.contract,
            franka_pose_ring=pose_ring,
            expected_F_T_EE=envelope.expected_F_T_EE,
            rh56_feedback_source=rh56_feedback_source,
            camera_source=self.camera_owner,
            maximum_pose_skew_s=self.maximum_pose_skew_s,
            maximum_franka_action_age_s=self.maximum_franka_action_age_s,
            maximum_franka_age_s=self.maximum_franka_age_s,
            maximum_rh56_age_s=self.maximum_rh56_age_s,
            maximum_future_skew_s=self.maximum_future_skew_s,
            maximum_velocity_dt_s=self.maximum_velocity_dt_s,
            point_feature_dim=self.point_feature_dim,
            policy_rgbd_resolution=self.policy_rgbd_resolution,
            requested_object_mask_mode=self.requested_object_mask_mode,
            effective_object_mask_mode=self.effective_object_mask_mode,
            fixed_sphere_completion_radius_m=(
                self.fixed_sphere_completion_radius_m
            ),
            support_plane_abcd=self.camera_owner.support_plane_abcd,
            support_plane_min_clearance_m=(
                self.camera_owner.support_plane_min_clearance_m
            ),
            feedback_mapper=self._feedback_mapper,
            fingertip_model=self._fingertip_model,
            proprio_builder=self._proprio_builder,
            visualization_sink=self.live_visualizer,
            monotonic=self._monotonic,
            realtime=self._realtime,
        )
        mapper = TransactionalV94ActionMapper(
            initial_arm_target_q_rad=poses[-1].q_rad,
            initial_hand_target_q_policy_order_rad=(mapped_hand.q_policy_order_rad),
            joint_limits_rad=envelope.joint_limits_rad,
            joint_limit_margin_rad=envelope.joint_limit_margin_rad,
            control_dt_s=self.contract.control_dt_s,
            q_hand_close_rad=self.contract.q_hand_close_rad,
            commissioned_max_arm_target_rate_rad_s=(
                getattr(
                    envelope,
                    "maximum_target_rate_rad_s",
                    envelope.maximum_velocity_rad_s,
                )
            ),
            arm_raw_gain_rad=self.arm_raw_gain_rad,
            target_filter_alpha=self.arm_target_filter_alpha,
            hand_target_filter_alpha=self.hand_target_filter_alpha,
            maximum_arm_target_step_rad=self.maximum_arm_target_step_rad,
            maximum_hand_target_step_rad=self.maximum_hand_target_step_rad,
            franka_action_contract_id=self.franka_action_contract_id,
        )
        source = TransactionalBoundedV94PolicyTickSource(
            observation_provider=observation_provider,
            policy=self.policy,
            action_mapper=mapper,
            maximum_actions_per_camera_frame=(self.maximum_actions_per_camera_frame),
            startup_non_actuated_policy_steps=(
                QD_G015_STARTUP_NON_ACTUATED_POLICY_STEPS
                if self.franka_action_contract_id
                == QD_G015_FRANKA_ACTION_CONTRACT_ID
                else 0
            ),
            maximum_object_pointcloud_age_s=(self.maximum_object_pointcloud_age_s),
            maximum_franka_action_age_s=self.maximum_franka_action_age_s,
            maximum_franka_hard_age_s=self.maximum_franka_age_s,
            franka_arrival_gate_enabled=self.franka_arrival_gate_enabled,
            franka_arrival_gate_tolerance_rad=(
                self.franka_arrival_gate_tolerance_rad
            ),
            franka_arrival_gate_timeout_s=self.franka_arrival_gate_timeout_s,
            policy_io_recorder=self.policy_io_recorder,
            rh56_hardware_command_shaper=(
                TransactionalRH56HardwareCommandShaper(
                    initial_angle_set_register_order=np.asarray(
                        initial_rh56_command, dtype=np.int32
                    ),
                    policy_rate_hz=self.contract.policy_rate_hz,
                    hardware_rate_hz=self.rh56_hardware_rate_hz,
                    maximum_register_delta_per_update=np.asarray(
                        self.rh56_maximum_register_delta_per_update,
                        dtype=np.int32,
                    ),
                    minimum_angle_set_register_order=np.asarray(
                        self.rh56_minimum_angle_set_register_order,
                        dtype=np.int32,
                    ),
                    maximum_angle_set_register_order=np.asarray(
                        self.rh56_maximum_angle_set_register_order,
                        dtype=np.int32,
                    ),
                )
            ),
            monotonic=self._monotonic,
            realtime=self._realtime,
        )
        self._built = True
        return source

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        camera_failure: Optional[BaseException] = None
        if self.camera_owner.is_open:
            try:
                self.camera_owner.close()
            except BaseException as exc:
                camera_failure = exc
        if self.live_visualizer is not None:
            # The viewer never owns hardware.  Its cleanup must not turn a
            # verified dual-device stop into a runtime failure.
            try:
                self.live_visualizer.close()
            except BaseException:
                pass
        if camera_failure is not None:
            raise camera_failure


__all__ = [
    "D435ObjectCameraOwner",
    "LiveD435ProviderFactory",
    "PrewarmedD435CameraHandoff",
    "PersistentV94ObservationOwner",
    "ProductionV94PolicyTickSourceFactory",
    "RH56FeedbackHistorySource",
    "V94CameraSnapshotSource",
    "V94LiveObservationError",
    "open_prevalidated_d435_handoff",
]
