#!/usr/bin/env python3
"""Read-only V94 policy preview using the installed camera, FR3, and RH56.

This process intentionally contains no Franka control-handle creation and no
RH56 register write.  Camera tracking and RH56 feedback run in producer
threads; the main 60 Hz loop is the sole Franka read-only owner and evaluates
the checkpoint over latest, timestamp-checked observations.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any, Deque, Mapping, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    workspace = Path(__file__).resolve().parents[2]
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    from sim2real.deployment.bundle import DeployBundle, load_checkpoint_safely
    from sim2real.io import FrankaStateReader, InspireStateReader
    from sim2real.policy import RollingStudentPolicy
    from sim2real.contracts.actions import V94ActionMapper
    from sim2real.contracts.v94 import INITIAL_PREVIOUS_ACTION13, V94Contract
    from sim2real.observation.kinematics import (
        KinematicVelocityTracker,
        RH56FeedbackMapper,
        RH56FingertipKinematics,
        T_base_policy_palm_from_franka,
    )
    from sim2real.observation.model import (
        MaskedRGBDProjector,
        PolicyHistory,
        Proprio67Builder,
    )
    from sim2real.deployment.verify import verify_v94_bundle
else:
    from sim2real.deployment.bundle import DeployBundle, load_checkpoint_safely
    from sim2real.io import FrankaStateReader, InspireStateReader
    from sim2real.policy import RollingStudentPolicy
    from sim2real.contracts.actions import V94ActionMapper
    from sim2real.contracts.v94 import INITIAL_PREVIOUS_ACTION13, V94Contract
    from sim2real.observation.kinematics import (
        KinematicVelocityTracker,
        RH56FeedbackMapper,
        RH56FingertipKinematics,
        T_base_policy_palm_from_franka,
    )
    from .model import (
        MaskedRGBDProjector,
        PolicyHistory,
        Proprio67Builder,
    )
    from sim2real.deployment.verify import verify_v94_bundle


SIM2REAL_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
from sim2real.workspace_paths import DEFAULT_V94_DEPLOY_BUNDLE

DEFAULT_BUNDLE = DEFAULT_V94_DEPLOY_BUNDLE
DEFAULT_PCD_CONFIG = (
    WORKSPACE_ROOT / "perception" / "configs" / "d435_default.yaml"
)
MAX_HOST_CLOCK_OFFSET_JUMP_S = 0.010
MAX_HOST_CLOCK_PAIR_SPAN_S = 0.002
HOST_CLOCK_PAIR_READ_ATTEMPTS = 3
BACKGROUND_JOIN_TIMEOUT_S = 5.0
DEFAULT_MIN_ACTION_COMPUTE_RESERVE_S = 0.012
ACTION_COMPUTE_RESERVE_WARMUP_FACTOR = 2.0
DEFAULT_COMPUTE_THREADS = 1
DEFAULT_MAX_CAMERA_DROPOUT_S = 0.500
MAX_COLOR_DEPTH_TIMESTAMP_SKEW_S = 0.008
DEFAULT_RH56_READ_RATE_HZ = 60.0
CAMERA_REJECTION_LOG_EVERY_N = 30
PROVIDER_OUTPUT_MODES = ("mask_only", "full_packet")
RETRYABLE_CAMERA_TIMEOUT_BACKOFF_S = 0.001
MAX_POLICY_ACTIONS_PER_CAMERA_FRAME = 2
FIXED_ROI_INITIALIZATION_MAX_ATTEMPTS = 3
FIXED_ROI_INITIALIZATION_MAX_FRAME_READS = 12
FIXED_ROI_INITIALIZATION_MAX_RETRYABLE_CAMERA_TIMEOUTS = 3
FIXED_ROI_INITIALIZATION_MIN_MASK_MARGIN_PX = 8
SAM2_INITIALIZATION_MIN_IMAGE_MARGIN_PX = 2
POLICY_REUSE_HOLD_SCHEMA_VERSION = 1
POLICY_REUSE_HOLD_REASON = "camera_frame_policy_reuse_limit"
POLICY_REUSE_GUARD_SEMANTICS = (
    "third_eligible_tick_holds_without_policy_state_or_action_evidence"
)


@dataclass(frozen=True)
class _CameraSample:
    color_bgr: np.ndarray
    depth_m: Optional[np.ndarray]
    mask: np.ndarray
    camera_K: np.ndarray
    distortion: np.ndarray
    distortion_model: str
    depth_scale_m_per_unit: float
    timestamp_s: float
    frame_id: int
    mask_valid: bool
    message: str
    provider_timings_ms: np.ndarray
    retrieved_at_s: float
    timestamp_domain: str
    color_depth_timestamp_skew_s: float
    color_depth_epoch_timestamp_skew_s: float
    rejected_timestamp_skew_frames: int
    last_rejected_color_depth_skew_s: float
    published_at_s: float
    dropped_queued_framesets: int
    sensor_frame_number: int
    depth_sensor_frame_number: int = 0
    rejected_transport_stale_frames: int = 0
    last_rejected_transport_age_s: float = 0.0
    retrieved_monotonic_s: float = 0.0
    host_clock_pair_span_s: float = 0.0
    capture_diagnostic_json: str = "{}"
    mask_source: str = "unknown"
    online_sam2_status: str = "unknown"
    object_mask_area_px: int = 0
    object_mask_bbox_xyxy: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.int32)
    )
    # Production D435 samples retain their owned Z16 buffer and defer metric
    # conversion until the projector has selected object-mask pixels.  The
    # optional metric field remains for reviewed providers/test doubles that
    # expose only ``depth_m``.
    depth_raw: Optional[np.ndarray] = None
    # provider_timings_ms keeps its six-element stored ABI. Index 4 means
    # mask_gate for the production mask-only path and pcd for full_packet.
    provider_output_mode: str = "mask_only"
    requested_object_mask_mode: str = "unknown"
    effective_object_mask_mode: str = "unknown"
    effective_mask_publication_mode: str = "unknown"
    effective_recovery_publication_mode: str = "unknown"
    policy_semantic_mask: Optional[np.ndarray] = None
    policy_semantic_valid: bool = False
    policy_semantic_source: str = ""

    @property
    def provider_published_mask_source(self) -> str:
        """Unambiguous alias for the provider's final publication decision."""

        return str(self.mask_source)

    @property
    def provider_published_mask_message(self) -> str:
        return str(self.message)

    @property
    def provider_online_sam2_status(self) -> str:
        return str(self.online_sam2_status)


@dataclass(frozen=True)
class _HandSample:
    angle_targets: np.ndarray
    angle_act: np.ndarray
    positions: np.ndarray
    forces: np.ndarray
    currents: np.ndarray
    errors: np.ndarray
    statuses: np.ndarray
    temperatures_c: np.ndarray
    q_policy_rad: np.ndarray
    dq_policy_rad_s: np.ndarray
    timestamp_s: float
    snapshot_span_s: float


@dataclass(frozen=True)
class _DeadlineMetrics:
    scheduled_tick_monotonic_s: float
    tick_started_monotonic_s: float
    tick_finished_monotonic_s: float
    start_lateness_s: float
    computation_s: float
    next_deadline_overrun_s: float


@dataclass(frozen=True)
class _HostClockSample:
    realtime_s: float
    monotonic_s: float
    pair_span_s: float
    realtime_minus_monotonic_s: float
    offset_delta_from_start_s: float
    offset_delta_from_previous_s: float


@dataclass(frozen=True)
class _FreshnessBudgetMetrics:
    pointcloud_age_s: float
    remaining_s: float
    required_reserve_s: float
    eligible: bool


@dataclass(frozen=True)
class _OnlineSAM2RuntimeSelection:
    """Auditable live-only selection layered over the provider YAML.

    Service enablement remains disable-only.  Mask publication may be selected
    in memory as guarded or legacy so one immutable YAML can support an A/B
    deployment without changing checkpoint or calibration inputs.
    """

    source_config_enabled: bool
    disable_override_requested: bool
    effective_config_enabled: bool
    source_mask_publication_mode: str
    effective_mask_publication_mode: str
    effective_recovery_publication_mode: str
    requested_object_mask_mode: Optional[str]

    @property
    def mode(self) -> str:
        if self.disable_override_requested:
            return "disabled_by_live_cli"
        if self.effective_config_enabled:
            return "enabled_by_config"
        return "disabled_by_config"

    @property
    def effective_object_mask_mode(self) -> str:
        """Canonical algorithm selected after every in-memory override."""

        publication = str(self.effective_mask_publication_mode).strip().lower()
        recovery = str(self.effective_recovery_publication_mode).strip().lower()
        if not self.effective_config_enabled:
            return (
                "adaptive_only"
                if publication == "adaptive_fusion"
                else "semantic_sam2_unavailable"
            )
        if publication == "semantic_sam2":
            return "legacy"
        if publication == "guarded_sam2_primary":
            if recovery == "unified_three_evidence":
                return "guarded_v2"
        if publication == "adaptive_fusion":
            if recovery == "legacy_double_confirm":
                return "guarded_v1"
        return "custom"


class _FreshnessFrameGate:
    """Hold policy mutation until a rejected camera frame is replaced.

    A freshness rejection is a discontinuity, not a missing 60 Hz logical
    observation.  The rejected frame and every repeated observation of it are
    held without mutating policy state.  A replacement within the velocity
    tracker's safe finite-difference interval preserves normal 30/60 Hz history
    semantics.  Only a longer gap (or a rejection after mutable projection or
    policy work) requires fail-closed rebootstrap.
    """

    def __init__(self) -> None:
        self.blocked_frame_id: Optional[int] = None
        self.rejected_frames = 0
        self.budget_rejected_frames = 0
        self.transient_stale_rejected_frames = 0
        self.unusable_initial_observation_rejected_frames = 0
        self.wait_ticks = 0
        self.rejection_episode_active = False
        self.replacement_ready = False
        self.forced_rebootstrap_pending = False
        self.rebootstrap_count = 0

    def waiting_for_newer_frame(self, frame_id: int) -> bool:
        current = int(frame_id)
        if self.blocked_frame_id is None:
            return False
        if current == self.blocked_frame_id:
            self.wait_ticks += 1
            return True
        self.blocked_frame_id = None
        self.replacement_ready = True
        return False

    def reject(
        self,
        frame_id: int,
        *,
        reason: str = "budget",
        force_rebootstrap: bool = False,
    ) -> None:
        if reason not in {"budget", "stale", "unusable_initial_observation"}:
            raise ValueError(f"unknown camera freshness rejection reason {reason!r}")
        self.blocked_frame_id = int(frame_id)
        self.rejection_episode_active = True
        self.replacement_ready = False
        self.forced_rebootstrap_pending = bool(
            self.forced_rebootstrap_pending or force_rebootstrap
        )
        self.rejected_frames += 1
        if reason == "budget":
            self.budget_rejected_frames += 1
        elif reason == "stale":
            self.transient_stale_rejected_frames += 1
        else:
            self.unusable_initial_observation_rejected_frames += 1

    def resolve_replacement(
        self,
        frame_id: int,
        *,
        policy_state_gap_s: Optional[float],
        maximum_continuity_gap_s: float,
    ) -> bool:
        """Resolve a replacement and report whether continuity must reset."""

        if not self.rejection_episode_active:
            return False
        if self.blocked_frame_id is not None or not self.replacement_ready:
            return False
        # ``waiting_for_newer_frame`` is the only operation that can make a
        # replacement ready, so this frame is necessarily different from the
        # rejected one.  Keep the argument explicit for readable call sites and
        # reject nonsensical IDs before logical state is reset.
        if int(frame_id) < 0:
            raise ValueError("replacement camera frame_id must be non-negative")
        maximum = float(maximum_continuity_gap_s)
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("maximum continuity gap must be finite and positive")
        if policy_state_gap_s is None:
            long_gap = False
        else:
            gap = float(policy_state_gap_s)
            if not np.isfinite(gap) or gap < 0.0:
                raise ValueError("policy state gap must be finite and non-negative")
            long_gap = gap > maximum
        rebootstrap = bool(self.forced_rebootstrap_pending or long_gap)
        self.rejection_episode_active = False
        self.replacement_ready = False
        self.forced_rebootstrap_pending = False
        if rebootstrap:
            self.rebootstrap_count += 1
        return rebootstrap


class _CameraPolicyReuseGuard:
    """Limit accepted policy actions without treating a hold as an action.

    The guard is keyed by the point cloud that actually enters the model, not
    merely the latest provider frame.  Querying whether a tick must hold is
    side-effect free; the accepted-use counter advances only after a complete
    post-forward proposal has passed every freshness/deadline check.
    """

    def __init__(
        self, maximum_actions_per_frame: int = MAX_POLICY_ACTIONS_PER_CAMERA_FRAME
    ) -> None:
        maximum = int(maximum_actions_per_frame)
        if maximum <= 0:
            raise ValueError("maximum policy actions per frame must be positive")
        self.maximum_actions_per_frame = maximum
        self.last_accepted_frame_id: Optional[int] = None
        self.accepted_uses_for_last_frame = 0
        self.accepted_action_count = 0
        self.hold_count = 0

    def accepted_uses(self, frame_id: int) -> int:
        current = int(frame_id)
        if current < 0:
            raise ValueError("point-cloud frame_id must be non-negative")
        if self.last_accepted_frame_id is None or current > self.last_accepted_frame_id:
            return 0
        if current < self.last_accepted_frame_id:
            raise RuntimeError(
                "point-cloud frame_id regressed behind the last accepted action: "
                f"current={current}, last={self.last_accepted_frame_id}"
            )
        return self.accepted_uses_for_last_frame

    def must_hold(self, frame_id: int) -> bool:
        return self.accepted_uses(frame_id) >= self.maximum_actions_per_frame

    def accept(self, frame_id: int) -> int:
        current = int(frame_id)
        uses = self.accepted_uses(current)
        if uses >= self.maximum_actions_per_frame:
            raise RuntimeError(
                "policy action would exceed the per-camera-frame reuse limit: "
                f"frame_id={current}, accepted_uses={uses}, "
                f"limit={self.maximum_actions_per_frame}"
            )
        if self.last_accepted_frame_id != current:
            self.last_accepted_frame_id = current
            self.accepted_uses_for_last_frame = 0
        self.accepted_uses_for_last_frame += 1
        self.accepted_action_count += 1
        return self.accepted_uses_for_last_frame

    def record_hold(self, frame_id: int) -> int:
        current = int(frame_id)
        uses = self.accepted_uses(current)
        if uses != self.maximum_actions_per_frame:
            raise RuntimeError(
                "camera reuse hold requires an exactly exhausted action budget: "
                f"frame_id={current}, accepted_uses={uses}, "
                f"limit={self.maximum_actions_per_frame}"
            )
        self.hold_count += 1
        return uses


def _camera_rejection_diagnostic(
    camera: _CameraSample,
    *,
    observed_at_s: float,
    reason: str,
    blocked_frame_id_before: Optional[int],
    previous_diagnostic_frame_id: Optional[int],
    candidate_capture_s: Optional[float] = None,
    pointcloud_capture_s: Optional[float] = None,
    required_action_compute_reserve_s: Optional[float] = None,
    freshness_budget_remaining_s: Optional[float] = None,
    observation_action_compute_s: Optional[float] = None,
) -> dict[str, Any]:
    """Build a no-pickle timing record for a rejected/held camera tick.

    The split timing fields distinguish an unchanged provider frame from a
    changing stream whose new frames already exceed the 100 ms budget during
    capture/retrieval/publication.
    """

    observed = float(observed_at_s)
    capture = float(camera.timestamp_s)
    retrieved = float(camera.retrieved_at_s)
    published = float(camera.published_at_s)
    candidate = capture if candidate_capture_s is None else float(candidate_capture_s)
    point_capture = (
        float("nan") if pointcloud_capture_s is None else float(pointcloud_capture_s)
    )
    required_reserve = (
        float("nan")
        if required_action_compute_reserve_s is None
        else float(required_action_compute_reserve_s)
    )
    budget_remaining = (
        float("nan")
        if freshness_budget_remaining_s is None
        else float(freshness_budget_remaining_s)
    )
    observation_action_compute = (
        float("nan")
        if observation_action_compute_s is None
        else float(observation_action_compute_s)
    )
    finite_values = (observed, capture, retrieved, published, candidate)
    if not all(np.isfinite(value) for value in finite_values):
        raise ValueError("camera rejection timing values must be finite")
    for name, provided, value in (
        (
            "required action compute reserve",
            required_action_compute_reserve_s,
            required_reserve,
        ),
        (
            "freshness budget remaining",
            freshness_budget_remaining_s,
            budget_remaining,
        ),
        (
            "observation-to-action compute",
            observation_action_compute_s,
            observation_action_compute,
        ),
    ):
        if provided is not None and not np.isfinite(value):
            raise ValueError(f"camera rejection {name} must be finite")
        if (
            provided is not None
            and name != "freshness budget remaining"
            and value < 0.0
        ):
            raise ValueError(f"camera rejection {name} cannot be negative")
    current = int(camera.frame_id)
    blocked = -1 if blocked_frame_id_before is None else int(blocked_frame_id_before)
    previous = (
        -1
        if previous_diagnostic_frame_id is None
        else int(previous_diagnostic_frame_id)
    )
    return {
        "reason": str(reason),
        "camera_frame_id": current,
        "camera_sensor_frame_number": int(camera.sensor_frame_number),
        "camera_depth_sensor_frame_number": int(camera.depth_sensor_frame_number),
        "camera_mask_valid": bool(camera.mask_valid),
        "camera_message": str(camera.message),
        "object_mask_area_px": int(camera.object_mask_area_px),
        "object_mask_bbox_xyxy": np.asarray(
            camera.object_mask_bbox_xyxy, dtype=np.int32
        ).reshape(4).copy(),
        "provider_timings_ms": np.asarray(
            camera.provider_timings_ms, dtype=np.float64
        ).reshape(6).copy(),
        "provider_output_mode": str(
            getattr(camera, "provider_output_mode", "mask_only")
        ),
        "blocked_frame_id_before": blocked,
        "same_frame_as_blocked": bool(blocked >= 0 and current == blocked),
        "new_frame_since_previous_diagnostic": bool(
            previous >= 0 and current != previous
        ),
        "previous_diagnostic_frame_id": previous,
        "observed_at_s": observed,
        "camera_capture_timestamp_s": capture,
        "camera_retrieved_at_s": retrieved,
        "camera_published_at_s": published,
        "capture_to_retrieval_s": retrieved - capture,
        "retrieval_to_publication_s": published - retrieved,
        "capture_to_publication_s": published - capture,
        "capture_to_observation_s": observed - capture,
        "publication_to_observation_s": observed - published,
        "candidate_capture_timestamp_s": candidate,
        "candidate_capture_to_observation_s": observed - candidate,
        "pointcloud_capture_timestamp_s": point_capture,
        "pointcloud_capture_to_observation_s": (
            float("nan") if not np.isfinite(point_capture) else observed - point_capture
        ),
        "required_action_compute_reserve_s": required_reserve,
        "freshness_budget_remaining_s": budget_remaining,
        "observation_action_compute_s": observation_action_compute,
    }


def _stack_camera_rejection_diagnostics(
    records: Sequence[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Stack optional rejection records without object/pickle arrays."""

    payload: dict[str, np.ndarray] = {
        "camera_rejection_diagnostic_schema_version": np.asarray(2, dtype=np.int32),
        "camera_rejection_diagnostic_count": np.asarray(len(records), dtype=np.int64),
    }
    if not records:
        return payload
    expected_keys = set(records[0])
    for index, record in enumerate(records):
        if set(record) != expected_keys:
            raise ValueError(
                f"camera rejection diagnostic {index} has inconsistent fields"
            )
    for key in sorted(expected_keys):
        values = [np.asarray(record[key]) for record in records]
        try:
            stacked = np.stack(values)
        except ValueError as exc:
            raise ValueError(
                f"camera rejection diagnostic field {key} has inconsistent shapes"
            ) from exc
        if stacked.dtype == object:
            raise ValueError(
                f"camera rejection diagnostic field {key} cannot use object dtype"
            )
        payload[f"camera_rejection_{key}"] = stacked
    return payload


def _policy_reuse_hold_diagnostic(
    camera: _CameraSample,
    *,
    hold_index: int,
    pointcloud_frame_id: int,
    pointcloud_capture_timestamp_s: float,
    pointcloud_retrieved_at_s: float,
    pointcloud_published_at_s: float,
    observation_clock: _HostClockSample,
    hold_clock: _HostClockSample,
    timing: _DeadlineMetrics,
    previous_action_tick_started_monotonic_s: float,
    camera_is_new: bool,
    action_uses_before_hold: int,
    maximum_policy_actions_per_camera_frame: int,
) -> dict[str, Any]:
    """Build evidence for an eligible tick intentionally held before policy state.

    This record is deliberately separate from the action audit.  In particular,
    none of its rows can be mistaken for a model output or an actuator proposal.
    """

    index = int(hold_index)
    point_frame = int(pointcloud_frame_id)
    uses = int(action_uses_before_hold)
    maximum = int(maximum_policy_actions_per_camera_frame)
    previous_action_tick = float(previous_action_tick_started_monotonic_s)
    point_capture = float(pointcloud_capture_timestamp_s)
    point_retrieved = float(pointcloud_retrieved_at_s)
    point_published = float(pointcloud_published_at_s)
    finite_values = (
        point_capture,
        point_retrieved,
        point_published,
        previous_action_tick,
        observation_clock.realtime_s,
        observation_clock.monotonic_s,
        hold_clock.realtime_s,
        hold_clock.monotonic_s,
        timing.scheduled_tick_monotonic_s,
        timing.tick_started_monotonic_s,
        timing.tick_finished_monotonic_s,
    )
    if index < 0 or point_frame < 0:
        raise ValueError("policy reuse hold IDs must be non-negative")
    if not all(np.isfinite(value) for value in finite_values):
        raise ValueError("policy reuse hold timing values must be finite")
    if maximum <= 0 or uses != maximum:
        raise ValueError(
            "policy reuse hold must occur at the exact configured action limit"
        )
    interval_from_previous_action = (
        timing.tick_started_monotonic_s - previous_action_tick
    )
    if interval_from_previous_action < 0.0:
        raise ValueError("policy reuse hold preceded its previous accepted action")

    return {
        "diagnostic_index": index,
        "reason": POLICY_REUSE_HOLD_REASON,
        "camera_frame_id": int(camera.frame_id),
        "pointcloud_frame_id": point_frame,
        "camera_new_frame": bool(camera_is_new),
        "camera_capture_timestamp_s": float(camera.timestamp_s),
        "camera_retrieved_at_s": float(camera.retrieved_at_s),
        "camera_published_at_s": float(camera.published_at_s),
        "pointcloud_capture_timestamp_s": point_capture,
        "pointcloud_retrieved_at_s": point_retrieved,
        "pointcloud_published_at_s": point_published,
        "host_observation_realtime_s": float(observation_clock.realtime_s),
        "host_observation_monotonic_s": float(observation_clock.monotonic_s),
        "host_hold_realtime_s": float(hold_clock.realtime_s),
        "host_hold_monotonic_s": float(hold_clock.monotonic_s),
        "host_hold_clock_pair_span_s": float(hold_clock.pair_span_s),
        "control_scheduled_monotonic_s": timing.scheduled_tick_monotonic_s,
        "control_tick_started_monotonic_s": timing.tick_started_monotonic_s,
        "control_tick_finished_monotonic_s": timing.tick_finished_monotonic_s,
        "control_start_lateness_s": timing.start_lateness_s,
        "control_computation_s": timing.computation_s,
        "control_next_deadline_overrun_s": timing.next_deadline_overrun_s,
        "control_tick_interval_from_previous_action_s": (
            interval_from_previous_action
        ),
        "pointcloud_age_at_hold_s": hold_clock.realtime_s - point_capture,
        "pointcloud_capture_to_hold_s": hold_clock.realtime_s - point_capture,
        "pointcloud_publish_to_hold_s": hold_clock.realtime_s - point_published,
        "action_uses_before_hold": uses,
        "maximum_policy_actions_per_camera_frame": maximum,
        "new_policy_action_evidence": False,
        "policy_state_mutated": False,
        "write": False,
        "waits_for_new_camera_frame": True,
    }


def _stack_policy_reuse_hold_diagnostics(
    records: Sequence[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Stack optional third-tick hold evidence without object/pickle arrays."""

    payload: dict[str, np.ndarray] = {
        "policy_hold_schema_version": np.asarray(
            POLICY_REUSE_HOLD_SCHEMA_VERSION, dtype=np.int32
        ),
        "policy_hold_count": np.asarray(len(records), dtype=np.int64),
    }
    if not records:
        return payload
    expected_keys = set(records[0])
    for index, record in enumerate(records):
        if set(record) != expected_keys:
            raise ValueError(f"policy reuse hold {index} has inconsistent fields")
    for key in sorted(expected_keys):
        values = [np.asarray(record[key]) for record in records]
        try:
            stacked = np.stack(values)
        except ValueError as exc:
            raise ValueError(
                f"policy reuse hold field {key} has inconsistent shapes"
            ) from exc
        if stacked.dtype == object:
            raise ValueError(f"policy reuse hold field {key} cannot use object dtype")
        payload[f"policy_hold_{key}"] = stacked
    return payload


def _record_camera_rejection_diagnostic(
    records: list[dict[str, Any]],
    camera: _CameraSample,
    *,
    observed_at_s: float,
    reason: str,
    blocked_frame_id_before: Optional[int],
    candidate_capture_s: Optional[float] = None,
    pointcloud_capture_s: Optional[float] = None,
    required_action_compute_reserve_s: Optional[float] = None,
    freshness_budget_remaining_s: Optional[float] = None,
    observation_action_compute_s: Optional[float] = None,
    quiet: bool = False,
    print_every_n: int = CAMERA_REJECTION_LOG_EVERY_N,
) -> dict[str, Any]:
    previous_record = None if not records else records[-1]
    previous_frame_id = (
        None if previous_record is None else int(previous_record["camera_frame_id"])
    )
    record = _camera_rejection_diagnostic(
        camera,
        observed_at_s=observed_at_s,
        reason=reason,
        blocked_frame_id_before=blocked_frame_id_before,
        previous_diagnostic_frame_id=previous_frame_id,
        candidate_capture_s=candidate_capture_s,
        pointcloud_capture_s=pointcloud_capture_s,
        required_action_compute_reserve_s=required_action_compute_reserve_s,
        freshness_budget_remaining_s=freshness_budget_remaining_s,
        observation_action_compute_s=observation_action_compute_s,
    )
    record["diagnostic_index"] = len(records)
    if record["same_frame_as_blocked"]:
        stream_classification = "same_provider_frame_wait"
    elif record["new_frame_since_previous_diagnostic"]:
        stream_classification = "new_provider_frame_rejected"
    else:
        stream_classification = "first_provider_frame_rejected"
    record["stream_classification"] = stream_classification
    records.append(record)
    interval = int(print_every_n)
    if interval <= 0:
        raise ValueError("camera rejection print interval must be positive")
    classification_changed = bool(
        previous_record is not None
        and (
            str(previous_record["reason"]) != str(record["reason"])
            or str(previous_record["stream_classification"])
            != stream_classification
        )
    )
    should_print = bool(
        not quiet
        and (
            record["diagnostic_index"] == 0
            or classification_changed
            or record["diagnostic_index"] % interval == 0
        )
    )
    console = {
        "diagnostic_index": record["diagnostic_index"],
        "reason": record["reason"],
        "stream_classification": stream_classification,
        "camera_frame_id": record["camera_frame_id"],
        "sensor_frame_number": record["camera_sensor_frame_number"],
        "camera_mask_valid": record["camera_mask_valid"],
        "camera_message": record["camera_message"],
        "object_mask_area_px": record["object_mask_area_px"],
        "object_mask_bbox_xyxy": np.asarray(
            record["object_mask_bbox_xyxy"], dtype=np.int32
        ).tolist(),
        "same_frame_as_blocked": record["same_frame_as_blocked"],
        "capture_to_retrieval_s": record["capture_to_retrieval_s"],
        "retrieval_to_publication_s": record["retrieval_to_publication_s"],
        "capture_to_publication_s": record["capture_to_publication_s"],
        "capture_to_observation_s": record["capture_to_observation_s"],
        "candidate_capture_to_observation_s": record[
            "candidate_capture_to_observation_s"
        ],
    }
    if should_print:
        print(
            "[read-only camera hold] " + json.dumps(console, sort_keys=True),
            file=sys.stderr,
        )
    return record


def _reset_policy_continuity(
    *, history: PolicyHistory, velocity_tracker: KinematicVelocityTracker
) -> None:
    """Clear the two mutable logical-observation continuity owners."""

    history.clear()
    velocity_tracker.reset()


class _CameraFrameWatchdog:
    """Fail if the provider stops publishing changing D435 frame IDs.

    Camera capture age and provider liveness are intentionally separate.  A
    frame just beyond the action-age limit is a transient hold condition; an
    unchanged frame ID for a bounded monotonic interval is a hard dropout.
    """

    def __init__(self) -> None:
        self.last_frame_id: Optional[int] = None
        self.last_change_monotonic_s: Optional[float] = None
        self.last_observed_monotonic_s: Optional[float] = None
        self.frame_changes = 0
        self.reuse_observations = 0
        self.maximum_unchanged_s = 0.0

    def observe(
        self,
        frame_id: int,
        *,
        monotonic_s: float,
        maximum_unchanged_s: float,
        provider_diagnostic: str = "",
    ) -> None:
        observed = float(monotonic_s)
        limit = float(maximum_unchanged_s)
        if not np.isfinite(observed) or not np.isfinite(limit) or limit <= 0.0:
            raise ValueError("D435 frame watchdog inputs are invalid")
        if (
            self.last_observed_monotonic_s is not None
            and observed < self.last_observed_monotonic_s
        ):
            raise RuntimeError("D435 frame watchdog monotonic clock regressed")
        self.last_observed_monotonic_s = observed

        current = int(frame_id)
        if self.last_frame_id is None or current != self.last_frame_id:
            self.last_frame_id = current
            self.last_change_monotonic_s = observed
            self.frame_changes += 1
            return
        if self.last_change_monotonic_s is None:
            raise RuntimeError("D435 frame watchdog has no change timestamp")
        self.reuse_observations += 1
        unchanged_s = observed - self.last_change_monotonic_s
        self.maximum_unchanged_s = max(self.maximum_unchanged_s, unchanged_s)
        if unchanged_s > limit:
            detail = str(provider_diagnostic).strip()
            raise RuntimeError(
                "camera provider hard dropout (publication): frame_id="
                f"{current} remained unchanged for {unchanged_s:.6f}s; "
                f"limit={limit:.6f}s"
                + ("; " + detail if detail else "")
            )


def _provider_progress_diagnostic(provider: Any, *, monotonic_s: float) -> str:
    """Describe where the camera provider is currently spending time.

    This heartbeat is diagnostic only.  It cannot make a frame valid or alter
    a watchdog deadline, and tolerates providers/test doubles that predate the
    staged heartbeat.
    """

    stage = str(getattr(provider, "last_pipeline_stage", "unknown"))
    frame_id = getattr(provider, "last_pipeline_frame_id", None)
    started = getattr(
        provider, "last_pipeline_stage_started_monotonic_s", None
    )
    age_text = "unknown"
    try:
        age = float(monotonic_s) - float(started)
        if np.isfinite(age) and age >= 0.0:
            age_text = f"{age:.6f}s"
    except (TypeError, ValueError):
        pass
    return (
        f"provider_stage={stage}; provider_stage_frame_id={frame_id!r}; "
        f"provider_stage_age={age_text}"
    )


def _calibrated_action_compute_reserve_s(
    *,
    minimum_reserve_s: float,
    warmup_observation_action_s: float,
    warmup_factor: float = ACTION_COMPUTE_RESERVE_WARMUP_FACTOR,
) -> float:
    """Return a conservative reserve for the full observation-to-action path.

    ``warmup_observation_action_s`` is measured from the freshness decision
    immediately before projection through projection, kinematics, observation
    construction/history, policy inference, target mapping, and the final
    dual-clock sample.  The configured floor protects against timer noise and
    unusually fast warm-up; the multiplier provides runtime jitter margin.
    There is deliberately no cap: an unexpectedly slow complete path must fail
    closed instead of silently weakening the camera freshness contract.
    """

    minimum = float(minimum_reserve_s)
    warmup = float(warmup_observation_action_s)
    factor = float(warmup_factor)
    if not all(np.isfinite(value) for value in (minimum, warmup, factor)):
        raise ValueError("action compute reserve inputs must be finite")
    if minimum <= 0.0 or warmup < 0.0 or factor < 1.0:
        raise ValueError("action compute reserve inputs are invalid")
    return max(minimum, factor * warmup)


def _freshness_budget_metrics(
    *,
    pointcloud_age_s: float,
    maximum_age_s: float,
    required_reserve_s: float,
) -> _FreshnessBudgetMetrics:
    age = float(pointcloud_age_s)
    maximum = float(maximum_age_s)
    reserve = float(required_reserve_s)
    if not all(np.isfinite(value) for value in (age, maximum, reserve)):
        raise ValueError("point-cloud freshness budget values must be finite")
    if maximum <= 0.0 or reserve <= 0.0:
        raise ValueError("point-cloud freshness budget limits must be positive")
    remaining = maximum - age
    return _FreshnessBudgetMetrics(
        pointcloud_age_s=age,
        remaining_s=remaining,
        required_reserve_s=reserve,
        eligible=remaining >= reserve,
    )


class _ComputeThreadGuard:
    """Apply explicit, process-scoped OpenCV/BLAS limits and restore them.

    NumPy is necessarily imported before CLI parsing, so changing
    OPENBLAS_NUM_THREADS here would be misleading and too late.  threadpoolctl
    changes the already-loaded BLAS pool at runtime; its context restores the
    prior setting.  OpenCV exposes a separate process-global setting, which is
    also saved, verified, and restored.
    """

    def __init__(self, requested_threads: int) -> None:
        requested = int(requested_threads)
        if requested <= 0:
            raise ValueError("compute_threads must be positive")
        self.requested_threads = requested
        self.opencv_threads_before: Optional[int] = None
        self.opencv_threads_active: Optional[int] = None
        self.blas_pools_before: list[dict[str, Any]] = []
        self.blas_pools_active: list[dict[str, Any]] = []
        self._cv2: Any = None
        self._threadpool_info: Any = None
        self._limiter: Any = None
        self._active = False

    @staticmethod
    def _compact_blas_pools(values: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "internal_api": str(value.get("internal_api", "")),
                "prefix": str(value.get("prefix", "")),
                "num_threads": int(value.get("num_threads", -1)),
                "threading_layer": str(value.get("threading_layer", "")),
                "version": str(value.get("version", "")),
            }
            for value in values
            if value.get("user_api") == "blas"
        ]

    def start(self) -> "_ComputeThreadGuard":
        if self._active:
            raise RuntimeError("compute thread guard is already active")
        try:
            import cv2
            from threadpoolctl import threadpool_info, threadpool_limits
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV and threadpoolctl are required for verified CPU thread limits"
            ) from exc

        self._cv2 = cv2
        self._threadpool_info = threadpool_info
        self.opencv_threads_before = int(cv2.getNumThreads())
        self.blas_pools_before = self._compact_blas_pools(threadpool_info())
        self._limiter = threadpool_limits(
            limits=self.requested_threads, user_api="blas"
        )
        self._limiter.__enter__()
        try:
            cv2.setNumThreads(self.requested_threads)
            self.opencv_threads_active = int(cv2.getNumThreads())
            self.blas_pools_active = self._compact_blas_pools(threadpool_info())
            if self.opencv_threads_active != self.requested_threads:
                raise RuntimeError(
                    "OpenCV rejected the requested thread limit: "
                    f"requested={self.requested_threads}, "
                    f"active={self.opencv_threads_active}"
                )
            if not self.blas_pools_active:
                raise RuntimeError("no loaded BLAS pool was available for verification")
            mismatched = [
                pool
                for pool in self.blas_pools_active
                if pool["num_threads"] != self.requested_threads
            ]
            if mismatched:
                raise RuntimeError(
                    "BLAS thread limit verification failed: "
                    + json.dumps(mismatched, sort_keys=True)
                )
            self._active = True
            return self
        except BaseException:
            try:
                cv2.setNumThreads(self.opencv_threads_before)
            finally:
                self._limiter.__exit__(None, None, None)
                self._limiter = None
            raise

    def close(self) -> None:
        if not self._active:
            return
        errors = []
        try:
            self._cv2.setNumThreads(self.opencv_threads_before)
            restored = int(self._cv2.getNumThreads())
            if restored != self.opencv_threads_before:
                errors.append(
                    "OpenCV thread restore mismatch: "
                    f"expected={self.opencv_threads_before}, actual={restored}"
                )
        except BaseException as exc:
            errors.append(f"OpenCV thread restore failed: {type(exc).__name__}: {exc}")
        try:
            self._limiter.__exit__(None, None, None)
        except BaseException as exc:
            errors.append(f"BLAS thread restore failed: {type(exc).__name__}: {exc}")
        finally:
            self._limiter = None
            self._active = False
        if errors:
            raise RuntimeError("; ".join(errors))

    @property
    def blas_pools_before_json(self) -> str:
        return json.dumps(self.blas_pools_before, sort_keys=True)

    @property
    def blas_pools_active_json(self) -> str:
        return json.dumps(self.blas_pools_active, sort_keys=True)


class _HostClockGuard:
    """Fail closed if the epoch clock jumps relative to the monotonic clock."""

    def __init__(
        self,
        *,
        maximum_offset_jump_s: float = MAX_HOST_CLOCK_OFFSET_JUMP_S,
        start_realtime_s: Optional[float] = None,
        start_monotonic_s: Optional[float] = None,
    ) -> None:
        limit = float(maximum_offset_jump_s)
        if not np.isfinite(limit) or limit <= 0.0:
            raise ValueError(
                "maximum host clock offset jump must be finite and positive"
            )
        if (start_realtime_s is None) != (start_monotonic_s is None):
            raise ValueError("host clock baseline requires both realtime and monotonic")
        if start_realtime_s is None:
            start_realtime_s, start_monotonic_s, start_pair_span_s = self._read_pair()
        else:
            start_pair_span_s = 0.0
        realtime = float(start_realtime_s)
        monotonic = float(start_monotonic_s)
        if not np.isfinite(realtime) or not np.isfinite(monotonic):
            raise ValueError("host clock baseline must be finite")
        self.maximum_offset_jump_s = limit
        self.start_realtime_s = realtime
        self.start_monotonic_s = monotonic
        self.start_pair_span_s = float(start_pair_span_s)
        self.start_offset_s = realtime - monotonic
        self._previous_monotonic_s = monotonic
        self._previous_offset_s = self.start_offset_s

    @staticmethod
    def _read_pair() -> Tuple[float, float, float]:
        # Sandwich realtime between monotonic samples.  A scheduling pause in a
        # simple time.time() -> time.monotonic() pair can make sensor age look
        # younger by almost the whole pause.  Retry a bounded number of times,
        # then return a conservative realtime upper bound at monotonic_after.
        best_span_s = float("inf")
        for _ in range(HOST_CLOCK_PAIR_READ_ATTEMPTS):
            monotonic_before_s = float(time.monotonic())
            realtime_s = float(time.time())
            monotonic_after_s = float(time.monotonic())
            pair_span_s = monotonic_after_s - monotonic_before_s
            if pair_span_s < 0.0:
                raise RuntimeError(
                    "host realtime discontinuity: monotonic clock regressed "
                    "during paired sampling"
                )
            best_span_s = min(best_span_s, pair_span_s)
            if pair_span_s <= MAX_HOST_CLOCK_PAIR_SPAN_S:
                return (
                    realtime_s + pair_span_s,
                    monotonic_after_s,
                    pair_span_s,
                )
        raise RuntimeError(
            "host realtime discontinuity: paired clock sampling span exceeded "
            f"{MAX_HOST_CLOCK_PAIR_SPAN_S:.6f}s after "
            f"{HOST_CLOCK_PAIR_READ_ATTEMPTS} attempts "
            f"(best={best_span_s:.6f}s)"
        )

    def sample(
        self,
        *,
        realtime_s: Optional[float] = None,
        monotonic_s: Optional[float] = None,
    ) -> _HostClockSample:
        if (realtime_s is None) != (monotonic_s is None):
            raise ValueError("host clock sample requires both realtime and monotonic")
        if realtime_s is None:
            realtime_s, monotonic_s, pair_span_s = self._read_pair()
        else:
            pair_span_s = 0.0
        realtime = float(realtime_s)
        monotonic = float(monotonic_s)
        if not np.isfinite(realtime) or not np.isfinite(monotonic):
            raise RuntimeError("host realtime discontinuity: non-finite clock sample")
        if monotonic < self._previous_monotonic_s:
            raise RuntimeError(
                "host realtime discontinuity: monotonic clock regressed "
                f"by {self._previous_monotonic_s - monotonic:.6f}s"
            )
        offset = realtime - monotonic
        from_start = offset - self.start_offset_s
        from_previous = offset - self._previous_offset_s
        if (
            abs(from_start) > self.maximum_offset_jump_s
            or abs(from_previous) > self.maximum_offset_jump_s
        ):
            raise RuntimeError(
                "host realtime discontinuity: time.time()-time.monotonic() "
                "offset changed "
                f"from_start={from_start:+.6f}s, "
                f"from_previous={from_previous:+.6f}s, "
                f"limit={self.maximum_offset_jump_s:.6f}s"
            )
        self._previous_monotonic_s = monotonic
        self._previous_offset_s = offset
        return _HostClockSample(
            realtime_s=realtime,
            monotonic_s=monotonic,
            pair_span_s=float(pair_span_s),
            realtime_minus_monotonic_s=offset,
            offset_delta_from_start_s=from_start,
            offset_delta_from_previous_s=from_previous,
        )


class _Latest:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._value: Any = None
        self._error: Optional[BaseException] = None
        # Camera capture timeouts are explicitly retryable in the formal
        # mask-only producer.  Retain a constant-space account of those
        # retries instead of swallowing the only evidence that distinguishes
        # a USB/SDK transport stall from a tracker/mask rejection.
        self._publication_count = 0
        self._transient_error_count = 0
        self._consecutive_transient_error_count = 0
        self._transient_episode_started_monotonic_s: Optional[float] = None
        self._last_transient_error_monotonic_s: Optional[float] = None
        self._last_transient_error = ""

    def put(self, value: Any) -> None:
        with self._condition:
            self._value = value
            self._publication_count += 1
            self._consecutive_transient_error_count = 0
            self._transient_episode_started_monotonic_s = None
            self._condition.notify_all()

    def note_transient(self, error: BaseException) -> None:
        """Record one retryable producer failure without latching a fault."""

        observed = time.monotonic()
        detail = " ".join(
            f"{type(error).__name__}: {error}".split()
        )[:2000]
        with self._condition:
            if self._consecutive_transient_error_count == 0:
                self._transient_episode_started_monotonic_s = observed
            self._transient_error_count += 1
            self._consecutive_transient_error_count += 1
            self._last_transient_error_monotonic_s = observed
            self._last_transient_error = detail
            self._condition.notify_all()

    def fail(self, error: BaseException) -> None:
        with self._condition:
            if self._error is None:
                self._error = error
                self._condition.notify_all()

    def get(self) -> Any:
        with self._condition:
            if self._error is not None:
                raise RuntimeError(f"background reader failed: {self._error}")
            return self._value

    def raise_if_failed(self) -> None:
        with self._condition:
            if self._error is not None:
                raise RuntimeError(f"background reader failed: {self._error}")

    def diagnostics_snapshot(self) -> dict[str, Any]:
        """Return detached producer evidence without touching the SDK."""

        with self._condition:
            return {
                "publication_count": int(self._publication_count),
                "transient_error_count": int(self._transient_error_count),
                "consecutive_transient_error_count": int(
                    self._consecutive_transient_error_count
                ),
                "transient_episode_started_monotonic_s": (
                    self._transient_episode_started_monotonic_s
                ),
                "last_transient_error_monotonic_s": (
                    self._last_transient_error_monotonic_s
                ),
                "last_transient_error": str(self._last_transient_error),
                "terminal_error": (
                    None
                    if self._error is None
                    else " ".join(
                        f"{type(self._error).__name__}: {self._error}".split()
                    )[:2000]
                ),
            }

    def wait_for_frame_change(self, frame_id: int, *, timeout_s: float) -> bool:
        """Wait for a different published frame ID or a reader failure.

        The predicate is checked while holding the same condition used by
        ``put`` and ``fail``, so a publication immediately before this call is
        not missed.  ``False`` means only that the bounded wait expired; callers
        must retry their normal sensor and watchdog checks rather than treating
        a timeout as a usable camera frame.
        """

        timeout = float(timeout_s)
        if not np.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("frame-change wait timeout must be finite and positive")
        previous = int(frame_id)

        def frame_changed_or_failed() -> bool:
            if self._error is not None:
                return True
            value = self._value
            return value is not None and int(value.frame_id) != previous

        with self._condition:
            ready = self._condition.wait_for(
                frame_changed_or_failed,
                timeout=timeout,
            )
            if self._error is not None:
                raise RuntimeError(f"background reader failed: {self._error}")
            return bool(ready)


def _wait_for_camera_replacement(
    camera_latest: _Latest,
    *,
    blocked_frame_id: int,
    control_dt_s: float,
    maximum_dropout_s: float,
) -> bool:
    """Phase-align a held loop to the next camera publication.

    Two policy periods cover one nominal 30 Hz camera interval.  The hard
    dropout limit remains the upper bound.  A timeout deliberately returns to
    the main loop so Franka/RH56 freshness and the camera watchdog are checked
    again; it never turns a held frame into an accepted observation.
    """

    period = float(control_dt_s)
    dropout = float(maximum_dropout_s)
    if not np.isfinite(period) or not np.isfinite(dropout):
        raise ValueError("camera replacement wait limits must be finite")
    if period <= 0.0 or dropout <= 0.0:
        raise ValueError("camera replacement wait limits must be positive")
    return camera_latest.wait_for_frame_change(
        int(blocked_frame_id),
        timeout_s=min(2.0 * period, dropout),
    )


def _deadline_metrics(
    *,
    scheduled_tick_monotonic_s: float,
    tick_started_monotonic_s: float,
    tick_finished_monotonic_s: float,
    control_dt_s: float,
) -> _DeadlineMetrics:
    scheduled = float(scheduled_tick_monotonic_s)
    started = float(tick_started_monotonic_s)
    finished = float(tick_finished_monotonic_s)
    period = float(control_dt_s)
    if not all(np.isfinite(value) for value in (scheduled, started, finished, period)):
        raise ValueError("deadline timestamps and control period must be finite")
    if period <= 0.0 or finished < started:
        raise ValueError("invalid deadline interval")
    return _DeadlineMetrics(
        scheduled_tick_monotonic_s=scheduled,
        tick_started_monotonic_s=started,
        tick_finished_monotonic_s=finished,
        start_lateness_s=max(0.0, started - scheduled),
        computation_s=finished - started,
        next_deadline_overrun_s=max(0.0, finished - (scheduled + period)),
    )


def _require_deadline(metrics: _DeadlineMetrics, *, maximum_overrun_s: float) -> None:
    limit = float(maximum_overrun_s)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("maximum_overrun_s must be finite and positive")
    worst = max(metrics.start_lateness_s, metrics.next_deadline_overrun_s)
    if worst > limit:
        raise RuntimeError(
            "policy control deadline overrun " f"{worst:.6f}s exceeds {limit:.6f}s"
        )


def _require_sensor_age(
    name: str,
    age_s: float,
    *,
    maximum_age_s: float,
    maximum_future_skew_s: float,
) -> None:
    age = float(age_s)
    maximum = float(maximum_age_s)
    future = float(maximum_future_skew_s)
    if not all(np.isfinite(value) for value in (age, maximum, future)):
        raise ValueError(f"{name} age limits must be finite")
    if maximum <= 0.0 or future < 0.0:
        raise ValueError(f"{name} age limits are invalid")
    if age < -future:
        raise RuntimeError(
            f"{name} timestamp is {-age:.6f}s in the future; "
            f"limit={future:.6f}s"
        )
    if age > maximum:
        raise RuntimeError(f"{name} observation is stale: age={age:.6f}s")


def _hold_for_transient_camera_staleness(
    *,
    gate: _FreshnessFrameGate,
    watchdog: _CameraFrameWatchdog,
    frame_id: int,
    age_s: float,
    monotonic_s: float,
    maximum_age_s: float,
    maximum_future_skew_s: float,
    maximum_dropout_s: float,
    provider_diagnostic: str = "",
) -> bool:
    """Return True when D435 compute must hold until a newer frame.

    Future/non-finite timestamps and a hard provider dropout remain fatal.
    Ordinary capture-age expiry is not a process failure: the exact frame is
    locked out so callers can skip every policy-state mutation and reset their
    periodic schedule while waiting for a replacement.
    """

    age = float(age_s)
    maximum = float(maximum_age_s)
    future = float(maximum_future_skew_s)
    if not all(np.isfinite(value) for value in (age, maximum, future)):
        raise ValueError("D435 age limits must be finite")
    if maximum <= 0.0 or future < 0.0:
        raise ValueError("D435 age limits are invalid")
    if age < -future:
        raise RuntimeError(
            f"D435 timestamp is {-age:.6f}s in the future; limit={future:.6f}s"
        )

    watchdog.observe(
        frame_id,
        monotonic_s=monotonic_s,
        maximum_unchanged_s=maximum_dropout_s,
        provider_diagnostic=provider_diagnostic,
    )
    if gate.waiting_for_newer_frame(frame_id):
        return True
    if age > maximum:
        gate.reject(frame_id, reason="stale")
        return True
    return False


def _hold_for_unusable_initial_object_observation(
    *,
    gate: _FreshnessFrameGate,
    frame_id: int,
    usable: bool,
) -> bool:
    """Lock out an unusable frame before the first point-cloud publication.

    Tracker recovery and depth holes are ordinary bounded startup conditions,
    not reasons to pass an empty observation to the policy or to fail on the
    first formal provider publication.  The existing replacement-frame gate,
    startup timeout, and camera publication watchdog remain the owners of the
    wait and its upper bounds.
    """

    if bool(usable):
        return False
    gate.reject(
        int(frame_id),
        reason="unusable_initial_observation",
        force_rebootstrap=True,
    )
    return True


def _validate_camera_contract(camera: _CameraSample, contract: V94Contract) -> None:
    if camera.camera_K.shape != (3, 3) or not np.all(np.isfinite(camera.camera_K)):
        raise RuntimeError("live D435 intrinsics are invalid")
    if not np.allclose(camera.camera_K, contract.camera_K, atol=1.0e-5, rtol=0.0):
        maximum = float(np.max(np.abs(camera.camera_K - contract.camera_K)))
        raise RuntimeError(
            "live D435 intrinsics differ from V94 contract: " f"max_abs={maximum:.9g}"
        )
    scale = float(camera.depth_scale_m_per_unit)
    if not np.isfinite(scale) or not np.isclose(
        scale, contract.depth_scale_m_per_unit, atol=1.0e-9, rtol=0.0
    ):
        raise RuntimeError(
            "live D435 depth scale differs from V94 contract: "
            f"live={scale!r}, expected={contract.depth_scale_m_per_unit!r}"
        )
    if (
        camera.distortion.shape != (5,)
        or not np.all(np.isfinite(camera.distortion))
        or not np.allclose(camera.distortion, 0.0, atol=1.0e-9, rtol=0.0)
    ):
        raise RuntimeError(
            "live D435 distortion is not the zero-coefficient V94 calibration"
        )
    if camera.timestamp_domain not in {
        "timestamp_domain.global_time",
        "timestamp_domain.system_time",
    }:
        raise RuntimeError(
            "live D435 does not provide an epoch-compatible capture timestamp"
        )
    if (
        not np.isfinite(camera.retrieved_at_s)
        or not np.isfinite(camera.color_depth_timestamp_skew_s)
        or camera.color_depth_timestamp_skew_s < 0.0
        or camera.color_depth_timestamp_skew_s
        > MAX_COLOR_DEPTH_TIMESTAMP_SKEW_S
    ):
        raise RuntimeError("live D435 RGB-D capture timestamps are not synchronized")
    if (
        not np.isfinite(camera.published_at_s)
        or camera.published_at_s < camera.retrieved_at_s - 0.05
    ):
        raise RuntimeError("live D435 provider publication timestamp is invalid")


def _stack_audit_records(records: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not records:
        raise RuntimeError("read-only preview produced zero valid policy steps")
    expected_keys = set(records[0])
    if not expected_keys:
        raise ValueError("audit record cannot be empty")
    for index, record in enumerate(records):
        if set(record) != expected_keys:
            raise ValueError(f"audit record {index} has inconsistent fields")
    payload: dict[str, np.ndarray] = {}
    for key in sorted(expected_keys):
        values = [np.asarray(record[key]) for record in records]
        try:
            stacked = np.stack(values)
        except ValueError as exc:
            raise ValueError(f"audit field {key} has inconsistent shapes") from exc
        if stacked.dtype == object:
            raise ValueError(f"audit field {key} cannot use object dtype")
        payload[key] = stacked
    payload["audit_schema_version"] = np.asarray(1, dtype=np.int32)
    # Legacy ``hardware_writes`` means robot actuator/register commands in the
    # original schema.  State the scope explicitly because starting the D435
    # does apply camera options (emitter, laser, preset, auto exposure).
    payload["hardware_writes"] = np.asarray(False)
    payload["hardware_writes_semantics"] = np.asarray(
        "robot_actuator_or_register_commands_only"
    )
    payload["robot_command_writes"] = np.asarray(False)
    payload["camera_configuration_writes"] = np.asarray(True)
    return payload


def _require_preview_complete(
    camera_latest: _Latest,
    hand_latest: _Latest,
    records: Sequence[dict[str, Any]],
) -> None:
    camera_latest.raise_if_failed()
    hand_latest.raise_if_failed()
    if not records:
        raise RuntimeError("read-only preview produced zero valid policy steps")


def _run_policy_preview_forward(
    policy: Any,
    point_history: np.ndarray,
    valid_history: np.ndarray,
    proprio_history: np.ndarray,
    *,
    accept_output: bool,
) -> Tuple[Optional[Any], float]:
    started = time.perf_counter()
    output = policy.act(point_history, valid_history, proprio_history)
    elapsed_s = time.perf_counter() - started
    return (output if accept_output else None), float(elapsed_s)


def _immutable_owned_array(value: Any, *, dtype: Any) -> np.ndarray:
    """Transfer an owned camera-stage array without a redundant full copy.

    The RealSense wrapper already copies SDK frame buffers before returning an
    ``RGBDFrame`` and ``depth_m``/the boolean mask are newly allocated arrays.
    Keep that ownership boundary explicit: third-party providers that return a
    view or non-contiguous storage are copied here.  The published result is
    read-only so the provider and policy threads cannot mutate shared pixels.
    """

    result = np.asarray(value, dtype=dtype)
    if not result.flags.owndata or not result.flags.c_contiguous:
        result = np.array(result, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _camera_worker(
    provider: Any,
    stop: threading.Event,
    latest: _Latest,
    provider_output_mode: str = "full_packet",
) -> None:
    output_mode = str(provider_output_mode)
    if output_mode not in PROVIDER_OUTPUT_MODES:
        latest.fail(
            ValueError(
                "provider_output_mode must be one of "
                f"{PROVIDER_OUTPUT_MODES}, got {output_mode!r}"
            )
        )
        stop.set()
        return
    discard_cold_provider_frame = True
    provider_step_index = 0
    try:
        while not stop.is_set():
            provider_step_index += 1
            stage = "cold_discard" if discard_cold_provider_frame else "formal"
            try:
                if output_mode == "mask_only":
                    step_mask_only = getattr(provider, "step_mask_only", None)
                    if not callable(step_mask_only):
                        raise RuntimeError(
                            "provider does not implement the reviewed "
                            "step_mask_only API"
                        )
                    frame, mask_result = step_mask_only()
                else:
                    frame, mask_result, _object, _packet = provider.step()
            except BaseException as exc:
                # A single D435 capture attempt has its own fixed 100 ms
                # deadline.  Only the production mask-only path may retry that
                # explicit timeout marker.  No sample is published and the
                # cold/formal provider stage is preserved.  The camera
                # owner's independent 0.20 s publication watchdog remains the
                # absolute liveness bound; the runtime then requests both
                # verified robot stop paths and closes this worker.
                if (
                    output_mode == "mask_only"
                    and bool(
                        getattr(exc, "retryable_camera_timeout", False)
                    )
                ):
                    note_transient = getattr(latest, "note_transient", None)
                    if callable(note_transient):
                        note_transient(exc)
                    if stop.wait(RETRYABLE_CAMERA_TIMEOUT_BACKOFF_S):
                        break
                    continue
                raise RuntimeError(
                    "object provider step failed: "
                    f"index={provider_step_index}, stage={stage}, "
                    f"output_mode={output_mode}; "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if stop.is_set():
                break
            if discard_cold_provider_frame:
                # The first provider step can include one-time tracker/model
                # work.  It is deliberately neither published nor eligible for
                # a formal policy capture.
                discard_cold_provider_frame = False
                continue
            intrinsics = frame.intrinsics
            camera_K = np.asarray(
                [
                    [intrinsics.fx, 0.0, intrinsics.ppx],
                    [0.0, intrinsics.fy, intrinsics.ppy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            color_bgr = _immutable_owned_array(
                frame.color_bgr, dtype=np.uint8
            )
            frame_depth_raw = getattr(frame, "depth_raw", None)
            if frame_depth_raw is None:
                depth_raw = None
                depth_m = _immutable_owned_array(
                    frame.depth_m, dtype=np.float32
                )
            else:
                depth_raw = _immutable_owned_array(
                    frame_depth_raw, dtype=np.uint16
                )
                depth_m = None
            object_mask = _immutable_owned_array(
                np.asarray(mask_result.mask) > 0, dtype=np.bool_
            )
            object_mask_valid = bool(mask_result.valid)
            raw_policy_semantic_mask = getattr(
                mask_result, "policy_semantic_mask", None
            )
            policy_semantic_valid = bool(
                getattr(mask_result, "policy_semantic_valid", False)
                and raw_policy_semantic_mask is not None
            )
            if policy_semantic_valid:
                policy_semantic_mask = _immutable_owned_array(
                    np.asarray(raw_policy_semantic_mask) > 0,
                    dtype=np.bool_,
                )
                if policy_semantic_mask.shape != object_mask.shape:
                    raise RuntimeError(
                        "provider policy semantic mask shape differs from RGB-D frame"
                    )
                policy_semantic_valid = bool(
                    np.count_nonzero(policy_semantic_mask) > 0
                )
            else:
                policy_semantic_mask = None
            object_mask_bbox = np.asarray(
                getattr(mask_result, "bbox_xyxy", np.zeros(4)), dtype=np.int32
            ).reshape(4)
            latest.put(
                _CameraSample(
                    color_bgr=color_bgr,
                    depth_m=depth_m,
                    mask=object_mask,
                    camera_K=camera_K,
                    distortion=np.asarray(
                        getattr(intrinsics, "distortion", ()), dtype=np.float64
                    ).copy(),
                    distortion_model=str(getattr(intrinsics, "model", "")),
                    depth_scale_m_per_unit=float(frame.depth_scale),
                    timestamp_s=float(frame.timestamp),
                    frame_id=int(frame.frame_id),
                    mask_valid=object_mask_valid,
                    message=str(mask_result.message),
                    mask_source=str(
                        getattr(
                            provider,
                            "published_mask_source",
                            getattr(
                                provider, "_last_output_mask_source", "unknown"
                            ),
                        )
                    ),
                    online_sam2_status=str(
                        getattr(
                            provider,
                            "online_sam2_status",
                            getattr(
                                provider, "_last_online_sam2_status", "unknown"
                            ),
                        )
                    ),
                    provider_timings_ms=np.asarray(
                        [
                            float(provider.last_timings_ms.get(name, 0.0))
                            for name in (
                                "camera",
                                "tracker",
                                "sam2",
                                "sam2_reinit",
                                (
                                    "mask_gate"
                                    if output_mode == "mask_only"
                                    else "pcd"
                                ),
                                "total",
                            )
                        ],
                        dtype=np.float64,
                    ),
                    retrieved_at_s=float(frame.retrieved_at_s),
                    timestamp_domain=str(frame.timestamp_domain),
                    color_depth_timestamp_skew_s=float(
                        frame.color_depth_timestamp_skew_s
                    ),
                    color_depth_epoch_timestamp_skew_s=float(
                        frame.color_depth_epoch_timestamp_skew_s
                    ),
                    rejected_timestamp_skew_frames=int(
                        frame.rejected_timestamp_skew_frames
                    ),
                    last_rejected_color_depth_skew_s=(
                        0.0
                        if frame.last_rejected_color_depth_skew_s is None
                        else float(frame.last_rejected_color_depth_skew_s)
                    ),
                    # Evaluated after the RGB/depth/mask ownership boundary
                    # above and just before publishing this immutable sample.
                    published_at_s=time.time(),
                    dropped_queued_framesets=int(frame.dropped_queued_framesets),
                    sensor_frame_number=int(frame.sensor_frame_number),
                    depth_sensor_frame_number=int(
                        getattr(frame, "depth_sensor_frame_number", 0)
                    ),
                    rejected_transport_stale_frames=int(
                        getattr(frame, "rejected_transport_stale_frames", 0)
                    ),
                    last_rejected_transport_age_s=(
                        0.0
                        if getattr(frame, "last_rejected_transport_age_s", None)
                        is None
                        else float(frame.last_rejected_transport_age_s)
                    ),
                    retrieved_monotonic_s=(
                        0.0
                        if getattr(frame, "retrieved_monotonic_s", None) is None
                        else float(frame.retrieved_monotonic_s)
                    ),
                    host_clock_pair_span_s=(
                        0.0
                        if getattr(frame, "host_clock_pair_span_s", None) is None
                        else float(frame.host_clock_pair_span_s)
                    ),
                    capture_diagnostic_json=json.dumps(
                        getattr(frame, "capture_diagnostic", None) or {},
                        sort_keys=True,
                    ),
                    object_mask_area_px=(
                        int(np.count_nonzero(object_mask))
                        if object_mask_valid
                        else 0
                    ),
                    object_mask_bbox_xyxy=(
                        object_mask_bbox.copy()
                        if object_mask_valid
                        else np.zeros(4, dtype=np.int32)
                    ),
                    depth_raw=depth_raw,
                    provider_output_mode=output_mode,
                    requested_object_mask_mode=str(
                        getattr(
                            provider,
                            "deployment_requested_object_mask_mode",
                            "unknown",
                        )
                    ),
                    effective_object_mask_mode=str(
                        getattr(
                            provider,
                            "deployment_effective_object_mask_mode",
                            "unknown",
                        )
                    ),
                    effective_mask_publication_mode=str(
                        getattr(
                            provider,
                            "deployment_effective_mask_publication_mode",
                            "unknown",
                        )
                    ),
                    effective_recovery_publication_mode=str(
                        getattr(
                            provider,
                            "deployment_effective_recovery_publication_mode",
                            "unknown",
                        )
                    ),
                    policy_semantic_mask=policy_semantic_mask,
                    policy_semantic_valid=policy_semantic_valid,
                    policy_semantic_source=str(
                        getattr(mask_result, "policy_semantic_source", "")
                    ),
                )
            )
    except BaseException as exc:
        latest.fail(exc)
        stop.set()


def _hand_worker(
    reader: InspireStateReader,
    mapper: RH56FeedbackMapper,
    stop: threading.Event,
    latest: _Latest,
    read_rate_hz: float = DEFAULT_RH56_READ_RATE_HZ,
) -> None:
    rate_hz = float(read_rate_hz)
    if not np.isfinite(rate_hz) or rate_hz <= 0.0:
        latest.fail(ValueError("RH56 read rate must be finite and positive"))
        stop.set()
        return
    period_s = 1.0 / rate_hz
    previous_q = None
    previous_time = None
    previous_poll_started_monotonic_s = None
    try:
        reader.start()
        while not stop.is_set():
            if previous_poll_started_monotonic_s is not None:
                # The compact snapshot is still a synchronous three-transaction
                # serial read.  Bound its producer rate explicitly instead of
                # spinning at the transport's maximum throughput and competing
                # continuously with the D435/60 Hz inference path.  Scheduling
                # is start-to-start and never catches up in a burst after an
                # overrun; sensor age remains independently fail-closed below.
                remaining_s = (
                    previous_poll_started_monotonic_s
                    + period_s
                    - time.monotonic()
                )
                if remaining_s > 0.0 and stop.wait(remaining_s):
                    break
            if stop.is_set():
                break
            previous_poll_started_monotonic_s = time.monotonic()
            snapshot_started = time.time()
            observation = reader.read()
            snapshot_finished = time.time()
            feedback = mapper.map(observation.angles)
            if previous_q is None:
                dq = np.zeros(6, dtype=np.float32)
            else:
                dt = observation.captured_at_s - previous_time
                if not np.isfinite(dt) or dt <= 0.0 or dt > 1.0:
                    raise RuntimeError(f"unsafe RH56 feedback interval {dt!r}s")
                dq = ((feedback.q_policy_order_rad - previous_q) / dt).astype(
                    np.float32
                )
            previous_q = feedback.q_policy_order_rad.copy()
            previous_time = observation.captured_at_s
            latest.put(
                _HandSample(
                    angle_targets=observation.angle_targets.copy(),
                    angle_act=observation.angles.copy(),
                    positions=observation.positions.copy(),
                    forces=observation.forces.copy(),
                    currents=observation.currents.copy(),
                    errors=observation.errors.copy(),
                    statuses=observation.statuses.copy(),
                    temperatures_c=observation.temperatures_c.copy(),
                    q_policy_rad=feedback.q_policy_order_rad.copy(),
                    dq_policy_rad_s=dq,
                    timestamp_s=float(observation.captured_at_s),
                    snapshot_span_s=float(snapshot_finished - snapshot_started),
                )
            )
    except BaseException as exc:
        latest.fail(exc)
        stop.set()
    finally:
        try:
            reader.close()
        except BaseException as exc:
            latest.fail(exc)
            stop.set()


def _nearest_pose(
    history: Deque[Tuple[float, np.ndarray]], timestamp_s: float, maximum_delta_s: float
) -> Tuple[np.ndarray, float]:
    if not history:
        raise RuntimeError("Franka pose history is empty")
    delta, pose = min(
        ((abs(timestamp - timestamp_s), value) for timestamp, value in history),
        key=lambda item: item[0],
    )
    if delta > maximum_delta_s:
        raise RuntimeError(
            f"camera/Franka capture-time pose skew {delta:.6f}s exceeds "
            f"{maximum_delta_s:.6f}s"
        )
    return pose.copy(), float(delta)


def _capture_pose_is_ready(
    history: Deque[Tuple[float, np.ndarray]],
    timestamp_s: float,
    maximum_delta_s: float,
) -> bool:
    """Report whether startup history can represent a camera capture safely.

    Camera and Franka readers start independently.  The first published camera
    frame can predate the first pose-history sample even though both streams are
    healthy.  Such a frame is ineligible for warm-up and must be replaced; this
    helper never widens the formal capture-time skew bound.
    """

    maximum = float(maximum_delta_s)
    timestamp = float(timestamp_s)
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("maximum pose delta must be finite and positive")
    if not np.isfinite(timestamp):
        raise ValueError("camera capture timestamp must be finite")
    if not history:
        return False
    return bool(
        min(abs(float(pose_time) - timestamp) for pose_time, _pose in history)
        <= maximum
    )


def _require_safe_bbox_initialization(provider: Any) -> None:
    """Reject unsafe initialization masks before any live worker is started.

    GrabCut needs visible background between its mask and prompt rectangle, so
    its historical prompt-margin gate remains strict.  A SAM2 box is only a
    prompt, not a crop: SAM2 is explicitly allowed to recover object pixels
    just outside that box.  SAM2 instead receives area-ratio and full-image
    boundary checks.
    """

    evidence = getattr(provider, "last_bbox_initialization_evidence", None)
    if evidence is None:
        raise RuntimeError(
            "object tracker ROI initialization provided no structured mask evidence"
        )
    source = str(getattr(evidence, "source", "")).strip().lower()
    valid = bool(getattr(evidence, "valid", False))
    mask_area = int(getattr(evidence, "mask_area_px", 0))
    mask_bbox_area = int(getattr(evidence, "mask_bbox_area_px", 0))
    prompt_area = int(getattr(evidence, "prompt_bbox_area_px", 0))
    solid_bbox_mask = bool(getattr(evidence, "solid_bbox_mask", False))
    full_prompt_box_mask = bool(
        getattr(evidence, "full_prompt_box_mask", False)
    )
    if not valid or mask_area <= 0 or mask_bbox_area <= 0 or prompt_area <= 0:
        raise RuntimeError(
            "object tracker ROI initialization mask evidence is invalid: "
            f"source={source!r}, area={mask_area}, bbox_area={mask_bbox_area}, "
            f"prompt_area={prompt_area}"
        )
    if source == "box" or source.startswith("box_fallback"):
        raise RuntimeError(
            "object tracker ROI initialization used unsafe box fallback: "
            f"source={source!r}, area={mask_area}, bbox_area={mask_bbox_area}; "
            "move/tighten the ROI around the object"
        )
    if solid_bbox_mask or full_prompt_box_mask or mask_area >= mask_bbox_area:
        raise RuntimeError(
            "object tracker ROI initialization produced an unsafe solid-box mask: "
            f"source={source!r}, area={mask_area}, bbox_area={mask_bbox_area}, "
            f"prompt_area={prompt_area}; move/tighten the ROI around the object"
        )
    prompt_bbox = np.asarray(
        getattr(evidence, "prompt_bbox_xyxy", None)
    )
    mask_bbox = np.asarray(
        getattr(evidence, "mask_bbox_xyxy", None)
    )
    if (
        prompt_bbox.shape != (4,)
        or mask_bbox.shape != (4,)
        or not np.issubdtype(prompt_bbox.dtype, np.integer)
        or not np.issubdtype(mask_bbox.dtype, np.integer)
    ):
        raise RuntimeError(
            "object tracker ROI initialization provided no integer prompt/mask "
            "bbox evidence"
        )
    px1, py1, px2, py2 = (int(value) for value in prompt_bbox.tolist())
    mx1, my1, mx2, my2 = (int(value) for value in mask_bbox.tolist())
    if source in ("sam2", "online_sam2_box"):
        sam2_cfg = getattr(provider, "cfg", {}).get("sam2", {})
        maximum_mask_ratio = float(
            sam2_cfg.get("max_mask_area_ratio", 4.0)
        )
        maximum_bbox_ratio = float(
            sam2_cfg.get("max_bbox_area_ratio", 6.0)
        )
        if (
            not np.isfinite(maximum_mask_ratio)
            or maximum_mask_ratio <= 0.0
            or not np.isfinite(maximum_bbox_ratio)
            or maximum_bbox_ratio <= 0.0
        ):
            raise RuntimeError("SAM2 initialization area-ratio limits are invalid")
        if (
            mask_area > maximum_mask_ratio * prompt_area
            or mask_bbox_area > maximum_bbox_ratio * prompt_area
        ):
            raise RuntimeError(
                "SAM2 initialization mask grew implausibly beyond its box "
                f"prompt: mask/prompt={mask_area / prompt_area:.3f}, "
                f"bbox/prompt={mask_bbox_area / prompt_area:.3f}"
            )
        camera_cfg = getattr(provider, "cfg", {}).get("camera", {})
        width = int(camera_cfg.get("width", 0))
        height = int(camera_cfg.get("height", 0))
        if width <= 0 or height <= 0:
            raise RuntimeError(
                "SAM2 initialization requires configured camera dimensions"
            )
        image_margins = (mx1, my1, width - mx2, height - my2)
        if min(image_margins) < SAM2_INITIALIZATION_MIN_IMAGE_MARGIN_PX:
            raise RuntimeError(
                "SAM2 initialization mask touches the camera image boundary: "
                f"left/top/right/bottom margins={image_margins}px, required>="
                f"{SAM2_INITIALIZATION_MIN_IMAGE_MARGIN_PX}px"
            )
        return

    margins = (mx1 - px1, my1 - py1, px2 - mx2, py2 - my2)
    if min(margins) < FIXED_ROI_INITIALIZATION_MIN_MASK_MARGIN_PX:
        raise RuntimeError(
            "object tracker initialization mask is too close to the ROI prompt "
            "boundary: "
            f"left/top/right/bottom margins={margins}px, required>="
            f"{FIXED_ROI_INITIALIZATION_MIN_MASK_MARGIN_PX}px; reselect a "
            "centered ROI with visible background on all four sides"
        )


def _camera_frame_progress_marker(
    frame: Any,
) -> Optional[Tuple[str, float]]:
    """Return the best available marker for fixed-ROI retry freshness.

    ``RGBDFrame.frame_id`` is the provider-local accepted-frame sequence and is
    therefore preferred over sensor metadata.  The timestamp is a fallback for
    reviewed camera doubles that do not expose a sequence number.  A frame with
    no usable marker remains eligible: in that case freshness cannot be proved
    and each read is conservatively treated as one initialization attempt.
    """

    for attribute in (
        "frame_id",
        "sensor_frame_number",
        "depth_sensor_frame_number",
    ):
        if not hasattr(frame, attribute):
            continue
        value = getattr(frame, attribute)
        try:
            numeric = float(value)
            integral = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if not np.isfinite(numeric) or numeric != float(integral):
            continue
        return attribute, float(integral)

    for attribute in ("timestamp", "timestamp_s"):
        if not hasattr(frame, attribute):
            continue
        try:
            numeric = float(getattr(frame, attribute))
        except (TypeError, ValueError, OverflowError):
            continue
        if np.isfinite(numeric):
            return attribute, numeric
    return None


def _start_and_initialize_provider(
    provider: Any, roi: Optional[Sequence[int]]
) -> Any:
    try:
        provider.start()
        if roi is None:
            if not provider.select_and_initialize():
                raise RuntimeError(
                    "object tracker initialization was cancelled or failed"
                )
        else:
            x, y, width, height = (int(value) for value in roi)
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                raise ValueError(
                    "--roi must be non-negative x y and positive width height"
                )
            bbox = np.asarray([x, y, x + width, y + height], dtype=np.int32)
            attempts = 0
            frame_reads = 0
            retryable_camera_timeouts = 0
            previous_marker: Optional[Tuple[str, float]] = None
            while attempts < FIXED_ROI_INITIALIZATION_MAX_ATTEMPTS:
                if frame_reads >= FIXED_ROI_INITIALIZATION_MAX_FRAME_READS:
                    raise RuntimeError(
                        "object tracker ROI initialization could not obtain "
                        "enough fresh increasing camera frames: "
                        f"completed {attempts}/"
                        f"{FIXED_ROI_INITIALIZATION_MAX_ATTEMPTS} attempts in "
                        f"{frame_reads} frame reads"
                    )
                frame_reads += 1
                try:
                    frame = provider.camera.get_frame()
                except BaseException as exc:
                    # The formal camera worker already treats the D435's
                    # explicit bounded capture-timeout marker as retryable.
                    # Apply the same rule while rebuilding the tracker from
                    # the camera-only ROI preflight: a single USB/SDK frame
                    # hole is not evidence that the saved ROI or mask is bad.
                    # Keep this startup retry count finite so a disconnected
                    # camera still fails before either actuator owner starts.
                    if not bool(
                        getattr(exc, "retryable_camera_timeout", False)
                    ):
                        raise
                    retryable_camera_timeouts += 1
                    if (
                        retryable_camera_timeouts
                        > FIXED_ROI_INITIALIZATION_MAX_RETRYABLE_CAMERA_TIMEOUTS
                    ):
                        raise RuntimeError(
                            "object tracker ROI initialization exceeded "
                            f"{FIXED_ROI_INITIALIZATION_MAX_RETRYABLE_CAMERA_TIMEOUTS} "
                            "retryable camera timeouts in "
                            f"{frame_reads} frame reads; last={type(exc).__name__}: "
                            f"{exc}"
                        ) from exc
                    time.sleep(RETRYABLE_CAMERA_TIMEOUT_BACKOFF_S)
                    continue
                marker = _camera_frame_progress_marker(frame)
                if (
                    marker is not None
                    and previous_marker is not None
                    and marker[0] == previous_marker[0]
                    and marker[1] <= previous_marker[1]
                ):
                    # A duplicate or out-of-order camera frame is not a new
                    # segmentation opportunity and must not spend an attempt.
                    continue
                if marker is not None:
                    previous_marker = marker

                attempts += 1
                if provider.initialize_from_bbox(frame, bbox):
                    break
            else:
                raise RuntimeError(
                    "object tracker ROI initialization failed after "
                    f"{FIXED_ROI_INITIALIZATION_MAX_ATTEMPTS} fresh-frame "
                    "attempts"
                )
        _require_safe_bbox_initialization(provider)
        return provider
    except BaseException as exc:
        try:
            provider.stop()
        except BaseException as stop_exc:
            raise RuntimeError(
                "object provider initialization failed and provider.stop also "
                f"failed: init={type(exc).__name__}: {exc}; "
                f"stop={type(stop_exc).__name__}: {stop_exc}"
            ) from stop_exc
        raise


OBJECT_MASK_MODE_TO_PROVIDER_PUBLICATION_MODE = {
    "guarded": "guarded_sam2_primary",
    "guarded_v2": "guarded_sam2_primary",
    "guarded_v1": "adaptive_fusion",
    "legacy": "semantic_sam2",
}

OBJECT_MASK_MODE_TO_RECOVERY_PUBLICATION_MODE = {
    # ``guarded`` remains the stable production spelling.  Explicit versioned
    # names make recorded RGB-D A/B runs reproducible without editing YAML.
    "guarded": "unified_three_evidence",
    "guarded_v2": "unified_three_evidence",
    "guarded_v1": "legacy_double_confirm",
}


def _apply_online_sam2_live_override(
    config: dict[str, Any],
    *,
    disable_online_sam2: bool,
    object_mask_mode: Optional[str] = None,
) -> _OnlineSAM2RuntimeSelection:
    """Apply non-persistent online-SAM2 deployment overrides in memory.

    ``guarded``/``guarded_v2`` select temporal SAM2 plus the unified final
    publication guard without running adaptive tracking on every frame.
    ``guarded_v1`` preserves the former sequential confirmation path and
    ``legacy`` preserves exact-frame SAM2-only publication for controlled A/B
    comparisons. ``None`` leaves the YAML selection untouched.
    """

    online_config = config.get("online_sam2")
    if not isinstance(online_config, dict):
        raise ValueError("provider online_sam2 configuration must be a mapping")
    source_enabled = bool(online_config.get("enabled", False))
    source_publication_mode = str(
        online_config.get("mask_publication_mode", "adaptive_fusion")
    ).strip()
    requested_mode = None
    if object_mask_mode is not None:
        requested_mode = str(object_mask_mode).strip().lower()
        if requested_mode not in OBJECT_MASK_MODE_TO_PROVIDER_PUBLICATION_MODE:
            raise ValueError(
                "object_mask_mode must be guarded, guarded_v2, guarded_v1, "
                "or legacy"
            )
        online_config["mask_publication_mode"] = (
            OBJECT_MASK_MODE_TO_PROVIDER_PUBLICATION_MODE[requested_mode]
        )
        recovery_mode = OBJECT_MASK_MODE_TO_RECOVERY_PUBLICATION_MODE.get(
            requested_mode
        )
        if recovery_mode is not None:
            tracker_config = config.setdefault("tracker", {})
            if not isinstance(tracker_config, dict):
                raise ValueError("provider tracker configuration must be a mapping")
            tracker_config["recovery_publication_mode"] = recovery_mode
    override_requested = bool(disable_online_sam2)
    if override_requested:
        if requested_mode == "legacy":
            raise ValueError(
                "--disable-online-sam2 is incompatible with "
                "--object-mask-mode legacy because legacy publishes the "
                "online SAM2 mask directly"
            )
        online_config["enabled"] = False
        # This switch is an adaptive-only A/B mode. Do not retain the
        # production requirement that bbox initialization use the service
        # which the same switch intentionally disables.
        online_config["require_for_bbox_init"] = False
        online_config["mask_publication_mode"] = "adaptive_fusion"
    effective_enabled = bool(online_config.get("enabled", False))
    if override_requested and effective_enabled:
        raise RuntimeError("live online-SAM2 disable override did not take effect")
    effective_publication_mode = str(
        online_config.get("mask_publication_mode", "adaptive_fusion")
    ).strip()
    effective_recovery_mode = (
        "semantic_sam2_direct"
        if effective_publication_mode == "semantic_sam2"
        else str(
            config.get("tracker", {}).get(
                "recovery_publication_mode", "legacy_double_confirm"
            )
        ).strip()
    )
    return _OnlineSAM2RuntimeSelection(
        source_config_enabled=source_enabled,
        disable_override_requested=override_requested,
        effective_config_enabled=effective_enabled,
        source_mask_publication_mode=source_publication_mode,
        effective_mask_publication_mode=effective_publication_mode,
        effective_recovery_publication_mode=effective_recovery_mode,
        requested_object_mask_mode=requested_mode,
    )


def _require_runtime_frame_timeout_contract(
    config: Mapping[str, Any],
    *,
    expected_timeout_ms: int,
) -> None:
    """Validate the formal camera timeout before constructing the provider."""

    expected = expected_timeout_ms
    if (
        isinstance(expected, (bool, np.bool_))
        or not isinstance(expected, (int, np.integer))
        or int(expected) <= 0
    ):
        raise ValueError(
            "expected_timeout_ms must be a positive integer"
        )
    camera_cfg = config.get("camera")
    if not isinstance(camera_cfg, Mapping):
        raise RuntimeError(
            "point-cloud config has no camera mapping for the formal "
            "runtime timeout contract"
        )
    actual = camera_cfg.get("runtime_frame_timeout_ms")
    if (
        isinstance(actual, (bool, np.bool_))
        or not isinstance(actual, (int, np.integer))
        or int(actual) != int(expected)
    ):
        raise RuntimeError(
            "formal D435 runtime frame timeout differs from the sealed "
            f"deployment value: actual={actual!r} "
            f"expected={int(expected)}"
        )


def _initialize_provider(
    config_path: Path,
    roi: Optional[Sequence[int]],
    *,
    disable_online_sam2: bool = False,
    object_mask_mode: Optional[str] = None,
    required_runtime_frame_timeout_ms: Optional[int] = None,
) -> Tuple[Any, _OnlineSAM2RuntimeSelection]:
    source_path = WORKSPACE_ROOT / "perception"
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))
    from dynamic_pcd.config import load_config
    from dynamic_pcd.provider.object_pcd_provider import ObjectPCDProvider

    config = load_config(str(config_path))
    if required_runtime_frame_timeout_ms is not None:
        _require_runtime_frame_timeout_contract(
            config,
            expected_timeout_ms=required_runtime_frame_timeout_ms,
        )
    online_sam2_selection = _apply_online_sam2_live_override(
        config,
        disable_online_sam2=disable_online_sam2,
        object_mask_mode=object_mask_mode,
    )
    provider = ObjectPCDProvider(config)
    # These are immutable run-selection facts, not evolving provider state.
    # Publishing them on the provider lets the existing worker attach the
    # exact requested/effective mode tuple to every immutable camera sample.
    provider.deployment_requested_object_mask_mode = str(
        online_sam2_selection.requested_object_mask_mode or "config_default"
    )
    provider.deployment_effective_object_mask_mode = str(
        online_sam2_selection.effective_object_mask_mode
    )
    provider.deployment_effective_mask_publication_mode = str(
        online_sam2_selection.effective_mask_publication_mode
    )
    provider.deployment_effective_recovery_publication_mode = str(
        online_sam2_selection.effective_recovery_publication_mode
    )
    return _start_and_initialize_provider(provider, roi), online_sam2_selection


def _shutdown_readonly_resources(
    *,
    stop: threading.Event,
    threads: Sequence[threading.Thread],
    camera_thread: Optional[threading.Thread],
    franka_reader: Any,
    provider: Any,
    join_timeout_s: float = BACKGROUND_JOIN_TIMEOUT_S,
) -> None:
    timeout = float(join_timeout_s)
    if not np.isfinite(timeout) or timeout < 0.0:
        raise ValueError("background join timeout must be finite and non-negative")
    errors = []
    stop.set()
    for thread in threads:
        try:
            thread.join(timeout=timeout)
        except BaseException as exc:
            errors.append(
                f"failed to join {thread.name}: {type(exc).__name__}: {exc}"
            )
    alive = [thread for thread in threads if thread.is_alive()]

    if franka_reader is not None:
        try:
            franka_reader.close()
        except BaseException as exc:
            errors.append(f"Franka close failed: {type(exc).__name__}: {exc}")

    camera_alive = camera_thread is not None and camera_thread.is_alive()
    if provider is not None:
        if camera_alive:
            # provider.step() and provider.stop() are not documented as safe to
            # call concurrently.  Leave process teardown to release the device
            # instead of racing a still-running step.
            errors.append(
                "camera reader is still alive inside provider.step; "
                "provider.stop was not called concurrently"
            )
        else:
            try:
                provider.stop()
            except BaseException as exc:
                errors.append(f"provider stop failed: {type(exc).__name__}: {exc}")

    if alive:
        names = ", ".join(thread.name for thread in alive)
        errors.append(f"background readers still alive after join: {names}")
    if errors:
        raise RuntimeError("read-only cleanup failed: " + "; ".join(errors))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate V94 against live read-only FR3/RH56/D435 observations; "
            "never write hardware commands."
        )
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--pcd-config", type=Path, default=DEFAULT_PCD_CONFIG)
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="effective capture duration after the first accepted policy step",
    )
    parser.add_argument(
        "--startup-timeout-s",
        type=float,
        default=10.0,
        help="bounded wait for fresh camera/hand/Franka samples before capture",
    )
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument(
        "--disable-online-sam2",
        action="store_true",
        help=(
            "live-only A/B override: do not create or schedule the temporal "
            "online-SAM2 service; keep the adaptive RGB-D tracker active"
        ),
    )
    parser.add_argument(
        "--provider-output-mode",
        choices=PROVIDER_OUTPUT_MODES,
        default="mask_only",
        help=(
            "mask_only skips the provider's duplicate 1024-point history and "
            "packet while retaining tracker/identity/recovery gates; "
            "full_packet is the diagnostic A/B baseline"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-camera-age-s", type=float, default=0.10)
    parser.add_argument(
        "--max-camera-dropout-s",
        type=float,
        default=DEFAULT_MAX_CAMERA_DROPOUT_S,
        help=(
            "hard-fail if the provider frame ID does not change for this "
            "monotonic interval; camera age expiry before that only holds"
        ),
    )
    parser.add_argument("--max-hand-age-s", type=float, default=0.50)
    parser.add_argument(
        "--rh56-read-rate-hz",
        type=float,
        default=DEFAULT_RH56_READ_RATE_HZ,
        help=(
            "rate-limit the read-only three-transaction RH56 feedback snapshot; "
            "freshness is still checked independently on every 60 Hz policy tick"
        ),
    )
    parser.add_argument("--max-pose-skew-s", type=float, default=0.025)
    parser.add_argument(
        "--min-action-compute-reserve-s",
        type=float,
        default=DEFAULT_MIN_ACTION_COMPUTE_RESERVE_S,
        help=(
            "minimum capture-to-action time reserved before projection/forward; "
            "the effective reserve is max(this, 2x the complete discarded "
            "projection-to-action warm-up)"
        ),
    )
    parser.add_argument(
        "--compute-threads",
        type=int,
        default=DEFAULT_COMPUTE_THREADS,
        help=(
            "verified process-scoped OpenCV and BLAS thread limit; previous "
            "settings are restored on exit"
        ),
    )
    parser.add_argument(
        "--max-deadline-overrun-s",
        type=float,
        default=0.020,
        help="fail if a 60 Hz tick starts or finishes this far past schedule",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="reserved; always refused before any device is opened",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress per-step JSON; the audit NPZ remains complete",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    provider = None
    franka_reader = None
    camera_thread = None
    compute_thread_guard = None
    stop = threading.Event()
    threads = []
    cleanup_complete = False
    try:
        if args.execute:
            raise RuntimeError(
                "live_v94_preview is read-only and has no hardware write path"
            )
        duration = float(args.duration)
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration must be finite and positive")
        for name in (
            "startup_timeout_s",
            "max_camera_age_s",
            "max_camera_dropout_s",
            "max_hand_age_s",
            "rh56_read_rate_hz",
            "max_pose_skew_s",
            "min_action_compute_reserve_s",
            "max_deadline_overrun_s",
        ):
            value = float(getattr(args, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if int(args.compute_threads) <= 0:
            raise ValueError("compute_threads must be positive")
        host_clock_guard = _HostClockGuard()

        # Complete offline acceptance happens before opening a camera, serial
        # port, or Franka read-only connection.
        verification = verify_v94_bundle(args.bundle)
        bundle = DeployBundle(args.bundle)
        contract = V94Contract.from_bundle(bundle)
        policy = RollingStudentPolicy(load_checkpoint_safely(bundle.checkpoint_bytes()))
        runtime_config = json.loads(
            (SIM2REAL_ROOT / "configs" / "config.json").read_text(
                encoding="utf-8"
            )
        )
        profile_path = (
            SIM2REAL_ROOT / runtime_config["commissioning_profile"]
        ).resolve()
        profile_payload = profile_path.read_bytes()
        profile = json.loads(profile_payload.decode("utf-8"))
        profile_sha256 = hashlib.sha256(profile_payload).hexdigest()
        if profile["calibration"]["id"] != contract.calibration_id:
            raise RuntimeError("commissioning and checkpoint calibration IDs disagree")
        if profile["calibration"]["camera_serial"] != contract.camera_serial:
            raise RuntimeError("commissioning and checkpoint camera serials disagree")

        provider, online_sam2_selection = _initialize_provider(
            args.pcd_config.resolve(),
            args.roi,
            disable_online_sam2=bool(args.disable_online_sam2),
        )
        online_sam2_provider_enabled = bool(
            getattr(provider, "online_sam2_enabled", False)
        )
        online_sam2_manager_created = bool(
            getattr(provider, "online_sam2_manager", None) is not None
        )
        online_sam2_executor_created = bool(
            getattr(provider, "_online_sam2_executor", None) is not None
        )
        if online_sam2_selection.disable_override_requested and (
            online_sam2_provider_enabled
            or online_sam2_manager_created
            or online_sam2_executor_created
        ):
            raise RuntimeError(
                "live online-SAM2 disable override left runtime resources active"
            )
        print(
            "[read-only perception] "
            f"online_sam2_mode={online_sam2_selection.mode}; "
            "source_config_enabled="
            f"{online_sam2_selection.source_config_enabled}; "
            "effective_config_enabled="
            f"{online_sam2_selection.effective_config_enabled}; "
            f"provider_enabled={online_sam2_provider_enabled}; "
            f"manager_created={online_sam2_manager_created}; "
            f"executor_created={online_sam2_executor_created}; "
            f"provider_output_mode={args.provider_output_mode}",
            file=sys.stderr,
        )
        # Provider initialization loads the tracker/SAM/OpenCV stack.  Apply
        # and verify runtime limits only after those libraries exist, but
        # before starting the steady-state camera/hand producer threads.
        compute_thread_guard = _ComputeThreadGuard(args.compute_threads).start()
        print(
            "[read-only compute threads] "
            f"OpenCV={compute_thread_guard.opencv_threads_active}; "
            f"BLAS={compute_thread_guard.blas_pools_active_json}; "
            "runtime-scoped limits active; BLAS environment unchanged",
            file=sys.stderr,
        )
        if provider.extrinsics.calibration_id != contract.calibration_id:
            raise RuntimeError("live provider calibration ID differs from V94")
        if provider.extrinsics.camera_serial != contract.camera_serial:
            raise RuntimeError("live provider camera serial differs from V94")
        if not np.allclose(
            np.asarray(provider.extrinsics.T_base_camera, dtype=np.float64),
            contract.T_base_camera_optical,
            # The provider intentionally stores this YAML transform as
            # float32; 1e-7 accepts only that sub-micrometre rounding while
            # still rejecting a different calibration or camera pose.
            atol=1.0e-7,
            rtol=0.0,
        ):
            raise RuntimeError("live provider T_base_camera differs from V94")

        hand_feedback_mapper = RH56FeedbackMapper(
            q_hand_close_rad=contract.q_hand_close_rad
        )
        fingertip_model = RH56FingertipKinematics(contract)
        camera_latest = _Latest()
        hand_latest = _Latest()
        hand_reader = InspireStateReader(
            port=profile["inspire"]["port"],
            baud=int(profile["inspire"]["baud"]),
            hand_id=int(profile["inspire"]["hand_id"]),
            timeout_s=float(runtime_config["inspire"]["serial_timeout_s"]),
            debug=False,
            snapshot_mode="compact_policy",
        )
        camera_thread = threading.Thread(
            target=_camera_worker,
            args=(provider, stop, camera_latest, args.provider_output_mode),
            name="v94-camera-readonly",
            daemon=True,
        )
        hand_thread = threading.Thread(
            target=_hand_worker,
            args=(
                hand_reader,
                hand_feedback_mapper,
                stop,
                hand_latest,
                float(args.rh56_read_rate_hz),
            ),
            name="v94-rh56-readonly",
            daemon=True,
        )
        for thread in (camera_thread, hand_thread):
            thread.start()
            threads.append(thread)

        franka_reader = FrankaStateReader(
            profile["franka"]["ip"], enforce_realtime=False
        )
        franka_reader.start()
        projector = MaskedRGBDProjector(
            camera_K=contract.camera_K,
            T_base_camera_optical=contract.T_base_camera_optical,
            image_size=(contract.camera_width, contract.camera_height),
            depth_range_m=contract.depth_range_m,
        )
        proprio_builder = Proprio67Builder(
            q_home_rad=contract.q_home_rad,
            q_hand_close_rad=contract.q_hand_close_rad,
        )
        velocity_tracker = KinematicVelocityTracker(maximum_dt_s=0.10)
        history = PolicyHistory()
        pose_history: Deque[Tuple[float, np.ndarray]] = deque(maxlen=64)
        point_frame = None
        point_capture_palm = None
        point_pose_skew_s = None
        point_camera_retrieved_at_s = None
        point_camera_published_at_s = None
        last_camera_id = None
        console_logs = []
        audit_records = []
        camera_rejection_diagnostics: list[dict[str, Any]] = []
        policy_reuse_hold_diagnostics: list[dict[str, Any]] = []
        startup_started = time.monotonic()
        capture_started: Optional[float] = None
        next_tick: Optional[float] = None
        previous_tick_started: Optional[float] = None
        policy_warmup_complete = False
        policy_warmup_inference_s: Optional[float] = None
        maximum_policy_warmup_inference_s = 0.0
        observation_action_warmup_s: Optional[float] = None
        maximum_observation_action_warmup_s = 0.0
        policy_warmup_count = 0
        startup_pose_alignment_wait_frames = 0
        action_compute_reserve_s: Optional[float] = None
        freshness_frame_gate = _FreshnessFrameGate()
        camera_frame_watchdog = _CameraFrameWatchdog()
        camera_policy_reuse_guard = _CameraPolicyReuseGuard()
        last_policy_state_timestamp_s: Optional[float] = None
        step = 0
        while not stop.is_set():
            loop_time = time.monotonic()
            if capture_started is None:
                if loop_time - startup_started >= float(args.startup_timeout_s):
                    break
            elif loop_time - capture_started >= duration:
                break
            tick_started = time.monotonic()
            scheduled_tick = tick_started if next_tick is None else next_tick
            if next_tick is not None:
                _require_deadline(
                    _deadline_metrics(
                        scheduled_tick_monotonic_s=scheduled_tick,
                        tick_started_monotonic_s=tick_started,
                        tick_finished_monotonic_s=tick_started,
                        control_dt_s=contract.control_dt_s,
                    ),
                    maximum_overrun_s=float(args.max_deadline_overrun_s),
                )
            franka = franka_reader.read()
            T_base_palm = T_base_policy_palm_from_franka(
                T_base_ee=franka.T_base_ee,
                F_T_EE=franka.F_T_EE,
                T_flange_policy_palm=contract.T_flange_policy_palm,
            )
            pose_history.append((franka.captured_at_s, T_base_palm.copy()))
            camera = camera_latest.get()
            hand = hand_latest.get()
            if camera is None or hand is None:
                # Startup is not a logical policy history step.  Reset the
                # periodic schedule so delayed first samples cannot trigger a
                # catch-up burst when they finally arrive.
                next_tick = None
                wait = contract.control_dt_s - (time.monotonic() - tick_started)
                if wait > 0.0:
                    time.sleep(wait)
                continue
            observation_clock = host_clock_guard.sample()
            now = observation_clock.realtime_s
            hand_age_s = now - hand.timestamp_s
            camera_age_s = now - camera.timestamp_s
            franka_age_s = now - franka.captured_at_s
            maximum_future_skew_s = float(
                runtime_config["observation"]["max_future_skew_s"]
            )
            # RH56 and Franka retain strict age semantics even while camera
            # compute is held.  A D435 dropout must never mask either device.
            _require_sensor_age(
                "RH56",
                hand_age_s,
                maximum_age_s=float(args.max_hand_age_s),
                maximum_future_skew_s=maximum_future_skew_s,
            )
            _require_sensor_age(
                "Franka",
                franka_age_s,
                maximum_age_s=float(args.max_camera_age_s),
                maximum_future_skew_s=maximum_future_skew_s,
            )
            if not np.array_equal(hand.angle_targets, np.full(6, -1)):
                raise RuntimeError(
                    "read-only preview requires verified RH56 ANGLE_SET=-1 on all axes"
                )
            if np.any(hand.errors):
                raise RuntimeError(f"RH56 reports errors: {hand.errors.tolist()}")
            if np.any(np.isin(hand.statuses, np.asarray([5, 6, 7]))):
                raise RuntimeError(
                    f"RH56 reports protection/fault status: {hand.statuses.tolist()}"
                )
            maximum_hand_temperature = int(
                runtime_config["observation"]["max_hand_temperature_c"]
            )
            if int(np.max(hand.temperatures_c)) > maximum_hand_temperature:
                raise RuntimeError(
                    "RH56 temperature exceeds read-only limit: "
                    f"{hand.temperatures_c.tolist()}"
                )
            _validate_camera_contract(camera, contract)
            if franka.robot_mode != "idle":
                raise RuntimeError(
                    f"read-only preview requires Franka idle mode, got {franka.robot_mode}"
                )
            if franka.current_errors:
                raise RuntimeError(
                    f"Franka reports active errors: {franka.current_errors}"
                )
            if float(np.max(np.abs(franka.dq))) > 0.02:
                raise RuntimeError(
                    "Franka is not stationary enough for read-only preview"
                )
            blocked_frame_id_before = freshness_frame_gate.blocked_frame_id
            if _hold_for_transient_camera_staleness(
                gate=freshness_frame_gate,
                watchdog=camera_frame_watchdog,
                frame_id=camera.frame_id,
                age_s=camera_age_s,
                monotonic_s=observation_clock.monotonic_s,
                maximum_age_s=float(args.max_camera_age_s),
                maximum_future_skew_s=maximum_future_skew_s,
                maximum_dropout_s=float(args.max_camera_dropout_s),
                provider_diagnostic=_provider_progress_diagnostic(
                    provider, monotonic_s=observation_clock.monotonic_s
                ),
            ):
                # The frame is either already locked out or has just crossed
                # the capture-age limit.  Do not project, advance velocity or
                # policy history, map an action, or create a catch-up burst.
                _record_camera_rejection_diagnostic(
                    camera_rejection_diagnostics,
                    camera,
                    observed_at_s=observation_clock.realtime_s,
                    reason=(
                        "same_frame_capture_age_wait"
                        if blocked_frame_id_before == camera.frame_id
                        else "capture_age_rejected"
                    ),
                    blocked_frame_id_before=blocked_frame_id_before,
                    quiet=bool(args.quiet),
                )
                next_tick = None
                _wait_for_camera_replacement(
                    camera_latest,
                    blocked_frame_id=camera.frame_id,
                    control_dt_s=contract.control_dt_s,
                    maximum_dropout_s=float(args.max_camera_dropout_s),
                )
                continue
            policy_state_gap_s = (
                None
                if last_policy_state_timestamp_s is None
                else franka.captured_at_s - last_policy_state_timestamp_s
            )
            if freshness_frame_gate.resolve_replacement(
                camera.frame_id,
                policy_state_gap_s=policy_state_gap_s,
                maximum_continuity_gap_s=velocity_tracker.maximum_dt_s,
            ):
                # A long/contaminated rejection episode cannot be bridged by a
                # finite difference or a four-frame policy history.  This
                # replacement is used only for another discarded warm-up; a
                # subsequent fresh provider frame begins formal inference.
                _reset_policy_continuity(
                    history=history, velocity_tracker=velocity_tracker
                )
                projector = MaskedRGBDProjector(
                    camera_K=contract.camera_K,
                    T_base_camera_optical=contract.T_base_camera_optical,
                    image_size=(contract.camera_width, contract.camera_height),
                    depth_range_m=contract.depth_range_m,
                )
                point_frame = None
                point_capture_palm = None
                point_pose_skew_s = None
                point_camera_retrieved_at_s = None
                point_camera_published_at_s = None
                policy_warmup_complete = False
                last_policy_state_timestamp_s = None
                previous_tick_started = None
                next_tick = None
                print(
                    "[read-only policy rebootstrap] "
                    f"replacement_frame_id={camera.frame_id}; "
                    f"policy_state_gap_s={policy_state_gap_s!r}; "
                    "velocity/history/point-fallback cleared; replacement "
                    "reserved for discarded warm-up",
                    file=sys.stderr,
                )
            camera_is_new = last_camera_id != camera.frame_id
            if policy_warmup_complete and point_frame is None and not camera_is_new:
                # A formal capture must start with a provider frame newer than
                # the one used for the discarded model warm-up.
                next_tick = None
                _wait_for_camera_replacement(
                    camera_latest,
                    blocked_frame_id=camera.frame_id,
                    control_dt_s=contract.control_dt_s,
                    maximum_dropout_s=float(args.max_camera_dropout_s),
                )
                continue
            if (
                camera_is_new
                and point_frame is None
                and _hold_for_unusable_initial_object_observation(
                    gate=freshness_frame_gate,
                    frame_id=camera.frame_id,
                    usable=bool(camera.mask_valid),
                )
            ):
                _record_camera_rejection_diagnostic(
                    camera_rejection_diagnostics,
                    camera,
                    observed_at_s=observation_clock.realtime_s,
                    reason="initial_object_mask_invalid",
                    blocked_frame_id_before=None,
                    quiet=bool(args.quiet),
                )
                next_tick = None
                _wait_for_camera_replacement(
                    camera_latest,
                    blocked_frame_id=camera.frame_id,
                    control_dt_s=contract.control_dt_s,
                    maximum_dropout_s=float(args.max_camera_dropout_s),
                )
                continue
            freshness_budget: Optional[_FreshnessBudgetMetrics] = None
            if policy_warmup_complete:
                if action_compute_reserve_s is None:
                    raise RuntimeError("policy compute reserve was not calibrated")
                candidate_capture_s = (
                    camera.timestamp_s
                    if camera_is_new
                    else point_frame.captured_at_s
                )
                budget_clock = host_clock_guard.sample()
                freshness_budget = _freshness_budget_metrics(
                    pointcloud_age_s=(
                        budget_clock.realtime_s - candidate_capture_s
                    ),
                    maximum_age_s=float(args.max_camera_age_s),
                    required_reserve_s=action_compute_reserve_s,
                )
                if not freshness_budget.eligible:
                    # Do not project, update the velocity estimator, append a
                    # logical 60 Hz history item, or invoke the policy when the
                    # observation cannot conservatively survive to action time.
                    blocked_frame_id_before = freshness_frame_gate.blocked_frame_id
                    freshness_frame_gate.reject(camera.frame_id)
                    _record_camera_rejection_diagnostic(
                        camera_rejection_diagnostics,
                        camera,
                        observed_at_s=budget_clock.realtime_s,
                        reason="compute_reserve_rejected_before_projection",
                        blocked_frame_id_before=blocked_frame_id_before,
                        candidate_capture_s=candidate_capture_s,
                        pointcloud_capture_s=(
                            None
                            if point_frame is None
                            else point_frame.captured_at_s
                        ),
                        required_action_compute_reserve_s=(
                            action_compute_reserve_s
                        ),
                        freshness_budget_remaining_s=(
                            freshness_budget.remaining_s
                        ),
                        quiet=bool(args.quiet),
                    )
                    next_tick = None
                    _wait_for_camera_replacement(
                        camera_latest,
                        blocked_frame_id=camera.frame_id,
                        control_dt_s=contract.control_dt_s,
                        maximum_dropout_s=float(args.max_camera_dropout_s),
                    )
                    continue
                # The reserve protects exactly the work beginning after this
                # successful pre-compute freshness decision.  Reuse the
                # decision clock so timer overhead cannot create an optimistic
                # gap between eligibility and the measured compute interval.
                action_compute_started_monotonic_s = budget_clock.monotonic_s
            else:
                # Discarded warm-up uses the same boundary even though it has
                # no calibrated reserve yet.  The resulting complete-path
                # measurement calibrates the next formal camera frame.
                action_compute_started_monotonic_s = (
                    host_clock_guard.sample().monotonic_s
                )
            had_usable_point_frame_before_projection = bool(
                point_frame is not None
                and np.sum(point_frame.valid) >= 1.0
            )
            if camera_is_new:
                if (
                    not policy_warmup_complete
                    and not _capture_pose_is_ready(
                        pose_history,
                        camera.timestamp_s,
                        float(args.max_pose_skew_s),
                    )
                ):
                    # This camera sample predates the startup pose history (or
                    # arrived too far ahead of it).  It cannot initialize the
                    # point cloud.  Wait for a new provider frame without
                    # mutating projection, velocity, history, or policy state.
                    startup_pose_alignment_wait_frames += 1
                    next_tick = None
                    _wait_for_camera_replacement(
                        camera_latest,
                        blocked_frame_id=camera.frame_id,
                        control_dt_s=contract.control_dt_s,
                        maximum_dropout_s=float(args.max_camera_dropout_s),
                    )
                    continue
                capture_palm, capture_skew_s = _nearest_pose(
                    pose_history,
                    camera.timestamp_s,
                    float(args.max_pose_skew_s),
                )
                projected = projector.project(
                    color_bgr=camera.color_bgr,
                    depth_m=(
                        camera.depth_m if camera.depth_raw is None else None
                    ),
                    depth_raw=camera.depth_raw,
                    depth_scale_m_per_unit=(
                        camera.depth_scale_m_per_unit
                        if camera.depth_raw is not None
                        else None
                    ),
                    object_mask=(
                        camera.mask if camera.mask_valid else np.zeros_like(camera.mask)
                    ),
                    T_base_palm_at_capture=capture_palm,
                    captured_at_s=camera.timestamp_s,
                    frame_id=camera.frame_id,
                )
                if projected.frame_id == camera.frame_id:
                    point_capture_palm = capture_palm.copy()
                    point_pose_skew_s = float(capture_skew_s)
                    point_camera_retrieved_at_s = float(camera.retrieved_at_s)
                    point_camera_published_at_s = float(camera.published_at_s)
                point_frame = projected
                last_camera_id = camera.frame_id
            if point_frame is None or np.sum(point_frame.valid) < 1.0:
                if (
                    not had_usable_point_frame_before_projection
                    and _hold_for_unusable_initial_object_observation(
                        gate=freshness_frame_gate,
                        frame_id=camera.frame_id,
                        usable=False,
                    )
                ):
                    point_frame = None
                    point_capture_palm = None
                    point_pose_skew_s = None
                    point_camera_retrieved_at_s = None
                    point_camera_published_at_s = None
                    _record_camera_rejection_diagnostic(
                        camera_rejection_diagnostics,
                        camera,
                        observed_at_s=observation_clock.realtime_s,
                        reason="initial_object_pointcloud_empty",
                        blocked_frame_id_before=None,
                        quiet=bool(args.quiet),
                    )
                    next_tick = None
                    _wait_for_camera_replacement(
                        camera_latest,
                        blocked_frame_id=camera.frame_id,
                        control_dt_s=contract.control_dt_s,
                        maximum_dropout_s=float(args.max_camera_dropout_s),
                    )
                    continue
                raise RuntimeError("no valid or stale-palm object observation")
            if point_capture_palm is None or point_pose_skew_s is None:
                raise RuntimeError("object point cloud has no capture-time palm pose")
            if (
                point_camera_retrieved_at_s is None
                or point_camera_published_at_s is None
            ):
                raise RuntimeError("object point cloud has no camera publication timing")
            if policy_warmup_complete:
                if freshness_budget is None or action_compute_reserve_s is None:
                    raise RuntimeError("formal point-cloud compute budget is missing")
                point_check_clock = host_clock_guard.sample()
                actual_point_age_s = (
                    point_check_clock.realtime_s - point_frame.captured_at_s
                )
                if point_frame.captured_at_s != candidate_capture_s:
                    # A valid-looking camera update can still fall back to the
                    # projector's previous good cloud.  Budget that older cloud,
                    # not the optimistic latest camera timestamp.
                    freshness_budget = _freshness_budget_metrics(
                        pointcloud_age_s=actual_point_age_s,
                        maximum_age_s=float(args.max_camera_age_s),
                        required_reserve_s=action_compute_reserve_s,
                    )
                    if not freshness_budget.eligible:
                        blocked_frame_id_before = (
                            freshness_frame_gate.blocked_frame_id
                        )
                        freshness_frame_gate.reject(
                            camera.frame_id, force_rebootstrap=True
                        )
                        _record_camera_rejection_diagnostic(
                            camera_rejection_diagnostics,
                            camera,
                            observed_at_s=point_check_clock.realtime_s,
                            reason="point_fallback_compute_reserve_rejected",
                            blocked_frame_id_before=blocked_frame_id_before,
                            candidate_capture_s=point_frame.captured_at_s,
                            pointcloud_capture_s=point_frame.captured_at_s,
                            required_action_compute_reserve_s=(
                                action_compute_reserve_s
                            ),
                            freshness_budget_remaining_s=(
                                freshness_budget.remaining_s
                            ),
                            quiet=bool(args.quiet),
                        )
                        next_tick = None
                        _wait_for_camera_replacement(
                            camera_latest,
                            blocked_frame_id=camera.frame_id,
                            control_dt_s=contract.control_dt_s,
                            maximum_dropout_s=float(args.max_camera_dropout_s),
                        )
                        continue
                elif actual_point_age_s > float(args.max_camera_age_s):
                    # Projection itself exhausted the strict bound despite the
                    # calibrated reserve.  Do not mutate logical policy state;
                    # wait for a newer provider frame.  A later accepted action
                    # still undergoes the independent post-forward age check.
                    blocked_frame_id_before = freshness_frame_gate.blocked_frame_id
                    freshness_frame_gate.reject(
                        camera.frame_id,
                        reason="stale",
                        force_rebootstrap=True,
                    )
                    _record_camera_rejection_diagnostic(
                        camera_rejection_diagnostics,
                        camera,
                        observed_at_s=point_check_clock.realtime_s,
                        reason="projection_exhausted_capture_age",
                        blocked_frame_id_before=blocked_frame_id_before,
                        candidate_capture_s=point_frame.captured_at_s,
                        pointcloud_capture_s=point_frame.captured_at_s,
                        required_action_compute_reserve_s=(
                            action_compute_reserve_s
                        ),
                        freshness_budget_remaining_s=(
                            float(args.max_camera_age_s) - actual_point_age_s
                        ),
                        quiet=bool(args.quiet),
                    )
                    next_tick = None
                    _wait_for_camera_replacement(
                        camera_latest,
                        blocked_frame_id=camera.frame_id,
                        control_dt_s=contract.control_dt_s,
                        maximum_dropout_s=float(args.max_camera_dropout_s),
                    )
                    continue
            elif now - point_frame.captured_at_s > float(args.max_camera_age_s):
                # The projector can deliberately retain the last valid cloud
                # when the new mask is unusable.  If that fallback has expired,
                # hold exactly like a stale raw camera frame; never warm up the
                # policy on it and let the independent frame watchdog detect a
                # sustained provider dropout.
                blocked_frame_id_before = freshness_frame_gate.blocked_frame_id
                freshness_frame_gate.reject(
                    camera.frame_id,
                    reason="stale",
                    force_rebootstrap=True,
                )
                _record_camera_rejection_diagnostic(
                    camera_rejection_diagnostics,
                    camera,
                    observed_at_s=observation_clock.realtime_s,
                    reason="warmup_point_fallback_capture_age",
                    blocked_frame_id_before=blocked_frame_id_before,
                    candidate_capture_s=point_frame.captured_at_s,
                    pointcloud_capture_s=point_frame.captured_at_s,
                    quiet=bool(args.quiet),
                )
                next_tick = None
                _wait_for_camera_replacement(
                    camera_latest,
                    blocked_frame_id=camera.frame_id,
                    control_dt_s=contract.control_dt_s,
                    maximum_dropout_s=float(args.max_camera_dropout_s),
                )
                continue

            if (
                policy_warmup_complete
                and camera_policy_reuse_guard.must_hold(point_frame.frame_id)
            ):
                # This is an explicit formal 60 Hz opportunity, but the actual
                # point cloud has already produced two accepted actions.  Keep
                # the tick as separate hold evidence and stop before velocity,
                # policy history, model inference, or target mapping can mutate.
                if previous_tick_started is None:
                    raise RuntimeError(
                        "policy reuse hold has no previous accepted action tick"
                    )
                hold_clock = host_clock_guard.sample()
                hold_finished = time.monotonic()
                hold_timing = _deadline_metrics(
                    scheduled_tick_monotonic_s=scheduled_tick,
                    tick_started_monotonic_s=tick_started,
                    tick_finished_monotonic_s=hold_finished,
                    control_dt_s=contract.control_dt_s,
                )
                _require_deadline(
                    hold_timing,
                    maximum_overrun_s=float(args.max_deadline_overrun_s),
                )
                accepted_uses = camera_policy_reuse_guard.record_hold(
                    point_frame.frame_id
                )
                hold_record = _policy_reuse_hold_diagnostic(
                    camera,
                    hold_index=len(policy_reuse_hold_diagnostics),
                    pointcloud_frame_id=point_frame.frame_id,
                    pointcloud_capture_timestamp_s=point_frame.captured_at_s,
                    pointcloud_retrieved_at_s=point_camera_retrieved_at_s,
                    pointcloud_published_at_s=point_camera_published_at_s,
                    observation_clock=observation_clock,
                    hold_clock=hold_clock,
                    timing=hold_timing,
                    previous_action_tick_started_monotonic_s=(
                        previous_tick_started
                    ),
                    camera_is_new=camera_is_new,
                    action_uses_before_hold=accepted_uses,
                    maximum_policy_actions_per_camera_frame=(
                        camera_policy_reuse_guard.maximum_actions_per_frame
                    ),
                )
                policy_reuse_hold_diagnostics.append(hold_record)
                if not args.quiet:
                    print(
                        "[read-only policy reuse hold] "
                        + json.dumps(
                            {
                                "diagnostic_index": hold_record[
                                    "diagnostic_index"
                                ],
                                "reason": hold_record["reason"],
                                "camera_frame_id": hold_record[
                                    "camera_frame_id"
                                ],
                                "pointcloud_frame_id": hold_record[
                                    "pointcloud_frame_id"
                                ],
                                "action_uses_before_hold": hold_record[
                                    "action_uses_before_hold"
                                ],
                                "new_policy_action_evidence": False,
                                "policy_state_mutated": False,
                                "write": False,
                            },
                            sort_keys=True,
                        ),
                        file=sys.stderr,
                    )
                # Do not advance ``previous_tick_started``: the next action row
                # must expose the complete action-to-action interval containing
                # this hold.  Rephase to a genuinely newer provider frame.
                next_tick = None
                _wait_for_camera_replacement(
                    camera_latest,
                    blocked_frame_id=camera.frame_id,
                    control_dt_s=contract.control_dt_s,
                    maximum_dropout_s=float(args.max_camera_dropout_s),
                )
                continue

            velocity = velocity_tracker.update(
                captured_at_s=franka.captured_at_s,
                T_base_palm=T_base_palm,
                hand_q_policy_order_rad=hand.q_policy_rad,
            )
            tips = fingertip_model.positions_base(
                angle_act_register_order=hand.angle_act,
                T_base_palm=T_base_palm,
            )
            proprio = proprio_builder.build(
                franka_q_rad=franka.q,
                franka_dq_rad_s=franka.dq,
                rh56_virtual_q_policy_order_rad=hand.q_policy_rad,
                rh56_virtual_dq_policy_order_rad_s=hand.dq_policy_rad_s,
                T_base_palm=T_base_palm,
                palm_linear_velocity_base_m_s=velocity.linear_base_m_s,
                palm_angular_velocity_base_rad_s=velocity.angular_base_rad_s,
                fingertip_positions_base_m=tips,
                # No policy command is executed in this entry point.  Keep the
                # observation honest instead of feeding back hypothetical
                # actions against a stationary robot.
                previous_executed_action13=INITIAL_PREVIOUS_ACTION13,
            )
            point_history, valid_history, proprio_history = history.append(
                point_frame, proprio
            )
            # This timestamp mirrors the actual finite-difference/history
            # baseline, not merely emitted audit rows.  Any later rejection in
            # this tick is marked force-rebootstrap because these objects have
            # already advanced and cannot be rolled back safely.
            last_policy_state_timestamp_s = float(franka.captured_at_s)
            is_discarded_warmup = not policy_warmup_complete
            output, policy_inference_s = _run_policy_preview_forward(
                policy,
                point_history,
                valid_history,
                proprio_history,
                # Warm-up is still logically discarded below, but retaining
                # its raw output locally lets us time the exact target-mapping
                # tail.  This preview has no hardware write path.
                accept_output=True,
            )
            # Produce a one-step target proposal from the current measured
            # state.  Do not accumulate unexecuted targets across preview ticks.
            # The same mapping is run during discarded warm-up so calibration
            # covers the full work protected by the pre-compute reserve.
            target_mapper = V94ActionMapper(
                initial_arm_target_q_rad=franka.q,
                initial_hand_target_q_policy_order_rad=hand.q_policy_rad,
                joint_limits_rad=contract.joint_limits_rad,
                q_hand_close_rad=contract.q_hand_close_rad,
            )
            mapped = target_mapper.map(output.action13, measured_q_rad=franka.q)
            action_clock = host_clock_guard.sample()
            observation_action_compute_s = (
                action_clock.monotonic_s - action_compute_started_monotonic_s
            )
            if (
                not np.isfinite(observation_action_compute_s)
                or observation_action_compute_s < 0.0
            ):
                raise RuntimeError(
                    "observation-to-action compute timer is invalid: "
                    f"elapsed={observation_action_compute_s!r}"
                )
            if is_discarded_warmup:
                policy_warmup_inference_s = float(policy_inference_s)
                maximum_policy_warmup_inference_s = max(
                    maximum_policy_warmup_inference_s,
                    policy_warmup_inference_s,
                )
                observation_action_warmup_s = float(
                    observation_action_compute_s
                )
                maximum_observation_action_warmup_s = max(
                    maximum_observation_action_warmup_s,
                    observation_action_warmup_s,
                )
                policy_warmup_count += 1
                calibrated_reserve_s = _calibrated_action_compute_reserve_s(
                    minimum_reserve_s=float(
                        args.min_action_compute_reserve_s
                    ),
                    warmup_observation_action_s=observation_action_warmup_s,
                )
                action_compute_reserve_s = max(
                    0.0 if action_compute_reserve_s is None else action_compute_reserve_s,
                    calibrated_reserve_s,
                )
                if action_compute_reserve_s >= float(args.max_camera_age_s):
                    raise RuntimeError(
                        "calibrated action compute reserve leaves no strict "
                        "camera freshness budget: "
                        f"reserve={action_compute_reserve_s:.6f}s, "
                        f"max_camera_age={float(args.max_camera_age_s):.6f}s, "
                        "warmup_observation_action="
                        f"{observation_action_warmup_s:.6f}s, "
                        f"warmup_policy={policy_warmup_inference_s:.6f}s"
                    )
                print(
                    "[read-only freshness budget] "
                    f"max_camera_age={float(args.max_camera_age_s):.6f}s; "
                    "effective_action_compute_reserve="
                    f"{action_compute_reserve_s:.6f}s; "
                    "configured_minimum="
                    f"{float(args.min_action_compute_reserve_s):.6f}s; "
                    "discarded_warmup_observation_action="
                    f"{observation_action_warmup_s:.6f}s; "
                    f"discarded_warmup_policy={policy_warmup_inference_s:.6f}s; "
                    f"discarded_warmup_count={policy_warmup_count}; "
                    f"warmup_factor={ACTION_COMPUTE_RESERVE_WARMUP_FACTOR:.1f}",
                    file=sys.stderr,
                )
                # Do not accept one-time NumPy/BLAS/model initialization as a
                # formal observation/action.  Its complete-path timing remains
                # a conservative calibration sample.  Clear every logical
                # state owner touched by the discarded inference and require a
                # newer camera frame before beginning the real capture.
                policy_warmup_complete = True
                _reset_policy_continuity(
                    history=history, velocity_tracker=velocity_tracker
                )
                last_policy_state_timestamp_s = None
                projector = MaskedRGBDProjector(
                    camera_K=contract.camera_K,
                    T_base_camera_optical=contract.T_base_camera_optical,
                    image_size=(contract.camera_width, contract.camera_height),
                    depth_range_m=contract.depth_range_m,
                )
                point_frame = None
                point_capture_palm = None
                point_pose_skew_s = None
                point_camera_retrieved_at_s = None
                point_camera_published_at_s = None
                next_tick = None
                previous_tick_started = None
                _wait_for_camera_replacement(
                    camera_latest,
                    blocked_frame_id=camera.frame_id,
                    control_dt_s=contract.control_dt_s,
                    maximum_dropout_s=float(args.max_camera_dropout_s),
                )
                continue
            action_wall_time_s = action_clock.realtime_s
            franka_age_at_action_s = action_wall_time_s - franka.captured_at_s
            hand_age_at_action_s = action_wall_time_s - hand.timestamp_s
            pointcloud_age_at_action_s = (
                action_wall_time_s - point_frame.captured_at_s
            )
            # Freshness is checked again after projection/FK/inference.  A
            # sample that expires during computation must not be reported as
            # an accepted observation -> action proposal.
            _require_sensor_age(
                "RH56 at action",
                hand_age_at_action_s,
                maximum_age_s=float(args.max_hand_age_s),
                maximum_future_skew_s=maximum_future_skew_s,
            )
            if (
                np.isfinite(pointcloud_age_at_action_s)
                and pointcloud_age_at_action_s > float(args.max_camera_age_s)
            ):
                # History/velocity/projector were necessarily touched to reach
                # this post-forward check.  The proposal is not accepted and a
                # replacement must force a full continuity rebootstrap; there
                # is no safe partial rollback of that rejected logical tick.
                blocked_frame_id_before = freshness_frame_gate.blocked_frame_id
                freshness_frame_gate.reject(
                    camera.frame_id,
                    reason="stale",
                    force_rebootstrap=True,
                )
                _record_camera_rejection_diagnostic(
                    camera_rejection_diagnostics,
                    camera,
                    observed_at_s=action_wall_time_s,
                    reason="post_forward_pointcloud_age_rejected",
                    blocked_frame_id_before=blocked_frame_id_before,
                    candidate_capture_s=point_frame.captured_at_s,
                    pointcloud_capture_s=point_frame.captured_at_s,
                    required_action_compute_reserve_s=action_compute_reserve_s,
                    freshness_budget_remaining_s=freshness_budget.remaining_s,
                    observation_action_compute_s=observation_action_compute_s,
                    quiet=bool(args.quiet),
                )
                next_tick = None
                _wait_for_camera_replacement(
                    camera_latest,
                    blocked_frame_id=camera.frame_id,
                    control_dt_s=contract.control_dt_s,
                    maximum_dropout_s=float(args.max_camera_dropout_s),
                )
                continue
            _require_sensor_age(
                (
                    "D435 point cloud at action "
                    f"(pre_age={now - point_frame.captured_at_s:.6f}s, "
                    f"policy={policy_inference_s:.6f}s, "
                    f"provider_total={camera.provider_timings_ms[-1]:.3f}ms)"
                ),
                pointcloud_age_at_action_s,
                maximum_age_s=float(args.max_camera_age_s),
                maximum_future_skew_s=maximum_future_skew_s,
            )
            _require_sensor_age(
                "Franka at action",
                franka_age_at_action_s,
                maximum_age_s=float(args.max_camera_age_s),
                maximum_future_skew_s=maximum_future_skew_s,
            )
            projector_mask_provenance = (
                projector.last_effective_object_mask_provenance
            )
            console_record = {
                "step": step,
                "timestamp_s": franka.captured_at_s,
                "camera_frame_id": point_frame.frame_id,
                "camera_age_s": now - point_frame.captured_at_s,
                "hand_age_s": hand_age_s,
                "point_status": point_frame.status,
                "valid_points": int(np.sum(point_frame.valid)),
                "source_valid_points": int(point_frame.source_valid_points),
                "pointcloud_age_at_action_s": pointcloud_age_at_action_s,
                "pointcloud_freshness_budget_before_compute_s": (
                    freshness_budget.remaining_s
                ),
                "action_compute_reserve_s": action_compute_reserve_s,
                "observation_action_compute_s": observation_action_compute_s,
                "action_compute_reserve_margin_s": (
                    action_compute_reserve_s - observation_action_compute_s
                ),
                "freshness_budget_rejected_frames": (
                    freshness_frame_gate.budget_rejected_frames
                ),
                "camera_transient_stale_rejected_frames": (
                    freshness_frame_gate.transient_stale_rejected_frames
                ),
                "policy_rebootstrap_count": freshness_frame_gate.rebootstrap_count,
                "requested_object_mask_mode": camera.requested_object_mask_mode,
                "effective_object_mask_mode": camera.effective_object_mask_mode,
                "effective_provider_mask_publication_mode": (
                    camera.effective_mask_publication_mode
                ),
                "provider_published_mask_source": (
                    camera.provider_published_mask_source
                ),
                "provider_online_sam2_status": camera.provider_online_sam2_status,
                "camera_object_mask_area_px": camera.object_mask_area_px,
                "camera_object_mask_bbox_xyxy": np.asarray(
                    camera.object_mask_bbox_xyxy, dtype=np.int32
                ).tolist(),
                "projector_effective_policy_mask_provenance": (
                    projector_mask_provenance.kind
                ),
                "projector_effective_policy_mask_source_frame_id": (
                    projector_mask_provenance.source_frame_id
                ),
                "projector_effective_policy_mask_area_px": (
                    projector_mask_provenance.area_px
                ),
                "projector_effective_policy_mask_bbox_xyxy": (
                    projector_mask_provenance.bbox_xyxy
                ),
                "pointcloud_capture_to_publish_s": (
                    point_camera_published_at_s - point_frame.captured_at_s
                ),
                "policy_inference_s": policy_inference_s,
                "raw_policy_action13": output.action13.tolist(),
                "franka_target_q_rad": mapped.franka_target_q_rad.tolist(),
                "rh56_angle_set": mapped.rh56_angle_set_register_order.tolist(),
                "predicted_hold_logit_diagnostic_only": output.predicted_hold_logit,
                "proposal_mode": "one_step_from_measured_idle",
                "write": False,
            }
            tick_finished = time.monotonic()
            timing = _deadline_metrics(
                scheduled_tick_monotonic_s=scheduled_tick,
                tick_started_monotonic_s=tick_started,
                tick_finished_monotonic_s=tick_finished,
                control_dt_s=contract.control_dt_s,
            )
            _require_deadline(
                timing,
                maximum_overrun_s=float(args.max_deadline_overrun_s),
            )
            policy_action_use_index = camera_policy_reuse_guard.accept(
                point_frame.frame_id
            )
            tick_interval_s = (
                0.0
                if previous_tick_started is None
                else tick_started - previous_tick_started
            )
            sensor_timestamps = np.asarray(
                [franka.captured_at_s, hand.timestamp_s, point_frame.captured_at_s],
                dtype=np.float64,
            )
            audit_records.append(
                {
                    "step": np.asarray(step, dtype=np.int64),
                    "timestamp_s": np.asarray(franka.captured_at_s),
                    "wall_observation_time_s": np.asarray(now),
                    "host_observation_realtime_s": np.asarray(
                        observation_clock.realtime_s
                    ),
                    "host_observation_monotonic_s": np.asarray(
                        observation_clock.monotonic_s
                    ),
                    "host_observation_clock_pair_span_s": np.asarray(
                        observation_clock.pair_span_s
                    ),
                    "host_observation_realtime_minus_monotonic_s": np.asarray(
                        observation_clock.realtime_minus_monotonic_s
                    ),
                    "host_observation_clock_offset_delta_from_start_s": np.asarray(
                        observation_clock.offset_delta_from_start_s
                    ),
                    "host_observation_clock_offset_delta_from_previous_s": np.asarray(
                        observation_clock.offset_delta_from_previous_s
                    ),
                    "host_action_realtime_s": np.asarray(action_clock.realtime_s),
                    "host_action_monotonic_s": np.asarray(action_clock.monotonic_s),
                    "host_action_clock_pair_span_s": np.asarray(
                        action_clock.pair_span_s
                    ),
                    "host_action_realtime_minus_monotonic_s": np.asarray(
                        action_clock.realtime_minus_monotonic_s
                    ),
                    "host_action_clock_offset_delta_from_start_s": np.asarray(
                        action_clock.offset_delta_from_start_s
                    ),
                    "host_action_clock_offset_delta_from_previous_s": np.asarray(
                        action_clock.offset_delta_from_previous_s
                    ),
                    "control_scheduled_monotonic_s": np.asarray(
                        timing.scheduled_tick_monotonic_s
                    ),
                    "control_tick_started_monotonic_s": np.asarray(
                        timing.tick_started_monotonic_s
                    ),
                    "control_tick_finished_monotonic_s": np.asarray(
                        timing.tick_finished_monotonic_s
                    ),
                    "control_tick_interval_s": np.asarray(tick_interval_s),
                    "control_tick_interval_valid": np.asarray(
                        previous_tick_started is not None
                    ),
                    "control_start_lateness_s": np.asarray(timing.start_lateness_s),
                    "control_computation_s": np.asarray(timing.computation_s),
                    "control_next_deadline_overrun_s": np.asarray(
                        timing.next_deadline_overrun_s
                    ),
                    "sensor_capture_span_s": np.asarray(
                        float(np.max(sensor_timestamps) - np.min(sensor_timestamps))
                    ),
                    "franka_age_s": np.asarray(franka_age_s),
                    "hand_age_s": np.asarray(hand_age_s),
                    "camera_latest_age_s": np.asarray(camera_age_s),
                    "pointcloud_age_s": np.asarray(now - point_frame.captured_at_s),
                    "franka_age_at_action_s": np.asarray(franka_age_at_action_s),
                    "hand_age_at_action_s": np.asarray(hand_age_at_action_s),
                    "pointcloud_age_at_action_s": np.asarray(
                        pointcloud_age_at_action_s
                    ),
                    "pointcloud_age_before_reserved_compute_s": np.asarray(
                        freshness_budget.pointcloud_age_s
                    ),
                    "pointcloud_freshness_budget_before_compute_s": np.asarray(
                        freshness_budget.remaining_s
                    ),
                    "action_compute_reserve_s": np.asarray(
                        action_compute_reserve_s
                    ),
                    "observation_action_compute_s": np.asarray(
                        observation_action_compute_s
                    ),
                    "action_compute_reserve_margin_s": np.asarray(
                        action_compute_reserve_s - observation_action_compute_s
                    ),
                    "freshness_budget_rejected_frames_cumulative": np.asarray(
                        freshness_frame_gate.budget_rejected_frames, dtype=np.int64
                    ),
                    "camera_transient_stale_rejected_frames_cumulative": np.asarray(
                        freshness_frame_gate.transient_stale_rejected_frames,
                        dtype=np.int64,
                    ),
                    "camera_freshness_rejected_frames_cumulative": np.asarray(
                        freshness_frame_gate.rejected_frames, dtype=np.int64
                    ),
                    "freshness_budget_wait_ticks_cumulative": np.asarray(
                        freshness_frame_gate.wait_ticks, dtype=np.int64
                    ),
                    "policy_rebootstrap_count_cumulative": np.asarray(
                        freshness_frame_gate.rebootstrap_count, dtype=np.int64
                    ),
                    "discarded_policy_warmup_inferences_cumulative": np.asarray(
                        policy_warmup_count, dtype=np.int64
                    ),
                    "camera_watchdog_max_unchanged_s": np.asarray(
                        camera_frame_watchdog.maximum_unchanged_s
                    ),
                    "pointcloud_capture_to_retrieval_s": np.asarray(
                        point_camera_retrieved_at_s - point_frame.captured_at_s
                    ),
                    "pointcloud_retrieval_to_publish_s": np.asarray(
                        point_camera_published_at_s - point_camera_retrieved_at_s
                    ),
                    "pointcloud_publish_to_observation_s": np.asarray(
                        now - point_camera_published_at_s
                    ),
                    "pointcloud_publish_to_action_s": np.asarray(
                        action_wall_time_s - point_camera_published_at_s
                    ),
                    "camera_pose_skew_s": np.asarray(point_pose_skew_s),
                    "camera_new_frame": np.asarray(camera_is_new),
                    "camera_latest_frame_id": np.asarray(
                        camera.frame_id, dtype=np.int64
                    ),
                    "camera_latest_timestamp_s": np.asarray(camera.timestamp_s),
                    "camera_mask_valid": np.asarray(camera.mask_valid),
                    "camera_message": np.asarray(camera.message),
                    "requested_object_mask_mode": np.asarray(
                        camera.requested_object_mask_mode
                    ),
                    "effective_object_mask_mode": np.asarray(
                        camera.effective_object_mask_mode
                    ),
                    "effective_provider_mask_publication_mode": np.asarray(
                        camera.effective_mask_publication_mode
                    ),
                    "effective_provider_recovery_publication_mode": np.asarray(
                        camera.effective_recovery_publication_mode
                    ),
                    "provider_published_mask_source": np.asarray(
                        camera.provider_published_mask_source
                    ),
                    "provider_published_mask_message": np.asarray(
                        camera.provider_published_mask_message
                    ),
                    "provider_online_sam2_status": np.asarray(
                        camera.provider_online_sam2_status
                    ),
                    "camera_object_mask_area_px": np.asarray(
                        camera.object_mask_area_px, dtype=np.int64
                    ),
                    "camera_object_mask_bbox_xyxy": np.asarray(
                        camera.object_mask_bbox_xyxy, dtype=np.int32
                    ).reshape(4).copy(),
                    "projector_effective_policy_mask_provenance": np.asarray(
                        projector_mask_provenance.kind
                    ),
                    "projector_effective_policy_mask_source_frame_id": np.asarray(
                        -1
                        if projector_mask_provenance.source_frame_id is None
                        else projector_mask_provenance.source_frame_id,
                        dtype=np.int64,
                    ),
                    "projector_effective_policy_mask_area_px": np.asarray(
                        projector_mask_provenance.area_px,
                        dtype=np.int64,
                    ),
                    "projector_effective_policy_mask_bbox_valid": np.asarray(
                        projector_mask_provenance.bbox_xyxy is not None
                    ),
                    "projector_effective_policy_mask_bbox_xyxy": np.asarray(
                        (
                            (0, 0, 0, 0)
                            if projector_mask_provenance.bbox_xyxy is None
                            else projector_mask_provenance.bbox_xyxy
                        ),
                        dtype=np.int32,
                    ),
                    "camera_retrieved_at_s": np.asarray(camera.retrieved_at_s),
                    "camera_published_at_s": np.asarray(camera.published_at_s),
                    "camera_capture_to_retrieval_s": np.asarray(
                        camera.retrieved_at_s - camera.timestamp_s
                    ),
                    "camera_timestamp_domain": np.asarray(camera.timestamp_domain),
                    "camera_color_depth_timestamp_skew_s": np.asarray(
                        camera.color_depth_timestamp_skew_s
                    ),
                    "camera_color_depth_epoch_timestamp_skew_s": np.asarray(
                        camera.color_depth_epoch_timestamp_skew_s
                    ),
                    "camera_rejected_timestamp_skew_frames": np.asarray(
                        camera.rejected_timestamp_skew_frames, dtype=np.int64
                    ),
                    "camera_last_rejected_color_depth_skew_s": np.asarray(
                        camera.last_rejected_color_depth_skew_s
                    ),
                    "camera_dropped_queued_framesets": np.asarray(
                        camera.dropped_queued_framesets, dtype=np.int64
                    ),
                    "camera_sensor_frame_number": np.asarray(
                        camera.sensor_frame_number, dtype=np.int64
                    ),
                    "camera_depth_sensor_frame_number": np.asarray(
                        camera.depth_sensor_frame_number, dtype=np.int64
                    ),
                    "camera_rejected_transport_stale_frames": np.asarray(
                        camera.rejected_transport_stale_frames, dtype=np.int64
                    ),
                    "camera_last_rejected_transport_age_s": np.asarray(
                        camera.last_rejected_transport_age_s
                    ),
                    "camera_retrieved_monotonic_s": np.asarray(
                        camera.retrieved_monotonic_s
                    ),
                    "camera_host_clock_pair_span_s": np.asarray(
                        camera.host_clock_pair_span_s
                    ),
                    "camera_capture_diagnostic_json": np.asarray(
                        camera.capture_diagnostic_json
                    ),
                    "camera_provider_timings_ms": camera.provider_timings_ms.astype(
                        np.float64, copy=True
                    ),
                    "camera_K": camera.camera_K.astype(np.float64, copy=True),
                    "camera_distortion": camera.distortion.astype(
                        np.float64, copy=True
                    ),
                    "camera_distortion_model": np.asarray(camera.distortion_model),
                    "camera_depth_scale_m_per_unit": np.asarray(
                        camera.depth_scale_m_per_unit
                    ),
                    "T_base_camera_optical": contract.T_base_camera_optical.astype(
                        np.float64, copy=True
                    ),
                    "pointcloud_frame_id": np.asarray(
                        point_frame.frame_id, dtype=np.int64
                    ),
                    "policy_action_use_index_for_camera_frame": np.asarray(
                        policy_action_use_index, dtype=np.int32
                    ),
                    "pointcloud_timestamp_s": np.asarray(point_frame.captured_at_s),
                    "pointcloud_status": np.asarray(point_frame.status),
                    "pointcloud_source_valid_points": np.asarray(
                        point_frame.source_valid_points, dtype=np.int32
                    ),
                    "pointcloud_current_xyzrgb_palm": point_frame.xyzrgb_palm.astype(
                        np.float32, copy=True
                    ),
                    "pointcloud_current_valid": point_frame.valid.astype(
                        np.float32, copy=True
                    ),
                    "pointcloud_history_xyzrgb_palm": point_history.astype(
                        np.float32, copy=True
                    ),
                    "pointcloud_valid_history": valid_history.astype(
                        np.float32, copy=True
                    ),
                    "proprio_history67": proprio_history.astype(np.float32, copy=True),
                    "proprio_current67": proprio.astype(np.float32, copy=True),
                    "T_base_palm": T_base_palm.astype(np.float64, copy=True),
                    "T_base_palm_at_pointcloud_capture": (
                        point_capture_palm.astype(np.float64, copy=True)
                    ),
                    "fingertip_positions_base_m": tips.astype(np.float64, copy=True),
                    "palm_linear_velocity_base_m_s": (
                        velocity.linear_base_m_s.astype(np.float32, copy=True)
                    ),
                    "palm_angular_velocity_base_rad_s": (
                        velocity.angular_base_rad_s.astype(np.float32, copy=True)
                    ),
                    "franka_timestamp_s": np.asarray(franka.captured_at_s),
                    "franka_q_rad": franka.q.astype(np.float64, copy=True),
                    "franka_dq_rad_s": franka.dq.astype(np.float64, copy=True),
                    "franka_q_desired_rad": franka.q_desired.astype(
                        np.float64, copy=True
                    ),
                    "franka_dq_desired_rad_s": franka.dq_desired.astype(
                        np.float64, copy=True
                    ),
                    "franka_tau_j": franka.tau_j.astype(np.float64, copy=True),
                    "franka_tau_ext_hat_filtered": (
                        franka.tau_ext_hat_filtered.astype(np.float64, copy=True)
                    ),
                    "franka_T_base_ee": franka.T_base_ee.astype(np.float64, copy=True),
                    "franka_T_base_ee_desired": franka.T_base_ee_desired.astype(
                        np.float64, copy=True
                    ),
                    "franka_F_T_EE": franka.F_T_EE.astype(np.float64, copy=True),
                    "franka_desired_cartesian_velocity": (
                        franka.desired_cartesian_velocity.astype(np.float64, copy=True)
                    ),
                    "franka_external_wrench_base": (
                        franka.external_wrench_base.astype(np.float64, copy=True)
                    ),
                    "franka_joint_contact": franka.joint_contact.astype(
                        np.float64, copy=True
                    ),
                    "franka_joint_collision": franka.joint_collision.astype(
                        np.float64, copy=True
                    ),
                    "franka_cartesian_contact": franka.cartesian_contact.astype(
                        np.float64, copy=True
                    ),
                    "franka_cartesian_collision": (
                        franka.cartesian_collision.astype(np.float64, copy=True)
                    ),
                    "franka_robot_mode": np.asarray(franka.robot_mode),
                    "franka_current_errors_json": np.asarray(
                        json.dumps(list(franka.current_errors))
                    ),
                    "franka_control_command_success_rate": np.asarray(
                        franka.control_command_success_rate
                    ),
                    "franka_F_x_Cee_m": franka.F_x_Cee_m.astype(np.float64, copy=True),
                    "franka_I_ee_kg_m2": franka.I_ee_kg_m2.astype(
                        np.float64, copy=True
                    ),
                    "franka_masses_kg": np.asarray(
                        [franka.m_ee_kg, franka.m_load_kg, franka.m_total_kg],
                        dtype=np.float64,
                    ),
                    "rh56_timestamp_s": np.asarray(hand.timestamp_s),
                    "rh56_snapshot_span_s": np.asarray(hand.snapshot_span_s),
                    "rh56_angle_set_register_order": hand.angle_targets.astype(
                        np.int32, copy=True
                    ),
                    "rh56_angle_act_register_order": hand.angle_act.astype(
                        np.int32, copy=True
                    ),
                    "rh56_position_act_register_order": hand.positions.astype(
                        np.int32, copy=True
                    ),
                    "rh56_force_act_register_order": hand.forces.astype(
                        np.int32, copy=True
                    ),
                    "rh56_current_register_order": hand.currents.astype(
                        np.int32, copy=True
                    ),
                    "rh56_error_register_order": hand.errors.astype(
                        np.int32, copy=True
                    ),
                    "rh56_status_register_order": hand.statuses.astype(
                        np.int32, copy=True
                    ),
                    "rh56_temperature_c_register_order": (
                        hand.temperatures_c.astype(np.int32, copy=True)
                    ),
                    "rh56_virtual_q_policy_order_rad": hand.q_policy_rad.astype(
                        np.float32, copy=True
                    ),
                    "rh56_virtual_dq_policy_order_rad_s": (
                        hand.dq_policy_rad_s.astype(np.float32, copy=True)
                    ),
                    "raw_policy_action13": output.action13.astype(
                        np.float32, copy=True
                    ),
                    "policy_inference_s": np.asarray(policy_inference_s),
                    "executed_policy_action13": (
                        mapped.executed_policy_action13.astype(np.float32, copy=True)
                    ),
                    "predicted_privileged32": (
                        output.predicted_privileged32.astype(np.float32, copy=True)
                    ),
                    "predicted_future_motion24": output.future_motion24.astype(
                        np.float32, copy=True
                    ),
                    "predicted_hold6": output.predicted_hold6.astype(
                        np.float32, copy=True
                    ),
                    "predicted_hold_logit": np.asarray(output.predicted_hold_logit),
                    "franka_target_q_rad": mapped.franka_target_q_rad.astype(
                        np.float32, copy=True
                    ),
                    "rh56_target_q_policy_order_rad": (
                        mapped.rh56_target_q_policy_order_rad.astype(
                            np.float32, copy=True
                        )
                    ),
                    "rh56_proposed_angle_set_register_order": (
                        mapped.rh56_angle_set_register_order.astype(np.int32, copy=True)
                    ),
                    "action_clipped": np.asarray(mapped.clipped),
                    "action_reasons_json": np.asarray(json.dumps(list(mapped.reasons))),
                    "proposal_mode": np.asarray("one_step_from_measured_idle"),
                    "write": np.asarray(False),
                }
            )
            if not args.quiet:
                print(json.dumps(console_record, sort_keys=True))
            console_logs.append(console_record)
            if capture_started is None:
                capture_started = tick_started
            step += 1
            previous_tick_started = tick_started
            next_tick = scheduled_tick + contract.control_dt_s
            wait = next_tick - time.monotonic()
            if wait > 0.0:
                time.sleep(wait)

        if capture_started is None:
            camera_latest.raise_if_failed()
            hand_latest.raise_if_failed()
            raise RuntimeError(
                "no fresh complete observation arrived within startup timeout "
                f"{float(args.startup_timeout_s):.3f}s"
            )
        _require_preview_complete(camera_latest, hand_latest, audit_records)
        if camera_policy_reuse_guard.accepted_action_count != len(audit_records):
            raise RuntimeError(
                "camera reuse guard action count does not match audit evidence"
            )
        if camera_policy_reuse_guard.hold_count != len(
            policy_reuse_hold_diagnostics
        ):
            raise RuntimeError(
                "camera reuse guard hold count does not match hold evidence"
            )
        audit_payload = _stack_audit_records(audit_records)
        if hashlib.sha256(profile_path.read_bytes()).hexdigest() != profile_sha256:
            raise RuntimeError(
                "commissioning profile changed during read-only shadow capture"
            )
        audit_payload.update(
            {
                "bundle_contract": np.asarray(verification.bundle_contract),
                "bundle_status": np.asarray(verification.bundle_status),
                "checkpoint_sha256": np.asarray(verification.checkpoint_sha256),
                "checkpoint_iteration": np.asarray(
                    verification.checkpoint_iteration, dtype=np.int64
                ),
                "calibration_id": np.asarray(contract.calibration_id),
                "camera_serial": np.asarray(contract.camera_serial),
                "policy_control_dt_s": np.asarray(contract.control_dt_s),
                "requested_capture_duration_s": np.asarray(duration),
                "maximum_policy_actions_per_camera_frame": np.asarray(
                    camera_policy_reuse_guard.maximum_actions_per_frame,
                    dtype=np.int32,
                ),
                "policy_action_evidence_count": np.asarray(
                    camera_policy_reuse_guard.accepted_action_count,
                    dtype=np.int64,
                ),
                "policy_reuse_guard_hold_count": np.asarray(
                    camera_policy_reuse_guard.hold_count, dtype=np.int64
                ),
                "policy_action_or_reuse_hold_opportunity_count": np.asarray(
                    camera_policy_reuse_guard.accepted_action_count
                    + camera_policy_reuse_guard.hold_count,
                    dtype=np.int64,
                ),
                "policy_reuse_guard_semantics": np.asarray(
                    POLICY_REUSE_GUARD_SEMANTICS
                ),
                "startup_elapsed_s": np.asarray(capture_started - startup_started),
                "host_clock_start_realtime_s": np.asarray(
                    host_clock_guard.start_realtime_s
                ),
                "host_clock_start_monotonic_s": np.asarray(
                    host_clock_guard.start_monotonic_s
                ),
                "host_clock_start_pair_span_s": np.asarray(
                    host_clock_guard.start_pair_span_s
                ),
                "host_clock_start_realtime_minus_monotonic_s": np.asarray(
                    host_clock_guard.start_offset_s
                ),
                "configured_max_host_clock_offset_jump_s": np.asarray(
                    host_clock_guard.maximum_offset_jump_s
                ),
                "configured_max_host_clock_pair_span_s": np.asarray(
                    MAX_HOST_CLOCK_PAIR_SPAN_S
                ),
                "discarded_cold_provider_frames": np.asarray(1, dtype=np.int32),
                "startup_pose_alignment_wait_frames": np.asarray(
                    startup_pose_alignment_wait_frames, dtype=np.int64
                ),
                "pcd_config_path": np.asarray(str(args.pcd_config.resolve())),
                "commissioning_profile_path": np.asarray(str(profile_path)),
                "commissioning_profile_sha256": np.asarray(profile_sha256),
                "commissioning_profile_id": np.asarray(
                    str(profile.get("profile_id", ""))
                ),
                "provider_output_mode": np.asarray(args.provider_output_mode),
                "rh56_snapshot_mode": np.asarray(hand_reader.snapshot_mode),
                "configured_rh56_read_rate_hz": np.asarray(
                    float(args.rh56_read_rate_hz), dtype=np.float64
                ),
                "camera_depth_transport_representation": np.asarray(
                    "owned_readonly_z16"
                ),
                "camera_depth_metric_conversion": np.asarray(
                    "masked_pixels_only_float32"
                ),
                "online_sam2_runtime_mode": np.asarray(
                    online_sam2_selection.mode
                ),
                "requested_object_mask_mode": np.asarray(
                    str(
                        online_sam2_selection.requested_object_mask_mode
                        or "config_default"
                    )
                ),
                "effective_object_mask_mode": np.asarray(
                    online_sam2_selection.effective_object_mask_mode
                ),
                "effective_provider_mask_publication_mode": np.asarray(
                    online_sam2_selection.effective_mask_publication_mode
                ),
                "effective_provider_recovery_publication_mode": np.asarray(
                    online_sam2_selection.effective_recovery_publication_mode
                ),
                "online_sam2_source_config_enabled": np.asarray(
                    online_sam2_selection.source_config_enabled
                ),
                "online_sam2_disable_override_requested": np.asarray(
                    online_sam2_selection.disable_override_requested
                ),
                "online_sam2_effective_config_enabled": np.asarray(
                    online_sam2_selection.effective_config_enabled
                ),
                "online_sam2_provider_enabled": np.asarray(
                    online_sam2_provider_enabled
                ),
                "online_sam2_manager_created": np.asarray(
                    online_sam2_manager_created
                ),
                "online_sam2_executor_created": np.asarray(
                    online_sam2_executor_created
                ),
                "discarded_policy_warmup_inferences": np.asarray(
                    policy_warmup_count, dtype=np.int32
                ),
                "configured_max_camera_age_s": np.asarray(
                    float(args.max_camera_age_s)
                ),
                "configured_max_camera_dropout_s": np.asarray(
                    float(args.max_camera_dropout_s)
                ),
                "configured_min_action_compute_reserve_s": np.asarray(
                    float(args.min_action_compute_reserve_s)
                ),
                "action_compute_reserve_warmup_factor": np.asarray(
                    ACTION_COMPUTE_RESERVE_WARMUP_FACTOR
                ),
                "discarded_policy_warmup_inference_s": np.asarray(
                    policy_warmup_inference_s
                ),
                "maximum_discarded_policy_warmup_inference_s": np.asarray(
                    maximum_policy_warmup_inference_s
                ),
                "discarded_observation_action_warmup_s": np.asarray(
                    observation_action_warmup_s
                ),
                "maximum_discarded_observation_action_warmup_s": np.asarray(
                    maximum_observation_action_warmup_s
                ),
                "action_compute_reserve_calibration_scope": np.asarray(
                    "freshness_decision_through_target_mapping_and_action_clock"
                ),
                "effective_action_compute_reserve_s": np.asarray(
                    action_compute_reserve_s
                ),
                "freshness_budget_rejected_frames": np.asarray(
                    freshness_frame_gate.budget_rejected_frames, dtype=np.int64
                ),
                "camera_transient_stale_rejected_frames": np.asarray(
                    freshness_frame_gate.transient_stale_rejected_frames,
                    dtype=np.int64,
                ),
                "camera_unusable_initial_observation_rejected_frames": np.asarray(
                    freshness_frame_gate.unusable_initial_observation_rejected_frames,
                    dtype=np.int64,
                ),
                "camera_freshness_rejected_frames": np.asarray(
                    freshness_frame_gate.rejected_frames, dtype=np.int64
                ),
                "freshness_budget_wait_ticks": np.asarray(
                    freshness_frame_gate.wait_ticks, dtype=np.int64
                ),
                "policy_rebootstrap_count": np.asarray(
                    freshness_frame_gate.rebootstrap_count, dtype=np.int64
                ),
                "camera_watchdog_frame_changes": np.asarray(
                    camera_frame_watchdog.frame_changes, dtype=np.int64
                ),
                "camera_watchdog_reuse_observations": np.asarray(
                    camera_frame_watchdog.reuse_observations, dtype=np.int64
                ),
                "camera_watchdog_max_unchanged_s": np.asarray(
                    camera_frame_watchdog.maximum_unchanged_s
                ),
                "configured_max_hand_age_s": np.asarray(float(args.max_hand_age_s)),
                "configured_max_pose_skew_s": np.asarray(
                    float(args.max_pose_skew_s)
                ),
                "configured_max_deadline_overrun_s": np.asarray(
                    float(args.max_deadline_overrun_s)
                ),
                "configured_compute_threads": np.asarray(
                    int(args.compute_threads), dtype=np.int32
                ),
                "opencv_threads_before": np.asarray(
                    compute_thread_guard.opencv_threads_before, dtype=np.int32
                ),
                "opencv_threads_active": np.asarray(
                    compute_thread_guard.opencv_threads_active, dtype=np.int32
                ),
                "blas_pools_before_json": np.asarray(
                    compute_thread_guard.blas_pools_before_json
                ),
                "blas_pools_active_json": np.asarray(
                    compute_thread_guard.blas_pools_active_json
                ),
                "blas_thread_limit_mechanism": np.asarray(
                    "threadpoolctl_runtime_scoped"
                ),
                "blas_environment_modified": np.asarray(False),
                "policy_q_home_rad": contract.q_home_rad.astype(np.float32, copy=True),
                "policy_joint_limits_rad": contract.joint_limits_rad.astype(
                    np.float32, copy=True
                ),
                "policy_q_hand_close_rad": contract.q_hand_close_rad.astype(
                    np.float32, copy=True
                ),
                "policy_T_flange_palm": contract.T_flange_policy_palm.astype(
                    np.float64, copy=True
                ),
            }
        )
        audit_payload.update(
            _stack_camera_rejection_diagnostics(camera_rejection_diagnostics)
        )
        audit_payload.update(
            _stack_policy_reuse_hold_diagnostics(
                policy_reuse_hold_diagnostics
            )
        )
        if args.output is not None:
            output_path = args.output.expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_path, **audit_payload)
        _shutdown_readonly_resources(
            stop=stop,
            threads=threads,
            camera_thread=camera_thread,
            franka_reader=franka_reader,
            provider=provider,
        )
        cleanup_complete = True
        compute_thread_guard.close()
        compute_thread_guard = None
        print(
            f"[read-only preview complete] steps={len(console_logs)}; no Franka "
            "control handle was created and no RH56 register was written",
            file=sys.stderr,
        )
        return 0
    except KeyboardInterrupt:
        return 130
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(
            f"[read-only preview failed] {type(exc).__name__}: {exc}", file=sys.stderr
        )
        return 2
    finally:
        if not cleanup_complete:
            try:
                _shutdown_readonly_resources(
                    stop=stop,
                    threads=threads,
                    camera_thread=camera_thread,
                    franka_reader=franka_reader,
                    provider=provider,
                )
            except BaseException as cleanup_exc:
                print(
                    "[read-only preview cleanup failed] "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                    file=sys.stderr,
                )
        if compute_thread_guard is not None:
            try:
                compute_thread_guard.close()
            except BaseException as thread_restore_exc:
                print(
                    "[read-only preview CPU thread restore failed] "
                    f"{type(thread_restore_exc).__name__}: {thread_restore_exc}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    raise SystemExit(main())
