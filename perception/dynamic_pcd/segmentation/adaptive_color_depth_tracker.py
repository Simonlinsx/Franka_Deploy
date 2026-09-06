from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from dynamic_pcd.segmentation.roi_depth_tracker import ROIDepthTracker, ROITrackerState
from dynamic_pcd.types import MaskResult, RGBDFrame
from dynamic_pcd.utils.geometry import (
    bbox_area,
    bbox_from_mask,
    bbox_to_mask,
    clip_bbox,
    depth_stats,
    enlarge_bbox,
)


def _binary_mask_iou(first: np.ndarray, second: np.ndarray) -> Tuple[float, int, int]:
    """Return IoU and foreground areas using exact ``mask > 0`` semantics."""

    def as_binary_u8(value: np.ndarray) -> np.ndarray:
        array = np.asarray(value)
        if array.dtype == np.bool_:
            # NumPy bool arrays are byte-sized 0/1 values.  The view avoids a
            # full-frame copy when callers already applied ``mask > 0``.
            return array.view(np.uint8)
        if array.dtype == np.uint8:
            # Threshold in OpenCV keeps arbitrary positive uint8 masks exact
            # while avoiding NumPy's temporary bool plus uint8 conversion.
            return cv2.threshold(array, 0, 1, cv2.THRESH_BINARY)[1]
        return (array > 0).astype(np.uint8)

    first_u8 = as_binary_u8(first)
    second_u8 = as_binary_u8(second)
    first_area = int(cv2.countNonZero(first_u8))
    second_area = int(cv2.countNonZero(second_u8))
    intersection = int(cv2.countNonZero(cv2.bitwise_and(first_u8, second_u8)))
    union = first_area + second_area - intersection
    iou = 0.0 if union <= 0 else float(intersection / union)
    return iou, first_area, second_area


@dataclass
class AdaptiveTrackerSnapshot:
    """Complete rollback state for guarded re-detection and identity rejection."""

    state: Optional[ROITrackerState]
    fixed_bbox_xyxy: Optional[np.ndarray]
    anchor_hs: Optional[np.ndarray]
    anchor_sv: Optional[np.ndarray]
    adaptive_hs: Optional[np.ndarray]
    adaptive_sv: Optional[np.ndarray]
    background_hs: Optional[np.ndarray]
    background_sv: Optional[np.ndarray]
    depth_half_width: float
    confidence: float
    previous_gray: Optional[np.ndarray]
    temporal_mask_score: Optional[np.ndarray]
    motion_mask_iou: float
    temporal_mask_stabilized: bool
    temporal_motion_source: str
    recovery_hypothesis: Optional["_RecoveryHypothesis"]
    maneuver_hypothesis: Optional["_ManeuverHypothesis"]
    appearance_freeze_remaining: int
    committed_motion: Optional["_CommittedMotionState"]


@dataclass(frozen=True)
class TargetAppearanceProbabilityROI:
    """Immutable, frame-bound target-appearance probability evidence.

    The probability image is evaluated once over the bounding rectangle of
    ``root_mask``.  Read-only appearance queries for that mask, or any strict
    subset of it, can reuse the result without repeating BGR-to-HSV conversion
    and histogram back-projection.  The evidence is bound to the exact camera
    frame timestamp bits and to a monotonic appearance-model revision; stale
    evidence is therefore never reused after online appearance adaptation or
    tracker rollback.

    ``root_mask`` and ``probability`` are backed by immutable ``bytes`` and
    are consequently non-writeable NumPy views.  ``root_mask_digest`` gives
    provider-level caches an exact content identity without weakening the
    tracker-side subset check.
    """

    frame_id: int
    timestamp_f64_bits: int
    appearance_revision: int
    image_shape_hw: Tuple[int, int]
    root_mask_digest: str
    bbox_xyxy: Tuple[int, int, int, int]
    root_mask: np.ndarray = field(repr=False, compare=False)
    probability: np.ndarray = field(repr=False, compare=False)
    root_mask_bytes: bytes = field(repr=False, compare=False)
    probability_bytes: bytes = field(repr=False, compare=False)
    _tracker_token: object = field(repr=False, compare=False)


@dataclass
class _ComponentCandidate:
    mask: np.ndarray
    bbox_xyxy: np.ndarray
    center_uv: Tuple[float, float]
    area: float
    depth_median: float
    confidence: float
    color_score: float
    search_scope: str = "tracking"
    motion_conflict_reason: str = ""


@dataclass
class _FlowPrediction:
    mask: np.ndarray
    bbox_xyxy: np.ndarray
    center_uv: Tuple[float, float]
    confidence: float
    affine_matrix: np.ndarray
    message: str


@dataclass
class _RecoveryHypothesis:
    """One fail-closed, not-yet-published lost-target hypothesis."""

    mask: np.ndarray
    bbox_xyxy: np.ndarray
    center_uv: Tuple[float, float]
    area: float
    depth_median: float
    confidence: float
    hits: int
    last_frame_id: int
    timestamp_s: float


@dataclass
class _ManeuverHypothesis:
    """Unpublished evidence that the target made a real abrupt maneuver."""

    mask: np.ndarray
    bbox_xyxy: np.ndarray
    center_uv: Tuple[float, float]
    area: float
    depth_median: float
    confidence: float
    hits: int
    last_frame_id: int
    timestamp_s: float
    velocity_uv_s: Optional[Tuple[float, float]] = None


@dataclass
class _CommittedMotionState:
    """Motion learned only from identity-gated, accepted target observations.

    This state is deliberately independent from ``ROITrackerState``.  The
    latter may advance internally during publication probation or a guarded
    SAM reinitialization.  Neither a tentative recovery mask nor an occluder
    is therefore able to teach the constant-velocity predictor.
    """

    center_uv: Tuple[float, float]
    bbox_xyxy: np.ndarray
    timestamp_s: float
    frame_id: int
    velocity_uv_s: Tuple[float, float]
    velocity_samples: int


class AdaptiveColorDepthTracker(ROIDepthTracker):
    """High-rate single-object tracker learned from an external prompt mask.

    This component deliberately is not a prompt detector.  A manual box,
    SAM2, or an open-vocabulary detector supplies the initial mask.  The fast
    path then intersects a learned, target-vs-background HSV likelihood with a
    depth band and selects one temporally consistent connected component.

    The initial appearance model remains an immutable anchor.  A second model
    adapts only on high-confidence frames, which absorbs moderate illumination
    changes without drifting onto the table or a distractor.  Any failed gate
    returns an empty mask and ``valid=False`` immediately (fail closed).
    """

    manages_identity = True

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        # A revision is intentionally monotonic for the lifetime of this
        # tracker.  Snapshot restore must invalidate, not resurrect, any
        # probability evidence computed against an older appearance model.
        self._appearance_revision = 0
        self._appearance_probability_roi_token = object()
        self.anchor_hs: Optional[np.ndarray] = None
        self.anchor_sv: Optional[np.ndarray] = None
        self.adaptive_hs: Optional[np.ndarray] = None
        self.adaptive_sv: Optional[np.ndarray] = None
        self.background_hs: Optional[np.ndarray] = None
        self.background_sv: Optional[np.ndarray] = None
        self.depth_half_width = float(cfg.get("depth_tolerance", 0.04))
        self.last_confidence = 0.0
        self.previous_gray: Optional[np.ndarray] = None
        # Soft, motion-compensated mask evidence.  A binary previous mask is
        # not enough to distinguish a one-frame edge flicker from a persistent
        # silhouette change: once a noisy pixel enters, a binary retain gate
        # can keep it forever.  This score provides symmetric two-frame
        # confirmation while the affine flow removes image-space motion.
        self.temporal_mask_score: Optional[np.ndarray] = None
        self.last_motion_mask_iou = 0.0
        self.last_temporal_stabilized = False
        self.last_temporal_motion_source = "none"
        self.recovery_hypothesis: Optional[_RecoveryHypothesis] = None
        self.maneuver_hypothesis: Optional[_ManeuverHypothesis] = None
        self.appearance_freeze_remaining = 0
        self.committed_motion: Optional[_CommittedMotionState] = None

    @property
    def confidence(self) -> float:
        return float(self.last_confidence)

    @property
    def appearance_revision(self) -> int:
        """Monotonic identity of the currently published appearance model."""

        return int(self._appearance_revision)

    def _advance_appearance_revision(self) -> None:
        self._appearance_revision += 1

    @property
    def lost(self) -> bool:
        return self.state is None or not bool(self.state.valid)

    @property
    def tracking_status(self) -> Dict[str, Any]:
        return {
            "valid": not self.lost,
            "confidence": self.confidence,
            "appearance_revision": self.appearance_revision,
            "lost_count": int(self.state.lost_count) if self.state is not None else 0,
            "recovery_pending_count": (
                int(self.recovery_hypothesis.hits)
                if self.recovery_hypothesis is not None
                else 0
            ),
            "maneuver_pending_count": (
                int(self.maneuver_hypothesis.hits)
                if self.maneuver_hypothesis is not None
                else 0
            ),
            "motion_mask_iou": float(self.last_motion_mask_iou),
            "temporal_mask_stabilized": bool(self.last_temporal_stabilized),
            "temporal_motion_source": str(self.last_temporal_motion_source),
            "committed_velocity_uv_px_s": (
                tuple(float(value) for value in self.committed_motion.velocity_uv_s)
                if self.committed_motion is not None
                else (0.0, 0.0)
            ),
            "committed_velocity_samples": (
                int(self.committed_motion.velocity_samples)
                if self.committed_motion is not None
                else 0
            ),
            "bbox_xyxy": (
                self.state.bbox_xyxy.copy() if self.state is not None else None
            ),
        }

    def initialize(
        self,
        frame: RGBDFrame,
        mask: Optional[np.ndarray] = None,
        bbox_xyxy: Optional[np.ndarray] = None,
    ) -> MaskResult:
        box_prompt_only = mask is None
        result = super().initialize(frame, mask=mask, bbox_xyxy=bbox_xyxy)
        if self.state is None:
            raise RuntimeError("adaptive tracker initialization produced no state")

        foreground_core = self._appearance_training_mask(
            result.mask, result.bbox_xyxy, box_prompt_only=box_prompt_only
        )
        self._initialize_appearance(
            frame,
            foreground_core=foreground_core,
            foreground_exclusion=result.mask,
            bbox=result.bbox_xyxy,
        )

        z5, _, z95 = self._configured_depth_stats(frame, result.mask)
        configured = float(self.cfg.get("depth_tolerance", 0.04))
        spread_margin = float(self.cfg.get("depth_spread_margin", 0.008))
        max_half_width = float(self.cfg.get("max_depth_tolerance", 0.08))
        observed_half_width = max(0.0, 0.5 * (z95 - z5) + spread_margin)
        self.depth_half_width = min(
            max_half_width, max(configured, observed_half_width)
        )
        self.last_confidence = 1.0
        self.state.valid = True
        self.previous_gray = cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2GRAY)
        self.temporal_mask_score = result.mask.astype(np.float32)
        self.last_motion_mask_iou = 1.0
        self.last_temporal_stabilized = False
        self.last_temporal_motion_source = "init"
        self.recovery_hypothesis = None
        self.maneuver_hypothesis = None
        self.appearance_freeze_remaining = 0
        initial_y, initial_x = np.nonzero(np.asarray(result.mask) > 0)
        committed_center = (
            (float(initial_x.mean()), float(initial_y.mean()))
            if initial_x.size > 0
            else (
                float(self.state.center_uv[0]),
                float(self.state.center_uv[1]),
            )
        )
        self.committed_motion = _CommittedMotionState(
            center_uv=(
                float(committed_center[0]),
                float(committed_center[1]),
            ),
            bbox_xyxy=self.state.bbox_xyxy.astype(np.int32).copy(),
            timestamp_s=float(frame.timestamp),
            frame_id=int(frame.frame_id),
            velocity_uv_s=(0.0, 0.0),
            velocity_samples=0,
        )
        result.score = 1.0
        result.message = (
            f"adaptive init z={self.state.depth_median:.3f}m "
            f"area={int(self.state.area)} depth_band=+/-{self.depth_half_width:.3f}m"
        )
        return result

    def update(self, frame: RGBDFrame) -> MaskResult:
        if self.state is None:
            raise RuntimeError("tracker not initialized")
        if self.anchor_hs is None or self.background_hs is None:
            return self._reject(frame, "appearance model unavailable")

        current_gray = cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2GRAY)
        previous_lost_count = int(self.state.lost_count)
        flow_bridged_recovery = False

        # Once a frame has been rejected, the last gray image and mask belong
        # to the last *visible* target.  A KLT result computed across a complete
        # occlusion can still be non-None while actually following the hand or
        # background texture.  Always try the learned appearance/depth recovery
        # path first while LOST.  A one-frame KLT grace path remains available
        # only as a fallback for a briefly missing, low-colour-texture target.
        flow = None
        if previous_lost_count == 0:
            flow = self._predict_with_optical_flow(current_gray)
            candidate, reason = self._best_component(frame, flow)
        else:
            candidate, reason = self._best_component(frame, None)
            flow_grace = max(
                0, int(self.cfg.get("occlusion_recovery_flow_grace_frames", 2))
            )
            if candidate is None and previous_lost_count <= flow_grace:
                grace_flow = self._predict_with_optical_flow(current_gray)
                min_flow_confidence = float(
                    self.cfg.get("component_flow_motion_min_confidence", 0.45)
                )
                if (
                    grace_flow is not None
                    and float(getattr(grace_flow, "confidence", 0.0))
                    >= min_flow_confidence
                ):
                    flow_candidate, flow_reason = self._best_component(
                        frame, grace_flow
                    )
                    if flow_candidate is not None:
                        candidate = flow_candidate
                        flow = grace_flow
                        flow_bridged_recovery = True
                    else:
                        reason = f"{reason}; flow fallback: {flow_reason}"
        if candidate is None:
            self._expire_recovery_hypothesis(frame.frame_id, reason)
            self.maneuver_hypothesis = None
            return self._reject(frame, reason)

        maneuver_recovered = False
        motion_conflict_reason = str(getattr(candidate, "motion_conflict_reason", ""))
        if motion_conflict_reason:
            if not bool(self.cfg.get("component_maneuver_recovery_enabled", True)):
                self.maneuver_hypothesis = None
                return self._reject(frame, motion_conflict_reason)
            maneuver_recovered, maneuver_message = self._confirm_maneuver(
                frame, candidate
            )
            if not maneuver_recovered:
                return self._reject(frame, maneuver_message)
            # The old-direction KLT/soft-mask state is intentionally bypassed
            # on the commit frame. The three raw candidates already passed
            # identity, size, depth and uniqueness gates.
            flow = None
            flow_bridged_recovery = False
        else:
            self.maneuver_hypothesis = None

        appearance_candidate = candidate
        appearance_flow = flow
        if previous_lost_count == 0:
            # Smooth, texture-poor objects often have fewer than four KLT
            # corners.  Once the raw component has independently passed all
            # color/depth/identity gates, its centroid and scale provide a
            # bounded translation/scale transform for mask stabilization.
            # This fallback cannot run when the object is absent because there
            # is no accepted candidate in that case.
            if flow is None:
                flow = self._geometry_flow_from_candidate(candidate)
            if flow is not None:
                consensus_error = self._tracked_motion_consensus_error(candidate, flow)
                if consensus_error is not None:
                    return self._reject(frame, consensus_error)
                candidate = self._stabilize_tracked_candidate(frame, candidate, flow)
            else:
                # No trustworthy motion transform means that a soft mask from
                # the previous coordinate system must not be blended in.
                self.temporal_mask_score = candidate.mask.astype(np.float32)
                self.last_motion_mask_iou = 0.0
                self.last_temporal_stabilized = False
                self.last_temporal_motion_source = "none"

        recovered_after_loss = bool(maneuver_recovered)
        if previous_lost_count > 0:
            if not bool(self.cfg.get("occlusion_recovery_enabled", True)):
                self.recovery_hypothesis = None
                return self._reject(frame, "lost recovery is disabled")
            if maneuver_recovered:
                self.recovery_hypothesis = None
            elif flow_bridged_recovery:
                # A reliable KLT bridge is tied to pixels from the last visible
                # target and the candidate has just re-passed the ordinary
                # immutable size/appearance/depth gates. Commit it internally
                # on this exact frame so tracking can continue, but rely on the
                # provider's source-agnostic multi-frame quarantine before any
                # recovered mask/PCD becomes public.
                self.recovery_hypothesis = None
            else:
                confirmed, confirmation_message = self._confirm_occlusion_recovery(
                    frame, candidate
                )
                if not confirmed:
                    return self._reject(frame, confirmation_message)
            recovered_after_loss = True
            self.temporal_mask_score = candidate.mask.astype(np.float32)
            self.last_motion_mask_iou = 1.0
            self.last_temporal_stabilized = False
            self.last_temporal_motion_source = (
                "maneuver"
                if maneuver_recovered
                else ("recovery-klt" if flow_bridged_recovery else "recovery")
            )

        base_motion_limit = float(self.cfg.get("component_max_motion_px", 24.0))
        raw_tracked_motion = float(
            np.hypot(
                appearance_candidate.center_uv[0] - self.state.center_uv[0],
                appearance_candidate.center_uv[1] - self.state.center_uv[1],
            )
        )
        reliable_flow = bool(
            appearance_flow is not None
            and float(getattr(appearance_flow, "confidence", 0.0))
            >= float(self.cfg.get("component_flow_motion_min_confidence", 0.45))
        )
        # A >base displacement can be physically real and KLT-supported, but
        # appearance/depth alone cannot prove that an identical neighbouring
        # instance did not replace the target.  Advance the internal temporal
        # state so the next exact frame can continue tracking, while returning
        # an empty public mask for this first large-motion observation.  The
        # provider then applies its multi-frame publication quarantine before
        # any point cloud can reach a viewer or robot.
        motion_probation = bool(
            not self._time_aware_motion_enabled()
            and previous_lost_count == 0
            and base_motion_limit > 0.0
            and raw_tracked_motion > base_motion_limit
            and reliable_flow
        )

        alpha = float(self.cfg.get("state_update_alpha", 0.65))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        self.state.depth_median = (
            alpha * candidate.depth_median + (1.0 - alpha) * self.state.depth_median
        )
        self.state.center_uv = (
            alpha * candidate.center_uv[0] + (1.0 - alpha) * self.state.center_uv[0],
            alpha * candidate.center_uv[1] + (1.0 - alpha) * self.state.center_uv[1],
        )
        self.state.bbox_xyxy = candidate.bbox_xyxy.astype(np.int32)
        self.state.area = candidate.area
        self.state.last_mask = candidate.mask.copy()
        self.state.lost_count = 0
        self.state.valid = True
        self.last_confidence = candidate.confidence
        self.previous_gray = current_gray
        self.recovery_hypothesis = None

        if motion_probation:
            self.appearance_freeze_remaining = max(
                self.appearance_freeze_remaining,
                max(
                    1,
                    int(self.cfg.get("motion_probation_appearance_freeze_frames", 3)),
                ),
            )
            self.last_confidence = 0.0
            h, w = frame.depth_raw.shape
            return MaskResult(
                mask=np.zeros((h, w), dtype=np.uint8),
                bbox_xyxy=candidate.bbox_xyxy.copy(),
                score=0.0,
                valid=False,
                message=(
                    "adaptive motion probation: "
                    f"motion={raw_tracked_motion:.1f}px "
                    f"flow_conf={float(appearance_flow.confidence):.3f}; "
                    "current point cloud quarantined"
                ),
            )

        if previous_lost_count > 0:
            self.appearance_freeze_remaining = max(
                0,
                int(self.cfg.get("occlusion_recovery_appearance_freeze_frames", 5)),
            )
        elif self.appearance_freeze_remaining > 0:
            self.appearance_freeze_remaining -= 1

        update_min = float(self.cfg.get("appearance_update_min_confidence", 0.82))
        if (
            previous_lost_count == 0
            and self.appearance_freeze_remaining == 0
            and candidate.confidence >= update_min
        ):
            # Learn only the eroded current/flow consensus core. Updating from
            # the entire candidate made a weak flow halo self-reinforcing:
            # boundary pixels entered the appearance model, then expanded the
            # next frame's mask again. The core still adapts to illumination
            # without teaching the tracker its own uncertain boundary.
            update_mask = self._appearance_update_core(
                appearance_candidate, appearance_flow
            )
            if update_mask is not None:
                self._update_adaptive_appearance(frame, update_mask)

        # Update the predictor only after every ordinary identity gate has
        # accepted this observation.  Pending recovery hypotheses, empty
        # sensor frames, motion probation and rejected occluders return above
        # and therefore cannot poison the learned velocity.
        self._commit_motion_observation(
            frame,
            candidate.center_uv,
            candidate.bbox_xyxy,
        )

        return MaskResult(
            mask=candidate.mask,
            bbox_xyxy=candidate.bbox_xyxy,
            score=candidate.confidence,
            valid=True,
            message=(
                f"adaptive "
                f"{'recovered-' + candidate.search_scope if recovered_after_loss else 'track'} "
                f"conf={candidate.confidence:.3f} "
                f"color={candidate.color_score:.3f} "
                f"z={self.state.depth_median:.3f}m area={int(candidate.area)}"
            ),
        )

    def snapshot_state(self) -> AdaptiveTrackerSnapshot:
        return AdaptiveTrackerSnapshot(
            state=copy.deepcopy(self.state),
            fixed_bbox_xyxy=self._copy_array(self.fixed_bbox_xyxy),
            anchor_hs=self._copy_array(self.anchor_hs),
            anchor_sv=self._copy_array(self.anchor_sv),
            adaptive_hs=self._copy_array(self.adaptive_hs),
            adaptive_sv=self._copy_array(self.adaptive_sv),
            background_hs=self._copy_array(self.background_hs),
            background_sv=self._copy_array(self.background_sv),
            depth_half_width=float(self.depth_half_width),
            confidence=float(self.last_confidence),
            previous_gray=self._copy_array(self.previous_gray),
            temporal_mask_score=self._copy_array(self.temporal_mask_score),
            motion_mask_iou=float(self.last_motion_mask_iou),
            temporal_mask_stabilized=bool(self.last_temporal_stabilized),
            temporal_motion_source=str(self.last_temporal_motion_source),
            recovery_hypothesis=copy.deepcopy(self.recovery_hypothesis),
            maneuver_hypothesis=copy.deepcopy(self.maneuver_hypothesis),
            appearance_freeze_remaining=int(self.appearance_freeze_remaining),
            committed_motion=copy.deepcopy(self.committed_motion),
        )

    def restore_state(self, state: Optional[AdaptiveTrackerSnapshot]) -> None:
        if state is None:
            self.state = None
            self.fixed_bbox_xyxy = None
            self.anchor_hs = None
            self.anchor_sv = None
            self.adaptive_hs = None
            self.adaptive_sv = None
            self.background_hs = None
            self.background_sv = None
            self.previous_gray = None
            self.temporal_mask_score = None
            self.last_motion_mask_iou = 0.0
            self.last_temporal_stabilized = False
            self.last_temporal_motion_source = "none"
            self.last_confidence = 0.0
            self.recovery_hypothesis = None
            self.maneuver_hypothesis = None
            self.appearance_freeze_remaining = 0
            self.committed_motion = None
            self._advance_appearance_revision()
            return
        if not isinstance(state, AdaptiveTrackerSnapshot):
            # Compatibility with callers that saved only the legacy state.
            self.state = copy.deepcopy(state)
            self.temporal_mask_score = (
                None
                if self.state is None or self.state.last_mask is None
                else self.state.last_mask.astype(np.float32)
            )
            self.last_motion_mask_iou = 0.0
            self.last_temporal_stabilized = False
            self.last_temporal_motion_source = "legacy_restore"
            self.recovery_hypothesis = None
            self.maneuver_hypothesis = None
            self.appearance_freeze_remaining = 0
            self.committed_motion = None
            return
        self.state = copy.deepcopy(state.state)
        self.fixed_bbox_xyxy = self._copy_array(state.fixed_bbox_xyxy)
        self.anchor_hs = self._copy_array(state.anchor_hs)
        self.anchor_sv = self._copy_array(state.anchor_sv)
        self.adaptive_hs = self._copy_array(state.adaptive_hs)
        self.adaptive_sv = self._copy_array(state.adaptive_sv)
        self.background_hs = self._copy_array(state.background_hs)
        self.background_sv = self._copy_array(state.background_sv)
        self.depth_half_width = float(state.depth_half_width)
        self.last_confidence = float(state.confidence)
        self.previous_gray = self._copy_array(state.previous_gray)
        self.temporal_mask_score = self._copy_array(state.temporal_mask_score)
        self.last_motion_mask_iou = float(state.motion_mask_iou)
        self.last_temporal_stabilized = bool(state.temporal_mask_stabilized)
        self.last_temporal_motion_source = str(state.temporal_motion_source)
        self.recovery_hypothesis = copy.deepcopy(state.recovery_hypothesis)
        self.maneuver_hypothesis = copy.deepcopy(state.maneuver_hypothesis)
        self.appearance_freeze_remaining = int(state.appearance_freeze_remaining)
        self.committed_motion = copy.deepcopy(state.committed_motion)
        # Never restore a revision number from the snapshot: doing so would
        # make pre-restore evidence appear current again (an ABA bug).
        self._advance_appearance_revision()

    def reinitialize_with_mask(
        self,
        frame: RGBDFrame,
        mask: np.ndarray,
        *,
        max_depth_jump_m: Optional[float] = None,
        update_adaptive: bool = False,
    ) -> MaskResult:
        """Guarded same-target re-detection from a SAM/open-vocabulary mask.

        Use :meth:`initialize` for an intentional user-requested target change.
        This method retains the original appearance anchor and rejects a
        re-detection that does not look like the tracked target.  Stateful
        per-frame segmenters should use the default strict depth gate.  A
        category-level semantic detector may explicitly override it because a
        user can carry the prompted object to another camera depth while it is
        completely hidden; ``0`` disables only that raw-depth gate, while the
        immutable appearance and size gates remain mandatory.
        """

        if self.state is None:
            return self.initialize(frame, mask=mask)
        snapshot = self.snapshot_state()
        mask_u8 = (mask > 0).astype(np.uint8)
        bbox = bbox_from_mask(mask_u8, min_area=int(self.cfg.get("min_area", 80)))
        if bbox is None:
            return self._invalid_from_snapshot(frame, snapshot, "reinit empty mask")

        color_score = self._mean_color_probability(frame, mask_u8, bbox)
        reinit_min = float(self.cfg.get("reinit_min_color_score", 0.62))
        if color_score < reinit_min:
            return self._invalid_from_snapshot(
                frame,
                snapshot,
                f"reinit color={color_score:.3f}<{reinit_min:.3f}",
            )

        area = float(mask_u8.sum())
        reject_reason = self._size_reject_reason(area, bbox, check_last=False)
        if reject_reason:
            return self._invalid_from_snapshot(
                frame, snapshot, f"reinit {reject_reason}"
            )

        if snapshot.state is not None:
            x1, y1, x2, y2 = bbox.astype(float)
            center = (0.5 * (x1 + x2), 0.5 * (y1 + y2))
            center_jump = float(
                np.hypot(
                    center[0] - snapshot.state.center_uv[0],
                    center[1] - snapshot.state.center_uv[1],
                )
            )
            max_center_jump = float(self.cfg.get("reinit_max_center_jump_px", 90.0))
            if max_center_jump > 0.0 and center_jump > max_center_jump:
                return self._invalid_from_snapshot(
                    frame,
                    snapshot,
                    f"reinit center jump {center_jump:.1f}>" f"{max_center_jump:.1f}px",
                )

            _, candidate_depth, _ = depth_stats(frame.depth_m, mask_u8)
            if candidate_depth <= 0.0:
                return self._invalid_from_snapshot(
                    frame, snapshot, "reinit has no valid depth"
                )
            max_depth_jump = float(
                self.cfg.get("reinit_max_depth_jump_m", 0.12)
                if max_depth_jump_m is None
                else max_depth_jump_m
            )
            depth_jump = abs(candidate_depth - snapshot.state.depth_median)
            if max_depth_jump > 0.0 and depth_jump > max_depth_jump:
                return self._invalid_from_snapshot(
                    frame,
                    snapshot,
                    f"reinit depth jump {depth_jump:.3f}>" f"{max_depth_jump:.3f}m",
                )

        result = ROIDepthTracker.initialize(self, frame, mask=mask_u8)
        if self.state is None:
            return self._invalid_from_snapshot(frame, snapshot, "reinit no state")
        if snapshot.state is not None:
            self.state.initial_area = snapshot.state.initial_area
            self.state.initial_bbox_area = snapshot.state.initial_bbox_area
        # Preserve the immutable target/background models.  Only the adaptive
        # foreground model may learn from this already-validated re-detection.
        self.anchor_hs = self._copy_array(snapshot.anchor_hs)
        self.anchor_sv = self._copy_array(snapshot.anchor_sv)
        self.adaptive_hs = self._copy_array(snapshot.adaptive_hs)
        self.adaptive_sv = self._copy_array(snapshot.adaptive_sv)
        self.background_hs = self._copy_array(snapshot.background_hs)
        self.background_sv = self._copy_array(snapshot.background_sv)
        self.depth_half_width = snapshot.depth_half_width
        self.last_confidence = color_score
        self.previous_gray = cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2GRAY)
        self.temporal_mask_score = mask_u8.astype(np.float32)
        self.last_motion_mask_iou = 1.0
        self.last_temporal_stabilized = False
        self.last_temporal_motion_source = "reinit"
        self.recovery_hypothesis = None
        self.maneuver_hypothesis = None
        self.appearance_freeze_remaining = max(
            0,
            int(self.cfg.get("occlusion_recovery_appearance_freeze_frames", 5)),
        )
        # Re-detections remain provisional until the provider-level recovery
        # quarantine commits. Never let a rejected SAM/prompt observation
        # modify appearance by default; normal tracked core updates can adapt
        # safely on subsequent frames.
        if update_adaptive:
            self._update_adaptive_appearance(frame, mask_u8, alpha_override=0.10)
        result.score = color_score
        result.message = f"adaptive reinit color={color_score:.3f}"
        return result

    def reinitialize_from_bbox(
        self, frame: RGBDFrame, bbox_xyxy: np.ndarray
    ) -> MaskResult:
        h, w = frame.depth_raw.shape
        bbox = clip_bbox(bbox_xyxy, w, h)
        return self.reinitialize_with_mask(frame, bbox_to_mask(bbox, (h, w)))

    def _best_component(
        self, frame: RGBDFrame, flow: Optional[_FlowPrediction]
    ) -> Tuple[Optional[_ComponentCandidate], str]:
        assert self.state is not None
        h, w = frame.depth_raw.shape
        lost_without_flow = self.state.lost_count > 0 and flow is None
        fast_recovery = lost_without_flow and bool(
            self.cfg.get("occlusion_recovery_enabled", True)
        )
        global_recovery = bool(
            fast_recovery
            and not self.roi_locked
            and self.cfg.get("occlusion_global_search_enabled", True)
            and self.state.lost_count
            >= max(
                1,
                int(self.cfg.get("occlusion_global_search_after_frames", 2)),
            )
        )
        if self.roi_locked and self.fixed_bbox_xyxy is not None:
            roi = clip_bbox(self.fixed_bbox_xyxy, w, h)
        elif global_recovery:
            roi = np.asarray([0, 0, w, h], dtype=np.int32)
        elif fast_recovery:
            if self._time_aware_motion_enabled():
                roi = self._time_aware_search_bbox(frame, w, h)
            else:
                roi = enlarge_bbox(
                    self.state.bbox_xyxy,
                    float(self.cfg.get("occlusion_recovery_search_scale", 4.0)),
                    w,
                    h,
                )
        else:
            base_scale = float(self.cfg.get("roi_scale", 2.2))
            loss_growth = float(self.cfg.get("lost_roi_growth", 0.25))
            max_scale = float(self.cfg.get("max_roi_scale", 4.0))
            scale = min(
                max_scale,
                base_scale * (1.0 + loss_growth * min(self.state.lost_count, 4)),
            )
            search_anchor = flow.bbox_xyxy if flow is not None else self.state.bbox_xyxy
            roi = enlarge_bbox(search_anchor, scale, w, h)
            if self._time_aware_motion_enabled() and flow is None:
                predicted_roi = self._time_aware_search_bbox(frame, w, h)
                roi = clip_bbox(
                    np.asarray(
                        [
                            min(int(roi[0]), int(predicted_roi[0])),
                            min(int(roi[1]), int(predicted_roi[1])),
                            max(int(roi[2]), int(predicted_roi[2])),
                            max(int(roi[3]), int(predicted_roi[3])),
                        ],
                        dtype=np.int32,
                    ),
                    w,
                    h,
                )
        x1, y1, x2, y2 = roi.astype(int)

        color_crop = frame.color_bgr[y1:y2, x1:x2]
        # Convert only the search ROI to metric depth. ``frame.depth_m``
        # materializes the complete 848x480 float image even though ordinary
        # tracking searches a small bbox neighborhood.  Elementwise conversion
        # after slicing is numerically identical for every consumed pixel.
        depth_crop = frame.depth_raw[y1:y2, x1:x2].astype(np.float32) * float(
            frame.depth_scale
        )
        if color_crop.size == 0 or depth_crop.size == 0:
            return None, "empty search ROI"
        hsv_crop = cv2.cvtColor(color_crop, cv2.COLOR_BGR2HSV)
        probability = self._color_probability(hsv_crop)

        if global_recovery:
            threshold = float(
                self.cfg.get(
                    "occlusion_global_color_probability_threshold",
                    self.cfg.get(
                        "occlusion_recovery_color_probability_threshold",
                        self.cfg.get("color_probability_threshold", 0.56),
                    ),
                )
            )
        elif fast_recovery:
            threshold = float(
                self.cfg.get(
                    "occlusion_recovery_color_probability_threshold",
                    self.cfg.get("color_probability_threshold", 0.56),
                )
            )
        else:
            threshold = float(self.cfg.get("color_probability_threshold", 0.56))
        valid_depth = np.isfinite(depth_crop) & (depth_crop > 0.0)
        recovery_depth_tolerance = float(
            self.cfg.get(
                "occlusion_global_depth_tolerance_m",
                max(self.depth_half_width, 0.10),
            )
            if global_recovery
            else self.depth_half_width
        )
        depth_ok = valid_depth & (
            np.abs(depth_crop - self.state.depth_median) <= recovery_depth_tolerance
        )
        strong_color = probability >= threshold
        if flow is not None:
            flow_crop = flow.mask[y1:y2, x1:x2]
            # Motion-compensated Schmitt trigger: a predicted target pixel may
            # remain at the weak threshold, while a brand-new pixel must pass
            # a stricter entry threshold.  The previous 5x5 dilated weak mask
            # created a 2-pixel halo which was written back and expanded again
            # on every frame.
            flow_support = flow_crop > 0
            weak_floor = float(self.cfg.get("flow_color_probability_floor", 0.42))
            enter_threshold = float(
                self.cfg.get("flow_enter_color_probability_threshold", 0.64)
            )
            appearance_ok = (flow_support & (probability >= weak_floor)) | (
                (~flow_support) & (probability >= enter_threshold)
            )
        else:
            flow_crop = None
            appearance_ok = strong_color
        # Keep semantic silhouette extraction separate from depth validity.
        # In deployment the D435 can report background/invalid depth on a
        # reflective object edge even while RGB and SAM2 still see that edge.
        # Pixel-wise ``appearance & depth`` therefore eroded a correct object
        # mask (notably the visible top of the sphere).  Depth still gates the
        # *whole component* below and the policy projector independently
        # rejects unusable depth; it must not rewrite semantic membership.
        depth_pixel_gate = bool(
            self.cfg.get("semantic_mask_depth_pixel_gate_enabled", True)
        )
        candidate_mask = (
            appearance_ok.astype(np.uint8)
            if fast_recovery or not depth_pixel_gate
            else (appearance_ok & depth_ok).astype(np.uint8)
        )

        morph_kernel = int(self.cfg.get("morph_kernel", 3))
        if morph_kernel > 1:
            kernel = np.ones((morph_kernel, morph_kernel), dtype=np.uint8)
            candidate_mask = cv2.morphologyEx(
                candidate_mask, cv2.MORPH_OPEN, kernel, iterations=1
            )
            candidate_mask = cv2.morphologyEx(
                candidate_mask, cv2.MORPH_CLOSE, kernel, iterations=1
            )

        if global_recovery and bool(
            self.cfg.get("occlusion_fragment_merge_enabled", True)
        ):
            candidate_mask = self._merge_global_recovery_fragments(
                candidate_mask,
                depth_crop,
                valid_depth,
            )

        num, labels, stats, centroids = cv2.connectedComponentsWithStats(
            candidate_mask, connectivity=8
        )
        min_area = int(self.cfg.get("min_area", 80))
        candidates = []
        rejection = "no color+depth component"
        for label in range(1, num):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            local = labels == label
            bx = int(stats[label, cv2.CC_STAT_LEFT]) + x1
            by = int(stats[label, cv2.CC_STAT_TOP]) + y1
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            bbox = np.asarray([bx, by, bx + bw, by + bh], dtype=np.int32)
            cx = float(centroids[label, 0]) + x1
            cy = float(centroids[label, 1]) + y1
            motion_conflict_reason = ""

            if fast_recovery:
                size_reason = self._occlusion_size_reject_reason(
                    area,
                    bbox,
                    check_last_position=(
                        not global_recovery and not self._time_aware_motion_enabled()
                    ),
                    global_recovery=global_recovery,
                )
                if size_reason:
                    rejection = size_reason
                    continue
            else:
                size_reason = self._component_size_reject_reason(area, bbox)
                if size_reason:
                    rejection = size_reason
                    continue

            predicted_center = self._motion_predicted_center(frame)
            expected_center = (
                flow.center_uv
                if flow is not None
                else (
                    predicted_center
                    if predicted_center is not None
                    else self.state.center_uv
                )
            )
            dx = cx - expected_center[0]
            dy = cy - expected_center[1]
            center_jump = float(np.hypot(dx, dy))
            if self._time_aware_motion_enabled():
                # The unified timestamp/speed/prediction gate below owns this
                # decision. Keeping a single rejection point lets a genuine
                # abrupt maneuver enter fail-closed multi-frame probation.
                max_center_jump = 0.0
            else:
                max_center_jump = float(
                    0.0
                    if global_recovery
                    else (
                        self.cfg.get(
                            "occlusion_recovery_max_center_displacement_px",
                            self.cfg.get("component_max_center_jump_px", 100.0),
                        )
                        if fast_recovery
                        else self.cfg.get("component_max_center_jump_px", 100.0)
                    )
                )
            if max_center_jump > 0.0 and center_jump > max_center_jump:
                rejection = (
                    f"motion prediction residual {center_jump:.1f}>"
                    f"{max_center_jump:.1f}px"
                    if self._time_aware_motion_enabled()
                    else (f"center jump {center_jump:.1f}>" f"{max_center_jump:.1f}px")
                )
                continue

            # Appearance and depth cannot distinguish two identical objects,
            # so a raw motion bound is still required.  A single fixed pixel
            # threshold is not resolution/scale invariant, though: in the live
            # 848x480 stream a normal hand nudge moved the cylinder 47 px in one
            # frame and the old 24 px gate immediately declared it LOST.  Scale
            # the ordinary bound by the *previous accepted* bbox diagonal.  A
            # reliable KLT prediction may use the separately bounded optical-
            # flow range because expected-center and mask-overlap gates above
            # already tie that candidate to pixels from the previous target.
            raw_center_jump = float(
                np.hypot(
                    cx - self.state.center_uv[0],
                    cy - self.state.center_uv[1],
                )
            )
            if self._time_aware_motion_enabled():
                motion_reason = (
                    ""
                    if global_recovery and self.maneuver_hypothesis is None
                    else self._time_aware_motion_reject_reason(
                        frame,
                        (cx, cy),
                        apply_prediction=True,
                        max_frame_gap_override=(
                            int(
                                self.cfg.get(
                                    "occlusion_recovery_motion_max_frame_gap", 6
                                )
                            )
                            if fast_recovery
                            else None
                        ),
                    )
                )
                if motion_reason:
                    if bool(
                        self.cfg.get("component_maneuver_recovery_enabled", True)
                    ) and motion_reason.startswith("motion prediction residual"):
                        motion_conflict_reason = motion_reason
                    else:
                        rejection = motion_reason
                        continue
                max_motion = self._time_aware_motion_distance_limit_px(frame)
            else:
                max_motion = float(
                    0.0
                    if global_recovery
                    else (
                        self.cfg.get(
                            "occlusion_recovery_max_center_displacement_px",
                            self.cfg.get("component_max_motion_px", 24.0),
                        )
                        if fast_recovery
                        else self._component_motion_limit_px(flow)
                    )
                )
                if max_motion > 0.0 and raw_center_jump > max_motion:
                    rejection = f"motion {raw_center_jump:.1f}>{max_motion:.1f}px"
                    continue
            if (
                flow is None
                and self.state.lost_count > 0
                and bool(self.cfg.get("lost_recovery_requires_flow", True))
                and not fast_recovery
            ):
                rejection = "lost recovery has no reliable optical flow"
                continue

            component_depth_support = (
                local if fast_recovery or depth_pixel_gate else (local & depth_ok)
            )
            values = depth_crop[component_depth_support]
            values = values[np.isfinite(values) & (values > 0.0)]
            min_valid_depth_ratio = float(
                self.cfg.get("occlusion_recovery_min_valid_depth_ratio", 0.25)
                if fast_recovery
                else self.cfg.get(
                    "component_min_valid_depth_ratio",
                    0.25 if not depth_pixel_gate else 0.0,
                )
            )
            required_depth = max(
                min_area,
                int(np.ceil(area * max(0.0, min_valid_depth_ratio))),
            )
            if values.size < required_depth:
                rejection = "component has insufficient valid depth"
                continue
            zmed = float(np.median(values))
            # A fragment group may have been expanded through a tight depth
            # hull to reconstruct non-pink highlights or small RGB-D gaps.
            # Score appearance only on its original strong-colour support;
            # the expanded pixels are still checked by depth and size gates.
            color_support = local & strong_color
            color_score = float(
                np.mean(probability[color_support])
                if np.any(color_support)
                else np.mean(probability[local])
            )
            if fast_recovery:
                max_depth_jump = float(
                    self.cfg.get(
                        "occlusion_global_max_depth_jump_m",
                        max(recovery_depth_tolerance, 0.10),
                    )
                    if global_recovery
                    else self.cfg.get("occlusion_recovery_max_depth_jump_m", 0.06)
                )
                depth_jump = abs(zmed - self.state.depth_median)
                if max_depth_jump > 0.0 and depth_jump > max_depth_jump:
                    rejection = (
                        f"recovery depth jump {depth_jump:.3f}>"
                        f"{max_depth_jump:.3f}m"
                    )
                    continue

            # LOST may begin after a slowly advancing hand has reduced the
            # last accepted mask to a small fragment. Recovery must therefore
            # compare against the immutable prompt-mask scale, not that
            # poisoned last-visible area. Otherwise a 25% fragment can be
            # confirmed and the tracker immediately chatters LOST again.
            reference_area = max(
                1.0,
                float(
                    self.state.initial_area
                    if fast_recovery and self.state.initial_area > 0.0
                    else self.state.area
                ),
            )
            area_ratio = area / reference_area
            area_score = float(np.exp(-abs(np.log(max(area_ratio, 1e-6)))))
            if global_recovery:
                # Position is intentionally not an identity hard gate after a
                # complete occlusion.  Appearance, depth, size, uniqueness and
                # multi-frame consistency remain mandatory.
                center_score = 1.0
            elif max_center_jump > 0.0:
                center_score = float(
                    np.exp(-0.5 * (center_jump / max(1.0, 0.55 * max_center_jump)) ** 2)
                )
            else:
                center_score = 1.0
            depth_score = max(
                0.0,
                1.0
                - abs(zmed - self.state.depth_median)
                / max(recovery_depth_tolerance, 1e-6),
            )
            if flow_crop is not None:
                flow_overlap = float(
                    np.logical_and(local, flow_crop > 0).sum() / max(1.0, area)
                )
                min_flow_overlap = float(self.cfg.get("flow_min_mask_overlap", 0.15))
                if flow_overlap < min_flow_overlap:
                    rejection = (
                        f"flow overlap {flow_overlap:.3f}<" f"{min_flow_overlap:.3f}"
                    )
                    continue
                confidence = (
                    0.35 * color_score
                    + 0.20 * area_score
                    + 0.15 * center_score
                    + 0.10 * depth_score
                    + 0.20 * flow_overlap * flow.confidence
                )
            elif global_recovery:
                confidence = 0.58 * color_score + 0.24 * area_score + 0.18 * depth_score
            else:
                confidence = (
                    0.50 * color_score
                    + 0.22 * area_score
                    + 0.18 * center_score
                    + 0.10 * depth_score
                )
            if global_recovery:
                min_confidence = float(
                    self.cfg.get(
                        "occlusion_global_min_confidence",
                        self.cfg.get("occlusion_recovery_min_confidence", 0.68),
                    )
                )
            elif fast_recovery:
                min_confidence = float(
                    self.cfg.get(
                        "occlusion_recovery_min_confidence",
                        self.cfg.get("component_min_confidence", 0.62),
                    )
                )
            else:
                min_confidence = float(self.cfg.get("component_min_confidence", 0.62))
            if confidence < min_confidence:
                rejection = (
                    f"component confidence {confidence:.3f}<{min_confidence:.3f}"
                )
                continue

            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[y1:y2, x1:x2][local] = 1
            component = _ComponentCandidate(
                mask=full_mask,
                bbox_xyxy=bbox,
                center_uv=(cx, cy),
                area=area,
                depth_median=zmed,
                confidence=float(np.clip(confidence, 0.0, 1.0)),
                color_score=color_score,
                search_scope=(
                    "global"
                    if global_recovery
                    else ("local" if fast_recovery else "flow")
                ),
                motion_conflict_reason=motion_conflict_reason,
            )
            candidates.append(component)

        if not candidates:
            return None, rejection
        candidates.sort(key=lambda item: item.confidence, reverse=True)
        if len(candidates) > 1:
            margin = candidates[0].confidence - candidates[1].confidence
            if global_recovery:
                min_margin = float(
                    self.cfg.get(
                        "occlusion_global_min_score_margin",
                        self.cfg.get("occlusion_recovery_min_score_margin", 0.08),
                    )
                )
            elif fast_recovery:
                min_margin = float(
                    self.cfg.get(
                        "occlusion_recovery_min_score_margin",
                        self.cfg.get("component_min_score_margin", 0.04),
                    )
                )
            else:
                min_margin = float(self.cfg.get("component_min_score_margin", 0.04))
            if min_margin > 0.0 and margin < min_margin:
                return (
                    None,
                    f"ambiguous components margin {margin:.3f}<" f"{min_margin:.3f}",
                )
        return candidates[0], rejection

    def _time_aware_motion_enabled(self) -> bool:
        mode = (
            str(self.cfg.get("component_motion_model", "legacy_per_frame"))
            .strip()
            .lower()
        )
        return mode in {
            "timestamp_velocity",
            "time_aware_velocity",
            "committed_velocity",
        }

    def _has_committed_velocity(self) -> bool:
        return bool(
            self.committed_motion is not None
            and int(self.committed_motion.velocity_samples) > 0
        )

    def _motion_delta(self, frame: RGBDFrame) -> Optional[Tuple[int, float]]:
        motion = self.committed_motion
        if motion is None:
            return None
        frame_gap = int(frame.frame_id) - int(motion.frame_id)
        dt_s = float(frame.timestamp) - float(motion.timestamp_s)
        if frame_gap <= 0 or not np.isfinite(dt_s) or dt_s <= 0.0:
            return None
        return int(frame_gap), float(dt_s)

    def _motion_predicted_center(
        self, frame: RGBDFrame
    ) -> Optional[Tuple[float, float]]:
        motion = self.committed_motion
        delta = self._motion_delta(frame)
        if motion is None or delta is None:
            return None
        _, dt_s = delta
        if not self._has_committed_velocity():
            return (
                float(motion.center_uv[0]),
                float(motion.center_uv[1]),
            )
        return (
            float(motion.center_uv[0] + motion.velocity_uv_s[0] * dt_s),
            float(motion.center_uv[1] + motion.velocity_uv_s[1] * dt_s),
        )

    def _motion_prediction_residual_limit_px(self) -> float:
        motion = self.committed_motion
        bbox = (
            np.asarray(motion.bbox_xyxy, dtype=np.float64).reshape(4)
            if motion is not None
            else (
                np.asarray(self.state.bbox_xyxy, dtype=np.float64).reshape(4)
                if self.state is not None
                else np.asarray([0.0, 0.0, 1.0, 1.0])
            )
        )
        width = max(1.0, float(bbox[2] - bbox[0]))
        height = max(1.0, float(bbox[3] - bbox[1]))
        diagonal = float(np.hypot(width, height))
        ratio = max(
            0.0,
            float(
                self.cfg.get(
                    "component_motion_prediction_residual_bbox_diagonal_ratio",
                    0.80,
                )
            ),
        )
        minimum = max(
            0.0,
            float(self.cfg.get("component_motion_prediction_residual_min_px", 12.0)),
        )
        hard_cap = float(
            self.cfg.get("component_motion_prediction_residual_max_px", 48.0)
        )
        limit = max(minimum, ratio * diagonal)
        if hard_cap > 0.0:
            limit = min(limit, hard_cap)
        return float(limit)

    def _time_aware_motion_distance_limit_px(self, frame: RGBDFrame) -> float:
        delta = self._motion_delta(frame)
        if delta is None:
            return 0.0
        _, dt_s = delta
        max_speed = max(
            0.0,
            float(self.cfg.get("component_motion_max_speed_px_s", 2400.0)),
        )
        return float(max_speed * dt_s)

    def _time_aware_motion_reject_reason(
        self,
        frame: RGBDFrame,
        center_uv: Tuple[float, float],
        *,
        apply_prediction: bool = True,
        max_frame_gap_override: Optional[int] = None,
    ) -> str:
        """Validate motion against sensor time and committed target velocity.

        Scalar speed bounds make the gate invariant to 30 Hz frame skips.
        After one accepted displacement, a bounded prediction residual also
        preserves direction and catches a plausible-speed reversal or an
        identical neighbouring instance.  No tentative recovery observation
        is consumed by this method.
        """

        motion = self.committed_motion
        if motion is None:
            return ""
        delta = self._motion_delta(frame)
        if delta is None:
            return "motion has non-advancing frame id/timestamp"
        frame_gap, dt_s = delta
        max_frame_gap = max(
            1,
            int(
                self.cfg.get("component_motion_max_frame_gap", 3)
                if max_frame_gap_override is None
                else max_frame_gap_override
            ),
        )
        max_timestamp_gap = max(
            0.0,
            float(self.cfg.get("component_motion_max_timestamp_gap_s", 0.20)),
        )
        if frame_gap > max_frame_gap:
            return f"motion frame gap {frame_gap}>{max_frame_gap}"
        if max_timestamp_gap > 0.0 and dt_s > max_timestamp_gap:
            return f"motion timestamp gap {dt_s:.3f}>" f"{max_timestamp_gap:.3f}s"

        dx = float(center_uv[0]) - float(motion.center_uv[0])
        dy = float(center_uv[1]) - float(motion.center_uv[1])
        displacement = float(np.hypot(dx, dy))
        speed = displacement / max(dt_s, 1.0e-9)
        max_speed = max(
            0.0,
            float(self.cfg.get("component_motion_max_speed_px_s", 2400.0)),
        )
        if max_speed > 0.0 and speed > max_speed + 1.0e-6:
            return (
                f"motion speed {speed:.1f}>{max_speed:.1f}px/s "
                f"gap={frame_gap} dt={dt_s:.4f}s"
            )

        if apply_prediction and self._has_committed_velocity():
            predicted = self._motion_predicted_center(frame)
            assert predicted is not None
            residual = float(
                np.hypot(
                    float(center_uv[0]) - float(predicted[0]),
                    float(center_uv[1]) - float(predicted[1]),
                )
            )
            residual_limit = self._motion_prediction_residual_limit_px()
            if residual_limit > 0.0 and residual > residual_limit:
                return (
                    f"motion prediction residual {residual:.1f}>"
                    f"{residual_limit:.1f}px gap={frame_gap} "
                    f"dt={dt_s:.4f}s"
                )
        return ""

    def _time_aware_search_bbox(
        self,
        frame: RGBDFrame,
        image_width: int,
        image_height: int,
    ) -> np.ndarray:
        """Search the physically reachable timestamped image region.

        Before velocity bootstrap the direction is unknown, so the speed
        envelope expands symmetrically.  Afterwards the ROI follows the
        committed prediction and needs only the scale-aware residual margin.
        """

        motion = self.committed_motion
        if motion is None:
            assert self.state is not None
            return enlarge_bbox(
                self.state.bbox_xyxy,
                float(self.cfg.get("roi_scale", 2.2)),
                image_width,
                image_height,
            )
        bbox = np.asarray(motion.bbox_xyxy, dtype=np.float64).reshape(4)
        predicted = self._motion_predicted_center(frame)
        delta = self._motion_delta(frame)
        if predicted is None or delta is None:
            return clip_bbox(bbox.astype(np.int32), image_width, image_height)
        _, dt_s = delta
        shift_x = float(predicted[0]) - float(motion.center_uv[0])
        shift_y = float(predicted[1]) - float(motion.center_uv[1])
        predicted_bbox = bbox + np.asarray(
            [shift_x, shift_y, shift_x, shift_y], dtype=np.float64
        )
        # Search the complete physically reachable envelope, not only the old
        # velocity ray. This is what lets the first bounce/reversal candidate
        # be observed and quarantined. The much tighter prediction residual is
        # still enforced before any mask is accepted or published.
        residual = (
            self._motion_prediction_residual_limit_px()
            + self._time_aware_motion_distance_limit_px(frame)
        )
        x1 = min(float(bbox[0]), float(predicted_bbox[0])) - residual
        y1 = min(float(bbox[1]), float(predicted_bbox[1])) - residual
        x2 = max(float(bbox[2]), float(predicted_bbox[2])) + residual
        y2 = max(float(bbox[3]), float(predicted_bbox[3])) + residual
        return clip_bbox(
            np.asarray(
                [
                    int(np.floor(x1)),
                    int(np.floor(y1)),
                    int(np.ceil(x2)),
                    int(np.ceil(y2)),
                ],
                dtype=np.int32,
            ),
            image_width,
            image_height,
        )

    def _commit_motion_observation(
        self,
        frame: RGBDFrame,
        center_uv: Tuple[float, float],
        bbox_xyxy: np.ndarray,
    ) -> None:
        center = (float(center_uv[0]), float(center_uv[1]))
        bbox = np.asarray(bbox_xyxy, dtype=np.int32).reshape(4).copy()
        previous = self.committed_motion
        if previous is None:
            self.committed_motion = _CommittedMotionState(
                center_uv=center,
                bbox_xyxy=bbox,
                timestamp_s=float(frame.timestamp),
                frame_id=int(frame.frame_id),
                velocity_uv_s=(0.0, 0.0),
                velocity_samples=0,
            )
            return
        delta = self._motion_delta(frame)
        if delta is None:
            return
        _, dt_s = delta
        measured = (
            (center[0] - float(previous.center_uv[0])) / dt_s,
            (center[1] - float(previous.center_uv[1])) / dt_s,
        )
        if int(previous.velocity_samples) <= 0:
            velocity = measured
        else:
            beta = float(
                np.clip(
                    float(self.cfg.get("component_motion_velocity_ema_beta", 0.70)),
                    0.0,
                    1.0,
                )
            )
            velocity = (
                beta * float(previous.velocity_uv_s[0]) + (1.0 - beta) * measured[0],
                beta * float(previous.velocity_uv_s[1]) + (1.0 - beta) * measured[1],
            )
        self.committed_motion = _CommittedMotionState(
            center_uv=center,
            bbox_xyxy=bbox,
            timestamp_s=float(frame.timestamp),
            frame_id=int(frame.frame_id),
            velocity_uv_s=(float(velocity[0]), float(velocity[1])),
            velocity_samples=int(previous.velocity_samples) + 1,
        )

    def _confirm_maneuver(
        self,
        frame: RGBDFrame,
        candidate: _ComponentCandidate,
    ) -> Tuple[bool, str]:
        """Confirm a real bounce/turn without learning from one conflict.

        Every candidate already passed immutable appearance, depth, scale and
        connected-component uniqueness gates in :meth:`_best_component`.
        This second layer requires three temporally coherent observations.
        Only the third observation resets committed velocity; the first two
        remain unpublished and cannot modify normal tracker/appearance state.
        """

        required = max(
            3,
            int(self.cfg.get("component_maneuver_confirm_frames", 3)),
        )
        previous = self.maneuver_hypothesis
        consistent = previous is not None
        pair_velocity: Optional[Tuple[float, float]] = None
        detail = "new prediction-conflict candidate"
        if previous is not None:
            frame_gap = int(frame.frame_id) - int(previous.last_frame_id)
            dt_s = float(frame.timestamp) - float(previous.timestamp_s)
            max_frame_gap = max(
                1,
                int(self.cfg.get("component_maneuver_confirm_max_frame_gap", 3)),
            )
            max_dt_s = max(
                0.0,
                float(
                    self.cfg.get(
                        "component_maneuver_confirm_max_timestamp_step_s", 0.20
                    )
                ),
            )
            center_step = float(
                np.hypot(
                    candidate.center_uv[0] - previous.center_uv[0],
                    candidate.center_uv[1] - previous.center_uv[1],
                )
            )
            pair_speed = center_step / max(dt_s, 1.0e-9)
            max_speed = max(
                0.0,
                float(self.cfg.get("component_motion_max_speed_px_s", 2400.0)),
            )
            area_ratio = float(candidate.area) / max(1.0, float(previous.area))
            max_area_ratio = max(
                1.0,
                float(self.cfg.get("component_maneuver_confirm_max_area_ratio", 1.60)),
            )
            depth_step = abs(
                float(candidate.depth_median) - float(previous.depth_median)
            )
            max_depth_step = max(
                0.0,
                float(
                    self.cfg.get("component_maneuver_confirm_max_depth_step_m", 0.030)
                ),
            )
            if dt_s > 0.0:
                pair_velocity = (
                    (candidate.center_uv[0] - previous.center_uv[0]) / dt_s,
                    (candidate.center_uv[1] - previous.center_uv[1]) / dt_s,
                )
            prediction_residual = 0.0
            residual_limit = self._motion_prediction_residual_limit_px()
            if previous.velocity_uv_s is not None and dt_s > 0.0:
                predicted = (
                    previous.center_uv[0] + previous.velocity_uv_s[0] * dt_s,
                    previous.center_uv[1] + previous.velocity_uv_s[1] * dt_s,
                )
                prediction_residual = float(
                    np.hypot(
                        candidate.center_uv[0] - predicted[0],
                        candidate.center_uv[1] - predicted[1],
                    )
                )
            consistent = (
                1 <= frame_gap <= max_frame_gap
                and dt_s > 0.0
                and (max_dt_s <= 0.0 or dt_s <= max_dt_s)
                and (max_speed <= 0.0 or pair_speed <= max_speed)
                and (1.0 / max_area_ratio) <= area_ratio <= max_area_ratio
                and (max_depth_step <= 0.0 or depth_step <= max_depth_step)
                and (
                    previous.velocity_uv_s is None
                    or residual_limit <= 0.0
                    or prediction_residual <= residual_limit
                )
            )
            detail = (
                f"gap={frame_gap} dt={dt_s:.3f}s "
                f"speed={pair_speed:.1f}px/s "
                f"prediction_residual={prediction_residual:.1f}px "
                f"area_x={area_ratio:.3f} depth_step={depth_step:.3f}m"
            )

        hits = previous.hits + 1 if consistent and previous is not None else 1
        stored_velocity = (
            pair_velocity if consistent and pair_velocity is not None else None
        )
        hypothesis = _ManeuverHypothesis(
            mask=candidate.mask.copy(),
            bbox_xyxy=candidate.bbox_xyxy.astype(np.int32).copy(),
            center_uv=(
                float(candidate.center_uv[0]),
                float(candidate.center_uv[1]),
            ),
            area=float(candidate.area),
            depth_median=float(candidate.depth_median),
            confidence=float(candidate.confidence),
            hits=int(hits),
            last_frame_id=int(frame.frame_id),
            timestamp_s=float(frame.timestamp),
            velocity_uv_s=(
                (float(stored_velocity[0]), float(stored_velocity[1]))
                if stored_velocity is not None
                else None
            ),
        )
        self.maneuver_hypothesis = hypothesis
        if hits < required:
            return (
                False,
                f"maneuver recovery pending {hits}/{required}: {detail}",
            )

        # Three accepted pieces of evidence now authorize a motion reset. Use
        # only the latest pair so a sharp bounce does not retain old-direction
        # velocity through an EMA tail.
        if stored_velocity is None:
            return False, "maneuver recovery lacks a valid final pair velocity"
        self.committed_motion = _CommittedMotionState(
            center_uv=hypothesis.center_uv,
            bbox_xyxy=hypothesis.bbox_xyxy.copy(),
            timestamp_s=float(hypothesis.timestamp_s),
            frame_id=int(hypothesis.last_frame_id),
            velocity_uv_s=(
                float(stored_velocity[0]),
                float(stored_velocity[1]),
            ),
            velocity_samples=1,
        )
        self.maneuver_hypothesis = None
        self.recovery_hypothesis = None
        return True, f"maneuver recovery confirmed {hits}/{required}: {detail}"

    def _component_motion_limit_px(self, flow: Optional[_FlowPrediction]) -> float:
        """Return a scale-aware, identity-bounded per-frame motion limit.

        ``component_max_motion_px`` remains the minimum tolerance and setting it
        to zero keeps the historical "disabled" behavior.  The adaptive part
        depends only on the previous accepted target, never on an untrusted
        current component.  High-confidence KLT may extend the limit up to its
        own guarded displacement range; low-confidence flow receives no such
        privilege.
        """

        assert self.state is not None
        base = float(self.cfg.get("component_max_motion_px", 24.0))
        if base <= 0.0:
            return 0.0

        # Without an independent motion observation, keep the strict base
        # limit. An identical same-depth neighbour is otherwise impossible to
        # distinguish from a teleported target in a single RGB-D frame.
        limit = base

        if flow is not None:
            min_confidence = float(
                self.cfg.get("component_flow_motion_min_confidence", 0.45)
            )
            if float(getattr(flow, "confidence", 0.0)) >= min_confidence:
                bbox = np.asarray(self.state.bbox_xyxy, dtype=np.float64).reshape(4)
                width = max(1.0, float(bbox[2] - bbox[0]))
                height = max(1.0, float(bbox[3] - bbox[1]))
                diagonal = float(np.hypot(width, height))
                ratio = max(
                    0.0,
                    float(self.cfg.get("component_motion_bbox_diagonal_ratio", 0.80)),
                )
                limit = max(
                    limit,
                    ratio * diagonal,
                    float(self.cfg.get("flow_max_displacement_px", 100.0)),
                )

        cap = float(self.cfg.get("component_max_motion_cap_px", 100.0))
        if cap > 0.0:
            limit = min(limit, max(base, cap))
        return float(limit)

    def _merge_global_recovery_fragments(
        self,
        seed_mask: np.ndarray,
        depth_crop: np.ndarray,
        valid_depth: np.ndarray,
    ) -> np.ndarray:
        """Join nearby, same-depth colour fragments before global ranking.

        The target in the real cover/uncover log was split into several
        800--1100 px colour islands although its prompt mask was about 3200 px.
        Treating each island as an object caused ambiguity and one-frame
        recover/loss chatter. Nearby islands are grouped only when their depth
        is coherent; their convex hull is then intersected with a tight depth
        band. Distant category instances and a closer hand remain separate.
        """

        seed = (np.asarray(seed_mask) > 0).astype(np.uint8)
        if int(seed.sum()) == 0 or self.state is None:
            return seed

        max_gap = max(0, int(self.cfg.get("occlusion_fragment_max_gap_px", 12)))
        if max_gap <= 0:
            return seed
        radius = max(1, int(np.ceil(0.5 * max_gap)))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        dilated = cv2.dilate(seed, kernel, iterations=1)
        group_count, group_labels, group_stats, _ = cv2.connectedComponentsWithStats(
            dilated, connectivity=8
        )
        if group_count <= 1:
            return seed

        initial_bbox_area = max(1.0, float(self.state.initial_bbox_area))
        max_bbox_ratio = float(
            self.cfg.get("occlusion_global_max_bbox_area_ratio", 2.50)
        )
        if max_bbox_ratio > 0.0:
            group_bbox_areas = group_stats[1:, cv2.CC_STAT_WIDTH].astype(
                np.float64
            ) * group_stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float64)
            # If every dilation group already spans an impossible target bbox,
            # no later component/depth test can make it eligible. This cheap
            # guard avoids a second full-frame connected-components pass in an
            # adversarial field of many nearby coloured highlights.
            if not np.any(group_bbox_areas <= initial_bbox_area * max_bbox_ratio):
                return seed

        # Label the original (undilated) islands once.  The previous
        # implementation ran connectedComponentsWithStats on a full-frame
        # boolean image for *every* dilation group.  A normal lab scene can
        # contain hundreds of tiny target-coloured highlights, turning LOST
        # recovery into an 80--110 ms operation even though only a handful of
        # nearby islands can possibly be merged.  Each original component is
        # wholly contained in exactly one dilated group, so a single component
        # pass plus cropped per-group work is equivalent and bounded.
        component_count, component_labels, component_stats, _ = (
            cv2.connectedComponentsWithStats(seed, connectivity=8)
        )
        if component_count <= 2:
            return seed

        foreground_flat = np.flatnonzero(seed.reshape(-1))
        component_ids_flat = component_labels.reshape(-1)[foreground_flat]
        first_flat = np.full(component_count, seed.size, dtype=np.int64)
        np.minimum.at(first_flat, component_ids_flat, foreground_flat)
        component_group = np.zeros(component_count, dtype=np.int32)
        valid_component_ids = np.arange(1, component_count, dtype=np.int32)
        first_positions = first_flat[valid_component_ids]
        present = first_positions < seed.size
        component_group[valid_component_ids[present]] = group_labels.reshape(-1)[
            first_positions[present]
        ]
        components_by_group = {}
        for component_id in valid_component_ids:
            group_id = int(component_group[component_id])
            if group_id > 0:
                components_by_group.setdefault(group_id, []).append(int(component_id))

        # Begin with every original island.  Only groups that pass all merge
        # gates add depth-supported hull pixels below.
        output = seed.copy()
        initial_area = max(1.0, float(self.state.initial_area))
        full_component_area = initial_area * float(
            self.cfg.get("occlusion_global_min_area_ratio", 0.55)
        )
        max_depth_spread = float(
            self.cfg.get("occlusion_fragment_max_depth_spread_m", 0.08)
        )
        expand_tolerance = float(
            self.cfg.get("occlusion_fragment_depth_tolerance_m", 0.05)
        )

        for group_id, component_ids in components_by_group.items():
            if len(component_ids) <= 1:
                continue

            component_areas = component_stats[component_ids, cv2.CC_STAT_AREA].astype(
                float
            )
            # Two independently full-sized objects are never merged into one
            # category candidate, even when they are physically close.
            if int(np.count_nonzero(component_areas >= full_component_area)) >= 2:
                continue

            gx = int(group_stats[group_id, cv2.CC_STAT_LEFT])
            gy = int(group_stats[group_id, cv2.CC_STAT_TOP])
            gw = int(group_stats[group_id, cv2.CC_STAT_WIDTH])
            gh = int(group_stats[group_id, cv2.CC_STAT_HEIGHT])
            sx = slice(gx, gx + gw)
            sy = slice(gy, gy + gh)
            group_seed = (seed[sy, sx] > 0) & (group_labels[sy, sx] == group_id)
            ys, xs = np.nonzero(group_seed)
            if xs.size < 3:
                continue
            component_boxes = component_stats[component_ids]
            left = int(component_boxes[:, cv2.CC_STAT_LEFT].min())
            top = int(component_boxes[:, cv2.CC_STAT_TOP].min())
            right = int(
                np.max(
                    component_boxes[:, cv2.CC_STAT_LEFT]
                    + component_boxes[:, cv2.CC_STAT_WIDTH]
                )
            )
            bottom = int(
                np.max(
                    component_boxes[:, cv2.CC_STAT_TOP]
                    + component_boxes[:, cv2.CC_STAT_HEIGHT]
                )
            )
            group_bbox_area = float((right - left) * (bottom - top))
            if (
                max_bbox_ratio > 0.0
                and group_bbox_area > initial_bbox_area * max_bbox_ratio
            ):
                continue

            local_depth = depth_crop[sy, sx]
            local_valid_depth = valid_depth[sy, sx]
            seed_depths = local_depth[group_seed & local_valid_depth]
            seed_depths = seed_depths[np.isfinite(seed_depths) & (seed_depths > 0.0)]
            if seed_depths.size < max(8, int(0.10 * xs.size)):
                continue
            z5, zmed, z95 = np.percentile(seed_depths, [5.0, 50.0, 95.0])
            if float(z95 - z5) > max_depth_spread:
                continue

            points = np.column_stack((xs, ys)).astype(np.int32)
            hull = cv2.convexHull(points)
            hull_mask = np.zeros_like(group_seed, dtype=np.uint8)
            cv2.fillConvexPoly(hull_mask, hull, 1)
            tight_depth = local_valid_depth & (
                np.abs(local_depth - float(zmed)) <= expand_tolerance
            )
            expanded = (hull_mask > 0) & tight_depth
            if int(expanded.sum()) < int(group_seed.sum()):
                continue
            output_crop = output[sy, sx]
            output_crop[expanded] = 1
        return output

    def _predict_with_optical_flow(
        self, current_gray: np.ndarray
    ) -> Optional[_FlowPrediction]:
        """Predict the prior prompt mask with guarded sparse KLT + affine flow."""

        if not bool(self.cfg.get("optical_flow_enabled", True)):
            return None
        if (
            self.state is None
            or self.state.last_mask is None
            or self.previous_gray is None
        ):
            return None
        if self.previous_gray.shape != current_gray.shape:
            return None

        feature_mask = (self.state.last_mask > 0).astype(np.uint8) * 255
        feature_gray = self.previous_gray
        feature_search_mask = feature_mask
        feature_origin = np.zeros(2, dtype=np.float32)
        if bool(self.cfg.get("flow_feature_crop_enabled", False)):
            # goodFeaturesToTrack only accepts corners inside feature_mask, so
            # scanning the rest of the full image is unnecessary. Keep a halo
            # around the exact prior-mask bbox so the 5x5 corner block and its
            # gradients see the same native-resolution pixels as before. KLT
            # itself still runs on the complete native images below; this crop
            # therefore cannot extend the accepted displacement/recovery
            # envelope or introduce an artificial flow boundary.
            halo = max(4, int(self.cfg.get("flow_feature_crop_halo_px", 4)))
            height, width = feature_mask.shape
            x1, y1, x2, y2 = (int(value) for value in self.state.bbox_xyxy)
            x1 = max(0, x1 - halo)
            y1 = max(0, y1 - halo)
            x2 = min(width, x2 + halo)
            y2 = min(height, y2 + halo)
            if x2 <= x1 or y2 <= y1:
                return None
            feature_gray = self.previous_gray[y1:y2, x1:x2]
            feature_search_mask = feature_mask[y1:y2, x1:x2]
            feature_origin = np.asarray([x1, y1], dtype=np.float32)

        max_corners = int(self.cfg.get("flow_max_corners", 80))
        points = cv2.goodFeaturesToTrack(
            feature_gray,
            maxCorners=max_corners,
            qualityLevel=float(self.cfg.get("flow_quality_level", 0.01)),
            minDistance=float(self.cfg.get("flow_min_distance", 4.0)),
            mask=feature_search_mask,
            blockSize=5,
        )
        min_points = int(self.cfg.get("flow_min_points", 4))
        if points is None or len(points) < min_points:
            return None
        points = points + feature_origin.reshape(1, 1, 2)

        win = int(self.cfg.get("flow_window_size", 21))
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            20,
            0.01,
        )
        next_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            current_gray,
            points,
            None,
            winSize=(win, win),
            maxLevel=int(self.cfg.get("flow_pyramid_levels", 3)),
            criteria=criteria,
        )
        if next_points is None or forward_status is None:
            return None
        back_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray,
            self.previous_gray,
            next_points,
            None,
            winSize=(win, win),
            maxLevel=int(self.cfg.get("flow_pyramid_levels", 3)),
            criteria=criteria,
        )
        if back_points is None or backward_status is None:
            return None

        forward_ok = forward_status.reshape(-1).astype(bool)
        backward_ok = backward_status.reshape(-1).astype(bool)
        source_all = points.reshape(-1, 2)
        target_all = next_points.reshape(-1, 2)
        back_all = back_points.reshape(-1, 2)
        fb_error = np.linalg.norm(source_all - back_all, axis=1)
        good = (
            forward_ok
            & backward_ok
            & np.isfinite(fb_error)
            & (fb_error <= float(self.cfg.get("flow_fb_max_error", 1.5)))
        )
        if int(good.sum()) < min_points:
            return None
        source = source_all[good]
        target = target_all[good]
        affine, inliers = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(
                self.cfg.get("flow_ransac_reprojection_px", 2.5)
            ),
            maxIters=500,
            confidence=0.99,
            refineIters=10,
        )
        if affine is None or inliers is None or not np.isfinite(affine).all():
            return None
        inlier_count = int(inliers.reshape(-1).sum())
        inlier_ratio = inlier_count / max(1, len(source))
        min_inlier_ratio = float(self.cfg.get("flow_min_inlier_ratio", 0.55))
        if inlier_count < min_points or inlier_ratio < min_inlier_ratio:
            return None

        a, b, _ = (float(v) for v in affine[0])
        c, d, _ = (float(v) for v in affine[1])
        scale = float(np.sqrt(max(0.0, a * a + c * c)))
        rotation_deg = float(np.degrees(np.arctan2(c, a)))
        min_scale = float(self.cfg.get("flow_min_scale", 0.75))
        max_scale = float(self.cfg.get("flow_max_scale", 1.35))
        max_rotation = float(self.cfg.get("flow_max_rotation_deg", 35.0))
        max_displacement = float(self.cfg.get("flow_max_displacement_px", 100.0))
        if not min_scale <= scale <= max_scale:
            return None
        if abs(rotation_deg) > max_rotation:
            return None
        center = np.asarray(
            [self.state.center_uv[0], self.state.center_uv[1], 1.0],
            dtype=np.float64,
        )
        predicted_center = affine @ center
        center_displacement = float(
            np.hypot(
                predicted_center[0] - center[0],
                predicted_center[1] - center[1],
            )
        )
        if center_displacement > max_displacement:
            return None

        h, w = current_gray.shape
        predicted = cv2.warpAffine(
            self.state.last_mask.astype(np.uint8),
            affine,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        bbox = bbox_from_mask(predicted, min_area=int(self.cfg.get("min_area", 80)))
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox.astype(float)
        point_score = min(1.0, inlier_count / max(8.0, 0.25 * max_corners))
        confidence = float(np.clip(inlier_ratio * point_score, 0.0, 1.0))
        return _FlowPrediction(
            mask=(predicted > 0).astype(np.uint8),
            bbox_xyxy=bbox,
            center_uv=(0.5 * (x1 + x2), 0.5 * (y1 + y2)),
            confidence=confidence,
            affine_matrix=np.asarray(affine, dtype=np.float32),
            message=(
                f"KLT inliers={inlier_count}/{len(source)} "
                f"scale={scale:.3f} rot={rotation_deg:.1f}deg"
            ),
        )

    def _geometry_flow_from_candidate(
        self, candidate: _ComponentCandidate
    ) -> Optional[_FlowPrediction]:
        """Build a bounded translation/scale prior for a texture-poor target.

        This is intentionally downstream of component validation.  It cannot
        invent a target from the previous mask: if color/depth segmentation
        found no current object, :meth:`update` rejects before calling here.
        The transform only places the previous soft silhouette into the
        already accepted component's coordinate system.
        """

        if not bool(self.cfg.get("temporal_mask_geometry_fallback_enabled", True)):
            return None
        if self.state is None or self.state.last_mask is None:
            return None
        previous = np.asarray(self.state.last_mask) > 0
        current = np.asarray(candidate.mask) > 0
        previous_area = float(previous.sum())
        current_area = float(current.sum())
        if previous_area <= 0.0 or current_area <= 0.0:
            return None

        previous_y, previous_x = np.nonzero(previous)
        current_y, current_x = np.nonzero(current)
        previous_center = np.asarray(
            [float(previous_x.mean()), float(previous_y.mean())],
            dtype=np.float64,
        )
        current_center = np.asarray(
            [float(current_x.mean()), float(current_y.mean())],
            dtype=np.float64,
        )
        displacement = float(np.linalg.norm(current_center - previous_center))
        configured_max_displacement = float(
            self.cfg.get("temporal_mask_geometry_max_displacement_px", 0.0)
        )
        max_displacement = (
            configured_max_displacement
            if configured_max_displacement > 0.0
            else self._component_motion_limit_px(None)
        )
        if max_displacement > 0.0 and displacement > max_displacement:
            return None

        previous_bbox = np.asarray(self.state.bbox_xyxy, dtype=np.float64)
        current_bbox = np.asarray(candidate.bbox_xyxy, dtype=np.float64)
        previous_w = max(1.0, float(previous_bbox[2] - previous_bbox[0]))
        previous_h = max(1.0, float(previous_bbox[3] - previous_bbox[1]))
        current_w = max(1.0, float(current_bbox[2] - current_bbox[0]))
        current_h = max(1.0, float(current_bbox[3] - current_bbox[1]))
        scale_x = current_w / previous_w
        scale_y = current_h / previous_h
        max_anisotropy = max(
            1.0,
            float(self.cfg.get("temporal_mask_geometry_max_anisotropy", 1.20)),
        )
        anisotropy = max(scale_x, scale_y) / max(1.0e-6, min(scale_x, scale_y))
        if anisotropy > max_anisotropy:
            return None
        scale = float(np.sqrt(max(1.0e-8, scale_x * scale_y)))
        min_scale = float(self.cfg.get("temporal_mask_geometry_min_scale", 0.85))
        max_scale = float(self.cfg.get("temporal_mask_geometry_max_scale", 1.18))
        if not min_scale <= scale <= max_scale:
            return None

        affine = np.asarray(
            [
                [
                    scale,
                    0.0,
                    current_center[0] - scale * previous_center[0],
                ],
                [
                    0.0,
                    scale,
                    current_center[1] - scale * previous_center[1],
                ],
            ],
            dtype=np.float32,
        )
        h, w = current.shape
        predicted = cv2.warpAffine(
            previous.astype(np.uint8),
            affine,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        bbox = bbox_from_mask(predicted, min_area=int(self.cfg.get("min_area", 80)))
        if bbox is None:
            return None
        area_ratio = current_area / previous_area
        shape_score = float(np.exp(-abs(np.log(max(area_ratio, 1.0e-6)))))
        confidence = float(np.clip(0.55 + 0.20 * shape_score, 0.0, 0.75))
        return _FlowPrediction(
            mask=(predicted > 0).astype(np.uint8),
            bbox_xyxy=bbox,
            center_uv=(float(current_center[0]), float(current_center[1])),
            confidence=confidence,
            affine_matrix=affine,
            message=(
                f"geometry fallback shift={displacement:.1f}px " f"scale={scale:.3f}"
            ),
        )

    def _component_size_reject_reason(self, area: float, bbox: np.ndarray) -> str:
        assert self.state is not None
        min_area_ratio = float(self.cfg.get("min_area_ratio", 0.35))
        if self.state.area > 0.0 and area < self.state.area * min_area_ratio:
            return (
                f"area shrink {int(self.state.area)}->{int(area)} "
                f"min_x={min_area_ratio:.2f}"
            )
        return self._size_reject_reason(area, bbox, check_last=True)

    def _occlusion_size_reject_reason(
        self,
        area: float,
        bbox: np.ndarray,
        check_last_position: bool = True,
        global_recovery: bool = False,
    ) -> str:
        """Apply tighter last-visible size gates during no-flow recovery.

        The ordinary tracker intentionally tolerates gradual scale changes.
        After complete occlusion, however, accepting a substantially different
        component is an instance-switch risk.  Compare against the frozen last
        valid state; no appearance/depth/size model is updated while LOST.
        """

        assert self.state is not None
        reference_area = max(
            1.0,
            float(
                self.state.initial_area
                if self.state.initial_area > 0.0
                else self.state.area
            ),
        )
        area_ratio = float(area) / reference_area
        min_area_ratio = float(
            self.cfg.get(
                "occlusion_global_min_area_ratio",
                self.cfg.get("occlusion_recovery_min_area_ratio", 0.60),
            )
            if global_recovery
            else self.cfg.get("occlusion_recovery_min_area_ratio", 0.60)
        )
        max_area_ratio = float(
            self.cfg.get(
                "occlusion_global_max_area_ratio",
                self.cfg.get("occlusion_recovery_max_area_ratio", 1.60),
            )
            if global_recovery
            else self.cfg.get("occlusion_recovery_max_area_ratio", 1.60)
        )
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            return (
                f"recovery area ratio {area_ratio:.3f} outside "
                f"[{min_area_ratio:.3f},{max_area_ratio:.3f}]"
            )

        reference_bbox_area = max(
            1.0,
            float(
                self.state.initial_bbox_area
                if self.state.initial_bbox_area > 0.0
                else bbox_area(self.state.bbox_xyxy)
            ),
        )
        bbox_ratio = float(bbox_area(bbox)) / reference_bbox_area
        min_bbox_ratio = float(
            self.cfg.get(
                "occlusion_global_min_bbox_area_ratio",
                self.cfg.get("occlusion_recovery_min_bbox_area_ratio", 0.60),
            )
            if global_recovery
            else self.cfg.get("occlusion_recovery_min_bbox_area_ratio", 0.60)
        )
        max_bbox_ratio = float(
            self.cfg.get(
                "occlusion_global_max_bbox_area_ratio",
                self.cfg.get("occlusion_recovery_max_bbox_area_ratio", 1.60),
            )
            if global_recovery
            else self.cfg.get("occlusion_recovery_max_bbox_area_ratio", 1.60)
        )
        if bbox_ratio < min_bbox_ratio or bbox_ratio > max_bbox_ratio:
            return (
                f"recovery bbox ratio {bbox_ratio:.3f} outside "
                f"[{min_bbox_ratio:.3f},{max_bbox_ratio:.3f}]"
            )

        if check_last_position:
            min_last_iou = float(
                self.cfg.get("occlusion_recovery_min_last_bbox_iou", 0.15)
            )
            last_iou = self._bbox_iou(self.state.bbox_xyxy, bbox)
            if min_last_iou > 0.0 and last_iou < min_last_iou:
                return f"recovery last-bbox IoU {last_iou:.3f}<" f"{min_last_iou:.3f}"
        return ""

    def _expire_recovery_hypothesis(self, frame_id: int, reason: str) -> None:
        """Keep a pending hypothesis across a short sensor dropout.

        D435 aligned depth commonly contains a blank or fragmented frame when
        an occluding hand leaves.  The previous implementation cleared the
        first recovery hit immediately, making a nominal two-frame confirmation
        require two perfectly consecutive masks.  Ambiguous observations still
        clear immediately; a plain missing/depth-hole frame is tolerated only
        inside the configured freshness window.
        """

        hypothesis = self.recovery_hypothesis
        if hypothesis is None:
            return
        if str(reason).startswith("ambiguous components"):
            self.recovery_hypothesis = None
            return
        max_gap = max(
            1,
            int(self.cfg.get("occlusion_recovery_confirm_max_frame_gap", 3)),
        )
        if int(frame_id) - int(hypothesis.last_frame_id) > max_gap:
            self.recovery_hypothesis = None

    def _confirm_occlusion_recovery(
        self,
        frame: RGBDFrame,
        candidate: _ComponentCandidate,
    ) -> Tuple[bool, str]:
        """Require a unique no-flow candidate to persist across fresh frames.

        The first observation is only a hypothesis and is never published.
        This prevents a passing hand, reflection, or one-frame depth artifact
        from immediately becoming the tracked target.  A consistent candidate
        is accepted on the configured second/third observation.
        """

        required = max(2, int(self.cfg.get("occlusion_recovery_confirm_frames", 2)))
        previous = self.recovery_hypothesis
        consistent = previous is not None
        detail = "new candidate"
        if self._time_aware_motion_enabled():
            committed_delta = self._motion_delta(frame)
            recovery_horizon = max(
                1,
                int(self.cfg.get("occlusion_recovery_motion_max_frame_gap", 6)),
            )
            # Within a short loss, retain target direction using only the last
            # committed state. After a longer complete occlusion that
            # prediction is no longer authoritative; the two tentative
            # candidates are then checked pairwise for bounded speed below,
            # without writing either velocity into committed state.
            if (
                committed_delta is not None
                and int(committed_delta[0]) <= recovery_horizon
            ):
                motion_reason = self._time_aware_motion_reject_reason(
                    frame,
                    candidate.center_uv,
                    apply_prediction=True,
                    max_frame_gap_override=recovery_horizon,
                )
                if motion_reason:
                    self.recovery_hypothesis = None
                    return (
                        False,
                        "occlusion recovery motion rejected: " + motion_reason,
                    )
        if previous is not None:
            frame_gap = int(frame.frame_id) - int(previous.last_frame_id)
            max_frame_gap = max(
                1,
                int(self.cfg.get("occlusion_recovery_confirm_max_frame_gap", 2)),
            )
            timestamp_step = float(frame.timestamp) - float(previous.timestamp_s)
            candidate_iou = self._bbox_iou(previous.bbox_xyxy, candidate.bbox_xyxy)
            min_iou = float(self.cfg.get("occlusion_recovery_confirm_bbox_iou", 0.60))
            area_ratio = float(candidate.area) / max(1.0, float(previous.area))
            max_area_ratio = max(
                1.0,
                float(self.cfg.get("occlusion_recovery_confirm_area_ratio", 1.25)),
            )
            depth_step = abs(
                float(candidate.depth_median) - float(previous.depth_median)
            )
            max_depth_step = float(
                self.cfg.get("occlusion_recovery_confirm_depth_step_m", 0.015)
            )
            if self._time_aware_motion_enabled():
                max_timestamp_step = max(
                    0.0,
                    float(
                        self.cfg.get(
                            "occlusion_recovery_confirm_max_timestamp_step_s",
                            0.20,
                        )
                    ),
                )
                predicted = self._motion_predicted_center(frame)
                prediction_residual = (
                    float(
                        np.hypot(
                            candidate.center_uv[0] - predicted[0],
                            candidate.center_uv[1] - predicted[1],
                        )
                    )
                    if predicted is not None
                    else 0.0
                )
                center_step = float(
                    np.hypot(
                        candidate.center_uv[0] - previous.center_uv[0],
                        candidate.center_uv[1] - previous.center_uv[1],
                    )
                )
                pair_speed = center_step / max(timestamp_step, 1.0e-9)
                max_speed = max(
                    0.0,
                    float(self.cfg.get("component_motion_max_speed_px_s", 2400.0)),
                )
                consistent = (
                    1 <= frame_gap <= max_frame_gap
                    and timestamp_step > 0.0
                    and (
                        max_timestamp_step <= 0.0
                        or timestamp_step <= max_timestamp_step
                    )
                    and (max_speed <= 0.0 or pair_speed <= max_speed)
                    and (1.0 / max_area_ratio) <= area_ratio <= max_area_ratio
                    and depth_step <= max_depth_step
                )
                detail = (
                    f"gap={frame_gap} dt={timestamp_step:.3f}s "
                    f"pair_speed={pair_speed:.1f}px/s "
                    f"prediction_residual={prediction_residual:.1f}px "
                    f"iou={candidate_iou:.3f} area_x={area_ratio:.3f} "
                    f"depth_step={depth_step:.3f}m"
                )
            else:
                center_step = float(
                    np.hypot(
                        candidate.center_uv[0] - previous.center_uv[0],
                        candidate.center_uv[1] - previous.center_uv[1],
                    )
                )
                max_center_step = float(
                    self.cfg.get("occlusion_recovery_confirm_center_step_px", 6.0)
                )
                consistent = (
                    1 <= frame_gap <= max_frame_gap
                    and center_step <= max_center_step
                    and candidate_iou >= min_iou
                    and (1.0 / max_area_ratio) <= area_ratio <= max_area_ratio
                    and depth_step <= max_depth_step
                )
                detail = (
                    f"gap={frame_gap} center_step={center_step:.1f}px "
                    f"iou={candidate_iou:.3f} area_x={area_ratio:.3f} "
                    f"depth_step={depth_step:.3f}m"
                )

        hits = previous.hits + 1 if consistent and previous is not None else 1
        self.recovery_hypothesis = _RecoveryHypothesis(
            mask=candidate.mask.copy(),
            bbox_xyxy=candidate.bbox_xyxy.astype(np.int32).copy(),
            center_uv=(
                float(candidate.center_uv[0]),
                float(candidate.center_uv[1]),
            ),
            area=float(candidate.area),
            depth_median=float(candidate.depth_median),
            confidence=float(candidate.confidence),
            hits=int(hits),
            last_frame_id=int(frame.frame_id),
            timestamp_s=float(frame.timestamp),
        )
        if hits < required:
            return (
                False,
                f"occlusion recovery pending {hits}/{required}: {detail}",
            )
        return True, f"occlusion recovery confirmed {hits}/{required}: {detail}"

    @staticmethod
    def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
        a = np.asarray(first, dtype=np.float64).reshape(4)
        b = np.asarray(second, dtype=np.float64).reshape(4)
        x1 = max(float(a[0]), float(b[0]))
        y1 = max(float(a[1]), float(b[1]))
        x2 = min(float(a[2]), float(b[2]))
        y2 = min(float(a[3]), float(b[3]))
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
        area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
        union = area_a + area_b - intersection
        return 0.0 if union <= 0.0 else float(intersection / union)

    def _tracked_motion_consensus_error(
        self,
        candidate: _ComponentCandidate,
        flow: _FlowPrediction,
    ) -> Optional[str]:
        """Reject a sudden silhouette change after affine motion compensation."""

        # A low-confidence affine is a weak hint, not veto authority.  Let the
        # independently learned color/depth/size gates decide and let temporal
        # smoothing bypass this frame.  Previously even a four-point, low-score
        # KLT estimate could reject an otherwise strong exact-frame component.
        min_flow_confidence = float(
            self.cfg.get("tracked_motion_min_flow_confidence", 0.45)
        )
        if float(getattr(flow, "confidence", 1.0)) < min_flow_confidence:
            return None

        current = np.asarray(candidate.mask) > 0
        predicted = np.asarray(flow.mask) > 0
        iou, _, predicted_area_count = _binary_mask_iou(current, predicted)
        predicted_area = float(predicted_area_count)
        if predicted_area <= 0.0:
            return "motion consensus has an empty predicted mask"
        area_ratio = float(candidate.area) / predicted_area
        min_area_ratio = float(self.cfg.get("tracked_motion_min_area_ratio", 0.70))
        max_area_ratio = float(self.cfg.get("tracked_motion_max_area_ratio", 1.20))
        min_iou = float(self.cfg.get("tracked_motion_min_mask_iou", 0.50))
        if not min_area_ratio <= area_ratio <= max_area_ratio:
            return (
                f"motion mask area_x={area_ratio:.3f} outside "
                f"[{min_area_ratio:.3f},{max_area_ratio:.3f}]"
            )
        if min_iou > 0.0 and iou < min_iou:
            return f"motion mask IoU {iou:.3f}<{min_iou:.3f}"
        return None

    def _stabilize_tracked_candidate(
        self,
        frame: RGBDFrame,
        candidate: _ComponentCandidate,
        flow: _FlowPrediction,
    ) -> _ComponentCandidate:
        """Confirm silhouette changes in motion-compensated coordinates.

        The color/depth Schmitt gate decides which pixels *may* belong to the
        target, but its binary result can still flicker at a reflective edge.
        Blend that current evidence with a soft previous score after applying
        the exact same affine transform used by KLT.  With the deployment
        weights, one absent old pixel is held for one frame and one new pixel
        must be observed twice.  Translation/scale/rotation are already
        compensated, so this does not create a stationary-image lag.

        The blend is deliberately conditional.  Low-confidence flow or a
        materially different raw contour bypasses smoothing rather than
        turning an occluder into persistent foreground.  A retained pixel also
        needs coherent current-frame depth and the total hold area is capped.
        """

        current = np.asarray(candidate.mask) > 0
        predicted = np.asarray(flow.mask) > 0
        raw_iou, _, predicted_area_count = _binary_mask_iou(current, predicted)
        self.last_motion_mask_iou = float(raw_iou)
        self.last_temporal_stabilized = False
        self.last_temporal_motion_source = (
            "geometry"
            if str(getattr(flow, "message", "")).startswith("geometry fallback")
            else "klt"
        )

        def bypass() -> _ComponentCandidate:
            self.temporal_mask_score = current.astype(np.float32)
            return candidate

        if not bool(self.cfg.get("temporal_mask_stability_enabled", True)):
            return bypass()
        min_flow_confidence = float(
            self.cfg.get("temporal_mask_min_flow_confidence", 0.45)
        )
        if float(flow.confidence) < min_flow_confidence:
            return bypass()
        predicted_area = float(predicted_area_count)
        if predicted_area <= 0.0:
            return bypass()
        area_ratio = float(candidate.area) / predicted_area
        min_iou = float(self.cfg.get("temporal_mask_min_raw_iou", 0.65))
        min_area_ratio = float(self.cfg.get("temporal_mask_min_raw_area_ratio", 0.85))
        max_area_ratio = float(self.cfg.get("temporal_mask_max_raw_area_ratio", 1.15))
        if (
            raw_iou < min_iou
            or area_ratio < min_area_ratio
            or area_ratio > max_area_ratio
        ):
            return bypass()

        affine = getattr(flow, "affine_matrix", None)
        previous_score = self.temporal_mask_score
        if (
            affine is None
            or previous_score is None
            or previous_score.shape != current.shape
        ):
            return bypass()
        affine = np.asarray(affine, dtype=np.float32)
        if affine.shape != (2, 3) or not np.isfinite(affine).all():
            return bypass()

        h, w = current.shape
        warped_score = cv2.warpAffine(
            np.asarray(previous_score, dtype=np.float32),
            affine,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

        # Keep the native-resolution warp exactly as-is. Changing its output
        # origin changes OpenCV's fixed-point interpolation coordinates by up
        # to 1/32, which can move a soft edge across the temporal threshold.
        # The remaining pointwise and connected-component work is exactly
        # equivalent inside the union of the non-zero warped support and the
        # already-validated current component. On the 848x480 deployment
        # stream that union is normally only about 45x50 pixels.
        current_u8 = current.view(np.uint8)
        warped_nonzero = cv2.compare(warped_score, 0.0, cv2.CMP_NE)
        support_boxes = []
        for support_mask in (current_u8, warped_nonzero):
            sx, sy, sw, sh = cv2.boundingRect(support_mask)
            if sw > 0 and sh > 0:
                support_boxes.append((sx, sy, sx + sw, sy + sh))
        if not support_boxes:
            return bypass()
        rx1 = min(box[0] for box in support_boxes)
        ry1 = min(box[1] for box in support_boxes)
        rx2 = max(box[2] for box in support_boxes)
        ry2 = max(box[3] for box in support_boxes)
        support_slice = (slice(ry1, ry2), slice(rx1, rx2))
        current_local = current[support_slice]
        warped_score_local = warped_score[support_slice]
        current_weight = float(
            np.clip(
                self.cfg.get("temporal_mask_current_weight", 0.45),
                0.0,
                1.0,
            )
        )
        score = (
            1.0 - current_weight
        ) * warped_score_local + current_weight * current_local.astype(np.float32)
        score = np.clip(score, 0.0, 1.0)
        threshold = float(
            np.clip(self.cfg.get("temporal_mask_threshold", 0.50), 0.0, 1.0)
        )
        stable = score >= threshold

        # A semantic deployment may retain a briefly missing, motion-warped
        # edge without asking D435 depth to decide whether that RGB pixel still
        # belongs to the object.  The point-cloud projector separately rejects
        # invalid/background/closer-hand depth.  Legacy depth-owned trackers
        # keep the stricter behavior by default.
        retained = stable & ~current_local
        retention_depth_gate = bool(
            self.cfg.get("temporal_mask_retention_depth_gate_enabled", True)
        )
        if np.any(retained) and retention_depth_gate:
            tolerance = float(
                self.cfg.get(
                    "temporal_mask_retention_depth_tolerance_m",
                    self.depth_half_width,
                )
            )
            retained_depth = frame.depth_raw[support_slice][retained].astype(
                np.float32
            ) * float(frame.depth_scale)
            coherent_retained = (
                np.isfinite(retained_depth)
                & (retained_depth > 0.0)
                & (np.abs(retained_depth - float(candidate.depth_median)) <= tolerance)
            )
            if not bool(np.all(coherent_retained)):
                rejected_retained = retained.copy()
                rejected_retained[retained] = ~coherent_retained
                stable[rejected_retained] = False
                score[rejected_retained] = 0.0
            retained = stable & ~current_local

        max_hold_pixels = max(
            int(self.cfg.get("temporal_mask_max_hold_pixels", 48)),
            int(
                np.ceil(
                    float(self.cfg.get("temporal_mask_max_hold_fraction", 0.08))
                    * predicted_area
                )
            ),
        )
        if int(retained.sum()) > max_hold_pixels:
            return bypass()

        # Keep the connected target component with the largest overlap with
        # the already-validated current component.  Soft evidence from a
        # detached reflection can therefore never become another output blob.
        stable_u8 = stable.astype(np.uint8)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            stable_u8, connectivity=8
        )
        if count <= 1:
            return bypass()
        overlaps = np.bincount(
            labels[current_local].reshape(-1), minlength=count
        ).astype(np.int64)
        overlaps[0] = 0
        label = int(np.argmax(overlaps))
        if label <= 0 or int(overlaps[label]) == 0:
            return bypass()
        stable = labels == label
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < float(self.cfg.get("min_area", 80)):
            return bypass()

        bx = int(stats[label, cv2.CC_STAT_LEFT]) + rx1
        by = int(stats[label, cv2.CC_STAT_TOP]) + ry1
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        bbox = np.asarray([bx, by, bx + bw, by + bh], dtype=np.int32)
        # Adding an integer crop origin to a floating local centroid may round
        # one ulp differently from the original full-frame integer moment.
        # Reconstruct the exact global moment so long replay state remains
        # bitwise identical.
        stable_y, stable_x = np.nonzero(stable)
        area_count = int(stable_x.size)
        center = (
            float(
                (stable_x.sum(dtype=np.int64) + np.int64(rx1) * np.int64(area_count))
                / area_count
            ),
            float(
                (stable_y.sum(dtype=np.int64) + np.int64(ry1) * np.int64(area_count))
                / area_count
            ),
        )
        valid_depths = frame.depth_raw[support_slice][stable].astype(
            np.float32
        ) * float(frame.depth_scale)
        valid_depths = valid_depths[np.isfinite(valid_depths) & (valid_depths > 0.0)]
        depth_median = (
            float(np.median(valid_depths))
            if valid_depths.size
            else float(candidate.depth_median)
        )

        # Preserve sub-threshold *current* pixels in the soft score so a real
        # new edge can cross the threshold on its second observation.  Remove
        # every unrelated soft island before the next warp.
        score[~(stable | current_local)] = 0.0
        temporal_score = np.zeros_like(warped_score, dtype=np.float32)
        temporal_score[support_slice] = score
        self.temporal_mask_score = temporal_score
        self.last_temporal_stabilized = not np.array_equal(stable, current_local)
        stable_full = np.zeros_like(current, dtype=np.uint8)
        stable_full[support_slice] = stable.astype(np.uint8)
        return _ComponentCandidate(
            mask=stable_full,
            bbox_xyxy=bbox,
            center_uv=center,
            area=area,
            depth_median=depth_median,
            confidence=float(candidate.confidence),
            color_score=float(candidate.color_score),
            search_scope=candidate.search_scope,
        )

    def _appearance_update_core(
        self,
        candidate: _ComponentCandidate,
        flow: Optional[_FlowPrediction],
    ) -> Optional[np.ndarray]:
        """Return a high-certainty core for adaptive appearance learning."""

        if flow is None:
            return None
        min_flow_confidence = float(
            self.cfg.get("appearance_update_min_flow_confidence", 0.35)
        )
        if float(flow.confidence) < min_flow_confidence:
            return None
        core = np.logical_and(
            np.asarray(candidate.mask) > 0,
            np.asarray(flow.mask) > 0,
        ).astype(np.uint8)
        kernel_size = max(1, int(self.cfg.get("appearance_update_erode_kernel", 3)))
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            core = cv2.erode(core, kernel, iterations=1)
        min_pixels = max(
            int(self.cfg.get("appearance_update_min_core_pixels", 40)),
            int(
                np.ceil(
                    float(self.cfg.get("appearance_update_min_core_fraction", 0.35))
                    * max(1.0, float(candidate.area))
                )
            ),
        )
        if int(core.sum()) < min_pixels:
            return None
        return core

    def _reject(self, frame: RGBDFrame, reason: str) -> MaskResult:
        assert self.state is not None
        self.state.lost_count += 1
        self.state.valid = False
        self.last_confidence = 0.0
        h, w = frame.depth_raw.shape
        # Never turn a stale mask into a current point cloud.  Keep only the
        # previous bbox as a search/re-detection hint.
        return MaskResult(
            mask=np.zeros((h, w), dtype=np.uint8),
            bbox_xyxy=self.state.bbox_xyxy.copy(),
            score=0.0,
            valid=False,
            message=f"adaptive lost {self.state.lost_count}: {reason}",
        )

    def _invalid_from_snapshot(
        self,
        frame: RGBDFrame,
        snapshot: AdaptiveTrackerSnapshot,
        reason: str,
    ) -> MaskResult:
        self.restore_state(snapshot)
        assert self.state is not None
        self.state.lost_count += 1
        self.state.valid = False
        self.last_confidence = 0.0
        h, w = frame.depth_raw.shape
        return MaskResult(
            mask=np.zeros((h, w), dtype=np.uint8),
            bbox_xyxy=self.state.bbox_xyxy.copy(),
            score=0.0,
            valid=False,
            message=f"adaptive {reason}",
        )

    def _initialize_appearance(
        self,
        frame: RGBDFrame,
        foreground_core: np.ndarray,
        foreground_exclusion: np.ndarray,
        bbox: np.ndarray,
    ) -> None:
        """Learn target from a deep core and background outside the full seed.

        ``foreground_core`` is deliberately conservative appearance evidence.
        ``foreground_exclusion`` is the complete, already-validated semantic
        seed and is used only to keep every target-boundary pixel out of the
        background model.  Keeping those roles separate avoids teaching a
        motion-blurred target rim as negative evidence when the foreground
        histogram uses a deeply eroded core.
        """

        expected_shape = frame.depth_raw.shape
        core = self._validated_appearance_seed_mask(
            foreground_core,
            expected_shape=expected_shape,
            label="foreground core",
        )
        exclusion = self._validated_appearance_seed_mask(
            foreground_exclusion,
            expected_shape=expected_shape,
            label="foreground exclusion",
        )
        if np.any(np.logical_and(core > 0, exclusion == 0)):
            raise ValueError(
                "appearance foreground core must be contained in the full "
                "foreground exclusion"
            )

        hsv = cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2HSV)
        anchor_hs, anchor_sv = self._histograms(hsv, core)

        h, w = expected_shape
        ring_bbox = enlarge_bbox(
            bbox,
            float(self.cfg.get("background_ring_scale", 1.8)),
            w,
            h,
        )
        ring = bbox_to_mask(ring_bbox, (h, w))
        dilation = int(self.cfg.get("background_exclusion_dilation", 7))
        if dilation > 1:
            kernel = np.ones((dilation, dilation), dtype=np.uint8)
            excluded = cv2.dilate(exclusion, kernel, iterations=1)
        else:
            excluded = exclusion
        background = ring & (excluded == 0).astype(np.uint8)
        if int(background.sum()) < 40:
            background = (excluded == 0).astype(np.uint8)
        background_hs, background_sv = self._histograms(hsv, background)

        # Publish the six coupled histograms only after every input and both
        # histogram computations have succeeded.  Invalid initialization data
        # can therefore never leave a half-written appearance model behind.
        self.anchor_hs = anchor_hs
        self.anchor_sv = anchor_sv
        self.adaptive_hs = anchor_hs.copy()
        self.adaptive_sv = anchor_sv.copy()
        self.background_hs = background_hs
        self.background_sv = background_sv
        self._advance_appearance_revision()

    @staticmethod
    def _validated_appearance_seed_mask(
        mask: np.ndarray,
        *,
        expected_shape: Tuple[int, int],
        label: str,
    ) -> np.ndarray:
        array = np.asarray(mask)
        if array.shape != expected_shape:
            raise ValueError(
                f"appearance {label} shape {array.shape} does not match "
                f"frame shape {expected_shape}"
            )
        if not (
            np.issubdtype(array.dtype, np.bool_)
            or (np.issubdtype(array.dtype, np.number) and not np.iscomplexobj(array))
        ):
            raise ValueError(f"appearance {label} must be a real numeric mask")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"appearance {label} contains non-finite values")
        mask_u8 = (array > 0).astype(np.uint8)
        if cv2.countNonZero(mask_u8) == 0:
            raise RuntimeError(f"appearance {label} is empty")
        return mask_u8

    def _appearance_training_mask(
        self,
        mask: np.ndarray,
        bbox: np.ndarray,
        box_prompt_only: bool,
    ) -> np.ndarray:
        mask_u8 = (mask > 0).astype(np.uint8)
        if box_prompt_only:
            x1, y1, x2, y2 = bbox.astype(float)
            shrink = float(self.cfg.get("box_prompt_seed_scale", 0.65))
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            bw = max(2.0, (x2 - x1) * shrink)
            bh = max(2.0, (y2 - y1) * shrink)
            seed = bbox_to_mask(
                np.asarray([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]),
                mask_u8.shape,
            )
            return mask_u8 & seed

        erode = int(self.cfg.get("appearance_seed_erode", 11))
        if erode > 1:
            kernel = np.ones((erode, erode), dtype=np.uint8)
            inner = cv2.erode(mask_u8, kernel, iterations=1)
            inner_pixels = int(inner.sum())
            seed_pixels = int(mask_u8.sum())
            if inner_pixels >= 20 and inner_pixels >= 0.25 * seed_pixels:
                return inner
        return mask_u8

    def _histograms(
        self, hsv: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        mask_u8 = (mask > 0).astype(np.uint8)
        if int(mask_u8.sum()) == 0:
            raise RuntimeError("cannot learn appearance from an empty mask")
        h_bins = int(self.cfg.get("hue_bins", 24))
        s_bins = int(self.cfg.get("saturation_bins", 16))
        v_bins = int(self.cfg.get("value_bins", 16))
        hs = cv2.calcHist(
            [hsv], [0, 1], mask_u8, [h_bins, s_bins], [0, 180, 0, 256]
        ).astype(np.float32)
        sv = cv2.calcHist(
            [hsv], [1, 2], mask_u8, [s_bins, v_bins], [0, 256, 0, 256]
        ).astype(np.float32)
        return self._smooth_hist(hs, cyclic_first_axis=True), self._smooth_hist(
            sv, cyclic_first_axis=False
        )

    def _smooth_hist(self, hist: np.ndarray, cyclic_first_axis: bool) -> np.ndarray:
        total = float(hist.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError("invalid non-finite or zero-sum appearance histogram")
        # Normalize before adding the smoothing prior.  Otherwise deeply
        # eroding a uniform seed changes only its sample count, yet changes
        # the prior's relative weight and can substantially alter likelihood
        # ratios in unseen colour bins.  The learned distribution must depend
        # on pixel colours, not on how many identical pixels were sampled.
        hist = hist / total
        smoothing = float(self.cfg.get("histogram_smoothing", 1e-6))
        if not np.isfinite(smoothing) or smoothing < 0.0:
            raise ValueError("histogram_smoothing must be finite and non-negative")
        hist = hist + smoothing
        hist = cv2.GaussianBlur(hist, (3, 3), sigmaX=0.8, sigmaY=0.8)
        if cyclic_first_axis:
            hist = (
                0.20 * np.roll(hist, 1, axis=0)
                + 0.60 * hist
                + 0.20 * np.roll(hist, -1, axis=0)
            )
        total = float(hist.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError("invalid zero-sum appearance histogram")
        return (hist / total).astype(np.float32)

    def _color_probability(self, hsv: np.ndarray) -> np.ndarray:
        assert self.anchor_hs is not None
        assert self.anchor_sv is not None
        assert self.adaptive_hs is not None
        assert self.adaptive_sv is not None
        assert self.background_hs is not None
        assert self.background_sv is not None
        anchor_weight = float(
            np.clip(self.cfg.get("appearance_anchor_weight", 0.70), 0.0, 1.0)
        )
        fg_hs_hist = (
            anchor_weight * self.anchor_hs + (1.0 - anchor_weight) * self.adaptive_hs
        )
        fg_sv_hist = (
            anchor_weight * self.anchor_sv + (1.0 - anchor_weight) * self.adaptive_sv
        )

        bg_weight = float(self.cfg.get("background_likelihood_weight", 1.0))
        eps = np.float32(1e-12)
        # The likelihood ratio is constant within each histogram bin. Compute it
        # once on the tiny 24x16 / 16x16 tables, then use OpenCV's C++
        # back-projection for the full image. The former NumPy path allocated
        # three int32 index images plus four advanced-index result images and
        # cost ~25 ms for a full 848x480 LOST scan on this deployment host.
        hs_ratio_hist = fg_hs_hist / (fg_hs_hist + bg_weight * self.background_hs + eps)
        sv_ratio_hist = fg_sv_hist / (fg_sv_hist + bg_weight * self.background_sv + eps)
        hsv_f32 = hsv.astype(np.float32)
        hs_ratio = cv2.calcBackProject(
            [hsv_f32],
            [0, 1],
            hs_ratio_hist.astype(np.float32),
            [0.0, 180.0, 0.0, 256.0],
            1.0,
        )
        sv_ratio = cv2.calcBackProject(
            [hsv_f32],
            [1, 2],
            sv_ratio_hist.astype(np.float32),
            [0.0, 256.0, 0.0, 256.0],
            1.0,
        )

        # Hue is unreliable for gray/white pixels.  This per-pixel weighting is
        # generic and learned histograms still determine every accepted color.
        saturation_reliability = np.clip((hsv_f32[..., 1] - 16.0) / 96.0, 0.0, 1.0)
        hs_weight = 0.25 + 0.55 * saturation_reliability
        probability = hs_weight * hs_ratio + (1.0 - hs_weight) * sv_ratio
        return probability.astype(np.float32)

    def _mean_color_probability(
        self, frame: RGBDFrame, mask: np.ndarray, bbox: np.ndarray
    ) -> float:
        h, w = frame.depth_raw.shape
        bbox = clip_bbox(bbox, w, h)
        x1, y1, x2, y2 = bbox.astype(int)
        local_mask = mask[y1:y2, x1:x2].astype(bool)
        if not np.any(local_mask):
            return 0.0
        hsv = cv2.cvtColor(frame.color_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        probability = self._color_probability(hsv)
        return float(np.mean(probability[local_mask]))

    def _appearance_model_available(self) -> bool:
        return bool(
            self.anchor_hs is not None
            and self.anchor_sv is not None
            and self.adaptive_hs is not None
            and self.adaptive_sv is not None
            and self.background_hs is not None
            and self.background_sv is not None
        )

    @staticmethod
    def _timestamp_f64_bits(timestamp: float) -> int:
        """Return the exact IEEE-754 identity used by frame-bound evidence."""

        return int(np.float64(float(timestamp)).view(np.uint64).item())

    def target_appearance_probability_roi(
        self,
        frame: RGBDFrame,
        root_mask: np.ndarray,
    ) -> Optional[TargetAppearanceProbabilityROI]:
        """Compute immutable appearance probability once for one mask root.

        The returned evidence may be supplied to the two read-only appearance
        query methods below.  They reuse it only for the exact same frame and
        appearance revision, and only when the query mask is a subset of this
        root.  Any mismatch silently falls back to the legacy computation.
        Empty roots return ``None`` because there is no useful probability
        rectangle to cache.
        """

        if not self._appearance_model_available():
            return None
        mask_u8 = (np.asarray(root_mask) > 0).astype(np.uint8)
        if mask_u8.shape != frame.depth_raw.shape:
            raise ValueError("appearance support mask shape does not match RGB-D frame")
        if cv2.countNonZero(mask_u8) == 0:
            return None

        x, y, width, height = cv2.boundingRect(mask_u8)
        hsv = cv2.cvtColor(
            frame.color_bgr[y : y + height, x : x + width],
            cv2.COLOR_BGR2HSV,
        )
        probability = np.ascontiguousarray(
            self._color_probability(hsv), dtype=np.float32
        )

        # Immutable byte backing prevents a caller from changing the evidence
        # after its digest and model/frame bindings have been recorded.
        root_mask_bytes = np.ascontiguousarray(mask_u8).tobytes(order="C")
        frozen_root = np.ndarray(
            mask_u8.shape,
            dtype=np.uint8,
            buffer=root_mask_bytes,
            order="C",
        )
        probability_bytes = probability.tobytes(order="C")
        frozen_probability = np.ndarray(
            probability.shape,
            dtype=np.float32,
            buffer=probability_bytes,
            order="C",
        )
        return TargetAppearanceProbabilityROI(
            frame_id=int(frame.frame_id),
            timestamp_f64_bits=self._timestamp_f64_bits(frame.timestamp),
            appearance_revision=self.appearance_revision,
            image_shape_hw=(int(mask_u8.shape[0]), int(mask_u8.shape[1])),
            root_mask_digest=hashlib.sha256(root_mask_bytes).hexdigest(),
            bbox_xyxy=(int(x), int(y), int(x + width), int(y + height)),
            root_mask=frozen_root,
            probability=frozen_probability,
            root_mask_bytes=root_mask_bytes,
            probability_bytes=probability_bytes,
            _tracker_token=self._appearance_probability_roi_token,
        )

    def _reusable_target_appearance_probability(
        self,
        frame: RGBDFrame,
        mask_u8: np.ndarray,
        query_bbox_xywh: Tuple[int, int, int, int],
        probability_roi: Optional[TargetAppearanceProbabilityROI],
    ) -> Optional[np.ndarray]:
        """Return a query-sized probability view only for valid evidence."""

        evidence = probability_roi
        if not isinstance(evidence, TargetAppearanceProbabilityROI):
            return None
        if evidence._tracker_token is not self._appearance_probability_roi_token:
            return None
        if int(evidence.frame_id) != int(frame.frame_id):
            return None
        if int(evidence.timestamp_f64_bits) != self._timestamp_f64_bits(
            frame.timestamp
        ):
            return None
        if int(evidence.appearance_revision) != self.appearance_revision:
            return None
        expected_shape = (int(frame.depth_raw.shape[0]), int(frame.depth_raw.shape[1]))
        if tuple(evidence.image_shape_hw) != expected_shape:
            return None
        if (
            evidence.root_mask.shape != expected_shape
            or evidence.root_mask.dtype != np.uint8
            or evidence.root_mask.flags.writeable
            or evidence.root_mask.base is not evidence.root_mask_bytes
        ):
            return None

        ex1, ey1, ex2, ey2 = (int(value) for value in evidence.bbox_xyxy)
        if not (0 <= ex1 < ex2 <= expected_shape[1]):
            return None
        if not (0 <= ey1 < ey2 <= expected_shape[0]):
            return None
        if (
            evidence.probability.shape != (ey2 - ey1, ex2 - ex1)
            or evidence.probability.dtype != np.float32
            or evidence.probability.flags.writeable
            or evidence.probability.base is not evidence.probability_bytes
        ):
            return None

        x, y, width, height = query_bbox_xywh
        x2 = x + width
        y2 = y + height
        if x < ex1 or y < ey1 or x2 > ex2 or y2 > ey2:
            return None
        query_crop = mask_u8[y:y2, x:x2]
        root_crop = evidence.root_mask[y:y2, x:x2]
        if np.any(np.logical_and(query_crop > 0, root_crop == 0)):
            return None
        return evidence.probability[
            y - ey1 : y2 - ey1,
            x - ex1 : x2 - ex1,
        ]

    def _target_appearance_probability_view(
        self,
        frame: RGBDFrame,
        mask_u8: np.ndarray,
        probability_roi: Optional[TargetAppearanceProbabilityROI],
    ) -> Tuple[int, int, int, int, np.ndarray, np.ndarray]:
        x, y, width, height = cv2.boundingRect(mask_u8)
        local_mask = mask_u8[y : y + height, x : x + width].astype(bool)
        probability = self._reusable_target_appearance_probability(
            frame,
            mask_u8,
            (int(x), int(y), int(width), int(height)),
            probability_roi,
        )
        if probability is None:
            hsv = cv2.cvtColor(
                frame.color_bgr[y : y + height, x : x + width],
                cv2.COLOR_BGR2HSV,
            )
            probability = self._color_probability(hsv)
        return int(x), int(y), int(width), int(height), local_mask, probability

    def target_appearance_support_stats(
        self,
        frame: RGBDFrame,
        mask: np.ndarray,
        *,
        threshold: Optional[float] = None,
        probability_roi: Optional[TargetAppearanceProbabilityROI] = None,
    ) -> Optional[Tuple[float, float, int]]:
        """Score arbitrary pixels against the immutable target appearance.

        The provider uses this only as a read-only guard for pixels which a
        semantic mask added outside the independently tracked target core.  It
        intentionally does not update either the anchor or adaptive
        appearance model.

        Returns ``(mean_probability, supported_fraction, pixel_count)`` or
        ``None`` when the target appearance model is not available.
        """

        if not self._appearance_model_available():
            return None
        mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
        if mask_u8.shape != frame.depth_raw.shape:
            raise ValueError("appearance support mask shape does not match RGB-D frame")
        if cv2.countNonZero(mask_u8) == 0:
            return (0.0, 0.0, 0)
        _, _, _, _, local_mask, probability = (
            self._target_appearance_probability_view(
                frame,
                mask_u8,
                probability_roi,
            )
        )
        values = probability[local_mask]
        support_threshold = float(
            self.cfg.get("flow_color_probability_floor", 0.42)
            if threshold is None
            else threshold
        )
        return (
            float(np.mean(values)),
            float(np.mean(values >= support_threshold)),
            int(values.size),
        )

    def target_appearance_supported_mask(
        self,
        frame: RGBDFrame,
        mask: np.ndarray,
        *,
        threshold: Optional[float] = None,
        probability_roi: Optional[TargetAppearanceProbabilityROI] = None,
    ) -> Optional[np.ndarray]:
        """Return immutable-appearance pixels inside an arbitrary mask.

        This is the pixelwise counterpart of
        :meth:`target_appearance_support_stats`.  It is deliberately read-only:
        the anchor/adaptive histograms are scored but never updated.  The
        provider uses it for read-only startup silhouette validation and after
        the source-agnostic final guard detects a foreign appendage, so an
        occluding RH56 finger can be removed while the still-visible object
        core remains publishable.
        """

        if not self._appearance_model_available():
            return None
        mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
        if mask_u8.shape != frame.depth_raw.shape:
            raise ValueError("appearance support mask shape does not match RGB-D frame")
        supported = np.zeros_like(mask_u8)
        if cv2.countNonZero(mask_u8) == 0:
            return supported
        x, y, width, height, local_mask, probability = (
            self._target_appearance_probability_view(
                frame,
                mask_u8,
                probability_roi,
            )
        )
        support_threshold = float(
            self.cfg.get("flow_color_probability_floor", 0.42)
            if threshold is None
            else threshold
        )
        supported[y : y + height, x : x + width] = np.logical_and(
            local_mask, probability >= support_threshold
        ).astype(np.uint8)
        return supported

    def _update_adaptive_appearance(
        self,
        frame: RGBDFrame,
        mask: np.ndarray,
        alpha_override: Optional[float] = None,
    ) -> None:
        if self.adaptive_hs is None or self.adaptive_sv is None:
            return
        mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
        if cv2.countNonZero(mask_u8) == 0:
            return
        x, y, width, height = cv2.boundingRect(mask_u8)
        mask_crop = mask_u8[y : y + height, x : x + width]
        hsv = cv2.cvtColor(
            frame.color_bgr[y : y + height, x : x + width],
            cv2.COLOR_BGR2HSV,
        )
        try:
            current_hs, current_sv = self._histograms(hsv, mask_crop)
        except RuntimeError:
            return
        alpha = (
            float(alpha_override)
            if alpha_override is not None
            else float(self.cfg.get("appearance_update_alpha", 0.025))
        )
        alpha = float(np.clip(alpha, 0.0, 0.25))
        self.adaptive_hs = self._renormalize(
            (1.0 - alpha) * self.adaptive_hs + alpha * current_hs
        )
        self.adaptive_sv = self._renormalize(
            (1.0 - alpha) * self.adaptive_sv + alpha * current_sv
        )
        self._advance_appearance_revision()

    @staticmethod
    def _renormalize(hist: np.ndarray) -> np.ndarray:
        total = float(hist.sum())
        if total <= 0.0:
            return hist.astype(np.float32)
        return (hist / total).astype(np.float32)

    @staticmethod
    def _copy_array(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
        return None if value is None else value.copy()
