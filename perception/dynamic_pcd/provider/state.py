"""State records for guarded object point-cloud provider transactions.

The records in this module carry evidence, proofs, pending commits, and
continuity state. They intentionally contain no camera, tracker, publication,
or hardware behavior; :mod:`object_pcd_provider` remains the transactional
owner and re-exports these names for import compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from dynamic_pcd.types import MaskResult, RGBDFrame

@dataclass
class _OnlineMaskGeometry:
    """Cheap, source-agnostic RGB-D geometry for one external mask."""

    mask: np.ndarray
    bbox_xyxy: np.ndarray
    area: float
    bbox_area: float
    centroid_xy: np.ndarray
    valid_depth_ratio: float
    depth_median: float
    depth_spread: float


@dataclass(frozen=True)
class _FrameMaskEvidence:
    """Immutable geometry token for one provider-owned exact-frame mask.

    This is a computation cache, never an authority shortcut.  A hit requires
    object identity with the read-only binary mask backed by ``mask_bytes`` as
    well as the exact frame/timestamp/target generation.  Equal external
    arrays intentionally miss.  ``authority_eligible`` records provenance so
    a sanitizer-owned token can never be recycled as raw/full/bootstrap
    evidence.
    """

    frame_id: int
    timestamp_bits: int
    target_generation: int
    mask: np.ndarray
    mask_bytes: bytes
    mask_digest: str
    evidence_kind: str
    authority_eligible: bool
    geometry: _OnlineMaskGeometry
    appearance_probability_roi: Optional[Any]


@dataclass
class _OnlineRecoveryHypothesis:
    """A validated but not yet robot-facing online recovery observation."""

    frame_id: int
    bbox_xyxy: np.ndarray
    center_xy: np.ndarray
    area: float
    depth_median: float
    timestamp: float
    center_velocity_px_s: np.ndarray
    hits: int
    source: str = "lost_recovery"


@dataclass
class _TrustedOnlineSAM2Publication:
    """Last exact SAM2 mask which crossed the final publication boundary."""

    frame_id: int
    mask: np.ndarray
    bbox_xyxy: np.ndarray
    center_xy: np.ndarray
    area: float
    depth_median: float
    timestamp: float
    center_velocity_px_s: np.ndarray
    log_area_rate_s: float
    velocity_samples: int
    mask_evidence: Optional[_FrameMaskEvidence] = None


@dataclass
class _FinalCleanPublication:
    """Last mask which crossed the source-agnostic publication boundary."""

    frame_id: int
    timestamp: float
    mask: np.ndarray
    bbox_xyxy: np.ndarray
    area: float
    center_xy: np.ndarray
    center_velocity_px_s: np.ndarray
    log_area_rate_s: float
    motion_samples: int
    mask_evidence: Optional[_FrameMaskEvidence] = None


@dataclass
class _FullTargetGeometryAuthority:
    """Clean, unsanitized exact-SAM scale history for guarded recovery.

    The robot-facing final-clean history intentionally follows the visible
    target core during an occlusion.  That smaller partial mask is useful for
    policy input and motion, but it must not become the ruler used to decide
    whether the full object has reappeared.  This compact authority advances
    only after an unsanitized exact-SAM mask crosses the final RGB-D boundary.
    """

    frame_id: int
    timestamp: float
    area: float
    bbox_area: float
    log_area_rate_s: float
    scale_samples: int


@dataclass
class _DeepOcclusionBroadSeed:
    """One non-publishing broad-core observation before deep occlusion.

    The ordinary broad-core contract intentionally requires at least half of
    the fixed target anchor to remain visible.  A hand can hide more than that
    while a still-useful, current-frame target fragment remains.  This seed is
    therefore only a transaction precursor: it binds one exact raw SAM mask,
    its immutable p=.20/p=.42 appearance subsets, a robot-facing subset of the
    fixed full-target anchor, and the exact strict owner expected to be
    finally committed for this frame.  A different RGB-D tuple must confirm
    it before the broader fragment may be published.
    """

    frame_id: int
    timestamp_bits: int
    target_generation: int
    online_step_count: int
    expected_owner_frame_id: int
    expected_owner_timestamp_bits: int
    expected_owner_hits: int
    expected_owner_mask_digest: str
    raw_mask_digest: str
    broad_mask_digest: str
    broad_pixels: int
    strict_mask_digest: str
    strict_pixels: int
    candidate_mask: np.ndarray
    candidate_mask_digest: str
    candidate_pixels: int
    candidate_bbox_xyxy: np.ndarray
    candidate_center_xy: np.ndarray
    candidate_depth_median: float
    fixed_anchor_mask: np.ndarray
    fixed_anchor_digest: str
    fixed_anchor_frame_id: int
    fixed_anchor_pixels: int
    aligned_anchor_mask: np.ndarray
    aligned_anchor_digest: str
    aligned_anchor_pixels: int
    broad_raw_fraction_bits: int
    strict_raw_fraction_bits: int
    candidate_anchor_fraction_bits: int
    candidate_broad_fraction_bits: int
    config_fingerprint: str
    seed_fingerprint: str


@dataclass
class _BroadVisibleCoreProof:
    """Frame-bound proof for one broader, still non-authoritative target core.

    The normal 0.42 appearance core is deliberately conservative.  During a
    confirmed hand occlusion it can become too small to preserve the visible
    object's geometry even though the exact SAM envelope still contains a
    clean, high-recall target subset.  This token binds that broader subset to
    the current raw exact mask, a strict current-frame identity core, and one
    non-ratcheting fixed anchor.  The anchor is evidence only: the full broad
    candidate is published, while no full/trusted/appearance authority is
    advanced.
    """

    frame_id: int
    timestamp_bits: int
    target_generation: int
    owner_frame_id: int
    owner_timestamp_bits: int
    owner_hits: int
    owner_mask_digest: str
    candidate_mask_digest: str
    candidate_pixels: int
    strict_mask: np.ndarray
    strict_mask_digest: str
    strict_pixels: int
    anchor_mask: np.ndarray
    anchor_mask_digest: str
    anchor_frame_id: int
    anchor_bbox_xyxy: np.ndarray
    anchor_center_xy: np.ndarray
    anchor_pixels: int
    aligned_anchor_mask: np.ndarray
    aligned_anchor_mask_digest: str
    aligned_anchor_bbox_xyxy: np.ndarray
    aligned_anchor_center_xy: np.ndarray
    aligned_anchor_pixels: int
    overlap_pixels: int
    shift_xy: np.ndarray
    broad_probability_threshold_bits: int
    strict_probability_threshold_bits: int
    min_anchor_overlap_fraction_bits: int
    min_candidate_overlap_fraction_bits: int
    max_candidate_anchor_area_ratio_bits: int
    alignment_radius_px: int
    # Deep-occlusion mode is a distinct two-frame transaction.  Defaults keep
    # every existing standard broad-core caller/test byte-for-byte compatible.
    deep_occlusion_mode: bool = False
    deep_seed: Optional[_DeepOcclusionBroadSeed] = None
    deep_seed_fingerprint: str = ""
    deep_pair_aligned_seed_mask: Optional[np.ndarray] = None
    deep_pair_aligned_seed_digest: str = ""
    deep_pair_overlap_pixels: int = 0
    deep_pair_seed_fraction_bits: int = 0
    deep_pair_current_fraction_bits: int = 0
    deep_pair_iou_bits: int = 0
    deep_pair_center_step_bits: int = 0
    deep_pair_depth_step_bits: int = 0
    deep_pair_depth_limit_bits: int = 0
    deep_config_fingerprint: str = ""


@dataclass(frozen=True)
class _DeepOcclusionBroadStageSeal:
    """One frame-scoped immutable latch for an already computed deep proof.

    Stage performs the expensive appearance, fixed-anchor, RGB-D, PCD and
    pair calculations once.  Publication/final/history boundaries still
    validate exact frame bytes, mask tokens, owner, config and every proof
    field through this provider-side latch; they no longer rerun the same
    image operations three times inside one 20 Hz tick.
    """

    pending: "_PendingFullTargetGeometryCommit"
    proof: _BroadVisibleCoreProof
    proof_fingerprint: str
    frame: RGBDFrame
    frame_content_digest: str
    raw_evidence: _FrameMaskEvidence
    final_evidence: _FrameMaskEvidence
    previous_owner: "_VisibleExactPartialContinuity"
    previous_owner_fingerprint: str
    frame_id: int
    timestamp_bits: int
    target_generation: int
    pending_source: str
    pending_eligibility_kind: str
    pending_raw_exact_eligible: bool
    pending_bootstrap_eligible: bool


@dataclass
class _LostReappearanceVisibleCoreSeed:
    """First exact visible-core observation while guarded tracking is LOST.

    This seed is deliberately not a publication authority.  It binds one
    fail-closed exact SAM observation to the immutable full-target anchor so a
    second independent RGB-D tuple can prove that a rapidly uncovering target
    is real without weakening the normal three-evidence recovery protocol.
    """

    frame_id: int
    timestamp_bits: int
    target_generation: int
    online_step_count: int
    raw_mask: np.ndarray
    raw_mask_digest: str
    raw_pixels: int
    raw_bbox_xyxy: np.ndarray
    raw_center_xy: np.ndarray
    raw_depth_median: float
    candidate_mask: np.ndarray
    candidate_mask_digest: str
    candidate_pixels: int
    candidate_bbox_xyxy: np.ndarray
    candidate_center_xy: np.ndarray
    strict_mask: np.ndarray
    strict_mask_digest: str
    strict_pixels: int
    anchor_mask: np.ndarray
    anchor_mask_digest: str
    anchor_frame_id: int
    anchor_timestamp_bits: int
    anchor_bbox_xyxy: np.ndarray
    anchor_center_xy: np.ndarray
    anchor_pixels: int
    aligned_anchor_mask: np.ndarray
    aligned_anchor_mask_digest: str
    aligned_anchor_pixels: int
    anchor_shift_xy: np.ndarray
    anchor_overlap_pixels: int
    broad_probability_threshold_bits: int
    strict_probability_threshold_bits: int
    min_candidate_raw_fraction_bits: int
    min_strict_raw_fraction_bits: int
    min_candidate_anchor_fraction_bits: int
    min_anchor_candidate_fraction_bits: int
    anchor_alignment_radius_px: int
    boundary_seed_fine_core: bool = False
    boundary_seed_fine_model_fingerprint: str = ""


@dataclass
class _LostReappearanceVisibleCoreProof:
    """Frame-bound proof for the second LOST reappearance observation."""

    frame_id: int
    timestamp_bits: int
    target_generation: int
    seed: _LostReappearanceVisibleCoreSeed
    raw_mask_digest: str
    raw_pixels: int
    candidate_mask_digest: str
    candidate_pixels: int
    strict_mask: np.ndarray
    strict_mask_digest: str
    strict_pixels: int
    anchor_mask: np.ndarray
    anchor_mask_digest: str
    anchor_frame_id: int
    anchor_timestamp_bits: int
    aligned_anchor_mask: np.ndarray
    aligned_anchor_mask_digest: str
    aligned_anchor_pixels: int
    anchor_shift_xy: np.ndarray
    anchor_overlap_pixels: int
    pair_aligned_seed_mask: np.ndarray
    pair_aligned_seed_digest: str
    pair_shift_xy: np.ndarray
    pair_overlap_pixels: int
    pair_bbox_iou_bits: int
    pair_center_step_bits: int
    pair_growth_ratio_bits: int
    recovery_seed_mask_digest: str


@dataclass(frozen=True)
class _BoundarySeedFineAppearanceModel:
    """Immutable fine colour discriminator learned from a clipped seed.

    The ordinary adaptive HSV model intentionally tolerates illumination and
    therefore cannot always distinguish a red target from skin.  This narrow
    Lab a/b likelihood table is created only for an initial mask touching the
    effective image boundary.  It never grants full/trusted authority; it can
    only help the existing two-frame LOST transaction select a visible target
    component from a hand+target SAM union.
    """

    target_generation: int
    seed_frame_id: int
    seed_timestamp_bits: int
    seed_mask: np.ndarray
    seed_mask_digest: str
    seed_pixels: int
    seed_bbox_xyxy: np.ndarray
    seed_center_xy: np.ndarray
    likelihood_table: np.ndarray
    likelihood_table_digest: str
    bins: int
    probability_threshold_bits: int
    model_fingerprint: str


@dataclass(frozen=True)
class _BoundarySeedFineVisibleCoreFrameProof:
    """Exact-frame binding for one boundary-seed fine visible core."""

    frame_id: int
    timestamp_bits: int
    target_generation: int
    model: _BoundarySeedFineAppearanceModel
    raw_evidence: _FrameMaskEvidence
    candidate_evidence: _FrameMaskEvidence
    aligned_anchor_mask: np.ndarray
    aligned_anchor_digest: str
    anchor_shift_xy: np.ndarray
    overlap_pixels: int
    candidate_fraction_bits: int
    anchor_fraction_bits: int
    strict_evidence: _FrameMaskEvidence
    relaxed_expansion: bool
    relaxed_probability_threshold_bits: int
    relaxed_max_area_ratio_bits: int
    relaxed_min_strict_retention_bits: int
    relaxed_max_bbox_extent_ratio_bits: int
    relaxed_max_center_step_bits: int
    relaxed_area_ratio_bits: int
    relaxed_strict_retention_bits: int
    relaxed_bbox_width_ratio_bits: int
    relaxed_bbox_height_ratio_bits: int
    relaxed_center_step_bits: int


@dataclass(frozen=True)
class _ExternalOccluderInvasionProof:
    """One exact-frame, owner-bound external-occluder explanation.

    The proof deliberately stores the aligned *predecessor* bytes which were
    used before final-clean history can advance.  It grants no publication
    authority by itself: consumers must still bind the current raw/core frame
    tokens, the same partial-owner object, and every structured metric below.
    """

    frame_id: int
    timestamp_bits: int
    target_generation: int
    raw_mask_digest: str
    core_mask_digest: str
    aligned_predecessor_mask: np.ndarray
    aligned_predecessor_digest: str
    previous_owner: "_VisibleExactPartialContinuity"
    previous_owner_frame_id: int
    previous_owner_mask_digest: str
    previous_owner_fingerprint: str
    core_pixels: int
    raw_pixels: int
    aligned_pixels: int
    loss_pixels: int
    raw_loss_pixels: int
    foreign_loss_pixels: int
    bridged_loss_pixels: int
    bridge_components: int
    anchor_fraction_bits: int
    raw_loss_fraction_bits: int
    foreign_loss_fraction_bits: int
    bridge_fraction_bits: int
    appearance_mean_bits: int
    appearance_support_bits: int


@dataclass(frozen=True)
class _SanitizerExternalOccluderStageSeal:
    """Monotonic binding for one prepare-time external-occluder proof.

    The mutable pending transaction may lose or replace its public proof
    field before final history is written.  This separate frozen record keeps
    the original proof identity, exact mask bytes and the prepare-time
    ``required`` decision so such a mutation can only fail closed.
    """

    pending: "_PendingFullTargetGeometryCommit"
    frame_id: int
    timestamp_bits: int
    target_generation: int
    raw_evidence: Optional[_FrameMaskEvidence]
    core_evidence: Optional[_FrameMaskEvidence]
    raw_mask_bytes_digest: str
    core_mask_bytes_digest: str
    pending_mask_digest: str
    pending_source: str
    pending_eligibility_kind: str
    pending_raw_exact_eligible: bool
    pending_bootstrap_eligible: bool
    required: bool
    staged: bool
    proof: Optional[_ExternalOccluderInvasionProof]
    proof_fingerprint: str


@dataclass(frozen=True)
class _VisibleExactPartialGateReport:
    """Frozen machine-readable result of the visible-partial gate.

    Human-readable rejection strings are diagnostics only.  A repair path may
    consume this report only when the exact candidate token, partial owner and
    threshold configuration still match.  Metrics and limits are stored as
    IEEE-754 payloads so a boundary value cannot change while retaining an
    equal-looking formatted string.
    """

    frame_id: int
    timestamp_bits: int
    target_generation: int
    candidate_evidence: Optional[_FrameMaskEvidence]
    candidate_mask_digest: str
    previous_owner: Optional["_VisibleExactPartialContinuity"]
    previous_owner_frame_id: int
    previous_owner_fingerprint: str
    config_fingerprint: str
    previous_geometry_kind: str
    allow_boundary_support: bool
    accepted: bool
    failure_kind: str
    violations: Tuple[str, ...]
    metric_bits: Tuple[Tuple[str, int], ...]
    limit_bits: Tuple[Tuple[str, int], ...]


@dataclass(frozen=True)
class _SourceDecomposedVisibleCoreProof:
    """One exact raw rejection decomposed into target core plus removed pixels.

    The raw mask remains audit evidence only.  ``core_evidence`` is the sole
    robot-facing and partial-motion token; this proof grants no full, trusted,
    bootstrap, tracker or appearance authority.
    """

    frame_id: int
    timestamp_bits: int
    target_generation: int
    raw_evidence: _FrameMaskEvidence
    core_evidence: _FrameMaskEvidence
    raw_report: _VisibleExactPartialGateReport
    core_report: _VisibleExactPartialGateReport
    raw_report_fingerprint: str
    core_report_fingerprint: str
    previous_owner: "_VisibleExactPartialContinuity"
    previous_owner_fingerprint: str
    config_fingerprint: str
    removed_mask_digest: str
    removed_pixels: int
    aligned_predecessor_mask: np.ndarray
    aligned_predecessor_digest: str
    fixed_anchor_mask: np.ndarray
    fixed_anchor_digest: str
    fixed_anchor_frame_id: int
    stable_core_raw_motion_decoupled: bool
    external_occluder_required: bool
    external_occluder_proof: Optional[_ExternalOccluderInvasionProof]
    external_occluder_proof_fingerprint: str


@dataclass(frozen=True)
class _SourceDecomposedVisibleCoreStageSeal:
    """Provider-side monotonic latch for a source-decomposition transaction."""

    pending: "_PendingFullTargetGeometryCommit"
    proof: _SourceDecomposedVisibleCoreProof
    proof_fingerprint: str
    frame: RGBDFrame
    frame_content_digest: str
    frame_id: int
    timestamp_bits: int
    target_generation: int
    pending_source: str
    pending_eligibility_kind: str
    pending_raw_exact_eligible: bool
    pending_bootstrap_eligible: bool


@dataclass
class _TrackingContractionVisibleCoreSeed:
    """Fail-closed first exact frame of a TRACKING -> LOST contraction.

    This object owns no publication authority.  It freezes the exact raw and
    appearance-sanitized bytes from one current RGB-D tuple together with the
    already-confirmed partial owner and its non-ratcheting broad anchor.  A
    later frame may use it only as local, read-only evidence.
    """

    frame_id: int
    timestamp_bits: int
    target_generation: int
    online_step_count: int
    raw_mask: np.ndarray
    raw_mask_digest: str
    raw_bbox_xyxy: np.ndarray
    raw_center_xy: np.ndarray
    raw_depth_median: float
    core_mask: np.ndarray
    core_mask_digest: str
    core_bbox_xyxy: np.ndarray
    core_center_xy: np.ndarray
    core_pixels: int
    aligned_predecessor_mask: np.ndarray
    aligned_predecessor_digest: str
    aligned_predecessor_pixels: int
    fixed_anchor_mask: np.ndarray
    fixed_anchor_digest: str
    fixed_anchor_frame_id: int
    fixed_anchor_pixels: int
    provisional_owner: _VisibleExactPartialContinuity
    previous_owner: _VisibleExactPartialContinuity
    previous_owner_digest: str
    previous_owner_frame_id: int
    final_clean_frame_id: int
    final_clean_digest: str
    trusted_frame_id: int
    trusted_digest: str
    full_geometry_frame_id: int
    full_geometry_area_bits: int
    authority_fingerprint: str
    initial_external_occluder_proven: bool
    initial_external_occluder_proof: _ExternalOccluderInvasionProof
    initial_external_proof_fingerprint: str
    trusted_continuity_reason: str
    seed_fingerprint: str


@dataclass
class _TrackingContractionVisibleCoreProof:
    """Frame-bound second exact-frame contraction proof."""

    frame_id: int
    timestamp_bits: int
    target_generation: int
    seed: _TrackingContractionVisibleCoreSeed
    raw_mask_digest: str
    core_mask_digest: str
    core_pixels: int
    aligned_seed_mask: np.ndarray
    aligned_seed_digest: str
    aligned_seed_pixels: int
    aligned_predecessor_mask: np.ndarray
    aligned_predecessor_digest: str
    aligned_predecessor_pixels: int
    pair_overlap_pixels: int
    pair_retention_bits: int
    pair_area_ratio_bits: int
    relocated_equal_area_pair: bool
    relocated_area_floor_bits: int
    pair_bbox_iou_bits: int
    pair_center_step_bits: int
    core_raw_fraction_bits: int
    fixed_anchor_overlap_pixels: int
    fixed_anchor_candidate_fraction_bits: int
    fixed_anchor_reference_fraction_bits: int
    depth_step_bits: int
    depth_limit_bits: int
    appearance_mean_bits: int
    appearance_support_bits: int
    external_occluder_proven: bool
    initial_external_proof_fingerprint: str
    recovery_seed_mask_digest: str


@dataclass
class _PendingFullTargetGeometryCommit:
    """Frame-scoped raw exact-SAM evidence awaiting final acceptance.

    The mask copy and digest bind both trusted-SAM and full-target geometry
    authority to the exact bytes returned by the current-frame video RPC.
    Final sanitization, clipping, probation bypass, or any other known-partial
    decision revokes ``raw_exact_eligible`` without discarding final-clean
    visibility history.
    """

    frame_id: int
    timestamp: float
    target_generation: int
    mask: np.ndarray
    mask_digest: str
    source: str
    raw_exact_eligible: bool
    bootstrap_eligible: bool
    eligibility_reason: str
    eligibility_kind: str
    area: float
    bbox_area: float
    confirmed_log_area_rate_s: Optional[float] = None
    confirmed_scale_samples: int = 0
    # A narrow, non-authoritative proof produced only by the later-RGB-D
    # recovery path.  It binds a sanitized visible core to the recovery-local
    # reference that admitted it.  The final history boundary may reuse this
    # proof to avoid measuring the core against an older full silhouette, but
    # it still has to revalidate the current raw/sanitized frame tokens and
    # the normal final-guard envelope.
    recovery_local_visible_core_digest: Optional[str] = None
    recovery_local_reference_digest: Optional[str] = None
    recovery_local_reference_frame_id: Optional[int] = None
    recovery_local_reference_timestamp_bits: Optional[int] = None
    recovery_local_reference_target_generation: Optional[int] = None
    recovery_local_core_pixels: int = 0
    recovery_local_reference_pixels: int = 0
    recovery_local_aligned_reference_mask: Optional[np.ndarray] = None
    recovery_local_aligned_reference_digest: Optional[str] = None
    recovery_local_aligned_reference_bbox_xyxy: Optional[np.ndarray] = None
    recovery_local_aligned_reference_center_xy: Optional[np.ndarray] = None
    broad_visible_core_proof: Optional[_BroadVisibleCoreProof] = None
    deep_occlusion_broad_stage_seal: Optional[
        _DeepOcclusionBroadStageSeal
    ] = None
    lost_reappearance_visible_core_proof: Optional[
        _LostReappearanceVisibleCoreProof
    ] = None
    tracking_contraction_visible_core_proof: Optional[
        _TrackingContractionVisibleCoreProof
    ] = None
    sanitizer_external_occluder_proof: Optional[
        _ExternalOccluderInvasionProof
    ] = None
    sanitizer_external_occluder_proof_required: bool = False
    sanitizer_external_occluder_stage_seal: Optional[
        _SanitizerExternalOccluderStageSeal
    ] = None
    source_decomposed_visible_core_proof: Optional[
        _SourceDecomposedVisibleCoreProof
    ] = None
    source_decomposed_visible_core_stage_seal: Optional[
        _SourceDecomposedVisibleCoreStageSeal
    ] = None


@dataclass
class _GuardedV2BootstrapHypothesis:
    """One finally-published raw exact mask awaiting stable confirmation."""

    frame_id: int
    timestamp: float
    online_step_count: int
    target_generation: int
    mask: np.ndarray
    mask_digest: str
    bbox_xyxy: np.ndarray
    center_xy: np.ndarray
    area: float
    bbox_area: float
    depth_median: float


@dataclass(frozen=True)
class _GuardedV2BootstrapTransientGap:
    """One fail-closed inference hole inside provisional commissioning.

    This token carries no mask authority.  It only proves that exactly one
    provider step between two bootstrap candidates was classified as local
    inference unavailability rather than visual contradiction.  The normal
    seed/candidate time, frame, identity, geometry, and RGB-D gates still own
    the resumed publication.
    """

    frame_id: int
    timestamp: float
    online_step_count: int
    target_generation: int


@dataclass
class _FullTargetScaleTransitionHypothesis:
    """One complete exact-SAM shrink awaiting a coherent second frame.

    A single small mask can be an occlusion.  This state is therefore only
    seeded by a finally-published raw exact mask whose appearance and topology
    still look complete, and it never owns recovery geometry by itself.
    """

    frame_id: int
    timestamp: float
    mask: np.ndarray
    bbox_xyxy: np.ndarray
    center_xy: np.ndarray
    area: float
    bbox_area: float
    depth_median: float


@dataclass
class _VisibleExactPartialContinuity:
    """Finally-published partial motion evidence without geometry authority.

    The normal member is a byte-identical raw exact-SAM publication.  A
    sanitizer may also contribute its finally-published immutable-appearance
    core, but only as a strict majority subset of a temporally accepted raw
    exact mask.  Such a core can carry policy motion to the boundary; it can
    never teach full/trusted geometry.  ``core_*`` always describes the
    published mask; the unprefixed geometry is the containing raw motion
    envelope and is admission-only.
    """

    frame_id: int
    timestamp: float
    mask: np.ndarray
    mask_digest: str
    core_bbox_xyxy: np.ndarray
    core_center_xy: np.ndarray
    core_area: float
    bbox_xyxy: np.ndarray
    center_xy: np.ndarray
    area: float
    bbox_area: float
    depth_median: float
    center_velocity_px_s: np.ndarray
    velocity_samples: int
    hits: int
    evidence_kind: str
    sanitized_handoff_eligible: bool
    sanitized_anchor_bbox_xyxy: Optional[np.ndarray]
    sanitized_anchor_center_xy: Optional[np.ndarray]
    sanitized_anchor_area: Optional[float]
    sanitized_anchor_mask: Optional[np.ndarray]
    sanitized_anchor_digest: Optional[str]
    sanitized_anchor_frame_id: Optional[int]
    sanitized_anchor_phase: str
    mask_evidence: Optional[_FrameMaskEvidence] = None
    # Separate from the legacy sanitizer anchor above.  This fixed mask is
    # created only by the broad-visible-core proof from the last exact raw
    # target before deep occlusion, then propagated byte-for-byte.  It is never
    # shrunk by later visible fragments and never grants full/trusted authority.
    broad_visible_anchor_mask: Optional[np.ndarray] = None
    broad_visible_anchor_digest: Optional[str] = None
    broad_visible_anchor_bbox_xyxy: Optional[np.ndarray] = None
    broad_visible_anchor_center_xy: Optional[np.ndarray] = None
    broad_visible_anchor_area: Optional[float] = None
    broad_visible_anchor_frame_id: Optional[int] = None
    # Set only after the two-frame deep-occlusion broad transaction crosses
    # the final RGB-D/history boundary.  It permits subsequent *fresh* exact
    # frames to use the same fixed ruler; it never grants raw/full/trusted or
    # tracker authority and is cleared with the partial owner on any gap.
    deep_occlusion_broad_confirmed: bool = False


@dataclass
class _BoundaryExitContinuity:
    """Finally-published evidence for one continuing image exit.

    The mask is either raw exact SAM or a proven immutable-appearance subset;
    geometry remains the containing raw motion envelope in both cases.
    """

    frame_id: int
    timestamp: float
    mask: np.ndarray
    mask_digest: str
    bbox_xyxy: np.ndarray
    center_xy: np.ndarray
    area: float
    bbox_area: float
    depth_median: float
    edges: Tuple[str, ...]
    hits: int
    evidence_kind: str = "raw_exact"
    mask_evidence: Optional[_FrameMaskEvidence] = None


@dataclass
class _RecoveryPublicationHypothesis:
    """Exact-frame evidence held before a recovered track may be published."""

    frame_id: int
    mask: np.ndarray
    bbox_xyxy: np.ndarray
    center_xy: np.ndarray
    area: float
    depth_median: float
    center_reference: np.ndarray
    timestamp: float
    center_velocity_px_s: np.ndarray
    hits: int
    target_generation: int
    timestamp_bits: int
    mask_digest: str
    evidence_kind: str
    # Provenance of the exact mask bytes carried by this protocol state.
    # This is distinct from ``evidence_kind`` above, which describes the
    # 2-SAM/RGB-D phase.  A sanitized core is always non-authoritative.
    mask_evidence_kind: str


@dataclass(frozen=True)
class _PendingRecoveryPublicationCommit:
    """Exact guarded-primary 3/3 proof awaiting the final history boundary."""

    frame_id: int
    timestamp_bits: int
    target_generation: int
    mask_digest: str
    hits: int
    evidence_kind: str


@dataclass(frozen=True)
class BBoxInitializationEvidence:
    """Structured evidence retained from the mask before tracker mutation."""

    source: str
    valid: bool
    mask_area_px: int
    mask_bbox_xyxy: np.ndarray
    mask_bbox_area_px: int
    prompt_bbox_xyxy: np.ndarray
    prompt_bbox_area_px: int
    solid_bbox_mask: bool
    full_prompt_box_mask: bool


@dataclass(frozen=True)
class InitialSAMHaloCleanupEvidence:
    """Decision/provenance for the one-shot provisional SAM seed cleanup.

    The cleanup is deliberately not a reusable publication sanitizer.  It is
    evaluated only while binding a newly selected SAM mask, before either the
    local tracker or temporal SAM2 history becomes robot-facing state.
    """

    source: str
    attempted: bool
    applied: bool
    raw_area_px: int
    clean_area_px: int
    deleted_area_px: int
    raw_mask_digest: str
    clean_mask_digest: str
    status: str


@dataclass(frozen=True)
class OnlineSAM2BBoxPrewarmEvidence:
    """Non-publication CUDA warmup and exact-seed provenance.

    The discarded track exists only to exercise the first temporal propagate
    kernels before the 20 Hz rollout starts.  The following exact reseed
    restores the accepted initialization frame/mask as the sole temporal
    authority, so the duplicate frame can never become a policy observation.
    """

    enabled: bool
    attempted: bool
    passed: bool
    seed_frame_id: Optional[int]
    discarded_track_frame_id: Optional[int]
    discarded_track_valid: bool
    discarded_track_rpc_ms: float
    discarded_track_frame_ids: Tuple[int, ...]
    discarded_track_rpc_ms_values: Tuple[float, ...]
    attempt_count: int
    required_stable_tracks: int
    achieved_stable_tracks: int
    max_attempts: int
    max_rpc_ms: float
    exact_reseed_rpc_ms: float
    total_ms: float
    status: str


@dataclass
class _MaskPipelineResult:
    """One exact RGB-D frame after all tracker and SAM mask gates."""

    frame: RGBDFrame
    mask_result: MaskResult
    camera_ms: float
    tracker_ms: float
    online_sam2_ms: float
    sam2_reinit_ms: float


@dataclass
class _PendingPublicationCommit:
    """Tracker side effects deferred until the final RGB-D boundary passes."""

    frame_id: int
    mask: np.ndarray
    source: str

