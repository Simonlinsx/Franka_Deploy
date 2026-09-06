from __future__ import annotations

import numpy as np

from .snapshot import GraspCandidates, VisualizationSnapshot
from .types import GraspResult, PointCloudObservation


def _strict_candidate_bool(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


def _shared_candidate_metadata(
    candidates: tuple, key: str, default: object
) -> object:
    """Return one provenance value and reject candidate-to-candidate drift."""

    if not candidates:
        return default
    values = [candidate.metadata.get(key, default) for candidate in candidates]
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"candidate metadata {key!r} is inconsistent")
    return first


def make_snapshot(
    observation: PointCloudObservation,
    result: GraspResult,
    *,
    scene_excludes_object: bool = True,
) -> VisualizationSnapshot:
    """Combine a synchronized point-cloud observation and inference result."""

    if result.reference_frame != observation.reference_frame:
        raise ValueError(
            "result/reference frame mismatch: "
            f"{result.reference_frame!r} != {observation.reference_frame!r}"
        )
    candidates = tuple(result.candidates)
    count = len(candidates)
    canonical = np.empty((count, 4, 4), dtype=np.float64)
    scores = np.empty(count, dtype=np.float32)
    type_ids = np.empty(count, dtype=np.int32)
    collision_free = np.empty(count, dtype=np.bool_)
    collision_checked = np.empty(count, dtype=np.bool_)
    widths = np.empty(count, dtype=np.float32)
    depths = np.empty(count, dtype=np.float32)
    source_indices = np.empty(count, dtype=np.int64)

    hand_presence = [candidate.T_reference_hand is not None for candidate in candidates]
    angle_presence = [candidate.hand_angles is not None for candidate in candidates]
    if any(hand_presence) and not all(hand_presence):
        raise ValueError("either all or no candidates must provide T_reference_hand")
    if any(angle_presence) and not all(angle_presence):
        raise ValueError("either all or no candidates must provide hand_angles")
    hand_poses = np.empty((count, 4, 4), dtype=np.float64) if all(hand_presence) and count else None
    hand_angles = np.empty((count, 6), dtype=np.float32) if all(angle_presence) and count else None

    for index, candidate in enumerate(candidates):
        canonical[index] = candidate.T_reference_grasp
        scores[index] = candidate.score
        type_ids[index] = candidate.grasp_type_id
        reported_free = _strict_candidate_bool(
            candidate.collision_free, f"candidates[{index}].collision_free"
        )
        checked = _strict_candidate_bool(
            candidate.metadata.get("collision_checked", False),
            f"candidates[{index}].metadata.collision_checked",
        )
        # A positive collision result without evidence that the check ran is
        # demoted, never serialized as collision-free.
        collision_checked[index] = checked
        collision_free[index] = reported_free and checked
        widths[index] = candidate.width_m
        depths[index] = candidate.depth_m
        source_indices[index] = candidate.source_index
        if hand_poses is not None:
            hand_poses[index] = candidate.T_reference_hand
        if hand_angles is not None:
            hand_angles[index] = candidate.hand_angles

    widths_payload = widths if np.isfinite(widths).all() else None
    depths_payload = depths if np.isfinite(depths).all() else None
    grasps = GraspCandidates(
        canonical_poses=canonical,
        scores=scores,
        type_ids=type_ids,
        collision_free=collision_free,
        collision_checked=collision_checked,
        selected_index=result.selected_index if count else -1,
        approach_axis_local=np.asarray((1.0, 0.0, 0.0), dtype=np.float32),
        hand_poses=hand_poses,
        hand_angles=hand_angles,
        widths_m=widths_payload,
        depths_m=depths_payload,
        source_indices=source_indices,
    )
    representation_hash = str(
        _shared_candidate_metadata(
            candidates,
            "representation_checkpoint_sha256",
            result.checkpoint_sha256,
        )
    )
    decision_hashes_value = _shared_candidate_metadata(
        candidates, "decision_checkpoint_sha256s", ()
    )
    if not isinstance(decision_hashes_value, (tuple, list)):
        raise ValueError(
            "candidate metadata decision_checkpoint_sha256s must be a sequence"
        )
    decision_hashes = tuple(str(value) for value in decision_hashes_value)
    official_source_commit = str(
        _shared_candidate_metadata(candidates, "official_source_commit", "")
    )

    return VisualizationSnapshot(
        scene_points=observation.scene_points,
        scene_colors=observation.scene_colors,
        object_points=observation.object_points,
        object_colors=observation.object_colors,
        grasps=grasps,
        reference_frame=observation.reference_frame,
        T_reference_camera=observation.T_reference_camera,
        frame_id=max(0, int(observation.frame_id)),
        timestamp_s=max(0.0, float(observation.timestamp_s)),
        calibration_id=observation.calibration_id,
        camera_serial=observation.camera_serial,
        scene_excludes_object=scene_excludes_object,
        inference_points=result.inference_points,
        model_name=result.model_name or result.backend_name,
        checkpoint_sha256=result.checkpoint_sha256,
        representation_checkpoint_sha256=representation_hash,
        decision_checkpoint_sha256s=decision_hashes,
        official_source_commit=official_source_commit,
    )
