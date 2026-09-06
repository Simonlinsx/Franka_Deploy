"""Explicit diagnostic mapping of generic grasps to an Inspire hand type.

The geometric commissioning backend predicts neither semantic grasp type nor
hand-specific decision score.  This helper is therefore opt-in and labels its
result as diagnostic.  It is useful for validating frames, URDF FK, and mesh
placement before the official AnyDex neural environment is available.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping, Any

import numpy as np

from .inspire_mapping import load_inspire_mapping, map_two_finger_grasps
from .snapshot import GraspCandidates, VisualizationSnapshot, validate_snapshot


def attach_diagnostic_inspire_hand(
    snapshot: VisualizationSnapshot,
    *,
    grasp_type_id: int,
    mapping: str | Path | Mapping[str, Any],
) -> VisualizationSnapshot:
    """Attach official kinematics for one explicitly fixed semantic hand type.

    Canonical poses, scores, candidate order, object/scene clouds, and frame
    provenance remain unchanged.  Only type IDs, hand poses, and six actuator
    values are added.  The depth already stored on each canonical candidate is
    applied exactly as in the official mesh visualizer.
    """

    validate_snapshot(snapshot)
    grasps = snapshot.grasps
    count = grasps.count
    if count == 0:
        return snapshot
    if grasps.widths_m is None:
        raise ValueError("diagnostic Inspire mapping requires grasp widths")
    widths = np.asarray(grasps.widths_m, dtype=np.float64)
    depths = (
        np.zeros(count, dtype=np.float64)
        if grasps.depths_m is None
        else np.asarray(grasps.depths_m, dtype=np.float64)
    )
    if not np.all(np.isfinite(widths)) or not np.all(np.isfinite(depths)):
        raise ValueError("diagnostic Inspire mapping requires finite widths and depths")

    two_finger = np.zeros((count, 17), dtype=np.float64)
    two_finger[:, 0] = np.asarray(grasps.scores, dtype=np.float64)
    two_finger[:, 1] = widths
    two_finger[:, 2] = 0.03
    two_finger[:, 3] = depths
    poses = np.asarray(grasps.canonical_poses, dtype=np.float64)
    two_finger[:, 4:13] = poses[:, :3, :3].reshape(count, 9)
    two_finger[:, 13:16] = poses[:, :3, 3]
    two_finger[:, 16] = -1

    data = mapping if isinstance(mapping, Mapping) else load_inspire_mapping(mapping)
    mapped = map_two_finger_grasps(
        two_finger,
        grasp_type_id,
        data,
        frame_id=snapshot.reference_frame,
    )
    updated_grasps = GraspCandidates(
        canonical_poses=grasps.canonical_poses,
        scores=grasps.scores,
        type_ids=mapped.grasp_types.astype(np.int32),
        collision_free=grasps.collision_free,
        selected_index=grasps.selected_index,
        approach_axis_local=grasps.approach_axis_local,
        hand_poses=mapped.pose_matrices(apply_depth=True),
        hand_angles=mapped.angles.astype(np.float32),
        widths_m=mapped.widths.astype(np.float32),
        depths_m=mapped.depths.astype(np.float32),
        source_indices=grasps.source_indices,
    )
    model_name = snapshot.model_name
    suffix = f" + diagnostic fixed Inspire type {int(grasp_type_id)} (not decision-model output)"
    if suffix not in model_name:
        model_name += suffix
    updated = replace(snapshot, grasps=updated_grasps, model_name=model_name)
    validate_snapshot(updated)
    return updated

