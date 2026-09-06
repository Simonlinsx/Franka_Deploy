"""Explicit mechanical frame composition for the FR3/RH56 adapter stack.

The AnyDex ``hand`` pose is its generated source-mesh frame, not a Franka EE
or flange frame.  Local upstream code establishes only the source-axis
permutation.  The installed yaw and source-origin datum are installation-
specific inputs: for V7 they are derived from the keyed adapter/FR3 CAD and
the official RH56 source datum.  This module still requires both explicitly
and never substitutes the unrelated 44 mm TCP configured on the upstream UR
robot.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .control_plan import inverse_rigid_transform, validate_rigid_transform


# T_mount_source: source +X -> mounting +Z, source +Y -> mounting +X,
# source +Z -> mounting +Y.  This is the exact pure rotation returned by the
# official AnyDex UR wrapper for InspireHandR.
T_MOUNT_SOURCE_AXIS_BASIS = np.asarray(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
T_MOUNT_SOURCE_AXIS_BASIS.setflags(write=False)


def rotation_z(angle_rad: float) -> np.ndarray:
    angle = float(angle_rad)
    if not np.isfinite(angle):
        raise ValueError("assembled_yaw_rad must be finite")
    cosine, sine = np.cos(angle), np.sin(angle)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return result


def compose_T_flange_hand_source(
    *,
    fr3_face_to_rh56_seating_plane_m: float,
    assembled_yaw_rad: float,
    seating_to_source_origin_m: Sequence[float],
) -> np.ndarray:
    """Compose measured ``T_F_hand`` for the snapshot/AnyDex source frame.

    ``seating_to_source_origin_m`` is expressed in the mounting-plane frame
    before the source-axis permutation.  It must be obtained from a verified
    mating datum or measurement and explicitly commissioned; the Link111
    visualization mesh by itself is not mating-surface CAD.
    """

    offset = float(fr3_face_to_rh56_seating_plane_m)
    if not np.isfinite(offset) or offset <= 0.0:
        raise ValueError("FR3-face to RH56-seating offset must be finite and positive")
    translation = np.asarray(seating_to_source_origin_m, dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("seating_to_source_origin_m must contain three finite values")

    T_F_seating = np.eye(4, dtype=np.float64)
    T_F_seating[2, 3] = offset
    T_seating_source = rotation_z(assembled_yaw_rad) @ T_MOUNT_SOURCE_AXIS_BASIS
    # The measured origin vector uses seating-frame axes and therefore is not
    # post-multiplied by the source rotation.
    T_seating_source = np.array(T_seating_source, copy=True)
    T_seating_source[:3, 3] = translation
    return validate_rigid_transform(
        T_F_seating @ T_seating_source, "T_F_hand_source"
    )


def compose_T_EE_hand_source(
    F_T_EE: np.ndarray, T_F_hand_source: np.ndarray
) -> np.ndarray:
    """Convert the mechanical flange transform to the configured Franka EE."""

    flange_to_ee = validate_rigid_transform(F_T_EE, "F_T_EE")
    flange_to_hand = validate_rigid_transform(
        T_F_hand_source, "T_F_hand_source"
    )
    return validate_rigid_transform(
        inverse_rigid_transform(flange_to_ee) @ flange_to_hand,
        "T_EE_hand_source",
    )


def compose_T_reference_EE(
    T_reference_hand_source: np.ndarray, T_EE_hand_source: np.ndarray
) -> np.ndarray:
    """Return the Franka EE target for one AnyDex source-frame hand target."""

    reference_to_hand = validate_rigid_transform(
        T_reference_hand_source, "T_reference_hand_source"
    )
    ee_to_hand = validate_rigid_transform(
        T_EE_hand_source, "T_EE_hand_source"
    )
    return validate_rigid_transform(
        reference_to_hand @ inverse_rigid_transform(ee_to_hand),
        "T_reference_EE",
    )


__all__ = [
    "T_MOUNT_SOURCE_AXIS_BASIS",
    "compose_T_EE_hand_source",
    "compose_T_flange_hand_source",
    "compose_T_reference_EE",
    "rotation_z",
]
