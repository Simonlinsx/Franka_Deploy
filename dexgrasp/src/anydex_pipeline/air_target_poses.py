"""One hardware-free source of truth for installed-tool air target poses.

The official AnyDex snapshot stores the nominal hand/contact pose.  The
installed PLA commissioning path deliberately stops short of that pose:

* ``final_air`` retreats from nominal by ``air_retreat_distance_m`` along
  ``-approach``;
* ``pregrasp`` retreats by one additional
  ``air_pregrasp_distance_m`` along the same direction.

Keeping this calculation in a pure module prevents the planner, collision
audit, executor binding, and viewer from silently displaying different
targets.  Nothing here imports a camera, GUI, serial driver, or libfranka.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .control_plan import inverse_rigid_transform, validate_rigid_transform


@dataclass(frozen=True)
class AirTargetPoses:
    """Exact nominal/final-air/pregrasp transforms in one reference frame."""

    approach_reference: np.ndarray
    T_reference_EE_nominal: np.ndarray
    T_reference_EE_pregrasp: np.ndarray
    T_reference_EE_final_air: np.ndarray
    T_reference_hand_final_air: np.ndarray

    def __post_init__(self) -> None:
        approach = np.asarray(self.approach_reference, dtype=np.float64)
        if approach.shape != (3,) or not np.all(np.isfinite(approach)):
            raise ValueError("approach_reference must be a finite 3-vector")
        norm = float(np.linalg.norm(approach))
        if not np.isclose(norm, 1.0, atol=1.0e-10, rtol=0.0):
            raise ValueError("approach_reference must be a unit vector")
        approach = np.array(approach, copy=True)
        approach.setflags(write=False)
        object.__setattr__(self, "approach_reference", approach)
        for name in (
            "T_reference_EE_nominal",
            "T_reference_EE_pregrasp",
            "T_reference_EE_final_air",
            "T_reference_hand_final_air",
        ):
            object.__setattr__(
                self,
                name,
                validate_rigid_transform(getattr(self, name), name),
            )


def derive_air_target_poses(
    canonical_pose_reference: np.ndarray,
    hand_pose_reference: np.ndarray,
    approach_axis_local: np.ndarray,
    T_EE_hand: np.ndarray,
    *,
    retreat_distance_m: float,
    pregrasp_extra_distance_m: float,
) -> AirTargetPoses:
    """Derive the exact installed-air target contract for one candidate.

    ``canonical_pose_reference`` supplies only the approach direction;
    ``hand_pose_reference`` supplies the official nominal wrist pose.  The
    mount transform follows ``T_A_B`` notation and maps hand-frame points into
    the Franka EE frame.
    """

    canonical = validate_rigid_transform(
        canonical_pose_reference, "candidate canonical pose"
    )
    hand = validate_rigid_transform(hand_pose_reference, "candidate hand pose")
    mount = validate_rigid_transform(T_EE_hand, "tool.T_EE_hand")
    axis = np.asarray(approach_axis_local, dtype=np.float64)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError("approach_axis_local must be a finite 3-vector")
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-12:
        raise ValueError("approach_axis_local must be non-zero")

    retreat = float(retreat_distance_m)
    extra = float(pregrasp_extra_distance_m)
    if not np.isfinite(retreat) or retreat <= 0.0:
        raise ValueError("air retreat distance must be finite and positive")
    if not np.isfinite(extra) or extra <= 0.0:
        raise ValueError("air pregrasp extra distance must be finite and positive")

    approach = canonical[:3, :3] @ (axis / norm)
    approach /= np.linalg.norm(approach)
    nominal = validate_rigid_transform(
        hand @ inverse_rigid_transform(mount), "nominal T_reference_EE"
    )
    final_air = np.array(nominal, copy=True)
    final_air[:3, 3] -= retreat * approach
    final_air = validate_rigid_transform(final_air, "T_reference_EE_final_air")
    pregrasp = np.array(final_air, copy=True)
    pregrasp[:3, 3] -= extra * approach
    pregrasp = validate_rigid_transform(pregrasp, "T_reference_EE_pregrasp")
    final_hand = validate_rigid_transform(
        final_air @ mount, "T_reference_hand_final_air"
    )
    return AirTargetPoses(
        approach_reference=approach,
        T_reference_EE_nominal=nominal,
        T_reference_EE_pregrasp=pregrasp,
        T_reference_EE_final_air=final_air,
        T_reference_hand_final_air=final_hand,
    )


__all__ = ["AirTargetPoses", "derive_air_target_poses"]
