"""Reusable motion-planning primitives for dynamic manipulation.

This package owns geometry, prediction, interception, and trajectory planning.
It must not open robot or camera devices.  Transactional hardware execution
remains in :mod:`sim2real` until the corresponding runtime adapters are moved
behind the :mod:`robot_control` interfaces.
"""

from .kinematics import (
    cartesian_joint_correction,
    panda_T_base_policy_palm,
    pose_preserving_joint_target,
)

__all__ = [
    "cartesian_joint_correction",
    "panda_T_base_policy_palm",
    "pose_preserving_joint_target",
]
