"""Pure Panda kinematics used by tabletop interception planning.

The functions in this module are deterministic and hardware-inert.  They are
kept separate from the online planner state machine so geometry can be tested
without constructing a deployment runtime.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


_PANDA_ORIGINS = (
    ((0.0, 0.0, 0.333), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (-np.pi / 2.0, 0.0, 0.0)),
    ((0.0, -0.316, 0.0), (np.pi / 2.0, 0.0, 0.0)),
    ((0.0825, 0.0, 0.0), (np.pi / 2.0, 0.0, 0.0)),
    ((-0.0825, 0.384, 0.0), (-np.pi / 2.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (np.pi / 2.0, 0.0, 0.0)),
    ((0.088, 0.0, 0.0), (np.pi / 2.0, 0.0, 0.0)),
)


def _rotation_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rx = np.asarray(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
    ry = np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    rz = np.asarray(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
    return rz @ ry @ rx


def panda_T_base_policy_palm(
    q_rad: np.ndarray,
    T_flange_policy_palm: np.ndarray,
) -> np.ndarray:
    """Return deterministic Panda forward kinematics for the policy palm."""

    q = np.asarray(q_rad, dtype=np.float64)
    flange_palm = np.asarray(T_flange_policy_palm, dtype=np.float64)
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        raise ValueError("Franka FK requires seven finite joints")
    if flange_palm.shape != (4, 4) or not np.all(np.isfinite(flange_palm)):
        raise ValueError("T_flange_policy_palm must be finite [4,4]")
    transform = np.eye(4, dtype=np.float64)
    for joint, (xyz, rpy) in zip(q, _PANDA_ORIGINS):
        origin = np.eye(4, dtype=np.float64)
        origin[:3, :3] = _rotation_rpy(*rpy)
        origin[:3, 3] = xyz
        rotation = np.eye(4, dtype=np.float64)
        cosine, sine = np.cos(joint), np.sin(joint)
        rotation[:3, :3] = (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        )
        transform = transform @ origin @ rotation
    link8 = np.eye(4, dtype=np.float64)
    link8[2, 3] = 0.107
    return transform @ link8 @ flange_palm


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    value = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(value) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew = np.asarray(
        (
            value[2, 1] - value[1, 2],
            value[0, 2] - value[2, 0],
            value[1, 0] - value[0, 1],
        ),
        dtype=np.float64,
    )
    if angle < 1.0e-8:
        return 0.5 * skew
    sine = float(np.sin(angle))
    if abs(sine) < 1.0e-8:
        raise ValueError("online planner orientation difference is singular")
    return (angle / (2.0 * sine)) * skew


def _numeric_spatial_jacobian(
    q_rad: np.ndarray,
    T_flange_policy_palm: np.ndarray,
) -> np.ndarray:
    q = np.asarray(q_rad, dtype=np.float64)
    base = panda_T_base_policy_palm(q, T_flange_policy_palm)
    epsilon = 1.0e-5
    jacobian = np.zeros((6, 7), dtype=np.float64)
    for joint in range(7):
        shifted = q.copy()
        shifted[joint] += epsilon
        moved = panda_T_base_policy_palm(shifted, T_flange_policy_palm)
        jacobian[:3, joint] = (moved[:3, 3] - base[:3, 3]) / epsilon
        relative = moved[:3, :3] @ base[:3, :3].T
        jacobian[3:, joint] = _rotation_vector(relative) / epsilon
    return jacobian


def cartesian_joint_correction(
    q_rad: np.ndarray,
    translation_base_m: np.ndarray,
    *,
    T_flange_policy_palm: np.ndarray,
    damping: float,
) -> np.ndarray:
    """Return the minimum-norm 6D pose-preserving correction for translation."""

    translation = np.asarray(translation_base_m, dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("translation_base_m must contain three finite values")
    damping_value = float(damping)
    if not np.isfinite(damping_value) or damping_value <= 0.0:
        raise ValueError("damping must be finite and positive")
    jacobian = _numeric_spatial_jacobian(q_rad, T_flange_policy_palm)
    target = np.concatenate([translation, np.zeros(3)])
    regularized = jacobian @ jacobian.T + (damping_value**2) * np.eye(6)
    return jacobian.T @ np.linalg.solve(regularized, target)


def pose_preserving_joint_target(
    q_seed_rad: np.ndarray,
    translation_base_m: np.ndarray,
    *,
    T_flange_policy_palm: np.ndarray,
    joint_limits_rad: np.ndarray,
    joint_limit_margin_rad: float,
    damping: float,
    target_rotation_base: Optional[np.ndarray] = None,
    local_precision_refinement: bool = False,
) -> np.ndarray:
    """Solve a bounded translated-palm target while preserving orientation."""

    seed = np.asarray(q_seed_rad, dtype=np.float64)
    translation = np.asarray(translation_base_m, dtype=np.float64)
    limits = np.asarray(joint_limits_rad, dtype=np.float64)
    if seed.shape != (7,) or not np.all(np.isfinite(seed)):
        raise ValueError("pose target seed must contain seven finite joints")
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("pose target translation must contain three finite values")
    if limits.shape != (7, 2) or not np.all(np.isfinite(limits)):
        raise ValueError("pose target joint limits must be finite [7,2]")
    margin = float(joint_limit_margin_rad)
    damping_value = float(damping)
    if not np.isfinite(margin) or margin <= 0.0:
        raise ValueError("pose target joint margin must be positive")
    if not np.isfinite(damping_value) or damping_value <= 0.0:
        raise ValueError("pose target damping must be positive")
    lower = limits[:, 0] + margin
    upper = limits[:, 1] - margin
    if np.any(lower >= upper) or np.any(seed < lower) or np.any(seed > upper):
        raise ValueError("pose target seed violates the joint-limit margin")
    reference = panda_T_base_policy_palm(seed, T_flange_policy_palm)
    target_position = reference[:3, 3] + translation
    if target_rotation_base is None:
        target_rotation = reference[:3, :3]
    else:
        target_rotation = np.asarray(target_rotation_base, dtype=np.float64)
        if target_rotation.shape != (3, 3) or not np.all(
            np.isfinite(target_rotation)
        ):
            raise ValueError("pose target rotation must be finite [3,3]")
    if not isinstance(local_precision_refinement, (bool, np.bool_)):
        raise ValueError("local_precision_refinement must be boolean")
    refine_locally = bool(local_precision_refinement)
    q = seed.copy()
    for _ in range(32 if refine_locally else 24):
        current = panda_T_base_policy_palm(q, T_flange_policy_palm)
        error = np.concatenate(
            [
                target_position - current[:3, 3],
                _rotation_vector(target_rotation @ current[:3, :3].T),
            ]
        )
        if (
            float(np.linalg.norm(error[:3])) <= 5.0e-4
            and float(np.linalg.norm(error[3:])) <= 2.0e-3
        ):
            return q
        jacobian = _numeric_spatial_jacobian(q, T_flange_policy_palm)
        local_damping = (
            min(damping_value, 0.01)
            if refine_locally
            and float(np.linalg.norm(error[:3])) <= 5.0e-3
            and float(np.linalg.norm(error[3:])) <= 1.0e-2
            else damping_value
        )
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + (local_damping**2) * np.eye(6),
            error,
        )
        maximum = float(np.max(np.abs(delta)))
        if maximum > 0.08:
            delta *= 0.08 / maximum
        q = np.clip(q + delta, lower, upper)
    current = panda_T_base_policy_palm(q, T_flange_policy_palm)
    position_error = float(np.linalg.norm(target_position - current[:3, 3]))
    orientation_error = float(
        np.linalg.norm(_rotation_vector(target_rotation @ current[:3, :3].T))
    )
    raise RuntimeError(
        "online intercept IK did not converge: "
        f"position_error={position_error:.6f}m "
        f"orientation_error={orientation_error:.6f}rad"
    )


__all__ = [
    "cartesian_joint_correction",
    "panda_T_base_policy_palm",
    "pose_preserving_joint_target",
]
