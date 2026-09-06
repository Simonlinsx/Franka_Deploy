"""Internal immutable-value validation helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np

from sim2real.contracts.actions import V94MappedAction


def _finite_scalar(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


def _readonly_float_vector(value: object, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    result = result.copy()
    result.setflags(write=False)
    return result


def _readonly_register_vector(value: object, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (6,) or not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must contain six finite integer values")
    numeric = raw.astype(np.float64)
    if not np.all(numeric == np.rint(numeric)):
        raise ValueError(f"{name} must contain integer values")
    result = numeric.astype(np.int32)
    if np.any(result < 0) or np.any(result > 1000):
        raise ValueError(f"{name} must lie in [0,1000]")
    result.setflags(write=False)
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric <= 0.0:
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _optional_positive_rate_vector(
    value: object, size: int, name: str
) -> Optional[np.ndarray]:
    if value is None:
        return None
    raw = np.asarray(value, dtype=np.float64)
    if raw.shape == ():
        raw = np.full(size, float(raw), dtype=np.float64)
    if raw.shape != (size,) or not np.all(np.isfinite(raw)) or np.any(raw <= 0.0):
        raise ValueError(f"{name} must be a positive scalar or {size}-vector")
    result = raw.copy()
    result.setflags(write=False)
    return result


def _immutable_mapped_action(value: V94MappedAction) -> V94MappedAction:
    if not isinstance(value, V94MappedAction):
        raise TypeError("mapped action must be V94MappedAction")
    return V94MappedAction(
        raw_policy_action13=_readonly_float_vector(
            value.raw_policy_action13, 13, "raw_policy_action13"
        ),
        executed_policy_action13=_readonly_float_vector(
            value.executed_policy_action13, 13, "executed_policy_action13"
        ),
        franka_target_q_rad=_readonly_float_vector(
            value.franka_target_q_rad, 7, "franka_target_q_rad"
        ),
        rh56_target_q_policy_order_rad=_readonly_float_vector(
            value.rh56_target_q_policy_order_rad,
            6,
            "rh56_target_q_policy_order_rad",
        ),
        rh56_angle_set_register_order=_readonly_register_vector(
            value.rh56_angle_set_register_order,
            "rh56_angle_set_register_order",
        ),
        clipped=bool(value.clipped),
        reasons=tuple(str(reason) for reason in value.reasons),
        hold_arm_target=bool(value.hold_arm_target),
    )
