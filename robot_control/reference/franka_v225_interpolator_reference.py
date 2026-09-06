"""Offline NumPy references for the accepted Franka controller variants.

The persistent 1 kHz trajectory generator is shared by the legacy V225/V258
mapping and the q_d-relative g015 mapping.  Keeping both policy-rate adapters
named here prevents a controller migration from silently changing old replay
or checkpoint semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

SERVO_DT_S = 0.001
TARGET_LOWPASS_CUTOFF_HZ = 100.0
TRACKING_NATURAL_FREQUENCY_HZ = 6.0
TRACKING_DAMPING_RATIO = 1.0
MAX_COMMAND_VELOCITY_RAD_S = 0.50
MAX_COMMAND_ACCELERATION_RAD_S2 = 5.0
MAX_COMMAND_JERK_RAD_S3 = 250.0

SAFE_JOINT_LOWER_RAD = np.asarray(
    [-2.6937, -1.7337, -2.8507, -2.9921, -2.7565, 0.5945, -2.9659],
    dtype=np.float64,
)
SAFE_JOINT_UPPER_RAD = np.asarray(
    [2.6937, 1.7337, 2.8507, -0.2018, 2.7565, 4.4669, 2.9659],
    dtype=np.float64,
)


def _vector7(value: object, name: str, *, dtype=np.float64) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != (7,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain seven finite values")
    return result.copy()


def map_v225_20hz_arm_action(
    previous_target_q_rad: object,
    measured_q_rad: object,
    action_arm: object,
) -> np.ndarray:
    """Map the first seven normalized policy outputs to a held target."""

    previous = _vector7(previous_target_q_rad, "previous", dtype=np.float32)
    measured = _vector7(measured_q_rad, "measured", dtype=np.float32)
    action = np.clip(
        _vector7(action_arm, "action", dtype=np.float32), -1.0, 1.0
    )
    raw = previous + np.float32(0.045) * action
    filtered = np.float32(0.40) * raw + np.float32(0.60) * previous
    delta = np.clip(filtered - previous, -np.float32(0.045), np.float32(0.045))
    candidate = previous + delta
    lower = np.maximum(
        SAFE_JOINT_LOWER_RAD.astype(np.float32), measured - np.float32(0.05)
    )
    upper = np.minimum(
        SAFE_JOINT_UPPER_RAD.astype(np.float32), measured + np.float32(0.05)
    )
    return np.clip(candidate, lower, upper).astype(np.float32, copy=False)


def map_qd_g015_20hz_arm_action(
    current_shaper_q_d_rad: object,
    action_arm: object,
) -> np.ndarray:
    """Map q_d-g015 actions without a previous-target or measured-q envelope."""

    q_d = _vector7(
        current_shaper_q_d_rad, "current_shaper_q_d_rad", dtype=np.float32
    )
    action = np.clip(
        _vector7(action_arm, "action", dtype=np.float32),
        np.float32(-1.0),
        np.float32(1.0),
    )
    target = q_d + np.float32(0.06) * action
    return np.clip(
        target,
        SAFE_JOINT_LOWER_RAD.astype(np.float32),
        SAFE_JOINT_UPPER_RAD.astype(np.float32),
    ).astype(np.float32, copy=False)


@dataclass(frozen=True)
class DesiredCommandState:
    q_rad: np.ndarray
    dq_rad_s: np.ndarray
    ddq_rad_s2: np.ndarray
    filtered_target_rad: np.ndarray


class V225FrankaCommandInterpolator:
    """100 Hz target filter plus 6 Hz critically damped 1 kHz interpolation."""

    def __init__(
        self,
        start_q_rad: object,
        *,
        initial_dq_rad_s: object | None = None,
        initial_ddq_rad_s2: object | None = None,
    ) -> None:
        self.q_rad = np.clip(
            _vector7(start_q_rad, "start_q_rad"),
            SAFE_JOINT_LOWER_RAD,
            SAFE_JOINT_UPPER_RAD,
        )
        zeros = np.zeros(7, dtype=np.float64)
        self.dq_rad_s = (
            zeros.copy()
            if initial_dq_rad_s is None
            else _vector7(initial_dq_rad_s, "initial_dq_rad_s")
        )
        self.ddq_rad_s2 = (
            zeros.copy()
            if initial_ddq_rad_s2 is None
            else _vector7(initial_ddq_rad_s2, "initial_ddq_rad_s2")
        )
        self.filtered_target_rad = self.q_rad.copy()
        self._lowpass_gain = SERVO_DT_S / (
            SERVO_DT_S + 1.0 / (2.0 * math.pi * TARGET_LOWPASS_CUTOFF_HZ)
        )
        self._omega = 2.0 * math.pi * TRACKING_NATURAL_FREQUENCY_HZ

    @property
    def state(self) -> DesiredCommandState:
        return DesiredCommandState(
            self.q_rad.copy(),
            self.dq_rad_s.copy(),
            self.ddq_rad_s2.copy(),
            self.filtered_target_rad.copy(),
        )

    def step(self, held_target_q_rad: object) -> DesiredCommandState:
        held = np.clip(
            _vector7(held_target_q_rad, "held_target_q_rad"),
            SAFE_JOINT_LOWER_RAD,
            SAFE_JOINT_UPPER_RAD,
        )
        self.filtered_target_rad = (
            self._lowpass_gain * held
            + (1.0 - self._lowpass_gain) * self.filtered_target_rad
        )
        desired_acceleration = (
            self._omega**2 * (self.filtered_target_rad - self.q_rad)
            - 2.0
            * TRACKING_DAMPING_RATIO
            * self._omega
            * self.dq_rad_s
        )
        jerk_limited_acceleration = self.ddq_rad_s2 + np.clip(
            desired_acceleration - self.ddq_rad_s2,
            -MAX_COMMAND_JERK_RAD_S3 * SERVO_DT_S,
            MAX_COMMAND_JERK_RAD_S3 * SERVO_DT_S,
        )
        jerk_limited_acceleration = np.clip(
            jerk_limited_acceleration,
            -MAX_COMMAND_ACCELERATION_RAD_S2,
            MAX_COMMAND_ACCELERATION_RAD_S2,
        )
        acceleration_to_velocity_gain = (
            MAX_COMMAND_JERK_RAD_S3 / MAX_COMMAND_ACCELERATION_RAD_S2
        )
        safe_maximum_acceleration = np.minimum(
            acceleration_to_velocity_gain
            * (MAX_COMMAND_VELOCITY_RAD_S - self.dq_rad_s),
            MAX_COMMAND_ACCELERATION_RAD_S2,
        )
        safe_minimum_acceleration = np.maximum(
            acceleration_to_velocity_gain
            * (-MAX_COMMAND_VELOCITY_RAD_S - self.dq_rad_s),
            -MAX_COMMAND_ACCELERATION_RAD_S2,
        )
        acceleration = np.clip(
            jerk_limited_acceleration,
            safe_minimum_acceleration,
            safe_maximum_acceleration,
        )
        self.dq_rad_s = self.dq_rad_s + acceleration * SERVO_DT_S
        self.q_rad = np.clip(
            self.q_rad + self.dq_rad_s * SERVO_DT_S,
            SAFE_JOINT_LOWER_RAD,
            SAFE_JOINT_UPPER_RAD,
        )
        self.ddq_rad_s2 = acceleration
        return self.state


class ServoSubstepAccumulator:
    """Produce the exact 8,8,9 virtual-1-kHz schedule at 120 Hz."""

    def __init__(self) -> None:
        self.remainder_s = 0.0

    def substeps_for(self, physics_dt_s: float) -> int:
        if not math.isfinite(physics_dt_s) or physics_dt_s <= 0.0:
            raise ValueError("physics_dt_s must be finite and positive")
        self.remainder_s += float(physics_dt_s)
        count = int(math.floor((self.remainder_s + 1.0e-15) / SERVO_DT_S))
        self.remainder_s -= count * SERVO_DT_S
        return count


__all__ = [
    "DesiredCommandState",
    "MAX_COMMAND_ACCELERATION_RAD_S2",
    "MAX_COMMAND_JERK_RAD_S3",
    "MAX_COMMAND_VELOCITY_RAD_S",
    "SERVO_DT_S",
    "ServoSubstepAccumulator",
    "V225FrankaCommandInterpolator",
    "map_qd_g015_20hz_arm_action",
    "map_v225_20hz_arm_action",
]
