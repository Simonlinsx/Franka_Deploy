"""Standalone reference for the accepted V225/V226 Franka motion generator.

The policy updates a held joint target at 20 Hz. This module converts that
piecewise-constant target into a continuous 1 kHz joint-position command. The
returned command is intended to be sent through libfranka with its rate limiter
enabled and its internal low-pass filter disabled; the 100 Hz filter is already
applied here.

This file does not connect to hardware. It is deliberately NumPy-only so the
real controller can use it as an offline numerical regression reference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


POLICY_DT_S = 0.05
SERVO_DT_S = 0.001
TARGET_LOWPASS_CUTOFF_HZ = 100.0
TRACKING_NATURAL_FREQUENCY_HZ = 6.0
TRACKING_DAMPING_RATIO = 1.0
MAX_ACCELERATION_RAD_S2 = 5.0
MAX_JERK_RAD_S3 = 250.0

ARM_ACTION_GAIN_RAD = 0.045
ARM_TARGET_ALPHA = 0.40
ARM_MAX_TARGET_STEP_RAD = 0.045
ARM_MEASURED_ENVELOPE_RAD = 0.05

SAFE_JOINT_LOWER_RAD = np.asarray(
    (-2.6937, -1.7337, -2.8507, -2.9921, -2.7565, 0.5945, -2.9659),
    dtype=np.float64,
)
SAFE_JOINT_UPPER_RAD = np.asarray(
    (2.6937, 1.7337, 2.8507, -0.2018, 2.7565, 4.4669, 2.9659),
    dtype=np.float64,
)


def _vector7(value: object, name: str, *, dtype: np.dtype = np.float64) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != (7,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain seven finite values")
    return result.copy()


def map_policy_action_to_held_target(
    previous_target_q_rad: object,
    measured_q_rad: object,
    action_arm: object,
    *,
    joint_lower_rad: Sequence[float] = SAFE_JOINT_LOWER_RAD,
    joint_upper_rad: Sequence[float] = SAFE_JOINT_UPPER_RAD,
) -> np.ndarray:
    """Apply the exact float32 V225/V226 20 Hz arm action mapping."""

    previous = _vector7(
        previous_target_q_rad, "previous_target_q_rad", dtype=np.float32
    )
    measured = _vector7(measured_q_rad, "measured_q_rad", dtype=np.float32)
    action = np.clip(
        _vector7(action_arm, "action_arm", dtype=np.float32),
        np.float32(-1.0),
        np.float32(1.0),
    )
    lower = _vector7(joint_lower_rad, "joint_lower_rad", dtype=np.float32)
    upper = _vector7(joint_upper_rad, "joint_upper_rad", dtype=np.float32)

    raw_target = previous + np.float32(ARM_ACTION_GAIN_RAD) * action
    filtered_target = (
        np.float32(ARM_TARGET_ALPHA) * raw_target
        + np.float32(1.0 - ARM_TARGET_ALPHA) * previous
    )
    target_delta = np.clip(
        filtered_target - previous,
        -np.float32(ARM_MAX_TARGET_STEP_RAD),
        np.float32(ARM_MAX_TARGET_STEP_RAD),
    )
    next_target = previous + target_delta
    safe_lower = np.maximum(
        lower, measured - np.float32(ARM_MEASURED_ENVELOPE_RAD)
    )
    safe_upper = np.minimum(
        upper, measured + np.float32(ARM_MEASURED_ENVELOPE_RAD)
    )
    return np.clip(next_target, safe_lower, safe_upper).astype(
        np.float32, copy=False
    )


@dataclass(frozen=True)
class DesiredCommandState:
    q_rad: np.ndarray
    dq_rad_s: np.ndarray
    ddq_rad_s2: np.ndarray
    filtered_target_rad: np.ndarray


class InterpolatedJointPositionGenerator:
    """Stateful 1 kHz target filter and critically damped interpolator."""

    def __init__(
        self,
        start_q_rad: object,
        *,
        start_dq_rad_s: object | None = None,
        start_ddq_rad_s2: object | None = None,
        cutoff_frequency_hz: float = TARGET_LOWPASS_CUTOFF_HZ,
        natural_frequency_hz: float = TRACKING_NATURAL_FREQUENCY_HZ,
        damping_ratio: float = TRACKING_DAMPING_RATIO,
    ) -> None:
        if cutoff_frequency_hz <= 0.0:
            raise ValueError("cutoff_frequency_hz must be positive")
        if natural_frequency_hz <= 0.0 or damping_ratio <= 0.0:
            raise ValueError("tracking frequency and damping must be positive")

        q = _vector7(start_q_rad, "start_q_rad")
        zeros = np.zeros(7, dtype=np.float64)
        self.q_rad = q
        self.dq_rad_s = (
            zeros.copy()
            if start_dq_rad_s is None
            else _vector7(start_dq_rad_s, "start_dq_rad_s")
        )
        self.ddq_rad_s2 = (
            zeros.copy()
            if start_ddq_rad_s2 is None
            else _vector7(start_ddq_rad_s2, "start_ddq_rad_s2")
        )
        self.filtered_target_rad = q.copy()
        self.lowpass_gain = SERVO_DT_S / (
            SERVO_DT_S + 1.0 / (2.0 * math.pi * cutoff_frequency_hz)
        )
        self.omega_rad_s = 2.0 * math.pi * natural_frequency_hz
        self.damping_ratio = float(damping_ratio)

    @property
    def state(self) -> DesiredCommandState:
        return DesiredCommandState(
            q_rad=self.q_rad.copy(),
            dq_rad_s=self.dq_rad_s.copy(),
            ddq_rad_s2=self.ddq_rad_s2.copy(),
            filtered_target_rad=self.filtered_target_rad.copy(),
        )

    def step(self, held_target_q_rad: object) -> DesiredCommandState:
        """Advance one 1 ms packet and return the pre-guard command."""

        held = np.clip(
            _vector7(held_target_q_rad, "held_target_q_rad"),
            SAFE_JOINT_LOWER_RAD,
            SAFE_JOINT_UPPER_RAD,
        )
        self.filtered_target_rad = (
            self.lowpass_gain * held
            + (1.0 - self.lowpass_gain) * self.filtered_target_rad
        )

        desired_acceleration = (
            self.omega_rad_s**2 * (self.filtered_target_rad - self.q_rad)
            - 2.0
            * self.damping_ratio
            * self.omega_rad_s
            * self.dq_rad_s
        )
        acceleration = self.ddq_rad_s2 + np.clip(
            (desired_acceleration - self.ddq_rad_s2) / SERVO_DT_S,
            -MAX_JERK_RAD_S3,
            MAX_JERK_RAD_S3,
        ) * SERVO_DT_S
        acceleration = np.clip(
            acceleration,
            -MAX_ACCELERATION_RAD_S2,
            MAX_ACCELERATION_RAD_S2,
        )

        velocity = self.dq_rad_s + acceleration * SERVO_DT_S
        position = self.q_rad + velocity * SERVO_DT_S
        self.q_rad = np.clip(
            position, SAFE_JOINT_LOWER_RAD, SAFE_JOINT_UPPER_RAD
        )
        self.dq_rad_s = velocity
        self.ddq_rad_s2 = acceleration
        return self.state

    def advance(self, held_target_q_rad: object, packets: int) -> DesiredCommandState:
        if packets <= 0:
            raise ValueError("packets must be positive")
        for _ in range(int(packets)):
            self.step(held_target_q_rad)
        return self.state


__all__ = [
    "ARM_ACTION_GAIN_RAD",
    "ARM_MAX_TARGET_STEP_RAD",
    "ARM_MEASURED_ENVELOPE_RAD",
    "ARM_TARGET_ALPHA",
    "DesiredCommandState",
    "InterpolatedJointPositionGenerator",
    "MAX_ACCELERATION_RAD_S2",
    "MAX_JERK_RAD_S3",
    "POLICY_DT_S",
    "SAFE_JOINT_LOWER_RAD",
    "SAFE_JOINT_UPPER_RAD",
    "SERVO_DT_S",
    "TARGET_LOWPASS_CUTOFF_HZ",
    "TRACKING_DAMPING_RATIO",
    "TRACKING_NATURAL_FREQUENCY_HZ",
    "map_policy_action_to_held_target",
]
