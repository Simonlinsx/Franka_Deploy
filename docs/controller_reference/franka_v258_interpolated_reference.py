"""Deployment reference for the V258 FR3 20 Hz command path and 29D state.

The policy publishes a held joint target at 20 Hz.  A stateful 1 kHz motion
generator turns that target into a continuous desired trajectory.  The 29D
observation is constructed from software-owned controller state and the latest
measured joint position, so it is available on the real robot without any
simulation-only signal.

This module is intentionally NumPy-only and does not connect to hardware.  It
is the numerical oracle for a deployment implementation and for the matching
batched Isaac Lab implementation.
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
MAX_VELOCITY_RAD_S = 0.5
MAX_ACCELERATION_RAD_S2 = 5.0
MAX_JERK_RAD_S3 = 250.0

ARM_ACTION_GAIN_RAD = 0.045
ARM_TARGET_ALPHA = 0.40
ARM_MAX_TARGET_STEP_RAD = 0.045
ARM_MEASURED_ENVELOPE_RAD = 0.05

TARGET_ERROR_SCALE_RAD = 0.018
TRACKING_ERROR_SCALE_RAD = 0.05

SAFE_JOINT_LOWER_RAD = np.asarray(
    (-2.6937, -1.7337, -2.8507, -2.9921, -2.7565, 0.5945, -2.9659),
    dtype=np.float64,
)
SAFE_JOINT_UPPER_RAD = np.asarray(
    (2.6937, 1.7337, 2.8507, -0.2018, 2.7565, 4.4669, 2.9659),
    dtype=np.float64,
)

# Exact 29D layout consumed by the current deployable Students.
CONTROLLER_STATE_DIM = 29
TARGET_LAG_SLICE = slice(0, 7)
TRACKING_ERROR_SLICE = slice(7, 14)
DESIRED_VELOCITY_SLICE = slice(14, 21)
DESIRED_ACCELERATION_SLICE = slice(21, 28)
EXECUTION_ALPHA_INDEX = 28


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
    """Apply the float32 20 Hz arm action mapping used by V258."""

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
    """Stateful V258 1 kHz target filter and bounded second-order tracker."""

    def __init__(
        self,
        start_q_rad: object,
        *,
        start_dq_rad_s: object | None = None,
        start_ddq_rad_s2: object | None = None,
        cutoff_frequency_hz: float = TARGET_LOWPASS_CUTOFF_HZ,
        natural_frequency_hz: float = TRACKING_NATURAL_FREQUENCY_HZ,
        damping_ratio: float = TRACKING_DAMPING_RATIO,
        max_velocity_rad_s: float = MAX_VELOCITY_RAD_S,
        max_acceleration_rad_s2: float = MAX_ACCELERATION_RAD_S2,
        max_jerk_rad_s3: float = MAX_JERK_RAD_S3,
    ) -> None:
        for name, value in (
            ("cutoff_frequency_hz", cutoff_frequency_hz),
            ("natural_frequency_hz", natural_frequency_hz),
            ("damping_ratio", damping_ratio),
            ("max_velocity_rad_s", max_velocity_rad_s),
            ("max_acceleration_rad_s2", max_acceleration_rad_s2),
            ("max_jerk_rad_s3", max_jerk_rad_s3),
        ):
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")

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
        self.omega_rad_s = 2.0 * math.pi * float(natural_frequency_hz)
        self.damping_ratio = float(damping_ratio)
        self.max_velocity_rad_s = float(max_velocity_rad_s)
        self.max_acceleration_rad_s2 = float(max_acceleration_rad_s2)
        self.max_jerk_rad_s3 = float(max_jerk_rad_s3)

    @property
    def state(self) -> DesiredCommandState:
        return DesiredCommandState(
            q_rad=self.q_rad.copy(),
            dq_rad_s=self.dq_rad_s.copy(),
            ddq_rad_s2=self.ddq_rad_s2.copy(),
            filtered_target_rad=self.filtered_target_rad.copy(),
        )

    def step(self, held_target_q_rad: object) -> DesiredCommandState:
        """Advance one 1 ms packet and return the generated desired state."""

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
        bounded_desired_acceleration = np.clip(
            desired_acceleration,
            -self.max_acceleration_rad_s2,
            self.max_acceleration_rad_s2,
        )
        acceleration = self.ddq_rad_s2 + np.clip(
            (bounded_desired_acceleration - self.ddq_rad_s2) / SERVO_DT_S,
            -self.max_jerk_rad_s3,
            self.max_jerk_rad_s3,
        ) * SERVO_DT_S
        acceleration = np.clip(
            acceleration,
            -self.max_acceleration_rad_s2,
            self.max_acceleration_rad_s2,
        )
        velocity = np.clip(
            self.dq_rad_s + acceleration * SERVO_DT_S,
            -self.max_velocity_rad_s,
            self.max_velocity_rad_s,
        )
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


def build_controller_state_observation(
    held_target_rad: object,
    desired_position_rad: object,
    measured_position_rad: object,
    desired_velocity_rad_s: object,
    desired_acceleration_rad_s2: object,
    *,
    execution_alpha: float = 1.0,
) -> np.ndarray:
    """Construct the exact float32 29D state appended after the 67D block.

    Layout::

        0:7   clip((held_target - q_d) / 0.018, -4, 4)
        7:14  clip((q_d - measured_q) / 0.05, -4, 4)
        14:21 clip(dq_d / 0.5, -1, 1)
        21:28 clip(ddq_d / 5.0, -1, 1)
        28    execution alpha; exactly 1.0 on the deployed controller
    """

    if not np.isfinite(execution_alpha) or not 0.0 <= execution_alpha <= 1.0:
        raise ValueError("execution_alpha must be finite and in [0, 1]")
    held = _vector7(held_target_rad, "held_target_rad")
    desired_q = _vector7(desired_position_rad, "desired_position_rad")
    measured_q = _vector7(measured_position_rad, "measured_position_rad")
    desired_dq = _vector7(desired_velocity_rad_s, "desired_velocity_rad_s")
    desired_ddq = _vector7(
        desired_acceleration_rad_s2, "desired_acceleration_rad_s2"
    )
    result = np.empty(CONTROLLER_STATE_DIM, dtype=np.float32)
    result[TARGET_LAG_SLICE] = np.clip(
        (held - desired_q) / TARGET_ERROR_SCALE_RAD, -4.0, 4.0
    )
    result[TRACKING_ERROR_SLICE] = np.clip(
        (desired_q - measured_q) / TRACKING_ERROR_SCALE_RAD, -4.0, 4.0
    )
    result[DESIRED_VELOCITY_SLICE] = np.clip(
        desired_dq / MAX_VELOCITY_RAD_S, -1.0, 1.0
    )
    result[DESIRED_ACCELERATION_SLICE] = np.clip(
        desired_ddq / MAX_ACCELERATION_RAD_S2, -1.0, 1.0
    )
    result[EXECUTION_ALPHA_INDEX] = np.float32(execution_alpha)
    return result


def controller_state_from_generator(
    generator: InterpolatedJointPositionGenerator,
    held_target_rad: object,
    measured_position_rad: object,
    *,
    execution_alpha: float = 1.0,
) -> np.ndarray:
    """Convenience wrapper for a policy thread reading a generator snapshot."""

    state = generator.state
    return build_controller_state_observation(
        held_target_rad,
        state.q_rad,
        measured_position_rad,
        state.dq_rad_s,
        state.ddq_rad_s2,
        execution_alpha=execution_alpha,
    )


__all__ = [
    "ARM_ACTION_GAIN_RAD",
    "ARM_MAX_TARGET_STEP_RAD",
    "ARM_MEASURED_ENVELOPE_RAD",
    "ARM_TARGET_ALPHA",
    "CONTROLLER_STATE_DIM",
    "DESIRED_ACCELERATION_SLICE",
    "DESIRED_VELOCITY_SLICE",
    "DesiredCommandState",
    "EXECUTION_ALPHA_INDEX",
    "InterpolatedJointPositionGenerator",
    "MAX_ACCELERATION_RAD_S2",
    "MAX_JERK_RAD_S3",
    "MAX_VELOCITY_RAD_S",
    "POLICY_DT_S",
    "SAFE_JOINT_LOWER_RAD",
    "SAFE_JOINT_UPPER_RAD",
    "SERVO_DT_S",
    "TARGET_ERROR_SCALE_RAD",
    "TARGET_LAG_SLICE",
    "TARGET_LOWPASS_CUTOFF_HZ",
    "TRACKING_DAMPING_RATIO",
    "TRACKING_ERROR_SCALE_RAD",
    "TRACKING_ERROR_SLICE",
    "TRACKING_NATURAL_FREQUENCY_HZ",
    "build_controller_state_observation",
    "controller_state_from_generator",
    "map_policy_action_to_held_target",
]
