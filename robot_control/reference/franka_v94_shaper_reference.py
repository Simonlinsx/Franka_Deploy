"""Reference copy of the Franka command shaping used by V94 deployment.

This module is for simulator alignment and offline trajectory comparison.  It
does not open either robot and is not imported by the real-hardware execution
path.

The production controller receives a held absolute joint-position target and
updates an exact desired-command history at 1 kHz.  The state below is the
previous desired command (q_d, dq_d, ddq_d), not the measured robot state.
Under normal 1 ms FCI periods, :class:`V94FrankaCommandShaper` mirrors the
native C++ algorithm, including its velocity and joint/episode barriers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


SERVO_DT_S = 0.001
MAX_COMMAND_VELOCITY_RAD_S = 0.50
MAX_COMMAND_ACCELERATION_RAD_S2 = 4.0
MAX_COMMAND_JERK_RAD_S3 = 120.0
MAX_EPISODE_DELTA_RAD = 1.21
MAX_TRACKING_ERROR_RAD = 0.01
MAX_UNCONTROLLED_CONTINUATION_PACKETS = 20

VELOCITY_BARRIER_GAIN_PER_S = 1.0
POSITION_BARRIER_GAIN_PER_S = 1.0
POSITION_ACCELERATION_BARRIER_GAIN_PER_S = 1.0
POSITION_TARGET_RESERVE_RAD = 0.060
POSITION_TRACKING_RESERVE_RAD = MAX_TRACKING_ERROR_RAD

SAFE_JOINT_LOWER_RAD = np.asarray(
    [-2.6937, -1.7337, -2.8507, -2.9921, -2.7565, 0.5945, -2.9659],
    dtype=np.float64,
)
SAFE_JOINT_UPPER_RAD = np.asarray(
    [2.6937, 1.7337, 2.8507, -0.2018, 2.7565, 4.4669, 2.9659],
    dtype=np.float64,
)


def _vector7(value: object, name: str, *, dtype: np.dtype = np.float64) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != (7,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain seven finite values")
    return result.copy()


@dataclass(frozen=True)
class DesiredCommandState:
    """Previous desired command, corresponding to FCI q_d/dq_d/ddq_d."""

    q_rad: np.ndarray
    dq_rad_s: np.ndarray
    ddq_rad_s2: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "q_rad", _vector7(self.q_rad, "q_rad"))
        object.__setattr__(self, "dq_rad_s", _vector7(self.dq_rad_s, "dq_rad_s"))
        object.__setattr__(
            self, "ddq_rad_s2", _vector7(self.ddq_rad_s2, "ddq_rad_s2")
        )


def map_incremental_arm_action(
    previous_target_q_rad: object,
    measured_q_rad: object,
    action_arm: object,
    *,
    raw_gain_rad: float,
    target_filter_alpha: float,
    maximum_target_step_rad: float,
    measured_arm_envelope_rad: float = 0.05,
    joint_lower_rad: object = SAFE_JOINT_LOWER_RAD,
    joint_upper_rad: object = SAFE_JOINT_UPPER_RAD,
) -> np.ndarray:
    """Map one normalized arm action to the held absolute joint target.

    This mirrors :class:`sim2real.contracts.actions.V94ActionMapper` using float32:

        raw = previous + raw_gain * clip(action, -1, 1)
        filtered = alpha * raw + (1 - alpha) * previous
        next = previous + clip(filtered - previous, +/-maximum_target_step)

    The resulting target is then bounded to ``measured_q +/- envelope`` and
    the safe joint interval.  The V212 20 Hz contract uses
    ``raw_gain=0.045``, ``alpha=0.40`` and ``maximum_target_step=0.045``;
    therefore a saturated action normally advances the held target by
    0.018 rad per policy tick.
    """

    scalar_values = {
        "raw_gain_rad": raw_gain_rad,
        "target_filter_alpha": target_filter_alpha,
        "maximum_target_step_rad": maximum_target_step_rad,
        "measured_arm_envelope_rad": measured_arm_envelope_rad,
    }
    for name, value in scalar_values.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if float(target_filter_alpha) > 1.0:
        raise ValueError("target_filter_alpha must not exceed one")

    previous = _vector7(previous_target_q_rad, "previous_target_q_rad", dtype=np.float32)
    measured = _vector7(measured_q_rad, "measured_q_rad", dtype=np.float32)
    action = _vector7(action_arm, "action_arm", dtype=np.float32)
    lower = _vector7(joint_lower_rad, "joint_lower_rad", dtype=np.float32)
    upper = _vector7(joint_upper_rad, "joint_upper_rad", dtype=np.float32)

    executed = np.clip(action, np.float32(-1.0), np.float32(1.0))
    raw_target = previous + np.float32(raw_gain_rad) * executed
    next_target = (
        np.float32(target_filter_alpha) * raw_target
        + (np.float32(1.0) - np.float32(target_filter_alpha)) * previous
    )
    target_delta = np.clip(
        next_target - previous,
        -np.float32(maximum_target_step_rad),
        np.float32(maximum_target_step_rad),
    )
    next_target = previous + target_delta
    safe_lower = np.maximum(
        lower, measured - np.float32(measured_arm_envelope_rad)
    )
    safe_upper = np.minimum(
        upper, measured + np.float32(measured_arm_envelope_rad)
    )
    return np.clip(next_target, safe_lower, safe_upper).astype(
        np.float32, copy=False
    )


def map_current_v94_arm_action(
    previous_target_q_rad: object,
    measured_q_rad: object,
    action_arm: object,
    *,
    joint_lower_rad: object = SAFE_JOINT_LOWER_RAD,
    joint_upper_rad: object = SAFE_JOINT_UPPER_RAD,
) -> np.ndarray:
    """Backward-compatible mapper for the legacy V94 60 Hz contract."""

    return map_incremental_arm_action(
        previous_target_q_rad,
        measured_q_rad,
        action_arm,
        raw_gain_rad=0.015,
        target_filter_alpha=0.20,
        maximum_target_step_rad=0.015,
        measured_arm_envelope_rad=0.05,
        joint_lower_rad=joint_lower_rad,
        joint_upper_rad=joint_upper_rad,
    )


def map_v212_20hz_arm_action(
    previous_target_q_rad: object,
    measured_q_rad: object,
    action_arm: object,
    *,
    joint_lower_rad: object = SAFE_JOINT_LOWER_RAD,
    joint_upper_rad: object = SAFE_JOINT_UPPER_RAD,
) -> np.ndarray:
    """Map one V212 XYZ 20 Hz action using its checkpoint contract."""

    return map_incremental_arm_action(
        previous_target_q_rad,
        measured_q_rad,
        action_arm,
        raw_gain_rad=0.045,
        target_filter_alpha=0.40,
        maximum_target_step_rad=0.045,
        measured_arm_envelope_rad=0.05,
        joint_lower_rad=joint_lower_rad,
        joint_upper_rad=joint_upper_rad,
    )


class V94FrankaCommandShaper:
    """Normal-period reference for the native 1 kHz command shaper."""

    def __init__(
        self,
        start_q_rad: object,
        *,
        initial_dq_rad_s: object | None = None,
        initial_ddq_rad_s2: object | None = None,
    ) -> None:
        self.start_q_rad = _vector7(start_q_rad, "start_q_rad")
        zeros = np.zeros(7, dtype=np.float64)
        self.state = DesiredCommandState(
            q_rad=self.start_q_rad,
            dq_rad_s=(
                zeros
                if initial_dq_rad_s is None
                else _vector7(initial_dq_rad_s, "initial_dq_rad_s")
            ),
            ddq_rad_s2=(
                zeros
                if initial_ddq_rad_s2 is None
                else _vector7(initial_ddq_rad_s2, "initial_ddq_rad_s2")
            ),
        )

    def step(self, held_target_q_rad: object) -> DesiredCommandState:
        """Advance exactly one normal 1 ms command packet."""

        target = _vector7(held_target_q_rad, "held_target_q_rad")
        reference = self.state
        next_q = np.empty(7, dtype=np.float64)
        next_dq = np.empty(7, dtype=np.float64)
        next_ddq = np.empty(7, dtype=np.float64)

        dt = SERVO_DT_S
        dt_squared = dt * dt
        velocity_barrier_denominator = (
            1.0 + VELOCITY_BARRIER_GAIN_PER_S * dt
        )

        for index in range(7):
            episode_lower = self.start_q_rad[index] - MAX_EPISODE_DELTA_RAD
            episode_upper = self.start_q_rad[index] + MAX_EPISODE_DELTA_RAD
            raw_position_lower = max(SAFE_JOINT_LOWER_RAD[index], episode_lower)
            raw_position_upper = min(SAFE_JOINT_UPPER_RAD[index], episode_upper)
            command_position_lower = (
                raw_position_lower + POSITION_TRACKING_RESERVE_RAD
            )
            command_position_upper = (
                raw_position_upper - POSITION_TRACKING_RESERVE_RAD
            )
            target_position_lower = (
                raw_position_lower + POSITION_TARGET_RESERVE_RAD
            )
            target_position_upper = (
                raw_position_upper - POSITION_TARGET_RESERVE_RAD
            )
            bounded_target = float(
                np.clip(
                    target[index], target_position_lower, target_position_upper
                )
            )

            q0 = float(reference.q_rad[index])
            dq0 = float(reference.dq_rad_s[index])
            ddq0 = float(reference.ddq_rad_s2[index])
            error = bounded_target - q0
            stopping_velocity = math.sqrt(
                2.0 * MAX_COMMAND_ACCELERATION_RAD_S2 * abs(error)
            )
            desired_velocity = math.copysign(
                min(MAX_COMMAND_VELOCITY_RAD_S, stopping_velocity), error
            )
            desired_acceleration = (desired_velocity - dq0) / dt

            lower = max(
                -MAX_COMMAND_ACCELERATION_RAD_S2,
                ddq0 - MAX_COMMAND_JERK_RAD_S3 * dt,
            )
            upper = min(
                MAX_COMMAND_ACCELERATION_RAD_S2,
                ddq0 + MAX_COMMAND_JERK_RAD_S3 * dt,
            )

            for packet in range(
                1, MAX_UNCONTROLLED_CONTINUATION_PACKETS + 2
            ):
                steps = float(packet)
                base_position = q0 + steps * dq0 * dt
                acceleration_position_coefficient = (
                    0.5 * steps * (steps + 1.0) * dt_squared
                )
                lower = max(
                    lower,
                    (command_position_lower - base_position)
                    / acceleration_position_coefficient,
                )
                upper = min(
                    upper,
                    (command_position_upper - base_position)
                    / acceleration_position_coefficient,
                )

                barrier_acceleration_coefficient = (
                    steps * dt
                    + POSITION_BARRIER_GAIN_PER_S
                    * acceleration_position_coefficient
                )
                lower = max(
                    lower,
                    (
                        -POSITION_BARRIER_GAIN_PER_S
                        * (q0 - command_position_lower)
                        - dq0
                        * (
                            1.0
                            + POSITION_BARRIER_GAIN_PER_S * steps * dt
                        )
                    )
                    / barrier_acceleration_coefficient,
                )
                upper = min(
                    upper,
                    (
                        POSITION_BARRIER_GAIN_PER_S
                        * (command_position_upper - q0)
                        - dq0
                        * (
                            1.0
                            + POSITION_BARRIER_GAIN_PER_S * steps * dt
                        )
                    )
                    / barrier_acceleration_coefficient,
                )

                position_gain = (
                    POSITION_BARRIER_GAIN_PER_S
                    * POSITION_ACCELERATION_BARRIER_GAIN_PER_S
                )
                velocity_gain = (
                    POSITION_BARRIER_GAIN_PER_S
                    + POSITION_ACCELERATION_BARRIER_GAIN_PER_S
                )
                acceleration_barrier_coefficient = (
                    1.0
                    + position_gain * acceleration_position_coefficient
                    + velocity_gain * steps * dt
                )
                lower = max(
                    lower,
                    (
                        -position_gain
                        * (
                            q0
                            - command_position_lower
                            + steps * dq0 * dt
                        )
                        - velocity_gain * dq0
                    )
                    / acceleration_barrier_coefficient,
                )
                upper = min(
                    upper,
                    (
                        position_gain
                        * (
                            command_position_upper
                            - q0
                            - steps * dq0 * dt
                        )
                        - velocity_gain * dq0
                    )
                    / acceleration_barrier_coefficient,
                )

            lower = max(
                lower,
                VELOCITY_BARRIER_GAIN_PER_S
                * (-MAX_COMMAND_VELOCITY_RAD_S - dq0)
                / velocity_barrier_denominator,
            )
            upper = min(
                upper,
                VELOCITY_BARRIER_GAIN_PER_S
                * (MAX_COMMAND_VELOCITY_RAD_S - dq0)
                / velocity_barrier_denominator,
            )
            if lower > upper + 1.0e-12:
                raise RuntimeError(
                    "no safe jerk-limited command exists: "
                    f"axis={index + 1} q={q0} dq={dq0} ddq={ddq0} "
                    f"lower={lower} upper={upper}"
                )
            if lower > upper:
                collapsed = upper + 0.5 * (lower - upper)
                lower = collapsed
                upper = collapsed

            acceleration = float(
                np.clip(desired_acceleration, lower, upper)
            )
            velocity = dq0 + acceleration * dt
            position = q0 + velocity * dt

            # Reconstruct derivatives from the exact position command, just as
            # FCI does with backward Euler.
            next_q[index] = position
            next_dq[index] = (position - q0) / dt
            next_ddq[index] = (next_dq[index] - dq0) / dt
            jerk = (next_ddq[index] - ddq0) / dt
            if (
                abs(next_dq[index]) > MAX_COMMAND_VELOCITY_RAD_S + 1.0e-5
                or abs(next_ddq[index])
                > MAX_COMMAND_ACCELERATION_RAD_S2 + 1.0e-5
                or abs(jerk) > MAX_COMMAND_JERK_RAD_S3 + 1.0e-5
            ):
                raise RuntimeError("reference shaper left derivative envelope")

        self.state = DesiredCommandState(next_q, next_dq, next_ddq)
        return self.state


class ServoSubstepAccumulator:
    """Schedule exact 1 ms shaper ticks inside a slower physics loop.

    For ``physics_dt_s=1/120``, successive calls return ``8, 8, 9`` and repeat.
    Six 120 Hz physics steps therefore execute exactly 50 shaper ticks during
    one 20 Hz held policy target.
    """

    def __init__(self, *, servo_dt_s: float = SERVO_DT_S) -> None:
        if not math.isfinite(servo_dt_s) or servo_dt_s <= 0.0:
            raise ValueError("servo_dt_s must be finite and positive")
        self.servo_dt_s = float(servo_dt_s)
        self.remainder_s = 0.0

    def substeps_for(self, physics_dt_s: float) -> int:
        if not math.isfinite(physics_dt_s) or physics_dt_s <= 0.0:
            raise ValueError("physics_dt_s must be finite and positive")
        self.remainder_s += float(physics_dt_s)
        count = int(
            math.floor((self.remainder_s + 1.0e-15) / self.servo_dt_s)
        )
        self.remainder_s -= count * self.servo_dt_s
        return count


__all__ = [
    "DesiredCommandState",
    "MAX_COMMAND_ACCELERATION_RAD_S2",
    "MAX_COMMAND_JERK_RAD_S3",
    "MAX_COMMAND_VELOCITY_RAD_S",
    "SAFE_JOINT_LOWER_RAD",
    "SAFE_JOINT_UPPER_RAD",
    "SERVO_DT_S",
    "ServoSubstepAccumulator",
    "V94FrankaCommandShaper",
    "map_incremental_arm_action",
    "map_current_v94_arm_action",
    "map_v212_20hz_arm_action",
]
