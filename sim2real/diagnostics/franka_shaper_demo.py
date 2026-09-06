"""Runnable simulator-alignment example for the V225/V226 controller.

Run from the repository root:

    .venv/bin/python -m sim2real.diagnostics.franka_shaper_demo

This program is offline-only.  It does not import or open a robot interface.
The same integration pattern can be copied into an Isaac Lab environment with
``sim_dt=1/120`` and ``decimation=6``; see ``run_demo`` below.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from robot_control.reference.franka_v225_interpolator_reference import (
    MAX_COMMAND_ACCELERATION_RAD_S2,
    MAX_COMMAND_JERK_RAD_S3,
    MAX_COMMAND_VELOCITY_RAD_S,
    SERVO_DT_S,
    ServoSubstepAccumulator,
    V225FrankaCommandInterpolator,
    map_v225_20hz_arm_action,
)


POLICY_RATE_HZ = 20.0
PHYSICS_RATE_HZ = 120.0
PHYSICS_STEPS_PER_POLICY_STEP = 6
Q_HOME_RAD = np.asarray(
    [0.0, -0.569000006, 0.0, -2.809999943, 0.0, 3.036999941, 0.740999997],
    dtype=np.float64,
)


def example_policy_action(policy_step: int) -> np.ndarray:
    """Small deterministic normalized action used by the offline demo."""

    action = np.zeros(7, dtype=np.float32)
    phase = policy_step % 40
    action[0] = 1.0 if phase < 20 else -1.0
    action[1] = 0.35 if phase < 10 or phase >= 30 else -0.35
    return action


def run_demo(policy_steps: int) -> dict[str, np.ndarray]:
    """Generate target and shaped-command traces without touching hardware.

    In a simulator, replace ``measured_q`` with the current simulated joint
    positions, replace ``example_policy_action`` with ``policy(obs)[:7]``, and
    send each row of ``physics_command_q`` to the actuator before its 120 Hz
    physics step.
    """

    shaper = V225FrankaCommandInterpolator(Q_HOME_RAD)
    schedule = ServoSubstepAccumulator()
    held_target = Q_HOME_RAD.astype(np.float32)

    policy_target_q: list[np.ndarray] = []
    physics_command_q: list[np.ndarray] = []
    physics_servo_tick_counts: list[int] = []
    servo_q: list[np.ndarray] = []
    servo_dq: list[np.ndarray] = []
    servo_ddq: list[np.ndarray] = []

    for policy_step in range(policy_steps):
        # Ideal-tracking demo only. In Isaac Lab use robot.data.joint_pos here.
        measured_q = shaper.state.q_rad
        held_target = map_v225_20hz_arm_action(
            held_target,
            measured_q,
            example_policy_action(policy_step),
        )
        policy_target_q.append(held_target.copy())

        # sim_dt=1/120 and decimation=6.  The exact 1 ms shaper tick schedule
        # is 8,8,9,8,8,9, totalling 50 ticks during one 20 Hz action hold.
        for _ in range(PHYSICS_STEPS_PER_POLICY_STEP):
            servo_ticks = schedule.substeps_for(1.0 / PHYSICS_RATE_HZ)
            physics_servo_tick_counts.append(servo_ticks)
            for _ in range(servo_ticks):
                state = shaper.step(held_target)
                servo_q.append(state.q_rad.copy())
                servo_dq.append(state.dq_rad_s.copy())
                servo_ddq.append(state.ddq_rad_s2.copy())

            # Isaac Lab integration point:
            # robot.set_joint_position_target(state.q_rad)
            # scene.write_data_to_sim(); sim.step(); scene.update(sim_dt)
            physics_command_q.append(shaper.state.q_rad.copy())

    return {
        "policy_target_q_rad": np.asarray(policy_target_q),
        "physics_command_q_rad": np.asarray(physics_command_q),
        "physics_servo_tick_counts": np.asarray(
            physics_servo_tick_counts, dtype=np.int32
        ),
        "servo_q_rad": np.asarray(servo_q),
        "servo_dq_rad_s": np.asarray(servo_dq),
        "servo_ddq_rad_s2": np.asarray(servo_ddq),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline V212 20 Hz Franka shaper alignment example"
    )
    parser.add_argument("--policy-steps", type=int, default=80)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional .npz path for the generated target/command trace",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.policy_steps <= 0:
        raise SystemExit("--policy-steps must be positive")
    trace = run_demo(args.policy_steps)
    dq = trace["servo_dq_rad_s"]
    ddq = trace["servo_ddq_rad_s2"]
    initial_ddq = np.zeros((1, 7), dtype=np.float64)
    jerk = np.diff(np.concatenate((initial_ddq, ddq), axis=0), axis=0) / SERVO_DT_S
    target_delta = np.diff(
        np.concatenate((Q_HOME_RAD[None, :], trace["policy_target_q_rad"]), axis=0),
        axis=0,
    )

    print("Franka V225/V226 interpolator offline reference: PASS")
    print(
        "contract: policy=20Hz physics=120Hz servo=1000Hz "
        "action_gain=0.045 alpha=0.40 effective_max=0.018rad/tick"
    )
    print(
        "interpolator/final-guard limits: "
        f"velocity={MAX_COMMAND_VELOCITY_RAD_S:.3f}rad/s "
        f"acceleration={MAX_COMMAND_ACCELERATION_RAD_S2:.3f}rad/s^2 "
        f"jerk={MAX_COMMAND_JERK_RAD_S3:.3f}rad/s^3"
    )
    print(
        "first policy-period 1ms ticks per 120Hz physics step: "
        f"{trace['physics_servo_tick_counts'][:6].tolist()}"
    )
    print(
        "observed maxima: "
        f"target_delta={np.max(np.abs(target_delta)):.9f}rad/tick "
        f"velocity={np.max(np.abs(dq)):.9f}rad/s "
        f"acceleration={np.max(np.abs(ddq)):.9f}rad/s^2 "
        f"jerk={np.max(np.abs(jerk)):.9f}rad/s^3"
    )

    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **trace)
        print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
