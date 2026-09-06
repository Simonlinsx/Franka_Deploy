#!/usr/bin/env python3
from __future__ import annotations

"""Run a 10 Hz EEF policy through a Cartesian impedance torque controller.

The policy action convention is:

    [dx, dy, dz, drx, dry, drz, gripper]

The policy updates a filtered target pose. The 1 kHz torque loop applies a
Cartesian impedance law:

    tau = J^T (K * pose_error - D * ee_twist) + tau_nullspace + coriolis

This is intentionally conservative and dry-run by default.
"""

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from run_eef_policy_env import (
    AsyncGripperController,
    HandCraftedPolicy,
    PolicyObservation,
    add_vec,
    apply_delta_rotation,
    clip_norm,
    format_vec,
    matmul3,
    matrix_to_rotvec,
    norm,
    rotation_distance,
    rotation_from_pose,
    scale_vec,
    sub_vec,
    translation_from_pose,
    transpose3,
)


def parse_vec(values: Iterable[str], expected_len: int, name: str) -> List[float]:
    parsed = [float(value) for value in values]
    if len(parsed) != expected_len:
        raise argparse.ArgumentTypeError(f"{name} expects exactly {expected_len} values")
    return parsed


def parse_vec3(values: Iterable[str]) -> List[float]:
    return parse_vec(values, 3, "vec3")


def filter_rotation(
    current: List[List[float]], target: List[List[float]], alpha: float
) -> List[List[float]]:
    alpha = min(1.0, max(0.0, alpha))
    relative = matmul3(target, transpose3(current))
    incremental_rotvec = scale_vec(matrix_to_rotvec(relative), alpha)
    return apply_delta_rotation(current, incremental_rotvec, "base")


def jacobian_from_model(model, state, order: str) -> np.ndarray:
    # libfranka arrays are column-major. Keep this configurable for diagnostics.
    return np.asarray(model.zero_jacobian(state), dtype=float).reshape((6, 7), order=order)


def saturate_rate(target: np.ndarray, previous: np.ndarray, max_delta: float) -> np.ndarray:
    return previous + np.clip(target - previous, -max_delta, max_delta)


def clip_norm_np(values: np.ndarray, max_norm: float) -> np.ndarray:
    value_norm = float(np.linalg.norm(values))
    if value_norm <= max_norm or value_norm == 0.0:
        return values
    return values * (max_norm / value_norm)


@dataclass
class ImpedanceState:
    initial_xyz: List[float]
    initial_rotation: List[List[float]]
    raw_target_xyz: List[float]
    raw_target_rotation: List[List[float]]
    filtered_target_xyz: List[float]
    filtered_target_rotation: List[List[float]]
    q_nullspace: np.ndarray


class CartesianImpedancePolicyEnv:
    def __init__(
        self,
        robot,
        model,
        pylibfranka,
        policy_hz: float,
        translational_stiffness: List[float],
        rotational_stiffness: List[float],
        translational_damping: List[float],
        rotational_damping: List[float],
        target_filter_tau: float,
        max_velocity: float,
        max_angular_velocity: float,
        max_radius: float,
        max_rotation_radius: float,
        max_force: float,
        max_task_torque: float,
        nullspace_stiffness: float,
        nullspace_damping: float,
        max_delta_tau: float,
        max_abs_tau: float,
        jacobian_order: str,
        rotation_frame: str,
    ) -> None:
        self.robot = robot
        self.model = model
        self.pylibfranka = pylibfranka
        self.policy_dt = 1.0 / policy_hz
        self.max_step = max_velocity / policy_hz
        self.max_rotation_step = max_angular_velocity / policy_hz
        self.stiffness = np.diag(translational_stiffness + rotational_stiffness)
        self.damping = np.diag(translational_damping + rotational_damping)
        self.target_filter_tau = target_filter_tau
        self.max_radius = max_radius
        self.max_rotation_radius = max_rotation_radius
        self.max_force = max_force
        self.max_task_torque = max_task_torque
        self.nullspace_stiffness = nullspace_stiffness
        self.nullspace_damping = nullspace_damping
        self.max_delta_tau = max_delta_tau
        self.max_abs_tau = max_abs_tau
        self.jacobian_order = jacobian_order
        self.rotation_frame = rotation_frame
        self.previous_tau: np.ndarray | None = None

        state = self.robot.read_once()
        self.state = self._state_from_robot_state(state)

    def _state_from_robot_state(self, robot_state) -> ImpedanceState:
        pose = list(robot_state.O_T_EE)
        xyz = translation_from_pose(pose)
        rotation = rotation_from_pose(pose)
        q = np.asarray(robot_state.q, dtype=float)
        return ImpedanceState(
            initial_xyz=list(xyz),
            initial_rotation=[row[:] for row in rotation],
            raw_target_xyz=list(xyz),
            raw_target_rotation=[row[:] for row in rotation],
            filtered_target_xyz=list(xyz),
            filtered_target_rotation=[row[:] for row in rotation],
            q_nullspace=q.copy(),
        )

    def reset_from_state(self, robot_state) -> None:
        self.state = self._state_from_robot_state(robot_state)
        self.previous_tau = np.asarray(robot_state.tau_J_d, dtype=float)

    def observation(self, robot_state, elapsed: float, policy_step: int) -> PolicyObservation:
        return PolicyObservation(
            q=list(robot_state.q),
            current_xyz=translation_from_pose(list(robot_state.O_T_EE)),
            command_xyz=list(self.state.raw_target_xyz),
            command_rotation=[row[:] for row in self.state.raw_target_rotation],
            gripper_width=None,
            elapsed=elapsed,
            policy_step=policy_step,
        )

    def set_policy_action(
        self, delta_xyz: List[float], delta_rotvec: List[float]
    ) -> tuple[List[float], List[float]]:
        safe_delta = clip_norm(delta_xyz, self.max_step)
        safe_delta_rotvec = clip_norm(delta_rotvec, self.max_rotation_step)

        next_target_xyz = add_vec(self.state.raw_target_xyz, safe_delta)
        next_target_rotation = apply_delta_rotation(
            self.state.raw_target_rotation, safe_delta_rotvec, self.rotation_frame
        )

        radius = norm(sub_vec(next_target_xyz, self.state.initial_xyz))
        if radius > self.max_radius:
            raise ValueError(
                f"target would be {radius:.4f} m from rollout start, exceeding "
                f"--max-radius={self.max_radius:.4f}."
            )
        rotation_radius = rotation_distance(self.state.initial_rotation, next_target_rotation)
        if rotation_radius > self.max_rotation_radius:
            raise ValueError(
                f"target would be {rotation_radius:.4f} rad from rollout start orientation, "
                f"exceeding --max-rotation-radius={self.max_rotation_radius:.4f}."
            )

        self.state.raw_target_xyz = next_target_xyz
        self.state.raw_target_rotation = next_target_rotation
        return safe_delta, safe_delta_rotvec

    def update_filtered_target(self, dt: float) -> None:
        if self.target_filter_tau <= 0.0:
            alpha = 1.0
        else:
            alpha = 1.0 - math.exp(-dt / self.target_filter_tau)

        self.state.filtered_target_xyz = [
            current + alpha * (target - current)
            for current, target in zip(self.state.filtered_target_xyz, self.state.raw_target_xyz)
        ]
        self.state.filtered_target_rotation = filter_rotation(
            self.state.filtered_target_rotation, self.state.raw_target_rotation, alpha
        )

    def torque_command(self, robot_state, dt: float):
        self.update_filtered_target(dt)

        current_pose = list(robot_state.O_T_EE)
        current_xyz = translation_from_pose(current_pose)
        current_rotation = rotation_from_pose(current_pose)
        position_error = np.asarray(sub_vec(self.state.filtered_target_xyz, current_xyz))
        orientation_error = np.asarray(
            matrix_to_rotvec(matmul3(self.state.filtered_target_rotation, transpose3(current_rotation)))
        )
        error = np.concatenate([position_error, orientation_error])

        jacobian = jacobian_from_model(self.model, robot_state, self.jacobian_order)
        dq = np.asarray(robot_state.dq, dtype=float)
        ee_twist = jacobian @ dq

        wrench = self.stiffness @ error - self.damping @ ee_twist
        wrench[:3] = clip_norm_np(wrench[:3], self.max_force)
        wrench[3:] = clip_norm_np(wrench[3:], self.max_task_torque)

        tau_task = jacobian.T @ wrench
        tau_nullspace = np.zeros(7)
        if self.nullspace_stiffness > 0.0 or self.nullspace_damping > 0.0:
            q = np.asarray(robot_state.q, dtype=float)
            jacobian_transpose_pinv = np.linalg.pinv(jacobian.T, rcond=1e-4)
            nullspace_projector = np.eye(7) - jacobian.T @ jacobian_transpose_pinv
            tau_nullspace = nullspace_projector @ (
                self.nullspace_stiffness * (self.state.q_nullspace - q)
                - self.nullspace_damping * dq
            )

        coriolis = np.asarray(self.model.coriolis(robot_state), dtype=float)
        tau = tau_task + tau_nullspace + coriolis

        if self.previous_tau is None:
            self.previous_tau = np.asarray(robot_state.tau_J_d, dtype=float)
        tau = saturate_rate(tau, self.previous_tau, self.max_delta_tau)
        tau = np.clip(tau, -self.max_abs_tau, self.max_abs_tau)
        self.previous_tau = tau
        return self.pylibfranka.Torques(tau.tolist())


def damping_from_stiffness(stiffness: List[float], damping_ratio: float) -> List[float]:
    return [2.0 * damping_ratio * math.sqrt(value) for value in stiffness]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="172.16.0.2", help="Franka Control FCI IP address")
    parser.add_argument(
        "--policy",
        choices=[
            "hold",
            "up",
            "down",
            "x-plus",
            "x-minus",
            "y-plus",
            "y-minus",
            "sine-z",
            "circle-xy",
            "rx-plus",
            "rx-minus",
            "ry-plus",
            "ry-minus",
            "rz-plus",
            "rz-minus",
        ],
        default="hold",
        help="hand-crafted placeholder policy",
    )
    parser.add_argument("--policy-hz", type=float, default=10.0, help="policy frequency in Hz")
    parser.add_argument("--duration", type=float, default=5.0, help="rollout duration in seconds")
    parser.add_argument("--max-velocity", type=float, default=0.02, help="max EEF target velocity, m/s")
    parser.add_argument(
        "--max-angular-velocity",
        type=float,
        default=0.05,
        help="max EEF target angular velocity, rad/s",
    )
    parser.add_argument("--script-velocity", type=float, default=0.005, help="scripted m/s")
    parser.add_argument(
        "--script-angular-velocity",
        type=float,
        default=0.02,
        help="scripted rad/s",
    )
    parser.add_argument("--sine-amplitude", type=float, default=0.005, help="sine-z amplitude, m")
    parser.add_argument("--circle-radius", type=float, default=0.005, help="circle-xy radius, m")
    parser.add_argument("--max-radius", type=float, default=0.04, help="max target radius, m")
    parser.add_argument(
        "--max-rotation-radius",
        type=float,
        default=0.10,
        help="max target orientation radius, rad",
    )
    parser.add_argument(
        "--rotation-frame",
        choices=["base", "eef"],
        default="base",
        help="frame for rotational action deltas",
    )
    parser.add_argument(
        "--trans-stiffness",
        nargs=3,
        metavar=("KX", "KY", "KZ"),
        default=["80", "80", "80"],
        help="Cartesian translational stiffness [N/m]",
    )
    parser.add_argument(
        "--rot-stiffness",
        nargs=3,
        metavar=("KRX", "KRY", "KRZ"),
        default=["5", "5", "5"],
        help="Cartesian rotational stiffness [Nm/rad]",
    )
    parser.add_argument(
        "--damping-ratio",
        type=float,
        default=1.0,
        help="damping ratio used as D=2*zeta*sqrt(K)",
    )
    parser.add_argument(
        "--target-filter-tau",
        type=float,
        default=0.20,
        help="first-order target filter time constant, seconds; 0 disables filtering",
    )
    parser.add_argument("--max-force", type=float, default=20.0, help="max task force norm, N")
    parser.add_argument(
        "--max-task-torque",
        type=float,
        default=4.0,
        help="max task rotational torque norm, Nm",
    )
    parser.add_argument(
        "--nullspace-stiffness",
        type=float,
        default=3.0,
        help="joint nullspace stiffness toward initial q",
    )
    parser.add_argument(
        "--nullspace-damping",
        type=float,
        default=4.0,
        help="joint nullspace damping",
    )
    parser.add_argument(
        "--max-delta-tau",
        type=float,
        default=0.5,
        help="max torque change per control tick, Nm",
    )
    parser.add_argument("--max-abs-tau", type=float, default=20.0, help="absolute torque clip, Nm")
    parser.add_argument(
        "--jacobian-order",
        choices=["F", "C"],
        default="F",
        help="reshape order for libfranka Jacobian array; F is libfranka default",
    )
    parser.add_argument(
        "--startup-hold",
        type=float,
        default=0.5,
        help="seconds to hold current target before policy starts",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=1.0,
        help="seconds to relax target to current pose before finishing",
    )
    parser.add_argument(
        "--gripper-policy",
        choices=["hold", "open", "close", "toggle", "sine"],
        default="hold",
        help="normalized gripper policy; 1=open, 0=closed",
    )
    parser.add_argument("--gripper-toggle-period", type=float, default=1.0)
    parser.add_argument("--use-gripper", action="store_true")
    parser.add_argument("--gripper-open-width", type=float, default=0.08)
    parser.add_argument("--gripper-closed-width", type=float, default=0.0)
    parser.add_argument("--gripper-speed", type=float, default=0.04)
    parser.add_argument("--gripper-min-interval", type=float, default=0.2)
    parser.add_argument("--gripper-homing", action="store_true")
    parser.add_argument("--execute", action="store_true", help="actually command torque control")
    parser.add_argument("--enforce-realtime", action="store_true")
    args = parser.parse_args()

    if args.policy_hz <= 0.0:
        raise ValueError("--policy-hz must be positive")
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive")
    if args.max_velocity <= 0.0 or args.max_angular_velocity <= 0.0:
        raise ValueError("velocity limits must be positive")
    if args.target_filter_tau < 0.0:
        raise ValueError("--target-filter-tau must be non-negative")
    if args.startup_hold < 0.0:
        raise ValueError("--startup-hold must be non-negative")
    if args.settle_time < 0.0:
        raise ValueError("--settle-time must be non-negative")

    trans_stiffness = parse_vec3(args.trans_stiffness)
    rot_stiffness = parse_vec3(args.rot_stiffness)
    if any(value <= 0.0 for value in trans_stiffness + rot_stiffness):
        raise ValueError("stiffness values must be positive")
    trans_damping = damping_from_stiffness(trans_stiffness, args.damping_ratio)
    rot_damping = damping_from_stiffness(rot_stiffness, args.damping_ratio)

    try:
        import pylibfranka
    except ImportError:
        print(
            "pylibfranka is not installed. Activate the .venv and install pylibfranka first.",
            file=sys.stderr,
        )
        return 2

    realtime_config = (
        pylibfranka.RealtimeConfig.kEnforce
        if args.enforce_realtime
        else pylibfranka.RealtimeConfig.kIgnore
    )
    robot = pylibfranka.Robot(args.ip, realtime_config)
    model = robot.load_model()
    initial_state = robot.read_once()
    start_pose = list(initial_state.O_T_EE)
    start_xyz = translation_from_pose(start_pose)

    policy = HandCraftedPolicy(
        args.policy,
        args.policy_hz,
        args.script_velocity,
        args.script_angular_velocity,
        args.sine_amplitude,
        args.circle_radius,
        args.gripper_policy,
        args.gripper_toggle_period,
    )

    print("Current q:       ", format_vec(initial_state.q))
    print("Start xyz:       ", format_vec(start_xyz))
    print(f"Policy:           {args.policy} @ {args.policy_hz:.2f} Hz")
    print(f"Duration:         {args.duration:.2f} s")
    print("Action format:    [dx, dy, dz, drx, dry, drz, gripper]")
    print("Trans stiffness:  ", format_vec(trans_stiffness))
    print("Rot stiffness:    ", format_vec(rot_stiffness))
    print("Trans damping:    ", format_vec(trans_damping))
    print("Rot damping:      ", format_vec(rot_damping))
    print(f"Target filter tau:{args.target_filter_tau:.3f} s")
    print(f"Max force:        {args.max_force:.2f} N")
    print(f"Max task torque:  {args.max_task_torque:.2f} Nm")
    print(f"Max delta tau:    {args.max_delta_tau:.2f} Nm/tick")
    print(f"Nullspace K/D:    {args.nullspace_stiffness:.2f} / {args.nullspace_damping:.2f}")

    if not args.execute:
        print("Dry-run only. Add --execute to run torque impedance control.")
        return 0

    env = CartesianImpedancePolicyEnv(
        robot=robot,
        model=model,
        pylibfranka=pylibfranka,
        policy_hz=args.policy_hz,
        translational_stiffness=trans_stiffness,
        rotational_stiffness=rot_stiffness,
        translational_damping=trans_damping,
        rotational_damping=rot_damping,
        target_filter_tau=args.target_filter_tau,
        max_velocity=args.max_velocity,
        max_angular_velocity=args.max_angular_velocity,
        max_radius=args.max_radius,
        max_rotation_radius=args.max_rotation_radius,
        max_force=args.max_force,
        max_task_torque=args.max_task_torque,
        nullspace_stiffness=args.nullspace_stiffness,
        nullspace_damping=args.nullspace_damping,
        max_delta_tau=args.max_delta_tau,
        max_abs_tau=args.max_abs_tau,
        jacobian_order=args.jacobian_order,
        rotation_frame=args.rotation_frame,
    )

    gripper_controller = None
    if args.use_gripper:
        gripper_controller = AsyncGripperController(
            pylibfranka=pylibfranka,
            ip=args.ip,
            open_width=args.gripper_open_width,
            closed_width=args.gripper_closed_width,
            speed=args.gripper_speed,
            min_interval=args.gripper_min_interval,
            homing=args.gripper_homing,
        )

    control = robot.start_torque_control()
    elapsed = 0.0
    next_policy_time = 0.0
    policy_step = 0
    last_print_step = -1

    try:
        hold_elapsed = 0.0
        while hold_elapsed <= args.startup_hold:
            state, period = control.readOnce()
            dt = period.to_sec()
            if hold_elapsed == 0.0:
                env.reset_from_state(state)
            hold_elapsed += dt
            control.writeOnce(env.torque_command(state, dt))

        while elapsed < args.duration:
            state, period = control.readOnce()
            dt = period.to_sec()
            elapsed += dt

            if elapsed >= next_policy_time:
                obs = env.observation(state, elapsed, policy_step)
                raw_action = policy.act(obs)
                safe_delta, safe_delta_rotvec = env.set_policy_action(
                    raw_action.delta_xyz, raw_action.delta_rotvec
                )
                if gripper_controller is not None and raw_action.gripper is not None:
                    gripper_controller.command_normalized(raw_action.gripper)
                if policy_step != last_print_step and policy_step % max(1, int(args.policy_hz)) == 0:
                    gripper_text = (
                        ""
                        if raw_action.gripper is None
                        else f" gripper={raw_action.gripper:.3f}"
                    )
                    print(
                        f"t={elapsed:.2f}s "
                        f"dxyz={format_vec(safe_delta)} "
                        f"drot={format_vec(safe_delta_rotvec)} "
                        f"target={format_vec(env.state.raw_target_xyz)}{gripper_text}"
                    )
                    last_print_step = policy_step
                policy_step += 1
                next_policy_time += env.policy_dt

            control.writeOnce(env.torque_command(state, dt))

        settle_elapsed = 0.0
        while settle_elapsed < args.settle_time:
            state, period = control.readOnce()
            dt = period.to_sec()
            settle_elapsed += dt
            pose = list(state.O_T_EE)
            env.state.raw_target_xyz = translation_from_pose(pose)
            env.state.raw_target_rotation = rotation_from_pose(pose)
            control.writeOnce(env.torque_command(state, dt))

        final_command = env.torque_command(state, dt)
        final_command.motion_finished = True
        control.writeOnce(final_command)
    except BaseException:
        try:
            robot.stop()
        except Exception:
            pass
        raise
    finally:
        if gripper_controller is not None:
            gripper_controller.close()

    final_state = robot.read_once()
    print("Final xyz:      ", format_vec(translation_from_pose(list(final_state.O_T_EE))))
    print("Final q:        ", format_vec(final_state.q))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
