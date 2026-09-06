#!/usr/bin/env python3
from __future__ import annotations

"""Run a 10 Hz EEF 6D-pose policy on FR3 with a 1 kHz Cartesian pose loop.

This is a minimal "robot_env" style inference script. The policy runs at
policy_hz and outputs an action with this convention:

    [dx, dy, dz, drx, dry, drz, gripper]

delta_xyz is in meters. delta_rotvec is a rotation vector in radians. The robot
control loop treats policy deltas as velocity targets, ramps velocity with
acceleration limits, and streams Cartesian velocity commands at the FCI rate.

The built-in policies are hand-crafted placeholders. Replace
HandCraftedPolicy.act() with your model inference when a learned policy is ready.
"""

import argparse
import math
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List

def parse_xyz(values: Iterable[str]) -> List[float]:
    xyz = [float(value) for value in values]
    if len(xyz) != 3:
        raise argparse.ArgumentTypeError("expected exactly 3 values")
    return xyz


def format_vec(values: Iterable[float], precision: int = 4) -> str:
    return "[" + ", ".join(f"{value:.{precision}f}" for value in values) + "]"


def translation_from_pose(O_T_EE: List[float]) -> List[float]:
    return [O_T_EE[12], O_T_EE[13], O_T_EE[14]]


def pose_with_translation(O_T_EE: List[float], xyz: List[float]) -> List[float]:
    pose = list(O_T_EE)
    pose[12], pose[13], pose[14] = xyz
    return pose


def rotation_from_pose(O_T_EE: List[float]) -> List[List[float]]:
    # O_T_EE is column-major. Return a row-major 3x3 rotation matrix.
    return [
        [O_T_EE[0], O_T_EE[4], O_T_EE[8]],
        [O_T_EE[1], O_T_EE[5], O_T_EE[9]],
        [O_T_EE[2], O_T_EE[6], O_T_EE[10]],
    ]


def pose_with_rotation_translation(
    O_T_EE: List[float], rotation: List[List[float]], xyz: List[float]
) -> List[float]:
    pose = list(O_T_EE)
    pose[0], pose[4], pose[8] = rotation[0]
    pose[1], pose[5], pose[9] = rotation[1]
    pose[2], pose[6], pose[10] = rotation[2]
    pose[12], pose[13], pose[14] = xyz
    return pose


def add_vec(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def sub_vec(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def scale_vec(values: List[float], scale: float) -> List[float]:
    return [value * scale for value in values]


def norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def clip_norm(values: List[float], max_norm: float) -> List[float]:
    value_norm = norm(values)
    if value_norm <= max_norm or value_norm == 0.0:
        return values
    return scale_vec(values, max_norm / value_norm)


def max_policy_step(max_velocity: float, policy_hz: float) -> float:
    return max_velocity / policy_hz


def move_toward_vec(current: List[float], target: List[float], max_delta: float) -> List[float]:
    delta = sub_vec(target, current)
    return add_vec(current, clip_norm(delta, max_delta))


def matmul3(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def matvec3(a: List[List[float]], v: List[float]) -> List[float]:
    return [sum(a[i][j] * v[j] for j in range(3)) for i in range(3)]


def transpose3(a: List[List[float]]) -> List[List[float]]:
    return [[a[j][i] for j in range(3)] for i in range(3)]


def eye3() -> List[List[float]]:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def rotvec_to_matrix(rotvec: List[float]) -> List[List[float]]:
    theta = norm(rotvec)
    if theta < 1e-12:
        return eye3()
    x, y, z = [value / theta for value in rotvec]
    c = math.cos(theta)
    s = math.sin(theta)
    one_minus_c = 1.0 - c
    return [
        [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
        [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
        [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
    ]


def matrix_to_rotvec(rotation: List[List[float]]) -> List[float]:
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    cos_theta = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
    theta = math.acos(cos_theta)
    if theta < 1e-12:
        return [0.0, 0.0, 0.0]
    if abs(math.pi - theta) < 1e-4:
        # Fallback for near-pi rotations. This path should rarely matter for
        # clipped policy deltas, but keeps the conversion numerically defined.
        axis = [
            math.sqrt(max(0.0, (rotation[0][0] + 1.0) / 2.0)),
            math.sqrt(max(0.0, (rotation[1][1] + 1.0) / 2.0)),
            math.sqrt(max(0.0, (rotation[2][2] + 1.0) / 2.0)),
        ]
        axis[1] = math.copysign(axis[1], rotation[0][1] + rotation[1][0])
        axis[2] = math.copysign(axis[2], rotation[0][2] + rotation[2][0])
        return scale_vec(axis, theta)
    denom = 2.0 * math.sin(theta)
    axis = [
        (rotation[2][1] - rotation[1][2]) / denom,
        (rotation[0][2] - rotation[2][0]) / denom,
        (rotation[1][0] - rotation[0][1]) / denom,
    ]
    return scale_vec(axis, theta)


def rotation_distance(a: List[List[float]], b: List[List[float]]) -> float:
    return norm(matrix_to_rotvec(matmul3(b, transpose3(a))))


def apply_delta_rotation(
    rotation: List[List[float]], delta_rotvec: List[float], frame: str
) -> List[List[float]]:
    delta_rotation = rotvec_to_matrix(delta_rotvec)
    if frame == "base":
        return matmul3(delta_rotation, rotation)
    if frame == "eef":
        return matmul3(rotation, delta_rotation)
    raise ValueError(f"unknown rotation frame: {frame}")


@dataclass
class PolicyObservation:
    q: List[float]
    current_xyz: List[float]
    command_xyz: List[float]
    command_rotation: List[List[float]]
    gripper_width: float | None
    elapsed: float
    policy_step: int


@dataclass
class PolicyAction:
    """Policy action: 6D EEF delta plus optional normalized gripper target."""

    delta_xyz: List[float]
    delta_rotvec: List[float]
    gripper: float | None = None


class HandCraftedPolicy:
    """Small scripted policies that output 6D EEF deltas at policy_hz."""

    def __init__(
        self,
        name: str,
        policy_hz: float,
        script_velocity: float,
        script_angular_velocity: float,
        sine_amplitude: float,
        circle_radius: float,
        gripper_policy: str,
        gripper_toggle_period: float,
    ) -> None:
        self.name = name
        self.policy_dt = 1.0 / policy_hz
        self.script_velocity = script_velocity
        self.script_angular_velocity = script_angular_velocity
        self.sine_amplitude = sine_amplitude
        self.circle_radius = circle_radius
        self.gripper_policy = gripper_policy
        self.gripper_toggle_period = gripper_toggle_period

    def act(self, obs: PolicyObservation) -> PolicyAction:
        """Return [dx, dy, dz, drx, dry, drz, gripper] for the next policy interval.

        gripper is normalized: 1.0 means open, 0.0 means closed. None means no
        gripper command for this step.
        """
        if self.name == "hold":
            velocity = [0.0, 0.0, 0.0]
            angular_velocity = [0.0, 0.0, 0.0]
        elif self.name == "up":
            velocity = [0.0, 0.0, self.script_velocity]
            angular_velocity = [0.0, 0.0, 0.0]
        elif self.name == "down":
            velocity = [0.0, 0.0, -self.script_velocity]
            angular_velocity = [0.0, 0.0, 0.0]
        elif self.name == "x-plus":
            velocity = [self.script_velocity, 0.0, 0.0]
            angular_velocity = [0.0, 0.0, 0.0]
        elif self.name == "x-minus":
            velocity = [-self.script_velocity, 0.0, 0.0]
            angular_velocity = [0.0, 0.0, 0.0]
        elif self.name == "y-plus":
            velocity = [0.0, self.script_velocity, 0.0]
            angular_velocity = [0.0, 0.0, 0.0]
        elif self.name == "y-minus":
            velocity = [0.0, -self.script_velocity, 0.0]
            angular_velocity = [0.0, 0.0, 0.0]
        elif self.name == "sine-z":
            omega = 2.0 * math.pi / 4.0
            z_velocity = self.sine_amplitude * omega * math.cos(omega * obs.elapsed)
            velocity = [0.0, 0.0, z_velocity]
            angular_velocity = [0.0, 0.0, 0.0]
        elif self.name == "circle-xy":
            omega = self.script_velocity / max(self.circle_radius, 1e-6)
            velocity = [
                -self.circle_radius * omega * math.sin(omega * obs.elapsed),
                self.circle_radius * omega * math.cos(omega * obs.elapsed),
                0.0,
            ]
            angular_velocity = [0.0, 0.0, 0.0]
        elif self.name == "rx-plus":
            velocity = [0.0, 0.0, 0.0]
            angular_velocity = [self.script_angular_velocity, 0.0, 0.0]
        elif self.name == "rx-minus":
            velocity = [0.0, 0.0, 0.0]
            angular_velocity = [-self.script_angular_velocity, 0.0, 0.0]
        elif self.name == "ry-plus":
            velocity = [0.0, 0.0, 0.0]
            angular_velocity = [0.0, self.script_angular_velocity, 0.0]
        elif self.name == "ry-minus":
            velocity = [0.0, 0.0, 0.0]
            angular_velocity = [0.0, -self.script_angular_velocity, 0.0]
        elif self.name == "rz-plus":
            velocity = [0.0, 0.0, 0.0]
            angular_velocity = [0.0, 0.0, self.script_angular_velocity]
        elif self.name == "rz-minus":
            velocity = [0.0, 0.0, 0.0]
            angular_velocity = [0.0, 0.0, -self.script_angular_velocity]
        else:
            raise ValueError(f"unknown hand-crafted policy: {self.name}")

        return PolicyAction(
            delta_xyz=scale_vec(velocity, self.policy_dt),
            delta_rotvec=scale_vec(angular_velocity, self.policy_dt),
            gripper=self._gripper_action(obs),
        )

    def _gripper_action(self, obs: PolicyObservation) -> float | None:
        if self.gripper_policy == "hold":
            return None
        if self.gripper_policy == "open":
            return 1.0
        if self.gripper_policy == "close":
            return 0.0
        if self.gripper_policy == "toggle":
            phase = int(obs.elapsed / max(self.gripper_toggle_period, 1e-6))
            return 1.0 if phase % 2 == 0 else 0.0
        if self.gripper_policy == "sine":
            return 0.5 + 0.5 * math.sin(2.0 * math.pi * obs.elapsed / 2.0)
        raise ValueError(f"unknown gripper policy: {self.gripper_policy}")


class AsyncGripperController:
    """Send Franka Hand width commands outside the 1 kHz arm control loop."""

    def __init__(
        self,
        pylibfranka,
        ip: str,
        open_width: float,
        closed_width: float,
        speed: float,
        min_interval: float,
        homing: bool,
    ) -> None:
        self.gripper = pylibfranka.Gripper(ip)
        if homing:
            self.gripper.homing()
        state = self.gripper.read_once()
        self.max_width = state.max_width
        self.open_width = min(open_width, self.max_width)
        self.closed_width = max(0.0, min(closed_width, self.open_width))
        self.speed = speed
        self.min_interval = min_interval
        self._queue: queue.Queue[float | None] = queue.Queue(maxsize=1)
        self._last_width: float | None = None
        self._last_command_time = 0.0
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

        print(
            "Gripper: "
            f"width={state.width:.4f} m, max_width={self.max_width:.4f} m, "
            f"open_width={self.open_width:.4f} m, closed_width={self.closed_width:.4f} m"
        )

    def width_from_action(self, gripper_action: float) -> float:
        normalized = max(0.0, min(1.0, gripper_action))
        return self.closed_width + normalized * (self.open_width - self.closed_width)

    def command_normalized(self, gripper_action: float) -> None:
        self.command_width(self.width_from_action(gripper_action))

    def command_width(self, width: float) -> None:
        width = max(self.closed_width, min(self.open_width, width))
        if self._last_width is not None and abs(width - self._last_width) < 0.002:
            return
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        self._queue.put_nowait(width)
        self._last_width = width

    def close(self) -> None:
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        self._queue.put_nowait(None)
        self._thread.join(timeout=2.0)

    def _worker(self) -> None:
        while True:
            width = self._queue.get()
            if width is None:
                try:
                    self.gripper.stop()
                except Exception:
                    pass
                return

            sleep_time = self.min_interval - (time.monotonic() - self._last_command_time)
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            try:
                self.gripper.move(width, self.speed)
                self._last_command_time = time.monotonic()
            except Exception as exc:
                print(f"Gripper command failed: {exc}", file=sys.stderr)


class FrankaEEFPolicyEnv:
    """EEF delta-action environment backed by pylibfranka Cartesian velocity control."""

    def __init__(
        self,
        robot,
        pylibfranka,
        policy_hz: float,
        max_velocity: float,
        max_angular_velocity: float,
        max_acceleration: float,
        max_angular_acceleration: float,
        max_radius: float,
        max_rotation_radius: float,
        rotation_frame: str,
    ) -> None:
        self.robot = robot
        self.pylibfranka = pylibfranka
        self.policy_dt = 1.0 / policy_hz
        self.max_step = max_policy_step(max_velocity, policy_hz)
        self.max_rotation_step = max_policy_step(max_angular_velocity, policy_hz)
        self.max_acceleration = max_acceleration
        self.max_angular_acceleration = max_angular_acceleration
        self.max_radius = max_radius
        self.max_rotation_radius = max_rotation_radius
        self.rotation_frame = rotation_frame

        initial_state = self.robot.read_once()
        self.initial_pose = list(initial_state.O_T_EE)
        self.initial_xyz = translation_from_pose(self.initial_pose)
        self.initial_rotation = rotation_from_pose(self.initial_pose)
        self.command_xyz = list(self.initial_xyz)
        self.command_rotation = [row[:] for row in self.initial_rotation]
        self.current_linear_velocity = [0.0, 0.0, 0.0]
        self.current_angular_velocity = [0.0, 0.0, 0.0]
        self.desired_linear_velocity = [0.0, 0.0, 0.0]
        self.desired_angular_velocity = [0.0, 0.0, 0.0]

    def reset_from_state(self, state) -> None:
        self.initial_pose = list(state.O_T_EE)
        self.initial_xyz = translation_from_pose(self.initial_pose)
        self.initial_rotation = rotation_from_pose(self.initial_pose)
        self.command_xyz = list(self.initial_xyz)
        self.command_rotation = [row[:] for row in self.initial_rotation]
        self.current_linear_velocity = [0.0, 0.0, 0.0]
        self.current_angular_velocity = [0.0, 0.0, 0.0]
        self.desired_linear_velocity = [0.0, 0.0, 0.0]
        self.desired_angular_velocity = [0.0, 0.0, 0.0]

    def observation(self, state, elapsed: float, policy_step: int) -> PolicyObservation:
        return PolicyObservation(
            q=list(state.q),
            current_xyz=translation_from_pose(list(state.O_T_EE)),
            command_xyz=list(self.command_xyz),
            command_rotation=[row[:] for row in self.command_rotation],
            gripper_width=None,
            elapsed=elapsed,
            policy_step=policy_step,
        )

    def set_policy_action(
        self, delta_xyz: List[float], delta_rotvec: List[float], elapsed: float
    ) -> PolicyAction:
        safe_delta = clip_norm(delta_xyz, self.max_step)
        safe_delta_rotvec = clip_norm(delta_rotvec, self.max_rotation_step)
        self.desired_linear_velocity = scale_vec(safe_delta, 1.0 / self.policy_dt)
        self.desired_angular_velocity = scale_vec(safe_delta_rotvec, 1.0 / self.policy_dt)
        return PolicyAction(delta_xyz=safe_delta, delta_rotvec=safe_delta_rotvec)

    def set_hold_action(self) -> None:
        self.desired_linear_velocity = [0.0, 0.0, 0.0]
        self.desired_angular_velocity = [0.0, 0.0, 0.0]

    def is_stopped(self) -> bool:
        return (
            norm(self.current_linear_velocity) < 1e-4
            and norm(self.current_angular_velocity) < 1e-4
            and norm(self.desired_linear_velocity) < 1e-4
            and norm(self.desired_angular_velocity) < 1e-4
        )

    def step_command(self, dt: float) -> tuple[List[float], List[List[float]]]:
        self.current_linear_velocity = move_toward_vec(
            self.current_linear_velocity,
            self.desired_linear_velocity,
            self.max_acceleration * dt,
        )
        self.current_angular_velocity = move_toward_vec(
            self.current_angular_velocity,
            self.desired_angular_velocity,
            self.max_angular_acceleration * dt,
        )
        next_target = add_vec(self.command_xyz, scale_vec(self.current_linear_velocity, dt))
        next_rotation = apply_delta_rotation(
            self.command_rotation,
            scale_vec(self.current_angular_velocity, dt),
            self.rotation_frame,
        )
        radius = norm(sub_vec(next_target, self.initial_xyz))
        if radius > self.max_radius:
            raise ValueError(
                f"next target would be {radius:.4f} m from the rollout start, exceeding "
                f"--max-radius={self.max_radius:.4f}. Stop or raise --max-radius intentionally."
            )
        rotation_radius = rotation_distance(self.initial_rotation, next_rotation)
        if rotation_radius > self.max_rotation_radius:
            raise ValueError(
                f"next target would be {rotation_radius:.4f} rad from the rollout start "
                f"orientation, exceeding --max-rotation-radius={self.max_rotation_radius:.4f}."
            )

        self.command_xyz = next_target
        self.command_rotation = next_rotation
        return self.command_xyz, self.command_rotation

    def velocity_command(self):
        angular_velocity = list(self.current_angular_velocity)
        if self.rotation_frame == "eef":
            angular_velocity = matvec3(self.command_rotation, angular_velocity)
        return self.pylibfranka.CartesianVelocities(
            list(self.current_linear_velocity) + angular_velocity
        )

    def zero_velocity_command(self):
        return self.pylibfranka.CartesianVelocities([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def simulate_rollout(
    start_pose: List[float],
    policy: HandCraftedPolicy,
    policy_hz: float,
    duration: float,
    settle_time: float,
    max_velocity: float,
    max_angular_velocity: float,
    max_acceleration: float,
    max_angular_acceleration: float,
    max_radius: float,
    max_rotation_radius: float,
    rotation_frame: str,
) -> Dict[str, List[float] | float | None]:
    policy_dt = 1.0 / policy_hz
    control_dt = 0.001
    max_step = max_policy_step(max_velocity, policy_hz)
    max_rotation_step = max_policy_step(max_angular_velocity, policy_hz)
    start_xyz = translation_from_pose(start_pose)
    command_xyz = list(start_xyz)
    command_rotation = rotation_from_pose(start_pose)
    initial_rotation = [row[:] for row in command_rotation]
    current_linear_velocity = [0.0, 0.0, 0.0]
    current_angular_velocity = [0.0, 0.0, 0.0]
    desired_linear_velocity = [0.0, 0.0, 0.0]
    desired_angular_velocity = [0.0, 0.0, 0.0]
    max_seen_radius = 0.0
    max_seen_rotation_radius = 0.0
    policy_step = 0
    next_policy_time = 0.0
    elapsed = 0.0
    final_gripper: float | None = None

    def integrate_step(dt: float) -> None:
        nonlocal command_xyz
        nonlocal command_rotation
        nonlocal current_linear_velocity
        nonlocal current_angular_velocity
        nonlocal max_seen_radius
        nonlocal max_seen_rotation_radius

        current_linear_velocity = move_toward_vec(
            current_linear_velocity,
            desired_linear_velocity,
            max_acceleration * dt,
        )
        current_angular_velocity = move_toward_vec(
            current_angular_velocity,
            desired_angular_velocity,
            max_angular_acceleration * dt,
        )
        command_xyz = add_vec(command_xyz, scale_vec(current_linear_velocity, dt))
        command_rotation = apply_delta_rotation(
            command_rotation, scale_vec(current_angular_velocity, dt), rotation_frame
        )
        max_seen_radius = max(max_seen_radius, norm(sub_vec(command_xyz, start_xyz)))
        max_seen_rotation_radius = max(
            max_seen_rotation_radius, rotation_distance(initial_rotation, command_rotation)
        )
        if max_seen_radius > max_radius:
            raise ValueError(
                f"dry-run target would be {max_seen_radius:.4f} m from start, exceeding "
                f"--max-radius={max_radius:.4f}"
            )
        if max_seen_rotation_radius > max_rotation_radius:
            raise ValueError(
                f"dry-run target would be {max_seen_rotation_radius:.4f} rad from start "
                f"orientation, exceeding --max-rotation-radius={max_rotation_radius:.4f}"
            )

    while elapsed < duration:
        if elapsed + 1e-12 >= next_policy_time:
            obs = PolicyObservation(
                [],
                command_xyz,
                command_xyz,
                [row[:] for row in command_rotation],
                None,
                elapsed,
                policy_step,
            )
            policy_action = policy.act(obs)
            action = clip_norm(policy_action.delta_xyz, max_step)
            rot_action = clip_norm(policy_action.delta_rotvec, max_rotation_step)
            desired_linear_velocity = scale_vec(action, 1.0 / policy_dt)
            desired_angular_velocity = scale_vec(rot_action, 1.0 / policy_dt)
            final_gripper = policy_action.gripper
            policy_step += 1
            next_policy_time += policy_dt

        dt = min(control_dt, duration - elapsed)
        integrate_step(dt)
        elapsed += dt

    desired_linear_velocity = [0.0, 0.0, 0.0]
    desired_angular_velocity = [0.0, 0.0, 0.0]
    settle_elapsed = 0.0
    while settle_elapsed < settle_time and (
        norm(current_linear_velocity) >= 1e-4 or norm(current_angular_velocity) >= 1e-4
    ):
        dt = min(control_dt, settle_time - settle_elapsed)
        integrate_step(dt)
        settle_elapsed += dt

    return {
        "final_xyz": command_xyz,
        "delta_xyz": sub_vec(command_xyz, start_xyz),
        "max_radius": max_seen_radius,
        "final_rotvec_from_start": matrix_to_rotvec(
            matmul3(command_rotation, transpose3(initial_rotation))
        ),
        "max_rotation_radius": max_seen_rotation_radius,
        "final_gripper": final_gripper,
    }


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
    parser.add_argument(
        "--max-velocity",
        type=float,
        default=0.03,
        help="max EEF translation velocity used for action clipping, m/s",
    )
    parser.add_argument(
        "--max-angular-velocity",
        type=float,
        default=0.30,
        help="max EEF angular velocity used for action clipping, rad/s",
    )
    parser.add_argument(
        "--max-acceleration",
        type=float,
        default=0.05,
        help="max EEF translation acceleration for velocity ramping, m/s^2",
    )
    parser.add_argument(
        "--max-angular-acceleration",
        type=float,
        default=0.05,
        help="max EEF angular acceleration for velocity ramping, rad/s^2",
    )
    parser.add_argument(
        "--script-velocity",
        type=float,
        default=0.01,
        help="velocity used by directional scripted policies, m/s",
    )
    parser.add_argument(
        "--script-angular-velocity",
        type=float,
        default=0.15,
        help="angular velocity used by rotational scripted policies, rad/s",
    )
    parser.add_argument(
        "--sine-amplitude",
        type=float,
        default=0.01,
        help="sine-z amplitude in meters",
    )
    parser.add_argument(
        "--circle-radius",
        type=float,
        default=0.01,
        help="circle-xy radius in meters",
    )
    parser.add_argument(
        "--max-radius",
        type=float,
        default=0.08,
        help="max distance from rollout start, meters",
    )
    parser.add_argument(
        "--max-rotation-radius",
        type=float,
        default=0.50,
        help="max orientation change from rollout start, radians",
    )
    parser.add_argument(
        "--rotation-frame",
        choices=["base", "eef"],
        default="base",
        help="frame for delta_rotvec actions: base frame or current EEF frame",
    )
    parser.add_argument(
        "--startup-hold",
        type=float,
        default=0.30,
        help="seconds to stream the current pose before starting policy actions",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=1.0,
        help="seconds allowed to ramp commanded velocity to zero before finishing",
    )
    parser.add_argument(
        "--gripper-policy",
        choices=["hold", "open", "close", "toggle", "sine"],
        default="hold",
        help="hand-crafted gripper action policy; normalized 1=open, 0=closed",
    )
    parser.add_argument(
        "--gripper-toggle-period",
        type=float,
        default=1.0,
        help="seconds per open/close phase for --gripper-policy toggle",
    )
    parser.add_argument(
        "--use-gripper",
        action="store_true",
        help="connect to Franka Hand and execute gripper actions asynchronously",
    )
    parser.add_argument(
        "--gripper-open-width",
        type=float,
        default=0.08,
        help="Franka Hand open width in meters, clipped to gripper max_width",
    )
    parser.add_argument(
        "--gripper-closed-width",
        type=float,
        default=0.0,
        help="Franka Hand closed width in meters",
    )
    parser.add_argument(
        "--gripper-speed",
        type=float,
        default=0.04,
        help="Franka Hand move speed in m/s",
    )
    parser.add_argument(
        "--gripper-min-interval",
        type=float,
        default=0.2,
        help="minimum time between gripper move commands, seconds",
    )
    parser.add_argument(
        "--gripper-homing",
        action="store_true",
        help="run Franka Hand homing before the rollout",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually move the robot; without this flag the script only prints a dry-run plan",
    )
    parser.add_argument(
        "--enforce-realtime",
        action="store_true",
        help="require real-time scheduling. Use after RT kernel setup.",
    )
    args = parser.parse_args()

    if args.policy_hz <= 0.0:
        raise ValueError("--policy-hz must be positive")
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive")
    if args.max_velocity <= 0.0:
        raise ValueError("--max-velocity must be positive")
    if args.max_angular_velocity <= 0.0:
        raise ValueError("--max-angular-velocity must be positive")
    if args.max_acceleration <= 0.0:
        raise ValueError("--max-acceleration must be positive")
    if args.max_angular_acceleration <= 0.0:
        raise ValueError("--max-angular-acceleration must be positive")
    if args.max_radius <= 0.0:
        raise ValueError("--max-radius must be positive")
    if args.max_rotation_radius <= 0.0:
        raise ValueError("--max-rotation-radius must be positive")
    if args.startup_hold < 0.0:
        raise ValueError("--startup-hold must be non-negative")
    if args.settle_time < 0.0:
        raise ValueError("--settle-time must be non-negative")

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
    initial_state = robot.read_once()
    start_pose = list(initial_state.O_T_EE)
    start_xyz = translation_from_pose(start_pose)
    max_step = max_policy_step(args.max_velocity, args.policy_hz)
    max_rotation_step = max_policy_step(args.max_angular_velocity, args.policy_hz)

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

    print("Current q:      ", format_vec(initial_state.q))
    print("Start xyz:      ", format_vec(start_xyz))
    print(f"Policy:          {args.policy} @ {args.policy_hz:.2f} Hz")
    print(f"Duration:        {args.duration:.2f} s")
    print(f"Max velocity:    {args.max_velocity:.4f} m/s")
    print(f"Max policy step: {max_step:.4f} m")
    print(f"Max radius:      {args.max_radius:.4f} m")
    print(f"Max angular vel: {args.max_angular_velocity:.4f} rad/s")
    print(f"Max rot step:    {max_rotation_step:.4f} rad")
    print(f"Max rot radius:  {args.max_rotation_radius:.4f} rad")
    print(f"Max accel:       {args.max_acceleration:.4f} m/s^2")
    print(f"Max angular acc: {args.max_angular_acceleration:.4f} rad/s^2")
    print(f"Rotation frame:  {args.rotation_frame}")
    print(f"Startup hold:    {args.startup_hold:.2f} s")
    print(f"Settle time:     {args.settle_time:.2f} s")
    print(f"Gripper policy:  {args.gripper_policy}")
    print("Action format:   [dx, dy, dz, drx, dry, drz, gripper]")

    dry_run = simulate_rollout(
        start_pose,
        policy,
        args.policy_hz,
        args.duration,
        args.settle_time,
        args.max_velocity,
        args.max_angular_velocity,
        args.max_acceleration,
        args.max_angular_acceleration,
        args.max_radius,
        args.max_rotation_radius,
        args.rotation_frame,
    )
    print("Dry-run final:  ", format_vec(dry_run["final_xyz"]))
    print("Dry-run delta:  ", format_vec(dry_run["delta_xyz"]))
    print(f"Dry-run radius:  {dry_run['max_radius']:.4f} m")
    print("Dry-run rotvec: ", format_vec(dry_run["final_rotvec_from_start"]))
    print(f"Dry-run rot rad: {dry_run['max_rotation_radius']:.4f} rad")
    if dry_run["final_gripper"] is not None:
        print(f"Dry-run gripper: {dry_run['final_gripper']:.3f} (1=open, 0=closed)")

    if not args.execute:
        print("Dry-run only. Add --execute to run the 1 kHz robot control loop.")
        return 0

    env = FrankaEEFPolicyEnv(
        robot=robot,
        pylibfranka=pylibfranka,
        policy_hz=args.policy_hz,
        max_velocity=args.max_velocity,
        max_angular_velocity=args.max_angular_velocity,
        max_acceleration=args.max_acceleration,
        max_angular_acceleration=args.max_angular_acceleration,
        max_radius=args.max_radius,
        max_rotation_radius=args.max_rotation_radius,
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

    control = robot.start_cartesian_velocity_control(pylibfranka.ControllerMode.CartesianImpedance)
    elapsed = 0.0
    next_policy_time = 0.0
    policy_step = 0
    last_print_step = -1

    try:
        hold_elapsed = 0.0
        while hold_elapsed <= args.startup_hold:
            state, period = control.readOnce()
            if hold_elapsed == 0.0:
                env.reset_from_state(state)
            hold_elapsed += period.to_sec()
            control.writeOnce(env.zero_velocity_command())

        while elapsed < args.duration:
            state, period = control.readOnce()
            dt = period.to_sec()
            elapsed += dt

            if elapsed >= next_policy_time:
                obs = env.observation(state, elapsed, policy_step)
                raw_action = policy.act(obs)
                safe_action = env.set_policy_action(
                    raw_action.delta_xyz, raw_action.delta_rotvec, elapsed
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
                        f"dxyz={format_vec(safe_action.delta_xyz)} "
                        f"drot={format_vec(safe_action.delta_rotvec)} "
                        f"v={format_vec(env.desired_linear_velocity)} "
                        f"w={format_vec(env.desired_angular_velocity)}{gripper_text}"
                    )
                    last_print_step = policy_step
                policy_step += 1
                next_policy_time += env.policy_dt

            env.step_command(dt)
            control.writeOnce(env.velocity_command())

        env.set_hold_action()
        settle_elapsed = 0.0
        while settle_elapsed < args.settle_time and not env.is_stopped():
            _, period = control.readOnce()
            dt = period.to_sec()
            settle_elapsed += dt
            env.step_command(dt)
            control.writeOnce(env.velocity_command())

        final_command = env.zero_velocity_command()
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
    print("Final xyz:     ", format_vec(translation_from_pose(list(final_state.O_T_EE))))
    print("Final q:       ", format_vec(final_state.q))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
