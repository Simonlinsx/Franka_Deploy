#!/usr/bin/env python3
from __future__ import annotations

"""Move the FR3 end-effector orientation while preserving Cartesian position.

This is a diagnostic script for Cartesian pose control. It sends one slow,
minimum-jerk orientation trajectory instead of a 10 Hz policy stream.
"""

import argparse
import math
import sys
from typing import Iterable, List

MINIMUM_JERK_PEAK_VELOCITY_FACTOR = 1.875


def parse_vec3(values: Iterable[str]) -> List[float]:
    vec = [float(value) for value in values]
    if len(vec) != 3:
        raise argparse.ArgumentTypeError("expected exactly 3 values")
    return vec


def format_vec(values: Iterable[float], precision: int = 4) -> str:
    return "[" + ", ".join(f"{value:.{precision}f}" for value in values) + "]"


def translation_from_pose(O_T_EE: List[float]) -> List[float]:
    return [O_T_EE[12], O_T_EE[13], O_T_EE[14]]


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


def norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def scale_vec(values: List[float], scale: float) -> List[float]:
    return [value * scale for value in values]


def matmul3(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


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


def minimum_jerk_blend(alpha: float) -> float:
    alpha = min(1.0, max(0.0, alpha))
    return alpha * alpha * alpha * (10.0 + alpha * (-15.0 + 6.0 * alpha))


def interpolate_rotation(
    start: List[List[float]], target: List[List[float]], alpha: float
) -> List[List[float]]:
    relative = matmul3(target, transpose3(start))
    rotvec = matrix_to_rotvec(relative)
    return matmul3(rotvec_to_matrix(scale_vec(rotvec, minimum_jerk_blend(alpha))), start)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="172.16.0.2", help="Franka Control FCI IP address")
    parser.add_argument(
        "--relative-rotvec",
        nargs=3,
        metavar=("DRX", "DRY", "DRZ"),
        required=True,
        help="relative EEF rotation vector in radians",
    )
    parser.add_argument(
        "--frame",
        choices=["base", "eef"],
        default="base",
        help="frame for --relative-rotvec",
    )
    parser.add_argument(
        "--max-angle",
        type=float,
        default=0.10,
        help="maximum allowed orientation change in this run, radians",
    )
    parser.add_argument(
        "--max-angular-velocity",
        type=float,
        default=0.03,
        help="maximum approximate angular velocity, rad/s",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=4.0,
        help="minimum trajectory duration, seconds",
    )
    parser.add_argument(
        "--startup-hold",
        type=float,
        default=0.50,
        help="seconds to stream the current pose before rotating",
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

    if args.max_angle <= 0.0:
        raise ValueError("--max-angle must be positive")
    if args.max_angular_velocity <= 0.0:
        raise ValueError("--max-angular-velocity must be positive")
    if args.min_duration <= 0.0:
        raise ValueError("--min-duration must be positive")
    if args.startup_hold < 0.0:
        raise ValueError("--startup-hold must be non-negative")

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
    state = robot.read_once()
    start_pose = list(state.O_T_EE)
    start_xyz = translation_from_pose(start_pose)
    start_rotation = rotation_from_pose(start_pose)
    delta_rotvec = parse_vec3(args.relative_rotvec)
    target_rotation = apply_delta_rotation(start_rotation, delta_rotvec, args.frame)
    angle = rotation_distance(start_rotation, target_rotation)

    if angle > args.max_angle:
        raise ValueError(
            f"orientation change is {angle:.4f} rad, exceeding "
            f"--max-angle={args.max_angle:.4f}. Use a smaller rotvec or raise "
            "--max-angle intentionally."
        )

    duration = max(
        args.min_duration,
        MINIMUM_JERK_PEAK_VELOCITY_FACTOR * angle / args.max_angular_velocity,
    )

    print("Current q:       ", format_vec(state.q))
    print("Current xyz:     ", format_vec(start_xyz))
    print("Delta rotvec:    ", format_vec(delta_rotvec))
    print(f"Frame:            {args.frame}")
    print(f"Angle:            {angle:.4f} rad")
    print(f"Max angular vel:  {args.max_angular_velocity:.4f} rad/s")
    print(f"Startup hold:     {args.startup_hold:.2f} s")
    print(f"Duration:         {duration:.2f} s")
    print("Translation:      preserving current O_T_EE translation")

    if not args.execute:
        print("Dry-run only. Add --execute to move the robot.")
        return 0

    control = robot.start_cartesian_pose_control(pylibfranka.ControllerMode.CartesianImpedance)
    elapsed = 0.0

    try:
        hold_elapsed = 0.0
        active_start_pose: List[float] | None = None
        while hold_elapsed <= args.startup_hold:
            active_state, period = control.readOnce()
            if active_start_pose is None:
                active_start_pose = list(active_state.O_T_EE)
                start_xyz = translation_from_pose(active_start_pose)
                start_rotation = rotation_from_pose(active_start_pose)
                target_rotation = apply_delta_rotation(start_rotation, delta_rotvec, args.frame)
            hold_elapsed += period.to_sec()
            control.writeOnce(
                pylibfranka.CartesianPose(
                    pose_with_rotation_translation(active_start_pose, start_rotation, start_xyz)
                )
            )

        while elapsed < duration:
            _, period = control.readOnce()
            elapsed += period.to_sec()
            alpha = min(1.0, elapsed / duration)
            command_rotation = interpolate_rotation(start_rotation, target_rotation, alpha)
            control.writeOnce(
                pylibfranka.CartesianPose(
                    pose_with_rotation_translation(active_start_pose, command_rotation, start_xyz)
                )
            )

        final_command = pylibfranka.CartesianPose(
            pose_with_rotation_translation(active_start_pose, target_rotation, start_xyz)
        )
        final_command.motion_finished = True
        control.writeOnce(final_command)
    except BaseException:
        try:
            robot.stop()
        except Exception:
            pass
        raise

    final_state = robot.read_once()
    print("Final xyz:       ", format_vec(translation_from_pose(list(final_state.O_T_EE))))
    print("Final q:         ", format_vec(final_state.q))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
