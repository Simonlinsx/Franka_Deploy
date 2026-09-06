#!/usr/bin/env python3
"""Move the FR3 end-effector by Cartesian position while preserving orientation.

The command is conservative by default:
- dry-run unless --execute is passed
- preserves the current O_T_EE rotation
- rejects large Cartesian moves unless --max-distance is raised explicitly
- uses a smooth cosine trajectory with a translation speed limit

Coordinates are in the robot base frame O. For a normally mounted FR3, +z is up.
"""

import argparse
import math
import sys
from typing import Iterable, List


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


def cosine_interpolate(start: List[float], target: List[float], alpha: float) -> List[float]:
    blend = 0.5 - 0.5 * math.cos(math.pi * alpha)
    return [a + blend * (b - a) for a, b in zip(start, target)]


def norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="172.16.0.2", help="Franka Control FCI IP address")
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--relative",
        nargs=3,
        metavar=("DX", "DY", "DZ"),
        help="relative end-effector translation in base frame, meters",
    )
    target_group.add_argument(
        "--target",
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="absolute end-effector translation in base frame, meters",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=0.05,
        help="maximum allowed translation distance in this run, default: 0.05 m",
    )
    parser.add_argument(
        "--max-velocity",
        type=float,
        default=0.02,
        help="maximum approximate translation velocity, default: 0.02 m/s",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=3.0,
        help="minimum trajectory duration, default: 3.0 s",
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

    if args.max_distance <= 0.0:
        raise ValueError("--max-distance must be positive")
    if args.max_velocity <= 0.0:
        raise ValueError("--max-velocity must be positive")
    if args.min_duration <= 0.0:
        raise ValueError("--min-duration must be positive")

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

    if args.relative is not None:
        delta_xyz = parse_xyz(args.relative)
        target_xyz = [value + delta for value, delta in zip(start_xyz, delta_xyz)]
    else:
        target_xyz = parse_xyz(args.target)
        delta_xyz = [target - start for target, start in zip(target_xyz, start_xyz)]

    distance = norm(delta_xyz)
    if distance > args.max_distance:
        raise ValueError(
            f"translation distance is {distance:.4f} m, exceeding "
            f"--max-distance={args.max_distance:.4f}. Move in smaller steps or explicitly raise "
            "--max-distance after checking the workspace."
        )

    # For x = x0 + 0.5 * (1 - cos(pi*t/T)) * dx, peak speed is pi/2 * |dx| / T.
    duration = max(args.min_duration, (math.pi / 2.0) * distance / args.max_velocity)

    print("Current q:   ", format_vec(state.q))
    print("Current xyz: ", format_vec(start_xyz))
    print("Target xyz:  ", format_vec(target_xyz))
    print("Delta xyz:   ", format_vec(delta_xyz))
    print(f"Distance:    {distance:.4f} m")
    print(f"Duration:    {duration:.2f} s")
    print("Orientation: preserving current O_T_EE rotation")

    if not args.execute:
        print("Dry-run only. Add --execute to move the robot.")
        return 0

    control = robot.start_cartesian_pose_control(pylibfranka.ControllerMode.CartesianImpedance)
    elapsed = 0.0

    try:
        while elapsed < duration:
            _, period = control.readOnce()
            elapsed += period.to_sec()
            alpha = min(1.0, elapsed / duration)
            command_xyz = cosine_interpolate(start_xyz, target_xyz, alpha)
            control.writeOnce(pylibfranka.CartesianPose(pose_with_translation(start_pose, command_xyz)))

        final_command = pylibfranka.CartesianPose(pose_with_translation(start_pose, target_xyz))
        final_command.motion_finished = True
        control.writeOnce(final_command)
    except BaseException:
        try:
            robot.stop()
        except Exception:
            pass
        raise

    final_state = robot.read_once()
    print("Final xyz:   ", format_vec(translation_from_pose(list(final_state.O_T_EE))))
    print("Final q:     ", format_vec(final_state.q))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
