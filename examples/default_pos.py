#!/usr/bin/env python3
"""Move a Franka Research 3 to a specified joint configuration.

The script is conservative by default:
- dry-run unless --execute is passed
- rejects targets outside FR3 joint limits
- rejects large single-step moves unless --max-delta is raised explicitly
- uses a smooth cosine trajectory with a velocity limit
"""

import argparse
import math
import sys
from typing import Iterable, List


FR3_JOINT_LIMITS = [
    (-2.9007, 2.9007),
    (-1.8361, 1.8361),
    (-2.9007, 2.9007),
    (-3.0770, -0.1169),
    (-2.8763, 2.8763),
    (0.4398, 4.6216),
    (-3.0508, 3.0508),
]

PRESETS = {
    "gello-neutral": [0.0, 0.0, 0.0, -1.571, 0.0, 1.571, 0.0],
    "default": [0.0, 0.0, 0.0, -1.571, 0.0, 1.571, 0.0],
}



def parse_q(values: Iterable[str]) -> List[float]:
    q = [float(value) for value in values]
    if len(q) != 7:
        raise argparse.ArgumentTypeError("expected exactly 7 joint values")
    return q


def format_q(q: Iterable[float]) -> str:
    return "[" + ", ".join(f"{value:.4f}" for value in q) + "]"


def validate_target(q_target: List[float], margin: float) -> None:
    for i, (q, (lower, upper)) in enumerate(zip(q_target, FR3_JOINT_LIMITS), start=1):
        if q < lower + margin or q > upper - margin:
            raise ValueError(
                f"target joint {i}={q:.4f} is outside the safe range "
                f"[{lower + margin:.4f}, {upper - margin:.4f}]"
            )


def cosine_interpolate(q_start: List[float], q_target: List[float], alpha: float) -> List[float]:
    blend = 0.5 - 0.5 * math.cos(math.pi * alpha)
    return [q0 + blend * (q1 - q0) for q0, q1 in zip(q_start, q_target)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="172.16.0.2", help="Franka Control FCI IP address")
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--target",
        nargs=7,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="absolute target joint configuration in radians",
    )
    target_group.add_argument(
        "--relative",
        nargs=7,
        metavar=("D1", "D2", "D3", "D4", "D5", "D6", "D7"),
        help="relative joint delta in radians, applied to the current q",
    )
    target_group.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        help="named target joint configuration",
    )
    parser.add_argument(
        "--max-delta",
        type=float,
        default=0.35,
        help="maximum allowed absolute change for any joint in this run, default: 0.35 rad",
    )
    parser.add_argument(
        "--max-velocity",
        type=float,
        default=0.08,
        help="maximum approximate joint velocity, default: 0.08 rad/s",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=4.0,
        help="minimum trajectory duration, default: 4.0 s",
    )
    parser.add_argument(
        "--joint-limit-margin",
        type=float,
        default=0.02,
        help="required margin from FR3 joint limits, default: 0.02 rad",
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

    if args.max_delta <= 0.0:
        raise ValueError("--max-delta must be positive")
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
    q_start = list(state.q)

    if args.target is not None:
        q_target = parse_q(args.target)
    elif args.relative is not None:
        delta = parse_q(args.relative)
        q_target = [q + dq for q, dq in zip(q_start, delta)]
    else:
        q_target = PRESETS[args.preset]

    validate_target(q_target, args.joint_limit_margin)

    deltas = [q1 - q0 for q0, q1 in zip(q_start, q_target)]
    max_abs_delta = max(abs(delta) for delta in deltas)
    if max_abs_delta > args.max_delta:
        raise ValueError(
            f"largest joint change is {max_abs_delta:.4f} rad, exceeding "
            f"--max-delta={args.max_delta:.4f}. Move in smaller steps or explicitly raise "
            "--max-delta after checking the workspace."
        )

    # For q = q0 + 0.5 * (1 - cos(pi*t/T)) * dq, peak speed is pi/2 * |dq| / T.
    duration = max(args.min_duration, (math.pi / 2.0) * max_abs_delta / args.max_velocity)

    print("Current q: ", format_q(q_start))
    print("Target q:  ", format_q(q_target))
    print("Delta q:   ", format_q(deltas))
    print(f"Max delta: {max_abs_delta:.4f} rad")
    print(f"Duration:  {duration:.2f} s")

    if not args.execute:
        print("Dry-run only. Add --execute to move the robot.")
        return 0

    control = robot.start_joint_position_control(pylibfranka.ControllerMode.JointImpedance)
    elapsed = 0.0

    try:
        while elapsed < duration:
            _, period = control.readOnce()
            elapsed += period.to_sec()
            alpha = min(1.0, elapsed / duration)
            control.writeOnce(pylibfranka.JointPositions(cosine_interpolate(q_start, q_target, alpha)))

        final_command = pylibfranka.JointPositions(q_target)
        final_command.motion_finished = True
        control.writeOnce(final_command)
    except BaseException:
        try:
            robot.stop()
        except Exception:
            pass
        raise

    final_state = robot.read_once()
    print("Final q:   ", format_q(final_state.q))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())