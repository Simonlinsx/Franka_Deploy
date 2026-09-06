#!/usr/bin/env python3
"""Read one Franka Research 3 robot state via FCI.

This example is intentionally read-only. It connects to the robot, reads one
state sample, and prints joint positions plus the end-effector position.
"""

import argparse
import math
import sys


def _translation(values):
    return [float(values[12]), float(values[13]), float(values[14])]


def _translation_error(first, second):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ip",
        default="172.16.0.2",
        help="FCI IP address of the Franka Control unit, default: 172.16.0.2",
    )
    parser.add_argument(
        "--enforce-realtime",
        action="store_true",
        help="Require real-time scheduling. Use this only after RT kernel setup.",
    )
    args = parser.parse_args()

    try:
        import pylibfranka
    except ImportError:
        print(
            "pylibfranka is not installed. Install libfranka/pylibfranka first.",
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

    print("Connected to Franka robot")
    print(f"q: {list(state.q)}")
    print(
        "O_T_EE translation: "
        f"x={state.O_T_EE[12]:.4f}, y={state.O_T_EE[13]:.4f}, z={state.O_T_EE[14]:.4f}"
    )
    commanded_pose = getattr(state, "O_T_EE_c", None)
    if commanded_pose is not None:
        measured_xyz = _translation(state.O_T_EE)
        commanded_xyz = _translation(commanded_pose)
        print(
            "O_T_EE_c translation: "
            f"x={commanded_xyz[0]:.4f}, y={commanded_xyz[1]:.4f}, "
            f"z={commanded_xyz[2]:.4f}"
        )
        print(
            "measured-commanded translation error: "
            f"{_translation_error(measured_xyz, commanded_xyz) * 1000.0:.3f} mm"
        )
    desired_pose = getattr(state, "O_T_EE_d", None)
    if desired_pose is not None:
        measured_xyz = _translation(state.O_T_EE)
        desired_xyz = _translation(desired_pose)
        print(
            "O_T_EE_d translation: "
            f"x={desired_xyz[0]:.4f}, y={desired_xyz[1]:.4f}, "
            f"z={desired_xyz[2]:.4f}"
        )
        print(
            "measured-desired translation error: "
            f"{_translation_error(measured_xyz, desired_xyz) * 1000.0:.3f} mm"
        )
    print(f"dq: {list(state.dq)}")
    print(f"control_command_success_rate: {state.control_command_success_rate:.9f}")
    print(
        "configured dynamics: "
        f"m_ee={state.m_ee:.6f} kg, m_load={state.m_load:.6f} kg, "
        f"m_total={state.m_total:.6f} kg"
    )
    print(f"F_x_Cee: {list(state.F_x_Cee)}")
    print(f"I_ee: {list(state.I_ee)}")
    print(f"F_T_EE: {list(state.F_T_EE)}")
    print(f"F_x_Cload: {list(state.F_x_Cload)}")
    print(f"I_load: {list(state.I_load)}")
    print(f"F_x_Ctotal: {list(state.F_x_Ctotal)}")
    print(f"I_total: {list(state.I_total)}")
    print(f"robot_mode: {state.robot_mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
