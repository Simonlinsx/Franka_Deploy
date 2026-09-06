#!/usr/bin/env python3
"""Move an explicitly bare FR3 flange to the supervised default joint pose.

This commissioning utility never imports or connects the Inspire hand.  It
requires the live Franka state to report F_T_EE=identity and zero configured
end-effector dynamics before the first control handle is created.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


FLANGE_TOKEN = "FR3_FLANGE_EMPTY"
CLEAR_TOKEN = "FR3_WORKSPACE_CLEAR"
STOP_TOKEN = "FR3_STOP_READY"
DEFAULT_Q = np.asarray(
    [0.0, 0.0, 0.0, -np.pi / 2.0, 0.0, np.pi / 2.0, 0.0],
    dtype=np.float64,
)
COMMISSIONING_JOINT_LIMITS = np.asarray(
    [
        [-2.7437, 2.7437],
        [-1.7837, 1.7837],
        [-2.9007, 2.9007],
        [-3.0421, -0.1518],
        [-2.8065, 2.8065],
        [0.5445, 4.5169],
        [-3.0159, 3.0159],
    ],
    dtype=np.float64,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Move a bare/unmounted FR3 flange to q=[0,0,0,-pi/2,0,pi/2,0] "
            "at at most 0.05 rad/s. Inspire is never connected."
        )
    )
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument("--confirm-flange-empty", metavar=FLANGE_TOKEN)
    parser.add_argument("--confirm-workspace-clear", metavar=CLEAR_TOKEN)
    parser.add_argument("--confirm-stop-ready", metavar=STOP_TOKEN)
    return parser


def _require_confirmations(args: argparse.Namespace) -> None:
    expected = (
        ("--confirm-flange-empty", args.confirm_flange_empty, FLANGE_TOKEN),
        ("--confirm-workspace-clear", args.confirm_workspace_clear, CLEAR_TOKEN),
        ("--confirm-stop-ready", args.confirm_stop_ready, STOP_TOKEN),
    )
    missing = [
        "{} {}".format(flag, token)
        for flag, actual, token in expected
        if actual != token
    ]
    if missing:
        raise ValueError("exact confirmations required: " + "; ".join(missing))


def run_default(robot_ip: str) -> None:
    # Lazy import: no confirmation means no libfranka import or connection.
    from anydex_pipeline.franka_sequence_driver import (
        FrankaMotionLimits,
        FrankaSequenceDriver,
    )

    limits = FrankaMotionLimits(
        expected_F_T_EE=np.eye(4),
        expected_m_ee_kg=0.0,
        expected_F_x_Cee_m=np.zeros(3),
        expected_I_ee_kg_m2=np.zeros((3, 3)),
        joint_limits_rad=COMMISSIONING_JOINT_LIMITS,
        joint_limit_margin_rad=0.05,
        max_joint_speed_rad_s=0.05,
        max_joint_segment_rad=0.20,
        min_joint_duration_s=6.0,
        joint_arrival_tolerance_rad=0.02,
    )
    arm = FrankaSequenceDriver.connect(robot_ip, limits, enforce_realtime=False)
    failure = None
    try:
        initial = arm.robot.read_once()
        # Same fail-closed state gate used before each generated segment.
        arm._validate_state(initial, require_idle=True, enforce_success=False)
        print(
            "[Franka bare] initial_q={}".format(
                np.round(np.asarray(initial.q, dtype=np.float64), 6).tolist()
            )
        )
        print(
            "[Franka bare] target_q={} max_speed=0.05rad/s".format(
                np.round(DEFAULT_Q, 6).tolist()
            )
        )
        arm.move_joints(DEFAULT_Q)
        final = arm.robot.read_once()
        arm._validate_state(final, require_idle=True, enforce_success=False)
        error = float(
            np.max(np.abs(np.asarray(final.q, dtype=np.float64) - DEFAULT_Q))
        )
        if error > limits.joint_arrival_tolerance_rad:
            raise RuntimeError(
                "final joint error {:.6f}rad exceeds {:.6f}rad".format(
                    error, limits.joint_arrival_tolerance_rad
                )
            )
        print(
            "[Franka bare] default reached; final_q={} max_error={:.6f}rad".format(
                np.round(np.asarray(final.q, dtype=np.float64), 6).tolist(), error
            )
        )
    except BaseException as exc:
        failure = exc
    finally:
        try:
            arm.stop()
        except BaseException as stop_exc:
            if failure is None:
                failure = RuntimeError("Franka stop failed: {}".format(stop_exc))
            else:
                failure = RuntimeError(
                    "{}; STOP UNCONFIRMED: {}".format(failure, stop_exc)
                )
    if failure is not None:
        raise failure


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _require_confirmations(args)
        run_default(str(args.robot_ip))
    except KeyboardInterrupt:
        print("[Franka bare] interrupted; use the physical stop if motion remains", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print("[Franka bare] failed: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
