#!/usr/bin/env python3
"""Offline microbenchmark for the Franka 1 kHz validation/command path.

This script imports no pylibfranka module and creates no hardware connection.
It compares the full segment-boundary provenance check with the dynamic-only
validator used for every active-control sample.  Timing is diagnostic rather
than a pass/fail gate because Python/CPU scheduling differs across hosts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anydex_pipeline.franka_sequence_driver import (  # noqa: E402
    FrankaMotionLimits,
    FrankaSequenceDriver,
    _PreparedCartesianInterpolation,
)


class Errors:
    """Exercise the same native-bool strategy as pylibfranka.Errors."""

    __module__ = "pylibfranka._pylibfranka"

    def __bool__(self) -> bool:
        return False


def _franka_pose(transform: np.ndarray) -> list[float]:
    return np.asarray(transform, dtype=np.float64).reshape(16, order="F").tolist()


def _state() -> SimpleNamespace:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [0.50, 0.0, 0.30]
    return SimpleNamespace(
        robot_mode=SimpleNamespace(name="Move"),
        current_errors=Errors(),
        cartesian_contact=np.zeros(6),
        cartesian_collision=np.zeros(6),
        joint_contact=np.zeros(7),
        joint_collision=np.zeros(7),
        control_command_success_rate=1.0,
        q=np.asarray([0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]),
        O_T_EE=_franka_pose(pose),
        F_T_EE=_franka_pose(np.eye(4)),
        m_ee=0.607,
        m_load=0.0,
        m_total=0.607,
        F_x_Cee=np.asarray([0.0, 0.0, 0.076]),
        I_ee=np.diag([0.00151, 0.00169, 0.000442]).reshape(9, order="F"),
    )


def _measure(label: str, function, iterations: int) -> float:
    for _ in range(min(200, iterations)):
        function()
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    microseconds = (time.perf_counter() - started) * 1.0e6 / iterations
    print(f"{label:42s} {microseconds:9.3f} us/sample")
    return microseconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10_000)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    state = _state()
    limits = FrankaMotionLimits(
        expected_F_T_EE=np.eye(4),
        expected_m_ee_kg=0.607,
        expected_F_x_Cee_m=[0.0, 0.0, 0.076],
        expected_I_ee_kg_m2=np.diag([0.00151, 0.00169, 0.000442]),
        workspace_min_m=[0.30, -0.30, 0.10],
        workspace_max_m=[0.80, 0.30, 0.70],
    )
    driver = FrankaSequenceDriver(object(), object(), limits)
    realtime, _ = driver._prepare_realtime_validation(
        state, require_idle=False
    )

    start = np.eye(4, dtype=np.float64)
    start[:3, 3] = [0.50, 0.0, 0.30]
    target = start.copy()
    angle = 0.05
    target[:3, :3] = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target[:3, 3] = [0.51, 0.01, 0.29]
    prepared_pose = _PreparedCartesianInterpolation(start, target)
    pose_command_buffer = [0.0] * 16

    print("offline=True hardware_imported=False thresholds=0.95/0.80")
    full_us = _measure(
        "full boundary validator (pose/load/inertia)",
        lambda: driver._validate_state(state, require_idle=False),
        args.iterations,
    )
    realtime_us = _measure(
        "1 kHz dynamic validator",
        lambda: realtime.validate(state, require_idle=False),
        args.iterations,
    )
    pose_us = _measure(
        "prepared Cartesian command sampler",
        lambda: prepared_pose.fill_command_values(0.37, pose_command_buffer),
        args.iterations,
    )
    print(f"validator speedup={full_us / realtime_us:.2f}x")
    print(f"dynamic_validator_plus_pose={realtime_us + pose_us:.3f} us/sample")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
