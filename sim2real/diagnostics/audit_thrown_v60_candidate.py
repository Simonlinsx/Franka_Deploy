#!/usr/bin/env python3
"""Hardware-inert audit of the transferred V60 thrown-task candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from sim2real.deployment.bundle import DeployBundle, load_checkpoint_safely
from sim2real.deployment.runner import DEFAULT_BUNDLE
from sim2real.observation.camera_profile import resolve_runtime_camera_contract
from sim2real.tasks.launcher import materialize_task_config
from sim2real.tasks.thrown_contract import (
    assess_v57_camera_visibility,
    resolve_v57_thrown_task_contract,
)
from sim2real.contracts.v94 import V94Contract


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    WORKSPACE
    / "data/checkpoints/thrown/thrown_v60_palmcatch_dagger_milddr_e60_a050_candidate_20260814"
    / "thrown_v60_palmcatch_dagger_milddr_e60_a050_candidate_20260814"
    / "checkpoint.pt"
)
PINNED_CHECKPOINTS = {
    "f6ac6400c9a6d280942300705a168d77a5cf70fa6ce32eb26790a805c6e043e7": (
        "milddr-e60"
    ),
    "a3c56e90c81914eebd189e12c2cd4dfee73674ac2b05a4b5ed1b1b8c4447c666": (
        "r3-reach-e100"
    ),
}
CONSERVATIVE_JOINT_LIMITS = np.asarray(
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
RUNTIME_MOTION_GATE_LIMITS = np.asarray(
    [
        [-2.9007, 2.9007],
        [-1.8361, 1.8361],
        [-2.9007, 2.9007],
        [-3.0770, -0.1169],
        [-2.8763, 2.8763],
        [0.4398, 4.6216],
        [-3.0508, 3.0508],
    ],
    dtype=np.float64,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    return parser


def run(checkpoint_path: Path) -> dict[str, object]:
    checkpoint = checkpoint_path.expanduser().resolve()
    actual_sha = _sha256(checkpoint)
    candidate_name = PINNED_CHECKPOINTS.get(actual_sha)
    if candidate_name is None:
        raise ValueError("V60 checkpoint bytes differ from the pinned candidate")
    loaded = load_checkpoint_safely(checkpoint.read_bytes())
    config, metadata = materialize_task_config("thrown_object_v60")
    base = V94Contract.from_bundle(DeployBundle(DEFAULT_BUNDLE))
    camera = resolve_runtime_camera_contract(
        base.with_runtime_policy_rate_hz(20), config
    )
    task = resolve_v57_thrown_task_contract(config)
    if task is None:
        raise ValueError("materialized V60 task has no simulation contract")
    q_home = np.asarray(task.franka_q_home_rad, dtype=np.float64)
    conservative_inside = np.logical_and(
        q_home >= CONSERVATIVE_JOINT_LIMITS[:, 0],
        q_home <= CONSERVATIVE_JOINT_LIMITS[:, 1],
    )
    runtime_gate_inside = np.logical_and(
        q_home >= RUNTIME_MOTION_GATE_LIMITS[:, 0],
        q_home <= RUNTIME_MOTION_GATE_LIMITS[:, 1],
    )
    visibility = assess_v57_camera_visibility(task, camera)
    return {
        "schema": "thrown_v60_supervised_rollout_audit_v3",
        "checkpoint": str(checkpoint),
        "candidate_name": candidate_name,
        "checkpoint_sha256": actual_sha,
        "checkpoint_epoch": int(loaded.iteration),
        "checkpoint_control_dt_s": float(loaded.metadata["control_dt"]),
        "checkpoint_task": str(loaded.metadata.get("task", "")),
        "task_config": str(config),
        "task_config_status": metadata.get("commissioning_status"),
        "audit_hardware_writes": False,
        "robot_execution_enabled": metadata.get("robot_execution_enabled") is True,
        "maximum_supervised_execute_steps": metadata.get(
            "maximum_supervised_execute_steps"
        ),
        "franka_q_home_rad": q_home.tolist(),
        "q_home_inside_conservative_bundle_limits": bool(
            np.all(conservative_inside)
        ),
        "q_home_conservative_failed_joints_1_based": (
            np.flatnonzero(~conservative_inside) + 1
        ).tolist(),
        "q_home_inside_recorded_runtime_gate": bool(np.all(runtime_gate_inside)),
        "q_home_runtime_gate_min_margin_rad": float(
            np.min(
                np.minimum(
                    q_home - RUNTIME_MOTION_GATE_LIMITS[:, 0],
                    RUNTIME_MOTION_GATE_LIMITS[:, 1] - q_home,
                )
            )
        ),
        "camera_visibility": visibility.as_dict(),
        "single_tick_reset_and_execute_admitted": bool(
            metadata.get("robot_execution_enabled") is True
            and int(metadata.get("maximum_supervised_execute_steps", 0)) >= 1
        ),
        "full_rollout_admitted": bool(
            metadata.get("commissioning_status") == "accepted"
            and metadata.get("robot_execution_enabled") is True
            and metadata.get("maximum_supervised_execute_steps") == 72
        ),
        "full_rollout_horizon_ticks": 72,
        "full_rollout_blocking_reasons": [],
        "full_rollout_advisories": [
            "V60 alpha=0.5 global object support exceeds the pinned 1.50 m depth range",
            "current-frame policy points outside [0.25,1.50] m remain fail-closed",
            "checkpoint simulation success is low; admission is operator-supervised, not a performance guarantee",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        print(json.dumps(run(args.checkpoint), sort_keys=True, indent=2))
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"V60 candidate audit: FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
