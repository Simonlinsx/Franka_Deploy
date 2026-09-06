#!/usr/bin/env python3
"""Select the highest-scoring RH56/FR3-feasible AnyDex candidate offline.

The selector never opens a camera, serial port, or Franka connection.  It
filters the official candidate list by the configured RH56 register envelope,
then delegates exact fixed-q7 IK, joint-margin, dense path, and bare-arm
self-collision checks to ``plan_installed_air_candidate.sh``.  The first
passing candidate in descending score order is published as the joint plan.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Optional, Sequence
import uuid

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anydex_pipeline.air_audit_readiness import (  # noqa: E402
    validate_air_candidate_commissioning,
)
from anydex_pipeline.control_config import load_control_config  # noqa: E402
from anydex_pipeline.control_plan import is_official_snapshot  # noqa: E402
from anydex_pipeline.snapshot import load_snapshot_npz  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/fr3_rh56_v7_sim2real_supervised.json"
DEFAULT_Q7_RAD = 1.3525


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline score-ordered AnyDex candidate selection using RH56 "
            "ranges plus exact FR3 IK/joint/self-collision planning"
        )
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--q7-rad", type=float, default=DEFAULT_Q7_RAD)
    parser.add_argument(
        "--start-q-rad",
        type=float,
        nargs=7,
        default=None,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"),
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="maximum ranked candidates to try; 0 means all",
    )
    return parser


def _last_error(output: str) -> str:
    lines = [line.strip() for line in str(output).splitlines() if line.strip()]
    return lines[-1] if lines else "planner returned no diagnostic"


def select_candidate(
    args: argparse.Namespace,
    *,
    runner=subprocess.run,
) -> int:
    snapshot_path = args.snapshot.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError("output already exists: {}".format(output_path))
    config, resolved_config = load_control_config(config_path)
    snapshot = load_snapshot_npz(snapshot_path)
    if not is_official_snapshot(snapshot):
        raise ValueError("snapshot is not provenance-complete official AnyDexGrasp output")
    if snapshot.grasps.hand_angles is None or snapshot.grasps.hand_poses is None:
        raise ValueError("official snapshot has no RH56 pose/register candidates")
    if args.max_candidates < 0:
        raise ValueError("--max-candidates must be nonnegative")
    q_start = np.asarray(
        config["franka"]["default_q_rad"]
        if args.start_q_rad is None
        else args.start_q_rad,
        dtype=np.float64,
    )
    if q_start.shape != (7,) or not np.all(np.isfinite(q_start)):
        raise ValueError("start q must contain seven finite radians")

    order = sorted(
        range(snapshot.grasps.count),
        key=lambda index: (-float(snapshot.grasps.scores[index]), int(index)),
    )
    if args.max_candidates:
        order = order[: int(args.max_candidates)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    for rank, index in enumerate(order, start=1):
        score = float(snapshot.grasps.scores[index])
        targets = np.asarray(snapshot.grasps.hand_angles[index], dtype=np.float64)
        try:
            validate_air_candidate_commissioning(config, targets)
        except ValueError as exc:
            failures.append((index, "RH56: {}".format(exc)))
            print(
                "[auto-select] skip rank={} index={} score={:.6f}: {}".format(
                    rank, index, score, exc
                ),
                flush=True,
            )
            continue

        temporary = output_path.with_name(
            ".{}.candidate-{}.{}.tmp".format(
                output_path.name, index, uuid.uuid4().hex
            )
        )
        command = [
            str(ROOT / "scripts/plan_installed_air_candidate.sh"),
            "--snapshot", str(snapshot_path),
            "--config", str(resolved_config),
            "--candidate-index", str(index),
            "--q7-rad", "{:.17g}".format(float(args.q7_rad)),
            "--start-q-rad",
            *["{:.17g}".format(float(value)) for value in q_start],
            "--output", str(temporary),
        ]
        try:
            completed = runner(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if int(completed.returncode) != 0:
                reason = _last_error(completed.stdout)
                failures.append((index, "FR3: {}".format(reason)))
                print(
                    "[auto-select] skip rank={} index={} score={:.6f}: {}".format(
                        rank, index, score, reason
                    ),
                    flush=True,
                )
                continue
            if not temporary.is_file():
                raise RuntimeError("passing planner did not create its joint plan")
            try:
                os.link(temporary, output_path)
            except FileExistsError as exc:
                raise FileExistsError("output appeared during selection") from exc
            print(
                "[auto-select] PASS rank={} index={} score={:.6f} targets={}".format(
                    rank,
                    index,
                    score,
                    np.rint(targets).astype(int).tolist(),
                ),
                flush=True,
            )
            print("SELECTED_INDEX={}".format(index), flush=True)
            print("JOINT_PLAN={}".format(output_path), flush=True)
            return int(index)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    summary = "; ".join(
        "{}: {}".format(index, reason) for index, reason in failures[:8]
    )
    if len(failures) > 8:
        summary += "; ... {} more".format(len(failures) - 8)
    raise RuntimeError(
        "no candidate passed RH56 range + FR3 IK/joint/self-collision checks: "
        + summary
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        select_candidate(args)
        return 0
    except KeyboardInterrupt:
        print("[auto-select] interrupted; no hardware was opened", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print("[auto-select] failed: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
