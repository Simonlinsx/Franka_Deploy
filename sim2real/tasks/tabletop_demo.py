#!/usr/bin/env python3
"""Launch a 20 Hz motion-triggered tabletop intercept through V94 runtime."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Optional, Sequence

from sim2real.replay_actions import main as replay_main
from .launcher import materialize_task_config

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PLAN_ROOT = WORKSPACE_ROOT / "dexgrasp/runs/replay_inputs/tabletop_online_planner_v4"
CHECKPOINT_ROOT = (
    WORKSPACE_ROOT
    / "data/checkpoints/table/inspire_directional_demo_v364_v371_speed600_20hz_noflow_20260816"
    / "inspire_directional_demo_v364_v371_speed600_20hz_noflow_20260816"
    / "runtime_checkpoints"
)
DEFAULT_PROFILE = (
    WORKSPACE_ROOT / "dexgrasp/configs/fr3_rh56_v94_seq286_20hz_commissioned.json"
)
SCENARIOS = {
    "preposition-check": (
        "preposition_check.zip",
        "large pink foam cylinder",
        None,
    ),
    "cylinder-posy-low": (
        "cylinder_posy_low.zip",
        "large pink foam cylinder",
        "v364_cylinder_+Y_0.02-0.20mps_inference.pt",
    ),
    "cylinder-negy-low": (
        "cylinder_negy_low.zip",
        "large pink foam cylinder",
        "v365_cylinder_-Y_0.02-0.20mps_inference.pt",
    ),
    "cylinder-posy-high": (
        "cylinder_posy_high.zip",
        "large pink foam cylinder",
        "v366_cylinder_+Y_0.20-0.40mps_inference.pt",
    ),
    "cylinder-negy-high": (
        "cylinder_negy_high.zip",
        "red foam cylinder",
        "v367_cylinder_-Y_0.20-0.40mps_inference.pt",
    ),
    "sphere-posy-low": (
        "sphere_posy_low.zip",
        "orange ball",
        "v368_sphere_+Y_0.02-0.20mps_inference.pt",
    ),
    "sphere-negy-low": (
        "sphere_negy_low.zip",
        "orange ball",
        "v369_sphere_-Y_0.02-0.20mps_inference.pt",
    ),
    "sphere-posy-high": (
        "sphere_posy_high.zip",
        "orange ball",
        "v370_sphere_+Y_0.20-0.40mps_inference.pt",
    ),
    "sphere-negy-high": (
        "sphere_negy_high.zip",
        "small ball",
        "v371_sphere_-Y_0.20-0.40mps_inference.pt",
    ),
    "sphere-posy-board-collision": (
        "sphere_posy_board_collision.zip",
        "orange ball",
        "v368_sphere_+Y_0.02-0.20mps_inference.pt",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=tuple(SCENARIOS))
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--object-text", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--yes-i-am-supervising", action="store_true")
    parser.add_argument("--live-visualization", action="store_true")
    parser.add_argument("--verbose-console", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    filename, default_text, checkpoint_filename = SCENARIOS[args.scenario]
    plan = (PLAN_ROOT / filename).resolve()
    if not plan.is_file():
        raise SystemExit(
            "tabletop motion plans are missing; run "
            "`.venv/bin/python -m sim2real.diagnostics.build_tabletop_intercept_replays`"
        )
    config, _ = materialize_task_config("tabletop")
    run_id = args.run_id or (
        "tabletop-plan-"
        + args.scenario
        + "-"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    video = WORKSPACE_ROOT / "dexgrasp/runs" / f"{run_id}.mp4"
    forwarded = [
        "--actions",
        str(plan),
        "--policy-rate-hz",
        "20",
        "--profile",
        str(args.profile.expanduser().resolve()),
        "--pcd-config",
        str(config),
        "--policy-rgbd-resolution",
        "424x240",
        "--object-mask-mode",
        "guarded_v2",
        "--object-text",
        str(args.object_text or default_text),
        "--run-id",
        run_id,
        "--record-video",
        str(video),
    ]
    if checkpoint_filename is not None:
        checkpoint = (CHECKPOINT_ROOT / checkpoint_filename).resolve()
        if not checkpoint.is_file():
            raise SystemExit(f"tabletop checkpoint is missing: {checkpoint}")
        forwarded.extend(["--checkpoint", str(checkpoint)])
    if args.live_visualization:
        forwarded.extend(["--live-visualization", "--live-visualization-rate-hz", "10"])
    if args.verbose_console:
        forwarded.append("--verbose-console")
    if args.execute:
        if not args.yes_i_am_supervising:
            raise SystemExit(
                "--execute also requires --yes-i-am-supervising"
            )
        forwarded.extend(["--execute", "--yes-i-am-supervising"])
    return replay_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
