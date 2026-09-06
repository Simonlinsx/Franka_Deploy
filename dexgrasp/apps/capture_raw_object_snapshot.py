#!/usr/bin/env python3
"""Capture one calibrated D435 object/scene snapshot without inference.

This stage is deliberately camera-only.  It imports no Franka or Inspire
transport and stores an empty grasp set for the separate official environment
to populate later.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (ROOT / "src", WORKSPACE / "perception", ROOT / "apps"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from anydex_pipeline.snapshot import (  # noqa: E402
    GraspCandidates,
    VisualizationSnapshot,
    save_snapshot_npz,
)
from capture_object_grasps import capture_observation  # noqa: E402


DEFAULT_CAMERA_CONFIG = WORKSPACE / "perception/configs/d435_default.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture calibrated scene/object point clouds into a raw snapshot; "
            "no grasp inference and no robot/hand connection"
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="pixel ROI; omit for interactive selection",
    )
    parser.add_argument("--sam2", action="store_true")
    parser.add_argument("--scene-stride", type=int, default=4)
    parser.add_argument("--settle-valid-frames", type=int, default=5)
    parser.add_argument("--capture-timeout-s", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.config.expanduser().is_file():
        raise ValueError(f"camera config does not exist: {args.config}")
    if args.output.expanduser().suffix.lower() != ".npz":
        raise ValueError("--output must end in .npz")
    if args.output.expanduser().exists() and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output}")
    if args.scene_stride < 1 or args.settle_valid_frames < 1:
        raise ValueError("--scene-stride and --settle-valid-frames must be >= 1")
    if not np.isfinite(args.capture_timeout_s) or args.capture_timeout_s <= 0:
        raise ValueError("--capture-timeout-s must be finite and positive")
    if args.roi is not None:
        x1, y1, x2, y2 = (int(value) for value in args.roi)
        if min(x1, y1, x2, y2) < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("--roi requires non-negative X1 Y1 X2 Y2 with X2>X1,Y2>Y1")


def raw_snapshot_from_observation(observation) -> VisualizationSnapshot:
    empty_grasps = GraspCandidates(
        canonical_poses=np.empty((0, 4, 4), dtype=np.float64),
        scores=np.empty((0,), dtype=np.float32),
        selected_index=-1,
    )
    return VisualizationSnapshot(
        scene_points=observation.scene_points,
        scene_colors=observation.scene_colors,
        object_points=observation.object_points,
        object_colors=observation.object_colors,
        grasps=empty_grasps,
        reference_frame=observation.reference_frame,
        T_reference_camera=observation.T_reference_camera,
        frame_id=max(0, int(observation.frame_id)),
        timestamp_s=max(0.0, float(observation.timestamp_s)),
        calibration_id=observation.calibration_id,
        camera_serial=observation.camera_serial,
        scene_excludes_object=True,
        model_name="raw calibrated D435 object capture (no inference)",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_args(args)
        args.config = args.config.expanduser().resolve()
        args.output = args.output.expanduser()
        print(
            "[safety] camera-only capture; no Franka/Inspire transport is imported or opened"
        )
        observation = capture_observation(args)
        if len(observation.object_points) < 20:
            raise RuntimeError("raw capture contains fewer than 20 object points")
        snapshot = raw_snapshot_from_observation(observation)
        output = save_snapshot_npz(args.output.expanduser(), snapshot)
        print(
            "[raw snapshot] scene={} object={} frame_id={} reference={} camera={} calibration={}".format(
                len(snapshot.scene_points),
                len(snapshot.object_points),
                snapshot.frame_id,
                snapshot.reference_frame,
                snapshot.camera_serial,
                snapshot.calibration_id,
            )
        )
        print(f"[saved] {output.resolve()}")
        return 0
    except KeyboardInterrupt:
        print("[interrupted] camera capture stopped; no robot/hand was connected", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[failed] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
