#!/usr/bin/env python3
"""Create a deterministic no-camera snapshot for installation smoke tests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anydex_pipeline.backends.geometric import GeometricGraspBackend
from anydex_pipeline.snapshot import save_snapshot_npz
from anydex_pipeline.snapshot_bridge import make_snapshot
from anydex_pipeline.types import PointCloudObservation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "runs/synthetic_demo.npz")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(7)
    obj = rng.uniform((-0.04, -0.025, 0.0), (0.04, 0.025, 0.06), (2500, 3))
    obj += np.asarray((0.52, 0.02, 0.12))
    table_xy = rng.uniform((0.25, -0.35), (0.80, 0.35), (9000, 2))
    table = np.column_stack((table_xy, np.full(len(table_xy), 0.10)))
    observation = PointCloudObservation(
        scene_points=table,
        scene_colors=np.full((len(table), 3), (0.38, 0.42, 0.45), dtype=np.float32),
        object_points=obj,
        object_colors=np.full((len(obj), 3), (0.9, 0.25, 0.08), dtype=np.float32),
        reference_frame="robot_base",
        T_reference_camera=np.eye(4),
        frame_id=0,
    )
    result = GeometricGraspBackend(top_k=8).infer(observation)
    snapshot = make_snapshot(observation, result)
    output = save_snapshot_npz(args.output, snapshot)
    print(f"[ok] wrote {output.resolve()} with {snapshot.grasps.count} candidates")
    if args.show:
        from anydex_pipeline.visualization import show_snapshot

        show_snapshot(snapshot, window_name="AnyDexGrasp synthetic smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
