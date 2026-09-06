#!/usr/bin/env python3
"""Run the official robot-free AnyDex backend on a saved PCD snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anydex_pipeline.official_pipeline import OfficialAnyDexBackend
from anydex_pipeline.snapshot import load_snapshot_npz, save_snapshot_npz
from anydex_pipeline.snapshot_bridge import make_snapshot
from anydex_pipeline.types import PointCloudObservation


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replace diagnostic poses in a snapshot with official AnyDexGrasp poses"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "weights/logs/model/checkpoint.tar.18",
    )
    parser.add_argument(
        "--inspire-model-dir",
        type=Path,
        default=ROOT / "weights/logs/model/inspire_model/obj140",
    )
    parser.add_argument(
        "--upstream-root", type=Path, default=ROOT / "third_party/AnyDexGrasp"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--decision-score-threshold", type=float, default=None)
    parser.add_argument("--trust-official-checkpoints", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    source = load_snapshot_npz(args.input)
    observation = PointCloudObservation(
        scene_points=source.scene_points,
        scene_colors=source.scene_colors,
        object_points=source.object_points,
        object_colors=source.object_colors,
        reference_frame=source.reference_frame,
        T_reference_camera=source.T_reference_camera,
        frame_id=source.frame_id,
        timestamp_s=source.timestamp_s,
        calibration_id=source.calibration_id,
        camera_serial=source.camera_serial,
    )
    backend = OfficialAnyDexBackend(
        checkpoint_path=args.checkpoint,
        upstream_root=args.upstream_root,
        inspire_model_dir=args.inspire_model_dir,
        device=args.device,
        top_k=args.top_k,
        decision_score_threshold=args.decision_score_threshold,
        trust_official_checkpoints=args.trust_official_checkpoints,
    )
    result = backend.infer(observation)
    if not result.candidates:
        raise RuntimeError("official AnyDexGrasp returned no candidate after filtering")
    snapshot = make_snapshot(
        observation, result, scene_excludes_object=source.scene_excludes_object
    )
    output = save_snapshot_npz(args.output, snapshot)
    print(
        f"[ok] official candidates={len(result.candidates)} "
        f"inference={result.inference_time_s:.3f}s output={output.resolve()}"
    )
    if args.show:
        from anydex_pipeline.visualization import VisualizationStyle, show_snapshot

        show_snapshot(
            snapshot,
            style=VisualizationStyle(show_selected_hand_frame=True),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
