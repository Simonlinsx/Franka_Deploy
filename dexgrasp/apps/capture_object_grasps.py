#!/usr/bin/env python3
"""Capture one calibrated object cloud, infer grasps, and visualize them.

This entry point is perception-only.  It has no imports from Franka or Inspire
control packages and contains no robot/serial command-line options.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
SRC = ROOT / "src"
DYNAMIC_PCD_ROOT = WORKSPACE / "perception"
for path in (str(SRC), str(DYNAMIC_PCD_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from anydex_pipeline.backends.geometric import GeometricGraspBackend
from anydex_pipeline.snapshot import save_snapshot_npz
from anydex_pipeline.snapshot_bridge import make_snapshot
from anydex_pipeline.types import PointCloudObservation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture scene + segmented object point clouds, infer grasp poses, "
            "and overlay them in Open3D. No robot or hand is controlled."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DYNAMIC_PCD_ROOT / "configs/d435_default.yaml",
        help="Calibrated dynamic_object_pcd configuration",
    )
    parser.add_argument(
        "--backend",
        choices=("geometric", "official"),
        default="geometric",
        help="official requires the legacy AnyDex CUDA environment and checkpoint",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--inspire-model-dir", type=Path, default=None)
    parser.add_argument("--upstream-root", type=Path, default=ROOT / "third_party/AnyDexGrasp")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--trust-official-checkpoints",
        action="store_true",
        help="Allow pickle-backed torch.load only for checkpoints from the official folder",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--geometric-inspire-type",
        type=int,
        choices=range(1, 9),
        default=4,
        metavar="1..8",
        help=(
            "Fixed diagnostic hand type for geometric backend (default 4, "
            "Large_Diameter); it is not neural decision-model output"
        ),
    )
    parser.add_argument(
        "--hand-mesh-resolution",
        choices=("full", "simplified"),
        default="full",
    )
    parser.add_argument("--no-hand-mesh", action="store_true")
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Numeric ROI; omit to select the object interactively",
    )
    parser.add_argument("--scene-stride", type=int, default=4)
    parser.add_argument("--settle-valid-frames", type=int, default=5)
    parser.add_argument("--capture-timeout-s", type=float, default=20.0)
    parser.add_argument(
        "--sam2",
        action="store_true",
        help="Use SAM2 initialization from the dynamic point-cloud config",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Snapshot NPZ path (default: runs/grasp_snapshot_<time>.npz)",
    )
    parser.add_argument("--no-viewer", action="store_true")
    return parser


def create_backend(args):
    if args.backend == "geometric":
        return GeometricGraspBackend(top_k=args.top_k)
    if args.checkpoint is None:
        raise ValueError("--backend official requires --checkpoint")
    from anydex_pipeline.official_pipeline import OfficialAnyDexBackend

    return OfficialAnyDexBackend(
        checkpoint_path=args.checkpoint,
        upstream_root=args.upstream_root,
        device=args.device,
        top_k=args.top_k,
        inspire_model_dir=args.inspire_model_dir,
        trust_official_checkpoints=args.trust_official_checkpoints,
    )


def capture_observation(args) -> PointCloudObservation:
    from dynamic_pcd.config import load_config
    from dynamic_pcd.provider.object_pcd_provider import ObjectPCDProvider

    config = load_config(str(args.config.resolve()))
    config.setdefault("sam2", {})["enabled"] = bool(args.sam2)
    provider = ObjectPCDProvider(config)
    provider.start()
    try:
        if args.roi is None:
            initialized = provider.select_and_initialize()
        else:
            frame = provider.camera.get_frame()
            initialized = provider.initialize_from_bbox(
                frame, np.asarray(args.roi, dtype=np.int32)
            )
        if not initialized:
            raise RuntimeError("object ROI initialization failed or was cancelled")

        valid_count = 0
        deadline = time.monotonic() + args.capture_timeout_s
        last = None
        while time.monotonic() < deadline:
            frame, mask, obj, packet = provider.step()
            if mask.valid and obj.valid and packet.valid and len(obj.points) >= 20:
                valid_count += 1
                last = (frame, mask, obj, packet)
                if valid_count >= args.settle_valid_frames:
                    break
            else:
                valid_count = 0
        if last is None or valid_count < args.settle_valid_frames:
            raise RuntimeError(
                "timed out waiting for a stable valid object point cloud; "
                "reselect a tighter ROI or check depth coverage"
            )

        frame, mask, obj, packet = last
        scene = provider.extractor.extract_scene(
            frame,
            exclude_mask=mask.mask,
            stride=max(1, int(args.scene_stride)),
        )
        print(
            "[capture] "
            f"frame={packet.frame_id} object_points={len(obj.points)} "
            f"scene_points={len(scene.points)} frame={packet.reference_frame} "
            f"calibration_id={packet.calibration_id}"
        )
        return PointCloudObservation(
            scene_points=scene.points,
            scene_colors=scene.colors,
            object_points=obj.points,
            object_colors=obj.colors,
            reference_frame=packet.reference_frame,
            T_reference_camera=provider.extrinsics.T_base_camera,
            frame_id=packet.frame_id,
            timestamp_s=packet.timestamp,
            calibration_id=packet.calibration_id or "",
            camera_serial=packet.camera_serial or "",
        )
    finally:
        provider.stop()


def main() -> int:
    args = build_parser().parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be >= 1")
    if args.scene_stride < 1 or args.settle_valid_frames < 1:
        raise SystemExit("--scene-stride and --settle-valid-frames must be >= 1")
    if args.capture_timeout_s <= 0:
        raise SystemExit("--capture-timeout-s must be positive")

    print("[safety] perception-only mode: no Franka/Inspire transport is imported or opened")
    observation = capture_observation(args)
    backend = create_backend(args)
    result = backend.infer(observation)
    if not result.candidates:
        raise RuntimeError("the backend returned no grasp candidates")
    snapshot = make_snapshot(observation, result, scene_excludes_object=True)
    if args.backend == "geometric":
        from anydex_pipeline.inspire_commissioning import (
            attach_diagnostic_inspire_hand,
        )

        mapping_path = (
            args.upstream_root
            / "generate_mesh_and_pointcloud/inspire_urdf/width_12Dangle_6Dangle.json"
        )
        snapshot = attach_diagnostic_inspire_hand(
            snapshot,
            grasp_type_id=args.geometric_inspire_type,
            mapping=mapping_path,
        )
        print(
            "[hand][diagnostic] geometric backend uses fixed Inspire type="
            f"{args.geometric_inspire_type}; this is not neural decision output"
        )

    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = ROOT / "runs" / f"grasp_snapshot_{stamp}.npz"
    output = save_snapshot_npz(output, snapshot)

    print(
        f"[grasp] backend={result.backend_name} candidates={len(result.candidates)} "
        f"inference={result.inference_time_s * 1000.0:.1f}ms selected={result.selected_index}"
    )
    for index, candidate in enumerate(result.candidates):
        position = candidate.T_reference_grasp[:3, 3]
        approach = candidate.T_reference_grasp[:3, 0]
        displayed_type = int(snapshot.grasps.type_ids[index])
        displayed_width = float(snapshot.grasps.widths_m[index])
        print(
            f"  #{index:02d} score={candidate.score:.4f} "
            f"xyz={np.array2string(position, precision=4)} "
            f"approach(+X)={np.array2string(approach, precision=4)} "
            f"width={displayed_width:.4f}m type={displayed_type}"
        )
    print(f"[saved] {output.resolve()}")

    if not args.no_viewer:
        from anydex_pipeline.visualization import VisualizationStyle, show_snapshot

        hand_builder = None
        if not args.no_hand_mesh and snapshot.grasps.hand_poses is not None:
            from anydex_pipeline.inspire_hand_model import (
                SelectedInspireHandLinkMeshBuilder,
            )

            hand_builder = SelectedInspireHandLinkMeshBuilder(
                args.upstream_root,
                mesh_resolution=args.hand_mesh_resolution,
            )

        show_snapshot(
            snapshot,
            window_name="AnyDexGrasp | scene + object + grasp + Inspire links",
            style=VisualizationStyle(show_selected_hand_frame=True),
            selected_hand_link_mesh_builder=hand_builder,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
