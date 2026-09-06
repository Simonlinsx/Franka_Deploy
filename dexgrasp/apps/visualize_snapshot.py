#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anydex_pipeline.snapshot import load_snapshot_npz, save_snapshot_npz


def main() -> int:
    parser = argparse.ArgumentParser(description="Open a saved scene/object/grasp snapshot")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument(
        "--selected-index",
        type=int,
        default=None,
        help=(
            "Visualize this candidate index instead of the selection stored in "
            "the snapshot. The snapshot file is never modified."
        ),
    )
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=ROOT / "third_party/AnyDexGrasp",
        help="Official AnyDex checkout containing the Inspire URDF/STLs/mapping",
    )
    parser.add_argument(
        "--diagnostic-inspire-type",
        type=int,
        choices=range(1, 9),
        default=None,
        metavar="1..8",
        help=(
            "Attach a fixed Inspire semantic type when this old/geometric snapshot "
            "has no hand pose. This is diagnostic FK, not decision-model output."
        ),
    )
    parser.add_argument(
        "--hand-mesh-resolution",
        choices=("full", "simplified"),
        default="full",
        help="full displays official link STLs; simplified displays link AABBs",
    )
    parser.add_argument("--no-hand-mesh", action="store_true")
    parser.add_argument(
        "--save-enriched",
        type=Path,
        default=None,
        help="Optionally save a snapshot after diagnostic hand mapping",
    )
    args = parser.parse_args()
    snapshot = load_snapshot_npz(args.snapshot)
    if args.diagnostic_inspire_type is not None:
        from anydex_pipeline.inspire_commissioning import (
            attach_diagnostic_inspire_hand,
        )

        mapping_path = (
            args.upstream_root
            / "generate_mesh_and_pointcloud/inspire_urdf/width_12Dangle_6Dangle.json"
        )
        snapshot = attach_diagnostic_inspire_hand(
            snapshot,
            grasp_type_id=args.diagnostic_inspire_type,
            mapping=mapping_path,
        )
        print(
            "[hand][diagnostic] fixed Inspire type="
            f"{args.diagnostic_inspire_type}; this is not neural decision output"
        )
        if args.save_enriched is not None:
            saved = save_snapshot_npz(args.save_enriched, snapshot)
            print(f"[saved] enriched snapshot: {saved.resolve()}")
    stored_selected = int(snapshot.grasps.selected_index)
    if args.selected_index is not None:
        selected = int(args.selected_index)
        if selected < 0 or selected >= snapshot.grasps.count:
            parser.error(
                "--selected-index must be within 0.."
                f"{snapshot.grasps.count - 1}, got {selected}"
            )
        snapshot = replace(
            snapshot,
            grasps=replace(snapshot.grasps, selected_index=selected),
        )
        print(
            "[selection] in-memory override: "
            f"stored={stored_selected} visualized={selected}; snapshot unchanged"
        )
    print(
        f"[snapshot] frame={snapshot.frame_id} reference={snapshot.reference_frame} "
        f"scene={len(snapshot.scene_points)} object={len(snapshot.object_points)} "
        f"grasps={snapshot.grasps.count}"
    )
    builder = None
    if not args.no_hand_mesh and snapshot.grasps.hand_poses is not None:
        from anydex_pipeline.inspire_hand_model import (
            SelectedInspireHandLinkMeshBuilder,
        )

        builder = SelectedInspireHandLinkMeshBuilder(
            args.upstream_root,
            mesh_resolution=args.hand_mesh_resolution,
        )
        selected = int(snapshot.grasps.selected_index)
        grasp_type = int(snapshot.grasps.type_ids[selected])
        width = float(snapshot.grasps.widths_m[selected])
        print(
            f"[hand] rendering 13 official Inspire URDF link meshes; "
            f"selected={selected} type={grasp_type} width={width:.4f}m"
        )

    from anydex_pipeline.visualization import VisualizationStyle, show_snapshot

    show_snapshot(
        snapshot,
        window_name=(
            "AnyDexGrasp | scene + object + grasp + Inspire links | "
            f"candidate {snapshot.grasps.selected_index} | {args.snapshot.name}"
        ),
        style=VisualizationStyle(show_selected_hand_frame=True),
        selected_hand_link_mesh_builder=builder,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
