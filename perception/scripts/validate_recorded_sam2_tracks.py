#!/usr/bin/env python3
"""Validate selected 60 Hz throw tracks through SAM2 at policy cadence.

This is an offline, camera-only diagnostic.  It never opens Franka or RH56.
Each ``--track`` value is ``NAME:START:END:X1:Y1:X2:Y2`` where START is the
recording index carrying the automatic grounding bbox.  Frames are sampled at
``--stride`` (three by default: 60 Hz camera -> 20 Hz policy cadence).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (ROOT, WORKSPACE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamic_pcd.camera.recorded_rgbd_camera import RecordedRGBDCase
from dynamic_pcd.segmentation.sam2_video_runtime import SAM2VideoServiceManager
from dynamic_pcd.utils.geometry import bbox_from_mask, draw_mask_overlay
from sim2real.observation.model import MaskedRGBDProjector


def _track_spec(value: str) -> tuple[str, int, int, np.ndarray]:
    parts = value.split(":")
    if len(parts) != 7:
        raise argparse.ArgumentTypeError(
            "track must be NAME:START:END:X1:Y1:X2:Y2"
        )
    name = parts[0].strip()
    try:
        start, end, x1, y1, x2, y2 = (int(v) for v in parts[1:])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("track fields after NAME must be integers") from exc
    if not name or start < 0 or end < start or x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError("track has invalid name/range/bbox")
    return name, start, end, np.asarray([x1, y1, x2, y2], dtype=np.int32)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track", type=_track_spec, action="append", required=True)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--request-timeout-ms", type=int, default=2000)
    return parser


def _percentile(values: list[float], q: float) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return None if not finite else float(np.percentile(finite, q))


def main() -> int:
    args = _parser().parse_args()
    case = RecordedRGBDCase(args.case_dir)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    output.mkdir(parents=True)

    projector = MaskedRGBDProjector(
        camera_K=np.asarray(case.manifest["camera_K"], dtype=np.float64),
        T_base_camera_optical=np.asarray(
            case.manifest["T_base_camera"], dtype=np.float64
        ),
        image_size=(case.width, case.height),
        depth_range_m=(0.25, 1.20),
        num_points=128,
        minimum_valid_points=16,
    )
    manager = SAM2VideoServiceManager(
        request_timeout_ms=args.request_timeout_ms,
        startup_timeout_s=60.0,
    )
    summaries: dict[str, dict[str, object]] = {}
    all_rows: list[dict[str, object]] = []
    try:
        health = manager.start()
        for name, start, end, bbox in args.track:
            if end >= len(case):
                raise ValueError(f"track {name!r} end is outside recording")
            manager.reset()
            track_dir = output / name
            track_dir.mkdir()
            mask_dir = track_dir / "mask"
            mask_dir.mkdir()
            writer = cv2.VideoWriter(
                str(track_dir / "mask_overlay.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(case.manifest["nominal_fps"]) / float(args.stride),
                (case.width, case.height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"cannot create overlay for {name}")
            rows: list[dict[str, object]] = []
            try:
                for ordinal, index in enumerate(range(start, end + 1, args.stride)):
                    frame = case.frame(index)
                    before = time.perf_counter()
                    result = (
                        manager.initialize_box(frame.color_bgr, bbox, frame.frame_id)
                        if ordinal == 0
                        else manager.track(frame.color_bgr, frame.frame_id)
                    )
                    elapsed_ms = (time.perf_counter() - before) * 1000.0
                    mask = (
                        np.asarray(result.mask, dtype=bool)
                        if result.valid
                        else np.zeros((case.height, case.width), dtype=bool)
                    )
                    area = int(np.count_nonzero(mask))
                    detected_bbox = bbox_from_mask(mask.astype(np.uint8))
                    mask_bbox = (
                        np.zeros(4, dtype=np.int32)
                        if detected_bbox is None
                        else np.asarray(detected_bbox, dtype=np.int32)
                    )
                    depth_m = frame.depth_raw.astype(np.float64) * frame.depth_scale
                    selected_depth = depth_m[mask]
                    nonzero = selected_depth[selected_depth > 0.0]
                    depth_nonzero_fraction = (
                        0.0 if area == 0 else float(nonzero.size / area)
                    )
                    in_production = (
                        (selected_depth >= 0.25) & (selected_depth <= 1.20)
                    )
                    in_diagnostic = (
                        (selected_depth >= 0.25) & (selected_depth <= 2.00)
                    )
                    production_fraction = (
                        0.0 if area == 0 else float(np.mean(in_production))
                    )
                    diagnostic_fraction = (
                        0.0 if area == 0 else float(np.mean(in_diagnostic))
                    )
                    point_frame = projector.project(
                        color_bgr=frame.color_bgr,
                        depth_raw=frame.depth_raw,
                        depth_scale_m_per_unit=frame.depth_scale,
                        object_mask=mask,
                        T_base_palm_at_capture=np.eye(4, dtype=np.float64),
                        captured_at_s=frame.timestamp,
                        frame_id=frame.frame_id,
                    )
                    row = {
                        "track": name,
                        "index": index,
                        "frame_id": int(frame.frame_id),
                        "valid": bool(result.valid),
                        "mask_area_px": area,
                        "bbox_xyxy": list(map(int, mask_bbox)),
                        "sam2_ms": float(elapsed_ms),
                        "depth_nonzero_fraction": depth_nonzero_fraction,
                        "depth_0p25_1p2_fraction": production_fraction,
                        "depth_0p25_2p0_fraction": diagnostic_fraction,
                        "depth_p50_m": (
                            None if nonzero.size == 0 else float(np.median(nonzero))
                        ),
                        "policy128_status": str(point_frame.status),
                        "policy128_source_points": int(point_frame.source_valid_points),
                        "message": str(result.message),
                    }
                    rows.append(row)
                    all_rows.append(row)
                    mask_path = mask_dir / f"{index:06d}.png"
                    if not cv2.imwrite(
                        str(mask_path), mask.astype(np.uint8) * np.uint8(255)
                    ):
                        raise RuntimeError(f"cannot write {mask_path}")
                    overlay = draw_mask_overlay(
                        frame.color_bgr,
                        mask.astype(np.uint8),
                        np.asarray(mask_bbox, dtype=np.int32),
                        text=(
                            f"{name} idx={index} valid={result.valid} "
                            f"area={area} pcd={point_frame.status}"
                        ),
                    )
                    writer.write(overlay)
            finally:
                writer.release()

            valid = [bool(row["valid"]) for row in rows]
            fresh = [row["policy128_status"] == "fresh" for row in rows]
            sam2_ms = [float(row["sam2_ms"]) for row in rows]
            summaries[name] = {
                "start_index": start,
                "end_index": end,
                "stride": args.stride,
                "frames": len(rows),
                "initialization_bbox_xyxy": bbox.tolist(),
                "mask_valid_fraction": float(np.mean(valid)),
                "policy128_fresh_fraction": float(np.mean(fresh)),
                "sam2_ms_p50": _percentile(sam2_ms, 50),
                "sam2_ms_p95": _percentile(sam2_ms, 95),
                "sam2_ms_max": _percentile(sam2_ms, 100),
                "depth_0p25_1p2_fraction_p50": _percentile(
                    [float(row["depth_0p25_1p2_fraction"]) for row in rows], 50
                ),
                "depth_0p25_2p0_fraction_p50": _percentile(
                    [float(row["depth_0p25_2p0_fraction"]) for row in rows], 50
                ),
                "diagnostic_depth_0p25_2p0_raw_min16_fraction": float(
                    np.mean(
                        [
                            int(row["mask_area_px"])
                            * float(row["depth_0p25_2p0_fraction"])
                            >= 16.0
                            for row in rows
                            if bool(row["valid"])
                        ]
                    )
                ),
            }
        payload = {
            "schema": "recorded_sam2_throw_validation_v1",
            "source_case": str(case.path),
            "camera_serial": str(case.manifest.get("camera_serial", "")),
            "nominal_fps": float(case.manifest["nominal_fps"]),
            "policy_stride": int(args.stride),
            "sam2_health": health,
            "hardware_interfaces_opened": False,
            "tracks": summaries,
        }
        (output / "summary.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with (output / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = list(all_rows[0]) if all_rows else []
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        manager.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
