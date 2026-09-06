#!/usr/bin/env python3
"""Diagnose formal 128-point coverage across throw depth envelopes.

The input is one output directory from ``validate_recorded_throw_end_to_end``.
Only its current-frame masks and the matching lossless recorded RGB-D tuples
are consumed.  Franka and RH56 modules are intentionally not imported.

This tool is diagnostic: changing ``z_max`` here does not modify a deployment
configuration and cannot establish robot-base accuracy while the selected
camera profile calibration is still provisional.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (ROOT, WORKSPACE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamic_pcd.calibration import resolve_extrinsics  # noqa: E402
from dynamic_pcd.camera.recorded_rgbd_camera import RecordedRGBDCase  # noqa: E402
from dynamic_pcd.config import load_config  # noqa: E402
from sim2real.observation.model import MaskedRGBDProjector  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(values, q))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("acceptance_dir", type=Path)
    parser.add_argument("--z-max", type=float, action="append")
    parser.add_argument("--output", type=Path)
    return parser


def _projector(
    case: RecordedRGBDCase,
    T_base_camera: np.ndarray,
    *,
    z_min: float,
    z_max: float,
) -> MaskedRGBDProjector:
    intrinsics = case.intrinsics
    camera_K = np.asarray(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return MaskedRGBDProjector(
        camera_K=camera_K,
        T_base_camera_optical=np.asarray(T_base_camera, dtype=np.float64),
        image_size=(case.width, case.height),
        depth_range_m=(float(z_min), float(z_max)),
        num_points=128,
        minimum_valid_points=16,
        point_feature_dim=3,
        maximum_mask_depth_deviation_m=0.055,
    )


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no frame rows in {path}")
    return rows


def _mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != shape:
        raise ValueError(f"invalid current-frame mask {path}")
    return image > 0


def _audit_envelope(
    *,
    acceptance_dir: Path,
    case: RecordedRGBDCase,
    rows: list[dict[str, Any]],
    T_base_camera: np.ndarray,
    z_min: float,
    z_max: float,
) -> dict[str, Any]:
    per_track: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        per_track.setdefault(str(row["track"]), []).append(row)

    visible_count = fresh_count = 0
    source_points: list[float] = []
    centers: list[np.ndarray] = []
    frame_records: list[dict[str, Any]] = []
    tracks: dict[str, Any] = {}
    for track, track_rows in sorted(per_track.items()):
        projector = _projector(
            case,
            T_base_camera,
            z_min=z_min,
            z_max=z_max,
        )
        track_visible = track_fresh = 0
        track_source: list[float] = []
        for row in sorted(track_rows, key=lambda value: int(value["index"])):
            index = int(row["index"])
            reference_visible = str(row["reference_visible"]).lower() == "true"
            frame = case.frame(index)
            mask = _mask(
                acceptance_dir / track / "mask" / f"{index:06d}.png",
                frame.depth_raw.shape,
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
            if not reference_visible:
                continue
            visible_count += 1
            track_visible += 1
            source_points.append(float(point_frame.source_valid_points))
            track_source.append(float(point_frame.source_valid_points))
            frame_record = {
                "track": track,
                "index": index,
                "frame_id": int(frame.frame_id),
                "status": str(point_frame.status),
                "source_points": int(point_frame.source_valid_points),
                "center_robot_base_m": None,
            }
            if point_frame.status != "fresh":
                frame_records.append(frame_record)
                continue
            fresh_count += 1
            track_fresh += 1
            valid = np.asarray(point_frame.valid, dtype=bool)
            points = np.asarray(point_frame.xyzrgb_palm, dtype=np.float64)[valid, :3]
            if points.size:
                center = np.median(points, axis=0)
                centers.append(center)
                frame_record["center_robot_base_m"] = center.tolist()
            frame_records.append(frame_record)
        tracks[track] = {
            "reviewed_visible_frames": track_visible,
            "fresh_frames": track_fresh,
            "fresh_fraction": (
                0.0 if track_visible == 0 else track_fresh / float(track_visible)
            ),
            "source_points_p05": _percentile(track_source, 5),
            "source_points_p50": _percentile(track_source, 50),
        }

    center_array = (
        np.asarray(centers, dtype=np.float64)
        if centers
        else np.empty((0, 3), dtype=np.float64)
    )
    return {
        "z_min_m": float(z_min),
        "z_max_m": float(z_max),
        "reviewed_visible_frames": visible_count,
        "fresh_frames": fresh_count,
        "fresh_fraction": (
            0.0 if visible_count == 0 else fresh_count / float(visible_count)
        ),
        "source_points_p05": _percentile(source_points, 5),
        "source_points_p50": _percentile(source_points, 50),
        "fresh_center_robot_base_m": (
            None
            if center_array.size == 0
            else {
                "p05": np.percentile(center_array, 5, axis=0).tolist(),
                "p50": np.percentile(center_array, 50, axis=0).tolist(),
                "p95": np.percentile(center_array, 95, axis=0).tolist(),
            }
        ),
        "tracks": tracks,
        "frame_records": frame_records,
    }


def main() -> int:
    args = _parser().parse_args()
    acceptance_dir = args.acceptance_dir.expanduser().resolve()
    summary_path = acceptance_dir / "summary.json"
    frames_path = acceptance_dir / "frames.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema") != "thrown_object_end_to_end_replay_v1":
        raise ValueError("input is not a thrown-object end-to-end replay")
    case = RecordedRGBDCase(Path(str(summary["source_case"])))
    config_path = Path(str(summary["config"])).resolve()
    if _sha256(config_path) != str(summary["config_sha256"]):
        raise ValueError("acceptance config content changed")
    cfg = load_config(str(config_path))
    extrinsics = resolve_extrinsics(cfg["extrinsics"])
    if extrinsics.reference_frame != "robot_base":
        raise ValueError("diagnostic requires robot_base extrinsics")
    rows = _load_rows(frames_path)
    z_values = sorted(set(args.z_max or [1.2, 1.5, 2.0, 3.0]))
    z_min = float(cfg["camera"]["z_min"])
    if any(not np.isfinite(value) or value <= z_min for value in z_values):
        raise ValueError("every z_max must be finite and above camera z_min")

    result = {
        "schema": "thrown_object_pointcloud_envelope_diagnostic_v1",
        "acceptance_dir": str(acceptance_dir),
        "acceptance_summary_sha256": _sha256(summary_path),
        "frames_csv_sha256": _sha256(frames_path),
        "source_case": str(case.path),
        "source_manifest_sha256": _sha256(case.path / "manifest.json"),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "calibration": str(extrinsics.source),
        "calibration_sha256": _sha256(Path(extrinsics.source)),
        "calibration_id": extrinsics.calibration_id,
        "reference_frame": extrinsics.reference_frame,
        "projector_contract": {
            "num_points": 128,
            "minimum_valid_points": 16,
            "point_feature_dim": 3,
            "maximum_mask_depth_deviation_m": 0.055,
        },
        "envelopes": [
            _audit_envelope(
                acceptance_dir=acceptance_dir,
                case=case,
                rows=rows,
                T_base_camera=extrinsics.T_base_camera,
                z_min=z_min,
                z_max=value,
            )
            for value in z_values
        ],
        "hardware_interfaces_opened": False,
        "franka_opened": False,
        "rh56_opened": False,
        "robot_motion": False,
        "diagnostic_only": True,
        "production_accepted": False,
        "reason": (
            "z envelope sweep does not change deployment config; provisional "
            "424x240@60 calibration and catch-workspace accuracy still require "
            "independent physical holdouts"
        ),
    }
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else acceptance_dir / "pointcloud_envelope_diagnostic.json"
    )
    if output.exists():
        raise FileExistsError(output)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
