#!/usr/bin/env python3
"""Validate reviewed thrown-object masks as catch-workspace policy clouds.

This is an offline acceptance boundary.  It consumes a completed
``validate_recorded_throw_end_to_end`` result, independently reviewed masks,
the immutable RGB-D recording, and a passing physical RGB-D report containing
an independent robot-base reference.  It never imports or opens Franka/RH56.

Each ``--track`` is ``NAME:REFERENCE_MASK_DIR``.  The reference filenames are
the original 60 Hz recording indices (for example ``000123.png``); only rows
already evaluated at the production 20 Hz cadence are accepted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (ROOT, WORKSPACE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamic_pcd.camera.recorded_rgbd_camera import RecordedRGBDCase  # noqa: E402
from dynamic_pcd.calibration import resolve_extrinsics  # noqa: E402
from dynamic_pcd.config import load_config  # noqa: E402
from sim2real.observation.model import MaskedRGBDProjector  # noqa: E402
from sim2real.tasks.thrown_contract import (  # noqa: E402
    resolve_v57_thrown_task_contract,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: Iterable[float], q: float) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return None if not finite else float(np.percentile(finite, q))


def _track(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition(":")
    path = Path(raw_path).expanduser().resolve()
    if not separator or not name.strip() or not path.is_dir():
        raise argparse.ArgumentTypeError(
            "track must be NAME:EXISTING_REFERENCE_MASK_DIR"
        )
    return name.strip(), path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("acceptance_dir", type=Path)
    parser.add_argument(
        "--physical-rgbd-validation-report", type=Path, required=True
    )
    parser.add_argument("--track", type=_track, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-workspace-visible-frames", type=int, default=20)
    parser.add_argument(
        "--minimum-reference-depth-fraction", type=float, default=0.95
    )
    parser.add_argument("--minimum-reference-inside-fraction", type=float, default=0.90)
    parser.add_argument("--minimum-fresh-128-fraction", type=float, default=0.95)
    parser.add_argument("--minimum-output-inside-fraction", type=float, default=0.90)
    parser.add_argument("--max-center-error-p95-mm", type=float, default=15.0)
    parser.add_argument("--max-center-error-max-mm", type=float, default=25.0)
    return parser


def _finite_vector(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return vector


def _workspace(
    manifest: dict[str, Any],
    *,
    expected_minimum: np.ndarray,
    expected_maximum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    overlay = manifest.get("live_workspace_overlay")
    if not isinstance(overlay, dict):
        raise ValueError("recording has no commissioned live_workspace_overlay")
    if overlay.get("reference_frame") != "robot_base":
        raise ValueError("workspace overlay is not expressed in robot_base")
    if overlay.get("all_vertices_inside_image") is not True:
        raise ValueError("workspace overlay was not fully visible")
    if overlay.get("lossless_frames_annotated") is not False:
        raise ValueError("workspace overlay must not modify lossless RGB-D frames")
    minimum = _finite_vector(overlay.get("minimum_base_m"), "workspace minimum")
    maximum = _finite_vector(overlay.get("maximum_base_m"), "workspace maximum")
    if np.any(maximum <= minimum):
        raise ValueError("workspace bounds are inverted or empty")
    if not np.array_equal(minimum, np.asarray(expected_minimum, dtype=np.float64)):
        raise ValueError("recording workspace minimum differs from V57 target box")
    if not np.array_equal(maximum, np.asarray(expected_maximum, dtype=np.float64)):
        raise ValueError("recording workspace maximum differs from V57 target box")
    vertices = np.asarray(overlay.get("vertices_base_m"), dtype=np.float64)
    if vertices.shape != (8, 3) or not np.all(np.isfinite(vertices)):
        raise ValueError("workspace overlay does not bind eight finite vertices")
    expected = np.asarray(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=np.float64,
    )
    if not np.array_equal(vertices, expected):
        raise ValueError("workspace overlay vertices do not match its bounds")
    return minimum, maximum


def _physical_report(
    path: Path,
    *,
    serial: str,
    calibration_id: str,
    calibration_sha256: str,
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("kind") != (
        "eye_to_hand_rgbd_physical_validation"
    ):
        raise ValueError("invalid physical RGB-D validation report schema")
    if report.get("physical_rgbd_status") != "pass":
        raise ValueError("physical RGB-D validation report is not PASS")
    if report.get("independent_known_base_evidence_present") is not True:
        raise ValueError("physical report has no independent robot-base reference")
    if report.get("failures") or report.get("missing_evidence"):
        raise ValueError("physical report contains failures or missing evidence")
    if str(report.get("camera_serial")) != str(serial):
        raise ValueError("physical report camera serial mismatch")
    if str(report.get("calibration_id")) != str(calibration_id):
        raise ValueError("physical report calibration id mismatch")
    if str(report.get("calibration_sha256")) != str(calibration_sha256):
        raise ValueError("physical report calibration SHA mismatch")
    return report


def _bind_runtime_calibration(
    *,
    cfg: dict[str, Any],
    calibration_path: Path,
    manifest: dict[str, Any],
) -> None:
    """Require the runtime config and lossless recording to bind one transform."""

    extrinsics = resolve_extrinsics(cfg.get("extrinsics"))
    if not extrinsics.calibrated or extrinsics.reference_frame != "robot_base":
        raise ValueError("runtime config does not resolve calibrated robot_base output")
    if Path(extrinsics.source).resolve() != calibration_path:
        raise ValueError("runtime config resolves a different calibration file")
    if str(extrinsics.calibration_id or "") != str(manifest.get("calibration_id") or ""):
        raise ValueError("recording calibration id conflicts with runtime calibration")
    if str(extrinsics.camera_serial or "") != str(manifest.get("camera_serial") or ""):
        raise ValueError("runtime calibration camera serial conflicts with recording")
    recorded_transform = np.asarray(manifest.get("T_base_camera"), dtype=np.float32)
    if (
        recorded_transform.shape != (4, 4)
        or not np.all(np.isfinite(recorded_transform))
        or not np.array_equal(recorded_transform, extrinsics.T_base_camera)
    ):
        raise ValueError("recording T_base_camera differs from runtime calibration")

    camera = cfg.get("camera") or {}
    expected_profile = (
        str(camera.get("serial") or ""),
        int(camera.get("width", -1)),
        int(camera.get("height", -1)),
        int(camera.get("fps", -1)),
    )
    recorded_profile = (
        str(manifest.get("camera_serial") or ""),
        int(manifest.get("image_width", -1)),
        int(manifest.get("image_height", -1)),
        int(manifest.get("nominal_fps", -1)),
    )
    if expected_profile != recorded_profile:
        raise ValueError(
            "runtime camera profile differs from recording: "
            f"runtime={expected_profile} recording={recorded_profile}"
        )
    if expected_profile[1:] != (424, 240, 60):
        raise ValueError("thrown-catch acceptance requires native 424x240@60")
    if not bool((cfg.get("extrinsics") or {}).get("strict_camera_serial", False)):
        raise ValueError("runtime config must enforce strict_camera_serial")


def _projector(case: RecordedRGBDCase, cfg: dict[str, Any]) -> MaskedRGBDProjector:
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
        T_base_camera_optical=np.asarray(case.manifest["T_base_camera"], dtype=np.float64),
        image_size=(case.width, case.height),
        depth_range_m=(float(cfg["camera"]["z_min"]), float(cfg["camera"]["z_max"])),
        num_points=128,
        minimum_valid_points=16,
        point_feature_dim=3,
        maximum_mask_depth_deviation_m=0.055,
    )


def _mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != shape:
        raise ValueError(f"invalid reviewed mask: {path}")
    return image > 0


def _points(frame: Any) -> np.ndarray:
    valid = np.asarray(frame.valid, dtype=np.float32)
    points = np.asarray(frame.xyzrgb_palm, dtype=np.float32)
    if valid.shape != (128,) or points.shape != (128, 3):
        raise ValueError("projector violated the policy 128x3 contract")
    return np.asarray(points[valid > 0.5], dtype=np.float64)


def _exact_current_128(frame: Any, *, frame_id: int, timestamp: float) -> bool:
    points = _points(frame)
    return bool(
        str(frame.status) == "fresh"
        and int(frame.frame_id) == int(frame_id)
        and float(frame.captured_at_s) == float(timestamp)
        and points.shape == (128, 3)
        and np.all(np.isfinite(points))
    )


def _inside(points: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    return np.all((value >= minimum) & (value <= maximum), axis=1)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("end-to-end result has no evaluated rows")
    return rows


def main() -> int:
    args = _parser().parse_args()
    if args.minimum_workspace_visible_frames < 1:
        raise ValueError("minimum workspace visible frames must be positive")
    for name in (
        "minimum_reference_depth_fraction",
        "minimum_reference_inside_fraction",
        "minimum_fresh_128_fraction",
        "minimum_output_inside_fraction",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0,1]")
    if args.max_center_error_p95_mm <= 0.0 or args.max_center_error_max_mm <= 0.0:
        raise ValueError("center-error limits must be positive")

    acceptance = args.acceptance_dir.expanduser().resolve(strict=True)
    summary_path = acceptance / "summary.json"
    frames_path = acceptance / "frames.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema") != "thrown_object_end_to_end_replay_v1":
        raise ValueError("input is not a thrown-object end-to-end result")
    if summary.get("mask_and_latency_accepted") is not True:
        raise ValueError("mask and latency acceptance did not PASS")
    if summary.get("hardware_interfaces_opened") is not False:
        raise ValueError("end-to-end result opened a hardware interface")
    case = RecordedRGBDCase(Path(str(summary["source_case"])))
    manifest_path = case.path / "manifest.json"
    if _sha256(manifest_path) != str(summary["source_manifest_sha256"]):
        raise ValueError("source recording manifest changed")
    config_path = Path(str(summary["config"])).resolve(strict=True)
    calibration_path = Path(str(summary["calibration"])).resolve(strict=True)
    if _sha256(config_path) != str(summary["config_sha256"]):
        raise ValueError("acceptance config changed")
    calibration_sha = _sha256(calibration_path)
    if calibration_sha != str(summary["calibration_sha256"]):
        raise ValueError("acceptance calibration changed")
    manifest = dict(case.manifest)
    if manifest.get("camera_only") is not True or any(
        manifest.get(name) is not False
        for name in (
            "franka_interface_opened",
            "rh56_interface_opened",
            "robot_command_writes",
        )
    ):
        raise ValueError("recording is not a camera-only fail-closed artifact")
    if str(manifest.get("camera_serial")) != str(summary["camera_serial"]):
        raise ValueError("recording/acceptance camera serial mismatch")
    if str(manifest.get("calibration_source_sha256")) != calibration_sha:
        raise ValueError("recording did not bind the accepted calibration bytes")
    cfg = load_config(str(config_path))
    _bind_runtime_calibration(
        cfg=cfg,
        calibration_path=calibration_path,
        manifest=manifest,
    )
    if str(summary.get("camera_profile")) != "424x240@60Hz":
        raise ValueError("end-to-end acceptance did not use native 424x240@60")
    if int(summary.get("policy_cadence_hz", -1)) != 20:
        raise ValueError("end-to-end acceptance did not evaluate the 20 Hz policy cadence")
    task_contract = resolve_v57_thrown_task_contract(config_path)
    if task_contract is None:
        raise ValueError("end-to-end config does not bind a V57 thrown task contract")
    selected_curriculum = task_contract.selected_curriculum
    minimum, maximum = _workspace(
        manifest,
        expected_minimum=selected_curriculum.target_minimum_base_m,
        expected_maximum=selected_curriculum.target_maximum_base_m,
    )
    support_minimum = minimum - task_contract.object_half_extent_base_m
    support_maximum = maximum + task_contract.object_half_extent_base_m
    physical_path = args.physical_rgbd_validation_report.expanduser().resolve(strict=True)
    _physical_report(
        physical_path,
        serial=str(manifest["camera_serial"]),
        calibration_id=str(manifest["calibration_id"]),
        calibration_sha256=calibration_sha,
    )
    rows = _read_rows(frames_path)
    row_map = {(str(row["track"]), int(row["index"])): row for row in rows}
    if len(row_map) != len(rows):
        raise ValueError("duplicate track/index rows in end-to-end result")

    track_map = dict(args.track)
    if len(track_map) != len(args.track):
        raise ValueError("duplicate --track name")
    expected_tracks = set(summary.get("tracks", {}))
    if set(track_map) != expected_tracks:
        raise ValueError("reviewed track names do not exactly match acceptance tracks")

    records: list[dict[str, Any]] = []
    reviewed_visible = reference_depth = workspace_visible = fresh_128 = 0
    center_errors: list[float] = []
    for track_name, reference_dir in sorted(track_map.items()):
        actual_projector = _projector(case, cfg)
        reference_projector = _projector(case, cfg)
        reference_paths = sorted(reference_dir.glob("*.png"))
        if not reference_paths:
            raise ValueError(f"track {track_name} has no reviewed masks")
        for reference_path in reference_paths:
            try:
                index = int(reference_path.stem)
            except ValueError as exc:
                raise ValueError(f"non-integer reference mask {reference_path}") from exc
            row = row_map.get((track_name, index))
            if row is None:
                raise ValueError(
                    f"track {track_name} reference {index} was not evaluated at 20 Hz"
                )
            frame = case.frame(index)
            reference_mask = _mask(reference_path, frame.depth_raw.shape)
            if not bool(reference_mask.any()):
                continue
            reviewed_visible += 1
            actual_mask = _mask(
                acceptance / track_name / "mask" / f"{index:06d}.png",
                frame.depth_raw.shape,
            )
            reference_frame = reference_projector.project(
                color_bgr=frame.color_bgr,
                depth_raw=frame.depth_raw,
                depth_scale_m_per_unit=frame.depth_scale,
                object_mask=reference_mask,
                T_base_palm_at_capture=np.eye(4, dtype=np.float64),
                captured_at_s=frame.timestamp,
                frame_id=frame.frame_id,
            )
            actual_frame = actual_projector.project(
                color_bgr=frame.color_bgr,
                depth_raw=frame.depth_raw,
                depth_scale_m_per_unit=frame.depth_scale,
                object_mask=actual_mask,
                T_base_palm_at_capture=np.eye(4, dtype=np.float64),
                captured_at_s=frame.timestamp,
                frame_id=frame.frame_id,
            )
            reference_points = _points(reference_frame)
            reference_fresh = bool(
                str(reference_frame.status) == "fresh"
                and int(reference_frame.frame_id) == int(frame.frame_id)
                and float(reference_frame.captured_at_s) == float(frame.timestamp)
                and reference_points.size > 0
                and np.all(np.isfinite(reference_points))
            )
            if reference_fresh:
                reference_depth += 1
            reference_center = (
                None if not reference_fresh else np.median(reference_points, axis=0)
            )
            reference_inside_fraction = (
                0.0
                if not reference_fresh
                else float(
                    np.mean(
                        _inside(reference_points, support_minimum, support_maximum)
                    )
                )
            )
            reference_in_workspace = bool(
                reference_fresh
                and np.all(reference_center >= minimum)
                and np.all(reference_center <= maximum)
                and reference_inside_fraction
                >= float(args.minimum_reference_inside_fraction)
            )
            actual_points = _points(actual_frame)
            exact_128 = _exact_current_128(
                actual_frame, frame_id=frame.frame_id, timestamp=frame.timestamp
            )
            actual_inside_fraction = (
                0.0
                if not exact_128
                else float(
                    np.mean(_inside(actual_points, support_minimum, support_maximum))
                )
            )
            center_error = None
            if reference_in_workspace:
                workspace_visible += 1
                if exact_128 and actual_inside_fraction >= float(
                    args.minimum_output_inside_fraction
                ):
                    fresh_128 += 1
                    center_error = float(
                        np.linalg.norm(np.median(actual_points, axis=0) - reference_center)
                    )
                    center_errors.append(center_error)
            records.append(
                {
                    "track": track_name,
                    "index": index,
                    "frame_id": int(frame.frame_id),
                    "reference_mask_sha256": _sha256(reference_path),
                    "reference_depth_fresh": reference_fresh,
                    "reference_center_robot_base_m": (
                        None if reference_center is None else reference_center.tolist()
                    ),
                    "reference_inside_workspace_fraction": reference_inside_fraction,
                    "reference_in_workspace": reference_in_workspace,
                    "actual_status": str(actual_frame.status),
                    "actual_source_points": int(actual_frame.source_valid_points),
                    "actual_exact_current_128": exact_128,
                    "actual_inside_workspace_fraction": actual_inside_fraction,
                    "center_error_m": center_error,
                }
            )

    reference_fraction = (
        0.0 if reviewed_visible == 0 else reference_depth / float(reviewed_visible)
    )
    fresh_fraction = (
        0.0 if workspace_visible == 0 else fresh_128 / float(workspace_visible)
    )
    center_p95 = _percentile(center_errors, 95)
    center_max = _percentile(center_errors, 100)
    gates = {
        "mask_and_latency": True,
        "physical_rgbd_with_independent_base": True,
        "reference_depth_fraction": reference_fraction
        >= float(args.minimum_reference_depth_fraction),
        "minimum_workspace_visible_frames": workspace_visible
        >= int(args.minimum_workspace_visible_frames),
        "fresh_exact_current_128_fraction": fresh_fraction
        >= float(args.minimum_fresh_128_fraction),
        "center_error_p95": center_p95 is not None
        and center_p95 <= float(args.max_center_error_p95_mm) / 1000.0,
        "center_error_max": center_max is not None
        and center_max <= float(args.max_center_error_max_mm) / 1000.0,
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    result = {
        "schema_version": 1,
        "kind": "thrown_catch_workspace_pointcloud_acceptance",
        "status": "pass" if all(gates.values()) else "fail",
        "acceptance_dir": str(acceptance),
        "acceptance_summary_sha256": _sha256(summary_path),
        "acceptance_frames_sha256": _sha256(frames_path),
        "source_case": str(case.path),
        "source_manifest_sha256": _sha256(manifest_path),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "calibration": str(calibration_path),
        "calibration_sha256": calibration_sha,
        "calibration_id": str(manifest["calibration_id"]),
        "physical_rgbd_validation_report": str(physical_path),
        "physical_rgbd_validation_report_sha256": _sha256(physical_path),
        "camera_serial": str(manifest["camera_serial"]),
        "camera_profile": f"{case.width}x{case.height}@{manifest['nominal_fps']}Hz",
        "reference_frame": "robot_base",
        "simulation_task_contract": str(task_contract.source_path),
        "simulation_task_contract_sha256": task_contract.source_sha256,
        "simulation_curriculum": selected_curriculum.name,
        "catch_reference_center_base_m": (
            task_contract.catch_reference_center_base_m.tolist()
        ),
        "workspace_minimum_base_m": minimum.tolist(),
        "workspace_maximum_base_m": maximum.tolist(),
        "object_support_minimum_base_m": support_minimum.tolist(),
        "object_support_maximum_base_m": support_maximum.tolist(),
        "policy_cadence_hz": 20,
        "reviewed_visible_frames": reviewed_visible,
        "reference_depth_fresh_frames": reference_depth,
        "reference_depth_fresh_fraction": reference_fraction,
        "workspace_visible_frames": workspace_visible,
        "fresh_exact_current_128_frames": fresh_128,
        "fresh_exact_current_128_fraction": fresh_fraction,
        "center_error_p95_m": center_p95,
        "center_error_max_m": center_max,
        "limits": {
            "minimum_workspace_visible_frames": int(args.minimum_workspace_visible_frames),
            "minimum_reference_depth_fraction": float(args.minimum_reference_depth_fraction),
            "minimum_reference_inside_fraction": float(args.minimum_reference_inside_fraction),
            "minimum_fresh_128_fraction": float(args.minimum_fresh_128_fraction),
            "minimum_output_inside_fraction": float(args.minimum_output_inside_fraction),
            "max_center_error_p95_m": float(args.max_center_error_p95_mm) / 1000.0,
            "max_center_error_max_m": float(args.max_center_error_max_mm) / 1000.0,
        },
        "gates": gates,
        "frame_records": records,
        "hardware_interfaces_opened": False,
        "franka_opened": False,
        "rh56_opened": False,
        "robot_motion": False,
        "production_accepted": all(gates.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["production_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
