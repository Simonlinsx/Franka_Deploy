#!/usr/bin/env python3
"""Fit and audit a provisional affine correction for aligned depth.

This is an offline evidence tool.  It never opens a camera or robot, never
edits a runtime configuration, and never treats robot kinematics derived from
the active eye-to-hand session as an independent metrology reference.

The fitted model is intentionally small::

    corrected_z = scale * raw_z + offset_m

The original camera ray is scaled by ``corrected_z / raw_z`` before applying
the frozen ``T_base_camera``.  A report is accepted only when the supplied
stations span the requested distance, leave-one-station-out errors pass, and
at least one report contains an independently known base-frame reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import yaml


SCHEMA_VERSION = 1
KIND = "aligned_depth_affine_candidate_audit"


class DepthAffineAuditError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mapping(path: Path) -> Dict[str, Any]:
    try:
        if path.suffix.lower() in (".yaml", ".yml"):
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError, json.JSONDecodeError) as exc:
        raise DepthAffineAuditError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DepthAffineAuditError(f"{path} must contain a mapping")
    return value


def _rigid(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise DepthAffineAuditError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-9):
        raise DepthAffineAuditError(f"{name} has an invalid bottom row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise DepthAffineAuditError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6):
        raise DepthAffineAuditError(f"{name} rotation determinant is not +1")
    return matrix


def _percentiles(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise DepthAffineAuditError("metric array is empty or non-finite")
    return {
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
    }


def _write_exclusive_atomic(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise DepthAffineAuditError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{time.monotonic_ns()}"
    )
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _station_rows(
    report_path: Path, T_camera_base: np.ndarray
) -> Tuple[Dict[str, Any], List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    report = _load_mapping(report_path)
    if report.get("kind") != "eye_to_hand_rgbd_physical_station":
        raise DepthAffineAuditError(f"unexpected report kind in {report_path}")
    reference = report.get("known_base_reference")
    if not isinstance(reference, dict):
        raise DepthAffineAuditError(f"missing known_base_reference in {report_path}")
    rows: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for frame in report.get("frames") or []:
        if not isinstance(frame, dict):
            continue
        rgbd = frame.get("rgbd")
        if not isinstance(rgbd, dict):
            continue
        raw_camera = np.asarray(rgbd.get("marker_center_camera_depth_m"), dtype=np.float64)
        known_base = np.asarray(rgbd.get("known_marker_center_base_m"), dtype=np.float64)
        if (
            raw_camera.shape != (3,)
            or known_base.shape != (3,)
            or not np.all(np.isfinite(raw_camera))
            or not np.all(np.isfinite(known_base))
            or raw_camera[2] <= 0.0
        ):
            raise DepthAffineAuditError(f"malformed RGB-D row in {report_path}")
        known_camera = (T_camera_base @ np.r_[known_base, 1.0])[:3]
        rows.append((raw_camera, known_camera, known_base))
    if not rows:
        raise DepthAffineAuditError(f"no valid RGB-D rows in {report_path}")
    metadata = {
        "path": str(report_path.resolve()),
        "sha256": _sha256(report_path),
        "station_id": str(report.get("station_id") or report_path.stem),
        "distance_band": report.get("distance_band"),
        "image_region": report.get("image_region"),
        "independent_known_base": reference.get("independent_of_calibration") is True,
        "known_base_source": reference.get("source"),
        "frame_count": len(rows),
    }
    return metadata, rows


def _fit(stations: Sequence[Mapping[str, Any]], indices: Iterable[int]) -> Tuple[float, float]:
    raw_z: List[float] = []
    known_z: List[float] = []
    for index in indices:
        for raw, known_camera, _known_base in stations[index]["rows"]:
            raw_z.append(float(raw[2]))
            known_z.append(float(known_camera[2]))
    design = np.column_stack((raw_z, np.ones(len(raw_z), dtype=np.float64)))
    scale, offset = np.linalg.lstsq(design, known_z, rcond=None)[0]
    if not math.isfinite(float(scale)) or not math.isfinite(float(offset)):
        raise DepthAffineAuditError("affine fit is non-finite")
    return float(scale), float(offset)


def _evaluate(
    station: Mapping[str, Any], scale: float, offset_m: float, T_base_camera: np.ndarray
) -> Dict[str, Any]:
    errors_3d_m: List[float] = []
    errors_z_m: List[float] = []
    raw_z_m: List[float] = []
    known_z_m: List[float] = []
    for raw, known_camera, known_base in station["rows"]:
        corrected_z = scale * float(raw[2]) + offset_m
        if corrected_z <= 0.0:
            raise DepthAffineAuditError("affine correction produced non-positive depth")
        corrected_camera = raw * (corrected_z / float(raw[2]))
        corrected_base = (T_base_camera @ np.r_[corrected_camera, 1.0])[:3]
        delta = corrected_base - known_base
        errors_3d_m.append(float(np.linalg.norm(delta)))
        errors_z_m.append(abs(float(delta[2])))
        raw_z_m.append(float(raw[2]))
        known_z_m.append(float(known_camera[2]))
    return {
        "raw_z_camera_m": _percentiles(raw_z_m),
        "known_z_camera_m": _percentiles(known_z_m),
        "corrected_base_error_3d_m": _percentiles(errors_3d_m),
        "corrected_base_error_z_abs_m": _percentiles(errors_z_m),
    }


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    calibration_path = Path(args.calibration).expanduser().resolve()
    calibration = _load_mapping(calibration_path)
    T_base_camera = _rigid(calibration.get("T_base_camera"), "T_base_camera")
    T_camera_base = np.linalg.inv(T_base_camera)
    expected_serial = str(args.expected_serial)
    actual_serial = str((calibration.get("camera") or {}).get("serial") or "")
    if actual_serial != expected_serial:
        raise DepthAffineAuditError(
            f"calibration serial {actual_serial!r} != expected {expected_serial!r}"
        )

    stations: List[Dict[str, Any]] = []
    for report_name in args.reports:
        metadata, rows = _station_rows(Path(report_name).expanduser().resolve(), T_camera_base)
        metadata["rows"] = rows
        stations.append(metadata)
    if len(stations) < 3:
        raise DepthAffineAuditError("at least three station reports are required")

    scale, offset_m = _fit(stations, range(len(stations)))
    station_results: List[Dict[str, Any]] = []
    station_median_raw_z: List[float] = []
    for station in stations:
        metrics = _evaluate(station, scale, offset_m, T_base_camera)
        station_median_raw_z.append(metrics["raw_z_camera_m"]["median"])
        station_results.append({
            **{key: value for key, value in station.items() if key != "rows"},
            "fit_using_all_stations": metrics,
        })

    loso_results: List[Dict[str, Any]] = []
    all_loso_p95_3d_m: List[float] = []
    all_loso_max_3d_m: List[float] = []
    for holdout_index, station in enumerate(stations):
        train = [index for index in range(len(stations)) if index != holdout_index]
        fold_scale, fold_offset = _fit(stations, train)
        metrics = _evaluate(station, fold_scale, fold_offset, T_base_camera)
        all_loso_p95_3d_m.append(metrics["corrected_base_error_3d_m"]["p95"])
        all_loso_max_3d_m.append(metrics["corrected_base_error_3d_m"]["max"])
        loso_results.append({
            "held_out_station_id": station["station_id"],
            "training_station_ids": [stations[index]["station_id"] for index in train],
            "scale": fold_scale,
            "offset_m": fold_offset,
            "held_out_metrics": metrics,
        })

    distance_span_m = float(max(station_median_raw_z) - min(station_median_raw_z))
    independent_count = sum(
        station["independent_known_base"] for station in stations
    )
    failures: List[str] = []
    if distance_span_m + 1.0e-12 < args.minimum_distance_span_m:
        failures.append(
            f"distance span {distance_span_m:.6f}m < {args.minimum_distance_span_m:.6f}m"
        )
    if args.require_independent_known_base and independent_count < 1:
        failures.append("no independently known robot_base reference station")
    worst_loso_p95 = max(all_loso_p95_3d_m)
    worst_loso_max = max(all_loso_max_3d_m)
    if worst_loso_p95 > args.max_loso_3d_p95_mm / 1000.0:
        failures.append(
            f"LOSO 3D p95 {1000.0 * worst_loso_p95:.3f}mm > "
            f"{args.max_loso_3d_p95_mm:.3f}mm"
        )
    if worst_loso_max > args.max_loso_3d_max_mm / 1000.0:
        failures.append(
            f"LOSO 3D max {1000.0 * worst_loso_max:.3f}mm > "
            f"{args.max_loso_3d_max_mm:.3f}mm"
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "decision": "accepted_candidate" if not failures else "refused_candidate",
        "failures": failures,
        "deployment_mutated": False,
        "hardware_opened": False,
        "model": {
            "equation": "corrected_z_m = scale * raw_z_m + offset_m",
            "ray_correction": "p_camera_corrected = p_camera_raw * corrected_z_m / raw_z_m",
            "scale": scale,
            "offset_m": offset_m,
        },
        "calibration": {
            "path": str(calibration_path),
            "sha256": _sha256(calibration_path),
            "calibration_id": calibration.get("calibration_id"),
            "camera_serial": actual_serial,
            "T_base_camera_unchanged": True,
        },
        "gates": {
            "minimum_distance_span_m": float(args.minimum_distance_span_m),
            "require_independent_known_base": bool(args.require_independent_known_base),
            "max_loso_3d_p95_mm": float(args.max_loso_3d_p95_mm),
            "max_loso_3d_max_mm": float(args.max_loso_3d_max_mm),
        },
        "evidence": {
            "station_count": len(stations),
            "independent_known_base_station_count": int(independent_count),
            "station_median_raw_depth_span_m": distance_span_m,
            "worst_loso_3d_p95_m": worst_loso_p95,
            "worst_loso_3d_max_m": worst_loso_max,
            "stations": station_results,
            "leave_one_station_out": loso_results,
        },
        "warning": (
            "This artifact is an offline candidate audit, not a RealSense factory "
            "calibration and not authorization to edit or enable a robot-facing runtime."
        ),
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--expected-serial", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-distance-span-m", type=float, required=True)
    parser.add_argument("--max-loso-3d-p95-mm", type=float, required=True)
    parser.add_argument("--max-loso-3d-max-mm", type=float, required=True)
    parser.add_argument("--require-independent-known-base", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in (
        "minimum_distance_span_m",
        "max_loso_3d_p95_mm",
        "max_loso_3d_max_mm",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be finite and positive")
    try:
        report = audit(args)
        _write_exclusive_atomic(Path(args.output), report)
    except (DepthAffineAuditError, OSError, ValueError) as exc:
        print(f"depth affine audit: ERROR: {type(exc).__name__}: {exc}")
        return 2
    print(json.dumps({
        "decision": report["decision"],
        "output": str(Path(args.output).expanduser().resolve()),
        "scale": report["model"]["scale"],
        "offset_m": report["model"]["offset_m"],
        "distance_span_m": report["evidence"]["station_median_raw_depth_span_m"],
        "worst_loso_3d_p95_m": report["evidence"]["worst_loso_3d_p95_m"],
        "failures": report["failures"],
    }, sort_keys=True))
    return 0 if report["decision"] == "accepted_candidate" else 1


if __name__ == "__main__":
    raise SystemExit(main())
