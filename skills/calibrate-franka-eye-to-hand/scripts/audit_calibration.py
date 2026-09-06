#!/usr/bin/env python3
"""Fail-closed structural and screening audit for a calibration YAML."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


def _load_mapping(path: Path) -> Dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _number(mapping: Dict[str, Any], *keys: str) -> Optional[float]:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) else None


def _calibration_id(transform: np.ndarray, serial: Optional[str]) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(transform, dtype="<f8").tobytes())
    digest.update((serial or "unknown").encode("utf-8"))
    return "eye-to-hand-" + digest.hexdigest()[:16]


def _check_max(
    failures: List[str], label: str, value: Optional[float], maximum: float
) -> None:
    if value is None:
        failures.append(f"missing/non-finite {label}")
    elif value > maximum:
        failures.append(f"{label}={value:.9g} exceeds {maximum:.9g}")


def _check_min(
    failures: List[str], label: str, value: Optional[float], minimum: float
) -> None:
    if value is None:
        failures.append(f"missing/non-finite {label}")
    elif value < minimum:
        failures.append(f"{label}={value:.9g} is below {minimum:.9g}")


def audit(
    calibration_path: Path,
    *,
    runtime_path: Optional[Path],
    expected_serial: Optional[str],
    expected_dictionary: Optional[str],
    expected_marker_id: Optional[int],
    expected_marker_length_m: Optional[float],
) -> Dict[str, Any]:
    failures: List[str] = []
    warnings: List[str] = []
    try:
        record = _load_mapping(calibration_path)
    except ValueError as exc:
        return {
            "static_checks_pass": False,
            "decision": "rejected",
            "calibration": str(calibration_path),
            "failures": [str(exc)],
            "warnings": [],
        }

    if record.get("schema_version") != 1:
        failures.append("schema_version must be 1")
    if record.get("kind") != "camera_robot_extrinsic_calibration":
        failures.append("kind must be camera_robot_extrinsic_calibration")
    if record.get("calibration_type") != "eye_to_hand":
        failures.append("calibration_type must be eye_to_hand")
    if record.get("base_frame") != "robot_base":
        failures.append("base_frame must be robot_base for this deployment")
    if not record.get("camera_frame"):
        failures.append("camera_frame is missing")

    transform: Optional[np.ndarray] = None
    try:
        transform = np.asarray(record.get("T_base_camera"), dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError(f"shape is {transform.shape}, expected (4, 4)")
        if not np.all(np.isfinite(transform)):
            raise ValueError("contains NaN or infinity")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
            raise ValueError("last row is not [0, 0, 0, 1]")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("rotation is not orthonormal")
        determinant = float(np.linalg.det(rotation))
        if not np.isclose(determinant, 1.0, atol=1e-5):
            raise ValueError(f"rotation determinant is {determinant:.9g}, not +1")
    except (TypeError, ValueError) as exc:
        failures.append(f"invalid T_base_camera: {exc}")
        transform = None

    camera = record.get("camera") or {}
    if not isinstance(camera, dict):
        failures.append("camera section must be a mapping")
        camera = {}
    serial_value = camera.get("serial") if isinstance(camera, dict) else None
    serial = None if serial_value in (None, "") else str(serial_value)
    if serial is None:
        failures.append("camera.serial is missing")
    if expected_serial is not None and serial != expected_serial:
        failures.append(
            f"camera serial {serial!r} does not match expected {expected_serial!r}"
        )
    if transform is not None and serial is not None:
        expected_id = _calibration_id(transform, serial)
        recorded_id = str(record.get("calibration_id") or "")
        if recorded_id != expected_id:
            failures.append(
                f"calibration_id {recorded_id!r} does not match {expected_id!r}"
            )

    depth_scale = _number(camera, "depth_scale")
    _check_min(failures, "camera.depth_scale", depth_scale, 1.0e-12)
    intrinsics = camera.get("intrinsics") or {}
    if not isinstance(intrinsics, dict):
        failures.append("camera.intrinsics must be a mapping")
        intrinsics = {}
    for label in ("width", "height", "fx", "fy"):
        _check_min(
            failures,
            f"camera.intrinsics.{label}",
            _number(intrinsics, label),
            1.0e-12,
        )
    for label in ("ppx", "ppy"):
        if _number(intrinsics, label) is None:
            failures.append(f"missing/non-finite camera.intrinsics.{label}")

    target = record.get("target") or {}
    if not isinstance(target, dict):
        failures.append("target section must be a mapping")
        target = {}
    if target.get("type") not in ("aruco", "charuco"):
        failures.append("target.type must be aruco or charuco")
    if expected_dictionary is not None and target.get("dictionary") != expected_dictionary:
        failures.append(
            f"target dictionary {target.get('dictionary')!r} does not match "
            f"expected {expected_dictionary!r}"
        )
    if expected_marker_id is not None:
        try:
            marker_id = int(target.get("marker_id"))
        except (TypeError, ValueError, OverflowError):
            marker_id = None
        if marker_id != expected_marker_id:
            failures.append(
                f"target marker_id {target.get('marker_id')!r} does not match "
                f"expected {expected_marker_id}"
            )
    if expected_marker_length_m is not None:
        marker_length = _number(target, "marker_length_m")
        if marker_length is None or not np.isclose(
            marker_length, expected_marker_length_m, atol=1.0e-9, rtol=0.0
        ):
            failures.append(
                f"target marker_length_m {marker_length!r} does not match "
                f"expected {expected_marker_length_m:.9g}"
            )

    solver = record.get("solver") or {}
    quality = solver.get("quality") if isinstance(solver, dict) else {}
    quality = quality if isinstance(quality, dict) else {}
    if quality.get("status") != "pass":
        failures.append(f"solver quality status is {quality.get('status')!r}, not 'pass'")
    quality_warnings = quality.get("warnings") or []
    if quality_warnings:
        failures.append(f"final solver warnings are not empty: {quality_warnings}")

    _check_min(failures, "solver.sample_count", _number(solver, "sample_count"), 15)
    _check_min(
        failures,
        "translation_span_m",
        _number(quality, "motion_diversity", "translation_span_m"),
        0.08,
    )
    _check_min(
        failures,
        "rotation_span_deg",
        _number(quality, "motion_diversity", "rotation_span_deg"),
        30.0,
    )
    _check_max(
        failures,
        "reprojection_error_p95_px",
        _number(quality, "reprojection_error_px", "p95"),
        1.0,
    )
    _check_max(
        failures,
        "translation_error_p95_m",
        _number(quality, "translation_error_m", "p95"),
        0.010,
    )
    _check_max(
        failures,
        "rotation_error_p95_deg",
        _number(quality, "rotation_error_deg", "p95"),
        2.0,
    )

    if runtime_path is not None:
        try:
            runtime = _load_mapping(runtime_path)
            runtime_camera = runtime.get("camera") or {}
            runtime_extrinsics = runtime.get("extrinsics") or {}
            if not isinstance(runtime_camera, dict):
                failures.append("runtime camera section must be a mapping")
                runtime_camera = {}
            if not isinstance(runtime_extrinsics, dict):
                failures.append("runtime extrinsics section must be a mapping")
                runtime_extrinsics = {}
            if str(runtime_camera.get("serial") or "") != str(serial or ""):
                failures.append("runtime camera.serial does not match calibration serial")
            for label in ("width", "height"):
                runtime_value = _number(runtime_camera, label)
                calibrated_value = _number(intrinsics, label)
                if (
                    runtime_value is None
                    or calibrated_value is None
                    or runtime_value != calibrated_value
                ):
                    failures.append(
                        f"runtime camera.{label} does not match calibrated profile"
                    )
            _check_min(
                failures,
                "runtime camera.fps",
                _number(runtime_camera, "fps"),
                1.0,
            )
            for key in (
                "require_calibration",
                "require_quality_pass",
                "strict_camera_serial",
            ):
                if runtime_extrinsics.get(key) is not True:
                    failures.append(f"runtime extrinsics.{key} must be true")
            configured_file = runtime_extrinsics.get("calibration_file")
            if not isinstance(configured_file, str) or not configured_file.strip():
                failures.append("runtime extrinsics.calibration_file is missing")
            else:
                configured_path = Path(configured_file)
                if not configured_path.is_absolute():
                    configured_path = runtime_path.parent / configured_path
                if configured_path.resolve() != calibration_path.resolve():
                    failures.append(
                        "runtime calibration_file resolves to "
                        f"{configured_path.resolve()}, not {calibration_path.resolve()}"
                    )
        except ValueError as exc:
            failures.append(str(exc))

    if runtime_path is None:
        warnings.append("runtime config was not checked")
    warnings.append(
        "static checks do not include holdout closure, mount rigidity, "
        "multi-distance depth, known-base-point, or task-clearance evidence"
    )

    return {
        "static_checks_pass": not failures,
        "decision": "provisional" if not failures else "rejected",
        "calibration": str(calibration_path.resolve()),
        "runtime_config": None if runtime_path is None else str(runtime_path.resolve()),
        "calibration_id": record.get("calibration_id"),
        "camera_serial": serial,
        "method": solver.get("method") if isinstance(solver, dict) else None,
        "sample_count": solver.get("sample_count") if isinstance(solver, dict) else None,
        "failures": failures,
        "warnings": warnings,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a final fixed-camera Franka eye-to-hand calibration"
    )
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--expected-serial")
    parser.add_argument("--expected-dictionary")
    parser.add_argument("--expected-marker-id", type=int)
    parser.add_argument("--expected-marker-length-m", type=float)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = audit(
        args.calibration,
        runtime_path=args.runtime_config,
        expected_serial=args.expected_serial,
        expected_dictionary=args.expected_dictionary,
        expected_marker_id=args.expected_marker_id,
        expected_marker_length_m=args.expected_marker_length_m,
    )
    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        status = "STATIC PASS" if result["static_checks_pass"] else "STATIC FAIL"
        print(f"[{status}] {result['calibration']}")
        print(
            f"  id={result.get('calibration_id')} serial={result.get('camera_serial')} "
            f"method={result.get('method')} samples={result.get('sample_count')}"
        )
        for message in result["failures"]:
            print(f"  [FAIL] {message}")
        for message in result["warnings"]:
            print(f"  [WARN] {message}")
    return 0 if result["static_checks_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
