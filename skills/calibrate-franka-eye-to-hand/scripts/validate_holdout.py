#!/usr/bin/env python3
"""Validate eye-to-hand closure on a dataset excluded from calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml


def _load(path: Path) -> Dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _transform(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{label} must have shape (4, 4), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} contains NaN or infinity")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-5):
        raise ValueError(f"{label} has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5):
        raise ValueError(f"{label} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-5):
        raise ValueError(f"{label} rotation determinant is not +1")
    return matrix


def _error(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    delta = np.linalg.inv(first) @ second
    translation_m = float(np.linalg.norm(delta[:3, 3]))
    cosine = float(np.clip((np.trace(delta[:3, :3]) - 1.0) * 0.5, -1.0, 1.0))
    rotation_deg = float(np.degrees(np.arccos(cosine)))
    return translation_m, rotation_deg


def _metrics(values: np.ndarray) -> Dict[str, float]:
    return {
        "median": float(np.median(values)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _camera_serial(record: Dict[str, Any]) -> str:
    camera = record.get("camera") or {}
    return "" if not isinstance(camera, dict) else str(camera.get("serial") or "")


def validate(
    calibration_path: Path,
    holdout_path: Path,
    *,
    minimum_samples: int,
    max_translation_p95_mm: float,
    max_rotation_p95_deg: float,
) -> Dict[str, Any]:
    failures: List[str] = []
    try:
        calibration = _load(calibration_path)
        holdout = _load(holdout_path)
        T_base_camera = _transform(
            calibration.get("T_base_camera"), "calibration.T_base_camera"
        )
        T_ee_target = _transform(
            calibration.get("T_ee_target"), "calibration.T_ee_target"
        )
    except ValueError as exc:
        return {
            "holdout_pass": False,
            "failures": [str(exc)],
            "calibration": str(calibration_path),
            "holdout": str(holdout_path),
        }

    calibration_serial = _camera_serial(calibration)
    holdout_serial = _camera_serial(holdout)
    if not calibration_serial or holdout_serial != calibration_serial:
        failures.append(
            f"holdout camera serial {holdout_serial!r} does not match "
            f"calibration serial {calibration_serial!r}"
        )
    if holdout.get("target") != calibration.get("target"):
        failures.append("holdout target metadata does not exactly match calibration")

    samples = holdout.get("samples")
    if not isinstance(samples, list):
        samples = []
        failures.append("holdout.samples must be a list")
    if len(samples) < minimum_samples:
        failures.append(
            f"holdout has {len(samples)} samples; at least {minimum_samples} required"
        )

    per_pose: List[Dict[str, Any]] = []
    for index, sample in enumerate(samples):
        try:
            if not isinstance(sample, dict):
                raise ValueError("sample must be a mapping")
            T_base_ee = _transform(sample.get("T_base_ee"), f"sample[{index}].T_base_ee")
            T_camera_target = _transform(
                sample.get("T_camera_target"),
                f"sample[{index}].T_camera_target",
            )
            translation_m, rotation_deg = _error(
                T_base_ee @ T_ee_target,
                T_base_camera @ T_camera_target,
            )
            per_pose.append(
                {
                    "index": index,
                    "frame_id": sample.get("frame_id"),
                    "translation_error_mm": translation_m * 1000.0,
                    "rotation_error_deg": rotation_deg,
                }
            )
        except ValueError as exc:
            failures.append(str(exc))

    translation_metrics: Dict[str, float] = {}
    rotation_metrics: Dict[str, float] = {}
    if per_pose:
        translation_values = np.asarray(
            [item["translation_error_mm"] for item in per_pose], dtype=np.float64
        )
        rotation_values = np.asarray(
            [item["rotation_error_deg"] for item in per_pose], dtype=np.float64
        )
        translation_metrics = _metrics(translation_values)
        rotation_metrics = _metrics(rotation_values)
        if translation_metrics["p95"] > max_translation_p95_mm:
            failures.append(
                f"holdout translation p95 {translation_metrics['p95']:.3f} mm "
                f"exceeds {max_translation_p95_mm:.3f} mm"
            )
        if rotation_metrics["p95"] > max_rotation_p95_deg:
            failures.append(
                f"holdout rotation p95 {rotation_metrics['p95']:.3f} deg "
                f"exceeds {max_rotation_p95_deg:.3f} deg"
            )

    return {
        "holdout_pass": not failures,
        "calibration": str(calibration_path.resolve()),
        "holdout": str(holdout_path.resolve()),
        "calibration_id": calibration.get("calibration_id"),
        "camera_serial": calibration_serial,
        "sample_count": len(per_pose),
        "limits": {
            "translation_p95_mm": max_translation_p95_mm,
            "rotation_p95_deg": max_rotation_p95_deg,
        },
        "translation_error_mm": translation_metrics,
        "rotation_error_deg": rotation_metrics,
        "per_pose": per_pose,
        "failures": failures,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate excluded eye-to-hand poses against a final calibration"
    )
    parser.add_argument("calibration", type=Path)
    parser.add_argument("holdout", type=Path)
    parser.add_argument("--minimum-samples", type=int, default=5)
    parser.add_argument("--max-translation-p95-mm", type=float, required=True)
    parser.add_argument("--max-rotation-p95-deg", type=float, required=True)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.minimum_samples < 1:
        raise SystemExit("--minimum-samples must be at least 1")
    for label, value in (
        ("--max-translation-p95-mm", args.max_translation_p95_mm),
        ("--max-rotation-p95-deg", args.max_rotation_p95_deg),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise SystemExit(f"{label} must be a finite positive number")
    result = validate(
        args.calibration,
        args.holdout,
        minimum_samples=args.minimum_samples,
        max_translation_p95_mm=args.max_translation_p95_mm,
        max_rotation_p95_deg=args.max_rotation_p95_deg,
    )
    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        status = "HOLDOUT PASS" if result["holdout_pass"] else "HOLDOUT FAIL"
        print(f"[{status}] {result.get('holdout')}")
        print(
            f"  id={result.get('calibration_id')} serial={result.get('camera_serial')} "
            f"samples={result.get('sample_count')}"
        )
        for name, values in (
            ("translation_mm", result.get("translation_error_mm") or {}),
            ("rotation_deg", result.get("rotation_error_deg") or {}),
        ):
            if values:
                print(
                    f"  {name}: median={values['median']:.3f} "
                    f"rms={values['rms']:.3f} p95={values['p95']:.3f} "
                    f"max={values['max']:.3f}"
                )
        for message in result["failures"]:
            print(f"  [FAIL] {message}")
    return 0 if result["holdout_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
