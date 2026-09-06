#!/usr/bin/env python3
"""Validate an unchanged T_base_camera after a calibration target remount.

The first stationary pose defines a session-local ``T_ee_target`` nuisance
transform.  It is excluded from the score.  The remaining independent poses
test whether the frozen ``T_base_camera`` closes under robot translations and
rotations.  This is a holdout validator, not a hand-eye solver: it never changes
or estimates ``T_base_camera`` and it never commands robot motion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import yaml


class ReusedExtrinsicHoldoutError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ReusedExtrinsicHoldoutError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReusedExtrinsicHoldoutError(f"{path} must contain a YAML mapping")
    return value


def _transform(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ReusedExtrinsicHoldoutError(f"{label} must be finite 4x4")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5, rtol=0.0):
        raise ReusedExtrinsicHoldoutError(f"{label} has invalid last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0.0):
        raise ReusedExtrinsicHoldoutError(f"{label} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5, rtol=0.0):
        raise ReusedExtrinsicHoldoutError(f"{label} rotation determinant is not +1")
    return matrix


def _error(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    delta = np.linalg.inv(first) @ second
    translation_m = float(np.linalg.norm(delta[:3, 3]))
    cosine = float(np.clip((np.trace(delta[:3, :3]) - 1.0) * 0.5, -1.0, 1.0))
    return translation_m, float(np.degrees(np.arccos(cosine)))


def _metrics(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ReusedExtrinsicHoldoutError("metric vector is empty or non-finite")
    return {
        "median": float(np.median(array)),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _motion_span(transforms: Sequence[np.ndarray]) -> tuple[float, float]:
    maximum_translation_m = 0.0
    maximum_rotation_deg = 0.0
    for left in range(len(transforms)):
        for right in range(left + 1, len(transforms)):
            translation_m, rotation_deg = _error(transforms[left], transforms[right])
            maximum_translation_m = max(maximum_translation_m, translation_m)
            maximum_rotation_deg = max(maximum_rotation_deg, rotation_deg)
    return maximum_translation_m, maximum_rotation_deg


def validate(
    calibration_path: Path,
    dataset_path: Path,
    *,
    minimum_holdout_samples: int,
    minimum_translation_span_m: float,
    minimum_rotation_span_deg: float,
    max_translation_p95_mm: float,
    max_rotation_p95_deg: float,
) -> Dict[str, Any]:
    calibration = _load(calibration_path)
    dataset = _load(dataset_path)
    if calibration.get("kind") != "camera_robot_extrinsic_calibration":
        raise ReusedExtrinsicHoldoutError("invalid calibration kind")
    if calibration.get("calibration_type") != "eye_to_hand":
        raise ReusedExtrinsicHoldoutError("calibration is not eye_to_hand")
    if dataset.get("kind") != "eye_to_hand_dataset":
        raise ReusedExtrinsicHoldoutError("invalid holdout dataset kind")
    calibration_camera = calibration.get("camera") or {}
    dataset_camera = dataset.get("camera") or {}
    if str(calibration_camera.get("serial") or "") != str(
        dataset_camera.get("serial") or ""
    ):
        raise ReusedExtrinsicHoldoutError("camera serial mismatch")
    calibration_target = calibration.get("target") or {}
    dataset_target = dataset.get("target") or {}
    for key in ("type", "dictionary", "marker_id", "marker_length_m"):
        if calibration_target.get(key) != dataset_target.get(key):
            raise ReusedExtrinsicHoldoutError(f"target contract mismatch: {key}")
    expected_intrinsics = calibration_camera.get("intrinsics") or {}
    observed_intrinsics = dataset_camera.get("intrinsics") or {}
    for key in ("width", "height"):
        if int(expected_intrinsics.get(key, -1)) != int(
            observed_intrinsics.get(key, -2)
        ):
            raise ReusedExtrinsicHoldoutError(f"intrinsics mismatch: {key}")
    for key in ("fx", "fy", "ppx", "ppy"):
        if not np.isclose(
            float(expected_intrinsics.get(key, math.nan)),
            float(observed_intrinsics.get(key, math.nan)),
            atol=0.05,
            rtol=0.0,
        ):
            raise ReusedExtrinsicHoldoutError(f"intrinsics mismatch: {key}")
    samples = dataset.get("samples")
    if not isinstance(samples, list):
        raise ReusedExtrinsicHoldoutError("dataset.samples must be a list")
    required_total = minimum_holdout_samples + 1
    if len(samples) < required_total:
        raise ReusedExtrinsicHoldoutError(
            f"dataset has {len(samples)} samples; need one reference plus "
            f"{minimum_holdout_samples} holdouts"
        )
    T_base_camera = _transform(
        calibration.get("T_base_camera"), "calibration.T_base_camera"
    )
    transforms_base_ee = []
    transforms_camera_target = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ReusedExtrinsicHoldoutError(f"sample[{index}] must be a mapping")
        transforms_base_ee.append(
            _transform(sample.get("T_base_ee"), f"sample[{index}].T_base_ee")
        )
        transforms_camera_target.append(
            _transform(
                sample.get("T_camera_target"),
                f"sample[{index}].T_camera_target",
            )
        )
    # This one nuisance transform is established before scoring.  No element
    # of T_base_camera is fitted or modified.
    T_ee_target_session = (
        np.linalg.inv(transforms_base_ee[0])
        @ T_base_camera
        @ transforms_camera_target[0]
    )
    holdout_base_ee = transforms_base_ee[1:]
    translation_span_m, rotation_span_deg = _motion_span(holdout_base_ee)
    failures = []
    if translation_span_m < minimum_translation_span_m:
        failures.append(
            f"translation span {translation_span_m:.6f} m is below "
            f"{minimum_translation_span_m:.6f} m"
        )
    if rotation_span_deg < minimum_rotation_span_deg:
        failures.append(
            f"rotation span {rotation_span_deg:.6f} deg is below "
            f"{minimum_rotation_span_deg:.6f} deg"
        )
    per_pose = []
    for index in range(1, len(samples)):
        translation_m, rotation_deg = _error(
            transforms_base_ee[index] @ T_ee_target_session,
            T_base_camera @ transforms_camera_target[index],
        )
        per_pose.append(
            {
                "dataset_index": index,
                "frame_id": samples[index].get("frame_id"),
                "translation_error_mm": translation_m * 1000.0,
                "rotation_error_deg": rotation_deg,
            }
        )
    translation = _metrics([record["translation_error_mm"] for record in per_pose])
    rotation = _metrics([record["rotation_error_deg"] for record in per_pose])
    if translation["p95"] > max_translation_p95_mm:
        failures.append(
            f"translation p95 {translation['p95']:.3f} mm exceeds "
            f"{max_translation_p95_mm:.3f} mm"
        )
    if rotation["p95"] > max_rotation_p95_deg:
        failures.append(
            f"rotation p95 {rotation['p95']:.3f} deg exceeds "
            f"{max_rotation_p95_deg:.3f} deg"
        )
    return {
        "schema_version": 1,
        "kind": "reused_eye_to_hand_extrinsic_session_holdout",
        "decision": "holdout_pass" if not failures else "holdout_fail",
        "calibration": {
            "path": str(calibration_path.resolve()),
            "sha256": _sha256(calibration_path),
            "calibration_id": calibration.get("calibration_id"),
            "T_base_camera_reused_unchanged": True,
        },
        "dataset": {
            "path": str(dataset_path.resolve()),
            "sha256": _sha256(dataset_path),
            "sample_count": len(samples),
            "reference_sample_index": 0,
            "holdout_sample_count": len(per_pose),
        },
        "scope": {
            "offline_only": True,
            "robot_motion_commanded": False,
            "T_base_camera_fitted": False,
            "session_T_ee_target_fitted_from_reference_only": True,
            "reference_sample_excluded_from_metrics": True,
        },
        "T_ee_target_session": T_ee_target_session.tolist(),
        "motion_diversity": {
            "translation_span_m": translation_span_m,
            "rotation_span_deg": rotation_span_deg,
        },
        "limits": {
            "minimum_holdout_samples": minimum_holdout_samples,
            "minimum_translation_span_m": minimum_translation_span_m,
            "minimum_rotation_span_deg": minimum_rotation_span_deg,
            "max_translation_p95_mm": max_translation_p95_mm,
            "max_rotation_p95_deg": max_rotation_p95_deg,
        },
        "translation_error_mm": translation,
        "rotation_error_deg": rotation,
        "per_pose": per_pose,
        "failures": failures,
    }


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise ReusedExtrinsicHoldoutError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-holdout-samples", type=int, default=5)
    parser.add_argument("--minimum-translation-span-m", type=float, default=0.08)
    parser.add_argument("--minimum-rotation-span-deg", type=float, default=20.0)
    parser.add_argument("--max-translation-p95-mm", type=float, default=10.0)
    parser.add_argument("--max-rotation-p95-deg", type=float, default=1.5)
    args = parser.parse_args()
    try:
        result = validate(
            args.calibration.resolve(),
            args.dataset.resolve(),
            minimum_holdout_samples=args.minimum_holdout_samples,
            minimum_translation_span_m=args.minimum_translation_span_m,
            minimum_rotation_span_deg=args.minimum_rotation_span_deg,
            max_translation_p95_mm=args.max_translation_p95_mm,
            max_rotation_p95_deg=args.max_rotation_p95_deg,
        )
        _write_json_exclusive(args.output, result)
    except (ReusedExtrinsicHoldoutError, ValueError, OSError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    print(
        "REUSED_EXTRINSIC_HOLDOUT_JSON="
        + json.dumps(
            {
                "decision": result["decision"],
                "output": str(args.output.resolve()),
                "holdout_sample_count": result["dataset"]["holdout_sample_count"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0 if result["decision"] == "holdout_pass" else 1


if __name__ == "__main__":
    sys.exit(main())
