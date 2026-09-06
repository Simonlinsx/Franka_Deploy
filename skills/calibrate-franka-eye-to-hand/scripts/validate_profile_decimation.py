#!/usr/bin/env python3
"""Compare ArUco PnP before and after an exact 2x RGB profile decimation.

This is an offline diagnostic only.  It never opens a camera or robot.  The
result can support reusing an unchanged eye-to-hand extrinsic when live
intrinsics are the exact half-resolution counterpart, but it cannot replace
native-profile physical holdouts or aligned-depth validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping, Sequence

import cv2
import numpy as np
import yaml


WORKSPACE = Path(__file__).resolve().parents[3]
CALIBRATION_ROOT = WORKSPACE / "beta" / "dynamic_object_pcd"
if str(CALIBRATION_ROOT) not in sys.path:
    sys.path.insert(0, str(CALIBRATION_ROOT))

from dynamic_pcd.calibration.aruco import (  # noqa: E402
    ArucoMarkerSpec,
    detect_aruco_marker,
)
from dynamic_pcd.calibration.transforms import transform_error  # noqa: E402
from dynamic_pcd.types import CameraIntrinsics  # noqa: E402


class ProfileDecimationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProfileDecimationError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProfileDecimationError(f"{path} must contain a YAML mapping")
    return value


def _intrinsics(record: Mapping[str, Any]) -> CameraIntrinsics:
    try:
        return CameraIntrinsics(
            width=int(record["width"]),
            height=int(record["height"]),
            fx=float(record["fx"]),
            fy=float(record["fy"]),
            ppx=float(record["ppx"]),
            ppy=float(record["ppy"]),
            model=str(record.get("model", "")),
            distortion=tuple(float(x) for x in record.get("distortion", ())),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProfileDecimationError("invalid calibration intrinsics") from exc


def _validate_contract(
    parent: Mapping[str, Any], derived: Mapping[str, Any]
) -> tuple[CameraIntrinsics, CameraIntrinsics, ArucoMarkerSpec]:
    for name, document in (("parent", parent), ("derived", derived)):
        if document.get("kind") != "camera_robot_extrinsic_calibration":
            raise ProfileDecimationError(f"{name} is not an extrinsic calibration")
        if document.get("calibration_type") != "eye_to_hand":
            raise ProfileDecimationError(f"{name} is not eye_to_hand")
    parent_camera = parent.get("camera") or {}
    derived_camera = derived.get("camera") or {}
    if str(parent_camera.get("serial")) != str(derived_camera.get("serial")):
        raise ProfileDecimationError("parent and derived camera serials differ")
    parent_intrinsics = _intrinsics(parent_camera.get("intrinsics") or {})
    derived_intrinsics = _intrinsics(derived_camera.get("intrinsics") or {})
    if (
        parent_intrinsics.width != 2 * derived_intrinsics.width
        or parent_intrinsics.height != 2 * derived_intrinsics.height
    ):
        raise ProfileDecimationError("resolution is not an exact 2x decimation")
    for name in ("fx", "fy", "ppx", "ppy"):
        parent_value = float(getattr(parent_intrinsics, name))
        derived_value = float(getattr(derived_intrinsics, name))
        if not math.isclose(parent_value / 2.0, derived_value, abs_tol=1e-9):
            raise ProfileDecimationError(
                f"derived {name} is not exactly parent {name}/2"
            )
    parent_target = parent.get("target") or {}
    derived_target = derived.get("target") or {}
    target_fields = ("type", "dictionary", "marker_id", "marker_length_m")
    if any(parent_target.get(key) != derived_target.get(key) for key in target_fields):
        raise ProfileDecimationError("parent and derived target contracts differ")
    if parent_target.get("type") != "aruco":
        raise ProfileDecimationError("only a single ArUco target is supported")
    spec = ArucoMarkerSpec(
        dictionary=str(parent_target["dictionary"]),
        marker_id=int(parent_target["marker_id"]),
        marker_length_m=float(parent_target["marker_length_m"]),
    )
    return parent_intrinsics, derived_intrinsics, spec


def _metrics(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ProfileDecimationError("metrics require finite non-empty values")
    return {
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise ProfileDecimationError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{os.getpid()}"
    )
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


def evaluate(
    *,
    parent_path: Path,
    derived_path: Path,
    image_paths: Sequence[Path],
    max_translation_p95_mm: float,
    max_rotation_p95_deg: float,
    max_scaled_corner_p95_px: float,
    max_derived_reprojection_p95_px: float,
) -> Dict[str, Any]:
    if len(image_paths) < 5:
        raise ProfileDecimationError("at least five independent images are required")
    parent = _load_yaml(parent_path)
    derived = _load_yaml(derived_path)
    parent_k, derived_k, target = _validate_contract(parent, derived)
    records = []
    failures = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ProfileDecimationError(f"cannot read image {image_path}")
        if image.shape[:2] != (parent_k.height, parent_k.width):
            raise ProfileDecimationError(
                f"{image_path} is {image.shape[1]}x{image.shape[0]}, expected "
                f"{parent_k.width}x{parent_k.height}"
            )
        reduced = cv2.resize(
            image,
            (derived_k.width, derived_k.height),
            interpolation=cv2.INTER_AREA,
        )
        full = detect_aruco_marker(image, parent_k, target)
        half = detect_aruco_marker(reduced, derived_k, target)
        if (
            not full.valid
            or not half.valid
            or full.T_camera_target is None
            or half.T_camera_target is None
        ):
            failures.append(
                f"target detection failed for {image_path.name}: "
                f"full={full.message}; half={half.message}"
            )
            continue
        translation_m, rotation_rad = transform_error(
            full.T_camera_target, half.T_camera_target
        )
        scaled_full_corners = np.asarray(full.corners, dtype=np.float64) / 2.0
        half_corners = np.asarray(half.corners, dtype=np.float64)
        corner_error = np.linalg.norm(scaled_full_corners - half_corners, axis=1)
        records.append(
            {
                "image": str(image_path.resolve()),
                "image_sha256": _sha256(image_path),
                "full_reprojection_error_px": float(full.reprojection_error_px),
                "derived_reprojection_error_px": float(half.reprojection_error_px),
                "translation_delta_m": float(translation_m),
                "rotation_delta_deg": float(np.degrees(rotation_rad)),
                "scaled_corner_error_px": {
                    "median": float(np.median(corner_error)),
                    "p95": float(np.percentile(corner_error, 95)),
                    "max": float(np.max(corner_error)),
                },
            }
        )
    if len(records) != len(image_paths):
        failures.append(f"valid pairs {len(records)}/{len(image_paths)}")
    aggregate: Dict[str, Any] = {}
    if records:
        aggregate = {
            "translation_delta_m": _metrics(
                record["translation_delta_m"] for record in records
            ),
            "rotation_delta_deg": _metrics(
                record["rotation_delta_deg"] for record in records
            ),
            "scaled_corner_p95_px": _metrics(
                record["scaled_corner_error_px"]["p95"] for record in records
            ),
            "derived_reprojection_error_px": _metrics(
                record["derived_reprojection_error_px"] for record in records
            ),
        }
        checks = (
            (
                aggregate["translation_delta_m"]["p95"] * 1000.0
                <= max_translation_p95_mm,
                "translation delta p95",
            ),
            (
                aggregate["rotation_delta_deg"]["p95"] <= max_rotation_p95_deg,
                "rotation delta p95",
            ),
            (
                aggregate["scaled_corner_p95_px"]["p95"]
                <= max_scaled_corner_p95_px,
                "scaled corner p95",
            ),
            (
                aggregate["derived_reprojection_error_px"]["p95"]
                <= max_derived_reprojection_p95_px,
                "derived reprojection p95",
            ),
        )
        failures.extend(label for passed, label in checks if not passed)
    return {
        "schema_version": 1,
        "kind": "eye_to_hand_profile_decimation_diagnostic",
        "decision": "diagnostic_pass" if not failures else "diagnostic_fail",
        "scope": {
            "offline_only": True,
            "camera_opened": False,
            "franka_opened": False,
            "rh56_opened": False,
            "robot_motion": False,
            "physical_holdout_replacement": False,
        },
        "parent_calibration": {
            "path": str(parent_path.resolve()),
            "sha256": _sha256(parent_path),
        },
        "derived_calibration": {
            "path": str(derived_path.resolve()),
            "sha256": _sha256(derived_path),
        },
        "target": target.to_dict(),
        "thresholds": {
            "max_translation_p95_mm": max_translation_p95_mm,
            "max_rotation_p95_deg": max_rotation_p95_deg,
            "max_scaled_corner_p95_px": max_scaled_corner_p95_px,
            "max_derived_reprojection_p95_px": max_derived_reprojection_p95_px,
        },
        "sample_count": len(records),
        "aggregate": aggregate,
        "failures": failures,
        "records": records,
        "note": (
            "OpenCV INTER_AREA downsampling is a profile-equivalence diagnostic; "
            "native 424x240@60 physical holdouts remain mandatory."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-calibration", type=Path, required=True)
    parser.add_argument("--derived-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-translation-p95-mm", type=float, default=5.0)
    parser.add_argument("--max-rotation-p95-deg", type=float, default=1.0)
    parser.add_argument("--max-scaled-corner-p95-px", type=float, default=1.5)
    parser.add_argument("--max-derived-reprojection-p95-px", type=float, default=1.0)
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        result = evaluate(
            parent_path=args.parent_calibration.resolve(),
            derived_path=args.derived_calibration.resolve(),
            image_paths=tuple(path.resolve() for path in args.images),
            max_translation_p95_mm=args.max_translation_p95_mm,
            max_rotation_p95_deg=args.max_rotation_p95_deg,
            max_scaled_corner_p95_px=args.max_scaled_corner_p95_px,
            max_derived_reprojection_p95_px=args.max_derived_reprojection_p95_px,
        )
        _write_json_exclusive(args.output, result)
    except (ProfileDecimationError, ValueError, OSError, cv2.error) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    print(
        "PROFILE_DECIMATION_JSON="
        + json.dumps(
            {
                "decision": result["decision"],
                "output": str(args.output.resolve()),
                "sample_count": result["sample_count"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0 if result["decision"] == "diagnostic_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
