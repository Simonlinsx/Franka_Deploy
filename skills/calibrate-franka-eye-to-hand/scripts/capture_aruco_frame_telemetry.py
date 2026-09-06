#!/usr/bin/env python3
"""Capture per-frame ArUco telemetry from one fixed RealSense camera.

This is a camera-only, read-only diagnostic.  It opens exactly the requested
RealSense serial, applies one explicit ArUco specification, and writes the
unaggregated observations to JSON.  It neither imports a Franka interface nor
opens an FCI connection.

The coherence calculation deliberately mirrors the stationary calibration
collector: a deterministic SE(3) medoid is selected and candidate poses are
gated independently by translation and rotation.  This makes it possible to
inspect failures such as ``116/120 coherent`` without losing the four rejected
frames.  Supplying ``--abnormal-images-dir`` additionally keeps lossless raw
PNGs for only the frames classified as abnormal.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from dynamic_pcd.calibration.aruco import ArucoMarkerSpec, detect_aruco_marker
from dynamic_pcd.calibration.transforms import average_transforms, transform_error
from dynamic_pcd.camera.realsense_camera import RealSenseCamera
from dynamic_pcd.config import load_config


SCHEMA_VERSION = 1
KIND = "realsense_aruco_frame_telemetry"
DEFAULT_MAX_REPROJECTION_ERROR_PX = 1.0
DEFAULT_MAX_TRANSLATION_DEVIATION_M = 0.003
DEFAULT_MAX_ROTATION_DEVIATION_DEG = 1.0


def _finite_or_none(value: float) -> Optional[float]:
    number = float(value)
    return number if np.isfinite(number) else None


def _utc_timestamp(timestamp_s: float) -> str:
    return datetime.fromtimestamp(float(timestamp_s), timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _focus_laplacian_variance(image_bgr: np.ndarray) -> float:
    gray = (
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if image_bgr.ndim == 3
        else image_bgr
    )
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _frame_record(capture_index: int, frame: Any, detection: Any) -> Dict[str, Any]:
    transform = detection.T_camera_target
    timestamp_s = float(frame.timestamp)
    return {
        "capture_index": int(capture_index),
        "sequence_number": int(capture_index + 1),
        "frame_id": int(frame.frame_id),
        "host_timestamp_unix_s": timestamp_s,
        "host_timestamp_utc": _utc_timestamp(timestamp_s),
        "focus_laplacian_variance": _focus_laplacian_variance(frame.color_bgr),
        "detection": {
            "valid": bool(detection.valid),
            "marker_id": int(detection.marker_id),
            "corners_px": np.asarray(detection.corners, dtype=np.float64)
            .reshape(-1, 2)
            .tolist(),
            "reprojection_error_px": _finite_or_none(
                detection.reprojection_error_px
            ),
            "T_camera_target": (
                np.asarray(transform, dtype=np.float64).tolist()
                if transform is not None
                else None
            ),
            "message": str(detection.message),
        },
        "diagnostics": {
            "reprojection_gate_pass": False,
            "coherence_candidate": False,
            "translation_from_medoid_m": None,
            "rotation_from_medoid_deg": None,
            "coherent_pose_inlier": False,
            "abnormal_reasons": [],
            "raw_image": None,
        },
    }


def analyze_pose_coherence(
    frame_records: Sequence[Mapping[str, Any]],
    *,
    max_reprojection_error_px: float,
    max_translation_deviation_m: float,
    max_rotation_deviation_deg: float,
) -> Dict[str, Any]:
    """Classify frame poses using the collector's deterministic medoid gate.

    The input mappings are mutated only through their nested ``diagnostics``
    dictionaries.  Returning a useful result for zero or one valid candidates
    is intentional: this is telemetry, not an acceptance gate.
    """

    if not np.isfinite(max_reprojection_error_px) or max_reprojection_error_px < 0:
        raise ValueError("max_reprojection_error_px must be finite and non-negative")
    if (
        not np.isfinite(max_translation_deviation_m)
        or max_translation_deviation_m <= 0
    ):
        raise ValueError("max_translation_deviation_m must be positive")
    if (
        not np.isfinite(max_rotation_deviation_deg)
        or max_rotation_deviation_deg <= 0
    ):
        raise ValueError("max_rotation_deviation_deg must be positive")

    candidate_indices: List[int] = []
    transforms: List[np.ndarray] = []
    reprojection_errors: List[float] = []
    detected_valid_count = 0
    invalid_detection_indices: List[int] = []
    reprojection_outlier_indices: List[int] = []

    for frame_index, record in enumerate(frame_records):
        detection = record["detection"]
        diagnostics = record["diagnostics"]
        if not bool(detection["valid"]) or detection["T_camera_target"] is None:
            invalid_detection_indices.append(frame_index)
            diagnostics["abnormal_reasons"].append("detection_invalid")
            continue
        detected_valid_count += 1
        reprojection = detection["reprojection_error_px"]
        if reprojection is None or float(reprojection) > max_reprojection_error_px:
            reprojection_outlier_indices.append(frame_index)
            diagnostics["abnormal_reasons"].append("reprojection_gate_failed")
            continue
        diagnostics["reprojection_gate_pass"] = True
        diagnostics["coherence_candidate"] = True
        candidate_indices.append(frame_index)
        transforms.append(np.asarray(detection["T_camera_target"], dtype=np.float64))
        reprojection_errors.append(float(reprojection))

    if not transforms:
        return {
            "detected_valid_count": detected_valid_count,
            "reprojection_gate_pass_count": 0,
            "coherent_pose_count": 0,
            "coherent_pose_fraction": 0.0,
            "medoid_capture_index": None,
            "coherent_capture_indices": [],
            "coherence_outlier_capture_indices": [],
            "invalid_detection_capture_indices": invalid_detection_indices,
            "reprojection_outlier_capture_indices": reprojection_outlier_indices,
            "aggregate_T_camera_target": None,
            "translation_jitter_p95_m": None,
            "rotation_jitter_p95_deg": None,
            "inlier_reprojection_error_p95_px": None,
            "all_reprojection_pass_translation_jitter_p95_m": None,
            "all_reprojection_pass_rotation_jitter_p95_deg": None,
            "all_reprojection_pass_reprojection_error_p95_px": None,
        }

    count = len(transforms)
    pairwise_cost = np.zeros((count, count), dtype=np.float64)
    rotation_limit_rad = float(np.radians(max_rotation_deviation_deg))
    for left in range(count):
        for right in range(left + 1, count):
            translation_m, rotation_rad = transform_error(
                transforms[left], transforms[right]
            )
            normalized = np.hypot(
                translation_m / max_translation_deviation_m,
                rotation_rad / rotation_limit_rad,
            )
            pairwise_cost[left, right] = normalized
            pairwise_cost[right, left] = normalized

    medoid_position = min(
        range(count),
        key=lambda index: (
            float(np.sum(pairwise_cost[index])),
            reprojection_errors[index],
            index,
        ),
    )
    medoid = transforms[medoid_position]
    # Keep a second, deliberately untrimmed view of the stationary-pose
    # observations.  The coherent aggregate below is useful for collection,
    # but its inlier-only p95 values cannot reveal a minority of unstable PnP
    # poses.  These residuals therefore include every valid frame that passed
    # the independent reprojection gate and use the deterministic medoid as
    # their common reference.
    all_translation_residuals: List[float] = []
    all_rotation_residuals_deg: List[float] = []
    for transform in transforms:
        translation_m, rotation_rad = transform_error(medoid, transform)
        all_translation_residuals.append(float(translation_m))
        all_rotation_residuals_deg.append(float(np.degrees(rotation_rad)))

    inlier_positions: List[int] = []
    coherence_outlier_indices: List[int] = []
    for position, (frame_index, transform) in enumerate(
        zip(candidate_indices, transforms)
    ):
        translation_m, rotation_rad = transform_error(medoid, transform)
        diagnostics = frame_records[frame_index]["diagnostics"]
        diagnostics["translation_from_medoid_m"] = float(translation_m)
        diagnostics["rotation_from_medoid_deg"] = float(np.degrees(rotation_rad))
        if (
            translation_m <= max_translation_deviation_m
            and rotation_rad <= rotation_limit_rad
        ):
            diagnostics["coherent_pose_inlier"] = True
            inlier_positions.append(position)
        else:
            diagnostics["abnormal_reasons"].append("pose_coherence_outlier")
            coherence_outlier_indices.append(frame_index)

    aggregate = average_transforms([transforms[index] for index in inlier_positions])
    translation_residuals: List[float] = []
    rotation_residuals_deg: List[float] = []
    for position in inlier_positions:
        translation_m, rotation_rad = transform_error(aggregate, transforms[position])
        translation_residuals.append(float(translation_m))
        rotation_residuals_deg.append(float(np.degrees(rotation_rad)))

    coherent_indices = [candidate_indices[index] for index in inlier_positions]
    return {
        "detected_valid_count": detected_valid_count,
        "reprojection_gate_pass_count": count,
        "coherent_pose_count": len(coherent_indices),
        "coherent_pose_fraction": (
            float(len(coherent_indices)) / float(len(frame_records))
            if frame_records
            else 0.0
        ),
        "medoid_capture_index": candidate_indices[medoid_position],
        "coherent_capture_indices": coherent_indices,
        "coherence_outlier_capture_indices": coherence_outlier_indices,
        "invalid_detection_capture_indices": invalid_detection_indices,
        "reprojection_outlier_capture_indices": reprojection_outlier_indices,
        "aggregate_T_camera_target": np.asarray(aggregate).tolist(),
        "translation_jitter_p95_m": float(
            np.percentile(translation_residuals, 95)
        ),
        "rotation_jitter_p95_deg": float(
            np.percentile(rotation_residuals_deg, 95)
        ),
        "inlier_reprojection_error_p95_px": float(
            np.percentile(
                [reprojection_errors[index] for index in inlier_positions], 95
            )
        ),
        "all_reprojection_pass_translation_jitter_p95_m": float(
            np.percentile(all_translation_residuals, 95)
        ),
        "all_reprojection_pass_rotation_jitter_p95_deg": float(
            np.percentile(all_rotation_residuals_deg, 95)
        ),
        "all_reprojection_pass_reprojection_error_p95_px": float(
            np.percentile(reprojection_errors, 95)
        ),
    }


def _intrinsics_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        if key == "distortion":
            if not np.allclose(left[key], right[key], rtol=0.0, atol=1.0e-12):
                return False
        elif isinstance(left[key], (int, float)):
            if not np.isclose(left[key], right[key], rtol=0.0, atol=1.0e-12):
                return False
        elif left[key] != right[key]:
            return False
    return True


def capture_telemetry(
    *,
    config_path: Path,
    camera_serial: str,
    target_spec: ArucoMarkerSpec,
    frame_count: int,
    output_path: Path,
    max_reprojection_error_px: float,
    max_translation_deviation_m: float,
    max_rotation_deviation_deg: float,
    minimum_focus: Optional[float] = None,
    abnormal_images_dir: Optional[Path] = None,
    timeout_ms: int = 1000,
    camera_factory: Any = RealSenseCamera,
    detector: Any = detect_aruco_marker,
) -> Dict[str, Any]:
    """Capture telemetry; injectable dependencies keep tests hardware-free."""

    if frame_count < 1:
        raise ValueError("frame_count must be at least 1")
    if timeout_ms < 1:
        raise ValueError("timeout_ms must be at least 1")
    if minimum_focus is not None and (
        not np.isfinite(minimum_focus) or minimum_focus < 0
    ):
        raise ValueError("minimum_focus must be finite and non-negative")
    if output_path.exists():
        raise FileExistsError("refusing to overwrite {}".format(output_path))
    if abnormal_images_dir is not None and abnormal_images_dir.exists():
        raise FileExistsError(
            "refusing to reuse abnormal image directory {}".format(
                abnormal_images_dir
            )
        )

    cfg = load_config(str(config_path))
    camera_cfg = dict(cfg["camera"])
    camera_cfg["serial"] = str(camera_serial)
    camera = camera_factory(camera_cfg)
    frame_records: List[Dict[str, Any]] = []
    cached_images: Optional[List[np.ndarray]] = (
        [] if abnormal_images_dir is not None else None
    )
    first_intrinsics: Optional[Dict[str, Any]] = None
    intrinsics_consistent = True
    started_unix_s = time.time()
    try:
        camera.start()
        if camera.device_serial != str(camera_serial):
            raise RuntimeError(
                "requested serial {}, opened {}".format(
                    camera_serial, camera.device_serial
                )
            )
        for capture_index in range(frame_count):
            frame = camera.get_frame(timeout_ms=timeout_ms)
            intrinsics = frame.intrinsics.to_dict()
            if first_intrinsics is None:
                first_intrinsics = intrinsics
            elif not _intrinsics_equal(first_intrinsics, intrinsics):
                intrinsics_consistent = False
            detection = detector(frame.color_bgr, frame.intrinsics, target_spec)
            record = _frame_record(capture_index, frame, detection)
            if minimum_focus is not None and (
                record["focus_laplacian_variance"] < minimum_focus
            ):
                record["diagnostics"]["abnormal_reasons"].append(
                    "focus_below_threshold"
                )
            frame_records.append(record)
            if cached_images is not None:
                cached_images.append(frame.color_bgr.copy())
    finally:
        camera.stop()
    finished_unix_s = time.time()

    if first_intrinsics is None:
        raise RuntimeError("camera produced no valid frames")
    coherence = analyze_pose_coherence(
        frame_records,
        max_reprojection_error_px=max_reprojection_error_px,
        max_translation_deviation_m=max_translation_deviation_m,
        max_rotation_deviation_deg=max_rotation_deviation_deg,
    )

    saved_images: List[str] = []
    if abnormal_images_dir is not None:
        assert cached_images is not None
        abnormal_images_dir.mkdir(parents=True, exist_ok=False)
        for record, image in zip(frame_records, cached_images):
            if not record["diagnostics"]["abnormal_reasons"]:
                continue
            image_path = abnormal_images_dir / "frame-{:06d}-id-{:06d}.png".format(
                int(record["capture_index"]), int(record["frame_id"])
            )
            if not cv2.imwrite(str(image_path), image):
                raise OSError("failed to write {}".format(image_path))
            resolved = str(image_path.resolve())
            record["diagnostics"]["raw_image"] = resolved
            saved_images.append(resolved)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "scope": {
            "camera_only": True,
            "read_only": True,
            "franka_fci_opened": False,
            "robot_motion_commanded": False,
        },
        "camera": {
            "requested_serial": str(camera_serial),
            "opened_serial": str(camera.device_serial),
            "name": camera.device_name,
            "depth_scale": float(camera.depth_scale),
            "intrinsics": first_intrinsics,
            "intrinsics_consistent_across_frames": intrinsics_consistent,
            "config_path": str(config_path.resolve()),
        },
        "target": target_spec.to_dict(),
        "capture": {
            "requested_frame_count": int(frame_count),
            "captured_frame_count": len(frame_records),
            "started_host_timestamp_unix_s": float(started_unix_s),
            "started_host_timestamp_utc": _utc_timestamp(started_unix_s),
            "finished_host_timestamp_unix_s": float(finished_unix_s),
            "finished_host_timestamp_utc": _utc_timestamp(finished_unix_s),
            "timeout_ms": int(timeout_ms),
        },
        "thresholds": {
            "max_reprojection_error_px": float(max_reprojection_error_px),
            "max_translation_deviation_m": float(max_translation_deviation_m),
            "max_rotation_deviation_deg": float(max_rotation_deviation_deg),
            "minimum_focus_laplacian_variance": (
                float(minimum_focus) if minimum_focus is not None else None
            ),
        },
        "coherence": coherence,
        "abnormal_images": saved_images,
        "frames": frame_records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--camera-serial", required=True)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--marker-id", type=int, required=True)
    parser.add_argument(
        "--marker-length-m",
        "--marker-length",
        dest="marker_length_m",
        type=float,
        required=True,
        help="measured outer black marker side in metres",
    )
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-reprojection-error-px",
        type=float,
        default=DEFAULT_MAX_REPROJECTION_ERROR_PX,
    )
    parser.add_argument(
        "--max-translation-deviation-m",
        type=float,
        default=DEFAULT_MAX_TRANSLATION_DEVIATION_M,
    )
    parser.add_argument(
        "--max-rotation-deviation-deg",
        type=float,
        default=DEFAULT_MAX_ROTATION_DEVIATION_DEG,
    )
    parser.add_argument(
        "--minimum-focus-laplacian-variance",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--abnormal-images-dir",
        type=Path,
        default=None,
        help="optional new directory for raw PNGs of abnormal frames only",
    )
    parser.add_argument("--timeout-ms", type=int, default=1000)
    return parser


def main() -> int:
    args = _parser().parse_args()
    target_spec = ArucoMarkerSpec(
        dictionary=args.dictionary,
        marker_id=args.marker_id,
        marker_length_m=args.marker_length_m,
    )
    payload = capture_telemetry(
        config_path=args.config,
        camera_serial=str(args.camera_serial),
        target_spec=target_spec,
        frame_count=args.frames,
        output_path=args.output,
        max_reprojection_error_px=args.max_reprojection_error_px,
        max_translation_deviation_m=args.max_translation_deviation_m,
        max_rotation_deviation_deg=args.max_rotation_deviation_deg,
        minimum_focus=args.minimum_focus_laplacian_variance,
        abnormal_images_dir=args.abnormal_images_dir,
        timeout_ms=args.timeout_ms,
    )
    coherence = payload["coherence"]
    print(
        "ARUCO_FRAME_TELEMETRY_JSON="
        + json.dumps(
            {
                "output": str(args.output.resolve()),
                "camera_serial": payload["camera"]["opened_serial"],
                "captured": payload["capture"]["captured_frame_count"],
                "valid": coherence["detected_valid_count"],
                "reprojection_gate_pass": coherence[
                    "reprojection_gate_pass_count"
                ],
                "coherent": coherence["coherent_pose_count"],
                "abnormal_images": len(payload["abnormal_images"]),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
