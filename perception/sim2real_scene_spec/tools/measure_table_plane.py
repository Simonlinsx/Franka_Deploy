#!/usr/bin/env python3
"""Measure the black tabletop plane in ``robot_base`` using only the D435.

This tool never imports or connects to a robot API.  It captures aligned D435
color/depth frames, verifies the calibrated device/profile, back-projects dark
pixels inside an explicit ROI, transforms them through ``T_base_camera``, and
fits a near-horizontal plane with RANSAC followed by an SVD refinement.

Do not run it while the robot or workcell is moving.  The output is a candidate
survey that must be visually reviewed before it is copied into the canonical
scene manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION = PACKAGE_ROOT / "source" / "calibration" / "fr3_d435_eye_to_hand.yaml"
DEFAULT_OUTPUT = PACKAGE_ROOT / "measurements" / "table_plane_measurement.yaml"


class MeasurementError(RuntimeError):
    pass


@dataclass(frozen=True)
class PlaneFit:
    coefficients: np.ndarray
    inliers: np.ndarray
    residuals_m: np.ndarray
    rms_m: float
    p95_m: float
    max_m: float
    tilt_deg: float


@dataclass(frozen=True)
class CaptureResult:
    points_base: np.ndarray
    per_frame_candidate_counts: Tuple[int, ...]
    roi_xyxy: Tuple[int, int, int, int]
    actual_intrinsics: Mapping[str, Any]
    depth_scale_m_per_unit: float
    camera_name: str
    camera_serial: str
    started_at: str
    completed_at: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_calibration(path: Path) -> Mapping[str, Any]:
    try:
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MeasurementError(f"cannot load calibration {path}: {exc}") from exc
    if not isinstance(values, dict):
        raise MeasurementError("calibration must contain a YAML mapping")
    if values.get("kind") != "camera_robot_extrinsic_calibration":
        raise MeasurementError("calibration kind is not camera_robot_extrinsic_calibration")
    if values.get("calibration_type") != "eye_to_hand":
        raise MeasurementError("only the fixed-camera eye_to_hand calibration is supported")
    if values.get("base_frame") != "robot_base":
        raise MeasurementError("calibration base frame must be robot_base")
    if values.get("camera_frame") != "camera_color_optical_frame":
        raise MeasurementError("calibration camera frame must be camera_color_optical_frame")

    transform = np.asarray(values.get("T_base_camera"), dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise MeasurementError("T_base_camera must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9, rtol=0.0):
        raise MeasurementError("T_base_camera has an invalid homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0.0):
        raise MeasurementError("T_base_camera rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8, rtol=0.0):
        raise MeasurementError("T_base_camera rotation determinant is not +1")

    camera = values.get("camera")
    if not isinstance(camera, dict) or not camera.get("serial"):
        raise MeasurementError("calibration has no camera serial")
    intrinsics = camera.get("intrinsics")
    if not isinstance(intrinsics, dict):
        raise MeasurementError("calibration has no camera intrinsics")
    for key in ("width", "height", "fx", "fy", "ppx", "ppy"):
        if key not in intrinsics:
            raise MeasurementError(f"calibration intrinsics missing {key}")
    return values


def _plane_from_three(points: np.ndarray) -> Optional[np.ndarray]:
    p0, p1, p2 = points
    normal = np.cross(p1 - p0, p2 - p0)
    norm = float(np.linalg.norm(normal))
    if not math.isfinite(norm) or norm < 1e-9:
        return None
    normal = normal / norm
    if normal[2] < 0.0:
        normal = -normal
    return np.r_[normal, -float(normal @ p0)]


def _refine_plane(points: np.ndarray) -> np.ndarray:
    center = np.mean(points, axis=0)
    covariance = (points - center).T @ (points - center) / max(1, len(points))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.all(np.isfinite(eigenvalues)):
        raise MeasurementError("plane covariance eigendecomposition failed")
    normal = eigenvectors[:, int(np.argmin(eigenvalues))]
    if normal[2] < 0.0:
        normal = -normal
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        raise MeasurementError("refined plane normal is degenerate")
    normal = normal / norm
    return np.r_[normal, -float(normal @ center)]


def _tilt_deg(coefficients: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(coefficients[2], -1.0, 1.0))))


def fit_horizontal_plane_ransac(
    points: np.ndarray,
    *,
    distance_threshold_m: float,
    max_tilt_deg: float,
    iterations: int,
    min_inliers: int,
    min_inlier_ratio: float,
    min_xy_span_m: float,
    seed: int,
) -> PlaneFit:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise MeasurementError(f"points must have shape [N,3], got {values.shape}")
    values = values[np.all(np.isfinite(values), axis=1)]
    if len(values) < max(3, min_inliers):
        raise MeasurementError(
            f"only {len(values)} finite candidates; need at least {min_inliers}"
        )
    if distance_threshold_m <= 0.0 or not math.isfinite(distance_threshold_m):
        raise MeasurementError("distance threshold must be finite and positive")
    if not 0.0 < max_tilt_deg < 90.0:
        raise MeasurementError("max tilt must be in (0, 90) degrees")
    if iterations < 1:
        raise MeasurementError("RANSAC iterations must be positive")

    rng = np.random.default_rng(seed)
    best_coefficients: Optional[np.ndarray] = None
    best_inliers: Optional[np.ndarray] = None
    best_count = -1
    best_rms = float("inf")
    max_points_for_hypotheses = min(len(values), 200000)
    if len(values) > max_points_for_hypotheses:
        hypothesis_pool = values[rng.choice(len(values), max_points_for_hypotheses, replace=False)]
    else:
        hypothesis_pool = values

    for _ in range(iterations):
        sample = hypothesis_pool[rng.choice(len(hypothesis_pool), 3, replace=False)]
        coefficients = _plane_from_three(sample)
        if coefficients is None or _tilt_deg(coefficients) > max_tilt_deg:
            continue
        residuals = np.abs(values @ coefficients[:3] + coefficients[3])
        inliers = residuals <= distance_threshold_m
        count = int(np.count_nonzero(inliers))
        if count < 3:
            continue
        rms = float(np.sqrt(np.mean(np.square(residuals[inliers]))))
        if count > best_count or (count == best_count and rms < best_rms):
            best_coefficients = coefficients
            best_inliers = inliers
            best_count = count
            best_rms = rms

    if best_coefficients is None or best_inliers is None:
        raise MeasurementError("RANSAC found no near-horizontal plane")

    required = max(min_inliers, int(math.ceil(min_inlier_ratio * len(values))))
    if best_count < required:
        raise MeasurementError(
            f"best plane has {best_count}/{len(values)} inliers; need at least {required}"
        )

    # Refit and reselect three times.  The threshold remains explicit and is
    # recorded in the output instead of being silently estimated from the data.
    inliers = best_inliers
    coefficients = best_coefficients
    for _ in range(3):
        coefficients = _refine_plane(values[inliers])
        if _tilt_deg(coefficients) > max_tilt_deg:
            raise MeasurementError("refined plane exceeds the configured tilt limit")
        residuals_all = np.abs(values @ coefficients[:3] + coefficients[3])
        new_inliers = residuals_all <= distance_threshold_m
        if int(np.count_nonzero(new_inliers)) < required:
            raise MeasurementError("plane refinement dropped below the required inlier count")
        if np.array_equal(new_inliers, inliers):
            break
        inliers = new_inliers

    inlier_points = values[inliers]
    spans = np.ptp(inlier_points[:, :2], axis=0)
    if np.any(spans < min_xy_span_m):
        raise MeasurementError(
            "fitted support is too small in base XY: "
            f"span={spans.tolist()} m, required each >= {min_xy_span_m} m"
        )

    signed_residuals = inlier_points @ coefficients[:3] + coefficients[3]
    absolute = np.abs(signed_residuals)
    return PlaneFit(
        coefficients=coefficients,
        inliers=inliers,
        residuals_m=signed_residuals,
        rms_m=float(np.sqrt(np.mean(np.square(signed_residuals)))),
        p95_m=float(np.percentile(absolute, 95.0)),
        max_m=float(np.max(absolute)),
        tilt_deg=_tilt_deg(coefficients),
    )


def _validate_roi(roi: Sequence[int], width: int, height: int) -> Tuple[int, int, int, int]:
    if len(roi) != 4:
        raise MeasurementError("ROI must contain x0 y0 x1 y1")
    x0, y0, x1, y1 = (int(v) for v in roi)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise MeasurementError(
            f"ROI {(x0, y0, x1, y1)} lies outside {width}x{height}"
        )
    return x0, y0, x1, y1


def _select_roi(color_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    try:
        import cv2
    except ImportError as exc:
        raise MeasurementError("--select-roi requires opencv-python") from exc
    print("[ROI] Drag a rectangle over exposed black tabletop only; Enter accepts, Esc cancels.")
    x, y, w, h = cv2.selectROI("Select black tabletop", color_bgr, showCrosshair=True)
    cv2.destroyWindow("Select black tabletop")
    if w <= 0 or h <= 0:
        raise MeasurementError("ROI selection was cancelled or empty")
    return int(x), int(y), int(x + w), int(y + h)


def _actual_intrinsics_dict(intrinsics: Any) -> Dict[str, Any]:
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
        "model": str(intrinsics.model),
        "distortion": [float(value) for value in intrinsics.coeffs],
    }


def _check_intrinsics(
    actual: Mapping[str, Any], expected: Mapping[str, Any], tolerance_px: float
) -> None:
    if int(actual["width"]) != int(expected["width"]) or int(actual["height"]) != int(expected["height"]):
        raise MeasurementError(
            "live color profile does not match calibration resolution: "
            f"actual={actual['width']}x{actual['height']} expected={expected['width']}x{expected['height']}"
        )
    for key in ("fx", "fy", "ppx", "ppy"):
        delta = abs(float(actual[key]) - float(expected[key]))
        if delta > tolerance_px:
            raise MeasurementError(
                f"live {key} differs from calibration by {delta:.6f}px, limit={tolerance_px:.6f}px"
            )


def _configure_depth_sensor(rs: Any, sensor: Any, emitter: bool, laser_power: float) -> None:
    try:
        if sensor.supports(rs.option.emitter_enabled):
            sensor.set_option(rs.option.emitter_enabled, 1.0 if emitter else 0.0)
        if sensor.supports(rs.option.laser_power):
            option_range = sensor.get_option_range(rs.option.laser_power)
            power = float(np.clip(laser_power, option_range.min, option_range.max))
            sensor.set_option(rs.option.laser_power, power)
    except Exception as exc:
        raise MeasurementError(f"failed to configure D435 depth sensor: {exc}") from exc


def capture_black_table_candidates(
    calibration: Mapping[str, Any], args: argparse.Namespace
) -> CaptureResult:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise MeasurementError("pyrealsense2 is required for live measurement") from exc

    camera_record = calibration["camera"]
    calibrated_intrinsics = camera_record["intrinsics"]
    expected_serial = str(camera_record["serial"])
    width = int(calibrated_intrinsics["width"])
    height = int(calibrated_intrinsics["height"])

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(expected_serial)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, int(args.fps))
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, int(args.fps))
    started_at = _utc_now()
    profile = None
    try:
        profile = pipeline.start(config)
        device = profile.get_device()
        serial = str(device.get_info(rs.camera_info.serial_number))
        camera_name = str(device.get_info(rs.camera_info.name))
        if serial != expected_serial:
            raise MeasurementError(
                f"connected camera serial {serial} does not match calibration {expected_serial}"
            )
        depth_sensor = device.first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
        _configure_depth_sensor(
            rs, depth_sensor, emitter=not args.no_emitter, laser_power=args.laser_power
        )
        align = rs.align(rs.stream.color)

        last_color: Optional[np.ndarray] = None
        for _ in range(int(args.warmup_frames)):
            frames = align.process(pipeline.wait_for_frames(timeout_ms=args.timeout_ms))
            color_frame = frames.get_color_frame()
            if color_frame:
                last_color = np.asanyarray(color_frame.get_data()).copy()
        if last_color is None:
            raise MeasurementError("no color frame received during warmup")

        if args.select_roi:
            roi = _select_roi(last_color)
            # Drop frames accumulated while the user was selecting the ROI.
            for _ in range(5):
                pipeline.wait_for_frames(timeout_ms=args.timeout_ms)
        else:
            roi = _validate_roi(args.roi, width, height)

        x0, y0, x1, y1 = roi
        transform = np.asarray(calibration["T_base_camera"], dtype=np.float64)
        all_points: List[np.ndarray] = []
        per_frame_counts: List[int] = []
        actual_intrinsics: Optional[Dict[str, Any]] = None

        for _ in range(int(args.frames)):
            frames = align.process(pipeline.wait_for_frames(timeout_ms=args.timeout_ms))
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                raise MeasurementError("received an incomplete aligned RGB-D frame")
            depth_raw = np.asanyarray(depth_frame.get_data())
            color_bgr = np.asanyarray(color_frame.get_data())
            live = _actual_intrinsics_dict(
                color_frame.profile.as_video_stream_profile().get_intrinsics()
            )
            _check_intrinsics(live, calibrated_intrinsics, args.intrinsics_tolerance_px)
            if actual_intrinsics is None:
                actual_intrinsics = live

            stride = int(args.stride)
            ys = np.arange(y0, y1, stride, dtype=np.int32)
            xs = np.arange(x0, x1, stride, dtype=np.int32)
            vv, uu = np.meshgrid(ys, xs, indexing="ij")
            sampled_depth = depth_raw[vv, uu].astype(np.float64) * depth_scale
            sampled_color = color_bgr[vv, uu].astype(np.uint8)
            value = np.max(sampled_color, axis=2)
            chroma = np.max(sampled_color, axis=2) - np.min(sampled_color, axis=2)
            valid = (
                np.isfinite(sampled_depth)
                & (sampled_depth >= args.depth_min_m)
                & (sampled_depth <= args.depth_max_m)
                & (value <= args.black_value_max)
                & (chroma <= args.black_chroma_max)
            )
            if not np.any(valid):
                per_frame_counts.append(0)
                continue

            z = sampled_depth[valid]
            u = uu[valid].astype(np.float64)
            v = vv[valid].astype(np.float64)
            x = (u - float(live["ppx"])) / float(live["fx"]) * z
            y = (v - float(live["ppy"])) / float(live["fy"]) * z
            points_camera = np.column_stack((x, y, z))
            points_base = points_camera @ transform[:3, :3].T + transform[:3, 3]
            if args.base_z_min_m is not None:
                points_base = points_base[points_base[:, 2] >= args.base_z_min_m]
            if args.base_z_max_m is not None:
                points_base = points_base[points_base[:, 2] <= args.base_z_max_m]
            per_frame_counts.append(int(len(points_base)))
            if len(points_base):
                all_points.append(points_base)

        if actual_intrinsics is None:
            raise MeasurementError("no complete frame was captured")
        if not all_points:
            raise MeasurementError("no black depth candidates survived the configured gates")
        points = np.concatenate(all_points, axis=0)
        if len(points) > args.max_candidates:
            rng = np.random.default_rng(args.seed)
            points = points[rng.choice(len(points), args.max_candidates, replace=False)]
        return CaptureResult(
            points_base=points,
            per_frame_candidate_counts=tuple(per_frame_counts),
            roi_xyxy=roi,
            actual_intrinsics=actual_intrinsics,
            depth_scale_m_per_unit=depth_scale,
            camera_name=camera_name,
            camera_serial=serial,
            started_at=started_at,
            completed_at=_utc_now(),
        )
    finally:
        if profile is not None:
            pipeline.stop()


def _build_measurement_document(
    calibration_path: Path,
    calibration: Mapping[str, Any],
    capture: CaptureResult,
    fit: PlaneFit,
    args: argparse.Namespace,
) -> Mapping[str, Any]:
    coefficients = fit.coefficients
    inlier_points = capture.points_base[fit.inliers]
    reference_xy = np.median(inlier_points[:, :2], axis=0)
    if abs(float(coefficients[2])) < 1e-9:
        raise MeasurementError("fitted plane has no usable z height")
    reference_z = -(
        float(coefficients[0]) * float(reference_xy[0])
        + float(coefficients[1]) * float(reference_xy[1])
        + float(coefficients[3])
    ) / float(coefficients[2])
    bounds_min = np.min(inlier_points, axis=0)
    bounds_max = np.max(inlier_points, axis=0)
    return {
        "schema_version": 1,
        "kind": "tabletop_plane_measurement",
        "measurement_status": "candidate_requires_visual_review",
        "measured_at_start": capture.started_at,
        "measured_at_end": capture.completed_at,
        "hardware_access": {
            "camera_read_only": True,
            "robot_connected": False,
            "robot_commanded": False,
        },
        "frames": {
            "base_frame": "robot_base",
            "camera_frame": "camera_color_optical_frame",
            "plane_equation": "a*x + b*y + c*z + d = 0",
            "normal_orientation": "c >= 0",
        },
        "units": {
            "length": "m",
            "angle": "deg",
            "pixel": "px",
        },
        "source_calibration": {
            "path": str(calibration_path.resolve()),
            "sha256": _sha256(calibration_path),
            "calibration_id": str(calibration["calibration_id"]),
            "T_base_camera": calibration["T_base_camera"],
        },
        "camera": {
            "name": capture.camera_name,
            "serial": capture.camera_serial,
            "stream": {
                "width": int(capture.actual_intrinsics["width"]),
                "height": int(capture.actual_intrinsics["height"]),
                "fps": int(args.fps),
                "depth_aligned_to_color": True,
            },
            "color_intrinsics": dict(capture.actual_intrinsics),
            "depth_scale_m_per_unit": float(capture.depth_scale_m_per_unit),
        },
        "candidate_selection": {
            "roi_xyxy_px": list(capture.roi_xyxy),
            "frames": int(args.frames),
            "warmup_frames": int(args.warmup_frames),
            "stride_px": int(args.stride),
            "black_value_max_8bit": int(args.black_value_max),
            "black_chroma_max_8bit": int(args.black_chroma_max),
            "camera_depth_range_m": [float(args.depth_min_m), float(args.depth_max_m)],
            "base_z_range_m": [args.base_z_min_m, args.base_z_max_m],
            "candidate_count_before_ransac": int(len(capture.points_base)),
            "per_frame_candidate_counts": list(capture.per_frame_candidate_counts),
            "note": "The sampled inlier bounds below are observation coverage, not physical table dimensions."
        },
        "robust_fit_configuration": {
            "algorithm": "near_horizontal_RANSAC_then_iterated_SVD",
            "ransac_iterations": int(args.ransac_iterations),
            "distance_threshold_m": float(args.distance_threshold_m),
            "max_tilt_deg": float(args.max_tilt_deg),
            "minimum_inliers": int(args.min_inliers),
            "minimum_inlier_ratio": float(args.min_inlier_ratio),
            "minimum_xy_span_m": float(args.min_xy_span_m),
            "random_seed": int(args.seed),
        },
        "result": {
            "plane_abcd": [float(value) for value in coefficients],
            "normal_robot_base": [float(value) for value in coefficients[:3]],
            "height": {
                "reference_xy_m": [float(value) for value in reference_xy],
                "z_m": float(reference_z),
                "note": "For a tilted table, z depends on x and y; use plane_abcd for collision geometry."
            },
            "tilt_from_robot_base_plus_z_deg": float(fit.tilt_deg),
            "residual_rms_m": float(fit.rms_m),
            "residual_p95_m": float(fit.p95_m),
            "residual_max_m": float(fit.max_m),
            "inlier_count": int(np.count_nonzero(fit.inliers)),
            "inlier_ratio": float(np.mean(fit.inliers)),
            "sampled_inlier_bounds_robot_base_m": {
                "min": [float(value) for value in bounds_min],
                "max": [float(value) for value in bounds_max],
            },
        },
        "review_checklist": [
            "Confirm the ROI contained exposed tabletop and excluded the robot, objects, table edge, and background.",
            "Overlay or inspect inlier coverage; sampled bounds are not a table-size survey.",
            "Repeat with at least three separated tabletop ROIs and compare height/normal.",
            "Do not copy this result into scene_manifest.json until repeatability and physical plausibility pass review."
        ],
    }


def _write_yaml(path: Path, document: Mapping[str, Any], force: bool) -> None:
    if path.exists() and not force:
        raise MeasurementError(f"output already exists: {path}; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _self_test() -> None:
    rng = np.random.default_rng(17)
    xy = rng.uniform([-0.2, -0.15], [0.3, 0.25], size=(5000, 2))
    # z = 0.042 + 0.01*x - 0.015*y, plus sub-millimetre noise.
    z = 0.042 + 0.01 * xy[:, 0] - 0.015 * xy[:, 1] + rng.normal(0.0, 0.0007, 5000)
    plane_points = np.column_stack((xy, z))
    outliers = rng.uniform([-0.3, -0.3, -0.1], [0.4, 0.35, 0.3], size=(800, 3))
    fit = fit_horizontal_plane_ransac(
        np.vstack((plane_points, outliers)),
        distance_threshold_m=0.003,
        max_tilt_deg=10.0,
        iterations=500,
        min_inliers=3000,
        min_inlier_ratio=0.5,
        min_xy_span_m=0.2,
        seed=4,
    )
    expected = np.array([-0.01, 0.015, 1.0, -0.042], dtype=np.float64)
    expected /= np.linalg.norm(expected[:3])
    if np.linalg.norm(fit.coefficients - expected) > 0.002:
        raise MeasurementError(
            f"self-test plane mismatch: got {fit.coefficients}, expected {expected}"
        )
    if fit.rms_m > 0.001:
        raise MeasurementError(f"self-test RMS is too high: {fit.rms_m}")
    print(
        "[PASS] offline plane-fit self-test "
        f"plane={fit.coefficients.tolist()} rms={fit.rms_m:.6f}m "
        f"inliers={int(np.count_nonzero(fit.inliers))}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    roi_group = parser.add_mutually_exclusive_group()
    roi_group.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="required non-interactive ROI around exposed tabletop",
    )
    roi_group.add_argument(
        "--select-roi",
        action="store_true",
        help="select the tabletop ROI interactively in the color frame",
    )
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--timeout-ms", type=int, default=2000)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--black-value-max", type=int, default=100)
    parser.add_argument("--black-chroma-max", type=int, default=50)
    parser.add_argument("--depth-min-m", type=float, default=0.25)
    parser.add_argument("--depth-max-m", type=float, default=1.20)
    parser.add_argument("--base-z-min-m", type=float, default=None)
    parser.add_argument("--base-z-max-m", type=float, default=None)
    parser.add_argument("--intrinsics-tolerance-px", type=float, default=2.0)
    parser.add_argument("--laser-power", type=float, default=180.0)
    parser.add_argument("--no-emitter", action="store_true")
    parser.add_argument("--distance-threshold-m", type=float, default=0.005)
    parser.add_argument("--max-tilt-deg", type=float, default=20.0)
    parser.add_argument("--ransac-iterations", type=int, default=1200)
    parser.add_argument("--min-inliers", type=int, default=1500)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.25)
    parser.add_argument("--min-xy-span-m", type=float, default=0.15)
    parser.add_argument("--max-candidates", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--confirm-camera-only",
        action="store_true",
        help="required acknowledgement before opening the calibrated D435",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run a synthetic offline plane-fit test; does not access hardware",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.self_test:
            _self_test()
            return 0
        if not args.confirm_camera_only:
            raise MeasurementError("--confirm-camera-only is required before camera access")
        if args.roi is None and not args.select_roi:
            raise MeasurementError("provide --roi X0 Y0 X1 Y1 or --select-roi")
        if args.frames < 3 or args.warmup_frames < 0 or args.stride < 1:
            raise MeasurementError("frames must be >=3, warmup >=0, and stride >=1")
        if not (0 <= args.black_value_max <= 255 and 0 <= args.black_chroma_max <= 255):
            raise MeasurementError("black thresholds must be in [0,255]")
        if not (0.0 < args.min_inlier_ratio <= 1.0):
            raise MeasurementError("--min-inlier-ratio must be in (0,1]")
        if args.max_candidates < args.min_inliers:
            raise MeasurementError("--max-candidates must be >= --min-inliers")
        if args.base_z_min_m is not None and args.base_z_max_m is not None:
            if args.base_z_min_m >= args.base_z_max_m:
                raise MeasurementError("base z minimum must be below maximum")

        calibration_path = args.calibration.resolve()
        calibration = _load_calibration(calibration_path)
        print("[Safety] Camera-only measurement. No robot API is imported or contacted.")
        print("[Safety] Keep the robot stationary and ensure the selected ROI is clear tabletop.")
        capture = capture_black_table_candidates(calibration, args)
        fit = fit_horizontal_plane_ransac(
            capture.points_base,
            distance_threshold_m=args.distance_threshold_m,
            max_tilt_deg=args.max_tilt_deg,
            iterations=args.ransac_iterations,
            min_inliers=args.min_inliers,
            min_inlier_ratio=args.min_inlier_ratio,
            min_xy_span_m=args.min_xy_span_m,
            seed=args.seed,
        )
        document = _build_measurement_document(
            calibration_path, calibration, capture, fit, args
        )
        _write_yaml(args.output.resolve(), document, force=args.force)
        result = document["result"]
        print(f"[Saved] {args.output.resolve()}")
        print(
            "[Table] plane="
            f"{result['plane_abcd']} height={result['height']['z_m']:.6f}m "
            f"at xy={result['height']['reference_xy_m']} "
            f"tilt={result['tilt_from_robot_base_plus_z_deg']:.4f}deg"
        )
        print(
            f"[Fit] RMS={result['residual_rms_m']:.6f}m "
            f"p95={result['residual_p95_m']:.6f}m "
            f"inliers={result['inlier_count']} ratio={result['inlier_ratio']:.3f}"
        )
        print("[Review required] Repeat on separated ROIs before updating scene_manifest.json.")
        return 0
    except MeasurementError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
