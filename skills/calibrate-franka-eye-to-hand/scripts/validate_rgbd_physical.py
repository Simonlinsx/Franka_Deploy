#!/usr/bin/env python3
"""Capture and evaluate physical RGB-D evidence for eye-to-hand calibration.

``capture-station`` opens one exact RealSense serial and, unless an independently
surveyed marker centre is supplied, the read-only Franka state adapter.  It has
no robot motion interface.  The operator must stage each near/middle/far pose
separately and run the capture only after the robot and target are stationary.

``evaluate`` is fully offline.  It checks several immutable station reports
against explicit, task-approved limits.  A PASS from this tool is only the
physical RGB-D portion of commissioning; static audit, holdout, rigidity,
runtime wiring, and task-clearance evidence remain separate requirements.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
from dynamic_pcd.calibration.stationary import (  # noqa: E402
    aggregate_stationary_target_poses,
)
from dynamic_pcd.calibration.transforms import (  # noqa: E402
    transform_error,
    transform_points,
    validate_rigid_transform,
)
from dynamic_pcd.camera.realsense_camera import RealSenseCamera  # noqa: E402
from dynamic_pcd.config import load_config  # noqa: E402
from dynamic_pcd.robot.franka import FrankaStateReader  # noqa: E402


class PhysicalValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class DepthPlaneFit:
    coefficients: np.ndarray
    inlier_count: int
    inlier_ratio: float
    residual_rms_m: float
    residual_p95_m: float
    residual_max_m: float


def _require_positive(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise PhysicalValidationError(f"{name} must be finite and positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PhysicalValidationError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PhysicalValidationError(f"{path} must contain a YAML mapping")
    return value


def _write_json_exclusive_atomic(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise PhysicalValidationError(f"refusing to overwrite {destination}")
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


def _load_calibration(path: Path, expected_serial: str) -> Dict[str, Any]:
    calibration = _load_yaml(path)
    if calibration.get("schema_version") != 1:
        raise PhysicalValidationError("calibration schema_version must be 1")
    if calibration.get("kind") != "camera_robot_extrinsic_calibration":
        raise PhysicalValidationError("calibration kind is not eye-to-hand calibration")
    if calibration.get("calibration_type") != "eye_to_hand":
        raise PhysicalValidationError("calibration_type must be eye_to_hand")
    if calibration.get("base_frame") != "robot_base":
        raise PhysicalValidationError("calibration base_frame must be robot_base")
    camera_frame = str(calibration.get("camera_frame") or "")
    if "color_optical_frame" not in camera_frame:
        raise PhysicalValidationError(
            "calibration camera_frame must be a color optical frame"
        )
    camera = calibration.get("camera")
    if not isinstance(camera, dict):
        raise PhysicalValidationError("calibration camera metadata is missing")
    serial = str(camera.get("serial") or "")
    if serial != str(expected_serial):
        raise PhysicalValidationError(
            f"calibration serial {serial!r} does not match {expected_serial!r}"
        )
    if not str(camera.get("name") or ""):
        raise PhysicalValidationError("calibration camera name is missing")
    try:
        depth_scale = float(camera.get("depth_scale"))
    except (TypeError, ValueError) as exc:
        raise PhysicalValidationError("calibration depth scale is missing") from exc
    if not math.isfinite(depth_scale) or depth_scale <= 0.0:
        raise PhysicalValidationError("calibration depth scale must be positive")
    if not str(calibration.get("calibration_id") or ""):
        raise PhysicalValidationError("calibration_id is missing")
    intrinsics = camera.get("intrinsics")
    if not isinstance(intrinsics, dict):
        raise PhysicalValidationError("calibration color intrinsics are missing")
    for name in ("width", "height", "fx", "fy", "ppx", "ppy", "distortion"):
        if name not in intrinsics:
            raise PhysicalValidationError(f"calibration intrinsics missing {name}")
    distortion = np.asarray(intrinsics["distortion"], dtype=np.float64)
    if distortion.ndim != 1 or not np.all(np.isfinite(distortion)):
        raise PhysicalValidationError("calibration distortion must be finite")
    if not np.allclose(distortion, 0.0, atol=1.0e-12, rtol=0.0):
        raise PhysicalValidationError(
            "this diagnostic currently requires zero color-distortion coefficients"
        )
    target = calibration.get("target")
    if not isinstance(target, dict) or target.get("type") != "aruco":
        raise PhysicalValidationError("physical RGB-D diagnostic requires ArUco target")
    for name in ("dictionary", "marker_id", "marker_length_m"):
        if name not in target:
            raise PhysicalValidationError(f"calibration target missing {name}")
    quality = (calibration.get("solver") or {}).get("quality") or {}
    if quality.get("status") != "pass" or quality.get("warnings") not in ([], None):
        raise PhysicalValidationError(
            "physical validation requires a warning-free quality-pass calibration"
        )
    calibration["T_base_camera"] = validate_rigid_transform(
        calibration.get("T_base_camera"), name="calibration.T_base_camera"
    )
    calibration["T_ee_target"] = validate_rigid_transform(
        calibration.get("T_ee_target"), name="calibration.T_ee_target"
    )
    return calibration


def _check_live_intrinsics(
    actual: Any,
    expected: Mapping[str, Any],
    tolerance_px: float,
) -> None:
    if int(actual.width) != int(expected["width"]) or int(actual.height) != int(
        expected["height"]
    ):
        raise PhysicalValidationError(
            "live color resolution differs from calibration: "
            f"{actual.width}x{actual.height} vs "
            f"{expected['width']}x{expected['height']}"
        )
    for name in ("fx", "fy", "ppx", "ppy"):
        difference = abs(float(getattr(actual, name)) - float(expected[name]))
        if difference > tolerance_px:
            raise PhysicalValidationError(
                f"live {name} differs from calibration by {difference:.6f}px; "
                f"limit={tolerance_px:.6f}px"
            )
    live_distortion = np.asarray(actual.distortion, dtype=np.float64)
    if not np.allclose(live_distortion, 0.0, atol=1.0e-12, rtol=0.0):
        raise PhysicalValidationError(
            "live color distortion is non-zero; deprojection adapter is required"
        )


def _fit_plane(
    points: np.ndarray,
    *,
    residual_limit_m: float,
    minimum_points: int,
    expected_normal: np.ndarray,
) -> DepthPlaneFit:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise PhysicalValidationError(
            f"plane points must have shape [N,3], got {values.shape}"
        )
    values = values[np.all(np.isfinite(values), axis=1)]
    if len(values) < minimum_points:
        raise PhysicalValidationError(
            f"only {len(values)} target depth points; need {minimum_points}"
        )
    if not math.isfinite(residual_limit_m) or residual_limit_m <= 0.0:
        raise PhysicalValidationError("depth plane residual limit must be positive")

    inliers = np.ones(len(values), dtype=bool)
    coefficients: Optional[np.ndarray] = None
    for _ in range(4):
        selected = values[inliers]
        center = np.mean(selected, axis=0)
        covariance = (selected - center).T @ (selected - center)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if not np.all(np.isfinite(eigenvalues)):
            raise PhysicalValidationError("depth plane eigendecomposition failed")
        normal = eigenvectors[:, int(np.argmin(eigenvalues))]
        normal /= np.linalg.norm(normal)
        if float(normal @ expected_normal) < 0.0:
            normal = -normal
        coefficients = np.r_[normal, -float(normal @ center)]
        residuals = np.abs(values @ normal + coefficients[3])
        next_inliers = residuals <= residual_limit_m
        if int(np.count_nonzero(next_inliers)) < minimum_points:
            raise PhysicalValidationError(
                "depth plane refinement retained only "
                f"{int(np.count_nonzero(next_inliers))} points; need {minimum_points}"
            )
        if np.array_equal(next_inliers, inliers):
            break
        inliers = next_inliers
    assert coefficients is not None
    signed = values[inliers] @ coefficients[:3] + coefficients[3]
    absolute = np.abs(signed)
    return DepthPlaneFit(
        coefficients=coefficients,
        inlier_count=int(np.count_nonzero(inliers)),
        inlier_ratio=float(np.mean(inliers)),
        residual_rms_m=float(np.sqrt(np.mean(np.square(signed)))),
        residual_p95_m=float(np.percentile(absolute, 95)),
        residual_max_m=float(np.max(absolute)),
    )


def measure_target_depth(
    frame: Any,
    detection: Any,
    *,
    T_base_camera: np.ndarray,
    known_center_base_m: Sequence[float],
    interior_scale: float,
    pixel_stride: int,
    depth_min_m: float,
    depth_max_m: float,
    max_depth_from_pnp_m: float,
    plane_residual_limit_m: float,
    minimum_depth_points: int,
) -> Dict[str, Any]:
    """Measure the target centre from aligned depth in one immutable frame."""

    if not detection.valid or detection.T_camera_target is None:
        raise PhysicalValidationError("ArUco detection is invalid")
    if not 0.0 < interior_scale < 1.0:
        raise PhysicalValidationError("interior_scale must be in (0,1)")
    if pixel_stride < 1:
        raise PhysicalValidationError("pixel_stride must be positive")
    if not 0.0 < depth_min_m < depth_max_m:
        raise PhysicalValidationError("depth range is invalid")
    if max_depth_from_pnp_m <= 0.0:
        raise PhysicalValidationError("max_depth_from_pnp_m must be positive")

    depth_raw = np.asarray(frame.depth_raw)
    height, width = depth_raw.shape
    if (height, width) != (int(frame.intrinsics.height), int(frame.intrinsics.width)):
        raise PhysicalValidationError(
            "aligned depth shape differs from color intrinsics"
        )
    corners = np.asarray(detection.corners, dtype=np.float64).reshape(4, 2)
    center_px = np.mean(corners, axis=0)
    interior = center_px + interior_scale * (corners - center_px)
    if np.any(interior[:, 0] < 0) or np.any(interior[:, 0] >= width):
        raise PhysicalValidationError("target interior lies outside image width")
    if np.any(interior[:, 1] < 0) or np.any(interior[:, 1] >= height):
        raise PhysicalValidationError("target interior lies outside image height")
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(interior).astype(np.int32), 1)
    vv, uu = np.nonzero(mask)
    retained = (uu % pixel_stride == 0) & (vv % pixel_stride == 0)
    uu = uu[retained]
    vv = vv[retained]
    candidate_count = int(len(uu))
    if candidate_count < minimum_depth_points:
        raise PhysicalValidationError(
            f"target interior has only {candidate_count} sampled pixels"
        )

    depth_m = depth_raw[vv, uu].astype(np.float64) * float(frame.depth_scale)
    pnp_transform = validate_rigid_transform(
        detection.T_camera_target, name="detection.T_camera_target"
    )
    pnp_center_camera = pnp_transform[:3, 3]
    valid = (
        np.isfinite(depth_m)
        & (depth_m >= depth_min_m)
        & (depth_m <= depth_max_m)
        & (np.abs(depth_m - float(pnp_center_camera[2])) <= max_depth_from_pnp_m)
    )
    valid_count = int(np.count_nonzero(valid))
    if valid_count < minimum_depth_points:
        raise PhysicalValidationError(
            f"only {valid_count}/{candidate_count} target depth pixels survived"
        )
    z = depth_m[valid]
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)
    intrinsics = frame.intrinsics
    x = (u - float(intrinsics.ppx)) / float(intrinsics.fx) * z
    y = (v - float(intrinsics.ppy)) / float(intrinsics.fy) * z
    points_camera = np.column_stack((x, y, z))
    plane = _fit_plane(
        points_camera,
        residual_limit_m=plane_residual_limit_m,
        minimum_points=minimum_depth_points,
        expected_normal=pnp_transform[:3, 2],
    )

    # The ray through the PnP target origin identifies the same physical point
    # on the depth plane; intersecting a fitted plane is much less noisy than a
    # single centre pixel.
    ray = np.array(
        [
            pnp_center_camera[0] / pnp_center_camera[2],
            pnp_center_camera[1] / pnp_center_camera[2],
            1.0,
        ],
        dtype=np.float64,
    )
    denominator = float(plane.coefficients[:3] @ ray)
    if abs(denominator) < 1.0e-9:
        raise PhysicalValidationError("target-centre ray is parallel to depth plane")
    ray_scale = -float(plane.coefficients[3]) / denominator
    if not math.isfinite(ray_scale) or ray_scale <= 0.0:
        raise PhysicalValidationError("depth-plane intersection is behind camera")
    depth_center_camera = ray_scale * ray
    depth_center_base = transform_points(
        T_base_camera, depth_center_camera.reshape(1, 3)
    )[0].astype(np.float64)
    pnp_center_base = transform_points(T_base_camera, pnp_center_camera.reshape(1, 3))[
        0
    ].astype(np.float64)
    known_center = np.asarray(known_center_base_m, dtype=np.float64).reshape(3)
    depth_pnp_vector = depth_center_camera - pnp_center_camera
    known_vector = depth_center_base - known_center
    return {
        "candidate_depth_pixel_count": candidate_count,
        "valid_depth_pixel_count": valid_count,
        "valid_depth_fraction": float(valid_count) / float(candidate_count),
        "depth_plane_camera_abcd": plane.coefficients.tolist(),
        "depth_plane_inlier_count": plane.inlier_count,
        "depth_plane_inlier_ratio": plane.inlier_ratio,
        "depth_plane_residual_rms_m": plane.residual_rms_m,
        "depth_plane_residual_p95_m": plane.residual_p95_m,
        "depth_plane_residual_max_m": plane.residual_max_m,
        "marker_center_camera_pnp_m": pnp_center_camera.tolist(),
        "marker_center_camera_depth_m": depth_center_camera.tolist(),
        "marker_center_base_pnp_m": pnp_center_base.tolist(),
        "marker_center_base_depth_m": depth_center_base.tolist(),
        "known_marker_center_base_m": known_center.tolist(),
        "depth_minus_pnp_camera_xyz_m": depth_pnp_vector.tolist(),
        "depth_minus_pnp_z_m": float(depth_pnp_vector[2]),
        "depth_minus_pnp_3d_m": float(np.linalg.norm(depth_pnp_vector)),
        "depth_minus_known_base_xyz_m": known_vector.tolist(),
        "depth_minus_known_base_z_m": float(known_vector[2]),
        "depth_minus_known_base_3d_m": float(np.linalg.norm(known_vector)),
    }


def _metrics(values: Sequence[float], *, absolute: bool = False) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if absolute:
        array = np.abs(array)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise PhysicalValidationError("metric values must be a non-empty finite vector")
    return {
        "median": float(np.median(array)),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _station_metrics(
    frame_records: Sequence[Mapping[str, Any]],
    *,
    total_frames: int,
) -> Dict[str, Any]:
    detected = [record for record in frame_records if record["detection_valid"]]
    reprojection_pass = [
        record for record in detected if record["reprojection_gate_pass"]
    ]
    rgbd = [record for record in reprojection_pass if record["rgbd_valid"]]
    result: Dict[str, Any] = {
        "captured_frame_count": len(frame_records),
        "detected_valid_count": len(detected),
        "reprojection_pass_count": len(reprojection_pass),
        "rgbd_valid_count": len(rgbd),
    }
    if reprojection_pass:
        result["pnp_target_z_camera_m"] = _metrics(
            [record["T_camera_target"][2][3] for record in reprojection_pass]
        )
        result["reprojection_error_px"] = _metrics(
            [record["reprojection_error_px"] for record in reprojection_pass]
        )
    if rgbd:
        result["valid_depth_fraction"] = _metrics(
            [record["rgbd"]["valid_depth_fraction"] for record in rgbd]
        )
        result["depth_plane_residual_p95_m"] = _metrics(
            [record["rgbd"]["depth_plane_residual_p95_m"] for record in rgbd]
        )
        result["depth_minus_pnp_z_abs_m"] = _metrics(
            [record["rgbd"]["depth_minus_pnp_z_m"] for record in rgbd],
            absolute=True,
        )
        result["depth_minus_pnp_z_signed_m"] = _metrics(
            [record["rgbd"]["depth_minus_pnp_z_m"] for record in rgbd]
        )
        result["depth_minus_pnp_3d_m"] = _metrics(
            [record["rgbd"]["depth_minus_pnp_3d_m"] for record in rgbd]
        )
        result["depth_minus_known_base_z_abs_m"] = _metrics(
            [record["rgbd"]["depth_minus_known_base_z_m"] for record in rgbd],
            absolute=True,
        )
        result["depth_minus_known_base_3d_m"] = _metrics(
            [record["rgbd"]["depth_minus_known_base_3d_m"] for record in rgbd]
        )
    if len(frame_records) != total_frames:
        raise PhysicalValidationError(
            f"captured {len(frame_records)} records, expected {total_frames}"
        )
    return result


def capture_station(
    *,
    config_path: Path,
    calibration_path: Path,
    output_path: Path,
    evidence_dir: Path,
    camera_serial: str,
    station_id: str,
    distance_band: str,
    image_region: str,
    frames: int,
    max_reprojection_error_px: float,
    min_reprojection_pass_frames: int,
    min_coherent_frames: int,
    max_pnp_translation_jitter_m: float,
    max_pnp_rotation_jitter_deg: float,
    max_untrimmed_translation_jitter_m: float,
    max_untrimmed_rotation_jitter_deg: float,
    max_untrimmed_reprojection_error_px: float,
    min_rgbd_valid_frames: int,
    interior_scale: float,
    pixel_stride: int,
    depth_min_m: float,
    depth_max_m: float,
    max_depth_from_pnp_m: float,
    plane_residual_limit_m: float,
    minimum_depth_points: int,
    intrinsics_tolerance_px: float,
    robot_ip: Optional[str],
    max_robot_translation_m: float,
    max_robot_rotation_deg: float,
    known_base_center_m: Optional[Sequence[float]],
    known_base_source: Optional[str],
    known_base_evidence: Optional[Path],
    session_target_report: Optional[Path] = None,
    camera_factory: Any = RealSenseCamera,
    robot_factory: Any = FrankaStateReader,
    detector: Any = detect_aruco_marker,
) -> Dict[str, Any]:
    """Capture one stationary evidence artifact; no motion API is available."""

    if output_path.exists():
        raise PhysicalValidationError(f"refusing to overwrite {output_path}")
    if evidence_dir.exists():
        raise PhysicalValidationError(
            f"refusing to reuse evidence directory {evidence_dir}"
        )
    if frames < 1:
        raise PhysicalValidationError("frames must be positive")
    if not 0.0 < interior_scale < 1.0:
        raise PhysicalValidationError("interior_scale must be in (0,1)")
    if not 0.0 < depth_min_m < depth_max_m:
        raise PhysicalValidationError("depth range is invalid")
    for name, count in (
        ("min_reprojection_pass_frames", min_reprojection_pass_frames),
        ("min_coherent_frames", min_coherent_frames),
        ("min_rgbd_valid_frames", min_rgbd_valid_frames),
        ("pixel_stride", pixel_stride),
        ("minimum_depth_points", minimum_depth_points),
    ):
        if (
            isinstance(count, bool)
            or not isinstance(count, (int, np.integer))
            or count < 1
        ):
            raise PhysicalValidationError(f"{name} must be a positive integer")
    if any(
        count > frames
        for count in (
            min_reprojection_pass_frames,
            min_coherent_frames,
            min_rgbd_valid_frames,
        )
    ):
        raise PhysicalValidationError("per-station frame requirements exceed frames")
    for name, value in (
        ("max_reprojection_error_px", max_reprojection_error_px),
        ("max_pnp_translation_jitter_m", max_pnp_translation_jitter_m),
        ("max_pnp_rotation_jitter_deg", max_pnp_rotation_jitter_deg),
        ("max_untrimmed_translation_jitter_m", max_untrimmed_translation_jitter_m),
        ("max_untrimmed_rotation_jitter_deg", max_untrimmed_rotation_jitter_deg),
        ("max_untrimmed_reprojection_error_px", max_untrimmed_reprojection_error_px),
        ("max_depth_from_pnp_m", max_depth_from_pnp_m),
        ("plane_residual_limit_m", plane_residual_limit_m),
        ("intrinsics_tolerance_px", intrinsics_tolerance_px),
        ("max_robot_translation_m", max_robot_translation_m),
        ("max_robot_rotation_deg", max_robot_rotation_deg),
    ):
        _require_positive(name, value)
    calibration_path = calibration_path.resolve()
    config_path = config_path.resolve()
    calibration_sha256 = _sha256(calibration_path)
    config_sha256 = _sha256(config_path)
    calibration = _load_calibration(calibration_path, camera_serial)
    camera_metadata = calibration["camera"]
    target = calibration["target"]
    target_spec = ArucoMarkerSpec(
        str(target["dictionary"]),
        int(target["marker_id"]),
        float(target["marker_length_m"]),
    )
    session_target_evidence: Optional[Dict[str, Any]] = None
    active_T_ee_target = calibration["T_ee_target"]
    if known_base_center_m is not None and session_target_report is not None:
        raise PhysicalValidationError(
            "--known-base-center-m and --session-target-report are mutually exclusive"
        )
    if known_base_center_m is None:
        if not robot_ip:
            raise PhysicalValidationError(
                "robot_ip is required unless an independent known base centre is supplied"
            )
        if session_target_report is None:
            known_reference = {
                "source": "robot_kinematics_using_calibration_T_ee_target",
                "independent_of_calibration": False,
                "evidence_path": None,
                "evidence_sha256": None,
            }
        else:
            report_path = session_target_report.resolve()
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise PhysicalValidationError(
                    f"cannot load session target report {report_path}: {exc}"
                ) from exc
            if not isinstance(report, dict):
                raise PhysicalValidationError(
                    "session target report must contain a JSON object"
                )
            if report.get("kind") != "reused_eye_to_hand_extrinsic_session_holdout":
                raise PhysicalValidationError("invalid session target report kind")
            if report.get("decision") != "holdout_pass":
                raise PhysicalValidationError(
                    "session target report must have decision=holdout_pass"
                )
            source = report.get("calibration") or {}
            if (
                str(source.get("sha256") or "") != calibration_sha256
                or str(source.get("calibration_id") or "")
                != str(calibration.get("calibration_id") or "")
                or source.get("T_base_camera_reused_unchanged") is not True
            ):
                raise PhysicalValidationError(
                    "session target report is not bound to this unchanged calibration"
                )
            active_T_ee_target = validate_rigid_transform(
                report.get("T_ee_target_session"),
                name="session_target_report.T_ee_target_session",
            )
            session_target_evidence = {
                "path": str(report_path),
                "sha256": _sha256(report_path),
            }
            known_reference = {
                "source": "robot_kinematics_using_session_T_ee_target",
                "independent_of_calibration": False,
                "evidence_path": str(report_path),
                "evidence_sha256": session_target_evidence["sha256"],
            }
    else:
        known_center = np.asarray(known_base_center_m, dtype=np.float64)
        if known_center.shape != (3,) or not np.all(np.isfinite(known_center)):
            raise PhysicalValidationError(
                "known base centre must contain three finite metres"
            )
        if not known_base_source or known_base_evidence is None:
            raise PhysicalValidationError(
                "independent known base centre requires source and evidence path"
            )
        if known_base_source not in (
            "surveyed_fixture",
            "robot_touch_off",
            "metrology_fixture",
        ):
            raise PhysicalValidationError(
                "known_base_source must be surveyed_fixture, robot_touch_off, "
                "or metrology_fixture"
            )
        evidence_path = known_base_evidence.resolve()
        if not evidence_path.is_file():
            raise PhysicalValidationError(
                f"known-base evidence does not exist: {evidence_path}"
            )
        known_reference = {
            "source": str(known_base_source),
            "independent_of_calibration": True,
            "evidence_path": str(evidence_path),
            "evidence_sha256": _sha256(evidence_path),
        }

    cfg = load_config(str(config_path))
    cfg["camera"]["serial"] = str(camera_serial)
    camera = camera_factory(cfg["camera"])
    robot = None if known_base_center_m is not None else robot_factory(str(robot_ip))
    frame_records: List[Dict[str, Any]] = []
    transforms: List[np.ndarray] = []
    reprojection_errors: List[float] = []
    T_before: Optional[np.ndarray] = None
    T_after: Optional[np.ndarray] = None
    first_intrinsics: Optional[Dict[str, Any]] = None
    snapshot_indices = {0, frames // 2, frames - 1}
    snapshots: List[Dict[str, Any]] = []
    started = time.time()
    try:
        camera.start()
        if str(camera.device_serial) != str(camera_serial):
            raise PhysicalValidationError(
                f"requested serial {camera_serial}, opened {camera.device_serial}"
            )
        expected_name = str(camera_metadata.get("name") or "")
        if expected_name and str(camera.device_name) != expected_name:
            raise PhysicalValidationError(
                f"calibration camera name {expected_name!r}, opened {camera.device_name!r}"
            )
        expected_depth_scale = float(camera_metadata.get("depth_scale"))
        if not np.isclose(
            float(camera.depth_scale), expected_depth_scale, atol=1.0e-12, rtol=0.0
        ):
            raise PhysicalValidationError(
                "live depth scale differs from calibration: "
                f"{camera.depth_scale} vs {expected_depth_scale}"
            )
        if robot is not None:
            robot.connect()
            T_before = validate_rigid_transform(
                robot.read_T_base_ee(), name="robot.T_base_ee_before"
            )
        for capture_index in range(frames):
            frame = camera.get_frame()
            _check_live_intrinsics(
                frame.intrinsics,
                camera_metadata["intrinsics"],
                intrinsics_tolerance_px,
            )
            if first_intrinsics is None:
                first_intrinsics = frame.intrinsics.to_dict()
            detection = detector(frame.color_bgr, frame.intrinsics, target_spec)
            record: Dict[str, Any] = {
                "capture_index": capture_index,
                "frame_id": int(frame.frame_id),
                "timestamp": float(frame.timestamp),
                "detection_valid": bool(detection.valid),
                "reprojection_error_px": (
                    float(detection.reprojection_error_px)
                    if np.isfinite(detection.reprojection_error_px)
                    else None
                ),
                "reprojection_gate_pass": False,
                "corners_px": np.asarray(detection.corners).reshape(-1, 2).tolist(),
                "T_camera_target": None,
                "rgbd_valid": False,
                "rgbd_error": None,
                "rgbd": None,
            }
            if (
                detection.valid
                and detection.T_camera_target is not None
                and detection.reprojection_error_px <= max_reprojection_error_px
            ):
                transform = validate_rigid_transform(
                    detection.T_camera_target, name="detection.T_camera_target"
                )
                record["reprojection_gate_pass"] = True
                record["T_camera_target"] = transform.tolist()
                transforms.append(transform)
                reprojection_errors.append(float(detection.reprojection_error_px))
                if known_base_center_m is None:
                    assert T_before is not None
                    known_center = (T_before @ active_T_ee_target)[:3, 3]
                else:
                    known_center = np.asarray(known_base_center_m, dtype=np.float64)
                try:
                    measurement = measure_target_depth(
                        frame,
                        detection,
                        T_base_camera=calibration["T_base_camera"],
                        known_center_base_m=known_center,
                        interior_scale=interior_scale,
                        pixel_stride=pixel_stride,
                        depth_min_m=depth_min_m,
                        depth_max_m=depth_max_m,
                        max_depth_from_pnp_m=max_depth_from_pnp_m,
                        plane_residual_limit_m=plane_residual_limit_m,
                        minimum_depth_points=minimum_depth_points,
                    )
                    record["rgbd_valid"] = True
                    record["rgbd"] = measurement
                except PhysicalValidationError as exc:
                    record["rgbd_error"] = str(exc)
            frame_records.append(record)
            if capture_index in snapshot_indices:
                snapshots.append(
                    {
                        "capture_index": capture_index,
                        "frame_id": int(frame.frame_id),
                        "timestamp": float(frame.timestamp),
                        "color_bgr": np.asarray(frame.color_bgr).copy(),
                        "depth_raw": np.asarray(frame.depth_raw).copy(),
                        "depth_scale": float(frame.depth_scale),
                        "intrinsics": frame.intrinsics.to_dict(),
                        "corners_px": np.asarray(detection.corners)
                        .reshape(-1, 2)
                        .copy(),
                        "T_camera_target": (
                            np.asarray(
                                detection.T_camera_target, dtype=np.float64
                            ).copy()
                            if detection.T_camera_target is not None
                            else np.full((4, 4), np.nan, dtype=np.float64)
                        ),
                        "reprojection_error_px": (
                            float(detection.reprojection_error_px)
                            if np.isfinite(detection.reprojection_error_px)
                            else float("nan")
                        ),
                    }
                )
        if robot is not None:
            T_after = validate_rigid_transform(
                robot.read_T_base_ee(), name="robot.T_base_ee_after"
            )
    finally:
        if robot is not None:
            robot.close()
        camera.stop()
    finished = time.time()

    if _sha256(calibration_path) != calibration_sha256:
        raise PhysicalValidationError("source calibration changed during capture")
    if _sha256(config_path) != config_sha256:
        raise PhysicalValidationError("camera config changed during capture")
    if known_reference["independent_of_calibration"]:
        evidence_path = Path(str(known_reference["evidence_path"]))
        if _sha256(evidence_path) != known_reference["evidence_sha256"]:
            raise PhysicalValidationError("known-base evidence changed during capture")
    if session_target_evidence is not None:
        report_path = Path(session_target_evidence["path"])
        if _sha256(report_path) != session_target_evidence["sha256"]:
            raise PhysicalValidationError(
                "session target report changed during capture"
            )

    evidence_dir.mkdir(parents=True, exist_ok=False)
    raw_evidence: List[Dict[str, Any]] = []
    for snapshot in snapshots:
        evidence_path = evidence_dir / (
            f"frame-{snapshot['capture_index']:06d}-id-{snapshot['frame_id']:06d}.npz"
        )
        np.savez_compressed(
            evidence_path,
            color_bgr=snapshot["color_bgr"],
            depth_raw=snapshot["depth_raw"],
            depth_scale_m_per_unit=np.asarray(snapshot["depth_scale"]),
            intrinsics_json=np.asarray(
                json.dumps(
                    snapshot["intrinsics"], separators=(",", ":"), sort_keys=True
                )
            ),
            corners_px=snapshot["corners_px"],
            T_camera_target=snapshot["T_camera_target"],
            reprojection_error_px=np.asarray(snapshot["reprojection_error_px"]),
            frame_id=np.asarray(snapshot["frame_id"]),
            timestamp=np.asarray(snapshot["timestamp"]),
        )
        raw_evidence.append(
            {
                "capture_index": snapshot["capture_index"],
                "frame_id": snapshot["frame_id"],
                "path": str(evidence_path.resolve()),
                "sha256": _sha256(evidence_path),
            }
        )

    failures: List[str] = []
    stationary_metrics: Dict[str, Any] = {}
    try:
        aggregate = aggregate_stationary_target_poses(
            transforms,
            reprojection_errors,
            total_frame_count=frames,
            min_valid_fraction=float(min_coherent_frames) / float(frames),
            min_valid_count=min_coherent_frames,
            max_translation_jitter_m=max_pnp_translation_jitter_m,
            max_rotation_jitter_deg=max_pnp_rotation_jitter_deg,
            min_all_reprojection_pass_count=min_reprojection_pass_frames,
            max_all_reprojection_pass_translation_jitter_m=(
                max_untrimmed_translation_jitter_m
            ),
            max_all_reprojection_pass_rotation_jitter_deg=(
                max_untrimmed_rotation_jitter_deg
            ),
            max_all_reprojection_pass_reprojection_error_px=(
                max_untrimmed_reprojection_error_px
            ),
        )
        stationary_metrics = {
            "coherent_count": len(aggregate.inlier_indices),
            "translation_jitter_p95_m": aggregate.translation_jitter_p95_m,
            "rotation_jitter_p95_deg": aggregate.rotation_jitter_p95_deg,
            "all_reprojection_pass_translation_jitter_p95_m": (
                aggregate.all_reprojection_pass_translation_jitter_p95_m
            ),
            "all_reprojection_pass_rotation_jitter_p95_deg": (
                aggregate.all_reprojection_pass_rotation_jitter_p95_deg
            ),
            "all_reprojection_pass_reprojection_error_p95_px": (
                aggregate.all_reprojection_pass_reprojection_error_p95_px
            ),
        }
    except ValueError as exc:
        failures.append(f"stationary PnP batch failed: {exc}")
    station_metrics = _station_metrics(frame_records, total_frames=frames)
    if station_metrics["rgbd_valid_count"] < min_rgbd_valid_frames:
        failures.append(
            f"RGB-D valid frames {station_metrics['rgbd_valid_count']}/{frames}; "
            f"need at least {min_rgbd_valid_frames}"
        )
    robot_stationary: Optional[Dict[str, Any]] = None
    if T_before is not None and T_after is not None:
        translation_m, rotation_rad = transform_error(T_before, T_after)
        robot_stationary = {
            "T_base_ee_before": T_before.tolist(),
            "T_base_ee_after": T_after.tolist(),
            "translation_m": translation_m,
            "rotation_deg": float(np.degrees(rotation_rad)),
        }
        if (
            translation_m > max_robot_translation_m
            or np.degrees(rotation_rad) > max_robot_rotation_deg
        ):
            failures.append(
                "robot moved during RGB-D capture: "
                f"{translation_m * 1000.0:.3f} mm, "
                f"{np.degrees(rotation_rad):.6f} deg"
            )
    document = {
        "schema_version": 1,
        "kind": "eye_to_hand_rgbd_physical_station",
        "station_id": station_id,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "distance_band": distance_band,
        "image_region": image_region,
        "station_local_pass": not failures,
        "failures": failures,
        "hardware_access": {
            "camera_read_only": True,
            "robot_state_read_only": robot is not None,
            "robot_motion_interface_present": False,
            "robot_commanded": False,
        },
        "capture": {
            "started_host_timestamp_unix_s": started,
            "finished_host_timestamp_unix_s": finished,
            "requested_frame_count": frames,
        },
        "source_calibration": {
            "path": str(calibration_path),
            "sha256": calibration_sha256,
            "calibration_id": str(calibration.get("calibration_id") or ""),
        },
        "source_camera_config": {
            "path": str(config_path.resolve()),
            "sha256": config_sha256,
            "profile": {
                "width": int(cfg["camera"]["width"]),
                "height": int(cfg["camera"]["height"]),
                "fps": int(cfg["camera"]["fps"]),
                "spatial_filter": bool(cfg["camera"].get("spatial_filter", False)),
                "temporal_filter": bool(cfg["camera"].get("temporal_filter", False)),
                "hole_filter": bool(cfg["camera"].get("hole_filter", False)),
                "preset": str(cfg["camera"].get("preset", "default")),
                "emitter": bool(cfg["camera"].get("emitter", True)),
                "laser_power": float(cfg["camera"].get("laser_power", 180.0)),
            },
        },
        "camera": {
            "requested_serial": str(camera_serial),
            "opened_serial": str(camera.device_serial),
            "name": str(camera.device_name),
            "depth_scale_m_per_unit": float(camera.depth_scale),
            "intrinsics": first_intrinsics,
        },
        "target": dict(target),
        "known_base_reference": known_reference,
        "robot_stationary": robot_stationary,
        "capture_configuration": {
            "max_reprojection_error_px": max_reprojection_error_px,
            "minimum_reprojection_pass_frames": min_reprojection_pass_frames,
            "minimum_coherent_frames": min_coherent_frames,
            "maximum_pnp_translation_jitter_m": max_pnp_translation_jitter_m,
            "maximum_pnp_rotation_jitter_deg": max_pnp_rotation_jitter_deg,
            "maximum_untrimmed_translation_jitter_m": (
                max_untrimmed_translation_jitter_m
            ),
            "maximum_untrimmed_rotation_jitter_deg": (
                max_untrimmed_rotation_jitter_deg
            ),
            "maximum_untrimmed_reprojection_error_px": (
                max_untrimmed_reprojection_error_px
            ),
            "minimum_rgbd_valid_frames": min_rgbd_valid_frames,
            "interior_scale": interior_scale,
            "pixel_stride": pixel_stride,
            "depth_range_m": [depth_min_m, depth_max_m],
            "maximum_depth_from_pnp_prefilter_m": max_depth_from_pnp_m,
            "plane_residual_limit_m": plane_residual_limit_m,
            "minimum_depth_points": minimum_depth_points,
            "intrinsics_tolerance_px": intrinsics_tolerance_px,
        },
        "stationary_pnp": stationary_metrics,
        "station_metrics": station_metrics,
        "frames": frame_records,
        "raw_rgbd_evidence": raw_evidence,
    }
    _write_json_exclusive_atomic(output_path, document)
    return document


def evaluate_station_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    expected_serial: str,
    expected_camera_name: str,
    minimum_stations: int,
    minimum_image_regions: int,
    minimum_distance_span_m: float,
    minimum_rgbd_valid_frames: int,
    minimum_raw_evidence_frames: int,
    minimum_depth_valid_fraction_p05: float,
    max_plane_residual_p95_m: float,
    max_depth_pnp_z_abs_p95_m: float,
    max_depth_pnp_z_abs_max_m: float,
    max_depth_pnp_3d_p95_m: float,
    max_known_base_z_abs_p95_m: float,
    max_known_base_z_abs_max_m: float,
    max_known_base_3d_p95_m: float,
    max_known_base_3d_max_m: float,
    max_station_signed_bias_range_m: float,
    require_independent_known_base: bool,
) -> Dict[str, Any]:
    for name, count in (
        ("minimum_stations", minimum_stations),
        ("minimum_image_regions", minimum_image_regions),
        ("minimum_rgbd_valid_frames", minimum_rgbd_valid_frames),
        ("minimum_raw_evidence_frames", minimum_raw_evidence_frames),
    ):
        if (
            isinstance(count, bool)
            or not isinstance(count, (int, np.integer))
            or count < 1
        ):
            raise PhysicalValidationError(f"{name} must be a positive integer")
    if not 0.0 < minimum_depth_valid_fraction_p05 <= 1.0:
        raise PhysicalValidationError(
            "minimum_depth_valid_fraction_p05 must be in (0,1]"
        )
    for name, value in (
        ("minimum_distance_span_m", minimum_distance_span_m),
        ("max_plane_residual_p95_m", max_plane_residual_p95_m),
        ("max_depth_pnp_z_abs_p95_m", max_depth_pnp_z_abs_p95_m),
        ("max_depth_pnp_z_abs_max_m", max_depth_pnp_z_abs_max_m),
        ("max_depth_pnp_3d_p95_m", max_depth_pnp_3d_p95_m),
        ("max_known_base_z_abs_p95_m", max_known_base_z_abs_p95_m),
        ("max_known_base_z_abs_max_m", max_known_base_z_abs_max_m),
        ("max_known_base_3d_p95_m", max_known_base_3d_p95_m),
        ("max_known_base_3d_max_m", max_known_base_3d_max_m),
        ("max_station_signed_bias_range_m", max_station_signed_bias_range_m),
    ):
        _require_positive(name, value)
    failures: List[str] = []
    missing_evidence: List[str] = []
    if len(reports) < minimum_stations:
        missing_evidence.append(
            f"only {len(reports)} station reports; need at least {minimum_stations}"
        )
    bands = {str(report.get("distance_band") or "") for report in reports}
    missing_bands = {"near", "middle", "far"} - bands
    if missing_bands:
        missing_evidence.append(
            "missing distance bands: " + ", ".join(sorted(missing_bands))
        )
    regions = {str(report.get("image_region") or "") for report in reports}
    regions.discard("")
    if len(regions) < minimum_image_regions:
        missing_evidence.append(
            f"only {len(regions)} image regions; need {minimum_image_regions}"
        )

    calibration_ids = set()
    calibration_hashes = set()
    all_rgbd_records: List[Mapping[str, Any]] = []
    station_biases: List[float] = []
    station_z: List[float] = []
    independent_known_base = False
    per_station: List[Dict[str, Any]] = []
    for report in reports:
        station_id = str(report.get("station_id") or "<unnamed>")
        if report.get("kind") != "eye_to_hand_rgbd_physical_station":
            failures.append(f"{station_id}: invalid report kind")
            continue
        source = report.get("source_calibration") or {}
        calibration_ids.add(str(source.get("calibration_id") or ""))
        calibration_hashes.add(str(source.get("sha256") or ""))
        camera = report.get("camera") or {}
        if str(camera.get("opened_serial") or "") != str(expected_serial):
            failures.append(f"{station_id}: camera serial mismatch")
        if str(camera.get("name") or "") != str(expected_camera_name):
            failures.append(
                f"{station_id}: camera name {camera.get('name')!r} does not match "
                f"{expected_camera_name!r}"
            )
        if not report.get("station_local_pass", False):
            failures.append(f"{station_id}: station_local_pass is false")
        raw_evidence = report.get("raw_rgbd_evidence")
        raw_count = len(raw_evidence) if isinstance(raw_evidence, list) else 0
        if raw_count < minimum_raw_evidence_frames:
            missing_evidence.append(
                f"{station_id}: only {raw_count} raw RGB-D evidence frames; "
                f"need {minimum_raw_evidence_frames}"
            )
        known_reference = report.get("known_base_reference") or {}
        independent_known_base = independent_known_base or bool(
            known_reference.get("independent_of_calibration", False)
            and known_reference.get("evidence_path")
            and known_reference.get("evidence_sha256")
        )
        frames = report.get("frames")
        if not isinstance(frames, list):
            failures.append(f"{station_id}: frames must be a list")
            continue
        rgbd_records = [
            record
            for record in frames
            if isinstance(record, dict)
            and bool(record.get("reprojection_gate_pass"))
            and bool(record.get("rgbd_valid"))
            and isinstance(record.get("rgbd"), dict)
        ]
        if len(rgbd_records) < minimum_rgbd_valid_frames:
            failures.append(
                f"{station_id}: only {len(rgbd_records)} RGB-D frames; "
                f"need {minimum_rgbd_valid_frames}"
            )
            continue
        all_rgbd_records.extend(rgbd_records)
        valid_fraction = _metrics(
            [record["rgbd"]["valid_depth_fraction"] for record in rgbd_records]
        )
        plane = _metrics(
            [record["rgbd"]["depth_plane_residual_p95_m"] for record in rgbd_records]
        )
        depth_z = _metrics(
            [record["rgbd"]["depth_minus_pnp_z_m"] for record in rgbd_records],
            absolute=True,
        )
        depth_3d = _metrics(
            [record["rgbd"]["depth_minus_pnp_3d_m"] for record in rgbd_records]
        )
        known_z = _metrics(
            [record["rgbd"]["depth_minus_known_base_z_m"] for record in rgbd_records],
            absolute=True,
        )
        known_3d = _metrics(
            [record["rgbd"]["depth_minus_known_base_3d_m"] for record in rgbd_records]
        )
        signed_bias = float(
            np.median(
                [record["rgbd"]["depth_minus_pnp_z_m"] for record in rgbd_records]
            )
        )
        target_z = float(
            np.median([record["T_camera_target"][2][3] for record in rgbd_records])
        )
        station_biases.append(signed_bias)
        station_z.append(target_z)
        metrics = {
            "station_id": station_id,
            "distance_band": report.get("distance_band"),
            "image_region": report.get("image_region"),
            "rgbd_valid_count": len(rgbd_records),
            "target_z_camera_median_m": target_z,
            "depth_valid_fraction": valid_fraction,
            "depth_plane_residual_p95_m": plane,
            "depth_minus_pnp_z_abs_m": depth_z,
            "depth_minus_pnp_3d_m": depth_3d,
            "depth_minus_known_base_z_abs_m": known_z,
            "depth_minus_known_base_3d_m": known_3d,
            "depth_minus_pnp_z_signed_median_m": signed_bias,
        }
        per_station.append(metrics)
        gates = (
            (
                valid_fraction["p05"] < minimum_depth_valid_fraction_p05,
                f"depth-valid p05 {valid_fraction['p05']:.6f} < {minimum_depth_valid_fraction_p05:.6f}",
            ),
            (
                plane["p95"] > max_plane_residual_p95_m,
                f"plane residual p95 {plane['p95'] * 1000.0:.3f} mm > {max_plane_residual_p95_m * 1000.0:.3f} mm",
            ),
            (
                depth_z["p95"] > max_depth_pnp_z_abs_p95_m,
                f"depth/PnP |z| p95 {depth_z['p95'] * 1000.0:.3f} mm > {max_depth_pnp_z_abs_p95_m * 1000.0:.3f} mm",
            ),
            (
                depth_z["max"] > max_depth_pnp_z_abs_max_m,
                f"depth/PnP |z| max {depth_z['max'] * 1000.0:.3f} mm > {max_depth_pnp_z_abs_max_m * 1000.0:.3f} mm",
            ),
            (
                depth_3d["p95"] > max_depth_pnp_3d_p95_m,
                f"depth/PnP 3D p95 {depth_3d['p95'] * 1000.0:.3f} mm > {max_depth_pnp_3d_p95_m * 1000.0:.3f} mm",
            ),
            (
                known_z["p95"] > max_known_base_z_abs_p95_m,
                f"known-base |z| p95 {known_z['p95'] * 1000.0:.3f} mm > {max_known_base_z_abs_p95_m * 1000.0:.3f} mm",
            ),
            (
                known_z["max"] > max_known_base_z_abs_max_m,
                f"known-base |z| max {known_z['max'] * 1000.0:.3f} mm > {max_known_base_z_abs_max_m * 1000.0:.3f} mm",
            ),
            (
                known_3d["p95"] > max_known_base_3d_p95_m,
                f"known-base 3D p95 {known_3d['p95'] * 1000.0:.3f} mm > {max_known_base_3d_p95_m * 1000.0:.3f} mm",
            ),
            (
                known_3d["max"] > max_known_base_3d_max_m,
                f"known-base 3D max {known_3d['max'] * 1000.0:.3f} mm > {max_known_base_3d_max_m * 1000.0:.3f} mm",
            ),
        )
        for failed, message in gates:
            if failed:
                failures.append(f"{station_id}: {message}")

    if len(calibration_ids) != 1 or "" in calibration_ids:
        failures.append("station reports do not share one non-empty calibration_id")
    if len(calibration_hashes) != 1 or "" in calibration_hashes:
        failures.append(
            "station reports do not share one non-empty calibration SHA-256"
        )
    distance_span_m = float(np.ptp(station_z)) if station_z else 0.0
    if distance_span_m < minimum_distance_span_m:
        missing_evidence.append(
            f"target distance span {distance_span_m:.6f} m is below "
            f"{minimum_distance_span_m:.6f} m"
        )
    bias_range_m = float(np.ptp(station_biases)) if station_biases else float("inf")
    if (
        not math.isfinite(bias_range_m)
        or bias_range_m > max_station_signed_bias_range_m
    ):
        failures.append(
            f"station signed depth/PnP bias range {bias_range_m * 1000.0:.3f} mm "
            f"exceeds {max_station_signed_bias_range_m * 1000.0:.3f} mm"
        )
    if require_independent_known_base and not independent_known_base:
        missing_evidence.append(
            "no independently surveyed/touch-off known-base marker centre with hashed evidence"
        )

    if failures:
        status = "rejected"
    elif missing_evidence:
        status = "provisional"
    else:
        status = "pass"
    return {
        "schema_version": 1,
        "kind": "eye_to_hand_rgbd_physical_validation",
        "physical_rgbd_status": status,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "camera_serial": str(expected_serial),
        "expected_camera_name": str(expected_camera_name),
        "calibration_id": next(iter(calibration_ids), ""),
        "calibration_sha256": next(iter(calibration_hashes), ""),
        "station_count": len(reports),
        "distance_bands": sorted(bands),
        "image_regions": sorted(regions),
        "target_distance_span_m": distance_span_m,
        "station_signed_depth_pnp_bias_range_m": bias_range_m,
        "independent_known_base_evidence_present": independent_known_base,
        "limits": {
            "minimum_stations": minimum_stations,
            "minimum_image_regions": minimum_image_regions,
            "minimum_distance_span_m": minimum_distance_span_m,
            "minimum_rgbd_valid_frames_per_station": minimum_rgbd_valid_frames,
            "minimum_raw_evidence_frames_per_station": minimum_raw_evidence_frames,
            "minimum_depth_valid_fraction_p05": minimum_depth_valid_fraction_p05,
            "max_plane_residual_p95_m": max_plane_residual_p95_m,
            "max_depth_pnp_z_abs_p95_m": max_depth_pnp_z_abs_p95_m,
            "max_depth_pnp_z_abs_max_m": max_depth_pnp_z_abs_max_m,
            "max_depth_pnp_3d_p95_m": max_depth_pnp_3d_p95_m,
            "max_known_base_z_abs_p95_m": max_known_base_z_abs_p95_m,
            "max_known_base_z_abs_max_m": max_known_base_z_abs_max_m,
            "max_known_base_3d_p95_m": max_known_base_3d_p95_m,
            "max_known_base_3d_max_m": max_known_base_3d_max_m,
            "max_station_signed_bias_range_m": max_station_signed_bias_range_m,
            "require_independent_known_base": require_independent_known_base,
        },
        "per_station": per_station,
        "failures": failures,
        "missing_evidence": missing_evidence,
        "scope_note": (
            "PASS covers only physical RGB-D validation; it is not full camera "
            "commissioning acceptance."
        ),
    }


def _capture_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "capture-station",
        help="capture one stationary read-only RGB-D/ArUco/base evidence report",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--camera-serial", required=True)
    parser.add_argument("--station-id", required=True)
    parser.add_argument(
        "--distance-band", choices=("near", "middle", "far"), required=True
    )
    parser.add_argument("--image-region", required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--max-reprojection-error-px", type=float, default=0.5)
    parser.add_argument("--min-reprojection-pass-frames", type=int, default=118)
    parser.add_argument("--min-coherent-frames", type=int, default=114)
    parser.add_argument("--max-pnp-translation-jitter-m", type=float, default=0.001)
    parser.add_argument("--max-pnp-rotation-jitter-deg", type=float, default=0.3)
    parser.add_argument(
        "--max-untrimmed-translation-jitter-m", type=float, default=0.001
    )
    parser.add_argument("--max-untrimmed-rotation-jitter-deg", type=float, default=0.3)
    parser.add_argument(
        "--max-untrimmed-reprojection-error-px", type=float, default=0.5
    )
    parser.add_argument("--min-rgbd-valid-frames", type=int, default=114)
    parser.add_argument("--interior-scale", type=float, default=0.65)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--depth-min-m", type=float, default=0.8)
    parser.add_argument("--depth-max-m", type=float, default=1.4)
    parser.add_argument("--max-depth-from-pnp-m", type=float, default=0.05)
    parser.add_argument("--plane-residual-limit-m", type=float, default=0.01)
    parser.add_argument("--minimum-depth-points", type=int, default=500)
    parser.add_argument("--intrinsics-tolerance-px", type=float, default=0.05)
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument("--max-robot-translation-m", type=float, default=0.0005)
    parser.add_argument("--max-robot-rotation-deg", type=float, default=0.1)
    parser.add_argument("--known-base-center-m", nargs=3, type=float)
    parser.add_argument(
        "--known-base-source",
        choices=("surveyed_fixture", "robot_touch_off", "metrology_fixture"),
    )
    parser.add_argument("--known-base-evidence", type=Path)
    parser.add_argument(
        "--session-target-report",
        type=Path,
        help=(
            "passed reused-extrinsic holdout report providing a hashed "
            "session-local T_ee_target after target remount"
        ),
    )
    parser.add_argument(
        "--confirm-stationary-read-only",
        action="store_true",
        help="required acknowledgement; this tool reads camera/robot state but never moves",
    )
    parser.set_defaults(command_func=_capture_main)


def _evaluate_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "evaluate", help="offline multi-station acceptance gate"
    )
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-serial", required=True)
    parser.add_argument("--expected-camera-name", required=True)
    parser.add_argument("--minimum-stations", type=int, default=3)
    parser.add_argument("--minimum-image-regions", type=int, default=3)
    parser.add_argument("--minimum-distance-span-m", type=float, required=True)
    parser.add_argument("--minimum-rgbd-valid-frames", type=int, default=114)
    parser.add_argument("--minimum-raw-evidence-frames", type=int, default=3)
    parser.add_argument("--minimum-depth-valid-fraction-p05", type=float, required=True)
    parser.add_argument("--max-plane-residual-p95-mm", type=float, required=True)
    parser.add_argument("--max-depth-pnp-z-abs-p95-mm", type=float, required=True)
    parser.add_argument("--max-depth-pnp-z-abs-max-mm", type=float, required=True)
    parser.add_argument("--max-depth-pnp-3d-p95-mm", type=float, required=True)
    parser.add_argument("--max-known-base-z-abs-p95-mm", type=float, required=True)
    parser.add_argument("--max-known-base-z-abs-max-mm", type=float, required=True)
    parser.add_argument("--max-known-base-3d-p95-mm", type=float, required=True)
    parser.add_argument("--max-known-base-3d-max-mm", type=float, required=True)
    parser.add_argument("--max-station-signed-bias-range-mm", type=float, required=True)
    parser.add_argument("--require-independent-known-base", action="store_true")
    parser.set_defaults(command_func=_evaluate_main)


def _capture_main(args: argparse.Namespace) -> int:
    if not args.confirm_stationary_read_only:
        raise PhysicalValidationError(
            "--confirm-stationary-read-only is required before hardware access"
        )
    document = capture_station(
        config_path=args.config,
        calibration_path=args.calibration,
        output_path=args.output,
        evidence_dir=args.evidence_dir,
        camera_serial=args.camera_serial,
        station_id=args.station_id,
        distance_band=args.distance_band,
        image_region=args.image_region,
        frames=args.frames,
        max_reprojection_error_px=args.max_reprojection_error_px,
        min_reprojection_pass_frames=args.min_reprojection_pass_frames,
        min_coherent_frames=args.min_coherent_frames,
        max_pnp_translation_jitter_m=args.max_pnp_translation_jitter_m,
        max_pnp_rotation_jitter_deg=args.max_pnp_rotation_jitter_deg,
        max_untrimmed_translation_jitter_m=args.max_untrimmed_translation_jitter_m,
        max_untrimmed_rotation_jitter_deg=args.max_untrimmed_rotation_jitter_deg,
        max_untrimmed_reprojection_error_px=args.max_untrimmed_reprojection_error_px,
        min_rgbd_valid_frames=args.min_rgbd_valid_frames,
        interior_scale=args.interior_scale,
        pixel_stride=args.pixel_stride,
        depth_min_m=args.depth_min_m,
        depth_max_m=args.depth_max_m,
        max_depth_from_pnp_m=args.max_depth_from_pnp_m,
        plane_residual_limit_m=args.plane_residual_limit_m,
        minimum_depth_points=args.minimum_depth_points,
        intrinsics_tolerance_px=args.intrinsics_tolerance_px,
        robot_ip=args.robot_ip,
        max_robot_translation_m=args.max_robot_translation_m,
        max_robot_rotation_deg=args.max_robot_rotation_deg,
        known_base_center_m=args.known_base_center_m,
        known_base_source=args.known_base_source,
        known_base_evidence=args.known_base_evidence,
        session_target_report=args.session_target_report,
    )
    print(
        "RGBD_STATION_JSON="
        + json.dumps(
            {
                "output": str(args.output.resolve()),
                "station_id": document["station_id"],
                "station_local_pass": document["station_local_pass"],
                "rgbd_valid_count": document["station_metrics"]["rgbd_valid_count"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0 if document["station_local_pass"] else 2


def _verify_hashed_file(path_value: Any, sha_value: Any, label: str) -> Dict[str, str]:
    path = Path(str(path_value or "")).expanduser().resolve()
    expected_sha = str(sha_value or "").lower()
    if not path.is_file():
        raise PhysicalValidationError(f"{label} file does not exist: {path}")
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise PhysicalValidationError(
            f"{label} SHA-256 mismatch: expected={expected_sha}, actual={actual_sha}"
        )
    return {"path": str(path), "sha256": actual_sha}


def _verify_station_report_provenance(
    report: Mapping[str, Any], report_path: Path
) -> Dict[str, Any]:
    report_artifact = _verify_hashed_file(
        report_path,
        _sha256(report_path),
        "station report",
    )
    calibration = report.get("source_calibration") or {}
    config = report.get("source_camera_config") or {}
    calibration_artifact = _verify_hashed_file(
        calibration.get("path"), calibration.get("sha256"), "source calibration"
    )
    config_artifact = _verify_hashed_file(
        config.get("path"), config.get("sha256"), "source camera config"
    )
    raw = report.get("raw_rgbd_evidence")
    if not isinstance(raw, list):
        raise PhysicalValidationError("raw_rgbd_evidence must be a list")
    raw_artifacts = [
        _verify_hashed_file(item.get("path"), item.get("sha256"), "raw RGB-D evidence")
        for item in raw
        if isinstance(item, dict)
    ]
    if len(raw_artifacts) != len(raw):
        raise PhysicalValidationError("raw_rgbd_evidence contains a non-mapping item")
    known = report.get("known_base_reference") or {}
    known_artifact = None
    if known.get("independent_of_calibration"):
        known_artifact = _verify_hashed_file(
            known.get("evidence_path"),
            known.get("evidence_sha256"),
            "known-base evidence",
        )
    return {
        "station_report": report_artifact,
        "source_calibration": calibration_artifact,
        "source_camera_config": config_artifact,
        "raw_rgbd_evidence": raw_artifacts,
        "known_base_evidence": known_artifact,
    }


def _evaluate_main(args: argparse.Namespace) -> int:
    reports = []
    provenance = []
    for path in args.reports:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PhysicalValidationError(f"cannot load report {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise PhysicalValidationError(f"report {path} must contain a JSON object")
        provenance.append(_verify_station_report_provenance(value, path.resolve()))
        reports.append(value)
    result = evaluate_station_reports(
        reports,
        expected_serial=args.expected_serial,
        expected_camera_name=args.expected_camera_name,
        minimum_stations=args.minimum_stations,
        minimum_image_regions=args.minimum_image_regions,
        minimum_distance_span_m=args.minimum_distance_span_m,
        minimum_rgbd_valid_frames=args.minimum_rgbd_valid_frames,
        minimum_raw_evidence_frames=args.minimum_raw_evidence_frames,
        minimum_depth_valid_fraction_p05=args.minimum_depth_valid_fraction_p05,
        max_plane_residual_p95_m=args.max_plane_residual_p95_mm / 1000.0,
        max_depth_pnp_z_abs_p95_m=args.max_depth_pnp_z_abs_p95_mm / 1000.0,
        max_depth_pnp_z_abs_max_m=args.max_depth_pnp_z_abs_max_mm / 1000.0,
        max_depth_pnp_3d_p95_m=args.max_depth_pnp_3d_p95_mm / 1000.0,
        max_known_base_z_abs_p95_m=args.max_known_base_z_abs_p95_mm / 1000.0,
        max_known_base_z_abs_max_m=args.max_known_base_z_abs_max_mm / 1000.0,
        max_known_base_3d_p95_m=args.max_known_base_3d_p95_mm / 1000.0,
        max_known_base_3d_max_m=args.max_known_base_3d_max_mm / 1000.0,
        max_station_signed_bias_range_m=(
            args.max_station_signed_bias_range_mm / 1000.0
        ),
        require_independent_known_base=args.require_independent_known_base,
    )
    result["source_station_reports"] = provenance
    _write_json_exclusive_atomic(args.output, result)
    print(
        "RGBD_PHYSICAL_VALIDATION_JSON="
        + json.dumps(result, separators=(",", ":"), sort_keys=True)
    )
    return {"pass": 0, "provisional": 1, "rejected": 2}[result["physical_rgbd_status"]]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _capture_parser(subparsers)
    _evaluate_parser(subparsers)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.command_func(args))
    except (PhysicalValidationError, ValueError, OSError, cv2.error) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
