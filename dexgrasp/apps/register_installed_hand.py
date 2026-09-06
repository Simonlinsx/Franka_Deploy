#!/usr/bin/env python3
"""Read-only D435 registration of an installed RH56 to the Franka EE frame.

The operator prompts only the rigid palm/wrist housing.  The program samples
the official AnyDex ``Link111`` mesh, applies the official source offset once,
fits mechanically valid yaw candidates, and reports ``T_EE_hand``.  It never
creates a Franka controller and never imports or writes the RH56 serial driver.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
SRC = ROOT / "src"
DYNAMIC_PCD_ROOT = WORKSPACE / "perception"
for path in (SRC, WORKSPACE, DYNAMIC_PCD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from anydex_pipeline.control_config import load_control_config
from anydex_pipeline.control_frames import T_MOUNT_SOURCE_AXIS_BASIS
from anydex_pipeline.control_plan import inverse_rigid_transform, validate_rigid_transform
from anydex_pipeline.inspire_hand_model import OFFICIAL_SOURCE_MESH_OFFSET_M
from anydex_pipeline.installed_hand_registration import (
    ICPOptions,
    RegistrationThresholds,
    compose_T_EE_hand,
    decompose_installed_mount,
    register_rigid_hand_model,
    rigid_delta,
    rotation_angle_deg,
    transform_points,
)


DEFAULT_CONTROL_CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"
DEFAULT_CAMERA_CONFIG = DYNAMIC_PCD_ROOT / "configs/d435_default.yaml"
DEFAULT_UPSTREAM = ROOT / "third_party/AnyDexGrasp"
SCHEMA_VERSION = 1


def _franka_matrix(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (16,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain 16 finite column-major values")
    return validate_rigid_transform(array.reshape((4, 4), order="F"), name)


def _matrix_json(matrix: np.ndarray) -> list[list[float]]:
    value = validate_rigid_transform(matrix, "serialized transform")
    return [[float(item) for item in row] for row in value]


def _vector_json(vector: Any) -> list[float]:
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("serialized vector must be finite")
    return [float(value) for value in values]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str, overwrite: bool) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists (pass --overwrite): {output}")
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{output.name}.",
            suffix=".tmp", dir=output.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray], overwrite: bool) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists (pass --overwrite): {output}")
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{output.name}.", suffix=".tmp",
            dir=output.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_state(robot: Any) -> dict[str, Any]:
    state = robot.read_once()
    return {
        "robot_mode": str(state.robot_mode),
        "q": np.asarray(state.q, dtype=np.float64).copy(),
        "dq": np.asarray(state.dq, dtype=np.float64).copy(),
        "O_T_EE": _franka_matrix(state.O_T_EE, "O_T_EE"),
        "F_T_EE": _franka_matrix(state.F_T_EE, "F_T_EE"),
        "read_at_unix_s": time.time(),
    }


def _voxel_cloud(points: np.ndarray, colors: np.ndarray, voxel_m: float) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    color_values = np.asarray(colors, dtype=np.float64)
    if color_values.shape == np.asarray(points).shape:
        point_cloud.colors = o3d.utility.Vector3dVector(np.clip(color_values, 0.0, 1.0))
    point_cloud = point_cloud.voxel_down_sample(float(voxel_m))
    output_points = np.asarray(point_cloud.points, dtype=np.float64)
    output_colors = np.asarray(point_cloud.colors, dtype=np.float64)
    if output_colors.shape != output_points.shape:
        output_colors = np.zeros_like(output_points)
    return output_points, output_colors


def _capture_batches(args: argparse.Namespace, control: dict[str, Any], robot: Any) -> dict[str, Any]:
    from dynamic_pcd.config import load_config
    from dynamic_pcd.provider.object_pcd_provider import ObjectPCDProvider

    camera_config_path = args.camera_config.expanduser().resolve()
    cfg = load_config(str(camera_config_path))
    cfg.setdefault("sam2", {})["enabled"] = False
    cfg.setdefault("pointcloud", {})["voxel_size"] = float(args.capture_voxel_m)
    cfg["pointcloud"]["center_policy_points"] = False
    provider = ObjectPCDProvider(cfg)

    expected_serial = str(control["calibration"]["camera_serial"])
    expected_calibration = str(control["calibration"]["id"])
    extrinsics = provider.extrinsics
    if not bool(extrinsics.calibrated):
        raise RuntimeError("eye-to-hand calibration is required")
    if str(extrinsics.reference_frame) != "robot_base":
        raise RuntimeError("registration requires calibration reference_frame=robot_base")
    if str(extrinsics.camera_serial or "") != expected_serial:
        raise RuntimeError("camera calibration serial does not match control profile")
    if str(extrinsics.calibration_id or "") != expected_calibration:
        raise RuntimeError("camera calibration ID does not match control profile")
    if str(getattr(extrinsics, "quality_status", "pass")) != "pass":
        raise RuntimeError("camera calibration quality is not pass")

    before = _read_state(robot)
    batches = []
    batch_colors = []
    frame_ids = []
    frame_timestamps = []
    batch_states = []
    roi_bbox = None
    scene_points = np.zeros((0, 3), dtype=np.float64)
    scene_colors = np.zeros((0, 3), dtype=np.float64)
    try:
        provider.start()
        if str(provider.camera.device_serial or "") != expected_serial:
            raise RuntimeError("opened D435 serial does not match control profile")
        print(
            "[prompt] 只框选 RH56 的刚性手掌/腕部白色外壳；"
            "排除手指、Franka 法兰、金属适配器、黑色线缆和背景"
        )
        if args.roi is None:
            initialized = provider.select_and_initialize()
        else:
            frame = provider.camera.get_frame(timeout_ms=args.frame_timeout_ms)
            initialized = provider.initialize_from_bbox(
                frame, np.asarray(args.roi, dtype=np.int32)
            )
        if not initialized:
            raise RuntimeError("hand ROI selection was cancelled or initialization failed")
        provider.set_roi_locked(True)

        last_frame = None
        last_mask = None
        for batch_index in range(args.capture_batches):
            points = []
            colors = []
            accepted = 0
            deadline = time.monotonic() + args.batch_timeout_s
            while accepted < args.frames_per_batch and time.monotonic() < deadline:
                frame, mask, obj, packet = provider.step()
                if not (mask.valid and obj.valid and packet.valid):
                    continue
                if len(obj.points) < args.min_points_per_frame:
                    continue
                points.append(np.asarray(obj.points, dtype=np.float64))
                colors.append(np.asarray(obj.colors, dtype=np.float64))
                frame_ids.append(int(frame.frame_id))
                frame_timestamps.append(float(frame.timestamp))
                roi_bbox = np.asarray(mask.bbox_xyxy, dtype=np.int32)
                last_frame = frame
                last_mask = mask.mask
                accepted += 1
            if accepted != args.frames_per_batch:
                raise RuntimeError(
                    f"batch {batch_index + 1} captured only "
                    f"{accepted}/{args.frames_per_batch} valid hand frames"
                )
            merged_points = np.concatenate(points, axis=0)
            merged_colors = np.concatenate(colors, axis=0)
            merged_points, merged_colors = _voxel_cloud(
                merged_points, merged_colors, args.registration_voxel_m
            )
            if len(merged_points) > args.max_points_per_batch:
                generator = np.random.default_rng(args.seed + batch_index)
                indices = generator.choice(
                    len(merged_points), args.max_points_per_batch, replace=False
                )
                merged_points = merged_points[indices]
                merged_colors = merged_colors[indices]
            if len(merged_points) < args.min_batch_points:
                raise RuntimeError(
                    f"batch {batch_index + 1} has only {len(merged_points)} "
                    f"points after filtering; require {args.min_batch_points}"
                )
            batches.append(merged_points)
            batch_colors.append(merged_colors)
            batch_states.append(_read_state(robot))
            print(
                f"[capture] batch={batch_index + 1}/{args.capture_batches} "
                f"frames={accepted} points={len(merged_points)}"
            )
        assert last_frame is not None
        scene = provider.extractor.extract_scene(
            last_frame, exclude_mask=last_mask, stride=args.scene_stride
        )
        scene_points = np.asarray(scene.points, dtype=np.float64)
        scene_colors = np.asarray(scene.colors, dtype=np.float64)
    finally:
        provider.stop()
    after = _read_state(robot)
    return {
        "before": before,
        "after": after,
        "batches": batches,
        "batch_colors": batch_colors,
        "scene_points": scene_points,
        "scene_colors": scene_colors,
        "roi_bbox": roi_bbox,
        "frame_ids": np.asarray(frame_ids, dtype=np.int64),
        "frame_timestamps": np.asarray(frame_timestamps, dtype=np.float64),
        "batch_states": batch_states,
        "T_base_camera": np.asarray(extrinsics.T_base_camera, dtype=np.float64),
        "calibration_id": str(extrinsics.calibration_id),
        "calibration_source": str(extrinsics.source),
        "camera_serial": str(provider.camera.device_serial),
        "camera_name": str(provider.camera.device_name),
        "camera_config_path": camera_config_path,
    }


def _save_preview(args: argparse.Namespace, control: dict[str, Any]) -> Path:
    """Save one current aligned color frame without opening Franka or RH56."""

    import cv2
    from dynamic_pcd.camera.realsense_camera import RealSenseCamera
    from dynamic_pcd.config import load_config

    camera_config_path = args.camera_config.expanduser().resolve()
    cfg = load_config(str(camera_config_path))
    camera = RealSenseCamera(cfg["camera"])
    try:
        camera.start()
        expected_serial = str(control["calibration"]["camera_serial"])
        if str(camera.device_serial or "") != expected_serial:
            raise RuntimeError("opened D435 serial does not match control profile")
        frame = None
        for _ in range(args.preview_warmup_frames + 1):
            frame = camera.get_frame(timeout_ms=args.frame_timeout_ms)
        assert frame is not None
        output = args.save_preview.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"preview exists (pass --overwrite): {output}")
        if not cv2.imwrite(str(output), frame.color_bgr):
            raise RuntimeError(f"failed to save preview image: {output}")
        print(
            f"[preview] saved {output} frame={frame.frame_id} "
            f"camera={camera.device_serial}; Franka was not opened"
        )
        return output
    finally:
        camera.stop()


def _load_link111_points(args: argparse.Namespace) -> tuple[np.ndarray, Path, Path, Any]:
    import open3d as o3d

    hand_root = (
        args.upstream_root.expanduser().resolve()
        / "generate_mesh_and_pointcloud/inspire_urdf/urdf-five3"
    )
    mesh_path = hand_root / "meshes/Link111.STL"
    urdf_path = hand_root / "robots/urdf-five3.urdf"
    if not mesh_path.is_file() or not urdf_path.is_file():
        raise FileNotFoundError("official Inspire Link111 mesh/URDF is missing")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty() or len(mesh.triangles) < 100:
        raise RuntimeError("Link111 mesh could not be loaded")
    extent = np.asarray(mesh.get_max_bound()) - np.asarray(mesh.get_min_bound())
    if not 0.05 < float(np.max(extent)) < 0.30:
        raise RuntimeError(
            "Link111 mesh scale is implausible; expected metres, "
            f"extent={extent.tolist()}"
        )
    o3d.utility.random.seed(int(args.seed))
    sampled = mesh.sample_points_uniformly(number_of_points=args.model_points)
    points = np.asarray(sampled.points, dtype=np.float64)
    # Raw STL is Link111 visual coordinates.  Apply H_T_Link111 exactly once.
    points = points + OFFICIAL_SOURCE_MESH_OFFSET_M[None, :]
    points = points[points[:, 0] >= float(args.model_min_source_x_m)]
    points, _ = _voxel_cloud(points, np.zeros_like(points), args.registration_voxel_m)
    if len(points) < 1000:
        raise RuntimeError("too few Link111 model points after rigid-palm crop")
    return points, mesh_path, urdf_path, mesh


def _yaw_values(prior_deg: float, search_deg: float, step_deg: float, opposite: bool) -> list[float]:
    centers = [float(prior_deg)]
    if opposite:
        centers.append(float(prior_deg) + 180.0)
    values = []
    count = int(np.floor((2.0 * search_deg) / step_deg + 1.0e-9))
    offsets = np.linspace(-search_deg, search_deg, count + 1)
    for center in centers:
        values.extend((center + offsets).tolist())
    unique = []
    for value in values:
        normalized = ((float(value) + 180.0) % 360.0) - 180.0
        if not any(abs(normalized - existing) < 1.0e-9 for existing in unique):
            unique.append(normalized)
    return unique


def _orientation_candidates(
    T_base_EE: np.ndarray,
    F_T_EE: np.ndarray,
    yaw_values_deg: Sequence[float],
) -> list[tuple[str, np.ndarray, float]]:
    candidates = []
    for yaw_deg in yaw_values_deg:
        angle = np.deg2rad(float(yaw_deg))
        cosine, sine = np.cos(angle), np.sin(angle)
        Rz = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        T_F_hand = np.eye(4, dtype=np.float64)
        T_F_hand[:3, :3] = Rz @ T_MOUNT_SOURCE_AXIS_BASIS[:3, :3]
        T_EE_hand = inverse_rigid_transform(F_T_EE) @ T_F_hand
        rotation = T_base_EE[:3, :3] @ T_EE_hand[:3, :3]
        candidates.append((f"assembled_yaw={yaw_deg:.3f}deg", rotation, float(yaw_deg)))
    return candidates


def _fit(args: argparse.Namespace, control: dict[str, Any], capture: dict[str, Any]) -> dict[str, Any]:
    model_points, mesh_path, urdf_path, mesh = _load_link111_points(args)
    before = capture["before"]
    after = capture["after"]
    pose_drift_m, pose_drift_deg = rigid_delta(before["O_T_EE"], after["O_T_EE"])
    q_drift = float(np.max(np.abs(after["q"] - before["q"])))
    if not np.allclose(before["F_T_EE"], after["F_T_EE"], atol=1.0e-9):
        raise RuntimeError("F_T_EE changed during static registration")
    expected_F_T_EE = np.asarray(control["franka"]["expected_F_T_EE"], dtype=np.float64)
    if not np.allclose(before["F_T_EE"], expected_F_T_EE, atol=1.0e-6):
        raise RuntimeError("live F_T_EE differs from the commissioning profile")

    yaw_prior_deg = (
        float(args.yaw_prior_deg)
        if args.yaw_prior_deg is not None
        else float(np.rad2deg(control["tool"]["assembled_yaw_rad"]))
    )
    yaw_values = _yaw_values(
        yaw_prior_deg, args.yaw_search_deg, args.yaw_step_deg,
        not args.no_opposite_branch,
    )
    candidates_with_yaw = _orientation_candidates(
        before["O_T_EE"], before["F_T_EE"], yaw_values
    )
    candidates = [(label, rotation) for label, rotation, _ in candidates_with_yaw]
    options = ICPOptions(
        correspondence_schedule_m=tuple(args.correspondence_schedule_m),
        iterations_per_stage=args.iterations_per_stage,
        trim_fraction=args.trim_fraction,
        min_correspondences=args.min_correspondences,
        lock_candidate_orientation=True,
    )
    thresholds = RegistrationThresholds(
        evaluation_inlier_distance_m=args.inlier_distance_m,
        min_observed_inlier_ratio=args.min_inlier_ratio,
        max_observed_p50_m=args.max_p50_m,
        max_observed_p90_m=args.max_p90_m,
        max_orientation_correction_deg=0.25,
        max_multistart_score_gap_ratio=args.ambiguity_score_ratio,
        ambiguity_min_orientation_separation_deg=30.0,
    )
    merged_points, merged_colors = _voxel_cloud(
        np.concatenate(capture["batches"], axis=0),
        np.concatenate(capture["batch_colors"], axis=0),
        args.registration_voxel_m,
    )
    merged = register_rigid_hand_model(
        model_points, merged_points, candidates, options=options, thresholds=thresholds
    )
    best_rotation = merged.T_reference_hand[:3, :3]
    local_candidates = [
        (label, rotation)
        for label, rotation in candidates
        if rotation_angle_deg(best_rotation.T @ rotation)
        <= max(2.1 * args.yaw_step_deg, 3.0)
    ]
    batch_results = [
        register_rigid_hand_model(
            model_points, points, local_candidates,
            options=options, thresholds=thresholds,
        )
        for points in capture["batches"]
    ]

    T_EE_hand = compose_T_EE_hand(before["O_T_EE"], merged.T_reference_hand)
    decomposition = decompose_installed_mount(
        T_EE_hand,
        before["F_T_EE"],
        fr3_face_to_rh56_seating_plane_m=float(
            control["tool"]["fr3_face_to_rh56_seating_plane_m"]
        ),
        T_mount_source_axis_basis=T_MOUNT_SOURCE_AXIS_BASIS,
    )
    batch_T_EE = [
        compose_T_EE_hand(before["O_T_EE"], result.T_reference_hand)
        for result in batch_results
    ]
    pairwise_translation = []
    pairwise_rotation = []
    for first in range(len(batch_T_EE)):
        for second in range(first + 1, len(batch_T_EE)):
            translation, rotation = rigid_delta(batch_T_EE[first], batch_T_EE[second])
            pairwise_translation.append(translation)
            pairwise_rotation.append(rotation)
    max_batch_translation = max(pairwise_translation, default=0.0)
    max_batch_rotation = max(pairwise_rotation, default=0.0)

    reasons = list(merged.rejection_reasons)
    if merged.metrics.observed_p95_m > args.max_p95_m:
        reasons.append("p95 observed-to-model distance is too high")
    if merged.metrics.observed_trimmed_rmse_m > args.max_trimmed_rmse_m:
        reasons.append("trimmed observed-to-model RMSE is too high")
    if any(not result.geometry_gate_passed for result in batch_results):
        reasons.append("at least one independent capture batch failed its geometry gate")
    if max_batch_translation > args.max_batch_translation_m:
        reasons.append("independent capture batches disagree in translation")
    if max_batch_rotation > args.max_batch_rotation_deg:
        reasons.append("independent capture batches disagree in yaw")
    if pose_drift_m > args.max_robot_pose_drift_m:
        reasons.append("Franka EE translated during the static capture")
    if pose_drift_deg > args.max_robot_pose_drift_deg:
        reasons.append("Franka EE rotated during the static capture")
    if q_drift > args.max_robot_joint_drift_rad:
        reasons.append("Franka joints changed during the static capture")
    if decomposition["mount_axis_residual_deg"] > args.max_mount_axis_residual_deg:
        reasons.append("fitted orientation violates the adapter mounting axis")
    seating_origin = decomposition["seating_to_hand_source_origin_m"]
    if float(np.linalg.norm(seating_origin[:2])) > args.max_seating_lateral_offset_m:
        reasons.append("seating-to-source lateral offset is mechanically implausible")
    if not args.min_seating_axial_offset_m <= float(seating_origin[2]) <= args.max_seating_axial_offset_m:
        reasons.append("seating-to-source axial offset is mechanically implausible")

    fitted_model_points = transform_points(merged.T_reference_hand, model_points)
    return {
        "model_points": model_points,
        "fitted_model_points": fitted_model_points,
        "merged_points": merged_points,
        "merged_colors": merged_colors,
        "mesh_path": mesh_path,
        "urdf_path": urdf_path,
        "mesh": mesh,
        "merged_result": merged,
        "batch_results": batch_results,
        "batch_T_EE_hand": batch_T_EE,
        "T_EE_hand": T_EE_hand,
        "decomposition": decomposition,
        "geometry_gate_passed": not reasons,
        "rejection_reasons": tuple(dict.fromkeys(reasons)),
        "pose_drift_m": pose_drift_m,
        "pose_drift_deg": pose_drift_deg,
        "q_drift_rad": q_drift,
        "max_batch_translation_m": max_batch_translation,
        "max_batch_rotation_deg": max_batch_rotation,
        "yaw_prior_deg": yaw_prior_deg,
        "yaw_values_deg": yaw_values,
    }


def _artifact(
    args: argparse.Namespace,
    control_path: Path,
    capture: dict[str, Any],
    fit: dict[str, Any],
    npz_path: Path,
) -> dict[str, Any]:
    before, after = capture["before"], capture["after"]
    result = fit["merged_result"]
    decomposition = fit["decomposition"]
    calibration_path = Path(capture["calibration_source"])
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "fr3_rh56_static_hand_mount_registration",
        "created_at_utc": now,
        "safety_scope": {
            "franka_access": "Robot.read_once only; no controller created",
            "inspire_access": "not imported or opened; no register written",
            "motion_command_created": False,
        },
        "matrix_convention": {
            "transform": "T_A_B maps coordinates in frame B into frame A",
            "serialization": "row-major nested 4x4",
            "translation_units": "metres",
            "angles": "degrees in reports; radians in assembled_yaw_rad",
        },
        "inputs": {
            "control_config": str(control_path),
            "control_config_sha256": _sha256(control_path),
            "camera_config": str(capture["camera_config_path"]),
            "camera_config_sha256": _sha256(capture["camera_config_path"]),
            "camera_serial": capture["camera_serial"],
            "camera_name": capture["camera_name"],
            "calibration_id": capture["calibration_id"],
            "calibration_source": capture["calibration_source"],
            "calibration_sha256": _sha256(calibration_path) if calibration_path.is_file() else "",
            "T_robot_base_camera": _matrix_json(capture["T_base_camera"]),
            "roi_bbox_xyxy": np.asarray(capture["roi_bbox"], dtype=int).tolist(),
            "frame_ids": capture["frame_ids"].astype(int).tolist(),
            "capture_batches": int(args.capture_batches),
            "frames_per_batch": int(args.frames_per_batch),
        },
        "franka_static_state": {
            "robot_mode_before": before["robot_mode"],
            "robot_mode_after": after["robot_mode"],
            "q_before_rad": _vector_json(before["q"]),
            "q_after_rad": _vector_json(after["q"]),
            "dq_before_rad_s": _vector_json(before["dq"]),
            "dq_after_rad_s": _vector_json(after["dq"]),
            "T_robot_base_EE_before": _matrix_json(before["O_T_EE"]),
            "T_robot_base_EE_after": _matrix_json(after["O_T_EE"]),
            "F_T_EE": _matrix_json(before["F_T_EE"]),
            "pose_drift_m": fit["pose_drift_m"],
            "pose_drift_deg": fit["pose_drift_deg"],
            "max_joint_drift_rad": fit["q_drift_rad"],
        },
        "source_geometry": {
            "rigid_link": "Link111",
            "source_geometry_frame": "AnyDex generated source mesh coordinates",
            "mesh_path": str(fit["mesh_path"]),
            "mesh_sha256": _sha256(fit["mesh_path"]),
            "urdf_path": str(fit["urdf_path"]),
            "urdf_sha256": _sha256(fit["urdf_path"]),
            "H_T_Link111_translation_m": _vector_json(OFFICIAL_SOURCE_MESH_OFFSET_M),
            "geometry_offset_applied_once": True,
            "complete_installed_collision_model": False,
        },
        "registration": {
            "method": "mechanically constrained yaw-grid + robust trimmed translation ICP",
            "yaw_prior_deg": fit["yaw_prior_deg"],
            "yaw_candidates_deg": fit["yaw_values_deg"],
            "selected_orientation": result.orientation_candidate_label,
            "metrics": result.metrics_dict(),
            "batch_metrics": [item.metrics_dict() for item in fit["batch_results"]],
            "max_batch_translation_m": fit["max_batch_translation_m"],
            "max_batch_rotation_deg": fit["max_batch_rotation_deg"],
            "ambiguous_180deg_branch": bool(result.ambiguous_multistart),
        },
        "result": {
            "T_robot_base_hand_source": _matrix_json(result.T_reference_hand),
            "T_EE_hand": _matrix_json(fit["T_EE_hand"]),
            "assembled_yaw_rad": float(decomposition["assembled_yaw_rad"]),
            "assembled_yaw_deg": float(decomposition["assembled_yaw_deg"]),
            "seating_to_hand_source_origin_m": _vector_json(
                decomposition["seating_to_hand_source_origin_m"]
            ),
            "mount_axis_residual_deg": float(decomposition["mount_axis_residual_deg"]),
        },
        "measurement_status": {
            "geometry_gate_passed": bool(fit["geometry_gate_passed"]),
            "rejection_reasons": list(fit["rejection_reasons"]),
            "operator_overlay_confirmed": False,
            "mount_transform_commissioned": False,
            "collision_model_validated": False,
            "default_path_collision_verified": False,
            "full_execution_ready": False,
            "scope": "candidate mount transform only; visual confirmation is required",
        },
        "point_cloud_artifact": str(npz_path),
    }


def _show(capture: dict[str, Any], fit: dict[str, Any]) -> None:
    import open3d as o3d

    scene = o3d.geometry.PointCloud()
    scene.points = o3d.utility.Vector3dVector(capture["scene_points"])
    colors = np.asarray(capture["scene_colors"], dtype=np.float64)
    if colors.shape == capture["scene_points"].shape:
        scene.colors = o3d.utility.Vector3dVector(np.clip(colors * 0.35, 0.03, 0.35))
    observed = o3d.geometry.PointCloud()
    observed.points = o3d.utility.Vector3dVector(fit["merged_points"])
    observed.paint_uniform_color([1.0, 0.35, 0.05])

    mesh = copy.deepcopy(fit["mesh"])
    mesh.translate(OFFICIAL_SOURCE_MESH_OFFSET_M, relative=True)
    mesh.transform(fit["merged_result"].T_reference_hand)
    mesh.paint_uniform_color([0.10, 0.75, 0.95])
    mesh.compute_vertex_normals()
    ee_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.055)
    ee_frame.transform(capture["before"]["O_T_EE"])
    hand_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.045)
    hand_frame.transform(fit["merged_result"].T_reference_hand)
    try:
        o3d.visualization.draw_geometries(
            [scene, observed, mesh, ee_frame, hand_frame],
            window_name="RH56 static registration | orange=observed cyan=Link111",
            width=1280,
            height=800,
        )
    except KeyboardInterrupt:
        print("[viewer] interrupted after artifacts were saved")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only static registration of installed RH56 Link111 to D435 "
            "and Franka O_T_EE. Produces a candidate T_EE_hand; never moves hardware."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONTROL_CONFIG)
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument(
        "--save-preview",
        type=Path,
        default=None,
        help="Save one D435 color frame and exit without opening Franka",
    )
    parser.add_argument("--preview-warmup-frames", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--capture-batches", type=int, default=3)
    parser.add_argument("--frames-per-batch", type=int, default=12)
    parser.add_argument("--batch-timeout-s", type=float, default=20.0)
    parser.add_argument("--frame-timeout-ms", type=int, default=2000)
    parser.add_argument("--min-points-per-frame", type=int, default=250)
    parser.add_argument("--min-batch-points", type=int, default=1000)
    parser.add_argument("--max-points-per-batch", type=int, default=10000)
    parser.add_argument("--scene-stride", type=int, default=4)
    parser.add_argument("--capture-voxel-m", type=float, default=0.002)
    parser.add_argument("--registration-voxel-m", type=float, default=0.002)
    parser.add_argument("--model-points", type=int, default=50000)
    parser.add_argument("--model-min-source-x-m", type=float, default=0.020)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--yaw-prior-deg", type=float, default=None)
    parser.add_argument("--yaw-search-deg", type=float, default=12.0)
    parser.add_argument("--yaw-step-deg", type=float, default=1.0)
    parser.add_argument("--no-opposite-branch", action="store_true")
    parser.add_argument(
        "--correspondence-schedule-m", type=float, nargs="+",
        default=(0.030, 0.015, 0.008),
    )
    parser.add_argument("--iterations-per-stage", type=int, default=15)
    parser.add_argument("--trim-fraction", type=float, default=0.78)
    parser.add_argument("--min-correspondences", type=int, default=150)
    parser.add_argument("--inlier-distance-m", type=float, default=0.008)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.55)
    parser.add_argument("--max-p50-m", type=float, default=0.0035)
    parser.add_argument("--max-p90-m", type=float, default=0.0075)
    parser.add_argument("--max-p95-m", type=float, default=0.0090)
    parser.add_argument("--max-trimmed-rmse-m", type=float, default=0.0050)
    parser.add_argument("--ambiguity-score-ratio", type=float, default=0.985)
    parser.add_argument("--max-batch-translation-m", type=float, default=0.003)
    parser.add_argument("--max-batch-rotation-deg", type=float, default=1.5)
    parser.add_argument("--max-robot-pose-drift-m", type=float, default=0.0003)
    parser.add_argument("--max-robot-pose-drift-deg", type=float, default=0.10)
    parser.add_argument("--max-robot-joint-drift-rad", type=float, default=0.001)
    parser.add_argument("--max-mount-axis-residual-deg", type=float, default=1.0)
    parser.add_argument("--max-seating-lateral-offset-m", type=float, default=0.050)
    parser.add_argument("--min-seating-axial-offset-m", type=float, default=-0.020)
    parser.add_argument("--max-seating-axial-offset-m", type=float, default=0.150)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.capture_batches < 2 or args.frames_per_batch < 2:
        raise ValueError("capture requires at least two batches and two frames per batch")
    if args.batch_timeout_s <= 0.0 or args.frame_timeout_ms <= 0:
        raise ValueError("capture timeouts must be positive")
    if args.min_points_per_frame < 100 or args.min_batch_points < 100:
        raise ValueError("point-count gates must be >= 100")
    if args.max_points_per_batch < args.min_batch_points:
        raise ValueError("max points per batch must be >= min batch points")
    if args.model_points < 2000:
        raise ValueError("--model-points must be >= 2000")
    if args.yaw_search_deg < 0.0 or args.yaw_step_deg <= 0.0:
        raise ValueError("yaw search must be nonnegative and step must be positive")
    if not 0.0 < args.registration_voxel_m <= 0.01:
        raise ValueError("registration voxel must be in (0, 0.01] metres")
    if args.preview_warmup_frames < 0:
        raise ValueError("preview warmup frames must be >= 0")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        "[safety] READ-ONLY STATIC REGISTRATION: Franka uses read_once only; "
        "no controller is created; Inspire transport is not imported or opened"
    )
    try:
        _validate_args(args)
        control, control_path = load_control_config(args.config)
        if args.save_preview is not None:
            _save_preview(args, control)
            return 0
        if args.output_prefix is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = ROOT / "runs" / f"fr3_rh56_static_registration_{stamp}"
        else:
            prefix = args.output_prefix.expanduser().resolve()
        json_path = Path(str(prefix) + ".json")
        npz_path = Path(str(prefix) + ".npz")
        for output in (json_path, npz_path):
            if output.exists() and not args.overwrite:
                raise FileExistsError(f"output exists (pass --overwrite): {output}")

        import pylibfranka

        robot = pylibfranka.Robot(
            str(control["franka"]["ip"]), pylibfranka.RealtimeConfig.kIgnore
        )
        capture = _capture_batches(args, control, robot)
        del robot
        fit = _fit(args, control, capture)
        arrays = {
            "artifact_type": np.asarray("fr3_rh56_static_hand_mount_registration_points"),
            "schema_version": np.asarray(SCHEMA_VERSION),
            "reference_frame": np.asarray("robot_base"),
            "observed_points": fit["merged_points"].astype(np.float32),
            "observed_colors": fit["merged_colors"].astype(np.float32),
            "model_points_hand_source": fit["model_points"].astype(np.float32),
            "fitted_model_points": fit["fitted_model_points"].astype(np.float32),
            "scene_points": capture["scene_points"].astype(np.float32),
            "scene_colors": capture["scene_colors"].astype(np.float32),
            "T_robot_base_camera": capture["T_base_camera"].astype(np.float64),
            "T_robot_base_EE": capture["before"]["O_T_EE"].astype(np.float64),
            "T_robot_base_hand_source": fit["merged_result"].T_reference_hand.astype(np.float64),
            "T_EE_hand": fit["T_EE_hand"].astype(np.float64),
            "frame_ids": capture["frame_ids"],
            "frame_timestamps_s": capture["frame_timestamps"],
            "roi_bbox_xyxy": np.asarray(capture["roi_bbox"], dtype=np.int32),
            "calibration_id": np.asarray(capture["calibration_id"]),
            "camera_serial": np.asarray(capture["camera_serial"]),
            "geometry_gate_passed": np.asarray(fit["geometry_gate_passed"]),
        }
        _atomic_npz(npz_path, arrays, args.overwrite)
        artifact = _artifact(args, control_path, capture, fit, npz_path)
        _atomic_text(
            json_path,
            json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            args.overwrite,
        )
        status = "PASS-CANDIDATE" if fit["geometry_gate_passed"] else "REJECTED"
        print(f"[registration] {status}")
        print("[result] T_EE_hand=")
        print(np.array2string(fit["T_EE_hand"], precision=8, suppress_small=True))
        print(
            "[result] yaw={:.3f}deg seating_to_source={}m residual={:.3f}deg".format(
                fit["decomposition"]["assembled_yaw_deg"],
                np.array2string(
                    fit["decomposition"]["seating_to_hand_source_origin_m"],
                    precision=6,
                ),
                fit["decomposition"]["mount_axis_residual_deg"],
            )
        )
        metrics = fit["merged_result"].metrics
        print(
            "[metrics] inlier={:.3f} p50={:.2f}mm p90={:.2f}mm "
            "p95={:.2f}mm batch_delta={:.2f}mm/{:.2f}deg".format(
                metrics.observed_inlier_ratio,
                metrics.observed_p50_m * 1000.0,
                metrics.observed_p90_m * 1000.0,
                metrics.observed_p95_m * 1000.0,
                fit["max_batch_translation_m"] * 1000.0,
                fit["max_batch_rotation_deg"],
            )
        )
        for reason in fit["rejection_reasons"]:
            print(f"[reject] {reason}")
        print(f"[saved] {json_path}")
        print(f"[saved] {npz_path}")
        print(
            "[commissioning] config was not modified; collision/path flags remain locked; "
            "inspect the overlay before accepting T_EE_hand"
        )
        if not args.no_viewer:
            _show(capture, fit)
        return 0 if fit["geometry_gate_passed"] else 3
    except KeyboardInterrupt:
        print("[cancelled] no motion command or RH56 write was created", file=sys.stderr)
        return 130
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"[fatal] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
