#!/usr/bin/env python3
"""Offline V94 point-cloud/filter/action-sensitivity A/B.

This program consumes a camera-only raw capture plus a read-only live-preview
audit for the fixed robot/proprioception context.  It never imports a hardware
reader.  Candidate masks replace only the newest point-cloud frame in the same
recorded policy history, which isolates perception sensitivity without claiming
that any diagnostic filter is already part of the deployment contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    workspace = Path(__file__).resolve().parents[2]
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    from sim2real.deployment.bundle import DeployBundle, load_checkpoint_safely
    from sim2real.policy import RollingStudentPolicy
    from sim2real.contracts.v94 import V94Contract
    from sim2real.observation.model import MaskedRGBDProjector
    from sim2real.observation.pointcloud_filters import FILTER_NAMES, build_mask_candidates
    from sim2real.deployment.verify import verify_v94_bundle
else:
    from sim2real.deployment.bundle import DeployBundle, load_checkpoint_safely
    from sim2real.policy import RollingStudentPolicy
    from sim2real.contracts.v94 import V94Contract
    from sim2real.observation.model import MaskedRGBDProjector
    from sim2real.observation.pointcloud_filters import FILTER_NAMES, build_mask_candidates
    from sim2real.deployment.verify import verify_v94_bundle


from sim2real.workspace_paths import DEFAULT_V94_DEPLOY_BUNDLE

DEFAULT_BUNDLE = DEFAULT_V94_DEPLOY_BUNDLE


def _load_npz(path: Path, label: str) -> Dict[str, np.ndarray]:
    resolved = path.expanduser().resolve()
    try:
        with np.load(resolved, allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in archive.files}
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load safe {label} NPZ {resolved}: {exc}") from exc


def _required(data: Dict[str, np.ndarray], name: str) -> np.ndarray:
    if name not in data:
        raise ValueError(f"input archive is missing {name}")
    return np.asarray(data[name])


def _scalar_bool(data: Dict[str, np.ndarray], name: str) -> bool:
    value = _required(data, name)
    if value.shape != () or value.dtype != np.bool_:
        raise ValueError(f"{name} must be a scalar bool")
    return bool(value)


def _atomic_save(path: Path, payload: Dict[str, np.ndarray]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=destination.stem + ".",
        suffix=".npz",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **payload)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _point_metrics(points: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, ...]:
    selected = np.asarray(points, dtype=np.float32)[
        np.asarray(valid, dtype=np.float32) > 0.5
    ]
    if selected.shape[0] == 0:
        nan3 = np.full(3, np.nan, dtype=np.float64)
        return nan3, nan3.copy(), np.asarray(np.nan), np.asarray(np.nan)
    xyz = selected[:, :3].astype(np.float64)
    rgb = selected[:, 3:].astype(np.float64)
    centroid = np.mean(xyz, axis=0)
    extent = np.max(xyz, axis=0) - np.min(xyz, axis=0)
    dark_fraction = float(np.mean(np.max(rgb, axis=1) < 0.10))
    radial_p90 = float(
        np.percentile(np.linalg.norm(xyz - centroid[None], axis=1), 90.0)
    )
    return centroid, extent, np.asarray(dark_fraction), np.asarray(radial_p90)


def analyze_pointcloud_ab(
    *,
    bundle_path: Path,
    capture_path: Path,
    state_audit_path: Path,
    state_index: int = -1,
    sphere_radius_m: Optional[float] = 0.030,
    sphere_shell_tolerance_m: float = 0.006,
    support_plane_clearance_m: float = 0.006,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Run the pure offline A/B and return NPZ payload plus JSON summary."""

    verification = verify_v94_bundle(bundle_path)
    bundle = DeployBundle(bundle_path)
    contract = V94Contract.from_bundle(bundle)
    scene_manifest = bundle.read_json("calibration/scene_manifest.json")
    try:
        support_plane_abcd = np.asarray(
            scene_manifest["scene"]["tabletop"]["plane_in_robot_base"],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("deployment bundle has no valid measured tabletop plane") from exc
    if support_plane_abcd.shape != (4,) or not np.all(np.isfinite(support_plane_abcd)):
        raise ValueError("deployment tabletop plane must contain four finite values")
    policy = RollingStudentPolicy(load_checkpoint_safely(bundle.checkpoint_bytes()))
    capture = _load_npz(capture_path, "raw capture")
    state = _load_npz(state_audit_path, "read-only state audit")

    if _scalar_bool(capture, "robot_command_writes"):
        raise ValueError("raw capture unexpectedly records robot command writes")
    if _scalar_bool(capture, "franka_interface_opened"):
        raise ValueError("raw capture unexpectedly opened Franka")
    if _scalar_bool(capture, "rh56_interface_opened"):
        raise ValueError("raw capture unexpectedly opened RH56")
    if _scalar_bool(state, "robot_command_writes"):
        raise ValueError("state audit contains robot command writes")
    capture_hash = str(_required(capture, "checkpoint_sha256").item())
    state_hash = str(_required(state, "checkpoint_sha256").item())
    if capture_hash != verification.checkpoint_sha256 or state_hash != capture_hash:
        raise ValueError("bundle/capture/state checkpoint hashes disagree")
    capture_calibration = str(_required(capture, "calibration_id").item())
    state_calibration = str(_required(state, "calibration_id").item())
    if (
        capture_calibration != contract.calibration_id
        or state_calibration != contract.calibration_id
    ):
        raise ValueError("bundle/capture/state calibration IDs disagree")

    rgb = _required(capture, "rgb")
    depth_raw = _required(capture, "depth_raw")
    raw_mask = _required(capture, "object_mask")
    frame_ids = _required(capture, "camera_frame_id")
    camera_time = _required(capture, "camera_timestamp_s")
    frame_count = int(rgb.shape[0])
    expected_image = (contract.camera_height, contract.camera_width)
    if rgb.shape != (frame_count, *expected_image, 3) or rgb.dtype != np.uint8:
        raise ValueError("capture rgb has the wrong shape/dtype")
    if depth_raw.shape != (frame_count, *expected_image) or depth_raw.dtype != np.uint16:
        raise ValueError("capture depth_raw has the wrong shape/dtype")
    if raw_mask.shape != (frame_count, *expected_image) or raw_mask.dtype != np.bool_:
        raise ValueError("capture object_mask has the wrong shape/dtype")
    if frame_ids.shape != (frame_count,) or camera_time.shape != (frame_count,):
        raise ValueError("capture frame IDs/timestamps have the wrong shape")
    if frame_count <= 0 or np.any(np.diff(frame_ids.astype(np.int64)) <= 0):
        raise ValueError("capture frame IDs must be nonempty and strictly increasing")
    if np.any(np.diff(camera_time.astype(np.float64)) <= 0.0):
        raise ValueError("capture timestamps must increase strictly")
    capture_K = _required(capture, "camera_K").astype(np.float64)
    capture_T = _required(capture, "T_base_camera_optical").astype(np.float64)
    capture_scale = float(_required(capture, "depth_scale_m_per_unit").item())
    if not np.allclose(capture_K, contract.camera_K, atol=1.0e-5, rtol=0.0):
        raise ValueError("capture camera intrinsics differ from contract")
    if not np.allclose(capture_T, contract.T_base_camera_optical, atol=1.0e-7):
        raise ValueError("capture camera transform differs from contract")
    if not np.isclose(capture_scale, contract.depth_scale_m_per_unit, atol=1.0e-9):
        raise ValueError("capture depth scale differs from contract")

    reference_points_all = _required(state, "pointcloud_history_xyzrgb_palm")
    reference_valid_all = _required(state, "pointcloud_valid_history")
    reference_proprio_all = _required(state, "proprio_history67")
    reference_actions_all = _required(state, "raw_policy_action13")
    palm_all = _required(state, "T_base_palm_at_pointcloud_capture")
    state_count = int(reference_points_all.shape[0])
    selected_index = int(state_index)
    if selected_index < 0:
        selected_index += state_count
    if selected_index < 0 or selected_index >= state_count:
        raise IndexError(f"state_index {state_index} is outside {state_count} records")
    reference_points = np.asarray(reference_points_all[selected_index], dtype=np.float32)
    reference_valid = np.asarray(reference_valid_all[selected_index], dtype=np.float32)
    reference_proprio = np.asarray(reference_proprio_all[selected_index], dtype=np.float32)
    reference_action_recorded = np.asarray(
        reference_actions_all[selected_index], dtype=np.float32
    )
    T_base_palm = np.asarray(palm_all[selected_index], dtype=np.float64)
    if reference_points.shape != (4, 128, 6):
        raise ValueError("state pointcloud history has the wrong shape")
    if reference_valid.shape != (4, 128) or reference_proprio.shape != (4, 67):
        raise ValueError("state valid/proprio history has the wrong shape")
    if T_base_palm.shape != (4, 4) or not np.all(np.isfinite(T_base_palm)):
        raise ValueError("state capture-time palm pose is invalid")
    reference_action_recomputed = policy.act(
        reference_points, reference_valid, reference_proprio
    ).action13
    replay_error = float(
        np.max(np.abs(reference_action_recomputed - reference_action_recorded))
    )
    if replay_error > 2.0e-5:
        raise RuntimeError(
            f"state audit policy replay mismatch before A/B: max_abs={replay_error:.9g}"
        )

    candidate_count = len(FILTER_NAMES)
    mask_point_count = np.zeros((frame_count, candidate_count), dtype=np.int32)
    projected_source_count = np.zeros((frame_count, candidate_count), dtype=np.int32)
    projected_points = np.zeros(
        (frame_count, candidate_count, 128, 6), dtype=np.float32
    )
    projected_valid = np.zeros((frame_count, candidate_count, 128), dtype=np.float32)
    point_status = np.full((frame_count, candidate_count), "not_run", dtype="<U32")
    point_centroid = np.full((frame_count, candidate_count, 3), np.nan, dtype=np.float64)
    point_extent = np.full((frame_count, candidate_count, 3), np.nan, dtype=np.float64)
    point_dark_fraction = np.full((frame_count, candidate_count), np.nan, dtype=np.float64)
    point_radial_p90 = np.full((frame_count, candidate_count), np.nan, dtype=np.float64)
    actions = np.full((frame_count, candidate_count, 13), np.nan, dtype=np.float32)
    robust_depth_center = np.full(frame_count, np.nan, dtype=np.float64)
    robust_depth_half_width = np.full(frame_count, np.nan, dtype=np.float64)
    sphere_center = np.full((frame_count, 3), np.nan, dtype=np.float64)
    sphere_residual_median = np.full(frame_count, np.nan, dtype=np.float64)
    sphere_residual_p90 = np.full(frame_count, np.nan, dtype=np.float64)
    sphere_inlier_fraction = np.zeros(frame_count, dtype=np.float64)
    sphere_fit_valid = np.zeros(frame_count, dtype=bool)

    for frame_index in range(frame_count):
        depth_m = depth_raw[frame_index].astype(np.float32) * np.float32(capture_scale)
        variants = build_mask_candidates(
            depth_m,
            raw_mask[frame_index],
            camera_K=capture_K,
            T_base_camera_optical=capture_T,
            sphere_radius_m=sphere_radius_m,
            sphere_shell_tolerance_m=sphere_shell_tolerance_m,
            support_plane_abcd=support_plane_abcd,
            support_plane_clearance_m=support_plane_clearance_m,
        )
        robust_depth_center[frame_index] = variants.robust_depth_center_m
        robust_depth_half_width[frame_index] = variants.robust_depth_half_width_m
        sphere_center[frame_index] = variants.sphere_fit.center_base_m
        sphere_residual_median[frame_index] = variants.sphere_fit.residual_median_m
        sphere_residual_p90[frame_index] = variants.sphere_fit.residual_p90_m
        sphere_inlier_fraction[frame_index] = variants.sphere_fit.shell_inlier_fraction
        sphere_fit_valid[frame_index] = variants.sphere_fit.valid
        for candidate_index, name in enumerate(FILTER_NAMES):
            candidate_mask = np.asarray(variants.masks[name], dtype=bool)
            mask_point_count[frame_index, candidate_index] = int(
                np.count_nonzero(candidate_mask)
            )
            # A fresh projector per candidate prevents its latest-good fallback
            # from hiding a candidate with too few points.
            projector = MaskedRGBDProjector(
                camera_K=capture_K,
                T_base_camera_optical=capture_T,
                image_size=(contract.camera_width, contract.camera_height),
                depth_range_m=contract.depth_range_m,
            )
            point_frame = projector.project(
                color_bgr=rgb[frame_index],
                depth_m=depth_m,
                object_mask=candidate_mask,
                T_base_palm_at_capture=T_base_palm,
                captured_at_s=float(camera_time[frame_index]),
                frame_id=int(frame_ids[frame_index]),
            )
            projected_source_count[frame_index, candidate_index] = (
                point_frame.source_valid_points
            )
            projected_points[frame_index, candidate_index] = point_frame.xyzrgb_palm
            projected_valid[frame_index, candidate_index] = point_frame.valid
            point_status[frame_index, candidate_index] = point_frame.status
            centroid, extent, dark, radial = _point_metrics(
                point_frame.xyzrgb_palm, point_frame.valid
            )
            point_centroid[frame_index, candidate_index] = centroid
            point_extent[frame_index, candidate_index] = extent
            point_dark_fraction[frame_index, candidate_index] = dark
            point_radial_p90[frame_index, candidate_index] = radial
            if float(np.sum(point_frame.valid)) < 1.0:
                continue
            history_points = reference_points.copy()
            history_valid = reference_valid.copy()
            history_points[-1] = point_frame.xyzrgb_palm
            history_valid[-1] = point_frame.valid
            actions[frame_index, candidate_index] = policy.act(
                history_points, history_valid, reference_proprio
            ).action13

    raw_index = FILTER_NAMES.index("raw")
    raw_actions = actions[:, raw_index]
    action_delta_vs_raw = actions - raw_actions[:, None, :]
    action_linf_vs_raw = np.max(np.abs(action_delta_vs_raw), axis=2)
    action_linf_vs_reference = np.max(
        np.abs(actions - reference_action_recomputed[None, None, :]), axis=2
    )
    raw_mask_count = np.maximum(mask_point_count[:, raw_index], 1)
    retention = mask_point_count / raw_mask_count[:, None]

    filter_summaries = []
    for index, name in enumerate(FILTER_NAMES):
        finite_action = np.all(np.isfinite(actions[:, index]), axis=1)
        delta = action_linf_vs_raw[finite_action, index]
        candidate_actions = actions[finite_action, index]
        if candidate_actions.shape[0] >= 2:
            step_linf = np.max(np.abs(np.diff(candidate_actions, axis=0)), axis=1)
            axis_std = np.std(candidate_actions, axis=0)
        else:
            step_linf = np.empty(0, dtype=np.float32)
            axis_std = np.empty(0, dtype=np.float32)
        finite_centroid = point_centroid[
            np.all(np.isfinite(point_centroid[:, index]), axis=1), index
        ]
        centroid_std_mm = (
            np.std(finite_centroid, axis=0) * 1000.0
            if finite_centroid.shape[0]
            else np.full(3, np.nan)
        )
        filter_summaries.append(
            {
                "filter": name,
                "valid_action_frames": int(np.count_nonzero(finite_action)),
                "mask_retention_median": float(np.median(retention[:, index])),
                "source_points_median": float(
                    np.median(projected_source_count[:, index])
                ),
                "action_linf_vs_raw_median": (
                    float(np.median(delta)) if delta.size else None
                ),
                "action_linf_vs_raw_p95": (
                    float(np.percentile(delta, 95.0)) if delta.size else None
                ),
                "action_linf_vs_raw_max": float(np.max(delta)) if delta.size else None,
                "temporal_action_step_linf_median": (
                    float(np.median(step_linf)) if step_linf.size else None
                ),
                "temporal_action_step_linf_p95": (
                    float(np.percentile(step_linf, 95.0)) if step_linf.size else None
                ),
                "temporal_action_step_linf_max": (
                    float(np.max(step_linf)) if step_linf.size else None
                ),
                "temporal_action_axis_std_mean": (
                    float(np.mean(axis_std)) if axis_std.size else None
                ),
                "temporal_action_axis_std_max": (
                    float(np.max(axis_std)) if axis_std.size else None
                ),
                "point_centroid_std_xyz_mm": (
                    centroid_std_mm.tolist()
                    if np.all(np.isfinite(centroid_std_mm))
                    else None
                ),
                "dark_point_fraction_median": (
                    float(np.nanmedian(point_dark_fraction[:, index]))
                    if np.any(np.isfinite(point_dark_fraction[:, index]))
                    else None
                ),
                "extent_norm_median_m": (
                    float(np.nanmedian(np.linalg.norm(point_extent[:, index], axis=1)))
                    if np.any(np.isfinite(point_extent[:, index]))
                    else None
                ),
            }
        )
    summary: Dict[str, object] = {
        "analysis_mode": "offline_replace_latest_point_frame_fixed_recorded_proprio",
        "capture_frames": frame_count,
        "state_index": selected_index,
        "reference_policy_replay_max_abs": replay_error,
        "sphere_radius_m": sphere_radius_m,
        "sphere_shell_tolerance_m": float(sphere_shell_tolerance_m),
        "support_plane_abcd": support_plane_abcd.tolist(),
        "support_plane_clearance_m": float(support_plane_clearance_m),
        "sphere_fit_valid_frames": int(np.count_nonzero(sphere_fit_valid)),
        "filters": filter_summaries,
        "live_policy_semantics_changed": False,
        "hardware_interfaces_opened": False,
        "robot_command_writes": False,
    }
    payload: Dict[str, np.ndarray] = {
        "analysis_schema_version": np.asarray(1, dtype=np.int32),
        "filter_names": np.asarray(FILTER_NAMES),
        "camera_frame_id": frame_ids.astype(np.int64),
        "camera_timestamp_s": camera_time.astype(np.float64),
        "candidate_mask_point_count": mask_point_count,
        "candidate_mask_retention": retention.astype(np.float64),
        "candidate_projected_source_count": projected_source_count,
        "candidate_point_xyzrgb_palm": projected_points,
        "candidate_point_valid": projected_valid,
        "candidate_point_status": point_status,
        "candidate_point_centroid_palm_m": point_centroid,
        "candidate_point_extent_palm_m": point_extent,
        "candidate_point_dark_fraction": point_dark_fraction,
        "candidate_point_radial_p90_m": point_radial_p90,
        "candidate_raw_policy_action13": actions,
        "candidate_action_delta_vs_raw13": action_delta_vs_raw,
        "candidate_action_linf_vs_raw": action_linf_vs_raw,
        "candidate_action_linf_vs_reference": action_linf_vs_reference,
        "robust_depth_center_m": robust_depth_center,
        "robust_depth_half_width_m": robust_depth_half_width,
        "sphere_fit_center_base_m": sphere_center,
        "sphere_fit_residual_median_m": sphere_residual_median,
        "sphere_fit_residual_p90_m": sphere_residual_p90,
        "sphere_fit_shell_inlier_fraction": sphere_inlier_fraction,
        "sphere_fit_valid": sphere_fit_valid,
        "support_plane_abcd": support_plane_abcd,
        "support_plane_clearance_m": np.asarray(
            support_plane_clearance_m, dtype=np.float64
        ),
        "reference_state_index": np.asarray(selected_index, dtype=np.int64),
        "reference_T_base_palm": T_base_palm,
        "reference_recorded_action13": reference_action_recorded,
        "reference_recomputed_action13": reference_action_recomputed,
        "reference_policy_replay_max_abs": np.asarray(replay_error),
        "checkpoint_sha256": np.asarray(verification.checkpoint_sha256),
        "calibration_id": np.asarray(contract.calibration_id),
        "analysis_mode": np.asarray(
            "offline_replace_latest_point_frame_fixed_recorded_proprio"
        ),
        "live_policy_semantics_changed": np.asarray(False),
        "hardware_interfaces_opened": np.asarray(False),
        "robot_command_writes": np.asarray(False),
        "summary_json": np.asarray(json.dumps(summary, sort_keys=True)),
    }
    return payload, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline-only raw/erosion/robust-depth/known-radius-sphere point-cloud "
            "and V94 action-sensitivity A/B."
        )
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--state-audit", type=Path, required=True)
    parser.add_argument("--state-index", type=int, default=-1)
    parser.add_argument(
        "--sphere-radius-m",
        type=float,
        default=0.030,
        help="known current test-object radius; use 0 to disable sphere candidate",
    )
    parser.add_argument("--sphere-shell-tolerance-m", type=float, default=0.006)
    parser.add_argument("--support-plane-clearance-m", type=float, default=0.006)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="reserved safety tripwire; this offline program always refuses it",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.execute:
        raise RuntimeError(
            "analyze_v94_pointcloud_ab is offline-only and cannot execute actions"
        )
    radius = float(args.sphere_radius_m)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("--sphere-radius-m must be finite and non-negative")
    tolerance = float(args.sphere_shell_tolerance_m)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("--sphere-shell-tolerance-m must be finite and positive")
    clearance = float(args.support_plane_clearance_m)
    if not np.isfinite(clearance) or clearance < 0.0:
        raise ValueError("--support-plane-clearance-m must be finite and non-negative")
    payload, summary = analyze_pointcloud_ab(
        bundle_path=args.bundle,
        capture_path=args.capture,
        state_audit_path=args.state_audit,
        state_index=int(args.state_index),
        sphere_radius_m=None if radius == 0.0 else radius,
        sphere_shell_tolerance_m=tolerance,
        support_plane_clearance_m=clearance,
    )
    _atomic_save(args.output, payload)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[offline A/B] saved={args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
