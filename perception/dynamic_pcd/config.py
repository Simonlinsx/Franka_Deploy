from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "camera": {
        "serial": None,
        "width": 848,
        "height": 480,
        "fps": 30,
        "z_min": 0.25,
        "z_max": 1.20,
        "emitter": True,
        "laser_power": 180,
        "preset": "default",
        "spatial_filter": False,
        "temporal_filter": False,
        "hole_filter": False,
    },
    "tracker": {
        "mode": "roi_depth",
        "interactive_roi_padding_px": 0,
        "roi_scale": 1.5,
        "lock_roi": False,
        "depth_tolerance": 0.04,
        "min_area": 80,
        "morph_kernel": 5,
        "max_center_jump": 0.20,
        "max_area_growth": 1.8,
        "max_area_vs_initial": 3.0,
        "max_bbox_growth": 4.0,
        "min_area_vs_initial": 0.25,
        "min_bbox_area_vs_initial": 0.30,
        "recovery_enabled": True,
        "recovery_search_scale": 5.0,
        "recovery_min_score": 0.35,
        "sam2_lost_search_scale": 3.0,
        "identity_enabled": True,
        "identity_max_hist_dist": 0.55,
        "identity_max_center_jump_px": 220.0,
        "identity_max_depth_jump": 0.25,
        "identity_max_area_ratio": 4.0,
        "identity_update_alpha": 0.0,
        "lost_after": 8,
        # adaptive_color_depth fast path (learned from any external prompt mask)
        "color_probability_threshold": 0.56,
        "component_min_confidence": 0.62,
        # Semantic pixels are owned by SAM2/appearance tracking.  D435 depth
        # validates a whole candidate and policy points, but never erodes the
        # semantic silhouette pixel-by-pixel.
        "semantic_mask_depth_pixel_gate_enabled": False,
        "component_min_valid_depth_ratio": 0.25,
        "component_max_center_jump_px": 100.0,
        "component_max_motion_px": 24.0,
        "component_motion_bbox_diagonal_ratio": 0.80,
        "component_max_motion_cap_px": 100.0,
        "component_flow_motion_min_confidence": 0.45,
        "motion_probation_appearance_freeze_frames": 3,
        "component_min_score_margin": 0.04,
        "min_area_ratio": 0.35,
        "state_update_alpha": 0.65,
        "lost_roi_growth": 0.25,
        "max_roi_scale": 4.0,
        "depth_spread_margin": 0.008,
        "max_depth_tolerance": 0.08,
        "background_ring_scale": 1.8,
        "background_exclusion_dilation": 7,
        "background_likelihood_weight": 1.0,
        # Learn foreground colour from a deep seed core.  Background
        # exclusion still uses the complete validated seed, so the removed
        # target rim cannot become negative appearance evidence.
        "appearance_seed_erode": 11,
        "box_prompt_seed_scale": 0.65,
        "hue_bins": 24,
        "saturation_bins": 16,
        "value_bins": 16,
        "histogram_smoothing": 1.0e-6,
        "appearance_anchor_weight": 0.70,
        "appearance_update_alpha": 0.025,
        "appearance_update_min_confidence": 0.82,
        "appearance_update_erode_kernel": 3,
        "appearance_update_min_core_pixels": 40,
        "appearance_update_min_core_fraction": 0.35,
        "appearance_update_min_flow_confidence": 0.35,
        "reinit_min_color_score": 0.62,
        "reinit_max_center_jump_px": 90.0,
        "reinit_max_depth_jump_m": 0.12,
        # A text prompt describes a category rather than a guaranteed physical
        # instance.  Semantic re-acquisition may therefore accept a large raw
        # camera-depth relocation, while online SAM2 fusion keeps the strict
        # reinit_max_depth_jump_m gate above.
        "prompt_reinit_max_depth_jump_m": 0.0,
        "lost_recovery_requires_flow": True,
        "occlusion_recovery_enabled": True,
        "occlusion_recovery_flow_grace_frames": 2,
        "occlusion_global_search_enabled": True,
        "occlusion_global_search_after_frames": 2,
        "occlusion_global_color_probability_threshold": 0.56,
        "occlusion_global_depth_tolerance_m": 0.10,
        "occlusion_global_max_depth_jump_m": 0.10,
        "occlusion_global_min_confidence": 0.68,
        "occlusion_global_min_score_margin": 0.08,
        "occlusion_global_min_area_ratio": 0.55,
        "occlusion_global_max_area_ratio": 2.50,
        "occlusion_global_min_bbox_area_ratio": 0.50,
        "occlusion_global_max_bbox_area_ratio": 2.50,
        "occlusion_fragment_merge_enabled": True,
        "occlusion_fragment_max_gap_px": 12,
        "occlusion_fragment_max_depth_spread_m": 0.08,
        "occlusion_fragment_depth_tolerance_m": 0.05,
        "occlusion_recovery_confirm_frames": 2,
        "occlusion_recovery_min_valid_depth_ratio": 0.25,
        "occlusion_recovery_confirm_max_frame_gap": 3,
        "occlusion_recovery_confirm_center_step_px": 12.0,
        "occlusion_recovery_confirm_bbox_iou": 0.35,
        "occlusion_recovery_confirm_area_ratio": 1.60,
        "occlusion_recovery_confirm_depth_step_m": 0.030,
        # Final source-agnostic quarantine before a recovered mask/PCD becomes
        # robot-facing. Internal tracker/SAM confirmation is necessary but not
        # sufficient: the committed state must survive these exact RGB-D
        # frames as well.
        "recovery_publish_confirm_frames": 3,
        "recovery_publish_confirm_max_frame_gap": 1,
        "recovery_publish_confirm_min_bbox_iou": 0.35,
        "recovery_publish_confirm_max_area_ratio": 1.35,
        "recovery_publish_confirm_max_depth_step_m": 0.04,
        "recovery_publish_confirm_max_center_step_m": 0.06,
        "recovery_publish_confirm_max_center_step_px": 24.0,
        "recovery_publish_confirm_max_timestamp_step_s": 0.20,
        "recovery_publish_min_valid_depth_ratio": 0.35,
        "recovery_publish_max_depth_spread_m": 0.18,
        # A guarded-v2 exact mask may publish a visible partial silhouette,
        # but it must retain this fraction of the motion-aligned previous
        # clean/full-authority support before it may teach full geometry.
        "publication_full_target_min_aligned_coverage": 0.90,
        # A two-frame, raw-exact perspective-scale transition has stricter
        # identity/topology/rate proofs and keeps the unchanged full mask as a
        # second anchor.  Its scale-normalized shape floor is therefore
        # separate from the ordinary partial-authority retention threshold.
        "publication_full_scale_transition_min_normalized_coverage": 0.85,
        # Full/trusted geometry may learn only from a mask whose immutable
        # target support is substantially stronger than ordinary visible-core
        # publication. Lower-support exact masks remain policy-visible partials.
        "publication_full_target_min_appearance_support_ratio": 0.90,
        # Match the deployed D435 effective-border entry policy.  This is
        # also the majority-support floor for a sanitizer-only late handoff;
        # it never authorizes raw off-boundary publication.
        "publication_boundary_exit_min_appearance_support_ratio": 0.60,
        # A confirmed edge exit may retain the reviewed motion-blurred final
        # sliver (about 0.58 support), but never on mean appearance alone.
        "publication_boundary_exit_continuation_min_appearance_support_ratio": 0.55,
        "optical_flow_enabled": True,
        "flow_max_corners": 80,
        "flow_min_points": 4,
        "flow_quality_level": 0.01,
        "flow_min_distance": 4.0,
        "flow_window_size": 21,
        "flow_pyramid_levels": 3,
        "flow_fb_max_error": 1.5,
        "flow_min_inlier_ratio": 0.55,
        "flow_ransac_reprojection_px": 2.5,
        "flow_min_scale": 0.75,
        "flow_max_scale": 1.35,
        "flow_max_rotation_deg": 35.0,
        "flow_max_displacement_px": 100.0,
        # Motion-compensated mask hysteresis. Weak evidence may retain only a
        # warped previous pixel; entering pixels need the stricter threshold.
        "flow_mask_dilation": 1,
        "flow_color_probability_floor": 0.42,
        "flow_enter_color_probability_threshold": 0.64,
        "flow_min_mask_overlap": 0.15,
        # Match the exact-current SAM2 consensus envelope.  A semantic contour
        # can legitimately correct a reflective edge by more than 20% while
        # the stricter temporal smoother below simply bypasses that frame.
        # The independent component/identity/initial-size gates still reject
        # a hand/background union.
        "tracked_motion_min_area_ratio": 0.60,
        "tracked_motion_max_area_ratio": 1.60,
        "tracked_motion_min_mask_iou": 0.50,
        "tracked_motion_min_flow_confidence": 0.45,
        # Motion-compensated two-frame silhouette confirmation.  The previous
        # soft score is affine-warped before blending, so real object motion is
        # not low-pass filtered in image coordinates.
        "temporal_mask_stability_enabled": True,
        "temporal_mask_min_flow_confidence": 0.45,
        "temporal_mask_geometry_fallback_enabled": True,
        # Zero derives the geometry-fallback range from the scale-aware
        # component motion gate above.
        "temporal_mask_geometry_max_displacement_px": 0.0,
        "temporal_mask_geometry_min_scale": 0.85,
        "temporal_mask_geometry_max_scale": 1.18,
        "temporal_mask_geometry_max_anisotropy": 1.20,
        "temporal_mask_min_raw_iou": 0.65,
        "temporal_mask_min_raw_area_ratio": 0.85,
        "temporal_mask_max_raw_area_ratio": 1.15,
        "temporal_mask_current_weight": 0.45,
        "temporal_mask_threshold": 0.50,
        "temporal_mask_max_hold_pixels": 48,
        "temporal_mask_max_hold_fraction": 0.08,
        "temporal_mask_retention_depth_tolerance_m": 0.045,
        "temporal_mask_retention_depth_gate_enabled": False,
    },
    "pointcloud": {
        "num_points": 1024,
        "history_len": 4,
        "use_rgb": False,
        "erode_kernel": 3,
        "stride": 1,
        "voxel_size": 0.003,
        "remove_outliers": True,
        "outlier_nb_neighbors": 20,
        "outlier_std_ratio": 1.5,
        "center_policy_points": True,
        "workspace_min": None,
        "workspace_max": None,
    },
    "extrinsics": {
        "calibration_file": None,
        "require_calibration": False,
        "require_quality_pass": False,
        "strict_camera_serial": True,
        "calibrated": None,
        "base_frame": "robot_base",
        "camera_frame": "camera_color_optical_frame",
        "T_base_camera": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
    "hover": {
        "object_workspace_min": [0.505, 0.008, -0.05],
        "object_workspace_max": [0.650, 0.104, 0.30],
        "eef_workspace_min": [0.480, -0.020, 0.300],
        "eef_workspace_max": [0.660, 0.130, 0.360],
        "target_z_min": 0.320,
        "target_z_max": 0.345,
        "hover_clearance": 0.150,
        "descent_clearance_guard": 0.015,
        "descent_velocity": 0.005,
        "max_descent_xy_error": 0.005,
        "max_top_drift": 0.010,
        "stable_frames": 30,
        "stable_seconds": 1.0,
        "max_packet_age": 0.15,
        "motion_max_packet_age": 0.25,
        "max_center_p95": 0.005,
        "max_center_step": 0.010,
        "max_fitted_speed": 0.010,
        "max_target_drift": 0.020,
        "min_raw_points": 200,
        "max_segment_distance": 0.030,
        "max_velocity": 0.010,
        "min_segment_duration": 3.0,
        "min_joint_margin": 0.05,
        "min_control_success_rate": 0.95,
        "max_final_error": 0.015,
    },
    "sam2": {
        "enabled": False,
        "require_for_bbox_init": False,
        "checkpoint": None,
        "model_cfg": None,
        "device": "cuda",
        "reinit_every": 10,
        "score_threshold": 0.5,
        "multimask_output": False,
        "keep_largest_component": True,
        "min_area": 20,
        "max_mask_area_ratio": 4.0,
        "max_bbox_area_ratio": 6.0,
    },
    "online_sam2": {
        # Persistent official SAM2 video memory.  It runs in a separate CUDA
        # service and its masks are still checked by the adaptive tracker's
        # strict appearance/size/depth reinitialization gate.
        "enabled": True,
        "guarded_v2_semantic_primary": True,
        # The first prompt silhouette is provisional: two adjacent, stable,
        # finally-published raw exact masks may replace its blurred geometry
        # and frozen appearance once, only inside this short startup window.
        "guarded_v2_bootstrap_commission_enabled": True,
        "guarded_v2_bootstrap_max_frame_delta": 12,
        "guarded_v2_bootstrap_max_elapsed_s": 0.40,
        "guarded_v2_bootstrap_candidate_max_frame_gap": 3,
        "guarded_v2_bootstrap_candidate_max_timestamp_gap_s": 0.20,
        # The first two exact masks of a fast target can differ by one pixel
        # on each side (43x43 -> 44x44 is 1.047x area) even when their extent,
        # appearance, depth, provenance and motion are all consistent.  Keep
        # this at a narrow 5% startup-only bound; the independent identity and
        # raw-exact byte gates still reject a hand union or sanitized
        # publication.
        "guarded_v2_bootstrap_stable_max_area_ratio": 1.05,
        "guarded_v2_bootstrap_stable_max_extent_ratio": 1.08,
        "guarded_v2_bootstrap_max_registered_contraction_fraction": 0.05,
        "guarded_v2_bootstrap_min_supported_extent_ratio": 0.90,
        "guarded_v2_bootstrap_max_supported_centroid_offset_ratio": 0.075,
        "guarded_v2_bootstrap_partial_seed_max_supported_centroid_offset_ratio": 0.10,
        "guarded_v2_bootstrap_max_unsupported_bbox_margin_ratio": 0.075,
        "guarded_v2_bootstrap_max_center_speed_px_s": 2400.0,
        "guarded_v2_bootstrap_max_depth_step_m": 0.04,
        "mask_publication_mode": "adaptive_fusion",
        "require_for_bbox_init": False,
        "bbox_init_depth_refine_enabled": False,
        "bbox_init_depth_seed_scale": 0.35,
        "bbox_init_depth_tolerance_m": 0.055,
        "bbox_init_depth_min_seed_pixels": 20,
        "bbox_init_depth_min_retained_ratio": 0.20,
        "bbox_init_depth_morph_kernel": 3,
        "bbox_init_clip_to_prompt": True,
        "bbox_init_clip_margin_px": 2,
        "service_addr": "tcp://127.0.0.1:5558",
        "service_autostart": True,
        # Full official VOS torch.compile prewarm is a startup-only operation
        # and may take minutes on the first cache miss. Runtime RPC deadlines
        # remain independently strict.
        "startup_timeout_s": 300.0,
        "request_timeout_s": 0.50,
        # The hot tiny/512 model is normally ready inside this same-frame
        # window.  An accepted SAM2 silhouette becomes the public mask while
        # the independent adaptive fallback state remains observationally
        # isolated from semantic-mask contamination.
        "tracked_frame_wait_timeout_s": 0.045,
        "tracked_publication_source": "sam2",
        # LOST recovery has no valid local observation.  The deployed
        # SAM2-tiny hot path measures roughly 27--34 ms, so a 30 ms deadline
        # systematically discarded correct recovery masks.  Use the same
        # bounded 45 ms exact-frame window as tracked publication; it remains
        # inside one 50 ms policy period and never pairs a late mask with a
        # newer depth frame.
        "frame_wait_timeout_s": 0.045,
        "semantic_frame_wait_timeout_s": 0.045,
        "semantic_previous_frame_grace_s": 0.015,
        "bbox_init_discarded_track_prewarm": True,
        # Initialization-only temporal warmup.  Formal publications retain
        # the independent 45 ms deadline below; warmup must demonstrate two
        # consecutive calls with 5 ms headroom before exact reseeding.
        "bbox_init_prewarm_max_attempts": 6,
        "bbox_init_prewarm_required_stable_tracks": 2,
        "bbox_init_prewarm_max_rpc_ms": 40.0,
        "initialization_timeout_s": 2.0,
        "checkpoint": "/home/qiaoguanren/code/franka/third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt",
        "model_config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "device": "cuda",
        "image_size": 512,
        "amp_dtype": "bfloat16",
        # Use the official SAM2 VOS-optimized predictor implementation. This
        # remains explicit so evaluation can run a provenance-bound false A/B.
        "vos_optimized": True,
        # Retain official full VOS torch.compile + max-autotune while disabling
        # CUDA Graph storage reuse, which is incompatible with SAM2's persistent
        # positional-encoding tensor cache under PyTorch 2.5.
        "vos_compile_mode": "max-autotune-no-cudagraphs",
        "reset_every_frames": 0,
        "min_mask_area": 20,
        "object_score_threshold": 0.0,
        # While the CPU tracker is valid, same-frame temporal output must still
        # agree with the adaptive mask before it may become robot-facing.
        "tracked_min_area_ratio": 0.60,
        "tracked_max_area_ratio": 1.60,
        "tracked_min_iou": 0.40,
        "tracked_min_adaptive_coverage": 0.65,
        "tracked_min_online_coverage": 0.65,
        "tracked_min_initial_bbox_area_ratio": 0.45,
        "tracked_max_initial_bbox_area_ratio": 1.35,
        "tracked_extra_core_dilation_px": 2,
        "tracked_extra_appearance_min_pixels": 24,
        "tracked_extra_appearance_min_fraction": 0.04,
        "tracked_extra_appearance_probability_threshold": 0.42,
        "tracked_extra_min_appearance_support_ratio": 0.65,
        "tracked_extra_min_appearance_mean": 0.45,
        "trusted_max_timestamp_step_s": 0.20,
        "trusted_max_center_speed_px_s": 2400.0,
        "trusted_max_log_area_rate_s": 12.0,
        "trusted_max_depth_step_m": 0.04,
        "trusted_max_depth_speed_m_s": 1.0,
        "trusted_max_initial_area_ratio": 2.50,
        "trusted_max_initial_bbox_area_ratio": 2.50,
        "trusted_min_valid_depth_ratio": 0.35,
        "trusted_max_depth_spread_m": 0.18,
        "trusted_appearance_probability_threshold": 0.50,
        "trusted_min_appearance_support_ratio": 0.65,
        "trusted_min_appearance_mean": 0.45,
        "trusted_extra_core_dilation_px": 2,
        "trusted_extra_appearance_min_pixels": 24,
        "trusted_extra_appearance_min_fraction": 0.04,
        "trusted_prediction_residual_min_px": 12.0,
        "trusted_prediction_residual_bbox_diagonal_ratio": 0.80,
        "trusted_prediction_residual_max_px": 72.0,
        "trusted_max_acceleration_px_s2": 12000.0,
        "trusted_reinit_min_area_vs_initial": 0.05,
        "trusted_reinit_min_bbox_area_vs_initial": 0.05,
        "trusted_recovery_max_area_ratio_cap": 2.20,
        # LOST recovery is compared to immutable prompt-mask geometry and RGB-D
        # quality, then held provisional until two consistent exact frames.
        "recovery_min_area_ratio": 0.40,
        "recovery_max_area_ratio": 1.80,
        "recovery_min_bbox_area_ratio": 0.35,
        "recovery_max_bbox_area_ratio": 2.20,
        "recovery_min_valid_depth_ratio": 0.35,
        "recovery_max_depth_spread_m": 0.18,
        "recovery_confirm_frames": 2,
        # Missing an exact-frame RPC deadline is not evidence that the object
        # disappeared. Preserve a recovery candidate across at most two missed
        # camera frames; stale masks are still never applied to newer RGB-D.
        "recovery_confirm_max_frame_gap": 3,
        "recovery_confirm_min_bbox_iou": 0.50,
        "recovery_confirm_max_area_ratio": 1.35,
        "recovery_confirm_max_depth_step_m": 0.04,
    },
    "prompting": {
        "service_addr": "tcp://127.0.0.1:5557",
        "service_autostart": True,
        "service_device": "auto",
        "mask_backend": "sam1",
        "startup_timeout_s": 45.0,
        "request_timeout_s": 35.0,
        "box_threshold": 0.25,
        "text_threshold": 0.20,
        "mask_threshold": 0.50,
        "top_k": 5,
        "auto_reacquire": True,
        "global_reacquire": True,
        # Async semantic masks are replayed from their exact source RGB-D frame
        # to the current buffer head with a fresh adaptive tracker.  The old
        # fixed-scale translation matcher remains available as a library helper
        # for offline diagnostics, but is not a production fallback.
        "replay_buffer_s": 2.0,
        "replay_max_frames": 12,
        "relocate_stale_mask": False,
        "stale_mask_after_frames": 3,
        "stale_mask_min_score": 0.70,
        "stale_mask_min_score_margin": 0.04,
        "reacquire_lost_frames": 3,
        "reacquire_cooldown_s": 0.5,
    },
    "runtime": {
        "vis": True,
        "show_pcd": True,
        "show_scene_pcd": False,
        "pcd_point_size": 3.0,
        "scene_stride": 4,
        "scene_update_every": 2,
        "scene_brightness": 0.32,
        "scene_color_floor": 0.08,
        "goal_clearance_m": None,
        "publish_zmq": False,
        "zmq_addr": "tcp://127.0.0.1:5556",
        "max_packet_hz": 30,
        "min_packet_hz": 20,
        "save_debug_masks": False,
        "debug_masks_dir": "mask_debug",
    },
}


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(
    path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}
        deep_update(cfg, file_cfg)
    if overrides:
        deep_update(cfg, overrides)
    calibration_file = cfg.get("extrinsics", {}).get("calibration_file")
    if calibration_file:
        calibration_path = Path(str(calibration_file)).expanduser()
        if not calibration_path.is_absolute() and path is not None:
            calibration_path = (
                Path(path).expanduser().resolve().parent / calibration_path
            )
        cfg["extrinsics"]["calibration_file"] = str(calibration_path.resolve())
    return cfg


def save_config(cfg: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
