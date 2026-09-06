#!/usr/bin/env python3
"""Deployment-aligned, read-only preview of the final policy object cloud.

This entry point deliberately reuses the production deployment's camera-only
object preflight, mask-only provider, SAM2 configuration, RGB-D projector and
live visualizer.  It never opens either robot interface and never evaluates or
commits a policy action.  Without a Franka sample the final 128 points are
published in ``robot_base`` (implemented by an explicitly recorded identity
``T_base_palm``); this command therefore does not claim palm-frame validation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Optional, Sequence

import numpy as np

from .observation.capture import _ComputeThreadGuard, _initialized_roi_evidence
from .console_output import compact_deployment_console, emit_operator_line
from .deployment.bundle import DeployBundle
from .observation.live_preview import (
    _Latest,
    _apply_online_sam2_live_override,
    _camera_worker,
    _initialize_provider,
    _require_runtime_frame_timeout_contract,
)
from .deployment.runner import (
    DEFAULT_BUNDLE,
    DEFAULT_PCD_CONFIG,
    DEFAULT_PROFILE,
    DEPLOYMENT_COMPUTE_THREADS,
    POLICY_RATE_HZ,
    DeploymentRequest,
    _parse_object_roi,
    _parse_object_text,
    _preflight_current_object_roi,
    _selected_point_feature_contract,
)
from .runtime.supervised_v94_runtime import D435_RUNTIME_FRAME_TIMEOUT_MS
from .observation.camera_profile import (
    load_resolved_task_config,
    resolve_runtime_task_contract,
    task_profile_policy_rate_hz,
    task_profile_rollout_trigger,
)
from sim2real.contracts.v94 import V94Contract
from .runtime.v94_live_observation_owner import (
    ProductionV94PolicyTickSourceFactory,
    _ThrownObjectRolloutTrigger,
)
from .observation.visualization import (
    V94LiveVisualizationSample,
    V94LiveVisualizer,
    mask_video_path_for_recording,
)
from .observation.model import (
    MaskedRGBDProjector,
    PolicyRGBDResolutionAdapter,
    resolve_policy_rgbd_resolution,
)
from .observation.roi_selector import (
    _GroundingSearchPreview,
    _grounding_candidate_depth_admission,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preview the exact deployment mask and final 128-point policy "
            "cloud without controlling Franka or opening RH56."
        )
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--pcd-config", type=Path, default=DEFAULT_PCD_CONFIG)
    parser.add_argument(
        "--object-mask-mode",
        choices=("guarded", "guarded_v2", "guarded_v1", "legacy"),
        default="guarded_v2",
        help=(
            "use unified guarded recovery (default/guarded_v2), the former "
            "guarded_v1 double-confirm path, or direct semantic-SAM2 legacy "
            "with explicit retained stale_palm compatibility"
        ),
    )
    parser.add_argument(
        "--policy-rgbd-resolution",
        choices=("848x480", "424x240"),
        default="848x480",
        help=(
            "policy point-cloud RGB-D/mask resolution; the task profile owns "
            "the native camera/SAM2 profile and any exact 2x decimation"
        ),
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--object-text",
        metavar="TEXT",
        help="deployment text grounding, for example 'red cube'",
    )
    target.add_argument(
        "--select-object-roi",
        action="store_true",
        help="select the deployment initialization box interactively",
    )
    target.add_argument(
        "--object-roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        help="use a numeric deployment initialization box",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="seconds after the first valid cloud; 0 runs until Ctrl+C",
    )
    parser.add_argument(
        "--visualization-rate-hz",
        type=float,
        default=10.0,
        help="lossy viewer refresh rate in 1..15 Hz",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="snapshot name; default uses the current wall-clock time",
    )
    parser.add_argument(
        "--save-directory",
        type=Path,
        default=None,
        help=(
            "final exact mask/cloud snapshot directory; default is "
            "dexgrasp/runs/<run-id>_observation_visualization"
        ),
    )
    parser.add_argument(
        "--record-video",
        type=Path,
        default=None,
        help=(
            "optional MP4 containing every published camera frame with the "
            "exact final policy mask inset; refuses to overwrite"
        ),
    )
    parser.add_argument(
        "--record-video-rate-hz",
        type=float,
        default=20.0,
        help="MP4 playback rate in 1..30 Hz (default: 20)",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=20,
        help="print one final-cloud diagnostic every N new camera frames",
    )
    parser.add_argument(
        "--test-rollout-trigger",
        action="store_true",
        help=(
            "camera-only thrown-task test: arm the production object-motion/"
            "entry trigger, capture the full post-trigger flight interval, "
            "and never open robot interfaces"
        ),
    )
    parser.add_argument(
        "--post-trigger-capture-s",
        type=float,
        default=3.0,
        help=(
            "seconds to keep capturing mask/point-cloud evidence after the "
            "throw is detected (default: 3; only used with "
            "--test-rollout-trigger)"
        ),
    )
    parser.add_argument(
        "--verbose-console",
        action="store_true",
        help="show full service/provider diagnostics instead of compact operator stages",
    )
    return parser


def _request_from_args(args: argparse.Namespace) -> DeploymentRequest:
    bundle = args.bundle.expanduser().resolve()
    profile = args.profile.expanduser().resolve()
    pcd_config = args.pcd_config.expanduser().resolve()
    checkpoint = (
        None if args.checkpoint is None else args.checkpoint.expanduser().resolve()
    )
    for name, path in (
        ("bundle", bundle),
        ("profile", profile),
        ("pcd config", pcd_config),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} is missing: {path}")
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {checkpoint}")
    task_policy_rate_hz = task_profile_policy_rate_hz(pcd_config)
    return DeploymentRequest(
        run_id=str(args.run_id),
        steps=1,
        bundle=bundle,
        profile=profile,
        pcd_config=pcd_config,
        execute=False,
        policy_rate_hz=(
            POLICY_RATE_HZ
            if task_policy_rate_hz is None
            else float(task_policy_rate_hz)
        ),
        checkpoint=checkpoint,
        object_roi_xywh=_parse_object_roi(args.object_roi),
        select_object_roi=bool(args.select_object_roi),
        object_text=_parse_object_text(args.object_text),
        object_mask_mode=str(getattr(args, "object_mask_mode", "guarded")),
        policy_rgbd_resolution=str(args.policy_rgbd_resolution),
        object_roi_source=(
            "text_grounding_camera_preflight"
            if args.object_text is not None
            else (
                "operator_numeric_camera_preflight"
                if args.object_roi is not None
                else "operator_interactive_camera_preflight"
            )
        ),
    )


def _validate_runtime_args(args: argparse.Namespace) -> None:
    duration = float(args.duration)
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError("--duration must be finite and non-negative")
    post_trigger_capture_s = float(args.post_trigger_capture_s)
    if (
        not math.isfinite(post_trigger_capture_s)
        or not 0.0 <= post_trigger_capture_s <= 10.0
    ):
        raise ValueError("--post-trigger-capture-s must be in 0..10")
    rate = float(args.visualization_rate_hz)
    if not math.isfinite(rate) or not 1.0 <= rate <= 15.0:
        raise ValueError("--visualization-rate-hz must be in 1..15")
    video_rate = float(args.record_video_rate_hz)
    if not math.isfinite(video_rate) or not 1.0 <= video_rate <= 30.0:
        raise ValueError("--record-video-rate-hz must be in 1..30")
    if args.record_video is not None:
        video_path = args.record_video.expanduser().resolve()
        if video_path.suffix.lower() != ".mp4":
            raise ValueError("--record-video path must end in .mp4")
        if video_path.exists():
            raise FileExistsError(f"--record-video refuses to overwrite: {video_path}")
        mask_video_path = mask_video_path_for_recording(video_path)
        if mask_video_path.exists():
            raise FileExistsError(
                f"--record-video refuses to overwrite mask sidecar: {mask_video_path}"
            )
    if isinstance(args.print_every, bool) or int(args.print_every) <= 0:
        raise ValueError("--print-every must be a positive integer")


def _mask_bbox_xyxy(mask: np.ndarray) -> Optional[list[int]]:
    rows, columns = np.nonzero(np.asarray(mask, dtype=bool))
    if columns.size == 0:
        return None
    return [
        int(columns.min()),
        int(rows.min()),
        int(columns.max()),
        int(rows.max()),
    ]


@contextmanager
def _silence_native_service_startup(enabled: bool):
    """Silence model subprocess chatter while retaining later operator logs."""

    if not enabled:
        yield
        return
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        stdout_fd = int(sys.__stdout__.fileno())
        stderr_fd = int(sys.__stderr__.fileno())
    except (AttributeError, OSError, ValueError):
        yield
        return
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, stdout_fd)
        os.dup2(null_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(null_fd)


def _initialize_continuous_text_grounded_provider(
    request: DeploymentRequest,
    *,
    compact_startup: bool,
) -> tuple[
    Any,
    Any,
    tuple[int, int, int, int],
    _GroundingSearchPreview,
    Any,
]:
    """Ground and retain one camera/SAM2 owner for the trigger diagnostic.

    Unlike deployment's isolated numeric-ROI handoff, this camera-only test
    has no actuator interfaces to isolate.  Retaining the exact initialized
    provider prevents a stale bbox and a second 26-second SAM2 compile between
    TRACKED and ARMED.
    """

    source_path = Path(__file__).resolve().parents[1] / "perception"
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))
    from dynamic_pcd.apps.realtime_masked_pcd import (
        prompt_service_launcher_args,
        prompt_service_launcher_path,
        validate_prompt_service_health,
        validate_prompting_config,
        wait_for_prompt_target,
    )
    from dynamic_pcd.config import load_config
    from dynamic_pcd.provider.object_pcd_provider import ObjectPCDProvider
    from dynamic_pcd.segmentation.prompt_protocol import validate_prompt
    from dynamic_pcd.segmentation.prompt_replay import RecentRGBDFrameBuffer
    from dynamic_pcd.segmentation.prompt_runtime import PromptServiceManager

    prompt = validate_prompt(str(request.object_text or ""))
    config = load_config(str(request.pcd_config))
    _require_runtime_frame_timeout_contract(
        config,
        expected_timeout_ms=D435_RUNTIME_FRAME_TIMEOUT_MS,
    )
    selection = _apply_online_sam2_live_override(
        config,
        disable_online_sam2=False,
        object_mask_mode=request.object_mask_mode,
    )
    prompt_cfg = config.get("prompting", {})
    validate_prompting_config(prompt_cfg)
    prompt_manager = PromptServiceManager(
        addr=str(prompt_cfg["service_addr"]),
        autostart=bool(prompt_cfg["service_autostart"]),
        startup_timeout_s=float(prompt_cfg["startup_timeout_s"]),
        request_timeout_ms=max(
            1, int(1000.0 * float(prompt_cfg["request_timeout_s"]))
        ),
        launcher_path=str(prompt_service_launcher_path(prompt_cfg)),
        launcher_args=prompt_service_launcher_args(prompt_cfg, prompt=prompt),
    )
    provider = ObjectPCDProvider(config)
    provider.deployment_requested_object_mask_mode = str(
        selection.requested_object_mask_mode or "config_default"
    )
    provider.deployment_effective_object_mask_mode = str(
        selection.effective_object_mask_mode
    )
    provider.deployment_effective_mask_publication_mode = str(
        selection.effective_mask_publication_mode
    )
    provider.deployment_effective_recovery_publication_mode = str(
        selection.effective_recovery_publication_mode
    )
    preview = _GroundingSearchPreview(prompt)
    committed = False
    try:
        with _silence_native_service_startup(compact_startup):
            prompt_health = prompt_manager.start()
            validate_prompt_service_health(prompt_cfg, prompt_health)
            provider.start()
        fps = max(1.0, float(config["camera"].get("fps", 30)))
        frame_buffer = RecentRGBDFrameBuffer(
            retention_s=float(prompt_cfg.get("replay_buffer_s", 2.0)),
            capacity_frames=max(
                2,
                int(
                    np.ceil(
                        float(prompt_cfg.get("replay_buffer_s", 2.0)) * fps
                    )
                )
                + 2,
            ),
        )
        detector_backend = str(
            prompt_cfg.get("detector_backend", "grounding_dino")
        ).strip().lower()
        print(
            "[Object grounding SEARCHING] live camera preview opened; "
            "hold the target naturally in view; small hand jitter is allowed; "
            "do not throw yet",
            flush=True,
        )
        while True:
            frame, result = wait_for_prompt_target(
                provider=provider,
                prompt_manager=prompt_manager,
                prompt=prompt,
                frame_buffer=frame_buffer,
                box_threshold=float(prompt_cfg["box_threshold"]),
                text_threshold=float(prompt_cfg["text_threshold"]),
                mask_threshold=float(prompt_cfg["mask_threshold"]),
                top_k=int(prompt_cfg["top_k"]),
                search_interval_s=float(prompt_cfg.get("search_interval_s", 0.5)),
                detector_backend=detector_backend,
                reference_bbox_xyxy=prompt_cfg.get("search_reference_roi_xyxy"),
                yolo_world_entry_filter_config=prompt_cfg,
                preview_callback=preview.show,
            )
            bbox = np.asarray(result.bbox_xyxy, dtype=np.int32).reshape(4)
            depth_ok, depth_reason = _grounding_candidate_depth_admission(
                frame,
                bbox,
                z_min_m=float(config["camera"]["z_min"]),
                z_max_m=float(config["camera"]["z_max"]),
                minimum_valid_depth_ratio=float(
                    config["tracker"].get(
                        "component_min_valid_depth_ratio", 0.25
                    )
                ),
                require_interior=True,
                T_base_camera=provider.extrinsics.T_base_camera,
                workspace_center_base_m=prompt_cfg.get(
                    "grounding_workspace_center_base_m"
                ),
                workspace_max_horizontal_distance_m=prompt_cfg.get(
                    "grounding_workspace_max_horizontal_distance_m"
                ),
            )
            if not depth_ok:
                preview.show(frame, "rejected", bbox, depth_reason)
                print(
                    "[Object grounding RETRY] "
                    f"bbox={bbox.tolist()} {depth_reason}; continuing search",
                    flush=True,
                )
                continue
            if detector_backend == "yolo_world":
                initialized = provider.initialize_from_bbox(frame, bbox)
            else:
                initialized = provider.initialize_from_mask(frame, result.mask)
            if not initialized:
                preview.show(
                    frame,
                    "rejected",
                    bbox,
                    "SAM2/tracker could not initialize this candidate",
                )
                print(
                    "[Object grounding RETRY] "
                    f"bbox={bbox.tolist()} SAM2/tracker initialization failed; "
                    "keep the target still and visible",
                    flush=True,
                )
                continue
            initialization = _initialized_roi_evidence(provider)
            source = str(
                np.asarray(initialization["initialization_mask_source"]).item()
            )
            if detector_backend == "yolo_world" and source != "online_sam2_box":
                raise RuntimeError(
                    "YOLO-World bbox did not initialize the required online SAM2 path"
                )
            roi = (
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[2] - bbox[0]),
                int(bbox[3] - bbox[1]),
            )
            preview.show(frame, "tracked", bbox, depth_reason)
            print(
                "[Object grounding TRACKED] "
                f"bbox={bbox.tolist()} {depth_reason}; "
                "switching directly to the live mask/point-cloud view",
                flush=True,
            )
            committed = True
            return provider, selection, roi, preview, prompt_manager
    finally:
        if not committed:
            prompt_manager.close()
            preview.close()
            provider.stop()


_SAM2_HEALTH_FIELDS = (
    "service",
    "protocol_version",
    "backend",
    "loaded",
    "initialized",
    "checkpoint",
    "checkpoint_sha256",
    "model_config",
    "model_config_sha256",
    "resolved_model_config_path",
    "device",
    "gpu_name",
    "image_size",
    "amp_dtype",
    "tf32",
    "fill_hole_area",
    "safe_history_frames",
    "reset_every_frames",
    "generation",
    "model_load_ms",
)


def _sha256_file_if_present(path: object) -> Optional[str]:
    raw = str(path or "").strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser().resolve()
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sam2_service_provenance(
    provider: Any, *, pcd_config_path: Path
) -> dict[str, object]:
    """Bind the already-running SAM2 health response to local model bytes.

    The manager's ZMQ client is thread-affine, so health is queried through
    the provider's existing executor.  This helper never starts a service,
    camera, Franka reader, RH56 transport, or controller.  Missing metadata is
    recorded as a failed identity check rather than inferred from file paths.
    """

    configured = dict(getattr(provider, "online_sam2_cfg", {}) or {})
    manager = getattr(provider, "online_sam2_manager", None)
    executor = getattr(provider, "_online_sam2_executor", None)
    pcd_path = Path(pcd_config_path).expanduser().resolve()
    configured_checkpoint = str(configured.get("checkpoint", ""))
    configured_model_config = str(configured.get("model_config", ""))
    expected_checkpoint_sha256 = _sha256_file_if_present(configured_checkpoint)
    result: dict[str, object] = {
        "query_attempted": False,
        "query_succeeded": False,
        "query_error": None,
        "pcd_config_path": str(pcd_path),
        "pcd_config_sha256": _sha256_file_if_present(pcd_path),
        "configured": {
            "enabled": bool(configured.get("enabled", False)),
            "service_addr": str(configured.get("service_addr", "")),
            "checkpoint": configured_checkpoint,
            "checkpoint_sha256_from_local_file": (expected_checkpoint_sha256),
            "model_config": configured_model_config,
            "device": str(configured.get("device", "")),
            "image_size": configured.get("image_size"),
            "amp_dtype": str(configured.get("amp_dtype", "")),
            "reset_every_frames": configured.get("reset_every_frames"),
            "mask_publication_mode": str(configured.get("mask_publication_mode", "")),
            "guarded_v2_semantic_primary": bool(
                configured.get("guarded_v2_semantic_primary", False)
            ),
        },
        "manager_present": manager is not None,
        "manager_owns_process": bool(getattr(manager, "owns_process", False)),
        "manager_process_pid": (
            getattr(getattr(manager, "process", None), "pid", None)
        ),
        "provider_reported_ready": bool(getattr(provider, "_online_sam2_ready", False)),
        "health": None,
        "identity_checks": {},
        "identity_valid_for_production": False,
    }

    health: dict[str, object] = {}
    if manager is None or executor is None:
        result["query_error"] = (
            "SAM2 manager/executor is unavailable; health was not inferred"
        )
    else:
        result["query_attempted"] = True
        try:
            future = executor.submit(manager.health, 1000)
            raw_health = future.result(timeout=2.0)
            if not isinstance(raw_health, dict):
                raise TypeError("SAM2 health response is not a mapping")
            for name in _SAM2_HEALTH_FIELDS:
                value = raw_health.get(name)
                if isinstance(value, np.generic):
                    value = value.item()
                if value is None or isinstance(value, (str, int, float, bool)):
                    health[name] = value
            result["query_succeeded"] = True
            result["health"] = health
        except Exception as exc:
            result["query_error"] = f"{type(exc).__name__}: {exc}"

    resolved_model_config_path = str(health.get("resolved_model_config_path", "") or "")
    expected_model_config_sha256 = _sha256_file_if_present(resolved_model_config_path)
    configured_summary = result["configured"]
    assert isinstance(configured_summary, dict)
    configured_summary["resolved_model_config_path_from_health"] = (
        resolved_model_config_path or None
    )
    configured_summary["model_config_sha256_from_local_file"] = (
        expected_model_config_sha256
    )

    checks = {
        "health_query_succeeded": bool(result["query_succeeded"]),
        "provider_reported_ready": bool(result["provider_reported_ready"]),
        "service_identity": health.get("service") == "dynamic-pcd-online-sam2",
        "protocol_version": health.get("protocol_version") == 1,
        "official_backend": health.get("backend") == "official-sam2-video-predictor",
        "model_loaded": health.get("loaded") is True,
        "checkpoint_path_identity": str(health.get("checkpoint", ""))
        == configured_checkpoint,
        "checkpoint_hash_identity": bool(
            expected_checkpoint_sha256
            and str(health.get("checkpoint_sha256", "")).lower()
            == expected_checkpoint_sha256.lower()
        ),
        "model_config_name_identity": str(health.get("model_config", ""))
        == configured_model_config,
        "model_config_hash_identity": bool(
            expected_model_config_sha256
            and str(health.get("model_config_sha256", "")).lower()
            == expected_model_config_sha256.lower()
        ),
        "device_identity": str(health.get("device", ""))
        == str(configured.get("device", "")),
        "image_size_identity": health.get("image_size") == configured.get("image_size"),
        "amp_dtype_identity": str(health.get("amp_dtype", ""))
        == str(configured.get("amp_dtype", "")),
        "reset_every_frames_identity": health.get("reset_every_frames")
        == configured.get("reset_every_frames"),
        "numeric_runtime_identity": bool(
            health.get("tf32") is True and health.get("fill_hole_area") == 0
        ),
        "pcd_config_hash_present": bool(result["pcd_config_sha256"]),
    }
    result["identity_checks"] = checks
    result["identity_valid_for_production"] = bool(checks and all(checks.values()))
    return result


def _formal_publication_watchdog_expired(
    *,
    rollout_trigger_enabled: bool,
    last_fresh_publication_monotonic_s: Optional[float],
    now_monotonic_s: float,
) -> bool:
    """Apply the 200 ms cloud watchdog only to the formal cloud benchmark.

    A thrown-object trigger deliberately observes exact current semantic masks
    while the guarded provider is fail-closed and rebuilding publication
    authority.  Treating that recovery interval as a dead camera terminates
    the diagnostic before the production trigger state machine can re-arm.
    Camera/provider failures are still propagated by ``_Latest`` and the
    trigger's sealed maximum-wait deadline remains in force.
    """

    if rollout_trigger_enabled or last_fresh_publication_monotonic_s is None:
        return False
    return (
        float(now_monotonic_s) - float(last_fresh_publication_monotonic_s)
        > 0.200
    )


def _pointcloud_temporal_fallback_config(provider: Any) -> dict[str, object]:
    cfg = getattr(provider, "cfg", {})
    pointcloud = cfg.get("pointcloud", {}) if isinstance(cfg, dict) else {}
    if not isinstance(pointcloud, dict):
        return {}
    allowed = (
        "temporal_fallback",
        "temporal_fallback_max_stale_s",
        "temporal_fallback_max_stale_steps",
        "temporal_fallback_max_image_speed_px_s",
    )
    return {name: pointcloud[name] for name in allowed if name in pointcloud}


def _current_projection_source_mask(camera: Any) -> np.ndarray:
    """Match the production observation owner's current-mask selection."""

    if bool(camera.mask_valid):
        return np.asarray(camera.mask, dtype=bool)
    semantic = getattr(camera, "policy_semantic_mask", None)
    if bool(getattr(camera, "policy_semantic_valid", False)) and semantic is not None:
        semantic_mask = np.asarray(semantic, dtype=bool)
        if semantic_mask.shape != np.asarray(camera.mask).shape:
            raise RuntimeError("policy semantic mask shape differs from camera mask")
        return semantic_mask
    return np.zeros_like(np.asarray(camera.mask), dtype=bool)


def _frame_trace_record(
    *,
    camera: Any,
    point_frame: Any,
    projector: MaskedRGBDProjector,
    point_coordinate_frame: str = "robot_base_via_identity_T_base_palm",
    palm_frame_geometry_validated: bool = False,
) -> dict[str, object]:
    """Build one explicit two-layer mask provenance record.

    ``provider_published_*`` always describes the current formal camera
    publication. ``projector_effective_policy_mask_*`` describes the mask that
    owns the returned policy cloud, which can come from an older frame during
    ``stale_palm`` and can be geometry-completed by the projector. Guarded
    deployment records that retained result for diagnosis but does not admit
    it as a policy input; only explicit legacy mode retains that compatibility.
    """

    provider_mask = np.asarray(camera.mask, dtype=bool)
    semantic_raw = getattr(camera, "policy_semantic_mask", None)
    semantic_mask = (
        None
        if semantic_raw is None
        else np.asarray(semantic_raw, dtype=bool)
    )
    semantic_valid = bool(
        getattr(camera, "policy_semantic_valid", False)
        and semantic_mask is not None
        and np.any(semantic_mask)
    )
    provenance = projector.last_effective_object_mask_provenance
    return {
        "schema_version": 2,
        "mask_provenance_schema": "provider_projector_split_v1",
        "camera_frame_id": int(camera.frame_id),
        "camera_timestamp_s": float(camera.timestamp_s),
        "requested_object_mask_mode": str(camera.requested_object_mask_mode),
        "effective_object_mask_mode": str(camera.effective_object_mask_mode),
        "effective_provider_mask_publication_mode": str(
            camera.effective_mask_publication_mode
        ),
        "effective_provider_recovery_publication_mode": str(
            camera.effective_recovery_publication_mode
        ),
        "provider_published_mask_valid": bool(camera.mask_valid),
        "provider_published_mask_coordinate_space": "source_camera_pixels",
        "provider_published_mask_area_px": int(np.count_nonzero(provider_mask)),
        "provider_published_mask_bbox_xyxy": _mask_bbox_xyxy(provider_mask),
        "provider_published_mask_source": str(camera.provider_published_mask_source),
        "provider_published_mask_message": str(camera.provider_published_mask_message),
        "provider_online_sam2_status": str(camera.provider_online_sam2_status),
        "exact_current_semantic_mask_valid": semantic_valid,
        "exact_current_semantic_mask_area_px": (
            0 if semantic_mask is None else int(np.count_nonzero(semantic_mask))
        ),
        "exact_current_semantic_mask_bbox_xyxy": (
            None if semantic_mask is None else _mask_bbox_xyxy(semantic_mask)
        ),
        "exact_current_semantic_mask_source": str(
            getattr(camera, "policy_semantic_source", "")
        ),
        "pointcloud_status": str(point_frame.status),
        "stale_palm_policy": (
            "legacy_explicit_compatibility"
            if str(camera.requested_object_mask_mode) == "legacy"
            and str(camera.effective_object_mask_mode) == "legacy"
            else "recoverable_fail_closed_no_policy_stage"
        ),
        "policy_input_eligible": bool(
            str(point_frame.status) in {"fresh", "motion_compensated"}
            or (
                str(point_frame.status) == "stale_palm"
                and str(camera.requested_object_mask_mode) == "legacy"
                and str(camera.effective_object_mask_mode) == "legacy"
            )
        ),
        "source_valid_points": int(point_frame.source_valid_points),
        "projector_effective_policy_mask_provenance": str(provenance.kind),
        "projector_effective_policy_mask_source_frame_id": (provenance.source_frame_id),
        "projector_effective_policy_mask_source_captured_realtime_s": (
            provenance.source_captured_at_s
        ),
        "projector_effective_policy_mask_coordinate_space": "policy_rgbd_pixels",
        "projector_effective_policy_mask_area_px": int(provenance.area_px),
        "projector_effective_policy_mask_bbox_xyxy": (
            None
            if provenance.bbox_xyxy is None
            else [int(value) for value in provenance.bbox_xyxy]
        ),
        "provider_timings_ms": np.asarray(
            camera.provider_timings_ms, dtype=np.float64
        ).tolist(),
        "point_coordinate_frame": str(point_coordinate_frame),
        "palm_frame_geometry_validated": bool(palm_frame_geometry_validated),
    }


def _fail_closed_visualization_sample(
    *,
    sequence: int,
    camera: Any,
    point_feature_dim: int,
    T_base_palm_at_capture: Optional[np.ndarray] = None,
    point_coordinate_frame: str = "robot_base_via_identity_T_base_palm",
) -> V94LiveVisualizationSample:
    """Record one exact camera frame with an explicitly empty policy input.

    Full-flight diagnostics must not hide guarded recovery frames by dropping
    them from the MP4, and must never make them look valid by retaining an old
    mask/cloud.  The semantic candidate remains available in ``frames.jsonl``;
    this viewer sample shows exactly what the policy would receive: nothing.
    """

    feature_dim = int(point_feature_dim)
    if feature_dim not in (3, 6):
        raise ValueError("point_feature_dim must be 3 or 6")
    empty_mask = np.zeros_like(np.asarray(camera.mask), dtype=bool)
    return V94LiveVisualizationSample(
        sequence=int(sequence),
        frame_id=int(camera.frame_id),
        captured_realtime_s=float(camera.timestamp_s),
        color_bgr=np.asarray(camera.color_bgr, dtype=np.uint8),
        object_mask=empty_mask,
        pointcloud_xyzrgb_palm=np.zeros((128, feature_dim), dtype=np.float32),
        pointcloud_valid=np.zeros(128, dtype=np.float32),
        T_base_palm_at_capture=(
            np.eye(4, dtype=np.float64)
            if T_base_palm_at_capture is None
            else np.asarray(T_base_palm_at_capture, dtype=np.float64)
        ),
        source_valid_points=0,
        point_coordinate_frame=str(point_coordinate_frame),
        requested_object_mask_mode=str(camera.requested_object_mask_mode),
        effective_object_mask_mode=str(camera.effective_object_mask_mode),
        effective_provider_mask_publication_mode=str(
            camera.effective_mask_publication_mode
        ),
        effective_provider_recovery_publication_mode=str(
            camera.effective_recovery_publication_mode
        ),
        provider_published_mask_valid=False,
        provider_published_mask_area_px=0,
        provider_published_mask_bbox_xyxy=None,
        provider_published_mask_source=str(camera.provider_published_mask_source),
        provider_published_mask_message=str(camera.provider_published_mask_message),
        provider_online_sam2_status=str(camera.provider_online_sam2_status),
        projector_effective_policy_mask_provenance=(
            "current_frame_fail_closed_empty_policy_input"
        ),
        projector_effective_policy_mask_source_frame_id=int(camera.frame_id),
        projector_effective_policy_mask_source_captured_realtime_s=float(
            camera.timestamp_s
        ),
        projector_effective_policy_mask_area_px=0,
        projector_effective_policy_mask_bbox_xyxy=None,
    )


def run(
    args: argparse.Namespace,
    *,
    read_only_policy_shadow: Optional[Any] = None,
) -> dict[str, object]:
    _validate_runtime_args(args)
    run_id = str(args.run_id or "").strip()
    if not run_id:
        run_id = time.strftime("perception-%Y%m%d-%H%M%S")
    args.run_id = run_id
    request = _request_from_args(args)
    trigger_config = (
        task_profile_rollout_trigger(request.pcd_config)
        if bool(args.test_rollout_trigger)
        else None
    )
    if bool(args.test_rollout_trigger) and trigger_config is None:
        raise ValueError(
            "--test-rollout-trigger requires the resolved thrown_object task config"
        )
    if bool(args.test_rollout_trigger):
        if read_only_policy_shadow is None:
            preparing_label = "camera-only test"
        else:
            preparing_label = "read-only policy shadow; robot writes remain zero"
        print(
            f"[Perception PREPARING] {preparing_label}; hold the target naturally "
            "in view while grounding/SAM2 warm up; small hand jitter is allowed "
            "(first compile may take ~30s)",
            flush=True,
        )

    bundle = DeployBundle(request.bundle)
    bundle.verify()
    bundle_contract = V94Contract.from_bundle(bundle)
    _resolved_task_config, runtime_task_profile = load_resolved_task_config(
        request.pcd_config
    )
    if str(runtime_task_profile.get("name", "")) in (
        "thrown_object",
        "thrown_object_v60",
        "thrown_object_v61",
    ):
        bundle_contract = bundle_contract.with_runtime_policy_rate_hz(
            request.policy_rate_hz
        )
    contract = resolve_runtime_task_contract(
        bundle_contract,
        request.pcd_config,
    )
    (
        point_feature_dim,
        point_feature_mode,
        _action_controller,
        fixed_sphere_completion_radius_m,
    ) = _selected_point_feature_contract(request)

    continuous_trigger_grounding = bool(
        trigger_config is not None and request.object_text is not None
    )
    roi: Optional[tuple[int, int, int, int]] = None
    if not continuous_trigger_grounding:
        # Non-trigger diagnostics retain deployment's isolated selector and
        # three-valid-frame preflight exactly.
        request = _preflight_current_object_roi(request)
        if request.object_roi_xywh is None:
            raise RuntimeError("deployment object preflight produced no numeric ROI")
        roi = tuple(int(value) for value in request.object_roi_xywh)

    save_directory = (
        args.save_directory.expanduser().resolve()
        if args.save_directory is not None
        else (
            Path(__file__).resolve().parents[1]
            / "dexgrasp"
            / "runs"
            / f"{run_id}_observation_visualization"
        )
    )
    record_video_path = (
        None if args.record_video is None else args.record_video.expanduser().resolve()
    )
    save_directory.mkdir(parents=True, exist_ok=True)
    frame_trace_path = save_directory / "frames.jsonl"
    if frame_trace_path.exists():
        raise FileExistsError(
            f"perception frame trace refuses to overwrite: {frame_trace_path}"
        )
    stop = threading.Event()
    camera_latest = _Latest()
    provider = None
    camera_thread: Optional[threading.Thread] = None
    compute_guard = None
    visualizer: Optional[V94LiveVisualizer] = None
    grounding_preview: Optional[_GroundingSearchPreview] = None
    grounding_prompt_manager: Optional[Any] = None
    frame_trace = None
    first_valid_monotonic_s: Optional[float] = None
    last_frame_id: Optional[int] = None
    fresh_frames = 0
    motion_compensated_frames = 0
    held_frames = 0
    invalid_frames = 0
    publication_monotonic_s: list[float] = []
    full128_publications = 0
    current_frame_publications = 0
    last_fresh_publication_monotonic_s: Optional[float] = None
    camera_stop_verified = False
    provider_stop_verified = False
    policy_shadow_started = False
    policy_shadow_closed = read_only_policy_shadow is None
    sam2_service: dict[str, object] = {
        "identity_valid_for_production": False,
        "query_error": "provider was not initialized",
    }
    started_monotonic_s = time.monotonic()
    trigger_started_monotonic_s: Optional[float] = None
    trigger_result: Optional[dict[str, object]] = None
    rollout_trigger = (
        None
        if trigger_config is None
        else _ThrownObjectRolloutTrigger(trigger_config)
    )
    post_trigger_capture_requested_s = (
        float(args.post_trigger_capture_s) if rollout_trigger is not None else 0.0
    )
    post_trigger_capture_started_monotonic_s: Optional[float] = None
    post_trigger_capture_deadline_monotonic_s: Optional[float] = None
    post_trigger_capture_elapsed_s = 0.0
    post_trigger_capture_completed = False
    post_trigger_camera_frames = 0
    post_trigger_fresh_frames = 0
    post_trigger_motion_compensated_frames = 0
    post_trigger_stale_frames = 0
    post_trigger_invalid_frames = 0
    post_trigger_semantic_visible_frames = 0
    post_trigger_fresh_semantic_visible_frames = 0
    post_trigger_usable_semantic_visible_frames = 0
    try:
        frame_trace = frame_trace_path.open("x", encoding="utf-8", buffering=1)
        compute_guard = _ComputeThreadGuard(DEPLOYMENT_COMPUTE_THREADS).start()
        if continuous_trigger_grounding:
            (
                provider,
                _online_selection,
                roi,
                grounding_preview,
                grounding_prompt_manager,
            ) = (
                _initialize_continuous_text_grounded_provider(
                    request,
                    compact_startup=not bool(args.verbose_console),
                )
            )
            request = replace(
                request,
                object_roi_xywh=roi,
                select_object_roi=False,
                object_roi_source="continuous_text_grounded_camera_only_trigger",
            )
        else:
            assert roi is not None
            provider, _online_selection = _initialize_provider(
                request.pcd_config,
                roi,
                disable_online_sam2=False,
                object_mask_mode=request.object_mask_mode,
                required_runtime_frame_timeout_ms=D435_RUNTIME_FRAME_TIMEOUT_MS,
            )
        assert roi is not None
        selected_bbox_xyxy = (
            roi[0],
            roi[1],
            roi[0] + roi[2],
            roi[1] + roi[3],
        )
        sam2_service = _sam2_service_provenance(
            provider, pcd_config_path=request.pcd_config
        )
        legacy_stale_palm_compatibility_enabled = bool(
            request.object_mask_mode == "legacy"
            and _online_selection.effective_object_mask_mode == "legacy"
        )
        extractor = getattr(provider, "extractor", None)
        support_plane = (
            None
            if extractor is None
            else getattr(extractor, "support_plane_abcd", None)
        )
        support_clearance = (
            0.0
            if extractor is None
            else float(getattr(extractor, "support_plane_min_clearance_m", 0.0))
        )
        policy_rgbd_adapter = PolicyRGBDResolutionAdapter(
            camera_K=contract.camera_K,
            source_image_size=(contract.camera_width, contract.camera_height),
            target_image_size=resolve_policy_rgbd_resolution(
                request.policy_rgbd_resolution
            ),
        )
        pointcloud_temporal_fallback_config = (
            _pointcloud_temporal_fallback_config(provider)
        )
        projector = MaskedRGBDProjector(
            camera_K=policy_rgbd_adapter.camera_K,
            T_base_camera_optical=contract.T_base_camera_optical,
            image_size=policy_rgbd_adapter.target_image_size,
            depth_range_m=contract.depth_range_m,
            point_feature_dim=point_feature_dim,
            maximum_mask_depth_deviation_m=0.055,
            fixed_sphere_completion_radius_m=(fixed_sphere_completion_radius_m),
            support_plane_abcd=support_plane,
            support_plane_min_clearance_m=support_clearance,
            **pointcloud_temporal_fallback_config,
        )
        visualizer = V94LiveVisualizer(
            update_rate_hz=float(args.visualization_rate_hz),
            save_directory=save_directory,
            record_video_path=record_video_path,
            record_video_rate_hz=float(args.record_video_rate_hz),
            selected_bbox_xyxy=selected_bbox_xyxy,
        )
        point_coordinate_frame = "robot_base_via_identity_T_base_palm"
        palm_frame_geometry_validated = False
        if read_only_policy_shadow is not None:
            read_only_policy_shadow.start(
                contract=contract,
                request=request,
                point_feature_dim=point_feature_dim,
            )
            policy_shadow_started = True
            point_coordinate_frame = "policy_palm"
            palm_frame_geometry_validated = True
        camera_thread = threading.Thread(
            target=_camera_worker,
            args=(provider, stop, camera_latest, "mask_only"),
            name="v94-perception-camera",
            daemon=True,
        )
        camera_thread.start()
        if grounding_prompt_manager is not None:
            grounding_prompt_manager.close()
            grounding_prompt_manager = None
        visualizer.open()
        print(
            "[Perception UI READY] live final-mask and point-cloud windows opened; "
            "keep holding naturally until the green Throw trigger ARMED line",
            flush=True,
        )
        if grounding_preview is not None:
            grounding_preview.close()
            grounding_preview = None

        if read_only_policy_shadow is None:
            print(
                "[Perception-only] deployment pipeline active: "
                f"feature_mode={point_feature_mode} "
                f"final_shape=(128,{point_feature_dim}); "
                f"policy_rgbd={request.policy_rgbd_resolution}; "
                "points=robot_base; Franka=not opened; RH56=not opened; "
                "policy/action/reset=disabled",
                flush=True,
            )
        else:
            print(
                "[Policy shadow READY] read-only Franka/RH56 state + "
                f"real {request.policy_rgbd_resolution} point cloud; "
                "policy starts only after throw detection; robot writes=0",
                flush=True,
            )
        print(
            "[Read-only shadow] Ctrl+C closes read-only interfaces and saves "
            "the last exact frame."
            if read_only_policy_shadow is not None
            else "[Perception-only] Ctrl+C stops and saves the last exact frame.",
            flush=True,
        )
        if rollout_trigger is not None:
            trigger_started_monotonic_s = time.monotonic()
            print(
                "[Throw trigger START] camera/model services ready; "
                "natural hand jitter is allowed; throw only after ARMED",
                flush=True,
            )

        retained_camera = None
        retained_T_base_palm = None
        # Camera-only perception deliberately uses identity.  The separate
        # read-only policy-shadow entry point supplies a fresh measured Franka
        # palm pose without exposing any actuator command path here.
        T_base_palm = (
            np.eye(4, dtype=np.float64)
            if read_only_policy_shadow is None
            else read_only_policy_shadow.sample_T_base_palm()
        )
        sequence = 0
        duration = float(args.duration)
        while not stop.is_set():
            loop_now = time.monotonic()
            if (
                post_trigger_capture_deadline_monotonic_s is not None
                and loop_now >= post_trigger_capture_deadline_monotonic_s
            ):
                assert post_trigger_capture_started_monotonic_s is not None
                post_trigger_capture_elapsed_s = (
                    loop_now - post_trigger_capture_started_monotonic_s
                )
                post_trigger_capture_completed = True
                print(
                    "[Throw trajectory CAPTURED] "
                    f"duration={post_trigger_capture_elapsed_s:.2f}s "
                    f"camera_frames={post_trigger_camera_frames} "
                    f"fresh={post_trigger_fresh_frames} "
                    f"motion_compensated={post_trigger_motion_compensated_frames} "
                    f"stale={post_trigger_stale_frames} "
                    f"invalid={post_trigger_invalid_frames}",
                    flush=True,
                )
                break
            if (
                trigger_config is not None
                and trigger_result is None
                and trigger_started_monotonic_s is not None
                and loop_now - trigger_started_monotonic_s
                >= trigger_config.maximum_wait_s
            ):
                raise RuntimeError(
                    "throw trigger timed out before an exact detection"
                )
            if _formal_publication_watchdog_expired(
                rollout_trigger_enabled=rollout_trigger is not None,
                last_fresh_publication_monotonic_s=(
                    last_fresh_publication_monotonic_s
                ),
                now_monotonic_s=time.monotonic(),
            ):
                raise RuntimeError(
                    "formal perception publication stalled for more than 200ms"
                )
            camera_latest.raise_if_failed()
            camera = camera_latest.get()
            if camera is None:
                stop.wait(0.005)
                continue
            if last_frame_id == int(camera.frame_id):
                camera_latest.wait_for_frame_change(
                    int(camera.frame_id), timeout_s=0.10
                )
                continue
            last_frame_id = int(camera.frame_id)

            if read_only_policy_shadow is not None:
                T_base_palm = read_only_policy_shadow.sample_T_base_palm()

            detected_this_frame = False
            if rollout_trigger is not None and trigger_result is None:
                trigger_mask, trigger_mask_kind = (
                    ProductionV94PolicyTickSourceFactory._throw_trigger_mask(camera)
                )
                decision = rollout_trigger.observe(
                    frame_id=int(camera.frame_id),
                    timestamp_s=float(camera.timestamp_s),
                    mask=trigger_mask,
                    mask_kind=trigger_mask_kind,
                )
                if decision.armed_now:
                    print(
                        "[Throw trigger ARMED] "
                        f"frame={decision.frame_id} basis={decision.reason}; "
                        "throw now",
                        flush=True,
                    )
                if decision.detected:
                    detected_at = time.monotonic()
                    detected_this_frame = True
                    trigger_result = {
                        "enabled": True,
                        "detected": True,
                        "mode": trigger_config.mode,
                        "reason": decision.reason,
                        "frame_id": decision.frame_id,
                        "camera_timestamp_s": decision.timestamp_s,
                        "wait_s": detected_at - trigger_started_monotonic_s,
                        "mask_kind": decision.mask_kind,
                        "mask_area_px": decision.mask_area_px,
                        "centroid_speed_px_s": decision.centroid_speed_px_s,
                        "centroid_displacement_px": (
                            decision.centroid_displacement_px
                        ),
                        "active_franka_owner_at_detection": False,
                        "active_rh56_owner_at_detection": False,
                    }
                    post_trigger_capture_started_monotonic_s = detected_at
                    post_trigger_capture_deadline_monotonic_s = (
                        detected_at + post_trigger_capture_requested_s
                    )
                    post_trigger_capture_completed = bool(
                        post_trigger_capture_requested_s == 0.0
                    )
                    print(
                        "[Throw trigger DETECTED] "
                        f"frame={decision.frame_id} reason={decision.reason} "
                        f"speed={decision.centroid_speed_px_s:.1f}px/s; "
                        "continuing full-flight mask/point-cloud capture for "
                        f"{post_trigger_capture_requested_s:.1f}s",
                        flush=True,
                    )

            projection_source_mask = _current_projection_source_mask(camera)
            policy_rgbd = policy_rgbd_adapter.adapt(
                color_bgr=camera.color_bgr,
                depth_m=(camera.depth_m if camera.depth_raw is None else None),
                depth_raw=camera.depth_raw,
                object_mask=projection_source_mask,
            )
            point_frame = projector.project(
                color_bgr=policy_rgbd.color_bgr,
                depth_m=policy_rgbd.depth_m,
                depth_raw=policy_rgbd.depth_raw,
                depth_scale_m_per_unit=(
                    camera.depth_scale_m_per_unit
                    if camera.depth_raw is not None
                    else None
                ),
                object_mask=policy_rgbd.object_mask,
                T_base_palm_at_capture=T_base_palm,
                captured_at_s=float(camera.timestamp_s),
                frame_id=int(camera.frame_id),
            )

            effective_mask = projector.last_effective_object_mask
            projector_mask_provenance = projector.last_effective_object_mask_provenance
            frame_trace.write(
                json.dumps(
                    _frame_trace_record(
                        camera=camera,
                        point_frame=point_frame,
                        projector=projector,
                        point_coordinate_frame=point_coordinate_frame,
                        palm_frame_geometry_validated=(
                            palm_frame_geometry_validated
                        ),
                    ),
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )

            if trigger_result is not None:
                post_trigger_camera_frames += 1
                semantic = getattr(camera, "policy_semantic_mask", None)
                semantic_visible = bool(
                    getattr(camera, "policy_semantic_valid", False)
                    and semantic is not None
                    and int(np.count_nonzero(np.asarray(semantic, dtype=bool)))
                    >= int(trigger_config.minimum_mask_area_px)
                )
                if semantic_visible:
                    post_trigger_semantic_visible_frames += 1
                if point_frame.status == "fresh":
                    post_trigger_fresh_frames += 1
                    if semantic_visible:
                        post_trigger_fresh_semantic_visible_frames += 1
                        post_trigger_usable_semantic_visible_frames += 1
                elif point_frame.status == "motion_compensated":
                    post_trigger_motion_compensated_frames += 1
                    if semantic_visible:
                        post_trigger_usable_semantic_visible_frames += 1
                elif point_frame.status == "stale_palm":
                    post_trigger_stale_frames += 1
                else:
                    post_trigger_invalid_frames += 1

            if point_frame.status in {"fresh", "motion_compensated"}:
                if effective_mask is None:
                    raise RuntimeError("usable policy cloud has no effective mask")
                points = np.asarray(point_frame.xyzrgb_palm)
                validity = np.asarray(point_frame.valid)
                if (
                    points.shape != (128, point_feature_dim)
                    or validity.shape != (128,)
                    or not np.all(np.isfinite(points))
                    or not np.all(np.isfinite(validity))
                ):
                    raise RuntimeError(
                        "usable formal128 publication has invalid shape or "
                        "non-finite values"
                    )
                retained_camera = replace(
                    camera,
                    mask=policy_rgbd_adapter.mask_to_source_resolution(effective_mask),
                )
                retained_T_base_palm = T_base_palm.copy()
                if point_frame.status == "fresh":
                    fresh_frames += 1
                else:
                    motion_compensated_frames += 1
                published_now = time.monotonic()
                publication_monotonic_s.append(published_now)
                last_fresh_publication_monotonic_s = published_now
                valid_count = int(np.count_nonzero(np.asarray(point_frame.valid) > 0.5))
                if valid_count == 128:
                    full128_publications += 1
                if (
                    int(point_frame.frame_id) == int(camera.frame_id)
                    and projector_mask_provenance.source_frame_id
                    == int(camera.frame_id)
                    and str(projector_mask_provenance.kind).startswith(
                        ("current_frame_", "current_rgb_")
                    )
                ):
                    current_frame_publications += 1
            elif point_frame.status == "stale_palm":
                held_frames += 1
                if not legacy_stale_palm_compatibility_enabled:
                    if trigger_result is not None:
                        sequence += 1
                        visualizer.try_publish(
                            _fail_closed_visualization_sample(
                                sequence=sequence,
                                camera=camera,
                                point_feature_dim=point_feature_dim,
                                T_base_palm_at_capture=T_base_palm,
                                point_coordinate_frame=point_coordinate_frame,
                            )
                        )
                    if held_frames % int(args.print_every) == 1:
                        print(
                            "[Perception-only][HOLD] "
                            f"frame={camera.frame_id} "
                            "status=stale_palm guarded_fail_closed=true "
                            f"retained_frame={point_frame.frame_id}",
                            flush=True,
                        )
                    if (
                        detected_this_frame
                        and post_trigger_capture_requested_s == 0.0
                    ):
                        break
                    continue
            else:
                invalid_frames += 1
                if trigger_result is not None:
                    sequence += 1
                    visualizer.try_publish(
                        _fail_closed_visualization_sample(
                            sequence=sequence,
                            camera=camera,
                            point_feature_dim=point_feature_dim,
                            T_base_palm_at_capture=T_base_palm,
                            point_coordinate_frame=point_coordinate_frame,
                        )
                    )
                if invalid_frames % int(args.print_every) == 1:
                    print(
                        "[Perception-only][HOLD] "
                        f"frame={camera.frame_id} status={point_frame.status} "
                        f"mask_valid={camera.mask_valid} "
                        f"mask_px={int(np.count_nonzero(camera.mask))} "
                        f"source_points={point_frame.source_valid_points}",
                        flush=True,
                    )
                if (
                    detected_this_frame
                    and post_trigger_capture_requested_s == 0.0
                ):
                    break
                continue

            if read_only_policy_shadow is not None:
                read_only_policy_shadow.maybe_infer(
                    point_frame=point_frame,
                    camera_frame_id=int(camera.frame_id),
                    camera_timestamp_s=float(camera.timestamp_s),
                    trigger_result=trigger_result,
                    trigger_detected_monotonic_s=(
                        post_trigger_capture_started_monotonic_s
                    ),
                )

            # ``retained_camera`` carries the exact effective policy mask,
            # expanded only for overlay on the task camera's native frame.
            source_camera = retained_camera
            source_T_base_palm = (
                T_base_palm
                if point_frame.status in {"fresh", "motion_compensated"}
                else retained_T_base_palm
            )
            if source_camera is None or source_T_base_palm is None:
                if (
                    detected_this_frame
                    and post_trigger_capture_requested_s == 0.0
                ):
                    break
                continue
            sequence += 1
            visualizer.try_publish(
                V94LiveVisualizationSample(
                    sequence=sequence,
                    frame_id=int(point_frame.frame_id),
                    captured_realtime_s=float(point_frame.captured_at_s),
                    color_bgr=source_camera.color_bgr,
                    object_mask=source_camera.mask,
                    pointcloud_xyzrgb_palm=point_frame.xyzrgb_palm,
                    pointcloud_valid=point_frame.valid,
                    T_base_palm_at_capture=source_T_base_palm,
                    source_valid_points=int(point_frame.source_valid_points),
                    requested_object_mask_mode=str(
                        source_camera.requested_object_mask_mode
                    ),
                    effective_object_mask_mode=str(
                        source_camera.effective_object_mask_mode
                    ),
                    effective_provider_mask_publication_mode=str(
                        source_camera.effective_mask_publication_mode
                    ),
                    effective_provider_recovery_publication_mode=str(
                        source_camera.effective_recovery_publication_mode
                    ),
                    provider_published_mask_valid=bool(source_camera.mask_valid),
                    provider_published_mask_area_px=int(
                        source_camera.object_mask_area_px
                    ),
                    provider_published_mask_bbox_xyxy=tuple(
                        int(value)
                        for value in np.asarray(
                            source_camera.object_mask_bbox_xyxy,
                            dtype=np.int32,
                        ).reshape(4)
                    ),
                    provider_published_mask_source=str(
                        source_camera.provider_published_mask_source
                    ),
                    provider_published_mask_message=str(
                        source_camera.provider_published_mask_message
                    ),
                    provider_online_sam2_status=str(
                        source_camera.provider_online_sam2_status
                    ),
                    projector_effective_policy_mask_provenance=str(
                        projector_mask_provenance.kind
                    ),
                    projector_effective_policy_mask_source_frame_id=(
                        projector_mask_provenance.source_frame_id
                    ),
                    projector_effective_policy_mask_source_captured_realtime_s=(
                        projector_mask_provenance.source_captured_at_s
                    ),
                    projector_effective_policy_mask_area_px=int(
                        projector_mask_provenance.area_px
                    ),
                    projector_effective_policy_mask_bbox_xyxy=(
                        projector_mask_provenance.bbox_xyxy
                    ),
                    point_coordinate_frame=point_coordinate_frame,
                )
            )
            if (
                detected_this_frame
                and post_trigger_capture_requested_s == 0.0
            ):
                break
            if first_valid_monotonic_s is None:
                first_valid_monotonic_s = time.monotonic()
            if sequence % int(args.print_every) == 0:
                valid = np.asarray(point_frame.valid) > 0.5
                xyz_palm = np.asarray(point_frame.xyzrgb_palm)[valid, :3]
                center = np.mean(xyz_palm, axis=0)
                print(
                    "[Perception-only] "
                    f"seq={sequence} camera_frame={point_frame.frame_id} "
                    f"status={point_frame.status} "
                    f"mask_px={int(np.count_nonzero(source_camera.mask))} "
                    f"source_points={point_frame.source_valid_points} "
                    f"final_valid={int(np.count_nonzero(valid))}/128 "
                    f"center_robot_base={np.array2string(center, precision=4)}",
                    flush=True,
                )
            if (
                rollout_trigger is None
                and duration > 0.0
                and first_valid_monotonic_s is not None
                and time.monotonic() - first_valid_monotonic_s >= duration
            ):
                break
    finally:
        stop.set()
        if grounding_prompt_manager is not None:
            grounding_prompt_manager.close()
            grounding_prompt_manager = None
        if grounding_preview is not None:
            grounding_preview.close()
            grounding_preview = None
        if camera_thread is not None:
            camera_thread.join(timeout=5.0)
        if provider is not None:
            # A camera worker that ignores the stop event must never be hidden
            # behind a successful perception-only summary.  Stop the provider
            # to unblock capture, then require the worker to terminate.
            provider.stop()
            provider_stop_verified = True
        if camera_thread is not None and camera_thread.is_alive():
            camera_thread.join(timeout=5.0)
            if camera_thread.is_alive():
                raise RuntimeError(
                    "perception camera worker did not stop after provider close"
                )
        camera_stop_verified = bool(
            camera_thread is not None and not camera_thread.is_alive()
        )
        if frame_trace is not None:
            frame_trace.close()
        if visualizer is not None:
            visualizer.close()
        if read_only_policy_shadow is not None:
            read_only_policy_shadow.close()
            policy_shadow_closed = True
        if compute_guard is not None:
            compute_guard.close()

    elapsed_s = max(time.monotonic() - started_monotonic_s, 1.0e-9)
    usable_frames = fresh_frames + motion_compensated_frames
    total_frames = usable_frames + held_frames + invalid_frames
    fresh_fraction = (
        float(fresh_frames) / float(total_frames) if total_frames > 0 else 0.0
    )
    usable_fraction = (
        float(usable_frames) / float(total_frames) if total_frames > 0 else 0.0
    )
    full128_fraction = (
        float(full128_publications) / float(usable_frames) if usable_frames > 0 else 0.0
    )
    current_frame_fraction = (
        float(current_frame_publications) / float(usable_frames)
        if usable_frames > 0
        else 0.0
    )
    gaps_ms = 1000.0 * np.diff(np.asarray(publication_monotonic_s, dtype=np.float64))
    cadence_p50_ms = float(np.percentile(gaps_ms, 50)) if gaps_ms.size else None
    cadence_p95_ms = float(np.percentile(gaps_ms, 95)) if gaps_ms.size else None
    cadence_max_ms = float(np.max(gaps_ms)) if gaps_ms.size else None
    cadence_over_50ms_fraction = (
        float(np.mean(gaps_ms > 50.0)) if gaps_ms.size else None
    )
    publication_rate_hz = (
        float(gaps_ms.size) / (float(np.sum(gaps_ms)) * 1.0e-3)
        if gaps_ms.size and float(np.sum(gaps_ms)) > 0.0
        else 0.0
    )
    post_trigger_minimum_camera_frames = int(
        math.ceil(19.0 * post_trigger_capture_requested_s)
    )
    post_trigger_visible_fresh_fraction = (
        float(post_trigger_fresh_semantic_visible_frames)
        / float(post_trigger_semantic_visible_frames)
        if post_trigger_semantic_visible_frames > 0
        else 0.0
    )
    post_trigger_visible_usable_fraction = (
        float(post_trigger_usable_semantic_visible_frames)
        / float(post_trigger_semantic_visible_frames)
        if post_trigger_semantic_visible_frames > 0
        else 0.0
    )
    post_trigger_capture = (
        None
        if rollout_trigger is None
        else {
            "requested_s": post_trigger_capture_requested_s,
            "elapsed_s": post_trigger_capture_elapsed_s,
            "completed": post_trigger_capture_completed,
            "minimum_camera_frames_at_19hz": (
                post_trigger_minimum_camera_frames
            ),
            "camera_frames": post_trigger_camera_frames,
            "fresh_frames": post_trigger_fresh_frames,
            "motion_compensated_frames": (
                post_trigger_motion_compensated_frames
            ),
            "usable_frames": (
                post_trigger_fresh_frames
                + post_trigger_motion_compensated_frames
            ),
            "stale_frames": post_trigger_stale_frames,
            "invalid_frames": post_trigger_invalid_frames,
            "exact_semantic_visible_frames": (
                post_trigger_semantic_visible_frames
            ),
            "fresh_exact_semantic_visible_frames": (
                post_trigger_fresh_semantic_visible_frames
            ),
            "usable_exact_semantic_visible_frames": (
                post_trigger_usable_semantic_visible_frames
            ),
            "fresh_fraction_while_exact_semantic_visible": (
                post_trigger_visible_fresh_fraction
            ),
            "usable_fraction_while_exact_semantic_visible": (
                post_trigger_visible_usable_fraction
            ),
        }
    )
    production_provenance = bool(
        str(request.object_mask_mode) == "guarded_v2"
        and str(_online_selection.effective_object_mask_mode) == "guarded_v2"
        and str(_online_selection.effective_mask_publication_mode)
        == "guarded_sam2_primary"
        and str(_online_selection.effective_recovery_publication_mode)
        == "unified_three_evidence"
    )
    _task_config, task_profile = load_resolved_task_config(request.pcd_config)
    task_profile_selected = bool(task_profile)
    task_profile_commissioned = bool(
        task_profile.get("commissioning_status") == "accepted"
    ) if task_profile_selected else True
    production_defaults_used = bool(
        str(request.object_mask_mode) == "guarded_v2"
        and (
            (
                task_profile_selected
                and str(request.policy_rgbd_resolution)
                == str(task_profile.get("policy_rgbd_resolution", ""))
            )
            or (
                request.pcd_config == DEFAULT_PCD_CONFIG.expanduser().resolve()
                and str(request.policy_rgbd_resolution) == "848x480"
            )
        )
    )
    acceptance_checks = {
        "production_provenance": production_provenance,
        "production_defaults_used": production_defaults_used,
        "task_profile_commissioned": task_profile_commissioned,
        "sam2_service_identity_valid": bool(
            sam2_service.get("identity_valid_for_production", False)
        ),
        "fresh_frames_at_least_60": fresh_frames >= 60,
        "publication_gap_samples_at_least_59": gaps_ms.size >= 59,
        "fresh_fraction_at_least_0p95": fresh_fraction >= 0.95,
        "full128_fraction_exactly_1": full128_fraction == 1.0,
        "current_frame_provenance_fraction_exactly_1": (current_frame_fraction == 1.0),
        "publication_gap_p95_at_most_50ms": bool(
            cadence_p95_ms is not None and cadence_p95_ms <= 50.0
        ),
        "publication_gap_max_at_most_100ms": bool(
            cadence_max_ms is not None and cadence_max_ms <= 100.0
        ),
        "publication_gap_over_50ms_fraction_at_most_0p02": bool(
            cadence_over_50ms_fraction is not None
            and cadence_over_50ms_fraction <= 0.02
        ),
        "publication_rate_at_least_19hz": publication_rate_hz >= 19.0,
        "camera_stop_verified": camera_stop_verified,
        "provider_stop_verified": provider_stop_verified,
        "robot_interfaces_closed": policy_shadow_closed,
        "robot_writes_zero": True,
    }
    if rollout_trigger is not None:
        acceptance_checks = {
            "rollout_trigger_detected": bool(
                trigger_result is not None and trigger_result.get("detected") is True
            ),
            "post_trigger_capture_completed": post_trigger_capture_completed,
            "post_trigger_camera_frames_at_least_19hz": bool(
                post_trigger_camera_frames >= post_trigger_minimum_camera_frames
            ),
            "post_trigger_exact_semantic_visible_frames_at_least_3": bool(
                post_trigger_capture_requested_s == 0.0
                or post_trigger_semantic_visible_frames >= 3
            ),
            "post_trigger_usable_fraction_while_visible_at_least_0p90": bool(
                post_trigger_capture_requested_s == 0.0
                or (
                    post_trigger_semantic_visible_frames >= 3
                    and post_trigger_visible_usable_fraction >= 0.90
                )
            ),
            "production_provenance": production_provenance,
            "production_defaults_used": production_defaults_used,
            "sam2_service_identity_valid": bool(
                sam2_service.get("identity_valid_for_production", False)
            ),
            "camera_stop_verified": camera_stop_verified,
            "provider_stop_verified": provider_stop_verified,
            "robot_interfaces_closed": policy_shadow_closed,
            "robot_writes_zero": True,
        }
    policy_shadow_summary = None
    if read_only_policy_shadow is not None:
        policy_shadow_summary = read_only_policy_shadow.finalize(
            save_directory=save_directory
        )
        acceptance_checks.update(
            {
                "policy_shadow_started": policy_shadow_started,
                "policy_shadow_interfaces_closed": policy_shadow_closed,
                "policy_shadow_inferences_at_least_3": int(
                    policy_shadow_summary.get("inference_count", 0)
                )
                >= 3,
                "policy_shadow_pretrigger_inferences_zero": int(
                    policy_shadow_summary.get("pretrigger_inference_count", -1)
                )
                == 0,
                "policy_shadow_outputs_finite": bool(
                    policy_shadow_summary.get("all_outputs_finite", False)
                ),
                "policy_shadow_robot_writes_zero": bool(
                    policy_shadow_summary.get("robot_hardware_writes", True)
                    is False
                ),
            }
        )
    production_ineligibility_reasons = [
        name for name, passed in acceptance_checks.items() if not passed
    ]
    accepted = not production_ineligibility_reasons
    return {
        "result": "PASS" if accepted else "FAILED",
        "accepted": accepted,
        "production_acceptance_eligible": (
            accepted if rollout_trigger is None else False
        ),
        "production_provenance_valid": production_provenance,
        "production_defaults_used": production_defaults_used,
        "task_profile": dict(task_profile) if task_profile_selected else None,
        "task_profile_commissioned": task_profile_commissioned,
        "diagnostic_only": rollout_trigger is not None,
        "production_ineligibility_reasons": (production_ineligibility_reasons),
        "acceptance_checks": acceptance_checks,
        "mode": (
            "thrown_rollout_trigger_policy_shadow"
            if read_only_policy_shadow is not None
            else (
                "thrown_rollout_trigger_camera_only"
                if rollout_trigger is not None
                else "deployment_aligned_perception_only"
            )
        ),
        "rollout_trigger": trigger_result,
        "post_trigger_capture": post_trigger_capture,
        "pointcloud_temporal_fallback": dict(
            pointcloud_temporal_fallback_config
        ),
        "run_id": run_id,
        "feature_mode": point_feature_mode,
        "final_policy_shape": [128, point_feature_dim],
        "requested_object_mask_mode": request.object_mask_mode,
        "effective_object_mask_mode": (_online_selection.effective_object_mask_mode),
        "effective_provider_mask_publication_mode": (
            _online_selection.effective_mask_publication_mode
        ),
        "effective_provider_recovery_publication_mode": (
            _online_selection.effective_recovery_publication_mode
        ),
        "sam2_temporal_service": sam2_service,
        "stale_palm_policy": (
            "legacy_explicit_compatibility"
            if legacy_stale_palm_compatibility_enabled
            else (
                "bounded_motion_compensated_then_fail_closed"
                if str(
                    pointcloud_temporal_fallback_config.get(
                        "temporal_fallback", "legacy_stale_palm"
                    )
                )
                == "motion_compensated"
                else "recoverable_fail_closed_no_policy_stage"
            )
        ),
        "fresh_frames": fresh_frames,
        "motion_compensated_frames": motion_compensated_frames,
        "usable_frames": usable_frames,
        "stale_palm_frames": held_frames,
        "invalid_frames": invalid_frames,
        "elapsed_s": elapsed_s,
        "effective_fresh_hz": fresh_frames / elapsed_s,
        "effective_usable_hz": usable_frames / elapsed_s,
        "publication_rate_hz": publication_rate_hz,
        "fresh_fraction": fresh_fraction,
        "usable_fraction": usable_fraction,
        "full128_publication_fraction": full128_fraction,
        "current_frame_provenance_fraction": current_frame_fraction,
        "publication_wall_gap_p50_ms": cadence_p50_ms,
        "publication_wall_gap_p95_ms": cadence_p95_ms,
        "publication_wall_gap_max_ms": cadence_max_ms,
        "publication_wall_gap_over_50ms_fraction": (cadence_over_50ms_fraction),
        "snapshot_directory": str(save_directory),
        "frame_trace_path": str(frame_trace_path),
        "record_video_path": (
            None if record_video_path is None else str(record_video_path)
        ),
        "record_mask_video_path": (
            None
            if record_video_path is None
            else str(mask_video_path_for_recording(record_video_path))
        ),
        "point_coordinate_frame": (
            "policy_palm"
            if read_only_policy_shadow is not None
            else "robot_base_via_identity_T_base_palm"
        ),
        "palm_frame_geometry_validated": read_only_policy_shadow is not None,
        "franka_access": (
            "read_only" if read_only_policy_shadow is not None else "not_opened"
        ),
        "rh56_access": (
            "read_only" if read_only_policy_shadow is not None else "not_opened"
        ),
        "franka_interface_opened": read_only_policy_shadow is not None,
        "rh56_interface_opened": read_only_policy_shadow is not None,
        "robot_hardware_interfaces_opened": read_only_policy_shadow is not None,
        "camera_interface_opened": True,
        "camera_stop_verified": camera_stop_verified,
        "provider_stop_verified": provider_stop_verified,
        "robot_hardware_writes": False,
        "camera_configuration_writes": True,
        "policy_inference": read_only_policy_shadow is not None,
        "policy_shadow": policy_shadow_summary,
        "robot_commands": False,
        "hardware_commands": False,
    }


def _save_compact_report(result: dict[str, object]) -> Path:
    run_id = str(result.get("run_id", "")).strip()
    if not run_id:
        raise RuntimeError("compact perception report has no run_id")
    output = (
        Path(__file__).resolve().parents[1]
        / "dexgrasp"
        / "runs"
        / f"{run_id}_perception_report.json"
    )
    payload = dict(result)
    payload["report_path"] = str(output)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            output.unlink()
        except OSError:
            pass
        raise
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        compact = bool(args.test_rollout_trigger and not args.verbose_console)
        with compact_deployment_console(enabled=compact):
            result = run(args)
        if compact:
            report_path = _save_compact_report(result)
            trigger = result.get("rollout_trigger")
            trigger_frame = (
                trigger.get("frame_id") if isinstance(trigger, dict) else None
            )
            post_capture = result.get("post_trigger_capture")
            post_capture_summary = (
                ""
                if not isinstance(post_capture, dict)
                else (
                    f"post_frames={post_capture.get('camera_frames')} "
                    "visible_usable="
                    f"{float(post_capture.get('usable_fraction_while_exact_semantic_visible', 0.0)):.3f} "
                )
            )
            emit_operator_line(
                (
                    "[Throw trigger test PASS] "
                    if result.get("result") == "PASS"
                    else "[Throw trigger test FAILED] "
                )
                + f"detected_frame={trigger_frame} "
                + post_capture_summary
                + f"video={result.get('record_video_path')} "
                + f"report={report_path}",
                error=result.get("result") != "PASS",
            )
        else:
            print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("result") == "PASS" else 1
    except KeyboardInterrupt:
        print(
            "V94 perception-only: interrupted; no robot stop was needed "
            "because no controller or RH56 interface was opened",
            file=sys.stderr,
        )
        return 130
    except SystemExit:
        raise
    except BaseException as exc:
        print(
            f"V94 perception-only: FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
