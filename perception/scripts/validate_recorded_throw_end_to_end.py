#!/usr/bin/env python3
"""Replay text entry through the production mask and policy-PCD chain.

This is an offline, camera-only acceptance tool.  It never imports or opens a
Franka/RH56 interface.  Each ``--track`` is
``NAME:SEARCH_START:EXPECTED_ENTRY:REFERENCE_MASK_DIR``.  The reference
directory contains the independently reviewed ``INDEX.png`` masks produced at
the 20 Hz policy cadence.  Text discovery is evaluated from SEARCH_START,
SAM2/provider state is initialized only from the accepted exact-frame bbox,
and all later masks and 128-point observations come from the current RGB-D
tuple.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (ROOT, WORKSPACE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamic_pcd.apps.realtime_masked_pcd import (  # noqa: E402
    boundary_entry_bbox_touches_image_edge,
    prompt_service_launcher_args,
    prompt_service_launcher_path,
    select_yolo_world_boundary_refinement_candidate,
    select_yolo_world_entry_candidate,
    select_yolo_world_startup_confirmation_candidate,
    validate_prompt_service_health,
    validate_prompting_config,
)
from dynamic_pcd.camera.recorded_rgbd_camera import (  # noqa: E402
    RecordedRGBDCase,
    RecordedRGBDCamera,
)
from dynamic_pcd.config import load_config  # noqa: E402
from dynamic_pcd.provider.object_pcd_provider import ObjectPCDProvider  # noqa: E402
from dynamic_pcd.segmentation.prompt_runtime import PromptServiceManager  # noqa: E402
from dynamic_pcd.utils.geometry import bbox_from_mask, draw_mask_overlay  # noqa: E402
from sim2real.observation.model import MaskedRGBDProjector  # noqa: E402


@dataclass(frozen=True)
class TrackSpec:
    name: str
    search_start: int
    expected_entry: int
    reference_dir: Path
    reference_indices: tuple[int, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_indices(path: Path) -> tuple[int, ...]:
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"reference mask directory missing: {path}")
    indices = []
    for item in sorted(path.glob("*.png")):
        try:
            indices.append(int(item.stem))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"reference mask filename is not an integer: {item.name}"
            ) from exc
    if not indices:
        raise argparse.ArgumentTypeError(f"no PNG reference masks in {path}")
    if len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("duplicate reference mask index")
    residues = {value % 3 for value in indices}
    if len(residues) != 1:
        raise argparse.ArgumentTypeError(
            "reference masks do not share one 60 Hz -> 20 Hz phase"
        )
    return tuple(indices)


def _track_spec(value: str) -> TrackSpec:
    parts = value.split(":", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "track must be NAME:SEARCH_START:EXPECTED_ENTRY:REFERENCE_MASK_DIR"
        )
    name = parts[0].strip()
    try:
        search_start = int(parts[1])
        expected_entry = int(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("track frame indices must be integers") from exc
    reference_dir = Path(parts[3]).expanduser().resolve()
    if not name or search_start < 0 or expected_entry < search_start:
        raise argparse.ArgumentTypeError("track name/range is invalid")
    return TrackSpec(
        name=name,
        search_start=search_start,
        expected_entry=expected_entry,
        reference_dir=reference_dir,
        reference_indices=_reference_indices(reference_dir),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--object-text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track", type=_track_spec, action="append", required=True)
    parser.add_argument("--search-stride", type=int, default=3)
    parser.add_argument("--max-entry-delay-frames", type=int, default=3)
    parser.add_argument("--minimum-visible-coverage", type=float, default=0.95)
    parser.add_argument("--maximum-visible-invalid-run", type=int, default=1)
    parser.add_argument("--maximum-absent-fp-fraction", type=float, default=0.01)
    parser.add_argument("--minimum-iou-p05", type=float, default=0.70)
    parser.add_argument("--minimum-recall-p05", type=float, default=0.78)
    parser.add_argument("--minimum-precision-p05", type=float, default=0.85)
    return parser


def _percentile(values: Iterable[float], q: float) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return None if not finite else float(np.percentile(finite, q))


def _longest_false_run(values: Iterable[bool]) -> int:
    longest = current = 0
    for value in values:
        if bool(value):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _mask_metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    intersection = int(np.logical_and(actual, reference).sum())
    union = int(np.logical_or(actual, reference).sum())
    actual_area = int(actual.sum())
    reference_area = int(reference.sum())
    return {
        "iou": 1.0 if union == 0 else intersection / float(union),
        "recall": (
            1.0 if reference_area == 0 else intersection / float(reference_area)
        ),
        "precision": 1.0 if actual_area == 0 else intersection / float(actual_area),
    }


def _bbox(candidate: Any, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    raw = np.asarray(candidate.bbox_xyxy, dtype=np.float64).reshape(4)
    raw[[0, 2]] = np.clip(raw[[0, 2]], 0.0, float(width))
    raw[[1, 3]] = np.clip(raw[[1, 3]], 0.0, float(height))
    return np.asarray(
        [
            int(np.floor(raw[0])),
            int(np.floor(raw[1])),
            int(np.ceil(raw[2])),
            int(np.ceil(raw[3])),
        ],
        dtype=np.int32,
    )


def _discover(
    *,
    case: RecordedRGBDCase,
    spec: TrackSpec,
    manager: PromptServiceManager,
    prompt_cfg: dict[str, Any],
    prompt: str,
    stride: int,
    max_delay_frames: int,
) -> tuple[int | None, Any | None, np.ndarray | None, list[dict[str, Any]]]:
    previous_gray = None
    attempts: list[dict[str, Any]] = []
    deadline = min(len(case) - 1, spec.expected_entry + max_delay_frames)
    for index in range(spec.search_start, deadline + 1, stride):
        frame = case.frame(index)
        started = time.perf_counter()
        result = manager.segment(
            frame.color_bgr,
            prompt=prompt,
            reference_bbox_xyxy=None,
            request_id=f"offline-{spec.name}-{index}",
            frame_id=int(frame.frame_id),
            frame_timestamp=float(frame.timestamp),
            frame_metadata={
                "tracker_generation": 0,
                "reason": "initial_or_reentry_search",
                "recording_index": index,
            },
            box_threshold=float(prompt_cfg["box_threshold"]),
            text_threshold=float(prompt_cfg["text_threshold"]),
            mask_threshold=float(prompt_cfg["mask_threshold"]),
            top_k=int(prompt_cfg["top_k"]),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        candidate, reason = select_yolo_world_entry_candidate(
            result,
            frame.color_bgr,
            previous_gray,
            prompt_cfg,
        )
        previous_gray = cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2GRAY)
        attempts.append(
            {
                "index": index,
                "frame_id": int(frame.frame_id),
                "elapsed_ms": elapsed_ms,
                "candidate_count": len(result.candidates),
                "accepted": candidate is not None,
                "reason": reason,
            }
        )
        if candidate is not None:
            return index, frame, _bbox(candidate, frame.depth_raw.shape), attempts
    return None, None, None, attempts


def _projector(case: RecordedRGBDCase, provider: ObjectPCDProvider, cfg: dict):
    intrinsics = case.intrinsics
    camera_K = np.asarray(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pointcloud_cfg = cfg.get("pointcloud", {})
    return MaskedRGBDProjector(
        camera_K=camera_K,
        T_base_camera_optical=np.asarray(
            provider.extrinsics.T_base_camera, dtype=np.float64
        ),
        image_size=(case.width, case.height),
        depth_range_m=(float(cfg["camera"]["z_min"]), float(cfg["camera"]["z_max"])),
        num_points=128,
        minimum_valid_points=16,
        point_feature_dim=(6 if bool(cfg["pointcloud"]["use_rgb"]) else 3),
        # Keep this validator byte-for-byte aligned with the live V94
        # observation owner.  Mixed D435 boundary/background depth is not a
        # policy point merely because its color pixel lies inside the SAM2
        # silhouette.
        maximum_mask_depth_deviation_m=0.055,
        temporal_fallback=str(
            pointcloud_cfg.get("temporal_fallback", "legacy_stale_palm")
        ),
        temporal_fallback_max_stale_s=float(
            pointcloud_cfg.get("temporal_fallback_max_stale_s", 0.25)
        ),
        temporal_fallback_max_stale_steps=int(
            pointcloud_cfg.get("temporal_fallback_max_stale_steps", 5)
        ),
        temporal_fallback_max_image_speed_px_s=float(
            pointcloud_cfg.get(
                "temporal_fallback_max_image_speed_px_s", 2400.0
            )
        ),
    )


def _read_reference(spec: TrackSpec, index: int, shape: tuple[int, int]) -> np.ndarray:
    path = spec.reference_dir / f"{index:06d}.png"
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != shape:
        raise RuntimeError(f"cannot read exact reference mask {path}")
    return image > 0


def _evaluate_result(
    *,
    provider: ObjectPCDProvider,
    spec: TrackSpec,
    index: int,
    frame: Any,
    mask_result: Any,
    projector: MaskedRGBDProjector,
    provider_ms: float | None,
    output_mask_dir: Path,
    overlay: cv2.VideoWriter,
) -> dict[str, Any] | None:
    if index not in spec.reference_indices:
        return None
    reference = _read_reference(spec, index, frame.depth_raw.shape)
    provider_actual = (
        np.asarray(mask_result.mask, dtype=bool)
        if bool(mask_result.valid)
        else np.zeros(frame.depth_raw.shape, dtype=bool)
    )
    semantic_raw = getattr(mask_result, "policy_semantic_mask", None)
    semantic_valid = bool(
        getattr(mask_result, "policy_semantic_valid", False)
        and semantic_raw is not None
    )
    semantic_actual = (
        np.asarray(semantic_raw, dtype=bool)
        if semantic_valid
        else np.zeros(frame.depth_raw.shape, dtype=bool)
    )
    if semantic_actual.shape != frame.depth_raw.shape:
        raise RuntimeError("policy semantic mask shape differs from RGB-D frame")
    # The exact-current semantic mask is the thrown policy's current object
    # silhouette.  Guarded publication remains separately audited below and
    # is still the only mask allowed to update provider identity authority.
    actual = semantic_actual if semantic_valid else provider_actual
    policy_mask_kind = (
        "exact_current_rgb_semantic_non_authoritative"
        if semantic_valid
        else ("guarded_provider_publication" if mask_result.valid else "empty")
    )
    project_started = time.perf_counter()
    point_frame = projector.project(
        color_bgr=frame.color_bgr,
        depth_raw=frame.depth_raw,
        depth_scale_m_per_unit=frame.depth_scale,
        object_mask=actual,
        T_base_palm_at_capture=np.eye(4, dtype=np.float64),
        captured_at_s=frame.timestamp,
        frame_id=frame.frame_id,
    )
    projector_ms = (time.perf_counter() - project_started) * 1000.0
    metrics = _mask_metrics(actual, reference)
    bbox = bbox_from_mask(actual.astype(np.uint8))
    bbox_xyxy = (
        np.zeros(4, dtype=np.int32)
        if bbox is None
        else np.asarray(bbox, dtype=np.int32)
    )
    path = output_mask_dir / f"{index:06d}.png"
    if not cv2.imwrite(str(path), actual.astype(np.uint8) * np.uint8(255)):
        raise RuntimeError(f"cannot write {path}")
    total_ms = None if provider_ms is None else float(provider_ms + projector_ms)
    overlay.write(
        draw_mask_overlay(
            frame.color_bgr,
            actual.astype(np.uint8),
            bbox_xyxy,
            text=(
                f"{spec.name} idx={index} ref={int(reference.sum())} "
                f"mask={int(actual.sum())} pcd={point_frame.status}"
            ),
        )
    )
    return {
        "track": spec.name,
        "index": index,
        "frame_id": int(frame.frame_id),
        "reference_visible": bool(reference.any()),
        "reference_area_px": int(reference.sum()),
        "mask_valid": bool(actual.any()),
        "mask_area_px": int(actual.sum()),
        "provider_mask_valid": bool(mask_result.valid),
        "provider_mask_area_px": int(provider_actual.sum()),
        "policy_semantic_valid": semantic_valid,
        "policy_semantic_mask_area_px": int(semantic_actual.sum()),
        "policy_mask_kind": policy_mask_kind,
        "iou": metrics["iou"],
        "recall": metrics["recall"],
        "precision": metrics["precision"],
        "policy128_status": str(point_frame.status),
        "policy128_source_points": int(point_frame.source_valid_points),
        "policy128_measured_frame_id": point_frame.measured_frame_id,
        "policy128_measured_at_s": point_frame.measured_at_s,
        "policy128_fallback_age_s": float(point_frame.fallback_age_s),
        "policy128_fallback_steps": int(point_frame.fallback_steps),
        "provider_ms": provider_ms,
        "projector_ms": projector_ms,
        "provider_plus_projector_ms": total_ms,
        "mask_source": str(getattr(mask_result, "source", "")),
        "publication_source": str(getattr(mask_result, "publication_source", "")),
        "online_sam2_status": str(
            getattr(provider, "_last_online_sam2_status", "")
        ),
        "online_sam2_geometry_reason": str(
            getattr(provider, "_last_online_sam2_geometry_reason", "")
        ),
        "trusted_continuity_reason": str(
            getattr(provider, "_trusted_online_sam2_continuity_reason", "")
        ),
        "visible_partial_reason": str(
            getattr(provider, "_last_visible_exact_partial_reason", "")
        ),
        "visible_partial_failure_kind": str(
            getattr(provider, "_last_visible_exact_partial_failure_kind", "")
        ),
        "visible_partial_repair_reason": str(
            getattr(provider, "_last_visible_exact_partial_repair_reason", "")
        ),
        "tracking_contraction_stage_reason": str(
            getattr(provider, "_last_tracking_contraction_stage_reason", "")
        ),
        "message": str(mask_result.message),
    }


def _track_summary(
    spec: TrackSpec,
    rows: list[dict[str, Any]],
    detected_index: int | None,
    attempts: list[dict[str, Any]],
    max_delay_frames: int,
) -> dict[str, Any]:
    visible = [row for row in rows if row["reference_visible"]]
    absent = [row for row in rows if not row["reference_visible"]]
    visible_valid = [bool(row["mask_valid"] and row["mask_area_px"] > 0) for row in visible]
    absent_fp = [bool(row["mask_valid"] and row["mask_area_px"] > 0) for row in absent]
    latency = [
        float(row["provider_plus_projector_ms"])
        for row in rows
        if row["provider_plus_projector_ms"] is not None
    ]
    delay = None if detected_index is None else detected_index - spec.expected_entry
    return {
        "search_start_index": spec.search_start,
        "expected_entry_index": spec.expected_entry,
        "detected_index": detected_index,
        "entry_delay_frames": delay,
        "entry_delay_s_at_60hz": None if delay is None else delay / 60.0,
        "grounding_attempts": attempts,
        "grounding_latency_ms_p95": _percentile(
            [float(value["elapsed_ms"]) for value in attempts], 95
        ),
        "grounding_detected_within_deadline": bool(
            delay is not None and 0 <= delay <= max_delay_frames
        ),
        "reviewed_visible_frames": len(visible),
        "reviewed_visible_valid_frames": int(sum(visible_valid)),
        "reviewed_visible_coverage": (
            0.0 if not visible else float(np.mean(visible_valid))
        ),
        "longest_visible_invalid_run": _longest_false_run(visible_valid),
        "reviewed_absent_frames": len(absent),
        "reviewed_absent_false_positive_frames": int(sum(absent_fp)),
        "reviewed_absent_false_positive_fraction": (
            0.0 if not absent else float(np.mean(absent_fp))
        ),
        "iou_p05": _percentile([float(row["iou"]) for row in visible], 5),
        "recall_p05": _percentile([float(row["recall"]) for row in visible], 5),
        "precision_p05": _percentile(
            [float(row["precision"]) for row in visible], 5
        ),
        "fresh_policy128_visible_frames": int(
            sum(row["policy128_status"] == "fresh" for row in visible)
        ),
        "fresh_policy128_visible_fraction": (
            0.0
            if not visible
            else float(
                np.mean([row["policy128_status"] == "fresh" for row in visible])
            )
        ),
        "usable_policy128_visible_frames": int(
            sum(
                row["policy128_status"] in {"fresh", "motion_compensated"}
                for row in visible
            )
        ),
        "usable_policy128_visible_fraction": (
            0.0
            if not visible
            else float(
                np.mean(
                    [
                        row["policy128_status"]
                        in {"fresh", "motion_compensated"}
                        for row in visible
                    ]
                )
            )
        ),
        "longest_visible_unusable_policy128_run": _longest_false_run(
            [
                row["policy128_status"] in {"fresh", "motion_compensated"}
                for row in visible
            ]
        ),
        "provider_plus_projector_ms_p50": _percentile(latency, 50),
        "provider_plus_projector_ms_p95": _percentile(latency, 95),
        "provider_plus_projector_ms_max": _percentile(latency, 100),
        "provider_plus_projector_over_50ms_fraction": (
            0.0 if not latency else float(np.mean(np.asarray(latency) > 50.0))
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.search_stride < 1 or args.max_entry_delay_frames < 0:
        raise ValueError("search stride/delay is invalid")
    case = RecordedRGBDCase(args.case_dir)
    config_path = args.config.expanduser().resolve()
    cfg = load_config(str(config_path))
    # Match the production thrown-object launcher.  The materialized YAML is
    # shared with recording and intentionally retains its source defaults;
    # deployment applies ``--object-mask-mode guarded_v2`` in memory.  An
    # acceptance replay must apply the same selection rather than silently
    # benchmarking direct semantic-SAM2 publication.
    cfg["online_sam2"]["enabled"] = True
    cfg["online_sam2"]["require_for_bbox_init"] = True
    cfg["online_sam2"]["guarded_v2_semantic_primary"] = True
    cfg["online_sam2"]["mask_publication_mode"] = "guarded_sam2_primary"
    cfg["tracker"]["recovery_publication_mode"] = "unified_three_evidence"
    prompt_cfg = dict(cfg["prompting"])
    validate_prompting_config(prompt_cfg)
    if str(prompt_cfg["detector_backend"]).strip().lower() != "yolo_world":
        raise ValueError("this fast-entry validator requires yolo_world")
    if str(case.manifest.get("camera_serial")) != str(cfg["camera"]["serial"]):
        raise ValueError("recording/config camera serial mismatch")
    if (case.width, case.height) != (
        int(cfg["camera"]["width"]),
        int(cfg["camera"]["height"]),
    ):
        raise ValueError("recording/config image size mismatch")
    for spec in args.track:
        if max(spec.reference_indices) >= len(case):
            raise ValueError(f"track {spec.name} reference index is outside recording")

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manager = PromptServiceManager(
        addr=str(prompt_cfg["service_addr"]),
        autostart=bool(prompt_cfg["service_autostart"]),
        startup_timeout_s=float(prompt_cfg["startup_timeout_s"]),
        request_timeout_ms=int(1000.0 * float(prompt_cfg["request_timeout_s"])),
        launcher_path=str(prompt_service_launcher_path(prompt_cfg)),
        launcher_args=prompt_service_launcher_args(
            prompt_cfg, prompt=str(args.object_text)
        ),
    )
    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    prompt_health = sam2_health = None
    camera = RecordedRGBDCamera(case, realtime=False)
    provider = ObjectPCDProvider(cfg, camera=camera)
    if not provider._guarded_v2_uses_semantic_primary():
        raise RuntimeError(
            "thrown-object acceptance must run guarded_v2 semantic-primary"
        )
    try:
        prompt_health = manager.start()
        validate_prompt_service_health(prompt_cfg, prompt_health)
        # The provider is the sole SAM2 service owner.  Its launcher arguments
        # bind mandatory compile prewarm to this task's native camera shape
        # (424x240 here; 848x480 for the tabletop profile).  Starting a generic
        # manager first would incorrectly create the historical 848x480
        # service and the provider must then reject its identity.
        provider.start()
        sam2_health = dict(provider._online_sam2_service_health or {})
        for spec in args.track:
            detected_index, detected_frame, detected_bbox, attempts = _discover(
                case=case,
                spec=spec,
                manager=manager,
                prompt_cfg=prompt_cfg,
                prompt=str(args.object_text),
                stride=int(args.search_stride),
                max_delay_frames=int(args.max_entry_delay_frames),
            )
            rows: list[dict[str, Any]] = []
            boundary_refinement: dict[str, Any] = {
                "attempted": False,
                "applied": False,
                "initial_index": detected_index,
                "selected_index": None,
                "attempts": [],
            }
            startup_confirmation: dict[str, Any] = {
                "attempted": False,
                "applied": False,
                "initial_index": detected_index,
                "selected_index": None,
                "attempts": [],
                "provider_step_ms": [],
            }
            if detected_index is not None:
                assert detected_frame is not None and detected_bbox is not None
                track_dir = output / spec.name
                mask_dir = track_dir / "mask"
                mask_dir.mkdir(parents=True)
                overlay = cv2.VideoWriter(
                    str(track_dir / "overlay.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(case.manifest["nominal_fps"]) / 3.0,
                    (case.width, case.height),
                )
                if not overlay.isOpened():
                    raise RuntimeError(f"cannot create overlay for {spec.name}")
                try:
                    if not provider.initialize_from_bbox(detected_frame, detected_bbox):
                        raise RuntimeError(
                            f"track {spec.name} detected bbox failed provider initialization"
                        )
                    projector = _projector(case, provider, cfg)
                    if detected_index in spec.reference_indices:
                        assert provider.last_mask_result is not None
                        row = _evaluate_result(
                            provider=provider,
                            spec=spec,
                            index=detected_index,
                            frame=detected_frame,
                            mask_result=provider.last_mask_result,
                            projector=projector,
                            provider_ms=None,
                            output_mask_dir=mask_dir,
                            overlay=overlay,
                        )
                        if row is not None:
                            rows.append(row)
                    last_processed_index = int(detected_index)
                    initial_boundary_entry = boundary_entry_bbox_touches_image_edge(
                        detected_bbox,
                        detected_frame.color_bgr.shape,
                    )
                    previous_gray = cv2.cvtColor(
                        detected_frame.color_bgr, cv2.COLOR_BGR2GRAY
                    )
                    refinement_enabled = bool(
                        prompt_cfg.get("boundary_entry_refinement_enabled", False)
                    )
                    if refinement_enabled and initial_boundary_entry:
                        boundary_refinement["attempted"] = True
                        max_ticks = int(
                            prompt_cfg.get("boundary_entry_refinement_max_ticks", 2)
                        )
                        for tick in range(1, max_ticks + 1):
                            refine_index = int(detected_index) + tick * int(
                                args.search_stride
                            )
                            if refine_index >= len(case):
                                break
                            camera.seek(refine_index)
                            frame_started = time.perf_counter()
                            refine_frame, _tracked_mask = provider.step_mask_only()
                            provider_ms = (
                                time.perf_counter() - frame_started
                            ) * 1000.0
                            if int(refine_frame.frame_id) != int(
                                case.records[refine_index]["frame_id"]
                            ):
                                raise RuntimeError(
                                    "provider returned the wrong boundary "
                                    "refinement frame"
                                )
                            if refine_index in spec.reference_indices:
                                row = _evaluate_result(
                                    provider=provider,
                                    spec=spec,
                                    index=refine_index,
                                    frame=refine_frame,
                                    mask_result=_tracked_mask,
                                    projector=projector,
                                    provider_ms=provider_ms,
                                    output_mask_dir=mask_dir,
                                    overlay=overlay,
                                )
                                if row is not None:
                                    rows.append(row)
                            prompt_started = time.perf_counter()
                            result = manager.segment(
                                refine_frame.color_bgr,
                                prompt=str(args.object_text),
                                reference_bbox_xyxy=None,
                                request_id=(
                                    f"offline-{spec.name}-boundary-refine-"
                                    f"{refine_index}"
                                ),
                                frame_id=int(refine_frame.frame_id),
                                frame_timestamp=float(refine_frame.timestamp),
                                frame_metadata={
                                    "tracker_generation": 0,
                                    "reason": "boundary_entry_refinement",
                                    "recording_index": refine_index,
                                },
                                box_threshold=float(prompt_cfg["box_threshold"]),
                                text_threshold=float(prompt_cfg["text_threshold"]),
                                mask_threshold=float(prompt_cfg["mask_threshold"]),
                                top_k=int(prompt_cfg["top_k"]),
                            )
                            prompt_ms = (
                                time.perf_counter() - prompt_started
                            ) * 1000.0
                            decision, reason = (
                                select_yolo_world_boundary_refinement_candidate(
                                    result,
                                    refine_frame.color_bgr,
                                    previous_gray,
                                    detected_bbox,
                                    prompt_cfg,
                                )
                            )
                            previous_gray = cv2.cvtColor(
                                refine_frame.color_bgr, cv2.COLOR_BGR2GRAY
                            )
                            record = {
                                "tick": tick,
                                "index": refine_index,
                                "frame_id": int(refine_frame.frame_id),
                                "provider_ms": provider_ms,
                                "prompt_ms": prompt_ms,
                                "candidate_count": len(result.candidates),
                                "accepted": decision is not None,
                                "reason": reason,
                            }
                            boundary_refinement["attempts"].append(record)
                            last_processed_index = refine_index
                            if decision is None:
                                continue
                            initialize_started = time.perf_counter()
                            if not provider.initialize_from_bbox(
                                refine_frame, decision.bbox_xyxy
                            ):
                                raise RuntimeError(
                                    f"track {spec.name} accepted boundary "
                                    "refinement failed atomic SAM2 initialization"
                                )
                            record["initialize_ms"] = (
                                time.perf_counter() - initialize_started
                            ) * 1000.0
                            boundary_refinement["applied"] = True
                            boundary_refinement["selected_index"] = refine_index
                            boundary_refinement["selected_bbox_xyxy"] = (
                                decision.bbox_xyxy.astype(int).tolist()
                            )
                            break
                    confirmation_enabled = bool(
                        prompt_cfg.get(
                            "startup_semantic_confirmation_enabled", False
                        )
                    )
                    if confirmation_enabled and initial_boundary_entry:
                        startup_confirmation["attempted"] = True
                        max_confirmation_ticks = int(
                            prompt_cfg.get(
                                "startup_semantic_confirmation_max_ticks", 10
                            )
                        )
                        search_every_ticks = int(
                            prompt_cfg.get(
                                "startup_semantic_confirmation_search_every_ticks",
                                2,
                            )
                        )
                        first_tick = (
                            (last_processed_index - int(detected_index))
                            // int(args.search_stride)
                            + 1
                        )
                        for tick in range(first_tick, max_confirmation_ticks + 1):
                            confirmation_index = int(detected_index) + tick * int(
                                args.search_stride
                            )
                            if confirmation_index >= len(case):
                                break
                            camera.seek(confirmation_index)
                            step_started = time.perf_counter()
                            confirmation_frame, tracked_mask = provider.step_mask_only()
                            provider_ms = (
                                time.perf_counter() - step_started
                            ) * 1000.0
                            startup_confirmation["provider_step_ms"].append(
                                provider_ms
                            )
                            if int(confirmation_frame.frame_id) != int(
                                case.records[confirmation_index]["frame_id"]
                            ):
                                raise RuntimeError(
                                    "provider returned the wrong startup "
                                    "confirmation frame"
                                )
                            if confirmation_index in spec.reference_indices:
                                row = _evaluate_result(
                                    provider=provider,
                                    spec=spec,
                                    index=confirmation_index,
                                    frame=confirmation_frame,
                                    mask_result=tracked_mask,
                                    projector=projector,
                                    provider_ms=provider_ms,
                                    output_mask_dir=mask_dir,
                                    overlay=overlay,
                                )
                                if row is not None:
                                    rows.append(row)
                            last_processed_index = confirmation_index
                            prior_gray = previous_gray
                            previous_gray = cv2.cvtColor(
                                confirmation_frame.color_bgr, cv2.COLOR_BGR2GRAY
                            )
                            if tick % search_every_ticks != 0:
                                continue
                            prompt_started = time.perf_counter()
                            result = manager.segment(
                                confirmation_frame.color_bgr,
                                prompt=str(args.object_text),
                                reference_bbox_xyxy=None,
                                request_id=(
                                    f"offline-{spec.name}-startup-confirm-"
                                    f"{confirmation_index}"
                                ),
                                frame_id=int(confirmation_frame.frame_id),
                                frame_timestamp=float(confirmation_frame.timestamp),
                                frame_metadata={
                                    "tracker_generation": 0,
                                    "reason": "startup_semantic_confirmation",
                                    "recording_index": confirmation_index,
                                },
                                box_threshold=float(prompt_cfg["box_threshold"]),
                                text_threshold=float(prompt_cfg["text_threshold"]),
                                mask_threshold=float(prompt_cfg["mask_threshold"]),
                                top_k=int(prompt_cfg["top_k"]),
                            )
                            prompt_ms = (
                                time.perf_counter() - prompt_started
                            ) * 1000.0
                            decision, reason = (
                                select_yolo_world_startup_confirmation_candidate(
                                    result,
                                    confirmation_frame.color_bgr,
                                    prior_gray,
                                    detected_bbox,
                                    float(confirmation_frame.timestamp)
                                    - float(detected_frame.timestamp),
                                    prompt_cfg,
                                )
                            )
                            record = {
                                "tick": tick,
                                "index": confirmation_index,
                                "frame_id": int(confirmation_frame.frame_id),
                                "provider_ms": provider_ms,
                                "prompt_ms": prompt_ms,
                                "candidate_count": len(result.candidates),
                                "accepted": decision is not None,
                                "reason": reason,
                            }
                            startup_confirmation["attempts"].append(record)
                            if decision is None:
                                continue
                            initialize_started = time.perf_counter()
                            if not provider.initialize_from_bbox(
                                confirmation_frame, decision.bbox_xyxy
                            ):
                                raise RuntimeError(
                                    f"track {spec.name} accepted startup "
                                    "confirmation failed atomic SAM2 initialization"
                                )
                            record["initialize_ms"] = (
                                time.perf_counter() - initialize_started
                            ) * 1000.0
                            startup_confirmation["applied"] = True
                            startup_confirmation["selected_index"] = (
                                confirmation_index
                            )
                            startup_confirmation["selected_bbox_xyxy"] = (
                                decision.bbox_xyxy.astype(int).tolist()
                            )
                            break
                    phase = spec.reference_indices[0] % 3
                    first = last_processed_index + 1
                    first += (phase - first) % 3
                    for index in range(first, max(spec.reference_indices) + 1, 3):
                        camera.seek(index)
                        started = time.perf_counter()
                        frame, mask_result = provider.step_mask_only()
                        elapsed_ms = (time.perf_counter() - started) * 1000.0
                        if int(frame.frame_id) != int(case.records[index]["frame_id"]):
                            raise RuntimeError("provider returned the wrong recorded frame")
                        row = _evaluate_result(
                            provider=provider,
                            spec=spec,
                            index=index,
                            frame=frame,
                            mask_result=mask_result,
                            projector=projector,
                            provider_ms=elapsed_ms,
                            output_mask_dir=mask_dir,
                            overlay=overlay,
                        )
                        if row is not None:
                            rows.append(row)
                finally:
                    overlay.release()
            all_rows.extend(rows)
            summaries[spec.name] = _track_summary(
                spec,
                rows,
                detected_index,
                attempts,
                int(args.max_entry_delay_frames),
            )
            summaries[spec.name]["boundary_entry_refinement"] = (
                boundary_refinement
            )
            summaries[spec.name]["startup_semantic_confirmation"] = (
                startup_confirmation
            )
    finally:
        try:
            provider.stop()
        finally:
            manager.close()

    visible_coverage = [value["reviewed_visible_coverage"] for value in summaries.values()]
    invalid_runs = [value["longest_visible_invalid_run"] for value in summaries.values()]
    absent_fp = [
        row["mask_valid"] and row["mask_area_px"] > 0
        for row in all_rows
        if not row["reference_visible"]
    ]
    visible_rows = [row for row in all_rows if row["reference_visible"]]
    usable_policy128 = [
        row["policy128_status"] in {"fresh", "motion_compensated"}
        for row in visible_rows
    ]
    cadence = [
        float(row["provider_plus_projector_ms"])
        for row in all_rows
        if row["provider_plus_projector_ms"] is not None
    ]
    gates = {
        "grounding_entry": all(
            value["grounding_detected_within_deadline"] for value in summaries.values()
        ),
        "visible_coverage": bool(visible_coverage)
        and float(np.mean([row["mask_valid"] and row["mask_area_px"] > 0 for row in visible_rows]))
        >= float(args.minimum_visible_coverage),
        "visible_invalid_run": bool(invalid_runs)
        and max(invalid_runs) <= int(args.maximum_visible_invalid_run),
        "policy128_usable_coverage": bool(usable_policy128)
        and float(np.mean(usable_policy128))
        >= float(args.minimum_visible_coverage),
        "policy128_unusable_run": bool(usable_policy128)
        and _longest_false_run(usable_policy128)
        <= int(args.maximum_visible_invalid_run),
        "absent_false_positive": (
            (0.0 if not absent_fp else float(np.mean(absent_fp)))
            <= float(args.maximum_absent_fp_fraction)
        ),
        "iou_p05": (_percentile([row["iou"] for row in visible_rows], 5) or 0.0)
        >= float(args.minimum_iou_p05),
        "recall_p05": (_percentile([row["recall"] for row in visible_rows], 5) or 0.0)
        >= float(args.minimum_recall_p05),
        "precision_p05": (
            _percentile([row["precision"] for row in visible_rows], 5) or 0.0
        )
        >= float(args.minimum_precision_p05),
        "cadence_p95": (_percentile(cadence, 95) or float("inf")) <= 50.0,
        "cadence_max": (_percentile(cadence, 100) or float("inf")) <= 100.0,
        "cadence_over_50ms_fraction": (
            0.0 if not cadence else float(np.mean(np.asarray(cadence) > 50.0))
        )
        <= 0.01,
    }
    calibration_path = Path(str(cfg["extrinsics"]["calibration_file"])).resolve()
    summary = {
        "schema": "thrown_object_end_to_end_replay_v1",
        "source_case": str(case.path),
        "source_manifest_sha256": _sha256(case.path / "manifest.json"),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "calibration": str(calibration_path),
        "calibration_sha256": _sha256(calibration_path),
        "camera_serial": str(case.manifest.get("camera_serial")),
        "camera_profile": f"{case.width}x{case.height}@{case.manifest['nominal_fps']}Hz",
        "object_text": str(args.object_text),
        "requested_object_mask_mode": "guarded_v2",
        "effective_mask_publication_mode": str(
            cfg["online_sam2"]["mask_publication_mode"]
        ),
        "effective_recovery_publication_mode": str(
            cfg["tracker"]["recovery_publication_mode"]
        ),
        "hardware_interfaces_opened": False,
        "franka_opened": False,
        "rh56_opened": False,
        "robot_motion": False,
        "policy_cadence_hz": 20,
        "prompt_health": prompt_health,
        "sam2_health": sam2_health,
        "tracks": summaries,
        "aggregate": {
            "reviewed_visible_frames": len(visible_rows),
            "reviewed_visible_coverage": (
                0.0
                if not visible_rows
                else float(
                    np.mean(
                        [row["mask_valid"] and row["mask_area_px"] > 0 for row in visible_rows]
                    )
                )
            ),
            "reviewed_absent_frames": len(absent_fp),
            "reviewed_absent_false_positive_fraction": (
                0.0 if not absent_fp else float(np.mean(absent_fp))
            ),
            "iou_p05": _percentile([row["iou"] for row in visible_rows], 5),
            "recall_p05": _percentile([row["recall"] for row in visible_rows], 5),
            "precision_p05": _percentile(
                [row["precision"] for row in visible_rows], 5
            ),
            "fresh_policy128_visible_fraction": (
                0.0
                if not visible_rows
                else float(
                    np.mean(
                        [row["policy128_status"] == "fresh" for row in visible_rows]
                    )
                )
            ),
            "usable_policy128_visible_fraction": (
                0.0
                if not visible_rows
                else float(np.mean(usable_policy128))
            ),
            "longest_visible_unusable_policy128_run": (
                _longest_false_run(usable_policy128)
            ),
            "provider_plus_projector_ms_p95": _percentile(cadence, 95),
            "provider_plus_projector_ms_max": _percentile(cadence, 100),
            "provider_plus_projector_over_50ms_fraction": (
                0.0 if not cadence else float(np.mean(np.asarray(cadence) > 50.0))
            ),
        },
        "gates": gates,
        "mask_and_latency_accepted": all(gates.values()),
        "pointcloud_acceptance_eligible": False,
        "pointcloud_acceptance_reason": (
            "catch workspace and 424x240@60 physical robot_base calibration "
            "are still provisional; measured and bounded motion-compensated "
            "128-point fractions are diagnostic only"
        ),
        "production_accepted": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(all_rows[0]) if all_rows else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["mask_and_latency_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
