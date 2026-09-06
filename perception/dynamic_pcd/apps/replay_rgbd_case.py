#!/usr/bin/env python3
"""Replay one lossless RGB-D case through the current mask/PCD pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from dynamic_pcd.camera.recorded_rgbd_camera import (
    EndOfRecording,
    RecordedRGBDCase,
    RecordedRGBDCamera,
)
from dynamic_pcd.config import load_config
from dynamic_pcd.provider.object_pcd_provider import ObjectPCDProvider
from dynamic_pcd.utils.geometry import draw_mask_overlay


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
DEFAULT_CONFIG = ROOT / "configs" / "d435_default.yaml"
TIMING_FIELDS = ("camera", "tracker", "sam2", "sam2_reinit", "mask_gate", "total")
OBJECT_MASK_MODES = ("guarded", "guarded_v2", "guarded_v1", "legacy")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline manual-ROI/SAM2/mask/object-PCD replay. The source is a "
            "lossless camera-only case; no hardware interface is opened."
        )
    )
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--disable-online-sam2",
        action="store_true",
        help="A/B only: disable the online SAM2 temporal service",
    )
    parser.add_argument(
        "--object-mask-mode",
        choices=OBJECT_MASK_MODES,
        default="guarded",
        help=(
            "replay an explicit mask path: guarded/guarded_v2 use production "
            "guarded_sam2_primary plus unified recovery; guarded_v1 retains "
            "adaptive_fusion/double confirmation; legacy uses direct semantic "
            "SAM2"
        ),
    )
    parser.add_argument(
        "--as-fast-as-possible",
        action="store_true",
        help="do not preserve recorded camera intervals",
    )
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--initialization-index",
        type=int,
        help=(
            "recording index of an exact-frame automatic-grounding bbox; "
            "requires --initialization-bbox"
        ),
    )
    parser.add_argument(
        "--initialization-bbox",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="exact-frame bbox produced by automatic grounding",
    )
    parser.add_argument("--vis", action="store_true")
    return parser


def _configure_object_mask_mode(
    cfg: Dict[str, Any],
    *,
    requested_mode: str,
    disable_online_sam2: bool,
) -> tuple[str, str]:
    """Apply one explicit replay mode and return its effective provenance."""

    mask_mode = str(requested_mode).strip().lower()
    if mask_mode not in OBJECT_MASK_MODES:
        raise ValueError(
            "object_mask_mode must be one of "
            f"{OBJECT_MASK_MODES}, got {mask_mode!r}"
        )
    online_cfg = cfg["online_sam2"]
    tracker_cfg = cfg["tracker"]
    if disable_online_sam2:
        if mask_mode == "legacy":
            raise ValueError(
                "--disable-online-sam2 is incompatible with legacy mask "
                "publication"
            )
        online_cfg["enabled"] = False
        online_cfg["require_for_bbox_init"] = False
        online_cfg["mask_publication_mode"] = "adaptive_fusion"
        tracker_cfg["recovery_publication_mode"] = (
            "legacy_double_confirm"
            if mask_mode == "guarded_v1"
            else "unified_three_evidence"
        )
        return "adaptive_only", str(
            tracker_cfg["recovery_publication_mode"]
        )

    online_cfg["enabled"] = True
    online_cfg["require_for_bbox_init"] = True
    if mask_mode == "legacy":
        online_cfg["mask_publication_mode"] = "semantic_sam2"
        return "legacy", "semantic_sam2_direct"
    tracker_cfg["recovery_publication_mode"] = (
        "legacy_double_confirm"
        if mask_mode == "guarded_v1"
        else "unified_three_evidence"
    )
    if mask_mode in ("guarded", "guarded_v2"):
        online_cfg["guarded_v2_semantic_primary"] = True
        online_cfg["mask_publication_mode"] = "guarded_sam2_primary"
        return "guarded_v2", "unified_three_evidence"
    online_cfg["mask_publication_mode"] = "adaptive_fusion"
    return "guarded_v1", "legacy_double_confirm"


def _stack(records: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    if not records:
        raise RuntimeError("replay produced no result frame")
    fields = set(records[0])
    for index, record in enumerate(records):
        if set(record) != fields:
            raise RuntimeError(f"replay record {index} has inconsistent fields")
    payload = {}
    for name in sorted(fields):
        try:
            payload[name] = np.stack(
                [np.asarray(record[name]) for record in records]
            )
        except ValueError as exc:
            raise RuntimeError(f"cannot stack replay field {name}") from exc
        if payload[name].dtype == object:
            raise RuntimeError(f"replay field {name} has object dtype")
    return payload


def _policy_projector(case: RecordedRGBDCase, cfg: Dict[str, Any]):
    if str(WORKSPACE) not in sys.path:
        sys.path.insert(0, str(WORKSPACE))
    try:
        from sim2real.observation.model import MaskedRGBDProjector
    except ImportError as exc:
        raise RuntimeError(
            "sim2real/observation/model.py is required for the formal 128-point "
            "adapter included in the handoff package"
        ) from exc
    depth_range = (
        float(cfg["camera"].get("z_min", 0.25)),
        float(cfg["camera"].get("z_max", 1.20)),
    )
    return MaskedRGBDProjector(
        camera_K=np.asarray(case.manifest["camera_K"], dtype=np.float64),
        T_base_camera_optical=np.asarray(
            case.manifest["T_base_camera"], dtype=np.float64
        ),
        image_size=(case.width, case.height),
        depth_range_m=depth_range,
        num_points=128,
        minimum_valid_points=16,
    )


def replay(args: argparse.Namespace) -> Path:
    # Keep the callable API compatible with older programmatic callers that
    # construct ``argparse.Namespace`` directly.  The CLI parser always
    # supplies these optional automatic-grounding fields, but their absence
    # must mean the documented manual-ROI path rather than an AttributeError.
    initialization_index = getattr(args, "initialization_index", None)
    initialization_bbox = getattr(args, "initialization_bbox", None)
    case = RecordedRGBDCase(args.case_dir)
    config_path = args.config.expanduser().resolve()
    cfg = load_config(str(config_path))
    active_config_sha256 = _sha256(config_path)
    capture_config_sha256 = str(
        case.manifest.get("source_config_sha256") or ""
    )
    config_matches_capture = bool(
        capture_config_sha256
        and active_config_sha256 == capture_config_sha256
    )
    if not config_matches_capture:
        print(
            "[Replay][WARN] active config differs from capture config; "
            "this is expected for an optimization A/B run, but it is not the "
            "original baseline configuration",
            file=sys.stderr,
            flush=True,
        )
    mask_mode = str(
        getattr(args, "object_mask_mode", "guarded")
    ).strip().lower()
    (
        effective_object_mask_mode,
        effective_recovery_publication_mode,
    ) = _configure_object_mask_mode(
        cfg,
        requested_mode=mask_mode,
        disable_online_sam2=bool(args.disable_online_sam2),
    )
    if args.max_frames is not None and int(args.max_frames) <= 0:
        raise ValueError("--max-frames must be positive")
    if (initialization_index is None) != (initialization_bbox is None):
        raise ValueError(
            "--initialization-index and --initialization-bbox must be used together"
        )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    mask_dir = output / "policy_mask"
    mask_dir.mkdir()

    camera = RecordedRGBDCamera(
        case,
        realtime=not bool(args.as_fast_as_possible),
        rate=float(args.rate),
    )
    provider = ObjectPCDProvider(cfg, camera=camera)
    projector = _policy_projector(case, cfg)
    records: List[Dict[str, Any]] = []
    overlay = None
    try:
        provider.start()
        if initialization_index is None:
            first = case.initialization_frame()
            stored_roi = case.manifest.get("provider_initialization_roi_xyxy")
            if stored_roi is None:
                raise RuntimeError(
                    "recording has no manual ROI; provide an exact automatic "
                    "--initialization-index and --initialization-bbox"
                )
            roi = np.asarray(stored_roi, dtype=np.int32)
            initialization_index = None
            initialization_source = "recorded_manual_roi"
        else:
            initialization_index = int(initialization_index)
            if not 0 <= initialization_index < len(case):
                raise ValueError("--initialization-index is outside the recording")
            first = case.frame(initialization_index)
            roi = np.asarray(initialization_bbox, dtype=np.int32)
            camera.seek(initialization_index + 1)
            initialization_source = "exact_frame_automatic_grounding_bbox"
        if not provider.initialize_from_bbox(first, roi):
            raise RuntimeError("recorded ROI failed to initialize the provider")
        camera.rebase_timing()
        overlay = cv2.VideoWriter(
            str(output / "overlay.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(case.manifest["nominal_fps"]),
            (case.width, case.height),
        )
        if not overlay.isOpened():
            raise RuntimeError("cannot create replay overlay.mp4")
        while args.max_frames is None or len(records) < int(args.max_frames):
            try:
                frame, mask_result = provider.step_mask_only()
            except EndOfRecording:
                break
            policy_mask = (
                np.asarray(mask_result.mask, dtype=bool)
                if bool(mask_result.valid)
                else np.zeros((case.height, case.width), dtype=bool)
            )
            point_frame = projector.project(
                color_bgr=frame.color_bgr,
                depth_raw=frame.depth_raw,
                depth_scale_m_per_unit=frame.depth_scale,
                object_mask=policy_mask,
                # Identity makes the returned XYZ coordinates robot_base. The
                # selected pixels/RGB/valid/stale behavior remain identical to
                # the formal adapter; a camera-only recording has no palm pose.
                T_base_palm_at_capture=np.eye(4, dtype=np.float64),
                captured_at_s=frame.timestamp,
                frame_id=frame.frame_id,
            )
            mask_path = mask_dir / f"{len(records):06d}.png"
            if not cv2.imwrite(
                str(mask_path), policy_mask.astype(np.uint8) * np.uint8(255)
            ):
                raise RuntimeError(f"cannot write {mask_path}")
            overlay_image = draw_mask_overlay(
                frame.color_bgr,
                policy_mask.astype(np.uint8),
                np.asarray(mask_result.bbox_xyxy, dtype=np.int32),
                text=(
                    f"valid={bool(mask_result.valid)} "
                    f"policy128={point_frame.status} "
                    f"source={point_frame.source_valid_points}"
                ),
            )
            overlay.write(overlay_image)
            if args.vis:
                cv2.imshow("offline object-PCD replay", overlay_image)
                if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                    break
            projector_provenance = (
                projector.last_effective_object_mask_provenance
            )
            projector_bbox = (
                np.zeros(4, dtype=np.int32)
                if projector_provenance.bbox_xyxy is None
                else np.asarray(
                    projector_provenance.bbox_xyxy, dtype=np.int32
                )
            )
            records.append(
                {
                    "frame_id": np.int64(frame.frame_id),
                    "camera_timestamp_s": np.float64(frame.timestamp),
                    "provider_published_mask_valid": np.bool_(mask_result.valid),
                    "mask_valid": np.bool_(mask_result.valid),
                    "provider_published_mask_area_px": np.int32(
                        np.count_nonzero(policy_mask)
                    ),
                    "provider_published_mask_bbox_xyxy": np.asarray(
                        mask_result.bbox_xyxy, dtype=np.int32
                    ).reshape(4),
                    "provider_published_mask_source": np.asarray(
                        str(provider.published_mask_source)
                    ),
                    "provider_published_mask_message": np.asarray(
                        str(mask_result.message)
                    ),
                    "provider_online_sam2_status": np.asarray(
                        str(provider.online_sam2_status)
                    ),
                    "projector_effective_policy_mask_provenance": np.asarray(
                        str(projector_provenance.kind)
                    ),
                    "projector_effective_policy_mask_source_frame_id": np.int64(
                        -1
                        if projector_provenance.source_frame_id is None
                        else projector_provenance.source_frame_id
                    ),
                    "projector_effective_policy_mask_area_px": np.int32(
                        projector_provenance.area_px
                    ),
                    "projector_effective_policy_mask_bbox_valid": np.bool_(
                        projector_provenance.bbox_xyxy is not None
                    ),
                    "projector_effective_policy_mask_bbox_xyxy": projector_bbox,
                    "policy128_xyzrgb_base": point_frame.xyzrgb_palm.astype(
                        np.float32, copy=True
                    ),
                    "policy128_valid": point_frame.valid.astype(
                        np.float32, copy=True
                    ),
                    "policy128_status": np.asarray(point_frame.status),
                    "policy128_source_valid_points": np.int32(
                        point_frame.source_valid_points
                    ),
                    "provider_timings_ms": np.asarray(
                        [
                            float(provider.last_timings_ms.get(name, 0.0))
                            for name in TIMING_FIELDS
                        ],
                        dtype=np.float64,
                    ),
                }
            )
    finally:
        try:
            provider.stop()
        finally:
            if overlay is not None:
                overlay.release()
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    payload = _stack(records)
    payload.update(
        {
            "schema": np.asarray("dynamic_object_pcd_replay_v2"),
            "source_case": np.asarray(str(case.path)),
            "source_case_schema": np.asarray(str(case.manifest["schema"])),
            "case_name": np.asarray(str(case.manifest["case"])),
            "config_path": np.asarray(str(config_path)),
            "requested_object_mask_mode": np.asarray(mask_mode),
            "object_mask_mode": np.asarray(mask_mode),
            "effective_object_mask_mode": np.asarray(
                effective_object_mask_mode
            ),
            "effective_provider_mask_publication_mode": np.asarray(
                str(
                    cfg.get("online_sam2", {}).get(
                        "mask_publication_mode", ""
                    )
                )
            ),
            "provider_mask_publication_mode": np.asarray(
                str(
                    cfg.get("online_sam2", {}).get(
                        "mask_publication_mode", ""
                    )
                )
            ),
            "effective_provider_recovery_publication_mode": np.asarray(
                effective_recovery_publication_mode
            ),
            "recovery_publication_mode": np.asarray(
                str(
                    cfg.get("tracker", {}).get(
                        "recovery_publication_mode", ""
                    )
                )
            ),
            "active_config_sha256": np.asarray(active_config_sha256),
            "capture_config_sha256": np.asarray(capture_config_sha256),
            "config_matches_capture": np.bool_(config_matches_capture),
            "online_sam2_enabled": np.bool_(
                cfg.get("online_sam2", {}).get("enabled", False)
            ),
            "policy128_coordinate_frame": np.asarray("robot_base"),
            "policy128_semantics": np.asarray(
                "formal MaskedRGBDProjector pixels/RGB/valid/fallback; "
                "identity T_base_palm because recording is camera-only"
            ),
            "T_base_camera": np.asarray(
                case.manifest["T_base_camera"], dtype=np.float64
            ),
            "camera_K": np.asarray(case.manifest["camera_K"], dtype=np.float64),
            "depth_scale_m_per_unit": np.float64(case.depth_scale),
            "timing_fields": np.asarray(TIMING_FIELDS),
            "hardware_interfaces_opened": np.bool_(False),
            "initialization_source": np.asarray(initialization_source),
            "initialization_recording_index": np.int64(
                -1 if initialization_index is None else initialization_index
            ),
            "initialization_frame_id": np.int64(first.frame_id),
            "initialization_bbox_xyxy": np.asarray(roi, dtype=np.int32),
        }
    )
    np.savez_compressed(output / "results.npz", **payload)
    summary = {
        "schema": "dynamic_object_pcd_replay_summary_v2",
        "case": str(case.manifest["case"]),
        "frames": len(records),
        "initialization_frame_evaluated": False,
        "evaluated_all_recorded_video_frames": len(records) == len(case),
        "provider_published_mask_valid_fraction": float(
            np.mean(payload["provider_published_mask_valid"])
        ),
        "mask_valid_fraction": float(
            np.mean(payload["provider_published_mask_valid"])
        ),
        "fresh_policy128_fraction": float(
            np.mean(payload["policy128_status"] == "fresh")
        ),
        "stale_policy128_fraction": float(
            np.mean(payload["policy128_status"] == "stale_palm")
        ),
        "online_sam2_enabled": bool(
            cfg.get("online_sam2", {}).get("enabled", False)
        ),
        "requested_object_mask_mode": mask_mode,
        "object_mask_mode": mask_mode,
        "effective_object_mask_mode": effective_object_mask_mode,
        "effective_provider_mask_publication_mode": str(
            cfg.get("online_sam2", {}).get("mask_publication_mode", "")
        ),
        "provider_mask_publication_mode": str(
            cfg.get("online_sam2", {}).get("mask_publication_mode", "")
        ),
        "effective_provider_recovery_publication_mode": str(
            effective_recovery_publication_mode
        ),
        "recovery_publication_mode": str(
            cfg.get("tracker", {}).get("recovery_publication_mode", "")
        ),
        "config_matches_capture": config_matches_capture,
        "hardware_interfaces_opened": False,
        "initialization_source": initialization_source,
        "initialization_recording_index": initialization_index,
        "initialization_frame_id": int(first.frame_id),
        "initialization_bbox_xyxy": np.asarray(roi, dtype=np.int32).tolist(),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        replay(args)
        return 0
    except KeyboardInterrupt:
        return 130
    except (EOFError, ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"[replay failed] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
