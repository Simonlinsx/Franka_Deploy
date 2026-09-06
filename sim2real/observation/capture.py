#!/usr/bin/env python3
"""Capture deployment-exact RGB/depth/masks without opening robot interfaces.

The camera/tracker runs continuously and all selected frames are copied into a
bounded in-memory list.  The NPZ is written only after the provider has stopped,
so compression and disk I/O cannot perturb the acquisition timing path.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    workspace = Path(__file__).resolve().parents[2]
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    from sim2real.deployment.bundle import DeployBundle
    from sim2real.observation.live_preview import (
        _ComputeThreadGuard,
        _initialize_provider,
    )
    from sim2real.contracts.v94 import V94Contract
    from sim2real.deployment.verify import verify_v94_bundle
else:
    from sim2real.deployment.bundle import DeployBundle
    from .live_preview import _ComputeThreadGuard, _initialize_provider
    from sim2real.contracts.v94 import V94Contract
    from sim2real.deployment.verify import verify_v94_bundle


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
from sim2real.workspace_paths import DEFAULT_V94_DEPLOY_BUNDLE

DEFAULT_BUNDLE = DEFAULT_V94_DEPLOY_BUNDLE
DEFAULT_PCD_CONFIG = (
    WORKSPACE_ROOT / "perception" / "configs" / "d435_default.yaml"
)
TIMING_FIELDS: Tuple[str, ...] = (
    "camera",
    "tracker",
    "sam2",
    "sam2_reinit",
    "mask_gate",
    "total",
)
MAX_STORED_FRAMES = 300
MAX_CONSECUTIVE_RETRYABLE_CAMERA_TIMEOUTS = 3
RETRYABLE_CAMERA_TIMEOUT_BACKOFF_S = 0.001


def _initialized_roi_evidence(provider: Any) -> Dict[str, np.ndarray]:
    """Return the actual prompt/mask provenance retained by the provider.

    Interactive ``selectROI`` runs do not have a CLI ``--roi`` value.  The
    provider evidence is therefore the authoritative source for the numeric
    prompt that initialized the tracker and must be written into the capture
    artifact instead of the old ``[-1, -1, -1, -1]`` placeholder.
    """

    evidence = getattr(provider, "last_bbox_initialization_evidence", None)
    if evidence is None:
        raise RuntimeError(
            "initialized provider retained no numeric ROI/mask evidence"
        )
    prompt = np.asarray(
        getattr(evidence, "prompt_bbox_xyxy", None)
    )
    mask_bbox = np.asarray(
        getattr(evidence, "mask_bbox_xyxy", None)
    )
    if (
        prompt.shape != (4,)
        or mask_bbox.shape != (4,)
        or not np.issubdtype(prompt.dtype, np.integer)
        or not np.issubdtype(mask_bbox.dtype, np.integer)
    ):
        raise RuntimeError(
            "initialized provider ROI/mask evidence must be integer xyxy[4]"
        )
    x1, y1, x2, y2 = (int(value) for value in prompt.tolist())
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise RuntimeError("initialized provider retained an invalid prompt ROI")
    source = str(getattr(evidence, "source", "")).strip()
    if not source:
        raise RuntimeError("initialized provider retained no mask source")
    return {
        "roi_xywh": np.asarray(
            [x1, y1, x2 - x1, y2 - y1], dtype=np.int32
        ),
        "initialization_prompt_bbox_xyxy": prompt.astype(
            np.int32, copy=True
        ),
        "initialization_mask_bbox_xyxy": mask_bbox.astype(
            np.int32, copy=True
        ),
        "initialization_mask_area_px": np.asarray(
            int(getattr(evidence, "mask_area_px", 0)), dtype=np.int32
        ),
        "initialization_mask_source": np.asarray(source),
    }


def _stack(records: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    if not records:
        raise RuntimeError("capture produced no valid stored RGB-D/mask frames")
    names = set(records[0])
    result: Dict[str, np.ndarray] = {}
    for index, record in enumerate(records):
        if set(record) != names:
            raise ValueError(f"capture record {index} has inconsistent fields")
    for name in sorted(names):
        try:
            value = np.stack([np.asarray(record[name]) for record in records])
        except ValueError as exc:
            raise ValueError(f"capture field {name} has inconsistent shapes") from exc
        if value.dtype == object:
            raise ValueError(f"capture field {name} cannot use object dtype")
        result[name] = value
    return result


def capture_valid_frames(
    provider: Any,
    *,
    stored_frames: int,
    sample_every_valid_frame: int = 1,
    maximum_attempts: Optional[int] = None,
    require_consecutive_valid_frames: bool = False,
    maximum_consecutive_retryable_camera_timeouts: int = (
        MAX_CONSECUTIVE_RETRYABLE_CAMERA_TIMEOUTS
    ),
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Collect a bounded, uniformly thinned sequence from ``step_mask_only``.

    This function performs no disk I/O.  Invalid provider publications remain
    counted but are never saved as if they were usable object observations.
    """

    wanted = int(stored_frames)
    stride = int(sample_every_valid_frame)
    if wanted <= 0 or wanted > MAX_STORED_FRAMES:
        raise ValueError(f"stored_frames must be in [1,{MAX_STORED_FRAMES}]")
    if stride <= 0:
        raise ValueError("sample_every_valid_frame must be positive")
    require_consecutive = bool(require_consecutive_valid_frames)
    if require_consecutive and stride != 1:
        raise ValueError(
            "consecutive valid-frame capture requires sample_every_valid_frame=1"
        )
    attempt_limit = (
        max(wanted * stride * 10, wanted + 30)
        if maximum_attempts is None
        else int(maximum_attempts)
    )
    if attempt_limit < wanted:
        raise ValueError("maximum_attempts cannot be smaller than stored_frames")
    retryable_timeout_limit = int(
        maximum_consecutive_retryable_camera_timeouts
    )
    if retryable_timeout_limit < 0:
        raise ValueError(
            "maximum_consecutive_retryable_camera_timeouts cannot be negative"
        )

    records: List[Dict[str, Any]] = []
    attempts = 0
    valid_publications = 0
    invalid_publications = 0
    retryable_camera_timeouts = 0
    consecutive_retryable_camera_timeouts = 0
    maximum_consecutive_retryable_camera_timeouts_observed = 0
    invalid_reasons: Counter[str] = Counter()
    previous_frame_id: Optional[int] = None
    while len(records) < wanted and attempts < attempt_limit:
        try:
            frame, mask_result = provider.step_mask_only()
        except BaseException as exc:
            # The RealSense owner marks only bounded SDK/frame-pair holes as
            # retryable.  Such a call produced no RGB-D publication, so it
            # must not spend a provider-step/mask budget.  A short burst is
            # tolerated here just as it is in the formal camera worker, while
            # sustained stream loss still fails before robot execution.
            if not bool(getattr(exc, "retryable_camera_timeout", False)):
                raise
            retryable_camera_timeouts += 1
            consecutive_retryable_camera_timeouts += 1
            maximum_consecutive_retryable_camera_timeouts_observed = max(
                maximum_consecutive_retryable_camera_timeouts_observed,
                consecutive_retryable_camera_timeouts,
            )
            if require_consecutive:
                # A capture hole breaks the proof that the stored frames form
                # one clean terminal sequence, even though it is recoverable.
                records.clear()
            if (
                consecutive_retryable_camera_timeouts
                > retryable_timeout_limit
            ):
                raise RuntimeError(
                    "capture exceeded "
                    f"{retryable_timeout_limit} consecutive retryable camera "
                    f"timeouts; total={retryable_camera_timeouts}; "
                    f"last={type(exc).__name__}: {exc}"
                ) from exc
            time.sleep(RETRYABLE_CAMERA_TIMEOUT_BACKOFF_S)
            continue
        consecutive_retryable_camera_timeouts = 0
        attempts += 1
        if previous_frame_id is not None and int(frame.frame_id) <= previous_frame_id:
            raise RuntimeError(
                "provider frame_id did not strictly increase during raw capture: "
                f"{int(frame.frame_id)} after {previous_frame_id}"
            )
        previous_frame_id = int(frame.frame_id)
        if not bool(mask_result.valid):
            invalid_publications += 1
            invalid_reasons[str(mask_result.message)] += 1
            if require_consecutive:
                # Preflight asks for a clean terminal streak, not merely N
                # valid masks with rejected publications between them.
                records.clear()
            continue
        sample_index = valid_publications
        valid_publications += 1
        if sample_index % stride != 0:
            continue

        color = np.asarray(frame.color_bgr)
        depth_raw = np.asarray(frame.depth_raw)
        mask = np.asarray(mask_result.mask, dtype=bool)
        if color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3:
            raise RuntimeError("provider color frame is not uint8 BGR")
        if depth_raw.dtype != np.uint16 or depth_raw.shape != color.shape[:2]:
            raise RuntimeError("provider depth_raw is not aligned uint16 depth")
        if mask.shape != depth_raw.shape:
            raise RuntimeError("provider mask is not aligned to RGB-D")
        if not np.isfinite(float(frame.timestamp)):
            raise RuntimeError("provider capture timestamp is not finite")
        retrieved = float(
            frame.timestamp if frame.retrieved_at_s is None else frame.retrieved_at_s
        )
        timings = np.asarray(
            [float(provider.last_timings_ms.get(name, 0.0)) for name in TIMING_FIELDS],
            dtype=np.float64,
        )
        records.append(
            {
                "rgb": color.copy(),
                "depth_raw": depth_raw.copy(),
                "object_mask": mask.copy(),
                "frame_camera_K": frame.intrinsics.as_matrix().astype(np.float64),
                "frame_camera_distortion": np.pad(
                    np.asarray(frame.intrinsics.distortion, dtype=np.float64),
                    (0, max(0, 5 - len(frame.intrinsics.distortion))),
                )[:5],
                "frame_depth_scale_m_per_unit": np.float64(frame.depth_scale),
                "camera_timestamp_s": np.float64(frame.timestamp),
                "camera_retrieved_at_s": np.float64(retrieved),
                "camera_frame_id": np.int64(frame.frame_id),
                "camera_sensor_frame_number": np.int64(frame.sensor_frame_number),
                "camera_depth_sensor_frame_number": np.int64(
                    frame.depth_sensor_frame_number
                ),
                "camera_timestamp_domain": np.asarray(str(frame.timestamp_domain)),
                "camera_provider_timings_ms": timings,
                "object_mask_area_px": np.int32(np.count_nonzero(mask)),
                "object_mask_bbox_xyxy": np.asarray(
                    mask_result.bbox_xyxy, dtype=np.int32
                ).reshape(4),
                "object_mask_message": np.asarray(str(mask_result.message)),
                "provider_published_mask_source": np.asarray(
                    str(
                        getattr(
                            provider,
                            "published_mask_source",
                            getattr(
                                provider,
                                "_last_output_mask_source",
                                "unknown",
                            ),
                        )
                    )
                ),
                "provider_published_mask_message": np.asarray(
                    str(mask_result.message)
                ),
                "provider_online_sam2_status": np.asarray(
                    str(
                        getattr(
                            provider,
                            "online_sam2_status",
                            getattr(
                                provider,
                                "_last_online_sam2_status",
                                "unknown",
                            ),
                        )
                    )
                ),
            }
        )
    if len(records) < wanted:
        raise RuntimeError(
            f"only captured {len(records)}/{wanted} stored valid frames after "
            f"{attempts} provider steps ({invalid_publications} invalid); "
            f"invalid_reasons={dict(invalid_reasons.most_common(8))}"
        )
    counters = {
        "provider_steps": attempts,
        "valid_publications": valid_publications,
        "invalid_publications": invalid_publications,
        "retryable_camera_timeouts": retryable_camera_timeouts,
        "maximum_consecutive_retryable_camera_timeouts": (
            maximum_consecutive_retryable_camera_timeouts_observed
        ),
        "stored_frames": len(records),
        "invalid_reasons": dict(invalid_reasons),
    }
    return _stack(records), counters


def _atomic_save_npz(
    path: Path, payload: Dict[str, np.ndarray], *, compressed: bool
) -> None:
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
        if compressed:
            np.savez_compressed(temporary, **payload)
        else:
            np.savez(temporary, **payload)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture D435 RGB/depth and the exact-frame provider mask in memory; "
            "never open Franka/RH56 and write the NPZ only after camera shutdown."
        )
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--pcd-config", type=Path, default=DEFAULT_PCD_CONFIG)
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H")
    )
    target.add_argument(
        "--object-text",
        metavar="TEXT",
        help=(
            "use the deployment camera-only text grounding path (for example "
            "'blue ball'); without --roi/--object-text, select a box manually"
        ),
    )
    parser.add_argument(
        "--object-mask-mode",
        choices=("guarded", "guarded_v2", "guarded_v1", "legacy"),
        default="guarded",
        help=(
            "use the same final mask publication mode as deployment "
            "(default: guarded)"
        ),
    )
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="store one of every N valid provider frames while tracking every frame",
    )
    parser.add_argument("--maximum-attempts", type=int)
    parser.add_argument(
        "--compute-threads",
        type=int,
        default=1,
        help="verified OpenCV/BLAS thread count during camera acquisition",
    )
    parser.add_argument("--disable-online-sam2", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--uncompressed",
        action="store_true",
        help="write a larger/faster NPZ after acquisition",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="reserved safety tripwire; this program always refuses it",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.execute:
        raise RuntimeError(
            "capture_v94_rgbd_mask is camera-only and has no robot execution path"
        )
    if int(args.frames) <= 0 or int(args.frames) > MAX_STORED_FRAMES:
        raise ValueError(f"--frames must be in [1,{MAX_STORED_FRAMES}]")
    if int(args.sample_every) <= 0:
        raise ValueError("--sample-every must be positive")
    if int(args.compute_threads) <= 0:
        raise ValueError("--compute-threads must be positive")

    # Complete bundle verification occurs before the camera is opened.
    verification = verify_v94_bundle(args.bundle)
    bundle = DeployBundle(args.bundle)
    contract = V94Contract.from_bundle(bundle)
    provider = None
    compute_thread_guard = None
    started = time.monotonic()
    acquisition_finished = started
    try:
        resolved_roi = args.roi
        object_text = (
            None if args.object_text is None else str(args.object_text).strip()
        )
        if args.object_text is not None and not object_text:
            raise ValueError("--object-text cannot be empty")
        if object_text is not None:
            # Reuse the deployment's camera-only text grounding implementation:
            # prompt detector -> SAM2 first-frame mask -> immutable numeric ROI.
            # It closes that camera/provider before the sequence capture reopens
            # the D435, exactly like formal deployment preflight.
            pointcloud_source = WORKSPACE_ROOT / "perception"
            if str(pointcloud_source) not in sys.path:
                sys.path.insert(0, str(pointcloud_source))
            from sim2real.observation.roi_selector import _select

            selection = _select(
                bundle_path=args.bundle,
                pcd_config_path=args.pcd_config,
                object_text=object_text,
            )
            resolved_roi = tuple(int(value) for value in selection["roi_xywh"])
        provider, online_selection = _initialize_provider(
            args.pcd_config.expanduser().resolve(),
            resolved_roi,
            disable_online_sam2=bool(args.disable_online_sam2),
            object_mask_mode=str(args.object_mask_mode),
        )
        initialization_evidence = _initialized_roi_evidence(provider)
        if resolved_roi is not None and not np.array_equal(
            initialization_evidence["roi_xywh"],
            np.asarray(resolved_roi, dtype=np.int32),
        ):
            raise RuntimeError(
                "provider clipped or changed the requested numeric ROI"
            )
        if provider.extrinsics.calibration_id != contract.calibration_id:
            raise RuntimeError("provider calibration ID differs from V94 contract")
        if provider.extrinsics.camera_serial != contract.camera_serial:
            raise RuntimeError("provider camera serial differs from V94 contract")
        if not np.allclose(
            np.asarray(provider.extrinsics.T_base_camera, dtype=np.float64),
            contract.T_base_camera_optical,
            atol=1.0e-7,
            rtol=0.0,
        ):
            raise RuntimeError("provider camera extrinsics differ from V94 contract")
        compute_thread_guard = _ComputeThreadGuard(
            int(args.compute_threads)
        ).start()
        payload, counters = capture_valid_frames(
            provider,
            stored_frames=int(args.frames),
            sample_every_valid_frame=int(args.sample_every),
            maximum_attempts=args.maximum_attempts,
        )
        acquisition_finished = time.monotonic()
    finally:
        try:
            if provider is not None:
                provider.stop()
        finally:
            if compute_thread_guard is not None:
                compute_thread_guard.close()

    first_K = np.asarray(payload["frame_camera_K"][0], dtype=np.float64)
    if not np.allclose(
        payload["frame_camera_K"], contract.camera_K[None], atol=1.0e-5, rtol=0.0
    ):
        raise RuntimeError("captured D435 intrinsics differ from V94 contract")
    if not np.allclose(payload["frame_camera_distortion"], 0.0, atol=1.0e-9):
        raise RuntimeError("captured D435 distortion differs from V94 contract")
    if not np.allclose(
        payload["frame_depth_scale_m_per_unit"],
        contract.depth_scale_m_per_unit,
        atol=1.0e-9,
        rtol=0.0,
    ):
        raise RuntimeError("captured D435 depth scale differs from V94 contract")
    if payload["rgb"].shape[1:] != (
        contract.camera_height,
        contract.camera_width,
        3,
    ):
        raise RuntimeError("captured image dimensions differ from V94 contract")

    payload.update(
        {
            "capture_schema_version": np.asarray(2, dtype=np.int32),
            "hardware_writes": np.asarray(False),
            "hardware_writes_semantics": np.asarray(
                "robot_actuator_or_register_commands_only"
            ),
            "robot_command_writes": np.asarray(False),
            "camera_configuration_writes": np.asarray(True),
            "franka_interface_opened": np.asarray(False),
            "rh56_interface_opened": np.asarray(False),
            "disk_io_during_acquisition": np.asarray(False),
            "color_channel_order": np.asarray("BGR"),
            "depth_storage": np.asarray("aligned_Z16"),
            "object_mask_semantics": np.asarray(
                "bool_HxW_true_is_policy_object"
            ),
            "camera_K": first_K,
            "depth_scale_m_per_unit": np.asarray(
                contract.depth_scale_m_per_unit, dtype=np.float64
            ),
            "depth_range_m": np.asarray(contract.depth_range_m, dtype=np.float64),
            "T_base_camera_optical": contract.T_base_camera_optical.astype(
                np.float64
            ),
            "camera_serial": np.asarray(contract.camera_serial),
            "calibration_id": np.asarray(contract.calibration_id),
            "bundle_contract": np.asarray(verification.bundle_contract),
            "checkpoint_sha256": np.asarray(
                verification.checkpoint_sha256
            ),
            "provider_timing_fields": np.asarray(TIMING_FIELDS),
            "sample_every_valid_frame": np.asarray(
                int(args.sample_every), dtype=np.int32
            ),
            "provider_steps": np.asarray(counters["provider_steps"], dtype=np.int32),
            "valid_publications": np.asarray(
                counters["valid_publications"], dtype=np.int32
            ),
            "invalid_publications": np.asarray(
                counters["invalid_publications"], dtype=np.int32
            ),
            "invalid_reason_names": np.asarray(
                sorted(counters["invalid_reasons"])
            ),
            "invalid_reason_counts": np.asarray(
                [
                    counters["invalid_reasons"][name]
                    for name in sorted(counters["invalid_reasons"])
                ],
                dtype=np.int32,
            ),
            "acquisition_elapsed_s": np.asarray(
                acquisition_finished - started, dtype=np.float64
            ),
            "online_sam2_runtime_mode": np.asarray(online_selection.mode),
            "object_text": np.asarray(object_text or ""),
            "requested_object_mask_mode": np.asarray(
                str(args.object_mask_mode)
            ),
            "effective_object_mask_mode": np.asarray(
                online_selection.effective_object_mask_mode
            ),
            "effective_mask_publication_mode": np.asarray(
                online_selection.effective_mask_publication_mode
            ),
            "effective_provider_mask_publication_mode": np.asarray(
                online_selection.effective_mask_publication_mode
            ),
            "effective_recovery_publication_mode": np.asarray(
                online_selection.effective_recovery_publication_mode
            ),
            "effective_provider_recovery_publication_mode": np.asarray(
                online_selection.effective_recovery_publication_mode
            ),
            "configured_compute_threads": np.asarray(
                int(args.compute_threads), dtype=np.int32
            ),
        }
    )
    payload.update(initialization_evidence)
    if args.output is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = WORKSPACE_ROOT / "dexgrasp" / "runs" / f"v94_raw_rgbd_mask_{stamp}.npz"
    else:
        output = args.output
    # This is intentionally the first output-file write after acquisition.
    _atomic_save_npz(output, payload, compressed=not bool(args.uncompressed))
    resolved = output.expanduser().resolve()
    print(
        "[camera-only capture] "
        f"saved={resolved} frames={int(args.frames)} "
        f"provider_steps={counters['provider_steps']} "
        f"invalid={counters['invalid_publications']} "
        f"elapsed_s={acquisition_finished - started:.3f} "
        "robot_writes=0 franka_opened=0 rh56_opened=0 "
        "disk_io_during_acquisition=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
