from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

from dynamic_pcd.utils.gui_env import (
    repair_gui_env_after_cv2_import,
    setup_gui_env,
)
setup_gui_env()
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import cv2
import numpy as np

repair_gui_env_after_cv2_import()

from dynamic_pcd.config import load_config
from dynamic_pcd.apps.interactive_hover import InteractiveHoverController
from dynamic_pcd.provider.object_pcd_provider import ObjectPCDProvider
from dynamic_pcd.ipc.zmq_pubsub import ZMQObjectPCDPublisher
from dynamic_pcd.segmentation.prompt_runtime import (
    AsyncPromptReacquirer,
    PromptRetrySchedule,
    PromptServiceManager,
)
from dynamic_pcd.segmentation.prompt_replay import (
    RecentRGBDFrameBuffer,
    replay_prompt_mask,
)
from dynamic_pcd.segmentation.sam2_video_backend import (
    ALLOWED_VOS_COMPILE_MODES,
    PRODUCTION_VOS_COMPILE_MODE,
)
from dynamic_pcd.utils.vis import Open3DLiveViewer, OpenCVPointCloudViewer, put_lines
from dynamic_pcd.utils.fps import DeadlineRateGate, FPSMeter, PacketRateMonitor


WIN = "Object Mask / RGB"

COMMAND_RESELECT = "reselect_target"
COMMAND_TOGGLE_LOCK = "toggle_roi_lock"
COMMAND_TOGGLE_PAUSE = "toggle_pause"
COMMAND_SAVE = "save_cloud"
COMMAND_HELP = "show_help"
COMMAND_QUIT = "quit"
COMMAND_PLAN_HOVER = "plan_hover"
COMMAND_ARM_MOTION = "arm_motion"
COMMAND_CONFIRM_MOTION = "confirm_motion"
COMMAND_CANCEL_MOTION = "cancel_motion"

# Open3D callbacks enqueue these commands; the perception thread executes them
# after returning from the viewer's event pump.
OPEN3D_COMMAND_BINDINGS = {
    "T": COMMAND_RESELECT,
    "L": COMMAND_TOGGLE_LOCK,
    "P": COMMAND_TOGGLE_PAUSE,
    "S": COMMAND_SAVE,
    "H": COMMAND_HELP,
    "D": COMMAND_PLAN_HOVER,
    "M": COMMAND_ARM_MOTION,
    "Y": COMMAND_CONFIRM_MOTION,
    "X": COMMAND_CANCEL_MOTION,
}


def command_from_cv_key(key: int):
    """Translate an OpenCV keycode into the shared interactive command set."""

    if key in (27, ord("q"), ord("Q")):
        return COMMAND_QUIT
    if key in (ord("r"), ord("R"), ord("t"), ord("T")):
        return COMMAND_RESELECT
    if key in (ord("l"), ord("L")):
        return COMMAND_TOGGLE_LOCK
    if key in (ord("p"), ord("P")):
        return COMMAND_TOGGLE_PAUSE
    if key in (ord("s"), ord("S")):
        return COMMAND_SAVE
    if key in (ord("h"), ord("H")):
        return COMMAND_HELP
    if key in (ord("d"), ord("D")):
        return COMMAND_PLAN_HOVER
    if key in (ord("m"), ord("M")):
        return COMMAND_ARM_MOTION
    if key in (ord("y"), ord("Y")):
        return COMMAND_CONFIRM_MOTION
    if key in (ord("x"), ord("X")):
        return COMMAND_CANCEL_MOTION
    return None


def print_key_help(prompt_mode: bool = False) -> None:
    target_action = (
        "T/r optional manual retry while LOST (automatic recovery is active)"
        if prompt_mode
        else "T/r reselect target"
    )
    print(
        f"[Keys] {target_action} | L lock/unlock ROI | P pause/resume | "
        "S save cloud | H help | q/ESC quit"
    )
    print(
        "[Robot keys] D dry-run plan | M arm one-shot hover | "
        "Y confirm armed motion | X cancel/disarm"
    )
    print("[Open3D view] R reset view | [ / ] point size | Q close only the 3D window")


def print_prompt_result(result, prefix: str = "[Prompt]") -> None:
    bbox = (
        None
        if result.bbox_xyxy is None
        else np.asarray(result.bbox_xyxy, dtype=np.int32).tolist()
    )
    timings = result.timings_ms or {}
    print(
        f"{prefix} valid={result.valid} score={result.score:.3f} "
        f"bbox={bbox} candidates={len(result.candidates)} "
        f"detector={timings.get('detector', 0.0):.1f}ms "
        f"segmenter={timings.get('segmenter', 0.0):.1f}ms | {result.message}"
    )
    for index, candidate in enumerate(result.candidates):
        candidate_bbox = np.asarray(
            candidate.bbox_xyxy, dtype=np.int32
        ).tolist()
        print(
            f"{prefix} candidate[{index}] label={candidate.label!r} "
            f"det={candidate.detector_score:.3f} rank={candidate.rank_score:.3f} "
            f"bbox={candidate_bbox} ref_iou={candidate.reference_iou} "
            f"mask_valid={candidate.mask_valid}"
        )


def select_yolo_world_entry_candidate(
    result: Any,
    image_bgr: np.ndarray,
    previous_gray: Optional[np.ndarray],
    config: Optional[Mapping[str, Any]] = None,
):
    """Select a compact exact-frame YOLO proposal linked to target entry.

    Open-vocabulary confidence alone is not sufficient for patterned beanbags:
    low-confidence proposals on fixed lab equipment can outrank the actual
    fast object.  This gate is deliberately applied only while discovering a
    new target.  SAM2 still owns the binary mask after the bbox is accepted.

    A candidate must have the expected compact image scale and either contain
    current-frame motion or have independently strong detector evidence.  The
    latter preserves acquisition of a clearly detected ball which is already
    stationary; weak static proposals remain fail-closed.
    """

    cfg = {} if config is None else dict(config)
    if not bool(cfg.get("entry_candidate_filter_enabled", True)):
        if not bool(getattr(result, "valid", False)):
            return None, "detector result is invalid"
        selected_index = getattr(result, "selected_index", None)
        for candidate in getattr(result, "candidates", ()):  # pragma: no branch
            if int(candidate.detector_index) == int(selected_index):
                return candidate, "entry candidate filter disabled"
        return None, "selected detector candidate is missing"

    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_bgr must be uint8 HxWx3")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prior = None if previous_gray is None else np.asarray(previous_gray)
    if prior is not None and (prior.dtype != np.uint8 or prior.shape != gray.shape):
        raise ValueError("previous_gray must match the current grayscale frame")

    short_side = float(min(gray.shape))
    min_extent = float(cfg.get("entry_candidate_min_extent_ratio", 0.03))
    max_extent = float(cfg.get("entry_candidate_max_extent_ratio", 0.15))
    max_aspect = float(cfg.get("entry_candidate_max_aspect_ratio", 2.5))
    difference_threshold = int(
        cfg.get("entry_candidate_motion_difference", 10)
    )
    min_motion_fraction = float(
        cfg.get("entry_candidate_min_motion_fraction", 0.15)
    )
    strong_static_score = float(
        cfg.get("entry_candidate_strong_static_score", 0.05)
    )
    motion = None if prior is None else cv2.absdiff(gray, prior)

    rejection_reasons = []
    for candidate in getattr(result, "candidates", ()):
        raw = np.asarray(candidate.bbox_xyxy, dtype=np.float64).reshape(4)
        if not np.all(np.isfinite(raw)):
            rejection_reasons.append("nonfinite_bbox")
            continue
        x1 = int(np.clip(np.floor(raw[0]), 0, gray.shape[1]))
        y1 = int(np.clip(np.floor(raw[1]), 0, gray.shape[0]))
        x2 = int(np.clip(np.ceil(raw[2]), 0, gray.shape[1]))
        y2 = int(np.clip(np.ceil(raw[3]), 0, gray.shape[0]))
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            rejection_reasons.append("empty_bbox")
            continue
        minimum = min(width, height) / short_side
        maximum = max(width, height) / short_side
        aspect = max(width / height, height / width)
        if minimum < min_extent:
            rejection_reasons.append("too_thin")
            continue
        if maximum > max_extent:
            rejection_reasons.append("too_large")
            continue
        if aspect > max_aspect:
            rejection_reasons.append("aspect")
            continue
        motion_fraction = 0.0
        if motion is not None:
            roi = motion[y1:y2, x1:x2]
            motion_fraction = float(np.mean(roi > difference_threshold))
        detector_score = float(candidate.detector_score)
        if (
            motion_fraction < min_motion_fraction
            and detector_score < strong_static_score
        ):
            rejection_reasons.append("weak_static")
            continue
        return candidate, (
            f"compact entry candidate det={detector_score:.4f} "
            f"motion={motion_fraction:.3f} bbox={[x1, y1, x2, y2]}"
        )
    summary = ",".join(rejection_reasons[:6]) or "no_candidates"
    return None, f"no admissible compact entry candidate ({summary})"


@dataclass(frozen=True)
class BoundaryEntryRefinementDecision:
    """One exact-frame semantic improvement of a clipped entry bbox.

    The first detector result remains usable immediately.  This record only
    authorizes replacing it when a later compact proposal is geometrically
    continuous with the same image-boundary entry.  It carries no mask or
    tracker authority by itself; the caller must run the accepted bbox through
    the normal exact-frame SAM2 initialization transaction.
    """

    candidate: Any
    bbox_xyxy: np.ndarray
    reason: str


@dataclass(frozen=True)
class StartupSemanticConfirmationDecision:
    """One strong interior confirmation of an image-boundary entry.

    A clipped target is still published immediately.  This record authorizes
    at most one later SAM2 reseed during a short startup window, after the
    open-vocabulary detector sees a compact, moving, sufficiently strong
    interior instance whose displacement is physically continuous with the
    original entry.  It is intentionally not a periodic re-grounding policy.
    """

    candidate: Any
    bbox_xyxy: np.ndarray
    reason: str


def _clipped_integer_bbox(
    bbox_xyxy: Any,
    image_shape: tuple[int, ...],
) -> Optional[np.ndarray]:
    height, width = int(image_shape[0]), int(image_shape[1])
    raw = np.asarray(bbox_xyxy, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(raw)):
        return None
    x1 = int(np.clip(np.floor(raw[0]), 0, width))
    y1 = int(np.clip(np.floor(raw[1]), 0, height))
    x2 = int(np.clip(np.ceil(raw[2]), 0, width))
    y2 = int(np.clip(np.ceil(raw[3]), 0, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.asarray([x1, y1, x2, y2], dtype=np.int32)


def boundary_entry_bbox_touches_image_edge(
    bbox_xyxy: Any,
    image_shape: tuple[int, ...],
) -> bool:
    """Return whether an exact integer bbox is clipped by the camera image."""

    bbox = _clipped_integer_bbox(bbox_xyxy, image_shape)
    if bbox is None:
        return False
    height, width = int(image_shape[0]), int(image_shape[1])
    x1, y1, x2, y2 = bbox.tolist()
    return bool(x1 == 0 or y1 == 0 or x2 == width or y2 == height)


def select_yolo_world_boundary_refinement_candidate(
    result: Any,
    image_bgr: np.ndarray,
    previous_gray: Optional[np.ndarray],
    initial_bbox_xyxy: Any,
    config: Optional[Mapping[str, Any]] = None,
) -> tuple[Optional[BoundaryEntryRefinementDecision], str]:
    """Select one bounded same-entry bbox after a clipped first detection.

    This is deliberately narrower than generic semantic re-detection.  The
    normal compact/motion entry filter must still pass, the first bbox must
    touch the image edge, and the new bbox must stay close in center and scale
    while either moving inward or revealing more of the object.  Callers may
    try this for only the configured number of initial policy ticks.
    """

    cfg = {} if config is None else dict(config)
    if not bool(cfg.get("boundary_entry_refinement_enabled", False)):
        return None, "boundary entry refinement is disabled"
    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_bgr must be uint8 HxWx3")
    initial = _clipped_integer_bbox(initial_bbox_xyxy, image.shape)
    if initial is None:
        return None, "initial boundary bbox is invalid"
    if not boundary_entry_bbox_touches_image_edge(initial, image.shape):
        return None, "initial bbox is not image-boundary clipped"

    candidate, entry_reason = select_yolo_world_entry_candidate(
        result,
        image,
        previous_gray,
        cfg,
    )
    if candidate is None:
        return None, "no current compact entry candidate: " + entry_reason
    current = _clipped_integer_bbox(candidate.bbox_xyxy, image.shape)
    if current is None:
        return None, "current boundary refinement bbox is invalid"

    ix1, iy1, ix2, iy2 = initial.astype(np.float64).tolist()
    cx1, cy1, cx2, cy2 = current.astype(np.float64).tolist()
    initial_center = np.asarray([(ix1 + ix2) * 0.5, (iy1 + iy2) * 0.5])
    current_center = np.asarray([(cx1 + cx2) * 0.5, (cy1 + cy2) * 0.5])
    center_step = float(np.linalg.norm(current_center - initial_center))
    short_side = float(min(image.shape[:2]))
    max_center_step = float(
        cfg.get("boundary_entry_refinement_max_center_step_ratio", 0.25)
    ) * short_side
    if center_step > max_center_step:
        return None, (
            "boundary refinement center jump exceeds limit: "
            f"{center_step:.2f}>{max_center_step:.2f}px"
        )

    initial_area = float((ix2 - ix1) * (iy2 - iy1))
    current_area = float((cx2 - cx1) * (cy2 - cy1))
    area_ratio = current_area / initial_area
    min_area_ratio = float(
        cfg.get("boundary_entry_refinement_min_area_ratio", 0.35)
    )
    max_area_ratio = float(
        cfg.get("boundary_entry_refinement_max_area_ratio", 4.0)
    )
    if not min_area_ratio <= area_ratio <= max_area_ratio:
        return None, (
            "boundary refinement area ratio outside limit: "
            f"{area_ratio:.3f} not in [{min_area_ratio:.3f},"
            f"{max_area_ratio:.3f}]"
        )

    min_inward = float(
        cfg.get("boundary_entry_refinement_min_inward_step_px", 1.0)
    )
    min_growth = float(
        cfg.get("boundary_entry_refinement_min_area_growth_ratio", 1.05)
    )
    height, width = image.shape[:2]
    inward_steps = []
    if ix1 == 0:
        inward_steps.append(cx1 - ix1)
    if iy1 == 0:
        inward_steps.append(cy1 - iy1)
    if ix2 == width:
        inward_steps.append(ix2 - cx2)
    if iy2 == height:
        # A bottom-entering object can remain clipped while its newly visible
        # top moves upward.  Count that as inward progress as well.
        inward_steps.append(max(iy1 - cy1, iy2 - cy2))
    best_inward = max(inward_steps, default=float("-inf"))
    if best_inward < min_inward and area_ratio < min_growth:
        return None, (
            "boundary refinement made neither inward nor area progress: "
            f"inward={best_inward:.2f}px area_ratio={area_ratio:.3f}"
        )

    reason = (
        f"same boundary entry center_step={center_step:.2f}px "
        f"area_ratio={area_ratio:.3f} inward={best_inward:.2f}px; "
        f"{entry_reason}"
    )
    return (
        BoundaryEntryRefinementDecision(
            candidate=candidate,
            bbox_xyxy=current.copy(),
            reason=reason,
        ),
        reason,
    )


def select_yolo_world_startup_confirmation_candidate(
    result: Any,
    image_bgr: np.ndarray,
    previous_gray: Optional[np.ndarray],
    initial_bbox_xyxy: Any,
    elapsed_s: float,
    config: Optional[Mapping[str, Any]] = None,
) -> tuple[Optional[StartupSemanticConfirmationDecision], str]:
    """Select one strong interior semantic confirmation after clipped entry."""

    cfg = {} if config is None else dict(config)
    if not bool(cfg.get("startup_semantic_confirmation_enabled", False)):
        return None, "startup semantic confirmation is disabled"
    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_bgr must be uint8 HxWx3")
    initial = _clipped_integer_bbox(initial_bbox_xyxy, image.shape)
    if initial is None:
        return None, "initial startup bbox is invalid"
    if not boundary_entry_bbox_touches_image_edge(initial, image.shape):
        return None, "startup confirmation requires an image-boundary entry"
    elapsed = float(elapsed_s)
    if not np.isfinite(elapsed) or elapsed <= 0.0:
        return None, "startup confirmation elapsed time must be positive"
    minimum_elapsed = float(
        cfg.get("startup_semantic_confirmation_min_elapsed_s", 0.0)
    )
    if elapsed < minimum_elapsed:
        return None, (
            "startup confirmation is before the stable confirmation window: "
            f"{elapsed:.3f}<{minimum_elapsed:.3f}s"
        )

    candidate, entry_reason = select_yolo_world_entry_candidate(
        result,
        image,
        previous_gray,
        cfg,
    )
    if candidate is None:
        return None, "no current compact startup candidate: " + entry_reason
    current = _clipped_integer_bbox(candidate.bbox_xyxy, image.shape)
    if current is None:
        return None, "current startup confirmation bbox is invalid"

    margin = int(cfg.get("startup_semantic_confirmation_interior_margin_px", 1))
    height, width = image.shape[:2]
    x1, y1, x2, y2 = current.tolist()
    if x1 < margin or y1 < margin or x2 > width - margin or y2 > height - margin:
        return None, "startup confirmation candidate is not interior"

    detector_score = float(candidate.detector_score)
    minimum_score = float(
        cfg.get("startup_semantic_confirmation_min_detector_score", 0.10)
    )
    if not np.isfinite(detector_score) or detector_score < minimum_score:
        return None, (
            "startup confirmation detector score below limit: "
            f"{detector_score:.4f}<{minimum_score:.4f}"
        )

    ix1, iy1, ix2, iy2 = initial.astype(np.float64).tolist()
    cx1, cy1, cx2, cy2 = current.astype(np.float64).tolist()
    initial_center = np.asarray([(ix1 + ix2) * 0.5, (iy1 + iy2) * 0.5])
    current_center = np.asarray([(cx1 + cx2) * 0.5, (cy1 + cy2) * 0.5])
    center_step = float(np.linalg.norm(current_center - initial_center))
    center_speed = center_step / elapsed
    max_center_speed = float(
        cfg.get("startup_semantic_confirmation_max_center_speed_px_s", 600.0)
    )
    if center_speed > max_center_speed:
        return None, (
            "startup confirmation center speed exceeds limit: "
            f"{center_speed:.1f}>{max_center_speed:.1f}px/s"
        )

    initial_area = float((ix2 - ix1) * (iy2 - iy1))
    current_area = float((cx2 - cx1) * (cy2 - cy1))
    area_ratio = current_area / initial_area
    min_area_ratio = float(
        cfg.get("startup_semantic_confirmation_min_area_ratio", 0.35)
    )
    max_area_ratio = float(
        cfg.get("startup_semantic_confirmation_max_area_ratio", 4.0)
    )
    if not min_area_ratio <= area_ratio <= max_area_ratio:
        return None, (
            "startup confirmation area ratio outside limit: "
            f"{area_ratio:.3f} not in "
            f"[{min_area_ratio:.3f},{max_area_ratio:.3f}]"
        )

    reason = (
        f"strong interior startup confirmation det={detector_score:.4f} "
        f"center_speed={center_speed:.1f}px/s area_ratio={area_ratio:.3f}; "
        f"{entry_reason}"
    )
    return (
        StartupSemanticConfirmationDecision(
            candidate=candidate,
            bbox_xyxy=current.copy(),
            reason=reason,
        ),
        reason,
    )


def wait_for_prompt_target(
    *,
    provider: Any,
    prompt_manager: PromptServiceManager,
    prompt: str,
    frame_buffer: Optional[RecentRGBDFrameBuffer],
    box_threshold: float,
    text_threshold: float,
    mask_threshold: float,
    top_k: int,
    search_interval_s: float = 0.5,
    detector_backend: str = "grounding_dino",
    reference_bbox_xyxy: Optional[Sequence[float]] = None,
    yolo_world_entry_filter_config: Optional[Mapping[str, Any]] = None,
    preview_callback: Optional[
        Callable[[Any, str, Optional[np.ndarray], str], None]
    ] = None,
):
    """Capture continuously until the configured text detector finds a target.

    The semantic worker may need hundreds of milliseconds, but camera capture
    never waits for it.  A two-second RGB-D ring retains every source-to-head
    interval; a valid source mask is replayed through those frames before it is
    returned.  Cheap frame-difference scoring remembers the strongest motion
    frame seen while the detector was busy, which makes a fast entering object
    much less likely to fall between low-rate semantic requests.  Ctrl+C is the
    unbounded-search cancellation path.
    """

    backend = str(detector_backend).strip().lower()
    if backend == "yolo_world":
        return _wait_for_yolo_world_target(
            provider=provider,
            prompt_manager=prompt_manager,
            prompt=prompt,
            frame_buffer=frame_buffer,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            mask_threshold=mask_threshold,
            top_k=top_k,
            search_interval_s=search_interval_s,
            reference_bbox_xyxy=reference_bbox_xyxy,
            entry_filter_config=yolo_world_entry_filter_config,
            preview_callback=preview_callback,
        )
    if backend != "grounding_dino":
        raise ValueError(
            "prompt detector_backend must be grounding_dino or yolo_world"
        )

    interval_s = float(search_interval_s)
    if not np.isfinite(interval_s) or not 0.05 <= interval_s <= 10.0:
        raise ValueError("prompt search_interval_s must be in 0.05..10.0")
    if frame_buffer is None:
        raise ValueError("automatic prompt search requires an RGB-D frame buffer")
    worker = AsyncPromptReacquirer(
        addr=prompt_manager.addr,
        timeout_ms=prompt_manager.request_timeout_ms,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        mask_threshold=mask_threshold,
        top_k=top_k,
    )
    attempt = 0
    last_submit_s = float("-inf")
    previous_gray = None
    best_motion_frame_id = None
    best_motion_score = -1.0
    try:
        while True:
            frame = provider.camera.get_frame()
            frame_buffer.append(frame)
            if preview_callback is not None:
                preview_callback(
                    frame,
                    "searching",
                    None,
                    f"Searching for {prompt!r}; hold the target still",
                )
            gray = cv2.resize(
                cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2GRAY),
                None,
                fx=0.25,
                fy=0.25,
                interpolation=cv2.INTER_AREA,
            )
            if previous_gray is not None:
                delta = cv2.absdiff(gray, previous_gray)
                # Mean excess above a noise-tolerant threshold is stable for
                # small fast objects and substantially cheaper than detection.
                motion_score = float(np.maximum(delta.astype(np.float32) - 18.0, 0.0).mean())
                if motion_score > best_motion_score:
                    best_motion_score = motion_score
                    best_motion_frame_id = int(frame.frame_id)
            previous_gray = gray

            response = worker.poll()
            if response is not None:
                request = response.request
                if response.error is not None:
                    print(
                        f"[Prompt SEARCHING][WARN] request={request.request_id} "
                        f"failed: {response.error}",
                        flush=True,
                    )
                elif response.result is not None:
                    result = response.result
                    print_prompt_result(result, prefix="[Prompt search]")
                    echo_error = prompt_response_echo_error(
                        request,
                        result,
                        expected_prompt=prompt,
                    )
                    window = frame_buffer.replay_window(
                        request.frame_id,
                        max_frames=12,
                    )
                    if (
                        echo_error is None
                        and result.valid
                        and result.bbox_xyxy is not None
                        and window is not None
                        and int(window.head_frame_id) == int(frame.frame_id)
                    ):
                        replay = replay_prompt_mask(
                            window,
                            result.mask,
                            provider.cfg["tracker"],
                        )
                        print(
                            "[Prompt SEARCHING replay] "
                            f"source={request.frame_id} head={frame.frame_id} "
                            f"valid={replay.valid} elapsed={replay.elapsed_ms:.1f}ms "
                            f"| {replay.message}",
                            flush=True,
                        )
                        if replay.valid and replay.bbox_xyxy is not None:
                            # Convert the semantic result to exact-current
                            # pixels before the production provider/SAM2 sees
                            # it. Prompt metadata stays available for audit.
                            result.mask = replay.mask
                            result.bbox_xyxy = replay.bbox_xyxy
                            result.frame_metadata = {
                                **dict(result.frame_metadata or {}),
                                "auto_grounding_source_frame_id": int(
                                    request.frame_id
                                ),
                                "auto_grounding_head_frame_id": int(
                                    frame.frame_id
                                ),
                                "auto_grounding_replay_frame_ids": list(
                                    replay.sampled_frame_ids
                                ),
                            }
                            result.frame_id = int(frame.frame_id)
                            result.frame_timestamp = float(frame.timestamp)
                            result.message = (
                                f"{result.message}; replayed to frame "
                                f"{frame.frame_id}"
                            )
                            if preview_callback is not None:
                                preview_callback(
                                    frame,
                                    "candidate",
                                    np.asarray(result.bbox_xyxy, dtype=np.int32),
                                    "Semantic candidate found; validating RGB-D",
                                )
                            return frame, result
                    elif echo_error is not None:
                        print(
                            f"[Prompt SEARCHING][WARN] rejected response: "
                            f"{echo_error}",
                            flush=True,
                        )

            now = time.monotonic()
            if not worker.busy and now - last_submit_s >= interval_s:
                candidate = (
                    None
                    if best_motion_frame_id is None
                    else frame_buffer.get(best_motion_frame_id)
                )
                if candidate is None:
                    candidate = frame
                attempt += 1
                request_id = worker.submit(
                    candidate.color_bgr,
                    prompt=prompt,
                    reference_bbox_xyxy=reference_bbox_xyxy,
                    frame_id=candidate.frame_id,
                    frame_timestamp=candidate.timestamp,
                    generation=0,
                    reason="initial_or_reentry_search",
                )
                if request_id is not None:
                    last_submit_s = now
                    print(
                        f"[Prompt SEARCHING] prompt={prompt!r} "
                        f"attempt={attempt} source_frame={candidate.frame_id} "
                        f"motion_score={max(0.0, best_motion_score):.3f}",
                        flush=True,
                    )
                    best_motion_frame_id = None
                    best_motion_score = -1.0
    finally:
        worker.close(join_timeout_s=1.0)


def _wait_for_yolo_world_target(
    *,
    provider: Any,
    prompt_manager: PromptServiceManager,
    prompt: str,
    frame_buffer: Optional[RecentRGBDFrameBuffer],
    box_threshold: float,
    text_threshold: float,
    mask_threshold: float,
    top_k: int,
    search_interval_s: float,
    reference_bbox_xyxy: Optional[Sequence[float]],
    entry_filter_config: Optional[Mapping[str, Any]],
    preview_callback: Optional[
        Callable[[Any, str, Optional[np.ndarray], str], None]
    ],
):
    """Run the verified fast detector synchronously on exact camera frames.

    Hot YOLO-World inference is below one 20 Hz policy period on the deployed
    GPU.  Keeping this call synchronous avoids applying a fast-moving object's
    bbox to newer pixels.  The returned rectangle is never treated as a mask;
    the caller must initialize SAM2 from ``bbox_xyxy`` on this exact frame.
    """

    interval_s = float(search_interval_s)
    if not np.isfinite(interval_s) or not 0.02 <= interval_s <= 10.0:
        raise ValueError("prompt search_interval_s must be in 0.02..10.0")
    if frame_buffer is None:
        raise ValueError("automatic prompt search requires an RGB-D frame buffer")
    attempt = 0
    last_submit_s = float("-inf")
    previous_gray = None
    while True:
        frame = provider.camera.get_frame()
        frame_buffer.append(frame)
        if preview_callback is not None:
            preview_callback(
                frame,
                "searching",
                None,
                f"Searching for {prompt!r}; hold the target still",
            )
        current_gray = cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2GRAY)
        now = time.monotonic()
        if now - last_submit_s < interval_s:
            continue
        attempt += 1
        last_submit_s = now
        result = prompt_manager.segment(
            frame.color_bgr,
            prompt=prompt,
            reference_bbox_xyxy=reference_bbox_xyxy,
            request_id=f"yolo-search-f{int(frame.frame_id)}-a{attempt}",
            frame_id=int(frame.frame_id),
            frame_timestamp=float(frame.timestamp),
            frame_metadata={
                "tracker_generation": 0,
                "reason": "initial_or_reentry_search",
            },
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            mask_threshold=mask_threshold,
            top_k=top_k,
        )
        print_prompt_result(result, prefix="[YOLO-World search]")
        candidate, entry_reason = select_yolo_world_entry_candidate(
            result,
            frame.color_bgr,
            previous_gray,
            entry_filter_config,
        )
        previous_gray = current_gray
        if candidate is None:
            if preview_callback is not None:
                preview_callback(
                    frame,
                    "rejected",
                    None,
                    "No admissible semantic candidate; keep target visible",
                )
            print(f"[YOLO-World SEARCHING] rejected: {entry_reason}", flush=True)
            continue
        bbox = np.asarray(candidate.bbox_xyxy, dtype=np.float64).reshape(4)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0.0, frame.color_bgr.shape[1])
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0.0, frame.color_bgr.shape[0])
        bbox = np.asarray(
            [
                int(np.floor(bbox[0])),
                int(np.floor(bbox[1])),
                int(np.ceil(bbox[2])),
                int(np.ceil(bbox[3])),
            ],
            dtype=np.int32,
        )
        result.mask.fill(0)
        result.mask[bbox[1] : bbox[3], bbox[0] : bbox[2]] = 1
        result.bbox_xyxy = bbox
        result.score = float(candidate.detector_score)
        result.selected_index = int(candidate.detector_index)
        result.valid = True
        result.message = f"{result.message}; {entry_reason}"
        result.frame_metadata = {
            **dict(result.frame_metadata or {}),
            "entry_candidate_filter": "compact_motion_or_strong_semantic_v1",
            "entry_candidate_filter_reason": entry_reason,
        }
        metadata = dict(result.frame_metadata or {})
        if metadata.get("prompt_output_kind") != "bbox_only":
            raise RuntimeError(
                "YOLO-World service did not identify its output as bbox_only"
            )
        if result.frame_id != int(frame.frame_id) or not np.isclose(
            float(result.frame_timestamp),
            float(frame.timestamp),
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise RuntimeError("YOLO-World response is not tied to the exact frame")
        if preview_callback is not None:
            preview_callback(
                frame,
                "candidate",
                bbox,
                "Semantic candidate found; validating depth and SAM2",
            )
        return frame, result


def validate_prompting_config(prompt_cfg: Mapping[str, Any]) -> None:
    """Validate prompt-service values before opening the camera or ZMQ ports."""

    if not isinstance(prompt_cfg, Mapping):
        raise ValueError("prompting must be a mapping")
    address = prompt_cfg.get("service_addr", "")
    if not isinstance(address, str) or not address.strip():
        raise ValueError("prompting.service_addr must be a non-empty string")
    detector_backend = str(
        prompt_cfg.get("detector_backend", "grounding_dino")
    ).strip().lower()
    if detector_backend not in ("grounding_dino", "yolo_world"):
        raise ValueError(
            "prompting.detector_backend must be grounding_dino or yolo_world"
        )
    service_device = prompt_cfg.get("service_device")
    if not isinstance(service_device, str) or not service_device.strip():
        raise ValueError(
            "prompting.service_device must be auto, cpu, cuda, or cuda:N"
        )
    normalized_device = service_device.strip().lower()
    if (
        normalized_device not in ("auto", "cpu", "cuda")
        and not (
            normalized_device.startswith("cuda:")
            and normalized_device[5:].isdigit()
        )
    ):
        raise ValueError(
            "prompting.service_device must be auto, cpu, cuda, or cuda:N"
        )
    mask_backend = prompt_cfg.get("mask_backend")
    if detector_backend == "grounding_dino" and (
        not isinstance(mask_backend, str)
        or mask_backend.strip().lower() not in ("sam1", "sam2")
    ):
        raise ValueError("prompting.mask_backend must be sam1 or sam2")
    if detector_backend == "yolo_world":
        weights = prompt_cfg.get("yolo_world_weights")
        if not isinstance(weights, str) or not weights.strip():
            raise ValueError(
                "prompting.yolo_world_weights must be a non-empty path"
            )
        try:
            confidence = float(prompt_cfg.get("yolo_world_confidence"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "prompting.yolo_world_confidence must be in [0.001, 1]"
            ) from exc
        if not np.isfinite(confidence) or not 0.001 <= confidence <= 1.0:
            raise ValueError(
                "prompting.yolo_world_confidence must be in [0.001, 1]"
            )
        image_size = prompt_cfg.get("yolo_world_image_size")
        try:
            numeric_image_size = float(image_size)
            parsed_image_size = int(image_size)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "prompting.yolo_world_image_size must be an integer in 320..1280"
            ) from exc
        if (
            isinstance(image_size, bool)
            or not np.isfinite(numeric_image_size)
            or numeric_image_size != float(parsed_image_size)
            or not 320 <= parsed_image_size <= 1280
        ):
            raise ValueError(
                "prompting.yolo_world_image_size must be an integer in 320..1280"
            )
        search_roi = prompt_cfg.get("search_reference_roi_xyxy")
        if search_roi is not None:
            try:
                search_roi_array = np.asarray(search_roi, dtype=np.float64).reshape(4)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "prompting.search_reference_roi_xyxy must be null or four finite XYXY values"
                ) from exc
            if (
                not np.all(np.isfinite(search_roi_array))
                or search_roi_array[0] < 0.0
                or search_roi_array[1] < 0.0
                or search_roi_array[2] <= search_roi_array[0]
                or search_roi_array[3] <= search_roi_array[1]
            ):
                raise ValueError(
                    "prompting.search_reference_roi_xyxy must be null or four finite ordered XYXY values"
                )
        workspace_center = prompt_cfg.get("grounding_workspace_center_base_m")
        workspace_radius = prompt_cfg.get(
            "grounding_workspace_max_horizontal_distance_m"
        )
        if (workspace_center is None) != (workspace_radius is None):
            raise ValueError(
                "prompting grounding workspace center and radius must be set together"
            )
        if workspace_center is not None:
            try:
                workspace_center_array = np.asarray(
                    workspace_center, dtype=np.float64
                ).reshape(3)
                workspace_radius_value = float(workspace_radius)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "prompting grounding workspace must be finite center[3] and positive radius"
                ) from exc
            if (
                not np.all(np.isfinite(workspace_center_array))
                or not np.isfinite(workspace_radius_value)
                or not 0.05 <= workspace_radius_value <= 2.0
            ):
                raise ValueError(
                    "prompting grounding workspace must be finite center[3] and radius in [0.05,2.0]"
                )
        if not isinstance(
            prompt_cfg.get("entry_candidate_filter_enabled", True), bool
        ):
            raise ValueError(
                "prompting.entry_candidate_filter_enabled must be true or false"
            )
        if not isinstance(
            prompt_cfg.get("boundary_entry_refinement_enabled", False), bool
        ):
            raise ValueError(
                "prompting.boundary_entry_refinement_enabled must be true or false"
            )
        if not isinstance(
            prompt_cfg.get("startup_semantic_confirmation_enabled", False), bool
        ):
            raise ValueError(
                "prompting.startup_semantic_confirmation_enabled must be true or false"
            )
        entry_ranges = {
            "entry_candidate_min_extent_ratio": (0.001, 0.25, 0.03),
            "entry_candidate_max_extent_ratio": (0.01, 1.0, 0.15),
            "entry_candidate_max_aspect_ratio": (1.0, 10.0, 2.5),
            "entry_candidate_min_motion_fraction": (0.0, 1.0, 0.15),
            "entry_candidate_strong_static_score": (0.0, 1.0, 0.05),
        }
        parsed_entry_values = {}
        for name, (minimum, maximum, default) in entry_ranges.items():
            try:
                value = float(prompt_cfg.get(name, default))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"prompting.{name} must be in [{minimum}, {maximum}]"
                ) from exc
            if not np.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(
                    f"prompting.{name} must be in [{minimum}, {maximum}]"
                )
            parsed_entry_values[name] = value
        if (
            parsed_entry_values["entry_candidate_min_extent_ratio"]
            >= parsed_entry_values["entry_candidate_max_extent_ratio"]
        ):
            raise ValueError(
                "prompting entry candidate minimum extent must be below maximum"
            )
        boundary_refinement_ranges = {
            "boundary_entry_refinement_max_center_step_ratio": (
                0.01,
                1.0,
                0.25,
            ),
            "boundary_entry_refinement_min_area_ratio": (0.05, 1.0, 0.35),
            "boundary_entry_refinement_max_area_ratio": (1.0, 10.0, 4.0),
            "boundary_entry_refinement_min_inward_step_px": (0.0, 50.0, 1.0),
            "boundary_entry_refinement_min_area_growth_ratio": (1.0, 4.0, 1.05),
        }
        parsed_boundary_values = {}
        for name, (minimum, maximum, default) in boundary_refinement_ranges.items():
            try:
                value = float(prompt_cfg.get(name, default))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"prompting.{name} must be in [{minimum}, {maximum}]"
                ) from exc
            if not np.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(
                    f"prompting.{name} must be in [{minimum}, {maximum}]"
                )
            parsed_boundary_values[name] = value
        if (
            parsed_boundary_values["boundary_entry_refinement_min_area_ratio"]
            > parsed_boundary_values["boundary_entry_refinement_max_area_ratio"]
        ):
            raise ValueError(
                "prompting boundary entry refinement minimum area ratio must "
                "not exceed maximum"
            )
        startup_confirmation_ranges = {
            "startup_semantic_confirmation_min_elapsed_s": (0.0, 2.0, 0.0),
            "startup_semantic_confirmation_min_detector_score": (
                0.001,
                1.0,
                0.10,
            ),
            "startup_semantic_confirmation_max_center_speed_px_s": (
                1.0,
                5000.0,
                600.0,
            ),
            "startup_semantic_confirmation_min_area_ratio": (0.05, 1.0, 0.35),
            "startup_semantic_confirmation_max_area_ratio": (1.0, 10.0, 4.0),
        }
        parsed_startup_values = {}
        for name, (minimum, maximum, default) in startup_confirmation_ranges.items():
            try:
                value = float(prompt_cfg.get(name, default))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"prompting.{name} must be in [{minimum}, {maximum}]"
                ) from exc
            if not np.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(
                    f"prompting.{name} must be in [{minimum}, {maximum}]"
                )
            parsed_startup_values[name] = value
        if (
            parsed_startup_values["startup_semantic_confirmation_min_area_ratio"]
            > parsed_startup_values["startup_semantic_confirmation_max_area_ratio"]
        ):
            raise ValueError(
                "prompting startup semantic confirmation minimum area ratio must "
                "not exceed maximum"
            )
        raw_refinement_ticks = prompt_cfg.get(
            "boundary_entry_refinement_max_ticks", 2
        )
        try:
            numeric_refinement_ticks = float(raw_refinement_ticks)
            refinement_ticks = int(raw_refinement_ticks)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "prompting.boundary_entry_refinement_max_ticks must be an "
                "integer in 1..4"
            ) from exc
        if (
            isinstance(raw_refinement_ticks, bool)
            or not np.isfinite(numeric_refinement_ticks)
            or numeric_refinement_ticks != float(refinement_ticks)
            or not 1 <= refinement_ticks <= 4
        ):
            raise ValueError(
                "prompting.boundary_entry_refinement_max_ticks must be an "
                "integer in 1..4"
            )
        startup_integer_ranges = {
            "startup_semantic_confirmation_max_ticks": (2, 30, 10),
            "startup_semantic_confirmation_search_every_ticks": (1, 10, 2),
            "startup_semantic_confirmation_interior_margin_px": (0, 32, 1),
        }
        parsed_startup_integers = {}
        for name, (minimum, maximum, default) in startup_integer_ranges.items():
            raw_value = prompt_cfg.get(name, default)
            try:
                numeric_value = float(raw_value)
                value = int(raw_value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"prompting.{name} must be an integer in "
                    f"{minimum}..{maximum}"
                ) from exc
            if (
                isinstance(raw_value, bool)
                or not np.isfinite(numeric_value)
                or numeric_value != float(value)
                or not minimum <= value <= maximum
            ):
                raise ValueError(
                    f"prompting.{name} must be an integer in "
                    f"{minimum}..{maximum}"
                )
            parsed_startup_integers[name] = value
        if (
            parsed_startup_integers[
                "startup_semantic_confirmation_search_every_ticks"
            ]
            > parsed_startup_integers["startup_semantic_confirmation_max_ticks"]
        ):
            raise ValueError(
                "prompting startup semantic confirmation search interval must "
                "not exceed its window"
            )
        raw_difference = prompt_cfg.get("entry_candidate_motion_difference", 10)
        try:
            numeric_difference = float(raw_difference)
            difference = int(raw_difference)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "prompting.entry_candidate_motion_difference must be an integer in 1..255"
            ) from exc
        if (
            isinstance(raw_difference, bool)
            or not np.isfinite(numeric_difference)
            or numeric_difference != float(difference)
            or not 1 <= difference <= 255
        ):
            raise ValueError(
                "prompting.entry_candidate_motion_difference must be an integer in 1..255"
            )
    for name in (
        "service_autostart",
        "auto_reacquire",
        "global_reacquire",
        "relocate_stale_mask",
    ):
        if not isinstance(prompt_cfg.get(name), bool):
            raise ValueError(f"prompting.{name} must be true or false")

    for name in (
        "startup_timeout_s",
        "request_timeout_s",
        "reacquire_cooldown_s",
        "replay_buffer_s",
    ):
        try:
            value = float(prompt_cfg.get(name))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"prompting.{name} must be a finite positive number") from exc
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"prompting.{name} must be a finite positive number")
    try:
        search_interval_s = float(prompt_cfg.get("search_interval_s", 0.5))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "prompting.search_interval_s must be in 0.05..10.0"
        ) from exc
    minimum_search_interval_s = 0.02 if detector_backend == "yolo_world" else 0.05
    if (
        not np.isfinite(search_interval_s)
        or not minimum_search_interval_s <= search_interval_s <= 10.0
    ):
        raise ValueError(
            "prompting.search_interval_s must be in "
            f"{minimum_search_interval_s:.2f}..10.0"
        )

    for name in (
        "box_threshold",
        "text_threshold",
        "mask_threshold",
        "stale_mask_min_score",
        "stale_mask_min_score_margin",
    ):
        try:
            value = float(prompt_cfg.get(name))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"prompting.{name} must be in [0, 1]") from exc
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"prompting.{name} must be in [0, 1]")

    for name in (
        "top_k",
        "reacquire_lost_frames",
        "stale_mask_after_frames",
        "replay_max_frames",
    ):
        raw_value = prompt_cfg.get(name)
        try:
            numeric = float(raw_value)
            value = int(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"prompting.{name} must be an integer >= 1") from exc
        minimum = 2 if name == "replay_max_frames" else 1
        if (
            isinstance(raw_value, bool)
            or not np.isfinite(numeric)
            or numeric != float(value)
            or value < minimum
        ):
            raise ValueError(
                f"prompting.{name} must be an integer >= {minimum}"
            )


def validate_online_sam2_config(online_cfg: Mapping[str, Any]) -> None:
    """Validate the independent stateful SAM2 video service configuration."""

    if not isinstance(online_cfg, Mapping):
        raise ValueError("online_sam2 must be a mapping")
    for name in ("enabled", "service_autostart", "vos_optimized"):
        if not isinstance(online_cfg.get(name), bool):
            raise ValueError(f"online_sam2.{name} must be true or false")
    vos_compile_mode = online_cfg.get("vos_compile_mode")
    if vos_compile_mode not in ALLOWED_VOS_COMPILE_MODES:
        raise ValueError(
            "online_sam2.vos_compile_mode must be one of "
            + ", ".join(ALLOWED_VOS_COMPILE_MODES)
        )
    if (
        online_cfg.get("vos_optimized") is True
        and vos_compile_mode != PRODUCTION_VOS_COMPILE_MODE
    ):
        raise ValueError(
            "online_sam2 optimized production runtime requires "
            f"vos_compile_mode={PRODUCTION_VOS_COMPILE_MODE}"
        )
    address = online_cfg.get("service_addr", "")
    if not isinstance(address, str) or not address.strip():
        raise ValueError("online_sam2.service_addr must be a non-empty string")
    for name in ("checkpoint", "model_config", "device"):
        value = online_cfg.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"online_sam2.{name} must be a non-empty string")
    if str(online_cfg.get("amp_dtype", "")).strip().lower() not in (
        "bfloat16",
        "float16",
    ):
        raise ValueError("online_sam2.amp_dtype must be bfloat16 or float16")
    for name in (
        "startup_timeout_s",
        "request_timeout_s",
        "frame_wait_timeout_s",
        "initialization_timeout_s",
    ):
        try:
            value = float(online_cfg.get(name))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"online_sam2.{name} must be a finite positive number"
            ) from exc
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"online_sam2.{name} must be a finite positive number"
            )
    for name, minimum in (
        ("image_size", 64),
        ("min_mask_area", 1),
        ("reset_every_frames", 0),
    ):
        raw = online_cfg.get(name)
        try:
            numeric = float(raw)
            value = int(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"online_sam2.{name} must be an integer >= {minimum}"
            ) from exc
        if (
            isinstance(raw, bool)
            or not np.isfinite(numeric)
            or numeric != float(value)
            or value < minimum
        ):
            raise ValueError(
                f"online_sam2.{name} must be an integer >= {minimum}"
            )
    if int(online_cfg["image_size"]) % 16 != 0:
        raise ValueError("online_sam2.image_size must be divisible by 16")
    if int(online_cfg["reset_every_frames"]) == 1:
        raise ValueError(
            "online_sam2.reset_every_frames must be 0 or at least 2"
        )
    try:
        object_score = float(online_cfg.get("object_score_threshold"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "online_sam2.object_score_threshold must be finite"
        ) from exc
    if not np.isfinite(object_score):
        raise ValueError("online_sam2.object_score_threshold must be finite")


def prompt_service_launcher_path(prompt_cfg: Mapping[str, Any]) -> Path:
    """Return the environment-specific prompt service launcher."""

    validate_prompting_config(prompt_cfg)
    root = Path(__file__).resolve().parents[2]
    backend = str(
        prompt_cfg.get("detector_backend", "grounding_dino")
    ).strip().lower()
    name = (
        "run_yolo_world_prompt_service.sh"
        if backend == "yolo_world"
        else "run_prompt_segmentation_service.sh"
    )
    return root / "scripts" / name


def prompt_service_launcher_args(
    prompt_cfg: Mapping[str, Any],
    *,
    prompt: Optional[str] = None,
) -> tuple:
    """Return normalized model options for an autostarted prompt service."""

    validate_prompting_config(prompt_cfg)
    backend = str(
        prompt_cfg.get("detector_backend", "grounding_dino")
    ).strip().lower()
    if backend == "yolo_world":
        args = (
            "--device",
            str(prompt_cfg["service_device"]).strip().lower(),
            "--weights",
            str(Path(str(prompt_cfg["yolo_world_weights"])).expanduser()),
            "--confidence",
            str(float(prompt_cfg["yolo_world_confidence"])),
            "--image-size",
            str(int(prompt_cfg["yolo_world_image_size"])),
        )
        if prompt is not None and str(prompt).strip():
            args += ("--preload-prompt", str(prompt).strip())
        return args
    return (
        "--device",
        str(prompt_cfg["service_device"]).strip().lower(),
        "--mask-backend",
        str(prompt_cfg["mask_backend"]).strip().lower(),
    )


def validate_prompt_service_health(
    prompt_cfg: Mapping[str, Any], health: Mapping[str, Any]
) -> None:
    """Reject a stale service from another configured detector backend."""

    validate_prompting_config(prompt_cfg)
    expected = str(
        prompt_cfg.get("detector_backend", "grounding_dino")
    ).strip().lower()
    actual_detector = str(health.get("detector_backend", "")).strip().lower()
    actual_mask = str(health.get("mask_backend", "")).strip().lower()
    if expected == "yolo_world":
        if actual_detector != "yolo_world" or actual_mask != "bbox_only":
            raise RuntimeError(
                "prompt service backend mismatch: configured "
                "yolo_world/bbox_only, running "
                f"{actual_detector or '<missing>'}/{actual_mask or '<missing>'}"
            )
    elif actual_detector not in ("native", "transformers"):
        raise RuntimeError(
            "prompt service backend mismatch: configured Grounding-DINO, "
            f"running {actual_detector or '<missing>'}"
        )


def prompt_response_echo_error(
    request,
    result,
    expected_prompt: str,
) -> Optional[str]:
    """Return a reason when a semantic response does not echo its source frame.

    The request ID is already checked by the production ZMQ client, but checking
    it again here keeps the application-side safety gate intact for alternate
    clients and tests.  A prompt mask is meaningful only for the exact image and
    tracker revision described by this metadata.
    """

    if request.prompt != expected_prompt:
        return "queued prompt no longer matches the configured prompt"
    if result.request_id != request.request_id:
        return "request_id mismatch"
    if result.prompt != request.prompt:
        return "prompt echo mismatch"
    if result.frame_id != request.frame_id:
        return "frame_id echo mismatch"
    if result.frame_timestamp is None:
        return "missing frame_timestamp echo"
    try:
        result_timestamp = float(result.frame_timestamp)
        request_timestamp = float(request.frame_timestamp)
    except (TypeError, ValueError) as exc:
        return f"invalid frame_timestamp echo: {exc}"
    if (
        not np.isfinite(result_timestamp)
        or not np.isfinite(request_timestamp)
        or not np.isclose(
            result_timestamp,
            request_timestamp,
            rtol=0.0,
            atol=1.0e-6,
        )
    ):
        return "frame_timestamp echo mismatch"
    metadata = result.frame_metadata
    if not isinstance(metadata, dict):
        return "frame_metadata is not an object"
    try:
        echoed_generation = int(metadata.get("tracker_generation", -1))
    except (TypeError, ValueError, OverflowError):
        return "invalid tracker_generation metadata"
    if echoed_generation != request.generation:
        return "tracker_generation metadata mismatch"
    if metadata.get("reason") != request.reason:
        return "reacquire reason metadata mismatch"
    return None


def save_object_pcd(save_dir: str, obj, packet) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    npz_path = Path(save_dir) / f"object_packet_{ts}.npz"
    np.savez_compressed(
        npz_path,
        pcd_current=packet.pcd_current,
        pcd_history=packet.pcd_history,
        pcd_reference=packet.pcd_reference,
        center=packet.center,
        velocity=packet.velocity,
        bbox_xyxy=packet.bbox_xyxy,
        valid=packet.valid,
        timestamp=packet.timestamp,
        frame_id=packet.frame_id,
        reference_frame=packet.reference_frame,
        point_frame=packet.point_frame,
        calibration_id=packet.calibration_id,
        camera_serial=packet.camera_serial,
    )
    print(f"[Saved] {npz_path}")

    try:
        import open3d as o3d
        if obj is not None and obj.valid and len(obj.points) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obj.points.astype(np.float64))
            if obj.colors is not None and len(obj.colors) == len(obj.points):
                pcd.colors = o3d.utility.Vector3dVector(obj.colors.astype(np.float64))
            ply_path = Path(save_dir) / f"object_cloud_{ts}.ply"
            o3d.io.write_point_cloud(str(ply_path), pcd)
            print(f"[Saved] {ply_path}")
    except Exception as e:
        print(f"[WARN] could not save PLY: {e}")


def record_packet_rates(
    *,
    packet_valid: bool,
    perception_monitor: PacketRateMonitor,
    publish_monitor: PacketRateMonitor,
    published: bool,
) -> None:
    """Record perception independently from optional ZMQ publication."""

    perception_monitor.tick(bool(packet_valid))
    if published:
        publish_monitor.tick(bool(packet_valid))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/d435_default.yaml")
    parser.add_argument("--publish_zmq", action="store_true")
    parser.add_argument("--zmq_addr", type=str, default=None)
    image_vis_group = parser.add_mutually_exclusive_group()
    image_vis_group.add_argument(
        "--vis",
        dest="vis",
        action="store_true",
        default=None,
        help="Enable the OpenCV RGB/mask window (prefer with --no_show_pcd)",
    )
    image_vis_group.add_argument(
        "--no_vis",
        dest="vis",
        action="store_false",
        help="Disable the OpenCV RGB/mask window",
    )
    parser.add_argument("--no_show_pcd", action="store_true")
    parser.add_argument(
        "--show_scene_pcd",
        action="store_true",
        help="Show dim scene cloud and highlighted segmented object together in robot_base",
    )
    parser.add_argument(
        "--scene_stride",
        type=int,
        default=None,
        help="Pixel stride for the lightweight display-only scene cloud (default: config/4)",
    )
    parser.add_argument(
        "--scene_update_every",
        type=int,
        default=None,
        help="Refresh scene geometry every N perception frames (default: config/2)",
    )
    parser.add_argument(
        "--goal_clearance_m",
        type=float,
        default=None,
        help="Visualize an EEF goal this many meters above the current object z95",
    )
    parser.add_argument("--save_dir", type=str, default="object_pcd_saves")
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Initialize from a numeric ROI and skip the interactive selector",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help=(
            "Open-vocabulary target description, e.g. 'green ball'. The "
            "configured text detector finds a bbox; online SAM2, not the "
            "rectangle, produces the object mask."
        ),
    )
    parser.add_argument(
        "--prompt_reference_roi",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Optional image-space hint for choosing among same-prompt instances",
    )
    parser.add_argument("--prompt_service_addr", type=str, default=None)
    parser.add_argument("--no_prompt_service_autostart", action="store_true")
    parser.add_argument("--prompt_startup_timeout_s", type=float, default=None)
    parser.add_argument("--prompt_timeout_s", type=float, default=None)
    parser.add_argument("--prompt_box_threshold", type=float, default=None)
    parser.add_argument("--prompt_text_threshold", type=float, default=None)
    parser.add_argument("--prompt_mask_threshold", type=float, default=None)
    parser.add_argument("--prompt_top_k", type=int, default=None)
    parser.add_argument("--prompt_reacquire_lost_frames", type=int, default=None)
    parser.add_argument("--prompt_reacquire_cooldown_s", type=float, default=None)
    parser.add_argument(
        "--no_prompt_auto_reacquire",
        action="store_true",
        help="Disable automatic background prompt re-acquisition after loss",
    )
    parser.add_argument(
        "--capture_only",
        type=str,
        default=None,
        help="Save one current color frame to this path and exit",
    )
    parser.add_argument("--sam2", action="store_true", help="Enable SAM2 from config for initialization/reinit")
    parser.add_argument(
        "--no_sam2",
        action="store_true",
        help="Disable SAM2 and use the configured manual ROI initializer",
    )
    online_sam2_group = parser.add_mutually_exclusive_group()
    online_sam2_group.add_argument(
        "--online_sam2",
        dest="online_sam2",
        action="store_true",
        default=None,
        help="Enable persistent online SAM2 video temporal candidates",
    )
    online_sam2_group.add_argument(
        "--no_online_sam2",
        dest="online_sam2",
        action="store_false",
        help="Disable persistent online SAM2 video temporal candidates",
    )
    parser.add_argument(
        "--online_sam2_image_size",
        type=int,
        default=None,
        help=(
            "Override online SAM2 square inference size (multiple of 16; "
            "useful for measured latency/quality tuning)"
        ),
    )
    parser.add_argument(
        "--online_sam2_frame_wait_ms",
        type=float,
        default=None,
        help=(
            "Maximum exact-current-frame SAM2 wait in milliseconds; late "
            "masks still advance temporal state but are never applied to new RGB-D"
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["roi_depth", "adaptive_color_depth", "sam2_reinit"],
    )
    parser.add_argument(
        "--depth_tolerance",
        type=float,
        default=None,
        help="Override ROI tracker depth band in meters",
    )
    parser.add_argument("--sam2_reinit_every", type=int, default=None, help="Override sam2.reinit_every; 1 means every frame, 0 disables reinit")
    parser.add_argument("--sam2_score_threshold", type=float, default=None, help="Override sam2.score_threshold")
    parser.add_argument("--save_debug_masks", action="store_true", help="Save SAM2/tracker debug mask overlays during initialization/reinit")
    parser.add_argument("--debug_masks_dir", type=str, default=None, help="Directory for --save_debug_masks output")
    parser.add_argument("--print_fps", action="store_true", help="Print perception loop FPS and stage timings")
    parser.add_argument(
        "--no_identity",
        action="store_true",
        help="Disable appearance-identity rejection for a tightly specified ROI",
    )
    parser.add_argument(
        "--lock_roi",
        action="store_true",
        help="Keep depth tracking inside the initial ROI (stationary commissioning only)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Exit after this many processed frames (useful for headless checks)",
    )
    parser.add_argument(
        "--print_center",
        action="store_true",
        help="Print the visible-cloud median center and z95 in its named frame",
    )
    parser.add_argument("--print_every", type=int, default=30, help="Print stats every N frames")
    parser.add_argument("--robot_ip", type=str, default="172.16.0.2")
    parser.add_argument(
        "--hover_clearance_m",
        type=float,
        default=0.10,
        help="One-shot EEF clearance above object cloud top (minimum 0.10m)",
    )
    parser.add_argument(
        "--motion_observe_timeout_s",
        type=float,
        default=30.0,
        help="Seconds the D/Y child may wait for a stable target",
    )
    parser.add_argument(
        "--motion_arm_timeout_s",
        type=float,
        default=5.0,
        help="Seconds after M during which Y may confirm motion",
    )
    parser.add_argument(
        "--enable_robot_motion",
        action="store_true",
        help="Allow M then Y to launch one guarded hover; disabled by default",
    )
    parser.add_argument("--confirm_calibration_id", type=str, default=None)
    parser.add_argument("--confirm_workspace_clear", action="store_true")
    parser.add_argument("--confirm_eef_clear", action="store_true")
    parser.add_argument("--confirm_descent_clear", action="store_true")
    args = parser.parse_args()
    if args.sam2 and args.no_sam2:
        parser.error("--sam2 and --no_sam2 are mutually exclusive")
    if args.prompt is not None:
        args.prompt = args.prompt.strip()
        if not args.prompt:
            parser.error("--prompt cannot be empty")
    if args.prompt is not None and args.roi is not None:
        parser.error("--prompt and --roi are mutually exclusive")
    if args.prompt_reference_roi is not None and args.prompt is None:
        parser.error("--prompt_reference_roi requires --prompt")
    if args.prompt is not None and args.sam2:
        parser.error("--prompt uses the external Grounded-SAM service; omit --sam2")
    if (
        args.prompt is not None
        and args.mode is not None
        and args.mode != "adaptive_color_depth"
    ):
        parser.error(
            "--prompt requires --mode adaptive_color_depth; omit --mode or "
            "select adaptive_color_depth"
        )
    if args.no_show_pcd and args.show_scene_pcd:
        parser.error("--no_show_pcd and --show_scene_pcd are mutually exclusive")
    if args.scene_stride is not None and args.scene_stride < 1:
        parser.error("--scene_stride must be >= 1")
    if args.scene_update_every is not None and args.scene_update_every < 1:
        parser.error("--scene_update_every must be >= 1")
    if args.goal_clearance_m is not None and args.goal_clearance_m <= 0:
        parser.error("--goal_clearance_m must be > 0")
    if args.depth_tolerance is not None and not 0.003 <= args.depth_tolerance <= 0.10:
        parser.error("--depth_tolerance must be in [0.003, 0.10] meters")
    if not 0.10 <= args.hover_clearance_m <= 0.30:
        parser.error("--hover_clearance_m must be in [0.10, 0.30] meters")
    if args.motion_observe_timeout_s <= 0:
        parser.error("--motion_observe_timeout_s must be positive")
    if not 2.0 <= args.motion_arm_timeout_s <= 30.0:
        parser.error("--motion_arm_timeout_s must be in [2, 30] seconds")
    for name in (
        "prompt_box_threshold",
        "prompt_text_threshold",
        "prompt_mask_threshold",
    ):
        value = getattr(args, name)
        if value is not None and (
            not np.isfinite(value) or not 0.0 <= value <= 1.0
        ):
            parser.error(f"--{name} must be in [0, 1]")
    if args.prompt_top_k is not None and args.prompt_top_k < 1:
        parser.error("--prompt_top_k must be >= 1")
    if (
        args.prompt_reacquire_lost_frames is not None
        and args.prompt_reacquire_lost_frames < 1
    ):
        parser.error("--prompt_reacquire_lost_frames must be >= 1")
    for name in (
        "prompt_startup_timeout_s",
        "prompt_timeout_s",
        "prompt_reacquire_cooldown_s",
    ):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0.0):
            parser.error(f"--{name} must be positive")

    cfg = load_config(args.config)
    online_sam2_cfg = cfg.setdefault("online_sam2", {})
    if not isinstance(online_sam2_cfg, dict):
        parser.error("online_sam2 must be a mapping")
    if args.online_sam2 is not None:
        online_sam2_cfg["enabled"] = bool(args.online_sam2)
    if args.online_sam2_image_size is not None:
        if (
            args.online_sam2_image_size < 64
            or args.online_sam2_image_size % 16 != 0
        ):
            parser.error(
                "--online_sam2_image_size must be >= 64 and divisible by 16"
            )
        online_sam2_cfg["image_size"] = int(
            args.online_sam2_image_size
        )
    if args.online_sam2_frame_wait_ms is not None:
        if (
            not np.isfinite(args.online_sam2_frame_wait_ms)
            or not 1.0 <= args.online_sam2_frame_wait_ms <= 200.0
        ):
            parser.error(
                "--online_sam2_frame_wait_ms must be in [1, 200]"
            )
        online_sam2_cfg["frame_wait_timeout_s"] = float(
            args.online_sam2_frame_wait_ms
        ) / 1000.0
    try:
        validate_online_sam2_config(online_sam2_cfg)
    except ValueError as exc:
        parser.error(str(exc))
    prompt_cfg = cfg.setdefault("prompting", {})
    if not isinstance(prompt_cfg, dict):
        parser.error("prompting must be a mapping")
    if args.prompt_service_addr is not None:
        prompt_cfg["service_addr"] = args.prompt_service_addr
    if args.no_prompt_service_autostart:
        prompt_cfg["service_autostart"] = False
    if args.prompt_startup_timeout_s is not None:
        prompt_cfg["startup_timeout_s"] = args.prompt_startup_timeout_s
    if args.prompt_timeout_s is not None:
        prompt_cfg["request_timeout_s"] = args.prompt_timeout_s
    if args.prompt_box_threshold is not None:
        prompt_cfg["box_threshold"] = args.prompt_box_threshold
    if args.prompt_text_threshold is not None:
        prompt_cfg["text_threshold"] = args.prompt_text_threshold
    if args.prompt_mask_threshold is not None:
        prompt_cfg["mask_threshold"] = args.prompt_mask_threshold
    if args.prompt_top_k is not None:
        prompt_cfg["top_k"] = args.prompt_top_k
    if args.prompt_reacquire_lost_frames is not None:
        prompt_cfg["reacquire_lost_frames"] = args.prompt_reacquire_lost_frames
    if args.prompt_reacquire_cooldown_s is not None:
        prompt_cfg["reacquire_cooldown_s"] = args.prompt_reacquire_cooldown_s
    if args.no_prompt_auto_reacquire:
        prompt_cfg["auto_reacquire"] = False
    try:
        validate_prompting_config(prompt_cfg)
    except ValueError as exc:
        parser.error(str(exc))
    if args.publish_zmq:
        cfg["runtime"]["publish_zmq"] = True
    if args.zmq_addr:
        cfg["runtime"]["zmq_addr"] = args.zmq_addr
    if args.vis is not None:
        cfg["runtime"]["vis"] = bool(args.vis)
    if args.no_show_pcd:
        cfg["runtime"]["show_pcd"] = False
    if args.show_scene_pcd:
        cfg["runtime"]["show_scene_pcd"] = True
    if args.scene_stride is not None:
        cfg["runtime"]["scene_stride"] = args.scene_stride
    if args.scene_update_every is not None:
        cfg["runtime"]["scene_update_every"] = args.scene_update_every
    if args.goal_clearance_m is not None:
        cfg["runtime"]["goal_clearance_m"] = args.goal_clearance_m
    if args.sam2:
        cfg["sam2"]["enabled"] = True
        if cfg["tracker"].get("mode", "roi_depth") == "roi_depth":
            cfg["tracker"]["mode"] = "sam2_reinit"
    if args.no_sam2:
        cfg["sam2"]["enabled"] = False
        if cfg["tracker"].get("mode") == "sam2_reinit":
            cfg["tracker"]["mode"] = "roi_depth"
    if args.mode:
        cfg["tracker"]["mode"] = args.mode
    if args.prompt is not None:
        # Heavy semantic models live in the external service.  The 20+ Hz
        # camera process uses only the generic adaptive tracker.
        cfg["tracker"]["mode"] = "adaptive_color_depth"
        cfg["sam2"]["enabled"] = False
    if args.depth_tolerance is not None:
        cfg["tracker"]["depth_tolerance"] = args.depth_tolerance
    if args.sam2_reinit_every is not None:
        cfg["sam2"]["reinit_every"] = args.sam2_reinit_every
    if args.sam2_score_threshold is not None:
        cfg["sam2"]["score_threshold"] = args.sam2_score_threshold
    if args.save_debug_masks:
        cfg["runtime"]["save_debug_masks"] = True
    if args.debug_masks_dir:
        cfg["runtime"]["debug_masks_dir"] = args.debug_masks_dir
    if args.no_identity:
        cfg["tracker"]["identity_enabled"] = False
    if args.lock_roi:
        if args.roi is None:
            parser.error("--lock_roi requires a numeric --roi")
        cfg["tracker"]["lock_roi"] = True

    if args.enable_robot_motion and not cfg["runtime"].get("publish_zmq", False):
        parser.error("--enable_robot_motion requires --publish_zmq (or runtime.publish_zmq: true)")

    provider = ObjectPCDProvider(cfg)
    publisher = ZMQObjectPCDPublisher(cfg["runtime"]["zmq_addr"]) if cfg["runtime"].get("publish_zmq", False) else None
    motion_controller = InteractiveHoverController(
        config_path=args.config,
        zmq_addr=cfg["runtime"]["zmq_addr"],
        robot_ip=args.robot_ip,
        clearance_m=args.hover_clearance_m,
        loaded_calibration_id=provider.extrinsics.calibration_id,
        enable_robot_motion=args.enable_robot_motion,
        confirm_calibration_id=args.confirm_calibration_id,
        confirm_workspace_clear=args.confirm_workspace_clear,
        confirm_eef_clear=args.confirm_eef_clear,
        confirm_descent_clear=args.confirm_descent_clear,
        observe_timeout_s=args.motion_observe_timeout_s,
        arm_timeout_s=args.motion_arm_timeout_s,
    )
    viewer = None
    prompt_manager = None
    prompt_worker = None
    tracking_generation = 0
    last_frame = None
    show_scene_pcd = bool(cfg["runtime"].get("show_scene_pcd", False))
    scene_stride = max(1, int(cfg["runtime"].get("scene_stride", 4)))
    scene_update_every = max(1, int(cfg["runtime"].get("scene_update_every", 2)))
    scene_brightness = float(cfg["runtime"].get("scene_brightness", 0.32))
    scene_color_floor = float(cfg["runtime"].get("scene_color_floor", 0.08))
    goal_clearance_m = cfg["runtime"].get("goal_clearance_m")
    if goal_clearance_m is not None:
        goal_clearance_m = float(goal_clearance_m)

    prompt_addr = str(prompt_cfg.get("service_addr", "tcp://127.0.0.1:5557"))
    prompt_timeout_ms = int(
        1000.0 * float(prompt_cfg.get("request_timeout_s", 35.0))
    )
    prompt_box_threshold = float(prompt_cfg.get("box_threshold", 0.25))
    prompt_text_threshold = float(prompt_cfg.get("text_threshold", 0.20))
    prompt_mask_threshold = float(prompt_cfg.get("mask_threshold", 0.50))
    prompt_top_k = int(prompt_cfg.get("top_k", 5))
    prompt_auto_reacquire = bool(prompt_cfg.get("auto_reacquire", True))
    prompt_global_reacquire = bool(prompt_cfg.get("global_reacquire", True))
    prompt_replay_buffer_s = float(prompt_cfg.get("replay_buffer_s", 2.0))
    prompt_replay_max_frames = int(prompt_cfg.get("replay_max_frames", 12))
    prompt_frame_buffer = None
    if args.prompt is not None:
        camera_fps = max(1.0, float(cfg["camera"].get("fps", 30)))
        prompt_frame_buffer = RecentRGBDFrameBuffer(
            retention_s=prompt_replay_buffer_s,
            capacity_frames=max(
                2,
                int(np.ceil(prompt_replay_buffer_s * camera_fps)) + 2,
            ),
        )
    prompt_lost_frames = int(prompt_cfg.get("reacquire_lost_frames", 3))
    prompt_cooldown_s = float(prompt_cfg.get("reacquire_cooldown_s", 0.5))
    prompt_retry = PromptRetrySchedule(prompt_cooldown_s)

    try:
        if args.prompt is not None:
            prompt_manager = PromptServiceManager(
                addr=prompt_addr,
                autostart=bool(prompt_cfg.get("service_autostart", True)),
                startup_timeout_s=float(
                    prompt_cfg.get("startup_timeout_s", 45.0)
                ),
                request_timeout_ms=prompt_timeout_ms,
                launcher_path=str(prompt_service_launcher_path(prompt_cfg)),
                launcher_args=prompt_service_launcher_args(
                    prompt_cfg, prompt=args.prompt
                ),
            )
            # Finish cold model loading before opening the camera. Otherwise
            # model startup creates a blind RGB-D interval precisely when an
            # entering object is most likely to be missed.
            prompt_health = prompt_manager.start()
            validate_prompt_service_health(prompt_cfg, prompt_health)
        provider.start()
        if args.capture_only:
            frame = provider.camera.get_frame()
            output_path = Path(args.capture_only).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), frame.color_bgr):
                raise RuntimeError(f"failed to save capture to {output_path}")
            print(f"[Captured] {output_path}")
            return
        if args.prompt is not None:
            assert prompt_manager is not None
            assert prompt_frame_buffer is not None
            reference_bbox = (
                None
                if args.prompt_reference_roi is None
                else np.asarray(args.prompt_reference_roi, dtype=np.float32)
            )
            if reference_bbox is not None:
                # A reference ROI is a deliberate one-shot instance hint used
                # by the diagnostic application.  Fully automatic deployment
                # passes no reference and uses the persistent SEARCHING loop.
                frame = provider.camera.get_frame()
                last_frame = frame
                prompt_frame_buffer.append(frame)
                initial_prompt_result = prompt_manager.segment(
                    frame.color_bgr,
                    prompt=args.prompt,
                    reference_bbox_xyxy=reference_bbox,
                    request_id=f"prompt-initial-f{frame.frame_id}",
                    frame_id=frame.frame_id,
                    frame_timestamp=frame.timestamp,
                    frame_metadata={"tracker_generation": 0, "reason": "initial"},
                    box_threshold=prompt_box_threshold,
                    text_threshold=prompt_text_threshold,
                    mask_threshold=prompt_mask_threshold,
                    top_k=prompt_top_k,
                )
                print_prompt_result(initial_prompt_result, prefix="[Prompt initial]")
                if not initial_prompt_result.valid:
                    print("[Prompt][ERROR] no valid target inside the reference ROI")
                    return
            else:
                print(
                    f"[Prompt] waiting for target {args.prompt!r}; "
                    f"{str(prompt_cfg.get('detector_backend', 'grounding_dino'))} "
                    "searches camera-only and Ctrl+C cancels",
                    flush=True,
                )
                frame, initial_prompt_result = wait_for_prompt_target(
                    provider=provider,
                    prompt_manager=prompt_manager,
                    prompt=args.prompt,
                    frame_buffer=prompt_frame_buffer,
                    box_threshold=prompt_box_threshold,
                    text_threshold=prompt_text_threshold,
                    mask_threshold=prompt_mask_threshold,
                    top_k=prompt_top_k,
                    search_interval_s=float(
                        prompt_cfg.get("search_interval_s", 0.5)
                    ),
                    detector_backend=str(
                        prompt_cfg.get("detector_backend", "grounding_dino")
                    ),
                    reference_bbox_xyxy=prompt_cfg.get(
                        "search_reference_roi_xyxy"
                    ),
                    yolo_world_entry_filter_config=prompt_cfg,
                )
                last_frame = frame
            if (
                dict(initial_prompt_result.frame_metadata or {}).get(
                    "prompt_output_kind"
                )
                == "bbox_only"
            ):
                ok = provider.initialize_from_bbox(
                    frame, initial_prompt_result.bbox_xyxy
                )
            else:
                ok = provider.initialize_from_mask(
                    frame,
                    initial_prompt_result.mask,
                    bbox_xyxy=initial_prompt_result.bbox_xyxy,
                    source="grounded_sam_prompt",
                )
            if ok:
                tracking_generation = 1
                prompt_worker = AsyncPromptReacquirer(
                    addr=prompt_addr,
                    timeout_ms=prompt_timeout_ms,
                    box_threshold=prompt_box_threshold,
                    text_threshold=prompt_text_threshold,
                    mask_threshold=prompt_mask_threshold,
                    top_k=prompt_top_k,
                )
        elif args.roi is not None:
            frame = provider.camera.get_frame()
            last_frame = frame
            bbox = np.asarray(args.roi, dtype=np.float32)
            ok = provider.initialize_from_bbox(frame, bbox)
        else:
            ok = provider.select_and_initialize()
        if not ok:
            return

        if cfg["runtime"].get("show_pcd", True):
            try:
                title = (
                    "Scene + segmented object [robot_base]"
                    if show_scene_pcd
                    else "Object point cloud [robot_base]"
                )
                viewer = Open3DLiveViewer(
                    title=title,
                    point_size=float(cfg["runtime"].get("pcd_point_size", 3.0)),
                    command_bindings=OPEN3D_COMMAND_BINDINGS,
                )
            except Exception as e:
                # Avoid an automatic cv2.namedWindow fallback here: a Qt/GL
                # backend failure can abort the whole process below Python and
                # would take the safety-critical perception publisher with it.
                print(f"[PCD viewer][WARN] Open3D initialization failed; display disabled: {e}")
                viewer = None
            if show_scene_pcd:
                print(
                    "[PCD legend] dim RGB=scene | orange=segmented object | "
                    "green=center | magenta=live EEF-goal candidate | axes=robot_base"
                )
                print(
                    f"[PCD display] scene_stride={scene_stride}, "
                    f"scene_refresh~{float(cfg['camera'].get('fps', 30)) / scene_update_every:.1f}Hz, "
                    f"goal_clearance={goal_clearance_m}m"
                )

        if cfg["runtime"].get("vis", True):
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        paused = False
        last_obj = None
        last_packet = None
        max_hz = float(cfg["runtime"].get("max_packet_hz", 30))
        min_hz = float(cfg["runtime"].get("min_packet_hz", 20))
        if min_hz <= 0.0 or min_hz > max_hz:
            raise ValueError(
                f"runtime.min_packet_hz must be in (0, max_packet_hz], "
                f"got {min_hz} with max_packet_hz={max_hz}"
            )
        publish_gate = DeadlineRateGate(max_hz)

        loop_meter = FPSMeter(window=60)
        perception_rate = PacketRateMonitor(window_s=2.0, lost_timeout_s=0.25)
        publish_rate = PacketRateMonitor(window_s=2.0, lost_timeout_s=0.25)
        scene_meter = FPSMeter(window=30)
        last_reported_packet_valid = None
        last_reported_recovery_key = None

        print_key_help(prompt_mode=args.prompt is not None)
        print("[Stats] overlay shows loop FPS / total latency / stage timings. Use --print_fps for terminal logs.")

        def schedule_prompt_reacquire(reason: str, force: bool = False) -> bool:
            if args.prompt is None or prompt_worker is None or last_frame is None:
                return False
            state = getattr(provider.tracker, "state", None)
            if reason == "manual_key" and provider.tracking_committed:
                print(
                    "[Prompt] tracker is healthy; ignore T/r so an old semantic "
                    "result cannot replace it. Re-acquisition is available after "
                    "tracking becomes LOST."
                )
                return False
            if prompt_worker.busy:
                if force:
                    print("[Prompt] acquisition already running")
                return False
            now = time.monotonic()
            if not prompt_retry.ready(now, force=force):
                return False
            # An automatic prompt retry is category re-detection, not an
            # instance-continuity hint. Keeping the old bbox here strongly
            # biased GroundingDINO toward background at the pre-occlusion
            # location and prevented a moved target from ever being selected.
            automatic_global = bool(
                prompt_global_reacquire
                and reason != "manual_key"
                and state is not None
                and not provider.tracking_committed
            )
            reference_bbox = (
                None
                if state is None or automatic_global
                else np.asarray(state.bbox_xyxy, dtype=np.float32)
            )
            request_id = prompt_worker.submit(
                last_frame.color_bgr,
                prompt=args.prompt,
                reference_bbox_xyxy=reference_bbox,
                frame_id=last_frame.frame_id,
                frame_timestamp=last_frame.timestamp,
                generation=tracking_generation,
                reason=reason,
            )
            if request_id is None:
                if force:
                    print("[Prompt] acquisition already running")
                return False
            attempt = prompt_retry.mark_submitted(now)
            print(
                f"[Prompt] queued async {reason} attempt={attempt} "
                f"request={request_id} "
                f"frame={last_frame.frame_id} reference_bbox="
                f"{None if reference_bbox is None else reference_bbox.astype(int).tolist()}. "
                "This request captured the newest available frame; semantic "
                "recovery works best while the target stays visible."
            )
            return True

        def queue_prompt_retry(cause: str) -> None:
            if not prompt_auto_reacquire:
                return
            prompt_retry.request_retry(cause)
            remaining_s = prompt_retry.cooldown_remaining_s(time.monotonic())
            timing = (
                "on this camera-loop iteration"
                if remaining_s <= 0.0
                else f"after {remaining_s:.2f}s start-to-start cooldown"
            )
            print(
                f"[Prompt][RETRY] cause={cause}; attempt="
                f"{prompt_retry.attempt + 1} will capture the newest frame "
                f"{timing}. No key press is required."
            )

        def apply_prompt_response(frame) -> bool:
            nonlocal tracking_generation
            nonlocal last_obj, last_packet
            if prompt_worker is None:
                return False
            response = prompt_worker.poll()
            if response is None:
                return False
            request = response.request
            if response.error is not None:
                print(
                    f"[Prompt][WARN] request={request.request_id} failed after "
                    f"{response.elapsed_s:.1f}s: {response.error}"
                )
                queue_prompt_retry("service_error")
                return False
            result = response.result
            if result is None:
                print(f"[Prompt][WARN] request={request.request_id} returned no result")
                queue_prompt_retry("empty_response")
                return False
            print_prompt_result(result, prefix="[Prompt async]")
            if request.generation != tracking_generation:
                print(
                    f"[Prompt] discard stale generation {request.generation}; "
                    f"current={tracking_generation}"
                )
                queue_prompt_retry("stale_generation")
                return False
            echo_error = prompt_response_echo_error(
                request,
                result,
                expected_prompt=args.prompt,
            )
            if echo_error is not None:
                print(f"[Prompt][WARN] discard response: {echo_error}")
                queue_prompt_retry("metadata_rejected")
                return False
            if motion_controller.active:
                print("[Prompt] discard response while robot job is active")
                queue_prompt_retry("robot_job_active")
                return False
            if not result.valid:
                print("[Prompt] no valid semantic candidate; tracker remains fail-closed")
                queue_prompt_retry("no_valid_candidate")
                return False
            if (
                provider.tracking_committed
                and last_packet is not None
                and last_packet.valid
            ):
                print(
                    "[Prompt] provider has a committed current packet while "
                    "semantic inference was running; discard the old result"
                )
                prompt_retry.clear()
                return False
            source_age_frames = max(0, int(frame.frame_id) - request.frame_id)
            if prompt_frame_buffer is None:
                print("[Prompt][WARN] semantic RGB-D replay buffer is unavailable")
                queue_prompt_retry("semantic_replay_buffer_unavailable")
                return False
            window = prompt_frame_buffer.replay_window(
                request.frame_id,
                max_frames=prompt_replay_max_frames,
            )
            if window is None:
                head = prompt_frame_buffer.head
                print(
                    f"[Prompt] source RGB-D frame {request.frame_id} is no "
                    f"longer in the {prompt_replay_buffer_s:.1f}s replay buffer "
                    f"(head={None if head is None else head.frame_id}); reject "
                    "the stale mask and capture a new semantic request"
                )
                queue_prompt_retry("semantic_source_frame_evicted")
                return False
            source_frame = window.frames[0]
            if not np.isclose(
                float(source_frame.timestamp),
                float(request.frame_timestamp),
                rtol=0.0,
                atol=1.0e-6,
            ):
                print(
                    "[Prompt][WARN] buffered source timestamp does not match "
                    f"request frame {request.frame_id}; reject without replay"
                )
                queue_prompt_retry("semantic_source_metadata_mismatch")
                return False
            if (
                int(window.head_frame_id) != int(frame.frame_id)
                or prompt_frame_buffer.head is None
                or int(prompt_frame_buffer.head.frame_id) != int(frame.frame_id)
            ):
                print(
                    "[Prompt][WARN] replay head is not the exact current RGB-D "
                    f"frame: replay={window.head_frame_id}, current={frame.frame_id}"
                )
                queue_prompt_retry("semantic_replay_head_mismatch")
                return False

            replay = replay_prompt_mask(
                window,
                result.mask,
                cfg["tracker"],
            )
            source_age_s = max(
                0.0, float(frame.timestamp) - float(source_frame.timestamp)
            )
            print(
                f"[Prompt replay] source={request.frame_id} "
                f"head={replay.target_frame_id} age={source_age_frames}frames/"
                f"{source_age_s:.3f}s sampled={list(replay.sampled_frame_ids)} "
                f"elapsed={replay.elapsed_ms:.1f}ms valid={replay.valid} | "
                f"{replay.message}"
            )
            if not replay.valid:
                # Even if a later frame looks plausible, one invalid replay
                # update means the source-to-current identity chain is broken.
                # Keep output fail-closed and immediately request fresh
                # semantics instead of applying any old/transplanted pixels.
                queue_prompt_retry(
                    f"semantic_replay_{replay.failure_code or 'invalid'}"
                )
                return False
            if replay.target_frame_id != int(frame.frame_id):
                print(
                    "[Prompt][WARN] discard non-current replay result: "
                    f"target={replay.target_frame_id}, current={frame.frame_id}"
                )
                queue_prompt_retry("semantic_replay_result_stale")
                return False
            bbox_only = (
                dict(result.frame_metadata or {}).get("prompt_output_kind")
                == "bbox_only"
            )
            if bbox_only:
                reinitialized = provider.reinitialize_same_target_from_bbox(
                    frame,
                    replay.bbox_xyxy,
                    source=f"yolo_world_reacquire_{request.reason}",
                    allow_category_relocation=True,
                )
            else:
                reinitialized = provider.reinitialize_same_target_from_mask(
                    frame,
                    replay.mask,
                    source=f"prompt_reacquire_{request.reason}",
                    allow_category_relocation=True,
                )
            if reinitialized:
                tracking_generation += 1
                last_obj = None
                last_packet = None
                prompt_retry.clear()
                motion_controller.disarm("prompt target re-acquired")
                print(
                    f"[Prompt] prompt category target re-acquired; generation="
                    f"{tracking_generation}, source_age={source_age_frames} frames. "
                    "The pre-reset packet is suppressed; fresh PCD resumes on "
                    "the next frame."
                )
                return True
            else:
                print(
                    f"[Prompt] semantic mask from {source_age_frames} frames ago "
                    "does not match the current appearance/depth/location; rejected"
                )
                queue_prompt_retry("current_frame_gate_rejected")
            return False

        def handle_commands(commands) -> bool:
            """Execute queued UI commands on the perception/main thread.

            Return True when the application should exit.  Manual target
            selection blocks packet publication, so a robot-side stale-data
            watchdog will stop motion while the selector is open.
            """

            nonlocal paused, last_obj, last_packet, tracking_generation
            for command in commands:
                motion_controller.poll()
                if command == COMMAND_QUIT:
                    if motion_controller.active:
                        motion_controller.request_cancel()
                    return True
                if command == COMMAND_TOGGLE_PAUSE:
                    if motion_controller.active:
                        print("[Paused][DENY] press X to stop the active robot job first")
                        continue
                    paused = not paused
                    print(f"[Paused] {paused}")
                elif command == COMMAND_RESELECT:
                    if motion_controller.active:
                        print("[Tracking][DENY] press X and wait for the robot job to stop before reselecting")
                        continue
                    motion_controller.disarm("target re-selection")
                    if args.prompt is not None:
                        schedule_prompt_reacquire("manual_key", force=True)
                    else:
                        print("[Tracking] select the moved target; publication is paused until selection finishes")
                        if provider.select_and_initialize():
                            tracking_generation += 1
                            paused = False
                            last_obj = None
                            last_packet = None
                            print(
                                f"[Tracking] target reinitialized; ROI mode="
                                f"{'LOCKED' if provider.roi_locked else 'FOLLOW'}"
                            )
                        else:
                            print("[Tracking] selection cancelled; keeping the previous target")
                elif command == COMMAND_TOGGLE_LOCK:
                    if motion_controller.active:
                        print("[Tracking][DENY] ROI mode cannot change during an active robot job")
                        continue
                    motion_controller.disarm("ROI mode changed")
                    locked = provider.toggle_roi_lock()
                    print(
                        "[Tracking] ROI mode="
                        + ("LOCKED (stationary target)" if locked else "FOLLOW (moving target)")
                    )
                elif command == COMMAND_SAVE:
                    if last_packet is None:
                        print("[Saved][WARN] no object packet is available yet")
                    else:
                        save_object_pcd(args.save_dir, last_obj, last_packet)
                elif command == COMMAND_HELP:
                    print_key_help(prompt_mode=args.prompt is not None)
                elif command == COMMAND_PLAN_HOVER:
                    if publisher is None:
                        print("[Interactive hover][DENY] D requires ZMQ publication")
                    else:
                        motion_controller.start_plan(
                            packet_valid=bool(last_packet is not None and last_packet.valid)
                        )
                elif command == COMMAND_ARM_MOTION:
                    if publisher is None:
                        print("[Interactive hover][DENY] M requires ZMQ publication")
                    else:
                        motion_controller.arm_motion(
                            packet_valid=bool(last_packet is not None and last_packet.valid),
                            roi_locked=provider.roi_locked,
                        )
                elif command == COMMAND_CONFIRM_MOTION:
                    if publisher is None:
                        print("[Interactive hover][DENY] Y requires ZMQ publication")
                    else:
                        motion_controller.confirm_motion(
                            packet_valid=bool(last_packet is not None and last_packet.valid),
                            roi_locked=provider.roi_locked,
                        )
                elif command == COMMAND_CANCEL_MOTION:
                    motion_controller.request_cancel()
            return False

        def collect_commands(cv_key=255):
            commands = []
            if viewer is not None and hasattr(viewer, "drain_commands"):
                commands.extend(viewer.drain_commands())
            cv_command = command_from_cv_key(cv_key)
            if cv_command is not None:
                commands.append(cv_command)
            return commands

        while True:
            motion_controller.poll()
            if paused:
                # Keep both GUI event loops alive while perception/publishing
                # is paused; in particular, P in Open3D must be able to resume.
                if viewer is not None and hasattr(viewer, "poll_events"):
                    try:
                        if not viewer.poll_events():
                            viewer.close()
                            viewer = None
                    except Exception as e:
                        print(f"[PCD viewer][WARN] disabled after error: {e}")
                        viewer.close()
                        viewer = None
                key = (
                    cv2.waitKey(30) & 0xFF
                    if cfg["runtime"].get("vis", True)
                    or isinstance(viewer, OpenCVPointCloudViewer)
                    else 255
                )
                if key == 255 and viewer is None:
                    time.sleep(0.03)
                if handle_commands(collect_commands(key)):
                    break
                continue

            frame, mask_res, obj, packet = provider.step()
            last_frame = frame
            last_obj = obj if packet.valid else None
            last_packet = packet
            if prompt_frame_buffer is not None:
                prompt_frame_buffer.append(frame)
            prompt_reset_applied = apply_prompt_response(frame)
            if prompt_reset_applied:
                # ``packet`` and ``obj`` were computed immediately before the
                # tracker reset.  Publishing or displaying either under the new
                # generation would mix two target states.  Skip this one output
                # cycle; the robot stale-data watchdog remains authoritative.
                loop_meter.tick()
                record_packet_rates(
                    packet_valid=packet.valid,
                    perception_monitor=perception_rate,
                    publish_monitor=publish_rate,
                    published=False,
                )
                if (
                    args.max_frames is not None
                    and packet.frame_id >= args.max_frames
                ):
                    print(f"[Done] reached --max_frames={args.max_frames}")
                    break
                key = (
                    cv2.waitKey(1) & 0xFF
                    if cfg["runtime"].get("vis", True)
                    or isinstance(viewer, OpenCVPointCloudViewer)
                    else 255
                )
                if handle_commands(collect_commands(key)):
                    break
                continue
            tracker_state = getattr(provider.tracker, "state", None)
            tracker_lost_count = int(
                getattr(tracker_state, "lost_count", 0)
            )
            tracker_is_valid = bool(
                getattr(tracker_state, "valid", mask_res.valid)
            )
            tracking_committed = provider.tracking_committed
            provider_uncommitted_frames = provider.uncommitted_frames
            tracking_status = getattr(
                provider.tracker, "tracking_status", {}
            )
            pending_count = (
                tracking_status.get("recovery_pending_count", 0)
                if isinstance(tracking_status, dict)
                else 0
            )
            if last_reported_packet_valid is None or bool(packet.valid) != bool(
                last_reported_packet_valid
            ):
                state_label = "VALID" if packet.valid else "LOST"
                print(
                    f"[Tracking state] frame={packet.frame_id} {state_label} "
                    f"tracker_valid={tracker_is_valid} "
                    f"committed={tracking_committed} "
                    f"uncommitted={provider_uncommitted_frames} "
                    f"lost_count={tracker_lost_count} "
                    f"recovery_pending={pending_count} | {packet.message}"
                )
                last_reported_packet_valid = bool(packet.valid)
            if not tracker_is_valid:
                global_after = max(
                    1,
                    int(
                        cfg["tracker"].get(
                            "occlusion_global_search_after_frames", 2
                        )
                    ),
                )
                recovery_mode = (
                    "GLOBAL"
                    if bool(
                        cfg["tracker"].get(
                            "occlusion_global_search_enabled", True
                        )
                    )
                    and not provider.roi_locked
                    and tracker_lost_count >= global_after
                    else "LOCAL"
                )
                recovery_key = (recovery_mode, int(pending_count))
                if recovery_key != last_reported_recovery_key:
                    print(
                        f"[Recovery] frame={packet.frame_id} "
                        f"mode={recovery_mode} lost_count={tracker_lost_count} "
                        f"confirm={pending_count}/"
                        f"{int(cfg['tracker'].get('occlusion_recovery_confirm_frames', 2))} "
                        f"| {mask_res.message}"
                    )
                    last_reported_recovery_key = recovery_key
            else:
                last_reported_recovery_key = None
            if tracking_committed and packet.valid:
                # A transient semantic failure must not arm an early retry for
                # some unrelated future loss after the lightweight tracker has
                # already recovered on its own.
                prompt_retry.clear()
            if (
                args.prompt is not None
                and prompt_auto_reacquire
                and not tracking_committed
                and (
                    prompt_retry.retry_pending
                    or max(
                        tracker_lost_count, provider_uncommitted_frames
                    )
                    >= prompt_lost_frames
                )
                and not motion_controller.active
            ):
                schedule_prompt_reacquire(
                    "automatic_retry"
                    if prompt_retry.retry_pending
                    else "automatic_lost"
                )
            if motion_controller.armed and not packet.valid:
                motion_controller.disarm("target packet became invalid")
            # No explicit ``dt`` here: the interval between successive step
            # completions includes the previous frame's publication and GUI
            # work, so this is the actual end-to-end throughput seen by users.
            loop_meter.tick()
            if args.max_frames is not None and packet.frame_id >= args.max_frames:
                print(f"[Done] reached --max_frames={args.max_frames}")
                break

            published = False
            if publisher is not None and publish_gate.ready():
                publisher.publish(packet)
                published = True
            record_packet_rates(
                packet_valid=packet.valid,
                perception_monitor=perception_rate,
                publish_monitor=publish_rate,
                published=published,
            )

            if publisher is None:
                publish_status = "OFF"
            else:
                publish_status = publish_rate.status(
                    min_valid_hz=min_hz, warmup_events=30
                )
            perception_status = perception_rate.status(
                min_valid_hz=min_hz, warmup_events=30
            )

            timings = getattr(provider, "last_timings_ms", {})
            if args.print_fps and packet.frame_id % max(1, args.print_every) == 0:
                print(
                    f"[FPS] frame={packet.frame_id} valid={packet.valid} "
                    f"loop={loop_meter.avg_fps:.1f}Hz ({loop_meter.avg_ms:.1f}ms) "
                    f"pub={publish_rate.publish_hz:.1f}Hz:{publish_status} "
                    f"valid_pcd={perception_rate.valid_hz:.1f}Hz "
                    f"valid_ratio={perception_rate.valid_ratio:.2f} "
                    f"target>={min_hz:.1f}Hz:{perception_status} "
                    f"scene={scene_meter.avg_fps:.1f}Hz "
                    f"camera={timings.get('camera', 0):.1f}ms "
                    f"tracker={timings.get('tracker', 0):.1f}ms "
                    f"sam2={timings.get('sam2', 0):.1f}ms "
                    f"pcd={timings.get('pcd', 0):.1f}ms "
                    f"total={timings.get('total', 0):.1f}ms "
                    f"mask_source={packet.debug.get('mask_source', 'unknown') if packet.debug else 'unknown'} "
                    f"mask_area={packet.debug.get('mask_area', 0) if packet.debug else 0} "
                    f"motion_iou={packet.debug.get('motion_mask_iou', 0.0) if packet.debug else 0.0:.3f} "
                    f"temporal={packet.debug.get('temporal_motion_source', 'none') if packet.debug else 'none'}:"
                    f"{'ON' if packet.debug and packet.debug.get('temporal_mask_stabilized', False) else 'OFF'} "
                    f"sam_area={packet.debug.get('online_sam2_mask_area', 0) if packet.debug else 0} "
                    f"sam_status={packet.debug.get('online_sam2_status', 'unknown') if packet.debug else 'unknown'} "
                    f"sam_exact={packet.debug.get('online_sam2_exact_count', 0) if packet.debug else 0}/"
                    f"{packet.debug.get('online_sam2_eligible_count', 0) if packet.debug else 0}="
                    f"{packet.debug.get('online_sam2_exact_frame_ratio', 0.0) if packet.debug else 0.0:.2f} "
                    f"sam_submit={packet.debug.get('online_sam2_submit_count', 0) if packet.debug else 0} "
                    f"sam_busy_skip={packet.debug.get('online_sam2_busy_skip_count', 0) if packet.debug else 0} "
                    f"sam_late={packet.debug.get('online_sam2_late_count', 0) if packet.debug else 0} "
                    f"sam_deadline_miss={packet.debug.get('online_sam2_deadline_miss_count', 0) if packet.debug else 0} "
                    f"raw_points={packet.debug.get('raw_points', 0) if packet.debug else 0}"
                )
            if args.print_center and packet.frame_id % max(1, args.print_every) == 0:
                center = (
                    "None"
                    if packet.center is None
                    else np.array2string(packet.center, precision=4)
                )
                z_p95 = None if packet.debug is None else packet.debug.get("visible_cloud_z_p95")
                z_text = "None" if z_p95 is None else f"{float(z_p95):.4f}"
                print(
                    f"[Center] frame={packet.frame_id} valid={packet.valid} "
                    f"frame_name={packet.reference_frame} visible_median={center} "
                    f"z95={z_text} calibration_id={packet.calibration_id}"
                )

            if cfg["runtime"].get("vis", True):
                debug_img = provider.make_debug_image(frame, mask_res, packet)
                prompt_status_line = (
                    "prompt OFF"
                    if args.prompt is None
                    else (
                        f"prompt {args.prompt!r} | semantic worker "
                        f"{'BUSY' if prompt_worker is not None and prompt_worker.busy else 'IDLE'} "
                        f"| generation {tracking_generation} | lost {tracker_lost_count} "
                        f"| uncommitted {provider_uncommitted_frames}"
                    )
                )
                target_key_text = (
                    "T/r manual retry (auto recovery ON)"
                    if args.prompt is not None
                    else "T/r reselect"
                )
                lines = [
                    f"loop {loop_meter.avg_fps:.1f} Hz | {loop_meter.avg_ms:.1f} ms | "
                    f"pub {publish_rate.publish_hz:.1f} Hz ({publish_status}) | valid PCD "
                    f"{perception_rate.valid_hz:.1f} Hz >= {min_hz:.1f}: {perception_status}",
                    f"camera {timings.get('camera', 0):.1f} ms | tracker {timings.get('tracker', 0):.1f} ms | sam2 {timings.get('sam2', 0):.1f} ms | pcd {timings.get('pcd', 0):.1f} ms",
                    f"mask {packet.debug.get('mask_area', 0) if packet.debug else 0} px "
                    f"[{packet.debug.get('mask_source', 'unknown') if packet.debug else 'unknown'}] "
                    f"| motion IoU {packet.debug.get('motion_mask_iou', 0.0) if packet.debug else 0.0:.3f} "
                    f"{packet.debug.get('temporal_motion_source', 'none') if packet.debug else 'none'} "
                    f"{'T-STABLE' if packet.debug and packet.debug.get('temporal_mask_stabilized', False) else 'RAW'} "
                    f"| SAM {packet.debug.get('online_sam2_mask_area', 0) if packet.debug else 0} px "
                    f"| points {packet.debug.get('raw_points', 0) if packet.debug else 0} "
                    f"| valid {packet.valid}",
                    (
                        f"ROI {'LOCKED' if provider.roi_locked else 'FOLLOW'} | "
                        f"{target_key_text} | L lock | D plan | M,Y move | X cancel"
                    ),
                    prompt_status_line,
                    f"robot {motion_controller.status}",
                    (
                        f"visible center [{packet.reference_frame}] "
                        f"{np.array2string(packet.center, precision=4) if packet.center is not None else 'None'} "
                        f"| z95 {packet.debug.get('visible_cloud_z_p95') if packet.debug else None}"
                    ),
                ]
                debug_img = put_lines(
                    debug_img,
                    lines,
                    org=(12, frame.color_bgr.shape[0] - 166),
                    scale=0.48,
                )
                cv2.imshow(WIN, debug_img)

            if viewer is not None:
                try:
                    if show_scene_pcd:
                        scene_points = None
                        scene_colors = None
                        if packet.frame_id % scene_update_every == 0:
                            exclude_mask = (
                                mask_res.mask
                                if (
                                    packet.valid
                                    and mask_res is not None
                                    and mask_res.valid
                                )
                                else None
                            )
                            scene = provider.extractor.extract_scene(
                                frame,
                                exclude_mask=exclude_mask,
                                stride=scene_stride,
                            )
                            scene_meter.tick()
                            scene_points = scene.points
                            scene_colors = scene.colors

                        object_points = (
                            obj.points
                            if packet.valid and obj is not None and obj.valid
                            else np.zeros((0, 3), dtype=np.float32)
                        )
                        center = packet.center if packet.valid else None
                        goal = None
                        z_p95 = (
                            packet.debug.get("visible_cloud_z_p95")
                            if packet.debug is not None
                            else None
                        )
                        if (
                            center is not None
                            and z_p95 is not None
                            and goal_clearance_m is not None
                        ):
                            goal = np.asarray(
                                [center[0], center[1], float(z_p95) + goal_clearance_m],
                                dtype=np.float32,
                            )
                        alive = viewer.update_composite(
                            scene_points,
                            scene_colors,
                            object_points,
                            center=center,
                            goal=goal,
                            scene_brightness=scene_brightness,
                            scene_color_floor=scene_color_floor,
                        )
                    else:
                        object_points = (
                            obj.points
                            if packet.valid and obj is not None and obj.valid
                            else np.zeros((0, 3), dtype=np.float32)
                        )
                        object_colors = (
                            obj.colors
                            if packet.valid and obj is not None and obj.valid
                            else None
                        )
                        alive = viewer.update(object_points, object_colors)
                    if not alive:
                        viewer.close()
                        viewer = None
                except Exception as e:
                    # Visualization is deliberately fail-open: perception keeps
                    # publishing, while the motion-side stale-data watchdog stays
                    # authoritative for robot safety.
                    print(f"[PCD viewer][WARN] disabled after error: {e}")
                    viewer.close()
                    viewer = None

            key = (
                cv2.waitKey(1) & 0xFF
                if cfg["runtime"].get("vis", True)
                or isinstance(viewer, OpenCVPointCloudViewer)
                else 255
            )
            if handle_commands(collect_commands(key)):
                break

    finally:
        # Stop the libfranka-owning child while fresh perception is still being
        # published.  Only after it exits may the publisher/camera be closed.
        motion_controller.shutdown()
        if viewer is not None:
            viewer.close()
        if publisher is not None:
            publisher.close()
        provider.stop()
        if prompt_worker is not None:
            # Mark the worker closed first.  If this application owns the model
            # service, stopping it below interrupts a long ZMQ request without
            # ever closing a worker-owned socket from the wrong thread.
            prompt_worker.request_close()
        if prompt_manager is not None:
            try:
                prompt_manager.close()
            except Exception as e:
                print(f"[Prompt][WARN] service cleanup failed: {e}")
        if prompt_worker is not None:
            prompt_worker.close(join_timeout_s=1.0)
        if cfg["runtime"].get("vis", True):
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
