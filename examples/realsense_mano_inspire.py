#!/usr/bin/env python3
"""RealSense -> WiLoR/MANO -> Inspire RH56BFX-2R real-time pipeline.

The default mode is preview-only and never opens a serial port.  Hardware
output requires explicit motion flags, port, calibration, retargeter, and axes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import signal
import sys
import time
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from inspire_mano_pipeline.calibration import PipelineCalibration
from inspire_mano_pipeline.camera import RealSenseSource
from inspire_mano_pipeline.model import CameraFrame, HARDWARE_JOINTS, RetargetOutput
from inspire_mano_pipeline.retargeting import (
    DEFAULT_DEX_ROOT,
    DexInspireRetargeter,
    GeometricInspireRetargeter,
)
from inspire_mano_pipeline.rh56_stream import (
    SafeRH56Stream,
    StreamState,
    TargetFrame,
)
from inspire_mano_pipeline.visualize import draw_overlay
from inspire_mano_pipeline.wilor_backend import (
    DEFAULT_WILOR_ROOT,
    WiLoRBackend,
    detector_label_for_physical_hand,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION = (
    SCRIPT_DIR / "inspire_mano_pipeline/inspire_rh56bfx_right.json"
)
NO_HAND_OPEN_CONFIRMATION = "CALIBRATED_OPEN"
SIX_DOF_CONFIRMATION = "RH56_SIX_DOF_REALTIME"
CURRENT_MONITOR_ONLY_CONFIRMATION = "RH56_ACTIVE_CURRENT_MONITOR_ONLY"
NO_HAND_FALLBACK_DELAY_SECONDS = 0.30
CALIBRATED_OPEN_START_TOLERANCE_UNITS = 30


class HardwareStopUnconfirmed(RuntimeError):
    """The program could not write and read back the final six-axis -1."""


class ImageSource:
    def __init__(self, path: Path, mirror: bool = False) -> None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not read image: {path}")
        self.image = cv2.flip(image, 1) if mirror else image
        self.frame_number = 0

    def start(self) -> None:
        return None

    def read(self, timeout_ms: int = 0) -> CameraFrame:
        del timeout_ms
        self.frame_number += 1
        return CameraFrame(
            color_bgr=self.image.copy(),
            depth_m=None,
            captured_at_monotonic=time.monotonic(),
            frame_number=self.frame_number,
        )

    def stop(self) -> None:
        return None


def parse_axes(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one axis is required")
    unknown = set(result).difference(HARDWARE_JOINTS)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown axes: {sorted(unknown)}")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("axis names must not be repeated")
    return result


def parse_operator_roi(value: str) -> tuple[float, float, float, float]:
    try:
        result = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "operator ROI must be x1,y1,x2,y2 normalized to 0..1"
        ) from exc
    if (
        len(result) != 4
        or not np.all(np.isfinite(result))
        or not (0.0 <= result[0] < result[2] <= 1.0)
        or not (0.0 <= result[1] < result[3] <= 1.0)
    ):
        raise argparse.ArgumentTypeError(
            "operator ROI must satisfy 0<=x1<x2<=1 and 0<=y1<y2<=1"
        )
    return result


def tracking_display_status(
    detection,
    diagnostics,
    calibration: PipelineCalibration,
) -> str:
    """Return a concise, operator-facing reason for the current control state."""

    if detection is not None:
        depth = getattr(detection, "control_palm_depth_m", None)
        if depth is None:
            depth = detection.palm_depth_m
        if depth is None:
            reason = getattr(detection, "palm_depth_reason", None)
            if not reason or reason in {"invalid", "not_estimated"}:
                return "DEPTH INVALID"
            return f"DEPTH INVALID: {reason.upper()}"
        if getattr(detection, "palm_depth_source", "measured") == "held":
            age_ms = 1000.0 * float(
                getattr(detection, "palm_depth_age_seconds", 0.0) or 0.0
            )
            return f"DEPTH HELD {age_ms:.0f}ms / CONTROL INHIBITED"
        if depth < calibration.min_palm_depth_m:
            return (
                f"DEPTH TOO NEAR {depth:.2f}m "
                f"(< {calibration.min_palm_depth_m:.2f}m)"
            )
        if depth > calibration.max_palm_depth_m:
            return (
                f"DEPTH TOO FAR {depth:.2f}m "
                f"(> {calibration.max_palm_depth_m:.2f}m)"
            )
        return "TRACKING OK"

    result = diagnostics.get("result") if isinstance(diagnostics, Mapping) else None
    labels = {
        "no_candidates": "NO HAND",
        "handedness_mismatch": "WRONG HAND / RIGHT REQUIRED",
        "multiple_hands": "MULTIPLE HANDS",
        "outside_operator_roi": "HAND OUTSIDE OPERATOR ROI",
        "invalid_keypoint_shape": "MANO INVALID KEYPOINTS",
        "nonfinite_keypoints": "MANO NON-FINITE",
        "degenerate_canonical_geometry": "MANO DEGENERATE GEOMETRY",
        "implausible_palm_width": "MANO IMPLAUSIBLE SCALE",
    }
    return labels.get(result, "NO VALID RIGHT HAND")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "RealSense BGR -> WiLoR MANO -> dexsuite Inspire retargeting. "
            "Preview-only unless hardware is explicitly enabled."
        )
    )
    parser.add_argument("--source", choices=("realsense", "image"), default="realsense")
    parser.add_argument("--image", type=Path, help="input image for --source image")
    parser.add_argument("--mirror-input", action="store_true", help="horizontally mirror --source image")
    parser.add_argument("--camera-serial")
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--operator-roi",
        type=parse_operator_roi,
        help=(
            "normalized x1,y1,x2,y2 region containing only the operator hand; "
            "useful when the camera also sees the Inspire hand"
        ),
    )
    parser.add_argument("--device", default="auto", help="auto, cuda:0, or cpu")
    parser.add_argument("--wilor-root", type=Path, default=DEFAULT_WILOR_ROOT)
    parser.add_argument("--dex-root", type=Path, default=DEFAULT_DEX_ROOT)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--retargeter", choices=("dex", "geometric"))
    parser.add_argument("--allow-multiple-right-hands", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0, help="seconds; 0 runs until q/Ctrl-C")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")

    hardware = parser.add_argument_group("hardware output (causes motion)")
    hardware.add_argument("--enable-hardware", action="store_true")
    hardware.add_argument("--confirm-hardware-motion", action="store_true")
    hardware.add_argument("--port")
    hardware.add_argument("--hand-id", type=int, default=1)
    hardware.add_argument(
        "--axes",
        type=parse_axes,
        default=None,
        help="comma-separated RH56 axes; thumb_rotate is disabled by default",
    )
    hardware.add_argument(
        "--no-hand-policy",
        choices=("disable", "open"),
        default=None,
        help=(
            "hardware behavior without a depth-valid physical right hand; "
            "default: disable"
        ),
    )
    hardware.add_argument(
        "--confirm-no-hand-open",
        help=(
            "exact token CALIBRATED_OPEN required for hardware fallback-open"
        ),
    )
    hardware.add_argument(
        "--confirm-wide-range",
        help="exact token RH56_WIDE_RANGE required outside commissioning mode",
    )
    hardware.add_argument(
        "--confirm-six-dof-motion",
        help=(
            "exact token RH56_SIX_DOF_REALTIME required when thumb_rotate is "
            "controlled together with bend axes"
        ),
    )
    hardware.add_argument(
        "--confirm-current-monitor-only",
        help=(
            "exact token RH56_ACTIVE_CURRENT_MONITOR_ONLY required when ACTIVE "
            "host current thresholds are telemetry-only"
        ),
    )
    return parser


def is_safe_commissioning_profile(
    calibration: PipelineCalibration, axes: Sequence[str]
) -> bool:
    """Content-based gate for the tokenless, first-motion profile."""

    if tuple(axes) != ("index",):
        return False
    index = calibration.axes["index"]
    return all(
        (
            calibration.detector_confidence >= 0.5,
            calibration.tracking_timeout_seconds <= 0.75,
            calibration.speed <= 80,
            calibration.force_limit <= 80,
            calibration.valid_frames_to_arm >= 5,
            calibration.arming_max_frame_gap_seconds <= 0.25,
            calibration.arming_max_target_delta_units <= 25,
            calibration.control_hz >= 20.0,
            calibration.feedback_hz >= 5.0,
            calibration.require_depth_for_hardware,
            calibration.min_palm_depth_m >= 0.15,
            calibration.max_palm_depth_m <= 1.50,
            calibration.preflight_stability_seconds >= 0.15,
            calibration.preflight_max_angle_delta <= 5,
            calibration.preflight_max_idle_current_ma <= 300,
            calibration.active_current_policy == "fault",
            calibration.stream_max_current_ma <= 600,
            index.enabled,
            abs(index.command_open - index.command_closed) <= 200,
            index.max_rate_units_per_second <= 80,
            all(
                axis.feedback_to_command_offset_units == 0
                and axis.feedback_to_command_valid_min is None
                and axis.feedback_to_command_valid_max is None
                for axis in calibration.axes.values()
            ),
            all(axis.hardware_speed is None for axis in calibration.axes.values()),
            not calibration.axes["thumb_rotate"].enabled,
        )
    )


def is_safe_thumb_rotate_commissioning_profile(
    calibration: PipelineCalibration, axes: Sequence[str]
) -> bool:
    """Content gate for the first live motion of the unverified sixth axis."""

    if tuple(axes) != ("thumb_rotate",):
        return False
    rotate = calibration.axes["thumb_rotate"]
    enabled = tuple(
        name for name in HARDWARE_JOINTS if calibration.axes[name].enabled
    )
    return all(
        (
            enabled == ("thumb_rotate",),
            calibration.detector_confidence >= 0.5,
            calibration.tracking_timeout_seconds <= 0.75,
            calibration.speed <= 80,
            calibration.force_limit <= 80,
            calibration.valid_frames_to_arm >= 5,
            calibration.arming_max_frame_gap_seconds <= 0.25,
            calibration.arming_max_target_delta_units <= 15,
            calibration.control_hz >= 20.0,
            calibration.feedback_hz >= 5.0,
            calibration.require_depth_for_hardware,
            calibration.min_palm_depth_m >= 0.15,
            calibration.max_palm_depth_m <= 1.50,
            calibration.preflight_stability_seconds >= 0.15,
            calibration.preflight_max_angle_delta <= 5,
            calibration.preflight_max_idle_current_ma <= 200,
            calibration.active_current_policy == "fault",
            calibration.stream_max_current_ma <= 400,
            calibration.stream_max_total_current_ma <= 400,
            rotate.enabled,
            rotate.command_open == 900,
            rotate.command_closed == 800,
            rotate.max_rate_units_per_second <= 40,
            tuple(
                calibration.axes[name].feedback_to_command_offset_units
                for name in HARDWARE_JOINTS
            ) == (0, 0, 0, 0, 0, 15),
            rotate.feedback_to_command_valid_min == 840,
            rotate.feedback_to_command_valid_max == 870,
            all(
                calibration.axes[name].feedback_to_command_valid_min is None
                and calibration.axes[name].feedback_to_command_valid_max is None
                for name in HARDWARE_JOINTS[:-1]
            ),
            all(axis.hardware_speed is None for axis in calibration.axes.values()),
            calibration.temporal_filter_axes == ("thumb_rotate",),
            calibration.temporal_median_window >= 3,
            calibration.temporal_ema_alpha <= 0.65,
        )
    )


def is_safe_six_dof_open1000_profile(
    calibration: PipelineCalibration, axes: Sequence[str]
) -> bool:
    """Strict content gate for the first joint six-DOF realtime profile."""

    if tuple(axes) != HARDWARE_JOINTS:
        return False
    expected = {
        "pinky": (0.05, 0.55, 1000, 0, 200.0, None),
        "ring": (0.05, 0.58, 1000, 0, 200.0, None),
        "middle": (0.05, 0.47, 1000, 0, 200.0, None),
        "index": (0.08, 0.40, 1000, 0, 200.0, None),
        "thumb_bend": (0.0, 0.60, 1000, 0, 160.0, None),
        "thumb_rotate": (0.20, 0.95, 1000, 900, 40.0, 80),
    }
    enabled = tuple(
        name for name in HARDWARE_JOINTS if calibration.axes[name].enabled
    )
    axis_content_matches = all(
        (
            axis.q_open,
            axis.q_closed,
            axis.command_open,
            axis.command_closed,
            axis.max_rate_units_per_second,
            axis.hardware_speed,
        )
        == expected[name]
        for name, axis in calibration.axes.items()
    )
    rotate = calibration.axes["thumb_rotate"]
    feedback_mapping_matches = all(
        calibration.axes[name].feedback_to_command_offset_units == 0
        and calibration.axes[name].feedback_to_command_valid_min is None
        and calibration.axes[name].feedback_to_command_valid_max is None
        for name in HARDWARE_JOINTS[:-1]
    ) and (
        rotate.feedback_to_command_offset_units == 15
        and rotate.feedback_to_command_valid_min == 885
        and rotate.feedback_to_command_valid_max == 1000
    )
    return all(
        (
            enabled == HARDWARE_JOINTS,
            axis_content_matches,
            feedback_mapping_matches,
            calibration.detector_confidence >= 0.5,
            calibration.tracking_timeout_seconds <= 0.75,
            calibration.valid_frames_to_arm >= 5,
            calibration.arming_max_frame_gap_seconds <= 0.25,
            calibration.arming_max_target_delta_units <= 25,
            calibration.speed == 200,
            calibration.force_limit <= 80,
            calibration.control_hz >= 20.0,
            calibration.feedback_hz >= 10.0,
            calibration.require_depth_for_hardware,
            calibration.min_palm_depth_m >= 0.15,
            calibration.max_palm_depth_m <= 1.50,
            calibration.preflight_stability_seconds >= 0.15,
            calibration.preflight_max_angle_delta <= 5,
            calibration.preflight_max_idle_current_ma <= 100,
            calibration.active_current_policy == "monitor_only",
            calibration.stream_max_current_ma == 400,
            calibration.stream_max_total_current_ma == 600,
            calibration.temporal_filter_axes == HARDWARE_JOINTS,
            calibration.temporal_median_window >= 3,
            calibration.temporal_ema_alpha <= 0.60,
        )
    )


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    calibration_was_explicit = args.calibration is not None
    retargeter_was_explicit = args.retargeter is not None
    axes_were_explicit = args.axes is not None
    no_hand_policy_was_explicit = args.no_hand_policy is not None
    if args.calibration is None:
        args.calibration = DEFAULT_CALIBRATION
    if args.retargeter is None:
        args.retargeter = "geometric"
    if args.axes is None:
        args.axes = parse_axes("pinky,ring,middle,index,thumb_bend")
    if args.no_hand_policy is None:
        args.no_hand_policy = "disable"
    if args.source == "image" and args.image is None:
        parser.error("--source image requires --image")
    if args.source == "realsense" and args.image is not None:
        parser.error("--image is only valid with --source image")
    if args.mirror_input and args.source != "image":
        parser.error("--mirror-input is only valid with --source image")
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("camera width, height and fps must be positive")
    if args.duration < 0 or args.max_frames < 0:
        parser.error("duration and max-frames cannot be negative")
    if args.source == "image" and args.headless and not args.duration and not args.max_frames:
        parser.error("headless image input requires --duration or --max-frames")
    if args.enable_hardware:
        if not args.confirm_hardware_motion:
            parser.error(
                "hardware output also requires --confirm-hardware-motion"
            )
        if args.source != "realsense":
            parser.error("hardware output is only allowed with live RealSense input")
        if not args.port:
            parser.error("hardware output requires an explicit --port")
        if not args.camera_serial:
            parser.error("hardware output requires an explicit --camera-serial")
        cuda_parts = args.device.split(":")
        if not (
            len(cuda_parts) == 2
            and cuda_parts[0] == "cuda"
            and cuda_parts[1].isdigit()
        ):
            parser.error(
                "hardware output requires explicit --device cuda:N; "
                "auto/cpu inference is too slow for the commissioning frame-gap "
                "and watchdog limits"
            )
        if not calibration_was_explicit:
            parser.error("hardware output requires an explicit --calibration")
        if not retargeter_was_explicit:
            parser.error("hardware output requires an explicit --retargeter")
        if not axes_were_explicit:
            parser.error("hardware output requires explicit --axes")
        if args.allow_multiple_right_hands:
            parser.error("hardware output forbids --allow-multiple-right-hands")
        if args.no_record:
            parser.error("hardware output requires recording; --no-record is preview-only")
        if args.retargeter != "geometric":
            parser.error(
                "dex hardware output is disabled until live WiLoR/MANO gesture "
                "calibration is completed; use --retargeter geometric"
            )
        try:
            gate_calibration = PipelineCalibration.load(args.calibration)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            parser.error(f"invalid hardware calibration: {exc}")
        disabled_requested = tuple(
            name for name in args.axes if not gate_calibration.axes[name].enabled
        )
        if disabled_requested:
            parser.error(
                "requested hardware axes are disabled by calibration: "
                + ",".join(disabled_requested)
            )
        index_commissioning = is_safe_commissioning_profile(
            gate_calibration, args.axes
        )
        thumb_rotate_commissioning = is_safe_thumb_rotate_commissioning_profile(
            gate_calibration, args.axes
        )
        six_dof_realtime = is_safe_six_dof_open1000_profile(
            gate_calibration, args.axes
        )
        monitor_only_current = (
            gate_calibration.active_current_policy == "monitor_only"
        )
        if monitor_only_current:
            if not six_dof_realtime:
                parser.error(
                    "monitor-only ACTIVE current thresholds are restricted to "
                    "the exact content-safe six-DOF open1000 profile"
                )
            if (
                args.confirm_current_monitor_only
                != CURRENT_MONITOR_ONLY_CONFIRMATION
            ):
                parser.error(
                    "monitor-only ACTIVE current thresholds require "
                    "--confirm-current-monitor-only "
                    f"{CURRENT_MONITOR_ONLY_CONFIRMATION}"
                )
        elif args.confirm_current_monitor_only is not None:
            parser.error(
                "--confirm-current-monitor-only requires the exact monitor-only "
                "six-DOF hardware profile"
            )
        joint_thumb_control = (
            "thumb_rotate" in args.axes and len(args.axes) > 1
        )
        if joint_thumb_control:
            if not six_dof_realtime:
                parser.error(
                    "joint thumb-rotation hardware output requires the exact "
                    "content-safe six-DOF open1000 profile and canonical six-axis list"
                )
            if args.confirm_six_dof_motion != SIX_DOF_CONFIRMATION:
                parser.error(
                    "joint six-DOF hardware output requires "
                    f"--confirm-six-dof-motion {SIX_DOF_CONFIRMATION}"
                )
        elif args.confirm_six_dof_motion is not None:
            parser.error(
                "--confirm-six-dof-motion requires the exact joint six-DOF profile"
            )
        commissioning = index_commissioning or thumb_rotate_commissioning
        if args.no_hand_policy == "open":
            if not no_hand_policy_was_explicit:
                parser.error(
                    "hardware fallback-open output requires explicit "
                    "--no-hand-policy open"
                )
            if args.confirm_no_hand_open != NO_HAND_OPEN_CONFIRMATION:
                parser.error(
                    "hardware fallback-open output requires "
                    "--confirm-no-hand-open CALIBRATED_OPEN"
                )
            if not (index_commissioning or six_dof_realtime):
                parser.error(
                    "hardware fallback-open is restricted to the content-safe "
                    "index commissioning or exact six-DOF open1000 profile"
                )
        elif args.confirm_no_hand_open is not None:
            parser.error(
                "--confirm-no-hand-open requires --no-hand-policy open"
            )
        if not commissioning and args.confirm_wide_range != "RH56_WIDE_RANGE":
            parser.error(
                "non-commissioning hardware output requires "
                "--confirm-wide-range RH56_WIDE_RANGE"
            )
        if commissioning and not (0.0 < args.duration <= 60.0):
            parser.error(
                "commissioning hardware output requires --duration in (0, 60] seconds"
            )
    elif args.confirm_hardware_motion:
        parser.error("--confirm-hardware-motion requires --enable-hardware")
    elif args.confirm_wide_range:
        parser.error("--confirm-wide-range requires --enable-hardware")
    elif args.confirm_six_dof_motion is not None:
        parser.error("--confirm-six-dof-motion requires --enable-hardware")
    elif args.confirm_current_monitor_only is not None:
        parser.error("--confirm-current-monitor-only requires --enable-hardware")
    elif args.confirm_no_hand_open is not None:
        parser.error("--confirm-no-hand-open requires hardware fallback-open")


def mask_targets(targets: Sequence[int], selected_axes: Sequence[str]) -> tuple[int, ...]:
    selected = set(selected_axes)
    return tuple(
        int(value) if name in selected else -1
        for name, value in zip(HARDWARE_JOINTS, targets)
    )


def open_fallback_targets(
    calibration: PipelineCalibration, selected_axes: Sequence[str]
) -> tuple[int, ...]:
    """Return a real RH56 open command without synthesizing MANO/qpos output."""

    selected = set(selected_axes)
    return tuple(
        calibration.axes[name].command_open if name in selected else -1
        for name in HARDWARE_JOINTS
    )


class NoHandOpenController:
    """Gate calibrated-open fallback and stable MANO reacquisition.

    This class only selects a source; it never publishes to the serial worker.
    In particular, fallback cannot be selected before valid MANO has already
    armed the worker and made ``stream_ever_active`` true.
    """

    def __init__(
        self,
        calibration: PipelineCalibration,
        selected_axes: Sequence[str],
        policy: str,
        fallback_delay_seconds: float = NO_HAND_FALLBACK_DELAY_SECONDS,
    ) -> None:
        if policy not in ("disable", "open"):
            raise ValueError(f"unknown no-hand policy: {policy}")
        self.calibration = calibration
        self.selected_axes = set(selected_axes)
        self.policy = policy
        self.fallback_delay_seconds = float(fallback_delay_seconds)
        if self.fallback_delay_seconds <= 0:
            raise ValueError("fallback_delay_seconds must be positive")
        self.fallback_active = False
        self.last_valid_mano_capture: Optional[float] = None
        self._reacquire_count = 0
        self._last_reacquire_capture: Optional[float] = None
        self._last_reacquire_targets: Optional[tuple[int, ...]] = None
        self._reacquisition_release_pending = False

    def _reset_reacquisition(self) -> None:
        self._reacquire_count = 0
        self._last_reacquire_capture = None
        self._last_reacquire_targets = None
        self._reacquisition_release_pending = False

    def note_mano_submit_result(self, accepted: bool) -> None:
        """Commit a fallback-to-MANO transition only after stream acceptance."""

        if not self._reacquisition_release_pending:
            return
        if accepted:
            self.fallback_active = False
            self._reset_reacquisition()
        else:
            # A stale, duplicate, or stopping frame is not evidence of a stable
            # accepted reacquisition sequence.  Stay open and collect a new run.
            self._reset_reacquisition()

    def choose(
        self,
        *,
        frame_captured_at_monotonic: float,
        detection_present: bool,
        depth_in_calibrated_range: bool,
        mano_targets: Optional[Sequence[int]],
        stream_ever_active: bool,
    ) -> str:
        captured_at = float(frame_captured_at_monotonic)
        if not np.isfinite(captured_at):
            raise ValueError("frame capture timestamp must be finite")

        if self.policy == "disable":
            usable = detection_present and (
                depth_in_calibrated_range
                or not self.calibration.require_depth_for_hardware
            )
            return "mano" if usable else "none"

        physical_mano_valid = detection_present and depth_in_calibrated_range
        if not physical_mano_valid:
            self._reset_reacquisition()
            if self.fallback_active:
                return "fallback_open"
            if (
                stream_ever_active
                and self.last_valid_mano_capture is not None
                and captured_at - self.last_valid_mano_capture
                >= self.fallback_delay_seconds
            ):
                self.fallback_active = True
                return "fallback_open"
            return "none"

        if mano_targets is None:
            raise ValueError("valid MANO control requires target values")
        targets = tuple(int(value) for value in mano_targets)
        if len(targets) != len(HARDWARE_JOINTS):
            raise ValueError("MANO control requires six target values")
        self.last_valid_mano_capture = captured_at
        if not self.fallback_active:
            return "mano"

        within_gap = (
            self._last_reacquire_capture is None
            or 0.0
            <= captured_at - self._last_reacquire_capture
            <= self.calibration.arming_max_frame_gap_seconds
        )
        within_delta = (
            self._last_reacquire_targets is None
            or all(
                abs(after - before)
                <= self.calibration.arming_max_target_delta_units
                for name, before, after in zip(
                    HARDWARE_JOINTS,
                    self._last_reacquire_targets,
                    targets,
                )
                if name in self.selected_axes
            )
        )
        self._reacquire_count = (
            self._reacquire_count + 1 if within_gap and within_delta else 1
        )
        self._last_reacquire_capture = captured_at
        self._last_reacquire_targets = targets
        if self._reacquire_count >= self.calibration.valid_frames_to_arm:
            self._reacquisition_release_pending = True
            return "mano"
        return "fallback_open"


def preview_control_source(
    policy: str, detection_present: bool, depth_in_calibrated_range: bool
) -> str:
    if policy == "disable":
        return "mano" if detection_present else "none"
    if policy != "open":
        raise ValueError(f"unknown no-hand policy: {policy}")
    return (
        "mano"
        if detection_present and depth_in_calibrated_range
        else "fallback_open"
    )


def selected_axes_are_calibrated_open(
    calibration: PipelineCalibration,
    selected_axes: Sequence[str],
    actual_angles: Optional[Sequence[int]],
    tolerance_units: int = CALIBRATED_OPEN_START_TOLERANCE_UNITS,
) -> bool:
    if actual_angles is None or len(actual_angles) != len(HARDWARE_JOINTS):
        return False
    selected = set(selected_axes)
    return all(
        name not in selected
        or abs(int(actual) - calibration.axes[name].command_open)
        <= tolerance_units
        for name, actual in zip(HARDWARE_JOINTS, actual_angles)
    )


def make_output_dir(requested: Optional[Path], overwrite: bool) -> Path:
    if requested is not None:
        path = requested.expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = SCRIPT_DIR / "inspire_mano_pipeline/output" / stamp
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory is not empty: {path}; pass --overwrite-output to reuse it"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_record(
    handle,
    frame,
    detection,
    output,
    inference_seconds: float,
    masked_targets=None,
    accepted=None,
    stream=None,
    tracking_age_seconds=None,
    depth_in_calibrated_range=None,
    session_id=None,
    control_source="none",
    mano_submit_count=0,
    fallback_submit_count=0,
    model_diagnostics=None,
) -> None:
    if control_source not in ("mano", "fallback_open", "none"):
        raise ValueError(f"invalid control_source: {control_source}")
    recorded_at_monotonic = time.monotonic()
    ready_at_monotonic = getattr(frame, "ready_at_monotonic", None)
    row = {
        "log_schema_version": 2,
        "session_id": session_id,
        "frame_number": frame.frame_number,
        "depth_frame_number": getattr(frame, "depth_frame_number", None),
        "captured_at_monotonic": frame.captured_at_monotonic,
        "host_frame_received_at_monotonic": frame.captured_at_monotonic,
        "frame_ready_at_monotonic": ready_at_monotonic,
        "recorded_at_monotonic": recorded_at_monotonic,
        "frame_copy_align_seconds": (
            max(0.0, ready_at_monotonic - frame.captured_at_monotonic)
            if ready_at_monotonic is not None
            else None
        ),
        "pipeline_age_seconds": max(
            0.0, recorded_at_monotonic - frame.captured_at_monotonic
        ),
        "color_sensor_timestamp_ms": getattr(
            frame, "color_sensor_timestamp_ms", None
        ),
        "depth_sensor_timestamp_ms": getattr(
            frame, "depth_sensor_timestamp_ms", None
        ),
        "sensor_timestamp_domain": getattr(
            frame, "sensor_timestamp_domain", None
        ),
        "right_hand_detected": detection is not None,
        "wilor_diagnostics": (
            dict(model_diagnostics) if model_diagnostics is not None else None
        ),
        "tracking_age_seconds": tracking_age_seconds,
        "is_right": detection.is_right if detection is not None else None,
        "bbox_xyxy": detection.bbox_xyxy.tolist() if detection is not None else None,
        "palm_depth_m": detection.palm_depth_m if detection is not None else None,
        "control_palm_depth_m": (
            getattr(detection, "control_palm_depth_m", None)
            if detection is not None
            else None
        ),
        "palm_depth_source": (
            getattr(detection, "palm_depth_source", None)
            if detection is not None
            else None
        ),
        "palm_depth_reason": (
            getattr(detection, "palm_depth_reason", None)
            if detection is not None
            else None
        ),
        "palm_depth_evidence_at_monotonic": (
            getattr(detection, "palm_depth_evidence_at_monotonic", None)
            if detection is not None
            else None
        ),
        "palm_depth_age_seconds": (
            getattr(detection, "palm_depth_age_seconds", None)
            if detection is not None
            else None
        ),
        "palm_depth_roi_sample_count": (
            getattr(detection, "palm_depth_roi_sample_count", None)
            if detection is not None
            else None
        ),
        "palm_depth_inlier_count": (
            getattr(detection, "palm_depth_inlier_count", None)
            if detection is not None
            else None
        ),
        "palm_depth_valid_pixel_count": (
            getattr(detection, "palm_depth_valid_pixel_count", None)
            if detection is not None
            else None
        ),
        "palm_depth_radius_px": (
            getattr(detection, "palm_depth_radius_px", None)
            if detection is not None
            else None
        ),
        "palm_depth_in_calibrated_range": depth_in_calibrated_range,
        "keypoints_3d_canonical": (
            detection.keypoints_3d_canonical.tolist()
            if detection is not None
            else None
        ),
        "global_orient": detection.global_orient.tolist() if detection is not None else None,
        "hand_pose": detection.hand_pose.tolist() if detection is not None else None,
        "betas": detection.betas.tolist() if detection is not None else None,
        "qpos": output.qpos.tolist() if output is not None else None,
        "raw_qpos": (
            output.raw_qpos.tolist()
            if output is not None and output.raw_qpos is not None
            else None
        ),
        "raw_hardware_targets": (
            list(output.raw_hardware_targets)
            if output is not None and output.raw_hardware_targets is not None
            else None
        ),
        "hardware_targets": list(output.hardware_targets) if output is not None else None,
        "masked_hardware_targets": (
            list(masked_targets) if masked_targets is not None else None
        ),
        "hardware_submit_accepted": accepted,
        "control_source": control_source,
        "mano_submit_count": int(mano_submit_count),
        "fallback_submit_count": int(fallback_submit_count),
        "accepted_mano_target_count": int(mano_submit_count),
        "accepted_fallback_open_target_count": int(fallback_submit_count),
        "hardware_state": stream.state.value if stream is not None else None,
        "last_sent_targets": (
            list(stream.last_sent_targets)
            if stream is not None and stream.last_sent_targets is not None
            else None
        ),
        "actual_angles": (
            list(stream.latest_angles)
            if stream is not None and stream.latest_angles is not None
            else None
        ),
        "errors": (
            list(stream.latest_errors)
            if stream is not None and stream.latest_errors is not None
            else None
        ),
        "statuses": (
            list(stream.latest_statuses)
            if stream is not None and stream.latest_statuses is not None
            else None
        ),
        "temperatures": (
            list(stream.latest_temperatures)
            if stream is not None and stream.latest_temperatures is not None
            else None
        ),
        "currents": (
            list(stream.latest_currents)
            if stream is not None and stream.latest_currents is not None
            else None
        ),
        "active_current_policy": (
            stream.calibration.active_current_policy
            if stream is not None
            else None
        ),
        "host_current_threshold_exceeded": (
            stream.active_current_threshold_exceeded
            if stream is not None
            else None
        ),
        "active_current_over_limit_sample_count": (
            stream.active_current_over_limit_sample_count
            if stream is not None
            else None
        ),
        "active_current_warning_event_count": (
            stream.active_current_warning_event_count
            if stream is not None
            else None
        ),
        "active_peak_abs_currents": (
            list(stream.active_peak_abs_currents)
            if stream is not None
            else None
        ),
        "active_max_selected_total_current_ma": (
            stream.active_max_selected_total_current_ma
            if stream is not None
            else None
        ),
        "device_current_limits": (
            list(stream.verified_device_current_limits)
            if stream is not None
            and stream.verified_device_current_limits is not None
            else None
        ),
        "hardware_stop_reason": stream.stop_reason if stream is not None else None,
        "hardware_ever_active": stream.ever_active if stream is not None else None,
        "hardware_accepted_target_count": (
            stream.accepted_target_count if stream is not None else None
        ),
        "hardware_motion_write_count": (
            stream.motion_write_count if stream is not None else None
        ),
        "retargeter": output.backend if output is not None else None,
        "inference_seconds": inference_seconds,
    }
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def write_run_metadata(
    path: Path,
    args: argparse.Namespace,
    calibration: PipelineCalibration,
    source,
    session_id: str,
) -> None:
    try:
        dex_version = importlib.metadata.version("dex_retargeting")
    except importlib.metadata.PackageNotFoundError:
        dex_version = "unknown"
    calibration_path = args.calibration.expanduser().resolve()
    calibration_sha256 = (
        hashlib.sha256(calibration_path.read_bytes()).hexdigest()
        if calibration_path.is_file()
        else None
    )
    metadata = {
        "metadata_schema_version": 1,
        "frame_log_schema_version": 2,
        "session_id": session_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "command": list(sys.argv),
        "source": args.source,
        "camera": {
            "name": getattr(source, "device_name", None),
            "serial": getattr(source, "device_serial", None),
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "aligned_depth_to_color": isinstance(source, RealSenseSource),
        },
        "model": {
            "wilor_root": str(args.wilor_root.expanduser().resolve()),
            "dex_root": str(args.dex_root.expanduser().resolve()),
            "retargeter": args.retargeter,
            # Keep the legacy field while making the detector/mirroring
            # convention explicit for replay and audit.
            "handedness": "right",
            "physical_handedness": "right",
            "input_mirrored": bool(args.mirror_input),
            "wilor_detector_label": detector_label_for_physical_hand(
                "right", bool(args.mirror_input)
            ),
            "strict_single_right_hand": not args.allow_multiple_right_hands,
            "strict_single_detected_hand": not args.allow_multiple_right_hands,
            "operator_roi": (
                list(getattr(args, "operator_roi", ()))
                if getattr(args, "operator_roi", None) is not None
                else None
            ),
        },
        "calibration": {
            "path": str(calibration_path),
            "sha256": calibration_sha256,
            "tracking_timeout_seconds": calibration.tracking_timeout_seconds,
            "valid_frames_to_arm": calibration.valid_frames_to_arm,
            "arming_max_frame_gap_seconds": (
                getattr(calibration, "arming_max_frame_gap_seconds", None)
            ),
            "arming_max_target_delta_units": (
                getattr(calibration, "arming_max_target_delta_units", None)
            ),
            "depth_range_m": [
                calibration.min_palm_depth_m,
                calibration.max_palm_depth_m,
            ],
            "active_current_monitoring": {
                "policy": getattr(calibration, "active_current_policy", "fault"),
                "per_axis_threshold_ma": getattr(
                    calibration, "stream_max_current_ma", None
                ),
                "selected_total_threshold_ma": (
                    getattr(calibration, "stream_max_total_current_ma", None)
                ),
                "feedback_hz": getattr(calibration, "feedback_hz", None),
            },
            "temporal_filter": {
                "axes": list(
                    getattr(calibration, "temporal_filter_axes", ())
                ),
                "median_window": getattr(
                    calibration, "temporal_median_window", 1
                ),
                "ema_alpha": getattr(calibration, "temporal_ema_alpha", 1.0),
            },
            "endpoint_hysteresis": {
                name: {
                    "open_enter_q": axis.open_enter_q,
                    "open_exit_q": axis.open_exit_q,
                    "closed_enter_q": axis.closed_enter_q,
                    "closed_exit_q": axis.closed_exit_q,
                }
                for name, axis in getattr(calibration, "axes", {}).items()
                if name in getattr(calibration, "temporal_filter_axes", ())
            },
            "feedback_to_command": {
                name: {
                    "offset_units": axis.feedback_to_command_offset_units,
                    "valid_actual_range": (
                        [
                            axis.feedback_to_command_valid_min,
                            axis.feedback_to_command_valid_max,
                        ]
                        if axis.feedback_to_command_valid_min is not None
                        and axis.feedback_to_command_valid_max is not None
                        else None
                    ),
                }
                for name, axis in getattr(calibration, "axes", {}).items()
            },
        },
        "hardware": {
            "enabled": args.enable_hardware,
            "port": args.port,
            "hand_id": args.hand_id,
            "selected_axes": list(args.axes),
            "no_hand_policy": getattr(args, "no_hand_policy", "disable"),
            "no_hand_fallback_delay_seconds": NO_HAND_FALLBACK_DELAY_SECONDS,
            "calibrated_open_start_tolerance_units": (
                CALIBRATED_OPEN_START_TOLERANCE_UNITS
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "dex_retargeting": dex_version,
            "asset_manifest": str(
                SCRIPT_DIR / "inspire_mano_pipeline/asset_manifest.json"
            ),
        },
    }
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> int:
    calibration = PipelineCalibration.load(args.calibration)
    no_hand_controller = NoHandOpenController(
        calibration, args.axes, args.no_hand_policy
    )
    output_dir = make_output_dir(args.output_dir, args.overwrite_output)
    session_id = uuid.uuid4().hex
    print(f"[config] calibration={args.calibration}")
    print(f"[output] {output_dir}")
    print("[safety] PREVIEW ONLY: serial port will not be opened" if not args.enable_hardware else "[safety] HARDWARE OUTPUT REQUESTED")

    if args.retargeter == "dex":
        retargeter = DexInspireRetargeter(calibration, dex_root=args.dex_root)
    else:
        retargeter = GeometricInspireRetargeter(calibration)

    print(f"[model] loading WiLoR from {args.wilor_root}")
    wilor = WiLoRBackend(
        wilor_root=args.wilor_root,
        device=args.device,
        hand_confidence=calibration.detector_confidence,
        handedness="right",
        input_mirrored=args.mirror_input,
        strict_single_hand=not args.allow_multiple_right_hands,
        operator_roi=args.operator_roi,
    )
    print(f"[model] WiLoR ready on {wilor.device}")

    if args.source == "realsense":
        source = RealSenseSource(
            width=args.width,
            height=args.height,
            fps=args.fps,
            serial=args.camera_serial,
        )
    else:
        source = ImageSource(args.image, mirror=args.mirror_input)

    stop_requested = False
    termination_signal: Optional[int] = None
    stream: Optional[SafeRH56Stream] = None

    def request_stop(signum=None, frame=None) -> None:
        del frame
        nonlocal stop_requested, termination_signal
        stop_requested = True
        if signum is not None:
            termination_signal = int(signum)
        if stream is not None:
            reason = (
                f"received signal {signum}"
                if signum is not None
                else "main loop stop requested"
            )
            stream.request_stop(reason)

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    record_handle = None
    first_overlay_saved = False
    frame_count = 0
    detection_count = 0
    mano_submit_count = 0
    fallback_submit_count = 0
    has_valid_mano_tracking = False
    smoothed_fps = 0.0
    started_at = time.monotonic()
    exit_code = 0
    last_detection_at: Optional[float] = None
    retarget_was_reset = False
    last_image_write = 0.0
    last_console_write = 0.0
    latest_overlay = None
    stop_unconfirmed_message: Optional[str] = None
    try:
        source.start()
        if isinstance(source, RealSenseSource):
            print(
                f"[camera] {source.device_name}, serial={source.device_serial}, "
                f"{args.width}x{args.height}@{args.fps}"
            )
            if (
                args.camera_serial is not None
                and source.device_serial != args.camera_serial
            ):
                raise RuntimeError(
                    "RealSense serial mismatch: requested "
                    f"{args.camera_serial}, opened {source.device_serial}"
                )
        if not args.no_record:
            write_run_metadata(
                output_dir / "run_metadata.json",
                args,
                calibration,
                source,
                session_id,
            )
        if args.enable_hardware:
            print(
                "[hardware] motion output is now enabled; "
                f"axes={','.join(args.axes)}, speed={calibration.speed}, "
                f"force={calibration.force_limit}g, "
                f"active_current_policy={calibration.active_current_policy}, "
                f"no_hand_policy={args.no_hand_policy}"
            )
            stream = SafeRH56Stream(
                calibration=calibration,
                port=args.port,
                hand_id=args.hand_id,
                selected_axes=args.axes,
            )
            stream.start()
            stream.wait_until_ready(timeout=20.0)
            if (
                stream.verified_streaming_speeds is None
                or stream.verified_streaming_forces is None
            ):
                raise RuntimeError(
                    "hardware stream became ready without verified SPEED_SET/FORCE_SET"
                )
            print(
                "[hardware] verified temporary register settings: "
                f"SPEED_SET={list(stream.verified_streaming_speeds)}, "
                f"FORCE_SET={list(stream.verified_streaming_forces)}; "
                "original settings are restored after ANGLE_SET=-1 at shutdown"
            )
            if calibration.active_current_policy == "monitor_only":
                print(
                    "[hardware] ACTIVE host current thresholds are MONITOR ONLY: "
                    f"per_axis={calibration.stream_max_current_ma}mA, "
                    f"selected_total={calibration.stream_max_total_current_ma}mA; "
                    "hard current protection remains with verified device "
                    "CURRENT_LIMIT/STATUS/ERROR; device CURRENT_LIMIT="
                    f"{list(stream.verified_device_current_limits or ())}"
                )
            feedback_offsets = {
                name: {
                    "offset": calibration.axes[name].feedback_to_command_offset_units,
                    "valid_actual_range": (
                        calibration.axes[name].feedback_to_command_valid_min,
                        calibration.axes[name].feedback_to_command_valid_max,
                    ),
                }
                for name in args.axes
                if calibration.axes[name].feedback_to_command_offset_units
            }
            if feedback_offsets:
                print(
                    "[hardware] calibrated ANGLE_ACT->ANGLE_SET seed/normal-stop "
                    f"conversion: {feedback_offsets}"
                )
            if args.no_hand_policy == "open" and not selected_axes_are_calibrated_open(
                calibration,
                args.axes,
                stream.latest_angles,
            ):
                exit_code = 1
                raise RuntimeError(
                    "hardware fallback-open requires selected axes to start within "
                    f"{CALIBRATED_OPEN_START_TOLERANCE_UNITS} units of each "
                    "calibrated command_open"
                )
            if args.no_hand_policy == "open":
                print(
                    "[hardware] selected axes verified at CALIBRATED_OPEN; "
                    "fallback remains inhibited until valid MANO has entered ACTIVE"
                )
            print(
                "[hardware] preflight passed; waiting for "
                f"{calibration.valid_frames_to_arm} fresh MANO frames"
            )

        if not args.no_record:
            record_handle = (output_dir / "mano_retarget.jsonl").open(
                "w", encoding="utf-8"
            )

        while not stop_requested:
            if args.duration and time.monotonic() - started_at >= args.duration:
                break
            if args.max_frames and frame_count >= args.max_frames:
                break
            frame = source.read()
            frame_count += 1
            if stop_requested:
                break
            infer_started = time.perf_counter()
            detection = wilor.predict(frame)
            inference_seconds = time.perf_counter() - infer_started
            if stop_requested:
                break
            instant_fps = 1.0 / max(inference_seconds, 1e-6)
            smoothed_fps = (
                instant_fps
                if smoothed_fps == 0.0
                else 0.85 * smoothed_fps + 0.15 * instant_fps
            )
            output = None
            command = None
            mano_command = None
            accepted = None
            control_source = "none"
            depth_in_calibrated_range = None
            if detection is not None:
                detection_count += 1
                last_detection_at = time.monotonic()
                retarget_was_reset = False
                output = retargeter.retarget(detection)
                if stop_requested:
                    break
                depth = detection.palm_depth_m
                depth_in_calibrated_range = (
                    getattr(detection, "palm_depth_source", "measured")
                    == "measured"
                    and depth is not None
                    and calibration.min_palm_depth_m
                    <= depth
                    <= calibration.max_palm_depth_m
                )
                mano_command = mask_targets(output.hardware_targets, args.axes)
            else:
                if (
                    last_detection_at is not None
                    and not retarget_was_reset
                    and time.monotonic() - last_detection_at
                    > calibration.tracking_timeout_seconds
                ):
                    retargeter.reset()
                    retarget_was_reset = True

            if stream is not None:
                control_source = no_hand_controller.choose(
                    frame_captured_at_monotonic=frame.captured_at_monotonic,
                    detection_present=detection is not None,
                    depth_in_calibrated_range=bool(depth_in_calibrated_range),
                    mano_targets=mano_command,
                    stream_ever_active=stream.ever_active,
                )
                if control_source == "mano":
                    assert detection is not None and mano_command is not None
                    command = mano_command
                    accepted = stream.submit(
                        TargetFrame(
                            targets=command,
                            captured_at_monotonic=detection.captured_at_monotonic,
                            frame_number=detection.frame_number,
                            source="mano",
                            depth_evidence_at_monotonic=getattr(
                                detection,
                                "palm_depth_evidence_at_monotonic",
                                detection.captured_at_monotonic,
                            ),
                            depth_source=getattr(
                                detection, "palm_depth_source", "measured"
                            ),
                        )
                    )
                    no_hand_controller.note_mano_submit_result(bool(accepted))
                    if accepted:
                        mano_submit_count += 1
                    elif depth_in_calibrated_range:
                        print(
                            f"[tracking] dropped stale MANO frame {detection.frame_number}"
                        )
                elif control_source == "fallback_open":
                    command = open_fallback_targets(calibration, args.axes)
                    accepted = stream.submit(
                        TargetFrame(
                            targets=command,
                            captured_at_monotonic=frame.captured_at_monotonic,
                            frame_number=frame.frame_number,
                            source="fallback_open",
                            depth_source="not_required",
                        )
                    )
                    if accepted:
                        fallback_submit_count += 1
                else:
                    # Preserve the old disable policy: an invalid-depth MANO
                    # target may be logged, but it is never submitted.
                    command = mano_command
                    accepted = False if detection is not None else None
                    if detection is not None and not depth_in_calibrated_range:
                        print(
                            "\n[tracking] MANO frame rejected: "
                            f"depth_source={getattr(detection, 'palm_depth_source', 'missing')}, "
                            f"reason={getattr(detection, 'palm_depth_reason', 'unknown')}"
                        )
            else:
                control_source = preview_control_source(
                    args.no_hand_policy,
                    detection is not None,
                    bool(depth_in_calibrated_range),
                )
                if control_source == "fallback_open":
                    command = open_fallback_targets(calibration, args.axes)

            if record_handle is not None:
                tracking_age_seconds = (
                    None
                    if last_detection_at is None
                    else max(0.0, time.monotonic() - last_detection_at)
                )
                write_record(
                    record_handle,
                    frame,
                    detection,
                    output,
                    inference_seconds,
                    masked_targets=command,
                    accepted=accepted,
                    stream=stream,
                    tracking_age_seconds=tracking_age_seconds,
                    depth_in_calibrated_range=depth_in_calibrated_range,
                    session_id=session_id,
                    control_source=control_source,
                    mano_submit_count=mano_submit_count,
                    fallback_submit_count=fallback_submit_count,
                    model_diagnostics=getattr(wilor, "last_diagnostics", None),
                )

            if stream is not None:
                control_label = (
                    "CALIBRATED_OPEN"
                    if control_source == "fallback_open"
                    else control_source.upper()
                )
                hardware_state = (
                    f"HARDWARE {stream.state.value.upper()} | CONTROL {control_label}"
                )
                if stream.state in (
                    StreamState.FAULT_LATCHED,
                    StreamState.STOP_UNCONFIRMED,
                ):
                    print(f"[hardware][FAULT] {stream.stop_reason}", file=sys.stderr)
                    exit_code = 3 if stream.state == StreamState.STOP_UNCONFIRMED else 2
                    break
            else:
                control_label = (
                    "CALIBRATED_OPEN"
                    if control_source == "fallback_open"
                    else control_source.upper()
                )
                hardware_state = f"PREVIEW ONLY | CONTROL {control_label}"

            overlay_output = output
            if control_source == "fallback_open" and command is not None:
                overlay_output = RetargetOutput(
                    qpos=np.zeros(len(HARDWARE_JOINTS), dtype=np.float32),
                    hardware_targets=tuple(command),
                    backend="calibrated-open",
                )
            overlay = draw_overlay(
                frame.color_bgr,
                detection,
                overlay_output,
                smoothed_fps,
                hardware_state=hardware_state,
                selected_axes=args.axes,
                tracking_status=tracking_display_status(
                    detection,
                    getattr(wilor, "last_diagnostics", None),
                    calibration,
                ),
                sent_targets=(
                    stream.last_sent_targets if stream is not None else None
                ),
                actual_angles=(stream.latest_angles if stream is not None else None),
                operator_roi=args.operator_roi,
            )
            latest_overlay = overlay
            if record_handle is not None:
                if not first_overlay_saved and detection is not None:
                    if not cv2.imwrite(
                        str(output_dir / "first_detection.jpg"), overlay
                    ):
                        raise IOError("failed to write first_detection.jpg")
                    np.savez_compressed(
                        output_dir / "first_detection_mano.npz",
                        frame_number=np.asarray(detection.frame_number),
                        global_orient=detection.global_orient,
                        hand_pose=detection.hand_pose,
                        betas=detection.betas,
                        vertices=detection.vertices,
                        vertices_2d=detection.vertices_2d,
                        mesh_faces=detection.mesh_faces,
                        keypoints_3d_raw=detection.keypoints_3d_raw,
                        keypoints_3d_canonical=detection.keypoints_3d_canonical,
                        keypoints_2d=detection.keypoints_2d,
                    )
                    first_overlay_saved = True
                if time.monotonic() - last_image_write >= 1.0:
                    if not cv2.imwrite(str(output_dir / "latest.jpg"), overlay):
                        raise IOError("failed to write latest.jpg")
                    last_image_write = time.monotonic()

            if not args.headless:
                cv2.imshow("RealSense -> MANO -> Inspire RH56", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            now = time.monotonic()
            if now - last_console_write >= 0.25:
                effective_depth = (
                    None
                    if detection is None
                    else getattr(detection, "control_palm_depth_m", None)
                )
                if effective_depth is None and detection is not None:
                    effective_depth = detection.palm_depth_m
                depth = "n/a"
                if effective_depth is not None and detection is not None:
                    depth = (
                        f"{effective_depth:.3f}m/"
                        f"{getattr(detection, 'palm_depth_source', 'measured')}"
                    )
                target_text = (
                    "not submitted"
                    if control_source == "none"
                    else (
                        str(list(command))
                        if command is not None
                        else (
                            str(list(output.hardware_targets))
                            if output is not None
                            else "no command"
                        )
                    )
                )
                print(
                    f"\r[frame {frame.frame_number}] "
                    f"infer={inference_seconds * 1000:.1f}ms "
                    f"depth={depth} control={control_label} "
                    f"targets={target_text} "
                    f"sent={list(stream.last_sent_targets) if stream is not None and stream.last_sent_targets is not None else 'n/a'} "
                    f"actual={list(stream.latest_angles) if stream is not None and stream.latest_angles is not None else 'n/a'}      ",
                    end="",
                    flush=True,
                )
                last_console_write = now
        print()
    finally:
        in_flight_exception = sys.exc_info()[1]
        if in_flight_exception is not None and exit_code == 0:
            exit_code = 130 if isinstance(in_flight_exception, KeyboardInterrupt) else 1
        if stream is not None:
            if termination_signal is not None:
                stream.request_stop(f"received signal {termination_signal}")
            try:
                stream.close()
            except Exception as exc:
                print(f"[hardware][STOP UNCONFIRMED] {exc}", file=sys.stderr)
                exit_code = 3
                stop_unconfirmed_message = str(exc)
            if not stream.stop_confirmed:
                print(
                    "[hardware][STOP UNCONFIRMED] immediately cut 24 V if the hand is moving",
                    file=sys.stderr,
                )
                exit_code = 3
                stop_unconfirmed_message = stream.stop_reason
            else:
                print(
                    "[hardware] verified final ANGLE_SET="
                    "[-1, -1, -1, -1, -1, -1] and physical feedback stopped"
                )
                mano_submit_count = int(
                    getattr(
                        stream,
                        "accepted_mano_target_count",
                        mano_submit_count,
                    )
                )
                fallback_submit_count = int(
                    getattr(
                        stream,
                        "accepted_fallback_open_target_count",
                        fallback_submit_count,
                    )
                )
                has_valid_mano_tracking = bool(
                    stream.ever_active
                    and mano_submit_count >= calibration.valid_frames_to_arm
                )
                if stream.state == StreamState.FAULT_LATCHED and exit_code == 0:
                    exit_code = 2
                if (
                    exit_code == 0
                    and (not stream.ever_active or stream.motion_write_count == 0)
                ):
                    print(
                        "[hardware][FAIL] run ended without entering ACTIVE and "
                        "sending a numeric motion target",
                        file=sys.stderr,
                    )
                    exit_code = 5
                if (
                    exit_code == 0
                    and fallback_submit_count > 0
                    and not has_valid_mano_tracking
                ):
                    print(
                        "[hardware][FAIL] CALIBRATED_OPEN fallback ran without "
                        "validated MANO tracking",
                        file=sys.stderr,
                    )
                    exit_code = 6
            if not args.no_record:
                final_exit_code = (
                    3
                    if stop_unconfirmed_message is not None
                    else (
                        128 + termination_signal
                        if termination_signal is not None
                        else exit_code
                    )
                )
                shutdown_status = {
                    "session_id": session_id,
                    "hardware_state": stream.state.value,
                    "stop_confirmed": stream.stop_confirmed,
                    "physical_stop_verified": stream.physical_stop_verified,
                    "stop_reason": stream.stop_reason,
                    "disable_error": stream.disable_error,
                    "ever_active": stream.ever_active,
                    "accepted_target_count": stream.accepted_target_count,
                    "motion_write_count": stream.motion_write_count,
                    "no_hand_policy": args.no_hand_policy,
                    "has_valid_mano_tracking": has_valid_mano_tracking,
                    "accepted_mano_target_count": mano_submit_count,
                    "accepted_fallback_open_target_count": fallback_submit_count,
                    "active_current_policy": calibration.active_current_policy,
                    "host_current_per_axis_threshold_ma": (
                        calibration.stream_max_current_ma
                    ),
                    "host_current_selected_total_threshold_ma": (
                        calibration.stream_max_total_current_ma
                    ),
                    "active_current_over_limit_sample_count": (
                        stream.active_current_over_limit_sample_count
                    ),
                    "active_current_warning_event_count": (
                        stream.active_current_warning_event_count
                    ),
                    "active_peak_abs_currents": list(
                        stream.active_peak_abs_currents
                    ),
                    "active_max_selected_total_current_ma": (
                        stream.active_max_selected_total_current_ma
                    ),
                    "verified_device_current_limits": (
                        list(stream.verified_device_current_limits)
                        if stream.verified_device_current_limits is not None
                        else None
                    ),
                    "verified_streaming_speeds": (
                        list(stream.verified_streaming_speeds)
                        if stream.verified_streaming_speeds is not None
                        else None
                    ),
                    "verified_streaming_forces": (
                        list(stream.verified_streaming_forces)
                        if stream.verified_streaming_forces is not None
                        else None
                    ),
                    "initial_command_seed": (
                        list(stream.initial_command_seed)
                        if stream.initial_command_seed is not None
                        else None
                    ),
                    "shutdown_hold_targets": (
                        list(stream.shutdown_hold_targets)
                        if stream.shutdown_hold_targets is not None
                        else None
                    ),
                    "shutdown_feedback": (
                        {
                            key: list(value)
                            for key, value in stream.shutdown_feedback.items()
                        }
                        if stream.shutdown_feedback is not None
                        else None
                    ),
                    "final_angle_targets": (
                        list(stream.final_angle_targets)
                        if stream.final_angle_targets is not None
                        else None
                    ),
                    "last_sent_targets": (
                        list(stream.last_sent_targets)
                        if stream.last_sent_targets is not None
                        else None
                    ),
                    "termination_signal": termination_signal,
                    "exit_code": final_exit_code,
                }
                (output_dir / "shutdown_status.json").write_text(
                    json.dumps(shutdown_status, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
        source.stop()
        if record_handle is not None and latest_overlay is not None:
            if not cv2.imwrite(str(output_dir / "latest.jpg"), latest_overlay):
                print("[record][WARN] failed to write final latest.jpg", file=sys.stderr)
        if record_handle is not None:
            record_handle.close()
        cv2.destroyAllWindows()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        if stop_unconfirmed_message is not None:
            raise HardwareStopUnconfirmed(stop_unconfirmed_message)

    print(
        f"[summary] frames={frame_count}, detections={detection_count}, "
        f"average_recent_inference={smoothed_fps:.2f} FPS, "
        f"mano_submits={mano_submit_count}, "
        f"calibrated_open_submits={fallback_submit_count}, "
        f"has_valid_mano_tracking={has_valid_mano_tracking}"
    )
    if detection_count == 0 and args.no_hand_policy != "open":
        print("[summary][FAIL] no right hand was detected", file=sys.stderr)
        return 4 if exit_code == 0 else exit_code
    if termination_signal is not None:
        return 128 + termination_signal
    return exit_code


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130
    except HardwareStopUnconfirmed as exc:
        print(f"[fatal][STOP UNCONFIRMED] {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"[fatal] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
