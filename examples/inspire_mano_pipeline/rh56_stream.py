from __future__ import annotations

import contextlib
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, ContextManager, Optional, Sequence, Tuple

import numpy as np

try:
    from examples.inspire_rh56_test import (
        JOINTS,
        REG_ANGLE_ACT,
        REG_ANGLE_SET,
        REG_CURRENT,
        REG_ERROR,
        REG_FORCE_SET,
        REG_POS_ACT,
        REG_SPEED_SET,
        REG_STATUS,
        REG_TEMP,
        LinuxSerial,
        RH56Error,
        RH56Hand,
        find_serial_port,
    )
except ModuleNotFoundError:  # Running the top-level script from examples/.
    from inspire_rh56_test import (  # type: ignore
        JOINTS,
        REG_ANGLE_ACT,
        REG_ANGLE_SET,
        REG_CURRENT,
        REG_ERROR,
        REG_FORCE_SET,
        REG_POS_ACT,
        REG_SPEED_SET,
        REG_STATUS,
        REG_TEMP,
        LinuxSerial,
        RH56Error,
        RH56Hand,
        find_serial_port,
    )

from .calibration import PipelineCalibration


class StreamState(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    WAITING_FOR_TRACKING = "waiting_for_tracking"
    ACTIVE = "active"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAULT_LATCHED = "fault_latched"
    STOP_UNCONFIRMED = "stop_unconfirmed"


class TrackingTimeout(RH56Error):
    pass


@dataclass(frozen=True)
class TargetFrame:
    targets: Tuple[int, ...]
    captured_at_monotonic: float
    frame_number: int
    source: str = "mano"
    depth_evidence_at_monotonic: Optional[float] = None
    depth_source: str = "measured"


HandContextFactory = Callable[[], ContextManager[RH56Hand]]


class SafeRH56Stream:
    """Single-owner, watchdog-protected streaming controller for RH56.

    The caller only publishes target frames.  All serial IO, including safety
    feedback and the final six-axis -1 command, happens in one worker thread.
    """

    def __init__(
        self,
        calibration: PipelineCalibration,
        port: Optional[str] = None,
        baud: int = 115200,
        hand_id: int = 1,
        serial_timeout: float = 0.15,
        selected_axes: Optional[Sequence[str]] = None,
        hand_context_factory: Optional[HandContextFactory] = None,
    ) -> None:
        self.calibration = calibration
        self.port = port
        self.baud = baud
        self.hand_id = hand_id
        self.serial_timeout = serial_timeout
        enabled = {
            name for name in JOINTS if calibration.axes[name].enabled
        }
        if selected_axes is None:
            self.selected_axes = enabled
        else:
            requested = set(selected_axes)
            unknown = requested.difference(JOINTS)
            if unknown:
                raise ValueError(f"unknown RH56 axes: {sorted(unknown)}")
            disabled = requested.difference(enabled)
            if disabled:
                raise ValueError(
                    "axes disabled by calibration: " + ", ".join(sorted(disabled))
                )
            self.selected_axes = requested
        if not self.selected_axes:
            raise ValueError("at least one calibrated axis must be selected")

        self._hand_context_factory = hand_context_factory
        self._state = StreamState.CREATED
        self._state_lock = threading.Lock()
        self._target_lock = threading.Lock()
        # request_stop() and every numeric motion write share this lock.  Once
        # request_stop() returns, no later motion write can start.
        self._motion_lock = threading.RLock()
        self._latest_target: Optional[TargetFrame] = None
        self._target_generation = 0
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[BaseException] = None
        self.stop_confirmed = False
        self.stop_reason = "not stopped"
        self.latest_angles: Optional[Tuple[int, ...]] = None
        self.latest_errors: Optional[Tuple[int, ...]] = None
        self.latest_statuses: Optional[Tuple[int, ...]] = None
        self.latest_temperatures: Optional[Tuple[int, ...]] = None
        self.latest_currents: Optional[Tuple[int, ...]] = None
        self.verified_device_current_limits: Optional[Tuple[int, ...]] = None
        self.active_current_threshold_exceeded = False
        self.active_current_over_limit_sample_count = 0
        self.active_current_warning_event_count = 0
        self.active_peak_abs_currents: Tuple[int, ...] = (0,) * 6
        self.active_max_selected_total_current_ma = 0
        self.latest_current_warning: Optional[str] = None
        self._active_current_warning_latched = False
        self.verified_streaming_speeds: Optional[Tuple[int, ...]] = None
        self.verified_streaming_forces: Optional[Tuple[int, ...]] = None
        self.last_sent_targets: Optional[Tuple[int, ...]] = None
        self.initial_command_seed: Optional[Tuple[int, ...]] = None
        self.shutdown_hold_targets: Optional[Tuple[int, ...]] = None
        self.shutdown_feedback: Optional[dict] = None
        self.physical_stop_verified = False
        self.final_angle_targets: Optional[Tuple[int, ...]] = None
        self.ever_active = False
        self.accepted_target_count = 0
        self.accepted_mano_target_count = 0
        self.accepted_fallback_open_target_count = 0
        self.motion_write_count = 0
        self.disable_error: Optional[str] = None

    @property
    def state(self) -> StreamState:
        with self._state_lock:
            return self._state

    def _set_state(self, state: StreamState) -> None:
        with self._state_lock:
            self._state = state

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("RH56 stream has already been started")
        self._set_state(StreamState.STARTING)
        self._thread = threading.Thread(
            target=self._run,
            name="rh56-serial-owner",
            daemon=False,
        )
        self._thread.start()

    def wait_until_ready(self, timeout: float = 10.0) -> None:
        if not self._ready_event.wait(timeout):
            raise TimeoutError("RH56 stream did not finish preflight in time")
        if self.error is not None:
            raise RH56Error(f"RH56 stream preflight failed: {self.error}")

    def submit(self, frame: TargetFrame) -> bool:
        if frame.source not in ("mano", "fallback_open"):
            raise ValueError(
                "target frame source must be 'mano' or 'fallback_open'"
            )
        if self._stop_event.is_set():
            return False
        if frame.source == "fallback_open" and (
            self.state != StreamState.ACTIVE or not self.ever_active
        ):
            return False
        if self.state not in (
            StreamState.WAITING_FOR_TRACKING,
            StreamState.ACTIVE,
        ):
            return False
        now = time.monotonic()
        targets = tuple(int(value) for value in frame.targets)
        if len(targets) != 6:
            raise ValueError("target frame must contain six RH56 values")
        if not np.isfinite(frame.captured_at_monotonic):
            raise ValueError("capture timestamp must be finite")
        age = now - frame.captured_at_monotonic
        if age < -0.10:
            raise ValueError("capture timestamp is in the future")
        if age > self.calibration.tracking_timeout_seconds:
            return False
        if frame.frame_number < 0:
            raise ValueError("frame_number cannot be negative")
        if frame.depth_source not in ("measured", "held", "not_required"):
            raise ValueError("unknown target depth_source")
        if frame.source == "mano":
            if frame.depth_source == "not_required":
                raise ValueError("MANO target requires measured or held depth")
            if (
                frame.depth_source == "held"
                and frame.depth_evidence_at_monotonic is None
            ):
                raise ValueError(
                    "held MANO depth requires the original measured-depth timestamp"
                )
            depth_evidence_at = (
                frame.captured_at_monotonic
                if frame.depth_evidence_at_monotonic is None
                else float(frame.depth_evidence_at_monotonic)
            )
            if not np.isfinite(depth_evidence_at):
                raise ValueError("depth evidence timestamp must be finite")
            if depth_evidence_at > frame.captured_at_monotonic + 0.01:
                raise ValueError("depth evidence cannot be newer than its target frame")
            if now - depth_evidence_at > self.calibration.tracking_timeout_seconds:
                return False
        else:
            depth_evidence_at = None
        for name, value in zip(JOINTS, targets):
            if name not in self.selected_axes:
                if value != -1:
                    raise ValueError(f"unselected axis {name} must have target -1")
                continue
            if not 0 <= value <= 1000:
                raise ValueError(f"axis {name} target must be in 0..1000")
            axis = self.calibration.axes[name]
            lower = min(axis.command_open, axis.command_closed)
            upper = max(axis.command_open, axis.command_closed)
            if not lower <= value <= upper:
                raise ValueError(
                    f"axis {name} target {value} is outside calibrated soft range "
                    f"{lower}..{upper}"
                )
        with self._motion_lock:
            if self._stop_event.is_set() or self.state not in (
                StreamState.WAITING_FOR_TRACKING,
                StreamState.ACTIVE,
            ):
                return False
            if frame.source == "fallback_open" and (
                self.state != StreamState.ACTIVE or not self.ever_active
            ):
                return False
            with self._target_lock:
                if (
                    self._latest_target is not None
                    and (
                        frame.captured_at_monotonic
                        <= self._latest_target.captured_at_monotonic
                        or frame.frame_number <= self._latest_target.frame_number
                    )
                ):
                    return False
                self._latest_target = TargetFrame(
                    targets=targets,
                    captured_at_monotonic=float(frame.captured_at_monotonic),
                    frame_number=int(frame.frame_number),
                    source=frame.source,
                    depth_evidence_at_monotonic=depth_evidence_at,
                    depth_source=(
                        frame.depth_source
                        if frame.source == "mano"
                        else "not_required"
                    ),
                )
                self._target_generation += 1
                self.accepted_target_count += 1
                if frame.source == "mano":
                    self.accepted_mano_target_count += 1
                else:
                    self.accepted_fallback_open_target_count += 1
        return True

    def request_stop(self, reason: str = "requested") -> None:
        with self._motion_lock:
            if self.error is None and self.stop_reason == "not stopped":
                self.stop_reason = reason
            self._stop_event.set()

    def close(self, timeout: float = 8.0) -> None:
        self.request_stop("close requested")
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError("RH56 serial owner did not stop in time")

    def _snapshot_target(self) -> tuple[Optional[TargetFrame], int]:
        with self._target_lock:
            return self._latest_target, self._target_generation

    def _command_from_actual(self, name: str, actual: int) -> int:
        """Convert ANGLE_ACT feedback into this profile's ANGLE_SET domain."""

        axis = self.calibration.axes[name]
        actual = int(actual)
        if (
            axis.feedback_to_command_valid_min is not None
            and axis.feedback_to_command_valid_max is not None
            and not (
                axis.feedback_to_command_valid_min
                <= actual
                <= axis.feedback_to_command_valid_max
            )
        ):
            raise RH56Error(
                f"axis {name} ANGLE_ACT {actual} is outside the validated "
                "feedback-to-command range "
                f"{axis.feedback_to_command_valid_min}.."
                f"{axis.feedback_to_command_valid_max}"
            )
        lower = min(axis.command_open, axis.command_closed)
        upper = max(axis.command_open, axis.command_closed)
        command = actual + axis.feedback_to_command_offset_units
        # A calibrated positive offset can cross only the protocol's physical
        # 1000 endpoint when ANGLE_ACT itself is already at that endpoint.  In
        # that one explicit case, saturating to 1000 is the feedback-equivalent
        # open hold; arbitrary soft-range clamping remains forbidden.
        if (
            command > upper
            and upper == 1000
            and axis.feedback_to_command_offset_units > 0
            and command - upper <= axis.feedback_to_command_offset_units
        ):
            command = upper
        elif (
            command < lower
            and lower == 0
            and axis.feedback_to_command_offset_units < 0
            and lower - command <= abs(axis.feedback_to_command_offset_units)
        ):
            command = lower
        if not lower <= command <= upper:
            raise RH56Error(
                f"axis {name} feedback-equivalent command {command} is outside "
                f"the calibrated command range {lower}..{upper}"
            )
        return command

    def _selected_commands_from_actual(
        self, actual: Sequence[int]
    ) -> Tuple[int, ...]:
        if len(actual) != len(JOINTS):
            raise RH56Error(f"expected six actual angles, got {actual}")
        return tuple(
            self._command_from_actual(name, int(value))
            if name in self.selected_axes
            else -1
            for name, value in zip(JOINTS, actual)
        )

    @contextlib.contextmanager
    def _default_hand_context(self):
        port = self.port or find_serial_port()
        with LinuxSerial(
            port,
            self.baud,
            timeout=self.serial_timeout,
            debug=False,
        ) as serial_port:
            yield RH56Hand(serial_port, self.hand_id)

    def _run(self) -> None:
        operation_error: Optional[BaseException] = None
        stop_error: Optional[BaseException] = None
        context_opened = False
        original_speeds: Optional[Tuple[int, ...]] = None
        original_forces: Optional[Tuple[int, ...]] = None
        context_factory = self._hand_context_factory or self._default_hand_context
        try:
            with context_factory() as opened_hand:
                context_opened = True
                hand = opened_hand
                try:
                    first_snapshot = hand.snapshot()
                    self._preflight(first_snapshot)
                    time.sleep(self.calibration.preflight_stability_seconds)
                    snapshot = hand.snapshot()
                    self._preflight(snapshot)
                    first_angles = tuple(int(value) for value in first_snapshot["angles"])
                    initial_angles = tuple(int(value) for value in snapshot["angles"])
                    if any(
                        abs(after - before)
                        > self.calibration.preflight_max_angle_delta
                        for before, after in zip(first_angles, initial_angles)
                    ):
                        raise RH56Error(
                            "preflight actual angles were not stable: "
                            f"first={first_angles}, second={initial_angles}"
                        )
                    self.latest_angles = initial_angles
                    current_limits_raw = snapshot.get("current_limits")
                    if current_limits_raw is not None:
                        current_limits = tuple(
                            int(value) for value in current_limits_raw
                        )
                        if len(current_limits) != len(JOINTS):
                            raise RH56Error(
                                "snapshot current_limits must contain six values"
                            )
                        if not all(0 <= value <= 1500 for value in current_limits):
                            raise RH56Error(
                                "snapshot contains invalid device CURRENT_LIMIT values: "
                                f"{current_limits}"
                            )
                        self.verified_device_current_limits = current_limits
                    if self.calibration.active_current_policy == "monitor_only":
                        if self.verified_device_current_limits is None:
                            raise RH56Error(
                                "monitor-only ACTIVE current policy requires verified "
                                "device CURRENT_LIMIT readback"
                            )
                        invalid_selected_limits = {
                            name: value
                            for name, value in zip(
                                JOINTS, self.verified_device_current_limits
                            )
                            if name in self.selected_axes and not 1 <= value <= 1500
                        }
                        if invalid_selected_limits:
                            raise RH56Error(
                                "monitor-only ACTIVE current policy requires enabled "
                                "device CURRENT_LIMIT on every selected axis: "
                                f"{invalid_selected_limits}"
                            )
                    original_speeds = tuple(int(value) for value in snapshot["speeds"])
                    original_forces = tuple(
                        int(value) for value in snapshot["force_limits"]
                    )
                    streaming_speeds = tuple(
                        (
                            self.calibration.axes[name].hardware_speed
                            if self.calibration.axes[name].hardware_speed is not None
                            else self.calibration.speed
                        )
                        if name in self.selected_axes
                        else value
                        for name, value in zip(JOINTS, original_speeds)
                    )
                    streaming_forces = tuple(
                        self.calibration.force_limit
                        if name in self.selected_axes
                        else value
                        for name, value in zip(JOINTS, original_forces)
                    )
                    hand.write_six_shorts(
                        REG_SPEED_SET, streaming_speeds, retries=0
                    )
                    speed_readback = tuple(
                        int(value)
                        for value in hand.read_six_shorts(REG_SPEED_SET, retries=0)
                    )
                    if speed_readback != streaming_speeds:
                        raise RH56Error(
                            "streaming SPEED_SET readback mismatch: "
                            f"expected={streaming_speeds}, actual={speed_readback}"
                        )
                    self.verified_streaming_speeds = speed_readback
                    hand.write_six_shorts(
                        REG_FORCE_SET, streaming_forces, retries=0
                    )
                    force_readback = tuple(
                        int(value)
                        for value in hand.read_six_shorts(REG_FORCE_SET, retries=0)
                    )
                    if force_readback != streaming_forces:
                        raise RH56Error(
                            "streaming FORCE_SET readback mismatch: "
                            f"expected={streaming_forces}, actual={force_readback}"
                        )
                    self.verified_streaming_forces = force_readback
                    self._set_state(StreamState.WAITING_FOR_TRACKING)
                    self._ready_event.set()

                    sent = self._selected_commands_from_actual(initial_angles)
                    self.initial_command_seed = sent
                    desired = sent
                    last_generation = 0
                    consecutive_valid = 0
                    last_valid_capture: Optional[float] = None
                    last_arming_targets: Optional[Tuple[int, ...]] = None
                    last_tick = time.monotonic()
                    next_feedback = last_tick
                    period = 1.0 / self.calibration.control_hz

                    while not self._stop_event.is_set():
                        loop_start = time.monotonic()
                        target_frame, generation = self._snapshot_target()
                        if self.state == StreamState.WAITING_FOR_TRACKING and (
                            target_frame is None
                            or target_frame.source != "mano"
                            or target_frame.depth_source != "measured"
                            or loop_start - target_frame.captured_at_monotonic
                            > self.calibration.tracking_timeout_seconds
                        ):
                            consecutive_valid = 0
                            last_valid_capture = None
                            last_arming_targets = None
                        if generation != last_generation and target_frame is not None:
                            last_generation = generation
                            if (
                                loop_start - target_frame.captured_at_monotonic
                                <= self.calibration.tracking_timeout_seconds
                                and (
                                    target_frame.source != "mano"
                                    or (
                                        target_frame.depth_source == "measured"
                                        and target_frame.depth_evidence_at_monotonic
                                        is not None
                                        and loop_start
                                        - target_frame.depth_evidence_at_monotonic
                                        <= self.calibration.tracking_timeout_seconds
                                    )
                                )
                                and (
                                    self.state != StreamState.WAITING_FOR_TRACKING
                                    or target_frame.source == "mano"
                                )
                            ):
                                within_frame_gap = (
                                    last_valid_capture is None
                                    or target_frame.captured_at_monotonic
                                    - last_valid_capture
                                    <= self.calibration.arming_max_frame_gap_seconds
                                )
                                within_target_delta = (
                                    last_arming_targets is None
                                    or all(
                                        abs(after - before)
                                        <= self.calibration.arming_max_target_delta_units
                                        for name, before, after in zip(
                                            JOINTS,
                                            last_arming_targets,
                                            target_frame.targets,
                                        )
                                        if name in self.selected_axes
                                    )
                                )
                                if within_frame_gap and within_target_delta:
                                    consecutive_valid += 1
                                else:
                                    consecutive_valid = 1
                                last_valid_capture = target_frame.captured_at_monotonic
                                last_arming_targets = target_frame.targets
                                desired = target_frame.targets
                            else:
                                consecutive_valid = 0
                                last_valid_capture = None
                                last_arming_targets = None
                        if (
                            self.state == StreamState.WAITING_FOR_TRACKING
                            and consecutive_valid
                            >= self.calibration.valid_frames_to_arm
                        ):
                            if self._stop_event.is_set():
                                break
                            self._read_safety_feedback(hand, sent, active=False)
                            if self._stop_event.is_set():
                                break
                            if hand.read_six_shorts(REG_ANGLE_SET, retries=0) != (-1,) * 6:
                                raise RH56Error(
                                    "ANGLE_SET changed after preflight; refusing to arm"
                                )
                            with self._motion_lock:
                                if self._stop_event.is_set():
                                    break
                                self._set_state(StreamState.ACTIVE)
                                self.ever_active = True

                        if self.state == StreamState.ACTIVE:
                            pose_stale = target_frame is None or (
                                loop_start - target_frame.captured_at_monotonic
                                > self.calibration.tracking_timeout_seconds
                            )
                            depth_stale = (
                                target_frame is not None
                                and target_frame.source == "mano"
                                and (
                                    target_frame.depth_evidence_at_monotonic is None
                                    or loop_start
                                    - target_frame.depth_evidence_at_monotonic
                                    > self.calibration.tracking_timeout_seconds
                                )
                            )
                            if pose_stale or depth_stale:
                                raise TrackingTimeout(
                                    "hardware control target or measured-depth evidence "
                                    "timed out; output has been latched off"
                                )
                            dt = min(max(loop_start - last_tick, 0.0), period)
                            sent = self._rate_limit(sent, desired, dt)
                            with self._motion_lock:
                                if self._stop_event.is_set():
                                    break
                                hand.write_six_shorts(REG_ANGLE_SET, sent, retries=0)
                                self.last_sent_targets = sent
                                self.motion_write_count += 1

                        if loop_start >= next_feedback:
                            self._read_safety_feedback(
                                hand, sent, active=self.ever_active
                            )
                            next_feedback = (
                                loop_start + 1.0 / self.calibration.feedback_hz
                            )
                        last_tick = loop_start
                        self._stop_event.wait(
                            max(0.0, period - (time.monotonic() - loop_start))
                        )
                except BaseException as exc:
                    operation_error = exc
                    self.error = exc
                    self.stop_reason = str(exc)
                finally:
                    self._set_state(StreamState.STOPPING)
                    try:
                        self._disable(
                            hand,
                            stop_in_place=(
                                self.ever_active and operation_error is None
                            ),
                            original_speeds=original_speeds,
                            original_forces=original_forces,
                        )
                        self.stop_confirmed = True
                    except BaseException as exc:
                        stop_error = exc
                        self.stop_confirmed = False
                        self.disable_error = str(exc)
                        prefix = "" if self.stop_reason == "not stopped" else self.stop_reason + "; "
                        self.stop_reason = prefix + f"shutdown disable failed: {exc}"
                        if self.error is None:
                            self.error = exc
        except BaseException as exc:
            if operation_error is None:
                operation_error = exc
                self.error = exc
                self.stop_reason = str(exc)
        finally:
            self._ready_event.set()
            if stop_error is not None:
                self.stop_confirmed = False
                self._set_state(StreamState.STOP_UNCONFIRMED)
            elif operation_error is not None:
                self._set_state(
                    StreamState.FAULT_LATCHED
                    if self.stop_confirmed
                    else StreamState.STOP_UNCONFIRMED
                )
            else:
                if not context_opened or not self.stop_confirmed:
                    self._set_state(StreamState.STOP_UNCONFIRMED)
                else:
                    self._set_state(StreamState.STOPPED)

    def _preflight(self, snapshot: dict) -> None:
        if tuple(snapshot["angle_targets"]) != (-1,) * 6:
            raise RH56Error("preflight requires all six ANGLE_SET targets to be -1")
        if any(snapshot["errors"]):
            raise RH56Error(f"preflight found actuator errors: {snapshot['errors']}")
        if max(snapshot["temperatures"]) >= 60:
            raise RH56Error(
                f"preflight found temperature >=60 C: {snapshot['temperatures']}"
            )
        if not all(status in (2, 0xFF) for status in snapshot["statuses"]):
            raise RH56Error(
                "preflight requires idle statuses in {2, 255}: "
                f"{snapshot['statuses']}"
            )
        if not all(0 <= angle <= 1000 for angle in snapshot["angles"]):
            raise RH56Error(f"preflight found invalid actual angles: {snapshot['angles']}")
        for name, angle in zip(JOINTS, snapshot["angles"]):
            if name not in self.selected_axes:
                continue
            try:
                self._command_from_actual(name, int(angle))
            except RH56Error as exc:
                raise RH56Error(
                    f"preflight actual angle for selected axis {name} cannot be "
                    f"converted to a safe command seed within the calibrated "
                    f"soft range: {exc}"
                ) from exc
        if max(abs(int(value)) for value in snapshot["currents"]) > (
            self.calibration.preflight_max_idle_current_ma
        ):
            raise RH56Error(
                "preflight idle current is too high: " f"{snapshot['currents']}"
            )
        selected_total_current = sum(
            abs(int(value))
            for name, value in zip(JOINTS, snapshot["currents"])
            if name in self.selected_axes
        )
        if selected_total_current > self.calibration.stream_max_total_current_ma:
            raise RH56Error(
                "preflight selected-axis total current is too high: "
                f"total={selected_total_current}, currents={snapshot['currents']}"
            )

    def _read_safety_feedback(
        self, hand: RH56Hand, sent: Tuple[int, ...], active: bool
    ) -> None:
        errors = tuple(hand.read(REG_ERROR, 6, retries=0))
        statuses = tuple(hand.read(REG_STATUS, 6, retries=0))
        temperatures = tuple(hand.read(REG_TEMP, 6, retries=0))
        angles = hand.read_six_shorts(REG_ANGLE_ACT, retries=0)
        currents = tuple(hand.read_six_shorts(REG_CURRENT, retries=0))
        self.latest_errors = errors
        self.latest_statuses = statuses
        self.latest_temperatures = temperatures
        self.latest_angles = angles
        self.latest_currents = currents
        if any(errors):
            raise RH56Error(f"actuator fault during streaming: {errors}")
        if max(temperatures) >= 60:
            raise RH56Error(f"actuator temperature reached 60 C: {temperatures}")
        if active:
            for name, command, status in zip(JOINTS, sent, statuses):
                allowed = (2, 0xFF) if command == -1 else (0, 1, 2)
                if status not in allowed:
                    raise RH56Error(
                        f"axis {name} returned unsafe status {status} while streaming"
                    )
        elif not all(status in (2, 0xFF) for status in statuses):
            raise RH56Error(
                f"non-idle actuator status while waiting for tracking: {statuses}"
            )

        abs_currents = tuple(abs(int(value)) for value in currents)
        selected_total_current = sum(
            value
            for name, value in zip(JOINTS, abs_currents)
            if name in self.selected_axes
        )
        if not active:
            if max(abs_currents) > self.calibration.preflight_max_idle_current_ma:
                raise RH56Error(
                    "actuator current exceeded idle limit "
                    f"{self.calibration.preflight_max_idle_current_ma} mA: {currents}"
                )
            if (
                selected_total_current
                > self.calibration.stream_max_total_current_ma
            ):
                raise RH56Error(
                    "pre-active selected-axis total current is too high: "
                    f"total={selected_total_current}, currents={currents}"
                )
            return

        self.active_peak_abs_currents = tuple(
            max(previous, current)
            for previous, current in zip(
                self.active_peak_abs_currents, abs_currents
            )
        )
        self.active_max_selected_total_current_ma = max(
            self.active_max_selected_total_current_ma,
            selected_total_current,
        )
        per_axis_peak = max(abs_currents)
        per_axis_exceeded = (
            per_axis_peak > self.calibration.stream_max_current_ma
        )
        total_exceeded = (
            selected_total_current
            > self.calibration.stream_max_total_current_ma
        )
        threshold_exceeded = per_axis_exceeded or total_exceeded
        self.active_current_threshold_exceeded = threshold_exceeded

        if self.calibration.active_current_policy == "fault":
            if per_axis_exceeded:
                raise RH56Error(
                    "actuator current exceeded "
                    f"{self.calibration.stream_max_current_ma} mA: {currents}"
                )
            if total_exceeded:
                raise RH56Error(
                    "selected-axis total current exceeded "
                    f"{self.calibration.stream_max_total_current_ma} mA: "
                    f"total={selected_total_current}, currents={currents}"
                )
            return

        if threshold_exceeded:
            self.active_current_over_limit_sample_count += 1
            warning = (
                "ACTIVE current telemetry crossed monitor-only thresholds: "
                f"per_axis_peak={per_axis_peak}/"
                f"{self.calibration.stream_max_current_ma}mA, "
                f"selected_total={selected_total_current}/"
                f"{self.calibration.stream_max_total_current_ma}mA, "
                f"currents={currents}"
            )
            self.latest_current_warning = warning
            if not self._active_current_warning_latched:
                self.active_current_warning_event_count += 1
                print(
                    "[hardware][CURRENT WARN] " + warning + "; continuing under "
                    "device CURRENT_LIMIT/STATUS/ERROR protection",
                    file=sys.stderr,
                )
            self._active_current_warning_latched = True
        else:
            self.latest_current_warning = None
            self._active_current_warning_latched = False

    def _rate_limit(
        self,
        previous: Tuple[int, ...],
        desired: Tuple[int, ...],
        dt: float,
    ) -> Tuple[int, ...]:
        result = []
        for name, before, target in zip(JOINTS, previous, desired):
            if name not in self.selected_axes:
                result.append(-1)
                continue
            if before == -1 or target == -1:
                raise RH56Error(f"selected axis {name} unexpectedly has target -1")
            limit = self.calibration.axes[name].max_rate_units_per_second * dt
            delta = float(np.clip(target - before, -limit, limit))
            result.append(int(round(before + delta)))
        return tuple(result)

    def _verify_post_release_stop(self, hand: RH56Hand) -> None:
        """Confirm that all actuators are electrically and mechanically idle."""

        samples = []
        for sample_index in range(5):
            sample = {
                "angles": tuple(hand.read_six_shorts(REG_ANGLE_ACT, retries=0)),
                "positions": tuple(hand.read_six_shorts(REG_POS_ACT, retries=0)),
                "currents": tuple(hand.read_six_shorts(REG_CURRENT, retries=0)),
                "errors": tuple(hand.read(REG_ERROR, 6, retries=0)),
                "statuses": tuple(hand.read(REG_STATUS, 6, retries=0)),
                "temperatures": tuple(hand.read(REG_TEMP, 6, retries=0)),
            }
            samples.append(sample)
            if sample_index < 4:
                time.sleep(0.10)

        verification = samples[-3:]
        for axis_index, name in enumerate(JOINTS):
            angle_span = max(
                int(sample["angles"][axis_index]) for sample in verification
            ) - min(int(sample["angles"][axis_index]) for sample in verification)
            position_span = max(
                int(sample["positions"][axis_index]) for sample in verification
            ) - min(
                int(sample["positions"][axis_index]) for sample in verification
            )
            if angle_span > 2 or position_span > 3:
                raise RH56Error(
                    f"post-release {name} feedback is still moving: "
                    f"angle_span={angle_span}, position_span={position_span}"
                )

        for sample in verification:
            if any(int(value) for value in sample["errors"]):
                raise RH56Error(
                    f"post-release actuator fault remains: {sample['errors']}"
                )
            if not all(
                int(status) in (2, 0xFF) for status in sample["statuses"]
            ):
                raise RH56Error(
                    f"post-release status is not idle: {sample['statuses']}"
                )
            currents = tuple(int(value) for value in sample["currents"])
            if any(abs(value) > 100 for value in currents):
                raise RH56Error(
                    f"post-release per-axis current is not idle: {currents}"
                )
            if sum(abs(value) for value in currents) > 200:
                raise RH56Error(
                    f"post-release total current is not idle: {currents}"
                )
            if max(int(value) for value in sample["temperatures"]) >= 60:
                raise RH56Error(
                    "post-release actuator temperature is unsafe: "
                    f"{sample['temperatures']}"
                )

        final = verification[-1]
        self.shutdown_feedback = {
            key: tuple(int(value) for value in final[key])
            for key in (
                "angles",
                "positions",
                "currents",
                "errors",
                "statuses",
                "temperatures",
            )
        }
        self.latest_angles = self.shutdown_feedback["angles"]
        self.latest_currents = self.shutdown_feedback["currents"]
        self.latest_errors = self.shutdown_feedback["errors"]
        self.latest_statuses = self.shutdown_feedback["statuses"]
        self.latest_temperatures = self.shutdown_feedback["temperatures"]
        self.physical_stop_verified = True

    def _disable(
        self,
        hand: RH56Hand,
        stop_in_place: bool,
        original_speeds: Optional[Tuple[int, ...]] = None,
        original_forces: Optional[Tuple[int, ...]] = None,
    ) -> None:
        degraded_errors = []
        if stop_in_place:
            try:
                actual = hand.read_six_shorts(REG_ANGLE_ACT, retries=0)
                if not all(0 <= int(value) <= 1000 for value in actual):
                    raise RH56Error(f"invalid actual angles during shutdown: {actual}")
                selected_hold = self._selected_commands_from_actual(actual)
                self.shutdown_hold_targets = selected_hold
                hand.write_six_shorts(REG_ANGLE_SET, selected_hold, retries=0)
                hold_readback = tuple(
                    hand.read_six_shorts(REG_ANGLE_SET, retries=0)
                )
                if hold_readback != selected_hold:
                    raise RH56Error(
                        "stop-in-place hold readback mismatch: "
                        f"expected={selected_hold}, actual={hold_readback}"
                    )
                time.sleep(0.10)
            except BaseException as exc:
                degraded_errors.append(f"stop-in-place hold failed: {exc}")

        # Disable and verify while the conservative streaming speed/force are
        # still active.  If this critical step fails, do not restore possibly
        # higher original settings on top of a remaining numeric target.
        hand.write_six_shorts(REG_ANGLE_SET, (-1,) * 6, retries=1)
        first_readback = tuple(hand.read_six_shorts(REG_ANGLE_SET, retries=1))
        self.final_angle_targets = first_readback
        if first_readback != (-1,) * 6:
            raise RH56Error(
                f"failed to verify ANGLE_SET=-1 before settings restore: {first_readback}"
            )

        try:
            self._verify_post_release_stop(hand)
        except BaseException as stop_exc:
            # Reassert release once more while conservative streaming settings
            # are still installed.  A target readback alone does not upgrade an
            # unverified physical stop to safe.
            try:
                hand.write_six_shorts(REG_ANGLE_SET, (-1,) * 6, retries=1)
                self.final_angle_targets = tuple(
                    hand.read_six_shorts(REG_ANGLE_SET, retries=1)
                )
            except BaseException as reassert_exc:
                raise RH56Error(
                    "physical stop verification failed and release reassertion "
                    f"also failed: stop={stop_exc}; reassert={reassert_exc}"
                ) from stop_exc
            raise RH56Error(
                f"physical stop verification failed: {stop_exc}"
            ) from stop_exc

        if original_speeds is not None:
            try:
                hand.write_six_shorts(REG_SPEED_SET, original_speeds, retries=0)
            except BaseException as exc:
                degraded_errors.append(f"speed restore failed: {exc}")
        if original_forces is not None:
            try:
                hand.write_six_shorts(REG_FORCE_SET, original_forces, retries=0)
            except BaseException as exc:
                degraded_errors.append(f"force restore failed: {exc}")
        # Restoration must never be the last device write.  Re-assert and
        # verify all-six -1 once more after restoring non-motion settings.
        hand.write_six_shorts(REG_ANGLE_SET, (-1,) * 6, retries=1)
        readback = tuple(hand.read_six_shorts(REG_ANGLE_SET, retries=1))
        self.final_angle_targets = readback
        if readback != (-1,) * 6:
            raise RH56Error(f"failed to verify final ANGLE_SET=-1: {readback}")
        if degraded_errors:
            self.stop_reason += "; " + "; ".join(degraded_errors)
