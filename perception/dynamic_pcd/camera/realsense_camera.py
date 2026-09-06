from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Deque, Dict, Optional

import numpy as np

from dynamic_pcd.types import CameraIntrinsics, RGBDFrame


PRESET_MAP = {
    "custom": 0,
    "default": 1,
    "hand": 2,
    "high_accuracy": 3,
    "high_density": 4,
    "medium_density": 5,
}

_EPOCH_TIMESTAMP_DOMAINS = {
    "timestamp_domain.global_time",
    "timestamp_domain.system_time",
}

_FRAME_DIAGNOSTIC_RING_SIZE = 16
_LATEST_FRAME_DRAIN_LIMIT = 64
_DEFAULT_MAX_COLOR_DEPTH_TIMESTAMP_SKEW_S = 0.008
_DEFAULT_MAX_RETRYABLE_FUTURE_SKEW_S = 0.10
_DIAGNOSTIC_METADATA_NAMES = (
    "frame_timestamp",
    "sensor_timestamp",
    "backend_timestamp",
    "time_of_arrival",
)


class _ColorDepthTimestampSkewError(RuntimeError):
    pass


class _NonIncreasingFrameSequenceError(RuntimeError):
    """A valid-timestamp frameset that is not newer than the accepted pair."""


class _CaptureTimestampStaleError(RuntimeError):
    """A retryable frame rejection, never an accepted freshness override."""

    def __init__(self, transport_age_s: float, maximum_transport_age_s: float):
        self.transport_age_s = float(transport_age_s)
        self.maximum_transport_age_s = float(maximum_transport_age_s)
        super().__init__(
            "RealSense frame was already stale when retrieved: "
            f"age={self.transport_age_s:.6f}s exceeds "
            f"{self.maximum_transport_age_s:.6f}s"
        )


class _CaptureTimestampFutureSkewError(RuntimeError):
    """A modest future-skew frame rejected while awaiting clock recovery.

    The frame is never accepted or clamped to host time.  This marker only
    lets the fixed-deadline capture loop inspect a subsequent strictly newer
    RGB-D pair after a transient D400 global-time resynchronization.
    """

    def __init__(self, future_skew_s: float, maximum_retryable_skew_s: float):
        self.future_skew_s = float(future_skew_s)
        self.maximum_retryable_skew_s = float(maximum_retryable_skew_s)
        super().__init__(
            "RealSense capture timestamp has retryable future skew: "
            f"skew={self.future_skew_s:.6f}s within "
            f"{self.maximum_retryable_skew_s:.6f}s"
        )


class RetryableCameraTimeout(RuntimeError):
    """No acceptable RGB-D frame arrived within one fixed capture deadline.

    This marker is intentionally narrow: callers may retry another bounded
    capture attempt, while a separate publication watchdog must still enforce
    the stream's absolute liveness limit.  Missing streams, malformed/future
    timestamps, and other camera contract failures do not use this type.
    """

    retryable_camera_timeout = True


class _StartupReadinessGate:
    """Require one uninterrupted run of trustworthy startup frames.

    The gate deliberately uses host monotonic time only for durations and the
    difference ``time.time() - time.monotonic()`` only as a clock-jump guard.
    A rejected frame never contributes to the next candidate run.
    """

    def __init__(
        self,
        *,
        required_consecutive_frames: int,
        maximum_clock_offset_jitter_s: float,
    ) -> None:
        required = int(required_consecutive_frames)
        jitter_s = float(maximum_clock_offset_jitter_s)
        if required < 2:
            raise ValueError(
                "RealSense startup_required_consecutive_frames must be >= 2"
            )
        if not np.isfinite(jitter_s) or jitter_s <= 0.0:
            raise ValueError(
                "RealSense startup_max_clock_offset_jitter_s must be finite "
                "and > 0"
            )
        self.required_consecutive_frames = required
        self.maximum_clock_offset_jitter_s = jitter_s
        self.consecutive_valid_frames = 0
        self.best_consecutive_valid_frames = 0
        self.observed_frames = 0
        self.rejected_frames = 0
        self.reset_count = 0
        self.last_rejection_reason: Optional[str] = None
        self.last_sensor_frame_number: Optional[int] = None
        self.last_depth_sensor_frame_number: Optional[int] = None
        self.last_retrieved_monotonic_s: Optional[float] = None
        self.last_host_realtime_minus_monotonic_s: Optional[float] = None
        self.sequence_clock_offset_min_s: Optional[float] = None
        self.sequence_clock_offset_max_s: Optional[float] = None
        self.skipped_sensor_frames = 0

    @property
    def ready(self) -> bool:
        return self.consecutive_valid_frames >= self.required_consecutive_frames

    @property
    def sequence_clock_offset_span_s(self) -> Optional[float]:
        if (
            self.sequence_clock_offset_min_s is None
            or self.sequence_clock_offset_max_s is None
        ):
            return None
        return (
            self.sequence_clock_offset_max_s - self.sequence_clock_offset_min_s
        )

    @staticmethod
    def _validated_frame_number(value: int, label: str) -> int:
        number = int(value)
        if isinstance(value, (float, np.floating)) and float(value) != float(number):
            raise ValueError(f"RealSense {label} frame number must be integral")
        if number < 0:
            raise ValueError(f"RealSense {label} frame number must be >= 0")
        return number

    def _record_observation(
        self,
        *,
        sensor_frame_number: Optional[int],
        depth_sensor_frame_number: Optional[int],
        retrieved_at_s: Optional[float],
        retrieved_monotonic_s: Optional[float],
    ) -> tuple[Optional[int], Optional[int], Optional[float]]:
        self.observed_frames += 1
        color_number: Optional[int] = None
        depth_number: Optional[int] = None
        host_clock_offset_s: Optional[float] = None
        if sensor_frame_number is not None:
            color_number = self._validated_frame_number(
                sensor_frame_number, "color"
            )
        if depth_sensor_frame_number is not None:
            depth_number = self._validated_frame_number(
                depth_sensor_frame_number, "depth"
            )
        if (retrieved_at_s is None) != (retrieved_monotonic_s is None):
            raise ValueError(
                "RealSense startup host timestamps must be supplied together"
            )
        if retrieved_at_s is not None and retrieved_monotonic_s is not None:
            realtime_s = float(retrieved_at_s)
            monotonic_s = float(retrieved_monotonic_s)
            if not np.isfinite(realtime_s) or not np.isfinite(monotonic_s):
                raise ValueError(
                    "RealSense startup host timestamps must be finite"
                )
            host_clock_offset_s = realtime_s - monotonic_s
            self.last_retrieved_monotonic_s = monotonic_s
            self.last_host_realtime_minus_monotonic_s = host_clock_offset_s
        return color_number, depth_number, host_clock_offset_s

    def _clear_sequence(
        self,
        reason: str,
        *,
        next_clock_offset_reference_s: Optional[float] = None,
    ) -> None:
        self.rejected_frames += 1
        self.reset_count += 1
        self.last_rejection_reason = str(reason)
        self.consecutive_valid_frames = 0
        self.sequence_clock_offset_min_s = next_clock_offset_reference_s
        self.sequence_clock_offset_max_s = next_clock_offset_reference_s

    def reject(
        self,
        reason: str,
        *,
        sensor_frame_number: Optional[int] = None,
        depth_sensor_frame_number: Optional[int] = None,
        retrieved_at_s: Optional[float] = None,
        retrieved_monotonic_s: Optional[float] = None,
    ) -> None:
        """Record a bad startup frame and clear the consecutive-good run."""

        try:
            color_number, depth_number, _offset_s = self._record_observation(
                sensor_frame_number=sensor_frame_number,
                depth_sensor_frame_number=depth_sensor_frame_number,
                retrieved_at_s=retrieved_at_s,
                retrieved_monotonic_s=retrieved_monotonic_s,
            )
        except (OverflowError, TypeError, ValueError):
            # The rejection itself may be malformed.  It still counts as one
            # observed bad frame, but no untrusted number becomes the anchor.
            color_number = None
            depth_number = None
        if color_number is not None:
            self.last_sensor_frame_number = color_number
        if depth_number is not None:
            self.last_depth_sensor_frame_number = depth_number
        self._clear_sequence(reason)

    def observe_valid(
        self,
        *,
        sensor_frame_number: int,
        depth_sensor_frame_number: int,
        retrieved_at_s: float,
        retrieved_monotonic_s: float,
    ) -> bool:
        """Add a timestamp-valid RGB-D frame and return readiness."""

        try:
            color_number, depth_number, host_clock_offset_s = (
                self._record_observation(
                    sensor_frame_number=sensor_frame_number,
                    depth_sensor_frame_number=depth_sensor_frame_number,
                    retrieved_at_s=retrieved_at_s,
                    retrieved_monotonic_s=retrieved_monotonic_s,
                )
            )
        except (OverflowError, TypeError, ValueError) as exc:
            self._clear_sequence(str(exc))
            return False
        assert color_number is not None
        assert depth_number is not None
        assert host_clock_offset_s is not None

        previous_color_number = self.last_sensor_frame_number
        previous_depth_number = self.last_depth_sensor_frame_number
        color_increases = (
            previous_color_number is None
            or color_number > previous_color_number
        )
        depth_increases = (
            previous_depth_number is None
            or depth_number > previous_depth_number
        )
        if (
            previous_color_number is not None
            and color_number > previous_color_number
        ):
            self.skipped_sensor_frames += max(
                0, color_number - previous_color_number - 1
            )
        self.last_sensor_frame_number = color_number
        self.last_depth_sensor_frame_number = depth_number
        if not color_increases or not depth_increases:
            self._clear_sequence(
                "RealSense startup sensor frame numbers must strictly increase: "
                f"color={color_number} after {previous_color_number}, "
                f"depth={depth_number} after {previous_depth_number}"
            )
            return False

        if self.sequence_clock_offset_min_s is None:
            next_min_s = host_clock_offset_s
            next_max_s = host_clock_offset_s
        else:
            assert self.sequence_clock_offset_max_s is not None
            next_min_s = min(
                self.sequence_clock_offset_min_s, host_clock_offset_s
            )
            next_max_s = max(
                self.sequence_clock_offset_max_s, host_clock_offset_s
            )
        clock_offset_span_s = next_max_s - next_min_s
        if clock_offset_span_s > self.maximum_clock_offset_jitter_s:
            self._clear_sequence(
                "RealSense startup host realtime-minus-monotonic offset jumped "
                f"by {clock_offset_span_s:.6f}s (limit "
                f"{self.maximum_clock_offset_jitter_s:.6f}s)",
                # The jump frame is not counted, but it is the reference for
                # determining whether the host clocks have become stable again.
                next_clock_offset_reference_s=host_clock_offset_s,
            )
            return False

        self.sequence_clock_offset_min_s = next_min_s
        self.sequence_clock_offset_max_s = next_max_s
        self.consecutive_valid_frames += 1
        self.best_consecutive_valid_frames = max(
            self.best_consecutive_valid_frames,
            self.consecutive_valid_frames,
        )
        return self.ready


def _validated_epoch_capture_timestamp(
    *,
    color_timestamp_ms: float,
    depth_timestamp_ms: float,
    color_domain: str,
    depth_domain: str,
    retrieved_at_s: float,
    color_sensor_timestamp_us: Optional[float] = None,
    depth_sensor_timestamp_us: Optional[float] = None,
    maximum_color_depth_skew_s: float = (
        _DEFAULT_MAX_COLOR_DEPTH_TIMESTAMP_SKEW_S
    ),
    maximum_transport_age_s: float = 1.0,
    maximum_future_skew_s: float = 0.05,
    maximum_retryable_future_skew_s: float = (
        _DEFAULT_MAX_RETRYABLE_FUTURE_SKEW_S
    ),
) -> tuple[float, float, float]:
    """Validate native timestamps and return the color capture epoch time.

    D435 global/system timestamps are epoch milliseconds. Hardware-clock
    timestamps cannot be compared with Franka/RH56 host timestamps without a
    separately measured clock transform, so this path rejects them.
    """

    values = (
        color_timestamp_ms,
        depth_timestamp_ms,
        retrieved_at_s,
        maximum_color_depth_skew_s,
        maximum_transport_age_s,
        maximum_future_skew_s,
        maximum_retryable_future_skew_s,
    )
    if not all(np.isfinite(float(value)) for value in values):
        raise RuntimeError("RealSense capture timestamps must be finite")
    if (
        maximum_color_depth_skew_s <= 0.0
        or maximum_transport_age_s <= 0.0
        or maximum_future_skew_s < 0.0
        or maximum_retryable_future_skew_s < maximum_future_skew_s
    ):
        raise ValueError("RealSense timestamp limits are invalid")
    color_domain = str(color_domain)
    depth_domain = str(depth_domain)
    if color_domain != depth_domain or color_domain not in _EPOCH_TIMESTAMP_DOMAINS:
        raise RuntimeError(
            "RealSense RGB-D timestamps require one shared global/system-time "
            f"domain, got color={color_domain!r}, depth={depth_domain!r}"
        )
    color_timestamp_s = float(color_timestamp_ms) * 1.0e-3
    depth_timestamp_s = float(depth_timestamp_ms) * 1.0e-3
    epoch_timestamp_skew_s = abs(color_timestamp_s - depth_timestamp_s)
    has_color_sensor_time = color_sensor_timestamp_us is not None
    has_depth_sensor_time = depth_sensor_timestamp_us is not None
    if has_color_sensor_time != has_depth_sensor_time:
        raise RuntimeError(
            "RealSense RGB-D sensor timestamps must be present for both streams"
        )
    if has_color_sensor_time:
        if not np.isfinite(float(color_sensor_timestamp_us)) or not np.isfinite(
            float(depth_sensor_timestamp_us)
        ):
            raise RuntimeError("RealSense RGB-D sensor timestamps must be finite")
        # D400 frame_timestamp metadata is in microseconds on the shared
        # device clock.  Unlike per-sensor global-time conversion, this is the
        # correct quantity for checking whether color and depth were captured
        # as one synchronized frameset.
        color_depth_skew_s = (
            abs(float(color_sensor_timestamp_us) - float(depth_sensor_timestamp_us))
            * 1.0e-6
        )
    else:
        color_depth_skew_s = epoch_timestamp_skew_s
    if color_depth_skew_s > float(maximum_color_depth_skew_s):
        raise _ColorDepthTimestampSkewError(
            "RealSense color/depth capture skew "
            f"{color_depth_skew_s:.6f}s exceeds "
            f"{float(maximum_color_depth_skew_s):.6f}s"
        )
    transport_age_s = float(retrieved_at_s) - color_timestamp_s
    if transport_age_s < -float(maximum_future_skew_s):
        future_skew_s = -transport_age_s
        if future_skew_s <= float(maximum_retryable_future_skew_s):
            raise _CaptureTimestampFutureSkewError(
                future_skew_s=future_skew_s,
                maximum_retryable_skew_s=float(
                    maximum_retryable_future_skew_s
                ),
            )
        raise RuntimeError(
            "RealSense capture timestamp is unexpectedly in the future: "
            f"age={transport_age_s:.6f}s"
        )
    if transport_age_s > float(maximum_transport_age_s):
        raise _CaptureTimestampStaleError(
            transport_age_s=transport_age_s,
            maximum_transport_age_s=float(maximum_transport_age_s),
        )
    return color_timestamp_s, color_depth_skew_s, epoch_timestamp_skew_s


class RealSenseCamera:
    """Aligned RGB-D capture wrapper for RealSense D400 cameras."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.rs = None
        self.pipeline = None
        self.frame_queue = None
        self.profile = None
        self.depth_sensor = None
        self.color_sensor = None
        self.align = None
        self.spatial_filter = None
        self.temporal_filter = None
        self.hole_filter = None
        self.depth_scale = 0.001
        self.frame_id = 0
        self.started = False
        self.device_serial: Optional[str] = None
        self.device_name: Optional[str] = None
        self.device_firmware_version: Optional[str] = None
        self.device_usb_type_descriptor: Optional[str] = None
        self.sdk_version: Optional[str] = None
        self.rejected_timestamp_skew_frames = 0
        self.last_rejected_color_depth_skew_s: Optional[float] = None
        self.dropped_queued_framesets = 0
        self.drained_pending_framesets = 0
        self.last_drained_pending_framesets = 0
        self.last_sensor_frame_number: Optional[int] = None
        self.last_depth_sensor_frame_number: Optional[int] = None
        # Dequeued and accepted frames are deliberately tracked separately.
        # A bad frameset is useful transport evidence, but it must never move
        # the acceptance anchor used to reject a duplicate/regressed stream.
        self.last_accepted_sensor_frame_number: Optional[int] = None
        self.last_accepted_depth_sensor_frame_number: Optional[int] = None
        self.last_retrieved_monotonic_s: Optional[float] = None
        self.last_host_realtime_minus_monotonic_s: Optional[float] = None
        self.last_host_clock_pair_span_s: Optional[float] = None
        self.runtime_frame_wait_timeouts = 0
        self.last_frame_wait_started_monotonic_s: Optional[float] = None
        self.last_frame_wait_failed_monotonic_s: Optional[float] = None
        self.rejected_transport_stale_frames = 0
        self.last_rejected_transport_age_s: Optional[float] = None
        self.rejected_future_skew_frames = 0
        self.last_rejected_future_skew_s: Optional[float] = None
        self.rejected_non_increasing_frames = 0
        self.last_rejected_frame_sequence_reason: Optional[str] = None
        self.frame_dequeue_count = 0
        self.frame_dequeue_diagnostics: Deque[Dict[str, Any]] = deque(
            maxlen=_FRAME_DIAGNOSTIC_RING_SIZE
        )
        self.startup_frames_observed = 0
        self.startup_rejected_frames = 0
        self.startup_reset_count = 0
        self.startup_consecutive_valid_frames = 0
        self.startup_best_consecutive_valid_frames = 0
        self.startup_last_rejection_reason: Optional[str] = None
        self.startup_clock_offset_span_s: Optional[float] = None
        self.startup_skipped_sensor_frames = 0

    @staticmethod
    def _host_clock_sample(
        *, monotonic_fn=None, realtime_fn=None
    ) -> tuple[float, float, float, float]:
        """Sample realtime between monotonic bounds after a frame dequeue."""

        monotonic_fn = monotonic_fn or time.monotonic
        realtime_fn = realtime_fn or time.time
        monotonic_before_s = float(monotonic_fn())
        realtime_s = float(realtime_fn())
        monotonic_after_s = float(monotonic_fn())
        values = (monotonic_before_s, realtime_s, monotonic_after_s)
        if not all(np.isfinite(value) for value in values):
            raise RuntimeError("RealSense host clock samples must be finite")
        pair_span_s = monotonic_after_s - monotonic_before_s
        if pair_span_s < 0.0:
            raise RuntimeError("RealSense host monotonic clock moved backwards")
        return (
            monotonic_before_s,
            realtime_s,
            monotonic_after_s,
            pair_span_s,
        )

    def _frame_metadata_snapshot(self, frame) -> Dict[str, Optional[float]]:
        snapshot: Dict[str, Optional[float]] = {}
        metadata_values = getattr(self.rs, "frame_metadata_value", None)
        for name in _DIAGNOSTIC_METADATA_NAMES:
            value: Optional[float] = None
            metadata = (
                None if metadata_values is None else getattr(metadata_values, name, None)
            )
            if metadata is not None:
                try:
                    if frame.supports_frame_metadata(metadata):
                        value = float(frame.get_frame_metadata(metadata))
                except Exception:
                    # Metadata is diagnostic-only. Unsupported/malformed fields
                    # must not alter the capture decision.
                    value = None
            snapshot[name] = value
        return snapshot

    def _record_dequeued_frameset(
        self,
        *,
        stage: str,
        frames,
        host_monotonic_before_s: float,
        host_realtime_s: float,
        host_monotonic_after_s: float,
        host_clock_pair_span_s: float,
    ) -> tuple[Dict[str, Any], Any, Any]:
        """Append one bounded diagnostic record for every validated candidate."""

        previous_record = (
            None
            if not self.frame_dequeue_diagnostics
            else self.frame_dequeue_diagnostics[-1]
        )
        self.frame_dequeue_count += 1
        record: Dict[str, Any] = {
            "dequeue_index": self.frame_dequeue_count,
            "stage": str(stage),
            "outcome": "dequeued",
            "host_monotonic_before_s": host_monotonic_before_s,
            "host_realtime_s": host_realtime_s,
            "host_monotonic_after_s": host_monotonic_after_s,
            "host_clock_pair_span_s": host_clock_pair_span_s,
        }
        # Append first so even a malformed frameset leaves evidence that one
        # item was dequeued. The bounded deque prevents diagnostic backlog.
        self.frame_dequeue_diagnostics.append(record)
        raw_depth_frame = frames.get_depth_frame()
        raw_color_frame = frames.get_color_frame()
        for prefix, frame in (
            ("color", raw_color_frame),
            ("depth", raw_depth_frame),
        ):
            record[f"{prefix}_frame_number"] = None
            record[f"{prefix}_timestamp_ms"] = None
            record[f"{prefix}_timestamp_domain"] = None
            for metadata_name in _DIAGNOSTIC_METADATA_NAMES:
                record[f"{prefix}_{metadata_name}_metadata"] = None
            if not frame:
                continue
            record[f"{prefix}_frame_number"] = int(frame.get_frame_number())
            record[f"{prefix}_timestamp_ms"] = float(frame.get_timestamp())
            record[f"{prefix}_timestamp_domain"] = str(
                frame.get_frame_timestamp_domain()
            )
            metadata = self._frame_metadata_snapshot(frame)
            for metadata_name, value in metadata.items():
                record[f"{prefix}_{metadata_name}_metadata"] = value

        for prefix in ("color", "depth"):
            current_number = record.get(f"{prefix}_frame_number")
            previous_number = (
                None
                if previous_record is None
                else previous_record.get(f"{prefix}_frame_number")
            )
            frame_delta = (
                None
                if current_number is None or previous_number is None
                else int(current_number) - int(previous_number)
            )
            record[f"{prefix}_frame_delta"] = frame_delta

        color_delta = record.get("color_frame_delta")
        depth_delta = record.get("depth_frame_delta")
        if color_delta is None or depth_delta is None:
            record["stream_progress"] = "unavailable"
        elif color_delta > 0 and depth_delta > 0:
            record["stream_progress"] = "both_advanced"
        elif color_delta > 0:
            record["stream_progress"] = "color_only"
        elif depth_delta > 0:
            record["stream_progress"] = "depth_only"
        else:
            record["stream_progress"] = "neither_advanced"

        color_timestamp_ms = record.get("color_timestamp_ms")
        depth_timestamp_ms = record.get("depth_timestamp_ms")
        record["color_depth_epoch_timestamp_skew_s"] = (
            None
            if color_timestamp_ms is None or depth_timestamp_ms is None
            else abs(float(color_timestamp_ms) - float(depth_timestamp_ms))
            * 1.0e-3
        )
        color_frame_timestamp = record.get(
            "color_frame_timestamp_metadata"
        )
        depth_frame_timestamp = record.get(
            "depth_frame_timestamp_metadata"
        )
        record["color_depth_sensor_timestamp_skew_s"] = (
            None
            if color_frame_timestamp is None or depth_frame_timestamp is None
            else abs(float(color_frame_timestamp) - float(depth_frame_timestamp))
            * 1.0e-6
        )
        return record, raw_color_frame, raw_depth_frame

    def _wait_for_latest_frameset(self, timeout_ms: int):
        """Return the newest frameset currently available from the SDK sink.

        The output ``frame_queue`` contains only composite framesets already
        synchronized by librealsense. A blocking read followed by non-blocking
        polling closes the race in which a newer complete frameset reaches the
        final sink just after the blocking read.

        Draining never weakens freshness: the returned candidate still passes
        the normal transport-age, RGB-D skew, and strictly-increasing sequence
        checks.  If the queue cannot quiesce within a deliberately generous
        bound, fail closed rather than publish a candidate that may not be the
        latest one.
        """

        assert self.frame_queue is not None
        candidate = self.frame_queue.wait_for_frame(int(timeout_ms))
        drained = 0
        poll_for_frame = getattr(self.frame_queue, "poll_for_frame", None)
        if poll_for_frame is not None:
            while True:
                pending = poll_for_frame()
                if not pending:
                    break
                candidate = pending
                drained += 1
                if drained >= _LATEST_FRAME_DRAIN_LIMIT:
                    self.drained_pending_framesets += drained
                    self.last_drained_pending_framesets = drained
                    raise RuntimeError(
                        "RealSense pending frame queue did not quiesce after "
                        f"{drained} non-blocking dequeues; refusing a "
                        "potentially non-latest frame"
                    )
        self.drained_pending_framesets += drained
        self.last_drained_pending_framesets = drained
        return candidate.as_frameset(), drained

    def recent_frame_dequeue_diagnostics(self) -> tuple[Dict[str, Any], ...]:
        """Return a detached snapshot of the fixed-length diagnostic ring."""

        return tuple(dict(record) for record in self.frame_dequeue_diagnostics)

    def _diagnostic_tail_text(self) -> str:
        if not self.frame_dequeue_diagnostics:
            return "last_dequeue=none"
        record = self.frame_dequeue_diagnostics[-1]
        outcomes = ",".join(
            str(item.get("outcome", "unknown"))
            for item in tuple(self.frame_dequeue_diagnostics)[-4:]
        )
        pair_progress = ",".join(
            "{index}:{color}/{depth}:d{color_delta}/{depth_delta}:"
            "{progress}:{outcome}".format(
                index=item.get("dequeue_index"),
                color=item.get("color_frame_number"),
                depth=item.get("depth_frame_number"),
                color_delta=item.get("color_frame_delta"),
                depth_delta=item.get("depth_frame_delta"),
                progress=item.get("stream_progress"),
                outcome=item.get("outcome", "unknown"),
            )
            for item in tuple(self.frame_dequeue_diagnostics)[-8:]
        )
        return (
            "last_dequeue=(index={index},stage={stage},outcome={outcome},"
            "color_frame={color_frame},depth_frame={depth_frame},"
            "color_timestamp_ms={color_timestamp},"
            "depth_timestamp_ms={depth_timestamp},"
            "color_frame_timestamp_metadata={color_frame_timestamp},"
            "depth_frame_timestamp_metadata={depth_frame_timestamp},"
            "rgbd_sensor_skew_s={sensor_skew},"
            "rgbd_epoch_skew_s={epoch_skew},"
            "host_pair_span_s={pair_span},transport_age_s={transport_age},"
            "drained_pending_before_candidate={drained_before}); "
            "drained_pending_total={drained_total}; "
            "recent_outcomes={outcomes}; recent_pair_progress={pair_progress}"
        ).format(
            index=record.get("dequeue_index"),
            stage=record.get("stage"),
            outcome=record.get("outcome"),
            color_frame=record.get("color_frame_number"),
            depth_frame=record.get("depth_frame_number"),
            color_timestamp=record.get("color_timestamp_ms"),
            depth_timestamp=record.get("depth_timestamp_ms"),
            color_frame_timestamp=record.get(
                "color_frame_timestamp_metadata"
            ),
            depth_frame_timestamp=record.get(
                "depth_frame_timestamp_metadata"
            ),
            sensor_skew=record.get(
                "color_depth_sensor_timestamp_skew_s"
            ),
            epoch_skew=record.get(
                "color_depth_epoch_timestamp_skew_s"
            ),
            pair_span=record.get("host_clock_pair_span_s"),
            transport_age=record.get("transport_age_s"),
            drained_before=record.get(
                "drained_pending_framesets_before_candidate"
            ),
            drained_total=self.drained_pending_framesets,
            outcomes=outcomes,
            pair_progress=pair_progress,
        )

    def _validated_startup_settings(self) -> Dict[str, float | int]:
        raw_required = self.cfg.get("startup_required_consecutive_frames", 30)
        try:
            required = int(raw_required)
            raw_required_float = float(raw_required)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                "RealSense startup_required_consecutive_frames must be an integer"
            ) from exc
        if not np.isfinite(raw_required_float) or raw_required_float != required:
            raise ValueError(
                "RealSense startup_required_consecutive_frames must be an integer"
            )
        startup_timeout_s = float(self.cfg.get("startup_timeout_s", 5.0))
        clock_jitter_s = float(
            self.cfg.get("startup_max_clock_offset_jitter_s", 0.005)
        )
        maximum_color_depth_skew_s = float(
            self.cfg.get(
                "max_color_depth_timestamp_skew_s",
                _DEFAULT_MAX_COLOR_DEPTH_TIMESTAMP_SKEW_S,
            )
        )
        maximum_transport_age_s = float(
            self.cfg.get("max_capture_to_retrieval_s", 0.10)
        )
        maximum_future_skew_s = float(
            self.cfg.get("max_timestamp_future_skew_s", 0.05)
        )
        _StartupReadinessGate(
            required_consecutive_frames=required,
            maximum_clock_offset_jitter_s=clock_jitter_s,
        )
        if not np.isfinite(startup_timeout_s) or startup_timeout_s <= 0.0:
            raise ValueError(
                "RealSense startup_timeout_s must be finite and > 0"
            )
        timestamp_limits = (
            maximum_color_depth_skew_s,
            maximum_transport_age_s,
            maximum_future_skew_s,
        )
        if not all(np.isfinite(value) for value in timestamp_limits):
            raise ValueError("RealSense timestamp limits must be finite")
        if (
            maximum_color_depth_skew_s <= 0.0
            or maximum_transport_age_s <= 0.0
            or maximum_future_skew_s < 0.0
        ):
            raise ValueError("RealSense timestamp limits are invalid")
        return {
            "required_consecutive_frames": required,
            "startup_timeout_s": startup_timeout_s,
            "maximum_clock_offset_jitter_s": clock_jitter_s,
            "maximum_color_depth_skew_s": maximum_color_depth_skew_s,
            "maximum_transport_age_s": maximum_transport_age_s,
            "maximum_future_skew_s": maximum_future_skew_s,
        }

    def _publish_startup_gate_diagnostics(
        self, gate: _StartupReadinessGate
    ) -> None:
        self.startup_frames_observed = gate.observed_frames
        self.startup_rejected_frames = gate.rejected_frames
        self.startup_reset_count = gate.reset_count
        self.startup_consecutive_valid_frames = gate.consecutive_valid_frames
        self.startup_best_consecutive_valid_frames = (
            gate.best_consecutive_valid_frames
        )
        self.startup_last_rejection_reason = gate.last_rejection_reason
        self.startup_clock_offset_span_s = gate.sequence_clock_offset_span_s
        self.startup_skipped_sensor_frames = gate.skipped_sensor_frames
        self.last_retrieved_monotonic_s = gate.last_retrieved_monotonic_s
        self.last_host_realtime_minus_monotonic_s = (
            gate.last_host_realtime_minus_monotonic_s
        )

    def _wait_for_startup_readiness(
        self,
        settings: Dict[str, float | int],
        *,
        monotonic_fn=None,
        realtime_fn=None,
    ) -> None:
        """Wait for consecutive trustworthy frames, bounded by one deadline."""

        assert self.frame_queue is not None
        assert self.rs is not None
        monotonic_fn = monotonic_fn or time.monotonic
        realtime_fn = realtime_fn or time.time
        gate = _StartupReadinessGate(
            required_consecutive_frames=int(
                settings["required_consecutive_frames"]
            ),
            maximum_clock_offset_jitter_s=float(
                settings["maximum_clock_offset_jitter_s"]
            ),
        )
        timeout_s = float(settings["startup_timeout_s"])
        deadline_s = float(monotonic_fn()) + timeout_s
        last_reason = "no frame was retrieved"

        while True:
            before_wait_s = float(monotonic_fn())
            if before_wait_s >= deadline_s:
                break
            remaining_ms = max(
                1, int(math.ceil((deadline_s - before_wait_s) * 1000.0))
            )
            sensor_frame_number: Optional[int] = None
            depth_sensor_frame_number: Optional[int] = None
            retrieved_monotonic_s: Optional[float] = None
            retrieved_at_s: Optional[float] = None
            color_sensor_timestamp_us: Optional[float] = None
            depth_sensor_timestamp_us: Optional[float] = None
            diagnostic_record: Optional[Dict[str, Any]] = None
            try:
                frames, _drained = self._wait_for_latest_frameset(remaining_ms)
                (
                    host_monotonic_before_s,
                    retrieved_at_s,
                    host_monotonic_after_s,
                    host_clock_pair_span_s,
                ) = self._host_clock_sample(
                    monotonic_fn=monotonic_fn,
                    realtime_fn=realtime_fn,
                )
                retrieved_monotonic_s = host_monotonic_before_s
                self.last_host_clock_pair_span_s = host_clock_pair_span_s
                diagnostic_record, raw_color_frame, raw_depth_frame = (
                    self._record_dequeued_frameset(
                        stage="startup",
                        frames=frames,
                        host_monotonic_before_s=host_monotonic_before_s,
                        host_realtime_s=retrieved_at_s,
                        host_monotonic_after_s=host_monotonic_after_s,
                        host_clock_pair_span_s=host_clock_pair_span_s,
                    )
                )
                diagnostic_record[
                    "drained_pending_framesets_before_candidate"
                ] = _drained
                diagnostic_record["drained_pending_framesets_total"] = (
                    self.drained_pending_framesets
                )
                if host_monotonic_after_s >= deadline_s:
                    last_reason = "frame arrived after the startup deadline"
                    diagnostic_record["outcome"] = "arrived_after_deadline"
                    gate.reject(
                        last_reason,
                        retrieved_at_s=retrieved_at_s,
                        retrieved_monotonic_s=retrieved_monotonic_s,
                    )
                    self._publish_startup_gate_diagnostics(gate)
                    break

                if not raw_depth_frame or not raw_color_frame:
                    diagnostic_record["outcome"] = "missing_rgbd_stream"
                    raise RuntimeError(
                        "failed to get synchronized depth/color frame"
                    )
                sensor_frame_number = int(
                    diagnostic_record["color_frame_number"]
                )
                depth_sensor_frame_number = int(
                    diagnostic_record["depth_frame_number"]
                )
                color_domain = str(
                    diagnostic_record["color_timestamp_domain"]
                )
                depth_domain = str(
                    diagnostic_record["depth_timestamp_domain"]
                )
                color_sensor_timestamp_us = diagnostic_record[
                    "color_frame_timestamp_metadata"
                ]
                depth_sensor_timestamp_us = diagnostic_record[
                    "depth_frame_timestamp_metadata"
                ]
                capture_timestamp_s, _skew_s, _epoch_skew_s = (
                    _validated_epoch_capture_timestamp(
                        color_timestamp_ms=float(
                            diagnostic_record["color_timestamp_ms"]
                        ),
                        depth_timestamp_ms=float(
                            diagnostic_record["depth_timestamp_ms"]
                        ),
                        color_domain=color_domain,
                        depth_domain=depth_domain,
                        retrieved_at_s=retrieved_at_s,
                        color_sensor_timestamp_us=color_sensor_timestamp_us,
                        depth_sensor_timestamp_us=depth_sensor_timestamp_us,
                        maximum_color_depth_skew_s=float(
                            settings["maximum_color_depth_skew_s"]
                        ),
                        maximum_transport_age_s=float(
                            settings["maximum_transport_age_s"]
                        ),
                        maximum_future_skew_s=float(
                            settings["maximum_future_skew_s"]
                        ),
                    )
                )
                diagnostic_record["transport_age_s"] = (
                    retrieved_at_s - capture_timestamp_s
                )
                diagnostic_record["outcome"] = "startup_timestamp_valid"
            except Exception as exc:
                last_reason = f"{type(exc).__name__}: {exc}"
                if diagnostic_record is not None:
                    diagnostic_record["error"] = last_reason
                    if diagnostic_record.get("outcome") in (
                        None,
                        "dequeued",
                        "startup_timestamp_valid",
                    ):
                        diagnostic_record["outcome"] = "startup_rejected"
                if isinstance(exc, _ColorDepthTimestampSkewError):
                    self.rejected_timestamp_skew_frames += 1
                    if (
                        color_sensor_timestamp_us is not None
                        and depth_sensor_timestamp_us is not None
                    ):
                        self.last_rejected_color_depth_skew_s = abs(
                            color_sensor_timestamp_us
                            - depth_sensor_timestamp_us
                        ) * 1.0e-6
                if isinstance(exc, _CaptureTimestampStaleError):
                    self.rejected_transport_stale_frames += 1
                    self.last_rejected_transport_age_s = exc.transport_age_s
                    if diagnostic_record is not None:
                        diagnostic_record["transport_age_s"] = (
                            exc.transport_age_s
                        )
                        diagnostic_record["outcome"] = (
                            "startup_rejected_transport_stale"
                        )
                gate.reject(
                    last_reason,
                    sensor_frame_number=sensor_frame_number,
                    depth_sensor_frame_number=depth_sensor_frame_number,
                    retrieved_at_s=retrieved_at_s,
                    retrieved_monotonic_s=retrieved_monotonic_s,
                )
                self._publish_startup_gate_diagnostics(gate)
                continue

            reset_count_before = gate.reset_count
            ready = gate.observe_valid(
                sensor_frame_number=sensor_frame_number,
                depth_sensor_frame_number=depth_sensor_frame_number,
                retrieved_at_s=retrieved_at_s,
                retrieved_monotonic_s=retrieved_monotonic_s,
            )
            if gate.reset_count > reset_count_before:
                diagnostic_record["outcome"] = "startup_gate_rejected"
                diagnostic_record["error"] = gate.last_rejection_reason
            else:
                diagnostic_record["outcome"] = "startup_accepted"
            self._publish_startup_gate_diagnostics(gate)
            if ready:
                self.last_sensor_frame_number = gate.last_sensor_frame_number
                self.last_depth_sensor_frame_number = (
                    gate.last_depth_sensor_frame_number
                )
                self.last_accepted_sensor_frame_number = (
                    gate.last_sensor_frame_number
                )
                self.last_accepted_depth_sensor_frame_number = (
                    gate.last_depth_sensor_frame_number
                )
                return
            if gate.last_rejection_reason is not None:
                last_reason = gate.last_rejection_reason

        self._publish_startup_gate_diagnostics(gate)
        raise RuntimeError(
            "RealSense startup readiness gate timed out after "
            f"{timeout_s:.3f}s: required "
            f"{gate.required_consecutive_frames} consecutive fresh, synchronized, "
            "strictly increasing frames with a stable host clock; observed "
            f"{gate.observed_frames}, rejected {gate.rejected_frames}, best run "
            f"{gate.best_consecutive_valid_frames}; last rejection: {last_reason}; "
            f"{self._diagnostic_tail_text()}"
        )

    def start(self) -> None:
        if self.started:
            return
        settings = self._validated_startup_settings()
        import pyrealsense2 as rs

        self.started = False
        self.last_sensor_frame_number = None
        self.last_depth_sensor_frame_number = None
        self.last_accepted_sensor_frame_number = None
        self.last_accepted_depth_sensor_frame_number = None
        self.rejected_timestamp_skew_frames = 0
        self.last_rejected_color_depth_skew_s = None
        self.rejected_transport_stale_frames = 0
        self.last_rejected_transport_age_s = None
        self.rejected_future_skew_frames = 0
        self.last_rejected_future_skew_s = None
        self.rejected_non_increasing_frames = 0
        self.last_rejected_frame_sequence_reason = None
        self.dropped_queued_framesets = 0
        self.drained_pending_framesets = 0
        self.last_drained_pending_framesets = 0
        self.runtime_frame_wait_timeouts = 0
        self.last_frame_wait_started_monotonic_s = None
        self.last_frame_wait_failed_monotonic_s = None
        self.last_retrieved_monotonic_s = None
        self.last_host_realtime_minus_monotonic_s = None
        self.last_host_clock_pair_span_s = None
        self.frame_dequeue_count = 0
        self.frame_dequeue_diagnostics.clear()
        self.startup_frames_observed = 0
        self.startup_rejected_frames = 0
        self.startup_reset_count = 0
        self.startup_consecutive_valid_frames = 0
        self.startup_best_consecutive_valid_frames = 0
        self.startup_last_rejection_reason = None
        self.startup_clock_offset_span_s = None
        self.startup_skipped_sensor_frames = 0
        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()

        serial = self.cfg.get("serial")
        if serial:
            config.enable_device(str(serial))

        width = int(self.cfg.get("width", 848))
        height = int(self.cfg.get("height", 480))
        fps = int(self.cfg.get("fps", 30))

        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        try:
            self.frame_queue = rs.frame_queue(1, False)
            self.profile = self.pipeline.start(config, self.frame_queue)
            device = self.profile.get_device()
            self.depth_sensor = device.first_depth_sensor()
            first_color_sensor = getattr(device, "first_color_sensor", None)
            self.color_sensor = (
                None
                if first_color_sensor is None
                else first_color_sensor()
            )
            self.sdk_version = (
                None
                if getattr(rs, "__version__", None) is None
                else str(getattr(rs, "__version__"))
            )
            try:
                self.device_serial = str(
                    device.get_info(rs.camera_info.serial_number)
                )
            except Exception:
                self.device_serial = None
            try:
                self.device_name = str(device.get_info(rs.camera_info.name))
            except Exception:
                self.device_name = None
            try:
                self.device_firmware_version = str(
                    device.get_info(rs.camera_info.firmware_version)
                )
            except Exception:
                self.device_firmware_version = None
            try:
                self.device_usb_type_descriptor = str(
                    device.get_info(rs.camera_info.usb_type_descriptor)
                )
            except Exception:
                self.device_usb_type_descriptor = None
            self.depth_scale = float(self.depth_sensor.get_depth_scale())
            self.align = rs.align(rs.stream.color)
            self.spatial_filter = rs.spatial_filter()
            self.temporal_filter = rs.temporal_filter()
            self.hole_filter = rs.hole_filling_filter()

            self._configure_depth_sensor()
            self._configure_color_sensor()
            self._wait_for_startup_readiness(settings)
        except BaseException:
            # Fail closed: a timeout, clock jump, malformed frame, or caller
            # interruption must not leave a half-started pipeline running.
            if self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
            self.started = False
            raise

        self.started = True
        print(
            f"[RealSense] started, serial={self.device_serial or 'unknown'}, "
            f"firmware={self.device_firmware_version or 'unknown'}, "
            f"usb={self.device_usb_type_descriptor or 'unknown'}, "
            f"sdk={self.sdk_version or 'unknown'}, "
            f"depth_scale={self.depth_scale}, startup_frames="
            f"{self.startup_consecutive_valid_frames}, startup_rejected="
            f"{self.startup_rejected_frames}, clock_offset_span_s="
            f"{float(self.startup_clock_offset_span_s or 0.0):.6f}"
        )

    def _configure_depth_sensor(self) -> None:
        rs = self.rs
        s = self.depth_sensor
        if s is None:
            return
        manual_exposure = self.cfg.get("depth_exposure")
        manual_gain = self.cfg.get("depth_gain")
        auto = bool(
            self.cfg.get(
                "depth_auto_exposure",
                manual_exposure is None and manual_gain is None,
            )
        )
        if auto and (manual_exposure is not None or manual_gain is not None):
            raise ValueError(
                "manual depth exposure/gain is incompatible with "
                "depth_auto_exposure=true"
            )
        try:
            # A visual preset can reset advanced depth controls, including
            # laser power.  Apply it first, then write and verify the explicit
            # emitter/laser/exposure contract below.
            if s.supports(rs.option.visual_preset):
                preset = str(self.cfg.get("preset", "default"))
                if preset in PRESET_MAP:
                    s.set_option(rs.option.visual_preset, PRESET_MAP[preset])
                    print(f"[RealSense] visual_preset={preset}")
            if s.supports(rs.option.emitter_enabled):
                s.set_option(rs.option.emitter_enabled, 1 if self.cfg.get("emitter", True) else 0)
        except Exception as e:
            print(f"[RealSense][WARN] failed to configure sensor: {e}")

        requested_laser = self.cfg.get("laser_power")
        if requested_laser is not None:
            if not s.supports(rs.option.laser_power):
                raise RuntimeError("depth sensor does not support laser_power")
            power = float(requested_laser)
            rng = s.get_option_range(rs.option.laser_power)
            if (
                not np.isfinite(power)
                or power < float(rng.min)
                or power > float(rng.max)
            ):
                raise ValueError(
                    f"laser_power={power} is outside [{rng.min}, {rng.max}]"
                )
            s.set_option(rs.option.laser_power, power)
            applied_power = float(s.get_option(rs.option.laser_power))
            if abs(applied_power - power) > max(float(rng.step) * 0.5, 1.0e-6):
                raise RuntimeError(
                    "depth sensor laser_power readback differs from request: "
                    f"requested={power} actual={applied_power}"
                )
            print(f"[RealSense] laser_power={applied_power}")

        # High-speed stereo needs a reproducible integration time.  Keep the
        # legacy default on auto, but treat an explicitly requested manual
        # contract as strict: silently clipping or ignoring it would make a
        # recorded throw incomparable with deployment.
        if s.supports(rs.option.enable_auto_exposure):
            s.set_option(rs.option.enable_auto_exposure, 1.0 if auto else 0.0)
        elif not auto:
            raise RuntimeError("depth sensor does not support manual exposure")
        applied = {"auto": auto}
        requested_options = []
        if manual_exposure is not None:
            requested_options.append(("depth_exposure", rs.option.exposure))
        if manual_gain is not None:
            requested_options.append(("depth_gain", rs.option.gain))
        for key, option in requested_options:
            requested = self.cfg.get(key)
            if requested is None:
                continue
            if not s.supports(option):
                raise RuntimeError(f"depth sensor does not support {key}")
            value = float(requested)
            rng = s.get_option_range(option)
            if (
                not np.isfinite(value)
                or value < float(rng.min)
                or value > float(rng.max)
            ):
                raise ValueError(
                    f"{key}={value} is outside [{rng.min}, {rng.max}]"
                )
            s.set_option(option, value)
            actual = float(s.get_option(option))
            if abs(actual - value) > max(float(rng.step) * 0.5, 1.0e-6):
                raise RuntimeError(
                    f"depth sensor {key} readback differs from request: "
                    f"requested={value} actual={actual}"
                )
            applied[key] = actual
        print(
            "[RealSense] depth_exposure "
            + " ".join(f"{key}={value}" for key, value in applied.items())
        )

    def _configure_color_sensor(self) -> None:
        """Apply an explicit RGB exposure contract when one is requested.

        The default remains librealsense auto exposure.  A configured manual
        exposure is never silently clipped because that would make a recorded
        A/B incomparable with its manifest.
        """

        rs = self.rs
        sensor = self.color_sensor
        if sensor is None:
            return
        manual_exposure = self.cfg.get("color_exposure")
        manual_gain = self.cfg.get("color_gain")
        auto = bool(
            self.cfg.get(
                "color_auto_exposure",
                manual_exposure is None and manual_gain is None,
            )
        )
        if auto and (manual_exposure is not None or manual_gain is not None):
            raise ValueError(
                "manual color exposure/gain is incompatible with "
                "color_auto_exposure=true"
            )
        if sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(rs.option.enable_auto_exposure, 1.0 if auto else 0.0)
        elif not auto:
            raise RuntimeError("RGB sensor does not support manual exposure")

        applied = {"auto": auto}
        for key, option in (
            ("color_exposure", rs.option.exposure),
            ("color_gain", rs.option.gain),
        ):
            requested = self.cfg.get(key)
            if requested is None:
                continue
            if not sensor.supports(option):
                raise RuntimeError(f"RGB sensor does not support {key}")
            value = float(requested)
            rng = sensor.get_option_range(option)
            if (
                not np.isfinite(value)
                or value < float(rng.min)
                or value > float(rng.max)
            ):
                raise ValueError(
                    f"{key}={value} is outside [{rng.min}, {rng.max}]"
                )
            sensor.set_option(option, value)
            applied[key] = float(sensor.get_option(option))
        print(
            "[RealSense] color_exposure "
            + " ".join(f"{key}={value}" for key, value in applied.items())
        )

    def get_frame(self, timeout_ms: int = 1000) -> RGBDFrame:
        if not self.started:
            self.start()
        assert self.frame_queue is not None
        assert self.pipeline is not None
        assert self.align is not None

        timeout_s = float(timeout_ms) * 1.0e-3
        if not np.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("RealSense frame timeout must be finite and > 0")
        deadline = time.monotonic() + timeout_s
        stale_rejected_this_call = 0
        skew_rejected_this_call = 0
        future_rejected_this_call = 0
        sequence_rejected_this_call = 0
        accepted_diagnostic: Optional[Dict[str, Any]] = None
        while True:
            wait_started_s = time.monotonic()
            self.last_frame_wait_started_monotonic_s = wait_started_s
            if wait_started_s >= deadline:
                raise RetryableCameraTimeout(
                    "RealSense produced no fresh synchronized RGB-D frame "
                    f"within the fixed {timeout_s:.3f}s deadline; "
                    f"transport_stale_rejected_this_call="
                    f"{stale_rejected_this_call}, "
                    f"rgbd_skew_rejected_this_call={skew_rejected_this_call}; "
                    f"future_skew_rejected_this_call={future_rejected_this_call}; "
                    f"non_increasing_rejected_this_call="
                    f"{sequence_rejected_this_call}; "
                    f"{self._diagnostic_tail_text()}"
                )
            remaining_ms = max(
                1, int(math.ceil((deadline - wait_started_s) * 1000.0))
            )
            try:
                frames, drained_before_candidate = (
                    self._wait_for_latest_frameset(remaining_ms)
                )
            except RuntimeError as exc:
                failed_s = time.monotonic()
                self.runtime_frame_wait_timeouts += 1
                self.last_frame_wait_failed_monotonic_s = failed_s
                since_last_s = (
                    None
                    if self.last_retrieved_monotonic_s is None
                    else failed_s - self.last_retrieved_monotonic_s
                )
                since_last_text = (
                    "none" if since_last_s is None else f"{since_last_s:.6f}s"
                )
                raise RetryableCameraTimeout(
                    "RealSense runtime frame wait failed: "
                    f"requested={remaining_ms}ms, "
                    f"elapsed={failed_s - wait_started_s:.6f}s, "
                    f"since_last_retrieval={since_last_text}, "
                    f"last_frame_id={self.frame_id}, "
                    f"last_sensor_frame={self.last_sensor_frame_number}, "
                    "last_accepted_sensor_pair="
                    f"{self.last_accepted_sensor_frame_number}/"
                    f"{self.last_accepted_depth_sensor_frame_number}, "
                    f"timeouts={self.runtime_frame_wait_timeouts}, "
                    f"transport_stale_rejected_this_call="
                    f"{stale_rejected_this_call}, "
                    f"rgbd_skew_rejected_this_call={skew_rejected_this_call}; "
                    f"future_skew_rejected_this_call={future_rejected_this_call}; "
                    f"non_increasing_rejected_this_call="
                    f"{sequence_rejected_this_call}; "
                    f"SDK: {exc}; {self._diagnostic_tail_text()}"
                ) from exc
            (
                host_monotonic_before_s,
                retrieved_at_s,
                host_monotonic_after_s,
                host_clock_pair_span_s,
            ) = self._host_clock_sample()
            self.last_retrieved_monotonic_s = host_monotonic_before_s
            self.last_host_realtime_minus_monotonic_s = (
                retrieved_at_s - host_monotonic_before_s
            )
            self.last_host_clock_pair_span_s = host_clock_pair_span_s
            diagnostic_record, raw_color_frame, raw_depth_frame = (
                self._record_dequeued_frameset(
                    stage="runtime",
                    frames=frames,
                    host_monotonic_before_s=host_monotonic_before_s,
                    host_realtime_s=retrieved_at_s,
                    host_monotonic_after_s=host_monotonic_after_s,
                    host_clock_pair_span_s=host_clock_pair_span_s,
                )
            )
            diagnostic_record["drained_pending_framesets_before_candidate"] = (
                drained_before_candidate
            )
            diagnostic_record["drained_pending_framesets_total"] = (
                self.drained_pending_framesets
            )
            if host_monotonic_after_s >= deadline:
                diagnostic_record["outcome"] = "arrived_after_deadline"
                raise RetryableCameraTimeout(
                    "RealSense frame arrived after the fixed runtime deadline; "
                    f"timeout={timeout_s:.3f}s; "
                    f"{self._diagnostic_tail_text()}"
                )
            if not raw_depth_frame or not raw_color_frame:
                diagnostic_record["outcome"] = "missing_rgbd_stream"
                raise RuntimeError(
                    "failed to get synchronized depth/color frame; "
                    f"{self._diagnostic_tail_text()}"
                )
            sensor_frame_number = int(
                diagnostic_record["color_frame_number"]
            )
            depth_sensor_frame_number = int(
                diagnostic_record["depth_frame_number"]
            )
            if self.last_sensor_frame_number is not None:
                self.dropped_queued_framesets += max(
                    0, sensor_frame_number - self.last_sensor_frame_number - 1
                )
            self.last_sensor_frame_number = sensor_frame_number
            self.last_depth_sensor_frame_number = depth_sensor_frame_number
            color_domain = str(diagnostic_record["color_timestamp_domain"])
            depth_domain = str(diagnostic_record["depth_timestamp_domain"])
            color_timestamp_ms = float(
                diagnostic_record["color_timestamp_ms"]
            )
            depth_timestamp_ms = float(
                diagnostic_record["depth_timestamp_ms"]
            )
            color_sensor_timestamp_us = diagnostic_record[
                "color_frame_timestamp_metadata"
            ]
            depth_sensor_timestamp_us = diagnostic_record[
                "depth_frame_timestamp_metadata"
            ]
            try:
                (
                    capture_timestamp_s,
                    color_depth_timestamp_skew_s,
                    color_depth_epoch_timestamp_skew_s,
                ) = _validated_epoch_capture_timestamp(
                    color_timestamp_ms=color_timestamp_ms,
                    depth_timestamp_ms=depth_timestamp_ms,
                    color_domain=color_domain,
                    depth_domain=depth_domain,
                    retrieved_at_s=retrieved_at_s,
                    color_sensor_timestamp_us=color_sensor_timestamp_us,
                    depth_sensor_timestamp_us=depth_sensor_timestamp_us,
                    maximum_color_depth_skew_s=float(
                        self.cfg.get(
                            "max_color_depth_timestamp_skew_s",
                            _DEFAULT_MAX_COLOR_DEPTH_TIMESTAMP_SKEW_S,
                        )
                    ),
                    maximum_transport_age_s=float(
                        self.cfg.get("max_capture_to_retrieval_s", 0.10)
                    ),
                    maximum_future_skew_s=float(
                        self.cfg.get("max_timestamp_future_skew_s", 0.05)
                    ),
                )
                diagnostic_record["transport_age_s"] = (
                    retrieved_at_s - capture_timestamp_s
                )
                color_increases = (
                    self.last_accepted_sensor_frame_number is None
                    or sensor_frame_number
                    > self.last_accepted_sensor_frame_number
                )
                depth_increases = (
                    self.last_accepted_depth_sensor_frame_number is None
                    or depth_sensor_frame_number
                    > self.last_accepted_depth_sensor_frame_number
                )
                if not color_increases or not depth_increases:
                    sequence_rejected_this_call += 1
                    self.rejected_non_increasing_frames += 1
                    reason = (
                        "RealSense runtime sensor frame numbers must both "
                        "advance beyond the last accepted RGB-D pair: "
                        f"color={sensor_frame_number} after "
                        f"{self.last_accepted_sensor_frame_number}, "
                        f"depth={depth_sensor_frame_number} after "
                        f"{self.last_accepted_depth_sensor_frame_number}"
                    )
                    self.last_rejected_frame_sequence_reason = reason
                    diagnostic_record["outcome"] = (
                        "rejected_non_increasing_sequence"
                    )
                    diagnostic_record["error"] = reason
                    if time.monotonic() >= deadline:
                        raise _NonIncreasingFrameSequenceError(reason)
                    continue
                self.last_accepted_sensor_frame_number = sensor_frame_number
                self.last_accepted_depth_sensor_frame_number = (
                    depth_sensor_frame_number
                )
                diagnostic_record["rejected_non_increasing_frames_total"] = (
                    self.rejected_non_increasing_frames
                )
                diagnostic_record["outcome"] = "accepted"
                accepted_diagnostic = diagnostic_record
                break
            except _CaptureTimestampStaleError as exc:
                self.rejected_transport_stale_frames += 1
                stale_rejected_this_call += 1
                self.last_rejected_transport_age_s = exc.transport_age_s
                diagnostic_record["transport_age_s"] = exc.transport_age_s
                diagnostic_record["outcome"] = "rejected_transport_stale"
                diagnostic_record["error"] = str(exc)
                if time.monotonic() >= deadline:
                    raise RetryableCameraTimeout(
                        "RealSense produced no transport-fresh RGB-D frame "
                        f"within the fixed {timeout_s:.3f}s deadline; rejected "
                        f"{stale_rejected_this_call} transport-stale frames "
                        f"without changing the configured freshness limit; "
                        f"{self._diagnostic_tail_text()}"
                    ) from exc
            except _CaptureTimestampFutureSkewError as exc:
                self.rejected_future_skew_frames += 1
                future_rejected_this_call += 1
                self.last_rejected_future_skew_s = exc.future_skew_s
                diagnostic_record["transport_age_s"] = -exc.future_skew_s
                diagnostic_record["outcome"] = "rejected_future_skew"
                diagnostic_record["error"] = str(exc)
                if time.monotonic() >= deadline:
                    raise RetryableCameraTimeout(
                        "RealSense produced no non-future RGB-D frame within "
                        f"the fixed {timeout_s:.3f}s deadline; rejected "
                        f"{future_rejected_this_call} modest future-skew "
                        "frames without accepting or clamping them; "
                        f"{self._diagnostic_tail_text()}"
                    ) from exc
            except _ColorDepthTimestampSkewError as exc:
                color_s = color_timestamp_ms * 1.0e-3
                depth_s = depth_timestamp_ms * 1.0e-3
                self.rejected_timestamp_skew_frames += 1
                skew_rejected_this_call += 1
                if (
                    color_sensor_timestamp_us is not None
                    and depth_sensor_timestamp_us is not None
                ):
                    self.last_rejected_color_depth_skew_s = (
                        abs(color_sensor_timestamp_us - depth_sensor_timestamp_us)
                        * 1.0e-6
                    )
                else:
                    self.last_rejected_color_depth_skew_s = abs(color_s - depth_s)
                diagnostic_record["outcome"] = "rejected_rgbd_skew"
                diagnostic_record["error"] = str(exc)
                if time.monotonic() >= deadline:
                    raise RetryableCameraTimeout(
                        "RealSense produced no synchronized RGB-D frame within "
                        f"the fixed {timeout_s:.3f}s deadline; rejected "
                        f"{skew_rejected_this_call} skewed frames this call; "
                        f"{self._diagnostic_tail_text()}"
                    ) from exc
            except _NonIncreasingFrameSequenceError as exc:
                raise RetryableCameraTimeout(
                    "RealSense produced no strictly newer RGB-D frame "
                    f"within the fixed {timeout_s:.3f}s deadline; rejected "
                    f"{sequence_rejected_this_call} repeated or regressed "
                    "framesets without accepting them; "
                    f"{self._diagnostic_tail_text()}"
                ) from exc
            except Exception as exc:
                # Domain, far-future, non-finite, and configuration errors are
                # not recoverable frame candidates and remain immediately
                # fatal.  Modest future skew is handled above only by dropping
                # that frame and waiting within the original fixed deadline.
                diagnostic_record["outcome"] = "fatal_timestamp_validation"
                diagnostic_record["error"] = f"{type(exc).__name__}: {exc}"
                raise RuntimeError(
                    "RealSense fatal timestamp validation failure (not "
                    f"retryable): {type(exc).__name__}: {exc}; "
                    f"{self._diagnostic_tail_text()}"
                ) from exc

        assert accepted_diagnostic is not None

        frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("failed to get aligned depth/color frame")

        if self.cfg.get("spatial_filter", False):
            depth_frame = self.spatial_filter.process(depth_frame).as_depth_frame()
        if self.cfg.get("temporal_filter", False):
            depth_frame = self.temporal_filter.process(depth_frame).as_depth_frame()
        if self.cfg.get("hole_filter", False):
            depth_frame = self.hole_filter.process(depth_frame).as_depth_frame()

        depth_raw = np.asanyarray(depth_frame.get_data()).copy()
        color_bgr = np.asanyarray(color_frame.get_data()).copy()

        # Depth was aligned to color above. Use the color optical intrinsics so
        # back-projected depth and the calibrated camera frame are identical.
        intr_rs = color_frame.profile.as_video_stream_profile().get_intrinsics()
        intr = CameraIntrinsics(
            width=int(intr_rs.width),
            height=int(intr_rs.height),
            fx=float(intr_rs.fx),
            fy=float(intr_rs.fy),
            ppx=float(intr_rs.ppx),
            ppy=float(intr_rs.ppy),
            model=str(intr_rs.model),
            distortion=tuple(float(value) for value in intr_rs.coeffs),
        )

        self.frame_id += 1
        return RGBDFrame(
            color_bgr=color_bgr,
            depth_raw=depth_raw,
            depth_scale=self.depth_scale,
            intrinsics=intr,
            timestamp=capture_timestamp_s,
            frame_id=self.frame_id,
            retrieved_at_s=retrieved_at_s,
            timestamp_domain=color_domain,
            depth_timestamp_s=depth_timestamp_ms * 1.0e-3,
            color_depth_timestamp_skew_s=color_depth_timestamp_skew_s,
            color_depth_epoch_timestamp_skew_s=(
                color_depth_epoch_timestamp_skew_s
            ),
            rejected_timestamp_skew_frames=self.rejected_timestamp_skew_frames,
            last_rejected_color_depth_skew_s=(
                self.last_rejected_color_depth_skew_s
            ),
            dropped_queued_framesets=self.dropped_queued_framesets,
            sensor_frame_number=sensor_frame_number,
            depth_sensor_frame_number=depth_sensor_frame_number,
            rejected_transport_stale_frames=(
                self.rejected_transport_stale_frames
            ),
            last_rejected_transport_age_s=self.last_rejected_transport_age_s,
            retrieved_monotonic_s=host_monotonic_before_s,
            host_clock_pair_span_s=host_clock_pair_span_s,
            capture_diagnostic=dict(accepted_diagnostic),
        )

    def stop(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        self.started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
