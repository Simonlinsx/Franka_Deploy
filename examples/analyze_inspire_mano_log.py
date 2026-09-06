#!/usr/bin/env python3
"""Summarize a RealSense -> MANO -> Inspire JSONL run without touching hardware."""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence


AXES = ("pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotate")
PALM_DEPTH_SOURCES = ("measured", "held", "missing")


class LogFormatError(ValueError):
    """A JSONL row cannot be analyzed without producing misleading statistics."""


def _finite_number(value, field: str, line_number: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LogFormatError(f"line {line_number}: {field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise LogFormatError(f"line {line_number}: {field} must be finite")
    return result


def _percentile(values: Sequence[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


@dataclass
class _VectorAccumulator:
    field: str
    sample_frames: int = 0
    previous: Optional[tuple[float, ...]] = None
    previous_record_index: Optional[int] = None

    def __post_init__(self) -> None:
        self.minimums: list[Optional[float]] = [None] * len(AXES)
        self.maximums: list[Optional[float]] = [None] * len(AXES)
        self.max_adjacent_jumps: list[Optional[float]] = [None] * len(AXES)

    def add(
        self, value, record_index: int, line_number: int
    ) -> Optional[tuple[float, ...]]:
        if value is None:
            return None
        if not isinstance(value, list) or len(value) != len(AXES):
            raise LogFormatError(
                f"line {line_number}: {self.field} must be null or a six-number array"
            )
        vector = tuple(
            _finite_number(item, f"{self.field}[{axis}]", line_number)
            for axis, item in zip(AXES, value)
        )
        for index, item in enumerate(vector):
            old_min = self.minimums[index]
            old_max = self.maximums[index]
            self.minimums[index] = item if old_min is None else min(old_min, item)
            self.maximums[index] = item if old_max is None else max(old_max, item)
        if (
            self.previous is not None
            and self.previous_record_index == record_index - 1
        ):
            for index, (before, after) in enumerate(zip(self.previous, vector)):
                jump = abs(after - before)
                old_jump = self.max_adjacent_jumps[index]
                self.max_adjacent_jumps[index] = (
                    jump if old_jump is None else max(old_jump, jump)
                )
        self.previous = vector
        self.previous_record_index = record_index
        self.sample_frames += 1
        return vector

    def summary(self) -> dict:
        return {
            "sample_frames": self.sample_frames,
            "axes": {
                axis: {
                    "min": self.minimums[index],
                    "max": self.maximums[index],
                    "max_adjacent_frame_jump": self.max_adjacent_jumps[index],
                }
                for index, axis in enumerate(AXES)
            },
        }


@dataclass
class _RawFilteredAccumulator:
    """Compare optional raw vectors with the control-facing filtered vectors.

    All range, adjustment, and jitter statistics use paired rows only.  A
    missing raw or filtered vector therefore cannot silently skew one side of
    the comparison.  Jitter is the absolute change between consecutive log
    rows which both contain a pair; detection gaps intentionally break
    adjacency, matching ``_VectorAccumulator`` semantics.
    """

    raw_field: str
    filtered_field: str
    paired_sample_frames: int = 0
    adjacent_pair_count: int = 0
    previous_raw: Optional[tuple[float, ...]] = None
    previous_filtered: Optional[tuple[float, ...]] = None
    previous_record_index: Optional[int] = None

    def __post_init__(self) -> None:
        self.raw_minimums: list[Optional[float]] = [None] * len(AXES)
        self.raw_maximums: list[Optional[float]] = [None] * len(AXES)
        self.filtered_minimums: list[Optional[float]] = [None] * len(AXES)
        self.filtered_maximums: list[Optional[float]] = [None] * len(AXES)
        self.raw_adjacent_jumps: list[list[float]] = [[] for _ in AXES]
        self.filtered_adjacent_jumps: list[list[float]] = [[] for _ in AXES]
        self.abs_adjustment_sums: list[float] = [0.0] * len(AXES)
        self.max_abs_adjustments: list[Optional[float]] = [None] * len(AXES)

    def add(
        self,
        raw: Optional[tuple[float, ...]],
        filtered: Optional[tuple[float, ...]],
        record_index: int,
    ) -> None:
        if raw is None or filtered is None:
            return
        for index, (raw_value, filtered_value) in enumerate(zip(raw, filtered)):
            old_raw_min = self.raw_minimums[index]
            old_raw_max = self.raw_maximums[index]
            old_filtered_min = self.filtered_minimums[index]
            old_filtered_max = self.filtered_maximums[index]
            self.raw_minimums[index] = (
                raw_value if old_raw_min is None else min(old_raw_min, raw_value)
            )
            self.raw_maximums[index] = (
                raw_value if old_raw_max is None else max(old_raw_max, raw_value)
            )
            self.filtered_minimums[index] = (
                filtered_value
                if old_filtered_min is None
                else min(old_filtered_min, filtered_value)
            )
            self.filtered_maximums[index] = (
                filtered_value
                if old_filtered_max is None
                else max(old_filtered_max, filtered_value)
            )
            adjustment = abs(filtered_value - raw_value)
            self.abs_adjustment_sums[index] += adjustment
            old_adjustment = self.max_abs_adjustments[index]
            self.max_abs_adjustments[index] = (
                adjustment
                if old_adjustment is None
                else max(old_adjustment, adjustment)
            )

        if (
            self.previous_raw is not None
            and self.previous_filtered is not None
            and self.previous_record_index == record_index - 1
        ):
            self.adjacent_pair_count += 1
            for index, (raw_before, raw_after) in enumerate(
                zip(self.previous_raw, raw)
            ):
                self.raw_adjacent_jumps[index].append(abs(raw_after - raw_before))
            for index, (filtered_before, filtered_after) in enumerate(
                zip(self.previous_filtered, filtered)
            ):
                self.filtered_adjacent_jumps[index].append(
                    abs(filtered_after - filtered_before)
                )

        self.previous_raw = raw
        self.previous_filtered = filtered
        self.previous_record_index = record_index
        self.paired_sample_frames += 1

    @staticmethod
    def _span(
        minimum: Optional[float], maximum: Optional[float]
    ) -> Optional[float]:
        if minimum is None or maximum is None:
            return None
        return maximum - minimum

    def summary(self, raw_sample_frames: int, filtered_sample_frames: int) -> dict:
        axes = {}
        for index, axis in enumerate(AXES):
            raw_p95 = _percentile(self.raw_adjacent_jumps[index], 0.95)
            filtered_p95 = _percentile(
                self.filtered_adjacent_jumps[index], 0.95
            )
            axes[axis] = {
                "raw_min": self.raw_minimums[index],
                "raw_max": self.raw_maximums[index],
                "raw_span": self._span(
                    self.raw_minimums[index], self.raw_maximums[index]
                ),
                "filtered_min": self.filtered_minimums[index],
                "filtered_max": self.filtered_maximums[index],
                "filtered_span": self._span(
                    self.filtered_minimums[index],
                    self.filtered_maximums[index],
                ),
                "raw_adjacent_jump_p95": raw_p95,
                "raw_adjacent_jump_max": (
                    max(self.raw_adjacent_jumps[index])
                    if self.raw_adjacent_jumps[index]
                    else None
                ),
                "filtered_adjacent_jump_p95": filtered_p95,
                "filtered_adjacent_jump_max": (
                    max(self.filtered_adjacent_jumps[index])
                    if self.filtered_adjacent_jumps[index]
                    else None
                ),
                "adjacent_jump_p95_reduction": (
                    raw_p95 - filtered_p95
                    if raw_p95 is not None and filtered_p95 is not None
                    else None
                ),
                "mean_abs_raw_to_filtered_delta": (
                    self.abs_adjustment_sums[index] / self.paired_sample_frames
                    if self.paired_sample_frames
                    else None
                ),
                "max_abs_raw_to_filtered_delta": self.max_abs_adjustments[index],
            }
        return {
            "available": self.paired_sample_frames > 0,
            "raw_sample_frames": int(raw_sample_frames),
            "filtered_sample_frames": int(filtered_sample_frames),
            "paired_sample_frames": self.paired_sample_frames,
            "adjacent_pair_count": self.adjacent_pair_count,
            "axes": axes,
        }


@dataclass
class _HardwareFeedbackAccumulator:
    """Aggregate optional per-frame RH56 feedback without requiring it."""

    actual_angle_sample_frames: int = 0
    current_sample_frames: int = 0
    temperature_sample_frames: int = 0
    error_sample_frames: int = 0
    frames_with_any_nonzero_error: int = 0

    def __post_init__(self) -> None:
        self.actual_angle_minimums: list[Optional[float]] = [None] * len(AXES)
        self.actual_angle_maximums: list[Optional[float]] = [None] * len(AXES)
        self.max_abs_currents: list[Optional[float]] = [None] * len(AXES)
        self.max_temperatures: list[Optional[float]] = [None] * len(AXES)
        self.nonzero_error_frames: list[int] = [0] * len(AXES)

    @staticmethod
    def _vector(value, field: str, line_number: int) -> Optional[tuple[float, ...]]:
        if value is None:
            return None
        if not isinstance(value, list) or len(value) != len(AXES):
            raise LogFormatError(
                f"line {line_number}: {field} must be null or a six-number array"
            )
        return tuple(
            _finite_number(item, f"{field}[{axis}]", line_number)
            for axis, item in zip(AXES, value)
        )

    def add(self, row: Mapping[str, object], line_number: int) -> None:
        actual_angles = self._vector(
            row.get("actual_angles"), "actual_angles", line_number
        )
        if actual_angles is not None:
            self.actual_angle_sample_frames += 1
            for index, value in enumerate(actual_angles):
                old_min = self.actual_angle_minimums[index]
                old_max = self.actual_angle_maximums[index]
                self.actual_angle_minimums[index] = (
                    value if old_min is None else min(old_min, value)
                )
                self.actual_angle_maximums[index] = (
                    value if old_max is None else max(old_max, value)
                )

        currents = self._vector(row.get("currents"), "currents", line_number)
        if currents is not None:
            self.current_sample_frames += 1
            for index, value in enumerate(currents):
                magnitude = abs(value)
                old_max = self.max_abs_currents[index]
                self.max_abs_currents[index] = (
                    magnitude if old_max is None else max(old_max, magnitude)
                )

        temperatures = self._vector(
            row.get("temperatures"), "temperatures", line_number
        )
        if temperatures is not None:
            self.temperature_sample_frames += 1
            for index, value in enumerate(temperatures):
                old_max = self.max_temperatures[index]
                self.max_temperatures[index] = (
                    value if old_max is None else max(old_max, value)
                )

        errors = self._vector(row.get("errors"), "errors", line_number)
        if errors is not None:
            self.error_sample_frames += 1
            has_nonzero_error = False
            for index, value in enumerate(errors):
                if value != 0:
                    self.nonzero_error_frames[index] += 1
                    has_nonzero_error = True
            self.frames_with_any_nonzero_error += int(has_nonzero_error)

    def summary(self) -> dict:
        available = bool(
            self.actual_angle_sample_frames
            or self.current_sample_frames
            or self.temperature_sample_frames
            or self.error_sample_frames
        )
        return {
            "available": available,
            "sample_frames": {
                "actual_angles": self.actual_angle_sample_frames,
                "currents": self.current_sample_frames,
                "temperatures": self.temperature_sample_frames,
                "errors": self.error_sample_frames,
            },
            "frames_with_any_nonzero_error": (
                self.frames_with_any_nonzero_error
                if self.error_sample_frames
                else None
            ),
            "axes": {
                axis: {
                    "actual_angle_min": self.actual_angle_minimums[index],
                    "actual_angle_max": self.actual_angle_maximums[index],
                    "max_abs_current_ma": self.max_abs_currents[index],
                    "max_temperature_c": self.max_temperatures[index],
                    "nonzero_error_frames": (
                        self.nonzero_error_frames[index]
                        if self.error_sample_frames
                        else None
                    ),
                }
                for index, axis in enumerate(AXES)
            },
        }


def _detection_value(row: Mapping[str, object], line_number: int) -> tuple[bool, bool]:
    """Return (right hand detected, row has the per-frame-v2 marker)."""

    if "right_hand_detected" in row:
        value = row["right_hand_detected"]
        if not isinstance(value, bool):
            raise LogFormatError(
                f"line {line_number}: right_hand_detected must be a JSON boolean"
            )
        # The current pipeline only emits right-hand detections.  Honor an
        # explicit false handedness if a future producer logs both hands.
        return value and row.get("is_right") is not False, True
    return row.get("is_right") is True, False


def _shutdown_not_checked() -> dict:
    return {
        "checked": False,
        "file_present": None,
        "path": None,
        "session_id": None,
        "session_matches_log": None,
        "stop_confirmed": None,
        "physical_stop_verified": None,
        "active_current_policy": None,
        "host_current_per_axis_threshold_ma": None,
        "host_current_selected_total_threshold_ma": None,
        "active_current_over_limit_sample_count": None,
        "active_current_warning_event_count": None,
        "active_peak_abs_currents": None,
        "active_max_selected_total_current_ma": None,
        "verified_device_current_limits": None,
        "shutdown_feedback": None,
        "shutdown_feedback_idle": None,
        "final_angle_targets": None,
        "final_all_minus_one": None,
        "safe_stop_verified": None,
        "verification_status": "not_checked",
    }


def _miss_summary(frames: Sequence[tuple[bool, float, object]]) -> dict:
    best = None
    start = None
    for index in range(len(frames) + 1):
        is_miss = index < len(frames) and not frames[index][0]
        if is_miss and start is None:
            start = index
        if not is_miss and start is not None:
            end = index - 1
            span = max(0.0, frames[end][1] - frames[start][1])
            candidate = {
                "frames": end - start + 1,
                "sample_span_seconds": span,
                "start_frame_number": frames[start][2],
                "end_frame_number": frames[end][2],
                "leading_censored": start == 0,
                "trailing_censored": end == len(frames) - 1,
            }
            rank = (candidate["frames"], candidate["sample_span_seconds"])
            if best is None or rank > best[0]:
                best = (rank, candidate)
            start = None
    if best is None:
        return {
            "frames": 0,
            "sample_span_seconds": 0.0,
            "start_frame_number": None,
            "end_frame_number": None,
            "leading_censored": False,
            "trailing_censored": False,
        }
    return best[1]


def _analyze_numbered_records(
    records: Iterable[tuple[int, Mapping[str, object]]]
) -> dict:
    rows = 0
    explicit_detection_rows = 0
    right_detections = 0
    latencies_ms: list[float] = []
    detected_latencies_ms: list[float] = []
    undetected_latencies_ms: list[float] = []
    pipeline_ages_ms: list[float] = []
    frame_copy_align_ms: list[float] = []
    rgbd_sensor_timestamp_deltas_ms: list[float] = []
    rgbd_frame_number_deltas: list[int] = []
    wilor_result_counts: collections.Counter[str] = collections.Counter()
    wilor_label_counts: collections.Counter[str] = collections.Counter()
    wilor_candidate_counts: list[int] = []
    depths: list[float] = []
    control_depths: list[float] = []
    palm_depth_source_counts: collections.Counter[str] = collections.Counter()
    palm_depth_reason_counts: collections.Counter[str] = collections.Counter()
    depth_range_flags = 0
    depth_range_true = 0
    tracking_ages: list[float] = []
    frame_states: list[tuple[bool, float, object]] = []
    qpos = _VectorAccumulator("qpos")
    raw_qpos = _VectorAccumulator("raw_qpos")
    targets = _VectorAccumulator("hardware_targets")
    raw_targets = _VectorAccumulator("raw_hardware_targets")
    masked_targets = _VectorAccumulator("masked_hardware_targets")
    sent_targets = _VectorAccumulator("last_sent_targets")
    qpos_filtering = _RawFilteredAccumulator("raw_qpos", "qpos")
    target_filtering = _RawFilteredAccumulator(
        "raw_hardware_targets", "hardware_targets"
    )
    hardware_feedback = _HardwareFeedbackAccumulator()
    hardware_states: set[str] = set()
    active_frames = 0
    submit_attempted_frames = 0
    submit_accepted_frames = 0
    submit_rejected_frames = 0
    motion_observation_frames = 0
    motion_target_changes = 0
    distinct_motion_targets: set[tuple[float, ...]] = set()
    previous_motion_target: Optional[tuple[float, ...]] = None
    schema_versions: set[object] = set()
    session_ids: set[str] = set()
    previous_timestamp: Optional[float] = None

    for record_index, (line_number, row) in enumerate(records, start=1):
        if not isinstance(row, Mapping):
            raise LogFormatError(f"line {line_number}: each JSONL value must be an object")
        rows += 1
        detected, explicit = _detection_value(row, line_number)
        explicit_detection_rows += int(explicit)
        right_detections += int(detected)

        diagnostics = row.get("wilor_diagnostics")
        if diagnostics is not None:
            if not isinstance(diagnostics, Mapping):
                raise LogFormatError(
                    f"line {line_number}: wilor_diagnostics must be an object or null"
                )
            result = diagnostics.get("result")
            candidate_count = diagnostics.get("candidate_count")
            labels = diagnostics.get("detector_labels")
            if not isinstance(result, str):
                raise LogFormatError(
                    f"line {line_number}: wilor_diagnostics.result must be a string"
                )
            if (
                isinstance(candidate_count, bool)
                or not isinstance(candidate_count, int)
                or candidate_count < 0
            ):
                raise LogFormatError(
                    f"line {line_number}: wilor_diagnostics.candidate_count "
                    "must be a non-negative integer"
                )
            if not isinstance(labels, list) or any(
                label not in ("left", "right") for label in labels
            ):
                raise LogFormatError(
                    f"line {line_number}: wilor_diagnostics.detector_labels "
                    "must be an array of left/right labels"
                )
            wilor_result_counts[result] += 1
            wilor_candidate_counts.append(candidate_count)
            wilor_label_counts.update(labels)

        timestamp = _finite_number(
            row.get("captured_at_monotonic"), "captured_at_monotonic", line_number
        )
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise LogFormatError(
                f"line {line_number}: captured_at_monotonic moved backwards"
            )
        previous_timestamp = timestamp
        frame_number = row.get("frame_number", record_index)
        frame_states.append((detected, timestamp, frame_number))

        latency = row.get("inference_seconds")
        if latency is not None:
            latency_seconds = _finite_number(latency, "inference_seconds", line_number)
            if latency_seconds < 0:
                raise LogFormatError(
                    f"line {line_number}: inference_seconds cannot be negative"
                )
            latencies_ms.append(latency_seconds * 1000.0)
            (detected_latencies_ms if detected else undetected_latencies_ms).append(
                latency_seconds * 1000.0
            )

        pipeline_age = row.get("pipeline_age_seconds")
        if pipeline_age is not None:
            age_seconds = _finite_number(
                pipeline_age, "pipeline_age_seconds", line_number
            )
            if age_seconds < 0:
                raise LogFormatError(
                    f"line {line_number}: pipeline_age_seconds cannot be negative"
                )
            pipeline_ages_ms.append(age_seconds * 1000.0)

        copy_align = row.get("frame_copy_align_seconds")
        if copy_align is not None:
            copy_align_seconds = _finite_number(
                copy_align, "frame_copy_align_seconds", line_number
            )
            if copy_align_seconds < 0:
                raise LogFormatError(
                    f"line {line_number}: frame_copy_align_seconds cannot be negative"
                )
            frame_copy_align_ms.append(copy_align_seconds * 1000.0)

        color_sensor_timestamp = row.get("color_sensor_timestamp_ms")
        depth_sensor_timestamp = row.get("depth_sensor_timestamp_ms")
        if color_sensor_timestamp is not None and depth_sensor_timestamp is not None:
            color_timestamp = _finite_number(
                color_sensor_timestamp, "color_sensor_timestamp_ms", line_number
            )
            depth_timestamp = _finite_number(
                depth_sensor_timestamp, "depth_sensor_timestamp_ms", line_number
            )
            rgbd_sensor_timestamp_deltas_ms.append(
                abs(color_timestamp - depth_timestamp)
            )

        depth_frame_number = row.get("depth_frame_number")
        if depth_frame_number is not None:
            if (
                isinstance(frame_number, bool)
                or not isinstance(frame_number, int)
                or isinstance(depth_frame_number, bool)
                or not isinstance(depth_frame_number, int)
            ):
                raise LogFormatError(
                    f"line {line_number}: RGB-D frame numbers must be integers"
                )
            rgbd_frame_number_deltas.append(frame_number - depth_frame_number)

        if detected:
            depth = row.get("palm_depth_m")
            if depth is not None:
                depth_value = _finite_number(depth, "palm_depth_m", line_number)
                if depth_value > 0:
                    depths.append(depth_value)
            # Before the stabilizer was introduced, palm_depth_m was both the
            # measured and control-facing depth.  Use it as a fallback only
            # when the new field is absent; an explicitly null control depth
            # must remain invalid.
            control_depth = (
                row.get("control_palm_depth_m")
                if "control_palm_depth_m" in row
                else depth
            )
            if control_depth is not None:
                control_depth_value = _finite_number(
                    control_depth, "control_palm_depth_m", line_number
                )
                if control_depth_value > 0:
                    control_depths.append(control_depth_value)

            palm_depth_source = row.get("palm_depth_source")
            if palm_depth_source is not None:
                if palm_depth_source not in PALM_DEPTH_SOURCES:
                    raise LogFormatError(
                        f"line {line_number}: palm_depth_source must be one of "
                        + ", ".join(PALM_DEPTH_SOURCES)
                    )
                palm_depth_source_counts[palm_depth_source] += 1

            palm_depth_reason = row.get("palm_depth_reason")
            if palm_depth_reason is not None:
                if not isinstance(palm_depth_reason, str) or not palm_depth_reason:
                    raise LogFormatError(
                        f"line {line_number}: palm_depth_reason must be a "
                        "non-empty string or null"
                    )
                palm_depth_reason_counts[palm_depth_reason] += 1
            in_range = row.get("palm_depth_in_calibrated_range")
            if in_range is not None:
                if not isinstance(in_range, bool):
                    raise LogFormatError(
                        f"line {line_number}: palm_depth_in_calibrated_range "
                        "must be a JSON boolean or null"
                    )
                depth_range_flags += 1
                depth_range_true += int(in_range)

        tracking_age = row.get("tracking_age_seconds")
        if tracking_age is not None:
            age = _finite_number(tracking_age, "tracking_age_seconds", line_number)
            if age >= 0:
                tracking_ages.append(age)

        filtered_qpos = qpos.add(row.get("qpos"), record_index, line_number)
        unfiltered_qpos = raw_qpos.add(
            row.get("raw_qpos"), record_index, line_number
        )
        qpos_filtering.add(unfiltered_qpos, filtered_qpos, record_index)
        filtered_targets = targets.add(
            row.get("hardware_targets"), record_index, line_number
        )
        unfiltered_targets = raw_targets.add(
            row.get("raw_hardware_targets"), record_index, line_number
        )
        target_filtering.add(
            unfiltered_targets, filtered_targets, record_index
        )
        masked_targets.add(
            row.get("masked_hardware_targets"), record_index, line_number
        )
        sent = sent_targets.add(
            row.get("last_sent_targets"), record_index, line_number
        )
        hardware_feedback.add(row, line_number)

        hardware_state = row.get("hardware_state")
        if hardware_state is not None:
            if not isinstance(hardware_state, str):
                raise LogFormatError(
                    f"line {line_number}: hardware_state must be a string or null"
                )
            hardware_states.add(hardware_state)
            active_frames += int(hardware_state == "active")

        accepted = row.get("hardware_submit_accepted")
        if accepted is not None:
            if not isinstance(accepted, bool):
                raise LogFormatError(
                    f"line {line_number}: hardware_submit_accepted must be a "
                    "JSON boolean or null"
                )
            submit_attempted_frames += 1
            submit_accepted_frames += int(accepted)
            submit_rejected_frames += int(not accepted)

        if sent is not None and any(value != -1 for value in sent):
            motion_observation_frames += 1
            distinct_motion_targets.add(sent)
            if previous_motion_target is not None and sent != previous_motion_target:
                motion_target_changes += 1
            previous_motion_target = sent

        if "log_schema_version" in row:
            schema_versions.add(row["log_schema_version"])
        if row.get("session_id") is not None:
            session_ids.add(str(row["session_id"]))

    if rows == 0:
        log_schema = "empty"
    elif explicit_detection_rows == rows:
        log_schema = "per_frame_v2"
    elif explicit_detection_rows == 0:
        log_schema = "legacy_detection_only"
    else:
        log_schema = "mixed"
    detection_metrics_available = log_schema == "per_frame_v2"
    total_frames = rows if detection_metrics_available else (0 if rows == 0 else None)
    detection_rate = (
        right_detections / rows if detection_metrics_available and rows else None
    )
    longest_miss = _miss_summary(frame_states) if detection_metrics_available else None
    hardware_feedback_summary = hardware_feedback.summary()

    return {
        "analyzer_schema_version": 4,
        "status": "ok" if detection_metrics_available else ("empty" if not rows else "partial"),
        "log_schema": log_schema,
        "log_rows": rows,
        "total_frames": total_frames,
        "right_hand": {
            "detected_frames": right_detections,
            "detection_rate": detection_rate,
        },
        "wilor": {
            "diagnostic_frames": len(wilor_candidate_counts),
            "frames_with_candidates": sum(
                count > 0 for count in wilor_candidate_counts
            ),
            "candidate_count_max": (
                max(wilor_candidate_counts) if wilor_candidate_counts else None
            ),
            "result_counts": dict(sorted(wilor_result_counts.items())),
            "detector_label_counts": dict(sorted(wilor_label_counts.items())),
        },
        "inference_latency": {
            "sample_count": len(latencies_ms),
            "p50_ms": _percentile(latencies_ms, 0.50),
            "p95_ms": _percentile(latencies_ms, 0.95),
            "detected_hand": {
                "sample_count": len(detected_latencies_ms),
                "p50_ms": _percentile(detected_latencies_ms, 0.50),
                "p95_ms": _percentile(detected_latencies_ms, 0.95),
            },
            "no_detected_hand": {
                "sample_count": len(undetected_latencies_ms),
                "p50_ms": _percentile(undetected_latencies_ms, 0.50),
                "p95_ms": _percentile(undetected_latencies_ms, 0.95),
            },
        },
        "pipeline_age": {
            "sample_count": len(pipeline_ages_ms),
            "p50_ms": _percentile(pipeline_ages_ms, 0.50),
            "p95_ms": _percentile(pipeline_ages_ms, 0.95),
        },
        "frame_copy_align": {
            "sample_count": len(frame_copy_align_ms),
            "p50_ms": _percentile(frame_copy_align_ms, 0.50),
            "p95_ms": _percentile(frame_copy_align_ms, 0.95),
        },
        "rgbd_sync": {
            "sensor_timestamp_sample_count": len(
                rgbd_sensor_timestamp_deltas_ms
            ),
            "sensor_timestamp_abs_delta_p50_ms": _percentile(
                rgbd_sensor_timestamp_deltas_ms, 0.50
            ),
            "sensor_timestamp_abs_delta_p95_ms": _percentile(
                rgbd_sensor_timestamp_deltas_ms, 0.95
            ),
            "frame_number_sample_count": len(rgbd_frame_number_deltas),
            "frame_number_abs_delta_max": (
                max(abs(value) for value in rgbd_frame_number_deltas)
                if rgbd_frame_number_deltas
                else None
            ),
            "frame_number_offset_min": (
                min(rgbd_frame_number_deltas)
                if rgbd_frame_number_deltas
                else None
            ),
            "frame_number_offset_max": (
                max(rgbd_frame_number_deltas)
                if rgbd_frame_number_deltas
                else None
            ),
            "frame_number_offset_span": (
                max(rgbd_frame_number_deltas) - min(rgbd_frame_number_deltas)
                if rgbd_frame_number_deltas
                else None
            ),
        },
        "depth": {
            "valid_frames": len(depths),
            "valid_rate_among_detections": (
                len(depths) / right_detections if right_detections else None
            ),
            "min_m": min(depths) if depths else None,
            "max_m": max(depths) if depths else None,
            "calibrated_range_sample_count": depth_range_flags,
            "in_calibrated_range_rate": (
                depth_range_true / depth_range_flags if depth_range_flags else None
            ),
            "control_valid_frames": len(control_depths),
            "control_valid_rate_among_detections": (
                len(control_depths) / right_detections
                if right_detections
                else None
            ),
            "control_min_m": min(control_depths) if control_depths else None,
            "control_max_m": max(control_depths) if control_depths else None,
            "source_sample_count": sum(palm_depth_source_counts.values()),
            "source_counts": {
                source: palm_depth_source_counts.get(source, 0)
                for source in PALM_DEPTH_SOURCES
            },
            "reason_sample_count": sum(palm_depth_reason_counts.values()),
            "reason_counts": dict(sorted(palm_depth_reason_counts.items())),
        },
        "longest_missed_detection": longest_miss,
        "max_tracking_age_seconds": max(tracking_ages) if tracking_ages else None,
        "qpos": qpos.summary(),
        "raw_qpos": raw_qpos.summary(),
        "hardware_targets": targets.summary(),
        "raw_hardware_targets": raw_targets.summary(),
        "masked_hardware_targets": masked_targets.summary(),
        "last_sent_targets": sent_targets.summary(),
        "filtering": {
            "available": bool(
                qpos_filtering.paired_sample_frames
                or target_filtering.paired_sample_frames
            ),
            "qpos": qpos_filtering.summary(
                raw_qpos.sample_frames, qpos.sample_frames
            ),
            "hardware_targets": target_filtering.summary(
                raw_targets.sample_frames, targets.sample_frames
            ),
        },
        "hardware": {
            "evidence_present": bool(
                hardware_states
                or submit_attempted_frames
                or sent_targets.sample_frames
                or hardware_feedback_summary["available"]
            ),
            "states_seen": sorted(hardware_states),
            "active_frames": active_frames,
            "ever_active": bool(active_frames or motion_observation_frames),
            "submission": {
                "attempted_frames": submit_attempted_frames,
                "accepted_frames": submit_accepted_frames,
                "rejected_frames": submit_rejected_frames,
                "acceptance_rate": (
                    submit_accepted_frames / submit_attempted_frames
                    if submit_attempted_frames
                    else None
                ),
            },
            # A JSONL row samples the worker's latest sent target; it is not a
            # one-to-one serial-write trace.  Keep these names explicit so the
            # report cannot overclaim the number of physical write operations.
            "motion": {
                "observed_frames": motion_observation_frames,
                "distinct_target_vectors": len(distinct_motion_targets),
                "observed_target_changes": motion_target_changes,
            },
            "feedback": hardware_feedback_summary,
        },
        "shutdown": _shutdown_not_checked(),
        "input_metadata": {
            "log_schema_versions": sorted(map(str, schema_versions)),
            "session_ids": sorted(session_ids),
        },
        "notes": (
            []
            if detection_metrics_available
            else [
                "Detection rate and missed intervals require per-frame rows with "
                "right_hand_detected; legacy logs contain detections only."
            ]
        ),
    }


def analyze_records(records: Iterable[Mapping[str, object]]) -> dict:
    """Analyze already-decoded rows; primarily useful for tests and notebooks."""

    return _analyze_numbered_records(enumerate(records, start=1))


def _analyze_shutdown_status(
    path: Path, log_session_ids: Sequence[str], hardware_evidence: bool
) -> dict:
    result = _shutdown_not_checked()
    result.update(
        {
            "checked": True,
            "file_present": path.is_file(),
            "path": str(path.resolve()),
            "verification_status": (
                "missing" if hardware_evidence else "not_applicable"
            ),
        }
    )
    if not path.is_file():
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LogFormatError(f"invalid shutdown status {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise LogFormatError(f"shutdown status {path} must contain a JSON object")

    def optional_nonnegative_integer(field: str) -> Optional[int]:
        value = payload.get(field)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise LogFormatError(
                f"shutdown status {path}: {field} must be a non-negative "
                "JSON integer or null"
            )
        return value

    def optional_nonnegative_number(field: str) -> Optional[float]:
        value = payload.get(field)
        if value is None:
            return None
        result = _finite_number(value, field, 0)
        if result < 0:
            raise LogFormatError(
                f"shutdown status {path}: {field} must be non-negative"
            )
        return result

    def optional_nonnegative_vector(field: str) -> Optional[list[float]]:
        value = payload.get(field)
        if value is None:
            return None
        if not isinstance(value, list) or len(value) != len(AXES):
            raise LogFormatError(
                f"shutdown status {path}: {field} must be null or a "
                "six-number array"
            )
        result = [
            _finite_number(item, f"{field}[{axis}]", 0)
            for axis, item in zip(AXES, value)
        ]
        if any(item < 0 for item in result):
            raise LogFormatError(
                f"shutdown status {path}: {field} values must be non-negative"
            )
        return result

    active_current_policy = payload.get("active_current_policy")
    if active_current_policy is not None and active_current_policy not in (
        "fault",
        "monitor_only",
    ):
        raise LogFormatError(
            f"shutdown status {path}: active_current_policy must be 'fault', "
            "'monitor_only', or null"
        )
    host_current_per_axis_threshold_ma = optional_nonnegative_integer(
        "host_current_per_axis_threshold_ma"
    )
    host_current_selected_total_threshold_ma = optional_nonnegative_integer(
        "host_current_selected_total_threshold_ma"
    )
    active_current_over_limit_sample_count = optional_nonnegative_integer(
        "active_current_over_limit_sample_count"
    )
    active_current_warning_event_count = optional_nonnegative_integer(
        "active_current_warning_event_count"
    )
    active_peak_abs_currents = optional_nonnegative_vector(
        "active_peak_abs_currents"
    )
    active_max_selected_total_current_ma = optional_nonnegative_number(
        "active_max_selected_total_current_ma"
    )
    verified_device_current_limits = optional_nonnegative_vector(
        "verified_device_current_limits"
    )

    stop_confirmed = payload.get("stop_confirmed")
    if not isinstance(stop_confirmed, bool):
        raise LogFormatError(
            f"shutdown status {path}: stop_confirmed must be a JSON boolean"
        )

    # This field was added when shutdown verification grew from checking only
    # the ANGLE_SET=-1 register to observing that all six actuators had also
    # become mechanically and electrically idle.  Missing means a valid legacy
    # file with weaker evidence, not a malformed file.
    physical_stop_verified = payload.get("physical_stop_verified")
    if (
        physical_stop_verified is not None
        and not isinstance(physical_stop_verified, bool)
    ):
        raise LogFormatError(
            f"shutdown status {path}: physical_stop_verified must be a JSON "
            "boolean or null"
        )

    feedback_value = payload.get("shutdown_feedback")
    shutdown_feedback = None
    shutdown_feedback_idle = None
    if feedback_value is not None:
        if not isinstance(feedback_value, Mapping):
            raise LogFormatError(
                f"shutdown status {path}: shutdown_feedback must be an object "
                "or null"
            )
        shutdown_feedback = {}
        for field in (
            "angles",
            "positions",
            "currents",
            "errors",
            "statuses",
            "temperatures",
        ):
            vector = feedback_value.get(field)
            if not isinstance(vector, list) or len(vector) != len(AXES):
                raise LogFormatError(
                    f"shutdown status {path}: shutdown_feedback.{field} must "
                    "be a six-number array"
                )
            shutdown_feedback[field] = [
                _finite_number(
                    value, f"shutdown_feedback.{field}[{axis}]", 0
                )
                for axis, value in zip(AXES, vector)
            ]
        currents = shutdown_feedback["currents"]
        shutdown_feedback_idle = bool(
            all(value == 0 for value in shutdown_feedback["errors"])
            and all(value in (2, 0xFF) for value in shutdown_feedback["statuses"])
            and all(abs(value) <= 100 for value in currents)
            and sum(abs(value) for value in currents) <= 200
            and max(shutdown_feedback["temperatures"]) < 60
        )

    final_value = payload.get("final_angle_targets")
    if final_value is None:
        final_targets = None
        final_all_minus_one = False
    else:
        if not isinstance(final_value, list) or len(final_value) != len(AXES):
            raise LogFormatError(
                f"shutdown status {path}: final_angle_targets must be null or "
                "a six-number array"
            )
        final_targets = [
            _finite_number(value, f"final_angle_targets[{axis}]", 0)
            for axis, value in zip(AXES, final_value)
        ]
        final_all_minus_one = all(value == -1 for value in final_targets)

    shutdown_session = payload.get("session_id")
    if shutdown_session is not None and not isinstance(shutdown_session, str):
        raise LogFormatError(
            f"shutdown status {path}: session_id must be a string or null"
        )
    if log_session_ids:
        session_matches = (
            shutdown_session is not None
            and set(log_session_ids) == {shutdown_session}
        )
    else:
        session_matches = None
    safe_stop = bool(
        stop_confirmed
        and physical_stop_verified is True
        and shutdown_feedback_idle is True
        and final_all_minus_one
        and session_matches is not False
    )
    result.update(
        {
            "session_id": shutdown_session,
            "session_matches_log": session_matches,
            "stop_confirmed": stop_confirmed,
            "physical_stop_verified": physical_stop_verified,
            "active_current_policy": active_current_policy,
            "host_current_per_axis_threshold_ma": (
                host_current_per_axis_threshold_ma
            ),
            "host_current_selected_total_threshold_ma": (
                host_current_selected_total_threshold_ma
            ),
            "active_current_over_limit_sample_count": (
                active_current_over_limit_sample_count
            ),
            "active_current_warning_event_count": (
                active_current_warning_event_count
            ),
            "active_peak_abs_currents": active_peak_abs_currents,
            "active_max_selected_total_current_ma": (
                active_max_selected_total_current_ma
            ),
            "verified_device_current_limits": verified_device_current_limits,
            "shutdown_feedback": shutdown_feedback,
            "shutdown_feedback_idle": shutdown_feedback_idle,
            "final_angle_targets": final_targets,
            "final_all_minus_one": final_all_minus_one,
            "safe_stop_verified": safe_stop,
            "verification_status": "safe" if safe_stop else "unsafe",
        }
    )
    return result


def analyze_jsonl(path: Path) -> dict:
    def rows():
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LogFormatError(
                        f"line {line_number}: invalid JSON: {exc.msg}"
                    ) from exc
                yield line_number, value

    summary = _analyze_numbered_records(rows())
    summary["input_path"] = str(path.expanduser().resolve())
    summary["shutdown"] = _analyze_shutdown_status(
        path.parent / "shutdown_status.json",
        summary["input_metadata"]["session_ids"],
        bool(summary["hardware"]["evidence_present"]),
    )
    shutdown = summary["shutdown"]
    if shutdown["verification_status"] == "missing":
        summary["notes"].append(
            "Hardware activity is present in the frame log, but "
            "shutdown_status.json is missing."
        )
    elif shutdown["verification_status"] == "unsafe":
        summary["notes"].append(
            "shutdown_status.json does not prove stop_confirmed=true, a "
            "physically and electrically idle six-axis hand, and all six "
            "final ANGLE_SET targets equal to -1 for the same session."
        )
    if (
        shutdown["active_current_policy"] == "monitor_only"
        and (
            (shutdown["active_current_over_limit_sample_count"] or 0) > 0
            or (shutdown["active_current_warning_event_count"] or 0) > 0
        )
    ):
        summary["notes"].append(
            "ACTIVE current policy was monitor_only and host observation "
            "thresholds were crossed; this is a current warning, not a "
            "host-current fault or a reason to override independently verified "
            "safe shutdown evidence."
        )
    return summary


def _format_number(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_text(summary: Mapping[str, object]) -> str:
    right = summary["right_hand"]
    wilor = summary["wilor"]
    latency = summary["inference_latency"]
    pipeline_age = summary["pipeline_age"]
    rgbd_sync = summary["rgbd_sync"]
    depth = summary["depth"]
    miss = summary["longest_missed_detection"]
    total = summary["total_frames"]
    rate = right["detection_rate"]
    lines = [
        f"日志: {summary.get('input_path', '<memory>')}",
        f"格式: {summary['log_schema']}  记录行: {summary['log_rows']}",
        f"总处理帧: {total if total is not None else '不可从旧日志推断'}",
        (
            f"右手检测: {right['detected_frames']} / "
            f"{total if total is not None else '?'} "
            f"({_format_number(rate * 100 if rate is not None else None, 2)}%)"
        ),
        (
            "WiLoR 候选: "
            f"candidate_frames={wilor['frames_with_candidates']}, "
            f"results={wilor['result_counts']}, "
            f"labels={wilor['detector_label_counts']}"
        ),
        (
            "推理延迟: "
            f"p50={_format_number(latency['p50_ms'], 2)} ms, "
            f"p95={_format_number(latency['p95_ms'], 2)} ms"
        ),
        (
            "  有效右手帧: "
            f"n={latency['detected_hand']['sample_count']}, "
            f"p50={_format_number(latency['detected_hand']['p50_ms'], 2)} ms, "
            f"p95={_format_number(latency['detected_hand']['p95_ms'], 2)} ms"
        ),
        (
            "  无有效右手帧: "
            f"n={latency['no_detected_hand']['sample_count']}, "
            f"p50={_format_number(latency['no_detected_hand']['p50_ms'], 2)} ms, "
            f"p95={_format_number(latency['no_detected_hand']['p95_ms'], 2)} ms"
        ),
        (
            "收帧至落日志: "
            f"p50={_format_number(pipeline_age['p50_ms'], 2)} ms, "
            f"p95={_format_number(pipeline_age['p95_ms'], 2)} ms"
        ),
        (
            "RGB-D 同步: "
            "sensor |Δt| p95="
            f"{_format_number(rgbd_sync['sensor_timestamp_abs_delta_p95_ms'], 2)} ms, "
            "frame |Δn| max="
            f"{_format_number(rgbd_sync['frame_number_abs_delta_max'], 0)}"
        ),
        (
            f"有效掌深: {depth['valid_frames']} / {right['detected_frames']} "
            f"({_format_number(depth['valid_rate_among_detections'] * 100 if depth['valid_rate_among_detections'] is not None else None, 2)}%), "
            f"范围={_format_number(depth['min_m'])}..{_format_number(depth['max_m'])} m"
        ),
        (
            f"有效控制掌深: {depth['control_valid_frames']} / "
            f"{right['detected_frames']} "
            f"({_format_number(depth['control_valid_rate_among_detections'] * 100 if depth['control_valid_rate_among_detections'] is not None else None, 2)}%), "
            f"范围={_format_number(depth['control_min_m'])}.."
            f"{_format_number(depth['control_max_m'])} m"
        ),
        (
            "掌深诊断计数: "
            f"source_samples={depth['source_sample_count']}, "
            f"sources={depth['source_counts']}, "
            f"reason_samples={depth['reason_sample_count']}, "
            f"reasons={depth['reason_counts']}"
        ),
    ]
    hardware = summary["hardware"]
    submission = hardware["submission"]
    motion = hardware["motion"]
    feedback = hardware["feedback"]
    lines.extend(
        [
            (
                "硬件状态: "
                f"ever_active={hardware['ever_active']}, "
                f"active_frames={hardware['active_frames']}, "
                f"states={hardware['states_seen']}"
            ),
            (
                "目标提交: "
                f"accepted={submission['accepted_frames']}, "
                f"rejected={submission['rejected_frames']}, "
                f"attempted={submission['attempted_frames']}"
            ),
            (
                "实际发送目标观测: "
                f"frames={motion['observed_frames']}, "
                f"distinct={motion['distinct_target_vectors']}, "
                f"changes={motion['observed_target_changes']}"
            ),
        ]
    )
    if feedback["available"]:
        samples = feedback["sample_frames"]
        lines.append(
            "硬件反馈样本: "
            f"actual_angles={samples['actual_angles']}, "
            f"currents={samples['currents']}, "
            f"temperatures={samples['temperatures']}, "
            f"errors={samples['errors']}, "
            "任一轴非零故障帧="
            f"{feedback['frames_with_any_nonzero_error']}"
        )
        lines.append("硬件反馈（实际角 min..max / |电流|max mA / 最高温度 °C / 非零故障帧）:")
        for axis in AXES:
            stats = feedback["axes"][axis]
            error_frames = stats["nonzero_error_frames"]
            lines.append(
                f"  {axis:12s} "
                f"{_format_number(stats['actual_angle_min'])}.."
                f"{_format_number(stats['actual_angle_max'])} / "
                f"{_format_number(stats['max_abs_current_ma'])} / "
                f"{_format_number(stats['max_temperature_c'])} / "
                f"{error_frames if error_frames is not None else 'n/a'}"
            )
    else:
        lines.append("硬件反馈: 无 per-frame actual/current/temperature/error 样本")
    if miss is None:
        lines.append("最长漏检: 不可从旧 detection-only 日志推断")
    else:
        lines.append(
            f"最长漏检: {miss['frames']} 帧, 采样跨度 "
            f"{_format_number(miss['sample_span_seconds'])} s"
        )
    for field, label in (
        ("qpos", "qpos"),
        ("hardware_targets", "未屏蔽硬件目标"),
        ("masked_hardware_targets", "按开放轴屏蔽后的目标"),
        ("last_sent_targets", "串口 worker 最近实际发送目标"),
    ):
        vector = summary[field]
        lines.append(f"{label}（min / max / 相邻处理帧最大跳变）:")
        for axis in AXES:
            stats = vector["axes"][axis]
            lines.append(
                f"  {axis:12s} {_format_number(stats['min'])} / "
                f"{_format_number(stats['max'])} / "
                f"{_format_number(stats['max_adjacent_frame_jump'])}"
            )
    filtering = summary.get("filtering")
    if isinstance(filtering, Mapping) and filtering.get("available"):
        for field, label in (
            ("qpos", "qpos"),
            ("hardware_targets", "未屏蔽硬件目标"),
        ):
            comparison = filtering.get(field)
            if not isinstance(comparison, Mapping) or not comparison.get("available"):
                continue
            lines.append(
                f"滤波对照 {label}（paired={comparison['paired_sample_frames']}, "
                f"adjacent_pairs={comparison['adjacent_pair_count']}；"
                "raw 范围 -> filtered 范围 / 相邻跳变 p95|max raw -> filtered / "
                "|raw-filter|max）:"
            )
            comparison_axes = comparison["axes"]
            for axis in AXES:
                stats = comparison_axes[axis]
                lines.append(
                    f"  {axis:12s} "
                    f"{_format_number(stats['raw_min'])}.."
                    f"{_format_number(stats['raw_max'])} -> "
                    f"{_format_number(stats['filtered_min'])}.."
                    f"{_format_number(stats['filtered_max'])} / "
                    f"{_format_number(stats['raw_adjacent_jump_p95'])}|"
                    f"{_format_number(stats['raw_adjacent_jump_max'])} -> "
                    f"{_format_number(stats['filtered_adjacent_jump_p95'])}|"
                    f"{_format_number(stats['filtered_adjacent_jump_max'])} / "
                    f"{_format_number(stats['max_abs_raw_to_filtered_delta'])}"
                )
    shutdown = summary["shutdown"]
    if shutdown["checked"]:
        lines.append(
            "停机证据: "
            f"status={shutdown['verification_status']}, "
            f"stop_confirmed={shutdown['stop_confirmed']}, "
            f"physical_stop_verified={shutdown['physical_stop_verified']}, "
            f"shutdown_feedback_idle={shutdown['shutdown_feedback_idle']}, "
            f"final_all_minus_one={shutdown['final_all_minus_one']}, "
            f"session_matches={shutdown['session_matches_log']}"
        )
        if shutdown["active_current_policy"] is not None:
            lines.append(
                "ACTIVE 电流策略: "
                f"policy={shutdown['active_current_policy']}, "
                "host_thresholds(per_axis/selected_total)="
                f"{shutdown['host_current_per_axis_threshold_ma']}/"
                f"{shutdown['host_current_selected_total_threshold_ma']} mA, "
                "over_limit_samples/events="
                f"{shutdown['active_current_over_limit_sample_count']}/"
                f"{shutdown['active_current_warning_event_count']}, "
                f"max_selected_total="
                f"{shutdown['active_max_selected_total_current_ma']} mA, "
                f"active_peaks={shutdown['active_peak_abs_currents']}, "
                "device_CURRENT_LIMIT="
                f"{shutdown['verified_device_current_limits']}"
            )
    else:
        lines.append("停机证据: 内存记录未检查同目录 shutdown_status.json")
    lines.extend(f"注意: {note}" for note in summary.get("notes", []))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a mano_retarget.jsonl file without opening camera or serial devices."
    )
    parser.add_argument("jsonl", type=Path, help="path to mano_retarget.jsonl")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = analyze_jsonl(args.jsonl)
    except (OSError, LogFormatError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(render_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
