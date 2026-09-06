from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

from .model import HARDWARE_JOINTS


def _finite_float(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a JSON boolean")
    return value


def _strict_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a JSON integer")
    return value


@dataclass(frozen=True)
class AxisCalibration:
    q_open: float
    q_closed: float
    command_open: int
    command_closed: int
    enabled: bool
    max_rate_units_per_second: float
    open_enter_q: float
    open_exit_q: float
    closed_enter_q: float
    closed_exit_q: float
    hardware_speed: Optional[int] = None
    feedback_to_command_offset_units: int = 0
    feedback_to_command_valid_min: Optional[int] = None
    feedback_to_command_valid_max: Optional[int] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "AxisCalibration":
        q_open = _finite_float(data["q_open"], "q_open")
        q_closed = _finite_float(data["q_closed"], "q_closed")
        hysteresis = data.get("hysteresis", {})
        if not isinstance(hysteresis, dict):
            raise ValueError("axis hysteresis must be a JSON object")
        feedback_valid_min_raw = data.get("feedback_to_command_valid_min")
        feedback_valid_max_raw = data.get("feedback_to_command_valid_max")
        hardware_speed_raw = data.get("hardware_speed")
        if (feedback_valid_min_raw is None) != (feedback_valid_max_raw is None):
            raise ValueError(
                "feedback_to_command_valid_min/max must be provided together"
            )
        feedback_valid_min = (
            None
            if feedback_valid_min_raw is None
            else _strict_int(
                feedback_valid_min_raw, "feedback_to_command_valid_min"
            )
        )
        feedback_valid_max = (
            None
            if feedback_valid_max_raw is None
            else _strict_int(
                feedback_valid_max_raw, "feedback_to_command_valid_max"
            )
        )
        axis = cls(
            q_open=q_open,
            q_closed=q_closed,
            command_open=int(data["command_open"]),
            command_closed=int(data["command_closed"]),
            enabled=_strict_bool(data.get("enabled", True), "enabled"),
            max_rate_units_per_second=_finite_float(
                data.get("max_rate_units_per_second", 180.0),
                "max_rate_units_per_second",
            ),
            open_enter_q=_finite_float(
                hysteresis.get("open_enter_q", q_open), "open_enter_q"
            ),
            open_exit_q=_finite_float(
                hysteresis.get("open_exit_q", q_open), "open_exit_q"
            ),
            closed_enter_q=_finite_float(
                hysteresis.get("closed_enter_q", q_closed), "closed_enter_q"
            ),
            closed_exit_q=_finite_float(
                hysteresis.get("closed_exit_q", q_closed), "closed_exit_q"
            ),
            hardware_speed=(
                None
                if hardware_speed_raw is None
                else _strict_int(hardware_speed_raw, "hardware_speed")
            ),
            feedback_to_command_offset_units=_strict_int(
                data.get("feedback_to_command_offset_units", 0),
                "feedback_to_command_offset_units",
            ),
            feedback_to_command_valid_min=feedback_valid_min,
            feedback_to_command_valid_max=feedback_valid_max,
        )
        if not axis.q_closed > axis.q_open:
            raise ValueError("q_closed must be greater than q_open")
        if not 0.0 <= axis.q_open < axis.q_closed <= 4.0:
            raise ValueError("q range must lie within 0..4 radians")
        for value in (axis.command_open, axis.command_closed):
            if not 0 <= value <= 1000:
                raise ValueError("RH56 commands must be in 0..1000")
        if axis.command_open <= axis.command_closed:
            raise ValueError(
                "RH56 calibration requires command_open > command_closed"
            )
        if not 1.0 <= axis.max_rate_units_per_second <= 1000.0:
            raise ValueError("max_rate_units_per_second must be in 1..1000")
        if axis.hardware_speed is not None and not 1 <= axis.hardware_speed <= 1000:
            raise ValueError("hardware_speed must be in 1..1000 when provided")
        if not -100 <= axis.feedback_to_command_offset_units <= 100:
            raise ValueError(
                "feedback_to_command_offset_units must be in -100..100"
            )
        if axis.feedback_to_command_offset_units != 0 and (
            axis.feedback_to_command_valid_min is None
            or axis.feedback_to_command_valid_max is None
        ):
            raise ValueError(
                "nonzero feedback_to_command_offset_units requires a validated "
                "feedback range"
            )
        if (
            axis.feedback_to_command_valid_min is None
        ) != (
            axis.feedback_to_command_valid_max is None
        ):
            raise ValueError(
                "feedback_to_command_valid_min/max must be provided together"
            )
        if (
            axis.feedback_to_command_valid_min is not None
            and axis.feedback_to_command_valid_max is not None
            and not (
                0
                <= axis.feedback_to_command_valid_min
                <= axis.feedback_to_command_valid_max
                <= 1000
            )
        ):
            raise ValueError(
                "feedback_to_command_valid_min/max must define a range in 0..1000"
            )
        if not (
            0.0
            <= axis.open_enter_q
            <= axis.q_open
            <= axis.open_exit_q
            < axis.closed_exit_q
            <= axis.q_closed
            <= axis.closed_enter_q
            <= 4.0
        ):
            raise ValueError(
                "axis hysteresis must satisfy open_enter_q <= q_open <= "
                "open_exit_q < closed_exit_q <= q_closed <= closed_enter_q"
            )
        return axis

    def map_qpos(self, qpos: float) -> int:
        if not np.isfinite(qpos):
            raise ValueError("qpos must be finite")
        fraction = np.clip(
            (float(qpos) - self.q_open) / (self.q_closed - self.q_open),
            0.0,
            1.0,
        )
        command = self.command_open + fraction * (
            self.command_closed - self.command_open
        )
        return int(round(float(command)))


@dataclass(frozen=True)
class PipelineCalibration:
    axes: Dict[str, AxisCalibration]
    tracking_timeout_seconds: float
    valid_frames_to_arm: int
    arming_max_frame_gap_seconds: float
    arming_max_target_delta_units: int
    detector_confidence: float
    speed: int
    force_limit: int
    control_hz: float
    feedback_hz: float
    require_depth_for_hardware: bool
    min_palm_depth_m: float
    max_palm_depth_m: float
    preflight_stability_seconds: float
    preflight_max_angle_delta: int
    preflight_max_idle_current_ma: int
    active_current_policy: str
    stream_max_current_ma: int
    stream_max_total_current_ma: int
    temporal_median_window: int
    temporal_ema_alpha: float
    temporal_filter_axes: Tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> "PipelineCalibration":
        def reject_constant(value: str):
            raise ValueError(f"non-standard JSON number is forbidden: {value}")

        data = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
        if data.get("schema_version") != 1:
            raise ValueError("unsupported or missing calibration schema_version")
        axes_data = data.get("axes")
        if not isinstance(axes_data, dict):
            raise ValueError("calibration must contain an axes object")
        if set(axes_data) != set(HARDWARE_JOINTS):
            raise ValueError(
                "calibration axes must exactly match: " + ", ".join(HARDWARE_JOINTS)
            )
        axes = {
            name: AxisCalibration.from_mapping(axes_data[name])
            for name in HARDWARE_JOINTS
        }
        safety = data.get("safety", {})
        temporal_filter = data.get("temporal_filter", {})
        if not isinstance(temporal_filter, dict):
            raise ValueError("temporal_filter must be a JSON object")
        raw_filter_axes = temporal_filter.get("axes", [])
        if (
            not isinstance(raw_filter_axes, list)
            or any(not isinstance(name, str) for name in raw_filter_axes)
            or len(set(raw_filter_axes)) != len(raw_filter_axes)
            or any(name not in HARDWARE_JOINTS for name in raw_filter_axes)
        ):
            raise ValueError(
                "temporal_filter.axes must be a unique list of RH56 axis names"
            )
        stream_max_current_ma = int(safety.get("stream_max_current_ma", 1000))
        active_current_policy = safety.get("active_current_policy", "fault")
        if not isinstance(active_current_policy, str):
            raise ValueError("active_current_policy must be a string")
        stream_max_total_current_ma = _strict_int(
            safety.get(
                "stream_max_total_current_ma",
                len(HARDWARE_JOINTS) * stream_max_current_ma,
            ),
            "stream_max_total_current_ma",
        )
        result = cls(
            axes=axes,
            tracking_timeout_seconds=_finite_float(
                safety.get("tracking_timeout_seconds", 0.75),
                "tracking_timeout_seconds",
            ),
            valid_frames_to_arm=int(safety.get("valid_frames_to_arm", 3)),
            arming_max_frame_gap_seconds=_finite_float(
                safety.get("arming_max_frame_gap_seconds", 0.25),
                "arming_max_frame_gap_seconds",
            ),
            arming_max_target_delta_units=_strict_int(
                safety.get("arming_max_target_delta_units", 50),
                "arming_max_target_delta_units",
            ),
            detector_confidence=_finite_float(
                data.get("detector_confidence", 0.5), "detector_confidence"
            ),
            speed=int(safety.get("speed", 120)),
            force_limit=int(safety.get("force_limit", 100)),
            control_hz=_finite_float(safety.get("control_hz", 20.0), "control_hz"),
            feedback_hz=_finite_float(
                safety.get("feedback_hz", 5.0), "feedback_hz"
            ),
            require_depth_for_hardware=_strict_bool(
                safety.get("require_depth_for_hardware", True),
                "require_depth_for_hardware",
            ),
            min_palm_depth_m=_finite_float(
                safety.get("min_palm_depth_m", 0.15), "min_palm_depth_m"
            ),
            max_palm_depth_m=_finite_float(
                safety.get("max_palm_depth_m", 1.50), "max_palm_depth_m"
            ),
            preflight_stability_seconds=_finite_float(
                safety.get("preflight_stability_seconds", 0.15),
                "preflight_stability_seconds",
            ),
            preflight_max_angle_delta=int(
                safety.get("preflight_max_angle_delta", 5)
            ),
            preflight_max_idle_current_ma=int(
                safety.get("preflight_max_idle_current_ma", 300)
            ),
            active_current_policy=active_current_policy,
            stream_max_current_ma=stream_max_current_ma,
            stream_max_total_current_ma=stream_max_total_current_ma,
            temporal_median_window=_strict_int(
                temporal_filter.get("median_window", 1), "median_window"
            ),
            temporal_ema_alpha=_finite_float(
                temporal_filter.get("ema_alpha", 1.0), "ema_alpha"
            ),
            temporal_filter_axes=tuple(raw_filter_axes),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not 0.10 <= self.tracking_timeout_seconds <= 5.0:
            raise ValueError("tracking_timeout_seconds must be in 0.10..5.0")
        if not 1 <= self.valid_frames_to_arm <= 60:
            raise ValueError("valid_frames_to_arm must be in 1..60")
        if not 0.03 <= self.arming_max_frame_gap_seconds <= (
            self.tracking_timeout_seconds
        ):
            raise ValueError(
                "arming_max_frame_gap_seconds must be in 0.03..tracking_timeout_seconds"
            )
        if not 0 <= self.arming_max_target_delta_units <= 1000:
            raise ValueError("arming_max_target_delta_units must be in 0..1000")
        if not 0 < self.detector_confidence <= 1:
            raise ValueError("detector_confidence must be in (0, 1]")
        if not 1 <= self.speed <= 300:
            raise ValueError("streaming speed must be in 1..300")
        if not 1 <= self.force_limit <= 300:
            raise ValueError("streaming force_limit must be in 1..300")
        if not 5.0 <= self.control_hz <= 100.0:
            raise ValueError("control_hz must be in 5..100")
        if not 0.5 <= self.feedback_hz <= self.control_hz:
            raise ValueError("feedback_hz must be in 0.5..control_hz")
        if self.tracking_timeout_seconds * self.control_hz < 3.0:
            raise ValueError("watchdog timeout must span at least three control ticks")
        if not 0.10 <= self.min_palm_depth_m < self.max_palm_depth_m <= 3.0:
            raise ValueError("palm depth range must lie within 0.10..3.0 m")
        if not 0.05 <= self.preflight_stability_seconds <= 2.0:
            raise ValueError("preflight_stability_seconds must be in 0.05..2.0")
        if not 0 <= self.preflight_max_angle_delta <= 20:
            raise ValueError("preflight_max_angle_delta must be in 0..20")
        if not 0 <= self.preflight_max_idle_current_ma <= 2000:
            raise ValueError("preflight_max_idle_current_ma must be in 0..2000")
        if self.active_current_policy not in ("fault", "monitor_only"):
            raise ValueError(
                "active_current_policy must be 'fault' or 'monitor_only'"
            )
        if not 100 <= self.stream_max_current_ma <= 3000:
            raise ValueError("stream_max_current_ma must be in 100..3000")
        if not (
            self.stream_max_current_ma
            <= self.stream_max_total_current_ma
            <= len(HARDWARE_JOINTS) * self.stream_max_current_ma
        ):
            raise ValueError(
                "stream_max_total_current_ma must be between "
                "stream_max_current_ma and six times stream_max_current_ma"
            )
        if self.preflight_max_idle_current_ma > self.stream_max_current_ma:
            raise ValueError(
                "preflight_max_idle_current_ma cannot exceed stream_max_current_ma"
            )
        if not 1 <= self.temporal_median_window <= 15:
            raise ValueError("median_window must be in 1..15")
        if self.temporal_median_window % 2 == 0:
            raise ValueError("median_window must be odd")
        if not 0.0 < self.temporal_ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")

    def map_qpos(self, qpos: Iterable[float]) -> Tuple[int, ...]:
        values = np.asarray(tuple(qpos), dtype=np.float64)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            raise ValueError("expected six finite qpos values")
        targets = []
        for name, value in zip(HARDWARE_JOINTS, values):
            axis = self.axes[name]
            targets.append(axis.map_qpos(float(value)) if axis.enabled else -1)
        return tuple(targets)
