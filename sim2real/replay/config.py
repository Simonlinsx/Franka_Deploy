"""Strict decoding of optional replay planner configuration."""

from __future__ import annotations

import json
from typing import Mapping, Optional

import numpy as np

from .models import (
    CANONICAL_ACTION_ORDER,
    TabletopInterceptReplayConfig,
    TabletopOnlinePlannerConfig,
)


def _optional_rate(value: object, name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be 20 or 60 Hz")
    try:
        result = float(np.asarray(value).reshape(()))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar")
    return result


def _validate_action_order(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, (list, tuple)) or tuple(str(item) for item in value) != (
        CANONICAL_ACTION_ORDER
    ):
        raise ValueError(
            "replay action_order must exactly match "
            + json.dumps(CANONICAL_ACTION_ORDER)
        )


def _tabletop_intercept_config(
    value: object,
    *,
    action_count: int,
) -> Optional[TabletopInterceptReplayConfig]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("tabletop_intercept must be an object")
    allowed = {
        "version",
        "preposition_end_index",
        "control_dt_s",
        "fit_sample_count",
        "motion_axis_palm",
        "intercept_center_palm_m",
        "speed_min_m_s",
        "speed_max_m_s",
        "heading_cos_min",
        "max_fit_residual_m",
        "max_lateral_miss_m",
        "trigger_ttc_max_s",
        "max_wait_ticks",
        "collision_replan",
        "collision_min_velocity_change_m_s",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            f"unknown tabletop_intercept fields: {sorted(unknown)}"
        )
    if value.get("version") != "tabletop_intercept_replay_v1":
        raise ValueError("unsupported tabletop_intercept version")

    def integer(name: str, minimum: int, maximum: int) -> int:
        raw = value.get(name)
        if isinstance(raw, (bool, np.bool_)) or not isinstance(
            raw, (int, np.integer)
        ):
            raise ValueError(f"tabletop_intercept {name} must be an integer")
        result = int(raw)
        if not minimum <= result <= maximum:
            raise ValueError(
                f"tabletop_intercept {name} must be in {minimum}..{maximum}"
            )
        return result

    def number(name: str, minimum: float, maximum: float) -> float:
        raw = value.get(name)
        if isinstance(raw, (bool, np.bool_)):
            raise ValueError(f"tabletop_intercept {name} must be finite")
        try:
            result = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"tabletop_intercept {name} must be finite"
            ) from exc
        if not np.isfinite(result) or not minimum <= result <= maximum:
            raise ValueError(
                f"tabletop_intercept {name} must be in {minimum:g}..{maximum:g}"
            )
        return result

    def vector(name: str) -> tuple[float, float, float]:
        try:
            result = np.asarray(value.get(name), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"tabletop_intercept {name} must contain three finite values"
            ) from exc
        if result.shape != (3,) or not np.all(np.isfinite(result)):
            raise ValueError(
                f"tabletop_intercept {name} must contain three finite values"
            )
        return tuple(float(item) for item in result)

    preposition = integer(
        "preposition_end_index", 1, max(1, int(action_count) - 2)
    )
    fit_samples = integer("fit_sample_count", 4, 12)
    max_wait = integer("max_wait_ticks", fit_samples, 180)
    dt = number("control_dt_s", 0.049, 0.051)
    axis_array = np.asarray(vector("motion_axis_palm"), dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis_array))
    if not np.isclose(axis_norm, 1.0, atol=1.0e-5, rtol=0.0):
        raise ValueError("tabletop_intercept motion_axis_palm must be unit length")
    center = vector("intercept_center_palm_m")
    if float(np.linalg.norm(np.asarray(center))) > 0.40:
        raise ValueError("tabletop_intercept center is outside the palm envelope")
    speed_min = number("speed_min_m_s", 0.01, 0.50)
    speed_max = number("speed_max_m_s", speed_min, 0.60)
    collision = value.get("collision_replan", False)
    if not isinstance(collision, (bool, np.bool_)):
        raise ValueError("tabletop_intercept collision_replan must be boolean")
    return TabletopInterceptReplayConfig(
        preposition_end_index=preposition,
        control_dt_s=dt,
        fit_sample_count=fit_samples,
        motion_axis_palm=tuple(float(item) for item in axis_array),
        intercept_center_palm_m=center,
        speed_min_m_s=speed_min,
        speed_max_m_s=speed_max,
        heading_cos_min=number("heading_cos_min", 0.5, 1.0),
        max_fit_residual_m=number("max_fit_residual_m", 0.001, 0.05),
        max_lateral_miss_m=number("max_lateral_miss_m", 0.005, 0.10),
        trigger_ttc_max_s=number("trigger_ttc_max_s", 0.05, 0.60),
        max_wait_ticks=max_wait,
        collision_replan=bool(collision),
        collision_min_velocity_change_m_s=number(
            "collision_min_velocity_change_m_s", 0.03, 0.40
        ),
    )


def _tabletop_online_planner_config(
    value: object,
    *,
    action_count: int,
) -> Optional[TabletopOnlinePlannerConfig]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("tabletop_online_planner must be an object")
    allowed = {
        "version",
        "control_dt_s",
        "template_start_index",
        "template_catch_index",
        "closure_end_index",
        "lift_end_index",
        "fit_sample_count",
        "motion_axis_base",
        "reference_intercept_center_base_m",
        "speed_min_m_s",
        "speed_max_m_s",
        "heading_cos_min",
        "max_fit_residual_m",
        "max_linear_prediction_s",
        "max_position_correction_xy_m",
        "max_position_correction_z_m",
        "correction_filter_alpha",
        "dls_damping",
        "max_joint_correction_rad",
        "max_joint_correction_step_rad",
        "max_target_step_rad",
        "max_orientation_correction_rad",
        "joint_limit_margin_rad",
        "tabletop_plane_base",
        "minimum_fingertip_clearance_m",
        "grasp_vertical_offset_m",
        "grasp_lift_m",
        "contact_prediction_latency_s",
        "collision_replan",
        "adaptive_speed_max_m_s",
        "approach_target_step_rad",
        "preshape_phase_step",
        "closure_trigger_ttc_s",
        "closure_cross_track_m",
        "closure_z_tolerance_m",
        "closure_past_tolerance_m",
        "capture_arm_error_rad",
        "joint_correction_reserve_rad",
        "capture_object_offset_palm_m",
        "arm_tracking_speed_rad_s",
        "minimum_intercept_horizon_s",
        "open_approach_lift_m",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            f"unknown tabletop_online_planner fields: {sorted(unknown)}"
        )
    version = value.get("version")
    if version not in {
        "tabletop_online_intercept_planner_v2",
        "tabletop_online_intercept_planner_v3",
        "tabletop_online_intercept_planner_v4",
        "tabletop_online_intercept_planner_v5",
    }:
        raise ValueError("unsupported tabletop_online_planner version")

    def integer(name: str, minimum: int, maximum: int) -> int:
        raw = value.get(name)
        if isinstance(raw, (bool, np.bool_)) or not isinstance(
            raw, (int, np.integer)
        ):
            raise ValueError(
                f"tabletop_online_planner {name} must be an integer"
            )
        result = int(raw)
        if not minimum <= result <= maximum:
            raise ValueError(
                f"tabletop_online_planner {name} must be in "
                f"{minimum}..{maximum}"
            )
        return result

    def number(name: str, minimum: float, maximum: float) -> float:
        raw = value.get(name)
        if isinstance(raw, (bool, np.bool_)):
            raise ValueError(
                f"tabletop_online_planner {name} must be finite"
            )
        try:
            result = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"tabletop_online_planner {name} must be finite"
            ) from exc
        if not np.isfinite(result) or not minimum <= result <= maximum:
            raise ValueError(
                f"tabletop_online_planner {name} must be in "
                f"{minimum:g}..{maximum:g}"
            )
        return result

    def vector(name: str, size: int) -> tuple[float, ...]:
        try:
            result = np.asarray(value.get(name), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"tabletop_online_planner {name} must contain {size} values"
            ) from exc
        if result.shape != (size,) or not np.all(np.isfinite(result)):
            raise ValueError(
                f"tabletop_online_planner {name} must contain {size} values"
            )
        return tuple(float(item) for item in result)

    start = integer("template_start_index", 1, max(1, action_count - 2))
    catch = integer(
        "template_catch_index", start + 1, max(start + 1, action_count - 1)
    )
    closure_end = integer(
        "closure_end_index", catch + 1, max(catch + 1, action_count - 1)
    )
    lift_end = integer(
        "lift_end_index",
        closure_end + 1,
        max(closure_end + 1, action_count - 1),
    )
    axis = np.asarray(vector("motion_axis_base", 3), dtype=np.float64)
    if not np.isclose(np.linalg.norm(axis), 1.0, atol=1.0e-5, rtol=0.0):
        raise ValueError(
            "tabletop_online_planner motion_axis_base must be unit length"
        )
    plane = np.asarray(vector("tabletop_plane_base", 4), dtype=np.float64)
    if not np.isclose(
        np.linalg.norm(plane[:3]), 1.0, atol=1.0e-3, rtol=0.0
    ):
        raise ValueError(
            "tabletop_online_planner tabletop plane normal must be unit length"
        )
    speed_min = number("speed_min_m_s", 0.01, 0.50)
    speed_max = number("speed_max_m_s", speed_min, 0.60)
    collision = value.get("collision_replan", False)
    if not isinstance(collision, (bool, np.bool_)):
        raise ValueError(
            "tabletop_online_planner collision_replan must be boolean"
        )
    adaptive = version in {
        "tabletop_online_intercept_planner_v3",
        "tabletop_online_intercept_planner_v4",
        "tabletop_online_intercept_planner_v5",
    }
    mpc = version == "tabletop_online_intercept_planner_v5"
    adaptive_speed_max = (
        number("adaptive_speed_max_m_s", speed_max, 0.60)
        if adaptive
        else speed_max
    )
    approach_target_step = (
        number("approach_target_step_rad", 0.01, 0.10)
        if adaptive
        else 0.0
    )
    preshape_phase_step = (
        integer("preshape_phase_step", 1, 3) if adaptive else 1
    )
    closure_trigger_ttc = (
        number("closure_trigger_ttc_s", 0.05, 1.00)
        if adaptive
        else 0.0
    )
    closure_cross_track = (
        number("closure_cross_track_m", 0.005, 0.08)
        if adaptive
        else 0.0
    )
    closure_z_tolerance = (
        number("closure_z_tolerance_m", 0.005, 0.05)
        if adaptive
        else 0.0
    )
    closure_past_tolerance = (
        number("closure_past_tolerance_m", 0.005, 0.06)
        if adaptive
        else 0.0
    )
    capture_arm_error = (
        number("capture_arm_error_rad", 0.01, 0.20)
        if adaptive
        else 0.0
    )
    joint_correction_reserve = (
        number("joint_correction_reserve_rad", 0.005, 0.10)
        if adaptive
        else 0.0
    )
    return TabletopOnlinePlannerConfig(
        version=str(version),
        control_dt_s=number("control_dt_s", 0.049, 0.051),
        template_start_index=start,
        template_catch_index=catch,
        closure_end_index=closure_end,
        lift_end_index=lift_end,
        fit_sample_count=integer("fit_sample_count", 3, 8),
        motion_axis_base=tuple(float(item) for item in axis),
        reference_intercept_center_base_m=vector(
            "reference_intercept_center_base_m", 3
        ),
        speed_min_m_s=speed_min,
        speed_max_m_s=speed_max,
        heading_cos_min=number("heading_cos_min", 0.5, 1.0),
        max_fit_residual_m=number("max_fit_residual_m", 0.001, 0.05),
        max_linear_prediction_s=number(
            "max_linear_prediction_s", 0.05, 2.0
        ),
        max_position_correction_xy_m=number(
            "max_position_correction_xy_m", 0.005, 0.25
        ),
        max_position_correction_z_m=number(
            "max_position_correction_z_m", 0.0, 0.04
        ),
        correction_filter_alpha=number(
            "correction_filter_alpha", 0.05, 1.0
        ),
        dls_damping=number("dls_damping", 0.001, 0.20),
        max_joint_correction_rad=number(
            "max_joint_correction_rad", 0.01, 0.75
        ),
        max_joint_correction_step_rad=number(
            "max_joint_correction_step_rad", 0.001, 0.05
        ),
        max_target_step_rad=number("max_target_step_rad", 0.02, 0.15),
        max_orientation_correction_rad=number(
            "max_orientation_correction_rad", 0.005, 0.20
        ),
        joint_limit_margin_rad=number(
            "joint_limit_margin_rad", 0.01, 0.20
        ),
        tabletop_plane_base=tuple(float(item) for item in plane),
        minimum_fingertip_clearance_m=number(
            "minimum_fingertip_clearance_m", 0.005, 0.08
        ),
        grasp_vertical_offset_m=(
            number(
                "grasp_vertical_offset_m",
                -0.050,
                0.0 if mpc else -0.015,
            )
            if "grasp_vertical_offset_m" in value
            else -0.030
        ),
        grasp_lift_m=number("grasp_lift_m", 0.05, 0.20),
        contact_prediction_latency_s=number(
            "contact_prediction_latency_s", 0.0, 0.20
        ),
        collision_replan=bool(collision),
        adaptive_speed_max_m_s=adaptive_speed_max,
        approach_target_step_rad=approach_target_step,
        preshape_phase_step=preshape_phase_step,
        closure_trigger_ttc_s=closure_trigger_ttc,
        closure_cross_track_m=closure_cross_track,
        closure_z_tolerance_m=closure_z_tolerance,
        closure_past_tolerance_m=closure_past_tolerance,
        capture_arm_error_rad=capture_arm_error,
        joint_correction_reserve_rad=joint_correction_reserve,
        capture_object_offset_palm_m=(
            vector("capture_object_offset_palm_m", 3) if mpc else None
        ),
        arm_tracking_speed_rad_s=(
            number("arm_tracking_speed_rad_s", 0.05, 1.0) if mpc else 0.0
        ),
        minimum_intercept_horizon_s=(
            number(
                "minimum_intercept_horizon_s",
                0.05,
                number("max_linear_prediction_s", 0.05, 2.0),
            )
            if mpc
            else 0.0
        ),
        open_approach_lift_m=(
            number("open_approach_lift_m", 0.0, 0.08) if mpc else 0.0
        ),
    )


def _readonly_optional_array(
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype,
    name: str,
) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape {shape}") from exc
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have finite shape {shape}")
    result = np.ascontiguousarray(result)
    result.setflags(write=False)
    return result

__all__ = [
    "_optional_rate",
    "_readonly_optional_array",
    "_tabletop_intercept_config",
    "_tabletop_online_planner_config",
    "_validate_action_order",
]
