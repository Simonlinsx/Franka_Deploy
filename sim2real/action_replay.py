"""Validated, transaction-safe policy-action replay for real deployment."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Optional
import zipfile

import numpy as np

MAX_REPLAY_ACTION_BYTES = 8 * 1024 * 1024
MAX_REPLAY_ACTIONS = 720
CANONICAL_ACTION_ORDER = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "thumb_rotation",
    "thumb_bending",
    "index",
    "middle",
    "ring",
    "little",
)


@dataclass(frozen=True)
class TabletopInterceptReplayConfig:
    """Sealed visual trigger for an exact-target tabletop replay plan."""

    preposition_end_index: int
    control_dt_s: float
    fit_sample_count: int
    motion_axis_palm: tuple[float, float, float]
    intercept_center_palm_m: tuple[float, float, float]
    speed_min_m_s: float
    speed_max_m_s: float
    heading_cos_min: float
    max_fit_residual_m: float
    max_lateral_miss_m: float
    trigger_ttc_max_s: float
    max_wait_ticks: int
    collision_replan: bool = False
    collision_min_velocity_change_m_s: float = 0.08


@dataclass(frozen=True)
class TabletopOnlinePlannerConfig:
    """Sealed 20 Hz visual correction contract for a grasp template."""

    version: str
    control_dt_s: float
    template_start_index: int
    template_catch_index: int
    closure_end_index: int
    lift_end_index: int
    fit_sample_count: int
    motion_axis_base: tuple[float, float, float]
    reference_intercept_center_base_m: tuple[float, float, float]
    speed_min_m_s: float
    speed_max_m_s: float
    heading_cos_min: float
    max_fit_residual_m: float
    max_linear_prediction_s: float
    max_position_correction_xy_m: float
    max_position_correction_z_m: float
    correction_filter_alpha: float
    dls_damping: float
    max_joint_correction_rad: float
    max_joint_correction_step_rad: float
    max_target_step_rad: float
    max_orientation_correction_rad: float
    joint_limit_margin_rad: float
    tabletop_plane_base: tuple[float, float, float, float]
    minimum_fingertip_clearance_m: float
    grasp_vertical_offset_m: float
    grasp_lift_m: float
    contact_prediction_latency_s: float
    collision_replan: bool = False
    adaptive_speed_max_m_s: float = 0.0
    approach_target_step_rad: float = 0.0
    preshape_phase_step: int = 1
    closure_trigger_ttc_s: float = 0.0
    closure_cross_track_m: float = 0.0
    closure_z_tolerance_m: float = 0.0
    closure_past_tolerance_m: float = 0.0
    capture_arm_error_rad: float = 0.0
    joint_correction_reserve_rad: float = 0.0
    capture_object_offset_palm_m: Optional[tuple[float, float, float]] = None
    arm_tracking_speed_rad_s: float = 0.0
    minimum_intercept_horizon_s: float = 0.0
    open_approach_lift_m: float = 0.0


@dataclass(frozen=True)
class ReplayActionSequence:
    actions13: np.ndarray
    sha256: str
    source_format: str
    declared_policy_rate_hz: Optional[float]
    recorded_time_s: Optional[np.ndarray] = None
    recorded_franka_target_q_rad: Optional[np.ndarray] = None
    recorded_rh56_angle_set_register_order: Optional[np.ndarray] = None
    recorded_action_contract: str = ""
    tabletop_intercept: Optional[TabletopInterceptReplayConfig] = None
    tabletop_online_planner: Optional[TabletopOnlinePlannerConfig] = None

    @property
    def action_count(self) -> int:
        return int(self.actions13.shape[0])


@dataclass(frozen=True)
class ReplayPolicyOutput:
    action13: np.ndarray
    exact_franka_target_q_rad: Optional[np.ndarray] = None
    exact_rh56_angle_set_register_order: Optional[np.ndarray] = None
    bypass_rh56_host_slew: bool = False
    replay_frame_index: Optional[int] = None
    replay_final_target_arrived: bool = False
    replay_arrival_error_rad: Optional[float] = None
    # Narrow online-planner exception: after closure authority is already
    # committed, an occluded closure tick may raise the palm just enough to
    # preserve the commissioned fingertip/table floor while curling RH56.
    occluded_clearance_lift: bool = False


def summarize_replay_actions(
    sequence: ReplayActionSequence, *, selected_steps: int
) -> Mapping[str, Any]:
    if not isinstance(sequence, ReplayActionSequence):
        raise TypeError("sequence must be ReplayActionSequence")
    if not 1 <= int(selected_steps) <= sequence.action_count:
        raise ValueError("selected_steps exceeds the replay action sequence")
    actions = np.asarray(sequence.actions13[: int(selected_steps)], dtype=np.float64)
    nominal_arm_path = np.cumsum(0.003 * actions[:, :7], axis=0)
    result: dict[str, Any] = {
        "max_abs_normalized_action": float(np.max(np.abs(actions))),
        "arm_nominal_gain_rad_per_tick": 0.003,
        "arm_nominal_final_delta_rad": nominal_arm_path[-1].tolist(),
        "arm_nominal_path_linf_rad": float(np.max(np.abs(nominal_arm_path))),
        "hand_action_min": np.min(actions[:, 7:], axis=0).tolist(),
        "hand_action_max": np.max(actions[:, 7:], axis=0).tolist(),
        "first_action13": actions[0].tolist(),
        "last_action13": actions[-1].tolist(),
    }
    recorded_q = sequence.recorded_franka_target_q_rad
    recorded_hand = sequence.recorded_rh56_angle_set_register_order
    if recorded_q is not None:
        q = np.asarray(recorded_q[: int(selected_steps)], dtype=np.float64)
        q_delta = q - q[0]
        q_tick = np.diff(q, axis=0)
        current_mapper_q = q[0] + nominal_arm_path
        first_change = np.flatnonzero(
            np.any(np.abs(q_delta) > 1.0e-8, axis=1)
        )
        max_recorded_tick = float(np.max(np.abs(q_tick), initial=0.0))
        max_mapper_error = float(np.max(np.abs(current_mapper_q - q)))
        result["recorded_target_audit"] = {
            "action_contract": sequence.recorded_action_contract or None,
            "first_recorded_motion_step": (
                None if first_change.size == 0 else int(first_change[0])
            ),
            "franka_recorded_final_delta_rad": q_delta[-1].tolist(),
            "franka_recorded_path_linf_rad": float(np.max(np.abs(q_delta))),
            "franka_recorded_tick_delta_linf_rad": max_recorded_tick,
            "franka_current_mapper_vs_recorded_max_error_rad": max_mapper_error,
            "franka_current_mapper_equivalent": bool(
                max_mapper_error <= 5.0e-4
            ),
            "franka_recorded_targets_fit_current_0p020_tick_guard": bool(
                max_recorded_tick <= 0.020 + 1.0e-9
            ),
        }
    if recorded_hand is not None:
        hand = np.asarray(recorded_hand[: int(selected_steps)], dtype=np.int32)
        hand_tick = np.diff(hand.astype(np.int64), axis=0)
        audit = result.setdefault("recorded_target_audit", {})
        audit["rh56_recorded_target_min_register_order"] = np.min(
            hand, axis=0
        ).tolist()
        audit["rh56_recorded_target_max_register_order"] = np.max(
            hand, axis=0
        ).tolist()
        audit["rh56_recorded_tick_delta_abs_max_register_order"] = np.max(
            np.abs(hand_tick), axis=0, initial=0
        ).tolist()
    intercept = sequence.tabletop_intercept
    if intercept is not None:
        result["tabletop_intercept"] = {
            "kind": "perception_triggered_exact_target_plan",
            "preposition_end_index": int(intercept.preposition_end_index),
            "fit_sample_count": int(intercept.fit_sample_count),
            "motion_axis_palm": list(intercept.motion_axis_palm),
            "intercept_center_palm_m": list(intercept.intercept_center_palm_m),
            "speed_range_m_s": [
                float(intercept.speed_min_m_s),
                float(intercept.speed_max_m_s),
            ],
            "heading_cos_min": float(intercept.heading_cos_min),
            "max_fit_residual_m": float(intercept.max_fit_residual_m),
            "max_lateral_miss_m": float(intercept.max_lateral_miss_m),
            "trigger_ttc_max_s": float(intercept.trigger_ttc_max_s),
            "max_wait_ticks": int(intercept.max_wait_ticks),
            "collision_replan": bool(intercept.collision_replan),
            "collision_min_velocity_change_m_s": float(
                intercept.collision_min_velocity_change_m_s
            ),
        }
    online = sequence.tabletop_online_planner
    if online is not None:
        result["tabletop_online_planner"] = {
            "kind": "online_visual_intercept_replanning",
            "version": str(online.version),
            "template_start_index": int(online.template_start_index),
            "template_catch_index": int(online.template_catch_index),
            "closure_end_index": int(online.closure_end_index),
            "lift_end_index": int(online.lift_end_index),
            "fit_sample_count": int(online.fit_sample_count),
            "motion_axis_base": list(online.motion_axis_base),
            "reference_intercept_center_base_m": list(
                online.reference_intercept_center_base_m
            ),
            "speed_range_m_s": [
                float(online.speed_min_m_s),
                float(online.speed_max_m_s),
            ],
            "max_position_correction_xy_m": float(
                online.max_position_correction_xy_m
            ),
            "max_position_correction_z_m": float(
                online.max_position_correction_z_m
            ),
            "max_joint_correction_rad": float(
                online.max_joint_correction_rad
            ),
            "max_target_step_rad": float(online.max_target_step_rad),
            "minimum_fingertip_clearance_m": float(
                online.minimum_fingertip_clearance_m
            ),
            "grasp_vertical_offset_m": float(
                online.grasp_vertical_offset_m
            ),
            "grasp_lift_m": float(online.grasp_lift_m),
            "contact_prediction_latency_s": float(
                online.contact_prediction_latency_s
            ),
            "collision_replan": bool(online.collision_replan),
            "adaptive_speed_max_m_s": float(
                online.adaptive_speed_max_m_s
            ),
            "approach_target_step_rad": float(
                online.approach_target_step_rad
            ),
            "preshape_phase_step": int(online.preshape_phase_step),
            "closure_trigger_ttc_s": float(
                online.closure_trigger_ttc_s
            ),
            "closure_cross_track_m": float(
                online.closure_cross_track_m
            ),
            "closure_z_tolerance_m": float(
                online.closure_z_tolerance_m
            ),
            "closure_past_tolerance_m": float(
                online.closure_past_tolerance_m
            ),
            "capture_arm_error_rad": float(
                online.capture_arm_error_rad
            ),
            "joint_correction_reserve_rad": float(
                online.joint_correction_reserve_rad
            ),
        }
        if online.capture_object_offset_palm_m is not None:
            result["tabletop_online_planner"][
                "capture_object_offset_palm_m"
            ] = list(online.capture_object_offset_palm_m)
            result["tabletop_online_planner"]["arm_tracking_speed_rad_s"] = float(
                online.arm_tracking_speed_rad_s
            )
            result["tabletop_online_planner"][
                "minimum_intercept_horizon_s"
            ] = float(online.minimum_intercept_horizon_s)
            result["tabletop_online_planner"]["open_approach_lift_m"] = float(
                online.open_approach_lift_m
            )
    return result


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


def _decode_v205_zip(
    payload: bytes,
) -> tuple[
    object,
    Optional[float],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    str,
    object,
    object,
]:
    """Read the bounded v205 replay interchange bundle without extracting it."""

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
            members = [
                item
                for item in bundle.infolist()
                if not item.is_dir()
                and not item.filename.startswith("__MACOSX/")
                and "/._" not in item.filename
            ]
            if not members:
                raise ValueError("replay ZIP contains no usable files")
            for item in members:
                path = Path(item.filename)
                if path.is_absolute() or ".." in path.parts or item.flag_bits & 0x1:
                    raise ValueError("replay ZIP contains an unsafe member")
                if item.file_size > MAX_REPLAY_ACTION_BYTES:
                    raise ValueError("replay ZIP member exceeds the size limit")

            metadata_candidates: list[tuple[str, Mapping[str, Any]]] = []
            for item in members:
                if not item.filename.lower().endswith(".json"):
                    continue
                try:
                    candidate = json.loads(bundle.read(item).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(candidate, Mapping) and {
                    "control_hz",
                    "action_contract",
                    "recommended_replay_fields",
                }.issubset(candidate):
                    metadata_candidates.append((item.filename, candidate))
            if len(metadata_candidates) != 1:
                raise ValueError(
                    "replay ZIP must contain exactly one replay metadata JSON"
                )
            _, metadata = metadata_candidates[0]
            declared_rate = _optional_rate(
                metadata.get("control_hz"), "replay control_hz"
            )
            policy_order = metadata.get("inspire_policy_order")
            register_order = metadata.get("inspire_register_order")
            if policy_order != list(CANONICAL_ACTION_ORDER[7:]):
                raise ValueError("replay ZIP Inspire policy order is not canonical")
            if register_order != [
                "little",
                "ring",
                "middle",
                "index",
                "thumb_bending",
                "thumb_rotation",
            ]:
                raise ValueError("replay ZIP Inspire register order is not canonical")
            recommended = metadata.get("recommended_replay_fields")
            if recommended != {
                "franka": "franka_joint_target_rad",
                "inspire": "inspire_angle_set_register_order",
            }:
                raise ValueError("replay ZIP recommended target fields are unsupported")

            npz_members = [
                item
                for item in members
                if item.filename.lower().endswith(".npz")
            ]
            if len(npz_members) != 1:
                raise ValueError("replay ZIP must contain exactly one NPZ payload")
            npz_payload = bundle.read(npz_members[0])
            with np.load(io.BytesIO(npz_payload), allow_pickle=False) as archive:
                required = {
                    "time_s",
                    "policy_action",
                    "franka_joint_target_rad",
                    "inspire_angle_set_register_order",
                }
                missing = required - set(archive.files)
                if missing:
                    raise ValueError(
                        f"replay ZIP NPZ is missing arrays: {sorted(missing)}"
                    )
                raw_actions = archive["policy_action"].copy()
                time_s = archive["time_s"].copy()
                franka_targets = archive["franka_joint_target_rad"].copy()
                rh56_targets = archive[
                    "inspire_angle_set_register_order"
                ].copy()
            try:
                expected_frames = int(metadata.get("frames"))
                control_dt_s = float(metadata.get("control_dt_s"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "replay ZIP metadata frames/control_dt_s are invalid"
                ) from exc
            if expected_frames != int(np.asarray(raw_actions).shape[0]):
                raise ValueError("replay ZIP metadata frame count differs from NPZ")
            if (
                declared_rate is None
                or not np.isfinite(control_dt_s)
                or control_dt_s <= 0.0
                or not np.isclose(
                    control_dt_s,
                    1.0 / declared_rate,
                    atol=1.0e-9,
                    rtol=0.0,
                )
            ):
                raise ValueError("replay ZIP control_hz/control_dt_s disagree")
            return (
                raw_actions,
                declared_rate,
                time_s,
                franka_targets,
                rh56_targets,
                str(metadata.get("action_contract", "")).strip(),
                metadata.get("tabletop_intercept"),
                metadata.get("tabletop_online_planner"),
            )
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ValueError(f"invalid replay ZIP: {exc}") from exc


def load_replay_actions_payload(
    payload: bytes,
    *,
    suffix: str,
    expected_policy_rate_hz: object,
    selected_steps: Optional[int] = None,
) -> ReplayActionSequence:
    """Decode and validate JSON/NPY/NPZ/CSV action rows without pickle."""

    if not isinstance(payload, bytes) or not payload:
        raise ValueError("replay action payload is empty")
    if len(payload) > MAX_REPLAY_ACTION_BYTES:
        raise ValueError(
            f"replay action payload exceeds {MAX_REPLAY_ACTION_BYTES} bytes"
        )
    extension = str(suffix).strip().lower()
    declared_rate: Optional[float] = None
    recorded_time: Optional[np.ndarray] = None
    recorded_franka_targets: Optional[np.ndarray] = None
    recorded_rh56_targets: Optional[np.ndarray] = None
    recorded_action_contract = ""
    raw_tabletop_intercept: object = None
    raw_tabletop_online_planner: object = None
    if extension == ".json":
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid replay JSON: {exc}") from exc
        if isinstance(decoded, Mapping):
            unknown = set(decoded) - {
                "actions",
                "action13",
                "policy_rate_hz",
                "action_order",
            }
            if unknown:
                raise ValueError(f"unknown replay JSON fields: {sorted(unknown)}")
            raw_actions = decoded.get("actions", decoded.get("action13"))
            declared_rate = _optional_rate(
                decoded.get("policy_rate_hz"), "replay policy_rate_hz"
            )
            _validate_action_order(decoded.get("action_order"))
        else:
            raw_actions = decoded
        source_format = "json"
    elif extension == ".npy":
        try:
            raw_actions = np.load(io.BytesIO(payload), allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid replay NPY: {exc}") from exc
        source_format = "npy"
    elif extension == ".npz":
        try:
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                allowed = {
                    "actions",
                    "action13",
                    "policy_action",
                    "policy_rate_hz",
                    "action_order",
                    "time_s",
                    "franka_joint_target_rad",
                    "inspire_angle_set_register_order",
                    "policy_step",
                    "franka_joint_position_rad",
                    "franka_joint_velocity_rad_s",
                    "inspire_joint_position_rad_policy_order",
                    "inspire_joint_velocity_rad_s_policy_order",
                    "inspire_joint_target_rad_policy_order",
                    "success",
                    "stable_hold",
                }
                unknown = set(archive.files) - allowed
                if unknown:
                    raise ValueError(f"unknown replay NPZ arrays: {sorted(unknown)}")
                action_key = next(
                    (
                        key
                        for key in ("actions", "action13", "policy_action")
                        if key in archive
                    ),
                    "",
                )
                if action_key not in archive:
                    raise ValueError(
                        "replay NPZ must contain actions, action13, or policy_action"
                    )
                raw_actions = archive[action_key].copy()
                if "policy_rate_hz" in archive:
                    declared_rate = _optional_rate(
                        archive["policy_rate_hz"], "replay policy_rate_hz"
                    )
                if "action_order" in archive:
                    _validate_action_order(archive["action_order"].tolist())
                if "time_s" in archive:
                    recorded_time = archive["time_s"].copy()
                if "franka_joint_target_rad" in archive:
                    recorded_franka_targets = archive[
                        "franka_joint_target_rad"
                    ].copy()
                if "inspire_angle_set_register_order" in archive:
                    recorded_rh56_targets = archive[
                        "inspire_angle_set_register_order"
                    ].copy()
        except (OSError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith(
                ("unknown replay", "replay NPZ", "replay action_order")
            ):
                raise
            raise ValueError(f"invalid replay NPZ: {exc}") from exc
        source_format = "npz"
    elif extension in (".csv", ".txt"):
        try:
            text = payload.decode("utf-8")
            if extension == ".csv" and text.lstrip().startswith("time_s,"):
                rows = list(csv.DictReader(io.StringIO(text)))
                action_names = [f"action_{index}" for index in range(13)]
                hand_names = [
                    "rh56_angle_set_little",
                    "rh56_angle_set_ring",
                    "rh56_angle_set_middle",
                    "rh56_angle_set_index",
                    "rh56_angle_set_thumb_bending",
                    "rh56_angle_set_thumb_rotation",
                ]
                if not rows or not all(
                    name in rows[0]
                    for name in action_names
                    + [f"franka_target_rad_{index}" for index in range(7)]
                    + hand_names
                ):
                    raise ValueError("replay CSV header is incomplete")
                raw_actions = np.asarray(
                    [[row[name] for name in action_names] for row in rows],
                    dtype=np.float64,
                )
                recorded_time = np.asarray(
                    [row["time_s"] for row in rows], dtype=np.float64
                )
                recorded_franka_targets = np.asarray(
                    [
                        [row[f"franka_target_rad_{index}"] for index in range(7)]
                        for row in rows
                    ],
                    dtype=np.float64,
                )
                recorded_rh56_targets = np.asarray(
                    [[row[name] for name in hand_names] for row in rows],
                    dtype=np.float64,
                )
            else:
                raw_actions = np.loadtxt(
                    io.StringIO(text),
                    dtype=np.float64,
                    delimiter="," if extension == ".csv" else None,
                    comments="#",
                    ndmin=2,
                )
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f"invalid replay {extension[1:].upper()}: {exc}") from exc
        source_format = extension[1:]
    elif extension == ".zip":
        (
            raw_actions,
            declared_rate,
            recorded_time,
            recorded_franka_targets,
            recorded_rh56_targets,
            recorded_action_contract,
            raw_tabletop_intercept,
            raw_tabletop_online_planner,
        ) = _decode_v205_zip(payload)
        source_format = "v205_replay_zip"
    else:
        raise ValueError(
            "replay actions must use .json, .npy, .npz, .csv, .txt, or .zip"
        )

    try:
        actions = np.asarray(raw_actions, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("replay actions must be a numeric [K,13] array") from exc
    if (
        actions.ndim != 2
        or actions.shape[1] != 13
        or not 1 <= actions.shape[0] <= MAX_REPLAY_ACTIONS
    ):
        raise ValueError(
            f"replay actions must have shape [K,13], K=1..{MAX_REPLAY_ACTIONS}; "
            f"actual={actions.shape}"
        )
    if not np.all(np.isfinite(actions)):
        raise ValueError("replay actions contain NaN or infinity")
    if np.any(actions < -1.0) or np.any(actions > 1.0):
        index = np.argwhere((actions < -1.0) | (actions > 1.0))[0]
        value = float(actions[tuple(index)])
        raise ValueError(
            "replay actions must already lie in [-1,1] without clipping: "
            f"row={int(index[0])} axis={int(index[1])} value={value:.9g}"
        )
    if selected_steps is not None:
        if (
            isinstance(selected_steps, bool)
            or not isinstance(selected_steps, (int, np.integer))
            or not 1 <= int(selected_steps) <= actions.shape[0]
        ):
            raise ValueError(f"selected replay steps must be in 1..{actions.shape[0]}")
    actions = np.ascontiguousarray(actions)
    actions.setflags(write=False)
    row_count = int(actions.shape[0])
    tabletop_intercept = _tabletop_intercept_config(
        raw_tabletop_intercept,
        action_count=row_count,
    )
    tabletop_online_planner = _tabletop_online_planner_config(
        raw_tabletop_online_planner,
        action_count=row_count,
    )
    if tabletop_intercept is not None and tabletop_online_planner is not None:
        raise ValueError(
            "replay cannot contain both tabletop_intercept and "
            "tabletop_online_planner"
        )
    time_values = (
        None
        if recorded_time is None
        else _readonly_optional_array(
            recorded_time,
            shape=(row_count,),
            dtype=np.dtype(np.float64),
            name="recorded time_s",
        )
    )
    if time_values is not None:
        if np.any(np.diff(time_values) <= 0.0):
            raise ValueError("recorded time_s must be strictly increasing")
        if declared_rate is None and row_count > 1:
            periods = np.diff(time_values)
            if np.allclose(periods, periods[0], atol=1.0e-6, rtol=0.0):
                declared_rate = float(1.0 / periods[0])
    expected_rate = _optional_rate(expected_policy_rate_hz, "expected policy rate")
    assert expected_rate is not None
    if declared_rate is not None and not np.isclose(
        declared_rate, expected_rate, atol=1.0e-6, rtol=0.0
    ):
        raise ValueError(
            "replay policy_rate_hz disagrees with --policy-rate-hz: "
            f"file={declared_rate:g} command={expected_rate:g}"
        )
    franka_targets = (
        None
        if recorded_franka_targets is None
        else _readonly_optional_array(
            recorded_franka_targets,
            shape=(row_count, 7),
            dtype=np.dtype(np.float32),
            name="recorded Franka targets",
        )
    )
    rh56_targets = (
        None
        if recorded_rh56_targets is None
        else _readonly_optional_array(
            recorded_rh56_targets,
            shape=(row_count, 6),
            dtype=np.dtype(np.float64),
            name="recorded RH56 targets",
        )
    )
    if rh56_targets is not None:
        if not np.all(rh56_targets == np.rint(rh56_targets)):
            raise ValueError("recorded RH56 targets must be integer registers")
        if np.any(rh56_targets < 0.0) or np.any(rh56_targets > 1000.0):
            raise ValueError("recorded RH56 targets must lie in [0,1000]")
        rh56_targets = np.asarray(rh56_targets, dtype=np.int32)
        rh56_targets.setflags(write=False)
    return ReplayActionSequence(
        actions13=actions,
        sha256=hashlib.sha256(payload).hexdigest(),
        source_format=source_format,
        declared_policy_rate_hz=declared_rate,
        recorded_time_s=time_values,
        recorded_franka_target_q_rad=franka_targets,
        recorded_rh56_angle_set_register_order=rh56_targets,
        recorded_action_contract=recorded_action_contract,
        tabletop_intercept=tabletop_intercept,
        tabletop_online_planner=tabletop_online_planner,
    )


def load_replay_actions(
    path: str | Path,
    *,
    expected_policy_rate_hz: object,
    selected_steps: Optional[int] = None,
) -> ReplayActionSequence:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"replay action file not found: {resolved}")
    size = resolved.stat().st_size
    if size > MAX_REPLAY_ACTION_BYTES:
        raise ValueError(f"replay action file exceeds {MAX_REPLAY_ACTION_BYTES} bytes")
    payload = resolved.read_bytes()
    if len(payload) != size:
        raise ValueError("replay action file changed while it was being read")
    return load_replay_actions_payload(
        payload,
        suffix=resolved.suffix,
        expected_policy_rate_hz=expected_policy_rate_hz,
        selected_steps=selected_steps,
    )


class TransactionalReplayActionPolicy:
    """Rollback-safe fixed-rate or measured-arrival replay source."""

    is_replay_policy = True
    transactional_state_contract = (
        "external_four_frame_history_lstm_recomputed_from_zero_v1"
    )

    def __init__(
        self,
        sequence: ReplayActionSequence,
        *,
        point_feature_dim: int,
        selected_steps: int,
        arrival_gated: bool = False,
        arrival_tolerance_rad: float = 0.015,
        q_home_rad: Optional[np.ndarray] = None,
    ) -> None:
        if not isinstance(sequence, ReplayActionSequence):
            raise TypeError("sequence must be ReplayActionSequence")
        if isinstance(point_feature_dim, bool) or int(point_feature_dim) not in (3, 6):
            raise ValueError("point_feature_dim must be 3 or 6")
        if not 1 <= int(selected_steps) <= sequence.action_count:
            raise ValueError("selected_steps exceeds the replay action sequence")
        self.sequence = sequence
        self.point_feature_dim = int(point_feature_dim)
        self.point_feature_mode = "xyz" if self.point_feature_dim == 3 else "xyzrgb"
        self.selected_steps = int(selected_steps)
        self.arrival_gated = bool(arrival_gated)
        tolerance = float(arrival_tolerance_rad)
        if not np.isfinite(tolerance) or not 0.005 <= tolerance <= 0.030:
            raise ValueError("arrival_tolerance_rad must be in 0.005..0.030")
        self.arrival_tolerance_rad = tolerance
        if q_home_rad is None:
            self.q_home_rad = np.zeros(7, dtype=np.float32)
        else:
            home = np.asarray(q_home_rad, dtype=np.float32)
            if home.shape != (7,) or not np.all(np.isfinite(home)):
                raise ValueError("q_home_rad must contain seven finite values")
            self.q_home_rad = home.copy()
        if self.arrival_gated and (
            sequence.recorded_franka_target_q_rad is None
            or sequence.recorded_rh56_angle_set_register_order is None
        ):
            raise ValueError("arrival-gated replay requires exact actuator targets")
        self.tabletop_intercept = sequence.tabletop_intercept
        if self.tabletop_intercept is not None:
            if not self.arrival_gated:
                raise ValueError(
                    "tabletop intercept replay requires --arrival-gated"
                )
            if self.selected_steps != sequence.action_count:
                raise ValueError(
                    "tabletop intercept replay requires the complete action plan"
                )
            if self.tabletop_intercept.preposition_end_index >= (
                self.selected_steps - 1
            ):
                raise ValueError(
                    "tabletop intercept plan has no post-trigger trajectory"
                )
        self._committed_replay_index = -1
        self._pending_sequence: Optional[int] = None
        self._pending_replay_index: Optional[int] = None
        self._pending_final_arrived = False
        self._pending_arrival_error_rad: Optional[float] = None
        self._replay_complete = False
        self._repeated_target_commands = 0
        self._intercept_armed = False
        self._intercept_triggered = False
        self._intercept_wait_ticks = 0
        self._intercept_samples: list[tuple[int, np.ndarray]] = []
        self._intercept_preimpact_velocity: Optional[np.ndarray] = None
        self._intercept_collision_seen = False
        self._intercept_post_collision_samples: list[
            tuple[int, np.ndarray]
        ] = []
        self._intercept_last_diagnostics: dict[str, object] = {
            "state": (
                "disabled" if self.tabletop_intercept is None else "prepositioning"
            )
        }
        self._pending_intercept_sample: Optional[tuple[int, np.ndarray]] = None
        self._pending_intercept_armed = False
        self._pending_intercept_triggered = False
        self._pending_intercept_preimpact_velocity: Optional[np.ndarray] = None
        self._pending_intercept_collision_seen = False
        self._pending_intercept_reset_post_samples = False
        self._pending_intercept_diagnostics: Optional[dict[str, object]] = None

    @property
    def replay_complete(self) -> bool:
        return bool(self._replay_complete)

    @property
    def completed_replay_frames(self) -> int:
        return max(0, self._committed_replay_index + 1)

    @property
    def repeated_target_commands(self) -> int:
        return int(self._repeated_target_commands)

    @property
    def diagnostics_snapshot(self) -> Mapping[str, object]:
        result = dict(self._intercept_last_diagnostics)
        result.update(
            {
                "enabled": self.tabletop_intercept is not None,
                "armed": bool(self._intercept_armed),
                "triggered": bool(self._intercept_triggered),
                "wait_ticks": int(self._intercept_wait_ticks),
                "collision_seen": bool(self._intercept_collision_seen),
                "samples": len(self._intercept_samples),
                "post_collision_samples": len(
                    self._intercept_post_collision_samples
                ),
            }
        )
        return result

    def _clear_pending_intercept(self) -> None:
        self._pending_intercept_sample = None
        self._pending_intercept_armed = False
        self._pending_intercept_triggered = False
        self._pending_intercept_preimpact_velocity = None
        self._pending_intercept_collision_seen = False
        self._pending_intercept_reset_post_samples = False
        self._pending_intercept_diagnostics = None

    @staticmethod
    def _object_center(
        pointcloud: np.ndarray,
        valid: np.ndarray,
    ) -> np.ndarray:
        points = np.asarray(pointcloud, dtype=np.float64)[-1, :, :3]
        validity = np.asarray(valid, dtype=np.float64)[-1] >= 0.5
        selected = points[validity]
        if selected.shape[0] < 16 or not np.all(np.isfinite(selected)):
            raise ValueError(
                "tabletop intercept requires at least 16 finite object points"
            )
        center = np.mean(selected, axis=0)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("tabletop intercept object center is invalid")
        return center

    def _fit_intercept(
        self,
        samples: list[tuple[int, np.ndarray]],
    ) -> dict[str, object]:
        config = self.tabletop_intercept
        assert config is not None
        count = int(config.fit_sample_count)
        if len(samples) < count:
            return {
                "state": "collecting_motion_samples",
                "fit_samples": len(samples),
                "required_fit_samples": count,
            }
        selected = samples[-count:]
        times = np.asarray(
            [item[0] * config.control_dt_s for item in selected],
            dtype=np.float64,
        )
        centers = np.stack([item[1] for item in selected], axis=0)
        centered_time = times - float(np.mean(times))
        design = np.stack(
            [centered_time, np.ones_like(centered_time)], axis=1
        )
        coefficients, _, _, _ = np.linalg.lstsq(design, centers, rcond=None)
        velocity = coefficients[0]
        predicted = design @ coefficients
        residual = float(
            np.sqrt(np.mean(np.sum((centers - predicted) ** 2, axis=1)))
        )
        speed = float(np.linalg.norm(velocity))
        axis = np.asarray(config.motion_axis_palm, dtype=np.float64)
        heading = float(np.dot(velocity, axis) / speed) if speed > 1.0e-9 else -1.0
        current = centers[-1]
        target = np.asarray(config.intercept_center_palm_m, dtype=np.float64)
        velocity_sq = float(np.dot(velocity, velocity))
        ttc = (
            float(np.dot(target - current, velocity) / velocity_sq)
            if velocity_sq > 1.0e-12
            else float("inf")
        )
        closest = current + velocity * max(0.0, ttc)
        lateral = float(np.linalg.norm(closest - target))
        stable = bool(
            config.speed_min_m_s <= speed <= config.speed_max_m_s
            and heading >= config.heading_cos_min
            and residual <= config.max_fit_residual_m
        )
        trigger_geometry = bool(
            stable
            and 0.0 <= ttc <= config.trigger_ttc_max_s
            and lateral <= config.max_lateral_miss_m
        )
        return {
            "state": "waiting_for_intercept",
            "fit_samples": count,
            "center_palm_m": current.tolist(),
            "velocity_palm_m_s": velocity.tolist(),
            "speed_m_s": speed,
            "heading_cos": heading,
            "fit_residual_m": residual,
            "ttc_s": ttc if np.isfinite(ttc) else None,
            "lateral_miss_m": lateral,
            "stable_motion": stable,
            "trigger_geometry": trigger_geometry,
            "_velocity": velocity,
        }

    def _stage_intercept_wait(
        self,
        *,
        sequence: int,
        pointcloud: np.ndarray,
        valid: np.ndarray,
    ) -> bool:
        config = self.tabletop_intercept
        assert config is not None
        if self._intercept_wait_ticks >= config.max_wait_ticks:
            raise RuntimeError(
                "tabletop intercept wait expired without a safe trajectory"
            )
        sample = (int(sequence), self._object_center(pointcloud, valid))
        self._pending_intercept_sample = sample
        self._pending_intercept_armed = not self._intercept_armed
        working = self._intercept_samples + [sample]
        fit_samples = working
        if config.collision_replan and self._intercept_collision_seen:
            fit_samples = self._intercept_post_collision_samples + [sample]
        diagnostics = self._fit_intercept(fit_samples)
        velocity_value = diagnostics.get("_velocity")
        velocity = (
            None
            if velocity_value is None
            else np.asarray(velocity_value, dtype=np.float64)
        )
        trigger = bool(diagnostics.get("trigger_geometry", False))
        if config.collision_replan:
            if self._intercept_preimpact_velocity is None:
                if bool(diagnostics.get("stable_motion", False)) and velocity is not None:
                    self._pending_intercept_preimpact_velocity = velocity.copy()
                trigger = False
                diagnostics["state"] = "waiting_for_collision"
            elif not self._intercept_collision_seen:
                if velocity is not None and (
                    float(
                        np.linalg.norm(
                            velocity - self._intercept_preimpact_velocity
                        )
                    )
                    >= config.collision_min_velocity_change_m_s
                ):
                    self._pending_intercept_collision_seen = True
                    self._pending_intercept_reset_post_samples = True
                    diagnostics["state"] = "collision_detected_refitting"
                else:
                    diagnostics["state"] = "waiting_for_collision"
                trigger = False
            else:
                diagnostics["state"] = (
                    "post_collision_trigger_ready"
                    if trigger
                    else "refitting_after_collision"
                )
        if trigger:
            diagnostics["state"] = "trigger_ready"
            self._pending_intercept_triggered = True
        diagnostics.pop("_velocity", None)
        self._pending_intercept_diagnostics = diagnostics
        return trigger

    def commit_replay_proposal(self, sequence: int) -> None:
        if not self.arrival_gated:
            index = int(sequence) - 1
            if not 0 <= index < self.selected_steps:
                raise RuntimeError("fixed-rate replay commit is outside the sequence")
            self._committed_replay_index = index
            return
        if self._pending_sequence != int(sequence) or self._pending_replay_index is None:
            raise RuntimeError("replay commit differs from the pending proposal")
        if self._pending_replay_index == self._committed_replay_index:
            self._repeated_target_commands += 1
        sample = self._pending_intercept_sample
        armed_transition = self._pending_intercept_armed
        triggered_transition = self._pending_intercept_triggered
        preimpact_velocity = self._pending_intercept_preimpact_velocity
        collision_transition = self._pending_intercept_collision_seen
        reset_post_samples = self._pending_intercept_reset_post_samples
        diagnostics = self._pending_intercept_diagnostics
        self._committed_replay_index = self._pending_replay_index
        self._replay_complete = bool(self._pending_final_arrived)
        self._pending_sequence = None
        self._pending_replay_index = None
        self._pending_final_arrived = False
        self._pending_arrival_error_rad = None
        if sample is not None:
            self._intercept_wait_ticks += 1
            self._intercept_samples.append((sample[0], sample[1].copy()))
            self._intercept_samples = self._intercept_samples[-24:]
            if self._intercept_collision_seen:
                self._intercept_post_collision_samples.append(
                    (sample[0], sample[1].copy())
                )
                self._intercept_post_collision_samples = (
                    self._intercept_post_collision_samples[-24:]
                )
        if armed_transition:
            self._intercept_armed = True
            print(
                "[Tabletop planner ARMED] pre-catch pose arrived; roll the object now",
                flush=True,
            )
        if preimpact_velocity is not None:
            self._intercept_preimpact_velocity = preimpact_velocity.copy()
        if collision_transition:
            self._intercept_collision_seen = True
            if reset_post_samples:
                self._intercept_post_collision_samples = (
                    []
                    if sample is None
                    else [(sample[0], sample[1].copy())]
                )
            print(
                "[Tabletop planner COLLISION] velocity change accepted; refitting post-impact flight",
                flush=True,
            )
        if triggered_transition:
            self._intercept_triggered = True
            print(
                "[Tabletop planner TRIGGERED] safe intercept window accepted; executing grasp template",
                flush=True,
            )
        if diagnostics is not None:
            self._intercept_last_diagnostics = dict(diagnostics)
        self._clear_pending_intercept()

    def discard_replay_proposal(self, sequence: int) -> None:
        if not self.arrival_gated:
            return
        if self._pending_sequence != int(sequence):
            raise RuntimeError("replay discard differs from the pending proposal")
        self._pending_sequence = None
        self._pending_replay_index = None
        self._pending_final_arrived = False
        self._pending_arrival_error_rad = None
        self._clear_pending_intercept()

    def action_for_sequence(
        self,
        sequence: int,
        pointcloud: np.ndarray,
        valid: np.ndarray,
        proprio: np.ndarray,
    ) -> ReplayPolicyOutput:
        if isinstance(sequence, bool) or not isinstance(sequence, (int, np.integer)):
            raise ValueError("replay sequence must be an integer")
        index = int(sequence) - 1
        if not self.arrival_gated and not 0 <= index < self.selected_steps:
            raise ValueError(
                f"replay sequence {sequence} is outside 1..{self.selected_steps}"
            )
        if np.asarray(pointcloud).shape != (4, 128, self.point_feature_dim):
            raise ValueError("replay received a malformed point-cloud history")
        if np.asarray(valid).shape != (4, 128) or np.asarray(proprio).shape != (4, 67):
            raise ValueError("replay received malformed valid/proprio history")
        if not self.arrival_gated:
            franka_target = self.sequence.recorded_franka_target_q_rad
            rh56_target = self.sequence.recorded_rh56_angle_set_register_order
            has_exact_targets = franka_target is not None and rh56_target is not None
            return ReplayPolicyOutput(
                action13=self.sequence.actions13[index].copy(),
                exact_franka_target_q_rad=(
                    None if franka_target is None else franka_target[index].copy()
                ),
                exact_rh56_angle_set_register_order=(
                    None if rh56_target is None else rh56_target[index].copy()
                ),
                bypass_rh56_host_slew=has_exact_targets,
                replay_frame_index=index,
            )
        if self._pending_sequence is not None:
            if self._pending_sequence != int(sequence):
                raise RuntimeError("replay already has a different pending proposal")
            assert self._pending_replay_index is not None
            index = self._pending_replay_index
            final_arrived = self._pending_final_arrived
            arrival_error = self._pending_arrival_error_rad
        else:
            final_arrived = False
            arrival_error: Optional[float] = None
            if self._committed_replay_index < 0:
                index = 0
            else:
                current_index = self._committed_replay_index
                assert self.sequence.recorded_franka_target_q_rad is not None
                measured_q = (
                    np.asarray(proprio, dtype=np.float32)[-1, :7]
                    + self.q_home_rad
                )
                target_q = self.sequence.recorded_franka_target_q_rad[current_index]
                arrival_error = float(
                    np.max(np.abs(measured_q.astype(np.float64) - target_q))
                )
                arrived = arrival_error <= self.arrival_tolerance_rad
                intercept = self.tabletop_intercept
                if intercept is not None and current_index < (
                    intercept.preposition_end_index
                ):
                    # The recorded successful policy trajectory is already a
                    # 20 Hz, native-rate-limited exact target stream.  Do not
                    # insert a stop at every waypoint: stream continuously to
                    # the preposition endpoint, while the exact-replay arm
                    # hold guard remains able to pause the entire transaction.
                    index = current_index + 1
                elif (
                    intercept is not None
                    and current_index == intercept.preposition_end_index
                    and not self._intercept_triggered
                ):
                    if arrived:
                        trigger = self._stage_intercept_wait(
                            sequence=int(sequence),
                            pointcloud=pointcloud,
                            valid=valid,
                        )
                        index = current_index + 1 if trigger else current_index
                    else:
                        index = current_index
                elif (
                    intercept is not None
                    and current_index + 1 < self.selected_steps
                ):
                    # Once triggered, preserve the demonstrated closure/lift
                    # timing at one exact waypoint per accepted 20 Hz tick.
                    index = current_index + 1
                elif arrived and current_index + 1 < self.selected_steps:
                    index = current_index + 1
                else:
                    index = current_index
                    final_arrived = bool(
                        arrived and current_index == self.selected_steps - 1
                    )
        if not 0 <= index < self.selected_steps:
            raise ValueError(
                f"replay frame {index} is outside 0..{self.selected_steps - 1}"
            )
        if self._pending_sequence is None:
            self._pending_sequence = int(sequence)
            self._pending_replay_index = index
            self._pending_final_arrived = final_arrived
            self._pending_arrival_error_rad = arrival_error
        franka_target = self.sequence.recorded_franka_target_q_rad
        rh56_target = self.sequence.recorded_rh56_angle_set_register_order
        has_exact_targets = franka_target is not None and rh56_target is not None
        return ReplayPolicyOutput(
            action13=self.sequence.actions13[index].copy(),
            exact_franka_target_q_rad=(
                None if franka_target is None else franka_target[index].copy()
            ),
            exact_rh56_angle_set_register_order=(
                None if rh56_target is None else rh56_target[index].copy()
            ),
            # The replay bundle already contains the simulator's actuator
            # setpoints.  Re-slewing RH56 on the host changes that trajectory;
            # the hand firmware's SPEED_SET remains the physical motion limit.
            bypass_rh56_host_slew=has_exact_targets,
            replay_frame_index=index,
            replay_final_target_arrived=final_arrived,
            replay_arrival_error_rad=arrival_error,
        )

    def act(
        self, pointcloud: np.ndarray, valid: np.ndarray, proprio: np.ndarray
    ) -> ReplayPolicyOutput:
        raise RuntimeError(
            "transactional replay must be addressed by committed sequence"
        )


__all__ = [
    "CANONICAL_ACTION_ORDER",
    "MAX_REPLAY_ACTION_BYTES",
    "MAX_REPLAY_ACTIONS",
    "ReplayActionSequence",
    "ReplayPolicyOutput",
    "TabletopInterceptReplayConfig",
    "TabletopOnlinePlannerConfig",
    "TransactionalReplayActionPolicy",
    "load_replay_actions",
    "load_replay_actions_payload",
    "summarize_replay_actions",
]
