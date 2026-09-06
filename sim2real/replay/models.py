"""Replay contracts and read-only summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

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

__all__ = [
    "CANONICAL_ACTION_ORDER",
    "MAX_REPLAY_ACTION_BYTES",
    "MAX_REPLAY_ACTIONS",
    "ReplayActionSequence",
    "ReplayPolicyOutput",
    "TabletopInterceptReplayConfig",
    "TabletopOnlinePlannerConfig",
    "summarize_replay_actions",
]
