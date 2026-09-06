"""Transaction-safe 20 Hz visual interception for moving tabletop objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from sim2real.action_replay import ReplayActionSequence, ReplayPolicyOutput
from sim2real.policy import (
    ActionControllerParameters,
    QD_G015_ACTION_CONTROLLER_CONTRACT_ID,
)
from sim2real.contracts.v94 import V94Contract
from sim2real.v94_kinematics import RH56FingertipKinematics
from motion_planning.kinematics import (
    _rotation_vector,
    cartesian_joint_correction,
    panda_T_base_policy_palm,
    pose_preserving_joint_target,
)

# A ball can become fully hidden by the RH56 during the last quarter-second
# of a closure that was already admitted by fresh spatial/TTC observations.
# The bounded continuation holds the Franka byte-exact and advances only the
# immutable demonstrated hand targets.  Five ticks corresponds to the
# observed late-closure self-occlusion boundary (step 8 of 13) at 20 Hz.
_MAXIMUM_OCCLUDED_CLOSURE_TICKS = 5

# A current exact observation may momentarily miss the normal live-fit
# residual while the hand is already in the capture corridor.  The board-
# collision task is especially susceptible because the post-impact centroid
# can deform by a few pixels.  Admit that frame only as a bounded continuation
# of an already observed live fit: current motion must still be strongly
# aligned, the planner must be holding the last live intercept, and all
# spatial/TTC/table-clearance gates remain authoritative.  This is not a stale
# point-cloud fallback and does not alter the ordinary capture corridor.
_DEGRADED_LIVE_FIT_CAPTURE_MAX_RESIDUAL_M = 0.025
_DEGRADED_LIVE_FIT_CAPTURE_MIN_HEADING_COS = 0.95
_DEGRADED_LIVE_FIT_CAPTURE_CROSS_TRACK_SLACK_M = 0.005

# Once closure has been admitted, a more-flexed RH56 target can transiently
# lower one modeled fingertip even when the Franka target is held byte-exact.
# Do not abort the grasp and do not lower the table floor: translate the palm
# upward by only the missing clearance, bounded to a small closure recovery.
_MAXIMUM_CLOSURE_CLEARANCE_LIFT_M = 0.020
_CLOSURE_CLEARANCE_LIFT_MARGIN_M = 0.001

# The installed RH56 reports calibrated per-axis FORCE_ACT values in grams.
# A rapid multi-finger rise while a fresh object cloud is already inside a
# bounded near-hand shell is independent evidence that the open hand has met
# the object.  It is used only to start the immutable closure at a frozen arm
# target after the exact command ACK; it never authorizes another descent.
_CONTACT_CAPTURE_FORCE_ABSOLUTE_G = 200
_CONTACT_CAPTURE_FORCE_DELTA_G = 150
_CONTACT_CAPTURE_MINIMUM_FINGER_AXES = 2
_CONTACT_CAPTURE_PLANAR_SHELL_M = 0.105
_CONTACT_CAPTURE_Z_SHELL_M = 0.035


def _minimum_jerk(fraction: float) -> float:
    value = float(np.clip(fraction, 0.0, 1.0))
    return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)


@dataclass(frozen=True)
class _OnlineProposal:
    sequence: int
    index: int
    correction_q_rad: np.ndarray
    target_q_rad: np.ndarray
    goal_q_rad: np.ndarray
    capture_q_rad: Optional[np.ndarray]
    lift_q_rad: Optional[np.ndarray]
    tabletop_height_correction_m: float
    live_fit_seen: bool
    diagnostics: Mapping[str, object]
    full_occluded_closure_committed: bool = False
    current_volume_capture_committed: bool = False
    force_contact_capture_committed: bool = False
    template_index: int = -1
    nominal_target_q_rad: Optional[np.ndarray] = None
    adaptive_phase: str = "legacy"
    preshape_index: int = -1
    closure_step: int = -1
    lift_step: int = -1


class _OrientationCorrectionConflict(RuntimeError):
    """A live correction cannot be composed with the current nominal wrist."""

    def __init__(self, *, required_rad: float, limit_rad: float) -> None:
        self.required_rad = float(required_rad)
        self.limit_rad = float(limit_rad)
        super().__init__(
            "online planner target exceeds orientation-correction envelope: "
            f"required={self.required_rad:.6f}rad "
            f"limit={self.limit_rad:.6f}rad"
        )


class _FingertipTableClearanceConflict(RuntimeError):
    """A proposed arm/hand target would cross the table-clearance floor."""

    def __init__(self, *, required_m: float, limit_m: float) -> None:
        self.required_m = float(required_m)
        self.limit_m = float(limit_m)
        super().__init__(
            "online planner target violates fingertip/table clearance: "
            f"required={self.required_m:.6f}m "
            f"limit={self.limit_m:.6f}m"
        )


class TransactionalTabletopOnlinePlannerPolicy:
    """Predict, approach, track through closure, and lift a moving object."""

    is_replay_policy = True
    transactional_state_contract = (
        "external_four_frame_history_lstm_recomputed_from_zero_v1"
    )

    def __init__(
        self,
        sequence: ReplayActionSequence,
        *,
        contract: V94Contract,
        action_controller: ActionControllerParameters,
        point_feature_dim: int,
        history_length: int,
        proprio_dim: int,
        selected_steps: int,
    ) -> None:
        config = sequence.tabletop_online_planner
        if config is None:
            raise ValueError("online tabletop planner metadata is missing")
        if config.version not in {
            "tabletop_online_intercept_planner_v2",
            "tabletop_online_intercept_planner_v3",
            "tabletop_online_intercept_planner_v4",
            "tabletop_online_intercept_planner_v5",
        }:
            raise ValueError("online tabletop planner bundle has an unknown version")
        if sequence.tabletop_intercept is not None:
            raise ValueError("online and preposition intercept plans are exclusive")
        if sequence.recorded_franka_target_q_rad is None or (
            sequence.recorded_rh56_angle_set_register_order is None
        ):
            raise ValueError("online tabletop planner requires exact actuator targets")
        if int(selected_steps) != sequence.action_count:
            raise ValueError("online tabletop planner requires the complete template")
        if action_controller.contract_id != QD_G015_ACTION_CONTROLLER_CONTRACT_ID:
            raise ValueError(
                "online tabletop planner requires the checkpoint q_d-g015 controller"
            )
        if (int(point_feature_dim), int(history_length), int(proprio_dim)) not in {
            (3, 8, 67),
            (3, 8, 96),
        }:
            raise ValueError(
                "online tabletop planner requires XYZ history=8 and proprio=67/96"
            )
        self.sequence = sequence
        self.config = config
        self.contract = contract
        self.action_controller = action_controller
        self.initial_previous_action13 = np.asarray(
            action_controller.initial_previous_action13, dtype=np.float32
        )
        self.initial_previous_action13.setflags(write=False)
        self.point_feature_dim = int(point_feature_dim)
        self.point_feature_mode = "xyz"
        self.history_length = int(history_length)
        self.proprio_dim = int(proprio_dim)
        self.selected_steps = int(selected_steps)
        self._fingertips = RH56FingertipKinematics(contract)
        self._committed_index = -1
        self._adaptive_enabled = bool(
            config.version
            in {
                "tabletop_online_intercept_planner_v3",
                "tabletop_online_intercept_planner_v4",
                "tabletop_online_intercept_planner_v5",
            }
        )
        self._moving_capture_enabled = bool(
            config.version
            in {
                "tabletop_online_intercept_planner_v4",
                "tabletop_online_intercept_planner_v5",
            }
        )
        self._mpc_capture_enabled = bool(
            config.version == "tabletop_online_intercept_planner_v5"
        )
        self._committed_template_index = -1
        self._committed_nominal_target_q = (
            contract.q_home_rad.astype(np.float64).copy()
        )
        self._committed_adaptive_phase = "approach"
        self._committed_preshape_index = -1
        self._committed_closure_step = -1
        self._committed_lift_step = -1
        self._committed_correction = np.zeros(7, dtype=np.float64)
        self._committed_target_q = contract.q_home_rad.astype(np.float64).copy()
        self._reference_grasp_q = np.asarray(
            sequence.recorded_franka_target_q_rad[
                config.closure_end_index
                if self._mpc_capture_enabled
                else config.template_catch_index
            ],
            dtype=np.float64,
        )
        self._reference_approach_q = np.asarray(
            sequence.recorded_franka_target_q_rad[
                config.template_catch_index - 1
                if self._mpc_capture_enabled
                else config.template_catch_index
            ],
            dtype=np.float64,
        )
        self._reference_lift_q = np.asarray(
            sequence.recorded_franka_target_q_rad[config.lift_end_index],
            dtype=np.float64,
        )
        self._reference_grasp_palm = panda_T_base_policy_palm(
            self._reference_grasp_q,
            contract.T_flange_policy_palm,
        )
        self._reference_object_from_palm_m = np.asarray(
            config.reference_intercept_center_base_m, dtype=np.float64
        ) - self._reference_grasp_palm[:3, 3]
        # V4 keeps the successful object's offset in the grasp-palm frame.
        # The earlier implementation stored a base-frame vector and then
        # added it to every measured palm position.  That silently moved the
        # capture window when the wrist rotated during a dynamic approach.
        if self._mpc_capture_enabled:
            self._reference_object_in_palm_m = np.asarray(
                config.capture_object_offset_palm_m, dtype=np.float64
            ).copy()
        else:
            self._reference_object_in_palm_m = (
                self._reference_grasp_palm[:3, :3].T
                @ self._reference_object_from_palm_m
            )
        self._committed_goal_q = self._reference_grasp_q.copy()
        self._capture_q: Optional[np.ndarray] = None
        self._lift_q: Optional[np.ndarray] = None
        self._committed_tabletop_height_correction_m: Optional[float] = None
        self._committed_live_fit_seen = False
        self._committed_full_occluded_closure = False
        self._committed_current_volume_capture = False
        self._committed_force_contact_capture = False
        self._last_committed_policy_sequence = 0
        self._last_rh56_feedback_sequence = 0
        self._last_rh56_feedback_forces_g: Optional[np.ndarray] = None
        self._pending: Optional[_OnlineProposal] = None
        self._replay_complete = False
        self._started_printed = False
        self._last_diagnostics: dict[str, object] = {
            "state": "startup_alignment",
            "online_replanning": True,
        }
        self._committed_diagnostics_trace: list[dict[str, object]] = []

    @property
    def replay_complete(self) -> bool:
        return bool(self._replay_complete)

    @property
    def completed_replay_frames(self) -> int:
        if self._adaptive_enabled:
            return max(0, self._committed_template_index + 1)
        return max(0, self._committed_index + 1)

    @property
    def repeated_target_commands(self) -> int:
        return 0

    @property
    def diagnostics_snapshot(self) -> Mapping[str, object]:
        snapshot = dict(self._last_diagnostics)
        snapshot["committed_trace"] = [
            dict(item) for item in self._committed_diagnostics_trace
        ]
        return snapshot

    def _centers_base(
        self,
        pointcloud: np.ndarray,
        valid: np.ndarray,
        proprio_history: np.ndarray,
    ) -> np.ndarray:
        points = np.asarray(pointcloud, dtype=np.float64)
        validity = np.asarray(valid, dtype=np.float64) >= 0.5
        proprio = np.asarray(proprio_history, dtype=np.float64)
        if proprio.ndim != 2 or proprio.shape[0] != points.shape[0] or (
            proprio.shape[1] < 7
        ):
            raise ValueError(
                "online tabletop planner requires pose-aligned proprio history"
            )
        centers = []
        for sample_points, sample_valid, sample_proprio in zip(
            points, validity, proprio
        ):
            selected = sample_points[sample_valid, :3]
            measured_q = sample_proprio[:7] + self.contract.q_home_rad
            if (
                selected.shape[0] < 16
                or not np.all(np.isfinite(selected))
                or not np.all(np.isfinite(measured_q))
            ):
                continue
            # Every point-cloud history entry is expressed in that entry's
            # capture-time policy-palm frame.  Applying the newest palm pose
            # to the whole history turns arm approach motion into fictitious
            # object velocity/height and can produce an unsafe intercept.
            transform = panda_T_base_policy_palm(
                measured_q, self.contract.T_flange_policy_palm
            )
            center_palm = np.mean(selected, axis=0)
            center_base = transform @ np.concatenate([center_palm, [1.0]])
            centers.append(center_base[:3])
        if not centers:
            raise ValueError(
                "online tabletop planner requires a fresh finite object center"
            )
        return np.asarray(centers, dtype=np.float64)

    def _predict_intercept(
        self,
        centers_base: np.ndarray,
        *,
        index: int,
        prediction_horizon_s: Optional[float] = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        config = self.config
        count = min(int(config.fit_sample_count), len(centers_base))
        # print("counting", config.fit_sample_count)
        # breakpoint()
        recent = centers_base[-count:]
        # A hand edge or a partial mask can move the surface centroid for one
        # frame even though the target itself is smooth.  Median filtering the
        # recent, current-palm-aligned centers prevents a single mask shape
        # change from becoming an arm correction.
        current = np.median(recent, axis=0)
        velocity = np.zeros(3, dtype=np.float64)
        residual = None
        speed = 0.0
        heading = None
        stable = False
        fit_quality = False
        fit_usable = False
        if count >= int(config.fit_sample_count):
            selected = recent
            times = np.arange(1 - count, 1, dtype=np.float64) * config.control_dt_s
            design = np.stack([times, np.ones_like(times)], axis=1)
            coefficients, _, _, _ = np.linalg.lstsq(design, selected, rcond=None)
            velocity = coefficients[0]
            fitted = design @ coefficients
            residual = float(
                np.sqrt(np.mean(np.sum((selected - fitted) ** 2, axis=1)))
            )
            speed = float(np.linalg.norm(velocity))
            axis = np.asarray(config.motion_axis_base, dtype=np.float64)
            heading = (
                float(np.dot(velocity, axis) / speed)
                if speed > 1.0e-9
                else 1.0
            )
            admitted_speed_max = (
                config.adaptive_speed_max_m_s
                if self._adaptive_enabled
                else config.speed_max_m_s
            )
            fit_quality = bool(
                speed <= admitted_speed_max + 1.0e-6
                and residual <= config.max_fit_residual_m
            )
            stable = bool(
                config.speed_min_m_s - 1.0e-6
                <= speed
                <= config.speed_max_m_s + 1.0e-6
                and (
                    heading >= config.heading_cos_min
                    or config.collision_replan
                )
                and residual <= config.max_fit_residual_m
            )
            # Once genuine live motion has been observed, deceleration or a
            # complete stop is still valid current-frame evidence.  Continue
            # following that measured center with its fitted (possibly zero)
            # velocity instead of freezing an earlier high-speed intercept or
            # substituting the scenario speed prior.  Fit quality and the
            # original workspace/depth gates remain mandatory.
            if self._adaptive_enabled:
                fit_usable = bool(
                    fit_quality
                    and (
                        heading >= config.heading_cos_min
                        or config.collision_replan
                    )
                    and (
                        speed >= 0.005
                        or self._committed_live_fit_seen
                    )
                )
            else:
                fit_usable = bool(
                    stable
                    or (
                        (
                            self._committed_live_fit_seen
                            or index >= config.template_catch_index
                        )
                        and fit_quality
                    )
                )
            if fit_usable:
                current = coefficients[1]
        remaining = max(
            0.0,
            (config.template_catch_index - index) * config.control_dt_s,
        )
        if prediction_horizon_s is not None:
            horizon = float(
                np.clip(
                    prediction_horizon_s,
                    0.0,
                    config.max_linear_prediction_s,
                )
            )
            horizon_source = "measured_arm_receding_horizon"
        elif index < config.template_catch_index:
            horizon = min(remaining, config.max_linear_prediction_s)
            horizon_source = "template_time_remaining"
        else:
            horizon = min(
                config.contact_prediction_latency_s,
                config.max_linear_prediction_s,
            )
            horizon_source = "contact_latency"
        axis = np.asarray(config.motion_axis_base, dtype=np.float64)
        nominal_velocity = axis * (
            0.5 * (config.speed_min_m_s + config.speed_max_m_s)
        )
        hold_last_goal = bool(
            not fit_usable
            and self._committed_live_fit_seen
            and index < config.template_catch_index
        )
        planning_velocity = velocity if fit_usable else nominal_velocity
        predicted = current + planning_velocity * horizon
        diagnostics: dict[str, object] = {
            "state": "tracking_and_replanning",
            "online_replanning": True,
            "template_index": int(index),
            "object_center_base_m": current.tolist(),
            "predicted_intercept_center_base_m": predicted.tolist(),
            "velocity_base_m_s": velocity.tolist(),
            "planning_velocity_base_m_s": planning_velocity.tolist(),
            "speed_m_s": speed,
            "heading_cos": heading,
            "fit_residual_m": residual,
            "stable_motion_fit": stable,
            "fit_quality": fit_quality,
            "admitted_speed_max_m_s": float(admitted_speed_max),
            "speed_outside_scenario_band": bool(
                speed < config.speed_min_m_s - 1.0e-6
                or speed > config.speed_max_m_s + 1.0e-6
            ),
            "velocity_source": "live_fit" if fit_usable else "scenario_prior",
            "hold_last_intercept_goal": hold_last_goal,
            "prediction_horizon_s": horizon,
            "prediction_horizon_source": horizon_source,
            "collision_replan": bool(config.collision_replan),
        }
        return predicted, diagnostics

    def _goal_q_for_intercept(
        self,
        predicted_center_base_m: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        desired = np.asarray(predicted_center_base_m, dtype=np.float64) - np.asarray(
            self.config.reference_intercept_center_base_m,
            dtype=np.float64,
        )
        horizontal_norm = float(np.linalg.norm(desired[:2]))
        if horizontal_norm > self.config.max_position_correction_xy_m + 1.0e-9:
            raise RuntimeError(
                "predicted object intercept is outside the commissioned XY workspace: "
                f"required={horizontal_norm:.4f}m "
                f"limit={self.config.max_position_correction_xy_m:.4f}m"
            )
        if abs(float(desired[2])) > self.config.max_position_correction_z_m + 1.0e-9:
            raise RuntimeError(
                "predicted object intercept is outside the commissioned Z workspace: "
                f"required={desired[2]:.4f}m "
                f"limit={self.config.max_position_correction_z_m:.4f}m"
            )
        # Re-solve from the last committed goal, while targeting the exact
        # translated reference-palm position.  This keeps the same IK branch
        # and makes steady 20 Hz replans one or two small iterations instead
        # of repeatedly solving a 0.2 m displacement from scratch.
        seed_palm = panda_T_base_policy_palm(
            self._committed_goal_q,
            self.contract.T_flange_policy_palm,
        )
        if self._mpc_capture_enabled:
            target_palm_position = (
                np.asarray(predicted_center_base_m, dtype=np.float64)
                - self._reference_grasp_palm[:3, :3]
                @ self._reference_object_in_palm_m
            )
            # The successful sphere replay leaves the rolling ball slightly
            # below the centre of the closing fingers on the real tabletop.
            # Apply only the bundle-sealed base-Z bias here; every resulting
            # arm/hand target still passes the ordinary fingertip and palm
            # clearance checks.  This is a Cartesian capture adjustment, not
            # a relaxation of the table or collision/reflex limits.
            target_palm_position[2] += self.config.grasp_vertical_offset_m
        else:
            target_palm_position = self._reference_grasp_palm[:3, 3] + desired
            target_palm_position[2] += (
                self.config.grasp_vertical_offset_m
                if self._moving_capture_enabled
                else 0.0
            )

        raw_goal = pose_preserving_joint_target(
            self._committed_goal_q,
            target_palm_position - seed_palm[:3, 3],
            T_flange_policy_palm=self.contract.T_flange_policy_palm,
            joint_limits_rad=self.contract.joint_limits_rad,
            joint_limit_margin_rad=self.config.joint_limit_margin_rad,
            damping=self.config.dls_damping,
            target_rotation_base=self._reference_grasp_palm[:3, :3],
            local_precision_refinement=True,
        )
        correction = raw_goal - self._reference_grasp_q
        maximum = float(np.max(np.abs(correction)))
        if maximum > self.config.max_joint_correction_rad + 1.0e-9:
            raise RuntimeError(
                "predicted object intercept exceeds the commissioned joint correction: "
                f"required={maximum:.4f}rad "
                f"limit={self.config.max_joint_correction_rad:.4f}rad"
            )
        return raw_goal, desired

    def _filtered_goal(self, raw_goal: np.ndarray) -> np.ndarray:
        delta = self.config.correction_filter_alpha * (
            np.asarray(raw_goal, dtype=np.float64) - self._committed_goal_q
        )
        delta = np.clip(
            delta,
            -self.config.max_joint_correction_step_rad,
            self.config.max_joint_correction_step_rad,
        )
        return self._committed_goal_q + delta

    def _mpc_prediction_horizon(
        self, measured_q_rad: np.ndarray
    ) -> tuple[float, float]:
        """Bound the next intercept horizon by measured arm time-to-go."""

        if not self._mpc_capture_enabled:
            raise RuntimeError("measured-arm MPC horizon is unavailable")
        measured = np.asarray(measured_q_rad, dtype=np.float64)
        if measured.shape != (7,) or not np.all(np.isfinite(measured)):
            raise ValueError("measured-arm MPC horizon requires finite q[7]")
        tracking_error = float(
            np.max(np.abs(self._committed_goal_q - measured))
        )
        horizon = float(
            np.clip(
                tracking_error / self.config.arm_tracking_speed_rad_s
                + self.config.contact_prediction_latency_s,
                self.config.minimum_intercept_horizon_s,
                self.config.max_linear_prediction_s,
            )
        )
        return horizon, tracking_error

    def _mpc_grasp_completion_horizon(
        self,
        *,
        phase: str,
        closure_step: int,
    ) -> float:
        """Predict where the object will be when the hand can actually grasp.

        The arm goal must not chase the object's position at the current
        camera tick while the RH56 is still part-way through its bounded
        closure.  Use the sealed remaining closure duration plus the existing
        observation/command latency, and recompute it from fresh motion every
        tick so a board collision or deceleration changes the intercept rather
        than leaving a fixed spatial lead behind.
        """

        if not self._mpc_capture_enabled:
            raise RuntimeError("grasp-completion horizon is unavailable")
        closure_intervals = (
            self.config.closure_end_index
            - self.config.template_catch_index
            + 1
        )
        if closure_intervals <= 0:
            raise RuntimeError("grasp-completion interval is invalid")
        if phase == "approach":
            remaining_intervals = closure_intervals
        elif phase == "closure":
            force_contact_first_tick = bool(
                int(closure_step) == -1
                and self._committed_force_contact_capture
                and self._committed_adaptive_phase == "closure"
                and self._committed_template_index
                == self.config.template_catch_index - 1
                and self._committed_closure_step == -1
            )
            if force_contact_first_tick:
                # Multi-finger contact is committed after the preceding open-
                # hand command has dual-ACKed.  ``-1`` is therefore a sealed
                # transition marker: the next policy command is closure step
                # zero and the full closure horizon remains.  No ordinary
                # closure state is permitted to use this sentinel.
                remaining_intervals = closure_intervals
            elif not 0 <= int(closure_step) < closure_intervals:
                raise RuntimeError("grasp-completion closure step is invalid")
            else:
                remaining_intervals = max(
                    0,
                    self.config.closure_end_index
                    - (self.config.template_catch_index + int(closure_step)),
                )
        else:
            raise RuntimeError(
                f"grasp-completion horizon is invalid in phase {phase!r}"
            )
        return float(
            remaining_intervals * self.config.control_dt_s
            + self.config.contact_prediction_latency_s
        )

    def _joint_correction(
        self,
        nominal_q: np.ndarray,
        desired_translation: np.ndarray,
    ) -> np.ndarray:
        config = self.config
        correction = cartesian_joint_correction(
            nominal_q,
            desired_translation,
            T_flange_policy_palm=self.contract.T_flange_policy_palm,
            damping=config.dls_damping,
        )
        return np.clip(
            correction,
            -config.max_joint_correction_rad,
            config.max_joint_correction_rad,
        )

    def _safe_target(
        self,
        *,
        nominal_q: np.ndarray,
        target_q: np.ndarray,
        hand_target: np.ndarray,
    ) -> dict[str, float]:
        config = self.config
        lower = self.contract.joint_limits_rad[:, 0] + config.joint_limit_margin_rad
        upper = self.contract.joint_limits_rad[:, 1] - config.joint_limit_margin_rad
        if np.any(target_q < lower) or np.any(target_q > upper):
            raise RuntimeError("online planner target violates joint-limit margin")
        target_step = float(
            np.max(np.abs(target_q - self._committed_target_q))
        )
        if target_step > config.max_target_step_rad + 1.0e-9:
            raise RuntimeError("online planner target exceeds target-step envelope")
        nominal_palm = panda_T_base_policy_palm(
            nominal_q, self.contract.T_flange_policy_palm
        )
        target_palm = panda_T_base_policy_palm(
            target_q, self.contract.T_flange_policy_palm
        )
        orientation = float(
            np.linalg.norm(
                _rotation_vector(
                    target_palm[:3, :3] @ nominal_palm[:3, :3].T
                )
            )
        )
        if orientation > config.max_orientation_correction_rad:
            raise _OrientationCorrectionConflict(
                required_rad=orientation,
                limit_rad=config.max_orientation_correction_rad,
            )
        plane = np.asarray(config.tabletop_plane_base, dtype=np.float64)
        fingertips = self._fingertips.positions_base(
            angle_act_register_order=hand_target,
            T_base_palm=target_palm,
        )
        tip_clearance = float(np.min(fingertips @ plane[:3] + plane[3]))
        palm_clearance = float(np.dot(target_palm[:3, 3], plane[:3]) + plane[3])
        if tip_clearance < config.minimum_fingertip_clearance_m:
            raise _FingertipTableClearanceConflict(
                required_m=tip_clearance,
                limit_m=config.minimum_fingertip_clearance_m,
            )
        if palm_clearance < 0.12:
            raise RuntimeError("online planner target violates palm/table clearance")
        return {
            "orientation_correction_rad": orientation,
            "target_step_rad": target_step,
            "minimum_fingertip_clearance_m": tip_clearance,
            "palm_clearance_m": palm_clearance,
        }

    def _safe_closure_target(
        self,
        *,
        nominal_q: np.ndarray,
        target_q: np.ndarray,
        hand_target: np.ndarray,
        actual_fingertips_palm_m: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Validate both commanded and physically observed closure geometry.

        RH56 target application is dual-ACKed, but ANGLE_ACT can lag the
        streamed target by several 20 Hz ticks.  Evaluating only the next,
        more-flexed target hand can therefore approve a descending wrist
        while the real fingers are still extended.  Fresh policy proprio
        contains the five measured fingertip positions in the palm frame, so
        use that physical hand shape at the proposed arm pose.  The immutable
        open preshape is retained only for the bounded occluded-closure path,
        where no fresh observation is available.
        """

        commanded = self._safe_target(
            nominal_q=nominal_q,
            target_q=target_q,
            hand_target=hand_target,
        )
        if actual_fingertips_palm_m is None:
            open_index = int(self.config.template_catch_index) - 1
            if not 0 <= open_index < self.sequence.action_count:
                raise RuntimeError("closure transition has no open-hand envelope")
            open_hand = np.asarray(
                self.sequence.recorded_rh56_angle_set_register_order[open_index],
                dtype=np.int32,
            )
            transition = self._safe_target(
                nominal_q=nominal_q,
                target_q=target_q,
                hand_target=open_hand,
            )
            transition_clearance = float(
                transition["minimum_fingertip_clearance_m"]
            )
        else:
            local_tips = np.asarray(
                actual_fingertips_palm_m, dtype=np.float64
            )
            if local_tips.shape != (5, 3) or not np.all(
                np.isfinite(local_tips)
            ):
                raise RuntimeError(
                    "closure transition requires five finite measured fingertips"
                )
            target_palm = panda_T_base_policy_palm(
                target_q, self.contract.T_flange_policy_palm
            )
            target_tips = (
                local_tips @ target_palm[:3, :3].T
                + target_palm[:3, 3]
            )
            plane = np.asarray(
                self.config.tabletop_plane_base, dtype=np.float64
            )
            transition_clearance = float(
                np.min(target_tips @ plane[:3] + plane[3])
            )
            if transition_clearance < self.config.minimum_fingertip_clearance_m:
                raise _FingertipTableClearanceConflict(
                    required_m=transition_clearance,
                    limit_m=self.config.minimum_fingertip_clearance_m,
                )
        result = dict(commanded)
        result["commanded_hand_fingertip_clearance_m"] = float(
            commanded["minimum_fingertip_clearance_m"]
        )
        result["closure_transition_fingertip_clearance_m"] = float(
            transition_clearance
        )
        result["minimum_fingertip_clearance_m"] = min(
            result["commanded_hand_fingertip_clearance_m"],
            result["closure_transition_fingertip_clearance_m"],
        )
        return result

    @staticmethod
    def _slew_joint_target(
        start_q_rad: np.ndarray,
        goal_q_rad: np.ndarray,
        maximum_step_rad: float,
    ) -> np.ndarray:
        start = np.asarray(start_q_rad, dtype=np.float64)
        goal = np.asarray(goal_q_rad, dtype=np.float64)
        step = float(maximum_step_rad)
        if start.shape != (7,) or goal.shape != (7,) or not (
            np.all(np.isfinite(start)) and np.all(np.isfinite(goal))
        ):
            raise ValueError("adaptive tabletop joint targets must be finite [7]")
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("adaptive tabletop joint slew must be positive")
        return start + np.clip(goal - start, -step, step)

    def _adaptive_goal_for_intercept(
        self,
        predicted_center_base_m: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        """Return the furthest sealed, joint-reachable planar correction.

        A distant object is evidence about a future interception trajectory,
        not permission to command the hand to that instantaneous position.
        The v2 planner projected only in Cartesian space and could then fail
        at the joint-space boundary.  V3 additionally backs the planar goal
        toward the calibrated intercept until it retains an explicit reserve
        inside the already commissioned joint-correction envelope.
        """

        reference = np.asarray(
            self.config.reference_intercept_center_base_m, dtype=np.float64
        )
        requested = np.asarray(predicted_center_base_m, dtype=np.float64)
        if requested.shape != (3,) or not np.all(np.isfinite(requested)):
            raise RuntimeError("adaptive predicted intercept is not finite")
        delta = requested - reference
        if abs(float(delta[2])) > self.config.max_position_correction_z_m + 1.0e-9:
            raise RuntimeError(
                "predicted object intercept is outside the commissioned Z workspace: "
                f"required={delta[2]:.4f}m "
                f"limit={self.config.max_position_correction_z_m:.4f}m"
            )
        requested_xy_norm = float(np.linalg.norm(delta[:2]))
        workspace_scale = 1.0
        if requested_xy_norm > self.config.max_position_correction_xy_m:
            workspace_scale = (
                self.config.max_position_correction_xy_m / requested_xy_norm
            )
        bounded_xy = delta[:2] * workspace_scale
        correction_limit = (
            self.config.max_joint_correction_rad
            - self.config.joint_correction_reserve_rad
        )
        if correction_limit <= 0.0:
            raise RuntimeError("adaptive joint-correction reserve is invalid")

        def solve(planar_scale: float):
            candidate = reference.copy()
            candidate[:2] += bounded_xy * float(planar_scale)
            candidate[2] += delta[2]
            try:
                goal, desired = self._goal_q_for_intercept(candidate)
            except RuntimeError:
                return None
            maximum = float(np.max(np.abs(goal - self._reference_grasp_q)))
            if maximum > correction_limit + 1.0e-9:
                return None
            return goal, desired, maximum

        selected = None
        selected_scale = 0.0
        # This is a 20 Hz control path, so do not optimize continuously along
        # the workspace boundary with dozens of repeated numerical IK solves.
        # Large collision replans start at the audited 75% workspace choice.
        # Trying full, then 75%, then 50% repeated the numerical IK solve in
        # one 20 Hz tick and could age the Franka state past 50 ms.  A dense
        # boundary audit shows 75% remains inside the unchanged joint reserve;
        # the exact check below is still authoritative and 50% is fail-safe.
        if (
            self._moving_capture_enabled
            and requested_xy_norm
            > 0.45 * self.config.max_position_correction_xy_m
        ):
            candidate_scales = (
                (0.75, 0.5)
                if self.config.collision_replan
                else (1.0, 0.5)
            )
        else:
            candidate_scales = (1.0, 0.5, 0.0)

        for candidate_scale in candidate_scales:
            candidate = solve(candidate_scale)
            if candidate is not None:
                selected = candidate
                selected_scale = candidate_scale
                break
        if selected is None:
            raise RuntimeError(
                "calibrated tabletop intercept is outside the reserved "
                "joint-correction envelope"
            )
        goal, desired, maximum = selected
        return goal, desired, {
            "requested_intercept_center_base_m": requested.tolist(),
            "workspace_xy_projection_scale": float(workspace_scale),
            "joint_reachability_projection_scale": float(selected_scale),
            "reserved_joint_correction_max_rad": float(maximum),
            "reserved_joint_correction_limit_rad": float(correction_limit),
        }

    def _adaptive_capture_gate(
        self,
        *,
        diagnostics: Mapping[str, object],
        measured_q_rad: np.ndarray,
        goal_q_rad: np.ndarray,
        hand_target: np.ndarray,
        preshape_ready: bool,
        arm_projection_horizon_s: Optional[float] = None,
    ) -> tuple[bool, dict[str, object]]:
        current = np.asarray(
            diagnostics.get("object_center_base_m"), dtype=np.float64
        )
        velocity = np.asarray(
            diagnostics.get("velocity_base_m_s"), dtype=np.float64
        )
        if current.shape != (3,) or velocity.shape != (3,) or not (
            np.all(np.isfinite(current)) and np.all(np.isfinite(velocity))
        ):
            raise RuntimeError("adaptive capture evidence is malformed")
        measured_q = np.asarray(measured_q_rad, dtype=np.float64)
        measured_palm = panda_T_base_policy_palm(
            measured_q, self.contract.T_flange_policy_palm
        )
        capture_q = measured_q
        projected_arm_horizon = 0.0
        if arm_projection_horizon_s is not None:
            projected_arm_horizon = float(arm_projection_horizon_s)
            if (
                not self._mpc_capture_enabled
                or not np.isfinite(projected_arm_horizon)
                or projected_arm_horizon < 0.0
                or projected_arm_horizon
                > self.config.max_linear_prediction_s + 1.0e-9
            ):
                raise RuntimeError("capture arm-projection horizon is invalid")
            reachable_step = (
                self.config.arm_tracking_speed_rad_s
                * projected_arm_horizon
            )
            capture_q = measured_q + np.clip(
                np.asarray(goal_q_rad, dtype=np.float64) - measured_q,
                -reachable_step,
                reachable_step,
            )
        capture_palm = panda_T_base_policy_palm(
            capture_q, self.contract.T_flange_policy_palm
        )
        if self._moving_capture_enabled:
            capture_center = (
                capture_palm[:3, 3]
                + capture_palm[:3, :3] @ self._reference_object_in_palm_m
            )
            # The grasp palm is deliberately commanded below the observed
            # object center.  Compensate that sealed base-Z offset when
            # evaluating object/hand alignment instead of consuming almost
            # the entire capture Z tolerance with the intended grasp bias.
            capture_center = np.asarray(capture_center, dtype=np.float64).copy()
            if not self._mpc_capture_enabled:
                capture_center[2] -= self.config.grasp_vertical_offset_m
            elif arm_projection_horizon_s is None:
                # The v5 sphere approach deliberately keeps the open fingers
                # above the table.  Evaluate the final contact volume at the
                # same measured XY/orientation while removing that sealed
                # approach-only lift; closure then descends along the audited
                # nominal arm/hand path.
                capture_center[2] -= self.config.open_approach_lift_m
            # With an explicit completion-horizon projection, ``capture_q``
            # already represents the future grasp pose; subtracting the open
            # approach lift again would double-count the descent.
        else:
            capture_center = (
                measured_palm[:3, 3] + self._reference_object_from_palm_m
            )
        measured_capture_center = (
            measured_palm[:3, 3]
            + measured_palm[:3, :3] @ self._reference_object_in_palm_m
        )
        measured_capture_center = np.asarray(
            measured_capture_center, dtype=np.float64
        ).copy()
        if self._mpc_capture_enabled:
            # This is the current physical open-hand capture volume, not the
            # future arm projection above.  Remove only the sealed approach
            # lift so its center stays at the demonstrated contact volume.
            measured_capture_center[2] -= self.config.open_approach_lift_m
        current_error = capture_center - current
        configured_axis = np.asarray(
            self.config.motion_axis_base, dtype=np.float64
        )
        speed = float(np.linalg.norm(velocity))
        planar_velocity = np.asarray(velocity, dtype=np.float64).copy()
        planar_velocity[2] = 0.0
        planar_speed = float(np.linalg.norm(planar_velocity))
        if self.config.collision_replan and planar_speed >= 0.005:
            travel_axis = planar_velocity / planar_speed
        else:
            travel_axis = configured_axis
        current_along = float(np.dot(current_error, travel_axis))
        current_lateral_vector = current_error - current_along * travel_axis
        current_lateral = float(np.linalg.norm(current_lateral_vector[:2]))
        current_z_error = float(current_error[2])
        measured_error = measured_capture_center - current
        measured_along = float(np.dot(measured_error, travel_axis))
        measured_lateral_vector = measured_error - measured_along * travel_axis
        measured_lateral = float(np.linalg.norm(measured_lateral_vector[:2]))
        measured_z_error = float(measured_error[2])
        speed_along = float(np.dot(velocity, travel_axis))
        temporal_object_center = current.copy()
        temporal_alignment_used = bool(
            self._mpc_capture_enabled
            and arm_projection_horizon_s is not None
            and speed_along >= 0.005
        )
        if temporal_alignment_used:
            # Compare the projected hand and the rolling object at the same
            # future instant.  The previous gate compared this future hand
            # pose with the object's current center, allowing closure while
            # the arm was still hundreds of milliseconds from the intercept.
            # Keep Z at the fresh observed tabletop height; raw fitted Z
            # velocity is centroid/depth noise for a rolling object.
            temporal_object_center[:2] += (
                velocity[:2] * projected_arm_horizon
            )
        error = capture_center - temporal_object_center
        along = float(np.dot(error, travel_axis))
        lateral_vector = error - along * travel_axis
        lateral = float(np.linalg.norm(lateral_vector[:2]))
        z_error = float(error[2])
        arm_error = float(np.max(np.abs(goal_q_rad - measured_q)))
        arm_ready = bool(arm_error <= self.config.capture_arm_error_rad)
        plane = np.asarray(self.config.tabletop_plane_base, dtype=np.float64)
        current_fingertips = self._fingertips.positions_base(
            angle_act_register_order=np.asarray(hand_target, dtype=np.int32),
            T_base_palm=measured_palm,
        )
        current_tip_clearance = float(
            np.min(current_fingertips @ plane[:3] + plane[3])
        )
        current_palm_clearance = float(
            np.dot(measured_palm[:3, 3], plane[:3]) + plane[3]
        )
        current_geometry_safe = bool(
            current_tip_clearance
            >= self.config.minimum_fingertip_clearance_m
            and current_palm_clearance >= 0.12
        )
        live_fit = bool(diagnostics.get("velocity_source") == "live_fit")
        fit_residual_value = diagnostics.get("fit_residual_m")
        heading_value = diagnostics.get("heading_cos")
        fit_residual = (
            float(fit_residual_value)
            if fit_residual_value is not None
            else float("nan")
        )
        heading = (
            float(heading_value)
            if heading_value is not None
            else float("nan")
        )
        degraded_live_fit_continuation = bool(
            self._mpc_capture_enabled
            and self.config.collision_replan
            and self._committed_live_fit_seen
            and diagnostics.get("velocity_source") == "scenario_prior"
            and diagnostics.get("hold_last_intercept_goal") is True
            and np.isfinite(fit_residual)
            and fit_residual
            <= _DEGRADED_LIVE_FIT_CAPTURE_MAX_RESIDUAL_M + 1.0e-12
            and np.isfinite(heading)
            and heading
            >= max(
                float(self.config.heading_cos_min),
                _DEGRADED_LIVE_FIT_CAPTURE_MIN_HEADING_COS,
            )
            and speed >= 0.005
            and speed
            <= float(self.config.adaptive_speed_max_m_s) + 1.0e-6
        )
        moving_fit_evidence = bool(live_fit or degraded_live_fit_continuation)
        stationary_capture = bool(
            self._committed_live_fit_seen
            and speed < 0.02
            and abs(current_along) <= self.config.closure_past_tolerance_m
        )
        temporal_along_tolerance = float(
            max(
                self.config.closure_past_tolerance_m,
                abs(speed_along)
                * self.config.contact_prediction_latency_s,
            )
        )
        if temporal_alignment_used:
            moving_capture = bool(
                moving_fit_evidence
                and abs(along) <= temporal_along_tolerance
            )
        else:
            moving_capture = bool(
                moving_fit_evidence
                and speed_along >= 0.005
                and current_along
                >= -self.config.closure_past_tolerance_m
                and current_along
                <= speed_along * self.config.closure_trigger_ttc_s
            )
        cross_track_limit = float(self.config.closure_cross_track_m)
        if degraded_live_fit_continuation:
            cross_track_limit += (
                _DEGRADED_LIVE_FIT_CAPTURE_CROSS_TRACK_SLACK_M
            )
        time_to_capture = (
            current_along / speed_along if speed_along >= 0.005 else None
        )
        # A rolling object can already be physically inside the open RH56
        # while completion-horizon extrapolation says it will have left by
        # the time the full demonstrated closure finishes.  If a fresh live
        # fit places it inside this commissioned current-hand volume, closing
        # now is safer than waiting until the hand self-occludes.  This path
        # remains collision-task, speed, geometry and table-clearance bound;
        # it never admits a stale point cloud or scenario-prior motion.
        current_volume_capture = bool(
            self._mpc_capture_enabled
            and self.config.collision_replan
            and temporal_alignment_used
            and live_fit
            and speed >= max(0.005, float(self.config.speed_min_m_s))
            and speed
            <= float(self.config.adaptive_speed_max_m_s) + 1.0e-6
            and abs(measured_along) <= cross_track_limit
            and measured_lateral <= cross_track_limit
            and abs(measured_z_error) <= self.config.closure_z_tolerance_m
        )
        accepted = bool(
            preshape_ready
            # A moving object may require closure before the arm reaches its
            # final predicted goal.  A stationary object has no such deadline
            # and must still wait for real arm arrival; otherwise a projected
            # future palm pose could close in mid-air.
            and (
                (
                    self._moving_capture_enabled
                    and (moving_capture or current_volume_capture)
                )
                or arm_ready
            )
            and current_geometry_safe
            and (
                current_volume_capture
                or (
                    lateral <= cross_track_limit
                    and abs(z_error) <= self.config.closure_z_tolerance_m
                    and (moving_capture or stationary_capture)
                )
            )
        )
        return accepted, {
            "capture_center_base_m": capture_center.tolist(),
            "capture_error_base_m": error.tolist(),
            "capture_along_track_m": along,
            "capture_cross_track_m": lateral,
            "capture_cross_track_limit_m": cross_track_limit,
            "capture_z_error_m": z_error,
            "capture_speed_along_m_s": speed_along,
            "capture_time_to_contact_s": time_to_capture,
            "capture_current_error_base_m": current_error.tolist(),
            "capture_current_along_track_m": current_along,
            "capture_current_cross_track_m": current_lateral,
            "capture_current_z_error_m": current_z_error,
            "capture_measured_center_base_m": measured_capture_center.tolist(),
            "capture_measured_error_base_m": measured_error.tolist(),
            "capture_measured_along_track_m": measured_along,
            "capture_measured_cross_track_m": measured_lateral,
            "capture_measured_z_error_m": measured_z_error,
            "capture_current_volume_committed": bool(
                accepted and current_volume_capture
            ),
            "capture_temporal_alignment_used": temporal_alignment_used,
            "capture_temporal_object_center_base_m": (
                temporal_object_center.tolist()
            ),
            "capture_temporal_along_tolerance_m": temporal_along_tolerance,
            "capture_arm_error_rad": arm_error,
            "capture_arm_ready": arm_ready,
            "capture_arm_projection_horizon_s": projected_arm_horizon,
            "capture_projected_arm_delta_rad": (
                capture_q - measured_q
            ).tolist(),
            "capture_uses_projected_arm_pose": bool(
                arm_projection_horizon_s is not None
            ),
            "capture_requires_final_goal_arrival": bool(
                not self._moving_capture_enabled
            ),
            "capture_current_geometry_safe": current_geometry_safe,
            "capture_current_fingertip_clearance_m": current_tip_clearance,
            "capture_current_palm_clearance_m": current_palm_clearance,
            "capture_grasp_vertical_offset_m": float(
                self.config.grasp_vertical_offset_m
            ),
            "capture_preshape_ready": bool(preshape_ready),
            "capture_degraded_live_fit_continuation": (
                degraded_live_fit_continuation
            ),
            "capture_degraded_live_fit_max_residual_m": (
                _DEGRADED_LIVE_FIT_CAPTURE_MAX_RESIDUAL_M
            ),
            "capture_temporal_alignment_committed": bool(
                accepted and temporal_alignment_used and moving_capture
            ),
            "capture_gate_accepted": accepted,
        }

    def _adaptive_approach_target(
        self,
        *,
        nominal_target_q_rad: np.ndarray,
        goal_q_rad: np.ndarray,
    ) -> np.ndarray:
        """Apply the live correction around the demonstrated arm path.

        Interpolating q_home toward two separate IK goals can introduce a
        transient wrist rotation even when both final goals share the exact
        same orientation.  V3 keeps the demonstrated approach as the nominal
        path and slews only its already commissioned Cartesian correction.
        """

        nominal = np.asarray(nominal_target_q_rad, dtype=np.float64)
        desired_correction = (
            np.asarray(goal_q_rad, dtype=np.float64) - self._reference_grasp_q
        )
        correction = self._committed_correction + np.clip(
            desired_correction - self._committed_correction,
            -self.config.max_joint_correction_step_rad,
            self.config.max_joint_correction_step_rad,
        )
        candidate = nominal + correction
        # Nominal and correction increments can have the same sign.  Preserve
        # the public per-tick target envelope after composing them.
        return self._committed_target_q + np.clip(
            candidate - self._committed_target_q,
            -self.config.max_target_step_rad,
            self.config.max_target_step_rad,
        )

    def _adaptive_action_for_sequence(
        self,
        sequence: int,
        pointcloud: np.ndarray,
        valid: np.ndarray,
        proprio: np.ndarray,
    ) -> ReplayPolicyOutput:
        runtime_index = int(sequence) - 1
        if self._pending is not None:
            if self._pending.sequence != int(sequence):
                raise RuntimeError("online planner already has another proposal")
            proposal = self._pending
            template_index = int(proposal.template_index)
            return ReplayPolicyOutput(
                action13=self.sequence.actions13[template_index].copy(),
                exact_franka_target_q_rad=proposal.target_q_rad.copy(),
                exact_rh56_angle_set_register_order=(
                    self.sequence.recorded_rh56_angle_set_register_order[
                        template_index
                    ].copy()
                ),
                bypass_rh56_host_slew=True,
                replay_frame_index=template_index,
            )

        config = self.config
        phase = self._committed_adaptive_phase
        preshape_index = self._committed_preshape_index
        closure_step = self._committed_closure_step
        lift_step = self._committed_lift_step
        nominal_target = self._committed_nominal_target_q.copy()
        target_q = self._committed_target_q.copy()
        goal_q = self._committed_goal_q.copy()
        capture_q = None if self._capture_q is None else self._capture_q.copy()
        lift_q = None if self._lift_q is None else self._lift_q.copy()
        tabletop_height_correction_m = (
            self._committed_tabletop_height_correction_m
        )
        live_fit_seen = self._committed_live_fit_seen
        full_occluded_closure_committed = (
            self._committed_full_occluded_closure
        )
        current_volume_capture_committed = (
            self._committed_current_volume_capture
        )
        diagnostics: dict[str, object]

        if phase in {"approach", "closure"}:
            centers = self._centers_base(pointcloud, valid, proprio)
            measured_q = (
                np.asarray(proprio[-1, :7], dtype=np.float64)
                + self.contract.q_home_rad
            )
            mpc_horizon = None
            grasp_completion_horizon = None
            if self._mpc_capture_enabled:
                arm_horizon, tracking_error = self._mpc_prediction_horizon(
                    measured_q
                )
                grasp_completion_horizon = (
                    self._mpc_grasp_completion_horizon(
                        phase=phase,
                        closure_step=closure_step,
                    )
                )
                # Do not move the open hand farther down-track merely because
                # closure has not started yet: the existing spatial/TTC gate
                # is defined against the measured open-hand capture volume.
                # Once that gate commits closure, lead by the remaining hand
                # actuation time so the final grasp, rather than the current
                # palm target, meets the moving object.
                effective_grasp_horizon = (
                    grasp_completion_horizon
                    if phase == "closure"
                    else 0.0
                )
                mpc_horizon = min(
                    config.max_linear_prediction_s,
                    max(arm_horizon, effective_grasp_horizon),
                )
            if self._moving_capture_enabled:
                # V4 is a receding-horizon controller.  RH56 preshape index
                # is not elapsed arm time and must not shorten prediction as
                # the fingers rotate into their open catch shape.  Lead by
                # the fixed bounded horizon throughout approach, then switch
                # to the configured one-tick contact latency during closure.
                fit_index = (
                    0
                    if phase == "approach"
                    else config.template_catch_index
                )
            else:
                fit_index = max(
                    0, min(preshape_index, config.template_catch_index)
                )
            predicted, diagnostics = self._predict_intercept(
                centers,
                index=fit_index,
                prediction_horizon_s=mpc_horizon,
            )
            if self._mpc_capture_enabled:
                diagnostics["arm_tracking_error_rad"] = tracking_error
                diagnostics["arm_tracking_speed_bound_rad_s"] = float(
                    self.config.arm_tracking_speed_rad_s
                )
                diagnostics["arm_prediction_horizon_s"] = float(arm_horizon)
                diagnostics["grasp_completion_horizon_s"] = float(
                    grasp_completion_horizon
                )
                diagnostics["grasp_completion_horizon_applied_s"] = float(
                    effective_grasp_horizon
                )
            reference_height = float(
                config.reference_intercept_center_base_m[2]
            )
            observed_height_correction = float(
                diagnostics["object_center_base_m"][2]
            ) - reference_height
            if tabletop_height_correction_m is None:
                if (
                    self._moving_capture_enabled
                    and abs(observed_height_correction)
                    > config.max_position_correction_z_m + 1.0e-9
                ):
                    # Start the arm immediately when ramp motion is detected,
                    # but never chase the ramp's Z height.  The nominal
                    # approach remains table-clear and the current-object
                    # capture gate cannot close until measured Z enters the
                    # commissioned corridor.
                    tabletop_height_correction_m = 0.0
                    height_source = "calibrated_table_height_during_entry"
                    diagnostics["entry_height_outside_capture_corridor"] = True
                else:
                    tabletop_height_correction_m = observed_height_correction
                    height_source = "first_committed_planar_observation"
            else:
                height_source = "locked_tabletop_episode_height"
            planar_predicted = np.asarray(predicted, dtype=np.float64).copy()
            planar_predicted[2] = (
                reference_height + tabletop_height_correction_m
            )
            diagnostics["observed_height_correction_m"] = (
                observed_height_correction
            )
            diagnostics["tabletop_height_correction_m"] = float(
                tabletop_height_correction_m
            )
            diagnostics["tabletop_height_source"] = height_source
            if (
                diagnostics.get("velocity_source") != "live_fit"
                and not live_fit_seen
            ):
                # A five-sample velocity fit is not available on the first
                # four ticks.  Begin the already audited nominal approach
                # immediately, but do not spend hundreds of milliseconds
                # solving a far, prior-only extrapolation which will be
                # replaced as soon as exact live velocity exists.
                nominal_intercept = np.asarray(
                    config.reference_intercept_center_base_m,
                    dtype=np.float64,
                ).copy()
                nominal_intercept[2] += tabletop_height_correction_m
                raw_goal, desired, reachability = (
                    self._adaptive_goal_for_intercept(nominal_intercept)
                )
                reachability["goal_source"] = (
                    "nominal_xy_until_first_live_motion_fit"
                )
            elif diagnostics["hold_last_intercept_goal"]:
                raw_goal = self._committed_goal_q.copy()
                committed_palm = panda_T_base_policy_palm(
                    raw_goal, self.contract.T_flange_policy_palm
                )
                desired = (
                    committed_palm[:3, 3]
                    - self._reference_grasp_palm[:3, 3]
                )
                reachability = {
                    "joint_reachability_projection_scale": 1.0,
                    "goal_source": "held_last_live_intercept",
                }
            else:
                raw_goal, desired, reachability = (
                    self._adaptive_goal_for_intercept(planar_predicted)
                )
                reachability["goal_source"] = "fresh_reachable_intercept"
            diagnostics.update(reachability)
            goal_q = self._filtered_goal(raw_goal)
            diagnostics["desired_position_correction_base_m"] = desired.tolist()
            diagnostics["raw_intercept_goal_q_rad"] = raw_goal.tolist()
            diagnostics["filtered_intercept_goal_q_rad"] = goal_q.tolist()
            live_fit_seen = bool(
                live_fit_seen or diagnostics.get("velocity_source") == "live_fit"
            )

        if phase == "approach":
            if preshape_index < 0:
                preshape_index = 0
            else:
                preshape_index = min(
                    config.template_catch_index - 1,
                    preshape_index + config.preshape_phase_step,
                )
            nominal_target = self._slew_joint_target(
                self._committed_nominal_target_q,
                (
                    self._reference_approach_q
                    if self._mpc_capture_enabled
                    else self._reference_grasp_q
                ),
                config.approach_target_step_rad,
            )
            target_q = self._adaptive_approach_target(
                nominal_target_q_rad=nominal_target,
                goal_q_rad=goal_q,
            )
            capture_hand_target = np.asarray(
                self.sequence.recorded_rh56_angle_set_register_order[
                    max(0, preshape_index)
                ],
                dtype=np.int32,
            )
            capture, capture_diagnostics = self._adaptive_capture_gate(
                diagnostics=diagnostics,
                measured_q_rad=measured_q,
                goal_q_rad=goal_q,
                hand_target=capture_hand_target,
                preshape_ready=bool(
                    preshape_index >= config.template_catch_index - 1
                ),
                arm_projection_horizon_s=(
                    grasp_completion_horizon
                    if self._mpc_capture_enabled
                    else None
                ),
            )
            diagnostics.update(capture_diagnostics)
            closure_count = config.closure_end_index - config.template_catch_index + 1
            lift_count = config.lift_end_index - config.closure_end_index
            last_start_runtime_index = self.selected_steps - closure_count - lift_count
            if capture:
                phase = "closure"
                closure_step = 0
                template_index = config.template_catch_index
                full_occluded_closure_committed = bool(
                    capture_diagnostics.get(
                        "capture_temporal_alignment_committed", False
                    )
                    or capture_diagnostics.get(
                        "capture_current_volume_committed", False
                    )
                )
                current_volume_capture_committed = bool(
                    capture_diagnostics.get(
                        "capture_current_volume_committed", False
                    )
                )
                diagnostics["state"] = "adaptive_visual_servo_closure"
            else:
                template_index = preshape_index
                diagnostics["state"] = "adaptive_predictive_approach"
                if runtime_index >= last_start_runtime_index:
                    raise RuntimeError(
                        "adaptive tabletop capture opportunity was not established "
                        "before the bounded lift deadline: "
                        f"arm_error={capture_diagnostics['capture_arm_error_rad']:.4f}rad "
                        f"along={capture_diagnostics['capture_along_track_m']:.4f}m "
                        f"cross={capture_diagnostics['capture_cross_track_m']:.4f}m "
                        f"z={capture_diagnostics['capture_z_error_m']:.4f}m"
                    )
                if (
                    not self._moving_capture_enabled
                    and not config.collision_replan
                    and diagnostics.get("velocity_source") == "live_fit"
                    and capture_diagnostics["capture_preshape_ready"]
                    and capture_diagnostics["capture_cross_track_m"]
                    <= config.closure_cross_track_m
                    and abs(capture_diagnostics["capture_z_error_m"])
                    <= config.closure_z_tolerance_m
                    and capture_diagnostics["capture_along_track_m"]
                    < -config.closure_past_tolerance_m
                ):
                    raise RuntimeError(
                        "object passed the adaptive capture corridor before the "
                        "Franka/RH56 ready state: "
                        f"arm_error={capture_diagnostics['capture_arm_error_rad']:.4f}rad "
                        f"along={capture_diagnostics['capture_along_track_m']:.4f}m "
                        f"cross={capture_diagnostics['capture_cross_track_m']:.4f}m "
                        f"z={capture_diagnostics['capture_z_error_m']:.4f}m"
                    )
        elif phase == "closure":
            next_step = closure_step + 1
            if next_step >= (
                config.closure_end_index - config.template_catch_index + 1
            ):
                raise RuntimeError("adaptive closure phase advanced out of bounds")
            closure_step = next_step
            template_index = config.template_catch_index + closure_step
            if current_volume_capture_committed:
                # The fresh object center was already inside the measured
                # open-hand volume when closure committed.  Keep that exact
                # arm pose while the fingers finish wrapping instead of
                # spending table clearance on the nominal descent.
                nominal_target = self._committed_nominal_target_q.copy()
                target_q = self._committed_target_q.copy()
                diagnostics["current_volume_closure_arm_held"] = True
            elif self._mpc_capture_enabled:
                recorded_nominal_target = np.asarray(
                    self.sequence.recorded_franka_target_q_rad[template_index],
                    dtype=np.float64,
                )
                closure_nominal_slew_limit = min(
                    config.max_target_step_rad,
                    config.arm_tracking_speed_rad_s * config.control_dt_s,
                )
                nominal_target = self._slew_joint_target(
                    self._committed_nominal_target_q,
                    recorded_nominal_target,
                    closure_nominal_slew_limit,
                )
                diagnostics["closure_nominal_slew_limit_rad"] = float(
                    closure_nominal_slew_limit
                )
                diagnostics["closure_nominal_slew_applied"] = bool(
                    not np.array_equal(
                        nominal_target, recorded_nominal_target
                    )
                )
            else:
                nominal_target = self._slew_joint_target(
                    self._committed_nominal_target_q,
                    self._reference_grasp_q,
                    config.approach_target_step_rad,
                )
            target_q = self._adaptive_approach_target(
                nominal_target_q_rad=nominal_target,
                goal_q_rad=goal_q,
            )
            diagnostics["state"] = "adaptive_visual_servo_closure"
            if template_index == config.closure_end_index:
                phase = "lift"
                diagnostics["capture_target_committed_on_success"] = True
        elif phase == "lift":
            if capture_q is None or lift_q is None:
                raise RuntimeError("adaptive lift has no committed capture target")
            lift_step += 1
            lift_count = config.lift_end_index - config.closure_end_index
            if not 0 <= lift_step < lift_count:
                raise RuntimeError("adaptive lift phase advanced out of bounds")
            template_index = config.closure_end_index + 1 + lift_step
            fraction = (lift_step + 1) / lift_count
            interpolation = _minimum_jerk(fraction)
            nominal_target = self._reference_grasp_q + interpolation * (
                self._reference_lift_q - self._reference_grasp_q
            )
            target_q = capture_q + interpolation * (lift_q - capture_q)
            diagnostics = {
                "state": "adaptive_lifting_captured_object",
                "online_replanning": False,
                "lift_phase": interpolation,
                "velocity_source": "frozen_after_closure",
            }
            if template_index == config.lift_end_index:
                phase = "hold"
        elif phase == "hold":
            if capture_q is None or lift_q is None:
                raise RuntimeError("adaptive hold has no committed lift target")
            # Every post-lift template sample is the same frozen arm/hand
            # hold.  Dynamic closure can begin later than the nominal catch
            # index, so replaying those identical samples one by one would
            # leave the transaction artificially incomplete.  Commit the
            # identical final hold sample directly after the audited lift.
            template_index = self.selected_steps - 1
            nominal_target = self._reference_lift_q.copy()
            target_q = lift_q.copy()
            diagnostics = {
                "state": "adaptive_post_lift_hold",
                "online_replanning": False,
                "lift_phase": 1.0,
                "velocity_source": "frozen_after_closure",
            }
        else:
            raise RuntimeError(f"unknown adaptive tabletop phase: {phase}")

        target_step_limit = config.max_target_step_rad
        if (
            self._mpc_capture_enabled
            and diagnostics.get("state") == "adaptive_visual_servo_closure"
        ):
            # Do not stream a target trajectory faster than the commissioned
            # arm can physically follow.  A larger position lead only builds
            # tracking error and makes collision-driven replans look jerky;
            # it cannot make the native 0.5 rad/s controller move faster.
            closure_target_slew_limit = min(
                config.max_target_step_rad,
                config.arm_tracking_speed_rad_s * config.control_dt_s,
            )
            target_step_limit = closure_target_slew_limit
            unsmoothed_target = target_q.copy()
            target_q = self._committed_target_q + np.clip(
                target_q - self._committed_target_q,
                -closure_target_slew_limit,
                closure_target_slew_limit,
            )
            diagnostics["closure_target_slew_limit_rad"] = float(
                closure_target_slew_limit
            )
            diagnostics["closure_target_slew_applied"] = bool(
                not np.array_equal(target_q, unsmoothed_target)
            )

        hand_target = np.asarray(
            self.sequence.recorded_rh56_angle_set_register_order[template_index],
            dtype=np.int32,
        )
        adaptive_correction = target_q - nominal_target
        closure_target_transaction = bool(
            self._mpc_capture_enabled
            and diagnostics.get("state") == "adaptive_visual_servo_closure"
        )
        if closure_target_transaction:
            raw_proprio = np.asarray(proprio, dtype=np.float64)
            if raw_proprio.ndim != 2 or raw_proprio.shape[1] < 54:
                raise RuntimeError(
                    "closure safety requires raw 67D/96D policy proprio"
                )
            actual_fingertips_palm_m = raw_proprio[-1, 39:54].reshape(5, 3)

            def validate_target(*, nominal_q, target_q, hand_target):
                return self._safe_closure_target(
                    nominal_q=nominal_q,
                    target_q=target_q,
                    hand_target=hand_target,
                    actual_fingertips_palm_m=actual_fingertips_palm_m,
                )

        else:
            validate_target = self._safe_target
        try:
            safety = validate_target(
                nominal_q=nominal_target,
                target_q=target_q,
                hand_target=hand_target,
            )
        except RuntimeError as original_error:
            # Use the same correction-only safety backoff as the commissioned
            # v2 path.  The nominal demonstrated motion is immutable; only
            # the live planar correction may be reduced.
            safety = None
            for scale in np.linspace(0.9, 0.0, 10):
                requested_correction = adaptive_correction * float(scale)
                unbounded_candidate = nominal_target + requested_correction
                candidate = self._committed_target_q + np.clip(
                    unbounded_candidate - self._committed_target_q,
                    -target_step_limit,
                    target_step_limit,
                )
                try:
                    candidate_safety = validate_target(
                        nominal_q=nominal_target,
                        target_q=candidate,
                        hand_target=hand_target,
                    )
                except RuntimeError:
                    continue
                target_q = candidate
                adaptive_correction = target_q - nominal_target
                safety = candidate_safety
                diagnostics["safety_correction_backoff_scale"] = float(scale)
                break
            if safety is None:
                # A receding-horizon wrist correction can cross the absolute
                # correction envelope by a few milliradians even though the
                # exact previously committed arm pose was safe and already at
                # the capture corridor.  Permit one hand-only closure tick at
                # that byte-exact arm target, then require a fresh safe arm
                # solution on the next tick.  This neither relaxes the 0.12
                # rad limit nor lets a persistent conflict close blindly.
                can_hold_one_orientation_tick = bool(
                    isinstance(
                        original_error, _OrientationCorrectionConflict
                    )
                    and self._committed_adaptive_phase == "closure"
                    and self._committed_template_index >= 0
                    and not bool(
                        self._last_diagnostics.get(
                            "closure_orientation_conflict_hold", False
                        )
                    )
                )
                if can_hold_one_orientation_tick:
                    held_target = self._committed_target_q.copy()
                    try:
                        held_safety = validate_target(
                            nominal_q=nominal_target,
                            target_q=held_target,
                            hand_target=hand_target,
                        )
                    except RuntimeError:
                        held_safety = None
                    if held_safety is not None:
                        target_q = held_target
                        adaptive_correction = target_q - nominal_target
                        safety = held_safety
                        diagnostics["state"] = (
                            "adaptive_closure_orientation_conflict_hold"
                        )
                        diagnostics[
                            "closure_orientation_conflict_hold"
                        ] = True
                        diagnostics[
                            "orientation_conflict_required_rad"
                        ] = float(original_error.required_rad)
                        diagnostics["orientation_conflict_limit_rad"] = float(
                            config.max_orientation_correction_rad
                        )
                        diagnostics[
                            "closure_arm_target_held_hand_advanced"
                        ] = True
                # Once closure has been spatially/TTC committed, a fresh
                # visual-servo arm correction must not abort the grasp merely
                # because it would move the closing fingertips below the
                # commissioned table floor.  Keep the exact last safe arm
                # target for this tick while advancing only the bounded RH56
                # closure template.  The current hand shape is rechecked at
                # that held pose, so this never lowers the clearance limit.
                can_hold_arm_during_closure = bool(
                    safety is None
                    and isinstance(
                        original_error, _FingertipTableClearanceConflict
                    )
                    # The last hand-closing sample atomically selects the
                    # subsequent lift phase before safety validation.  It is
                    # still a closure target transaction and must retain the
                    # same arm-hold fallback instead of aborting one tick
                    # before lift.
                    and closure_target_transaction
                    and self._committed_template_index >= 0
                )
                if can_hold_arm_during_closure:
                    held_target = self._committed_target_q.copy()
                    held_clearance_error = None
                    try:
                        held_safety = validate_target(
                            nominal_q=nominal_target,
                            target_q=held_target,
                            hand_target=hand_target,
                        )
                    except _FingertipTableClearanceConflict as error:
                        held_clearance_error = error
                        held_safety = None
                    except RuntimeError:
                        held_safety = None
                    if held_safety is not None:
                        target_q = held_target
                        adaptive_correction = target_q - nominal_target
                        safety = held_safety
                        diagnostics["state"] = (
                            "adaptive_closure_fingertip_clearance_hold"
                        )
                        diagnostics["fingertip_clearance_hold"] = True
                        diagnostics[
                            "rejected_fingertip_clearance_m"
                        ] = float(original_error.required_m)
                        diagnostics["fingertip_clearance_limit_m"] = float(
                            config.minimum_fingertip_clearance_m
                        )
                        diagnostics[
                            "closure_arm_target_held_hand_advanced"
                        ] = True
                    elif held_clearance_error is not None:
                        # The arm is already at its last safe committed pose,
                        # but the next demonstrated finger curl itself would
                        # cross the table floor.  Closing is already committed
                        # here, so replace further descent with the minimum
                        # pose-preserving upward correction.  This preserves
                        # the configured clearance instead of relaxing it and
                        # naturally transitions toward the subsequent lift.
                        required_lift = (
                            float(config.minimum_fingertip_clearance_m)
                            - float(held_clearance_error.required_m)
                            + _CLOSURE_CLEARANCE_LIFT_MARGIN_M
                        )
                        if (
                            np.isfinite(required_lift)
                            and 0.0 < required_lift
                            <= _MAXIMUM_CLOSURE_CLEARANCE_LIFT_M
                        ):
                            plane_normal = np.asarray(
                                config.tabletop_plane_base[:3],
                                dtype=np.float64,
                            )
                            normal_norm = float(np.linalg.norm(plane_normal))
                            if normal_norm <= 0.0 or not np.isfinite(normal_norm):
                                raise RuntimeError(
                                    "tabletop plane has no finite unit normal"
                                )
                            plane_normal = plane_normal / normal_norm
                            try:
                                clearance_target = pose_preserving_joint_target(
                                    held_target,
                                    plane_normal * required_lift,
                                    T_flange_policy_palm=(
                                        self.contract.T_flange_policy_palm
                                    ),
                                    joint_limits_rad=self.contract.joint_limits_rad,
                                    joint_limit_margin_rad=(
                                        config.joint_limit_margin_rad
                                    ),
                                    damping=config.dls_damping,
                                    local_precision_refinement=True,
                                )
                                clearance_safety = validate_target(
                                    nominal_q=self._committed_nominal_target_q,
                                    target_q=clearance_target,
                                    hand_target=hand_target,
                                )
                            except RuntimeError:
                                clearance_safety = None
                            if clearance_safety is not None:
                                nominal_target = (
                                    self._committed_nominal_target_q.copy()
                                )
                                target_q = clearance_target
                                adaptive_correction = target_q - nominal_target
                                safety = clearance_safety
                                diagnostics["state"] = (
                                    "adaptive_closure_clearance_lift"
                                )
                                diagnostics[
                                    "closure_clearance_lift_applied"
                                ] = True
                                diagnostics[
                                    "closure_clearance_lift_m"
                                ] = float(required_lift)
                                diagnostics[
                                    "closure_clearance_before_lift_m"
                                ] = float(held_clearance_error.required_m)
                                diagnostics[
                                    "fingertip_clearance_limit_m"
                                ] = float(config.minimum_fingertip_clearance_m)
                                diagnostics[
                                    "closure_arm_target_raised_hand_advanced"
                                ] = True
                if safety is None:
                    # A board impact or a lower-than-commissioned scene can
                    # make the next approach correction temporarily violate
                    # the wrist or fingertip envelope.  Before closure, keep
                    # the exact previous safe command and replan from the next
                    # fresh observation instead of terminating the episode.
                    # This does not lower either limit or advance preshape,
                    # closure, lift, or the intercept authority.
                    can_hold_previous_safe_target = bool(
                        isinstance(
                            original_error,
                            (
                                _OrientationCorrectionConflict,
                                _FingertipTableClearanceConflict,
                            ),
                        )
                        and phase == "approach"
                        and not diagnostics.get("capture_gate_accepted", False)
                        and self._committed_template_index >= 0
                    )
                    if not can_hold_previous_safe_target:
                        raise original_error
                    nominal_target = self._committed_nominal_target_q.copy()
                    target_q = self._committed_target_q.copy()
                    goal_q = self._committed_goal_q.copy()
                    adaptive_correction = self._committed_correction.copy()
                    template_index = int(self._committed_template_index)
                    preshape_index = int(self._committed_preshape_index)
                    closure_step = int(self._committed_closure_step)
                    lift_step = int(self._committed_lift_step)
                    tabletop_height_correction_m = (
                        self._committed_tabletop_height_correction_m
                    )
                    hand_target = np.asarray(
                        self.sequence.recorded_rh56_angle_set_register_order[
                            template_index
                        ],
                        dtype=np.int32,
                    )
                    safety = self._safe_target(
                        nominal_q=nominal_target,
                        target_q=target_q,
                        hand_target=hand_target,
                    )
                    if isinstance(
                        original_error, _OrientationCorrectionConflict
                    ):
                        diagnostics["state"] = (
                            "adaptive_orientation_conflict_hold"
                        )
                        diagnostics["orientation_conflict_hold"] = True
                        diagnostics["orientation_conflict_required_rad"] = float(
                            original_error.required_rad
                        )
                        diagnostics["orientation_conflict_limit_rad"] = float(
                            config.max_orientation_correction_rad
                        )
                        diagnostics[
                            "orientation_conflict_control_state_frozen"
                        ] = True
                    else:
                        diagnostics["state"] = (
                            "adaptive_approach_fingertip_clearance_hold"
                        )
                        diagnostics["approach_fingertip_clearance_hold"] = True
                        diagnostics["rejected_fingertip_clearance_m"] = float(
                            original_error.required_m
                        )
                        diagnostics["fingertip_clearance_limit_m"] = float(
                            config.minimum_fingertip_clearance_m
                        )
                        diagnostics[
                            "fingertip_clearance_control_state_frozen"
                        ] = True
        if diagnostics.get("capture_target_committed_on_success"):
            capture_q = target_q.copy()
            lift_q = pose_preserving_joint_target(
                capture_q,
                np.asarray([0.0, 0.0, config.grasp_lift_m]),
                T_flange_policy_palm=self.contract.T_flange_policy_palm,
                joint_limits_rad=self.contract.joint_limits_rad,
                joint_limit_margin_rad=config.joint_limit_margin_rad,
                damping=config.dls_damping,
                local_precision_refinement=True,
            )
            diagnostics["lift_goal_q_rad"] = lift_q.tolist()
        diagnostics.update(safety)
        diagnostics["adaptive_phase"] = phase
        diagnostics["runtime_index"] = runtime_index
        diagnostics["template_index"] = template_index
        diagnostics["preshape_index"] = preshape_index
        diagnostics["closure_step"] = closure_step
        diagnostics["lift_step"] = lift_step
        diagnostics["joint_correction_rad"] = adaptive_correction.tolist()
        proposal = _OnlineProposal(
            sequence=int(sequence),
            index=runtime_index,
            correction_q_rad=(target_q - nominal_target).copy(),
            target_q_rad=target_q.astype(np.float32),
            goal_q_rad=goal_q.copy(),
            capture_q_rad=None if capture_q is None else capture_q.copy(),
            lift_q_rad=None if lift_q is None else lift_q.copy(),
            tabletop_height_correction_m=float(
                tabletop_height_correction_m
                if tabletop_height_correction_m is not None
                else 0.0
            ),
            live_fit_seen=bool(live_fit_seen),
            full_occluded_closure_committed=bool(
                full_occluded_closure_committed
            ),
            current_volume_capture_committed=bool(
                current_volume_capture_committed
            ),
            force_contact_capture_committed=bool(
                self._committed_force_contact_capture
            ),
            diagnostics=diagnostics,
            template_index=template_index,
            nominal_target_q_rad=nominal_target.copy(),
            adaptive_phase=phase,
            preshape_index=preshape_index,
            closure_step=closure_step,
            lift_step=lift_step,
        )
        self._pending = proposal
        return ReplayPolicyOutput(
            action13=self.sequence.actions13[template_index].copy(),
            exact_franka_target_q_rad=proposal.target_q_rad.copy(),
            exact_rh56_angle_set_register_order=hand_target.copy(),
            bypass_rh56_host_slew=True,
            replay_frame_index=template_index,
        )

    def action_for_sequence(
        self,
        sequence: int,
        pointcloud: np.ndarray,
        valid: np.ndarray,
        proprio: np.ndarray,
    ) -> ReplayPolicyOutput:
        if isinstance(sequence, bool) or not isinstance(sequence, (int, np.integer)):
            raise ValueError("online planner sequence must be an integer")
        index = int(sequence) - 1
        if not 0 <= index < self.selected_steps:
            raise ValueError("online planner sequence is outside the template")
        if np.asarray(pointcloud).shape != (
            self.history_length,
            128,
            self.point_feature_dim,
        ):
            raise ValueError("online planner received malformed point-cloud history")
        if np.asarray(valid).shape != (self.history_length, 128) or (
            np.asarray(proprio).shape != (self.history_length, self.proprio_dim)
        ):
            raise ValueError("online planner received malformed proprio history")
        if self._adaptive_enabled:
            return self._adaptive_action_for_sequence(
                sequence, pointcloud, valid, proprio
            )
        if self._pending is not None:
            if self._pending.sequence != int(sequence):
                raise RuntimeError("online planner already has another proposal")
            proposal = self._pending
        else:
            nominal_q = np.asarray(
                self.sequence.recorded_franka_target_q_rad[index],
                dtype=np.float64,
            )
            hand = np.asarray(
                self.sequence.recorded_rh56_angle_set_register_order[index],
                dtype=np.int32,
            )
            capture_q = None if self._capture_q is None else self._capture_q.copy()
            lift_q = None if self._lift_q is None else self._lift_q.copy()
            tabletop_height_correction_m = (
                self._committed_tabletop_height_correction_m
            )
            if index <= self.config.closure_end_index:
                centers = self._centers_base(pointcloud, valid, proprio)
                predicted, diagnostics = self._predict_intercept(
                    centers,
                    index=index,
                )
                reference_height = float(
                    self.config.reference_intercept_center_base_m[2]
                )
                observed_height_correction = float(
                    diagnostics["object_center_base_m"][2]
                ) - reference_height
                if tabletop_height_correction_m is None:
                    tabletop_height_correction_m = observed_height_correction
                    height_source = "first_committed_planar_observation"
                else:
                    height_source = "locked_tabletop_episode_height"
                planar_predicted = np.asarray(predicted, dtype=np.float64).copy()
                planar_predicted[2] = (
                    reference_height + tabletop_height_correction_m
                )
                diagnostics["observed_height_correction_m"] = (
                    observed_height_correction
                )
                diagnostics["tabletop_height_correction_m"] = float(
                    tabletop_height_correction_m
                )
                diagnostics["tabletop_height_source"] = height_source
                if diagnostics["hold_last_intercept_goal"]:
                    raw_goal = self._committed_goal_q.copy()
                    committed_palm = panda_T_base_policy_palm(
                        raw_goal,
                        self.contract.T_flange_policy_palm,
                    )
                    desired = (
                        committed_palm[:3, 3]
                        - self._reference_grasp_palm[:3, 3]
                    )
                    diagnostics["velocity_source"] = (
                        "held_last_live_intercept_during_unstable_fit"
                    )
                else:
                    unbounded_planar = planar_predicted.copy()
                    planar_delta = (
                        unbounded_planar
                        - np.asarray(
                            self.config.reference_intercept_center_base_m,
                            dtype=np.float64,
                        )
                    )
                    planar_distance = float(np.linalg.norm(planar_delta[:2]))
                    boundary_projected = bool(
                        index < self.config.template_catch_index
                        and planar_distance
                        > self.config.max_position_correction_xy_m
                    )
                    if boundary_projected:
                        planar_predicted[:2] = (
                            np.asarray(
                                self.config.reference_intercept_center_base_m,
                                dtype=np.float64,
                            )[:2]
                            + planar_delta[:2]
                            * (
                                self.config.max_position_correction_xy_m
                                / planar_distance
                            )
                        )
                    diagnostics["workspace_boundary_projection"] = (
                        boundary_projected
                    )
                    diagnostics["unbounded_intercept_xy_distance_m"] = (
                        planar_distance
                    )
                    raw_goal, desired = self._goal_q_for_intercept(
                        planar_predicted
                    )
                goal_q = self._filtered_goal(raw_goal)
                diagnostics["desired_position_correction_base_m"] = desired.tolist()
                diagnostics["raw_intercept_goal_q_rad"] = raw_goal.tolist()
                diagnostics["filtered_intercept_goal_q_rad"] = goal_q.tolist()
            else:
                if capture_q is None or lift_q is None:
                    raise RuntimeError(
                        "online intercept lift started without an atomic capture target"
                    )
                goal_q = self._committed_goal_q.copy()
                diagnostics = {
                    "state": "lifting_captured_object",
                    "online_replanning": False,
                    "template_index": int(index),
                    "velocity_source": "frozen_after_closure",
                }
            if index < self.config.template_start_index:
                target_q = self.contract.q_home_rad.astype(np.float64).copy()
                diagnostics["state"] = "motion_fit_at_q_home"
                diagnostics["approach_phase"] = 0.0
            elif index <= self.config.template_catch_index:
                denominator = (
                    self.config.template_catch_index
                    - self.config.template_start_index
                )
                phase = _minimum_jerk(
                    (index - self.config.template_start_index) / denominator
                )
                target_q = self.contract.q_home_rad + phase * (
                    goal_q - self.contract.q_home_rad
                )
                diagnostics["state"] = "predicted_intercept_approach"
                diagnostics["approach_phase"] = phase
            elif index <= self.config.closure_end_index:
                target_q = goal_q.copy()
                diagnostics["state"] = "visual_servo_during_closure"
                diagnostics["approach_phase"] = 1.0
            elif index <= self.config.lift_end_index:
                denominator = (
                    self.config.lift_end_index - self.config.closure_end_index
                )
                phase = _minimum_jerk(
                    (index - self.config.closure_end_index) / denominator
                )
                target_q = capture_q + phase * (lift_q - capture_q)
                diagnostics["state"] = "lifting_captured_object"
                diagnostics["lift_phase"] = phase
            else:
                target_q = lift_q.copy()
                diagnostics["state"] = "post_lift_hold"
                diagnostics["lift_phase"] = 1.0
            filtered = target_q - nominal_q
            try:
                safety = self._safe_target(
                    nominal_q=nominal_q,
                    target_q=target_q,
                    hand_target=hand,
                )
            except RuntimeError as original_error:
                # The nominal template is pre-audited.  A Cartesian correction
                # can approach a joint/table margin as the demonstrated hand
                # changes shape.  Back off only the live correction, never the
                # nominal target.  Safety backoff may revoke correction faster
                # than its normal 20 mrad/tick filter; the q_d native 1 kHz
                # limiter still bounds physical velocity/acceleration/jerk.
                safety = None
                for scale in np.linspace(0.9, 0.0, 10):
                    backed_off = filtered * float(scale)
                    candidate = nominal_q + backed_off
                    try:
                        candidate_safety = self._safe_target(
                            nominal_q=nominal_q,
                            target_q=candidate,
                            hand_target=hand,
                        )
                    except RuntimeError:
                        continue
                    filtered = backed_off
                    target_q = candidate
                    safety = candidate_safety
                    diagnostics["safety_correction_backoff_scale"] = float(scale)
                    backoff_step = float(
                        np.max(
                            np.abs(
                                backed_off - self._committed_correction
                            )
                        )
                    )
                    diagnostics["safety_backoff_joint_step_rad"] = backoff_step
                    diagnostics["safety_backoff_exceeded_nominal_slew"] = bool(
                        backoff_step
                        > self.config.max_joint_correction_step_rad + 1.0e-9
                    )
                    break
                if safety is None:
                    raise original_error
            diagnostics.update(safety)
            diagnostics["joint_correction_rad"] = filtered.tolist()
            if index == self.config.closure_end_index:
                capture_q = target_q.copy()
                lift_q = pose_preserving_joint_target(
                    capture_q,
                    np.asarray([0.0, 0.0, self.config.grasp_lift_m]),
                    T_flange_policy_palm=self.contract.T_flange_policy_palm,
                    joint_limits_rad=self.contract.joint_limits_rad,
                    joint_limit_margin_rad=self.config.joint_limit_margin_rad,
                    damping=self.config.dls_damping,
                )
                diagnostics["capture_target_committed_on_success"] = True
                diagnostics["lift_goal_q_rad"] = lift_q.tolist()
            proposal = _OnlineProposal(
                sequence=int(sequence),
                index=index,
                correction_q_rad=filtered.copy(),
                target_q_rad=target_q.astype(np.float32),
                goal_q_rad=goal_q.copy(),
                capture_q_rad=None if capture_q is None else capture_q.copy(),
                lift_q_rad=None if lift_q is None else lift_q.copy(),
                tabletop_height_correction_m=float(
                    tabletop_height_correction_m
                    if tabletop_height_correction_m is not None
                    else 0.0
                ),
                live_fit_seen=bool(
                    self._committed_live_fit_seen
                    or diagnostics.get("velocity_source") == "live_fit"
                ),
                diagnostics=diagnostics,
            )
            self._pending = proposal
        hand_target = self.sequence.recorded_rh56_angle_set_register_order[index]
        return ReplayPolicyOutput(
            action13=self.sequence.actions13[index].copy(),
            exact_franka_target_q_rad=proposal.target_q_rad.copy(),
            exact_rh56_angle_set_register_order=hand_target.copy(),
            bypass_rh56_host_slew=True,
            replay_frame_index=index,
        )

    def action_for_occluded_closure(
        self,
        sequence: int,
        *,
        observation_hold_reason: str,
    ) -> Optional[ReplayPolicyOutput]:
        """Finish a bounded late committed closure without vision.

        This is deliberately not a stale-point-cloud fallback.  It is only a
        bounded completion transaction after the spatial/TTC capture gate has
        already committed closure.  The ordinary late-occlusion path holds
        Franka byte-exact.  A temporally aligned MPC commitment may instead
        advance the immutable demonstrated descent around its frozen
        intercept correction; it never replans from retained object geometry.
        RH56 always advances only through the demonstrated closure targets.
        """

        if isinstance(sequence, bool) or not isinstance(
            sequence, (int, np.integer)
        ):
            raise ValueError("occluded closure sequence must be an integer")
        runtime_index = int(sequence) - 1
        if not 0 <= runtime_index < self.selected_steps:
            raise ValueError("occluded closure sequence is outside the template")
        reason = str(observation_hold_reason).strip()
        if not reason.startswith(
            (
                "guarded_stale_palm_fail_closed:",
                "object_pointcloud_transient_invalid:",
            )
        ):
            return None
        if not (self._adaptive_enabled and self._mpc_capture_enabled):
            return None
        if self._pending is not None:
            raise RuntimeError("online planner already has another proposal")
        if self._committed_adaptive_phase != "closure":
            return None

        config = self.config
        committed_template = int(self._committed_template_index)
        remaining = int(config.closure_end_index - committed_template)
        maximum_occluded_ticks = _MAXIMUM_OCCLUDED_CLOSURE_TICKS
        if self._committed_full_occluded_closure:
            maximum_occluded_ticks = int(
                config.closure_end_index - config.template_catch_index
            )
        force_contact_first_tick = bool(
            self._committed_force_contact_capture
            and committed_template == config.template_catch_index - 1
            and int(self._committed_closure_step) == -1
        )
        if force_contact_first_tick:
            maximum_occluded_ticks += 1
        if not 1 <= remaining <= maximum_occluded_ticks:
            return None
        if force_contact_first_tick:
            expected_step = -1
        else:
            expected_step = committed_template - int(
                config.template_catch_index
            )
            if (
                committed_template < config.template_catch_index
                or committed_template >= config.closure_end_index
                or int(self._committed_closure_step) != expected_step
            ):
                raise RuntimeError(
                    "committed closure state is internally inconsistent"
                )

        template_index = committed_template + 1
        closure_step = expected_step + 1
        if (
            self._committed_full_occluded_closure
            and not self._committed_current_volume_capture
        ):
            recorded_nominal_target = np.asarray(
                self.sequence.recorded_franka_target_q_rad[template_index],
                dtype=np.float64,
            )
            closure_slew_limit = min(
                config.max_target_step_rad,
                config.arm_tracking_speed_rad_s * config.control_dt_s,
            )
            nominal_target = self._slew_joint_target(
                self._committed_nominal_target_q,
                recorded_nominal_target,
                closure_slew_limit,
            )
            target_q = self._adaptive_approach_target(
                nominal_target_q_rad=nominal_target,
                goal_q_rad=self._committed_goal_q,
            )
            target_q = self._committed_target_q + np.clip(
                target_q - self._committed_target_q,
                -closure_slew_limit,
                closure_slew_limit,
            )
        else:
            target_q = self._committed_target_q.copy()
            nominal_target = self._committed_nominal_target_q.copy()
        hand_target = np.asarray(
            self.sequence.recorded_rh56_angle_set_register_order[
                template_index
            ],
            dtype=np.int32,
        )
        clearance_lift_m = None
        try:
            safety = self._safe_closure_target(
                nominal_q=nominal_target,
                target_q=target_q,
                hand_target=hand_target,
            )
        except _FingertipTableClearanceConflict as clearance_error:
            required_lift = (
                float(config.minimum_fingertip_clearance_m)
                - float(clearance_error.required_m)
                + _CLOSURE_CLEARANCE_LIFT_MARGIN_M
            )
            if not (
                np.isfinite(required_lift)
                and 0.0 < required_lift
                <= _MAXIMUM_CLOSURE_CLEARANCE_LIFT_M
            ):
                raise
            plane_normal = np.asarray(
                config.tabletop_plane_base[:3], dtype=np.float64
            )
            normal_norm = float(np.linalg.norm(plane_normal))
            if normal_norm <= 0.0 or not np.isfinite(normal_norm):
                raise RuntimeError("tabletop plane has no finite unit normal")
            plane_normal = plane_normal / normal_norm
            target_q = pose_preserving_joint_target(
                self._committed_target_q,
                plane_normal * required_lift,
                T_flange_policy_palm=self.contract.T_flange_policy_palm,
                joint_limits_rad=self.contract.joint_limits_rad,
                joint_limit_margin_rad=config.joint_limit_margin_rad,
                damping=config.dls_damping,
                local_precision_refinement=True,
            )
            nominal_target = self._committed_nominal_target_q.copy()
            safety = self._safe_closure_target(
                nominal_q=nominal_target,
                target_q=target_q,
                hand_target=hand_target,
            )
            clearance_lift_m = float(required_lift)
        phase = "closure"
        capture_q = None if self._capture_q is None else self._capture_q.copy()
        lift_q = None if self._lift_q is None else self._lift_q.copy()
        diagnostics: dict[str, object] = {
            "state": "adaptive_occluded_closure_hand_only",
            "online_replanning": False,
            "adaptive_phase": phase,
            "runtime_index": runtime_index,
            "template_index": template_index,
            "closure_step": closure_step,
            "occluded_closure_hand_only": True,
            "occluded_closure_remaining_before_tick": remaining,
            "occluded_closure_maximum_ticks": (
                maximum_occluded_ticks
            ),
            "full_occluded_closure_committed": bool(
                self._committed_full_occluded_closure
            ),
            "observation_hold_reason": reason,
            "arm_target_held": bool(
                not self._committed_full_occluded_closure
                or self._committed_current_volume_capture
            ),
            "frozen_intercept_closure_arm_advanced": bool(
                self._committed_full_occluded_closure
                and not self._committed_current_volume_capture
            ),
            "current_volume_capture_committed": bool(
                self._committed_current_volume_capture
            ),
            "force_contact_capture_committed": bool(
                self._committed_force_contact_capture
            ),
            "velocity_source": "frozen_after_visual_closure_commit",
            **safety,
        }
        if clearance_lift_m is not None:
            diagnostics.update(
                {
                    "state": "adaptive_occluded_closure_clearance_lift",
                    "closure_clearance_lift_applied": True,
                    "closure_clearance_lift_m": clearance_lift_m,
                }
            )
        if template_index == config.closure_end_index:
            phase = "lift"
            capture_q = target_q.copy()
            lift_q = pose_preserving_joint_target(
                capture_q,
                np.asarray([0.0, 0.0, config.grasp_lift_m]),
                T_flange_policy_palm=self.contract.T_flange_policy_palm,
                joint_limits_rad=self.contract.joint_limits_rad,
                joint_limit_margin_rad=config.joint_limit_margin_rad,
                damping=config.dls_damping,
                local_precision_refinement=True,
            )
            diagnostics["adaptive_phase"] = phase
            diagnostics["capture_target_committed_on_success"] = True
            diagnostics["lift_goal_q_rad"] = lift_q.tolist()

        proposal = _OnlineProposal(
            sequence=int(sequence),
            index=runtime_index,
            correction_q_rad=(target_q - nominal_target).copy(),
            target_q_rad=target_q.astype(np.float32),
            goal_q_rad=self._committed_goal_q.copy(),
            capture_q_rad=None if capture_q is None else capture_q.copy(),
            lift_q_rad=None if lift_q is None else lift_q.copy(),
            tabletop_height_correction_m=float(
                self._committed_tabletop_height_correction_m
                if self._committed_tabletop_height_correction_m is not None
                else 0.0
            ),
            live_fit_seen=bool(self._committed_live_fit_seen),
            full_occluded_closure_committed=bool(
                self._committed_full_occluded_closure
            ),
            current_volume_capture_committed=bool(
                self._committed_current_volume_capture
            ),
            force_contact_capture_committed=bool(
                self._committed_force_contact_capture
            ),
            diagnostics=diagnostics,
            template_index=template_index,
            nominal_target_q_rad=nominal_target,
            adaptive_phase=phase,
            preshape_index=int(self._committed_preshape_index),
            closure_step=closure_step,
            lift_step=int(self._committed_lift_step),
        )
        self._pending = proposal
        return ReplayPolicyOutput(
            action13=self.sequence.actions13[template_index].copy(),
            exact_franka_target_q_rad=proposal.target_q_rad.copy(),
            exact_rh56_angle_set_register_order=hand_target.copy(),
            bypass_rh56_host_slew=True,
            replay_frame_index=template_index,
            occluded_clearance_lift=bool(clearance_lift_m is not None),
        )

    def action_for_frozen_post_closure(
        self,
        sequence: int,
        *,
        observation_hold_reason: str,
    ) -> Optional[ReplayPolicyOutput]:
        """Advance the frozen lift/hold template without an object cloud.

        The tick source separately requires a current Franka owner snapshot on
        every call.  This method never predicts, replans, or consumes retained
        object geometry after closure.
        """

        if isinstance(sequence, bool) or not isinstance(
            sequence, (int, np.integer)
        ):
            raise ValueError("frozen post-closure sequence must be an integer")
        runtime_index = int(sequence) - 1
        if not 0 <= runtime_index < self.selected_steps:
            raise ValueError("frozen post-closure sequence is outside the template")
        reason = str(observation_hold_reason).strip()
        if not reason.startswith(
            (
                "guarded_stale_palm_fail_closed:",
                "object_pointcloud_transient_invalid:",
            )
        ):
            return None
        if not (self._adaptive_enabled and self._mpc_capture_enabled):
            return None
        if self._pending is not None:
            raise RuntimeError("online planner already has another proposal")
        phase = self._committed_adaptive_phase
        if phase not in {"lift", "hold"}:
            return None
        if self._capture_q is None or self._lift_q is None:
            raise RuntimeError("frozen post-closure motion has no capture target")

        config = self.config
        lift_step = int(self._committed_lift_step)
        capture_q = self._capture_q.copy()
        lift_q = self._lift_q.copy()
        if phase == "lift":
            lift_step += 1
            lift_count = config.lift_end_index - config.closure_end_index
            if not 0 <= lift_step < lift_count:
                raise RuntimeError("frozen lift phase advanced out of bounds")
            template_index = config.closure_end_index + 1 + lift_step
            fraction = (lift_step + 1) / lift_count
            interpolation = _minimum_jerk(fraction)
            nominal_target = self._reference_grasp_q + interpolation * (
                self._reference_lift_q - self._reference_grasp_q
            )
            target_q = capture_q + interpolation * (lift_q - capture_q)
            next_phase = (
                "hold"
                if template_index == config.lift_end_index
                else "lift"
            )
            diagnostics: dict[str, object] = {
                "state": "adaptive_frozen_lift_without_object_cloud",
                "online_replanning": False,
                "lift_phase": interpolation,
                "velocity_source": "frozen_after_closure",
            }
        else:
            template_index = self.selected_steps - 1
            nominal_target = self._reference_lift_q.copy()
            target_q = lift_q.copy()
            next_phase = "hold"
            diagnostics = {
                "state": "adaptive_frozen_post_lift_hold_without_object_cloud",
                "online_replanning": False,
                "lift_phase": 1.0,
                "velocity_source": "frozen_after_closure",
            }
        hand_target = np.asarray(
            self.sequence.recorded_rh56_angle_set_register_order[
                template_index
            ],
            dtype=np.int32,
        )
        safety = self._safe_target(
            nominal_q=nominal_target,
            target_q=target_q,
            hand_target=hand_target,
        )
        diagnostics.update(
            {
                "adaptive_phase": next_phase,
                "runtime_index": runtime_index,
                "template_index": template_index,
                "closure_step": int(self._committed_closure_step),
                "lift_step": lift_step,
                "frozen_post_closure_without_pointcloud": True,
                "observation_hold_reason": reason,
                **safety,
            }
        )
        proposal = _OnlineProposal(
            sequence=int(sequence),
            index=runtime_index,
            correction_q_rad=(target_q - nominal_target).copy(),
            target_q_rad=target_q.astype(np.float32),
            goal_q_rad=self._committed_goal_q.copy(),
            capture_q_rad=capture_q,
            lift_q_rad=lift_q,
            tabletop_height_correction_m=float(
                self._committed_tabletop_height_correction_m
                if self._committed_tabletop_height_correction_m is not None
                else 0.0
            ),
            live_fit_seen=bool(self._committed_live_fit_seen),
            full_occluded_closure_committed=bool(
                self._committed_full_occluded_closure
            ),
            current_volume_capture_committed=bool(
                self._committed_current_volume_capture
            ),
            force_contact_capture_committed=bool(
                self._committed_force_contact_capture
            ),
            diagnostics=diagnostics,
            template_index=template_index,
            nominal_target_q_rad=nominal_target.copy(),
            adaptive_phase=next_phase,
            preshape_index=int(self._committed_preshape_index),
            closure_step=int(self._committed_closure_step),
            lift_step=lift_step,
        )
        self._pending = proposal
        return ReplayPolicyOutput(
            action13=self.sequence.actions13[template_index].copy(),
            exact_franka_target_q_rad=proposal.target_q_rad.copy(),
            exact_rh56_angle_set_register_order=hand_target.copy(),
            bypass_rh56_host_slew=True,
            replay_frame_index=template_index,
        )

    def commit_replay_proposal(self, sequence: int) -> None:
        if self._pending is None or self._pending.sequence != int(sequence):
            raise RuntimeError("online planner commit differs from its proposal")
        proposal = self._pending
        previous_adaptive_phase = self._committed_adaptive_phase
        self._committed_index = proposal.index
        if self._adaptive_enabled:
            if proposal.nominal_target_q_rad is None:
                raise RuntimeError("adaptive proposal lost its nominal target")
            self._committed_template_index = int(proposal.template_index)
            self._committed_nominal_target_q = np.asarray(
                proposal.nominal_target_q_rad, dtype=np.float64
            ).copy()
            self._committed_adaptive_phase = str(proposal.adaptive_phase)
            self._committed_preshape_index = int(proposal.preshape_index)
            self._committed_closure_step = int(proposal.closure_step)
            self._committed_lift_step = int(proposal.lift_step)
        self._committed_correction = proposal.correction_q_rad.copy()
        self._committed_target_q = proposal.target_q_rad.astype(
            np.float64
        ).copy()
        self._committed_goal_q = proposal.goal_q_rad.copy()
        self._capture_q = (
            None
            if proposal.capture_q_rad is None
            else proposal.capture_q_rad.copy()
        )
        self._lift_q = (
            None if proposal.lift_q_rad is None else proposal.lift_q_rad.copy()
        )
        self._committed_tabletop_height_correction_m = float(
            proposal.tabletop_height_correction_m
        )
        self._committed_live_fit_seen = bool(proposal.live_fit_seen)
        self._committed_full_occluded_closure = bool(
            proposal.full_occluded_closure_committed
        )
        self._committed_current_volume_capture = bool(
            proposal.current_volume_capture_committed
        )
        self._committed_force_contact_capture = bool(
            proposal.force_contact_capture_committed
        )
        self._last_committed_policy_sequence = int(sequence)
        self._last_diagnostics = dict(proposal.diagnostics)
        trace_keys = (
            "adaptive_phase",
            "state",
            "template_index",
            "object_center_base_m",
            "predicted_intercept_center_base_m",
            "velocity_base_m_s",
            "planning_velocity_base_m_s",
            "velocity_source",
            "speed_m_s",
            "prediction_horizon_s",
            "arm_prediction_horizon_s",
            "grasp_completion_horizon_s",
            "grasp_completion_horizon_applied_s",
            "arm_tracking_error_rad",
            "capture_gate_accepted",
            "capture_time_to_contact_s",
            "capture_along_track_m",
            "capture_cross_track_m",
            "capture_cross_track_limit_m",
            "capture_z_error_m",
            "capture_current_along_track_m",
            "capture_current_cross_track_m",
            "capture_current_z_error_m",
            "capture_measured_along_track_m",
            "capture_measured_cross_track_m",
            "capture_measured_z_error_m",
            "capture_current_volume_committed",
            "capture_arm_error_rad",
            "capture_arm_projection_horizon_s",
            "capture_uses_projected_arm_pose",
            "capture_degraded_live_fit_continuation",
            "capture_temporal_alignment_used",
            "capture_temporal_along_tolerance_m",
            "capture_temporal_alignment_committed",
            "full_occluded_closure_committed",
            "current_volume_capture_committed",
            "force_contact_capture_committed",
            "current_volume_closure_arm_held",
            "occluded_closure_maximum_ticks",
            "goal_source",
            "joint_reachability_projection_scale",
            "target_step_rad",
            "commanded_hand_fingertip_clearance_m",
            "closure_transition_fingertip_clearance_m",
            "minimum_fingertip_clearance_m",
            "orientation_conflict_hold",
            "closure_orientation_conflict_hold",
            "orientation_conflict_required_rad",
            "orientation_conflict_limit_rad",
            "fingertip_clearance_hold",
            "approach_fingertip_clearance_hold",
            "fingertip_clearance_control_state_frozen",
            "rejected_fingertip_clearance_m",
            "fingertip_clearance_limit_m",
            "closure_target_slew_limit_rad",
            "closure_target_slew_applied",
            "closure_nominal_slew_limit_rad",
            "closure_nominal_slew_applied",
            "closure_clearance_lift_applied",
            "closure_clearance_lift_m",
            "closure_clearance_before_lift_m",
            "occluded_closure_hand_only",
            "occluded_closure_remaining_before_tick",
            "occluded_closure_maximum_ticks",
            "arm_target_held",
            "frozen_post_closure_without_pointcloud",
        )
        self._committed_diagnostics_trace.append(
            {
                "sequence": int(sequence),
                "runtime_index": int(proposal.index),
                **{
                    key: proposal.diagnostics[key]
                    for key in trace_keys
                    if key in proposal.diagnostics
                },
            }
        )
        self._replay_complete = (
            proposal.template_index == self.selected_steps - 1
            if self._adaptive_enabled
            else proposal.index == self.selected_steps - 1
        )
        self._pending = None
        if not self._started_printed and proposal.index == 0:
            self._started_printed = True
            print(
                "[Tabletop online intercept START] object motion detected; "
                "predicting, approaching, closing and lifting at 20 Hz",
                flush=True,
            )
        if (
            self._adaptive_enabled
            and proposal.adaptive_phase != previous_adaptive_phase
        ):
            if proposal.adaptive_phase == "closure":
                print(
                    "[Tabletop capture CLOSURE] spatial/TTC gate passed; "
                    f"ttc={proposal.diagnostics.get('capture_time_to_contact_s')}s "
                    f"arm_error={proposal.diagnostics.get('capture_arm_error_rad')}rad",
                    flush=True,
                )
            elif proposal.adaptive_phase == "lift":
                print(
                    "[Tabletop capture LIFT] demonstrated closure completed; "
                    "capture pose frozen and 120 mm lift started",
                    flush=True,
                )
            elif proposal.adaptive_phase == "hold":
                print(
                    "[Tabletop capture HOLD] lift completed",
                    flush=True,
                )

    def commit_rh56_feedback_after_dual_ack(
        self,
        sequence: int,
        feedback: Mapping[str, object],
    ) -> None:
        """Commit a near-hand multi-finger contact as closure authority.

        The callback is invoked only after the exact Franka/RH56 command has
        dual-ACKed and the owner has copied a fresh complete RH56 feedback
        sample.  It can convert the already committed open-hand approach into
        a frozen-arm closure, but cannot alter the command that produced the
        feedback or grant another descent.
        """

        if isinstance(sequence, bool) or not isinstance(
            sequence, (int, np.integer)
        ):
            raise ValueError("RH56 feedback sequence must be an integer")
        current_sequence = int(sequence)
        if current_sequence != self._last_committed_policy_sequence:
            raise RuntimeError(
                "RH56 feedback does not bind the latest committed policy sequence"
            )
        if self._pending is not None:
            raise RuntimeError("RH56 feedback cannot overlap a policy proposal")
        if not isinstance(feedback, Mapping):
            raise TypeError("RH56 feedback must be a mapping")
        if feedback.get("fresh") is not True:
            return
        raw_forces = feedback.get("forces_g")
        raw_errors = feedback.get("errors")
        try:
            forces = np.asarray(raw_forces, dtype=np.int64)
            errors = np.asarray(raw_errors, dtype=np.int64)
        except (TypeError, ValueError):
            return
        if forces.shape != (6,) or errors.shape != (6,):
            return
        if np.any(errors != 0):
            return

        previous_sequence = int(self._last_rh56_feedback_sequence)
        previous_forces = self._last_rh56_feedback_forces_g
        self._last_rh56_feedback_sequence = current_sequence
        self._last_rh56_feedback_forces_g = forces.copy()
        if (
            previous_forces is None
            or previous_sequence != current_sequence - 1
            or self._committed_force_contact_capture
        ):
            return

        diagnostics = self._last_diagnostics
        try:
            measured_along = float(
                diagnostics["capture_measured_along_track_m"]
            )
            measured_cross = float(
                diagnostics["capture_measured_cross_track_m"]
            )
            measured_z = float(diagnostics["capture_measured_z_error_m"])
            speed = float(diagnostics["speed_m_s"])
            tip_clearance = float(
                diagnostics["capture_current_fingertip_clearance_m"]
            )
        except (KeyError, TypeError, ValueError):
            return
        finite_metrics = np.asarray(
            [measured_along, measured_cross, measured_z, speed, tip_clearance],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(finite_metrics)):
            return
        planar_shell = float(np.hypot(measured_along, measured_cross))
        force_delta = np.abs(forces - previous_forces)
        contact_axes = tuple(
            axis
            for axis in range(4)
            if abs(int(forces[axis])) >= _CONTACT_CAPTURE_FORCE_ABSOLUTE_G
            and int(force_delta[axis]) >= _CONTACT_CAPTURE_FORCE_DELTA_G
        )
        config = self.config
        eligible = bool(
            self._mpc_capture_enabled
            and config.collision_replan
            and self._committed_adaptive_phase == "approach"
            and self._committed_template_index == config.template_catch_index - 1
            and self._committed_closure_step == -1
            and self._committed_live_fit_seen
            and diagnostics.get("capture_preshape_ready") is True
            and diagnostics.get("capture_current_geometry_safe") is True
            and tip_clearance >= config.minimum_fingertip_clearance_m
            and planar_shell <= _CONTACT_CAPTURE_PLANAR_SHELL_M
            and abs(measured_z) <= _CONTACT_CAPTURE_Z_SHELL_M
            and 0.0 <= speed
            <= float(config.adaptive_speed_max_m_s) + 1.0e-6
            and len(contact_axes) >= _CONTACT_CAPTURE_MINIMUM_FINGER_AXES
        )
        diagnostics.update(
            {
                "force_contact_capture_candidate": bool(eligible),
                "force_contact_axes": list(contact_axes),
                "force_contact_current_g": forces.tolist(),
                "force_contact_delta_g": force_delta.tolist(),
                "force_contact_planar_error_m": planar_shell,
                "force_contact_z_error_m": measured_z,
            }
        )
        if not eligible:
            return

        capture_q = self._committed_target_q.copy()
        lift_q = pose_preserving_joint_target(
            capture_q,
            np.asarray([0.0, 0.0, config.grasp_lift_m]),
            T_flange_policy_palm=self.contract.T_flange_policy_palm,
            joint_limits_rad=self.contract.joint_limits_rad,
            joint_limit_margin_rad=config.joint_limit_margin_rad,
            damping=config.dls_damping,
            local_precision_refinement=True,
        )
        self._capture_q = capture_q
        self._lift_q = lift_q
        self._committed_adaptive_phase = "closure"
        self._committed_full_occluded_closure = True
        self._committed_current_volume_capture = True
        self._committed_force_contact_capture = True
        diagnostics.update(
            {
                "state": "adaptive_force_contact_capture_committed",
                "adaptive_phase": "closure",
                "capture_gate_accepted": True,
                "full_occluded_closure_committed": True,
                "current_volume_capture_committed": True,
                "force_contact_capture_committed": True,
                "capture_target_committed_after_dual_ack": True,
                "lift_goal_q_rad": lift_q.tolist(),
            }
        )
        if self._committed_diagnostics_trace:
            tail = self._committed_diagnostics_trace[-1]
            if int(tail.get("sequence", -1)) != current_sequence:
                raise RuntimeError(
                    "RH56 contact feedback lost its policy trace binding"
                )
            tail.update(
                {
                    "state": diagnostics["state"],
                    "adaptive_phase": "closure",
                    "capture_gate_accepted": True,
                    "full_occluded_closure_committed": True,
                    "current_volume_capture_committed": True,
                    "force_contact_capture_committed": True,
                    "force_contact_axes": list(contact_axes),
                    "force_contact_planar_error_m": planar_shell,
                    "force_contact_z_error_m": measured_z,
                }
            )
        print(
            "[Tabletop capture CONTACT] fresh near-hand object plus "
            f"multi-finger force confirmed axes={list(contact_axes)}; "
            "Franka frozen, RH56 closure starting",
            flush=True,
        )

    def discard_replay_proposal(self, sequence: int) -> None:
        if self._pending is None or self._pending.sequence != int(sequence):
            raise RuntimeError("online planner discard differs from its proposal")
        self._pending = None

    def act(
        self,
        pointcloud: np.ndarray,
        valid: np.ndarray,
        proprio: np.ndarray,
    ) -> ReplayPolicyOutput:
        raise RuntimeError(
            "transactional online planner must be addressed by sequence"
        )


__all__ = [
    "TransactionalTabletopOnlinePlannerPolicy",
    "cartesian_joint_correction",
    "panda_T_base_policy_palm",
    "pose_preserving_joint_target",
]
