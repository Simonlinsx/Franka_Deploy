"""Transactional action-replay policy and intercept state machine."""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from .models import ReplayActionSequence, ReplayPolicyOutput


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

__all__ = ["TransactionalReplayActionPolicy"]
