"""Hardware-free policy/action scaffold to be replaced by the simulation ckpt."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .contracts import FRANKA_DOF, INSPIRE_AXES, ObservationSample


@dataclass(frozen=True)
class BaselineAction:
    """Stage-one action contract.

    ``arm_joint_delta_rad`` is a per-policy-tick delta.  ``hand_targets`` uses
    the raw RH56 six-axis order; ``-1`` means no numeric target.  This contract
    is deliberately local to the dry-run baseline and is not claimed to match
    the future simulation checkpoint.
    """

    arm_joint_delta_rad: np.ndarray
    hand_targets: np.ndarray
    source: str


@dataclass(frozen=True)
class LimitedAction:
    arm_joint_delta_rad: np.ndarray
    arm_target_q_rad: np.ndarray
    hand_targets: np.ndarray
    clipped: bool
    reasons: Tuple[str, ...]


class BaselinePolicy:
    """Three deterministic policies for testing wiring without a checkpoint."""

    MODES = ("hold", "arm-sine", "preview-combined")

    def __init__(
        self,
        mode: str,
        *,
        policy_hz: float,
        arm_amplitude_rad: float = 0.01,
        arm_period_s: float = 4.0,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unknown baseline mode {mode!r}")
        if not np.isfinite(policy_hz) or policy_hz <= 0.0:
            raise ValueError("policy_hz must be finite and positive")
        if not np.isfinite(arm_amplitude_rad) or arm_amplitude_rad < 0.0:
            raise ValueError("arm_amplitude_rad must be finite and non-negative")
        if not np.isfinite(arm_period_s) or arm_period_s <= 0.0:
            raise ValueError("arm_period_s must be finite and positive")
        self.mode = mode
        self.policy_hz = float(policy_hz)
        self.arm_amplitude_rad = float(arm_amplitude_rad)
        self.arm_period_s = float(arm_period_s)
        self._last_offset_rad = 0.0

    def reset(self) -> None:
        self._last_offset_rad = 0.0

    def act(
        self, observation: Optional[ObservationSample], *, elapsed_s: float
    ) -> BaselineAction:
        if observation is not None and not observation.valid:
            return self._hold("invalid-observation-hold")
        if self.mode == "hold":
            return self._hold("baseline-hold")

        omega = 2.0 * math.pi / self.arm_period_s
        offset = self.arm_amplitude_rad * math.sin(omega * float(elapsed_s))
        delta = offset - self._last_offset_rad
        self._last_offset_rad = offset
        arm = np.zeros(FRANKA_DOF, dtype=np.float64)
        # Joint 7 is chosen only for an obvious offline signal.  This is not a
        # statement that q7 motion is collision-cleared on the installed hand.
        arm[6] = delta
        hand = np.full(len(INSPIRE_AXES), -1, dtype=np.int32)
        if self.mode == "preview-combined":
            phase = 0.5 + 0.5 * math.sin(omega * float(elapsed_s))
            hand[:5] = int(round(1000.0 - 100.0 * phase))
            # q6 stays at the commissioned open-side endpoint in this preview.
            hand[5] = 1000
        return BaselineAction(arm, hand, f"baseline-{self.mode}")

    @staticmethod
    def _hold(source: str) -> BaselineAction:
        return BaselineAction(
            np.zeros(FRANKA_DOF, dtype=np.float64),
            np.full(len(INSPIRE_AXES), -1, dtype=np.int32),
            source,
        )


class ActionLimiter:
    """Validate and clip baseline actions before any future executor sees them."""

    def __init__(
        self,
        *,
        initial_q_rad: Sequence[float],
        joint_limits_rad: Sequence[Sequence[float]],
        joint_limit_margin_rad: float,
        max_arm_step_rad: float,
        max_arm_episode_delta_rad: float,
        initial_hand_angles: Sequence[int],
        max_hand_step_units: int,
    ) -> None:
        self.initial_q = np.asarray(initial_q_rad, dtype=np.float64)
        self.command_q = self.initial_q.copy()
        self.joint_limits = np.asarray(joint_limits_rad, dtype=np.float64)
        self.margin = float(joint_limit_margin_rad)
        self.max_arm_step = float(max_arm_step_rad)
        self.max_arm_episode_delta = float(max_arm_episode_delta_rad)
        self.command_hand = np.asarray(initial_hand_angles, dtype=np.int32)
        self.max_hand_step = int(max_hand_step_units)
        if self.initial_q.shape != (7,) or not np.all(np.isfinite(self.initial_q)):
            raise ValueError("initial_q_rad must contain seven finite values")
        if self.joint_limits.shape != (7, 2) or not np.all(
            np.isfinite(self.joint_limits)
        ):
            raise ValueError("joint_limits_rad must be a finite (7,2) array")
        if (
            self.command_hand.shape != (6,)
            or np.any(self.command_hand < 0)
            or np.any(self.command_hand > 1000)
        ):
            raise ValueError("initial_hand_angles must contain six values in 0..1000")
        for name, value in (
            ("joint_limit_margin_rad", self.margin),
            ("max_arm_step_rad", self.max_arm_step),
            ("max_arm_episode_delta_rad", self.max_arm_episode_delta),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_hand_step <= 0:
            raise ValueError("max_hand_step_units must be positive")
        lower = self.joint_limits[:, 0] + self.margin
        upper = self.joint_limits[:, 1] - self.margin
        if np.any(self.initial_q <= lower) or np.any(self.initial_q >= upper):
            raise ValueError("initial_q_rad violates the joint-limit margin")

    def apply(self, action: BaselineAction) -> LimitedAction:
        reasons = []
        delta = np.asarray(action.arm_joint_delta_rad, dtype=np.float64)
        if delta.shape != (7,) or not np.all(np.isfinite(delta)):
            raise ValueError("arm_joint_delta_rad must contain seven finite values")
        clipped_delta = np.clip(delta, -self.max_arm_step, self.max_arm_step)
        if not np.array_equal(clipped_delta, delta):
            reasons.append("arm per-step delta clipped")
        target = self.command_q + clipped_delta
        episode_lower = self.initial_q - self.max_arm_episode_delta
        episode_upper = self.initial_q + self.max_arm_episode_delta
        safe_lower = np.maximum(episode_lower, self.joint_limits[:, 0] + self.margin)
        safe_upper = np.minimum(episode_upper, self.joint_limits[:, 1] - self.margin)
        clipped_target = np.clip(target, safe_lower, safe_upper)
        if not np.array_equal(clipped_target, target):
            reasons.append("arm episode/joint-limit envelope clipped")
        applied_delta = clipped_target - self.command_q
        self.command_q = clipped_target

        hand = np.asarray(action.hand_targets)
        if (
            hand.shape != (6,)
            or not np.all(np.isfinite(hand.astype(np.float64)))
            or not np.array_equal(
                hand.astype(np.float64), np.rint(hand.astype(np.float64))
            )
        ):
            raise ValueError("hand_targets must contain six finite integers")
        hand = hand.astype(np.int32)
        if np.any(hand < -1) or np.any(hand > 1000):
            raise ValueError("hand_targets must be -1 or lie in 0..1000")
        limited_hand = np.full(6, -1, dtype=np.int32)
        for index, requested in enumerate(hand):
            if requested == -1:
                continue
            change = int(requested) - int(self.command_hand[index])
            applied = int(np.clip(change, -self.max_hand_step, self.max_hand_step))
            if applied != change:
                reasons.append(f"hand {INSPIRE_AXES[index]} rate clipped")
            self.command_hand[index] += applied
            limited_hand[index] = self.command_hand[index]
        return LimitedAction(
            arm_joint_delta_rad=applied_delta.copy(),
            arm_target_q_rad=self.command_q.copy(),
            hand_targets=limited_hand,
            clipped=bool(reasons),
            reasons=tuple(reasons),
        )


__all__ = ["ActionLimiter", "BaselineAction", "BaselinePolicy", "LimitedAction"]
