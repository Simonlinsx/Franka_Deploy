"""Immutable commands, proposals, commits, and sample/hold transport."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Optional

import numpy as np

from sim2real.contracts.actions import V94MappedAction

from .authorization import ClosedLoopProtocolError
from .validation import (
    _finite_scalar,
    _immutable_mapped_action,
    _positive_integer,
    _readonly_float_vector,
    _readonly_register_vector,
    _strict_bool,
)


@dataclass(frozen=True)
class ClosedLoopCommand:
    """One policy command shared by both actuator owners."""

    sequence: int
    produced_monotonic_s: float
    observation_realtime_s: float
    previous_executed_action13_used: np.ndarray
    raw_policy_action13: np.ndarray
    executed_policy_action13: np.ndarray
    franka_target_q_rad: np.ndarray
    rh56_angle_set_register_order: np.ndarray
    hold_arm_target: bool = False
    # Normally identical to ``previous_executed_action13_used``.  A reset-
    # gated policy may, however, advance its recurrent/history previous-action
    # input for non-actuated startup ticks while the hardware dual-ACK ledger
    # correctly remains at its initial value.  Keeping both values explicit
    # avoids forging actuator commits merely to reproduce training startup.
    previous_policy_action13_used: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        sequence = _positive_integer(self.sequence, "command sequence")
        produced = _finite_scalar(self.produced_monotonic_s, "produced_monotonic_s")
        observation = _finite_scalar(
            self.observation_realtime_s, "observation_realtime_s"
        )
        previous = _readonly_float_vector(
            self.previous_executed_action13_used,
            13,
            "previous_executed_action13_used",
        )
        raw = _readonly_float_vector(
            self.raw_policy_action13, 13, "raw_policy_action13"
        )
        executed = _readonly_float_vector(
            self.executed_policy_action13, 13, "executed_policy_action13"
        )
        if np.any(executed < -1.0) or np.any(executed > 1.0):
            raise ValueError("executed_policy_action13 must lie in [-1,1]")
        arm = _readonly_float_vector(
            self.franka_target_q_rad, 7, "franka_target_q_rad"
        )
        hand = _readonly_register_vector(
            self.rh56_angle_set_register_order,
            "rh56_angle_set_register_order",
        )
        hold_arm = _strict_bool(self.hold_arm_target, "hold_arm_target")
        policy_previous = (
            previous
            if self.previous_policy_action13_used is None
            else _readonly_float_vector(
                self.previous_policy_action13_used,
                13,
                "previous_policy_action13_used",
            )
        )
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "produced_monotonic_s", produced)
        object.__setattr__(self, "observation_realtime_s", observation)
        object.__setattr__(self, "previous_executed_action13_used", previous)
        object.__setattr__(self, "raw_policy_action13", raw)
        object.__setattr__(self, "executed_policy_action13", executed)
        object.__setattr__(self, "franka_target_q_rad", arm)
        object.__setattr__(self, "rh56_angle_set_register_order", hand)
        object.__setattr__(self, "hold_arm_target", hold_arm)
        object.__setattr__(
            self, "previous_policy_action13_used", policy_previous
        )


@dataclass(frozen=True)
class ExecutedActionCommit:
    """Atomic view of the latest dual-acknowledged policy action."""

    sequence: int
    executed_policy_action13: np.ndarray

    def __post_init__(self) -> None:
        sequence = int(self.sequence)
        if sequence < 0:
            raise ValueError("committed sequence cannot be negative")
        action = _readonly_float_vector(
            self.executed_policy_action13, 13, "executed_policy_action13"
        )
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "executed_policy_action13", action)


@dataclass(frozen=True)
class TransactionalV94ActionProposal:
    """Immutable, rollback-safe result of one V94 target proposal."""

    sequence: int
    mapped: V94MappedAction
    prior_arm_target_q_rad: np.ndarray
    prior_hand_target_q_policy_order_rad: np.ndarray
    arm_target_rate_rad_s: np.ndarray
    hand_target_rate_rad_s: np.ndarray

    def __post_init__(self) -> None:
        sequence = _positive_integer(self.sequence, "proposal sequence")
        mapped = _immutable_mapped_action(self.mapped)
        prior_arm = _readonly_float_vector(
            self.prior_arm_target_q_rad, 7, "prior_arm_target_q_rad"
        )
        prior_hand = _readonly_float_vector(
            self.prior_hand_target_q_policy_order_rad,
            6,
            "prior_hand_target_q_policy_order_rad",
        )
        arm_rate = _readonly_float_vector(
            self.arm_target_rate_rad_s, 7, "arm_target_rate_rad_s"
        )
        hand_rate = _readonly_float_vector(
            self.hand_target_rate_rad_s, 6, "hand_target_rate_rad_s"
        )
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "mapped", mapped)
        object.__setattr__(self, "prior_arm_target_q_rad", prior_arm)
        object.__setattr__(
            self, "prior_hand_target_q_policy_order_rad", prior_hand
        )
        object.__setattr__(self, "arm_target_rate_rad_s", arm_rate)
        object.__setattr__(self, "hand_target_rate_rad_s", hand_rate)


class PolicyCommandSampleHold:
    """Atomic latest command sampled repeatedly by the 1 kHz arm loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._command: Optional[ClosedLoopCommand] = None

    def publish(self, command: ClosedLoopCommand) -> None:
        if not isinstance(command, ClosedLoopCommand):
            raise TypeError("command must be ClosedLoopCommand")
        with self._lock:
            expected = 1 if self._command is None else self._command.sequence + 1
            if command.sequence != expected:
                raise ClosedLoopProtocolError(
                    f"command sequence gap: expected={expected}, actual={command.sequence}"
                )
            self._command = command

    def sample(
        self, *, now_monotonic_s: object, maximum_age_s: object
    ) -> ClosedLoopCommand:
        now = _finite_scalar(now_monotonic_s, "now_monotonic_s")
        maximum_age = _finite_scalar(maximum_age_s, "maximum_age_s")
        if maximum_age <= 0.0:
            raise ValueError("maximum_age_s must be positive")
        with self._lock:
            command = self._command
        if command is None:
            raise ClosedLoopProtocolError("no policy command is available")
        age = now - command.produced_monotonic_s
        if age < 0.0:
            raise ClosedLoopProtocolError("policy command timestamp is in the future")
        if age > maximum_age:
            raise ClosedLoopProtocolError(
                f"policy command is stale: age={age:.6f}s, limit={maximum_age:.6f}s"
            )
        return command

__all__ = [
    "ClosedLoopCommand",
    "ExecutedActionCommit",
    "PolicyCommandSampleHold",
    "TransactionalV94ActionProposal",
]
