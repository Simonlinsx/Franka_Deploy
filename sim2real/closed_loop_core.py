"""Hardware-free safety protocol for a future V94 closed-loop executor.

This module deliberately contains no device imports and no actuator calls.  It
defines the concurrency and state contracts that the eventual hardware entry
point must satisfy:

* motion starts disarmed and needs a run-scoped, expiring authorization;
* deadman/E-stop loss and every runtime failure latch the session terminally;
* a 60 Hz policy command is sampled-and-held by a 1 kHz Franka loop;
* Franka and RH56 must acknowledge the same command before its action becomes
  the ``previous_executed_action13`` used by the next policy observation; and
* an RH56 implementation can bind all serial access to one owner thread.

The protocol is intentionally stricter than a latest-value queue.  There is at
most one command in flight, so a slow hand writer cannot silently skip policy
actions while the Franka loop consumes them.  Failure to meet that contract is
a commissioning result, not permission to weaken the action semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import threading
from typing import Optional, Tuple

import numpy as np

from sim2real.contracts.v94 import INITIAL_PREVIOUS_ACTION13
from sim2real.contracts.actions import (
    LEGACY_FRANKA_ACTION_CONTRACT_ID,
    QD_G015_FRANKA_ACTION_CONTRACT_ID,
    V94ActionMapper,
    V94MappedAction,
    contracted_joint_limits_float64,
    inward_float32_joint_limits,
    normalize_franka_action_contract_id,
)


AUTHORIZATION_SCOPE = "v94_closed_loop_franka_rh56"
COMMAND_CONSUMERS: Tuple[str, str] = ("franka", "rh56")


class ClosedLoopProtocolError(RuntimeError):
    """A fail-closed safety or command-protocol violation."""


class SafetyState(str, Enum):
    DISARMED = "disarmed"
    ARMED = "armed"
    FAULT_LATCHED = "fault_latched"


def _finite_scalar(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


def _readonly_float_vector(value: object, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    result = result.copy()
    result.setflags(write=False)
    return result


def _readonly_register_vector(value: object, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (6,) or not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must contain six finite integer values")
    numeric = raw.astype(np.float64)
    if not np.all(numeric == np.rint(numeric)):
        raise ValueError(f"{name} must contain integer values")
    result = numeric.astype(np.int32)
    if np.any(result < 0) or np.any(result > 1000):
        raise ValueError(f"{name} must lie in [0,1000]")
    result.setflags(write=False)
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric <= 0.0:
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _optional_positive_rate_vector(
    value: object, size: int, name: str
) -> Optional[np.ndarray]:
    if value is None:
        return None
    raw = np.asarray(value, dtype=np.float64)
    if raw.shape == ():
        raw = np.full(size, float(raw), dtype=np.float64)
    if raw.shape != (size,) or not np.all(np.isfinite(raw)) or np.any(raw <= 0.0):
        raise ValueError(f"{name} must be a positive scalar or {size}-vector")
    result = raw.copy()
    result.setflags(write=False)
    return result


def _immutable_mapped_action(value: V94MappedAction) -> V94MappedAction:
    if not isinstance(value, V94MappedAction):
        raise TypeError("mapped action must be V94MappedAction")
    return V94MappedAction(
        raw_policy_action13=_readonly_float_vector(
            value.raw_policy_action13, 13, "raw_policy_action13"
        ),
        executed_policy_action13=_readonly_float_vector(
            value.executed_policy_action13, 13, "executed_policy_action13"
        ),
        franka_target_q_rad=_readonly_float_vector(
            value.franka_target_q_rad, 7, "franka_target_q_rad"
        ),
        rh56_target_q_policy_order_rad=_readonly_float_vector(
            value.rh56_target_q_policy_order_rad,
            6,
            "rh56_target_q_policy_order_rad",
        ),
        rh56_angle_set_register_order=_readonly_register_vector(
            value.rh56_angle_set_register_order,
            "rh56_angle_set_register_order",
        ),
        clipped=bool(value.clipped),
        reasons=tuple(str(reason) for reason in value.reasons),
        hold_arm_target=bool(value.hold_arm_target),
    )


@dataclass(frozen=True)
class MotionAuthorization:
    """Run-scoped proof emitted by a trusted operator-approval boundary.

    Creating this value is not itself an operator prompt.  The hardware CLI
    must create it only after explicit user approval and must persist
    ``authorization_id`` in the run audit.  Keeping that boundary outside this
    hardware-free module makes it impossible for unit tests or imports to open
    a device accidentally.
    """

    run_id: str
    authorization_id: str
    issued_monotonic_s: float
    expires_monotonic_s: float
    scope: str = AUTHORIZATION_SCOPE

    def __post_init__(self) -> None:
        run_id = str(self.run_id).strip()
        authorization_id = str(self.authorization_id).strip()
        scope = str(self.scope).strip()
        issued = _finite_scalar(self.issued_monotonic_s, "issued_monotonic_s")
        expires = _finite_scalar(self.expires_monotonic_s, "expires_monotonic_s")
        if not run_id:
            raise ValueError("run_id must be non-empty")
        if not authorization_id:
            raise ValueError("authorization_id must be non-empty")
        if scope != AUTHORIZATION_SCOPE:
            raise ValueError("motion authorization scope is invalid")
        if expires <= issued:
            raise ValueError("motion authorization must expire after it is issued")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "authorization_id", authorization_id)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "issued_monotonic_s", issued)
        object.__setattr__(self, "expires_monotonic_s", expires)


class ClosedLoopSafetyGate:
    """Sticky, thread-safe authorization/deadman/E-stop gate.

    A fault cannot be cleared in place.  Recovery requires verified Franka
    rest, verified RH56 disable, process teardown, a new run ID, and a new
    operator authorization.  This prevents an exception handler from
    accidentally re-arming a partially failed session.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = SafetyState.DISARMED
        self._deadman_asserted: Optional[bool] = None
        self._estop_healthy: Optional[bool] = None
        self._authorization: Optional[MotionAuthorization] = None
        self._fault_reason: Optional[str] = None
        self._used_authorization_ids: set[str] = set()
        self._last_motion_check_monotonic_s: Optional[float] = None
        # Gate checks come from the policy/RH56 owner and the persistent
        # Franka owner concurrently.  Comparing their pre-call clock samples
        # against one process-global value can falsely report regression when
        # thread A samples first but thread B acquires this lock first.  Keep
        # the audit high-water mark above, while enforcing regression within
        # each caller thread where invocation order is well-defined.
        self._last_motion_check_by_thread: dict[threading.Thread, float] = {}

    @property
    def state(self) -> SafetyState:
        with self._lock:
            return self._state

    @property
    def fault_reason(self) -> Optional[str]:
        with self._lock:
            return self._fault_reason

    @property
    def authorization_id(self) -> Optional[str]:
        with self._lock:
            return (
                None
                if self._authorization is None
                else self._authorization.authorization_id
            )

    def _latch_locked(self, reason: str) -> None:
        detail = str(reason).strip()
        if not detail:
            detail = "unspecified closed-loop fault"
        if self._state is not SafetyState.FAULT_LATCHED:
            self._fault_reason = detail
        self._authorization = None
        self._state = SafetyState.FAULT_LATCHED

    def latch_fault(self, reason: str) -> None:
        with self._lock:
            self._latch_locked(reason)

    def update_interlocks(
        self, *, deadman_asserted: object, estop_healthy: object
    ) -> None:
        deadman = _strict_bool(deadman_asserted, "deadman_asserted")
        estop = _strict_bool(estop_healthy, "estop_healthy")
        with self._lock:
            self._deadman_asserted = deadman
            self._estop_healthy = estop
            if self._state is SafetyState.ARMED and (not deadman or not estop):
                reason = (
                    "external emergency stop opened"
                    if not estop
                    else "external deadman was released"
                )
                self._latch_locked(reason)
                raise ClosedLoopProtocolError(reason)

    def arm(
        self,
        authorization: MotionAuthorization,
        *,
        run_id: str,
        now_monotonic_s: object,
        franka_rest_verified: object,
        rh56_disabled_verified: object,
    ) -> None:
        if not isinstance(authorization, MotionAuthorization):
            raise TypeError("authorization must be MotionAuthorization")
        active_run_id = str(run_id).strip()
        now = _finite_scalar(now_monotonic_s, "now_monotonic_s")
        franka_safe = _strict_bool(franka_rest_verified, "franka_rest_verified")
        hand_safe = _strict_bool(rh56_disabled_verified, "rh56_disabled_verified")
        with self._lock:
            if self._state is SafetyState.FAULT_LATCHED:
                raise ClosedLoopProtocolError(
                    f"closed-loop fault is latched: {self._fault_reason}"
                )
            if self._state is not SafetyState.DISARMED:
                raise ClosedLoopProtocolError("closed-loop gate is already armed")
            if authorization.run_id != active_run_id:
                raise ClosedLoopProtocolError("motion authorization run ID mismatch")
            if authorization.authorization_id in self._used_authorization_ids:
                raise ClosedLoopProtocolError("motion authorization was already consumed")
            if not (
                authorization.issued_monotonic_s
                <= now
                < authorization.expires_monotonic_s
            ):
                raise ClosedLoopProtocolError("motion authorization is not active")
            if self._deadman_asserted is not True:
                raise ClosedLoopProtocolError("external deadman is not asserted")
            if self._estop_healthy is not True:
                raise ClosedLoopProtocolError("external emergency stop is not healthy")
            if not franka_safe:
                raise ClosedLoopProtocolError("Franka rest is not verified")
            if not hand_safe:
                raise ClosedLoopProtocolError("RH56 disabled state is not verified")
            self._authorization = authorization
            self._used_authorization_ids.add(authorization.authorization_id)
            self._last_motion_check_monotonic_s = now
            self._last_motion_check_by_thread = {
                threading.current_thread(): now,
            }
            self._state = SafetyState.ARMED

    def require_motion(self, *, run_id: str, now_monotonic_s: object) -> None:
        active_run_id = str(run_id).strip()
        now = _finite_scalar(now_monotonic_s, "now_monotonic_s")
        caller = threading.current_thread()
        with self._lock:
            if self._state is SafetyState.FAULT_LATCHED:
                raise ClosedLoopProtocolError(
                    f"closed-loop fault is latched: {self._fault_reason}"
                )
            if self._state is not SafetyState.ARMED or self._authorization is None:
                raise ClosedLoopProtocolError("closed-loop motion is not authorized")
            if self._authorization.run_id != active_run_id:
                self._latch_locked("active run ID changed after authorization")
                raise ClosedLoopProtocolError("active run ID changed after authorization")
            prior_for_caller = self._last_motion_check_by_thread.get(caller)
            if prior_for_caller is not None and now < prior_for_caller:
                self._latch_locked("motion authorization clock regressed")
                raise ClosedLoopProtocolError("motion authorization clock regressed")
            self._last_motion_check_by_thread[caller] = now
            self._last_motion_check_monotonic_s = max(
                now,
                self._last_motion_check_monotonic_s
                if self._last_motion_check_monotonic_s is not None
                else now,
            )
            if now >= self._authorization.expires_monotonic_s:
                self._latch_locked("motion authorization expired")
                raise ClosedLoopProtocolError("motion authorization expired")
            if self._deadman_asserted is not True or self._estop_healthy is not True:
                self._latch_locked("external motion interlock is not healthy")
                raise ClosedLoopProtocolError(
                    "external motion interlock is not healthy"
                )

    def disarm_after_verified_stop(
        self,
        *,
        run_id: str,
        franka_stop_verified: object,
        rh56_disabled_verified: object,
    ) -> None:
        active_run_id = str(run_id).strip()
        franka_safe = _strict_bool(franka_stop_verified, "franka_stop_verified")
        hand_safe = _strict_bool(rh56_disabled_verified, "rh56_disabled_verified")
        with self._lock:
            if self._state is SafetyState.FAULT_LATCHED:
                raise ClosedLoopProtocolError(
                    "a fault-latched session cannot be re-used after stop"
                )
            if self._state is not SafetyState.ARMED or self._authorization is None:
                raise ClosedLoopProtocolError("closed-loop gate is not armed")
            if self._authorization.run_id != active_run_id:
                raise ClosedLoopProtocolError("active run ID mismatch during disarm")
            if not franka_safe or not hand_safe:
                reason = (
                    "both Franka stop and RH56 disable must be verified before disarm"
                )
                self._latch_locked(reason)
                raise ClosedLoopProtocolError(reason)
            self._authorization = None
            self._last_motion_check_monotonic_s = None
            self._last_motion_check_by_thread.clear()
            self._state = SafetyState.DISARMED


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


class ExecutedActionLedger:
    """Two-consumer commit barrier for exact previous-action semantics.

    ``previous_executed_action13`` advances only after the first Franka
    ``writeOnce`` for a sequence and the verified RH56 target write for that
    same sequence have both succeeded.  A partial or late commit becomes a
    terminal protocol fault and must trigger both device stop paths.
    """

    def __init__(
        self,
        *,
        maximum_commit_latency_s: float,
        initial_previous_action13: np.ndarray = INITIAL_PREVIOUS_ACTION13,
    ) -> None:
        maximum_latency = _finite_scalar(
            maximum_commit_latency_s, "maximum_commit_latency_s"
        )
        if maximum_latency <= 0.0:
            raise ValueError("maximum_commit_latency_s must be positive")
        self.maximum_commit_latency_s = maximum_latency
        self._lock = threading.Lock()
        self._previous = _readonly_float_vector(
            initial_previous_action13, 13, "initial_previous_action13"
        )
        self._last_committed_sequence = 0
        self._pending: Optional[ClosedLoopCommand] = None
        self._acknowledged: set[str] = set()
        self._fault_reason: Optional[str] = None

    @property
    def fault_reason(self) -> Optional[str]:
        with self._lock:
            return self._fault_reason

    @property
    def last_committed_sequence(self) -> int:
        with self._lock:
            return self._last_committed_sequence

    @property
    def pending_sequence(self) -> Optional[int]:
        with self._lock:
            return None if self._pending is None else self._pending.sequence

    def consumer_acknowledged(self, consumer: str, *, sequence: int) -> bool:
        """Return whether one exact consumer ACKed an exact sequence.

        The check and pending-sequence comparison share the ledger lock.  It
        is therefore safe for an orchestrator to use as a one-device barrier
        before allowing the other actuator to start its transaction.  A
        sequence that has already dual-committed necessarily has both ACKs;
        any other non-pending sequence is a protocol error rather than a
        best-effort answer.
        """

        name = str(consumer).strip().lower()
        if name not in COMMAND_CONSUMERS:
            raise ValueError(f"consumer must be one of {COMMAND_CONSUMERS}")
        acknowledged_sequence = _positive_integer(
            sequence, "acknowledged sequence"
        )
        with self._lock:
            self._require_healthy_locked()
            if acknowledged_sequence == self._last_committed_sequence:
                return True
            command = self._pending
            if command is None or command.sequence != acknowledged_sequence:
                pending = None if command is None else command.sequence
                raise ClosedLoopProtocolError(
                    "acknowledgement query sequence is neither pending nor "
                    "the last committed sequence: "
                    f"query={acknowledged_sequence}, pending={pending}, "
                    f"committed={self._last_committed_sequence}"
                )
            return name in self._acknowledged

    def previous_executed_action13(self) -> np.ndarray:
        with self._lock:
            return self._previous.copy()

    def committed_snapshot(self) -> ExecutedActionCommit:
        """Return sequence and action under one lock acquisition."""

        with self._lock:
            return ExecutedActionCommit(
                sequence=self._last_committed_sequence,
                executed_policy_action13=self._previous,
            )

    def _require_healthy_locked(self) -> None:
        if self._fault_reason is not None:
            raise ClosedLoopProtocolError(
                f"action commit ledger fault is latched: {self._fault_reason}"
            )

    def _fault_locked(self, reason: str) -> None:
        if self._fault_reason is None:
            self._fault_reason = str(reason).strip() or "unspecified commit fault"

    def stage(
        self, command: ClosedLoopCommand, *, now_monotonic_s: object
    ) -> None:
        if not isinstance(command, ClosedLoopCommand):
            raise TypeError("command must be ClosedLoopCommand")
        now = _finite_scalar(now_monotonic_s, "now_monotonic_s")
        with self._lock:
            self._require_healthy_locked()
            if self._pending is not None:
                reason = "cannot stage a new command while another is uncommitted"
                self._fault_locked(reason)
                raise ClosedLoopProtocolError(reason)
            expected_sequence = self._last_committed_sequence + 1
            if command.sequence != expected_sequence:
                reason = "command sequence does not follow the last committed command"
                self._fault_locked(reason)
                raise ClosedLoopProtocolError(reason)
            if not np.array_equal(
                command.previous_executed_action13_used, self._previous
            ):
                reason = "policy observation used an uncommitted previous action"
                self._fault_locked(reason)
                raise ClosedLoopProtocolError(reason)
            age = now - command.produced_monotonic_s
            if age < 0.0:
                raise ClosedLoopProtocolError("staged command timestamp is in the future")
            if age > self.maximum_commit_latency_s:
                self._fault_locked("command was already too old when staged")
                raise ClosedLoopProtocolError(self._fault_reason)
            self._pending = command
            self._acknowledged.clear()

    def acknowledge(
        self,
        consumer: str,
        *,
        sequence: int,
        now_monotonic_s: object,
    ) -> bool:
        name = str(consumer).strip().lower()
        if name not in COMMAND_CONSUMERS:
            raise ValueError(f"consumer must be one of {COMMAND_CONSUMERS}")
        now = _finite_scalar(now_monotonic_s, "now_monotonic_s")
        with self._lock:
            self._require_healthy_locked()
            command = self._pending
            if command is None:
                raise ClosedLoopProtocolError("there is no command awaiting commit")
            acknowledged_sequence = _positive_integer(
                sequence, "acknowledged sequence"
            )
            if acknowledged_sequence != command.sequence:
                self._fault_locked(
                    f"{name} acknowledged sequence {acknowledged_sequence}, "
                    f"expected {command.sequence}"
                )
                raise ClosedLoopProtocolError(self._fault_reason)
            latency = now - command.produced_monotonic_s
            if latency < 0.0:
                self._fault_locked(f"{name} acknowledgement timestamp is in the future")
                raise ClosedLoopProtocolError(self._fault_reason)
            if latency > self.maximum_commit_latency_s:
                self._fault_locked(
                    f"{name} acknowledgement exceeded commit latency: "
                    f"actual={latency:.6f}s, "
                    f"limit={self.maximum_commit_latency_s:.6f}s"
                )
                raise ClosedLoopProtocolError(self._fault_reason)
            self._acknowledged.add(name)
            if self._acknowledged != set(COMMAND_CONSUMERS):
                return False
            self._previous = _readonly_float_vector(
                command.executed_policy_action13,
                13,
                "committed executed_policy_action13",
            )
            self._last_committed_sequence = command.sequence
            self._pending = None
            self._acknowledged.clear()
            return True

    def require_commit_deadline(self, *, now_monotonic_s: object) -> None:
        now = _finite_scalar(now_monotonic_s, "now_monotonic_s")
        with self._lock:
            self._require_healthy_locked()
            if self._pending is None:
                return
            latency = now - self._pending.produced_monotonic_s
            if latency < 0.0:
                self._fault_locked("commit deadline clock regressed")
                raise ClosedLoopProtocolError(self._fault_reason)
            if latency > self.maximum_commit_latency_s:
                missing = sorted(set(COMMAND_CONSUMERS) - self._acknowledged)
                self._fault_locked(
                    "command commit deadline expired; missing acknowledgements="
                    + ",".join(missing)
                )
                raise ClosedLoopProtocolError(self._fault_reason)

    def fail(self, consumer: str, *, sequence: int, reason: str) -> None:
        name = str(consumer).strip().lower()
        if name not in COMMAND_CONSUMERS:
            raise ValueError(f"consumer must be one of {COMMAND_CONSUMERS}")
        detail = str(reason).strip() or "unspecified actuator failure"
        with self._lock:
            pending = self._pending
            expected = None if pending is None else pending.sequence
            self._fault_locked(
                f"{name} failed sequence {int(sequence)} "
                f"(pending={expected}): {detail}"
            )


class TransactionalV94ActionMapper:
    """Rollback-safe wrapper around the exact V94 target mapping.

    A proposal runs the existing mapper in a temporary state.  The persistent
    arm/hand targets are updated only when :meth:`commit` observes an atomic
    :class:`ExecutedActionCommit` for the same proposal.  That commit can only
    be produced by :class:`ExecutedActionLedger` after both actuator owners
    acknowledge the command.

    ``joint_limit_margin_rad`` contracts the supplied limits before they reach
    :class:`V94ActionMapper`, matching the safe interval enforced by the FCI
    loop.  For the legacy accumulated-target contract, optional commissioned
    rate vectors reject, rather than silently reshape, a target transition
    that exceeds the verified policy-rate envelope.  The q_d g015 contract
    intentionally has no high-level target-rate guard; its separately bounded
    1 kHz generator owns velocity/acceleration/jerk shaping.
    """

    def __init__(
        self,
        *,
        initial_arm_target_q_rad: np.ndarray,
        initial_hand_target_q_policy_order_rad: np.ndarray,
        joint_limits_rad: np.ndarray,
        control_dt_s: float,
        q_hand_close_rad: np.ndarray,
        joint_limit_margin_rad: float = 0.0,
        commissioned_max_arm_target_rate_rad_s: object = None,
        commissioned_max_hand_target_rate_rad_s: object = None,
        measured_arm_envelope_rad: float = 0.05,
        arm_raw_gain_rad: float = 0.015,
        target_filter_alpha: float = 0.20,
        hand_target_filter_alpha: object = None,
        maximum_arm_target_step_rad: float = 0.015,
        maximum_hand_target_step_rad: float = 0.05,
        franka_action_contract_id: str = LEGACY_FRANKA_ACTION_CONTRACT_ID,
    ) -> None:
        control_dt = _finite_scalar(control_dt_s, "control_dt_s")
        margin = _finite_scalar(joint_limit_margin_rad, "joint_limit_margin_rad")
        if control_dt <= 0.0:
            raise ValueError("control_dt_s must be positive")
        if margin < 0.0:
            raise ValueError("joint_limit_margin_rad cannot be negative")
        safe_limits_f64 = contracted_joint_limits_float64(
            joint_limits_rad,
            margin_rad=margin,
        )
        safe_limits = inward_float32_joint_limits(
            joint_limits_rad,
            margin_rad=margin,
        )

        self.control_dt_s = control_dt
        self.joint_limit_margin_rad = margin
        self.franka_action_contract_id = normalize_franka_action_contract_id(
            franka_action_contract_id
        )
        self._joint_limits_f64 = safe_limits_f64
        self._joint_limits = safe_limits
        # Exact replay payloads are stored as float32.  Retain the legacy
        # nearest-float representation solely to recognize an endpoint whose
        # quantization landed outside the authoritative binary64 interval.
        # The command itself is always canonicalized to ``_joint_limits``.
        self._rounded_joint_limits = safe_limits_f64.astype(np.float32)
        self._q_hand_close = _readonly_float_vector(
            q_hand_close_rad, 6, "q_hand_close_rad"
        )
        if np.any(self._q_hand_close <= 0.0):
            raise ValueError("q_hand_close_rad must be positive")
        self._maximum_arm_rate = _optional_positive_rate_vector(
            commissioned_max_arm_target_rate_rad_s,
            7,
            "commissioned_max_arm_target_rate_rad_s",
        )
        self._maximum_hand_rate = _optional_positive_rate_vector(
            commissioned_max_hand_target_rate_rad_s,
            6,
            "commissioned_max_hand_target_rate_rad_s",
        )
        scalar_parameters = {
            "measured_arm_envelope_rad": measured_arm_envelope_rad,
            "arm_raw_gain_rad": arm_raw_gain_rad,
            "target_filter_alpha": target_filter_alpha,
            "hand_target_filter_alpha": (
                target_filter_alpha
                if hand_target_filter_alpha is None
                else hand_target_filter_alpha
            ),
            "maximum_arm_target_step_rad": maximum_arm_target_step_rad,
            "maximum_hand_target_step_rad": maximum_hand_target_step_rad,
        }
        self._mapper_parameters = {
            name: _finite_scalar(value, name)
            for name, value in scalar_parameters.items()
        }
        if self.franka_action_contract_id == LEGACY_FRANKA_ACTION_CONTRACT_ID:
            if any(value <= 0.0 for value in self._mapper_parameters.values()):
                raise ValueError("action mapper parameters must be positive")
        else:
            if (
                self._mapper_parameters["arm_raw_gain_rad"] < 0.0
                or self._mapper_parameters["target_filter_alpha"] < 0.0
                or self._mapper_parameters["maximum_arm_target_step_rad"] < 0.0
                or self._mapper_parameters["measured_arm_envelope_rad"] < 0.0
                or self._mapper_parameters["hand_target_filter_alpha"] <= 0.0
                or self._mapper_parameters["maximum_hand_target_step_rad"] <= 0.0
            ):
                raise ValueError("q_d-relative mapper parameters are invalid")
        if (
            self._mapper_parameters["target_filter_alpha"] > 1.0
            or self._mapper_parameters["hand_target_filter_alpha"] > 1.0
        ):
            raise ValueError("target filter alpha must not exceed one")

        self._arm_target = _readonly_float_vector(
            initial_arm_target_q_rad, 7, "initial_arm_target_q_rad"
        )
        self._hand_target = _readonly_float_vector(
            initial_hand_target_q_policy_order_rad,
            6,
            "initial_hand_target_q_policy_order_rad",
        )
        if np.any(self._arm_target < self._joint_limits[:, 0]) or np.any(
            self._arm_target > self._joint_limits[:, 1]
        ):
            raise ValueError(
                "initial_arm_target_q_rad is outside the margin-contracted limits"
            )
        if np.any(self._hand_target < 0.0) or np.any(
            self._hand_target > self._q_hand_close
        ):
            raise ValueError(
                "initial_hand_target_q_policy_order_rad is outside semantic limits"
            )
        self._lock = threading.Lock()
        self._last_committed_sequence = 0
        self._pending: Optional[TransactionalV94ActionProposal] = None
        self._fault_reason: Optional[str] = None

    @property
    def fault_reason(self) -> Optional[str]:
        with self._lock:
            return self._fault_reason

    @property
    def last_committed_sequence(self) -> int:
        with self._lock:
            return self._last_committed_sequence

    @property
    def pending_sequence(self) -> Optional[int]:
        with self._lock:
            return None if self._pending is None else self._pending.sequence

    def committed_targets(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self._arm_target.copy(), self._hand_target.copy()

    def _fault_locked(self, reason: str) -> None:
        if self._fault_reason is None:
            self._fault_reason = str(reason).strip() or "unspecified mapper fault"

    def _require_healthy_locked(self) -> None:
        if self._fault_reason is not None:
            raise ClosedLoopProtocolError(
                f"transactional action mapper fault is latched: {self._fault_reason}"
            )

    @staticmethod
    def _rate_violation(
        actual: np.ndarray,
        maximum: Optional[np.ndarray],
        *,
        label: str,
    ) -> Optional[str]:
        if maximum is None:
            return None
        excess = np.flatnonzero(
            np.abs(np.asarray(actual, dtype=np.float64))
            > maximum + np.float64(1.0e-7)
        )
        if len(excess) == 0:
            return None
        index = int(excess[0])
        return (
            f"{label} target rate exceeds commissioned limit at axis {index + 1}: "
            f"actual={abs(float(actual[index])):.9f}rad/s, "
            f"limit={float(maximum[index]):.9f}rad/s"
        )

    def propose(
        self,
        sequence: int,
        action13: np.ndarray,
        *,
        measured_q_rad: np.ndarray,
        shaper_q_d_rad: object = None,
        hold_arm_target: bool = False,
    ) -> TransactionalV94ActionProposal:
        proposal_sequence = _positive_integer(sequence, "proposal sequence")
        with self._lock:
            self._require_healthy_locked()
            if self._pending is not None:
                reason = "cannot propose while another mapper proposal is pending"
                self._fault_locked(reason)
                raise ClosedLoopProtocolError(reason)
            expected = self._last_committed_sequence + 1
            if proposal_sequence != expected:
                reason = (
                    f"mapper proposal sequence mismatch: expected={expected}, "
                    f"actual={proposal_sequence}"
                )
                self._fault_locked(reason)
                raise ClosedLoopProtocolError(reason)
            prior_arm = self._arm_target.copy()
            prior_hand = self._hand_target.copy()
            q_d = None
            if self.franka_action_contract_id == QD_G015_FRANKA_ACTION_CONTRACT_ID:
                if shaper_q_d_rad is None:
                    reason = (
                        "q_d g015 Franka action proposal requires shaper_q_d_rad"
                    )
                    self._fault_locked(reason)
                    raise ClosedLoopProtocolError(reason)
                try:
                    q_d = _readonly_float_vector(
                        shaper_q_d_rad, 7, "shaper_q_d_rad"
                    )
                except ValueError as exc:
                    reason = f"invalid q_d g015 shaper state: {exc}"
                    self._fault_locked(reason)
                    raise ClosedLoopProtocolError(reason) from exc
            temporary = V94ActionMapper(
                initial_arm_target_q_rad=prior_arm,
                initial_hand_target_q_policy_order_rad=prior_hand,
                joint_limits_rad=self._joint_limits,
                q_hand_close_rad=self._q_hand_close,
                franka_action_contract_id=self.franka_action_contract_id,
                **self._mapper_parameters,
            )
            mapped = temporary.map(
                action13,
                measured_q_rad=measured_q_rad,
                shaper_q_d_rad=q_d,
                hold_arm_target=bool(hold_arm_target),
            )
            arm_rate = (
                np.asarray(mapped.franka_target_q_rad, dtype=np.float64)
                - prior_arm.astype(np.float64)
            ) / self.control_dt_s
            hand_rate = (
                np.asarray(
                    mapped.rh56_target_q_policy_order_rad, dtype=np.float64
                )
                - prior_hand.astype(np.float64)
            ) / self.control_dt_s
            arm_violation = None
            if self.franka_action_contract_id == LEGACY_FRANKA_ACTION_CONTRACT_ID:
                arm_violation = self._rate_violation(
                    arm_rate, self._maximum_arm_rate, label="Franka"
                )
            violation = arm_violation or self._rate_violation(
                hand_rate, self._maximum_hand_rate, label="RH56"
            )
            if violation is not None:
                self._fault_locked(violation)
                raise ClosedLoopProtocolError(violation)
            proposal = TransactionalV94ActionProposal(
                sequence=proposal_sequence,
                mapped=mapped,
                prior_arm_target_q_rad=prior_arm,
                prior_hand_target_q_policy_order_rad=prior_hand,
                arm_target_rate_rad_s=arm_rate,
                hand_target_rate_rad_s=hand_rate,
            )
            self._pending = proposal
            return proposal

    @staticmethod
    def _register_order_to_policy_order(value: np.ndarray) -> np.ndarray:
        """Return RH56 register values in V94 semantic policy order."""

        registers = _readonly_register_vector(value, "exact RH56 target")
        # register: little, ring, middle, index, thumb_bending, thumb_rotation
        # policy:   thumb_rotation, thumb_bending, index, middle, ring, little
        return registers[np.asarray([5, 4, 3, 2, 1, 0], dtype=np.int64)]

    def propose_exact_targets(
        self,
        sequence: int,
        action13: np.ndarray,
        *,
        franka_target_q_rad: np.ndarray,
        rh56_angle_set_register_order: np.ndarray,
        measured_q_rad: np.ndarray,
        hold_arm_target: bool = False,
    ) -> TransactionalV94ActionProposal:
        """Propose simulator-recorded actuator targets without remapping them.

        This path is intentionally limited to replay payloads that carry both
        exact target arrays.  ``action13`` remains the dual-committed previous
        action used by the observation contract, but it is not mapped a second
        time. Continuous Franka motion is generated by the native V225 1 kHz
        interpolator followed by libfranka's official rate limiter.
        """

        proposal_sequence = _positive_integer(sequence, "proposal sequence")
        raw_action = _readonly_float_vector(action13, 13, "action13")
        if np.any(raw_action < -1.0) or np.any(raw_action > 1.0):
            raise ValueError("exact replay action13 must lie in [-1,1]")
        requested_arm = _readonly_float_vector(
            franka_target_q_rad, 7, "exact franka_target_q_rad"
        )
        measured = _readonly_float_vector(measured_q_rad, 7, "measured_q_rad")
        del measured  # freshness/tracking is enforced by the observation/native owners
        registers = _readonly_register_vector(
            rh56_angle_set_register_order,
            "exact rh56_angle_set_register_order",
        )
        with self._lock:
            self._require_healthy_locked()
            if self._pending is not None:
                reason = "cannot propose while another mapper proposal is pending"
                self._fault_locked(reason)
                raise ClosedLoopProtocolError(reason)
            expected = self._last_committed_sequence + 1
            if proposal_sequence != expected:
                reason = (
                    f"mapper proposal sequence mismatch: expected={expected}, "
                    f"actual={proposal_sequence}"
                )
                self._fault_locked(reason)
                raise ClosedLoopProtocolError(reason)
            prior_arm = self._arm_target.copy()
            prior_hand = self._hand_target.copy()
            next_arm = prior_arm if bool(hold_arm_target) else requested_arm.copy()
            boundary_canonicalized = False
            if not bool(hold_arm_target):
                lower_quantized_outward = (
                    self._rounded_joint_limits[:, 0].astype(np.float64)
                    < self._joint_limits_f64[:, 0]
                )
                upper_quantized_outward = (
                    self._rounded_joint_limits[:, 1].astype(np.float64)
                    > self._joint_limits_f64[:, 1]
                )
                use_lower = lower_quantized_outward & (
                    next_arm == self._rounded_joint_limits[:, 0]
                )
                use_upper = upper_quantized_outward & (
                    next_arm == self._rounded_joint_limits[:, 1]
                )
                next_arm[use_lower] = self._joint_limits[use_lower, 0]
                next_arm[use_upper] = self._joint_limits[use_upper, 1]
                boundary_canonicalized = bool(np.any(use_lower | use_upper))
            if np.any(next_arm < self._joint_limits[:, 0]) or np.any(
                next_arm > self._joint_limits[:, 1]
            ):
                reason = "exact Franka replay target is outside joint limits"
                self._fault_locked(reason)
                raise ClosedLoopProtocolError(reason)

            register_policy = self._register_order_to_policy_order(registers)
            next_hand = (
                self._q_hand_close
                * (
                    np.float32(1.0)
                    - register_policy.astype(np.float32) / np.float32(1000.0)
                )
            ).astype(np.float32)
            arm_rate = (
                next_arm.astype(np.float64) - prior_arm.astype(np.float64)
            ) / self.control_dt_s
            hand_rate = (
                next_hand.astype(np.float64) - prior_hand.astype(np.float64)
            ) / self.control_dt_s
            reasons = []
            if hold_arm_target:
                reasons.append("external hold gate held exact Franka replay target")
            if boundary_canonicalized:
                reasons.append("Franka float32 boundary canonicalized inward")
            mapped = V94MappedAction(
                raw_policy_action13=raw_action,
                executed_policy_action13=raw_action,
                franka_target_q_rad=next_arm,
                rh56_target_q_policy_order_rad=next_hand,
                rh56_angle_set_register_order=registers,
                clipped=bool(reasons),
                reasons=tuple(reasons),
                hold_arm_target=bool(hold_arm_target),
            )
            proposal = TransactionalV94ActionProposal(
                sequence=proposal_sequence,
                mapped=mapped,
                prior_arm_target_q_rad=prior_arm,
                prior_hand_target_q_policy_order_rad=prior_hand,
                arm_target_rate_rad_s=arm_rate,
                hand_target_rate_rad_s=hand_rate,
            )
            self._pending = proposal
            return proposal

    def discard_unstaged(self, proposal: TransactionalV94ActionProposal) -> None:
        """Discard an exact proposal before it is published to either actuator."""

        with self._lock:
            self._require_healthy_locked()
            if self._pending is not proposal:
                reason = "only the exact pending mapper proposal can be discarded"
                self._fault_locked(reason)
                raise ClosedLoopProtocolError(reason)
            self._pending = None

    def commit(
        self,
        proposal: TransactionalV94ActionProposal,
        *,
        action_ledger: ExecutedActionLedger,
    ) -> None:
        """Commit only after the ledger proves this action was dual-acknowledged."""

        if not isinstance(action_ledger, ExecutedActionLedger):
            raise TypeError("action_ledger must be ExecutedActionLedger")
        committed = action_ledger.committed_snapshot()
        with self._lock:
            self._require_healthy_locked()
            if self._pending is not proposal:
                reason = "mapper commit did not receive the exact pending proposal"
                self._fault_locked(reason)
                raise ClosedLoopProtocolError(reason)
            expected = self._last_committed_sequence + 1
            if proposal.sequence != expected:
                reason = "pending mapper proposal sequence changed before commit"
                self._fault_locked(reason)
                raise ClosedLoopProtocolError(reason)
            if committed.sequence != proposal.sequence:
                raise ClosedLoopProtocolError(
                    "mapper proposal has not been dual-acknowledged by the ledger"
                )
            if not np.array_equal(
                committed.executed_policy_action13,
                proposal.mapped.executed_policy_action13,
            ):
                reason = "ledger committed a different action for this mapper sequence"
                self._fault_locked(reason)
                raise ClosedLoopProtocolError(reason)
            if not np.array_equal(
                self._arm_target, proposal.prior_arm_target_q_rad
            ) or not np.array_equal(
                self._hand_target, proposal.prior_hand_target_q_policy_order_rad
            ):
                reason = "mapper committed targets changed after proposal"
                self._fault_locked(reason)
                raise ClosedLoopProtocolError(reason)
            self._arm_target = _readonly_float_vector(
                proposal.mapped.franka_target_q_rad,
                7,
                "committed franka_target_q_rad",
            )
            self._hand_target = _readonly_float_vector(
                proposal.mapped.rh56_target_q_policy_order_rad,
                6,
                "committed rh56_target_q_policy_order_rad",
            )
            self._last_committed_sequence = proposal.sequence
            self._pending = None


class SingleThreadOwner:
    """Bind a hardware endpoint (notably RH56 serial) to one thread forever."""

    def __init__(self, endpoint_name: str) -> None:
        name = str(endpoint_name).strip()
        if not name:
            raise ValueError("endpoint_name must be non-empty")
        self.endpoint_name = name
        self._lock = threading.Lock()
        self._owner_thread: Optional[threading.Thread] = None

    @property
    def owner_ident(self) -> Optional[int]:
        with self._lock:
            return None if self._owner_thread is None else self._owner_thread.ident

    def claim_for_current_thread(self) -> None:
        thread = threading.current_thread()
        with self._lock:
            if self._owner_thread is None:
                self._owner_thread = thread
                return
            if self._owner_thread is not thread:
                raise ClosedLoopProtocolError(
                    f"{self.endpoint_name} already has a different owner thread"
                )

    def require_current_thread(self) -> None:
        thread = threading.current_thread()
        with self._lock:
            if self._owner_thread is None:
                raise ClosedLoopProtocolError(
                    f"{self.endpoint_name} has no bound owner thread"
                )
            if self._owner_thread is not thread:
                raise ClosedLoopProtocolError(
                    f"{self.endpoint_name} access attempted by a non-owner thread"
                )


__all__ = [
    "AUTHORIZATION_SCOPE",
    "COMMAND_CONSUMERS",
    "ClosedLoopCommand",
    "ClosedLoopProtocolError",
    "ClosedLoopSafetyGate",
    "ExecutedActionCommit",
    "ExecutedActionLedger",
    "MotionAuthorization",
    "PolicyCommandSampleHold",
    "SafetyState",
    "SingleThreadOwner",
    "TransactionalV94ActionMapper",
    "TransactionalV94ActionProposal",
]
