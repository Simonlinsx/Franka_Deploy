"""Run-scoped authorization and fail-closed interlock state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import threading
from typing import Optional

from .constants import AUTHORIZATION_SCOPE
from .validation import _finite_scalar, _strict_bool


class ClosedLoopProtocolError(RuntimeError):
    """A fail-closed safety or command-protocol violation."""


class SafetyState(str, Enum):
    DISARMED = "disarmed"
    ARMED = "armed"
    FAULT_LATCHED = "fault_latched"


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

__all__ = [
    "ClosedLoopProtocolError",
    "ClosedLoopSafetyGate",
    "MotionAuthorization",
    "SafetyState",
]
