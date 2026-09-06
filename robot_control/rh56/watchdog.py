"""Independent single-owner watchdog boundary for the RH56 actuator.

The policy/runtime thread must never own the RH56 serial descriptor: camera
waits, observation assembly and policy inference may all block.  This module
therefore places the complete live RH56 lifecycle in one worker thread:

``session factory/open/arm -> bootstrap feedback -> execute -> disable -> close``.

Construction and import are hardware-inert.  The injected session factory is
called only by :meth:`RH56WatchdogOwner.start`, inside the worker.  Every
transport operation remains inside that same thread.  If no command heartbeat
arrives before the actuator's exact commissioned deadline, the worker first
publishes a terminal fault (so the combined runtime can latch its shared gate
and stop Franka), then performs the actuator's two-pass disable and physical
stop verification before closing the serial session.  The default/formal path
keeps the original command transaction with one complete feedback sample.  An
explicit supervised target-only path instead ACKs a verified target promptly
and samples complete safety feedback on the same owner thread at a separately
bounded rate; observations zero-order hold that immutable feedback cache.

There is intentionally no executable CLI in this module.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
import math
import threading
import time
from typing import Any, Callable, Deque, Optional, Tuple

from sim2real.closed_loop_core import ClosedLoopCommand, ExecutedActionLedger
from .actuator import (
    RH56ActuatorState,
    RH56SafetyFeedback,
    RH56StopReport,
    RH56TransactionalActuator,
    RH56TransactionalError,
)


class RH56WatchdogOwnerError(RuntimeError):
    """Terminal owner lifecycle, watchdog, request, or feedback-cache error."""


class RH56WatchdogOwnerState(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    FAULT_LATCHED = "fault_latched"
    STOPPED = "stopped"


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_float(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _bounded_positive_integer(
    value: object,
    name: str,
    *,
    maximum: int,
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if (
        not math.isfinite(numeric)
        or not numeric.is_integer()
        or not 1 <= int(numeric) <= maximum
    ):
        raise ValueError(f"{name} must be a positive integer <= {maximum}")
    return int(numeric)


def _advance_periodic_deadline(
    deadline_s: float,
    *,
    period_s: float,
    now_s: float,
) -> float:
    """Advance a periodic deadline without adding transaction duration.

    Scheduling the next poll from the completion timestamp produces a period
    of ``configured_period + serial_transaction_time``.  Advancing the prior
    absolute deadline keeps the long-run rate fixed while skipping, rather
    than replaying, any missed slots.
    """

    deadline = _finite_float(deadline_s, "periodic deadline")
    period = _positive_float(period_s, "periodic period")
    now = _finite_float(now_s, "periodic now")
    if deadline > now:
        return deadline
    elapsed_periods = math.floor((now - deadline) / period) + 1
    return deadline + float(elapsed_periods) * period


@dataclass(frozen=True)
class RH56OwnedSession:
    """An already-open, claimed and armed session created by the owner thread.

    The factory producing this object is responsible for closing a partially
    opened transport if setup raises before returning.  Once returned, the
    owner guarantees that ``close`` is invoked only after a disable attempt
    (or after confirming the actuator never reached ARMED).
    """

    actuator: RH56TransactionalActuator
    close: Callable[[], None]

    def __post_init__(self) -> None:
        if not isinstance(self.actuator, RH56TransactionalActuator):
            raise TypeError("owned session actuator is invalid")
        if not callable(self.close):
            raise TypeError("owned session close must be callable")


@dataclass(frozen=True)
class RH56TimestampedFeedback:
    """One immutable cached sample with monotonic and calibrated wall time."""

    feedback: RH56SafetyFeedback
    captured_realtime_s: float
    cached_monotonic_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.feedback, RH56SafetyFeedback):
            raise TypeError("feedback must be RH56SafetyFeedback")
        captured_realtime = _finite_float(
            self.captured_realtime_s, "captured_realtime_s"
        )
        cached = _finite_float(self.cached_monotonic_s, "cached_monotonic_s")
        if cached < self.feedback.captured_monotonic_s:
            raise ValueError("feedback cannot be cached before it was captured")
        object.__setattr__(self, "captured_realtime_s", captured_realtime)
        object.__setattr__(self, "cached_monotonic_s", cached)

    @property
    def captured_monotonic_s(self) -> float:
        return self.feedback.captured_monotonic_s


@dataclass(frozen=True)
class RH56FeedbackHistorySnapshot:
    """Read-only cache snapshot; creating one performs no transport access."""

    samples: Tuple[RH56TimestampedFeedback, ...]
    observed_monotonic_s: float
    latest_age_s: float
    maximum_age_s: float
    fresh: bool
    owner_state: RH56WatchdogOwnerState


@dataclass(frozen=True)
class RH56PhysicalTrackingSnapshot:
    """Read-only evidence that commanded RH56 motion appeared in feedback.

    ``challenged_axes`` records axes for which a numeric target was far enough
    from ``ANGLE_ACT`` to require physical-motion evidence.  ``verified_axes``
    is sticky evidence: the corresponding axis subsequently moved far enough
    in the commanded direction (or reported force-control contact).  The first
    outstanding challenge for an axis becomes ``failed`` if no progress
    appears before ``tracking_timeout_s``.  Once motion has been verified for
    an axis, later target/feedback gaps are allowed: a hand axis is expected to
    stop short of its free-space angle target after contacting a grasped
    object.  Fresh feedback, device errors, current and temperature remain
    monitored independently.  A policy may withdraw an unverified challenge
    by bringing its target back inside the significant-gap threshold; that
    inactive historical challenge does not invalidate physical proof already
    obtained on other axes, but a run containing only withdrawn challenges is
    not reported as verified.
    """

    verdict: str
    challenged_axes: Tuple[bool, ...]
    verified_axes: Tuple[bool, ...]
    contact_verified_axes: Tuple[bool, ...]
    failed_axes: Tuple[bool, ...]
    active_challenge_axes: Tuple[bool, ...]
    latest_feedback_fresh: bool
    latest_feedback_age_s: Optional[float]
    latest_target: Optional[Tuple[int, ...]]
    latest_angles: Optional[Tuple[int, ...]]
    latest_positions: Optional[Tuple[int, ...]]
    tracking_significant_gap_units: int
    tracking_min_progress_units: int
    tracking_contact_force_delta_g: int
    tracking_contact_force_absolute_g: int
    tracking_timeout_s: float
    failure_reason: Optional[str]


@dataclass(frozen=True)
class RH56OwnerFault:
    reason: str
    detected_monotonic_s: float
    pending_sequence: Optional[int]
    stop_confirmed: bool = False
    stop_error: Optional[str] = None


@dataclass(frozen=True)
class RH56OwnerStopResult:
    worker_stopped: bool
    rh56_disabled_verified: bool
    disable_attempted: bool
    close_attempted: bool
    serial_closed: bool
    stop_report: Optional[RH56StopReport]
    fault: Optional[RH56OwnerFault]


class RH56OwnerTicket:
    """One single-use command result without any actuator/transport access."""

    def __init__(self, *, sequence: int) -> None:
        self.sequence = int(sequence)
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._receipt: Optional[Any] = None
        self._error: Optional[BaseException] = None

    @property
    def done(self) -> bool:
        return self._event.is_set()

    def _complete(
        self,
        *,
        receipt: Optional[Any] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        with self._lock:
            if self._event.is_set():
                return
            self._receipt = receipt
            self._error = error
            self._event.set()

    def result(self, timeout_s: Optional[float] = None) -> Any:
        timeout = (
            None
            if timeout_s is None
            else _positive_float(timeout_s, "ticket timeout_s")
        )
        if not self._event.wait(timeout):
            raise RH56WatchdogOwnerError(
                f"RH56 owner response timed out for sequence {self.sequence}"
            )
        with self._lock:
            receipt = self._receipt
            error = self._error
        if error is not None:
            raise RH56WatchdogOwnerError(
                f"RH56 owner rejected sequence {self.sequence}: {error}"
            ) from error
        if receipt is None:
            raise RH56WatchdogOwnerError(
                f"RH56 owner returned no receipt for sequence {self.sequence}"
            )
        return receipt


@dataclass
class _ExecuteRequest:
    command: ClosedLoopCommand
    ticket: RH56OwnerTicket
    submitted_monotonic_s: float


class RH56WatchdogOwner:
    """Single worker that owns RH56 IO and enforces an independent watchdog."""

    def __init__(
        self,
        session_factory: Callable[[], RH56OwnedSession],
        action_ledger: ExecutedActionLedger,
        *,
        fault_callback: Optional[Callable[[RH56OwnerFault], None]] = None,
        monotonic: Callable[[], float] = time.monotonic,
        realtime: Callable[[], float] = time.time,
        startup_timeout_s: object = 2.0,
        response_timeout_s: object = 2.0,
        join_timeout_s: object = 2.0,
        bootstrap_feedback_samples: object = 2,
        feedback_history_capacity: object = 8,
        supervised_target_only: bool = False,
        feedback_period_s: object = 1.0 / 30.0,
        feedback_hard_age_s: object = 0.050,
        tracking_significant_gap_units: object = 30,
        tracking_min_progress_units: object = 3,
        tracking_contact_force_delta_g: object = 150,
        tracking_contact_force_absolute_g: object = 200,
        tracking_timeout_s: object = 0.75,
        thread_name: str = "rh56-watchdog-owner",
    ) -> None:
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        if not isinstance(action_ledger, ExecutedActionLedger):
            raise TypeError("action_ledger must be ExecutedActionLedger")
        if fault_callback is not None and not callable(fault_callback):
            raise TypeError("fault_callback must be callable")
        if not callable(monotonic) or not callable(realtime):
            raise TypeError("monotonic and realtime must be callable")
        if not isinstance(supervised_target_only, bool):
            raise TypeError("supervised_target_only must be boolean")
        name = str(thread_name).strip()
        if not name:
            raise ValueError("thread_name must be non-empty")

        self._session_factory = session_factory
        self._ledger = action_ledger
        self._fault_callback = fault_callback
        self._monotonic = monotonic
        self._realtime = realtime
        self._startup_timeout_s = _positive_float(
            startup_timeout_s, "startup_timeout_s"
        )
        self._response_timeout_s = _positive_float(
            response_timeout_s, "response_timeout_s"
        )
        self._join_timeout_s = _positive_float(join_timeout_s, "join_timeout_s")
        self._bootstrap_feedback_samples = _bounded_positive_integer(
            bootstrap_feedback_samples,
            "bootstrap_feedback_samples",
            maximum=16,
        )
        if self._bootstrap_feedback_samples < 2:
            raise ValueError("bootstrap_feedback_samples must be at least two")
        self._feedback_history_capacity = _bounded_positive_integer(
            feedback_history_capacity,
            "feedback_history_capacity",
            maximum=1024,
        )
        if self._feedback_history_capacity < self._bootstrap_feedback_samples:
            raise ValueError(
                "feedback_history_capacity cannot be smaller than bootstrap samples"
            )
        self._supervised_target_only = supervised_target_only
        self._feedback_period_s = _positive_float(
            feedback_period_s, "feedback_period_s"
        )
        self._feedback_hard_age_s = _positive_float(
            feedback_hard_age_s, "feedback_hard_age_s"
        )
        if self._feedback_period_s >= self._feedback_hard_age_s:
            raise ValueError("feedback_period_s must be below feedback_hard_age_s")
        self._tracking_significant_gap_units = _bounded_positive_integer(
            tracking_significant_gap_units,
            "tracking_significant_gap_units",
            maximum=1000,
        )
        self._tracking_min_progress_units = _bounded_positive_integer(
            tracking_min_progress_units,
            "tracking_min_progress_units",
            maximum=1000,
        )
        if (
            self._tracking_min_progress_units
            >= self._tracking_significant_gap_units
        ):
            raise ValueError(
                "tracking_min_progress_units must be below "
                "tracking_significant_gap_units"
            )
        self._tracking_position_min_progress_units = max(
            5, self._tracking_min_progress_units
        )
        self._tracking_contact_force_delta_g = _bounded_positive_integer(
            tracking_contact_force_delta_g,
            "tracking_contact_force_delta_g",
            maximum=32767,
        )
        self._tracking_contact_force_absolute_g = _bounded_positive_integer(
            tracking_contact_force_absolute_g,
            "tracking_contact_force_absolute_g",
            maximum=32767,
        )
        if (
            self._tracking_contact_force_delta_g
            > self._tracking_contact_force_absolute_g
        ):
            raise ValueError(
                "tracking_contact_force_delta_g cannot exceed "
                "tracking_contact_force_absolute_g"
            )
        self._tracking_timeout_s = _positive_float(
            tracking_timeout_s, "tracking_timeout_s"
        )
        self._thread_name = name

        self._condition = threading.Condition(threading.Lock())
        self._state = RH56WatchdogOwnerState.CREATED
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._worker_stopped = threading.Event()
        self._fault_event = threading.Event()
        self._stop_requested = False
        self._request: Optional[_ExecuteRequest] = None
        self._request_inflight = False
        self._final_tracking_settle_requested = False
        self._final_tracking_settle_active = False
        self._final_tracking_settle_deadline_s: Optional[float] = None
        self._first_command_readiness_verified = False
        self._fault: Optional[RH56OwnerFault] = None
        self._fault_callback_sent = False
        self._startup_error: Optional[BaseException] = None
        self._actuator_watchdog_timeout_s: Optional[float] = None
        self._default_feedback_maximum_age_s: Optional[float] = None
        self._feedback_history: Deque[RH56TimestampedFeedback] = deque(
            maxlen=self._feedback_history_capacity
        )
        self._tracking_latest_target: Optional[Tuple[int, ...]] = None
        self._tracking_challenged = [False] * 6
        self._tracking_verified = [False] * 6
        self._tracking_contact_verified = [False] * 6
        self._tracking_failed = [False] * 6
        self._tracking_active = [False] * 6
        self._tracking_direction = [0] * 6
        self._tracking_baseline_angle = [0] * 6
        self._tracking_baseline_position = [0] * 6
        self._tracking_baseline_force_g = [0] * 6
        # A streamed closure target can begin with a sub-threshold ANGLE_ACT
        # gap and cross the significant-gap threshold only after contact has
        # already started.  Preserve the force baseline from the beginning of
        # that same-direction target stream; otherwise the late challenge can
        # incorrectly treat a real force-controlled grasp as no motion.
        self._tracking_force_epoch_active = [False] * 6
        self._tracking_force_epoch_direction = [0] * 6
        self._tracking_force_epoch_baseline_g = [0] * 6
        self._tracking_deadline_s = [0.0] * 6
        self._tracking_failure_reason: Optional[str] = None
        self._stop_report: Optional[RH56StopReport] = None
        self._disable_attempted = False
        self._close_attempted = False
        self._serial_closed = False

    @property
    def state(self) -> RH56WatchdogOwnerState:
        with self._condition:
            return self._state

    @property
    def fault_event(self) -> threading.Event:
        return self._fault_event

    @property
    def fault_snapshot(self) -> Optional[RH56OwnerFault]:
        with self._condition:
            return self._fault

    @property
    def owner_ident(self) -> Optional[int]:
        with self._condition:
            thread = self._thread
        return None if thread is None else thread.ident

    @property
    def independent_watchdog_active(self) -> bool:
        """Explicit runtime evidence that an independent owner is live."""

        with self._condition:
            thread = self._thread
            return bool(
                self._state is RH56WatchdogOwnerState.RUNNING
                and thread is not None
                and thread.is_alive()
                and self._actuator_watchdog_timeout_s is not None
            )

    @property
    def fault_callback_configured(self) -> bool:
        """True only after start while a real terminal-fault sink is bound."""

        with self._condition:
            return bool(
                self._state is RH56WatchdogOwnerState.RUNNING
                and self._fault_callback is not None
            )

    @property
    def supervised_target_only(self) -> bool:
        return self._supervised_target_only

    @property
    def feedback_period_s(self) -> float:
        return self._feedback_period_s

    @property
    def feedback_hard_age_s(self) -> float:
        return self._feedback_hard_age_s

    @property
    def first_command_readiness_verified(self) -> bool:
        """True after complete bootstrap feedback and before target IO."""

        with self._condition:
            return bool(
                self._state is RH56WatchdogOwnerState.RUNNING
                and self._first_command_readiness_verified
            )

    @property
    def final_tracking_settle_active(self) -> bool:
        with self._condition:
            return bool(self._final_tracking_settle_active)

    def _now(self) -> float:
        return _finite_float(self._monotonic(), "monotonic clock")

    def _wall_now(self) -> float:
        return _finite_float(self._realtime(), "realtime clock")

    def start(self) -> None:
        """Start, open and arm in the worker; wait for two bootstrap samples."""

        with self._condition:
            if self._state is not RH56WatchdogOwnerState.CREATED:
                raise RH56WatchdogOwnerError("RH56 owner is single-use")
            self._state = RH56WatchdogOwnerState.STARTING
            thread = threading.Thread(
                target=self._worker_main,
                name=self._thread_name,
                daemon=False,
            )
            self._thread = thread
            thread.start()
        try:
            ready = self._ready.wait(self._startup_timeout_s)
        except BaseException:
            self.request_stop()
            raise
        if not ready:
            self.request_stop()
            raise RH56WatchdogOwnerError(
                "RH56 owner startup exceeded its bounded timeout"
            )
        with self._condition:
            error = self._startup_error
            state = self._state
        if error is not None or state is not RH56WatchdogOwnerState.RUNNING:
            detail = error if error is not None else self._fault
            raise RH56WatchdogOwnerError(
                f"RH56 owner failed during startup: {detail}"
            ) from error

    def submit(self, command: ClosedLoopCommand) -> RH56OwnerTicket:
        """Queue exactly one already-staged command; performs no device IO."""

        if not isinstance(command, ClosedLoopCommand):
            raise TypeError("command must be ClosedLoopCommand")
        with self._condition:
            if self._state is RH56WatchdogOwnerState.FAULT_LATCHED:
                raise RH56WatchdogOwnerError(
                    f"RH56 owner fault is latched: {self._fault}"
                )
            if self._state is not RH56WatchdogOwnerState.RUNNING:
                raise RH56WatchdogOwnerError("RH56 owner is not running")
            if self._stop_requested:
                raise RH56WatchdogOwnerError("RH56 owner stop is already requested")
            if (
                self._final_tracking_settle_requested
                or self._final_tracking_settle_active
            ):
                raise RH56WatchdogOwnerError(
                    "RH56 owner rejects commands during final tracking settle"
                )
            if self._request is not None or self._request_inflight:
                raise RH56WatchdogOwnerError(
                    "RH56 owner accepts only one in-flight request"
                )
            if self._ledger.pending_sequence != command.sequence:
                raise RH56WatchdogOwnerError(
                    f"RH56 sequence {command.sequence} is not staged in the ledger"
                )
            ticket = RH56OwnerTicket(sequence=command.sequence)
            self._request = _ExecuteRequest(
                command=command,
                ticket=ticket,
                submitted_monotonic_s=self._now(),
            )
            self._condition.notify_all()
            return ticket

    def begin_final_tracking_settle(
        self,
        *,
        timeout_s: Optional[float] = None,
    ) -> float:
        """Enter a bounded, no-write hold for final physical-motion proof.

        The already-committed target remains owned by the hand firmware.  The
        worker continues complete-feedback polling and refreshes only the
        actuator's sample-hold heartbeat so the shorter inter-command watchdog
        cannot pre-empt the existing physical-tracking deadline.  No policy
        sequence, ledger state, or RH56 register is changed.

        Returns the absolute owner deadline by which tracking must either be
        verified or faulted.
        """

        if not self._supervised_target_only:
            raise RH56WatchdogOwnerError(
                "final tracking settle requires supervised target-only mode"
            )
        timeout = (
            self._response_timeout_s
            if timeout_s is None
            else _positive_float(timeout_s, "final tracking settle timeout_s")
        )
        wait_deadline = time.monotonic() + timeout
        now = self._now()
        with self._condition:
            if self._state is RH56WatchdogOwnerState.FAULT_LATCHED:
                raise RH56WatchdogOwnerError(
                    f"RH56 owner fault is latched: {self._fault}"
                )
            if self._state is not RH56WatchdogOwnerState.RUNNING:
                raise RH56WatchdogOwnerError("RH56 owner is not running")
            if self._stop_requested:
                raise RH56WatchdogOwnerError(
                    "RH56 owner stop is already requested"
                )
            if (
                self._request is not None
                or self._request_inflight
            ):
                raise RH56WatchdogOwnerError(
                    "final tracking settle requires an idle request boundary"
                )
            if (
                self._final_tracking_settle_requested
                or self._final_tracking_settle_active
            ):
                raise RH56WatchdogOwnerError(
                    "final tracking settle is already active"
                )
            if self._ledger.pending_sequence is not None:
                raise RH56WatchdogOwnerError(
                    "final tracking settle is forbidden with a pending "
                    "ledger sequence"
                )
            if self._ledger.last_committed_sequence < 1:
                raise RH56WatchdogOwnerError(
                    "final tracking settle requires a committed policy sequence"
                )
            if any(self._tracking_failed):
                raise RH56WatchdogOwnerError(
                    self._tracking_failure_reason
                    or "RH56 physical tracking already failed"
                )
            challenged_indices = [
                axis
                for axis in range(6)
                if self._tracking_challenged[axis]
            ]
            if not challenged_indices or all(
                self._tracking_verified[axis]
                for axis in challenged_indices
            ):
                return now
            active_unverified = [
                axis
                for axis in challenged_indices
                if (
                    not self._tracking_verified[axis]
                    and self._tracking_active[axis]
                )
            ]
            if not active_unverified:
                raise RH56WatchdogOwnerError(
                    "RH56 challenged axis is unverified without an active "
                    "physical-tracking window"
                )
            tracking_deadline = max(
                self._tracking_deadline_s[axis]
                for axis in active_unverified
            )
            # One hard-age interval after the existing no-motion deadline is
            # enough for the already-scheduled feedback poll to expose either
            # progress or the unchanged-angle fault.  This is not a new
            # tracking window and can never exceed one timeout plus one
            # feedback-hard-age interval from this request.
            settle_deadline = min(
                tracking_deadline + self._feedback_hard_age_s,
                now + self._tracking_timeout_s + self._feedback_hard_age_s,
            )
            if settle_deadline <= now:
                raise RH56WatchdogOwnerError(
                    "RH56 physical-tracking settle deadline already elapsed"
                )
            self._final_tracking_settle_deadline_s = settle_deadline
            self._final_tracking_settle_requested = True
            self._condition.notify_all()
            while not self._final_tracking_settle_active:
                if (
                    self._state is not RH56WatchdogOwnerState.RUNNING
                    or self._fault is not None
                    or self._stop_requested
                ):
                    self._final_tracking_settle_requested = False
                    raise RH56WatchdogOwnerError(
                        f"RH56 final tracking settle rejected: {self._fault}"
                    )
                remaining = wait_deadline - time.monotonic()
                if remaining <= 0.0:
                    self._final_tracking_settle_requested = False
                    self._condition.notify_all()
                    raise RH56WatchdogOwnerError(
                        "RH56 final tracking settle readiness timed out"
                    )
                self._condition.wait(timeout=remaining)
            assert self._final_tracking_settle_deadline_s is not None
            return float(self._final_tracking_settle_deadline_s)

    def execute(
        self,
        command: ClosedLoopCommand,
        *,
        timeout_s: Optional[float] = None,
    ) -> Any:
        ticket = self.submit(command)
        timeout = self._response_timeout_s if timeout_s is None else timeout_s
        try:
            return ticket.result(timeout)
        except BaseException:
            if not ticket.done:
                self.request_stop()
            raise

    def raise_if_faulted(self) -> None:
        with self._condition:
            fault = self._fault
            state = self._state
        if fault is not None or state is RH56WatchdogOwnerState.FAULT_LATCHED:
            raise RH56WatchdogOwnerError(f"RH56 owner fault is latched: {fault}")

    def feedback_history_snapshot(
        self,
        *,
        maximum_age_s: Optional[float] = None,
        now_monotonic_s: Optional[float] = None,
    ) -> RH56FeedbackHistorySnapshot:
        """Copy the feedback ring without polling or touching the actuator."""

        now = self._now() if now_monotonic_s is None else _finite_float(
            now_monotonic_s, "now_monotonic_s"
        )
        with self._condition:
            samples = tuple(self._feedback_history)
            default_maximum = self._default_feedback_maximum_age_s
            state = self._state
        if not samples:
            raise RH56WatchdogOwnerError("RH56 feedback cache is empty")
        maximum = (
            default_maximum
            if maximum_age_s is None
            else _positive_float(maximum_age_s, "maximum_age_s")
        )
        if maximum is None:
            raise RH56WatchdogOwnerError(
                "RH56 feedback freshness limit is unavailable before startup"
            )
        latest_age = now - samples[-1].captured_monotonic_s
        if latest_age < 0.0:
            raise RH56WatchdogOwnerError(
                "RH56 feedback snapshot clock precedes the cached sample"
            )
        return RH56FeedbackHistorySnapshot(
            samples=samples,
            observed_monotonic_s=now,
            latest_age_s=latest_age,
            maximum_age_s=maximum,
            fresh=latest_age <= maximum,
            owner_state=state,
        )

    def require_fresh_feedback_history(
        self,
        *,
        minimum_samples: object = 2,
        maximum_age_s: Optional[float] = None,
        now_monotonic_s: Optional[float] = None,
    ) -> RH56FeedbackHistorySnapshot:
        minimum = _bounded_positive_integer(
            minimum_samples, "minimum_samples", maximum=1024
        )
        snapshot = self.feedback_history_snapshot(
            maximum_age_s=maximum_age_s,
            now_monotonic_s=now_monotonic_s,
        )
        if len(snapshot.samples) < minimum:
            raise RH56WatchdogOwnerError(
                f"RH56 feedback history has {len(snapshot.samples)} samples, "
                f"requires {minimum}"
            )
        if not snapshot.fresh:
            raise RH56WatchdogOwnerError(
                f"RH56 feedback is stale: age={snapshot.latest_age_s:.6f}s, "
                f"limit={snapshot.maximum_age_s:.6f}s"
            )
        return snapshot

    def latest_feedback_snapshot(
        self,
        *,
        maximum_age_s: Optional[float] = None,
        now_monotonic_s: Optional[float] = None,
    ) -> RH56TimestampedFeedback:
        return self.require_fresh_feedback_history(
            minimum_samples=1,
            maximum_age_s=maximum_age_s,
            now_monotonic_s=now_monotonic_s,
        ).samples[-1]

    def physical_tracking_snapshot(
        self,
        *,
        now_monotonic_s: Optional[float] = None,
    ) -> RH56PhysicalTrackingSnapshot:
        """Return motion-following evidence without performing RH56 IO."""

        now = self._now() if now_monotonic_s is None else _finite_float(
            now_monotonic_s, "now_monotonic_s"
        )
        with self._condition:
            challenged = tuple(self._tracking_challenged)
            verified = tuple(self._tracking_verified)
            contact_verified = tuple(self._tracking_contact_verified)
            failed = tuple(self._tracking_failed)
            active = tuple(self._tracking_active)
            latest_target = self._tracking_latest_target
            failure_reason = self._tracking_failure_reason
            latest_sample = (
                None if not self._feedback_history else self._feedback_history[-1]
            )
        latest_age: Optional[float] = None
        latest_fresh = False
        latest_angles: Optional[Tuple[int, ...]] = None
        latest_positions: Optional[Tuple[int, ...]] = None
        if latest_sample is not None:
            latest_age = max(
                0.0, now - latest_sample.feedback.captured_monotonic_s
            )
            latest_fresh = latest_age <= self._feedback_hard_age_s
            latest_angles = latest_sample.feedback.angles
            latest_positions = latest_sample.feedback.positions
        active_unverified = tuple(
            challenged[index]
            and not verified[index]
            and active[index]
            for index in range(6)
        )
        if any(failed):
            verdict = "failed"
        elif any(active_unverified):
            verdict = "challenged_unverified"
        elif any(challenged) and any(verified):
            verdict = "verified"
        elif any(challenged):
            verdict = "challenged_withdrawn_unverified"
        else:
            verdict = "not_exercised"
        return RH56PhysicalTrackingSnapshot(
            verdict=verdict,
            challenged_axes=challenged,
            verified_axes=verified,
            contact_verified_axes=contact_verified,
            failed_axes=failed,
            active_challenge_axes=active,
            latest_feedback_fresh=latest_fresh,
            latest_feedback_age_s=latest_age,
            latest_target=latest_target,
            latest_angles=latest_angles,
            latest_positions=latest_positions,
            tracking_significant_gap_units=self._tracking_significant_gap_units,
            tracking_min_progress_units=self._tracking_min_progress_units,
            tracking_contact_force_delta_g=(
                self._tracking_contact_force_delta_g
            ),
            tracking_contact_force_absolute_g=(
                self._tracking_contact_force_absolute_g
            ),
            tracking_timeout_s=self._tracking_timeout_s,
            failure_reason=failure_reason,
        )

    def request_stop(self) -> None:
        """Wake the owner immediately; never performs IO in the caller."""

        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()

    def stop_and_close(
        self,
        *,
        timeout_s: Optional[float] = None,
    ) -> RH56OwnerStopResult:
        """Request verified stop and use a bounded join (including SIGINT paths)."""

        self.request_stop()
        with self._condition:
            thread = self._thread
        if thread is None:
            return RH56OwnerStopResult(
                worker_stopped=True,
                rh56_disabled_verified=True,
                disable_attempted=False,
                close_attempted=False,
                serial_closed=True,
                stop_report=None,
                fault=None,
            )
        timeout = (
            self._join_timeout_s
            if timeout_s is None
            else _positive_float(timeout_s, "stop timeout_s")
        )
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise RH56WatchdogOwnerError(
                "RH56 owner did not stop before bounded join timeout"
            )
        with self._condition:
            fault = self._fault
            report = self._stop_report
            disable_attempted = self._disable_attempted
            close_attempted = self._close_attempted
            serial_closed = self._serial_closed
        return RH56OwnerStopResult(
            worker_stopped=True,
            rh56_disabled_verified=bool(
                report is not None and report.verified
            )
            or bool(fault is not None and fault.stop_confirmed),
            disable_attempted=disable_attempted,
            close_attempted=close_attempted,
            serial_closed=serial_closed,
            stop_report=report,
            fault=fault,
        )

    def _cache_feedback(self, feedback: RH56SafetyFeedback) -> None:
        cached = self._now()
        realtime_now = self._wall_now()
        age = cached - feedback.captured_monotonic_s
        if age < 0.0:
            raise RH56WatchdogOwnerError(
                "RH56 feedback timestamp is in the future at cache boundary"
            )
        sample = RH56TimestampedFeedback(
            feedback=feedback,
            captured_realtime_s=realtime_now - age,
            cached_monotonic_s=cached,
        )
        with self._condition:
            self._feedback_history.append(sample)
        self._evaluate_physical_tracking(feedback)

    def _register_physical_tracking_target(
        self,
        targets: Tuple[int, ...],
        *,
        now_monotonic_s: float,
    ) -> None:
        """Start/extend per-axis progress windows after a numeric write."""

        now = _finite_float(now_monotonic_s, "tracking target time")
        with self._condition:
            if not self._feedback_history:
                raise RH56WatchdogOwnerError(
                    "RH56 physical tracking cannot start without feedback"
                )
            feedback = self._feedback_history[-1].feedback
            previous_targets = self._tracking_latest_target
            self._tracking_latest_target = tuple(int(value) for value in targets)
            for axis, target in enumerate(self._tracking_latest_target):
                actual = int(feedback.angles[axis])
                gap = target - actual
                if previous_targets is not None:
                    previous_target = int(previous_targets[axis])
                    target_delta = target - previous_target
                    if target_delta != 0:
                        stream_direction = 1 if target_delta > 0 else -1
                        if (
                            not self._tracking_force_epoch_active[axis]
                            or self._tracking_force_epoch_direction[axis]
                            != stream_direction
                        ):
                            self._tracking_force_epoch_active[axis] = True
                            self._tracking_force_epoch_direction[axis] = (
                                stream_direction
                            )
                            self._tracking_force_epoch_baseline_g[axis] = int(
                                feedback.forces_g[axis]
                            )
                if abs(gap) < self._tracking_significant_gap_units:
                    self._tracking_active[axis] = False
                    self._tracking_direction[axis] = 0
                    continue
                direction = 1 if gap > 0 else -1
                if (
                    not self._tracking_challenged[axis]
                    and previous_targets is not None
                    and abs(target - actual)
                    < abs(int(previous_targets[axis]) - actual)
                ):
                    # Do not open a no-motion challenge when a streamed target
                    # is moving *toward* the latest measured position.  This
                    # occurs near an RH56 endpoint when ANGLE_ACT advances by
                    # more than one 20 Hz target increment; the command is
                    # reducing, not creating, the outstanding physical gap.
                    continue
                self._tracking_challenged[axis] = True
                # ANGLE_SET is a free-space target, not a promise that a hand
                # axis in contact will reach it.  Require one real motion proof
                # per axis, then rely on the independent fresh-feedback,
                # error/current/temperature guards.  Re-challenging a proven
                # axis aborts valid grasps when a finger or thumb rotation
                # naturally stops against the object.
                if self._tracking_verified[axis]:
                    self._tracking_active[axis] = False
                    self._tracking_direction[axis] = 0
                    continue
                # A continuing same-direction target stream must not keep
                # postponing the progress deadline.  Rebase only for the first
                # challenge or for a genuine reversal.
                if (
                    not self._tracking_active[axis]
                    or self._tracking_direction[axis] != direction
                ):
                    self._tracking_active[axis] = True
                    self._tracking_direction[axis] = direction
                    self._tracking_baseline_angle[axis] = actual
                    self._tracking_baseline_position[axis] = int(
                        feedback.positions[axis]
                    )
                    force_baseline_g = int(feedback.forces_g[axis])
                    if (
                        self._tracking_force_epoch_active[axis]
                        and self._tracking_force_epoch_direction[axis]
                        == direction
                    ):
                        force_baseline_g = int(
                            self._tracking_force_epoch_baseline_g[axis]
                        )
                    self._tracking_baseline_force_g[axis] = force_baseline_g
                    self._tracking_deadline_s[axis] = (
                        now + self._tracking_timeout_s
                    )

    def _evaluate_physical_tracking(
        self,
        feedback: RH56SafetyFeedback,
    ) -> None:
        """Verify progress or raise once an active progress window expires."""

        now = float(feedback.captured_monotonic_s)
        failure_axes = []
        with self._condition:
            targets = self._tracking_latest_target
            if targets is None:
                return
            for axis in range(6):
                if not self._tracking_active[axis]:
                    continue
                direction = self._tracking_direction[axis]
                angle_progress = direction * (
                    int(feedback.angles[axis])
                    - self._tracking_baseline_angle[axis]
                )
                position_progress = abs(
                    int(feedback.positions[axis])
                    - self._tracking_baseline_position[axis]
                )
                force_g = int(feedback.forces_g[axis])
                force_delta_g = abs(
                    force_g - self._tracking_baseline_force_g[axis]
                )
                # RH56 firmware on the installed hand reports status 0/1
                # while a force-controlled finger is visibly holding an
                # object, so status==3 alone is not usable contact evidence.
                # Accept a large FORCE_ACT change only when both its absolute
                # value and its change from the pre-command baseline are
                # significant.  Error/current/temperature/freshness guards
                # remain independent and fully active.
                force_contact = bool(
                    abs(force_g) >= self._tracking_contact_force_absolute_g
                    and force_delta_g >= self._tracking_contact_force_delta_g
                )
                contact = int(feedback.statuses[axis]) == 3 or force_contact
                if (
                    angle_progress >= self._tracking_min_progress_units
                    or position_progress
                    >= self._tracking_position_min_progress_units
                    or contact
                ):
                    self._tracking_verified[axis] = True
                    self._tracking_contact_verified[axis] = contact
                    self._tracking_active[axis] = False
                    self._tracking_direction[axis] = 0
                    continue
                if now >= self._tracking_deadline_s[axis]:
                    self._tracking_failed[axis] = True
                    self._tracking_active[axis] = False
                    failure_axes.append(axis)
            if failure_axes:
                axis_text = ",".join(str(axis) for axis in failure_axes)
                reason = (
                    "RH56 physical tracking made no ANGLE_ACT/POS_ACT progress "
                    f"on axes [{axis_text}] within "
                    f"{self._tracking_timeout_s:.3f}s"
                )
                self._tracking_failure_reason = reason
        if failure_axes:
            raise RH56WatchdogOwnerError(reason)

    def _fail_pending_ledger(self, reason: str) -> Optional[int]:
        pending = self._ledger.pending_sequence
        if pending is None:
            return None
        try:
            self._ledger.fail("rh56", sequence=pending, reason=reason)
        except BaseException:
            pass
        return pending

    def _publish_fault(self, reason: str) -> RH56OwnerFault:
        detail = str(reason).strip() or "unspecified RH56 owner fault"
        pending = self._fail_pending_ledger(detail)
        callback: Optional[Callable[[RH56OwnerFault], None]] = None
        with self._condition:
            if self._fault is None:
                self._fault = RH56OwnerFault(
                    reason=detail,
                    detected_monotonic_s=self._now(),
                    pending_sequence=pending,
                )
            self._state = RH56WatchdogOwnerState.FAULT_LATCHED
            fault = self._fault
            self._fault_event.set()
            if not self._fault_callback_sent:
                self._fault_callback_sent = True
                callback = self._fault_callback
            request = self._request
            self._request = None
            self._condition.notify_all()
        if request is not None:
            request.ticket._complete(error=RH56WatchdogOwnerError(detail))
        if callback is not None:
            try:
                callback(fault)
            except BaseException:
                # A reporting callback is never allowed to suppress disable.
                pass
        return fault

    def _finish_fault_stop(
        self,
        *,
        stop_report: Optional[RH56StopReport],
        stop_error: Optional[BaseException],
        stop_confirmed_override: bool = False,
    ) -> None:
        with self._condition:
            fault = self._fault
            if fault is None:
                return
            newly_confirmed = bool(
                stop_confirmed_override
                or (stop_report is not None and stop_report.verified)
            )
            stop_confirmed = bool(fault.stop_confirmed or newly_confirmed)
            if stop_confirmed:
                retained_stop_error = None
            elif stop_error is not None:
                retained_stop_error = str(stop_error)
            else:
                # A later close/finally path without a new stop result cannot
                # erase the earlier stop failure that the audit must retain.
                retained_stop_error = fault.stop_error
            self._fault = replace(
                fault,
                stop_confirmed=stop_confirmed,
                stop_error=retained_stop_error,
            )
            if stop_report is not None:
                self._stop_report = stop_report

    def _watchdog_expire(
        self,
        actuator: RH56TransactionalActuator,
        *,
        now: float,
    ) -> None:
        deadline = actuator.policy_watchdog_deadline_monotonic_s
        reason = (
            "RH56 independent owner watchdog reached the command heartbeat "
            f"deadline={deadline!r}, now={now:.9f}"
        )
        self._publish_fault(reason)
        stop_report: Optional[RH56StopReport] = None
        stop_error: Optional[BaseException] = None
        self._disable_attempted = True
        try:
            stop_report = actuator.enforce_policy_watchdog(
                now_monotonic_s=now
            )
            if stop_report is None:
                raise RH56TransactionalError(
                    "owner selected watchdog expiry before actuator deadline"
                )
        except BaseException as exc:
            stop_error = exc
            if actuator.state is RH56ActuatorState.ARMED:
                try:
                    stop_report = actuator.fault_and_disable(reason)
                    stop_error = None
                except BaseException as cleanup_exc:
                    stop_error = cleanup_exc
        self._finish_fault_stop(
            stop_report=stop_report,
            stop_error=stop_error,
        )

    def _normal_disable(self, actuator: RH56TransactionalActuator) -> None:
        self._disable_attempted = True
        report = actuator.disable_and_verify()
        with self._condition:
            self._stop_report = report

    def _worker_main(self) -> None:
        session: Optional[RH56OwnedSession] = None
        actuator: Optional[RH56TransactionalActuator] = None
        try:
            session = self._session_factory()
            if not isinstance(session, RH56OwnedSession):
                raise TypeError("session_factory did not return RH56OwnedSession")
            actuator = session.actuator
            if actuator.owner_ident != threading.get_ident():
                raise RH56WatchdogOwnerError(
                    "RH56 actuator was not claimed by its lifecycle worker"
                )
            if actuator.state is not RH56ActuatorState.ARMED:
                raise RH56WatchdogOwnerError(
                    "RH56 owned session did not return an armed actuator"
                )
            deadline = actuator.policy_watchdog_deadline_monotonic_s
            if deadline is None and not actuator.awaiting_first_numeric_command:
                raise RH56WatchdogOwnerError(
                    "RH56 bootstrap heartbeat/deadline is absent after arm"
                )
            with self._condition:
                self._actuator_watchdog_timeout_s = (
                    actuator.inter_command_watchdog_timeout_s
                )
                self._default_feedback_maximum_age_s = (
                    actuator.maximum_feedback_age_s
                )

            def poll_owner_safety() -> RH56SafetyFeedback:
                if self._supervised_target_only:
                    return actuator.poll_safety(defer_failure_cleanup=True)
                return actuator.poll_safety()

            # Formal C2 uses the arm timestamp as bootstrap heartbeat.
            # Supervised mode remains numeric-target-free here: safety polls
            # keep feedback fresh but do not create a policy heartbeat.
            for _ in range(self._bootstrap_feedback_samples):
                now = self._now()
                deadline = actuator.policy_watchdog_deadline_monotonic_s
                if deadline is None and actuator.awaiting_first_numeric_command:
                    pass
                elif deadline is None or now >= deadline:
                    self._watchdog_expire(actuator, now=now)
                    raise RH56WatchdogOwnerError(
                        "RH56 bootstrap heartbeat expired before readiness"
                    )
                feedback = poll_owner_safety()
                self._cache_feedback(feedback)

            now = self._now()
            deadline = actuator.policy_watchdog_deadline_monotonic_s
            if deadline is None and actuator.awaiting_first_numeric_command:
                pass
            elif deadline is None or now >= deadline:
                self._watchdog_expire(actuator, now=now)
                raise RH56WatchdogOwnerError(
                    "RH56 bootstrap heartbeat expired during feedback bootstrap"
                )
            if (
                self._supervised_target_only
                and self._feedback_hard_age_s
                > float(actuator.maximum_feedback_age_s) + 1.0e-12
            ):
                raise RH56WatchdogOwnerError(
                    "supervised feedback hard age exceeds the actuator freshness bound"
                )
            with self._condition:
                if not self._feedback_history:
                    raise RH56WatchdogOwnerError(
                        "RH56 bootstrap produced no cached feedback"
                    )
                last_feedback_captured_s = (
                    self._feedback_history[-1].captured_monotonic_s
                )
            next_feedback_deadline_s = (
                last_feedback_captured_s + self._feedback_period_s
            )
            with self._condition:
                self._state = RH56WatchdogOwnerState.RUNNING
                self._first_command_readiness_verified = True
                self._ready.set()
                self._condition.notify_all()

            while True:
                request: Optional[_ExecuteRequest] = None
                now = self._now()
                with self._condition:
                    settle_active = self._final_tracking_settle_active
                    settle_deadline = self._final_tracking_settle_deadline_s
                    stop_already_requested = self._stop_requested
                if settle_active and not stop_already_requested:
                    if settle_deadline is None or now >= settle_deadline:
                        raise RH56WatchdogOwnerError(
                            "RH56 bounded final physical-tracking settle expired "
                            "without a verified result"
                        )
                    # Owner-thread only, no transport operation and no ledger
                    # mutation.  The physical no-motion tracker below remains
                    # authoritative and retains its original deadline.
                    actuator.refresh_supervised_sample_hold_heartbeat()
                    now = self._now()
                deadline = actuator.policy_watchdog_deadline_monotonic_s
                awaiting_first = actuator.awaiting_first_numeric_command
                if deadline is None and not awaiting_first:
                    raise RH56WatchdogOwnerError(
                        "RH56 watchdog deadline disappeared while running"
                    )
                feedback_due = bool(
                    self._supervised_target_only
                    and now >= next_feedback_deadline_s
                )
                feedback_expired = bool(
                    self._supervised_target_only
                    and now
                    >= last_feedback_captured_s + self._feedback_hard_age_s
                )
                with self._condition:
                    if self._stop_requested:
                        request = self._request
                        self._request = None
                        action = "stop"
                    elif self._final_tracking_settle_requested:
                        self._final_tracking_settle_requested = False
                        action = "begin_final_tracking_settle"
                    elif deadline is not None and now >= deadline:
                        request = self._request
                        self._request = None
                        action = "watchdog"
                    elif feedback_expired:
                        request = self._request
                        self._request = None
                        action = "feedback_stale"
                    elif self._request is not None:
                        request = self._request
                        self._request = None
                        self._request_inflight = True
                        action = "execute"
                    elif feedback_due:
                        action = "poll_feedback"
                    else:
                        if awaiting_first and not self._supervised_target_only:
                            self._condition.wait(
                                timeout=max(
                                    0.001,
                                    float(actuator.maximum_feedback_age_s) * 0.5,
                                )
                            )
                            if self._stop_requested or self._request is not None:
                                continue
                            action = "poll_before_first"
                        else:
                            wake_deadlines = []
                            if deadline is not None:
                                wake_deadlines.append(deadline)
                            if self._supervised_target_only:
                                wake_deadlines.append(next_feedback_deadline_s)
                            if (
                                self._final_tracking_settle_active
                                and self._final_tracking_settle_deadline_s
                                is not None
                            ):
                                wake_deadlines.append(
                                    self._final_tracking_settle_deadline_s
                                )
                            if not wake_deadlines:
                                raise RH56WatchdogOwnerError(
                                    "RH56 owner has no watchdog or feedback wake deadline"
                                )
                            self._condition.wait(
                                timeout=max(0.0, min(wake_deadlines) - now)
                            )
                            continue

                if action == "begin_final_tracking_settle":
                    actuator.refresh_supervised_sample_hold_heartbeat()
                    if (
                        self._feedback_period_s
                        >= actuator.inter_command_watchdog_timeout_s
                    ):
                        raise RH56WatchdogOwnerError(
                            "RH56 feedback period cannot sustain the bounded "
                            "final sample-hold heartbeat"
                        )
                    with self._condition:
                        if self._stop_requested:
                            continue
                        self._final_tracking_settle_active = True
                        self._condition.notify_all()
                    continue

                if action == "stop":
                    if request is not None:
                        reason = "RH56 owner stopped before queued sequence executed"
                        self._fail_pending_ledger(reason)
                        request.ticket._complete(
                            error=RH56WatchdogOwnerError(reason)
                        )
                    self._normal_disable(actuator)
                    break

                if action == "watchdog":
                    if request is not None:
                        request.ticket._complete(
                            error=RH56WatchdogOwnerError(
                                "RH56 watchdog expired before queued execution"
                            )
                        )
                    self._watchdog_expire(actuator, now=now)
                    break

                if action == "feedback_stale":
                    if request is not None:
                        request.ticket._complete(
                            error=RH56WatchdogOwnerError(
                                "RH56 complete feedback became stale before queued execution"
                            )
                        )
                    raise RH56WatchdogOwnerError(
                        "RH56 complete feedback hard-age deadline expired: "
                        f"age={now - last_feedback_captured_s:.6f}s, "
                        f"limit={self._feedback_hard_age_s:.6f}s"
                    )

                if action in ("poll_before_first", "poll_feedback"):
                    feedback = poll_owner_safety()
                    if (
                        self._supervised_target_only
                        and feedback.captured_monotonic_s
                        - last_feedback_captured_s
                        > self._feedback_hard_age_s
                    ):
                        raise RH56WatchdogOwnerError(
                            "RH56 complete feedback interval exceeded hard age: "
                            f"actual={feedback.captured_monotonic_s - last_feedback_captured_s:.6f}s, "
                            f"limit={self._feedback_hard_age_s:.6f}s"
                        )
                    self._cache_feedback(feedback)
                    last_feedback_captured_s = feedback.captured_monotonic_s
                    next_feedback_deadline_s = _advance_periodic_deadline(
                        next_feedback_deadline_s,
                        period_s=self._feedback_period_s,
                        now_s=last_feedback_captured_s,
                    )
                    continue

                assert request is not None
                try:
                    if self._supervised_target_only:
                        receipt = actuator.execute_target_and_ack(
                            request.command,
                            action_ledger=self._ledger,
                        )
                        if bool(
                            getattr(receipt, "numeric_write_performed", True)
                        ):
                            self._register_physical_tracking_target(
                                tuple(int(value) for value in receipt.exact_target),
                                now_monotonic_s=receipt.acknowledged_monotonic_s,
                            )
                        # The target ticket is deliberately completed before a
                        # due complete-feedback poll.  This preserves the 60 Hz
                        # ledger path while the same serial owner continues the
                        # independently bounded physical-feedback stream.
                        with self._condition:
                            self._request_inflight = False
                            self._condition.notify_all()
                        request.ticket._complete(receipt=receipt)
                        now_after_ack = self._now()
                        if (
                            now_after_ack
                            >= last_feedback_captured_s
                            + self._feedback_hard_age_s
                        ):
                            raise RH56WatchdogOwnerError(
                                "RH56 complete feedback became stale during target ACK"
                            )
                        if (
                            now_after_ack >= next_feedback_deadline_s
                        ):
                            feedback = poll_owner_safety()
                            if (
                                feedback.captured_monotonic_s
                                - last_feedback_captured_s
                                > self._feedback_hard_age_s
                            ):
                                raise RH56WatchdogOwnerError(
                                    "RH56 complete feedback interval exceeded hard age "
                                    "after target ACK"
                                )
                            self._cache_feedback(feedback)
                            last_feedback_captured_s = (
                                feedback.captured_monotonic_s
                            )
                            next_feedback_deadline_s = (
                                _advance_periodic_deadline(
                                    next_feedback_deadline_s,
                                    period_s=self._feedback_period_s,
                                    now_s=last_feedback_captured_s,
                                )
                            )
                    else:
                        receipt = actuator.execute(
                            request.command,
                            action_ledger=self._ledger,
                        )
                        self._cache_feedback(receipt.feedback)
                        last_feedback_captured_s = (
                            receipt.feedback.captured_monotonic_s
                        )
                        request.ticket._complete(receipt=receipt)
                except BaseException as exc:
                    reason = f"RH56 owner sequence {request.command.sequence}: {exc}"
                    self._publish_fault(reason)
                    stop_report: Optional[RH56StopReport] = None
                    stop_error: Optional[BaseException] = None
                    # execute() normally performs this itself.  The explicit
                    # fallback covers a gate/precondition failure before its
                    # internal transaction try-block.
                    if actuator.stop_confirmed:
                        self._disable_attempted = True
                        stop_report = None
                        with self._condition:
                            existing = self._stop_report
                        if existing is not None:
                            stop_report = existing
                    elif actuator.state in (
                        RH56ActuatorState.ARMED,
                        RH56ActuatorState.FAULT_LATCHED,
                    ):
                        self._disable_attempted = True
                        try:
                            stop_report = actuator.fault_and_disable(reason)
                        except BaseException as cleanup_exc:
                            stop_error = cleanup_exc
                    else:
                        self._disable_attempted = True
                        if actuator.stop_confirmed:
                            stop_error = None
                        else:
                            stop_error = RH56TransactionalError(
                                actuator.fault_reason
                                or "RH56 execute stop was unconfirmed"
                            )
                    self._finish_fault_stop(
                        stop_report=stop_report,
                        stop_error=stop_error,
                        stop_confirmed_override=actuator.stop_confirmed,
                    )
                    request.ticket._complete(error=exc)
                    break
                finally:
                    with self._condition:
                        self._request_inflight = False
                        self._condition.notify_all()

        except BaseException as exc:
            with self._condition:
                starting = self._state is RH56WatchdogOwnerState.STARTING
                if starting:
                    self._startup_error = exc
            if self._fault is None:
                self._publish_fault(f"RH56 owner worker failed: {exc}")
            if (
                actuator is not None
                and not actuator.stop_confirmed
                and actuator.state in (
                    RH56ActuatorState.ARMED,
                    RH56ActuatorState.FAULT_LATCHED,
                )
            ):
                reason = f"RH56 owner worker failed: {exc}"
                self._disable_attempted = True
                stop_report: Optional[RH56StopReport] = None
                stop_error: Optional[BaseException] = None
                try:
                    stop_report = actuator.fault_and_disable(reason)
                except BaseException as cleanup_exc:
                    stop_error = cleanup_exc
                self._finish_fault_stop(
                    stop_report=stop_report,
                    stop_error=stop_error,
                )
            elif actuator is not None and actuator.stop_confirmed:
                self._disable_attempted = True
                self._finish_fault_stop(
                    stop_report=self._stop_report,
                    stop_error=None,
                    stop_confirmed_override=True,
                )
        finally:
            # A returned armed session is never closed before a disable was
            # attempted.  A setup callback that fails before returning retains
            # responsibility for closing its own partial transport.
            if session is not None:
                if (
                    actuator is not None
                    and actuator.state is RH56ActuatorState.ARMED
                    and not self._disable_attempted
                ):
                    self._disable_attempted = True
                    try:
                        self._normal_disable(actuator)
                    except BaseException as exc:
                        self._publish_fault(
                            f"RH56 final disable before close failed: {exc}"
                        )
                        self._finish_fault_stop(
                            stop_report=None,
                            stop_error=exc,
                        )
                self._close_attempted = True
                try:
                    session.close()
                    self._serial_closed = True
                except BaseException as exc:
                    self._serial_closed = False
                    self._publish_fault(f"RH56 owned serial close failed: {exc}")
                    self._finish_fault_stop(
                        stop_report=self._stop_report,
                        stop_error=None,
                        stop_confirmed_override=bool(
                            actuator is not None and actuator.stop_confirmed
                        ),
                    )
            with self._condition:
                self._final_tracking_settle_requested = False
                self._final_tracking_settle_active = False
                self._final_tracking_settle_deadline_s = None
                if self._fault is None:
                    self._state = RH56WatchdogOwnerState.STOPPED
                else:
                    self._state = RH56WatchdogOwnerState.FAULT_LATCHED
                self._ready.set()
                self._worker_stopped.set()
                self._condition.notify_all()


__all__ = [
    "RH56FeedbackHistorySnapshot",
    "RH56OwnedSession",
    "RH56OwnerFault",
    "RH56PhysicalTrackingSnapshot",
    "RH56OwnerStopResult",
    "RH56OwnerTicket",
    "RH56TimestampedFeedback",
    "RH56WatchdogOwner",
    "RH56WatchdogOwnerError",
    "RH56WatchdogOwnerState",
]
