"""Dual-actuator acknowledgement ledger and atomic previous-action commit."""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from sim2real.contracts.v94 import INITIAL_PREVIOUS_ACTION13

from .authorization import ClosedLoopProtocolError
from .commands import ClosedLoopCommand, ExecutedActionCommit
from .constants import COMMAND_CONSUMERS
from .validation import _finite_scalar, _positive_integer, _readonly_float_vector


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

__all__ = ["ExecutedActionLedger"]
