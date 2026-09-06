"""Fail-closed, single-owner RH56 command transactions for V94.

This module is deliberately hardware-transport agnostic.  It performs no
serial discovery, opens no device and imports no RH56 implementation.  A
future hardware entry point must inject one already-reviewed transport and
must keep that transport private to :class:`RH56TransactionalActuator`.

The actuator starts disarmed.  Arming requires all three independent proofs:

* a PASS report from the offline C2 commissioning preflight;
* the exact run-scoped :class:`~sim2real.closed_loop_core.MotionAuthorization`;
* an already-armed :class:`~sim2real.closed_loop_core.ClosedLoopSafetyGate`
  carrying that same authorization.

There is no force/override path.  The formal transaction keeps its reviewed
four-exchange write, exact target readback and complete safety-feedback sample.
An explicitly supervised, non-C2 session may instead use the separate target
transaction: one ``ANGLE_SET`` write response, one exact readback, then the
RH56 side of the shared ledger acknowledgement.  It performs no feedback read
and cannot be entered from the formal arm path.  Every runtime failure is
sticky and immediately runs the complete physical-stop verification.

The transport protocol includes absolute deadlines, but an implementation
must additionally configure its underlying serial timeout so a blocked OS or
USB call cannot outlive those deadlines.  A Python post-call time check cannot
pre-empt a blocking driver call.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import threading
import time
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from sim2real.closed_loop_core import (
    AUTHORIZATION_SCOPE,
    ClosedLoopCommand,
    ClosedLoopProtocolError,
    ClosedLoopSafetyGate,
    ExecutedActionLedger,
    MotionAuthorization,
    SafetyState,
    SingleThreadOwner,
)

RH56_AXIS_COUNT = 6
RH56_DISABLED_TARGETS: Tuple[int, ...] = (-1,) * RH56_AXIS_COUNT
RH56_FAULT_STATUSES = frozenset((5, 6, 7))
RH56_IDLE_STATUSES = frozenset((2, 0xFF))
RH56_C2_READINESS_LEVEL = "C2_BOUNDED_CLOSED_LOOP"
RH56_MAX_POLICY_WATCHDOG_S = 0.050
# The operator-supervised path may wait through a recoverable observation or
# USB scheduling gap while holding the last already-bounded target.  That
# inter-command gap is a different contract from the freshness of a newly
# produced command: every new command must still complete inside
# ``RH56_MAX_POLICY_WATCHDOG_S``.  Formal C2 never uses this wider bound.
RH56_MAX_SUPERVISED_INTER_COMMAND_WATCHDOG_S = 0.500
# A 20-byte write plus its 9-byte ACK occupies about 2.52 ms at 115200 8N1,
# but the installed CH340/RS-485 path has repeatedly delivered otherwise
# valid replies after the old 8 ms host window (13/240 writes in one sustained
# 20 Hz run).  Cutting the exchange short can leave that late ACK in flight
# and make the following exact-readback request fail too.  Give the write one
# complete Linux serial-attempt window instead.  The original 50 ms absolute
# command deadline is unchanged, and a numeric request is still never resent.
RH56_SUPERVISED_WRITE_RESPONSE_GRACE_S = 0.020
# The Linux transport caps each read-only attempt at 20 ms and the complete
# read/retry transaction at 50 ms.  This actuator always supplies the original
# ``produced_monotonic_s + 50 ms`` command deadline, so Franka ACK and the
# numeric write consume part of this value rather than creating a new window.
RH56_SUPERVISED_EXCHANGE_TIMEOUT_S = 0.050
# Leave one additional bounded host-scheduling window after accounting for
# the serial exchanges and verification sleeps in the release proof.  This is
# deliberately separate from the policy-command watchdog: stop verification
# must remain possible after the policy stream has already faulted.
RH56_STOP_RELEASE_SCHEDULING_MARGIN_S = 0.100

# 8N1 serial framing uses ten wire bits for each byte.  The count below is the
# protocol minimum for one reviewed compact transaction:
#   ANGLE_SET write request/response                20 + 9 bytes
#   ANGLE_SET exact read request/response             9 + 20 bytes
#   POS_ACT+ANGLE_ACT read request/response            9 + 32 bytes
#   FORCE_ACT through TEMP read request/response       9 + 50 bytes
# It assumes four request/response turns and no retry, USB packet delay,
# RS-485 direction turnaround, firmware processing or host scheduling delay.
RH56_COMPACT_TRANSACTION_WIRE_BYTES = 158
RH56_COMPACT_TRANSACTION_EXCHANGES = 4
RH56_SERIAL_BITS_PER_BYTE_8N1 = 10
_VERIFIED_PREFLIGHT_SEAL = object()
_SUPERVISED_PREFLIGHT_SEAL = object()
_REQUIRED_RH56_PREFLIGHT_CHECKS = (
    "evidence_physical_deadman_acceptance",
    "evidence_physical_emergency_stop_acceptance",
    "evidence_rh56_single_owner_serial_session",
    "evidence_rh56_full_six_axis_and_fingertip_fk",
    "evidence_rh56_full_thumb_rotation_range",
    "evidence_rh56_60hz_write_readback",
    "evidence_rh56_fault_disable_and_verified_stop",
    "evidence_dual_device_same_sequence_ack_transaction",
    "evidence_dual_device_rate_and_max_gap",
    "evidence_metric_rh56_sustained_command_rate_hz",
    "evidence_metric_dual_device_sustained_ack_rate_hz",
    "evidence_metric_dual_device_max_interaction_gap_s",
    "evidence_metric_policy_command_watchdog_timeout_s",
)


class RH56TransactionalError(ClosedLoopProtocolError):
    """A terminal RH56 command, feedback or ownership failure."""


class RH56TransportError(RH56TransactionalError):
    """The injected transport failed or violated its data contract."""


class RH56DeadlineExceeded(RH56TransactionalError):
    """A command/feedback transport operation missed an absolute deadline."""


class RH56StopUnconfirmed(RH56TransactionalError):
    """In-place hold, stable stop, or final target release was not proven."""


class RH56ActuatorState(str, Enum):
    DISARMED = "disarmed"
    ARMED = "armed"
    FAULT_LATCHED = "fault_latched"


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_float(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _bounded_integer(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}") from exc
    if (
        not np.isfinite(numeric)
        or numeric != round(numeric)
        or not minimum <= int(numeric) <= maximum
    ):
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    return int(numeric)


def _six_integers(
    values: object,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> Tuple[int, ...]:
    try:
        raw = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise RH56TransportError(f"{name} must contain exactly six integers") from exc
    if len(raw) != RH56_AXIS_COUNT:
        raise RH56TransportError(f"{name} returned {len(raw)} values, expected six")
    try:
        return tuple(
            _bounded_integer(
                value,
                f"{name}[{index}]",
                minimum=minimum,
                maximum=maximum,
            )
            for index, value in enumerate(raw)
        )
    except ValueError as exc:
        raise RH56TransportError(str(exc)) from exc


def _six_targets(values: object, name: str, *, allow_disabled: bool) -> Tuple[int, ...]:
    return _six_integers(
        values,
        name,
        minimum=-1 if allow_disabled else 0,
        maximum=1000,
    )


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


@dataclass(frozen=True)
class RH56SafetyFeedback:
    """One transport-atomic logical RH56 safety/physical-state sample.

    A real compact transport may obtain this logical sample using the two
    contiguous blocks ``POS_ACT+ANGLE_ACT`` and ``FORCE_ACT`` through ``TEMP``.
    ``captured_monotonic_s`` is the completion
    time of the last block, not the start time of the first request.
    """

    captured_monotonic_s: float
    positions: Tuple[int, ...]
    angles: Tuple[int, ...]
    forces_g: Tuple[int, ...]
    currents_ma: Tuple[int, ...]
    errors: Tuple[int, ...]
    statuses: Tuple[int, ...]
    temperatures_c: Tuple[int, ...]

    def __post_init__(self) -> None:
        captured = _finite_float(self.captured_monotonic_s, "captured_monotonic_s")
        positions = _six_integers(
            self.positions, "positions", minimum=-32768, maximum=32767
        )
        angles = _six_integers(self.angles, "angles", minimum=-32768, maximum=32767)
        forces = _six_integers(
            self.forces_g, "forces_g", minimum=-32768, maximum=32767
        )
        currents = _six_integers(
            self.currents_ma, "currents_ma", minimum=-32768, maximum=32767
        )
        errors = _six_integers(self.errors, "errors", minimum=0, maximum=255)
        statuses = _six_integers(self.statuses, "statuses", minimum=0, maximum=255)
        temperatures = _six_integers(
            self.temperatures_c,
            "temperatures_c",
            minimum=0,
            maximum=255,
        )
        object.__setattr__(self, "captured_monotonic_s", captured)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "angles", angles)
        object.__setattr__(self, "forces_g", forces)
        object.__setattr__(self, "currents_ma", currents)
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "statuses", statuses)
        object.__setattr__(self, "temperatures_c", temperatures)


@runtime_checkable
class RH56TransactionalTransport(Protocol):
    """Deadline-aware low-level operations owned by exactly one thread.

    Implementations must never retry a numeric write.  Side-effect-free reads
    may use a documented finite retry count only inside the caller's original
    absolute deadline.  ``hardware_backed`` is audit metadata, not an
    authorization switch; the actuator applies the same preflight and motion
    gate to fake and real transports.
    """

    hardware_backed: bool
    baud_rate: int
    transport_name: str

    def write_angle_set(
        self,
        values: Tuple[int, ...],
        *,
        deadline_monotonic_s: float,
    ) -> None: ...

    def read_angle_set(self, *, deadline_monotonic_s: float) -> Sequence[int]: ...

    def read_safety_feedback(
        self, *, deadline_monotonic_s: float
    ) -> RH56SafetyFeedback: ...


def _canonical_preflight_digest(report: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            dict(report),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "commissioning preflight report must be canonical JSON data"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, init=False)
class RH56ActuationPreflight:
    """Digest-bound proof that C2 is eligible for operator authorization.

    This is derived from the offline report; it does not authorize motion.
    ``commissioning_run_id`` identifies the evidence-producing commissioning
    run.  It is deliberately distinct from the later execution run ID carried
    by :class:`MotionAuthorization`.
    The later :class:`MotionAuthorization` and live safety gate remain
    mandatory and independent.
    """

    commissioning_run_id: str
    report_sha256: str
    commissioning_profile_sha256: str
    verified_command_watchdog_timeout_s: float
    verified_rh56_command_rate_hz: float
    verified_dual_ack_rate_hz: float
    verified_dual_ack_max_gap_s: float
    readiness_level: str = RH56_C2_READINESS_LEVEL

    def __init__(
        self,
        *,
        commissioning_run_id: str,
        report_sha256: str,
        commissioning_profile_sha256: str,
        verified_command_watchdog_timeout_s: float,
        verified_rh56_command_rate_hz: float,
        verified_dual_ack_rate_hz: float,
        verified_dual_ack_max_gap_s: float,
        readiness_level: str = RH56_C2_READINESS_LEVEL,
        _verification_seal: object = None,
    ) -> None:
        if _verification_seal is not _VERIFIED_PREFLIGHT_SEAL:
            raise TypeError(
                "RH56ActuationPreflight must be created from a verified report"
            )
        evidence_run = str(commissioning_run_id).strip()
        digest = str(report_sha256).strip().lower()
        profile_digest = str(commissioning_profile_sha256).strip().lower()
        readiness = str(readiness_level).strip()
        watchdog = _positive_float(
            verified_command_watchdog_timeout_s,
            "verified_command_watchdog_timeout_s",
        )
        command_rate = _positive_float(
            verified_rh56_command_rate_hz, "verified_rh56_command_rate_hz"
        )
        dual_rate = _positive_float(
            verified_dual_ack_rate_hz, "verified_dual_ack_rate_hz"
        )
        dual_gap = _positive_float(
            verified_dual_ack_max_gap_s, "verified_dual_ack_max_gap_s"
        )
        if not evidence_run:
            raise ValueError("commissioning_run_id must be non-empty")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("preflight report_sha256 must be 64 lowercase hex digits")
        if len(profile_digest) != 64 or any(
            c not in "0123456789abcdef" for c in profile_digest
        ):
            raise ValueError(
                "commissioning_profile_sha256 must be 64 lowercase hex digits"
            )
        if readiness != RH56_C2_READINESS_LEVEL:
            raise ValueError("preflight readiness level is not V94 C2")
        if watchdog > RH56_MAX_POLICY_WATCHDOG_S:
            raise ValueError("verified RH56 watchdog exceeds the C2 maximum")
        if command_rate < 60.0 or dual_rate < 60.0:
            raise ValueError("verified RH56/dual acknowledgement rate is below 60 Hz")
        if dual_gap > 1.0 / 30.0:
            raise ValueError("verified dual acknowledgement gap exceeds 1/30 s")
        object.__setattr__(self, "commissioning_run_id", evidence_run)
        object.__setattr__(self, "report_sha256", digest)
        object.__setattr__(self, "commissioning_profile_sha256", profile_digest)
        object.__setattr__(self, "verified_command_watchdog_timeout_s", watchdog)
        object.__setattr__(self, "verified_rh56_command_rate_hz", command_rate)
        object.__setattr__(self, "verified_dual_ack_rate_hz", dual_rate)
        object.__setattr__(self, "verified_dual_ack_max_gap_s", dual_gap)
        object.__setattr__(self, "readiness_level", readiness)

    @classmethod
    def from_report(cls, report: Mapping[str, Any]) -> "RH56ActuationPreflight":
        if not isinstance(report, Mapping):
            raise TypeError("commissioning preflight report must be a mapping")
        required_exact = {
            "result": "PASS",
            "readiness_level": RH56_C2_READINESS_LEVEL,
            "offline_only": True,
            "device_access": False,
            "hardware_writes": False,
            "robot_command_writes": False,
            "arming_state": SafetyState.DISARMED.value.upper(),
            "motion_authorization_created": False,
            "physical_motion_authorized": False,
            "future_authorization_scope_required": AUTHORIZATION_SCOPE,
            "eligible_for_operator_authorization": True,
        }
        mismatches = [
            f"{name}={report.get(name)!r}"
            for name, expected in required_exact.items()
            if report.get(name) != expected
        ]
        failed_checks = report.get("failed_checks")
        if not isinstance(failed_checks, list) or failed_checks:
            mismatches.append(f"failed_checks={failed_checks!r}")
        blockers = report.get("blockers")
        if not isinstance(blockers, list) or blockers:
            mismatches.append(f"blockers={blockers!r}")
        run_id = report.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            mismatches.append(f"run_id={run_id!r}")

        inputs = report.get("inputs")
        hashes: Mapping[str, Any] = {}
        if isinstance(inputs, Mapping) and isinstance(inputs.get("sha256"), Mapping):
            hashes = inputs["sha256"]
        profile_sha256 = hashes.get("commissioning_profile_sha256")
        if not isinstance(profile_sha256, str):
            mismatches.append(
                f"commissioning_profile_sha256={profile_sha256!r}"
            )

        checks = report.get("checks")
        indexed: dict[str, Mapping[str, Any]] = {}
        duplicate_codes: set[str] = set()
        if isinstance(checks, Sequence) and not isinstance(checks, (str, bytes)):
            for item in checks:
                if isinstance(item, Mapping) and isinstance(item.get("code"), str):
                    code = str(item["code"])
                    if code in indexed:
                        duplicate_codes.add(code)
                    indexed[code] = item
        if duplicate_codes:
            mismatches.append(
                "duplicate check codes=" + ",".join(sorted(duplicate_codes))
            )
        for code in _REQUIRED_RH56_PREFLIGHT_CHECKS:
            if indexed.get(code, {}).get("passed") is not True:
                mismatches.append(code)

        def metric(code: str) -> float:
            value = indexed.get(code, {}).get("actual")
            if isinstance(value, (bool, np.bool_)):
                return math.nan
            try:
                result = float(value)
            except (TypeError, ValueError):
                return math.nan
            return result if math.isfinite(result) else math.nan

        command_rate = metric("evidence_metric_rh56_sustained_command_rate_hz")
        dual_rate = metric("evidence_metric_dual_device_sustained_ack_rate_hz")
        dual_gap = metric("evidence_metric_dual_device_max_interaction_gap_s")
        watchdog = metric("evidence_metric_policy_command_watchdog_timeout_s")
        if not command_rate >= 60.0:
            mismatches.append("rh56_sustained_command_rate_hz")
        if not dual_rate >= 60.0:
            mismatches.append("dual_device_sustained_ack_rate_hz")
        if not 0.0 < dual_gap <= 1.0 / 30.0:
            mismatches.append("dual_device_max_interaction_gap_s")
        if not 0.0 < watchdog <= RH56_MAX_POLICY_WATCHDOG_S:
            mismatches.append("policy_command_watchdog_timeout_s")
        if mismatches:
            raise ValueError(
                "commissioning preflight is not eligible: " + ", ".join(mismatches)
            )
        return cls(
            commissioning_run_id=run_id.strip(),
            report_sha256=_canonical_preflight_digest(report),
            commissioning_profile_sha256=profile_sha256,
            verified_command_watchdog_timeout_s=watchdog,
            verified_rh56_command_rate_hz=command_rate,
            verified_dual_ack_rate_hz=dual_rate,
            verified_dual_ack_max_gap_s=dual_gap,
            _verification_seal=_VERIFIED_PREFLIGHT_SEAL,
        )


@dataclass(frozen=True, init=False)
class SupervisedRH56Preflight:
    """Interactive-run permit; explicitly not C2 commissioning evidence."""

    commissioning_run_id: str
    report_sha256: str
    commissioning_profile_sha256: str
    verified_command_watchdog_timeout_s: float
    verified_inter_command_watchdog_timeout_s: float
    readiness_level: str = "EXPERIMENTAL_OPERATOR_SUPERVISED_NON_C2"
    authorizes_c2: bool = False

    def __init__(
        self,
        *,
        run_id: str,
        confirmed_permit_sha256: str,
        commissioning_profile_sha256: str,
        watchdog_timeout_s: object,
        inter_command_watchdog_timeout_s: object | None = None,
        _seal: object = None,
    ) -> None:
        if _seal is not _SUPERVISED_PREFLIGHT_SEAL:
            raise TypeError(
                "SupervisedRH56Preflight must come from the interactive runner"
            )
        active_run = str(run_id).strip()
        if not active_run:
            raise ValueError("run_id must be non-empty")
        permit = str(confirmed_permit_sha256).strip().lower()
        profile = str(commissioning_profile_sha256).strip().lower()
        for value, name in ((permit, "permit"), (profile, "profile")):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"supervised {name} SHA-256 is invalid")
        watchdog = _positive_float(watchdog_timeout_s, "watchdog_timeout_s")
        if watchdog > RH56_MAX_POLICY_WATCHDOG_S:
            raise ValueError("supervised RH56 watchdog exceeds 50ms")
        inter_watchdog = _positive_float(
            watchdog
            if inter_command_watchdog_timeout_s is None
            else inter_command_watchdog_timeout_s,
            "inter_command_watchdog_timeout_s",
        )
        if inter_watchdog < watchdog:
            raise ValueError(
                "supervised RH56 inter-command watchdog cannot be below "
                "the command freshness watchdog"
            )
        if inter_watchdog > RH56_MAX_SUPERVISED_INTER_COMMAND_WATCHDOG_S:
            raise ValueError("supervised RH56 inter-command watchdog exceeds 500ms")
        object.__setattr__(self, "commissioning_run_id", f"interactive:{active_run}")
        object.__setattr__(self, "report_sha256", permit)
        object.__setattr__(self, "commissioning_profile_sha256", profile)
        object.__setattr__(self, "verified_command_watchdog_timeout_s", watchdog)
        object.__setattr__(
            self,
            "verified_inter_command_watchdog_timeout_s",
            inter_watchdog,
        )
        object.__setattr__(self, "readiness_level", "EXPERIMENTAL_OPERATOR_SUPERVISED_NON_C2")
        object.__setattr__(self, "authorizes_c2", False)


def _issue_supervised_rh56_preflight(
    *,
    run_id: str,
    confirmed_permit_sha256: str,
    commissioning_profile_sha256: str,
    watchdog_timeout_s: object = RH56_MAX_POLICY_WATCHDOG_S,
    inter_command_watchdog_timeout_s: object | None = None,
) -> SupervisedRH56Preflight:
    return SupervisedRH56Preflight(
        run_id=run_id,
        confirmed_permit_sha256=confirmed_permit_sha256,
        commissioning_profile_sha256=commissioning_profile_sha256,
        watchdog_timeout_s=watchdog_timeout_s,
        inter_command_watchdog_timeout_s=inter_command_watchdog_timeout_s,
        _seal=_SUPERVISED_PREFLIGHT_SEAL,
    )


@dataclass(frozen=True)
class RH56TransactionReceipt:
    sequence: int
    acknowledged_monotonic_s: float
    latency_s: float
    exact_readback: Tuple[int, ...]
    feedback: RH56SafetyFeedback
    dual_device_commit_completed: bool


@dataclass(frozen=True)
class RH56TargetReceipt:
    """Target-only receipt for an operator-supervised session.

    A normal successful write ACK is sufficient to commit the target.  An
    exact ANGLE_SET readback is issued only when that ACK is missing, so the
    common 20 Hz path uses one serial exchange instead of two.
    """

    sequence: int
    acknowledged_monotonic_s: float
    latency_s: float
    exact_target: Tuple[int, ...]
    exact_readback: Tuple[int, ...]
    write_response_received: bool
    application_verified_by_readback: bool
    dual_device_commit_completed: bool
    numeric_write_performed: bool = True


@dataclass(frozen=True)
class RH56StopReport:
    verified: bool
    disable_passes_verified: int
    feedback_samples_verified: int
    completed_monotonic_s: float


@dataclass(frozen=True)
class RH56WireRateEstimate:
    baud_rate: int
    wire_bytes_per_transaction: int
    request_response_exchanges: int
    bits_per_byte: int
    ideal_wire_time_s: float
    ideal_wire_only_max_rate_hz: float
    ideal_wire_utilization_at_60hz: float
    wire_only_60hz_possible: bool
    real_hardware_60hz_verified: bool
    feasibility: str


def estimate_compact_transaction_wire_rate(
    baud_rate: object = 115200,
) -> RH56WireRateEstimate:
    """Return a lower-bound wire estimate, never a hardware feasibility claim."""

    baud = _bounded_integer(baud_rate, "baud_rate", minimum=1, maximum=10_000_000)
    bits = RH56_COMPACT_TRANSACTION_WIRE_BYTES * RH56_SERIAL_BITS_PER_BYTE_8N1
    wire_time = bits / float(baud)
    ideal_rate = 1.0 / wire_time
    utilization = 60.0 * wire_time
    return RH56WireRateEstimate(
        baud_rate=baud,
        wire_bytes_per_transaction=RH56_COMPACT_TRANSACTION_WIRE_BYTES,
        request_response_exchanges=RH56_COMPACT_TRANSACTION_EXCHANGES,
        bits_per_byte=RH56_SERIAL_BITS_PER_BYTE_8N1,
        ideal_wire_time_s=wire_time,
        ideal_wire_only_max_rate_hz=ideal_rate,
        ideal_wire_utilization_at_60hz=utilization,
        wire_only_60hz_possible=wire_time <= 1.0 / 60.0,
        real_hardware_60hz_verified=False,
        feasibility="UNKNOWN_UNTIL_HARDWARE_COMMISSIONING",
    )


class RH56TransactionalActuator:
    """Synchronous, exact-sequence owner for one injected RH56 transport."""

    def __init__(
        self,
        transport: RH56TransactionalTransport,
        *,
        command_watchdog_timeout_s: float = RH56_MAX_POLICY_WATCHDOG_S,
        supervised_inter_command_watchdog_timeout_s: Optional[float] = None,
        maximum_feedback_age_s: float = 0.025,
        maximum_running_axis_current_ma: int = 1400,
        maximum_temperature_c: int = 60,
        stop_timeout_s: float = 1.0,
        stop_verify_samples: int = 3,
        stop_verify_interval_s: float = 0.100,
        stop_max_axis_current_ma: int = 100,
        stop_settle_max_axis_current_ma: Optional[int] = None,
        stop_max_angle_drift_units: int = 2,
        stop_max_position_drift_units: int = 3,
        feedback_to_command_offset_units: Sequence[int] = (0,) * 6,
        stop_hold_command_minimum: Sequence[int] = (0,) * 6,
        stop_hold_command_maximum: Sequence[int] = (1000,) * 6,
        monotonic=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        if not isinstance(transport, RH56TransactionalTransport):
            raise TypeError("transport must implement RH56TransactionalTransport")
        if not callable(monotonic) or not callable(sleep):
            raise ValueError("monotonic and sleep must be callable")
        timeout = _positive_float(
            command_watchdog_timeout_s, "command_watchdog_timeout_s"
        )
        if timeout > RH56_MAX_POLICY_WATCHDOG_S:
            raise ValueError(
                f"command_watchdog_timeout_s must not exceed "
                f"{RH56_MAX_POLICY_WATCHDOG_S:.3f}s"
            )
        supervised_inter_timeout = _positive_float(
            timeout
            if supervised_inter_command_watchdog_timeout_s is None
            else supervised_inter_command_watchdog_timeout_s,
            "supervised_inter_command_watchdog_timeout_s",
        )
        if supervised_inter_timeout < timeout:
            raise ValueError(
                "supervised inter-command watchdog cannot be below command freshness"
            )
        if (
            supervised_inter_timeout
            > RH56_MAX_SUPERVISED_INTER_COMMAND_WATCHDOG_S
        ):
            raise ValueError(
                "supervised inter-command watchdog exceeds "
                f"{RH56_MAX_SUPERVISED_INTER_COMMAND_WATCHDOG_S:.3f}s"
            )
        feedback_age = _positive_float(maximum_feedback_age_s, "maximum_feedback_age_s")
        stop_timeout = _positive_float(stop_timeout_s, "stop_timeout_s")
        stop_interval = _finite_float(
            stop_verify_interval_s, "stop_verify_interval_s"
        )
        if stop_interval < 0.0:
            raise ValueError("stop_verify_interval_s cannot be negative")
        samples = _bounded_integer(
            stop_verify_samples, "stop_verify_samples", minimum=2, maximum=10
        )
        running_current = _bounded_integer(
            maximum_running_axis_current_ma,
            "maximum_running_axis_current_ma",
            minimum=1,
            maximum=32767,
        )
        stop_current = _bounded_integer(
            stop_max_axis_current_ma,
            "stop_max_axis_current_ma",
            minimum=1,
            maximum=32767,
        )
        stop_settle_current = _bounded_integer(
            running_current
            if stop_settle_max_axis_current_ma is None
            else stop_settle_max_axis_current_ma,
            "stop_settle_max_axis_current_ma",
            minimum=1,
            maximum=32767,
        )
        temperature = _bounded_integer(
            maximum_temperature_c,
            "maximum_temperature_c",
            minimum=1,
            maximum=255,
        )
        angle_drift = _bounded_integer(
            stop_max_angle_drift_units,
            "stop_max_angle_drift_units",
            minimum=0,
            maximum=1000,
        )
        position_drift = _bounded_integer(
            stop_max_position_drift_units,
            "stop_max_position_drift_units",
            minimum=0,
            maximum=1000,
        )
        feedback_offsets = _six_integers(
            feedback_to_command_offset_units,
            "feedback_to_command_offset_units",
            minimum=-100,
            maximum=100,
        )
        stop_hold_minimum = _six_targets(
            stop_hold_command_minimum,
            "stop_hold_command_minimum",
            allow_disabled=False,
        )
        stop_hold_maximum = _six_targets(
            stop_hold_command_maximum,
            "stop_hold_command_maximum",
            allow_disabled=False,
        )
        if any(
            lower >= upper
            for lower, upper in zip(stop_hold_minimum, stop_hold_maximum)
        ):
            raise ValueError("stop hold command intervals are invalid")
        transport_name = str(transport.transport_name).strip()
        if not transport_name:
            raise ValueError("transport_name must be non-empty")
        hardware_backed = _require_bool(
            transport.hardware_backed, "transport.hardware_backed"
        )
        baud = _bounded_integer(
            transport.baud_rate,
            "transport.baud_rate",
            minimum=1,
            maximum=10_000_000,
        )

        self._transport = transport
        self.transport_name = transport_name
        self.hardware_backed = hardware_backed
        self.baud_rate = baud
        self.command_watchdog_timeout_s = timeout
        self.supervised_inter_command_watchdog_timeout_s = (
            supervised_inter_timeout
        )
        self.maximum_feedback_age_s = feedback_age
        self.maximum_running_axis_current_ma = running_current
        self.maximum_temperature_c = temperature
        self.stop_timeout_s = stop_timeout
        self.stop_verify_samples = samples
        self.stop_verify_interval_s = stop_interval
        self.stop_max_axis_current_ma = stop_current
        self.stop_settle_max_axis_current_ma = stop_settle_current
        self.stop_max_angle_drift_units = angle_drift
        self.stop_max_position_drift_units = position_drift
        self.feedback_to_command_offset_units = feedback_offsets
        self.stop_hold_command_minimum = stop_hold_minimum
        self.stop_hold_command_maximum = stop_hold_maximum
        self._monotonic = monotonic
        self._sleep = sleep
        self._owner = SingleThreadOwner(f"RH56 transport {transport_name}")
        self._state_lock = threading.Lock()
        self._state = RH56ActuatorState.DISARMED
        self._fault_reason: Optional[str] = None
        self._stop_confirmed = False
        self._run_id: Optional[str] = None
        self._authorization_id: Optional[str] = None
        self._preflight_sha256: Optional[str] = None
        self._safety_gate: Optional[ClosedLoopSafetyGate] = None
        self._pending_sequence: Optional[int] = None
        self._last_acknowledged_sequence = 0
        self._armed_monotonic_s: Optional[float] = None
        self._last_command_monotonic_s: Optional[float] = None
        self._last_feedback_monotonic_s: Optional[float] = None
        self._supervised_session = False
        self._supervised_awaiting_first_numeric_command = False
        self._last_supervised_target: Optional[Tuple[int, ...]] = None

    @property
    def state(self) -> RH56ActuatorState:
        with self._state_lock:
            return self._state

    @property
    def fault_reason(self) -> Optional[str]:
        with self._state_lock:
            return self._fault_reason

    @property
    def stop_confirmed(self) -> bool:
        with self._state_lock:
            return self._stop_confirmed

    @property
    def pending_sequence(self) -> Optional[int]:
        with self._state_lock:
            return self._pending_sequence

    @property
    def last_acknowledged_sequence(self) -> int:
        with self._state_lock:
            return self._last_acknowledged_sequence

    @property
    def owner_ident(self) -> Optional[int]:
        return self._owner.owner_ident

    @property
    def policy_watchdog_deadline_monotonic_s(self) -> Optional[float]:
        """Return the exact inter-command heartbeat deadline without IO.

        The value is available only while the actuator is armed.  Reading it
        is intentionally thread-safe so a lifecycle wrapper can expose audit
        state, but only the claimed owner may act on the deadline or touch the
        transport.  Formal C2 uses the 50 ms command watchdog.  The supervised
        path uses its separately sealed inter-command bound; this does not
        change the 50 ms produced-command transaction deadline.
        """

        with self._state_lock:
            if self._state is not RH56ActuatorState.ARMED:
                return None
            heartbeat = self._last_command_monotonic_s
            supervised = self._supervised_session
        if heartbeat is None:
            return None
        timeout = (
            self.supervised_inter_command_watchdog_timeout_s
            if supervised
            else self.command_watchdog_timeout_s
        )
        return heartbeat + timeout

    @property
    def inter_command_watchdog_timeout_s(self) -> float:
        """Current mode's sealed ACK-to-next-command watchdog bound."""

        with self._state_lock:
            supervised = self._supervised_session
        return (
            self.supervised_inter_command_watchdog_timeout_s
            if supervised
            else self.command_watchdog_timeout_s
        )

    @property
    def awaiting_first_numeric_command(self) -> bool:
        """True only for supervised arm before any numeric ANGLE_SET write."""

        with self._state_lock:
            return bool(
                self._state is RH56ActuatorState.ARMED
                and self._supervised_awaiting_first_numeric_command
                and self._last_command_monotonic_s is None
            )

    def claim_for_current_thread(self) -> None:
        self._owner.claim_for_current_thread()

    def _now(self) -> float:
        return _finite_float(self._monotonic(), "monotonic clock")

    def _require_owner(self) -> None:
        self._owner.require_current_thread()

    def arm(
        self,
        *,
        preflight: RH56ActuationPreflight,
        authorization: MotionAuthorization,
        safety_gate: ClosedLoopSafetyGate,
        run_id: str,
        now_monotonic_s: Optional[float] = None,
    ) -> None:
        """Formal C2 arm path; rejects experimental supervised permits."""

        if not isinstance(preflight, RH56ActuationPreflight):
            raise TypeError("C2 arm requires RH56ActuationPreflight")
        self._arm_with_preflight(
            preflight=preflight,
            authorization=authorization,
            safety_gate=safety_gate,
            run_id=run_id,
            now_monotonic_s=now_monotonic_s,
        )

    def arm_supervised(
        self,
        *,
        preflight: SupervisedRH56Preflight,
        authorization: MotionAuthorization,
        safety_gate: ClosedLoopSafetyGate,
        run_id: str,
        now_monotonic_s: Optional[float] = None,
    ) -> None:
        """Experimental non-C2 arm path; rejects formal C2 tokens."""

        if not isinstance(preflight, SupervisedRH56Preflight):
            raise TypeError("supervised arm requires SupervisedRH56Preflight")
        if preflight.authorizes_c2 is not False:
            raise RH56TransactionalError("supervised permit classification is invalid")
        self._arm_with_preflight(
            preflight=preflight,
            authorization=authorization,
            safety_gate=safety_gate,
            run_id=run_id,
            now_monotonic_s=now_monotonic_s,
        )

    def _arm_with_preflight(
        self,
        *,
        preflight: RH56ActuationPreflight | SupervisedRH56Preflight,
        authorization: MotionAuthorization,
        safety_gate: ClosedLoopSafetyGate,
        run_id: str,
        now_monotonic_s: Optional[float] = None,
    ) -> None:
        """Bind the exact preflight, authorization and live gate without IO."""

        self._require_owner()
        if not isinstance(preflight, (RH56ActuationPreflight, SupervisedRH56Preflight)):
            raise TypeError("preflight must be a sealed RH56 preflight")
        if not isinstance(authorization, MotionAuthorization):
            raise TypeError("authorization must be MotionAuthorization")
        if not isinstance(safety_gate, ClosedLoopSafetyGate):
            raise TypeError("safety_gate must be ClosedLoopSafetyGate")
        active_run = str(run_id).strip()
        if not active_run:
            raise ValueError("run_id must be non-empty")
        now = (
            self._now()
            if now_monotonic_s is None
            else _finite_float(now_monotonic_s, "now_monotonic_s")
        )
        with self._state_lock:
            if self._state is RH56ActuatorState.FAULT_LATCHED:
                raise RH56TransactionalError(
                    f"RH56 fault is latched: {self._fault_reason}"
                )
            if self._state is not RH56ActuatorState.DISARMED:
                raise RH56TransactionalError("RH56 actuator is already armed")
        if preflight.commissioning_run_id == active_run:
            raise RH56TransactionalError(
                "RH56 execution run ID must differ from the commissioning run ID"
            )
        if not math.isclose(
            preflight.verified_command_watchdog_timeout_s,
            self.command_watchdog_timeout_s,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RH56TransactionalError(
                "RH56 command watchdog differs from the commissioned preflight"
            )
        if isinstance(preflight, SupervisedRH56Preflight) and not math.isclose(
            preflight.verified_inter_command_watchdog_timeout_s,
            self.supervised_inter_command_watchdog_timeout_s,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RH56TransactionalError(
                "RH56 supervised inter-command watchdog differs from preflight"
            )
        if authorization.run_id != active_run:
            raise RH56TransactionalError("RH56 authorization run ID mismatch")
        if not (
            authorization.issued_monotonic_s <= now < authorization.expires_monotonic_s
        ):
            raise RH56TransactionalError("RH56 motion authorization is not active")
        if safety_gate.state is not SafetyState.ARMED:
            raise RH56TransactionalError("closed-loop safety gate is not armed")
        if safety_gate.authorization_id != authorization.authorization_id:
            raise RH56TransactionalError(
                "closed-loop safety gate authorization does not match RH56"
            )
        safety_gate.require_motion(run_id=active_run, now_monotonic_s=now)
        with self._state_lock:
            self._run_id = active_run
            self._authorization_id = authorization.authorization_id
            self._preflight_sha256 = preflight.report_sha256
            self._safety_gate = safety_gate
            self._pending_sequence = None
            self._last_acknowledged_sequence = 0
            self._armed_monotonic_s = now
            supervised = isinstance(preflight, SupervisedRH56Preflight)
            self._supervised_session = supervised
            self._last_command_monotonic_s = None if supervised else now
            self._supervised_awaiting_first_numeric_command = supervised
            self._last_supervised_target = None
            self._last_feedback_monotonic_s = now
            self._stop_confirmed = False
            self._state = RH56ActuatorState.ARMED

    def _require_armed(self, now: float) -> tuple[str, ClosedLoopSafetyGate]:
        with self._state_lock:
            if self._state is RH56ActuatorState.FAULT_LATCHED:
                raise RH56TransactionalError(
                    f"RH56 fault is latched: {self._fault_reason}"
                )
            if self._state is not RH56ActuatorState.ARMED:
                raise RH56TransactionalError("RH56 actuator is disarmed")
            run_id = self._run_id
            gate = self._safety_gate
        if run_id is None or gate is None:
            raise RH56TransactionalError("RH56 armed state is incomplete")
        gate.require_motion(run_id=run_id, now_monotonic_s=now)
        return run_id, gate

    @staticmethod
    def _before_deadline(now: float, deadline: float, operation: str) -> None:
        if now > deadline:
            raise RH56DeadlineExceeded(
                f"RH56 {operation} missed deadline by {now - deadline:.6f}s"
            )

    def _transport_write(
        self, targets: Tuple[int, ...], *, deadline: float, operation: str
    ) -> None:
        self._before_deadline(self._now(), deadline, operation)
        try:
            self._transport.write_angle_set(targets, deadline_monotonic_s=deadline)
        except BaseException as exc:
            raise RH56TransportError(f"{operation} failed: {exc}") from exc
        self._before_deadline(self._now(), deadline, operation)

    def _transport_read_targets(
        self, *, deadline: float, operation: str
    ) -> Tuple[int, ...]:
        self._before_deadline(self._now(), deadline, operation)
        try:
            raw = self._transport.read_angle_set(deadline_monotonic_s=deadline)
        except BaseException as exc:
            raise RH56TransportError(f"{operation} failed: {exc}") from exc
        result = _six_targets(raw, f"{operation} readback", allow_disabled=True)
        self._before_deadline(self._now(), deadline, operation)
        return result

    def _transport_write_once_then_verify(
        self,
        targets: Tuple[int, ...],
        *,
        deadline: float,
        operation: str,
    ) -> Tuple[int, ...]:
        """Issue one target write, then prove it with a read-only readback.

        Stop handling can have a multi-second overall deadline, but one missing
        write ACK must not consume that budget and prevent the mandatory
        disable passes.  The response wait is therefore capped at the same
        reviewed 20 ms used by supervised target writes.  A timed-out write is
        never repeated here: one read-only target query determines whether the
        hand applied it.  The concrete Linux transport independently caps that
        single read without extending ``deadline``.
        """

        write_failure: Optional[BaseException] = None
        write_deadline = min(
            deadline,
            self._now() + RH56_SUPERVISED_WRITE_RESPONSE_GRACE_S,
        )
        try:
            self._transport_write(
                targets,
                deadline=write_deadline,
                operation=f"{operation} write",
            )
        except BaseException as exc:
            write_failure = exc
        try:
            readback = self._transport_read_targets(
                deadline=deadline,
                operation=f"{operation} readback",
            )
        except BaseException as exc:
            if write_failure is not None:
                raise RH56TransportError(
                    f"{operation} write response missing and verification "
                    f"readback failed: write_error={write_failure}; "
                    f"readback_error={exc}"
                ) from exc
            raise
        if readback != targets:
            write_context = (
                ""
                if write_failure is None
                else f"; write_response_error={write_failure}"
            )
            raise RH56TransportError(
                f"{operation} readback mismatch: expected={targets}, "
                f"actual={readback}{write_context}"
            )
        return readback

    def _transport_read_feedback(
        self, *, deadline: float, operation: str
    ) -> RH56SafetyFeedback:
        self._before_deadline(self._now(), deadline, operation)
        try:
            feedback = self._transport.read_safety_feedback(
                deadline_monotonic_s=deadline
            )
        except BaseException as exc:
            raise RH56TransportError(f"{operation} failed: {exc}") from exc
        if not isinstance(feedback, RH56SafetyFeedback):
            raise RH56TransportError(f"{operation} did not return RH56SafetyFeedback")
        finished = self._now()
        self._before_deadline(finished, deadline, operation)
        age = finished - feedback.captured_monotonic_s
        if age < 0.0:
            raise RH56TransportError(f"{operation} feedback timestamp is in the future")
        if age > self.maximum_feedback_age_s:
            raise RH56TransportError(
                f"{operation} feedback is stale: age={age:.6f}s, "
                f"limit={self.maximum_feedback_age_s:.6f}s"
            )
        with self._state_lock:
            previous = self._last_feedback_monotonic_s
        if previous is not None and feedback.captured_monotonic_s < previous:
            raise RH56TransportError(f"{operation} feedback timestamp regressed")
        return feedback

    def _validate_running_feedback(self, feedback: RH56SafetyFeedback) -> None:
        if any(feedback.errors):
            raise RH56TransactionalError(
                f"RH56 actuator ERROR is nonzero: {feedback.errors}"
            )
        if max(feedback.temperatures_c) >= self.maximum_temperature_c:
            raise RH56TransactionalError(
                f"RH56 temperature reached {self.maximum_temperature_c} C: "
                f"{feedback.temperatures_c}"
            )
        for index, status in enumerate(feedback.statuses):
            if status in RH56_FAULT_STATUSES:
                raise RH56TransactionalError(
                    f"RH56 axis {index} returned fault status {status}"
                )
        if any(
            abs(current) > self.maximum_running_axis_current_ma
            for current in feedback.currents_ma
        ):
            raise RH56TransactionalError(
                "RH56 running current exceeded the commissioned per-axis "
                f"limit {self.maximum_running_axis_current_ma}mA: "
                f"{feedback.currents_ma}"
            )

    def _record_feedback(self, feedback: RH56SafetyFeedback) -> None:
        with self._state_lock:
            self._last_feedback_monotonic_s = feedback.captured_monotonic_s

    def _latch_fault(self, reason: str) -> None:
        detail = str(reason).strip() or "unspecified RH56 transactional fault"
        with self._state_lock:
            if self._fault_reason is None:
                self._fault_reason = detail
            self._state = RH56ActuatorState.FAULT_LATCHED
            self._stop_confirmed = False

    def _command_deadline(self, command: ClosedLoopCommand) -> float:
        return command.produced_monotonic_s + self.command_watchdog_timeout_s

    def execute(
        self,
        command: ClosedLoopCommand,
        *,
        action_ledger: ExecutedActionLedger,
    ) -> RH56TransactionReceipt:
        """Execute and ACK exactly one already-staged V94 command.

        The call is synchronous; therefore this owner can never have two
        device sequences in flight.  On an active-session failure it latches,
        marks the ledger failed and performs the complete stop protocol before
        returning an exception.
        """

        self._require_owner()
        if not isinstance(command, ClosedLoopCommand):
            raise TypeError("command must be ClosedLoopCommand")
        if not isinstance(action_ledger, ExecutedActionLedger):
            raise TypeError("action_ledger must be ExecutedActionLedger")
        now = self._now()
        self._require_armed(now)
        sequence = command.sequence
        try:
            with self._state_lock:
                command_heartbeat = self._last_command_monotonic_s
                awaiting_first = self._supervised_awaiting_first_numeric_command
                if command_heartbeat is None:
                    if not (awaiting_first and sequence == 1):
                        raise RH56TransactionalError(
                            "RH56 command heartbeat is absent outside supervised sequence 1"
                        )
                elif now - command_heartbeat > self.command_watchdog_timeout_s:
                    raise RH56TransactionalError(
                        "RH56 policy command watchdog expired before sequence"
                    )
                if self._pending_sequence is not None:
                    raise RH56TransactionalError(
                        "RH56 cannot start a second sequence while one is pending"
                    )
                expected = self._last_acknowledged_sequence + 1
                if sequence != expected:
                    raise RH56TransactionalError(
                        f"RH56 stale/out-of-order sequence: expected={expected}, "
                        f"actual={sequence}"
                    )
                if action_ledger.pending_sequence != sequence:
                    raise RH56TransactionalError(
                        f"RH56 sequence {sequence} is not the ledger pending sequence "
                        f"{action_ledger.pending_sequence}"
                    )
                self._pending_sequence = sequence
            deadline = self._command_deadline(command)
            self._before_deadline(now, deadline, "command start")
            targets = _six_targets(
                command.rh56_angle_set_register_order,
                "command ANGLE_SET",
                allow_disabled=False,
            )
            self._transport_write(
                targets, deadline=deadline, operation="ANGLE_SET batch write"
            )
            readback = self._transport_read_targets(
                deadline=deadline, operation="ANGLE_SET exact readback"
            )
            if readback != targets:
                raise RH56TransactionalError(
                    f"RH56 ANGLE_SET readback mismatch: expected={targets}, "
                    f"actual={readback}"
                )
            feedback = self._transport_read_feedback(
                deadline=deadline, operation="command safety feedback"
            )
            self._validate_running_feedback(feedback)
            acknowledged_at = self._now()
            self._before_deadline(acknowledged_at, deadline, "command acknowledgement")
            self._require_armed(acknowledged_at)
            if action_ledger.pending_sequence != sequence:
                raise RH56TransactionalError(
                    "ledger pending sequence changed before RH56 acknowledgement"
                )
            committed = action_ledger.acknowledge(
                "rh56",
                sequence=sequence,
                now_monotonic_s=acknowledged_at,
            )
            self._record_feedback(feedback)
            with self._state_lock:
                self._last_acknowledged_sequence = sequence
                self._last_command_monotonic_s = acknowledged_at
                self._supervised_awaiting_first_numeric_command = False
                self._pending_sequence = None
            return RH56TransactionReceipt(
                sequence=sequence,
                acknowledged_monotonic_s=acknowledged_at,
                latency_s=acknowledged_at - command.produced_monotonic_s,
                exact_readback=readback,
                feedback=feedback,
                dual_device_commit_completed=committed,
            )
        except BaseException as exc:
            try:
                action_ledger.fail("rh56", sequence=sequence, reason=str(exc))
            except BaseException:
                pass
            self._latch_fault(str(exc))
            stop_error: Optional[BaseException] = None
            try:
                self._disable_and_verify(preserve_fault=True)
            except BaseException as cleanup_exc:
                stop_error = cleanup_exc
            detail = f"RH56 sequence {sequence} failed: {exc}"
            if stop_error is not None:
                detail += f"; STOP UNCONFIRMED: {stop_error}"
            raise RH56TransactionalError(detail) from exc

    def execute_target_and_ack(
        self,
        command: ClosedLoopCommand,
        *,
        action_ledger: ExecutedActionLedger,
    ) -> RH56TargetReceipt:
        """Apply one supervised target using exactly write + exact readback.

        This transaction deliberately performs no safety-feedback read.  It is
        available only after :meth:`arm_supervised`; formal C2 callers retain
        :meth:`execute` and its reviewed four wire exchanges.  A failure is
        terminal and never retries the numeric target because a missing write
        response leaves application ambiguous.
        """

        self._require_owner()
        if not isinstance(command, ClosedLoopCommand):
            raise TypeError("command must be ClosedLoopCommand")
        if not isinstance(action_ledger, ExecutedActionLedger):
            raise TypeError("action_ledger must be ExecutedActionLedger")
        now = self._now()
        self._require_armed(now)
        sequence = command.sequence
        targets = _six_targets(
            command.rh56_angle_set_register_order,
            "supervised target ANGLE_SET",
            allow_disabled=False,
        )
        with self._state_lock:
            supervised_session = self._supervised_session
        if not supervised_session:
            raise RH56TransactionalError(
                "RH56 target-only transaction requires arm_supervised: "
                f"sequence={sequence} exact_target={targets} phase=PRECHECK "
                "APPLY_UNKNOWN=false RETRY_FORBIDDEN"
            )

        phase = "PRECHECK"
        apply_unknown = False
        try:
            with self._state_lock:
                command_heartbeat = self._last_command_monotonic_s
                awaiting_first = self._supervised_awaiting_first_numeric_command
                if command_heartbeat is None:
                    if not (awaiting_first and sequence == 1):
                        raise RH56TransactionalError(
                            "RH56 command heartbeat is absent outside "
                            "supervised sequence 1"
                        )
                elif (
                    now - command_heartbeat
                    > self.supervised_inter_command_watchdog_timeout_s
                ):
                    raise RH56TransactionalError(
                        "RH56 supervised inter-command watchdog expired "
                        "before sequence"
                    )
                if self._pending_sequence is not None:
                    raise RH56TransactionalError(
                        "RH56 cannot start a second sequence while one is pending"
                    )
                expected = self._last_acknowledged_sequence + 1
                if sequence != expected:
                    raise RH56TransactionalError(
                        f"RH56 stale/out-of-order sequence: expected={expected}, "
                        f"actual={sequence}"
                    )
                if action_ledger.pending_sequence != sequence:
                    raise RH56TransactionalError(
                        f"RH56 sequence {sequence} is not the ledger pending sequence "
                        f"{action_ledger.pending_sequence}"
                    )
                self._pending_sequence = sequence

            deadline = self._command_deadline(command)
            self._before_deadline(now, deadline, "supervised target start")
            with self._state_lock:
                previous_target = self._last_supervised_target
            numeric_write_performed = previous_target != targets
            write_failure: Optional[BaseException] = None
            if numeric_write_performed:
                phase = "ANGLE_SET_WRITE"
                apply_unknown = True
                write_deadline = min(
                    deadline,
                    self._now() + RH56_SUPERVISED_WRITE_RESPONSE_GRACE_S,
                )
                # Once the numeric request has been attempted it is never
                # resent: a missing response cannot tell us whether the hand
                # applied it.  Reserve the second exchange exclusively for
                # one exact readback.
                self._before_deadline(
                    self._now(),
                    write_deadline,
                    "supervised ANGLE_SET batch write",
                )
                try:
                    self._transport_write(
                        targets,
                        deadline=write_deadline,
                        operation="supervised ANGLE_SET batch write",
                    )
                except BaseException as exc:
                    write_failure = exc
                if write_failure is None:
                    # The device returned the protocol's successful write ACK.
                    # Do not immediately query ANGLE_SET again: the separate
                    # 20 Hz feedback stream already owns the remaining serial
                    # budget and provides physical ANGLE_ACT/POS_ACT progress.
                    readback = targets
                else:
                    phase = "ANGLE_SET_EXACT_READBACK"
                    # A missing ACK is ambiguous: never resend the numeric
                    # write.  Use exactly one side-effect-free read to prove
                    # whether the target was applied inside the original
                    # produced-command deadline.
                    try:
                        readback = self._transport_read_targets(
                            deadline=deadline,
                            operation="supervised ANGLE_SET exact readback",
                        )
                    except BaseException as exc:
                        raise RH56TransactionalError(
                            "ANGLE_SET write response missing and exact verification "
                            f"readback failed: write_error={write_failure}; "
                            f"readback_error={exc}"
                        ) from exc
                    if readback != targets:
                        raise RH56TransactionalError(
                            f"RH56 ANGLE_SET readback mismatch: expected={targets}, "
                            f"actual={readback}; write_response_error={write_failure}"
                        )
            else:
                # The single owner holds an exclusive serial descriptor and
                # is the only code path that can change ANGLE_SET.  Reusing a
                # previously exact-readback target therefore requires no wire
                # transaction.  Avoiding same-value rewrites is important:
                # the installed firmware can restart its motion plan on every
                # numeric write.
                phase = "EXCLUSIVE_SESSION_SAMPLE_HOLD"
                readback = targets
            apply_unknown = False
            phase = "LEDGER_ACK"
            acknowledged_at = self._now()
            self._before_deadline(
                acknowledged_at,
                deadline,
                "supervised target acknowledgement",
            )
            self._require_armed(acknowledged_at)
            if action_ledger.pending_sequence != sequence:
                raise RH56TransactionalError(
                    "ledger pending sequence changed before RH56 acknowledgement"
                )
            committed = action_ledger.acknowledge(
                "rh56",
                sequence=sequence,
                now_monotonic_s=acknowledged_at,
            )
            with self._state_lock:
                self._last_acknowledged_sequence = sequence
                self._last_command_monotonic_s = acknowledged_at
                self._supervised_awaiting_first_numeric_command = False
                self._last_supervised_target = targets
                self._pending_sequence = None
            return RH56TargetReceipt(
                sequence=sequence,
                acknowledged_monotonic_s=acknowledged_at,
                latency_s=acknowledged_at - command.produced_monotonic_s,
                exact_target=targets,
                exact_readback=readback,
                write_response_received=(
                    write_failure is None if numeric_write_performed else False
                ),
                application_verified_by_readback=bool(
                    numeric_write_performed and write_failure is not None
                ),
                dual_device_commit_completed=committed,
                numeric_write_performed=numeric_write_performed,
            )
        except BaseException as exc:
            detail = (
                "RH56 supervised target transaction failed: "
                f"sequence={sequence} exact_target={targets} phase={phase} "
                f"APPLY_UNKNOWN={'true' if apply_unknown else 'false'} "
                f"RETRY_FORBIDDEN: {exc}"
            )
            try:
                action_ledger.fail("rh56", sequence=sequence, reason=detail)
            except BaseException:
                pass
            # Supervised owner coordination is intentionally two-stage: make
            # the exact target failure visible first so its fault callback can
            # request Franka stop, then let that owner call fault_and_disable().
            # Unlike formal execute(), this method never blocks the fault edge
            # behind synchronous RH56 stop verification.
            self._latch_fault(detail)
            raise RH56TransactionalError(detail) from exc

    def refresh_supervised_sample_hold_heartbeat(self) -> float:
        """Keep an already-committed supervised target alive without RH56 IO.

        This narrow owner-thread operation exists only for the bounded final
        physical-tracking settle phase.  It is not a policy action: it neither
        writes ``ANGLE_SET`` nor reads the transport, advances a sequence, or
        acknowledges the action ledger.  The existing heartbeat must still be
        live when the settle phase begins; callers cannot use this method to
        revive an expired command stream.
        """

        self._require_owner()
        now = self._now()
        self._require_armed(now)
        with self._state_lock:
            heartbeat = self._last_command_monotonic_s
            supervised = self._supervised_session
            awaiting_first = self._supervised_awaiting_first_numeric_command
            pending = self._pending_sequence
            acknowledged = self._last_acknowledged_sequence
            target = self._last_supervised_target
            if not supervised:
                raise RH56TransactionalError(
                    "RH56 sample-hold heartbeat requires supervised mode"
                )
            if awaiting_first or heartbeat is None or target is None:
                raise RH56TransactionalError(
                    "RH56 sample-hold heartbeat requires a committed target"
                )
            if pending is not None:
                raise RH56TransactionalError(
                    "RH56 sample-hold heartbeat is forbidden during a "
                    "pending sequence"
                )
            if acknowledged < 1:
                raise RH56TransactionalError(
                    "RH56 sample-hold heartbeat has no acknowledged sequence"
                )
            if (
                now - heartbeat
                > self.supervised_inter_command_watchdog_timeout_s
            ):
                raise RH56TransactionalError(
                    "RH56 supervised inter-command watchdog expired before "
                    "sample-hold heartbeat"
                )
            self._last_command_monotonic_s = now
        return now

    def poll_safety(
        self,
        *,
        defer_failure_cleanup: bool = False,
    ) -> RH56SafetyFeedback:
        """Poll feedback while holding the most recent target, with watchdog.

        The default preserves the formal synchronous stop behavior.  A
        supervised owner may defer cleanup so it can publish the fault and
        request Franka stop before calling :meth:`fault_and_disable` itself.
        """

        self._require_owner()
        defer_cleanup = _require_bool(
            defer_failure_cleanup,
            "defer_failure_cleanup",
        )
        now = self._now()
        self._require_armed(now)
        with self._state_lock:
            heartbeat = self._last_command_monotonic_s
            awaiting_first = self._supervised_awaiting_first_numeric_command
            supervised = self._supervised_session
        if heartbeat is None and awaiting_first:
            # No numeric target exists yet, so this read-only readiness sample
            # is not constrained by the post-command 50 ms motion watchdog.
            # Supervised deployment may use its (still bounded) complete-
            # feedback freshness allowance, leaving room for one compact-read
            # retry before any ANGLE_SET write is possible.
            deadline = now + max(
                self.command_watchdog_timeout_s,
                self.maximum_feedback_age_s,
            )
            try:
                feedback = self._transport_read_feedback(
                    deadline=deadline,
                    operation="pre-first-command safety feedback",
                )
                self._validate_running_feedback(feedback)
                self._record_feedback(feedback)
                return feedback
            except BaseException as exc:
                self._latch_fault(str(exc))
                if not defer_cleanup:
                    try:
                        self._disable_and_verify(preserve_fault=True)
                    except BaseException:
                        pass
                raise RH56TransactionalError(
                    f"RH56 pre-first-command safety poll failed: {exc}"
                ) from exc
        inter_command_timeout = (
            self.supervised_inter_command_watchdog_timeout_s
            if supervised
            else self.command_watchdog_timeout_s
        )
        if heartbeat is None or now - heartbeat > inter_command_timeout:
            reason = "RH56 policy command watchdog expired before safety poll"
            self._latch_fault(reason)
            if not defer_cleanup:
                try:
                    self._disable_and_verify(preserve_fault=True)
                except BaseException as exc:
                    reason += f"; STOP UNCONFIRMED: {exc}"
            raise RH56TransactionalError(reason)
        # A wider supervised ACK-to-command allowance never widens one serial
        # transaction.  The read itself remains bounded by the 50 ms command
        # freshness contract.
        deadline = now + self.command_watchdog_timeout_s
        try:
            feedback = self._transport_read_feedback(
                deadline=deadline, operation="hold safety feedback"
            )
            self._validate_running_feedback(feedback)
            self._record_feedback(feedback)
            return feedback
        except BaseException as exc:
            self._latch_fault(str(exc))
            if defer_cleanup:
                raise RH56TransactionalError(
                    f"RH56 safety poll failed: {exc}"
                ) from exc
            stop_error: Optional[BaseException] = None
            try:
                self._disable_and_verify(preserve_fault=True)
            except BaseException as cleanup_exc:
                stop_error = cleanup_exc
            detail = f"RH56 safety poll failed: {exc}"
            if stop_error is not None:
                detail += f"; STOP UNCONFIRMED: {stop_error}"
            raise RH56TransactionalError(detail) from exc

    def enforce_policy_watchdog(
        self,
        *,
        now_monotonic_s: Optional[float] = None,
    ) -> Optional[RH56StopReport]:
        """Fail closed when the exact policy heartbeat deadline has elapsed.

        This method exists for an independent owner thread.  It performs no
        IO before the deadline.  At or after the deadline it latches a fault
        and runs the same measured-pose hold, stable-idle proof and two-pass
        target release as every other terminal RH56 failure.  Safety-gate loss
        cannot prevent this fail-safe stop path.
        """

        self._require_owner()
        now = (
            self._now()
            if now_monotonic_s is None
            else _finite_float(now_monotonic_s, "now_monotonic_s")
        )
        with self._state_lock:
            state = self._state
            heartbeat = self._last_command_monotonic_s
            fault_reason = self._fault_reason
            supervised = self._supervised_session
        if state is RH56ActuatorState.FAULT_LATCHED:
            raise RH56TransactionalError(
                f"RH56 fault is latched: {fault_reason}"
            )
        if state is not RH56ActuatorState.ARMED or heartbeat is None:
            raise RH56TransactionalError(
                "RH56 policy watchdog cannot run while actuator is disarmed"
            )
        inter_command_timeout = (
            self.supervised_inter_command_watchdog_timeout_s
            if supervised
            else self.command_watchdog_timeout_s
        )
        deadline = heartbeat + inter_command_timeout
        if now < deadline:
            return None
        reason = (
            "RH56 independent policy command watchdog expired: "
            f"age={now - heartbeat:.6f}s, "
            f"limit={inter_command_timeout:.6f}s"
        )
        self._latch_fault(reason)
        return self._disable_and_verify(preserve_fault=True)

    def fault_and_disable(self, reason: str) -> RH56StopReport:
        """Owner-only terminal fallback for faults detected outside execute.

        It is intentionally limited to the safe direction: callers cannot
        use it to arm the hand or write a numeric target.
        """

        self._require_owner()
        with self._state_lock:
            state = self._state
        if state is RH56ActuatorState.DISARMED:
            raise RH56TransactionalError(
                "RH56 is disarmed; refusing an unauthorised disable write"
            )
        self._latch_fault(reason)
        return self._disable_and_verify(preserve_fault=True)

    def _validate_stopped_feedback(
        self,
        feedback: RH56SafetyFeedback,
        *,
        phase: str,
        require_idle: bool = True,
        maximum_current_ma: Optional[int] = None,
    ) -> None:
        if any(feedback.errors):
            raise RH56StopUnconfirmed(
                f"{phase} actuator ERROR remains: {feedback.errors}"
            )
        if max(feedback.temperatures_c) >= self.maximum_temperature_c:
            raise RH56StopUnconfirmed(
                f"{phase} temperature is unsafe: {feedback.temperatures_c}"
            )
        fault_statuses = tuple(
            (axis, int(status))
            for axis, status in enumerate(feedback.statuses)
            if status in RH56_FAULT_STATUSES
        )
        if fault_statuses:
            raise RH56StopUnconfirmed(
                f"{phase} actuator fault status remains: {fault_statuses}"
            )
        current_limit = (
            self.stop_max_axis_current_ma
            if maximum_current_ma is None and require_idle
            else (
                self.maximum_running_axis_current_ma
                if maximum_current_ma is None
                else _bounded_integer(
                    maximum_current_ma,
                    f"{phase} maximum_current_ma",
                    minimum=1,
                    maximum=5000,
                )
            )
        )
        if any(abs(current) > current_limit for current in feedback.currents_ma):
            raise RH56StopUnconfirmed(
                f"{phase} current exceeded per-axis bound "
                f"{current_limit}mA: {feedback.currents_ma}"
            )
        if require_idle and any(
            status not in RH56_IDLE_STATUSES for status in feedback.statuses
        ):
            raise RH56StopUnconfirmed(
                f"{phase} status is not idle: {feedback.statuses}"
            )

    def _stop_tail_motion_error(
        self,
        samples: Sequence[RH56SafetyFeedback],
    ) -> Optional[str]:
        if len(samples) != self.stop_verify_samples:
            return (
                "stable idle tail is incomplete: "
                f"actual={len(samples)}, required={self.stop_verify_samples}"
            )
        problems = []
        for axis in range(RH56_AXIS_COUNT):
            angles = [sample.angles[axis] for sample in samples]
            positions = [sample.positions[axis] for sample in samples]
            if max(angles) - min(angles) > self.stop_max_angle_drift_units:
                problems.append(f"axis {axis} ANGLE_ACT still moving: {angles}")
            if max(positions) - min(positions) > self.stop_max_position_drift_units:
                problems.append(f"axis {axis} POS_ACT still moving: {positions}")
        return "; ".join(problems) if problems else None

    def _feedback_equivalent_hold_target(
        self, angles: Sequence[int]
    ) -> Tuple[int, ...]:
        """Convert ANGLE_ACT into the calibrated ANGLE_SET command domain."""

        actual = _six_integers(
            angles,
            "stop hold ANGLE_ACT",
            minimum=0,
            maximum=1000,
        )
        result = []
        for axis, (value, offset, lower, upper) in enumerate(
            zip(
                actual,
                self.feedback_to_command_offset_units,
                self.stop_hold_command_minimum,
                self.stop_hold_command_maximum,
            )
        ):
            command = value + offset
            # ANGLE_ACT and ANGLE_SET use calibrated domains.  At either
            # commissioned endpoint, physical tracking can overshoot by less
            # than the calibrated domain offset.  Clamp only that bounded
            # residual; a larger excursion still refuses the stop proof.
            if command > upper and command - upper <= abs(offset):
                command = upper
            elif command < lower and lower - command <= abs(offset):
                command = lower
            if not lower <= command <= upper:
                raise RH56StopUnconfirmed(
                    f"axis {axis} feedback-equivalent stop hold {command} is "
                    f"outside calibrated command interval {lower}..{upper}"
                )
            result.append(command)
        return tuple(result)

    def _disable_and_verify(self, *, preserve_fault: bool) -> RH56StopReport:
        """Hold the measured pose, prove it stationary, then release targets.

        ``ANGLE_SET=-1`` is a target-release command, not an immediate brake:
        firmware may continue following an internal ``POS_SET`` after release.
        The stop path therefore first snapshots ``ANGLE_ACT``, commits that
        exact six-axis vector as a bounded in-place hold, and rolls a tail of
        feedback samples until the physical state is stable.  Some installed
        firmware keeps STATUS=1 while any numeric ANGLE_SET is present even
        after ANGLE_ACT/POS_ACT have stopped changing.  In that case the
        motionless hold tail authorizes both disable passes, after which three
        fresh consecutive STATUS-idle, motionless samples are required.  Every
        operation shares one absolute ``stop_timeout_s``.

        If the hold phase cannot be established, both disable passes are still
        attempted in the remaining deadline.  Such a fallback is deliberately
        reported as STOP UNCONFIRMED even when ``-1`` reads back exactly.
        """

        deadline = self._now() + self.stop_timeout_s
        failures: list[str] = []
        hold_target: Optional[Tuple[int, ...]] = None
        hold_verified = False
        hold_tail_verified = False
        stable_tail_verified = False
        motionless_hold_verified = False
        stable_samples: deque[RH56SafetyFeedback] = deque(
            maxlen=self.stop_verify_samples
        )
        hold_motion_samples: deque[RH56SafetyFeedback] = deque(
            maxlen=self.stop_verify_samples
        )
        observed_samples = 0
        last_settle_detail = "no post-hold feedback was captured"

        # Reserve a bounded tail of the same overall deadline for the two
        # fail-safe disable write/readback passes plus a wholly fresh
        # post-disable tail.  Each post-disable sample contains one ANGLE_SET
        # read and the two compact feedback-register reads.  In the production
        # 5 s / three-sample / 20 ms configuration this is:
        #
        #   2 * (20 ms write ACK + 50 ms readback)
        # + 3 * (3 * 50 ms register reads)
        # + 2 * 20 ms verification sleeps
        # + 100 ms scheduling margin
        # = 730 ms.
        #
        # Short diagnostic stop budgets retain at least half their time for
        # establishing the measured hold.  The deployed five-second budget
        # receives the complete calculated release reserve.
        disable_passes_io_s = 2.0 * (
            RH56_SUPERVISED_WRITE_RESPONSE_GRACE_S
            + RH56_SUPERVISED_EXCHANGE_TIMEOUT_S
        )
        post_disable_reads_s = (
            float(self.stop_verify_samples)
            * 3.0
            * RH56_SUPERVISED_EXCHANGE_TIMEOUT_S
        )
        post_disable_sleeps_s = (
            float(self.stop_verify_samples - 1)
            * self.stop_verify_interval_s
        )
        calculated_release_reserve_s = (
            disable_passes_io_s
            + post_disable_reads_s
            + post_disable_sleeps_s
            + RH56_STOP_RELEASE_SCHEDULING_MARGIN_S
        )
        release_reserve_s = min(
            self.stop_timeout_s * 0.5,
            calculated_release_reserve_s,
        )
        settle_deadline = deadline - release_reserve_s

        try:
            seed = self._transport_read_feedback(
                deadline=settle_deadline,
                operation="stop hold seed feedback",
            )
            self._validate_stopped_feedback(
                seed,
                phase="stop hold seed",
                require_idle=False,
                maximum_current_ma=self.stop_settle_max_axis_current_ma,
            )
            hold_target = self._feedback_equivalent_hold_target(seed.angles)
            self._transport_write_once_then_verify(
                hold_target,
                deadline=settle_deadline,
                operation="stop hold",
            )
            hold_verified = True

            while self._now() < settle_deadline:
                readback = self._transport_read_targets(
                    deadline=settle_deadline,
                    operation=f"stop settle sample {observed_samples + 1} target readback",
                )
                if readback != hold_target:
                    raise RH56StopUnconfirmed(
                        "stop hold ANGLE_SET changed during settling: "
                        f"expected={hold_target}, actual={readback}"
                    )
                feedback = self._transport_read_feedback(
                    deadline=settle_deadline,
                    operation=f"stop settle sample {observed_samples + 1} feedback",
                )
                observed_samples += 1
                self._validate_stopped_feedback(
                    feedback,
                    phase=f"stop settle sample {observed_samples}",
                    require_idle=False,
                    maximum_current_ma=self.stop_settle_max_axis_current_ma,
                )
                hold_motion_samples.append(feedback)
                hold_motion_error = self._stop_tail_motion_error(
                    tuple(hold_motion_samples)
                )
                if any(
                    status not in RH56_IDLE_STATUSES
                    for status in feedback.statuses
                ):
                    stable_samples.clear()
                    last_settle_detail = (
                        f"latest status is not idle: {feedback.statuses}"
                    )
                else:
                    stable_samples.append(feedback)
                    motion_error = self._stop_tail_motion_error(
                        tuple(stable_samples)
                    )
                    if motion_error is None:
                        last_settle_detail = "stable idle tail verified"
                        hold_tail_verified = True
                        break
                    last_settle_detail = motion_error

                # STATUS=1 can be sticky while a numeric hold is installed on
                # the RH56 thumb-rotation axis.  A complete
                # running-current-bounded tail with bounded ANGLE_ACT and
                # POS_ACT drift is sufficient to release the numeric target,
                # but not yet sufficient for the final stop proof; that proof
                # is collected after both -1 passes below.
                if hold_motion_error is None:
                    motionless_hold_verified = True
                    last_settle_detail = (
                        "motionless hold tail verified; awaiting disabled idle tail"
                    )
                    break

                remaining = settle_deadline - self._now()
                if remaining <= 0.0:
                    break
                self._sleep(min(self.stop_verify_interval_s, remaining))
        except BaseException as exc:
            failures.append(f"in-place hold/settle: {exc}")

        if hold_verified and not (
            hold_tail_verified or motionless_hold_verified
        ):
            failures.append(
                "in-place hold did not reach a stable idle tail before the "
                f"reserved release deadline: observed={observed_samples}, "
                f"consecutive_idle={len(stable_samples)}, detail={last_settle_detail}"
            )

        verified_passes = 0
        for pass_index in (1, 2):
            try:
                self._transport_write_once_then_verify(
                    RH56_DISABLED_TARGETS,
                    deadline=deadline,
                    operation=f"disable pass {pass_index}",
                )
                verified_passes += 1
            except BaseException as exc:
                failures.append(f"disable pass {pass_index}: {exc}")

        # The final proof never reuses a pre-release sample.  After both -1
        # readbacks, collect a wholly fresh tail whose STATUS is idle, whose
        # current is within the strict idle cap, and whose ANGLE_ACT/POS_ACT
        # drift is bounded.  Transitional hold/braking samples above are
        # allowed up to the separately configured running-current cap.
        if verified_passes == 2:
            stable_samples.clear()
            stable_tail_verified = False
            post_disable_observed = 0
            post_disable_detail = "no post-disable feedback was captured"
            try:
                while self._now() < deadline:
                    readback = self._transport_read_targets(
                        deadline=deadline,
                        operation=(
                            "post-disable stable sample "
                            f"{post_disable_observed + 1} target readback"
                        ),
                    )
                    if readback != RH56_DISABLED_TARGETS:
                        raise RH56StopUnconfirmed(
                            f"post-disable ANGLE_SET changed: {readback}"
                        )
                    feedback = self._transport_read_feedback(
                        deadline=deadline,
                        operation=(
                            "post-disable stable sample "
                            f"{post_disable_observed + 1} feedback"
                        ),
                    )
                    post_disable_observed += 1
                    self._validate_stopped_feedback(
                        feedback,
                        phase=(
                            "post-disable stable sample "
                            f"{post_disable_observed}"
                        ),
                        require_idle=False,
                        maximum_current_ma=(
                            self.stop_settle_max_axis_current_ma
                        ),
                    )
                    if any(
                        status not in RH56_IDLE_STATUSES
                        for status in feedback.statuses
                    ):
                        stable_samples.clear()
                        post_disable_detail = (
                            f"latest status is not idle: {feedback.statuses}"
                        )
                    elif any(
                        abs(current) > self.stop_max_axis_current_ma
                        for current in feedback.currents_ma
                    ):
                        stable_samples.clear()
                        post_disable_detail = (
                            "latest idle current exceeds per-axis bound "
                            f"{self.stop_max_axis_current_ma}mA: "
                            f"{feedback.currents_ma}"
                        )
                    else:
                        self._validate_stopped_feedback(
                            feedback,
                            phase=(
                                "post-disable stable sample "
                                f"{post_disable_observed}"
                            ),
                            require_idle=True,
                        )
                        stable_samples.append(feedback)
                        motion_error = self._stop_tail_motion_error(
                            tuple(stable_samples)
                        )
                        if motion_error is None:
                            stable_tail_verified = True
                            post_disable_detail = "disabled stable idle tail verified"
                            break
                        post_disable_detail = motion_error
                    remaining = deadline - self._now()
                    if remaining <= 0.0:
                        break
                    self._sleep(min(self.stop_verify_interval_s, remaining))
            except BaseException as exc:
                failures.append(f"post-disable stable-tail verification: {exc}")
            if not stable_tail_verified:
                failures.append(
                    "disabled target did not reach a stable idle tail before "
                    f"the stop deadline: observed={post_disable_observed}, "
                    f"consecutive_idle={len(stable_samples)}, "
                    f"detail={post_disable_detail}"
                )

        if (
            failures
            or not hold_verified
            or not stable_tail_verified
            or verified_passes != 2
            or len(stable_samples) != self.stop_verify_samples
        ):
            reason = "; ".join(failures) or "incomplete RH56 stop proof"
            self._latch_fault(reason)
            raise RH56StopUnconfirmed(reason)

        completed = self._now()
        with self._state_lock:
            self._stop_confirmed = True
            self._pending_sequence = None
            self._last_feedback_monotonic_s = (
                stable_samples[-1].captured_monotonic_s
            )
            if not preserve_fault:
                self._state = RH56ActuatorState.DISARMED
                self._run_id = None
                self._authorization_id = None
                self._preflight_sha256 = None
                self._safety_gate = None
                self._armed_monotonic_s = None
                self._last_command_monotonic_s = None
                self._supervised_session = False
                self._supervised_awaiting_first_numeric_command = False
                self._last_supervised_target = None
        return RH56StopReport(
            verified=True,
            disable_passes_verified=verified_passes,
            feedback_samples_verified=len(stable_samples),
            completed_monotonic_s=completed,
        )

    def disable_and_verify(self) -> RH56StopReport:
        """Normal authorized shutdown; does not disarm the shared safety gate."""

        self._require_owner()
        with self._state_lock:
            state = self._state
        if state is RH56ActuatorState.DISARMED:
            raise RH56TransactionalError(
                "RH56 is disarmed; refusing an unauthorised disable write"
            )
        return self._disable_and_verify(
            preserve_fault=(state is RH56ActuatorState.FAULT_LATCHED)
        )


__all__ = [
    "RH56ActuationPreflight",
    "RH56ActuatorState",
    "RH56_COMPACT_TRANSACTION_EXCHANGES",
    "RH56_COMPACT_TRANSACTION_WIRE_BYTES",
    "RH56_C2_READINESS_LEVEL",
    "RH56DeadlineExceeded",
    "RH56_DISABLED_TARGETS",
    "RH56SafetyFeedback",
    "RH56StopReport",
    "RH56StopUnconfirmed",
    "RH56_SUPERVISED_EXCHANGE_TIMEOUT_S",
    "RH56_STOP_RELEASE_SCHEDULING_MARGIN_S",
    "RH56_SUPERVISED_WRITE_RESPONSE_GRACE_S",
    "RH56TargetReceipt",
    "RH56TransactionReceipt",
    "RH56TransactionalActuator",
    "RH56TransactionalError",
    "RH56TransactionalTransport",
    "RH56TransportError",
    "RH56WireRateEstimate",
    "SupervisedRH56Preflight",
    "estimate_compact_transaction_wire_rate",
]
