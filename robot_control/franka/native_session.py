"""Process-isolated Franka session proxy for the supervised V94 path.

This module deliberately contains no Franka or camera backend import.  It is
the parent-side lifecycle and event bridge for a future native servo child.
The wire codec and child launcher are injected so construction remains inert
and protocol conformance can be tested without opening a device.

The transport contract is a critical full-duplex ``AF_UNIX/SOCK_SEQPACKET``
socket plus a child-to-parent ``AF_UNIX/SOCK_DGRAM`` state channel.  Packets
are opaque bytes here; the production codec mirrors the fixed POD protocol
owned by the native servo implementation.  In particular, this module must
not silently fall back to JSON or pickle on either boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
import errno
import hashlib
import multiprocessing
import os
import select
import socket
import struct
import subprocess
import threading
import time
from typing import Any, Callable, Optional, Protocol, Sequence, Union, runtime_checkable
import uuid
import zlib

import numpy as np

from sim2real.closed_loop_core import ExecutedActionLedger, MotionAuthorization
from .session import (
    ExperimentalSupervisedFrankaEnvelope,
    FrankaJointTarget,
    FrankaPoseRing,
    FrankaPoseSample,
    FrankaTargetSampleHold,
    FrankaTargetSource,
    FrankaSessionMode,
    SupervisedFrankaPreflightToken,
)


class FrankaNativeSupervisedSessionError(RuntimeError):
    """The isolated Franka child or its parent-side protocol failed."""


_PARENT_TARGET_MAX_AGE_S = 0.050
_PARENT_TARGET_MAX_FUTURE_SKEW_S = 0.010
# libfranka's returned period includes the recovered current packet.  Thus a
# 21 ms return represents 20 missing 1 kHz packets, exactly the FCI fail-stop
# horizon rather than 21 packets of uncontrolled continuation.
_NATIVE_MAX_ISOLATED_CONTROL_PERIOD_MS = 21


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if result != value or result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _positive_integer(value: object, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


@dataclass(frozen=True)
class NativeHelloEvent:
    """The first child-to-parent packet of one process session."""

    child_pid: int
    session_nonce: bytes

    def __post_init__(self) -> None:
        object.__setattr__(self, "child_pid", _positive_integer(self.child_pid, "child_pid"))
        nonce = bytes(self.session_nonce)
        if len(nonce) != 16:
            raise ValueError("session_nonce must contain exactly 16 bytes")
        object.__setattr__(self, "session_nonce", nonce)


@dataclass(frozen=True)
class NativeIpcReadyEvent:
    """Same-Robot static/q-home preflight passed; actions remain forbidden."""

    measured_q_rad: Sequence[float]
    measured_dq_rad_s: Sequence[float]
    q_home_linf_error_rad: float
    robot_time_ms: int
    static_provenance_verified: bool
    realtime_scheduler_policy: int = 0
    realtime_scheduler_priority: int = 0
    realtime_cpu: int = 0
    realtime_affinity_cpu_count: int = 0


@dataclass(frozen=True)
class NativeActionReadyEvent:
    """The child passed the 100-consecutive-healthy-cycle action gate."""

    consecutive_healthy_cycles: int
    maximum_control_period_ms: int
    healthy_hold_robot_time_ms: int
    maximum_read_to_write_ns: int
    minimum_control_command_success_rate: float
    latest_control_command_success_rate: float
    measured_q_rad: Sequence[float]
    measured_dq_rad_s: Sequence[float]
    robot_time_ms: int
    active_read_count: int
    active_write_count: int
    status_flags: int
    cumulative_missed_robot_states: int


# Semantic compatibility name for reviewers referring to the action gate as
# HEALTHY.  The frozen wire kind is ACTION_READY.
NativeHealthyEvent = NativeActionReadyEvent


@dataclass(frozen=True)
class NativeStateEvent:
    """One state sampled by the child from its active Franka handle."""

    cycle: int
    realtime_s: float
    monotonic_s: float
    q_rad: Sequence[float]
    dq_rad_s: Sequence[float]
    T_base_eef: Sequence[Sequence[float]]
    robot_time_ms: int = 0
    active_target_sequence: int = 0
    active_observation_sequence: int = 0
    control_period_ms: int = 0
    robot_mode: int = 0
    control_command_success_rate: float = 0.0
    commanded_q_rad: Sequence[float] = ()
    # Protocol-v4 software-controller state captured after the final native
    # position limiter.  Empty defaults preserve legacy/fake event producers.
    shaper_q_d_rad: Optional[Sequence[float]] = None
    shaper_dq_d_rad_s: Optional[Sequence[float]] = None
    shaper_ddq_d_rad_s2: Optional[Sequence[float]] = None
    held_q_cmd_rad: Optional[Sequence[float]] = None
    controller_state29: Sequence[float] = ()
    status_flags: int = 0
    cumulative_missed_robot_states: int = 0
    last_read_to_write_ns: int = 0
    maximum_read_to_write_ns: int = 0
    telemetry_drop_count: int = 0
    health_flags: int = 0
    consecutive_healthy_cycles: int = 0
    active_read_count: int = 0
    active_write_count: int = 0
    target_ack_count: int = 0


@dataclass(frozen=True)
class NativeAckEvent:
    """The child wrote the exact target sequence to the active handle."""

    sequence: int
    observation_sequence: int
    target_produced_monotonic_ns: int
    target_age_at_write_ns: int
    control_cycle: int
    robot_time_ms: int
    applied_monotonic_ns: int
    control_period_ms: int
    target_q_rad: Sequence[float]
    commanded_q_rad: Sequence[float]
    measured_q_rad: Sequence[float]
    read_to_write_ns: int
    maximum_tracking_error_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence", _positive_integer(self.sequence, "ACK sequence"))


@dataclass(frozen=True)
class NativeStopVerificationSample:
    robot_time_ms: int
    robot_mode: int
    status_flags: int
    measured_dq_rad_s: Sequence[float]


@dataclass(frozen=True)
class NativeFaultEvent:
    """A fail-closed terminal child fault."""

    code: int
    reason: str
    sequence: int = 0
    control_cycle: int = 0
    robot_time_ms: int = 0
    system_errno: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _nonnegative_integer(self.code, "fault code"))
        object.__setattr__(
            self,
            "sequence",
            _nonnegative_integer(self.sequence, "fault sequence"),
        )
        reason = str(self.reason).strip()
        if not reason:
            raise ValueError("fault reason must be non-empty")
        object.__setattr__(self, "reason", reason)


@dataclass(frozen=True)
class NativeStopProofEvent:
    """Terminal child proof; it is never inferred from process exit alone."""

    stop_reason: int
    terminal_fault_code: int
    finish_attempted: bool
    finish_succeeded: bool
    robot_stop_attempted: bool
    robot_stop_succeeded: bool
    idle_dq_verified: bool
    active_handle_released: bool
    fault_reply_delivered: bool
    robot_backend_released: bool
    control_cycles: int
    last_target_sequence: int
    active_read_count: int
    active_write_count: int
    franka_ack_count: int
    stop_verification_samples: int
    stop_consecutive_idle_samples: int
    final_robot_mode: int
    maximum_control_period_ms: int
    maximum_stop_dq_rad_s: float
    maximum_read_to_write_ns: int
    pre_stop_robot_time_ms: int
    final_robot_time_ms: int
    telemetry_drop_count: int
    verified_samples: Sequence[NativeStopVerificationSample]
    detail: str
    parent_stop_verified: bool
    maximum_control_period_s: float
    fault_reason: Optional[str] = None

    def __post_init__(self) -> None:
        for name in (
            "finish_attempted",
            "finish_succeeded",
            "robot_stop_attempted",
            "robot_stop_succeeded",
            "idle_dq_verified",
            "active_handle_released",
            "fault_reply_delivered",
            "parent_stop_verified",
        ):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
            object.__setattr__(self, name, bool(getattr(self, name)))
        for name in (
            "active_read_count",
            "active_write_count",
            "franka_ack_count",
            "stop_reason",
            "terminal_fault_code",
            "control_cycles",
            "last_target_sequence",
            "stop_verification_samples",
            "stop_consecutive_idle_samples",
            "final_robot_mode",
            "maximum_control_period_ms",
            "maximum_read_to_write_ns",
            "final_robot_time_ms",
            "telemetry_drop_count",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_integer(getattr(self, name), name),
            )
        maximum_period = float(self.maximum_control_period_s)
        if not np.isfinite(maximum_period) or maximum_period < 0.0:
            raise ValueError("maximum_control_period_s must be finite and nonnegative")
        object.__setattr__(self, "maximum_control_period_s", maximum_period)
        detail = str(self.detail).strip()
        object.__setattr__(self, "detail", detail)
        if self.fault_reason is not None:
            reason = str(self.fault_reason).strip()
            object.__setattr__(self, "fault_reason", reason or None)


NativeChildEvent = Union[
    NativeHelloEvent,
    NativeIpcReadyEvent,
    NativeActionReadyEvent,
    NativeStateEvent,
    NativeAckEvent,
    NativeFaultEvent,
    NativeStopProofEvent,
]


class V94NativeMessageKind(IntEnum):
    ARM = 0x001
    HEARTBEAT = 0x002
    TARGET = 0x003
    STOP = 0x004
    HELLO = 0x101
    IPC_READY = 0x102
    STATE = 0x103
    ACK = 0x104
    FAULT = 0x105
    STOP_PROOF = 0x106
    ACTION_READY = 0x107


class V94NativeControllerMode(IntEnum):
    """ARM-selected native target validation contract."""

    LEGACY = 0
    QD_G015 = 1


V94_NATIVE_PROTOCOL_MAGIC = 0x46343956
V94_NATIVE_PROTOCOL_VERSION = 4
V94_NATIVE_MAX_PACKET_BYTES = 1024
V94_NATIVE_HEADER_BYTES = 56
V94_NATIVE_PAYLOAD_BYTES = {
    V94NativeMessageKind.ARM: 616,
    V94NativeMessageKind.HEARTBEAT: 8,
    V94NativeMessageKind.TARGET: 80,
    V94NativeMessageKind.STOP: 16,
    V94NativeMessageKind.HELLO: 164,
    V94NativeMessageKind.IPC_READY: 152,
    V94NativeMessageKind.STATE: 764,
    V94NativeMessageKind.ACK: 248,
    V94NativeMessageKind.FAULT: 200,
    V94NativeMessageKind.STOP_PROOF: 488,
    V94NativeMessageKind.ACTION_READY: 184,
}


def _controller_mode(value: object) -> V94NativeControllerMode:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("controller_mode must be legacy=0 or qd_g015=1")
    try:
        return V94NativeControllerMode(int(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("controller_mode must be legacy=0 or qd_g015=1") from exc


def _fixed_digest(value: object, size: int, name: str) -> bytes:
    if isinstance(value, str):
        text = value.strip().lower()
        if len(text) != size * 2:
            raise ValueError(f"{name} must contain {size * 2} hex characters")
        try:
            result = bytes.fromhex(text)
        except ValueError as exc:
            raise ValueError(f"{name} must be hexadecimal") from exc
    else:
        try:
            result = bytes(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain {size} bytes") from exc
    if len(result) != size:
        raise ValueError(f"{name} must contain exactly {size} bytes")
    return result


def _finite_vector(value: object, length: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite length-{length} vector") from exc
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite length-{length} vector")
    result = result.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class NativeHelloExpectation:
    """Exact native producer identity expected before ARM is permitted."""

    state_decimation: int
    safety_limits_schema: int
    libfranka_sha256: bytes
    libfranka_source_commit: bytes
    producer_build_sha256: bytes

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "state_decimation",
            _positive_integer(self.state_decimation, "state_decimation"),
        )
        object.__setattr__(
            self,
            "safety_limits_schema",
            _positive_integer(self.safety_limits_schema, "safety_limits_schema"),
        )
        object.__setattr__(
            self,
            "libfranka_sha256",
            _fixed_digest(self.libfranka_sha256, 32, "libfranka_sha256"),
        )
        object.__setattr__(
            self,
            "libfranka_source_commit",
            _fixed_digest(
                self.libfranka_source_commit,
                20,
                "libfranka_source_commit",
            ),
        )
        object.__setattr__(
            self,
            "producer_build_sha256",
            _fixed_digest(
                self.producer_build_sha256,
                32,
                "producer_build_sha256",
            ),
        )


@dataclass(frozen=True)
class NativeFrankaTelemetry:
    """Small compatibility surface consumed by ``BoundedV94C2Runtime``."""

    active_read_count: int
    active_write_count: int
    franka_ack_count: int
    maximum_control_period_s: float
    stop_requested: bool
    stop_verified: bool
    fault_reason: Optional[str]
    parent_heartbeat_count: int = 0
    maximum_parent_heartbeat_interval_s: float = 0.0
    realtime_scheduler_policy: int = 0
    realtime_scheduler_priority: int = 0
    realtime_cpu: int = 0
    realtime_affinity_cpu_count: int = 0
    maximum_read_to_write_ns: int = 0

    @classmethod
    def from_stop_proof(
        cls,
        proof: NativeStopProofEvent,
        ipc_ready: Optional[NativeIpcReadyEvent] = None,
    ) -> "NativeFrankaTelemetry":
        return cls(
            active_read_count=proof.active_read_count,
            active_write_count=proof.active_write_count,
            franka_ack_count=proof.franka_ack_count,
            maximum_control_period_s=proof.maximum_control_period_s,
            stop_requested=True,
            stop_verified=proof.parent_stop_verified,
            fault_reason=proof.fault_reason,
            realtime_scheduler_policy=(
                0 if ipc_ready is None else ipc_ready.realtime_scheduler_policy
            ),
            realtime_scheduler_priority=(
                0 if ipc_ready is None else ipc_ready.realtime_scheduler_priority
            ),
            realtime_cpu=0 if ipc_ready is None else ipc_ready.realtime_cpu,
            realtime_affinity_cpu_count=(
                0 if ipc_ready is None else ipc_ready.realtime_affinity_cpu_count
            ),
            maximum_read_to_write_ns=proof.maximum_read_to_write_ns,
        )

    @classmethod
    def failed_without_proof(cls, reason: str) -> "NativeFrankaTelemetry":
        return cls(
            active_read_count=0,
            active_write_count=0,
            franka_ack_count=0,
            maximum_control_period_s=0.0,
            stop_requested=True,
            stop_verified=False,
            fault_reason=str(reason).strip() or "native child stopped without proof",
            maximum_read_to_write_ns=0,
        )


@runtime_checkable
class NativeSupervisedPacketCodec(Protocol):
    """Opaque fixed-POD codec implemented from the native protocol header."""

    max_packet_bytes: int

    def encode_arm(
        self,
        *,
        run_id: str,
        authorization: Any,
        preflight_token: Any,
        envelope: Any,
        maximum_cycles: Optional[int],
        controller_mode: V94NativeControllerMode = V94NativeControllerMode.LEGACY,
    ) -> bytes:
        ...

    def encode_heartbeat(self, *, monotonic_ns: int) -> bytes:
        ...

    def encode_target(self, target: FrankaJointTarget) -> bytes:
        ...

    def encode_stop(self, *, reason: str) -> bytes:
        ...

    def decode_critical_packet(self, packet: bytes) -> NativeChildEvent:
        ...

    def decode_state_packet(self, packet: bytes) -> NativeStateEvent:
        ...


_HEADER = struct.Struct("<IHHIIQQ16sII")
_HELLO = struct.Struct("<IIII32s20s32s8d")
_ARM = struct.Struct("<32s32s32s32s16s6Q51dII8s")
_HEARTBEAT = struct.Struct("<Q")
_TARGET = struct.Struct("<QQQ7d")
_STOP = struct.Struct("<IIQ")
_IPC_READY = struct.Struct("<15dQ6I")
_ACTION_READY = struct.Struct("<IIQQ2d14d3QII")
_STATE = struct.Struct("<6QIId37dII3QII3Q29f28d")
_ACK = struct.Struct("<7QII21dQd")
_FAULT = struct.Struct("<IIQQQiI160s")
_STOP_SAMPLE = struct.Struct("<QII7d")
_STOP_PROOF_PREFIX = struct.Struct("<II8B5Q4Id4Q")

_STRUCT_SIZE_SENTINELS = {
    "header": (_HEADER.size, V94_NATIVE_HEADER_BYTES),
    "HELLO": (_HELLO.size, V94_NATIVE_PAYLOAD_BYTES[V94NativeMessageKind.HELLO]),
    "ARM": (_ARM.size, V94_NATIVE_PAYLOAD_BYTES[V94NativeMessageKind.ARM]),
    "HEARTBEAT": (
        _HEARTBEAT.size,
        V94_NATIVE_PAYLOAD_BYTES[V94NativeMessageKind.HEARTBEAT],
    ),
    "TARGET": (_TARGET.size, V94_NATIVE_PAYLOAD_BYTES[V94NativeMessageKind.TARGET]),
    "STOP": (_STOP.size, V94_NATIVE_PAYLOAD_BYTES[V94NativeMessageKind.STOP]),
    "IPC_READY": (
        _IPC_READY.size,
        V94_NATIVE_PAYLOAD_BYTES[V94NativeMessageKind.IPC_READY],
    ),
    "ACTION_READY": (
        _ACTION_READY.size,
        V94_NATIVE_PAYLOAD_BYTES[V94NativeMessageKind.ACTION_READY],
    ),
    "STATE": (_STATE.size, V94_NATIVE_PAYLOAD_BYTES[V94NativeMessageKind.STATE]),
    "ACK": (_ACK.size, V94_NATIVE_PAYLOAD_BYTES[V94NativeMessageKind.ACK]),
    "FAULT": (_FAULT.size, V94_NATIVE_PAYLOAD_BYTES[V94NativeMessageKind.FAULT]),
    "STOP_SAMPLE": (_STOP_SAMPLE.size, 72),
    "STOP_PROOF": (
        _STOP_PROOF_PREFIX.size + 3 * _STOP_SAMPLE.size + 160,
        V94_NATIVE_PAYLOAD_BYTES[V94NativeMessageKind.STOP_PROOF],
    ),
}
for _sentinel_name, (_python_size, _native_size) in _STRUCT_SIZE_SENTINELS.items():
    if _python_size != _native_size:
        raise RuntimeError(
            f"Python/native POD size mismatch for {_sentinel_name}: "
            f"{_python_size} != {_native_size}"
        )


def _wire_boolean(value: int, name: str) -> bool:
    if value not in (0, 1):
        raise FrankaNativeSupervisedSessionError(
            f"{name} is not a fixed-width wire boolean"
        )
    return bool(value)


def _decode_fixed_text(raw: bytes, *, length: Optional[int] = None) -> str:
    if length is None:
        terminator = raw.find(b"\x00")
        if terminator < 0:
            prefix = raw
        else:
            prefix = raw[:terminator]
            if any(raw[terminator:]):
                raise FrankaNativeSupervisedSessionError(
                    "fixed diagnostic text has nonzero trailing bytes"
                )
    else:
        if length < 0 or length > len(raw):
            raise FrankaNativeSupervisedSessionError(
                "diagnostic detail_bytes exceeds the fixed field"
            )
        prefix = raw[:length]
        if any(raw[length:]):
            raise FrankaNativeSupervisedSessionError(
                "fixed diagnostic text has nonzero trailing bytes"
            )
    try:
        return prefix.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise FrankaNativeSupervisedSessionError(
            "diagnostic text is not valid UTF-8"
        ) from exc


class V94NativePODCodec:
    """Strict Python mirror of frozen native protocol version 4."""

    max_packet_bytes = V94_NATIVE_MAX_PACKET_BYTES
    # Parent heartbeat traffic is owned by a CPython thread and can be delayed
    # by a GIL-holding observation/inference section.  Use the compiled 100 ms
    # supervisor ceiling for that process-liveness layer.
    heartbeat_timeout_ns = 100_000_000
    # ACTION_READY precedes the late RH56 owner startup.  Give that bounded
    # startup enough room without weakening the steady-target watchdog
    # once policy execution begins.
    first_target_timeout_ns = 5_000_000_000
    # This ARM field is the maximum receive-to-receive gap, not packet age.
    # During a recoverable observation or USB scheduling gap the servo
    # converges to and then holds the last accepted bounded target.  Allow that
    # no-new-action state for at most 500 ms.  Process liveness is still
    # guarded independently at 100 ms and every newly produced TARGET remains
    # capped at 50 ms by the parent and native timestamp checks.
    target_timeout_ns = 500_000_000
    # 720 policy targets at 60 Hz span 12 seconds.  Reserve three additional
    # seconds for the fixed healthy-control bootstrap, first-target handoff,
    # and final acknowledged hold/stop request without changing motion limits.
    maximum_session_duration_ns = 15_000_000_000

    _critical_child_kinds = frozenset(
        {
            V94NativeMessageKind.HELLO,
            V94NativeMessageKind.IPC_READY,
            V94NativeMessageKind.ACK,
            V94NativeMessageKind.FAULT,
            V94NativeMessageKind.STOP_PROOF,
            V94NativeMessageKind.ACTION_READY,
        }
    )

    def __init__(
        self,
        *,
        hello_expectation: NativeHelloExpectation,
        reference_q_rad: Sequence[float],
        maximum_target_count: int,
        expected_servo_cpu: Optional[int] = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not isinstance(hello_expectation, NativeHelloExpectation):
            raise TypeError("hello_expectation must be NativeHelloExpectation")
        if not callable(monotonic_ns):
            raise TypeError("monotonic_ns must be callable")
        self.hello_expectation = hello_expectation
        self.reference_q_rad = _finite_vector(
            reference_q_rad, 7, "reference_q_rad"
        )
        self.maximum_target_count = _positive_integer(
            maximum_target_count, "maximum_target_count"
        )
        if self.maximum_target_count > 720:
            raise ValueError("maximum_target_count exceeds supervised limit 720")
        if expected_servo_cpu is None:
            self.expected_servo_cpu = None
        else:
            self.expected_servo_cpu = _nonnegative_integer(
                expected_servo_cpu, "expected_servo_cpu"
            )
        self._monotonic_ns = monotonic_ns
        self._session_nonce: Optional[bytes] = None
        self._parent_packet_sequence = 0
        self._child_critical_sequence = 0
        self._child_state_sequence = 0
        self._last_child_critical_monotonic_ns = 0
        self._last_child_state_monotonic_ns = 0
        self._heartbeat_sequence = 0
        self._hello_validated = False
        self._arm_encoded = False
        self._ipc_ready_decoded = False
        self._action_ready_decoded = False
        self._stop_encoded = False
        self._last_target_sequence = 0
        self._pending_target_sequence: Optional[int] = None

    @property
    def session_nonce(self) -> Optional[bytes]:
        return self._session_nonce

    @staticmethod
    def protocol_crc32(packet_with_zero_crc: bytes) -> int:
        return zlib.crc32(packet_with_zero_crc) & 0xFFFFFFFF

    def _encode(self, kind: V94NativeMessageKind, payload: bytes) -> bytes:
        nonce = self._session_nonce
        if nonce is None:
            raise FrankaNativeSupervisedSessionError(
                f"cannot encode {kind.name} before validated HELLO"
            )
        expected_size = V94_NATIVE_PAYLOAD_BYTES[kind]
        if len(payload) != expected_size:
            raise FrankaNativeSupervisedSessionError(
                f"{kind.name} payload size changed: {len(payload)} != {expected_size}"
            )
        self._parent_packet_sequence += 1
        monotonic_ns = _positive_integer(
            self._monotonic_ns(), "packet monotonic_ns"
        )
        header = _HEADER.pack(
            V94_NATIVE_PROTOCOL_MAGIC,
            V94_NATIVE_PROTOCOL_VERSION,
            int(kind),
            len(payload),
            0,
            self._parent_packet_sequence,
            monotonic_ns,
            nonce,
            0,
            0,
        )
        packet = bytearray(header + payload)
        crc = self.protocol_crc32(bytes(packet))
        struct.pack_into("<I", packet, 48, crc)
        if len(packet) > self.max_packet_bytes:
            raise FrankaNativeSupervisedSessionError(
                f"{kind.name} packet exceeds frozen maximum"
            )
        return bytes(packet)

    def _validate_envelope(self, envelope: Any) -> ExperimentalSupervisedFrankaEnvelope:
        if not isinstance(envelope, ExperimentalSupervisedFrankaEnvelope):
            raise FrankaNativeSupervisedSessionError(
                "ARM requires ExperimentalSupervisedFrankaEnvelope"
            )
        checks = (
            (envelope.maximum_velocity_rad_s, 0.50, "velocity"),
            (envelope.maximum_target_rate_rad_s, 0.50, "target rate"),
            (envelope.maximum_acceleration_rad_s2, 5.0, "acceleration"),
            (envelope.maximum_jerk_rad_s3, 250.0, "jerk"),
            (envelope.maximum_tracking_error_rad, 0.01, "tracking"),
        )
        for values, expected, name in checks:
            vector = _finite_vector(values, 7, name)
            if not np.allclose(vector, expected, rtol=0.0, atol=1.0e-12):
                raise FrankaNativeSupervisedSessionError(
                    f"supervised {name} ceiling differs from native protocol"
                )
        scalar_checks = (
            (envelope.control_period_max_s, 0.002, "control period"),
            (envelope.read_to_write_deadline_s, 0.0008, "read-to-write"),
            (envelope.policy_command_max_age_s, 0.05, "target age"),
            (envelope.maximum_session_duration_s, 15.0, "session duration"),
            (envelope.expected_external_load_mass_kg, 0.0, "external load"),
        )
        for actual, expected, name in scalar_checks:
            if not np.isclose(float(actual), expected, rtol=0.0, atol=1.0e-12):
                raise FrankaNativeSupervisedSessionError(
                    f"supervised {name} differs from native protocol"
                )
        return envelope

    def encode_arm(
        self,
        *,
        run_id: str,
        authorization: Any,
        preflight_token: Any,
        envelope: Any,
        maximum_cycles: Optional[int],
        controller_mode: V94NativeControllerMode = V94NativeControllerMode.LEGACY,
    ) -> bytes:
        del maximum_cycles
        if not self._hello_validated or self._arm_encoded or self._stop_encoded:
            raise FrankaNativeSupervisedSessionError(
                "ARM is allowed exactly once after validated HELLO"
            )
        if not isinstance(authorization, MotionAuthorization):
            raise FrankaNativeSupervisedSessionError(
                "ARM requires MotionAuthorization"
            )
        if not isinstance(preflight_token, SupervisedFrankaPreflightToken):
            raise FrankaNativeSupervisedSessionError(
                "ARM requires SupervisedFrankaPreflightToken"
            )
        envelope = self._validate_envelope(envelope)
        active_run_id = str(run_id).strip()
        if authorization.run_id != active_run_id or preflight_token.run_id != active_run_id:
            raise FrankaNativeSupervisedSessionError("ARM run ID binding changed")
        now_s = self._monotonic_ns() / 1_000_000_000
        preflight_token.require_active(
            run_id=active_run_id,
            stage=FrankaSessionMode.SUPERVISED_V94,
            envelope=envelope,
            now_monotonic_s=now_s,
        )
        if not authorization.issued_monotonic_s <= now_s < authorization.expires_monotonic_s:
            raise FrankaNativeSupervisedSessionError(
                "motion authorization is not active at ARM"
            )
        try:
            authorization_id = uuid.UUID(authorization.authorization_id).bytes
        except (ValueError, AttributeError) as exc:
            raise FrankaNativeSupervisedSessionError(
                "authorization_id must be a canonical UUID"
            ) from exc
        values = (
            *self.reference_q_rad.tolist(),
            *np.asarray(envelope.safe_joint_lower_rad, dtype=np.float64).tolist(),
            *np.asarray(envelope.safe_joint_upper_rad, dtype=np.float64).tolist(),
            *np.asarray(envelope.expected_F_T_EE, dtype=np.float64)
            .reshape(-1, order="F")
            .tolist(),
            float(envelope.expected_end_effector_mass_kg),
            *np.asarray(envelope.expected_end_effector_com_m, dtype=np.float64).tolist(),
            *np.asarray(
                envelope.expected_end_effector_inertia_kg_m2,
                dtype=np.float64,
            )
            .reshape(-1, order="F")
            .tolist(),
            float(envelope.expected_external_load_mass_kg),
        )
        if len(values) != 51 or not np.all(np.isfinite(values)):
            raise FrankaNativeSupervisedSessionError(
                "ARM numeric payload is not the frozen 51-double layout"
            )
        selected_controller_mode = _controller_mode(controller_mode)
        payload = _ARM.pack(
            _fixed_digest(envelope.profile_sha256, 32, "profile_sha256"),
            _fixed_digest(envelope.binding_sha256, 32, "envelope_sha256"),
            _fixed_digest(preflight_token.report_sha256, 32, "permit_sha256"),
            hashlib.sha256(active_run_id.encode("utf-8")).digest(),
            authorization_id,
            int(round(authorization.issued_monotonic_s * 1_000_000_000)),
            int(round(authorization.expires_monotonic_s * 1_000_000_000)),
            self.heartbeat_timeout_ns,
            self.first_target_timeout_ns,
            self.target_timeout_ns,
            self.maximum_session_duration_ns,
            *values,
            self.maximum_target_count,
            int(selected_controller_mode),
            bytes(8),
        )
        packet = self._encode(V94NativeMessageKind.ARM, payload)
        self._arm_encoded = True
        return packet

    def encode_heartbeat(self, *, monotonic_ns: int) -> bytes:
        del monotonic_ns  # The packet header carries the fresh monotonic time.
        if not self._arm_encoded or self._stop_encoded:
            raise FrankaNativeSupervisedSessionError(
                "HEARTBEAT requires active ARM and no STOP"
            )
        self._heartbeat_sequence += 1
        return self._encode(
            V94NativeMessageKind.HEARTBEAT,
            _HEARTBEAT.pack(self._heartbeat_sequence),
        )

    def encode_target(self, target: FrankaJointTarget) -> bytes:
        if not isinstance(target, FrankaJointTarget):
            raise TypeError("target must be FrankaJointTarget")
        if target.source is not FrankaTargetSource.SUPERVISED_V94:
            raise FrankaNativeSupervisedSessionError(
                "native TARGET must be supervised V94"
            )
        if not self._action_ready_decoded or self._stop_encoded:
            raise FrankaNativeSupervisedSessionError(
                "TARGET is forbidden before ACTION_READY or after STOP"
            )
        if self._pending_target_sequence is not None:
            raise FrankaNativeSupervisedSessionError(
                "TARGET is forbidden while a prior target awaits write ACK"
            )
        if target.sequence != self._last_target_sequence + 1:
            raise FrankaNativeSupervisedSessionError(
                "TARGET sequence does not increase exactly"
            )
        if target.sequence > self.maximum_target_count:
            raise FrankaNativeSupervisedSessionError(
                "TARGET sequence exceeds ARM maximum_target_count"
            )
        command = target.closed_loop_command
        if command is None or command.sequence != target.sequence:
            raise FrankaNativeSupervisedSessionError(
                "TARGET lacks its exact observation sequence"
            )
        payload = _TARGET.pack(
            target.sequence,
            command.sequence,
            int(round(target.produced_monotonic_s * 1_000_000_000)),
            *target.target_q_rad,
        )
        packet = self._encode(V94NativeMessageKind.TARGET, payload)
        self._last_target_sequence = target.sequence
        self._pending_target_sequence = target.sequence
        return packet

    def encode_stop(self, *, reason: str) -> bytes:
        if not str(reason).strip():
            raise ValueError("STOP reason must be non-empty")
        if self._session_nonce is None or self._stop_encoded:
            raise FrankaNativeSupervisedSessionError(
                "STOP requires HELLO and is single-use"
            )
        payload = _STOP.pack(1, 0, _positive_integer(self._monotonic_ns(), "STOP time"))
        packet = self._encode(V94NativeMessageKind.STOP, payload)
        self._stop_encoded = True
        return packet

    def _decode_packet(
        self, packet: bytes, *, channel: str
    ) -> tuple[V94NativeMessageKind, bytes]:
        if not isinstance(packet, bytes):
            raise TypeError("native packet must be bytes")
        if len(packet) < _HEADER.size or len(packet) > self.max_packet_bytes:
            raise FrankaNativeSupervisedSessionError(
                "native packet length is outside frozen bounds"
            )
        (
            magic,
            version,
            raw_kind,
            payload_bytes,
            flags,
            sequence,
            monotonic_ns,
            nonce,
            expected_crc,
            reserved,
        ) = _HEADER.unpack_from(packet)
        try:
            kind = V94NativeMessageKind(raw_kind)
        except ValueError as exc:
            raise FrankaNativeSupervisedSessionError(
                f"unknown native packet kind 0x{raw_kind:x}"
            ) from exc
        if magic != V94_NATIVE_PROTOCOL_MAGIC or version != V94_NATIVE_PROTOCOL_VERSION:
            raise FrankaNativeSupervisedSessionError("native packet magic/version mismatch")
        if flags != 0 or reserved != 0:
            raise FrankaNativeSupervisedSessionError(
                "native packet flags/reserved must be zero"
            )
        exact_payload = V94_NATIVE_PAYLOAD_BYTES[kind]
        if payload_bytes != exact_payload or len(packet) != _HEADER.size + exact_payload:
            raise FrankaNativeSupervisedSessionError(
                f"native {kind.name} packet has the wrong exact size"
            )
        copy = bytearray(packet)
        copy[48:52] = b"\x00" * 4
        if self.protocol_crc32(bytes(copy)) != expected_crc:
            raise FrankaNativeSupervisedSessionError("native packet CRC32 mismatch")
        if monotonic_ns <= 0:
            raise FrankaNativeSupervisedSessionError(
                "native packet monotonic_ns must be positive"
            )
        if channel == "critical":
            if kind not in self._critical_child_kinds:
                raise FrankaNativeSupervisedSessionError(
                    f"{kind.name} is forbidden on the critical child channel"
                )
            expected_sequence = self._child_critical_sequence + 1
            last_time = self._last_child_critical_monotonic_ns
            if self._session_nonce is None:
                if kind is not V94NativeMessageKind.HELLO or sequence != 1:
                    raise FrankaNativeSupervisedSessionError(
                        "HELLO must be child critical packet sequence one"
                    )
                if nonce == bytes(16):
                    raise FrankaNativeSupervisedSessionError(
                        "HELLO session nonce must not be all zero"
                    )
            elif nonce != self._session_nonce:
                raise FrankaNativeSupervisedSessionError(
                    "native critical packet session nonce changed"
                )
        elif channel == "telemetry":
            if kind is not V94NativeMessageKind.STATE:
                raise FrankaNativeSupervisedSessionError(
                    "telemetry channel accepts STATE only"
                )
            if self._session_nonce is None or nonce != self._session_nonce:
                raise FrankaNativeSupervisedSessionError(
                    "native STATE session nonce changed"
                )
            expected_sequence = self._child_state_sequence + 1
            last_time = self._last_child_state_monotonic_ns
        else:
            raise ValueError("channel must be critical or telemetry")
        if sequence != expected_sequence:
            raise FrankaNativeSupervisedSessionError(
                f"native {channel} packet sequence changed: "
                f"expected={expected_sequence}, actual={sequence}"
            )
        if monotonic_ns < last_time:
            raise FrankaNativeSupervisedSessionError(
                f"native {channel} packet monotonic clock regressed"
            )
        if channel == "critical":
            self._child_critical_sequence = sequence
            self._last_child_critical_monotonic_ns = monotonic_ns
            if self._session_nonce is None:
                self._session_nonce = bytes(nonce)
        else:
            self._child_state_sequence = sequence
            self._last_child_state_monotonic_ns = monotonic_ns
        return kind, packet[_HEADER.size:]

    def _decode_hello(self, payload: bytes) -> NativeHelloEvent:
        values = _HELLO.unpack(payload)
        process_id, decimation, version, safety_schema = values[:4]
        libfranka_sha, source_commit, producer_sha = values[4:7]
        limits = np.asarray(values[7:], dtype=np.float64)
        expectation = self.hello_expectation
        if process_id <= 0:
            raise FrankaNativeSupervisedSessionError("HELLO process_id is invalid")
        if (
            decimation != expectation.state_decimation
            or version != V94_NATIVE_PROTOCOL_VERSION
            or safety_schema != expectation.safety_limits_schema
            or libfranka_sha != expectation.libfranka_sha256
            or source_commit != expectation.libfranka_source_commit
            or producer_sha != expectation.producer_build_sha256
        ):
            raise FrankaNativeSupervisedSessionError(
                "HELLO native producer identity/protocol changed"
            )
        expected_limits = np.asarray(
            [0.50, 5.0, 250.0, 0.01, 0.020, 1.21, 0.01, 0.0008],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(limits)) or not np.allclose(
            limits, expected_limits, rtol=0.0, atol=1.0e-12
        ):
            raise FrankaNativeSupervisedSessionError(
                "HELLO compiled safety ceilings changed"
            )
        assert self._session_nonce is not None
        self._hello_validated = True
        return NativeHelloEvent(
            child_pid=process_id,
            session_nonce=self._session_nonce,
        )

    def _decode_ipc_ready(self, payload: bytes) -> NativeIpcReadyEvent:
        if not self._arm_encoded or self._ipc_ready_decoded:
            raise FrankaNativeSupervisedSessionError(
                "IPC_READY requires exactly one prior ARM"
            )
        values = _IPC_READY.unpack(payload)
        q = np.asarray(values[:7])
        dq = np.asarray(values[7:14])
        (
            q_error,
            robot_time,
            static_verified,
            scheduler_policy,
            scheduler_priority,
            realtime_cpu,
            affinity_cpu_count,
            reserved,
        ) = values[14:]
        if reserved != 0 or static_verified != 1:
            raise FrankaNativeSupervisedSessionError(
                "IPC_READY static proof/reserved field is invalid"
            )
        if self.expected_servo_cpu is not None:
            expected_policy = int(getattr(os, "SCHED_FIFO", 1))
            expected_priority = int(os.sched_get_priority_max(expected_policy))
            if (
                scheduler_policy != expected_policy
                or scheduler_priority != expected_priority
                or realtime_cpu != self.expected_servo_cpu
                or affinity_cpu_count != 1
            ):
                raise FrankaNativeSupervisedSessionError(
                    "IPC_READY realtime scheduler/CPU proof differs from "
                    "the admitted single-CPU servo partition"
                )
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq)):
            raise FrankaNativeSupervisedSessionError("IPC_READY vectors are not finite")
        if not np.isfinite(q_error) or q_error < 0.0 or q_error > 0.01:
            raise FrankaNativeSupervisedSessionError(
                "IPC_READY q-home error exceeds native ceiling"
            )
        event = NativeIpcReadyEvent(
            measured_q_rad=q,
            measured_dq_rad_s=dq,
            q_home_linf_error_rad=q_error,
            robot_time_ms=robot_time,
            static_provenance_verified=True,
            realtime_scheduler_policy=scheduler_policy,
            realtime_scheduler_priority=scheduler_priority,
            realtime_cpu=realtime_cpu,
            realtime_affinity_cpu_count=affinity_cpu_count,
        )
        self._ipc_ready_decoded = True
        return event

    def _decode_action_ready(self, payload: bytes) -> NativeActionReadyEvent:
        if not self._ipc_ready_decoded or self._action_ready_decoded:
            raise FrankaNativeSupervisedSessionError(
                "ACTION_READY requires exactly one prior IPC_READY"
            )
        values = _ACTION_READY.unpack(payload)
        cycles, max_period, hold_time, max_rtw, min_success, latest_success = values[:6]
        q = np.asarray(values[6:13])
        dq = np.asarray(values[13:20])
        robot_time, active_reads, active_writes, status_flags, missed_states = values[20:]
        if (
            cycles < 100
            or max_period > _NATIVE_MAX_ISOLATED_CONTROL_PERIOD_MS
            or max_rtw > 800_000
            or not np.isfinite(min_success)
            or min_success < 0.99
            or not np.isfinite(latest_success)
            or latest_success < 0.99
            or not np.all(np.isfinite(q))
            or not np.all(np.isfinite(dq))
            or active_reads < 100
            or active_writes < 100
            or active_writes > active_reads
            or status_flags != 0
        ):
            raise FrankaNativeSupervisedSessionError(
                "ACTION_READY did not prove the frozen 100-cycle health gate"
            )
        event = NativeActionReadyEvent(
            consecutive_healthy_cycles=cycles,
            maximum_control_period_ms=max_period,
            healthy_hold_robot_time_ms=hold_time,
            maximum_read_to_write_ns=max_rtw,
            minimum_control_command_success_rate=min_success,
            latest_control_command_success_rate=latest_success,
            measured_q_rad=q,
            measured_dq_rad_s=dq,
            robot_time_ms=robot_time,
            active_read_count=active_reads,
            active_write_count=active_writes,
            status_flags=status_flags,
            cumulative_missed_robot_states=missed_states,
        )
        self._action_ready_decoded = True
        return event

    def _decode_ack(self, payload: bytes) -> NativeAckEvent:
        values = _ACK.unpack(payload)
        if values[8] != 0:
            raise FrankaNativeSupervisedSessionError("ACK reserved field is nonzero")
        target = np.asarray(values[9:16])
        commanded = np.asarray(values[16:23])
        measured = np.asarray(values[23:30])
        tracking = float(values[31])
        if (
            not np.all(np.isfinite(target))
            or not np.all(np.isfinite(commanded))
            or not np.all(np.isfinite(measured))
            or values[7] > _NATIVE_MAX_ISOLATED_CONTROL_PERIOD_MS
            or values[30] > 800_000
            or not np.isfinite(tracking)
            or tracking < 0.0
            or tracking > 0.01
        ):
            raise FrankaNativeSupervisedSessionError(
                "ACK violates frozen write evidence bounds"
            )
        event = NativeAckEvent(
            sequence=values[0],
            observation_sequence=values[1],
            target_produced_monotonic_ns=values[2],
            target_age_at_write_ns=values[3],
            control_cycle=values[4],
            robot_time_ms=values[5],
            applied_monotonic_ns=values[6],
            control_period_ms=values[7],
            target_q_rad=target,
            commanded_q_rad=commanded,
            measured_q_rad=measured,
            read_to_write_ns=values[30],
            maximum_tracking_error_rad=tracking,
        )
        if self._pending_target_sequence != event.sequence:
            raise FrankaNativeSupervisedSessionError(
                "ACK does not match the codec's pending TARGET"
            )
        self._pending_target_sequence = None
        return event

    def _decode_fault(self, payload: bytes) -> NativeFaultEvent:
        code, detail_bytes, cycle, sequence, robot_time, system_errno, reserved, raw = (
            _FAULT.unpack(payload)
        )
        if reserved != 0 or code <= 0 or code > 19:
            raise FrankaNativeSupervisedSessionError(
                "FAULT code/reserved field is invalid"
            )
        detail = _decode_fixed_text(raw, length=detail_bytes)
        if not detail:
            detail = "native child reported an empty diagnostic"
        return NativeFaultEvent(
            code=code,
            reason=detail,
            sequence=sequence,
            control_cycle=cycle,
            robot_time_ms=robot_time,
            system_errno=system_errno,
        )

    def _decode_stop_proof(self, payload: bytes) -> NativeStopProofEvent:
        prefix = _STOP_PROOF_PREFIX.unpack_from(payload)
        stop_reason, terminal_fault = prefix[:2]
        flags = tuple(
            _wire_boolean(value, f"STOP_PROOF flag[{index}]")
            for index, value in enumerate(prefix[2:10])
        )
        (
            control_cycles,
            last_target,
            active_reads,
            active_writes,
            ack_count,
        ) = prefix[10:15]
        (
            verification_count,
            consecutive_idle,
            final_mode,
            maximum_period_ms,
        ) = prefix[15:19]
        maximum_stop_dq = float(prefix[19])
        maximum_rtw, pre_stop_time, final_time, drop_count = prefix[20:24]
        offset = _STOP_PROOF_PREFIX.size
        samples = []
        for _index in range(3):
            sample = _STOP_SAMPLE.unpack_from(payload, offset)
            offset += _STOP_SAMPLE.size
            samples.append(
                NativeStopVerificationSample(
                    robot_time_ms=sample[0],
                    robot_mode=sample[1],
                    status_flags=sample[2],
                    measured_dq_rad_s=np.asarray(sample[3:]),
                )
            )
        detail = _decode_fixed_text(payload[offset:])
        if stop_reason not in (1, 2, 3, 4) or terminal_fault > 19:
            raise FrankaNativeSupervisedSessionError(
                "STOP_PROOF reason/fault code is invalid"
            )
        if active_writes > active_reads or ack_count > active_writes:
            raise FrankaNativeSupervisedSessionError(
                "STOP_PROOF counters are internally inconsistent"
            )
        if flags[1] and not flags[0]:
            raise FrankaNativeSupervisedSessionError(
                "STOP_PROOF reports finish success without a finish attempt"
            )
        if flags[3] and not flags[2]:
            raise FrankaNativeSupervisedSessionError(
                "STOP_PROOF reports Robot.stop success without an attempt"
            )
        if flags[1] and flags[2]:
            raise FrankaNativeSupervisedSessionError(
                "STOP_PROOF reports Robot.stop after a successful finish"
            )
        healthy_finish_path = bool(flags[0] and flags[1] and not flags[2])
        direct_stop_path = bool(not flags[0] and flags[2])
        failed_finish_stop_fallback = bool(flags[0] and not flags[1] and flags[2])
        allowed_stop_modes = {1}
        if (
            (direct_stop_path or failed_finish_stop_fallback)
            and flags[3]
        ):
            # An active-session fault can itself leave this libfranka/FR3 pair
            # in kReflex after a successful explicit Robot.stop.  The native
            # side accepts only the exact communication-only error tuple and
            # still proves three fresh low-dq samples.
            allowed_stop_modes.add(5)
            if terminal_fault == 0 and stop_reason in (1, 2):
                # Some requested-stop transitions report kOther without an
                # error tuple. Guiding/user-stop/recovery remain rejected.
                allowed_stop_modes.add(3)
        sample_times = [sample.robot_time_ms for sample in samples]
        sample_dq = np.asarray([sample.measured_dq_rad_s for sample in samples])
        def _stop_status_is_clear(sample: NativeStopVerificationSample) -> bool:
            requested_stop_communication_status = (1 << 0) | (1 << 5) | (1 << 6)
            return sample.status_flags == 0 or bool(
                sample.robot_mode in (1, 5)
                and sample.robot_mode in allowed_stop_modes
                and sample.status_flags == requested_stop_communication_status
            )

        raw_idle_verified = bool(
            verification_count >= 3
            and consecutive_idle >= 3
            and final_mode in allowed_stop_modes
            and pre_stop_time < sample_times[0] < sample_times[1] < sample_times[2]
            and final_time == sample_times[-1]
            and all(sample.robot_mode in allowed_stop_modes for sample in samples)
            and all(_stop_status_is_clear(sample) for sample in samples)
            and sample_dq.shape == (3, 7)
            and np.all(np.isfinite(sample_dq))
            and np.max(np.abs(sample_dq)) <= 0.01
        )
        idle_dq_claim = flags[4]
        if idle_dq_claim != raw_idle_verified:
            raise FrankaNativeSupervisedSessionError(
                "STOP_PROOF idle_dq_verified differs from parent raw-sample result"
            )
        if not np.isfinite(maximum_stop_dq) or maximum_stop_dq < 0.0:
            raise FrankaNativeSupervisedSessionError(
                "STOP_PROOF maximum stop velocity is invalid"
            )
        # Robot.stop() return status remains fault evidence but is not allowed
        # to contradict three fresh Idle samples.  Physical stop is recomputed
        # from the mutually exclusive terminal path plus released ownership and
        # raw post-stop samples, never from child exit or its boolean alone.
        parent_stop_verified = bool(
            (healthy_finish_path or direct_stop_path or failed_finish_stop_fallback)
            and flags[5]
            and flags[7]
            and raw_idle_verified
        )
        fault_reason = detail or (
            None if terminal_fault == 0 else f"terminal fault code {terminal_fault}"
        )
        return NativeStopProofEvent(
            stop_reason=stop_reason,
            terminal_fault_code=terminal_fault,
            finish_attempted=flags[0],
            finish_succeeded=flags[1],
            robot_stop_attempted=flags[2],
            robot_stop_succeeded=flags[3],
            idle_dq_verified=flags[4],
            active_handle_released=flags[5],
            fault_reply_delivered=flags[6],
            robot_backend_released=flags[7],
            control_cycles=control_cycles,
            last_target_sequence=last_target,
            active_read_count=active_reads,
            active_write_count=active_writes,
            franka_ack_count=ack_count,
            stop_verification_samples=verification_count,
            stop_consecutive_idle_samples=consecutive_idle,
            final_robot_mode=final_mode,
            maximum_control_period_ms=maximum_period_ms,
            maximum_stop_dq_rad_s=maximum_stop_dq,
            maximum_read_to_write_ns=maximum_rtw,
            pre_stop_robot_time_ms=pre_stop_time,
            final_robot_time_ms=final_time,
            telemetry_drop_count=drop_count,
            verified_samples=tuple(samples),
            detail=detail,
            parent_stop_verified=parent_stop_verified,
            maximum_control_period_s=maximum_period_ms / 1000.0,
            fault_reason=fault_reason,
        )

    def decode_critical_packet(self, packet: bytes) -> NativeChildEvent:
        kind, payload = self._decode_packet(packet, channel="critical")
        if kind is V94NativeMessageKind.HELLO:
            return self._decode_hello(payload)
        if kind is V94NativeMessageKind.IPC_READY:
            return self._decode_ipc_ready(payload)
        if kind is V94NativeMessageKind.ACTION_READY:
            return self._decode_action_ready(payload)
        if kind is V94NativeMessageKind.ACK:
            return self._decode_ack(payload)
        if kind is V94NativeMessageKind.FAULT:
            return self._decode_fault(payload)
        if kind is V94NativeMessageKind.STOP_PROOF:
            return self._decode_stop_proof(payload)
        raise FrankaNativeSupervisedSessionError(
            f"unsupported critical child packet {kind.name}"
        )

    def decode_state_packet(self, packet: bytes) -> NativeStateEvent:
        kind, payload = self._decode_packet(packet, channel="telemetry")
        assert kind is V94NativeMessageKind.STATE
        values = _STATE.unpack(payload)
        q = np.asarray(values[9:16])
        dq = np.asarray(values[16:23])
        transform = np.asarray(values[23:39]).reshape((4, 4), order="F")
        commanded = np.asarray(values[39:46])
        controller_state = np.asarray(values[56:85], dtype=np.float32)
        shaper_q_d = np.asarray(values[85:92])
        shaper_dq_d = np.asarray(values[92:99])
        shaper_ddq_d = np.asarray(values[99:106])
        held_q_cmd = np.asarray(values[106:113])
        finite = np.concatenate(
            (
                q,
                dq,
                transform.reshape(-1),
                commanded,
                controller_state.astype(np.float64),
                shaper_q_d,
                shaper_dq_d,
                shaper_ddq_d,
                held_q_cmd,
                [values[8]],
            )
        )
        if not np.all(np.isfinite(finite)):
            raise FrankaNativeSupervisedSessionError("STATE contains nonfinite data")
        if (
            np.any(controller_state[:14] < -4.0)
            or np.any(controller_state[:14] > 4.0)
            or np.any(controller_state[14:28] < -1.0)
            or np.any(controller_state[14:28] > 1.0)
            or controller_state[28] != np.float32(1.0)
        ):
            raise FrankaNativeSupervisedSessionError(
                "STATE controller_state29 violates the q_d observation bounds"
            )
        expected_controller_state = np.empty(29, dtype=np.float32)
        expected_controller_state[0:7] = np.clip(
            (held_q_cmd - shaper_q_d) / 0.018, -4.0, 4.0
        )
        expected_controller_state[7:14] = np.clip(
            (shaper_q_d - q) / 0.05, -4.0, 4.0
        )
        expected_controller_state[14:21] = np.clip(
            shaper_dq_d / 0.5, -1.0, 1.0
        )
        expected_controller_state[21:28] = np.clip(
            shaper_ddq_d / 5.0, -1.0, 1.0
        )
        expected_controller_state[28] = np.float32(1.0)
        if not np.array_equal(commanded, shaper_q_d):
            raise FrankaNativeSupervisedSessionError(
                "STATE commanded_q_rad differs from post-limit shaper_q_d_rad"
            )
        if not np.array_equal(controller_state, expected_controller_state):
            raise FrankaNativeSupervisedSessionError(
                "STATE controller_state29 is not coherent with its post-limit "
                "shaper snapshot"
            )
        return NativeStateEvent(
            cycle=values[0],
            robot_time_ms=values[1],
            monotonic_s=values[2] / 1_000_000_000,
            realtime_s=values[3] / 1_000_000_000,
            active_target_sequence=values[4],
            active_observation_sequence=values[5],
            control_period_ms=values[6],
            robot_mode=values[7],
            control_command_success_rate=values[8],
            q_rad=q,
            dq_rad_s=dq,
            T_base_eef=transform,
            commanded_q_rad=commanded,
            shaper_q_d_rad=shaper_q_d,
            shaper_dq_d_rad_s=shaper_dq_d,
            shaper_ddq_d_rad_s2=shaper_ddq_d,
            held_q_cmd_rad=held_q_cmd,
            controller_state29=controller_state,
            status_flags=values[46],
            cumulative_missed_robot_states=values[47],
            last_read_to_write_ns=values[48],
            maximum_read_to_write_ns=values[49],
            telemetry_drop_count=values[50],
            health_flags=values[51],
            consecutive_healthy_cycles=values[52],
            active_read_count=values[53],
            active_write_count=values[54],
            target_ack_count=values[55],
        )


@runtime_checkable
class NativeChildProcess(Protocol):
    @property
    def pid(self) -> Optional[int]:
        ...

    @property
    def exitcode(self) -> Optional[int]:
        ...

    def is_alive(self) -> bool:
        ...

    def join(self, timeout: Optional[float] = None) -> None:
        ...

    def terminate(self) -> None:
        ...


@dataclass(frozen=True)
class NativeChildEndpoint:
    """One child plus its critical and lossy-telemetry parent fds."""

    critical_sock: socket.socket
    telemetry_sock: socket.socket
    process: NativeChildProcess

    def __post_init__(self) -> None:
        for name, sock, expected_type in (
            ("critical", self.critical_sock, socket.SOCK_SEQPACKET),
            ("telemetry", self.telemetry_sock, socket.SOCK_DGRAM),
        ):
            if not isinstance(sock, socket.socket):
                raise TypeError(f"native child endpoint requires a {name} socket")
            if sock.family != socket.AF_UNIX:
                raise ValueError(f"native child {name} endpoint must use AF_UNIX")
            # ``socket.type`` is fixed by socketpair construction.  The native
            # child independently verifies SO_TYPE/SO_DOMAIN/SO_PEERCRED before
            # opening Robot; this parent check stays hardware- and syscall-light.
            socket_type = int(sock.type) & 0xF
            if socket_type != expected_type:
                raise ValueError(
                    f"native child {name} endpoint has the wrong socket type"
                )
        for name in ("is_alive", "join", "terminate"):
            if not callable(getattr(self.process, name, None)):
                raise TypeError(f"native child process is missing {name}()")


@runtime_checkable
class NativeChildLauncher(Protocol):
    construction_is_inert: bool

    def start(self) -> NativeChildEndpoint:
        ...


def _spawned_child_bootstrap(
    child_main: Callable[..., None],
    critical_sock: socket.socket,
    telemetry_sock: socket.socket,
    child_args: tuple[Any, ...],
) -> None:
    """Process entrypoint kept hardware-agnostic and pickleable."""

    try:
        child_main(critical_sock, telemetry_sock, *child_args)
    finally:
        critical_sock.close()
        telemetry_sock.close()


class SpawnedSeqpacketChildLauncher:
    """Start an injected child with isolated critical and telemetry channels.

    Merely constructing this launcher starts no thread, process, socket, or
    device.  The default ``spawn`` context also prevents accidental inheritance
    of parent-owned camera/serial/FCI handles.
    """

    construction_is_inert = True

    def __init__(
        self,
        child_main: Callable[..., None],
        *,
        child_args: Sequence[Any] = (),
        context_name: str = "spawn",
        process_name: str = "v94-franka-native-supervised",
    ) -> None:
        if not callable(child_main):
            raise TypeError("child_main must be callable")
        context = multiprocessing.get_context(str(context_name))
        name = str(process_name).strip()
        if not name:
            raise ValueError("process_name must be non-empty")
        self._child_main = child_main
        self._child_args = tuple(child_args)
        self._context = context
        self._process_name = name
        self._start_lock = threading.Lock()
        self._started = False

    def start(self) -> NativeChildEndpoint:
        with self._start_lock:
            if self._started:
                raise FrankaNativeSupervisedSessionError(
                    "native child launcher is single-use"
                )
            self._started = True
        parent_critical, child_critical = socket.socketpair(
            socket.AF_UNIX,
            socket.SOCK_SEQPACKET,
        )
        parent_telemetry, child_telemetry = socket.socketpair(
            socket.AF_UNIX,
            socket.SOCK_DGRAM,
        )
        try:
            # The parent critical writer must never sleep inside a send after
            # a TARGET header has captured its freshness timestamp.
            parent_critical.setblocking(False)
        except BaseException:
            parent_critical.close()
            child_critical.close()
            parent_telemetry.close()
            child_telemetry.close()
            raise
        process = self._context.Process(
            target=_spawned_child_bootstrap,
            args=(
                self._child_main,
                child_critical,
                child_telemetry,
                self._child_args,
            ),
            name=self._process_name,
            daemon=False,
        )
        try:
            process.start()
        except BaseException:
            parent_critical.close()
            child_critical.close()
            parent_telemetry.close()
            child_telemetry.close()
            raise
        child_critical.close()
        child_telemetry.close()
        return NativeChildEndpoint(
            critical_sock=parent_critical,
            telemetry_sock=parent_telemetry,
            process=process,
        )


class PopenChildProcessHandle:
    """Small ``subprocess.Popen`` adapter with an optional Linux pidfd."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self._process = process
        self._pidfd: Optional[int] = None
        pidfd_open = getattr(os, "pidfd_open", None)
        if callable(pidfd_open):
            try:
                self._pidfd = int(pidfd_open(process.pid, 0))
            except OSError:
                self._pidfd = None

    @property
    def pid(self) -> Optional[int]:
        return int(self._process.pid)

    @property
    def pidfd(self) -> Optional[int]:
        return self._pidfd

    @property
    def exitcode(self) -> Optional[int]:
        return self._process.poll()

    def is_alive(self) -> bool:
        return self._process.poll() is None

    def join(self, timeout: Optional[float] = None) -> None:
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return

    def terminate(self) -> None:
        self._process.terminate()

    def close(self) -> None:
        pidfd = self._pidfd
        self._pidfd = None
        if pidfd is not None:
            os.close(pidfd)


class PopenSeqpacketChildLauncher:
    """Exec a native child with exactly two inherited connected descriptors.

    ``argv_builder`` receives ``(critical_fd, telemetry_fd)`` and returns the
    complete argv.  Requiring that injection here avoids inventing CLI flags
    before the native executable's command-line contract is frozen.  Shell
    execution is never used.
    """

    construction_is_inert = True

    def __init__(
        self,
        argv_builder: Callable[[int, int], Sequence[str]],
        *,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        pinned_executable_fd: Optional[int] = None,
        prelaunch_validator: Optional[Callable[[], None]] = None,
    ) -> None:
        if not callable(argv_builder):
            raise TypeError("argv_builder must be callable")
        if prelaunch_validator is not None and not callable(
            prelaunch_validator
        ):
            raise TypeError("prelaunch_validator must be callable")
        if cwd is not None and not str(cwd).strip():
            raise ValueError("cwd must be non-empty when supplied")
        if env is not None:
            normalized_env: Optional[dict[str, str]] = {
                str(key): str(value) for key, value in env.items()
            }
        else:
            normalized_env = None
        if pinned_executable_fd is not None:
            if isinstance(pinned_executable_fd, bool):
                raise ValueError("pinned_executable_fd must be an open fd")
            executable_fd = int(pinned_executable_fd)
            if executable_fd < 3:
                raise ValueError("pinned_executable_fd must be at least 3")
            try:
                os.fstat(executable_fd)
            except OSError as exc:
                raise ValueError("pinned_executable_fd is not open") from exc
        else:
            executable_fd = None
        self._argv_builder = argv_builder
        self._cwd = None if cwd is None else str(cwd)
        self._env = normalized_env
        # The production caller opens and hashes this fd before constructing
        # the launcher.  Executing /proc/self/fd/N pins the checked inode even
        # if an offline rebuild atomically replaces the path before Popen.
        self._pinned_executable_fd = executable_fd
        self._prelaunch_validator = prelaunch_validator
        self._start_lock = threading.Lock()
        self._started = False

    def _release_pinned_executable_fd(self) -> None:
        executable_fd = getattr(self, "_pinned_executable_fd", None)
        self._pinned_executable_fd = None
        if executable_fd is not None:
            try:
                os.close(executable_fd)
            except OSError:
                pass

    def close(self) -> None:
        with self._start_lock:
            self._release_pinned_executable_fd()

    def __del__(self) -> None:
        self._release_pinned_executable_fd()

    def start(self) -> NativeChildEndpoint:
        with self._start_lock:
            if self._started:
                raise FrankaNativeSupervisedSessionError(
                    "native Popen launcher is single-use"
                )
            self._started = True
        parent_critical, child_critical = socket.socketpair(
            socket.AF_UNIX,
            socket.SOCK_SEQPACKET,
        )
        parent_telemetry, child_telemetry = socket.socketpair(
            socket.AF_UNIX,
            socket.SOCK_DGRAM,
        )
        executable_fd = self._pinned_executable_fd
        try:
            # The native child independently makes its own endpoint
            # nonblocking.  This is the opposite socketpair endpoint: set it
            # explicitly so a parent TARGET send has no blocking path.
            parent_critical.setblocking(False)
            argv = tuple(
                str(value)
                for value in self._argv_builder(
                    child_critical.fileno(),
                    child_telemetry.fileno(),
                )
            )
            if not argv or not argv[0].strip():
                raise ValueError("argv_builder returned an empty executable")
            # This callback is deliberately adjacent to Popen, after camera,
            # policy, and viewer initialization.  Production uses it to
            # re-read the frozen NIC IRQ/irqbalance admission token so a
            # host-state change in that initialization window cannot reach
            # native HELLO or an FCI handle.
            if self._prelaunch_validator is not None:
                self._prelaunch_validator()
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                # Never leave an undrained PIPE that can block the servo child.
                stderr=subprocess.DEVNULL,
                cwd=self._cwd,
                env=self._env,
                close_fds=True,
                pass_fds=(
                    child_critical.fileno(),
                    child_telemetry.fileno(),
                    *((executable_fd,) if executable_fd is not None else ()),
                ),
                shell=False,
            )
            # Open the optional pidfd while the pinned executable descriptor
            # is still live.  Otherwise Linux may immediately recycle that
            # descriptor number for the pidfd after the close below, making
            # the parent appear to retain the executable fd even though it
            # was correctly closed.
            process_handle = PopenChildProcessHandle(process)
        except BaseException:
            parent_critical.close()
            child_critical.close()
            parent_telemetry.close()
            child_telemetry.close()
            self._release_pinned_executable_fd()
            raise
        self._release_pinned_executable_fd()
        child_critical.close()
        child_telemetry.close()
        return NativeChildEndpoint(
            critical_sock=parent_critical,
            telemetry_sock=parent_telemetry,
            process=process_handle,
        )


class NativeFrankaTargetSampleHold(FrankaTargetSampleHold):
    """The runtime's target hold with an active child notification edge."""

    def __init__(self, on_publish: Callable[[FrankaJointTarget], None]) -> None:
        if not callable(on_publish):
            raise TypeError("on_publish must be callable")
        super().__init__()
        self._on_publish = on_publish

    def publish(self, target: FrankaJointTarget) -> None:
        # Preserve all sequence/timestamp/type validation from the original
        # hold.  Notification happens immediately after the target becomes the
        # unique published value; there is no polling/peek delivery loop.
        super().publish(target)
        self._on_publish(target)


class FrankaNativeSupervisedSessionProxy:
    """Parent-side stand-in for ``FrankaPersistentSession``.

    ``run()`` owns the child event pump.  Policy targets are pushed through
    :class:`NativeFrankaTargetSampleHold`; STATE updates the local pose ring;
    only a matching child ACK can acknowledge the parent Franka ledger edge.
    Process exit alone never becomes a physical-stop proof.
    """

    def __init__(
        self,
        *,
        run_id: str,
        envelope: Any,
        action_ledger: ExecutedActionLedger,
        codec: NativeSupervisedPacketCodec,
        launcher: NativeChildLauncher,
        controller_mode: V94NativeControllerMode = V94NativeControllerMode.LEGACY,
        heartbeat_interval_s: float = 0.010,
        event_poll_interval_s: float = 0.005,
        startup_timeout_s: float = 2.0,
        shutdown_timeout_s: float = 2.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        identifier = str(run_id).strip()
        if not identifier:
            raise ValueError("run_id must be non-empty")
        if not isinstance(action_ledger, ExecutedActionLedger):
            raise TypeError("action_ledger must be ExecutedActionLedger")
        maximum_packet = _positive_integer(
            getattr(codec, "max_packet_bytes", None),
            "codec.max_packet_bytes",
        )
        if maximum_packet > 1024 * 1024:
            raise ValueError("codec.max_packet_bytes exceeds the 1 MiB hard bound")
        required_codec_methods = (
            "encode_arm",
            "encode_heartbeat",
            "encode_target",
            "encode_stop",
            "decode_critical_packet",
            "decode_state_packet",
        )
        for name in required_codec_methods:
            if not callable(getattr(codec, name, None)):
                raise TypeError(f"codec is missing {name}()")
        if getattr(launcher, "construction_is_inert", None) is not True:
            raise ValueError("native child launcher construction must be inert")
        if not callable(getattr(launcher, "start", None)):
            raise TypeError("launcher is missing start()")
        pose_capacity = _positive_integer(
            getattr(envelope, "pose_ring_capacity", None),
            "envelope.pose_ring_capacity",
        )
        if pose_capacity < 2:
            raise ValueError("envelope.pose_ring_capacity must be at least two")

        self.run_id = identifier
        self.envelope = envelope
        self.action_ledger = action_ledger
        self.controller_mode = _controller_mode(controller_mode)
        self.pose_ring = FrankaPoseRing(pose_capacity)
        self.target_hold = NativeFrankaTargetSampleHold(self._publish_target)
        self._codec = codec
        self._launcher = launcher
        self._maximum_packet_bytes = maximum_packet
        self._heartbeat_interval_s = _positive_float(
            heartbeat_interval_s, "heartbeat_interval_s"
        )
        self._event_poll_interval_s = _positive_float(
            event_poll_interval_s, "event_poll_interval_s"
        )
        self._startup_timeout_s = _positive_float(
            startup_timeout_s, "startup_timeout_s"
        )
        self._shutdown_timeout_s = _positive_float(
            shutdown_timeout_s, "shutdown_timeout_s"
        )
        if not callable(monotonic):
            raise TypeError("monotonic must be callable")
        self._monotonic = monotonic

        self._state_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._endpoint: Optional[NativeChildEndpoint] = None
        self._run_started = False
        self._run_finished = False
        self._hello: Optional[NativeHelloEvent] = None
        self._ipc_ready = False
        self._ipc_ready_event: Optional[NativeIpcReadyEvent] = None
        self._action_ready = False
        self._bootstrap_ready = False
        self._stop_packet_sent = False
        self._pending_target: Optional[FrankaJointTarget] = None
        self._last_sent_sequence = 0
        self._last_ack_sequence = 0
        self._last_ack_control_cycle = 0
        self._pose_publication_count = 0
        self._latest_child_control_cycle = 0
        self._last_parent_heartbeat_monotonic_s: Optional[float] = None
        self._parent_heartbeat_count = 0
        self._maximum_parent_heartbeat_interval_s = 0.0
        self._async_failure: Optional[BaseException] = None
        self._last_telemetry: Optional[NativeFrankaTelemetry] = None

    @property
    def c2_bootstrap_ready(self) -> bool:
        with self._state_lock:
            return self._bootstrap_ready

    @property
    def last_telemetry(self) -> Optional[NativeFrankaTelemetry]:
        with self._state_lock:
            return self._last_telemetry

    @property
    def child_pid(self) -> Optional[int]:
        with self._state_lock:
            endpoint = self._endpoint
        return None if endpoint is None else endpoint.process.pid

    def _critical_endpoint_before_encode(
        self, *, operation: str
    ) -> NativeChildEndpoint:
        """Resolve the endpoint before a packet captures its header time."""

        with self._state_lock:
            endpoint = self._endpoint
        if endpoint is None:
            raise FrankaNativeSupervisedSessionError(
                f"cannot send {operation} before child start"
            )
        return endpoint

    @staticmethod
    def _require_critical_writable_before_encode(
        endpoint: NativeChildEndpoint, *, operation: str
    ) -> None:
        """Take one zero-wait writable boundary before encoding.

        Writability may change immediately afterward.  The actual send is
        therefore still MSG_DONTWAIT and never retries; this check exists to
        avoid creating a timestamped packet while the queue is already full.
        """

        try:
            _readable, writable, exceptional = select.select(
                [],
                [endpoint.critical_sock],
                [endpoint.critical_sock],
                0.0,
            )
        except (OSError, ValueError) as exc:
            raise FrankaNativeSupervisedSessionError(
                f"{operation} critical seqpacket writable check failed: {exc}"
            ) from exc
        if exceptional or endpoint.critical_sock not in writable:
            raise FrankaNativeSupervisedSessionError(
                f"{operation} critical seqpacket is not immediately writable; "
                "packet was not encoded or retried"
            )

    def _require_target_fresh_at_boundary(
        self,
        target: FrankaJointTarget,
        *,
        boundary: str,
        encoded: bool,
    ) -> None:
        now = self._monotonic()
        age_s = now - float(target.produced_monotonic_s)
        disposition = (
            "packet was encoded but not sent"
            if encoded
            else "packet was not encoded or sent"
        )
        if age_s > _PARENT_TARGET_MAX_AGE_S:
            raise FrankaNativeSupervisedSessionError(
                f"TARGET expired {boundary}: "
                f"age_s={age_s:.9f} limit_s={_PARENT_TARGET_MAX_AGE_S:.9f}; "
                f"{disposition}"
            )
        if age_s < -_PARENT_TARGET_MAX_FUTURE_SKEW_S:
            raise FrankaNativeSupervisedSessionError(
                "TARGET produced timestamp is too far in the future "
                f"{boundary}: "
                f"future_s={-age_s:.9f} "
                f"limit_s={_PARENT_TARGET_MAX_FUTURE_SKEW_S:.9f}; "
                f"{disposition}"
            )

    @staticmethod
    def _force_parent_critical_nonblocking(
        endpoint: NativeChildEndpoint,
    ) -> None:
        """Make even an injected launcher's parent critical fd zero-wait."""

        try:
            endpoint.critical_sock.setblocking(False)
            socket_reports_blocking = endpoint.critical_sock.getblocking()
            fd_reports_blocking = os.get_blocking(endpoint.critical_sock.fileno())
        except (OSError, ValueError) as exc:
            raise FrankaNativeSupervisedSessionError(
                "could not establish nonblocking parent critical seqpacket"
            ) from exc
        if socket_reports_blocking or fd_reports_blocking:
            raise FrankaNativeSupervisedSessionError(
                "parent critical seqpacket remained blocking after configuration"
            )

    @staticmethod
    def _terminate_endpoint_best_effort(endpoint: NativeChildEndpoint) -> None:
        """Independent stop edge after an uncertain critical send boundary."""

        try:
            if endpoint.process.is_alive():
                endpoint.process.terminate()
        except BaseException:
            pass

    def _send_packet_once_locked(
        self,
        endpoint: NativeChildEndpoint,
        packet: object,
        *,
        operation: str,
    ) -> None:
        if not isinstance(packet, bytes) or not packet:
            raise FrankaNativeSupervisedSessionError(
                f"{operation} codec output must be non-empty bytes"
            )
        if len(packet) > self._maximum_packet_bytes:
            raise FrankaNativeSupervisedSessionError(
                f"{operation} packet exceeds codec.max_packet_bytes"
            )
        flags = socket.MSG_DONTWAIT | getattr(socket, "MSG_NOSIGNAL", 0)
        try:
            # One call is one complete SOCK_SEQPACKET record.  In particular,
            # TARGET is never retried after EAGAIN because application status
            # would otherwise be ambiguous.
            try:
                sent = endpoint.critical_sock.send(packet, flags)
            except PermissionError as exc:
                if exc.errno != errno.EPERM:
                    raise
                # Some restricted Python syscall profiles deny send(2) with
                # flags while permitting write(2).  EPERM proves the first
                # call enqueued no record; O_NONBLOCK on the descriptor gives
                # this compatibility call the same zero-wait/no-retry
                # semantics.  EAGAIN and every ambiguous error still fault.
                sent = os.write(endpoint.critical_sock.fileno(), packet)
        except BlockingIOError as exc:
            self._terminate_endpoint_best_effort(endpoint)
            raise FrankaNativeSupervisedSessionError(
                f"{operation} critical seqpacket send would block "
                "(EAGAIN); packet was not retried and native terminate was "
                "requested"
            ) from exc
        except InterruptedError as exc:
            self._terminate_endpoint_best_effort(endpoint)
            raise FrankaNativeSupervisedSessionError(
                f"{operation} critical seqpacket send was interrupted; "
                "packet was not retried and native terminate was requested"
            ) from exc
        except OSError as exc:
            self._terminate_endpoint_best_effort(endpoint)
            raise FrankaNativeSupervisedSessionError(
                f"{operation} critical seqpacket send failed: "
                f"errno={exc.errno}; packet was not retried and native "
                "terminate was requested"
            ) from exc
        if sent != len(packet):
            self._terminate_endpoint_best_effort(endpoint)
            raise FrankaNativeSupervisedSessionError(
                f"short SOCK_SEQPACKET send during {operation}: "
                f"{sent}/{len(packet)}; packet was not retried and native "
                "terminate was requested"
            )

    def _send_encoded(
        self,
        encoder: Callable[[], bytes],
        *,
        operation: str,
        target: Optional[FrankaJointTarget] = None,
    ) -> None:
        # Encode and send share one lock because the production fixed-POD codec
        # owns the strictly increasing parent packet sequence.  All unbounded
        # lock acquisition and the zero-wait writable check happen before the
        # encoder captures its monotonic header timestamp.
        with self._send_lock:
            endpoint = self._critical_endpoint_before_encode(
                operation=operation
            )
            self._require_critical_writable_before_encode(
                endpoint,
                operation=operation,
            )
            if operation == "TARGET":
                if target is None:
                    raise FrankaNativeSupervisedSessionError(
                        "TARGET send requires its exact target freshness proof"
                    )
                if self._stop_requested.is_set():
                    raise FrankaNativeSupervisedSessionError(
                        "TARGET send was cancelled because stop is requested; "
                        "packet was not encoded or sent"
                    )
                self._require_target_fresh_at_boundary(
                    target,
                    boundary="before native encode/send boundary",
                    encoded=False,
                )
            elif target is not None:
                raise FrankaNativeSupervisedSessionError(
                    f"{operation} cannot carry TARGET freshness state"
                )
            packet = encoder()
            if operation == "TARGET":
                assert target is not None
                if self._stop_requested.is_set():
                    raise FrankaNativeSupervisedSessionError(
                        "TARGET send was cancelled after encode because stop is "
                        "requested; packet was not sent"
                    )
                # Encoding is normally tiny, but it is still Python and can be
                # descheduled.  Do not pass an already-expired header/payload
                # to the child after that gap.
                self._require_target_fresh_at_boundary(
                    target,
                    boundary="after native encode/before send boundary",
                    encoded=True,
                )
            self._send_packet_once_locked(
                endpoint,
                packet,
                operation=operation,
            )
            sent_monotonic_s = self._monotonic()
            if operation in ("ARM", "HEARTBEAT"):
                with self._state_lock:
                    previous = self._last_parent_heartbeat_monotonic_s
                    if operation == "HEARTBEAT":
                        self._parent_heartbeat_count += 1
                        if previous is not None:
                            self._maximum_parent_heartbeat_interval_s = max(
                                self._maximum_parent_heartbeat_interval_s,
                                sent_monotonic_s - previous,
                            )
                    self._last_parent_heartbeat_monotonic_s = sent_monotonic_s

    def _fail_pending_ledger(self, reason: str) -> None:
        pending = self.action_ledger.pending_sequence
        if pending is None:
            return
        try:
            self.action_ledger.fail(
                "franka",
                sequence=pending,
                reason=reason,
            )
        except BaseException:
            # The first protocol failure remains the useful root cause.
            pass

    def _latch_async_failure(self, failure: BaseException) -> None:
        with self._state_lock:
            if self._async_failure is None:
                self._async_failure = failure
        self._fail_pending_ledger(str(failure))
        self._stop_requested.set()

    def _publish_target(self, target: FrankaJointTarget) -> None:
        try:
            if target.source is not FrankaTargetSource.SUPERVISED_V94:
                raise FrankaNativeSupervisedSessionError(
                    "native supervised proxy rejects non-supervised targets"
                )
            with self._state_lock:
                if not self._run_started or self._run_finished:
                    raise FrankaNativeSupervisedSessionError(
                        "native child is not running"
                    )
                if not self._action_ready or not self._bootstrap_ready:
                    raise FrankaNativeSupervisedSessionError(
                        "native child has not completed measured-hold bootstrap"
                    )
                if self._stop_requested.is_set():
                    prior_failure = self._async_failure
                    if prior_failure is not None:
                        raise FrankaNativeSupervisedSessionError(
                            "native child already stopped after prior fault: "
                            f"{type(prior_failure).__name__}: {prior_failure}"
                        ) from prior_failure
                    raise FrankaNativeSupervisedSessionError(
                        "native child stop is already requested"
                    )
                expected = self._last_sent_sequence + 1
                if target.sequence != expected:
                    raise FrankaNativeSupervisedSessionError(
                        f"native target sequence gap: expected={expected}, "
                        f"actual={target.sequence}"
                    )
                if self._pending_target is not None:
                    raise FrankaNativeSupervisedSessionError(
                        "native child still has an unacknowledged target"
                    )
                if self.action_ledger.pending_sequence != target.sequence:
                    raise FrankaNativeSupervisedSessionError(
                        "parent ledger was not staged for the published target"
                    )
                self._pending_target = target
                self._last_sent_sequence = target.sequence
            self._send_encoded(
                lambda: self._codec.encode_target(target),
                operation="TARGET",
                target=target,
            )
        except BaseException as exc:
            self._latch_async_failure(exc)
            raise

    def _send_stop_once(self, reason: str) -> None:
        with self._state_lock:
            endpoint_exists = self._endpoint is not None
            if self._stop_packet_sent or not endpoint_exists:
                return
            self._stop_packet_sent = True
        try:
            self._send_encoded(
                lambda: self._codec.encode_stop(reason=reason),
                operation="STOP",
            )
        except BaseException as exc:
            # A failure may occur at the zero-wait writable/endpoint boundary
            # before _send_packet_once_locked() gets a chance to request the
            # independent terminate edge.  The one-shot STOP flag deliberately
            # prevents a retry, so cover every pre-send failure here as well.
            with self._state_lock:
                endpoint = self._endpoint
            if endpoint is not None:
                self._terminate_endpoint_best_effort(endpoint)
            self._latch_async_failure(exc)
            raise

    def request_clean_stop(self) -> None:
        """Idempotently request child stop; physical proof arrives separately."""

        self._stop_requested.set()
        with self._state_lock:
            endpoint_exists = self._endpoint is not None
            finished = self._run_finished
        if endpoint_exists and not finished:
            try:
                self._send_stop_once("parent clean stop requested")
            except BaseException:
                # Critical-channel failure must not remove the independent
                # SIGTERM stop edge.  Exit still does not count as STOP_PROOF.
                with self._state_lock:
                    endpoint = self._endpoint
                if endpoint is not None and endpoint.process.is_alive():
                    endpoint.process.terminate()

    def _receive_packet(
        self, timeout_s: float
    ) -> Optional[tuple[str, bytes]]:
        with self._state_lock:
            endpoint = self._endpoint
        if endpoint is None:
            raise FrankaNativeSupervisedSessionError("native child is not started")
        import select

        readable, _writable, _exceptional = select.select(
            [endpoint.critical_sock, endpoint.telemetry_sock],
            [],
            [],
            max(0.0, float(timeout_s)),
        )
        if not readable:
            return None
        # Critical replies always win if both descriptors are ready.  STATE is
        # explicitly lossy and must never delay ACK/FAULT/STOP_PROOF handling.
        channel = (
            "critical"
            if endpoint.critical_sock in readable
            else "telemetry"
        )
        sock = (
            endpoint.critical_sock
            if channel == "critical"
            else endpoint.telemetry_sock
        )
        # Read one byte beyond the frozen protocol maximum.  On both Unix
        # SEQPACKET and DGRAM sockets this consumes exactly one datagram; a
        # returned max+1 length proves the packet was oversized/truncated and
        # is therefore rejected just as MSG_TRUNC would be.
        packet = os.read(sock.fileno(), self._maximum_packet_bytes + 1)
        if len(packet) > self._maximum_packet_bytes:
            raise FrankaNativeSupervisedSessionError(
                "native child packet exceeded codec.max_packet_bytes"
            )
        if not packet:
            raise FrankaNativeSupervisedSessionError(
                f"native child closed its {channel} socket before STOP_PROOF"
            )
        return channel, bytes(packet)

    def _wait_for_critical_eof(
        self, endpoint: NativeChildEndpoint, *, deadline_s: float
    ) -> None:
        """Require the child to close critical fd after its one STOP_PROOF."""

        import select

        while self._monotonic() < deadline_s:
            readable, _writable, _exceptional = select.select(
                [endpoint.critical_sock],
                [],
                [],
                min(
                    self._event_poll_interval_s,
                    max(0.0, deadline_s - self._monotonic()),
                ),
            )
            if not readable:
                continue
            packet = os.read(
                endpoint.critical_sock.fileno(),
                self._maximum_packet_bytes + 1,
            )
            if packet:
                raise FrankaNativeSupervisedSessionError(
                    "critical packet arrived after terminal STOP_PROOF"
                )
            return
        raise FrankaNativeSupervisedSessionError(
            "native child did not close critical fd after STOP_PROOF"
        )

    def _handle_event(self, event: NativeChildEvent) -> Optional[NativeStopProofEvent]:
        if isinstance(event, NativeHelloEvent):
            with self._state_lock:
                if self._hello is not None:
                    raise FrankaNativeSupervisedSessionError("duplicate child HELLO")
                endpoint = self._endpoint
                if endpoint is None or endpoint.process.pid is None:
                    raise FrankaNativeSupervisedSessionError(
                        "child HELLO arrived without an exact spawned PID"
                    )
                if event.child_pid != endpoint.process.pid:
                    raise FrankaNativeSupervisedSessionError(
                        "HELLO process_id differs from the exact spawned child PID"
                    )
                self._hello = event
            return None

        with self._state_lock:
            hello = self._hello
        if hello is None:
            raise FrankaNativeSupervisedSessionError(
                "child event arrived before HELLO"
            )
        if isinstance(event, NativeIpcReadyEvent):
            with self._state_lock:
                if self._ipc_ready:
                    raise FrankaNativeSupervisedSessionError(
                        "duplicate child IPC_READY"
                    )
                self._ipc_ready = True
                self._ipc_ready_event = event
            return None
        if isinstance(event, NativeActionReadyEvent):
            with self._state_lock:
                if not self._ipc_ready:
                    raise FrankaNativeSupervisedSessionError(
                        "ACTION_READY arrived before IPC_READY"
                    )
                if self._action_ready:
                    raise FrankaNativeSupervisedSessionError(
                        "duplicate child ACTION_READY"
                    )
                if event.consecutive_healthy_cycles < 100:
                    raise FrankaNativeSupervisedSessionError(
                        "ACTION_READY proved fewer than 100 healthy cycles"
                    )
                self._action_ready = True
                self._bootstrap_ready = True
            return None
        if isinstance(event, NativeStateEvent):
            with self._state_lock:
                ipc_ready = self._ipc_ready
            if not ipc_ready:
                raise FrankaNativeSupervisedSessionError(
                    "STATE arrived before IPC_READY"
                )
            with self._state_lock:
                if event.cycle <= self._latest_child_control_cycle:
                    raise FrankaNativeSupervisedSessionError(
                        "STATE control cycle did not increase"
                    )
                self._latest_child_control_cycle = event.cycle
                self._pose_publication_count += 1
                pose_cycle = self._pose_publication_count
            # STATE datagrams may be dropped by design, whereas FrankaPoseRing
            # requires a dense local publication sequence.  Keep the raw child
            # control cycle separately and use the dense receive index here.
            sample = FrankaPoseSample(
                cycle=pose_cycle,
                realtime_s=event.realtime_s,
                monotonic_s=event.monotonic_s,
                q_rad=event.q_rad,
                dq_rad_s=event.dq_rad_s,
                T_base_eef=event.T_base_eef,
                status_flags=event.status_flags,
                controller_state29=(
                    None
                    if np.asarray(event.controller_state29).size == 0
                    else event.controller_state29
                ),
                shaper_q_d_rad=event.shaper_q_d_rad,
                shaper_dq_d_rad_s=event.shaper_dq_d_rad_s,
                shaper_ddq_d_rad_s2=event.shaper_ddq_d_rad_s2,
                held_q_cmd_rad=event.held_q_cmd_rad,
            )
            self.pose_ring.publish(sample)
            return None
        if isinstance(event, NativeAckEvent):
            with self._state_lock:
                target = self._pending_target
                expected = self._last_ack_sequence + 1
                if target is None:
                    raise FrankaNativeSupervisedSessionError(
                        "child ACK arrived without a pending target"
                    )
                if event.sequence != expected or event.sequence != target.sequence:
                    raise FrankaNativeSupervisedSessionError(
                        f"child ACK sequence mismatch: expected={expected}, "
                        f"actual={event.sequence}"
                    )
                command = target.closed_loop_command
                expected_produced_ns = int(
                    round(target.produced_monotonic_s * 1_000_000_000)
                )
                if (
                    command is None
                    or event.observation_sequence != command.sequence
                    or event.target_produced_monotonic_ns != expected_produced_ns
                    or not np.array_equal(
                        np.asarray(event.target_q_rad, dtype=np.float64),
                        target.target_q_rad,
                    )
                ):
                    raise FrankaNativeSupervisedSessionError(
                        "child ACK does not echo the exact published target"
                    )
                if (
                    event.control_cycle <= self._last_ack_control_cycle
                    or event.applied_monotonic_ns < event.target_produced_monotonic_ns
                    or event.target_age_at_write_ns
                    != event.applied_monotonic_ns
                    - event.target_produced_monotonic_ns
                    or event.target_age_at_write_ns > 50_000_000
                ):
                    raise FrankaNativeSupervisedSessionError(
                        "child ACK write time/cycle evidence is invalid"
                    )
            self.action_ledger.acknowledge(
                "franka",
                sequence=event.sequence,
                now_monotonic_s=self._monotonic(),
            )
            with self._state_lock:
                self._pending_target = None
                self._last_ack_sequence = event.sequence
                self._last_ack_control_cycle = event.control_cycle
            return None
        if isinstance(event, NativeFaultEvent):
            raise FrankaNativeSupervisedSessionError(
                f"native child FAULT code={event.code} sequence={event.sequence}: "
                f"{event.reason}"
            )
        if isinstance(event, NativeStopProofEvent):
            with self._state_lock:
                if self._last_telemetry is not None:
                    raise FrankaNativeSupervisedSessionError(
                        "duplicate child STOP_PROOF"
                    )
                telemetry = replace(
                    NativeFrankaTelemetry.from_stop_proof(
                        event, self._ipc_ready_event
                    ),
                    parent_heartbeat_count=self._parent_heartbeat_count,
                    maximum_parent_heartbeat_interval_s=(
                        self._maximum_parent_heartbeat_interval_s
                    ),
                )
                self._last_telemetry = telemetry
            if event.terminal_fault_code != 0:
                self._latch_async_failure(
                    FrankaNativeSupervisedSessionError(
                        "native STOP_PROOF records terminal fault "
                        f"{event.terminal_fault_code}: {event.fault_reason or event.detail}"
                    )
                )
            return event
        raise FrankaNativeSupervisedSessionError(
            f"codec returned unsupported event type: {type(event).__name__}"
        )

    def _drain_until_stop_proof(
        self,
        *,
        deadline_s: float,
        existing_failure: Optional[BaseException],
    ) -> tuple[Optional[NativeStopProofEvent], Optional[BaseException]]:
        proof: Optional[NativeStopProofEvent] = None
        failure = existing_failure
        while proof is None and self._monotonic() < deadline_s:
            with self._state_lock:
                endpoint = self._endpoint
            assert endpoint is not None
            if not endpoint.process.is_alive():
                # One final nonblocking read can collect a packet queued before
                # clean process exit.
                timeout = 0.0
            else:
                timeout = min(
                    self._event_poll_interval_s,
                    max(0.0, deadline_s - self._monotonic()),
                )
            try:
                received = self._receive_packet(timeout)
                if received is None:
                    if not endpoint.process.is_alive():
                        break
                    continue
                channel, packet = received
                if channel == "critical":
                    event = self._codec.decode_critical_packet(packet)
                    if isinstance(event, NativeStateEvent):
                        raise FrankaNativeSupervisedSessionError(
                            "STATE is forbidden on the critical channel"
                        )
                else:
                    event = self._codec.decode_state_packet(packet)
                    if not isinstance(event, NativeStateEvent):
                        raise FrankaNativeSupervisedSessionError(
                            "telemetry channel accepts STATE only"
                        )
                maybe_proof = self._handle_event(event)
                if maybe_proof is not None:
                    proof = maybe_proof
            except BaseException as exc:
                if failure is None:
                    failure = exc
                self._fail_pending_ledger(str(exc))
                if not endpoint.process.is_alive():
                    break
        return proof, failure

    def run(
        self,
        *,
        authorization: Any,
        preflight_token: Any,
        maximum_cycles: Optional[int] = None,
    ) -> NativeFrankaTelemetry:
        if maximum_cycles is not None:
            maximum_cycles = _positive_integer(maximum_cycles, "maximum_cycles")
        with self._state_lock:
            if self._run_started:
                raise FrankaNativeSupervisedSessionError(
                    "native supervised session is single-use"
                )
            self._run_started = True

        endpoint: Optional[NativeChildEndpoint] = None
        proof: Optional[NativeStopProofEvent] = None
        failure: Optional[BaseException] = None
        child_exitcode: Optional[int] = None
        armed = False
        next_heartbeat_s = self._monotonic() + self._heartbeat_interval_s
        startup_deadline_s = self._monotonic() + self._startup_timeout_s
        try:
            endpoint = self._launcher.start()
            if not isinstance(endpoint, NativeChildEndpoint):
                raise FrankaNativeSupervisedSessionError(
                    "native child launcher returned an invalid endpoint"
                )
            try:
                # Do not trust only the two built-in launchers: tests and
                # production integrations may inject another conforming
                # NativeChildLauncher.  This must precede publishing the
                # endpoint to any sender thread.
                self._force_parent_critical_nonblocking(endpoint)
            except BaseException:
                self._terminate_endpoint_best_effort(endpoint)
                endpoint.process.join(timeout=self._shutdown_timeout_s)
                endpoint.critical_sock.close()
                endpoint.telemetry_sock.close()
                close_process = getattr(endpoint.process, "close", None)
                if callable(close_process):
                    close_process()
                endpoint = None
                raise
            with self._state_lock:
                self._endpoint = endpoint

            while proof is None:
                with self._state_lock:
                    async_failure = self._async_failure
                    hello = self._hello
                    ready = self._ipc_ready
                if failure is None and async_failure is not None:
                    failure = async_failure
                    self._stop_requested.set()
                now = self._monotonic()
                if hello is None and now >= startup_deadline_s:
                    raise FrankaNativeSupervisedSessionError(
                        "native child HELLO timeout"
                    )
                if hello is not None and not armed and not self._stop_requested.is_set():
                    self._send_encoded(
                        lambda: self._codec.encode_arm(
                            run_id=self.run_id,
                            authorization=authorization,
                            preflight_token=preflight_token,
                            envelope=self.envelope,
                            maximum_cycles=maximum_cycles,
                            controller_mode=self.controller_mode,
                        ),
                        operation="ARM",
                    )
                    armed = True
                    next_heartbeat_s = now + self._heartbeat_interval_s
                if self._stop_requested.is_set():
                    self._send_stop_once("native session terminal cleanup")
                elif armed and now >= next_heartbeat_s:
                    self._send_encoded(
                        lambda: self._codec.encode_heartbeat(
                            monotonic_ns=int(self._monotonic() * 1_000_000_000)
                        ),
                        operation="HEARTBEAT",
                    )
                    next_heartbeat_s = now + self._heartbeat_interval_s

                timeout = self._event_poll_interval_s
                if armed and not self._stop_requested.is_set():
                    timeout = min(timeout, max(0.0, next_heartbeat_s - now))
                if not endpoint.process.is_alive():
                    # Drain a STOP_PROOF already queued before clean child exit;
                    # exit itself remains insufficient evidence.
                    timeout = 0.0
                received = self._receive_packet(timeout)
                if received is None:
                    if not endpoint.process.is_alive():
                        raise FrankaNativeSupervisedSessionError(
                            "native child exited before STOP_PROOF"
                        )
                    continue
                channel, packet = received
                if channel == "critical":
                    event = self._codec.decode_critical_packet(packet)
                    if isinstance(event, NativeStateEvent):
                        raise FrankaNativeSupervisedSessionError(
                            "STATE is forbidden on the critical channel"
                        )
                else:
                    event = self._codec.decode_state_packet(packet)
                    if not isinstance(event, NativeStateEvent):
                        raise FrankaNativeSupervisedSessionError(
                            "telemetry channel accepts STATE only"
                        )
                maybe_proof = self._handle_event(event)
                if maybe_proof is not None:
                    proof = maybe_proof
                with self._state_lock:
                    ready = self._ipc_ready
                if armed and not ready and self._monotonic() >= startup_deadline_s:
                    raise FrankaNativeSupervisedSessionError(
                        "native child READY timeout"
                    )
        except BaseException as exc:
            failure = exc if failure is None else failure
            self._latch_async_failure(failure)
        finally:
            self._stop_requested.set()
            if endpoint is not None:
                try:
                    self._send_stop_once("native session terminal cleanup")
                except BaseException as exc:
                    if failure is None:
                        failure = exc
                    if endpoint.process.is_alive():
                        endpoint.process.terminate()
                if proof is None:
                    proof, failure = self._drain_until_stop_proof(
                        deadline_s=self._monotonic() + self._shutdown_timeout_s,
                        existing_failure=failure,
                    )
                if proof is not None:
                    try:
                        self._wait_for_critical_eof(
                            endpoint,
                            deadline_s=self._monotonic()
                            + self._shutdown_timeout_s,
                        )
                    except BaseException as exc:
                        if failure is None:
                            failure = exc
                endpoint.process.join(timeout=self._shutdown_timeout_s)
                if endpoint.process.is_alive():
                    endpoint.process.terminate()
                    endpoint.process.join(timeout=self._shutdown_timeout_s)
                    if failure is None:
                        failure = FrankaNativeSupervisedSessionError(
                            "native child did not exit after STOP and was terminated"
                        )
                child_exitcode = endpoint.process.exitcode
                if (
                    proof is not None
                    and proof.stop_reason == 1
                    and proof.terminal_fault_code == 0
                    and proof.parent_stop_verified
                    and child_exitcode != 0
                    and failure is None
                ):
                    failure = FrankaNativeSupervisedSessionError(
                        "verified STOP_PROOF was followed by nonzero child exit: "
                        f"exitcode={child_exitcode}"
                    )
                try:
                    endpoint.critical_sock.close()
                    endpoint.telemetry_sock.close()
                except BaseException as exc:
                    if failure is None:
                        failure = exc
                close_process = getattr(endpoint.process, "close", None)
                if callable(close_process):
                    try:
                        close_process()
                    except BaseException as exc:
                        if failure is None:
                            failure = exc
            with self._state_lock:
                self._run_finished = True

        telemetry = self.last_telemetry
        if proof is None:
            if failure is None:
                failure = FrankaNativeSupervisedSessionError(
                    "native child returned no STOP_PROOF"
                )
            with self._state_lock:
                telemetry = replace(
                    NativeFrankaTelemetry.failed_without_proof(str(failure)),
                    parent_heartbeat_count=self._parent_heartbeat_count,
                    maximum_parent_heartbeat_interval_s=(
                        self._maximum_parent_heartbeat_interval_s
                    ),
                )
                self._last_telemetry = telemetry
        elif not proof.parent_stop_verified:
            if failure is None:
                failure = FrankaNativeSupervisedSessionError(
                    "native child STOP_PROOF did not verify requested physical "
                    "stop: "
                    f"finish_attempted={proof.finish_attempted}, "
                    f"finish_succeeded={proof.finish_succeeded}, "
                    f"robot_stop_attempted={proof.robot_stop_attempted}, "
                    f"robot_stop_succeeded={proof.robot_stop_succeeded}, "
                    f"idle_dq_verified={proof.idle_dq_verified}, "
                    f"stop_samples={proof.stop_verification_samples}, "
                    f"consecutive_idle={proof.stop_consecutive_idle_samples}, "
                    f"final_robot_mode={proof.final_robot_mode}, "
                    f"maximum_stop_dq_rad_s={proof.maximum_stop_dq_rad_s}, "
                    f"detail={proof.detail!r}, "
                    f"active_handle_released={proof.active_handle_released}, "
                    f"robot_backend_released={proof.robot_backend_released}, "
                    f"child_exitcode={child_exitcode}"
                )
        if proof is not None and proof.terminal_fault_code != 0 and failure is None:
            failure = FrankaNativeSupervisedSessionError(
                "native child terminated with fault code "
                f"{proof.terminal_fault_code}: {proof.fault_reason or proof.detail}"
            )
        if proof is not None and proof.franka_ack_count != self._last_ack_sequence:
            if failure is None:
                failure = FrankaNativeSupervisedSessionError(
                    "native STOP_PROOF ACK count differs from parent ledger bridge"
                )
        if failure is not None:
            if isinstance(failure, (KeyboardInterrupt, SystemExit)):
                raise failure
            raise FrankaNativeSupervisedSessionError(
                f"native supervised Franka session failed: "
                f"{type(failure).__name__}: {failure}"
            ) from failure
        assert telemetry is not None
        return telemetry


__all__ = [
    "FrankaNativeSupervisedSessionError",
    "FrankaNativeSupervisedSessionProxy",
    "NativeAckEvent",
    "NativeChildEndpoint",
    "NativeChildEvent",
    "NativeChildLauncher",
    "NativeChildProcess",
    "NativeFaultEvent",
    "NativeFrankaTargetSampleHold",
    "NativeFrankaTelemetry",
    "NativeHelloEvent",
    "NativeHelloExpectation",
    "NativeHealthyEvent",
    "NativeIpcReadyEvent",
    "NativeActionReadyEvent",
    "NativeStateEvent",
    "NativeStopVerificationSample",
    "NativeStopProofEvent",
    "NativeSupervisedPacketCodec",
    "PopenChildProcessHandle",
    "PopenSeqpacketChildLauncher",
    "SpawnedSeqpacketChildLauncher",
    "V94NativeMessageKind",
    "V94NativeControllerMode",
    "V94NativePODCodec",
    "V94_NATIVE_HEADER_BYTES",
    "V94_NATIVE_MAX_PACKET_BYTES",
    "V94_NATIVE_PAYLOAD_BYTES",
    "V94_NATIVE_PROTOCOL_MAGIC",
    "V94_NATIVE_PROTOCOL_VERSION",
]
