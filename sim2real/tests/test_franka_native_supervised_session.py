"""Hardware-inert tests for the process-isolated Franka parent proxy."""

from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import uuid
import zlib

import numpy as np
import pytest

from sim2real.closed_loop_core import (
    ClosedLoopCommand,
    ExecutedActionLedger,
    MotionAuthorization,
)
from sim2real.deployment.bundle import DeployBundle
from robot_control.franka.native_session import (
    FrankaNativeSupervisedSessionError,
    FrankaNativeSupervisedSessionProxy,
    NativeAckEvent,
    NativeActionReadyEvent,
    NativeFaultEvent,
    NativeHelloEvent,
    NativeIpcReadyEvent,
    NativeHelloExpectation,
    NativeStateEvent,
    NativeStopProofEvent,
    NativeStopVerificationSample,
    PopenSeqpacketChildLauncher,
    V94NativeControllerMode,
    V94NativeMessageKind,
    V94NativePODCodec,
    V94_NATIVE_HEADER_BYTES,
    V94_NATIVE_MAX_PACKET_BYTES,
    V94_NATIVE_PAYLOAD_BYTES,
    V94_NATIVE_PROTOCOL_MAGIC,
    V94_NATIVE_PROTOCOL_VERSION,
)
from robot_control.franka.session import (
    FrankaJointTarget,
    FrankaTargetSource,
    _issue_supervised_franka_preflight_token,
    load_experimental_supervised_franka_envelope,
)
from sim2real.contracts.v94 import V94Contract
from sim2real.contracts.v94 import INITIAL_PREVIOUS_ACTION13


_NONCE = bytes.fromhex("00112233445566778899aabbccddeeff")
_Q7 = struct.Struct("<7d")
_HEADER = struct.Struct("<IHHIIQQ16sII")
_HELLO = struct.Struct("<IIII32s20s32s8d")
_IPC_READY = struct.Struct("<15dQ6I")
_ACTION_READY = struct.Struct("<IIQQ2d14d3QII")
_STATE = struct.Struct("<6QIId37dII3QII3Q29f28d")
_ACK = struct.Struct("<7QII21dQd")
_STOP_PREFIX = struct.Struct("<II8B5Q4Id4Q")
_STOP_SAMPLE = struct.Struct("<QII7d")
_EXPECTED_LIBFRANKA = bytes.fromhex("11" * 32)
_EXPECTED_COMMIT = bytes.fromhex("22" * 20)
_EXPECTED_BUILD = bytes.fromhex("33" * 32)
_CPP_GOLDEN_TARGET_HEX = (
    "5639344604000300500000000000000008070605040302011817161514131211"
    "000102030405060708090a0b0c0d0e0f35311ec4000000002827262524232221"
    "383736353433323148474645444342410000000000000000000000000000f03f"
    "000000000000f0bf000000000000e03f000000000000e0bf0000000000000a40"
    "00000000000006c0"
)


def test_native_codec_target_count_boundary_is_seven_hundred_twenty() -> None:
    expectation = NativeHelloExpectation(
        state_decimation=16,
        safety_limits_schema=3,
        libfranka_sha256=_EXPECTED_LIBFRANKA,
        libfranka_source_commit=_EXPECTED_COMMIT,
        producer_build_sha256=_EXPECTED_BUILD,
    )
    V94NativePODCodec(
        hello_expectation=expectation,
        reference_q_rad=np.zeros(7, dtype=np.float64),
        maximum_target_count=720,
    )
    with pytest.raises(ValueError, match="supervised limit 720"):
        V94NativePODCodec(
            hello_expectation=expectation,
            reference_q_rad=np.zeros(7, dtype=np.float64),
            maximum_target_count=721,
        )
_FAKE_CHILD = r"""
import os
import socket
import struct
import sys
import time

critical = int(sys.argv[1])
telemetry = int(sys.argv[2])
mode = sys.argv[3]
nonce = bytes.fromhex("00112233445566778899aabbccddeeff")
q7 = struct.Struct("<7d")
os.write(critical, struct.pack("<cI16s", b"H", os.getpid(), nonce))
ack_count = 0

while True:
    packet = os.read(critical, 4096)
    if not packet:
        break
    kind = packet[:1]
    if kind == b"a":
        os.write(critical, b"I")
        base_mono = time.monotonic_ns()
        base_real = time.time_ns()
        os.write(telemetry, struct.pack("<cQQQ", b"S", 10, base_mono, base_real))
        os.write(
            telemetry,
            struct.pack("<cQQQ", b"S", 25, base_mono + 110_000_000,
                        base_real + 110_000_000)
        )
        os.write(critical, b"A")
    elif kind == b"h":
        pass
    elif kind == b"t":
        sequence = struct.unpack_from("<Q", packet, 1)[0]
        target = q7.unpack_from(packet, 9)
        if mode == "wrong_ack":
            sequence += 1
        os.write(critical, b"K" + struct.pack("<Q", sequence) + q7.pack(*target))
        if mode != "wrong_ack":
            ack_count += 1
    elif kind == b"s":
        os.write(critical, b"P" + struct.pack("<Q", ack_count))
        break

os.close(critical)
os.close(telemetry)
"""


def _native_packet(
    kind: V94NativeMessageKind,
    *,
    sequence: int,
    monotonic_ns: int,
    payload: bytes,
    nonce: bytes = _NONCE,
) -> bytes:
    assert len(payload) == V94_NATIVE_PAYLOAD_BYTES[kind]
    header = _HEADER.pack(
        V94_NATIVE_PROTOCOL_MAGIC,
        V94_NATIVE_PROTOCOL_VERSION,
        int(kind),
        len(payload),
        0,
        sequence,
        monotonic_ns,
        nonce,
        0,
        0,
    )
    packet = bytearray(header + payload)
    struct.pack_into("<I", packet, 48, zlib.crc32(packet) & 0xFFFFFFFF)
    return bytes(packet)


def _rewrite_packet(packet: bytes, *, kind: int | None = None) -> bytes:
    value = bytearray(packet)
    if kind is not None:
        struct.pack_into("<H", value, 6, kind)
    value[48:52] = bytes(4)
    struct.pack_into("<I", value, 48, zlib.crc32(value) & 0xFFFFFFFF)
    return bytes(value)


def _real_codec_fixture(*, expected_servo_cpu: int | None = None):
    workspace = Path(__file__).resolve().parents[2]
    profile_path = workspace / "dexgrasp/configs/fr3_rh56_v94_commissioning.json"
    contract = V94Contract.from_bundle(
        DeployBundle(workspace / "data/test_fixtures/sim2real/deploy.zip")
    )
    # Production admission uses the bundle's float32 q_home.  Promote that
    # exact value to the ARM wire's float64 fields; reading profile JSON as
    # float64 would silently exercise different values.
    q_home = np.asarray(contract.q_home_rad, dtype=np.float64)
    envelope = load_experimental_supervised_franka_envelope(profile_path)
    expectation = NativeHelloExpectation(
        state_decimation=16,
        safety_limits_schema=3,
        libfranka_sha256=_EXPECTED_LIBFRANKA,
        libfranka_source_commit=_EXPECTED_COMMIT,
        producer_build_sha256=_EXPECTED_BUILD,
    )
    codec = V94NativePODCodec(
        hello_expectation=expectation,
        reference_q_rad=q_home,
        maximum_target_count=3,
        monotonic_ns=lambda: 10_000_000_000,
        expected_servo_cpu=expected_servo_cpu,
    )
    authorization = MotionAuthorization(
        run_id="native-codec",
        authorization_id="12345678-1234-5678-1234-567812345678",
        issued_monotonic_s=5.0,
        expires_monotonic_s=20.0,
    )
    token = _issue_supervised_franka_preflight_token(
        envelope=envelope,
        run_id="native-codec",
        confirmed_permit_sha256="ab" * 32,
        issued_monotonic_s=5.0,
        expires_monotonic_s=20.0,
    )
    return codec, envelope, authorization, token, q_home


def _hello_payload(*, process_id: int = 1234) -> bytes:
    return _HELLO.pack(
        process_id,
        16,
        V94_NATIVE_PROTOCOL_VERSION,
        3,
        _EXPECTED_LIBFRANKA,
        _EXPECTED_COMMIT,
        _EXPECTED_BUILD,
        0.50,
        5.0,
        250.0,
        0.01,
        0.020,
        1.21,
        0.01,
        0.0008,
    )


def _prime_real_codec(
    codec,
    envelope,
    authorization,
    token,
    *,
    controller_mode=V94NativeControllerMode.LEGACY,
    action_ready_period_ms: int = 2,
):
    hello = codec.decode_critical_packet(
        _native_packet(
            V94NativeMessageKind.HELLO,
            sequence=1,
            monotonic_ns=1,
            payload=_hello_payload(),
        )
    )
    assert isinstance(hello, NativeHelloEvent)
    arm = codec.encode_arm(
        run_id="native-codec",
        authorization=authorization,
        preflight_token=token,
        envelope=envelope,
        maximum_cycles=None,
        controller_mode=controller_mode,
    )
    ipc_payload = _IPC_READY.pack(
        *np.zeros(14),
        0.0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
    )
    ipc = codec.decode_critical_packet(
        _native_packet(
            V94NativeMessageKind.IPC_READY,
            sequence=2,
            monotonic_ns=2,
            payload=ipc_payload,
        )
    )
    assert isinstance(ipc, NativeIpcReadyEvent)
    action_payload = _ACTION_READY.pack(
        100,
        action_ready_period_ms,
        100,
        500_000,
        0.99,
        0.995,
        *np.zeros(14),
        101,
        100,
        100,
        0,
        0,
    )
    action = codec.decode_critical_packet(
        _native_packet(
            V94NativeMessageKind.ACTION_READY,
            sequence=3,
            monotonic_ns=3,
            payload=action_payload,
        )
    )
    assert isinstance(action, NativeActionReadyEvent)
    return arm


def test_python_codec_accepts_21ms_returned_fci_period_and_rejects_22ms() -> None:
    codec, envelope, authorization, token, _q_home = _real_codec_fixture()
    _prime_real_codec(
        codec,
        envelope,
        authorization,
        token,
        action_ready_period_ms=21,
    )

    codec, envelope, authorization, token, _q_home = _real_codec_fixture()
    with pytest.raises(
        FrankaNativeSupervisedSessionError,
        match="100-cycle health gate",
    ):
        _prime_real_codec(
            codec,
            envelope,
            authorization,
            token,
            action_ready_period_ms=22,
        )


def test_ipc_ready_requires_exact_fifo_priority_and_single_servo_cpu() -> None:
    codec, envelope, authorization, token, _q_home = _real_codec_fixture(
        expected_servo_cpu=11
    )
    codec.decode_critical_packet(
        _native_packet(
            V94NativeMessageKind.HELLO,
            sequence=1,
            monotonic_ns=1,
            payload=_hello_payload(),
        )
    )
    codec.encode_arm(
        run_id="native-codec",
        authorization=authorization,
        preflight_token=token,
        envelope=envelope,
        maximum_cycles=None,
    )
    priority = os.sched_get_priority_max(os.SCHED_FIFO)
    payload = _IPC_READY.pack(
        *np.zeros(14),
        0.0,
        1,
        1,
        os.SCHED_FIFO,
        priority,
        11,
        1,
        0,
    )
    event = codec.decode_critical_packet(
        _native_packet(
            V94NativeMessageKind.IPC_READY,
            sequence=2,
            monotonic_ns=2,
            payload=payload,
        )
    )
    assert isinstance(event, NativeIpcReadyEvent)
    assert event.realtime_cpu == 11
    assert event.realtime_affinity_cpu_count == 1


@pytest.mark.parametrize(
    ("policy", "priority_offset", "cpu", "affinity_count"),
    [
        (0, 0, 11, 1),
        (os.SCHED_FIFO, -1, 11, 1),
        (os.SCHED_FIFO, 0, 10, 1),
        (os.SCHED_FIFO, 0, 11, 2),
    ],
)
def test_ipc_ready_rejects_inexact_realtime_proof(
    policy, priority_offset, cpu, affinity_count
) -> None:
    codec, envelope, authorization, token, _q_home = _real_codec_fixture(
        expected_servo_cpu=11
    )
    codec.decode_critical_packet(
        _native_packet(
            V94NativeMessageKind.HELLO,
            sequence=1,
            monotonic_ns=1,
            payload=_hello_payload(),
        )
    )
    codec.encode_arm(
        run_id="native-codec",
        authorization=authorization,
        preflight_token=token,
        envelope=envelope,
        maximum_cycles=None,
    )
    payload = _IPC_READY.pack(
        *np.zeros(14),
        0.0,
        1,
        1,
        policy,
        os.sched_get_priority_max(os.SCHED_FIFO) + priority_offset,
        cpu,
        affinity_count,
        0,
    )
    with pytest.raises(
        FrankaNativeSupervisedSessionError,
        match="realtime scheduler/CPU proof differs",
    ):
        codec.decode_critical_packet(
            _native_packet(
                V94NativeMessageKind.IPC_READY,
                sequence=2,
                monotonic_ns=2,
                payload=payload,
            )
        )


def _healthy_stop_payload(*, corrupt_last_mode: bool = False) -> bytes:
    prefix = _STOP_PREFIX.pack(
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        120,
        1,
        120,
        120,
        1,
        3,
        3,
        1,
        2,
        0.0,
        500_000,
        200,
        203,
        0,
    )
    samples = b"".join(
        _STOP_SAMPLE.pack(
            201 + index,
            2 if corrupt_last_mode and index == 2 else 1,
            0,
            *np.zeros(7),
        )
        for index in range(3)
    )
    return prefix + samples + bytes(160)


def _fault_stop_payload() -> bytes:
    prefix = _STOP_PREFIX.pack(
        3,
        14,
        0,
        0,
        1,
        0,  # Robot.stop return failure is retained as run-fault evidence.
        1,
        1,
        1,
        1,
        120,
        0,
        120,
        119,
        0,
        3,
        3,
        1,
        2,
        0.0,
        500_000,
        200,
        203,
        0,
    )
    samples = b"".join(
        _STOP_SAMPLE.pack(201 + index, 1, 0, *np.zeros(7))
        for index in range(3)
    )
    detail = b"read-to-write deadline" + bytes(160 - len(b"read-to-write deadline"))
    return prefix + samples + detail


def _fault_comm_only_stop_payload(*, extra_status_flag: int = 0) -> bytes:
    status_flags = (1 << 0) | (1 << 5) | (1 << 6) | extra_status_flag
    prefix = _STOP_PREFIX.pack(
        3,
        11,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        50,
        0,
        50,
        49,
        0,
        3,
        3,
        5,
        13,
        0.0,
        500_000,
        200,
        203,
        0,
    )
    samples = b"".join(
        _STOP_SAMPLE.pack(201 + index, 5, status_flags, *np.zeros(7))
        for index in range(3)
    )
    detail = b"active control period outside FCI recoverable range"
    return prefix + samples + detail + bytes(160 - len(detail))


def _requested_direct_stop_payload(*, mode: int = 5) -> bytes:
    status_flags = ((1 << 0) | (1 << 5) | (1 << 6)) if mode == 5 else 0
    prefix = _STOP_PREFIX.pack(
        1,
        0,
        0,  # Streaming STOP must not claim a graceful MotionFinished attempt.
        0,
        1,  # Active handle release is followed by explicit Robot.stop.
        1,
        1,
        1,
        0,
        1,
        120,
        1,
        120,
        119,
        1,
        3,
        3,
        mode,
        2,
        0.0,
        500_000,
        200,
        203,
        0,
    )
    samples = b"".join(
        _STOP_SAMPLE.pack(201 + index, mode, status_flags, *np.zeros(7))
        for index in range(3)
    )
    detail = b"parent requested supervised stop"
    return prefix + samples + detail + bytes(160 - len(detail))


def _finish_fallback_stop_payload() -> bytes:
    prefix = _STOP_PREFIX.pack(
        1,
        0,
        1,  # MotionFinished was attempted.
        0,  # It was rejected while the shaped command was still in flight.
        1,  # Explicit Robot.stop fallback was attempted after handle release.
        1,
        1,
        1,
        0,
        1,
        120,
        1,
        120,
        119,
        1,
        3,
        3,
        1,
        2,
        0.0,
        500_000,
        200,
        203,
        0,
    )
    samples = b"".join(
        _STOP_SAMPLE.pack(201 + index, 1, 0, *np.zeros(7))
        for index in range(3)
    )
    return prefix + samples + bytes(160)


class _FakePODCodec:
    """Tiny test-only codec; it is intentionally not the production ABI."""

    max_packet_bytes = 4096

    def __init__(self) -> None:
        self._produced_ns: dict[int, int] = {}

    def encode_arm(self, **_kwargs) -> bytes:
        return b"a"

    def encode_heartbeat(self, *, monotonic_ns: int) -> bytes:
        return b"h" + struct.pack("<Q", monotonic_ns)

    def encode_target(self, target: FrankaJointTarget) -> bytes:
        self._produced_ns[target.sequence] = int(
            round(target.produced_monotonic_s * 1_000_000_000)
        )
        return (
            b"t"
            + struct.pack("<Q", target.sequence)
            + _Q7.pack(*target.target_q_rad)
        )

    def encode_stop(self, *, reason: str) -> bytes:
        assert reason
        return b"s"

    def decode_critical_packet(self, packet: bytes):
        kind = packet[:1]
        if kind == b"H":
            _kind, pid, nonce = struct.unpack("<cI16s", packet)
            return NativeHelloEvent(child_pid=pid, session_nonce=nonce)
        if kind == b"I":
            return NativeIpcReadyEvent(
                measured_q_rad=np.zeros(7),
                measured_dq_rad_s=np.zeros(7),
                q_home_linf_error_rad=0.0,
                robot_time_ms=1,
                static_provenance_verified=True,
            )
        if kind == b"A":
            return NativeActionReadyEvent(
                consecutive_healthy_cycles=100,
                maximum_control_period_ms=1,
                healthy_hold_robot_time_ms=100,
                maximum_read_to_write_ns=300_000,
                minimum_control_command_success_rate=1.0,
                latest_control_command_success_rate=1.0,
                measured_q_rad=np.zeros(7),
                measured_dq_rad_s=np.zeros(7),
                robot_time_ms=101,
                active_read_count=100,
                active_write_count=100,
                status_flags=0,
                cumulative_missed_robot_states=0,
            )
        if kind == b"K":
            sequence = struct.unpack_from("<Q", packet, 1)[0]
            target = np.asarray(_Q7.unpack_from(packet, 9))
            produced_ns = self._produced_ns.get(sequence, 1)
            return NativeAckEvent(
                sequence=sequence,
                observation_sequence=sequence,
                target_produced_monotonic_ns=produced_ns,
                target_age_at_write_ns=1_000_000,
                control_cycle=101,
                robot_time_ms=101,
                applied_monotonic_ns=produced_ns + 1_000_000,
                control_period_ms=1,
                target_q_rad=target,
                commanded_q_rad=target,
                measured_q_rad=target,
                read_to_write_ns=300_000,
                maximum_tracking_error_rad=0.0,
            )
        if kind == b"F":
            return NativeFaultEvent(code=1, reason="synthetic fault")
        if kind == b"P":
            ack_count = struct.unpack_from("<Q", packet, 1)[0]
            samples = tuple(
                NativeStopVerificationSample(
                    robot_time_ms=200 + index,
                    robot_mode=1,
                    status_flags=0,
                    measured_dq_rad_s=np.zeros(7),
                )
                for index in range(3)
            )
            return NativeStopProofEvent(
                stop_reason=1,
                terminal_fault_code=0,
                finish_attempted=True,
                finish_succeeded=True,
                robot_stop_attempted=False,
                robot_stop_succeeded=False,
                idle_dq_verified=True,
                active_handle_released=True,
                fault_reply_delivered=False,
                robot_backend_released=True,
                control_cycles=110,
                last_target_sequence=ack_count,
                active_read_count=110,
                active_write_count=110,
                franka_ack_count=ack_count,
                stop_verification_samples=3,
                stop_consecutive_idle_samples=3,
                final_robot_mode=1,
                maximum_control_period_ms=1,
                maximum_stop_dq_rad_s=0.0,
                maximum_read_to_write_ns=300_000,
                pre_stop_robot_time_ms=199,
                final_robot_time_ms=202,
                telemetry_drop_count=0,
                verified_samples=samples,
                detail="",
                parent_stop_verified=True,
                maximum_control_period_s=0.001,
            )
        raise ValueError("malformed fake critical packet")

    def decode_state_packet(self, packet: bytes) -> NativeStateEvent:
        kind, cycle, monotonic_ns, realtime_ns = struct.unpack("<cQQQ", packet)
        if kind != b"S":
            raise ValueError("fake telemetry packet is not STATE")
        return NativeStateEvent(
            cycle=cycle,
            realtime_s=realtime_ns / 1_000_000_000,
            monotonic_s=monotonic_ns / 1_000_000_000,
            q_rad=np.zeros(7),
            dq_rad_s=np.zeros(7),
            T_base_eef=np.eye(4),
        )


def _launcher(mode: str = "normal") -> PopenSeqpacketChildLauncher:
    return PopenSeqpacketChildLauncher(
        lambda critical_fd, telemetry_fd: (
            sys.executable,
            "-c",
            _FAKE_CHILD,
            str(critical_fd),
            str(telemetry_fd),
            mode,
        )
    )


def _proxy(mode: str = "normal"):
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)
    proxy = FrankaNativeSupervisedSessionProxy(
        run_id="native-fake",
        envelope=SimpleNamespace(pose_ring_capacity=32),
        action_ledger=ledger,
        codec=_FakePODCodec(),
        launcher=_launcher(mode),
        heartbeat_interval_s=0.02,
        event_poll_interval_s=0.002,
        startup_timeout_s=1.0,
        shutdown_timeout_s=1.0,
    )
    return proxy, ledger


def _wait_until(predicate, *, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError("timed out waiting for fake child")


def _command(
    ledger: ExecutedActionLedger,
    *,
    produced_monotonic_s: float | None = None,
) -> ClosedLoopCommand:
    del ledger
    action = np.zeros(13, dtype=np.float32)
    target = np.zeros(7, dtype=np.float64)
    target[0] = 0.001
    return ClosedLoopCommand(
        sequence=1,
        produced_monotonic_s=(
            time.monotonic()
            if produced_monotonic_s is None
            else produced_monotonic_s
        ),
        observation_realtime_s=time.time(),
        previous_executed_action13_used=INITIAL_PREVIOUS_ACTION13,
        raw_policy_action13=action,
        executed_policy_action13=action,
        franka_target_q_rad=target,
        rh56_angle_set_register_order=np.full(6, 1000),
    )


def test_popen_launcher_construction_is_inert_and_uses_two_socket_types() -> None:
    launcher = _launcher()

    assert launcher.construction_is_inert is True
    endpoint = launcher.start()
    try:
        assert int(endpoint.critical_sock.type) & 0xF == socket.SOCK_SEQPACKET
        assert int(endpoint.telemetry_sock.type) & 0xF == socket.SOCK_DGRAM
        assert endpoint.critical_sock.getblocking() is False
        assert os.get_blocking(endpoint.critical_sock.fileno()) is False
        assert endpoint.process.pid is not None
        assert endpoint.process.is_alive()
        endpoint.process.terminate()
        endpoint.process.join(timeout=1.0)
    finally:
        endpoint.critical_sock.close()
        endpoint.telemetry_sock.close()
        endpoint.process.close()


def test_proxy_forces_an_injected_launcher_critical_fd_nonblocking() -> None:
    delegate = _launcher()

    class BlockingInjectedLauncher:
        construction_is_inert = True

        def __init__(self) -> None:
            self.endpoint = None

        def start(self):
            endpoint = delegate.start()
            endpoint.critical_sock.setblocking(True)
            self.endpoint = endpoint
            return endpoint

    launcher = BlockingInjectedLauncher()
    ledger = ExecutedActionLedger(maximum_commit_latency_s=1.0)
    proxy = FrankaNativeSupervisedSessionProxy(
        run_id="native-injected-blocking",
        envelope=SimpleNamespace(pose_ring_capacity=32),
        action_ledger=ledger,
        codec=_FakePODCodec(),
        launcher=launcher,
        heartbeat_interval_s=0.02,
        event_poll_interval_s=0.002,
        startup_timeout_s=1.0,
        shutdown_timeout_s=1.0,
    )
    errors: list[BaseException] = []

    def owner() -> None:
        try:
            proxy.run(authorization=object(), preflight_token=object())
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=owner)
    thread.start()
    _wait_until(lambda: proxy.c2_bootstrap_ready)
    assert launcher.endpoint is not None
    assert launcher.endpoint.critical_sock.getblocking() is False
    assert os.get_blocking(launcher.endpoint.critical_sock.fileno()) is False
    proxy.request_clean_stop()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert errors == []


def test_target_header_is_encoded_only_after_held_state_lock_is_released() -> None:
    proxy, ledger = _proxy()
    now = [10.0]
    proxy._monotonic = lambda: now[0]
    target = FrankaJointTarget.from_closed_loop_command(
        _command(ledger, produced_monotonic_s=10.0),
        source=FrankaTargetSource.SUPERVISED_V94,
    )
    endpoint = object()
    proxy._endpoint = endpoint
    proxy._require_critical_writable_before_encode = (
        lambda _endpoint, *, operation: None
    )
    encoded_at: list[float] = []
    sent: list[bytes] = []
    proxy._send_packet_once_locked = (
        lambda _endpoint, packet, *, operation: sent.append(packet)
    )
    lookup_entered = threading.Event()
    original_lookup = proxy._critical_endpoint_before_encode

    def notifying_lookup(*, operation: str):
        lookup_entered.set()
        return original_lookup(operation=operation)

    proxy._critical_endpoint_before_encode = notifying_lookup
    errors: list[BaseException] = []

    def sender() -> None:
        try:
            proxy._send_encoded(
                lambda: encoded_at.append(now[0]) or b"target",
                operation="TARGET",
                target=target,
            )
        except BaseException as exc:
            errors.append(exc)

    proxy._state_lock.acquire()
    try:
        thread = threading.Thread(target=sender)
        thread.start()
        assert lookup_entered.wait(timeout=1.0)
        now[0] = 10.020
    finally:
        proxy._state_lock.release()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert errors == []
    assert encoded_at == [10.020]
    assert sent == [b"target"]


def test_target_stale_after_held_state_lock_is_not_encoded_or_sent() -> None:
    proxy, ledger = _proxy()
    now = [10.0]
    proxy._monotonic = lambda: now[0]
    target = FrankaJointTarget.from_closed_loop_command(
        _command(ledger, produced_monotonic_s=10.0),
        source=FrankaTargetSource.SUPERVISED_V94,
    )
    proxy._endpoint = object()
    proxy._require_critical_writable_before_encode = (
        lambda _endpoint, *, operation: None
    )
    encoded: list[bool] = []
    sent: list[bytes] = []
    proxy._send_packet_once_locked = (
        lambda _endpoint, packet, *, operation: sent.append(packet)
    )
    lookup_entered = threading.Event()
    original_lookup = proxy._critical_endpoint_before_encode

    def notifying_lookup(*, operation: str):
        lookup_entered.set()
        return original_lookup(operation=operation)

    proxy._critical_endpoint_before_encode = notifying_lookup
    errors: list[BaseException] = []

    def sender() -> None:
        try:
            proxy._send_encoded(
                lambda: encoded.append(True) or b"target",
                operation="TARGET",
                target=target,
            )
        except BaseException as exc:
            errors.append(exc)

    proxy._state_lock.acquire()
    try:
        thread = threading.Thread(target=sender)
        thread.start()
        assert lookup_entered.wait(timeout=1.0)
        now[0] = 10.106
    finally:
        proxy._state_lock.release()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert "expired before native encode/send boundary" in str(errors[0])
    assert encoded == []
    assert sent == []


def test_target_that_expires_during_encode_is_never_sent() -> None:
    proxy, ledger = _proxy()
    now = [10.0]
    proxy._monotonic = lambda: now[0]
    target = FrankaJointTarget.from_closed_loop_command(
        _command(ledger, produced_monotonic_s=10.0),
        source=FrankaTargetSource.SUPERVISED_V94,
    )
    proxy._endpoint = object()
    proxy._require_critical_writable_before_encode = (
        lambda _endpoint, *, operation: None
    )
    sent: list[bytes] = []
    proxy._send_packet_once_locked = (
        lambda _endpoint, packet, *, operation: sent.append(packet)
    )

    def delayed_encoder() -> bytes:
        now[0] = 10.051
        return b"target"

    with pytest.raises(
        FrankaNativeSupervisedSessionError,
        match="expired after native encode/before send boundary",
    ):
        proxy._send_encoded(
            delayed_encoder,
            operation="TARGET",
            target=target,
        )

    assert sent == []


@pytest.mark.parametrize("operation", ("TARGET", "STOP"))
def test_nonblocking_eagain_is_one_send_no_retry_and_terminates(
    operation: str,
) -> None:
    proxy, _ledger = _proxy()

    class WouldBlockSocket:
        def __init__(self) -> None:
            self.calls = 0
            self.flags = 0

        def send(self, _packet: bytes, flags: int) -> int:
            self.calls += 1
            self.flags = flags
            raise BlockingIOError(errno.EAGAIN, "synthetic full queue")

    class LiveProcess:
        def __init__(self) -> None:
            self.terminate_calls = 0

        def is_alive(self) -> bool:
            return True

        def terminate(self) -> None:
            self.terminate_calls += 1

    critical = WouldBlockSocket()
    process = LiveProcess()
    endpoint = SimpleNamespace(critical_sock=critical, process=process)

    with pytest.raises(
        FrankaNativeSupervisedSessionError,
        match=r"would block \(EAGAIN\).*not retried.*terminate",
    ):
        proxy._send_packet_once_locked(
            endpoint,
            b"packet",
            operation=operation,
        )

    assert critical.calls == 1
    assert critical.flags & socket.MSG_DONTWAIT
    assert process.terminate_calls == 1


def test_stop_writable_precheck_failure_immediately_terminates_without_retry() -> None:
    proxy, _ledger = _proxy()

    class LiveProcess:
        def __init__(self) -> None:
            self.terminate_calls = 0

        def is_alive(self) -> bool:
            return True

        def terminate(self) -> None:
            self.terminate_calls += 1

    process = LiveProcess()
    proxy._endpoint = SimpleNamespace(process=process)
    writable_checks: list[str] = []
    encoded: list[bool] = []
    sent: list[bytes] = []

    def reject_writable(_endpoint, *, operation: str) -> None:
        writable_checks.append(operation)
        raise FrankaNativeSupervisedSessionError(
            "STOP critical seqpacket is not immediately writable"
        )

    proxy._require_critical_writable_before_encode = reject_writable
    proxy._codec.encode_stop = (
        lambda *, reason: encoded.append(bool(reason)) or b"stop"
    )
    proxy._send_packet_once_locked = (
        lambda _endpoint, packet, *, operation: sent.append(packet)
    )

    with pytest.raises(
        FrankaNativeSupervisedSessionError,
        match="not immediately writable",
    ):
        proxy._send_stop_once("synthetic pre-send failure")

    # STOP is deliberately a one-shot transaction.  The failed precheck is
    # not retried, but the independent native terminate edge is immediate.
    proxy._send_stop_once("must not retry")

    assert writable_checks == ["STOP"]
    assert encoded == []
    assert sent == []
    assert process.terminate_calls == 1


def test_popen_launcher_executes_the_exact_opened_inode_and_releases_parent_fd() -> None:
    executable_fd = os.open(sys.executable, os.O_RDONLY | os.O_CLOEXEC)
    launcher = PopenSeqpacketChildLauncher(
        lambda critical_fd, telemetry_fd: (
            f"/proc/self/fd/{executable_fd}",
            "-c",
            _FAKE_CHILD,
            str(critical_fd),
            str(telemetry_fd),
            "normal",
        ),
        pinned_executable_fd=executable_fd,
    )

    endpoint = launcher.start()
    try:
        with pytest.raises(OSError):
            os.fstat(executable_fd)
        assert endpoint.process.is_alive()
        endpoint.process.terminate()
        endpoint.process.join(timeout=1.0)
    finally:
        endpoint.critical_sock.close()
        endpoint.telemetry_sock.close()
        endpoint.process.close()


def test_popen_launcher_runs_validator_immediately_before_process_creation() -> None:
    calls: list[str] = []

    def reject_changed_host_state() -> None:
        calls.append("validated")
        raise RuntimeError("synthetic host admission drift")

    launcher = PopenSeqpacketChildLauncher(
        lambda critical_fd, telemetry_fd: (
            sys.executable,
            "-c",
            _FAKE_CHILD,
            str(critical_fd),
            str(telemetry_fd),
            "normal",
        ),
        prelaunch_validator=reject_changed_host_state,
    )
    with pytest.raises(RuntimeError, match="host admission drift"):
        launcher.start()
    assert calls == ["validated"]


def test_fake_child_bridges_state_target_ack_and_verified_stop() -> None:
    proxy, ledger = _proxy()
    result: list[object] = []
    errors: list[BaseException] = []

    def owner() -> None:
        try:
            result.append(
                proxy.run(authorization=object(), preflight_token=object())
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=owner)
    thread.start()
    _wait_until(lambda: proxy.c2_bootstrap_ready)
    _wait_until(lambda: len(proxy.pose_ring.snapshot()) == 2)
    poses = proxy.pose_ring.snapshot()
    assert [sample.cycle for sample in poses] == [1, 2]
    assert poses[-1].monotonic_s - poses[0].monotonic_s == pytest.approx(0.11)
    time.sleep(0.03)

    command = _command(ledger)
    ledger.stage(command, now_monotonic_s=time.monotonic())
    ledger.acknowledge(
        "rh56",
        sequence=command.sequence,
        now_monotonic_s=time.monotonic(),
    )
    proxy.target_hold.publish(
        FrankaJointTarget.from_closed_loop_command(
            command,
            source=FrankaTargetSource.SUPERVISED_V94,
        )
    )
    _wait_until(lambda: ledger.last_committed_sequence == 1)

    proxy.request_clean_stop()
    proxy.request_clean_stop()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert errors == []
    assert len(result) == 1
    telemetry = result[0]
    assert telemetry.stop_requested is True
    assert telemetry.stop_verified is True
    assert telemetry.franka_ack_count == 1
    assert telemetry.parent_heartbeat_count >= 1
    assert telemetry.maximum_read_to_write_ns == 300_000
    assert 0.0 < telemetry.maximum_parent_heartbeat_interval_s < 0.1
    assert proxy.last_telemetry is telemetry


def test_wrong_child_ack_never_acknowledges_parent_franka_edge() -> None:
    proxy, ledger = _proxy("wrong_ack")
    errors: list[BaseException] = []

    def owner() -> None:
        try:
            proxy.run(authorization=object(), preflight_token=object())
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=owner)
    thread.start()
    _wait_until(lambda: proxy.c2_bootstrap_ready)
    command = _command(ledger)
    ledger.stage(command, now_monotonic_s=time.monotonic())
    proxy.target_hold.publish(
        FrankaJointTarget.from_closed_loop_command(
            command,
            source=FrankaTargetSource.SUPERVISED_V94,
        )
    )
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], FrankaNativeSupervisedSessionError)
    assert "ACK sequence mismatch" in str(errors[0])
    assert ledger.last_committed_sequence == 0
    assert ledger.fault_reason is not None


def test_target_before_action_ready_fails_without_child_ack() -> None:
    proxy, ledger = _proxy()
    command = _command(ledger)
    ledger.stage(command, now_monotonic_s=time.monotonic())

    with pytest.raises(
        FrankaNativeSupervisedSessionError,
        match="native child is not running",
    ):
        proxy.target_hold.publish(
            FrankaJointTarget.from_closed_loop_command(
                command,
                source=FrankaTargetSource.SUPERVISED_V94,
            )
        )

    assert ledger.last_committed_sequence == 0
    assert ledger.fault_reason is not None


def test_target_after_async_child_fault_surfaces_the_first_fault() -> None:
    proxy, ledger = _proxy()
    first_fault = FrankaNativeSupervisedSessionError(
        "native child FAULT code=12 sequence=29: first useful reason"
    )
    proxy._run_started = True
    proxy._action_ready = True
    proxy._bootstrap_ready = True
    proxy._latch_async_failure(first_fault)

    with pytest.raises(
        FrankaNativeSupervisedSessionError,
        match=(
            "native child already stopped after prior fault: "
            "FrankaNativeSupervisedSessionError: "
            "native child FAULT code=12 sequence=29: first useful reason"
        ),
    ) as caught:
        proxy.target_hold.publish(
            FrankaJointTarget.from_closed_loop_command(
                _command(ledger),
                source=FrankaTargetSource.SUPERVISED_V94,
            )
        )

    assert caught.value.__cause__ is first_fault
    assert proxy._async_failure is first_fault


def test_frozen_python_struct_sizes_and_arm_target_layout() -> None:
    assert V94_NATIVE_HEADER_BYTES == 56
    assert V94_NATIVE_MAX_PACKET_BYTES == 1024
    assert V94_NATIVE_PAYLOAD_BYTES == {
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
    codec, envelope, authorization, token, _q_home = _real_codec_fixture()
    arm = _prime_real_codec(codec, envelope, authorization, token)
    header = _HEADER.unpack_from(arm)
    assert header[2] == V94NativeMessageKind.ARM
    assert header[3] == 616
    assert header[5] == 1
    assert header[7] == _NONCE
    arm_values = struct.Struct("<32s32s32s32s16s6Q51dII8s").unpack_from(
        arm, V94_NATIVE_HEADER_BYTES
    )
    assert arm_values[0] == bytes.fromhex(envelope.profile_sha256)
    assert arm_values[7] == 100_000_000  # parent heartbeat receipt gap
    assert arm_values[8] == 5_000_000_000  # first target after ACTION_READY
    assert arm_values[9] == 500_000_000  # bounded last-target hold gap
    assert arm_values[-2] == V94NativeControllerMode.LEGACY
    assert arm_values[-1] == bytes(8)
    assert envelope.policy_command_max_age_s == pytest.approx(0.050)


def test_protocol_v4_state_decodes_coherent_post_limit_shaper_snapshot() -> None:
    codec, envelope, authorization, token, _q_home = _real_codec_fixture()
    _prime_real_codec(codec, envelope, authorization, token)
    transform = np.eye(4, dtype=np.float64).reshape(-1, order="F")
    measured_q = np.linspace(-0.3, 0.3, 7)
    shaper_q_d = measured_q + np.linspace(-0.01, 0.01, 7)
    shaper_dq_d = np.linspace(-0.5, 0.5, 7)
    shaper_ddq_d = np.linspace(-5.0, 5.0, 7)
    held_q_cmd = shaper_q_d + 0.018 * np.linspace(-4.0, 4.0, 7)
    controller = np.concatenate(
        (
            np.clip((held_q_cmd - shaper_q_d) / 0.018, -4.0, 4.0),
            np.clip((shaper_q_d - measured_q) / 0.05, -4.0, 4.0),
            np.clip(shaper_dq_d / 0.5, -1.0, 1.0),
            np.clip(shaper_ddq_d / 5.0, -1.0, 1.0),
            np.asarray([1.0], dtype=np.float32),
        )
    ).astype(np.float32)
    payload = _STATE.pack(
        101,
        202,
        3_000_000_000,
        4_000_000_000,
        5,
        6,
        1,
        2,
        1.0,
        *measured_q,
        *np.linspace(-0.03, 0.03, 7),
        *transform,
        *shaper_q_d,
        0,
        0,
        100,
        200,
        3,
        0,
        100,
        101,
        101,
        5,
        *controller,
        *shaper_q_d,
        *shaper_dq_d,
        *shaper_ddq_d,
        *held_q_cmd,
    )
    event = codec.decode_state_packet(
        _native_packet(
            V94NativeMessageKind.STATE,
            sequence=1,
            monotonic_ns=3_000_000_000,
            payload=payload,
        )
    )
    assert np.array_equal(event.controller_state29, controller)
    assert np.array_equal(event.commanded_q_rad, shaper_q_d)
    assert np.array_equal(event.shaper_q_d_rad, shaper_q_d)
    assert np.array_equal(event.shaper_dq_d_rad_s, shaper_dq_d)
    assert np.array_equal(event.shaper_ddq_d_rad_s2, shaper_ddq_d)
    assert np.array_equal(event.held_q_cmd_rad, held_q_cmd)
    assert event.active_target_sequence == 5
    assert event.active_observation_sequence == 6

    incoherent = bytearray(payload)
    struct.pack_into("<d", incoherent, 708, held_q_cmd[0] + 0.001)
    with pytest.raises(
        FrankaNativeSupervisedSessionError,
        match="not coherent with its post-limit shaper snapshot",
    ):
        codec.decode_state_packet(
            _native_packet(
                V94NativeMessageKind.STATE,
                sequence=2,
                monotonic_ns=3_000_000_001,
                payload=bytes(incoherent),
            )
        )


def test_arm_encodes_explicit_qd_g015_controller_mode() -> None:
    codec, envelope, authorization, token, _q_home = _real_codec_fixture()
    arm = _prime_real_codec(
        codec,
        envelope,
        authorization,
        token,
        controller_mode=V94NativeControllerMode.QD_G015,
    )
    values = struct.Struct("<32s32s32s32s16s6Q51dII8s").unpack_from(
        arm, V94_NATIVE_HEADER_BYTES
    )
    assert values[-3] == 3
    assert values[-2] == V94NativeControllerMode.QD_G015
    assert values[-1] == bytes(8)


def test_production_arm_q_home_uses_exact_bundle_float32_wire_values() -> None:
    codec, envelope, authorization, token, _q_home = _real_codec_fixture()
    arm = _prime_real_codec(codec, envelope, authorization, token)
    values = struct.Struct("<32s32s32s32s16s6Q51dII8s").unpack_from(
        arm, V94_NATIVE_HEADER_BYTES
    )
    arm_values = values
    q_home = values[11:18]

    assert tuple(value.hex() for value in q_home) == (
        "0x0.0p+0",
        "-0x1.2353f80000000p-1",
        "0x0.0p+0",
        "-0x1.67ae140000000p+1",
        "0x0.0p+0",
        "0x1.84bc6a0000000p+1",
        "0x1.7b645a0000000p-1",
    )
    assert arm_values[1] == bytes.fromhex(envelope.binding_sha256)
    assert arm_values[2] == bytes.fromhex(token.report_sha256)
    assert arm_values[3] == hashlib.sha256(b"native-codec").digest()
    assert arm_values[4] == uuid.UUID(authorization.authorization_id).bytes
    assert arm_values[7:11] == (
        100_000_000,
        5_000_000_000,
        500_000_000,
        15_000_000_000,
    )
    assert arm_values[-3] == 3
    assert arm_values[-2] == V94NativeControllerMode.LEGACY
    assert arm_values[-1] == bytes(8)

    command = ClosedLoopCommand(
        sequence=1,
        produced_monotonic_s=10.0,
        observation_realtime_s=20.0,
        previous_executed_action13_used=INITIAL_PREVIOUS_ACTION13,
        raw_policy_action13=np.zeros(13, dtype=np.float32),
        executed_policy_action13=np.zeros(13, dtype=np.float32),
        franka_target_q_rad=np.arange(7, dtype=np.float64) * 0.001,
        rh56_angle_set_register_order=np.full(6, 1000),
    )
    target = FrankaJointTarget.from_closed_loop_command(
        command,
        source=FrankaTargetSource.SUPERVISED_V94,
    )
    target_packet = codec.encode_target(target)
    target_header = _HEADER.unpack_from(target_packet)
    target_payload = struct.Struct("<QQQ7d").unpack_from(
        target_packet, V94_NATIVE_HEADER_BYTES
    )
    assert target_header[2] == V94NativeMessageKind.TARGET
    assert target_header[5] == 2
    assert target_payload[:3] == (1, 1, 10_000_000_000)
    assert target_payload[3:] == pytest.approx(command.franka_target_q_rad)


@pytest.mark.parametrize("failure", ("zero_nonce", "crc", "sequence", "kind"))
def test_hello_rejects_zero_nonce_crc_sequence_and_unknown_kind(failure: str) -> None:
    codec, _envelope, _authorization, _token, _q_home = _real_codec_fixture()
    packet = _native_packet(
        V94NativeMessageKind.HELLO,
        sequence=2 if failure == "sequence" else 1,
        monotonic_ns=1,
        payload=_hello_payload(),
        nonce=bytes(16) if failure == "zero_nonce" else _NONCE,
    )
    if failure == "crc":
        value = bytearray(packet)
        value[-1] ^= 0x80
        packet = bytes(value)
    elif failure == "kind":
        packet = _rewrite_packet(packet, kind=0x7FFF)

    with pytest.raises(FrankaNativeSupervisedSessionError):
        codec.decode_critical_packet(packet)


def test_ack_is_write_evidence_and_stop_proof_is_parent_recomputed() -> None:
    codec, envelope, authorization, token, _q_home = _real_codec_fixture()
    _prime_real_codec(codec, envelope, authorization, token)
    target_q = np.arange(7, dtype=np.float64) * 0.001
    command = ClosedLoopCommand(
        sequence=1,
        produced_monotonic_s=10.0,
        observation_realtime_s=20.0,
        previous_executed_action13_used=INITIAL_PREVIOUS_ACTION13,
        raw_policy_action13=np.zeros(13, dtype=np.float32),
        executed_policy_action13=np.zeros(13, dtype=np.float32),
        franka_target_q_rad=target_q,
        rh56_angle_set_register_order=np.full(6, 1000),
    )
    codec.encode_target(
        FrankaJointTarget.from_closed_loop_command(
            command,
            source=FrankaTargetSource.SUPERVISED_V94,
        )
    )
    ack_payload = _ACK.pack(
        1,
        1,
        10_000_000_000,
        1_000_000,
        110,
        110,
        10_001_000_000,
        1,
        0,
        *target_q,
        *target_q,
        *target_q,
        300_000,
        0.0,
    )
    ack = codec.decode_critical_packet(
        _native_packet(
            V94NativeMessageKind.ACK,
            sequence=4,
            monotonic_ns=4,
            payload=ack_payload,
        )
    )
    assert isinstance(ack, NativeAckEvent)
    assert ack.sequence == ack.observation_sequence == 1
    assert ack.target_age_at_write_ns == 1_000_000

    proof = codec.decode_critical_packet(
        _native_packet(
            V94NativeMessageKind.STOP_PROOF,
            sequence=5,
            monotonic_ns=5,
            payload=_healthy_stop_payload(),
        )
    )
    assert isinstance(proof, NativeStopProofEvent)
    assert proof.finish_succeeded is True
    assert proof.robot_stop_attempted is False
    assert proof.parent_stop_verified is True


def test_stop_proof_child_idle_claim_cannot_override_raw_samples() -> None:
    codec, envelope, authorization, token, _q_home = _real_codec_fixture()
    _prime_real_codec(codec, envelope, authorization, token)

    with pytest.raises(
        FrankaNativeSupervisedSessionError,
        match="idle_dq_verified differs",
    ):
        codec.decode_critical_packet(
            _native_packet(
                V94NativeMessageKind.STOP_PROOF,
                sequence=4,
                monotonic_ns=4,
                payload=_healthy_stop_payload(corrupt_last_mode=True),
            )
        )


def test_fault_stop_path_uses_raw_idle_proof_even_if_robot_stop_returned_failure() -> None:
    codec, envelope, authorization, token, _q_home = _real_codec_fixture()
    _prime_real_codec(codec, envelope, authorization, token)

    proof = codec.decode_critical_packet(
        _native_packet(
            V94NativeMessageKind.STOP_PROOF,
            sequence=4,
            monotonic_ns=4,
            payload=_fault_stop_payload(),
        )
    )

    assert isinstance(proof, NativeStopProofEvent)
    assert proof.terminal_fault_code == 14
    assert proof.finish_attempted is False
    assert proof.robot_stop_attempted is True
    assert proof.robot_stop_succeeded is False
    assert proof.parent_stop_verified is True
    assert proof.fault_reason == "read-to-write deadline"


def test_fault_stop_accepts_exact_reflex_communication_only_proof() -> None:
    codec, envelope, authorization, token, _q_home = _real_codec_fixture()
    _prime_real_codec(codec, envelope, authorization, token)

    proof = codec.decode_critical_packet(
        _native_packet(
            V94NativeMessageKind.STOP_PROOF,
            sequence=4,
            monotonic_ns=4,
            payload=_fault_comm_only_stop_payload(),
        )
    )

    assert isinstance(proof, NativeStopProofEvent)
    assert proof.terminal_fault_code == 11
    assert proof.robot_stop_succeeded is True
    assert proof.idle_dq_verified is True
    assert proof.parent_stop_verified is True
    assert proof.maximum_control_period_ms == 13


def test_fault_stop_rejects_communication_tuple_with_contact_flag() -> None:
    codec, envelope, authorization, token, _q_home = _real_codec_fixture()
    _prime_real_codec(codec, envelope, authorization, token)

    with pytest.raises(
        FrankaNativeSupervisedSessionError,
        match="idle_dq_verified differs",
    ):
        codec.decode_critical_packet(
            _native_packet(
                V94NativeMessageKind.STOP_PROOF,
                sequence=4,
                monotonic_ns=4,
                payload=_fault_comm_only_stop_payload(extra_status_flag=1 << 1),
            )
        )


def test_requested_streaming_stop_uses_direct_stop_without_finish_attempt() -> None:
    codec, envelope, authorization, token, _q_home = _real_codec_fixture()
    _prime_real_codec(codec, envelope, authorization, token)

    proof = codec.decode_critical_packet(
        _native_packet(
            V94NativeMessageKind.STOP_PROOF,
            sequence=4,
            monotonic_ns=4,
            payload=_requested_direct_stop_payload(),
        )
    )

    assert isinstance(proof, NativeStopProofEvent)
    assert proof.stop_reason == 1
    assert proof.terminal_fault_code == 0
    assert proof.finish_attempted is False
    assert proof.finish_succeeded is False
    assert proof.robot_stop_attempted is True
    assert proof.robot_stop_succeeded is True
    assert proof.idle_dq_verified is True
    assert proof.parent_stop_verified is True
    assert proof.final_robot_mode == 5


def test_requested_streaming_stop_rejects_guiding_mode() -> None:
    codec, envelope, authorization, token, _q_home = _real_codec_fixture()
    _prime_real_codec(codec, envelope, authorization, token)

    with pytest.raises(
        FrankaNativeSupervisedSessionError,
        match="idle_dq_verified differs",
    ):
        codec.decode_critical_packet(
            _native_packet(
                V94NativeMessageKind.STOP_PROOF,
                sequence=4,
                monotonic_ns=4,
                payload=_requested_direct_stop_payload(mode=4),
            )
        )


def test_failed_finish_explicit_stop_fallback_uses_fresh_raw_idle_proof() -> None:
    codec, envelope, authorization, token, _q_home = _real_codec_fixture()
    _prime_real_codec(codec, envelope, authorization, token)

    proof = codec.decode_critical_packet(
        _native_packet(
            V94NativeMessageKind.STOP_PROOF,
            sequence=4,
            monotonic_ns=4,
            payload=_finish_fallback_stop_payload(),
        )
    )

    assert isinstance(proof, NativeStopProofEvent)
    assert proof.terminal_fault_code == 0
    assert proof.finish_attempted is True
    assert proof.finish_succeeded is False
    assert proof.robot_stop_attempted is True
    assert proof.robot_stop_succeeded is True
    assert proof.idle_dq_verified is True
    assert proof.parent_stop_verified is True


def test_production_native_symbols_are_explicitly_exported() -> None:
    import robot_control.franka.native_session as module

    assert {
        "FrankaNativeSupervisedSessionProxy",
        "NativeHelloExpectation",
        "PopenSeqpacketChildLauncher",
        "V94NativePODCodec",
        "V94_NATIVE_PAYLOAD_BYTES",
    }.issubset(set(module.__all__))


def test_cpp_and_python_fixed_pod_golden_packets_are_byte_exact(tmp_path: Path) -> None:
    """Compile the authoritative C++ codec and compare two complete packets."""

    workspace = Path(__file__).resolve().parents[2]
    native = workspace / "dexgrasp/native/v94_franka_servo"
    source = tmp_path / "golden.cpp"
    binary = tmp_path / "golden"
    source.write_text(
        r'''
#include "anydex/v94_franka_servo/protocol.hpp"
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>

namespace servo = anydex::v94_franka_servo;

void print_packet(const servo::EncodedPacket& packet) {
  for (std::size_t i = 0; i < packet.size; ++i) {
    std::cout << std::hex << std::setfill('0') << std::setw(2)
              << static_cast<unsigned>(packet.bytes[i]);
  }
  std::cout << "\n";
}

int main() {
  std::array<std::uint8_t, 16> nonce{};
  for (std::size_t i = 0; i < nonce.size(); ++i) {
    nonce[i] = static_cast<std::uint8_t>(i + 1U);
  }
  servo::TargetPayload target{};
  target.target_sequence = 7U;
  target.observation_sequence = 19U;
  target.produced_monotonic_ns = 123456789U;
  for (std::size_t i = 0; i < 7; ++i) {
    target.target_q_rad[i] = 0.125 * static_cast<double>(i + 1U);
  }
  servo::EncodedPacket target_packet{};
  if (servo::encode_payload(servo::MessageKind::kTarget, 3U, 999U,
                            nonce, target, &target_packet) !=
      servo::CodecError::kNone) return 2;
  print_packet(target_packet);

  servo::StopProofPayload proof{};
  proof.stop_reason = 1U;
  proof.terminal_fault_code = 0U;
  proof.finish_attempted = 1U;
  proof.finish_succeeded = 1U;
  proof.robot_stop_attempted = 0U;
  proof.robot_stop_succeeded = 0U;
  proof.idle_dq_verified = 1U;
  proof.active_handle_released = 1U;
  proof.fault_reply_delivered = 0U;
  proof.robot_backend_released = 1U;
  proof.control_cycles = 120U;
  proof.last_target_sequence = 1U;
  proof.active_read_count = 120U;
  proof.active_write_count = 120U;
  proof.target_ack_count = 1U;
  proof.stop_verification_samples = 3U;
  proof.stop_consecutive_idle_samples = 3U;
  proof.final_robot_mode = 1U;
  proof.maximum_control_period_ms = 2U;
  proof.maximum_stop_dq_rad_s = 0.0;
  proof.maximum_read_to_write_ns = 500000U;
  proof.pre_stop_robot_time_ms = 200U;
  proof.final_robot_time_ms = 203U;
  proof.telemetry_drop_count = 0U;
  for (std::size_t i = 0; i < 3; ++i) {
    proof.verified_samples[i].robot_time_ms = 201U + i;
    proof.verified_samples[i].robot_mode = 1U;
    proof.verified_samples[i].status_flags = 0U;
  }
  servo::EncodedPacket proof_packet{};
  if (servo::encode_payload(servo::MessageKind::kStopProof, 5U, 1000U,
                            nonce, proof, &proof_packet) !=
      servo::CodecError::kNone) return 3;
  print_packet(proof_packet);
  return 0;
}
''',
        encoding="utf-8",
    )
    subprocess.run(
        [
            "/usr/bin/c++",
            "-std=c++17",
            "-O2",
            "-I",
            str(native / "include"),
            str(source),
            str(native / "src/protocol.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    target_hex, proof_hex = completed.stdout.strip().splitlines()
    nonce = bytes(range(1, 17))
    python_target = _native_packet(
        V94NativeMessageKind.TARGET,
        sequence=3,
        monotonic_ns=999,
        nonce=nonce,
        payload=struct.pack(
            "<QQQ7d",
            7,
            19,
            123456789,
            *(0.125 * (index + 1) for index in range(7)),
        ),
    )
    python_proof = _native_packet(
        V94NativeMessageKind.STOP_PROOF,
        sequence=5,
        monotonic_ns=1000,
        nonce=nonce,
        payload=_healthy_stop_payload(),
    )

    assert bytes.fromhex(target_hex) == python_target
    assert bytes.fromhex(proof_hex) == python_proof


def test_frozen_cpp_target_golden_literal_matches_python_complete_packet() -> None:
    payload = struct.pack(
        "<QQQ7d",
        0x2122232425262728,
        0x3132333435363738,
        0x4142434445464748,
        0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        3.25,
        -2.75,
    )
    packet = _native_packet(
        V94NativeMessageKind.TARGET,
        sequence=0x0102030405060708,
        monotonic_ns=0x1112131415161718,
        nonce=bytes(range(16)),
        payload=payload,
    )

    assert packet.hex() == _CPP_GOLDEN_TARGET_HEX
