"""Wire-level regressions for a delayed RH56 write ACK crossing into a read."""

from __future__ import annotations

import importlib
import socket
import struct
import sys
import threading
import time

import pytest

_HAND_ID = 1
_ANGLE_SET = 1486
_TARGETS = (1000, 1000, 929, 990, 49, 1000)
_REQUEST_HEADER = b"\xEB\x90"
_RESPONSE_HEADER = b"\x90\xEB"
_READ_COMMAND = 0x11
_WRITE_COMMAND = 0x12


def _checksum(frame_without_checksum: bytes) -> int:
    return sum(frame_without_checksum[2:]) & 0xFF


def _request(command: int, address: int, data: bytes) -> bytes:
    if command == _READ_COMMAND:
        body = bytes((command, address & 0xFF, address >> 8, data[0]))
    else:
        body = bytes((command, address & 0xFF, address >> 8)) + data
    frame = bytearray(
        (*_REQUEST_HEADER, _HAND_ID, len(body), *body)
    )
    frame.append(_checksum(frame))
    return bytes(frame)


_WRITE_REQUEST = _request(
    _WRITE_COMMAND, _ANGLE_SET, struct.pack("<6h", *_TARGETS)
)
_READ_REQUEST = _request(_READ_COMMAND, _ANGLE_SET, b"\x0c")
_SECOND_WRITE_REQUEST = _request(
    _WRITE_COMMAND,
    _ANGLE_SET,
    struct.pack("<6h", *(1000, 1000, 930, 989, 50, 1000)),
)
_THIRD_WRITE_REQUEST = _request(
    _WRITE_COMMAND,
    _ANGLE_SET,
    struct.pack("<6h", *(1000, 1000, 931, 988, 51, 1000)),
)
_FEEDBACK_ADDRESS = 1534
_FEEDBACK_DATA = bytes(range(24))
_FEEDBACK_REQUEST = _request(
    _READ_COMMAND, _FEEDBACK_ADDRESS, bytes((len(_FEEDBACK_DATA),))
)


def _response(
    command: int,
    address: int,
    data: bytes,
    *,
    hand_id: int = _HAND_ID,
) -> bytes:
    frame = bytearray(
        (
            *_RESPONSE_HEADER,
            hand_id,
            len(data) + 3,
            command,
            address & 0xFF,
            (address >> 8) & 0xFF,
            *data,
        )
    )
    frame.append(_checksum(frame))
    return bytes(frame)


def _parse_response(
    frame: bytes,
    command: int,
    address: int,
    expected_data_length: int,
) -> bytes:
    if len(frame) < 8 or frame[:2] != _RESPONSE_HEADER:
        raise ValueError("invalid response header")
    if len(frame) != frame[3] + 5:
        raise ValueError("invalid response length")
    if _checksum(frame[:-1]) != frame[-1]:
        raise ValueError("bad response checksum")
    if frame[2] != _HAND_ID:
        raise ValueError("wrong hand ID")
    if frame[4] != command:
        raise ValueError(f"unexpected response command 0x{frame[4]:02X}")
    actual_address = frame[5] | (frame[6] << 8)
    if actual_address != address:
        raise ValueError("wrong response address")
    data = frame[7:-1]
    if len(data) != expected_data_length:
        raise ValueError("wrong response data length")
    return data


_WRITE_ACK = _response(_WRITE_COMMAND, _ANGLE_SET, b"\x01")
_READ_RESPONSE = _response(
    _READ_COMMAND, _ANGLE_SET, struct.pack("<6h", *_TARGETS)
)
_FEEDBACK_RESPONSE = _response(
    _READ_COMMAND, _FEEDBACK_ADDRESS, _FEEDBACK_DATA
)


def _bad_checksum(frame: bytes) -> bytes:
    corrupted = bytearray(frame)
    corrupted[-1] ^= 0x01
    return bytes(corrupted)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = sock.recv(size - len(result))
        if not chunk:
            raise RuntimeError("wire peer closed before one request was complete")
        result.extend(chunk)
    return bytes(result)


def _timed_out_write_then_exchange(
    chunks: tuple[bytes, ...],
    *,
    next_request: bytes = _READ_REQUEST,
    through_hand: bool = False,
):
    api = importlib.import_module("examples.inspire_rh56_test")
    host, device = socket.socketpair()
    device.settimeout(1.0)
    requests: list[bytes] = []
    peer_errors: list[BaseException] = []

    def peer() -> None:
        try:
            requests.append(_recv_exact(device, len(_WRITE_REQUEST)))
            requests.append(_recv_exact(device, len(next_request)))
            for index, chunk in enumerate(chunks):
                if index:
                    time.sleep(0.002)
                device.sendall(chunk)
        except BaseException as exc:
            peer_errors.append(exc)

    thread = threading.Thread(target=peer)
    thread.start()
    serial = api.LinuxSerial("/dev/not-used", 115200, 0.010)
    serial.fd = host.fileno()
    hand = api.RH56Hand(serial, _HAND_ID)
    # tcflush is a tty operation; the socketpair supplies only deterministic
    # wire framing for this hardware-inert regression.
    serial.discard_input = lambda: None  # type: ignore[method-assign]
    write_error = None
    read_result = None
    read_error = None
    try:
        try:
            if through_hand:
                hand.write_six_shorts(
                    _ANGLE_SET, _TARGETS, retries=0
                )
            else:
                serial.exchange(_WRITE_REQUEST)
        except BaseException as exc:
            write_error = exc
        serial.timeout = 0.050
        try:
            if through_hand:
                assert next_request == _READ_REQUEST
                read_result = hand.read(
                    _ANGLE_SET, 12, retries=0
                )
            else:
                read_result = serial.exchange(next_request)
        except BaseException as exc:
            read_error = exc
    finally:
        serial.fd = None
        thread.join(timeout=1.0)
        host.close()
        device.close()
        sys.modules.pop("examples.inspire_rh56_test", None)

    assert not thread.is_alive()
    assert peer_errors == []
    assert type(write_error).__name__ == "RH56Error"
    assert "timeout waiting for hand response" in str(write_error)
    assert requests == [_WRITE_REQUEST, next_request]
    return read_result, read_error


@pytest.mark.parametrize(
    "chunks",
    [
        (_WRITE_ACK + _READ_RESPONSE,),
        (_WRITE_ACK, _READ_RESPONSE),
    ],
    ids=("same-os-read", "separate-os-reads"),
)
def test_one_valid_late_write_ack_is_skipped_without_resending(chunks) -> None:
    response, error = _timed_out_write_then_exchange(
        chunks, through_hand=True
    )

    assert error is None
    assert response == struct.pack("<6h", *_TARGETS)


@pytest.mark.parametrize(
    "bad_ack",
    [
        _response(_WRITE_COMMAND, _ANGLE_SET + 1, b"\x01"),
        _response(_WRITE_COMMAND, _ANGLE_SET, b"\x00"),
        _response(_WRITE_COMMAND, _ANGLE_SET, b"\x01", hand_id=2),
        _bad_checksum(_WRITE_ACK),
    ],
    ids=("wrong-address", "negative-ack", "wrong-hand-id", "bad-checksum"),
)
def test_unproven_write_frame_is_not_silently_skipped(bad_ack: bytes) -> None:
    response, error = _timed_out_write_then_exchange(
        (bad_ack + _READ_RESPONSE,)
    )

    assert error is None
    assert response == bad_ack
    with pytest.raises(ValueError):
        _parse_response(response, _READ_COMMAND, _ANGLE_SET, 12)


def test_a_second_late_write_ack_is_not_silently_skipped() -> None:
    response, error = _timed_out_write_then_exchange(
        (_WRITE_ACK + _WRITE_ACK + _READ_RESPONSE,)
    )

    assert error is None
    assert response == _WRITE_ACK
    with pytest.raises(ValueError, match="unexpected response command 0x12"):
        _parse_response(response, _READ_COMMAND, _ANGLE_SET, 12)


def test_late_write_ack_without_exact_read_response_keeps_original_deadline() -> None:
    response, error = _timed_out_write_then_exchange((_WRITE_ACK,))

    assert response is None
    assert type(error).__name__ == "RH56Error"
    assert "validated one late write ACK" in str(error)


def test_same_address_next_write_cannot_claim_one_ambiguous_ack() -> None:
    response, error = _timed_out_write_then_exchange(
        (_WRITE_ACK,),
        next_request=_SECOND_WRITE_REQUEST,
    )

    assert response is None
    assert type(error).__name__ == "RH56Error"
    assert "validated one late write ACK" in str(error)


def test_same_address_next_write_requires_a_second_ack() -> None:
    response, error = _timed_out_write_then_exchange(
        (_WRITE_ACK + _WRITE_ACK,),
        next_request=_SECOND_WRITE_REQUEST,
    )

    assert error is None
    assert response == _WRITE_ACK


@pytest.mark.parametrize(
    ("ack_count", "must_succeed"),
    [(2, False), (3, True)],
    ids=("ack1-plus-ack2-cannot-prove-write3", "third-ack-proves-write3"),
)
def test_multiple_same_address_ack_debts_do_not_collapse(
    ack_count: int, must_succeed: bool
) -> None:
    api = importlib.import_module("examples.inspire_rh56_test")
    host, device = socket.socketpair()
    device.settimeout(1.0)
    requests: list[bytes] = []
    peer_errors: list[BaseException] = []

    def peer() -> None:
        try:
            for request in (
                _WRITE_REQUEST,
                _SECOND_WRITE_REQUEST,
                _THIRD_WRITE_REQUEST,
            ):
                requests.append(_recv_exact(device, len(request)))
            device.sendall(_WRITE_ACK * ack_count)
        except BaseException as exc:
            peer_errors.append(exc)

    thread = threading.Thread(target=peer)
    thread.start()
    serial = api.LinuxSerial("/dev/not-used", 115200, 0.010)
    serial.fd = host.fileno()
    serial.discard_input = lambda: None  # type: ignore[method-assign]
    first_errors: list[BaseException] = []
    third_response = None
    third_error = None
    try:
        for request in (_WRITE_REQUEST, _SECOND_WRITE_REQUEST):
            try:
                serial.exchange(request)
            except BaseException as exc:
                first_errors.append(exc)
        serial.timeout = 0.050
        try:
            third_response = serial.exchange(_THIRD_WRITE_REQUEST)
        except BaseException as exc:
            third_error = exc
    finally:
        serial.fd = None
        thread.join(timeout=1.0)
        host.close()
        device.close()
        sys.modules.pop("examples.inspire_rh56_test", None)

    assert not thread.is_alive()
    assert peer_errors == []
    assert len(first_errors) == 2
    assert all(type(error).__name__ == "RH56Error" for error in first_errors)
    assert requests == [
        _WRITE_REQUEST,
        _SECOND_WRITE_REQUEST,
        _THIRD_WRITE_REQUEST,
    ]
    if must_succeed:
        assert third_error is None
        assert third_response == _WRITE_ACK
    else:
        assert third_response is None
        assert type(third_error).__name__ == "RH56Error"
        assert "validated 2 late write ACKs" in str(third_error)


def test_verified_read_on_same_hand_is_a_barrier_before_stop_write() -> None:
    api = importlib.import_module("examples.inspire_rh56_test")
    host, device = socket.socketpair()
    device.settimeout(1.0)
    requests: list[bytes] = []
    peer_errors: list[BaseException] = []

    def peer() -> None:
        try:
            requests.append(_recv_exact(device, len(_WRITE_REQUEST)))
            requests.append(_recv_exact(device, len(_FEEDBACK_REQUEST)))
            # The old write ACK is genuinely lost.  The valid later read
            # response itself must clear that hand's ordered-stream debt.
            device.sendall(_FEEDBACK_RESPONSE)
            requests.append(_recv_exact(device, len(_SECOND_WRITE_REQUEST)))
            device.sendall(_WRITE_ACK)
        except BaseException as exc:
            peer_errors.append(exc)

    thread = threading.Thread(target=peer)
    thread.start()
    serial = api.LinuxSerial("/dev/not-used", 115200, 0.010)
    serial.fd = host.fileno()
    serial.discard_input = lambda: None  # type: ignore[method-assign]
    first_error = None
    feedback_response = None
    stop_write_response = None
    try:
        try:
            serial.exchange(_WRITE_REQUEST)
        except BaseException as exc:
            first_error = exc
        serial.timeout = 0.050
        feedback_response = serial.exchange(_FEEDBACK_REQUEST)
        stop_write_response = serial.exchange(_SECOND_WRITE_REQUEST)
    finally:
        serial.fd = None
        thread.join(timeout=1.0)
        host.close()
        device.close()
        sys.modules.pop("examples.inspire_rh56_test", None)

    assert not thread.is_alive()
    assert peer_errors == []
    assert type(first_error).__name__ == "RH56Error"
    assert feedback_response == _FEEDBACK_RESPONSE
    assert stop_write_response == _WRITE_ACK
    assert requests == [
        _WRITE_REQUEST,
        _FEEDBACK_REQUEST,
        _SECOND_WRITE_REQUEST,
    ]


def test_complete_frame_read_on_deadline_edge_is_parsed_immediately(
    monkeypatch,
) -> None:
    """A final readable wakeup must not strand a complete frame in the buffer."""

    api = importlib.import_module("examples.inspire_rh56_test")
    clock = [100.0]
    writes: list[bytes] = []
    reads: list[int] = []

    def monotonic() -> float:
        return clock[0]

    def write(_fd: int, data) -> int:
        payload = bytes(data)
        writes.append(payload)
        return len(payload)

    def select_ready(readable, _writable, _exceptional, timeout):
        assert readable == [123]
        assert timeout == pytest.approx(0.010)
        clock[0] = 100.0099
        return ([123], [], [])

    def read(_fd: int, size: int) -> bytes:
        reads.append(size)
        # Simulate the scheduler crossing the relative deadline immediately
        # after the kernel reported the complete response as readable.
        clock[0] = 100.0101
        return _WRITE_ACK

    monkeypatch.setattr(api.time, "monotonic", monotonic)
    monkeypatch.setattr(api.os, "write", write)
    monkeypatch.setattr(api.os, "read", read)
    monkeypatch.setattr(api.select, "select", select_ready)

    serial = api.LinuxSerial("/dev/not-used", 115200, 0.010)
    serial.fd = 123
    serial.discard_input = lambda: None  # type: ignore[method-assign]

    assert serial.exchange(_WRITE_REQUEST) == _WRITE_ACK
    assert writes == [_WRITE_REQUEST]
    assert reads == [4096]
