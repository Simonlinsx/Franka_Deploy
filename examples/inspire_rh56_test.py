#!/usr/bin/env python3
"""Minimal Linux test for an Inspire Robots RH56 hand over USB-RS485.

The wire format and register addresses come from Inspire Robots' official
"RH56 series user manual V1.09".  The default ``status`` action is read-only.
No third-party Python package is required.

Examples:
    python3 examples/inspire_rh56_test.py self-test
    python3 examples/inspire_rh56_test.py status --port /dev/ttyUSB0
    python3 examples/inspire_rh56_test.py nudge --port /dev/ttyUSB0 \
        --joint pinky --confirm-movement
    python3 examples/inspire_rh56_test.py sweep --port /dev/ttyUSB0 \
        --confirm-full-sweep
"""

import argparse
import fcntl
import glob
import os
import select
import struct
import sys
import termios
import time
import tty
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple


REQUEST_HEADER = b"\xEB\x90"
RESPONSE_HEADER = b"\x90\xEB"
READ_COMMAND = 0x11
WRITE_COMMAND = 0x12

REG_HAND_ID = 1000
REG_BAUD = 1002
REG_CLEAR_ERROR = 1004
REG_VOLTAGE = 1472
REG_POS_SET = 1474
REG_ANGLE_SET = 1486
REG_FORCE_SET = 1498
REG_SPEED_SET = 1522
REG_POS_ACT = 1534
REG_ANGLE_ACT = 1546
REG_FORCE_ACT = 1582
REG_CURRENT = 1594
REG_ERROR = 1606
REG_STATUS = 1612
REG_TEMP = 1618
REG_CURRENT_LIMIT = 1020

POSITION_TOLERANCE = 10
NUDGE_POSITION_TOLERANCE = 3
SWEEP_POSITION_TOLERANCE = 20
SWEEP_ENDPOINT_TOLERANCE = 30
OPEN_POSITION_MAX = 200
THUMB_ROTATE_NUDGE_MIN_ANGLE = 800
THUMB_ROTATE_NUDGE_MAX_ANGLE = 900
THUMB_ROTATE_NUDGE_MIN_BEND_CLEARANCE = 950
NUDGE_MAX_IDLE_CURRENT_MA = 100
NUDGE_MAX_TOTAL_IDLE_CURRENT_MA = 200
NUDGE_MAX_INACTIVE_ANGLE_DRIFT = 10
NUDGE_MAX_INACTIVE_POSITION_DRIFT = 25
PROBE_MIN_ANGLE_MOVEMENT = 3
PROBE_MIN_POSITION_MOVEMENT = 5
PROBE_CONFIRM_SAMPLES = 2
PROBE_MAX_FORCE_DELTA_G = 100
THUMB_ROTATE_FEEDBACK_TO_COMMAND_OFFSET = 15
THUMB_ROTATE_HOLD_VALID_MIN_ANGLE = 840
THUMB_ROTATE_HOLD_VALID_MAX_ANGLE = 870
THUMB_ROTATE_VISUAL_LOW_COMMAND = 855
THUMB_ROTATE_VISUAL_HIGH_COMMAND = 885
THUMB_ROTATE_VISUAL_CYCLES = 2
THUMB_ROTATE_VISUAL_ENDPOINT_TOLERANCE = 10
THUMB_ROTATE_VISUAL_PAUSE_SECONDS = 0.7
THUMB_ROTATE_VISUAL_SETTLE_SAMPLES = 3
THUMB_ROTATE_VISUAL_STABILITY_DELTA = 2
THUMB_ROTATE_VISUAL_IDLE_CURRENT_MA = 50
THUMB_ROTATE_FULL_CYCLE_TOKEN = "RH56_FULL_RANGE_1000_0_1000"
THUMB_ROTATE_FULL_CYCLE_MIN_BEND_CLEARANCE = 980
THUMB_ROTATE_FULL_CYCLE_MAX_JOINT_CURRENT_MA = 400
THUMB_ROTATE_FULL_CYCLE_MAX_TOTAL_CURRENT_MA = 600
THUMB_ROTATE_FULL_CYCLE_STOP_ANGLE_DRIFT = 2
THUMB_ROTATE_FULL_CYCLE_STOP_POSITION_DRIFT = 3
THUMB_ROTATE_REALTIME_OPEN_MIN_ANGLE = 885
THUMB_ROTATE_REALTIME_OPEN_ACCEPT_ANGLE = 980
THUMB_ROTATE_REALTIME_OPEN_TOLERANCE = 20
THUMB_ROTATE_REALTIME_OPEN_SPEED = 40
THUMB_ROTATE_REALTIME_OPEN_FORCE_LIMIT = 80
MIDDLE_POSITION_RECOVERY_TOKEN = "RH56_MIDDLE_POSITIVE_RANGE_RECOVERY"
MIDDLE_POSITION_RECOVERY_SPEED = 20
MIDDLE_POSITION_RECOVERY_FORCE_G = 300
MIDDLE_POSITION_RECOVERY_CURRENT_LIMIT_MA = 100
MIDDLE_POSITION_RECOVERY_WAYPOINTS = (0,)
MIDDLE_POSITION_RECOVERY_TIMEOUT_S = 8.0
CLEAR_ERROR_TOKEN = "RH56_CLEAR_LATCHED_ERROR"
CLEAR_ERROR_MAX_IDLE_CURRENT_MA = 100
CLEAR_ERROR_MAX_ANGLE_DRIFT = 3
CLEAR_ERROR_MAX_POSITION_DRIFT = 5
STALE_Q6_RECOVERY_TOKEN = "RH56_STALE_Q6_STATUS_RECOVERY"
STALE_Q6_RECOVERY_SPEED = 20
STALE_Q6_RECOVERY_FORCE_G = 80
STALE_Q6_RECOVERY_CURRENT_LIMIT_MA = 100
STALE_Q6_RECOVERY_MIN_COMMAND = 416
STALE_Q6_RECOVERY_MAX_COMMAND = 1000
STALE_Q6_RECOVERY_MIN_INDEX_ANGLE = 800
STALE_Q6_RECOVERY_MIN_THUMB_BEND_ANGLE = 950
STALE_Q6_RECOVERY_MAX_FORCE_ABS_G = 200
STALE_Q6_RECOVERY_MAX_ANGLE_DRIFT = 20
STALE_Q6_RECOVERY_MAX_POSITION_DRIFT = 40
STALE_Q6_RECOVERY_TIMEOUT_S = 1.5

JOINTS = ("pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotate")
JOINT_LABELS = (
    "小拇指",
    "无名指",
    "中指",
    "食指",
    "拇指弯曲",
    "拇指旋转",
)

STATUS_LABELS: Dict[int, str] = {
    0: "正在松开",
    1: "正在抓取",
    2: "位置到位",
    3: "力控到位",
    5: "电流保护停止",
    6: "堵转停止",
    7: "故障停止",
}

ERROR_BITS = (
    (0, "堵转"),
    (1, "过温"),
    (2, "过流"),
    (3, "电机异常"),
    (4, "通信故障"),
)

BAUD_CONSTANTS = {
    115200: termios.B115200,
    57600: termios.B57600,
    19200: termios.B19200,
}


class RH56Error(RuntimeError):
    """Communication, protocol, or safety error."""


def hex_bytes(data: bytes) -> str:
    return " ".join(f"{value:02X}" for value in data)


def checksum(frame_without_checksum: Sequence[int]) -> int:
    """Checksum is the low byte of the sum, excluding the two-byte header."""
    return sum(frame_without_checksum[2:]) & 0xFF


def validate_common(hand_id: int, address: int) -> None:
    if not 1 <= hand_id <= 254:
        raise ValueError("hand ID must be in 1..254")
    if not 0 <= address <= 0xFFFF:
        raise ValueError("register address must be in 0..65535")


def build_read_frame(hand_id: int, address: int, length: int) -> bytes:
    validate_common(hand_id, address)
    if not 1 <= length <= 255:
        raise ValueError("read length must be in 1..255 bytes")
    frame = bytearray(
        (
            *REQUEST_HEADER,
            hand_id,
            0x04,
            READ_COMMAND,
            address & 0xFF,
            (address >> 8) & 0xFF,
            length,
        )
    )
    frame.append(checksum(frame))
    return bytes(frame)


def build_write_frame(hand_id: int, address: int, data: bytes) -> bytes:
    validate_common(hand_id, address)
    if not 1 <= len(data) <= 252:
        raise ValueError("write data length must be in 1..252 bytes")
    frame = bytearray(
        (
            *REQUEST_HEADER,
            hand_id,
            len(data) + 3,
            WRITE_COMMAND,
            address & 0xFF,
            (address >> 8) & 0xFF,
            *data,
        )
    )
    frame.append(checksum(frame))
    return bytes(frame)


def parse_response(
    frame: bytes,
    hand_id: int,
    command: int,
    address: int,
    expected_data_length: int,
) -> bytes:
    if len(frame) < 8 or frame[:2] != RESPONSE_HEADER:
        raise RH56Error(f"invalid response header: {hex_bytes(frame)}")
    wire_length = frame[3]
    if len(frame) != wire_length + 5:
        raise RH56Error(
            f"invalid response length: field={wire_length}, bytes={len(frame)}"
        )
    if checksum(frame[:-1]) != frame[-1]:
        raise RH56Error(f"bad response checksum: {hex_bytes(frame)}")
    if frame[2] != hand_id:
        raise RH56Error(f"response ID {frame[2]} does not match requested ID {hand_id}")
    if frame[4] != command:
        raise RH56Error(f"unexpected response command 0x{frame[4]:02X}")
    response_address = frame[5] | (frame[6] << 8)
    if response_address != address:
        raise RH56Error(
            f"response address {response_address} does not match {address}"
        )
    data = frame[7:-1]
    if len(data) != expected_data_length:
        raise RH56Error(
            f"expected {expected_data_length} data bytes, received {len(data)}"
        )
    return data


def _request_identity(frame: bytes) -> Optional[Tuple[int, int, int]]:
    """Return ``(hand_id, command, address)`` for one valid request frame."""

    if (
        len(frame) < 8
        or frame[:2] != REQUEST_HEADER
        or len(frame) != frame[3] + 5
        or checksum(frame[:-1]) != frame[-1]
    ):
        return None
    hand_id = frame[2]
    if not 1 <= hand_id <= 254:
        return None
    command = frame[4]
    if command not in (READ_COMMAND, WRITE_COMMAND):
        return None
    if (
        (command == READ_COMMAND and frame[3] != 4)
        or (command == WRITE_COMMAND and frame[3] < 4)
    ):
        return None
    return hand_id, command, frame[5] | (frame[6] << 8)


def _exact_write_ack_identity(frame: bytes) -> Optional[Tuple[int, int]]:
    """Return ``(hand_id, address)`` only for a complete successful write ACK."""

    if (
        len(frame) != 9
        or frame[:2] != RESPONSE_HEADER
        or frame[3] != 4
        or checksum(frame[:-1]) != frame[-1]
        or not 1 <= frame[2] <= 254
        or frame[4] != WRITE_COMMAND
        or frame[7] != 0x01
    ):
        return None
    return frame[2], frame[5] | (frame[6] << 8)


def _is_exact_read_response(
    frame: bytes,
    hand_id: int,
    address: int,
    expected_data_length: int,
) -> bool:
    """Recognize a current read response before clearing older ACK ambiguity."""

    try:
        parse_response(
            frame,
            hand_id,
            READ_COMMAND,
            address,
            expected_data_length,
        )
        return True
    except RH56Error:
        return False


def find_serial_port() -> str:
    patterns = (
        "/dev/serial/by-id/*",
        "/dev/ttyUSB*",
        "/dev/ttyACM*",
    )
    ports: List[str] = []
    real_paths = set()
    for pattern in patterns:
        for candidate in sorted(glob.glob(pattern)):
            real_path = os.path.realpath(candidate)
            if real_path not in real_paths:
                real_paths.add(real_path)
                ports.append(candidate)
    if not ports:
        raise RH56Error(
            "no USB serial port found. For a CH340 shown by lsusb as 1a86:7523, "
            "check whether Ubuntu BRLTTY grabbed it before /dev/ttyUSB0 appeared"
        )
    if len(ports) > 1:
        raise RH56Error(
            "multiple serial ports found; pass --port explicitly: " + ", ".join(ports)
        )
    return ports[0]


class LinuxSerial:
    """Small 8N1 serial transport implemented with the Python standard library."""

    def __init__(self, port: str, baud: int, timeout: float, debug: bool = False):
        if baud not in BAUD_CONSTANTS:
            raise ValueError(f"unsupported baud rate: {baud}")
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.debug = debug
        self.fd: Optional[int] = None
        # The supervised transactional caller never retransmits a numeric
        # write.  If a complete write request was sent but its ACK was not
        # proven, retain one ACK debt for that device/register.  Matching ACKs
        # discharge debt in order; a strictly matching read response is an
        # ordering barrier for that hand's single serial stream.  Counting
        # debt prevents repeated same-address timeouts from letting ACK(W1/W2)
        # masquerade as ACK(W3).
        self._pending_late_write_acks: Dict[Tuple[int, int], int] = {}

    def __enter__(self) -> "LinuxSerial":
        try:
            self.fd = os.open(
                self.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK
            )
        except OSError as exc:
            raise RH56Error(f"cannot open {self.port}: {exc}") from exc

        try:
            if hasattr(termios, "TIOCEXCL"):
                fcntl.ioctl(self.fd, termios.TIOCEXCL)
            tty.setraw(self.fd, when=termios.TCSANOW)
            attrs = termios.tcgetattr(self.fd)
            speed = BAUD_CONSTANTS[self.baud]
            attrs[4] = speed
            attrs[5] = speed
            attrs[2] &= ~(termios.CSIZE | termios.PARENB | termios.CSTOPB)
            attrs[2] &= ~getattr(termios, "CRTSCTS", 0)
            attrs[2] |= termios.CS8 | termios.CREAD | termios.CLOCAL
            attrs[6][termios.VMIN] = 0
            attrs[6][termios.VTIME] = 0
            termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
            termios.tcflush(self.fd, termios.TCIOFLUSH)
        except Exception:
            os.close(self.fd)
            self.fd = None
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self._pending_late_write_acks.clear()

    def discard_input(self) -> None:
        """Discard unread response bytes without transmitting anything."""

        if self.fd is None:
            raise RH56Error("serial port is not open")
        termios.tcflush(self.fd, termios.TCIFLUSH)

    def exchange(self, request: bytes) -> bytes:
        if self.fd is None:
            raise RH56Error("serial port is not open")
        identity = _request_identity(request)
        write_key: Optional[Tuple[int, int]] = None
        if identity is not None and identity[1] == WRITE_COMMAND:
            write_key = (identity[0], identity[2])

        self.discard_input()
        if self.debug:
            print(f"TX: {hex_bytes(request)}")
        deadline = time.monotonic() + self.timeout
        buffer = bytearray()
        late_write_acks_skipped = 0
        write_may_have_been_sent = False
        write_debt_recorded = False

        def remember_unproven_write() -> None:
            nonlocal write_debt_recorded
            if (
                write_key is not None
                and write_may_have_been_sent
                and not write_debt_recorded
            ):
                self._pending_late_write_acks[write_key] = (
                    self._pending_late_write_acks.get(write_key, 0) + 1
                )
                write_debt_recorded = True

        def pop_buffered_response() -> Optional[bytes]:
            """Return the current reply, skipping proven late responses.

            This must be called immediately after every successful ``os.read``.
            A frame can become readable just before the select deadline while
            Python resumes just after it.  Deferring parsing to the next outer
            loop iteration would then report a timeout even though the complete
            response is already in ``buffer``.
            """

            nonlocal late_write_acks_skipped
            while True:
                header_at = buffer.find(RESPONSE_HEADER)
                if header_at < 0:
                    return None
                if header_at:
                    del buffer[:header_at]
                if len(buffer) < 4:
                    return None
                total_length = buffer[3] + 5
                if len(buffer) < total_length:
                    return None
                response = bytes(buffer[:total_length])
                del buffer[:total_length]
                ack_key = _exact_write_ack_identity(response)
                if (
                    ack_key is not None
                    and self._pending_late_write_acks.get(ack_key, 0) > 0
                ):
                    remaining_pending = (
                        self._pending_late_write_acks[ack_key] - 1
                    )
                    if remaining_pending:
                        self._pending_late_write_acks[ack_key] = (
                            remaining_pending
                        )
                    else:
                        del self._pending_late_write_acks[ack_key]
                    late_write_acks_skipped += 1
                    if self.debug:
                        print(
                            "RX (validated late write ACK, continuing): "
                            f"{hex_bytes(response)}"
                        )
                    continue
                if (
                    identity is not None
                    and identity[1] == READ_COMMAND
                    and len(request) >= 9
                    and _is_exact_read_response(
                        response,
                        identity[0],
                        identity[2],
                        request[7],
                    )
                ):
                    # An exact read response is an ordering barrier for every
                    # older transaction on this hand's single serial stream.
                    for pending_key in tuple(self._pending_late_write_acks):
                        if pending_key[0] == identity[0]:
                            del self._pending_late_write_acks[pending_key]
                elif write_key is not None and ack_key != write_key:
                    remember_unproven_write()
                if self.debug:
                    print(f"RX: {hex_bytes(response)}")
                return response

        try:
            view = memoryview(request)
            while view:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RH56Error("timeout while writing request to the hand")
                try:
                    # Mark the write conservatively before entering the kernel:
                    # an asynchronous exception immediately after os.write()
                    # returns must not lose the possible ACK debt.
                    if write_key is not None:
                        write_may_have_been_sent = True
                    written = os.write(self.fd, view)
                except BlockingIOError:
                    _, writable, _ = select.select([], [self.fd], [], remaining)
                    if not writable:
                        raise RH56Error(
                            "timeout while waiting for serial write readiness"
                        )
                    continue
                if written <= 0:
                    raise RH56Error("serial write made no progress")
                view = view[written:]

            while time.monotonic() < deadline:
                # A single os.read() may contain both the delayed write ACK
                # and the current read response.  Consume frames in-place so
                # the second frame is not lost when the first one is skipped.
                response = pop_buffered_response()
                if response is not None:
                    return response

                remaining = max(0.0, deadline - time.monotonic())
                readable, _, _ = select.select([self.fd], [], [], remaining)
                if not readable:
                    break
                try:
                    chunk = os.read(self.fd, 4096)
                except BlockingIOError:
                    continue
                if chunk:
                    buffer.extend(chunk)
                    # Do not defer this parse to the outer deadline test.  The
                    # complete frame may have arrived on the final select.
                    response = pop_buffered_response()
                    if response is not None:
                        return response
        except BaseException:
            remember_unproven_write()
            raise

        # Keep a final side-effect-free parse for injected transports where a
        # read can append bytes while advancing a synthetic deadline clock.
        response = pop_buffered_response()
        if response is not None:
            return response
        remember_unproven_write()
        received = hex_bytes(bytes(buffer)) if buffer else "<nothing>"
        late_context = (
            ""
            if not late_write_acks_skipped
            else (
                "; validated "
                + (
                    "one"
                    if late_write_acks_skipped == 1
                    else str(late_write_acks_skipped)
                )
                + " late write ACK"
                + ("s" if late_write_acks_skipped != 1 else "")
                + " while awaiting current response"
            )
        )
        raise RH56Error(
            f"timeout waiting for hand response; received {received}{late_context}"
        )


class RH56Hand:
    def __init__(self, serial_port: LinuxSerial, hand_id: int):
        self.serial = serial_port
        self.hand_id = hand_id

    def read(self, address: int, length: int, retries: int = 2) -> bytes:
        request = build_read_frame(self.hand_id, address, length)
        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                response = self.serial.exchange(request)
                return parse_response(
                    response, self.hand_id, READ_COMMAND, address, length
                )
            except RH56Error as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(0.05)
        raise RH56Error(f"read register {address} failed: {last_error}")

    def write(self, address: int, data: bytes, retries: int = 1) -> None:
        request = build_write_frame(self.hand_id, address, data)
        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                response = self.serial.exchange(request)
                ack = parse_response(
                    response, self.hand_id, WRITE_COMMAND, address, 1
                )
                if ack != b"\x01":
                    raise RH56Error(f"write register {address} returned ACK {ack.hex()}")
                return
            except RH56Error as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(0.05)
        raise RH56Error(f"write register {address} failed: {last_error}")

    def read_short(self, address: int) -> int:
        return struct.unpack("<h", self.read(address, 2))[0]

    def write_short(self, address: int, value: int) -> None:
        self.write(address, struct.pack("<h", value))

    def read_six_shorts(self, address: int, retries: int = 2) -> Tuple[int, ...]:
        return struct.unpack("<6h", self.read(address, 12, retries=retries))

    def write_six_shorts(
        self, address: int, values: Sequence[int], retries: int = 1
    ) -> None:
        values_tuple = tuple(int(value) for value in values)
        if len(values_tuple) != 6:
            raise ValueError("expected exactly six values")
        self.write(address, struct.pack("<6h", *values_tuple), retries=retries)

    def snapshot(self) -> Dict[str, object]:
        return {
            "hand_id": self.read(REG_HAND_ID, 1)[0],
            "baud_setting": self.read(REG_BAUD, 1)[0],
            "voltage_raw": self.read_short(REG_VOLTAGE),
            "position_targets": self.read_six_shorts(REG_POS_SET),
            "angle_targets": self.read_six_shorts(REG_ANGLE_SET),
            "force_limits": self.read_six_shorts(REG_FORCE_SET),
            "speeds": self.read_six_shorts(REG_SPEED_SET),
            "current_limits": self.read_six_shorts(REG_CURRENT_LIMIT),
            "positions": self.read_six_shorts(REG_POS_ACT),
            "angles": self.read_six_shorts(REG_ANGLE_ACT),
            "forces": self.read_six_shorts(REG_FORCE_ACT),
            "currents": self.read_six_shorts(REG_CURRENT),
            "errors": tuple(self.read(REG_ERROR, 6)),
            "statuses": tuple(self.read(REG_STATUS, 6)),
            "temperatures": tuple(self.read(REG_TEMP, 6)),
        }


def decode_error(value: int) -> str:
    if value == 0:
        return "正常"
    labels = [label for bit, label in ERROR_BITS if value & (1 << bit)]
    unknown = value & ~0x1F
    if unknown:
        labels.append(f"未知位0x{unknown:02X}")
    return "/".join(labels)


def print_snapshot(snapshot: Dict[str, object], port: str, baud: int) -> None:
    baud_names = {0: 115200, 1: 57600, 2: 19200}
    baud_setting = int(snapshot["baud_setting"])
    configured_baud = baud_names.get(baud_setting, f"未知({baud_setting})")
    print(f"已连接 Inspire RH56: port={port}, ID={snapshot['hand_id']}, baud={baud}")
    print(f"设备波特率配置寄存器: {configured_baud}")
    print(f"系统电压寄存器原始值: {snapshot['voltage_raw']}")
    print(f"电缸位置目标: {list(snapshot['position_targets'])}")
    print(f"目标角度: {list(snapshot['angle_targets'])}")
    print(f"速度设置: {list(snapshot['speeds'])}")
    print(f"力阈值(g): {list(snapshot['force_limits'])}")
    print(f"电流保护值(mA): {list(snapshot['current_limits'])}")
    print()
    print("自由度       角度值  电缸位置  受力(g)  电流(mA)  温度(°C)  状态          故障")
    print("-----------  ------  --------  -------  --------  --------  ------------  --------")
    rows = zip(
        JOINT_LABELS,
        snapshot["angles"],
        snapshot["positions"],
        snapshot["forces"],
        snapshot["currents"],
        snapshot["temperatures"],
        snapshot["statuses"],
        snapshot["errors"],
    )
    for label, angle, position, force, current, temperature, status, error in rows:
        status_text = STATUS_LABELS.get(status, f"未知({status})")
        print(
            f"{label:<11}  {angle:>6}  {position:>8}  {force:>7}  {current:>8}  "
            f"{temperature:>8}  {status_text:<12}  {decode_error(error)}"
        )


def clear_latched_error(
    hand: RH56Hand,
    initial_snapshot: Mapping[str, object],
) -> Dict[str, object]:
    """Clear a latched RH56 fault only from a proved disabled, idle state.

    Register 1004 and the one-byte value ``1`` are the vendor's documented
    CLEAR_ERROR operation. It is not a motion command. We nevertheless require
    all six ANGLE_SET registers to already be disabled and reject any active
    current, heat, error bit, or moving/contact state before the single write.
    """

    def six_ints(name: str, snapshot: Mapping[str, object]) -> Tuple[int, ...]:
        values = tuple(int(value) for value in snapshot[name])
        if len(values) != len(JOINTS):
            raise RH56Error(f"{name} must contain exactly six values")
        return values

    initial_targets = six_ints("angle_targets", initial_snapshot)
    initial_angles = six_ints("angles", initial_snapshot)
    initial_positions = six_ints("positions", initial_snapshot)
    initial_currents = six_ints("currents", initial_snapshot)
    initial_temperatures = six_ints("temperatures", initial_snapshot)
    initial_errors = six_ints("errors", initial_snapshot)
    initial_statuses = six_ints("statuses", initial_snapshot)

    if initial_targets != (-1,) * len(JOINTS):
        raise RH56Error(
            "clear-error refused: all six ANGLE_SET targets must already be -1"
        )
    if any(abs(value) > CLEAR_ERROR_MAX_IDLE_CURRENT_MA for value in initial_currents):
        raise RH56Error(
            "clear-error refused: idle current exceeds "
            f"{CLEAR_ERROR_MAX_IDLE_CURRENT_MA}mA: {initial_currents}"
        )
    if any(value >= 60 for value in initial_temperatures):
        raise RH56Error(
            f"clear-error refused: actuator temperature is unsafe: {initial_temperatures}"
        )
    if any(initial_errors):
        raise RH56Error(
            "clear-error refused: ERROR bits are still active; remove the "
            f"underlying fault first: {initial_errors}"
        )
    if any(status in (0, 1, 3) for status in initial_statuses):
        raise RH56Error(
            "clear-error refused: an actuator is moving or force-stopped: "
            f"{initial_statuses}"
        )
    if any(status not in (2, 5, 6, 7, 0xFF) for status in initial_statuses):
        raise RH56Error(
            f"clear-error refused: unexpected actuator status: {initial_statuses}"
        )
    if not any(status in (5, 6, 7) for status in initial_statuses):
        raise RH56Error("clear-error refused: no latched fault/protection status exists")

    hand.write(REG_CLEAR_ERROR, b"\x01", retries=0)
    time.sleep(0.10)
    final_snapshot = hand.snapshot()

    final_targets = six_ints("angle_targets", final_snapshot)
    final_angles = six_ints("angles", final_snapshot)
    final_positions = six_ints("positions", final_snapshot)
    final_currents = six_ints("currents", final_snapshot)
    final_temperatures = six_ints("temperatures", final_snapshot)
    final_errors = six_ints("errors", final_snapshot)
    final_statuses = six_ints("statuses", final_snapshot)

    if final_targets != (-1,) * len(JOINTS):
        raise RH56Error(
            f"clear-error post-check failed: ANGLE_SET changed: {final_targets}"
        )
    if any(abs(value) > CLEAR_ERROR_MAX_IDLE_CURRENT_MA for value in final_currents):
        raise RH56Error(
            f"clear-error post-check failed: current is not idle: {final_currents}"
        )
    if any(value >= 60 for value in final_temperatures):
        raise RH56Error(
            f"clear-error post-check failed: temperature is unsafe: {final_temperatures}"
        )
    if any(final_errors):
        raise RH56Error(
            f"clear-error post-check failed: ERROR remains: {final_errors}"
        )
    if any(status not in (2, 0xFF) for status in final_statuses):
        raise RH56Error(
            f"clear-error post-check failed: STATUS remains non-idle: {final_statuses}"
        )
    angle_drift = tuple(
        abs(after - before) for before, after in zip(initial_angles, final_angles)
    )
    position_drift = tuple(
        abs(after - before)
        for before, after in zip(initial_positions, final_positions)
    )
    if any(value > CLEAR_ERROR_MAX_ANGLE_DRIFT for value in angle_drift):
        raise RH56Error(
            "clear-error post-check failed: unexpected ANGLE_ACT movement: "
            f"{angle_drift}"
        )
    if any(value > CLEAR_ERROR_MAX_POSITION_DRIFT for value in position_drift):
        raise RH56Error(
            "clear-error post-check failed: unexpected POS_ACT movement: "
            f"{position_drift}"
        )
    return final_snapshot


def recover_stale_thumb_rotate_status(
    hand: RH56Hand,
    initial_snapshot: Mapping[str, object],
) -> Dict[str, object]:
    """Relatch a cleared-but-stale q6 STATUS=7 with a bounded pose hold.

    This is deliberately narrower than an ordinary nudge.  It accepts only a
    disabled, current-free, error-free and motionless hand whose sole non-idle
    status is thumb rotation 7.  The only numeric target is the calibrated
    feedback-equivalent q6 pose.  Conservative speed/current settings are
    installed temporarily, then every target is disabled and all settings are
    restored regardless of success.
    """

    def six_ints(name: str, snapshot: Mapping[str, object]) -> Tuple[int, ...]:
        values = tuple(int(value) for value in snapshot[name])
        if len(values) != len(JOINTS):
            raise RH56Error(f"{name} must contain exactly six values")
        return values

    def validate_seed(snapshot: Mapping[str, object], phase: str) -> None:
        targets = six_ints("angle_targets", snapshot)
        angles = six_ints("angles", snapshot)
        currents = six_ints("currents", snapshot)
        temperatures = six_ints("temperatures", snapshot)
        errors = six_ints("errors", snapshot)
        statuses = six_ints("statuses", snapshot)
        forces = six_ints("forces", snapshot)
        if targets != (-1,) * len(JOINTS):
            raise RH56Error(f"{phase}: all six ANGLE_SET targets must be -1")
        if any(abs(value) > CLEAR_ERROR_MAX_IDLE_CURRENT_MA for value in currents):
            raise RH56Error(f"{phase}: hand is not current-idle: {currents}")
        if any(value >= 60 for value in temperatures):
            raise RH56Error(f"{phase}: temperature is unsafe: {temperatures}")
        if any(errors):
            raise RH56Error(f"{phase}: ERROR bits remain active: {errors}")
        if statuses[:5] != (2, 2, 2, 2, 2) or statuses[5] != 7:
            raise RH56Error(
                f"{phase}: expected only thumb_rotate STATUS=7: {statuses}"
            )
        if max(abs(value) for value in forces) > STALE_Q6_RECOVERY_MAX_FORCE_ABS_G:
            raise RH56Error(f"{phase}: hand is not contact-free: {forces}")
        if angles[3] < STALE_Q6_RECOVERY_MIN_INDEX_ANGLE:
            raise RH56Error(
                f"{phase}: index clearance is insufficient: {angles[3]}"
            )
        if angles[4] < STALE_Q6_RECOVERY_MIN_THUMB_BEND_ANGLE:
            raise RH56Error(
                f"{phase}: thumb-bend clearance is insufficient: {angles[4]}"
            )

    validate_seed(initial_snapshot, "stale-q6 recovery seed")
    stable_samples = [initial_snapshot]
    for _ in range(2):
        time.sleep(0.10)
        sample = hand.snapshot()
        validate_seed(sample, "stale-q6 recovery stability sample")
        stable_samples.append(sample)
    for axis, name in enumerate(JOINTS):
        angles = [six_ints("angles", sample)[axis] for sample in stable_samples]
        positions = [six_ints("positions", sample)[axis] for sample in stable_samples]
        if max(angles) - min(angles) > 2 or max(positions) - min(positions) > 3:
            raise RH56Error(
                f"stale-q6 recovery refused: {name} is not stationary; "
                f"angles={angles} positions={positions}"
            )

    seed = stable_samples[-1]
    seed_angles = six_ints("angles", seed)
    seed_positions = six_ints("positions", seed)
    hold_target = seed_angles[5] + THUMB_ROTATE_FEEDBACK_TO_COMMAND_OFFSET
    if not STALE_Q6_RECOVERY_MIN_COMMAND <= hold_target <= STALE_Q6_RECOVERY_MAX_COMMAND:
        raise RH56Error(
            "stale-q6 recovery refused: feedback-equivalent q6 target "
            f"{hold_target} is outside {STALE_Q6_RECOVERY_MIN_COMMAND}.."
            f"{STALE_Q6_RECOVERY_MAX_COMMAND}"
        )

    original_speeds = six_ints("speeds", seed)
    original_forces = six_ints("force_limits", seed)
    original_current_limits = six_ints("current_limits", seed)
    q6_speed_address = REG_SPEED_SET + 2 * 5
    q6_force_address = REG_FORCE_SET + 2 * 5
    q6_current_limit_address = REG_CURRENT_LIMIT + 2 * 5
    hold_targets = (-1, -1, -1, -1, -1, hold_target)
    operation_error: Optional[BaseException] = None
    cleanup_errors: List[str] = []
    reached_idle = False

    try:
        hand.write_short(q6_speed_address, STALE_Q6_RECOVERY_SPEED)
        hand.write_short(q6_force_address, STALE_Q6_RECOVERY_FORCE_G)
        hand.write_short(
            q6_current_limit_address, STALE_Q6_RECOVERY_CURRENT_LIMIT_MA
        )
        if hand.read_short(q6_speed_address) != STALE_Q6_RECOVERY_SPEED:
            raise RH56Error("stale-q6 recovery speed readback mismatch")
        if hand.read_short(q6_force_address) != STALE_Q6_RECOVERY_FORCE_G:
            raise RH56Error("stale-q6 recovery force readback mismatch")
        if (
            hand.read_short(q6_current_limit_address)
            != STALE_Q6_RECOVERY_CURRENT_LIMIT_MA
        ):
            raise RH56Error("stale-q6 recovery current-limit readback mismatch")

        hand.write(REG_CLEAR_ERROR, b"\x01", retries=0)
        hand.write_six_shorts(REG_ANGLE_SET, hold_targets, retries=1)
        if tuple(hand.read_six_shorts(REG_ANGLE_SET, retries=1)) != hold_targets:
            raise RH56Error("stale-q6 recovery hold-target readback mismatch")

        deadline = time.monotonic() + STALE_Q6_RECOVERY_TIMEOUT_S
        while True:
            sample = hand.snapshot()
            targets = six_ints("angle_targets", sample)
            angles = six_ints("angles", sample)
            positions = six_ints("positions", sample)
            currents = six_ints("currents", sample)
            temperatures = six_ints("temperatures", sample)
            errors = six_ints("errors", sample)
            statuses = six_ints("statuses", sample)
            if targets != hold_targets:
                raise RH56Error(
                    f"stale-q6 recovery target changed unexpectedly: {targets}"
                )
            if any(errors):
                raise RH56Error(f"stale-q6 recovery ERROR appeared: {errors}")
            if any(value >= 60 for value in temperatures):
                raise RH56Error(
                    f"stale-q6 recovery temperature is unsafe: {temperatures}"
                )
            if abs(currents[5]) > STALE_Q6_RECOVERY_CURRENT_LIMIT_MA:
                raise RH56Error(
                    f"stale-q6 recovery q6 current exceeded limit: {currents}"
                )
            if sum(abs(value) for value in currents) > 200:
                raise RH56Error(
                    f"stale-q6 recovery total current exceeded limit: {currents}"
                )
            if any(
                abs(angles[index] - seed_angles[index]) > 3
                or abs(positions[index] - seed_positions[index]) > 5
                for index in range(5)
            ):
                raise RH56Error(
                    "stale-q6 recovery moved an inactive axis: "
                    f"angles={angles} positions={positions}"
                )
            if (
                abs(angles[5] - seed_angles[5])
                > STALE_Q6_RECOVERY_MAX_ANGLE_DRIFT
                or abs(positions[5] - seed_positions[5])
                > STALE_Q6_RECOVERY_MAX_POSITION_DRIFT
            ):
                raise RH56Error(
                    "stale-q6 recovery exceeded its in-place drift bound: "
                    f"angles={angles[5]}/{seed_angles[5]} "
                    f"positions={positions[5]}/{seed_positions[5]}"
                )
            if statuses[:5] != (2, 2, 2, 2, 2):
                raise RH56Error(
                    f"stale-q6 recovery changed an inactive status: {statuses}"
                )
            if statuses[5] == 2:
                reached_idle = True
                break
            if statuses[5] not in (0, 1, 7):
                raise RH56Error(
                    f"stale-q6 recovery returned unsafe q6 status: {statuses[5]}"
                )
            if time.monotonic() >= deadline:
                raise RH56Error(
                    "stale-q6 recovery timed out without clearing STATUS=7"
                )
            time.sleep(0.05)
    except BaseException as exc:
        operation_error = exc
    finally:
        try:
            _disable_all_targets_verified(hand)
        except BaseException as exc:
            cleanup_errors.append(f"disable failed: {exc}")
        for address, value, name in (
            (q6_speed_address, original_speeds[5], "speed"),
            (q6_force_address, original_forces[5], "force"),
            (q6_current_limit_address, original_current_limits[5], "current limit"),
        ):
            try:
                hand.write_short(address, value)
                if hand.read_short(address) != value:
                    raise RH56Error(f"{name} restore readback mismatch")
            except BaseException as exc:
                cleanup_errors.append(f"{name} restore failed: {exc}")

    if cleanup_errors:
        raise RH56Error(
            "stale-q6 recovery cleanup failed: " + "; ".join(cleanup_errors)
        ) from operation_error
    if operation_error is not None:
        raise RH56Error(f"stale-q6 recovery failed: {operation_error}") from operation_error
    if not reached_idle:
        raise RH56Error("stale-q6 recovery did not establish idle status")

    final_snapshot = hand.snapshot()
    final_targets = six_ints("angle_targets", final_snapshot)
    final_currents = six_ints("currents", final_snapshot)
    final_errors = six_ints("errors", final_snapshot)
    final_statuses = six_ints("statuses", final_snapshot)
    if (
        final_targets != (-1,) * len(JOINTS)
        or any(abs(value) > CLEAR_ERROR_MAX_IDLE_CURRENT_MA for value in final_currents)
        or any(final_errors)
        or any(status not in (2, 0xFF) for status in final_statuses)
    ):
        raise RH56Error(
            "stale-q6 recovery final disabled/idle proof failed: "
            f"targets={final_targets} currents={final_currents} "
            f"errors={final_errors} statuses={final_statuses}"
        )
    return final_snapshot


def _disable_all_targets_verified(hand: RH56Hand) -> None:
    """Best-effort motion stop used by diagnostic motion recovery paths."""

    disabled = (-1,) * len(JOINTS)
    failures: List[str] = []
    for verification_pass in (1, 2):
        try:
            hand.write_six_shorts(REG_ANGLE_SET, disabled, retries=1)
            readback = tuple(hand.read_six_shorts(REG_ANGLE_SET, retries=1))
            if readback != disabled:
                raise RH56Error(f"readback={readback}")
        except BaseException as exc:
            failures.append(f"pass {verification_pass}: {exc}")
    if failures:
        raise RH56Error(
            "failed to verify all-six ANGLE_SET=-1: " + "; ".join(failures)
        )


def _joint_motion_telemetry(
    hand: RH56Hand,
    joint_index: int,
    *,
    angle_actual: Optional[int] = None,
) -> Dict[str, int]:
    """Read the selected axis registers needed to diagnose one motion leg."""

    return {
        "angle_set": hand.read_short(REG_ANGLE_SET + 2 * joint_index),
        "angle_actual": (
            hand.read_short(REG_ANGLE_ACT + 2 * joint_index)
            if angle_actual is None
            else int(angle_actual)
        ),
        "position_set": hand.read_short(REG_POS_SET + 2 * joint_index),
        "position_actual": hand.read_short(REG_POS_ACT + 2 * joint_index),
        "force_actual": hand.read_short(REG_FORCE_ACT + 2 * joint_index),
    }


def wait_for_angle(
    hand: RH56Hand,
    joint_index: int,
    target: int,
    timeout: float = 12.0,
    tolerance: int = POSITION_TOLERANCE,
    allow_undefined_status: bool = False,
    monitor_all: bool = False,
    diagnostic_trace: bool = False,
    max_joint_current_ma: Optional[int] = None,
    max_total_current_ma: Optional[int] = None,
    inactive_reference_angles: Optional[Sequence[int]] = None,
    inactive_reference_positions: Optional[Sequence[int]] = None,
    max_inactive_angle_drift: Optional[int] = None,
    max_inactive_position_drift: Optional[int] = None,
    direction_reference_angle: Optional[int] = None,
    required_stable_samples: int = 1,
    endpoint_max_joint_current_ma: Optional[int] = None,
    endpoint_max_total_current_ma: Optional[int] = None,
    safety_check: Optional[Callable[[], None]] = None,
) -> int:
    if required_stable_samples < 1:
        raise ValueError("required_stable_samples must be positive")
    started_at = time.monotonic()
    deadline = time.monotonic() + timeout
    next_trace_at = started_at
    angle_address = REG_ANGLE_ACT + 2 * joint_index
    consecutive_reached = 0
    previous_reached_actual: Optional[int] = None
    previous_reached_position: Optional[int] = None
    while True:
        if safety_check is not None:
            safety_check()
        now = time.monotonic()
        if inactive_reference_angles is not None:
            actual_angles = tuple(
                hand.read_six_shorts(REG_ANGLE_ACT, retries=0)
            )
            actual = actual_angles[joint_index]
        else:
            actual_angles = None
            actual = hand.read_short(angle_address)
        if inactive_reference_positions is not None:
            actual_positions = tuple(
                hand.read_six_shorts(REG_POS_ACT, retries=0)
            )
        else:
            actual_positions = None
        if actual_angles is not None and max_inactive_angle_drift is not None:
            for index, (baseline, observed) in enumerate(
                zip(inactive_reference_angles, actual_angles)
            ):
                if index == joint_index:
                    continue
                drift = abs(int(observed) - int(baseline))
                if drift > max_inactive_angle_drift:
                    raise RH56Error(
                        f"inactive joint {JOINT_LABELS[index]} ANGLE_ACT drifted "
                        f"by {drift} while {JOINT_LABELS[joint_index]} was moving"
                    )
        if (
            actual_positions is not None
            and max_inactive_position_drift is not None
        ):
            for index, (baseline, observed) in enumerate(
                zip(inactive_reference_positions, actual_positions)
            ):
                if index == joint_index:
                    continue
                drift = abs(int(observed) - int(baseline))
                if drift > max_inactive_position_drift:
                    raise RH56Error(
                        f"inactive joint {JOINT_LABELS[index]} POS_ACT drifted "
                        f"by {drift} while {JOINT_LABELS[joint_index]} was moving"
                    )
        currents: Optional[Tuple[int, ...]] = None
        if monitor_all:
            errors = tuple(hand.read(REG_ERROR, 6))
            statuses = tuple(hand.read(REG_STATUS, 6))
            temperatures = tuple(hand.read(REG_TEMP, 6))
            currents = tuple(hand.read_six_shorts(REG_CURRENT, retries=0))
            error = errors[joint_index]
            status = statuses[joint_index]
            temperature = temperatures[joint_index]
            if any(errors):
                fault_index = next(
                    index for index, value in enumerate(errors) if value
                )
                raise RH56Error(
                    f"{JOINT_LABELS[fault_index]} reported "
                    f"{decode_error(errors[fault_index])} while "
                    f"{JOINT_LABELS[joint_index]} was moving to {target}"
                )
            hottest = max(temperatures)
            if hottest >= 60:
                hot_index = temperatures.index(hottest)
                raise RH56Error(
                    f"{JOINT_LABELS[hot_index]} reached {hottest} °C while "
                    f"{JOINT_LABELS[joint_index]} was moving to {target}"
                )
            for index, other_status in enumerate(statuses):
                if index != joint_index and other_status in (0, 1):
                    raise RH56Error(
                        f"inactive joint {JOINT_LABELS[index]} unexpectedly started moving"
                    )
        else:
            error = hand.read(REG_ERROR + joint_index, 1)[0]
            status = hand.read(REG_STATUS + joint_index, 1)[0]
            temperature = hand.read(REG_TEMP + joint_index, 1)[0]
            if diagnostic_trace or max_joint_current_ma is not None:
                currents = tuple(hand.read_six_shorts(REG_CURRENT, retries=0))
        if not monitor_all and error:
            raise RH56Error(
                f"{JOINT_LABELS[joint_index]} reported {decode_error(error)} "
                f"while moving to {target}"
            )
        if not monitor_all and temperature >= 60:
            raise RH56Error(
                f"{JOINT_LABELS[joint_index]} reached {temperature} °C "
                f"while moving to {target}"
            )
        selected_current = (
            int(currents[joint_index]) if currents is not None else 0
        )
        total_current = (
            sum(abs(int(value)) for value in currents)
            if currents is not None
            else 0
        )
        if (
            max_joint_current_ma is not None
            and abs(selected_current) > max_joint_current_ma
        ):
            raise RH56Error(
                f"{JOINT_LABELS[joint_index]} current exceeded "
                f"{max_joint_current_ma} mA while moving to {target}: "
                f"current={selected_current} mA"
            )
        if (
            max_total_current_ma is not None
            and total_current > max_total_current_ma
        ):
            raise RH56Error(
                f"all-axis total current exceeded {max_total_current_ma} mA "
                f"while moving {JOINT_LABELS[joint_index]} to {target}: "
                f"total={total_current} mA"
            )
        if direction_reference_angle is not None:
            commanded_delta = target - direction_reference_angle
            observed_delta = actual - direction_reference_angle
            if (
                commanded_delta
                and abs(observed_delta) >= PROBE_MIN_ANGLE_MOVEMENT
                and observed_delta * commanded_delta < 0
            ):
                raise RH56Error(
                    f"{JOINT_LABELS[joint_index]} moved in the wrong direction "
                    f"while moving to {target}: start={direction_reference_angle}, "
                    f"actual={actual}"
                )
        reached_target = (
            abs(actual - target) <= tolerance
            and (status == 2 or (allow_undefined_status and status == 0xFF))
            and (
                endpoint_max_joint_current_ma is None
                or abs(selected_current) <= endpoint_max_joint_current_ma
            )
            and (
                endpoint_max_total_current_ma is None
                or total_current <= endpoint_max_total_current_ma
            )
        )
        selected_position = (
            int(actual_positions[joint_index])
            if actual_positions is not None
            else None
        )
        feedback_stable = (
            previous_reached_actual is None
            or (
                abs(actual - previous_reached_actual)
                <= THUMB_ROTATE_VISUAL_STABILITY_DELTA
                and (
                    selected_position is None
                    or previous_reached_position is None
                    or abs(selected_position - previous_reached_position)
                    <= THUMB_ROTATE_FULL_CYCLE_STOP_POSITION_DRIFT
                )
            )
        )
        if reached_target and feedback_stable:
            consecutive_reached += 1
        elif reached_target:
            consecutive_reached = 1
        else:
            consecutive_reached = 0
        previous_reached_actual = actual if reached_target else None
        previous_reached_position = selected_position if reached_target else None
        endpoint_confirmed = consecutive_reached >= required_stable_samples
        if diagnostic_trace and (
            now >= next_trace_at or endpoint_confirmed or now >= deadline
        ):
            telemetry = _joint_motion_telemetry(
                hand, joint_index, angle_actual=actual
            )
            print(
                "[trace] "
                f"t={now - started_at:.2f}s "
                f"ANGLE_SET/ACT={telemetry['angle_set']}/{actual} "
                f"POS_SET/ACT={telemetry['position_set']}/"
                f"{telemetry['position_actual']} "
                f"FORCE={telemetry['force_actual']}g "
                f"CURRENT={selected_current}mA TOTAL={total_current}mA "
                f"STATUS={status} ERROR={error} TEMP={temperature}C"
            )
            next_trace_at = now + 0.50
        if endpoint_confirmed:
            return actual
        if status in (3, 5, 6, 7):
            status_text = STATUS_LABELS.get(status, str(status))
            raise RH56Error(
                f"{JOINT_LABELS[joint_index]} stopped before target: {status_text}, "
                f"actual={actual}, target={target}"
            )
        allowed_statuses = (0, 1, 2, 0xFF) if allow_undefined_status else (0, 1, 2)
        if status not in allowed_statuses:
            raise RH56Error(
                f"{JOINT_LABELS[joint_index]} returned unknown status {status}"
            )
        if now >= deadline:
            telemetry = _joint_motion_telemetry(
                hand, joint_index, angle_actual=actual
            )
            raise RH56Error(
                f"{JOINT_LABELS[joint_index]} did not reach {target} within "
                f"{timeout:.1f}s; ANGLE_SET/ACT="
                f"{telemetry['angle_set']}/{actual}, POS_SET/ACT="
                f"{telemetry['position_set']}/{telemetry['position_actual']}, "
                f"force={telemetry['force_actual']}g, "
                f"current={selected_current}mA, status={status}"
            )
        time.sleep(0.08)


def move_joint_path(
    hand: RH56Hand,
    joint_index: int,
    speed: int,
    force_limit: int,
    motion_timeout: float,
    title: str,
    build_path: Callable[[int, int], Sequence[Tuple[str, int]]],
    require_disabled_idle: bool = False,
    monitor_all: bool = False,
    stop_in_place_on_error: bool = False,
    diagnostic_trace: bool = False,
    max_joint_current_ma: Optional[int] = None,
    max_total_current_ma: Optional[int] = None,
    verify_all_disabled_on_exit: bool = False,
    batch_angle_write: bool = False,
    position_tolerance: Optional[int] = None,
    verify_inactive_axes_stationary: bool = False,
    return_target_from_actual: Optional[Callable[[int], int]] = None,
    verify_direction_during_motion: bool = False,
    restore_settings_on_exit: bool = True,
    endpoint_stable_samples: int = 1,
    endpoint_max_joint_current_ma: Optional[int] = None,
    endpoint_max_total_current_ma: Optional[int] = None,
    safety_check: Optional[Callable[[], None]] = None,
) -> None:
    if safety_check is not None:
        safety_check()
    snapshot = hand.snapshot()
    errors = snapshot["errors"]
    statuses = snapshot["statuses"]
    temperatures = snapshot["temperatures"]
    angle_targets = snapshot["angle_targets"]
    if any(errors):
        raise RH56Error("movement refused because at least one actuator reports a fault")
    position_idle = all(status == 2 for status in statuses)
    disabled_idle = all(target == -1 for target in angle_targets) and all(
        status in (2, 0xFF) for status in statuses
    )
    if require_disabled_idle and not disabled_idle:
        raise RH56Error(
            "full sweep requires all ANGLE_SET=-1 and each STATUS in {2, 0xFF}"
        )
    if not require_disabled_idle and not (position_idle or disabled_idle):
        details = ", ".join(
            f"{JOINT_LABELS[index]}={STATUS_LABELS.get(status, f'unknown({status})')}"
            for index, status in enumerate(statuses)
        )
        raise RH56Error(
            "movement refused: expected all STATUS=2, or all ANGLE_SET=-1 "
            "with each STATUS in {2, 0xFF}; "
            + details
        )
    if max(temperatures) >= 60:
        raise RH56Error("movement refused because an actuator is at least 60 °C")
    if require_disabled_idle and any(
        not 0 <= angle <= 1000 for angle in snapshot["angles"]
    ):
        raise RH56Error("full sweep refused because an actual angle is outside 0..1000")

    angle_address = REG_ANGLE_ACT + 2 * joint_index
    target_address = REG_ANGLE_SET + 2 * joint_index
    force_address = REG_FORCE_SET + 2 * joint_index
    speed_address = REG_SPEED_SET + 2 * joint_index
    original_angle = hand.read_short(angle_address)
    original_target = hand.read_short(target_address)
    original_force = hand.read_short(force_address)
    original_speed = hand.read_short(speed_address)
    if original_target != -1 and not 0 <= original_target <= 1000:
        raise RH56Error(
            f"movement refused because original target is {original_target}, not -1 or 0..1000"
        )
    if original_target == -1 and return_target_from_actual is not None:
        return_target = int(return_target_from_actual(original_angle))
    else:
        return_target = original_angle if original_target == -1 else original_target
    if not 0 <= return_target <= 1000:
        raise RH56Error(
            f"movement refused because return target is {return_target}, not 0..1000"
        )
    path = tuple(build_path(original_angle, return_target))
    if not path:
        raise ValueError("movement path must contain at least one waypoint")
    if path[-1][1] != return_target:
        raise ValueError("movement path must finish at the original position target")
    for _, target in path:
        if not 0 <= target <= 1000:
            raise ValueError(f"movement target {target} is outside 0..1000")
    path_text = " -> ".join(str(target) for _, target in path)
    print(
        f"{title} {JOINT_LABELS[joint_index]}: "
        f"actual={original_angle} -> {path_text}, "
        f"final_target={original_target}, "
        f"speed={speed}, force_limit={force_limit}g"
    )

    def write_angle_target_verified(value: int) -> None:
        if batch_angle_write:
            expected = tuple(
                value if index == joint_index else -1
                for index in range(len(JOINTS))
            )
            hand.write_six_shorts(REG_ANGLE_SET, expected, retries=1)
            readback = tuple(
                hand.read_six_shorts(REG_ANGLE_SET, retries=1)
            )
            if readback != expected:
                raise RH56Error(
                    "batch ANGLE_SET readback mismatch: "
                    f"wrote={expected}, readback={readback}"
                )
            return
        hand.write_short(target_address, value)
        readback = hand.read_short(target_address)
        if readback != value:
            raise RH56Error(
                f"ANGLE_SET readback mismatch: wrote={value}, "
                f"readback={readback}"
            )

    speed_change_attempted = False
    force_change_attempted = False
    target_change_attempted = False
    target_restored = False
    operation_error: Optional[BaseException] = None
    try:
        speed_change_attempted = True
        hand.write_short(speed_address, speed)
        force_change_attempted = True
        hand.write_short(force_address, force_limit)
        target_change_attempted = True
        for waypoint_index, (label, target) in enumerate(path):
            if safety_check is not None:
                safety_check()
            leg_start = hand.read_short(angle_address)
            tolerance = position_tolerance if position_tolerance is not None else (
                SWEEP_ENDPOINT_TOLERANCE
                if require_disabled_idle and target in (0, 1000)
                else (
                    SWEEP_POSITION_TOLERANCE
                    if require_disabled_idle
                    else POSITION_TOLERANCE
                )
            )
            write_angle_target_verified(target)
            reached = wait_for_angle(
                hand,
                joint_index,
                target,
                timeout=motion_timeout,
                tolerance=tolerance,
                allow_undefined_status=(
                    disabled_idle and abs(leg_start - target) <= tolerance
                ),
                monitor_all=monitor_all,
                diagnostic_trace=diagnostic_trace,
                max_joint_current_ma=max_joint_current_ma,
                max_total_current_ma=max_total_current_ma,
                inactive_reference_angles=(
                    snapshot["angles"] if verify_inactive_axes_stationary else None
                ),
                inactive_reference_positions=(
                    snapshot["positions"] if verify_inactive_axes_stationary else None
                ),
                max_inactive_angle_drift=(
                    NUDGE_MAX_INACTIVE_ANGLE_DRIFT
                    if verify_inactive_axes_stationary
                    else None
                ),
                max_inactive_position_drift=(
                    NUDGE_MAX_INACTIVE_POSITION_DRIFT
                    if verify_inactive_axes_stationary
                    else None
                ),
                direction_reference_angle=(
                    leg_start if verify_direction_during_motion else None
                ),
                required_stable_samples=endpoint_stable_samples,
                endpoint_max_joint_current_ma=endpoint_max_joint_current_ma,
                endpoint_max_total_current_ma=endpoint_max_total_current_ma,
                safety_check=safety_check,
            )
            commanded_delta = target - leg_start
            observed_delta = reached - leg_start
            if commanded_delta and observed_delta * commanded_delta <= 0:
                raise RH56Error(
                    f"{JOINT_LABELS[joint_index]} moved in the wrong direction: "
                    f"start={leg_start}, target={target}, actual={reached}"
                )
            print(f"{label}: target={target}, actual={reached}")
            if waypoint_index + 1 < len(path):
                time.sleep(0.3)
        if original_target == -1:
            write_angle_target_verified(-1)
        if verify_all_disabled_on_exit:
            _disable_all_targets_verified(hand)
            print("[safety] verified final all-six ANGLE_SET=-1")
        target_restored = True
    except BaseException as exc:
        operation_error = exc
    finally:
        recovery_errors: List[str] = []
        if target_change_attempted and not target_restored:
            try:
                if stop_in_place_on_error:
                    stop_target = hand.read_short(angle_address)
                    write_angle_target_verified(stop_target)
                    time.sleep(0.2)
                    write_angle_target_verified(-1)
                elif not verify_all_disabled_on_exit:
                    write_angle_target_verified(return_target)
                    wait_for_angle(
                        hand,
                        joint_index,
                        return_target,
                        timeout=motion_timeout,
                        allow_undefined_status=disabled_idle,
                        monitor_all=monitor_all,
                    )
                    if original_target == -1:
                        write_angle_target_verified(-1)
            except BaseException as exc:
                recovery_errors.append(f"目标角度恢复失败: {exc}")
            if verify_all_disabled_on_exit:
                try:
                    _disable_all_targets_verified(hand)
                    print(
                        "[recovery] motion output disabled; verified all-six "
                        "ANGLE_SET=-1"
                    )
                except BaseException as exc:
                    recovery_errors.append(f"六轴禁用失败: {exc}")
            try:
                if verify_all_disabled_on_exit:
                    restored_targets = tuple(
                        hand.read_six_shorts(REG_ANGLE_SET, retries=1)
                    )
                    if restored_targets != (-1,) * len(JOINTS):
                        raise RH56Error(
                            f"readback={restored_targets}, expected all -1"
                        )
                else:
                    restored_target = hand.read_short(target_address)
                    if restored_target != original_target:
                        raise RH56Error(
                            f"readback={restored_target}, expected={original_target}"
                        )
                target_restored = True
            except BaseException as exc:
                recovery_errors.append(f"目标角度最终校验失败: {exc}")
        target_is_safe = not target_change_attempted or target_restored
        speed_restored = not speed_change_attempted or not restore_settings_on_exit
        if target_is_safe and speed_change_attempted and restore_settings_on_exit:
            try:
                hand.write_short(speed_address, original_speed)
                restored_speed = hand.read_short(speed_address)
                if restored_speed != original_speed:
                    raise RH56Error(
                        f"readback={restored_speed}, expected={original_speed}"
                    )
                speed_restored = True
            except BaseException as exc:
                recovery_errors.append(f"速度恢复失败: {exc}")
        if (
            target_is_safe
            and speed_restored
            and force_change_attempted
            and restore_settings_on_exit
        ):
            try:
                hand.write_short(force_address, original_force)
                restored_force = hand.read_short(force_address)
                if restored_force != original_force:
                    raise RH56Error(
                        f"readback={restored_force}, expected={original_force}"
                    )
            except BaseException as exc:
                recovery_errors.append(f"力阈值恢复失败: {exc}")
        if not target_is_safe and (speed_change_attempted or force_change_attempted):
            recovery_errors.append("目标未恢复，已保留临时低速/低力参数")
        if recovery_errors:
            raise RH56Error(
                "RECOVERY FAILED; stop testing and power off the hand if it is still moving. "
                + "; ".join(recovery_errors)
            ) from operation_error
    if operation_error is not None:
        raise operation_error


def nudge(
    hand: RH56Hand,
    joint_index: int,
    delta: int,
    speed: int,
    force_limit: int,
    motion_timeout: float,
    batch_angle_write: bool = False,
) -> None:
    initial = hand.snapshot()
    if tuple(initial["angle_targets"]) != (-1,) * len(JOINTS):
        raise RH56Error("nudge requires all six ANGLE_SET targets to be -1")
    if any(
        abs(int(current)) > NUDGE_MAX_IDLE_CURRENT_MA
        for current in initial["currents"]
    ):
        raise RH56Error(
            f"nudge requires each idle current to be <= {NUDGE_MAX_IDLE_CURRENT_MA} mA"
        )
    if (
        sum(abs(int(current)) for current in initial["currents"])
        > NUDGE_MAX_TOTAL_IDLE_CURRENT_MA
    ):
        raise RH56Error(
            "nudge requires total idle current to be <= "
            f"{NUDGE_MAX_TOTAL_IDLE_CURRENT_MA} mA"
        )
    if joint_index == JOINTS.index("thumb_rotate") and any(
        int(angle) < THUMB_ROTATE_NUDGE_MIN_BEND_CLEARANCE
        for angle in initial["angles"][:5]
    ):
        raise RH56Error(
            "thumb_rotate nudge requires all five bend axes to be open at "
            f">={THUMB_ROTATE_NUDGE_MIN_BEND_CLEARANCE}"
        )

    def build_path(
        original_angle: int, return_target: int
    ) -> Sequence[Tuple[str, int]]:
        if joint_index == JOINTS.index("thumb_rotate"):
            target = nudge_target(
                original_angle,
                delta,
                minimum_angle=THUMB_ROTATE_NUDGE_MIN_ANGLE,
                maximum_angle=THUMB_ROTATE_NUDGE_MAX_ANGLE,
            )
        else:
            target = nudge_target(original_angle, delta)
        return (("到达测试点", target), ("返回起点", return_target))

    move_joint_path(
        hand,
        joint_index,
        speed,
        force_limit,
        motion_timeout,
        "低速点动",
        build_path,
        monitor_all=True,
        stop_in_place_on_error=False,
        diagnostic_trace=True,
        max_joint_current_ma=400,
        max_total_current_ma=600,
        verify_all_disabled_on_exit=True,
        batch_angle_write=batch_angle_write,
        position_tolerance=NUDGE_POSITION_TOLERANCE,
        verify_inactive_axes_stationary=True,
    )


def recover_middle_positive_position(hand: RH56Hand) -> None:
    """Recover an over-open middle actuator into the documented POS range.

    This is deliberately a raw-position maintenance operation, not a normal
    hand controller.  It is restricted to the middle axis, accepts only a
    small negative starting position, uses a low device current limit, and
    finishes at the device's own bounded target inside the documented 0..2000
    actuator interval.  The Franka interface is not imported or opened by this
    program.
    """

    joint_index = JOINTS.index("middle")
    snapshot = hand.snapshot()
    if tuple(snapshot["angle_targets"]) != (-1,) * len(JOINTS):
        raise RH56Error("middle position recovery requires all ANGLE_SET=-1")
    if any(int(value) for value in snapshot["errors"]):
        raise RH56Error("middle position recovery refused: actuator fault present")
    if not all(int(value) in (2, 0xFF) for value in snapshot["statuses"]):
        raise RH56Error("middle position recovery requires every actuator idle")
    if max(int(value) for value in snapshot["temperatures"]) >= 50:
        raise RH56Error("middle position recovery refused: temperature >=50 C")
    currents = tuple(int(value) for value in snapshot["currents"])
    if any(abs(value) > 50 for value in currents):
        raise RH56Error(
            "middle position recovery requires every idle current <=50 mA"
        )
    start_position = int(snapshot["positions"][joint_index])
    if not -200 <= start_position < 0:
        raise RH56Error(
            "middle position recovery is only for POS_ACT in -200..-1; "
            f"observed={start_position}"
        )
    if int(snapshot["angles"][joint_index]) != 1000:
        raise RH56Error(
            "middle position recovery requires saturated ANGLE_ACT=1000; "
            f"observed={snapshot['angles'][joint_index]}"
        )

    position_address = REG_POS_SET + 2 * joint_index
    speed_address = REG_SPEED_SET + 2 * joint_index
    force_address = REG_FORCE_SET + 2 * joint_index
    current_limit_address = REG_CURRENT_LIMIT + 2 * joint_index
    original_speed = hand.read_short(speed_address)
    original_force = hand.read_short(force_address)
    original_current_limit = hand.read_short(current_limit_address)
    baseline_angles = tuple(int(value) for value in snapshot["angles"])
    baseline_positions = tuple(int(value) for value in snapshot["positions"])

    def write_position_target_once_verified(value: int) -> int:
        write_error: Optional[BaseException] = None
        try:
            hand.write(position_address, struct.pack("<h", value), retries=0)
        except BaseException as exc:
            write_error = exc
        readback = hand.read_short(position_address)
        # This installed RH56 rewrites a raw open-end POS_SET=0 to its
        # calibrated internal open target (observed as 126).  Accept that
        # device-side positive clamp only within a narrow recovery envelope;
        # never resend the numeric command.
        accepted = readback == value or (
            value == 0 and 0 <= readback <= 200
        )
        if not accepted:
            context = "" if write_error is None else f"; write_error={write_error}"
            raise RH56Error(
                f"middle POS_SET readback mismatch: wrote={value}, "
                f"readback={readback}{context}"
            )
        if readback != value:
            print(
                "[middle-recovery] device clamped "
                f"POS_SET={value} to calibrated internal target={readback}"
            )
        return readback

    def release_outputs() -> None:
        hand.write_short(position_address, -1)
        if hand.read_short(position_address) != -1:
            raise RH56Error("middle POS_SET release readback is not -1")
        _disable_all_targets_verified(hand)

    def wait_for_position(target: int) -> int:
        started = time.monotonic()
        deadline = started + MIDDLE_POSITION_RECOVERY_TIMEOUT_S
        previous = hand.read_short(REG_POS_ACT + 2 * joint_index)
        stable = 0
        next_trace = started
        while True:
            now = time.monotonic()
            positions = tuple(hand.read_six_shorts(REG_POS_ACT, retries=0))
            angles = tuple(hand.read_six_shorts(REG_ANGLE_ACT, retries=0))
            live_currents = tuple(hand.read_six_shorts(REG_CURRENT, retries=0))
            forces = tuple(hand.read_six_shorts(REG_FORCE_ACT, retries=0))
            errors = tuple(hand.read(REG_ERROR, 6, retries=0))
            statuses = tuple(hand.read(REG_STATUS, 6, retries=0))
            temperatures = tuple(hand.read(REG_TEMP, 6, retries=0))
            actual = int(positions[joint_index])
            selected_current = abs(int(live_currents[joint_index]))
            total_current = sum(abs(int(value)) for value in live_currents)

            if any(errors):
                raise RH56Error(f"actuator fault during recovery: errors={errors}")
            if max(temperatures) >= 50:
                raise RH56Error(
                    f"temperature reached recovery limit: {temperatures}"
                )
            if selected_current > 150 or total_current > 250:
                raise RH56Error(
                    "current exceeded recovery envelope: "
                    f"middle={selected_current}mA total={total_current}mA"
                )
            if actual < previous - 3:
                raise RH56Error(
                    "middle moved farther past the open endpoint: "
                    f"previous={previous}, actual={actual}, target={target}"
                )
            if actual > target + 25:
                raise RH56Error(
                    f"middle overshot POS_SET={target}: POS_ACT={actual}"
                )
            for axis in range(len(JOINTS)):
                if axis == joint_index:
                    continue
                if abs(positions[axis] - baseline_positions[axis]) > 10:
                    raise RH56Error(
                        f"inactive {JOINT_LABELS[axis]} POS_ACT moved during recovery"
                    )
                if abs(angles[axis] - baseline_angles[axis]) > 5:
                    raise RH56Error(
                        f"inactive {JOINT_LABELS[axis]} ANGLE_ACT moved during recovery"
                    )
                if statuses[axis] in (0, 1):
                    raise RH56Error(
                        f"inactive {JOINT_LABELS[axis]} unexpectedly started moving"
                    )
            status = int(statuses[joint_index])
            if status in (3, 5, 6, 7):
                raise RH56Error(
                    "middle stopped before position target: "
                    f"status={STATUS_LABELS.get(status, status)} actual={actual}"
                )
            if status not in (0, 1, 2, 0xFF):
                raise RH56Error(f"middle returned unknown status {status}")

            # This maintenance action only needs to leave the invalid negative
            # range.  The installed force-aware firmware can settle before its
            # rewritten internal POS_SET while reporting STATUS=2 and zero
            # current.  Treat that stable, valid-range state as success rather
            # than driving harder merely to make POS_ACT equal the internal
            # target.
            reached = (
                0 <= actual <= 200
                and status in (2, 0xFF)
                and selected_current <= 50
            )
            stable = stable + 1 if reached and abs(actual - previous) <= 2 else 0
            if now >= next_trace or stable >= 3 or now >= deadline:
                print(
                    "[middle-recovery] "
                    f"target={target} POS_ACT={actual} ANGLE_ACT={angles[joint_index]} "
                    f"FORCE={forces[joint_index]}g CURRENT={live_currents[joint_index]}mA "
                    f"STATUS={status}"
                )
                next_trace = now + 0.25
            if stable >= 3:
                return actual
            if now >= deadline:
                raise RH56Error(
                    f"middle did not reach POS_SET={target} within "
                    f"{MIDDLE_POSITION_RECOVERY_TIMEOUT_S:.1f}s; POS_ACT={actual}"
                )
            previous = actual
            time.sleep(0.05)

    operation_error: Optional[BaseException] = None
    stopped = False
    print(
        "中指负位置受控回收: "
        f"POS_ACT={start_position} -> device-bounded positive target; speed="
        f"{MIDDLE_POSITION_RECOVERY_SPEED}, current_limit="
        f"{MIDDLE_POSITION_RECOVERY_CURRENT_LIMIT_MA}mA; 其余五轴保持禁用"
    )
    try:
        hand.write_short(speed_address, MIDDLE_POSITION_RECOVERY_SPEED)
        hand.write_short(force_address, MIDDLE_POSITION_RECOVERY_FORCE_G)
        hand.write_short(
            current_limit_address, MIDDLE_POSITION_RECOVERY_CURRENT_LIMIT_MA
        )
        if hand.read_short(speed_address) != MIDDLE_POSITION_RECOVERY_SPEED:
            raise RH56Error("middle recovery SPEED_SET readback mismatch")
        if hand.read_short(force_address) != MIDDLE_POSITION_RECOVERY_FORCE_G:
            raise RH56Error("middle recovery FORCE_SET readback mismatch")
        if (
            hand.read_short(current_limit_address)
            != MIDDLE_POSITION_RECOVERY_CURRENT_LIMIT_MA
        ):
            raise RH56Error("middle recovery CURRENT_LIMIT readback mismatch")
        if tuple(hand.read_six_shorts(REG_ANGLE_SET, retries=1)) != (-1,) * len(
            JOINTS
        ):
            raise RH56Error("ANGLE_SET changed before middle recovery motion")
        for target in MIDDLE_POSITION_RECOVERY_WAYPOINTS:
            effective_target = write_position_target_once_verified(target)
            actual = wait_for_position(effective_target)
            print(
                "[middle-recovery] waypoint PASS: "
                f"requested={target}, effective={effective_target}, actual={actual}"
            )
        release_outputs()
        stopped = True
    except BaseException as exc:
        operation_error = exc
    finally:
        cleanup_errors: List[str] = []
        if not stopped:
            try:
                release_outputs()
                stopped = True
                print("[middle-recovery] emergency release verified")
            except BaseException as exc:
                cleanup_errors.append(f"output release failed: {exc}")
        if stopped:
            try:
                hand.write_short(speed_address, original_speed)
                hand.write_short(force_address, original_force)
                hand.write_short(current_limit_address, original_current_limit)
                if hand.read_short(speed_address) != original_speed:
                    raise RH56Error("speed restore mismatch")
                if hand.read_short(force_address) != original_force:
                    raise RH56Error("force restore mismatch")
                if hand.read_short(current_limit_address) != original_current_limit:
                    raise RH56Error("current-limit restore mismatch")
            except BaseException as exc:
                cleanup_errors.append(f"setting restore failed: {exc}")
        if cleanup_errors:
            raise RH56Error(
                "MIDDLE RECOVERY CLEANUP FAILED; cut RH56 24 V if it is moving: "
                + "; ".join(cleanup_errors)
            ) from operation_error
    if operation_error is not None:
        raise operation_error

    final = hand.snapshot()
    final_position = int(final["positions"][joint_index])
    if not 0 <= final_position <= 200:
        raise RH56Error(
            f"middle recovery final POS_ACT={final_position}, expected 0..200"
        )
    if tuple(final["angle_targets"]) != (-1,) * len(JOINTS):
        raise RH56Error("middle recovery final ANGLE_SET is not all -1")
    if any(int(value) for value in final["errors"]):
        raise RH56Error(f"middle recovery final errors={final['errors']}")
    print(
        "[PASS] 中指已回到有效位置范围并停在端点内侧: "
        f"POS_ACT={final_position}, ANGLE_ACT={final['angles'][joint_index]}, "
        "Franka 未访问"
    )


def nudge_target(
    original_angle: int,
    signed_delta: int,
    *,
    minimum_angle: int = 0,
    maximum_angle: int = 1000,
) -> int:
    """Return an explicit-direction nudge target without silently reversing it."""

    if not 0 <= original_angle <= 1000:
        raise ValueError("nudge original angle must be in 0..1000")
    if not 20 <= abs(signed_delta) <= 100:
        raise ValueError("signed nudge delta must be -100..-20 or 20..100")
    if not 0 <= minimum_angle < maximum_angle <= 1000:
        raise ValueError("nudge soft range must satisfy 0<=min<max<=1000")
    if not minimum_angle <= original_angle <= maximum_angle:
        raise ValueError(
            f"nudge original angle {original_angle} is outside the soft range "
            f"{minimum_angle}..{maximum_angle}"
        )
    target = original_angle + signed_delta
    if not minimum_angle <= target <= maximum_angle:
        raise ValueError(
            f"nudge target {target} is outside {minimum_angle}..{maximum_angle}; "
            "use the opposite signed delta"
        )
    return target


def probe_thumb_rotate(
    hand: RH56Hand,
    delta: int,
    speed: int,
    force_limit: int,
    observation_timeout: float,
) -> None:
    """Issue one bounded batch target and stop after measurable axis-6 motion.

    This is deliberately not a position-reaching command.  It distinguishes an
    internal no-response window from a working drive without commanding a
    return leg.  After a successful response, cleanup first converts the fresh
    ANGLE_ACT into the measured neutral ANGLE_SET domain (actual + 15), then
    writes all six ANGLE_SET values to -1.  It never rewrites raw ANGLE_ACT as
    ANGLE_SET because those mappings are offset on the commissioned hand.
    """

    joint_index = JOINTS.index("thumb_rotate")
    initial = hand.snapshot()
    if tuple(initial["angle_targets"]) != (-1,) * len(JOINTS):
        raise RH56Error("probe requires all six ANGLE_SET targets to be -1")
    if any(initial["errors"]):
        raise RH56Error("probe refused because at least one actuator reports a fault")
    if not all(int(status) in (2, 0xFF) for status in initial["statuses"]):
        raise RH56Error("probe requires every actuator STATUS to be 2 or 0xFF")
    if max(int(value) for value in initial["temperatures"]) >= 60:
        raise RH56Error("probe refused because an actuator is at least 60 °C")
    if any(
        abs(int(current)) > NUDGE_MAX_IDLE_CURRENT_MA
        for current in initial["currents"]
    ):
        raise RH56Error(
            f"probe requires each idle current to be <= {NUDGE_MAX_IDLE_CURRENT_MA} mA"
        )
    if (
        sum(abs(int(current)) for current in initial["currents"])
        > NUDGE_MAX_TOTAL_IDLE_CURRENT_MA
    ):
        raise RH56Error(
            "probe requires total idle current to be <= "
            f"{NUDGE_MAX_TOTAL_IDLE_CURRENT_MA} mA"
        )
    if any(
        int(angle) < THUMB_ROTATE_NUDGE_MIN_BEND_CLEARANCE
        for angle in initial["angles"][:5]
    ):
        raise RH56Error(
            "thumb_rotate probe requires all five bend axes to be open at "
            f">={THUMB_ROTATE_NUDGE_MIN_BEND_CLEARANCE}"
        )

    start_angles = tuple(int(value) for value in initial["angles"])
    start_positions = tuple(int(value) for value in initial["positions"])
    start_forces = tuple(int(value) for value in initial["forces"])
    start_angle = start_angles[joint_index]
    start_position = start_positions[joint_index]
    target = nudge_target(
        start_angle,
        delta,
        minimum_angle=THUMB_ROTATE_NUDGE_MIN_ANGLE,
        maximum_angle=THUMB_ROTATE_NUDGE_MAX_ANGLE,
    )
    expected_targets = tuple(
        target if index == joint_index else -1 for index in range(len(JOINTS))
    )
    speed_address = REG_SPEED_SET + 2 * joint_index
    force_address = REG_FORCE_SET + 2 * joint_index
    original_speed = hand.read_short(speed_address)
    original_force = hand.read_short(force_address)
    print(
        "单程探测 拇指旋转: "
        f"ANGLE_ACT={start_angle} -> ANGLE_SET={target}, "
        f"POS_ACT={start_position}, speed={speed}, "
        f"force_limit={force_limit}g, window={observation_timeout:.1f}s"
    )

    speed_change_attempted = False
    force_change_attempted = False
    operation_error: Optional[BaseException] = None
    try:
        speed_change_attempted = True
        hand.write_short(speed_address, speed)
        if hand.read_short(speed_address) != speed:
            raise RH56Error("probe SPEED_SET readback mismatch")
        force_change_attempted = True
        hand.write_short(force_address, force_limit)
        if hand.read_short(force_address) != force_limit:
            raise RH56Error("probe FORCE_SET readback mismatch")
        hand.write_six_shorts(REG_ANGLE_SET, expected_targets, retries=1)
        target_readback = tuple(hand.read_six_shorts(REG_ANGLE_SET, retries=1))
        if target_readback != expected_targets:
            raise RH56Error(
                "probe batch ANGLE_SET readback mismatch: "
                f"wrote={expected_targets}, readback={target_readback}"
            )

        started_at = time.monotonic()
        deadline = started_at + observation_timeout
        next_trace_at = started_at
        consecutive_motion_samples = 0
        expected_direction = 1 if delta > 0 else -1
        while True:
            now = time.monotonic()
            angles = tuple(hand.read_six_shorts(REG_ANGLE_ACT, retries=0))
            positions = tuple(hand.read_six_shorts(REG_POS_ACT, retries=0))
            currents = tuple(hand.read_six_shorts(REG_CURRENT, retries=0))
            forces = tuple(hand.read_six_shorts(REG_FORCE_ACT, retries=0))
            errors = tuple(hand.read(REG_ERROR, 6, retries=0))
            statuses = tuple(hand.read(REG_STATUS, 6, retries=0))
            temperatures = tuple(hand.read(REG_TEMP, 6, retries=0))
            angle_delta = int(angles[joint_index]) - start_angle
            position_delta = int(positions[joint_index]) - start_position
            selected_current = int(currents[joint_index])
            total_current = sum(abs(int(value)) for value in currents)
            status = int(statuses[joint_index])

            if any(errors):
                fault_index = next(index for index, value in enumerate(errors) if value)
                raise RH56Error(
                    f"{JOINT_LABELS[fault_index]} reported "
                    f"{decode_error(errors[fault_index])} during probe"
                )
            if max(temperatures) >= 60:
                hot_index = temperatures.index(max(temperatures))
                raise RH56Error(
                    f"{JOINT_LABELS[hot_index]} reached "
                    f"{temperatures[hot_index]} °C during probe"
                )
            if abs(selected_current) > 400:
                raise RH56Error(
                    f"thumb_rotate current exceeded 400 mA: {selected_current} mA"
                )
            if total_current > 600:
                raise RH56Error(
                    f"all-axis total current exceeded 600 mA: {total_current} mA"
                )
            if status in (3, 5, 6, 7):
                raise RH56Error(
                    "thumb_rotate stopped during probe: "
                    f"{STATUS_LABELS.get(status, status)}"
                )
            if status not in (0, 1, 2):
                raise RH56Error(f"thumb_rotate returned unknown status {status}")
            for index, other_status in enumerate(statuses):
                if index != joint_index and other_status in (0, 1):
                    raise RH56Error(
                        f"inactive joint {JOINT_LABELS[index]} unexpectedly started moving"
                    )
                if index == joint_index:
                    continue
                angle_drift = abs(int(angles[index]) - start_angles[index])
                position_drift = abs(int(positions[index]) - start_positions[index])
                if angle_drift > NUDGE_MAX_INACTIVE_ANGLE_DRIFT:
                    raise RH56Error(
                        f"inactive joint {JOINT_LABELS[index]} ANGLE_ACT drifted "
                        f"by {angle_drift} during probe"
                    )
                if position_drift > NUDGE_MAX_INACTIVE_POSITION_DRIFT:
                    raise RH56Error(
                        f"inactive joint {JOINT_LABELS[index]} POS_ACT drifted "
                        f"by {position_drift} during probe"
                    )
            force_delta = abs(int(forces[joint_index]) - start_forces[joint_index])
            if force_delta > PROBE_MAX_FORCE_DELTA_G:
                raise RH56Error(
                    f"thumb_rotate force changed by {force_delta}g during probe"
                )
            if (
                abs(angle_delta) >= PROBE_MIN_ANGLE_MOVEMENT
                and angle_delta * expected_direction < 0
            ):
                raise RH56Error(
                    "thumb_rotate ANGLE_ACT moved in the wrong direction: "
                    f"delta={angle_delta}"
                )
            if (
                abs(position_delta) >= PROBE_MIN_POSITION_MOVEMENT
                and position_delta * expected_direction > 0
            ):
                raise RH56Error(
                    "thumb_rotate POS_ACT moved in the wrong direction: "
                    f"delta={position_delta}"
                )

            motion_sample = (
                abs(angle_delta) >= PROBE_MIN_ANGLE_MOVEMENT
                or abs(position_delta) >= PROBE_MIN_POSITION_MOVEMENT
            )
            consecutive_motion_samples = (
                consecutive_motion_samples + 1 if motion_sample else 0
            )
            detected = consecutive_motion_samples >= PROBE_CONFIRM_SAMPLES
            if now >= next_trace_at or detected or now >= deadline:
                position_set = hand.read_short(REG_POS_SET + 2 * joint_index)
                print(
                    "[probe] "
                    f"t={now - started_at:.2f}s "
                    f"ANGLE_SET/ACT={target}/{angles[joint_index]} "
                    f"dANGLE={angle_delta:+d} "
                    f"POS_SET/ACT={position_set}/{positions[joint_index]} "
                    f"dPOS={position_delta:+d} "
                    f"FORCE={forces[joint_index]}g "
                    f"CURRENT={selected_current}mA TOTAL={total_current}mA "
                    f"STATUS={status} ERROR={errors[joint_index]} "
                    f"TEMP={temperatures[joint_index]}C"
                )
                next_trace_at = now + 0.25
            if detected:
                print(
                    "[probe] measurable directional response detected; "
                    "ending without an automatic return command"
                )
                break
            if now >= deadline:
                raise RH56Error(
                    "thumb_rotate produced no measurable motion during probe; "
                    f"ANGLE_ACT={angles[joint_index]} (delta={angle_delta:+d}), "
                    f"POS_ACT={positions[joint_index]} (delta={position_delta:+d}), "
                    f"current={selected_current}mA, status={status}"
                )
            time.sleep(0.08)
    except BaseException as exc:
        operation_error = exc
    finally:
        recovery_errors: List[str] = []
        output_disabled = False
        if operation_error is None:
            try:
                hold_actual = hand.read_short(REG_ANGLE_ACT + 2 * joint_index)
                if not (
                    THUMB_ROTATE_HOLD_VALID_MIN_ANGLE
                    <= hold_actual
                    <= THUMB_ROTATE_HOLD_VALID_MAX_ANGLE
                ):
                    raise RH56Error(
                        f"ANGLE_ACT {hold_actual} is outside validated hold range "
                        f"{THUMB_ROTATE_HOLD_VALID_MIN_ANGLE}.."
                        f"{THUMB_ROTATE_HOLD_VALID_MAX_ANGLE}"
                    )
                hold_target = (
                    hold_actual + THUMB_ROTATE_FEEDBACK_TO_COMMAND_OFFSET
                )
                if not (
                    THUMB_ROTATE_NUDGE_MIN_ANGLE
                    <= hold_target
                    <= THUMB_ROTATE_NUDGE_MAX_ANGLE
                ):
                    raise RH56Error(
                        f"feedback-equivalent hold target {hold_target} is outside "
                        f"{THUMB_ROTATE_NUDGE_MIN_ANGLE}.."
                        f"{THUMB_ROTATE_NUDGE_MAX_ANGLE}"
                    )
                hold_targets = tuple(
                    hold_target if index == joint_index else -1
                    for index in range(len(JOINTS))
                )
                hand.write_six_shorts(REG_ANGLE_SET, hold_targets, retries=1)
                hold_readback = tuple(
                    hand.read_six_shorts(REG_ANGLE_SET, retries=1)
                )
                if hold_readback != hold_targets:
                    raise RH56Error(
                        "feedback-equivalent hold readback mismatch: "
                        f"wrote={hold_targets}, readback={hold_readback}"
                    )
                time.sleep(0.10)
                print(
                    "[probe] neutralized the internal position target with "
                    f"ANGLE_ACT/ANGLE_SET={hold_actual}/{hold_target}"
                )
            except BaseException as exc:
                recovery_errors.append(f"中性保持失败: {exc}")
        try:
            _disable_all_targets_verified(hand)
            output_disabled = True
            print(
                "[probe] verified all-six "
                "ANGLE_SET=-1"
            )
        except BaseException as exc:
            recovery_errors.append(f"六轴禁用失败: {exc}")
        if output_disabled and speed_change_attempted:
            try:
                hand.write_short(speed_address, original_speed)
                if hand.read_short(speed_address) != original_speed:
                    raise RH56Error("speed restore readback mismatch")
            except BaseException as exc:
                recovery_errors.append(f"速度恢复失败: {exc}")
        if output_disabled and force_change_attempted:
            try:
                hand.write_short(force_address, original_force)
                if hand.read_short(force_address) != original_force:
                    raise RH56Error("force restore readback mismatch")
            except BaseException as exc:
                recovery_errors.append(f"力阈值恢复失败: {exc}")
        if recovery_errors:
            raise RH56Error(
                "PROBE RECOVERY FAILED; cut 24 V if the hand is still moving. "
                + "; ".join(recovery_errors)
            ) from operation_error
    if operation_error is not None:
        raise operation_error


def visual_cycle_thumb_rotate(
    hand: RH56Hand,
    speed: int,
    force_limit: int,
    motion_timeout: float,
) -> None:
    """Run one bounded, visible axis-6 cycle and return to the start pose."""

    joint_index = JOINTS.index("thumb_rotate")
    def validate_preflight_snapshot(snapshot: Dict[str, object], phase: str) -> None:
        if tuple(snapshot["angle_targets"]) != (-1,) * len(JOINTS):
            raise RH56Error(
                f"visual-cycle {phase} requires all six ANGLE_SET targets to be -1"
            )
        if any(snapshot["errors"]):
            raise RH56Error(
                f"visual-cycle {phase} refused because an actuator reports a fault"
            )
        if not all(
            int(status) in (2, 0xFF) for status in snapshot["statuses"]
        ):
            raise RH56Error(
                f"visual-cycle {phase} requires every STATUS to be 2 or 0xFF"
            )
        if max(int(value) for value in snapshot["temperatures"]) >= 50:
            raise RH56Error(
                f"visual-cycle {phase} refused because an actuator is at least 50 °C"
            )
        if any(
            abs(int(current)) > NUDGE_MAX_IDLE_CURRENT_MA
            for current in snapshot["currents"]
        ):
            raise RH56Error(
                f"visual-cycle {phase} requires each idle current to be <= "
                f"{NUDGE_MAX_IDLE_CURRENT_MA} mA"
            )
        if (
            sum(abs(int(current)) for current in snapshot["currents"])
            > NUDGE_MAX_TOTAL_IDLE_CURRENT_MA
        ):
            raise RH56Error(
                f"visual-cycle {phase} requires total idle current to be <= "
                f"{NUDGE_MAX_TOTAL_IDLE_CURRENT_MA} mA"
            )
        if any(
            int(angle) < THUMB_ROTATE_NUDGE_MIN_BEND_CLEARANCE
            for angle in snapshot["angles"][:5]
        ):
            raise RH56Error(
                "thumb_rotate visual-cycle requires all five bend axes to be open "
                f"at >={THUMB_ROTATE_NUDGE_MIN_BEND_CLEARANCE}"
            )

    first_snapshot = hand.snapshot()
    validate_preflight_snapshot(first_snapshot, "first preflight")
    time.sleep(0.15)
    initial = hand.snapshot()
    validate_preflight_snapshot(initial, "second preflight")
    if any(
        abs(int(after) - int(before)) > 2
        for before, after in zip(first_snapshot["angles"], initial["angles"])
    ):
        raise RH56Error(
            "visual-cycle preflight ANGLE_ACT was not stable: "
            f"first={first_snapshot['angles']}, second={initial['angles']}"
        )
    if any(
        abs(int(after) - int(before)) > 3
        for before, after in zip(first_snapshot["positions"], initial["positions"])
    ):
        raise RH56Error(
            "visual-cycle preflight POS_ACT was not stable: "
            f"first={first_snapshot['positions']}, second={initial['positions']}"
        )

    start_angles = tuple(int(value) for value in initial["angles"])
    start_positions = tuple(int(value) for value in initial["positions"])
    start_forces = tuple(int(value) for value in initial["forces"])
    start_angle = start_angles[joint_index]
    if not (
        THUMB_ROTATE_HOLD_VALID_MIN_ANGLE
        <= start_angle
        <= THUMB_ROTATE_HOLD_VALID_MAX_ANGLE
    ):
        raise RH56Error(
            f"thumb_rotate start angle {start_angle} is outside the validated "
            f"visual-cycle range {THUMB_ROTATE_HOLD_VALID_MIN_ANGLE}.."
            f"{THUMB_ROTATE_HOLD_VALID_MAX_ANGLE}"
        )
    return_command = start_angle + THUMB_ROTATE_FEEDBACK_TO_COMMAND_OFFSET
    if not (
        THUMB_ROTATE_NUDGE_MIN_ANGLE
        <= return_command
        <= THUMB_ROTATE_NUDGE_MAX_ANGLE
    ):
        raise RH56Error(
            f"feedback-equivalent return command {return_command} is outside "
            f"{THUMB_ROTATE_NUDGE_MIN_ANGLE}..{THUMB_ROTATE_NUDGE_MAX_ANGLE}"
        )

    expected_low_actual = (
        THUMB_ROTATE_VISUAL_LOW_COMMAND
        - THUMB_ROTATE_FEEDBACK_TO_COMMAND_OFFSET
    )
    expected_high_actual = (
        THUMB_ROTATE_VISUAL_HIGH_COMMAND
        - THUMB_ROTATE_FEEDBACK_TO_COMMAND_OFFSET
    )
    waypoints: List[Tuple[str, int, int]] = []
    for cycle_index in range(1, THUMB_ROTATE_VISUAL_CYCLES + 1):
        waypoints.extend(
            (
                (
                    f"第{cycle_index}轮低位",
                    THUMB_ROTATE_VISUAL_LOW_COMMAND,
                    expected_low_actual,
                ),
                (
                    f"第{cycle_index}轮高位",
                    THUMB_ROTATE_VISUAL_HIGH_COMMAND,
                    expected_high_actual,
                ),
            )
        )
    waypoints.append(("返回起点", return_command, start_angle))
    speed_address = REG_SPEED_SET + 2 * joint_index
    force_address = REG_FORCE_SET + 2 * joint_index
    original_speed = hand.read_short(speed_address)
    original_force = hand.read_short(force_address)
    print(
        "拇指旋转可视往返: "
        f"start ANGLE_ACT={start_angle}, commands="
        f"{THUMB_ROTATE_VISUAL_LOW_COMMAND} -> "
        f"{THUMB_ROTATE_VISUAL_HIGH_COMMAND} -> {return_command}, "
        f"cycles={THUMB_ROTATE_VISUAL_CYCLES}, speed={speed}, "
        f"force_limit={force_limit}g"
    )

    def write_batch_target(command: int) -> None:
        expected = tuple(
            command if index == joint_index else -1
            for index in range(len(JOINTS))
        )
        hand.write_six_shorts(REG_ANGLE_SET, expected, retries=1)
        readback = tuple(hand.read_six_shorts(REG_ANGLE_SET, retries=1))
        if readback != expected:
            raise RH56Error(
                "visual-cycle ANGLE_SET readback mismatch: "
                f"wrote={expected}, readback={readback}"
            )

    def wait_for_endpoint(
        label: str,
        command: int,
        expected_actual: int,
        leg_start_angle: int,
        leg_start_position: int,
    ) -> int:
        expected_direction = 1 if expected_actual > leg_start_angle else -1
        started_at = time.monotonic()
        deadline = started_at + motion_timeout
        next_trace_at = started_at
        consecutive_idle_samples = 0
        previous_actual: Optional[int] = None
        previous_position: Optional[int] = None
        while True:
            now = time.monotonic()
            angles = tuple(hand.read_six_shorts(REG_ANGLE_ACT, retries=0))
            positions = tuple(hand.read_six_shorts(REG_POS_ACT, retries=0))
            currents = tuple(hand.read_six_shorts(REG_CURRENT, retries=0))
            forces = tuple(hand.read_six_shorts(REG_FORCE_ACT, retries=0))
            errors = tuple(hand.read(REG_ERROR, 6, retries=0))
            statuses = tuple(hand.read(REG_STATUS, 6, retries=0))
            temperatures = tuple(hand.read(REG_TEMP, 6, retries=0))
            actual = int(angles[joint_index])
            position = int(positions[joint_index])
            angle_delta = actual - leg_start_angle
            position_delta = position - leg_start_position
            selected_current = int(currents[joint_index])
            total_current = sum(abs(int(value)) for value in currents)
            status = int(statuses[joint_index])

            if any(errors):
                fault_index = next(index for index, value in enumerate(errors) if value)
                raise RH56Error(
                    f"{JOINT_LABELS[fault_index]} reported "
                    f"{decode_error(errors[fault_index])} during visual-cycle"
                )
            hottest = max(temperatures)
            if hottest >= 60:
                hot_index = temperatures.index(hottest)
                raise RH56Error(
                    f"{JOINT_LABELS[hot_index]} reached {hottest} °C "
                    "during visual-cycle"
                )
            if abs(selected_current) > 400:
                raise RH56Error(
                    "thumb_rotate current exceeded 400 mA during visual-cycle: "
                    f"{selected_current} mA"
                )
            if total_current > 600:
                raise RH56Error(
                    "all-axis total current exceeded 600 mA during visual-cycle: "
                    f"{total_current} mA"
                )
            if status in (3, 5, 6, 7):
                raise RH56Error(
                    "thumb_rotate stopped during visual-cycle: "
                    f"{STATUS_LABELS.get(status, status)}"
                )
            if status not in (0, 1, 2):
                raise RH56Error(f"thumb_rotate returned unknown status {status}")
            if not 830 <= actual <= 880:
                raise RH56Error(
                    f"thumb_rotate ANGLE_ACT left the visual safety envelope: {actual}"
                )
            for index, other_status in enumerate(statuses):
                if index != joint_index and other_status in (0, 1):
                    raise RH56Error(
                        f"inactive joint {JOINT_LABELS[index]} unexpectedly started moving"
                    )
                if index == joint_index:
                    continue
                angle_drift = abs(int(angles[index]) - start_angles[index])
                position_drift = abs(int(positions[index]) - start_positions[index])
                if angle_drift > NUDGE_MAX_INACTIVE_ANGLE_DRIFT:
                    raise RH56Error(
                        f"inactive joint {JOINT_LABELS[index]} ANGLE_ACT drifted "
                        f"by {angle_drift} during visual-cycle"
                    )
                if position_drift > NUDGE_MAX_INACTIVE_POSITION_DRIFT:
                    raise RH56Error(
                        f"inactive joint {JOINT_LABELS[index]} POS_ACT drifted "
                        f"by {position_drift} during visual-cycle"
                    )
            force_delta = abs(int(forces[joint_index]) - start_forces[joint_index])
            if force_delta > PROBE_MAX_FORCE_DELTA_G:
                raise RH56Error(
                    f"thumb_rotate force changed by {force_delta}g "
                    "during visual-cycle"
                )
            if (
                abs(angle_delta) >= PROBE_MIN_ANGLE_MOVEMENT
                and angle_delta * expected_direction < 0
            ):
                raise RH56Error(
                    "thumb_rotate moved in the wrong direction during "
                    f"{label}: delta={angle_delta:+d}"
                )
            if (
                abs(position_delta) >= PROBE_MIN_POSITION_MOVEMENT
                and position_delta * expected_direction > 0
            ):
                raise RH56Error(
                    "thumb_rotate POS_ACT moved in the wrong direction during "
                    f"{label}: delta={position_delta:+d}"
                )

            feedback_stable = (
                previous_actual is not None
                and previous_position is not None
                and abs(actual - previous_actual)
                <= THUMB_ROTATE_VISUAL_STABILITY_DELTA
                and abs(position - previous_position)
                <= THUMB_ROTATE_VISUAL_STABILITY_DELTA
            )
            endpoint_idle = (
                abs(actual - expected_actual)
                <= THUMB_ROTATE_VISUAL_ENDPOINT_TOLERANCE
                and status == 2
                and abs(selected_current) <= THUMB_ROTATE_VISUAL_IDLE_CURRENT_MA
                and feedback_stable
            )
            consecutive_idle_samples = (
                consecutive_idle_samples + 1 if endpoint_idle else 0
            )
            reached = (
                consecutive_idle_samples >= THUMB_ROTATE_VISUAL_SETTLE_SAMPLES
            )
            if now >= next_trace_at or reached or now >= deadline:
                print(
                    "[visual] "
                    f"{label} t={now - started_at:.2f}s "
                    f"ANGLE_SET/ACT={command}/{actual} "
                    f"expected_ACT={expected_actual} dANGLE={angle_delta:+d} "
                    f"POS_SET/ACT="
                    f"{hand.read_short(REG_POS_SET + 2 * joint_index)}/{position} "
                    f"dPOS={position_delta:+d} CURRENT={selected_current}mA "
                    f"TOTAL={total_current}mA STATUS={status} "
                    f"ERROR={errors[joint_index]} TEMP={temperatures[joint_index]}C"
                )
                next_trace_at = now + 0.25
            if reached:
                return actual
            if now >= deadline:
                raise RH56Error(
                    f"visual-cycle {label} did not settle within "
                    f"{motion_timeout:.1f}s; command={command}, actual={actual}, "
                    f"expected_actual={expected_actual}, current={selected_current}mA, "
                    f"status={status}"
                )
            previous_actual = actual
            previous_position = position
            time.sleep(0.08)

    speed_change_attempted = False
    force_change_attempted = False
    completed = False
    operation_error: Optional[BaseException] = None
    try:
        speed_change_attempted = True
        hand.write_short(speed_address, speed)
        if hand.read_short(speed_address) != speed:
            raise RH56Error("visual-cycle SPEED_SET readback mismatch")
        force_change_attempted = True
        hand.write_short(force_address, force_limit)
        if hand.read_short(force_address) != force_limit:
            raise RH56Error("visual-cycle FORCE_SET readback mismatch")
        if tuple(hand.read_six_shorts(REG_ANGLE_SET, retries=1)) != (-1,) * len(
            JOINTS
        ):
            raise RH56Error(
                "visual-cycle ANGLE_SET changed after preflight; refusing motion"
            )
        for waypoint_index, (label, command, expected_actual) in enumerate(waypoints):
            leg_start_angle = hand.read_short(REG_ANGLE_ACT + 2 * joint_index)
            leg_start_position = hand.read_short(REG_POS_ACT + 2 * joint_index)
            write_batch_target(command)
            reached = wait_for_endpoint(
                label,
                command,
                expected_actual,
                leg_start_angle,
                leg_start_position,
            )
            print(
                f"[visual] {label} stable: command={command}, ANGLE_ACT={reached}"
            )
            if waypoint_index + 1 < len(waypoints):
                time.sleep(THUMB_ROTATE_VISUAL_PAUSE_SECONDS)
        completed = True
    except BaseException as exc:
        operation_error = exc
    finally:
        recovery_errors: List[str] = []
        if completed:
            try:
                write_batch_target(return_command)
                time.sleep(0.10)
                print(
                    "[visual] neutral return target confirmed before release: "
                    f"ANGLE_ACT/ANGLE_SET={start_angle}/{return_command}"
                )
            except BaseException as exc:
                recovery_errors.append(f"中性返回确认失败: {exc}")
        try:
            _disable_all_targets_verified(hand)
            print("[visual] verified pre-restore all-six ANGLE_SET=-1")
            output_disabled = True
        except BaseException as exc:
            recovery_errors.append(f"六轴禁用失败: {exc}")
            output_disabled = False
        if output_disabled and speed_change_attempted:
            try:
                hand.write_short(speed_address, original_speed)
                if hand.read_short(speed_address) != original_speed:
                    raise RH56Error("speed restore readback mismatch")
            except BaseException as exc:
                recovery_errors.append(f"速度恢复失败: {exc}")
        if output_disabled and force_change_attempted:
            try:
                hand.write_short(force_address, original_force)
                if hand.read_short(force_address) != original_force:
                    raise RH56Error("force restore readback mismatch")
            except BaseException as exc:
                recovery_errors.append(f"力阈值恢复失败: {exc}")
        if output_disabled:
            try:
                _disable_all_targets_verified(hand)
                print("[visual] verified final all-six ANGLE_SET=-1")
            except BaseException as exc:
                recovery_errors.append(f"最终六轴禁用失败: {exc}")
        if recovery_errors:
            raise RH56Error(
                "VISUAL-CYCLE RECOVERY FAILED; cut 24 V if the hand is still moving. "
                + "; ".join(recovery_errors)
            ) from operation_error
    if operation_error is not None:
        raise operation_error


def full_cycle_thumb_rotate(
    hand: RH56Hand,
    speed: int,
    force_limit: int,
    motion_timeout: float,
) -> None:
    """Commission axis 6 over its rated open/closed endpoints once.

    This is intentionally separate from the generic sweep.  It is a supervised
    acceptance test for one already commissioned right hand, not a realtime
    control mode.  The official axis convention is 1000=open and 0=closed.
    """

    if speed != 40:
        raise RH56Error("thumb-full-cycle requires speed=40")
    if force_limit != 80:
        raise RH56Error("thumb-full-cycle requires force_limit=80g")
    if not 30 <= motion_timeout <= 180:
        raise RH56Error("thumb-full-cycle requires motion_timeout in 30..180s")

    joint_index = JOINTS.index("thumb_rotate")

    def validate_preflight(snapshot: Dict[str, object], phase: str) -> None:
        if tuple(snapshot["angle_targets"]) != (-1,) * len(JOINTS):
            raise RH56Error(
                f"thumb-full-cycle {phase} requires all six ANGLE_SET=-1"
            )
        if any(int(value) for value in snapshot["errors"]):
            raise RH56Error(
                f"thumb-full-cycle {phase} refused because an axis reports a fault"
            )
        if not all(
            int(status) in (2, 0xFF) for status in snapshot["statuses"]
        ):
            raise RH56Error(
                f"thumb-full-cycle {phase} requires every STATUS in {{2, 0xFF}}"
            )
        if max(int(value) for value in snapshot["temperatures"]) >= 50:
            raise RH56Error(
                f"thumb-full-cycle {phase} refused because an axis is at least 50 °C"
            )
        currents = tuple(int(value) for value in snapshot["currents"])
        if any(abs(value) > NUDGE_MAX_IDLE_CURRENT_MA for value in currents):
            raise RH56Error(
                f"thumb-full-cycle {phase} requires each idle current <= "
                f"{NUDGE_MAX_IDLE_CURRENT_MA} mA"
            )
        if sum(abs(value) for value in currents) > NUDGE_MAX_TOTAL_IDLE_CURRENT_MA:
            raise RH56Error(
                f"thumb-full-cycle {phase} requires total idle current <= "
                f"{NUDGE_MAX_TOTAL_IDLE_CURRENT_MA} mA"
            )
        if any(
            int(angle) < THUMB_ROTATE_FULL_CYCLE_MIN_BEND_CLEARANCE
            for angle in snapshot["angles"][:5]
        ):
            raise RH56Error(
                "thumb-full-cycle requires all five bend axes completely open at "
                f">={THUMB_ROTATE_FULL_CYCLE_MIN_BEND_CLEARANCE}"
            )
        start_angle = int(snapshot["angles"][joint_index])
        if not (
            THUMB_ROTATE_HOLD_VALID_MIN_ANGLE
            <= start_angle
            <= THUMB_ROTATE_HOLD_VALID_MAX_ANGLE
        ):
            raise RH56Error(
                f"thumb-full-cycle start ANGLE_ACT={start_angle} is outside the "
                f"commissioned {THUMB_ROTATE_HOLD_VALID_MIN_ANGLE}.."
                f"{THUMB_ROTATE_HOLD_VALID_MAX_ANGLE} window"
            )

    first = hand.snapshot()
    validate_preflight(first, "first preflight")
    time.sleep(0.15)
    second = hand.snapshot()
    validate_preflight(second, "second preflight")
    if any(
        abs(int(after) - int(before)) > 2
        for before, after in zip(first["angles"], second["angles"])
    ):
        raise RH56Error("thumb-full-cycle preflight ANGLE_ACT was not stable")
    if any(
        abs(int(after) - int(before)) > 3
        for before, after in zip(first["positions"], second["positions"])
    ):
        raise RH56Error("thumb-full-cycle preflight POS_ACT was not stable")

    def build_path(
        original_angle: int, return_target: int
    ) -> Sequence[Tuple[str, int]]:
        del original_angle, return_target
        return (
            ("到达张开端", 1000),
            ("闭合 1/4", 750),
            ("闭合 2/4", 500),
            ("闭合 3/4", 250),
            ("到达闭合端", 0),
            ("张开 1/4", 250),
            ("张开 2/4", 500),
            ("张开 3/4", 750),
            ("回到张开端", 1000),
        )

    print(
        "第六轴额定全行程验收：1000(张开) -> 0(闭合) -> "
        "1000(张开)，按 250 单位分段运行；全程保持五个弯曲轴不动，"
        "测试后停在张开端"
    )
    original_speed = int(second["speeds"][joint_index])
    original_force = int(second["force_limits"][joint_index])
    operation_error: Optional[BaseException] = None
    try:
        move_joint_path(
            hand,
            joint_index,
            speed,
            force_limit,
            motion_timeout,
            "额定全行程",
            build_path,
            require_disabled_idle=True,
            monitor_all=True,
            stop_in_place_on_error=False,
            diagnostic_trace=True,
            max_joint_current_ma=THUMB_ROTATE_FULL_CYCLE_MAX_JOINT_CURRENT_MA,
            max_total_current_ma=THUMB_ROTATE_FULL_CYCLE_MAX_TOTAL_CURRENT_MA,
            verify_all_disabled_on_exit=True,
            batch_angle_write=True,
            position_tolerance=SWEEP_ENDPOINT_TOLERANCE,
            verify_inactive_axes_stationary=True,
            return_target_from_actual=lambda actual: 1000,
            verify_direction_during_motion=True,
            restore_settings_on_exit=False,
            endpoint_stable_samples=3,
            endpoint_max_joint_current_ma=NUDGE_MAX_IDLE_CURRENT_MA,
            endpoint_max_total_current_ma=NUDGE_MAX_TOTAL_IDLE_CURRENT_MA,
        )
    except BaseException as exc:
        operation_error = exc

    # ANGLE_SET=-1 is a protocol release command, not proof that a moving
    # internal POS_SET trajectory stopped immediately.  Observe real feedback
    # after release before declaring the hand safe.
    stop_samples: List[Dict[str, Tuple[int, ...]]] = []
    stop_error: Optional[BaseException] = None
    try:
        if tuple(hand.read_six_shorts(REG_ANGLE_SET, retries=1)) != (-1,) * len(
            JOINTS
        ):
            raise RH56Error("final ANGLE_SET readback is not all -1")
        for sample_index in range(6):
            stop_samples.append(
                {
                    "angles": tuple(
                        hand.read_six_shorts(REG_ANGLE_ACT, retries=0)
                    ),
                    "positions": tuple(
                        hand.read_six_shorts(REG_POS_ACT, retries=0)
                    ),
                    "currents": tuple(
                        hand.read_six_shorts(REG_CURRENT, retries=0)
                    ),
                    "errors": tuple(hand.read(REG_ERROR, 6, retries=0)),
                    "statuses": tuple(hand.read(REG_STATUS, 6, retries=0)),
                    "temperatures": tuple(hand.read(REG_TEMP, 6, retries=0)),
                }
            )
            if sample_index < 5:
                time.sleep(0.15)
        verification_samples = stop_samples[-3:]
        for axis_index, axis_label in enumerate(JOINT_LABELS):
            angle_span = max(
                value["angles"][axis_index] for value in verification_samples
            ) - min(value["angles"][axis_index] for value in verification_samples)
            position_span = max(
                value["positions"][axis_index] for value in verification_samples
            ) - min(
                value["positions"][axis_index] for value in verification_samples
            )
            if (
                angle_span > THUMB_ROTATE_FULL_CYCLE_STOP_ANGLE_DRIFT
                or position_span > THUMB_ROTATE_FULL_CYCLE_STOP_POSITION_DRIFT
            ):
                raise RH56Error(
                    f"{axis_label} feedback still moving after release: "
                    f"angle_span={angle_span}, position_span={position_span}, "
                    f"samples={stop_samples}"
                )
        for sample in verification_samples:
            if any(sample["errors"]):
                raise RH56Error(
                    f"post-release actuator fault remains: errors={sample['errors']}"
                )
            if not all(status in (2, 0xFF) for status in sample["statuses"]):
                raise RH56Error(
                    "post-release status is not idle: "
                    f"statuses={sample['statuses']}"
                )
            if any(
                abs(current) > NUDGE_MAX_IDLE_CURRENT_MA
                for current in sample["currents"]
            ):
                raise RH56Error(
                    "post-release current did not return to the per-axis idle "
                    f"limit: currents={sample['currents']}"
                )
            total_current = sum(abs(current) for current in sample["currents"])
            if total_current > NUDGE_MAX_TOTAL_IDLE_CURRENT_MA:
                raise RH56Error(
                    "post-release total current did not return to idle: "
                    f"total={total_current}mA, currents={sample['currents']}"
                )
            if max(sample["temperatures"]) >= 60:
                raise RH56Error(
                    "post-release temperature is unsafe: "
                    f"temperatures={sample['temperatures']}"
                )
        print(
            "[full-cycle] verified ANGLE_SET=-1 and physical feedback stable: "
            f"angles={verification_samples[-1]['angles']}, "
            f"positions={verification_samples[-1]['positions']}, "
            f"currents={verification_samples[-1]['currents']}, "
            f"statuses={verification_samples[-1]['statuses']}"
        )
    except BaseException as exc:
        stop_error = exc

    if stop_error is not None:
        raise RH56Error(
            "FULL-CYCLE MOTION STOP UNCONFIRMED; cut 24 V now. "
            f"{stop_error}"
        ) from operation_error

    settings_error: Optional[BaseException] = None
    try:
        speed_address = REG_SPEED_SET + 2 * joint_index
        force_address = REG_FORCE_SET + 2 * joint_index
        hand.write_short(speed_address, original_speed)
        if hand.read_short(speed_address) != original_speed:
            raise RH56Error("speed restore readback mismatch")
        hand.write_short(force_address, original_force)
        if hand.read_short(force_address) != original_force:
            raise RH56Error("force restore readback mismatch")
        _disable_all_targets_verified(hand)
        print(
            "[full-cycle] restored SPEED_SET/FORCE_SET only after physical "
            "feedback stopped; re-verified all-six ANGLE_SET=-1"
        )
    except BaseException as exc:
        settings_error = exc
    if settings_error is not None:
        raise RH56Error(
            "FULL-CYCLE SETTINGS RESTORE FAILED; leave the hand untouched and "
            f"cut 24 V if any motion remains. {settings_error}"
        ) from operation_error
    if operation_error is not None:
        raise operation_error
    print("第六轴额定全行程验收完成")


def sweep(
    hand: RH56Hand,
    speed: int,
    force_limit: int,
    motion_timeout: float,
    joint_pause: float,
    include_thumb_rotate: bool,
    selected_joint: Optional[int],
) -> None:
    def build_path(
        original_angle: int, return_target: int
    ) -> Sequence[Tuple[str, int]]:
        del original_angle
        return (
            ("到达 0 端", 0),
            ("到达 1000 端", 1000),
            ("返回起点", return_target),
        )

    initial = hand.snapshot()
    if any(initial["errors"]):
        raise RH56Error("full sweep refused because at least one actuator reports a fault")
    if not all(target == -1 for target in initial["angle_targets"]):
        raise RH56Error("full sweep requires all ANGLE_SET=-1")
    if not all(status in (2, 0xFF) for status in initial["statuses"]):
        raise RH56Error("full sweep requires every STATUS to be 2 or 0xFF")
    if max(initial["temperatures"]) >= 60:
        raise RH56Error("full sweep refused because an actuator is at least 60 °C")
    if selected_joint is not None:
        joint_indices = (selected_joint,)
        scope = f"{JOINT_LABELS[selected_joint]}单轴"
    else:
        joint_count = len(JOINTS) if include_thumb_rotate else len(JOINTS) - 1
        joint_indices = tuple(range(joint_count))
        scope = "六自由度" if include_thumb_rotate else "五个手指弯曲轴"
    print(f"开始{scope}逐轴全行程测试（0 -> 1000 -> 原位置）")
    for progress, joint_index in enumerate(joint_indices, start=1):
        joint_label = JOINT_LABELS[joint_index]
        print(f"\n[{progress}/{len(joint_indices)}] {joint_label}")
        move_joint_path(
            hand,
            joint_index,
            speed,
            force_limit,
            motion_timeout,
            "全行程",
            build_path,
            require_disabled_idle=True,
            monitor_all=True,
            stop_in_place_on_error=True,
        )
        if progress < len(joint_indices):
            time.sleep(joint_pause)
    final = hand.snapshot()
    for key in ("angle_targets", "speeds", "force_limits"):
        if final[key] != initial[key]:
            raise RH56Error(f"full sweep finished but failed to restore {key}")
    if any(final["errors"]):
        raise RH56Error("full sweep finished with an actuator fault")
    for index in joint_indices:
        initial_angle = initial["angles"][index]
        tolerance = (
            SWEEP_ENDPOINT_TOLERANCE
            if initial_angle <= SWEEP_ENDPOINT_TOLERANCE
            or initial_angle >= 1000 - SWEEP_ENDPOINT_TOLERANCE
            else SWEEP_POSITION_TOLERANCE
        )
        if abs(final["angles"][index] - initial_angle) > tolerance:
            raise RH56Error(
                f"{JOINT_LABELS[index]} did not return close to its initial position"
            )
    print(f"{scope}逐轴全行程测试完成")


def open_thumb_rotate_to_realtime_start(
    hand: RH56Hand,
    speed: int,
    force_limit: int,
    motion_timeout: float,
    safety_check: Optional[Callable[[], None]] = None,
) -> None:
    """Move axis 6 only toward its validated realtime open-side endpoint."""

    if speed != THUMB_ROTATE_REALTIME_OPEN_SPEED:
        raise RH56Error(
            "thumb-rotate open requires "
            f"speed={THUMB_ROTATE_REALTIME_OPEN_SPEED}"
        )
    if force_limit != THUMB_ROTATE_REALTIME_OPEN_FORCE_LIMIT:
        raise RH56Error(
            "thumb-rotate open requires "
            f"force_limit={THUMB_ROTATE_REALTIME_OPEN_FORCE_LIMIT}g"
        )
    if not 5 <= motion_timeout <= 30:
        raise RH56Error("thumb-rotate open requires motion_timeout in 5..30s")

    joint_index = JOINTS.index("thumb_rotate")

    def validate_preflight(snapshot: Dict[str, object], phase: str) -> None:
        if tuple(snapshot["angle_targets"]) != (-1,) * len(JOINTS):
            raise RH56Error(
                f"thumb-rotate open {phase} requires all six ANGLE_SET=-1"
            )
        if any(int(value) for value in snapshot["errors"]):
            raise RH56Error(
                f"thumb-rotate open {phase} refused because an axis reports a fault"
            )
        if not all(
            int(status) in (2, 0xFF) for status in snapshot["statuses"]
        ):
            raise RH56Error(
                f"thumb-rotate open {phase} requires every STATUS in {{2, 0xFF}}"
            )
        if max(int(value) for value in snapshot["temperatures"]) >= 50:
            raise RH56Error(
                f"thumb-rotate open {phase} refused because an axis is at least 50 °C"
            )
        currents = tuple(int(value) for value in snapshot["currents"])
        if any(abs(value) > NUDGE_MAX_IDLE_CURRENT_MA for value in currents):
            raise RH56Error(
                f"thumb-rotate open {phase} requires each idle current <= "
                f"{NUDGE_MAX_IDLE_CURRENT_MA} mA"
            )
        if sum(abs(value) for value in currents) > NUDGE_MAX_TOTAL_IDLE_CURRENT_MA:
            raise RH56Error(
                f"thumb-rotate open {phase} requires total idle current <= "
                f"{NUDGE_MAX_TOTAL_IDLE_CURRENT_MA} mA"
            )
        if any(
            int(angle) < THUMB_ROTATE_FULL_CYCLE_MIN_BEND_CLEARANCE
            for angle in snapshot["angles"][:5]
        ):
            raise RH56Error(
                "thumb-rotate open requires all five bend axes completely open at "
                f">={THUMB_ROTATE_FULL_CYCLE_MIN_BEND_CLEARANCE}"
            )
        thumb_angle = int(snapshot["angles"][joint_index])
        if not THUMB_ROTATE_REALTIME_OPEN_MIN_ANGLE <= thumb_angle <= 1000:
            raise RH56Error(
                f"thumb-rotate open ANGLE_ACT={thumb_angle} is outside the "
                f"validated {THUMB_ROTATE_REALTIME_OPEN_MIN_ANGLE}..1000 start range"
            )

    if safety_check is not None:
        safety_check()
    first = hand.snapshot()
    validate_preflight(first, "first preflight")
    time.sleep(0.15)
    if safety_check is not None:
        safety_check()
    second = hand.snapshot()
    validate_preflight(second, "second preflight")
    if any(
        abs(int(after) - int(before)) > 2
        for before, after in zip(first["angles"], second["angles"])
    ):
        raise RH56Error("thumb-rotate open preflight ANGLE_ACT was not stable")
    if any(
        abs(int(after) - int(before)) > 3
        for before, after in zip(first["positions"], second["positions"])
    ):
        raise RH56Error("thumb-rotate open preflight POS_ACT was not stable")

    start_angle = int(second["angles"][joint_index])
    if start_angle >= THUMB_ROTATE_REALTIME_OPEN_ACCEPT_ANGLE:
        _disable_all_targets_verified(hand)
        print(
            "拇指旋转已在实时控制张开端："
            f"ANGLE_ACT={start_angle}, ANGLE_SET=-1"
        )
        return

    def build_path(
        original_angle: int, return_target: int
    ) -> Sequence[Tuple[str, int]]:
        del original_angle, return_target
        return (("到达实时控制张开端", 1000),)

    operation_error: Optional[BaseException] = None
    try:
        move_joint_path(
            hand,
            joint_index,
            speed,
            force_limit,
            motion_timeout,
            "单程张开",
            build_path,
            require_disabled_idle=True,
            monitor_all=True,
            diagnostic_trace=True,
            max_joint_current_ma=THUMB_ROTATE_FULL_CYCLE_MAX_JOINT_CURRENT_MA,
            max_total_current_ma=THUMB_ROTATE_FULL_CYCLE_MAX_TOTAL_CURRENT_MA,
            verify_all_disabled_on_exit=True,
            batch_angle_write=True,
            position_tolerance=THUMB_ROTATE_REALTIME_OPEN_TOLERANCE,
            verify_inactive_axes_stationary=True,
            return_target_from_actual=lambda actual: 1000,
            verify_direction_during_motion=True,
            endpoint_stable_samples=3,
            endpoint_max_joint_current_ma=NUDGE_MAX_IDLE_CURRENT_MA,
            endpoint_max_total_current_ma=NUDGE_MAX_TOTAL_IDLE_CURRENT_MA,
            safety_check=safety_check,
        )
    except BaseException as exc:
        operation_error = exc

    stop_samples: List[Dict[str, object]] = []
    stop_error: Optional[BaseException] = None
    final_angle: Optional[int] = None
    try:
        if tuple(hand.read_six_shorts(REG_ANGLE_SET, retries=1)) != (-1,) * len(
            JOINTS
        ):
            raise RH56Error("thumb-rotate open final ANGLE_SET is not all -1")
        for sample_index in range(4):
            if safety_check is not None:
                safety_check()
            sample = hand.snapshot()
            stop_samples.append(sample)
            if sample_index < 3:
                time.sleep(0.15)
        verification_samples = stop_samples[-3:]
        thumb_angles = [
            int(sample["angles"][joint_index]) for sample in verification_samples
        ]
        thumb_positions = [
            int(sample["positions"][joint_index]) for sample in verification_samples
        ]
        if max(thumb_angles) - min(thumb_angles) > (
            THUMB_ROTATE_FULL_CYCLE_STOP_ANGLE_DRIFT
        ) or max(thumb_positions) - min(thumb_positions) > (
            THUMB_ROTATE_FULL_CYCLE_STOP_POSITION_DRIFT
        ):
            raise RH56Error(
                "thumb-rotate feedback is still moving after release: "
                f"angles={thumb_angles}, positions={thumb_positions}"
            )
        for sample in verification_samples:
            currents = tuple(int(value) for value in sample["currents"])
            if any(int(value) for value in sample["errors"]):
                raise RH56Error(
                    f"post-release actuator fault remains: {sample['errors']}"
                )
            if not all(
                int(status) in (2, 0xFF) for status in sample["statuses"]
            ):
                raise RH56Error(
                    f"post-release status is not idle: {sample['statuses']}"
                )
            if any(abs(value) > NUDGE_MAX_IDLE_CURRENT_MA for value in currents):
                raise RH56Error(
                    f"post-release current did not return to idle: {currents}"
                )
            if sum(abs(value) for value in currents) > (
                NUDGE_MAX_TOTAL_IDLE_CURRENT_MA
            ):
                raise RH56Error(
                    f"post-release total current did not return to idle: {currents}"
                )
            if max(int(value) for value in sample["temperatures"]) >= 60:
                raise RH56Error(
                    f"post-release temperature is unsafe: {sample['temperatures']}"
                )
        final_angle = thumb_angles[-1]
        print(
            "[thumb-open] verified ANGLE_SET=-1 and physical feedback stopped: "
            f"ANGLE_ACT={final_angle}, POSITION_ACT={thumb_positions[-1]}"
        )
    except BaseException as exc:
        stop_error = exc

    if stop_error is not None:
        raise RH56Error(
            "THUMB-OPEN MOTION STOP UNCONFIRMED; cut 24 V now. "
            f"{stop_error}"
        ) from operation_error
    if operation_error is not None:
        raise operation_error
    if final_angle is None or final_angle < THUMB_ROTATE_REALTIME_OPEN_ACCEPT_ANGLE:
        raise RH56Error(
            f"thumb-rotate stopped at {final_angle}, expected at least "
            f"{THUMB_ROTATE_REALTIME_OPEN_ACCEPT_ANGLE}"
        )


def open_hand(
    hand: RH56Hand,
    speed: int,
    force_limit: int,
    motion_timeout: float,
    include_thumb_rotate: bool = False,
    simultaneous_bend_open: bool = False,
    safety_check: Optional[Callable[[], None]] = None,
    max_axis_current_ma: Optional[int] = None,
    endpoint_stable_samples: Optional[int] = None,
    max_inactive_drift_units: Optional[int] = None,
    feedback_callback: Optional[Callable[[Mapping[str, object]], None]] = None,
) -> None:
    if not isinstance(simultaneous_bend_open, bool):
        raise ValueError("simultaneous_bend_open must be boolean")
    if simultaneous_bend_open and include_thumb_rotate:
        raise ValueError(
            "simultaneous bend open cannot include thumb rotation"
        )
    if max_axis_current_ma is not None:
        if (
            isinstance(max_axis_current_ma, bool)
            or int(max_axis_current_ma) != float(max_axis_current_ma)
            or not 1 <= int(max_axis_current_ma) <= 1400
        ):
            raise ValueError("max_axis_current_ma must be an integer in 1..1400")
        current_cap = int(max_axis_current_ma)
    else:
        current_cap = None
    if endpoint_stable_samples is None:
        stable_required = 1
    elif (
        isinstance(endpoint_stable_samples, bool)
        or int(endpoint_stable_samples) != float(endpoint_stable_samples)
        or not 1 <= int(endpoint_stable_samples) <= 10
    ):
        raise ValueError("endpoint_stable_samples must be an integer in 1..10")
    else:
        stable_required = int(endpoint_stable_samples)
    if max_inactive_drift_units is None:
        inactive_drift_limit = None
    elif (
        isinstance(max_inactive_drift_units, bool)
        or int(max_inactive_drift_units) != float(max_inactive_drift_units)
        or not 1 <= int(max_inactive_drift_units) <= 20
    ):
        raise ValueError("max_inactive_drift_units must be an integer in 1..20")
    else:
        inactive_drift_limit = int(max_inactive_drift_units)
    if safety_check is not None:
        safety_check()
    snapshot = hand.snapshot()
    if tuple(int(value) for value in snapshot["angle_targets"]) != (-1,) * 6:
        raise RH56Error("open refused because ANGLE_SET is not all -1")
    if any(snapshot["errors"]):
        raise RH56Error("open refused because at least one actuator reports a fault")
    if max(snapshot["temperatures"]) >= 60:
        raise RH56Error("open refused because an actuator is at least 60 °C")
    if current_cap is not None:
        preflight_over_current = tuple(
            (index, int(current))
            for index, current in enumerate(snapshot["currents"])
            if abs(int(current)) > current_cap
        )
        if preflight_over_current:
            raise RH56Error(
                "open helper preflight exceeded host per-axis current cap "
                f"{current_cap}mA: {preflight_over_current}"
            )
    if any(int(status) not in (2, 0xFF) for status in snapshot["statuses"]):
        raise RH56Error("open refused because at least one actuator is not idle")

    if simultaneous_bend_open:
        print(
            "同时张开五个手指弯曲轴：一次写入 ANGLE_SET=1000；"
            "到位后统一设回 -1"
        )
        start_angles = tuple(int(value) for value in snapshot["angles"])
        expected_targets = (1000, 1000, 1000, 1000, 1000, -1)
        commanded_speeds = (int(speed),) * 5 + (
            int(snapshot["speeds"][5]),
        )
        commanded_forces = (int(force_limit),) * 5 + (
            int(snapshot["force_limits"][5]),
        )
        deadline = time.monotonic() + motion_timeout
        peak_currents = [0] * 5
        stable = 0
        try:
            if safety_check is not None:
                safety_check()
            hand.write_six_shorts(REG_SPEED_SET, commanded_speeds)
            hand.write_six_shorts(REG_FORCE_SET, commanded_forces)
            # One contiguous register transaction starts all five bend axes;
            # q6 remains disabled and therefore cannot be moved by this reset.
            hand.write_six_shorts(REG_ANGLE_SET, expected_targets)
            target_readback = tuple(hand.read_six_shorts(REG_ANGLE_SET))
            if target_readback != expected_targets:
                raise RH56Error(
                    "simultaneous bend ANGLE_SET readback failed: "
                    f"{target_readback}"
                )

            while True:
                if safety_check is not None:
                    safety_check()
                targets_before = tuple(hand.read_six_shorts(REG_ANGLE_SET))
                angles = tuple(hand.read_six_shorts(REG_ANGLE_ACT))
                positions = tuple(hand.read_six_shorts(REG_POS_ACT))
                currents = tuple(hand.read_six_shorts(REG_CURRENT))
                errors = tuple(hand.read(REG_ERROR, 6))
                statuses = tuple(hand.read(REG_STATUS, 6))
                temperatures = tuple(hand.read(REG_TEMP, 6))
                targets_after = tuple(hand.read_six_shorts(REG_ANGLE_SET))
                if (
                    targets_before != expected_targets
                    or targets_after != expected_targets
                ):
                    raise RH56Error(
                        "simultaneous bend ANGLE_SET changed during opening: "
                        f"before={targets_before}, after={targets_after}"
                    )
                for axis in range(5):
                    peak_currents[axis] = max(
                        peak_currents[axis], abs(int(currents[axis]))
                    )
                if current_cap is not None:
                    over_current = tuple(
                        (index, int(current))
                        for index, current in enumerate(currents)
                        if abs(int(current)) > current_cap
                    )
                    if over_current:
                        raise RH56Error(
                            "simultaneous open exceeded host per-axis current "
                            f"cap {current_cap}mA: {over_current}"
                        )
                if any(errors):
                    fault_index = next(
                        index for index, value in enumerate(errors) if value
                    )
                    raise RH56Error(
                        f"{JOINT_LABELS[fault_index]} reported "
                        f"{decode_error(errors[fault_index])} while opening"
                    )
                if max(temperatures) >= 60:
                    raise RH56Error(
                        "an actuator reached at least 60 °C while opening"
                    )
                for axis in range(5):
                    if int(angles[axis]) < start_angles[axis] - max(
                        4, SWEEP_ENDPOINT_TOLERANCE // 2
                    ):
                        raise RH56Error(
                            f"{JOINT_LABELS[axis]} moved opposite the open command"
                        )
                    if int(statuses[axis]) not in (0, 1, 2):
                        raise RH56Error(
                            f"{JOINT_LABELS[axis]} stopped while opening: "
                            f"{STATUS_LABELS.get(int(statuses[axis]), statuses[axis])}"
                        )
                if inactive_drift_limit is not None:
                    if (
                        abs(int(angles[5]) - start_angles[5])
                        > inactive_drift_limit
                    ):
                        raise RH56Error(
                            "inactive thumb rotation drifted during simultaneous "
                            "bend opening"
                        )
                    if int(statuses[5]) not in (2, 0xFF):
                        raise RH56Error(
                            "inactive thumb rotation left idle during "
                            "simultaneous bend opening"
                        )
                if safety_check is not None:
                    safety_check()
                if feedback_callback is not None:
                    feedback_callback(
                        {
                            "phase": "reset_open_bends_all_1000",
                            "angle_targets": targets_after,
                            "angles": angles,
                            "positions": positions,
                            "currents": currents,
                            "errors": errors,
                            "statuses": statuses,
                            "temperatures": temperatures,
                        }
                    )
                reached = all(
                    int(angles[axis]) >= 1000 - SWEEP_ENDPOINT_TOLERANCE
                    and int(positions[axis]) <= OPEN_POSITION_MAX
                    and int(statuses[axis]) == 2
                    for axis in range(5)
                )
                stable = stable + 1 if reached else 0
                if stable >= stable_required:
                    print(
                        "五轴同时张开完成: angles={} positions={} "
                        "peak_currents={}mA".format(
                            list(int(value) for value in angles[:5]),
                            list(int(value) for value in positions[:5]),
                            peak_currents,
                        )
                    )
                    break
                if time.monotonic() >= deadline:
                    raise RH56Error(
                        "five bend axes did not physically open within "
                        f"{motion_timeout:.1f}s; angles={angles[:5]}, "
                        f"positions={positions[:5]}, statuses={statuses[:5]}, "
                        f"peak_currents={tuple(peak_currents)}mA"
                    )
                time.sleep(0.08)
        except BaseException:
            # Hold the measured bend pose before disabling, matching the
            # existing per-axis cleanup without issuing another open command.
            try:
                hold_angles = tuple(hand.read_six_shorts(REG_ANGLE_ACT))
                hand.write_six_shorts(
                    REG_ANGLE_SET, tuple(hold_angles[:5]) + (-1,)
                )
            finally:
                hand.write_six_shorts(REG_ANGLE_SET, (-1,) * 6)
                hand.write_six_shorts(REG_SPEED_SET, (1000,) * 6)
                hand.write_six_shorts(REG_FORCE_SET, (500,) * 6)
            raise

        hand.write_six_shorts(REG_SPEED_SET, (1000,) * 6)
        hand.write_six_shorts(REG_FORCE_SET, (500,) * 6)
        hand.write_six_shorts(REG_ANGLE_SET, (-1,) * 6)
        disabled_readback = tuple(hand.read_six_shorts(REG_ANGLE_SET))
        if disabled_readback != (-1,) * 6:
            raise RH56Error(
                "failed to disable all ANGLE_SET after simultaneous opening: "
                f"{disabled_readback}"
            )
        print(
            "五个手指已同时张开，目标已设回 -1（不执行新动作）；"
            "拇指旋转轴未改变"
        )
        return

    print("依次张开五个手指弯曲轴；到位后将 ANGLE_SET 设回 -1")
    bend_open_min_angle = (
        THUMB_ROTATE_FULL_CYCLE_MIN_BEND_CLEARANCE
        if include_thumb_rotate
        else 1000 - SWEEP_ENDPOINT_TOLERANCE
    )
    for joint_index in range(len(JOINTS) - 1):
        if safety_check is not None:
            safety_check()
        target_address = REG_ANGLE_SET + 2 * joint_index
        speed_address = REG_SPEED_SET + 2 * joint_index
        force_address = REG_FORCE_SET + 2 * joint_index
        start_angles = tuple(hand.read_six_shorts(REG_ANGLE_ACT))
        start_positions = tuple(hand.read_six_shorts(REG_POS_ACT))
        start_angle = int(start_angles[joint_index])
        start_position = int(start_positions[joint_index])
        print(
            f"[{joint_index + 1}/5] {JOINT_LABELS[joint_index]}: "
            f"angle={start_angle}, position={start_position} -> open"
        )
        hand.write_short(speed_address, speed)
        hand.write_short(force_address, force_limit)
        hand.write_short(target_address, 1000)
        expected_targets = tuple(
            1000 if index == joint_index else -1 for index in range(6)
        )
        target_readback = tuple(hand.read_six_shorts(REG_ANGLE_SET))
        if target_readback != expected_targets:
            raise RH56Error(
                f"{JOINT_LABELS[joint_index]} masked ANGLE_SET readback failed: "
                f"{target_readback}"
            )

        deadline = time.monotonic() + motion_timeout
        peak_current = 0
        stable = 0
        try:
            while True:
                if safety_check is not None:
                    safety_check()
                targets_before = tuple(hand.read_six_shorts(REG_ANGLE_SET))
                angles = tuple(hand.read_six_shorts(REG_ANGLE_ACT))
                positions = tuple(hand.read_six_shorts(REG_POS_ACT))
                currents = hand.read_six_shorts(REG_CURRENT)
                errors = tuple(hand.read(REG_ERROR, 6))
                statuses = tuple(hand.read(REG_STATUS, 6))
                temperatures = tuple(hand.read(REG_TEMP, 6))
                targets_after = tuple(hand.read_six_shorts(REG_ANGLE_SET))
                if (
                    targets_before != expected_targets
                    or targets_after != expected_targets
                ):
                    raise RH56Error(
                        f"{JOINT_LABELS[joint_index]} ANGLE_SET changed during "
                        f"opening: before={targets_before}, after={targets_after}"
                    )
                angle = int(angles[joint_index])
                position = int(positions[joint_index])
                peak_current = max(peak_current, abs(currents[joint_index]))
                if current_cap is not None:
                    over_current = tuple(
                        (index, int(current))
                        for index, current in enumerate(currents)
                        if abs(int(current)) > current_cap
                    )
                    if over_current:
                        raise RH56Error(
                            "open helper exceeded host per-axis current cap "
                            f"{current_cap}mA: {over_current}"
                        )
                if any(errors):
                    fault_index = next(
                        index for index, value in enumerate(errors) if value
                    )
                    raise RH56Error(
                        f"{JOINT_LABELS[fault_index]} reported "
                        f"{decode_error(errors[fault_index])} while opening"
                    )
                if max(temperatures) >= 60:
                    raise RH56Error("an actuator reached at least 60 °C while opening")
                if inactive_drift_limit is not None:
                    for inactive_index in range(6):
                        if inactive_index == joint_index:
                            continue
                        if (
                            abs(
                                int(angles[inactive_index])
                                - int(start_angles[inactive_index])
                            )
                            > inactive_drift_limit
                        ):
                            raise RH56Error(
                                f"inactive {JOINT_LABELS[inactive_index]} drifted "
                                f"while opening {JOINT_LABELS[joint_index]}"
                            )
                        if int(statuses[inactive_index]) not in (2, 0xFF):
                            raise RH56Error(
                                f"inactive {JOINT_LABELS[inactive_index]} left idle "
                                f"while opening {JOINT_LABELS[joint_index]}"
                            )
                if angle < start_angle - max(4, SWEEP_ENDPOINT_TOLERANCE // 2):
                    raise RH56Error(
                        f"{JOINT_LABELS[joint_index]} moved opposite the open command"
                    )
                status = int(statuses[joint_index])
                if status not in (0, 1, 2):
                    raise RH56Error(
                        f"{JOINT_LABELS[joint_index]} stopped while opening: "
                        f"{STATUS_LABELS.get(status, status)}"
                    )
                if safety_check is not None:
                    safety_check()
                if feedback_callback is not None:
                    feedback_callback(
                        {
                            "phase": (
                                f"reset_open_bend_m{joint_index}_1000"
                            ),
                            "angle_targets": targets_after,
                            "angles": angles,
                            "positions": positions,
                            "currents": tuple(int(value) for value in currents),
                            "errors": errors,
                            "statuses": statuses,
                            "temperatures": temperatures,
                        }
                    )
                reached = (
                    angle >= bend_open_min_angle
                    and position <= OPEN_POSITION_MAX
                    and status == 2
                )
                stable = stable + 1 if reached else 0
                if stable >= stable_required:
                    print(
                        f"    已张开: angle={angle}, position={position}, "
                        f"peak_current={peak_current}mA"
                    )
                    break
                if time.monotonic() >= deadline:
                    raise RH56Error(
                        f"{JOINT_LABELS[joint_index]} did not physically open within "
                        f"{motion_timeout:.1f}s; angle={angle}, position={position}, "
                        f"status={status}, peak_current={peak_current}mA"
                    )
                time.sleep(0.08)
        except BaseException:
            hold_angle = int(hand.read_six_shorts(REG_ANGLE_ACT)[joint_index])
            hand.write_short(target_address, hold_angle)
            hand.write_short(target_address, -1)
            hand.write_short(speed_address, 1000)
            hand.write_short(force_address, 500)
            raise

        hand.write_short(speed_address, 1000)
        hand.write_short(force_address, 500)
        hand.write_short(target_address, -1)
        disabled_readback = tuple(hand.read_six_shorts(REG_ANGLE_SET))
        if disabled_readback != (-1,) * 6:
            raise RH56Error(
                f"failed to leave all ANGLE_SET disabled after opening "
                f"{JOINT_LABELS[joint_index]}: {disabled_readback}"
            )
        time.sleep(0.5)
    if include_thumb_rotate:
        print("五个弯曲轴已张开；开始低速单程张开拇指旋转轴")
        open_thumb_rotate_to_realtime_start(
            hand,
            speed,
            force_limit,
            motion_timeout,
            safety_check=safety_check,
        )
        print("六轴已处于实时控制张开起点，目标均已设回 -1")
    else:
        print("五个手指已张开，目标已设回 -1（不执行新动作）；拇指旋转轴未改变")


def write_check(hand: RH56Hand) -> None:
    """Verify the write/ACK path using the manual's -1 (no action) target."""
    address = REG_ANGLE_SET
    original_target = hand.read_short(address)
    if original_target != -1:
        raise RH56Error(
            "write-check refused: pinky ANGLE_SET is not -1, so a no-motion "
            f"write cannot be guaranteed (value={original_target})"
        )
    hand.write_short(address, -1)
    readback = hand.read_short(address)
    if readback != -1:
        raise RH56Error(f"write-check readback is {readback}, expected -1")
    print("无动作写入测试通过：ANGLE_SET(小拇指) -1 -> ACK -> readback -1")


def disable_all(hand: RH56Hand) -> None:
    """Disable all six angle targets and verify the no-motion command twice."""

    _disable_all_targets_verified(hand)
    print(
        "六轴已禁用并双重回读确认："
        "ANGLE_SET=[-1, -1, -1, -1, -1, -1]"
    )


def self_test() -> None:
    official_read = bytes.fromhex("EB 90 01 04 11 0A 06 0C 32")
    assert build_read_frame(1, 1546, 12) == official_read

    angle_data = struct.pack("<6h", 100, 100, 100, 100, 1000, 0)
    official_write = bytes.fromhex(
        "EB 90 01 0F 12 CE 05 64 00 64 00 64 00 64 00 E8 03 00 00 70"
    )
    assert build_write_frame(1, 1486, angle_data) == official_write

    official_response = bytes.fromhex(
        "90 EB 01 0F 11 0A 06 64 00 64 00 64 00 64 00 E8 03 00 00 AC"
    )
    parsed = parse_response(official_response, 1, READ_COMMAND, 1546, 12)
    assert struct.unpack("<6h", parsed) == (100, 100, 100, 100, 1000, 0)
    print("协议自检通过：读帧、写帧和官方 V1.09 示例完全一致")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test an Inspire RH56BFX/DFX hand through USB-RS485"
    )
    parser.add_argument(
        "action",
        nargs="?",
        choices=(
            "status",
            "clear-error",
            "recover-stale-q6-status",
            "write-check",
            "disable",
            "open",
            "recover-middle-position",
            "nudge",
            "probe",
            "visual-cycle",
            "thumb-full-cycle",
            "sweep",
            "self-test",
        ),
        default="status",
        help=(
            "status is read-only; clear-error clears one proved disabled/idle "
            "latched fault (no motion); recover-stale-q6-status installs one "
            "bounded feedback-equivalent q6 hold; write-check writes one -1 "
            "(no action); "
            "disable writes all six -1 targets (no action); "
            "open leaves five bend axes open and can explicitly open thumb "
            "rotation for realtime startup; nudge moves one joint; "
            "recover-middle-position performs one bounded raw-position repair "
            "for a middle actuator just beyond its open endpoint; "
            "probe performs a one-way thumb-rotation response test; "
            "visual-cycle performs one bounded visible thumb-rotation round trip; "
            "thumb-full-cycle performs one supervised rated axis-6 cycle; "
            "sweep tests each finger over 0..1000"
        ),
    )
    parser.add_argument("--port", help="serial device; auto-detected if unique")
    parser.add_argument("--baud", type=int, choices=tuple(BAUD_CONSTANTS), default=115200)
    parser.add_argument("--id", type=int, default=1, help="hand ID, factory default: 1")
    parser.add_argument("--timeout", type=float, default=0.5)
    parser.add_argument(
        "--motion-timeout",
        type=float,
        default=20.0,
        help="seconds to wait for each movement leg; independent of serial timeout",
    )
    parser.add_argument("--debug", action="store_true", help="print raw TX/RX frames")
    parser.add_argument(
        "--confirm-clear-error",
        metavar="TOKEN",
        help=(
            "exact confirmation token required for clear-error after a "
            "read-only status inspection"
        ),
    )
    parser.add_argument(
        "--confirm-stale-q6-recovery",
        metavar="TOKEN",
        help=(
            "exact confirmation token required for the bounded stale q6 "
            "status recovery"
        ),
    )
    parser.add_argument("--joint", choices=JOINTS, default="pinky")
    parser.add_argument(
        "--delta",
        type=int,
        default=20,
        help=(
            "signed nudge in protocol units: -100..-20 decreases ANGLE_SET, "
            "20..100 increases it"
        ),
    )
    parser.add_argument(
        "--speed", type=int, default=100, help="movement speed in protocol units"
    )
    parser.add_argument(
        "--force-limit",
        type=int,
        default=100,
        help="temporary movement force threshold in grams (action-specific range)",
    )
    parser.add_argument(
        "--confirm-movement",
        action="store_true",
        help=(
            "required for open/nudge/probe/visual-cycle/thumb-full-cycle; "
            "confirms the hand "
            "workspace is clear"
        ),
    )
    parser.add_argument(
        "--confirm-visual-cycle",
        action="store_true",
        help="required for visual-cycle; confirms continuous observation",
    )
    parser.add_argument(
        "--confirm-rated-thumb-cycle",
        metavar="TOKEN",
        help=(
            "exact confirmation token required for thumb-full-cycle; this action "
            "touches both rated axis-6 endpoints"
        ),
    )
    parser.add_argument(
        "--confirm-middle-position-recovery",
        metavar="TOKEN",
        help=(
            "exact confirmation token required for the middle raw-position "
            "recovery operation"
        ),
    )
    parser.add_argument(
        "--confirm-full-sweep",
        action="store_true",
        help="required for sweep; confirms continuous supervision and a clear workspace",
    )
    parser.add_argument(
        "--joint-pause",
        type=float,
        default=0.8,
        help="pause in seconds between joints during sweep",
    )
    parser.add_argument(
        "--include-thumb-rotate",
        action="store_true",
        help=(
            "explicitly allow thumb rotation during open/nudge/probe/visual-cycle/"
            "thumb-full-cycle/sweep; "
            "disabled by default to reduce self-collision risk"
        ),
    )
    parser.add_argument(
        "--batch-angle-write",
        action="store_true",
        help=(
            "for thumb_rotate nudge, write all six ANGLE_SET registers in one "
            "frame with the other five set to -1; matches the realtime path"
        ),
    )
    parser.add_argument(
        "--sweep-only",
        action="store_true",
        help="during sweep, test only the axis selected by --joint",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "self-test":
        self_test()
        return 0
    if not 1 <= args.id <= 254:
        print("error: --id must be in 1..254", file=sys.stderr)
        return 2
    if args.timeout <= 0:
        print("error: --timeout must be positive", file=sys.stderr)
        return 2
    if args.motion_timeout <= 0:
        print("error: --motion-timeout must be positive", file=sys.stderr)
        return 2
    if args.joint_pause < 0:
        print("error: --joint-pause must be non-negative", file=sys.stderr)
        return 2
    if args.action == "open":
        if not args.confirm_movement:
            print(
                "error: open requires --confirm-movement after clearing the hand workspace",
                file=sys.stderr,
            )
            return 2
        if not 1 <= args.speed <= 1000:
            print("error: --speed must be in 1..1000 for open", file=sys.stderr)
            return 2
        if not 1 <= args.force_limit <= 1000:
            print("error: --force-limit must be in 1..1000 for open", file=sys.stderr)
            return 2
        if args.include_thumb_rotate:
            if args.speed != THUMB_ROTATE_REALTIME_OPEN_SPEED:
                print(
                    "error: open --include-thumb-rotate requires "
                    f"--speed {THUMB_ROTATE_REALTIME_OPEN_SPEED}",
                    file=sys.stderr,
                )
                return 2
            if args.force_limit != THUMB_ROTATE_REALTIME_OPEN_FORCE_LIMIT:
                print(
                    "error: open --include-thumb-rotate requires "
                    f"--force-limit {THUMB_ROTATE_REALTIME_OPEN_FORCE_LIMIT}",
                    file=sys.stderr,
                )
                return 2
            if not 5 <= args.motion_timeout <= 30:
                print(
                    "error: open --include-thumb-rotate requires "
                    "--motion-timeout in 5..30 seconds",
                    file=sys.stderr,
                )
                return 2
    if (
        args.action == "clear-error"
        and args.confirm_clear_error != CLEAR_ERROR_TOKEN
    ):
        print(
            "error: clear-error requires --confirm-clear-error "
            f"{CLEAR_ERROR_TOKEN}",
            file=sys.stderr,
        )
        return 2
    if (
        args.action == "recover-stale-q6-status"
        and args.confirm_stale_q6_recovery != STALE_Q6_RECOVERY_TOKEN
    ):
        print(
            "error: recover-stale-q6-status requires "
            "--confirm-stale-q6-recovery "
            f"{STALE_Q6_RECOVERY_TOKEN}",
            file=sys.stderr,
        )
        return 2
    if (
        args.action == "recover-middle-position"
        and args.confirm_middle_position_recovery != MIDDLE_POSITION_RECOVERY_TOKEN
    ):
        print(
            "error: recover-middle-position requires "
            "--confirm-middle-position-recovery "
            f"{MIDDLE_POSITION_RECOVERY_TOKEN}",
            file=sys.stderr,
        )
        return 2
    if args.action in (
        "nudge",
        "probe",
        "visual-cycle",
        "thumb-full-cycle",
        "sweep",
    ):
        confirmation = (
            args.confirm_movement
            if args.action
            in ("nudge", "probe", "visual-cycle", "thumb-full-cycle")
            else args.confirm_full_sweep
        )
        required_flag = (
            "--confirm-movement"
            if args.action
            in ("nudge", "probe", "visual-cycle", "thumb-full-cycle")
            else "--confirm-full-sweep"
        )
        if not confirmation:
            print(
                f"error: {args.action} requires {required_flag} after clearing "
                "the hand workspace",
                file=sys.stderr,
            )
            return 2
        if args.action == "nudge" and not 20 <= abs(args.delta) <= 100:
            print(
                "error: --delta must be -100..-20 or 20..100",
                file=sys.stderr,
            )
            return 2
        if args.action == "probe" and not 20 <= abs(args.delta) <= 40:
            print(
                "error: probe --delta must be -40..-20 or 20..40",
                file=sys.stderr,
            )
            return 2
        if args.action == "visual-cycle" and not args.confirm_visual_cycle:
            print(
                "error: visual-cycle also requires --confirm-visual-cycle",
                file=sys.stderr,
            )
            return 2
        if (
            args.action == "thumb-full-cycle"
            and args.confirm_rated_thumb_cycle != THUMB_ROTATE_FULL_CYCLE_TOKEN
        ):
            print(
                "error: thumb-full-cycle requires "
                "--confirm-rated-thumb-cycle "
                f"{THUMB_ROTATE_FULL_CYCLE_TOKEN}",
                file=sys.stderr,
            )
            return 2
        minimum_speed = (
            1
            if args.action
            in ("nudge", "probe", "visual-cycle", "thumb-full-cycle")
            else 100
        )
        if not minimum_speed <= args.speed <= 300:
            print(
                f"error: --speed must be in {minimum_speed}..300 for {args.action}",
                file=sys.stderr,
            )
            return 2
        if not 1 <= args.force_limit <= 300:
            print("error: --force-limit must be in 1..300", file=sys.stderr)
            return 2
        if args.action == "probe" and not 0.5 <= args.motion_timeout <= 5:
            print(
                "error: probe requires --motion-timeout in 0.5..5 seconds",
                file=sys.stderr,
            )
            return 2
        if args.action == "visual-cycle":
            if args.speed != 40:
                print("error: visual-cycle requires --speed 40", file=sys.stderr)
                return 2
            if args.force_limit != 80:
                print(
                    "error: visual-cycle requires --force-limit 80",
                    file=sys.stderr,
                )
                return 2
            if not 5 <= args.motion_timeout <= 15:
                print(
                    "error: visual-cycle requires --motion-timeout in 5..15 seconds",
                    file=sys.stderr,
                )
                return 2
        if args.action == "thumb-full-cycle":
            if args.speed != 40:
                print(
                    "error: thumb-full-cycle requires --speed 40",
                    file=sys.stderr,
                )
                return 2
            if args.force_limit != 80:
                print(
                    "error: thumb-full-cycle requires --force-limit 80",
                    file=sys.stderr,
                )
                return 2
            if not 30 <= args.motion_timeout <= 180:
                print(
                    "error: thumb-full-cycle requires --motion-timeout in "
                    "30..180 seconds",
                    file=sys.stderr,
                )
                return 2
        if args.action == "sweep" and args.motion_timeout < 15:
            print("error: sweep requires --motion-timeout of at least 15s", file=sys.stderr)
            return 2
        if (
            args.joint == "thumb_rotate"
            and (
                args.action == "nudge"
                or args.action == "probe"
                or args.action == "visual-cycle"
                or args.action == "thumb-full-cycle"
                or (args.action == "sweep" and args.sweep_only)
            )
            and not args.include_thumb_rotate
        ):
            print(
                f"error: thumb_rotate {args.action} also requires "
                "--include-thumb-rotate",
                file=sys.stderr,
            )
            return 2
        if (
            args.action in ("probe", "visual-cycle", "thumb-full-cycle")
            and args.joint != "thumb_rotate"
        ):
            print(
                f"error: {args.action} is restricted to --joint thumb_rotate",
                file=sys.stderr,
            )
            return 2
    if args.batch_angle_write and not (
        args.action == "nudge" and args.joint == "thumb_rotate"
    ):
        print(
            "error: --batch-angle-write is restricted to thumb_rotate nudge",
            file=sys.stderr,
        )
        return 2

    try:
        port = args.port or find_serial_port()
        with LinuxSerial(port, args.baud, args.timeout, args.debug) as serial_port:
            hand = RH56Hand(serial_port, args.id)
            # ``disable`` is the stop path: do not delay it with the 15-register
            # diagnostic snapshot performed by the other actions.
            if args.action == "disable":
                disable_all(hand)
                return 0
            snapshot = hand.snapshot()
            print_snapshot(snapshot, port, args.baud)
            if args.action == "clear-error":
                print()
                final_snapshot = clear_latched_error(hand, snapshot)
                print(
                    "故障锁存清除并验证通过；六轴仍为 ANGLE_SET=-1，未执行运动命令"
                )
                print()
                print_snapshot(final_snapshot, port, args.baud)
            elif args.action == "recover-stale-q6-status":
                print()
                final_snapshot = recover_stale_thumb_rotate_status(hand, snapshot)
                print(
                    "拇指旋转陈旧状态恢复并验证通过；六轴已重新禁用，设置已恢复"
                )
                print()
                print_snapshot(final_snapshot, port, args.baud)
            elif args.action == "write-check":
                print()
                write_check(hand)
            elif args.action == "open":
                print()
                open_hand(
                    hand,
                    args.speed,
                    args.force_limit,
                    args.motion_timeout,
                    args.include_thumb_rotate,
                )
                print()
                print_snapshot(hand.snapshot(), port, args.baud)
            elif args.action == "nudge":
                print()
                nudge(
                    hand,
                    JOINTS.index(args.joint),
                    args.delta,
                    args.speed,
                    args.force_limit,
                    args.motion_timeout,
                    args.batch_angle_write,
                )
                print()
                print_snapshot(hand.snapshot(), port, args.baud)
            elif args.action == "recover-middle-position":
                print()
                recover_middle_positive_position(hand)
                print()
                print_snapshot(hand.snapshot(), port, args.baud)
            elif args.action == "probe":
                print()
                probe_thumb_rotate(
                    hand,
                    args.delta,
                    args.speed,
                    args.force_limit,
                    args.motion_timeout,
                )
                print()
                print_snapshot(hand.snapshot(), port, args.baud)
            elif args.action == "visual-cycle":
                print()
                visual_cycle_thumb_rotate(
                    hand,
                    args.speed,
                    args.force_limit,
                    args.motion_timeout,
                )
                print()
                print_snapshot(hand.snapshot(), port, args.baud)
            elif args.action == "thumb-full-cycle":
                print()
                full_cycle_thumb_rotate(
                    hand,
                    args.speed,
                    args.force_limit,
                    args.motion_timeout,
                )
                print()
                print_snapshot(hand.snapshot(), port, args.baud)
            elif args.action == "sweep":
                print()
                sweep(
                    hand,
                    args.speed,
                    args.force_limit,
                    args.motion_timeout,
                    args.joint_pause,
                    args.include_thumb_rotate,
                    JOINTS.index(args.joint) if args.sweep_only else None,
                )
                print()
                print_snapshot(hand.snapshot(), port, args.baud)
        return 0
    except KeyboardInterrupt:
        print("error: interrupted; remaining motion was cancelled", file=sys.stderr)
        return 130
    except (OSError, RH56Error, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
