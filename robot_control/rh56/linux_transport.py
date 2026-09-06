"""Deadline-bound Linux serial transport for transactional RH56 control.

Importing this module and constructing :class:`LinuxRH56TransactionalTransport`
perform no device discovery, module import, serial open, or register access.
The exclusive serial context is opened only by :meth:`open`/``with`` and must
be closed by the same owner after the transactional actuator has completed its
verified disable sequence.

The wire protocol and register definitions are deliberately reused from
``examples.inspire_rh56_test``.  Every individual RH56 request uses
``retries=0``.  Side-effect-free compact feedback and exact-target reads each
issue exactly one request inside one caller deadline and one per-register-read
total cap.  The protocol has no transaction sequence ID, so retransmitting an
identical read after a timeout would make two replies indistinguishable and
could leave the second reply in the stream for the next transaction.  Numeric
writes are never retried.  A small serial proxy binds each individual
``LinuxSerial.exchange`` to the caller's absolute monotonic deadline instead
of leaving the diagnostic module's usual fixed timeout in effect.
"""

from __future__ import annotations

import importlib
import math
import struct
import sys
import time
from typing import Any, Callable, Optional, Sequence, Tuple

from .actuator import (
    RH56SafetyFeedback,
    RH56TransactionalTransport,
)


DEFAULT_RH56_API_MODULE = "examples.inspire_rh56_test"
RH56_LINUX_TRANSPORT_NAME = "inspire-rh56-linux-serial"
# RH56 replies carry no request sequence number.  A compact register read is
# therefore sent exactly once: retrying the same request could accept the
# first reply as the retry's reply and leave a duplicate frame to corrupt the
# following transaction.
RH56_COMPACT_READ_ATTEMPTS = 1
# One side-effect-free register read must never inherit a multi-second caller
# deadline (notably the five-second physical-stop budget).  Fifty milliseconds
# is the existing produced-command freshness ceiling and exceeds the observed
# 34.3 ms worst complete-feedback command-window wait while still leaving the
# rest of a stop deadline available for hold, disable and stable-idle proof.
RH56_COMPACT_READ_TOTAL_TIMEOUT_S = 0.050


class RH56LinuxTransportError(RuntimeError):
    """A serial lifecycle, protocol, or response-contract failure."""


class RH56LinuxDeadlineExceeded(RH56LinuxTransportError):
    """An exchange could not complete inside its caller's absolute deadline."""


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


def _bounded_integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be an integer in {minimum}..{maximum}"
        ) from exc
    if (
        not math.isfinite(numeric)
        or not numeric.is_integer()
        or not minimum <= int(numeric) <= maximum
    ):
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    return int(numeric)


def _angle_targets(values: Sequence[int]) -> Tuple[int, ...]:
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise ValueError("ANGLE_SET must contain exactly six integers") from exc
    if len(raw) != 6:
        raise ValueError("ANGLE_SET must contain exactly six integers")
    return tuple(
        _bounded_integer(value, f"ANGLE_SET[{index}]", -1, 1000)
        for index, value in enumerate(raw)
    )


def _six_unsigned_settings(values: Sequence[int], name: str) -> Tuple[int, ...]:
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly six integers") from exc
    if len(raw) != 6:
        raise ValueError(f"{name} must contain exactly six integers")
    return tuple(
        _bounded_integer(value, f"{name}[{index}]", 0, 1000)
        for index, value in enumerate(raw)
    )


def _exact_bytes(value: object, expected_length: int, operation: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise RH56LinuxTransportError(
            f"{operation} returned {type(value).__name__}, expected bytes"
        )
    result = bytes(value)
    if len(result) != expected_length:
        raise RH56LinuxTransportError(
            f"{operation} returned {len(result)} bytes, expected {expected_length}"
        )
    return result


def _validate_register_map(api: Any) -> None:
    required = (
        "REG_ANGLE_SET",
        "REG_POS_ACT",
        "REG_ANGLE_ACT",
        "REG_FORCE_ACT",
        "REG_CURRENT",
        "REG_ERROR",
        "REG_STATUS",
        "REG_TEMP",
    )
    missing = [name for name in required if not hasattr(api, name)]
    if missing:
        raise RH56LinuxTransportError(
            "RH56 register map is incomplete: " + ", ".join(missing)
        )
    try:
        registers = {name: int(getattr(api, name)) for name in required}
    except (TypeError, ValueError) as exc:
        raise RH56LinuxTransportError(
            "RH56 register addresses must be integers"
        ) from exc
    adjacency = (
        ("REG_POS_ACT", "REG_ANGLE_ACT", 12),
        ("REG_FORCE_ACT", "REG_CURRENT", 12),
        ("REG_CURRENT", "REG_ERROR", 12),
        ("REG_ERROR", "REG_STATUS", 6),
        ("REG_STATUS", "REG_TEMP", 6),
    )
    for first, second, width in adjacency:
        if registers[second] != registers[first] + width:
            raise RH56LinuxTransportError(
                "RH56 compact register map is not contiguous: "
                f"{first}->{second}"
            )


class _DeadlineBoundSerial:
    """Give ``RH56Hand`` an exchange-only view of one ``LinuxSerial``.

    ``RH56Hand`` builds its request before calling ``exchange``.  Computing the
    remaining budget at this boundary keeps frame construction outside the
    blocking serial allowance and prevents a stale construction-time timeout
    (commonly 0.5 s) from outliving the transactional caller's watchdog.
    """

    def __init__(self, serial_port: Any, monotonic: Callable[[], float]) -> None:
        self._serial_port = serial_port
        self._monotonic = monotonic
        self._deadline_monotonic_s: Optional[float] = None

    def bind_deadline(self, deadline_monotonic_s: float) -> None:
        if self._deadline_monotonic_s is not None:
            raise RH56LinuxTransportError(
                "a second serial exchange was bound before the first completed"
            )
        self._deadline_monotonic_s = deadline_monotonic_s

    def clear_deadline(self) -> None:
        self._deadline_monotonic_s = None

    def discard_input(self) -> None:
        """Clear unread response bytes without writing a new request."""

        discard = getattr(self._serial_port, "discard_input", None)
        if callable(discard):
            discard()
            return
        # pyserial-compatible transports use this spelling.  The reviewed
        # standard-library LinuxSerial exposes ``discard_input``; this branch
        # keeps injected transports equally explicit.
        reset = getattr(self._serial_port, "reset_input_buffer", None)
        if callable(reset):
            reset()

    def exchange(self, request: bytes) -> bytes:
        deadline = self._deadline_monotonic_s
        if deadline is None:
            raise RH56LinuxTransportError(
                "RH56 serial exchange has no caller deadline"
            )
        now = _finite_float(self._monotonic(), "monotonic clock")
        remaining = deadline - now
        if remaining <= 0.0:
            raise RH56LinuxDeadlineExceeded(
                "RH56 serial deadline expired before exchange"
            )
        try:
            # LinuxSerial.exchange creates its relative deadline immediately
            # after entering the method.  Rebinding here (rather than at
            # transport construction) makes the blocking budget no larger
            # than the current caller budget at the exchange boundary.
            self._serial_port.timeout = remaining
            response = self._serial_port.exchange(request)
        except (KeyboardInterrupt, SystemExit):
            raise
        finally:
            self._deadline_monotonic_s = None
        finished = _finite_float(self._monotonic(), "monotonic clock")
        if finished > deadline:
            raise RH56LinuxDeadlineExceeded(
                "RH56 serial exchange exceeded caller deadline by "
                f"{finished - deadline:.6f}s"
            )
        return response


class LinuxRH56TransactionalTransport:
    """Concrete, context-managed :class:`RH56TransactionalTransport`.

    The constructor is inert.  Use ``with transport:`` (or explicitly call
    :meth:`open` and :meth:`close`) around the complete RH56 actuator lifetime.
    Closing the transport only closes the serial file descriptor; it never
    writes a release target and is therefore not a substitute for the
    actuator's verified disable protocol.
    """

    hardware_backed = True

    def __init__(
        self,
        port: Optional[str],
        *,
        baud_rate: int = 115200,
        hand_id: int = 1,
        debug: bool = False,
        api_module_name: str = DEFAULT_RH56_API_MODULE,
        module_loader: Callable[[str], Any] = importlib.import_module,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if port is not None and not str(port).strip():
            raise ValueError("port must be non-empty or None for delayed discovery")
        baud = _bounded_integer(baud_rate, "baud_rate", 1, 10_000_000)
        device_id = _bounded_integer(hand_id, "hand_id", 0, 254)
        module_name = str(api_module_name).strip()
        if not module_name:
            raise ValueError("api_module_name must be non-empty")
        if not callable(module_loader):
            raise ValueError("module_loader must be callable")
        if not callable(monotonic):
            raise ValueError("monotonic must be callable")

        self.port = None if port is None else str(port).strip()
        self.resolved_port: Optional[str] = None
        self.baud_rate = baud
        self.hand_id = device_id
        self.debug = bool(debug)
        self.transport_name = RH56_LINUX_TRANSPORT_NAME
        self._api_module_name = module_name
        self._module_loader = module_loader
        self._monotonic = monotonic
        self._api: Any = None
        self._serial_context: Any = None
        self._serial_proxy: Optional[_DeadlineBoundSerial] = None
        self._hand: Any = None
        self._open_attempted = False
        self._closed = False

    @property
    def is_open(self) -> bool:
        return self._hand is not None and not self._closed

    def _now(self) -> float:
        return _finite_float(self._monotonic(), "monotonic clock")

    def open(self) -> "LinuxRH56TransactionalTransport":
        """Lazily import the reviewed API and acquire its exclusive context."""

        if self.is_open:
            return self
        if self._closed:
            raise RH56LinuxTransportError("RH56 serial transport is closed")
        if self._open_attempted:
            raise RH56LinuxTransportError(
                "RH56 serial transport open already failed; construct a new instance"
            )
        self._open_attempted = True
        context: Any = None
        entered = False
        try:
            api = self._module_loader(self._api_module_name)
            _validate_register_map(api)
            if not callable(getattr(api, "LinuxSerial", None)):
                raise RH56LinuxTransportError("RH56 API has no LinuxSerial")
            if not callable(getattr(api, "RH56Hand", None)):
                raise RH56LinuxTransportError("RH56 API has no RH56Hand")
            if self.port is None:
                finder = getattr(api, "find_serial_port", None)
                if not callable(finder):
                    raise RH56LinuxTransportError(
                        "RH56 API has no delayed serial-port discovery"
                    )
                resolved_port = str(finder()).strip()
            else:
                resolved_port = self.port
            if not resolved_port:
                raise RH56LinuxTransportError("RH56 serial port resolved empty")

            # No exchange occurs during either constructor.  A zero initial
            # timeout makes accidental unbound use fail immediately; each
            # reviewed operation replaces it with the caller's remaining time.
            context = api.LinuxSerial(
                resolved_port,
                self.baud_rate,
                0.0,
                self.debug,
            )
            serial_port = context.__enter__()
            entered = True
            if serial_port is None:
                serial_port = context
            proxy = _DeadlineBoundSerial(serial_port, self._monotonic)
            hand = api.RH56Hand(proxy, self.hand_id)
        except (KeyboardInterrupt, SystemExit):
            if entered and context is not None:
                context.__exit__(*sys.exc_info())
            raise
        except Exception as exc:
            if entered and context is not None:
                try:
                    context.__exit__(*sys.exc_info())
                except Exception:
                    pass
            if isinstance(exc, RH56LinuxTransportError):
                raise
            detail = str(exc).strip() or type(exc).__name__
            raise RH56LinuxTransportError(
                f"opening RH56 serial transport failed: {detail}"
            ) from exc

        self._api = api
        self._serial_context = context
        self._serial_proxy = proxy
        self._hand = hand
        self.resolved_port = resolved_port
        return self

    def _require_open(self) -> Tuple[Any, Any, _DeadlineBoundSerial]:
        if not self.is_open:
            state = "closed" if self._closed else "not open"
            raise RH56LinuxTransportError(f"RH56 serial transport is {state}")
        assert self._api is not None
        assert self._hand is not None
        assert self._serial_proxy is not None
        return self._api, self._hand, self._serial_proxy

    def _deadline(self, value: object, operation: str) -> float:
        try:
            deadline = _finite_float(value, "deadline_monotonic_s")
        except ValueError as exc:
            raise RH56LinuxTransportError(str(exc)) from exc
        now = self._now()
        if deadline <= now:
            raise RH56LinuxDeadlineExceeded(
                f"RH56 {operation} deadline already expired"
            )
        return deadline

    def _invoke(
        self,
        operation: str,
        deadline_monotonic_s: object,
        callback: Callable[[], Any],
        serial_proxy: _DeadlineBoundSerial,
    ) -> Tuple[Any, float]:
        deadline = self._deadline(deadline_monotonic_s, operation)
        serial_proxy.bind_deadline(deadline)
        try:
            result = callback()
        except (RH56LinuxDeadlineExceeded, KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            detail = str(exc).strip() or type(exc).__name__
            raise RH56LinuxTransportError(
                f"RH56 {operation} failed: {type(exc).__name__}: {detail}"
            ) from exc
        finally:
            # Also clears a binding if a fake/malformed RH56Hand returned or
            # failed without invoking serial.exchange().
            serial_proxy.clear_deadline()
        finished = self._now()
        if finished > deadline:
            raise RH56LinuxDeadlineExceeded(
                f"RH56 {operation} exceeded caller deadline by "
                f"{finished - deadline:.6f}s"
            )
        return result, finished

    def _invoke_compact_read(
        self,
        operation: str,
        deadline_monotonic_s: object,
        callback: Callable[[], Any],
        serial_proxy: _DeadlineBoundSerial,
    ) -> Tuple[Any, float]:
        """Issue one side-effect-free compact read under one 50 ms cap.

        Numeric writes never enter this helper.  The request is deliberately
        not retransmitted because RH56 responses have no transaction sequence
        ID; a late first reply and a retry reply cannot be distinguished.
        """

        caller_deadline = self._deadline(deadline_monotonic_s, operation)
        total_deadline = min(
            caller_deadline,
            self._now() + RH56_COMPACT_READ_TOTAL_TIMEOUT_S,
        )
        return self._invoke(
            f"{operation} single read",
            total_deadline,
            callback,
            serial_proxy,
        )

    @staticmethod
    def _check_decode_deadline(
        now: float, deadline_monotonic_s: object, operation: str
    ) -> None:
        try:
            deadline = _finite_float(deadline_monotonic_s, "deadline_monotonic_s")
        except ValueError as exc:
            raise RH56LinuxTransportError(str(exc)) from exc
        if now > deadline:
            raise RH56LinuxDeadlineExceeded(
                f"RH56 {operation} decode exceeded caller deadline by "
                f"{now - deadline:.6f}s"
            )

    def write_angle_set(
        self,
        values: Tuple[int, ...],
        *,
        deadline_monotonic_s: float,
    ) -> None:
        api, hand, proxy = self._require_open()
        targets = _angle_targets(values)
        self._invoke(
            "ANGLE_SET batch write",
            deadline_monotonic_s,
            lambda: hand.write_six_shorts(
                int(api.REG_ANGLE_SET), targets, retries=0
            ),
            proxy,
        )

    def read_angle_set(self, *, deadline_monotonic_s: float) -> Tuple[int, ...]:
        api, hand, proxy = self._require_open()
        # ANGLE_SET verification is read-only and sent exactly once.  The
        # caller's original absolute deadline remains authoritative and a
        # numeric target is never resent.
        raw, _finished = self._invoke_compact_read(
            "ANGLE_SET batch readback",
            deadline_monotonic_s,
            lambda: hand.read(int(api.REG_ANGLE_SET), 12, retries=0),
            proxy,
        )
        packed = _exact_bytes(raw, 12, "ANGLE_SET batch readback")
        result = struct.unpack("<6h", packed)
        self._check_decode_deadline(
            self._now(), deadline_monotonic_s, "ANGLE_SET batch readback"
        )
        return result

    def _write_six_settings(
        self,
        register_name: str,
        values: Sequence[int],
        *,
        deadline_monotonic_s: float,
    ) -> None:
        api, hand, proxy = self._require_open()
        register = getattr(api, register_name, None)
        if register is None:
            raise RH56LinuxTransportError(f"RH56 API has no {register_name}")
        settings = _six_unsigned_settings(values, register_name)
        self._invoke(
            f"{register_name} batch write",
            deadline_monotonic_s,
            lambda: hand.write_six_shorts(int(register), settings, retries=0),
            proxy,
        )

    def _read_six_settings(
        self,
        register_name: str,
        *,
        deadline_monotonic_s: float,
    ) -> Tuple[int, ...]:
        api, hand, proxy = self._require_open()
        register = getattr(api, register_name, None)
        if register is None:
            raise RH56LinuxTransportError(f"RH56 API has no {register_name}")
        raw, _finished = self._invoke(
            f"{register_name} batch readback",
            deadline_monotonic_s,
            lambda: hand.read(int(register), 12, retries=0),
            proxy,
        )
        result = struct.unpack(
            "<6h", _exact_bytes(raw, 12, f"{register_name} batch readback")
        )
        settings = _six_unsigned_settings(result, register_name)
        self._check_decode_deadline(
            self._now(), deadline_monotonic_s, f"{register_name} batch readback"
        )
        return settings

    def write_speed_set(
        self, values: Sequence[int], *, deadline_monotonic_s: float
    ) -> None:
        self._write_six_settings(
            "REG_SPEED_SET", values, deadline_monotonic_s=deadline_monotonic_s
        )

    def read_speed_set(self, *, deadline_monotonic_s: float) -> Tuple[int, ...]:
        return self._read_six_settings(
            "REG_SPEED_SET", deadline_monotonic_s=deadline_monotonic_s
        )

    def write_force_set(
        self, values: Sequence[int], *, deadline_monotonic_s: float
    ) -> None:
        self._write_six_settings(
            "REG_FORCE_SET", values, deadline_monotonic_s=deadline_monotonic_s
        )

    def read_force_set(self, *, deadline_monotonic_s: float) -> Tuple[int, ...]:
        return self._read_six_settings(
            "REG_FORCE_SET", deadline_monotonic_s=deadline_monotonic_s
        )

    def read_safety_feedback(
        self, *, deadline_monotonic_s: float
    ) -> RH56SafetyFeedback:
        api, hand, proxy = self._require_open()
        position_angle_raw, _first_finished = self._invoke_compact_read(
            "POS_ACT/ANGLE_ACT compact read",
            deadline_monotonic_s,
            lambda: hand.read(int(api.REG_POS_ACT), 24, retries=0),
            proxy,
        )
        position_angle_packed = _exact_bytes(
            position_angle_raw, 24, "POS_ACT/ANGLE_ACT compact read"
        )
        position_angle = struct.unpack("<12h", position_angle_packed)

        safety_raw, last_response_monotonic_s = self._invoke_compact_read(
            "FORCE/CURRENT/ERROR/STATUS/TEMP compact read",
            deadline_monotonic_s,
            lambda: hand.read(int(api.REG_FORCE_ACT), 42, retries=0),
            proxy,
        )
        safety = _exact_bytes(
            safety_raw,
            42,
            "FORCE/CURRENT/ERROR/STATUS/TEMP compact read",
        )
        feedback = RH56SafetyFeedback(
            captured_monotonic_s=last_response_monotonic_s,
            positions=position_angle[:6],
            angles=position_angle[6:],
            forces_g=struct.unpack("<6h", safety[:12]),
            currents_ma=struct.unpack("<6h", safety[12:24]),
            errors=tuple(safety[24:30]),
            statuses=tuple(safety[30:36]),
            temperatures_c=tuple(safety[36:42]),
        )
        self._check_decode_deadline(
            self._now(), deadline_monotonic_s, "compact safety feedback"
        )
        return feedback

    def close(self) -> None:
        """Close the exclusive LinuxSerial context without writing hardware."""

        if self._closed:
            return
        context = self._serial_context
        proxy = self._serial_proxy
        self._hand = None
        self._serial_proxy = None
        self._serial_context = None
        self._api = None
        self._closed = True
        if proxy is not None:
            proxy.clear_deadline()
        if context is None:
            return
        try:
            context.__exit__(None, None, None)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            detail = str(exc).strip() or type(exc).__name__
            raise RH56LinuxTransportError(
                f"closing RH56 serial transport failed: {detail}"
            ) from exc

    def __enter__(self) -> "LinuxRH56TransactionalTransport":
        return self.open()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


# Make the structural relationship explicit for static reviewers without
# importing, constructing, or touching any RH56 hardware.
_transport_protocol_typecheck: RH56TransactionalTransport


__all__ = [
    "DEFAULT_RH56_API_MODULE",
    "LinuxRH56TransactionalTransport",
    "RH56_COMPACT_READ_ATTEMPTS",
    "RH56_COMPACT_READ_TOTAL_TIMEOUT_S",
    "RH56_LINUX_TRANSPORT_NAME",
    "RH56LinuxDeadlineExceeded",
    "RH56LinuxTransportError",
]
