from __future__ import annotations

import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest

from robot_control.rh56.linux_transport import (
    LinuxRH56TransactionalTransport,
    RH56_COMPACT_READ_ATTEMPTS,
    RH56_COMPACT_READ_TOTAL_TIMEOUT_S,
    RH56_LINUX_TRANSPORT_NAME,
    RH56LinuxDeadlineExceeded,
    RH56LinuxTransportError,
)
from robot_control.rh56.actuator import (
    RH56SafetyFeedback,
    RH56TransactionalTransport,
)


class _Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += float(seconds)


def _default_payloads():
    targets = (-1, 0, 1, 500, 999, 1000)
    force_limits = (80,) * 6
    speeds = (40,) * 6
    positions = (-32768, -10, 0, 10, 1234, 32767)
    angles = (1000, 900, 800, 700, 600, 500)
    forces = (-300, -20, 0, 20, 300, 1200)
    currents = (-1400, -10, 0, 10, 200, 1400)
    errors = (0, 1, 2, 4, 8, 16)
    statuses = (0, 1, 2, 3, 5, 0xFF)
    temperatures = (20, 21, 22, 23, 24, 25)
    return SimpleNamespace(
        targets=targets,
        force_limits=force_limits,
        speeds=speeds,
        positions=positions,
        angles=angles,
        forces=forces,
        currents=currents,
        errors=errors,
        statuses=statuses,
        temperatures=temperatures,
        angle_bytes=struct.pack("<6h", *targets),
        force_limit_bytes=struct.pack("<6h", *force_limits),
        speed_bytes=struct.pack("<6h", *speeds),
        position_angle_bytes=struct.pack("<12h", *(positions + angles)),
        safety_bytes=(
            struct.pack("<6h", *forces)
            + struct.pack("<6h", *currents)
            + bytes(errors)
            + bytes(statuses)
            + bytes(temperatures)
        ),
    )


def _fake_api(
    clock: _Clock,
    *,
    exchange_durations=(),
    exchange_failures=(),
    payload_overrides=None,
    exchange_failure=None,
    hand_constructor_failure=None,
    close_failure=None,
    drift_register_map: bool = False,
    hand_return_delay_s: float = 0.0,
):
    payloads = _default_payloads()
    responses = {
        (1486, 12): payloads.angle_bytes,
        (1498, 12): payloads.force_limit_bytes,
        (1522, 12): payloads.speed_bytes,
        (1534, 24): payloads.position_angle_bytes,
        (1582, 42): payloads.safety_bytes,
    }
    responses.update(payload_overrides or {})
    durations = list(exchange_durations)
    failures = list(exchange_failures)
    events = []
    holder = SimpleNamespace(
        contexts=[],
        hands=[],
        events=events,
        discovery_calls=0,
        exchange_count=0,
        payloads=payloads,
    )

    class Serial:
        def __init__(self, port, baud, timeout, debug):
            self.port = port
            self.baud = baud
            self.timeout = timeout
            self.debug = debug
            self.exit_count = 0
            holder.contexts.append(self)
            events.append(("serial_init", port, baud, timeout, debug))

        def __enter__(self):
            events.append(("serial_enter", self.port))
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.exit_count += 1
            events.append(("serial_exit", exc_type))
            if close_failure is not None:
                raise close_failure

        def exchange(self, request):
            holder.exchange_count += 1
            started = clock()
            events.append(("exchange_start", request, self.timeout, started))
            duration = durations.pop(0) if durations else 0.0
            clock.advance(duration)
            events.append(("exchange_done", request, clock()))
            per_exchange_failure = failures.pop(0) if failures else None
            if per_exchange_failure is not None:
                raise per_exchange_failure
            if exchange_failure is not None:
                raise exchange_failure
            return b"fake-response"

        def discard_input(self):
            events.append(("discard_input", clock()))

    class Hand:
        def __init__(self, serial_port, hand_id):
            events.append(("hand_init", hand_id))
            if hand_constructor_failure is not None:
                raise hand_constructor_failure
            self.serial = serial_port
            self.hand_id = hand_id
            holder.hands.append(self)

        def write_six_shorts(self, address, values, retries):
            values = tuple(values)
            events.append(("write", address, values, retries))
            self.serial.exchange(("write", address, values))

        def read(self, address, length, retries):
            events.append(("read", address, length, retries))
            self.serial.exchange(("read", address, length))
            clock.advance(hand_return_delay_s)
            return responses[(address, length)]

    def find_serial_port():
        holder.discovery_calls += 1
        events.append(("discover",))
        return "/dev/serial/by-id/fake-rh56"

    api = SimpleNamespace(
        REG_ANGLE_SET=1486,
        REG_FORCE_SET=1498,
        REG_SPEED_SET=1522,
        REG_POS_ACT=1534,
        REG_ANGLE_ACT=1547 if drift_register_map else 1546,
        REG_FORCE_ACT=1582,
        REG_CURRENT=1594,
        REG_ERROR=1606,
        REG_STATUS=1612,
        REG_TEMP=1618,
        LinuxSerial=Serial,
        RH56Hand=Hand,
        find_serial_port=find_serial_port,
    )
    return api, holder


def _transport(clock, api, loads, *, port="/dev/fake-rh56"):
    def load(name):
        loads.append(name)
        return api

    return LinuxRH56TransactionalTransport(
        port,
        module_loader=load,
        monotonic=clock,
    )


def test_module_import_does_not_import_rh56_api_or_open_a_device():
    code = (
        "import sys; "
        "assert 'examples.inspire_rh56_test' not in sys.modules; "
        "import robot_control.rh56.linux_transport; "
        "assert 'examples.inspire_rh56_test' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_construction_is_lazy_and_context_exposes_protocol_metadata():
    clock = _Clock()
    api, holder = _fake_api(clock)
    loads = []
    transport = _transport(clock, api, loads)

    assert loads == []
    assert holder.events == []
    assert transport.hardware_backed is True
    assert transport.baud_rate == 115200
    assert transport.transport_name == RH56_LINUX_TRANSPORT_NAME
    assert isinstance(transport, RH56TransactionalTransport)
    assert transport.is_open is False

    with transport as opened:
        assert opened is transport
        assert transport.is_open is True
        assert loads == ["examples.inspire_rh56_test"]
        assert holder.events[:3] == [
            ("serial_init", "/dev/fake-rh56", 115200, 0.0, False),
            ("serial_enter", "/dev/fake-rh56"),
            ("hand_init", 1),
        ]

    assert transport.is_open is False
    assert holder.contexts[0].exit_count == 1
    transport.close()
    assert holder.contexts[0].exit_count == 1
    with pytest.raises(RH56LinuxTransportError, match="closed"):
        transport.open()


def test_auto_port_discovery_is_delayed_until_open():
    clock = _Clock()
    api, holder = _fake_api(clock)
    loads = []
    transport = _transport(clock, api, loads, port=None)
    assert holder.discovery_calls == 0
    transport.open()
    assert holder.discovery_calls == 1
    assert transport.resolved_port == "/dev/serial/by-id/fake-rh56"
    transport.close()


def test_exact_batch_io_compact_unpack_and_captured_after_response():
    clock = _Clock()
    api, holder = _fake_api(
        clock,
        exchange_durations=(0.010, 0.011, 0.012, 0.013),
        hand_return_delay_s=0.001,
    )
    transport = _transport(clock, api, []).open()
    deadline = 10.200

    transport.write_angle_set(
        (1000, 900, 800, 700, 600, 500),
        deadline_monotonic_s=deadline,
    )
    assert transport.read_angle_set(deadline_monotonic_s=deadline) == (
        holder.payloads.targets
    )
    feedback = transport.read_safety_feedback(deadline_monotonic_s=deadline)

    assert isinstance(feedback, RH56SafetyFeedback)
    assert feedback.positions == holder.payloads.positions
    assert feedback.angles == holder.payloads.angles
    assert feedback.forces_g == holder.payloads.forces
    assert feedback.currents_ma == holder.payloads.currents
    assert feedback.errors == holder.payloads.errors
    assert feedback.statuses == holder.payloads.statuses
    assert feedback.temperatures_c == holder.payloads.temperatures

    io_events = [event for event in holder.events if event[0] in ("write", "read")]
    assert io_events == [
        ("write", 1486, (1000, 900, 800, 700, 600, 500), 0),
        ("read", 1486, 12, 0),
        ("read", 1534, 24, 0),
        ("read", 1582, 42, 0),
    ]
    exchange_starts = [
        event for event in holder.events if event[0] == "exchange_start"
    ]
    assert len(exchange_starts) == 4
    # Writes retain the caller deadline.  Every side-effect-free read is sent
    # once with the full 50 ms per-register cap.
    assert [event[2] for event in exchange_starts] == pytest.approx(
        [0.200, 0.050, 0.050, 0.050]
    )
    last_response_finished = [
        event for event in holder.events if event[0] == "exchange_done"
    ][-1][2]
    assert feedback.captured_monotonic_s == pytest.approx(
        last_response_finished + 0.001
    )
    assert feedback.captured_monotonic_s <= clock()
    transport.close()


def test_deadline_bound_speed_and_force_batch_setters_have_exact_readback():
    clock = _Clock()
    api, holder = _fake_api(clock)
    transport = _transport(clock, api, []).open()

    assert transport.read_speed_set(deadline_monotonic_s=11.0) == (40,) * 6
    assert transport.read_force_set(deadline_monotonic_s=11.0) == (80,) * 6
    transport.write_speed_set((35,) * 6, deadline_monotonic_s=11.0)
    transport.write_force_set((75,) * 6, deadline_monotonic_s=11.0)

    io_events = [event for event in holder.events if event[0] in ("write", "read")]
    assert io_events == [
        ("read", 1522, 12, 0),
        ("read", 1498, 12, 0),
        ("write", 1522, (35,) * 6, 0),
        ("write", 1498, (75,) * 6, 0),
    ]


def test_expired_deadline_fails_before_any_register_call():
    clock = _Clock()
    api, holder = _fake_api(clock)
    transport = _transport(clock, api, []).open()
    with pytest.raises(RH56LinuxDeadlineExceeded, match="already expired"):
        transport.read_angle_set(deadline_monotonic_s=clock())
    assert not any(event[0] in ("read", "write") for event in holder.events)
    transport.close()


def test_exchange_overrun_is_terminal_and_never_retried():
    clock = _Clock()
    api, holder = _fake_api(clock, exchange_durations=(0.051,))
    transport = _transport(clock, api, []).open()
    with pytest.raises(RH56LinuxDeadlineExceeded, match="exceeded caller deadline"):
        transport.write_angle_set(
            (1, 2, 3, 4, 5, 6), deadline_monotonic_s=10.050
        )
    assert holder.exchange_count == 1
    assert [event for event in holder.events if event[0] == "write"] == [
        ("write", 1486, (1, 2, 3, 4, 5, 6), 0)
    ]
    transport.close()


def test_feedback_reuses_one_absolute_deadline_and_stops_after_total_overrun():
    clock = _Clock()
    api, holder = _fake_api(clock, exchange_durations=(0.051, 0.001))
    transport = _transport(clock, api, []).open()
    with pytest.raises(RH56LinuxDeadlineExceeded):
        transport.read_safety_feedback(deadline_monotonic_s=10.050)
    assert holder.exchange_count == 1
    assert [event for event in holder.events if event[0] == "read"] == [
        ("read", 1534, 24, 0)
    ]
    transport.close()


def test_feedback_single_read_gets_only_total_deadline_budget():
    clock = _Clock()
    api, holder = _fake_api(clock, exchange_durations=(0.051,))
    transport = _transport(clock, api, []).open()
    with pytest.raises(RH56LinuxDeadlineExceeded):
        transport.read_safety_feedback(deadline_monotonic_s=10.050)
    starts = [event for event in holder.events if event[0] == "exchange_start"]
    assert [event[2] for event in starts] == pytest.approx([0.050])
    assert holder.exchange_count == 1
    transport.close()


def test_compact_feedback_timeout_is_not_retransmitted():
    clock = _Clock()
    api, holder = _fake_api(
        clock,
        exchange_durations=(0.020,),
        exchange_failures=(TimeoutError("synthetic read timeout"),),
    )
    transport = _transport(clock, api, []).open()

    with pytest.raises(RH56LinuxTransportError, match="synthetic read timeout"):
        transport.read_safety_feedback(deadline_monotonic_s=10.075)
    assert [event for event in holder.events if event[0] == "read"] == [
        ("read", 1534, 24, 0),
    ]
    assert holder.exchange_count == RH56_COMPACT_READ_ATTEMPTS == 1
    assert not any(event[0] == "discard_input" for event in holder.events)
    assert not any(event[0] == "write" for event in holder.events)
    transport.close()


def test_compact_feedback_persistent_timeout_is_bounded_to_one_read_and_no_write():
    clock = _Clock()
    api, holder = _fake_api(
        clock,
        exchange_durations=(0.050,),
        exchange_failures=(TimeoutError("synthetic read timeout"),),
    )
    transport = _transport(clock, api, []).open()

    with pytest.raises(
        RH56LinuxTransportError,
        match=r"POS_ACT/ANGLE_ACT compact read single read.*synthetic read timeout",
    ):
        # A multi-second caller/stop deadline must not be inherited by one
        # compact register read.
        transport.read_safety_feedback(deadline_monotonic_s=15.0)

    assert clock() == pytest.approx(
        10.0 + RH56_COMPACT_READ_TOTAL_TIMEOUT_S
    )
    assert [event for event in holder.events if event[0] == "read"] == [
        ("read", 1534, 24, 0),
    ]
    assert not any(event[0] == "discard_input" for event in holder.events)
    assert not any(event[0] == "write" for event in holder.events)
    transport.close()


def test_underlying_serial_error_is_not_retried():
    clock = _Clock()
    api, holder = _fake_api(
        clock,
        exchange_failure=OSError("synthetic USB disconnect"),
    )
    transport = _transport(clock, api, []).open()
    with pytest.raises(
        RH56LinuxTransportError, match="synthetic USB disconnect"
    ) as caught:
        transport.read_angle_set(deadline_monotonic_s=11.0)
    assert isinstance(caught.value.__cause__, OSError)
    assert holder.exchange_count == 1
    assert [event for event in holder.events if event[0] == "read"] == [
        ("read", 1486, 12, 0),
    ]
    assert not any(event[0] == "discard_input" for event in holder.events)
    transport.close()


def test_angle_set_exact_read_uses_one_request_with_full_cap():
    clock = _Clock()
    api, holder = _fake_api(
        clock,
        exchange_durations=(0.002,),
    )
    transport = _transport(clock, api, []).open()

    assert transport.read_angle_set(deadline_monotonic_s=10.050) == (
        holder.payloads.targets
    )

    assert [event for event in holder.events if event[0] == "read"] == [
        ("read", 1486, 12, 0),
    ]
    starts = [event for event in holder.events if event[0] == "exchange_start"]
    assert [event[2] for event in starts] == pytest.approx([0.050])
    assert not any(event[0] == "discard_input" for event in holder.events)
    assert not any(event[0] == "write" for event in holder.events)
    transport.close()


def test_one_numeric_write_then_failed_exact_read_never_resends_either_request():
    clock = _Clock()
    api, holder = _fake_api(
        clock,
        exchange_durations=(
            0.001,
            0.050,
        ),
        exchange_failures=(
            None,
            TimeoutError("synthetic exact-read timeout"),
        ),
    )
    transport = _transport(clock, api, []).open()

    transport.write_angle_set(
        holder.payloads.targets,
        deadline_monotonic_s=10.008,
    )
    with pytest.raises(
        RH56LinuxTransportError, match="synthetic exact-read timeout"
    ):
        transport.read_angle_set(deadline_monotonic_s=10.051)

    assert [event for event in holder.events if event[0] == "write"] == [
        ("write", 1486, holder.payloads.targets, 0)
    ]
    assert [event for event in holder.events if event[0] == "read"] == [
        ("read", 1486, 12, 0),
    ]
    assert not any(event[0] == "discard_input" for event in holder.events)
    transport.close()


@pytest.mark.parametrize(
    ("override", "message", "expected_reads"),
    [
        ({(1534, 24): b"short"}, "expected 24", [("read", 1534, 24, 0)]),
        (
            {(1582, 42): b"short"},
            "expected 42",
            [("read", 1534, 24, 0), ("read", 1582, 42, 0)],
        ),
    ],
)
def test_malformed_compact_blocks_fail_at_the_exact_boundary(
    override, message, expected_reads
):
    clock = _Clock()
    api, holder = _fake_api(clock, payload_overrides=override)
    transport = _transport(clock, api, []).open()
    with pytest.raises(RH56LinuxTransportError, match=message):
        transport.read_safety_feedback(deadline_monotonic_s=11.0)
    assert [event for event in holder.events if event[0] == "read"] == expected_reads
    transport.close()


def test_register_map_drift_is_rejected_before_serial_construction():
    clock = _Clock()
    api, holder = _fake_api(clock, drift_register_map=True)
    transport = _transport(clock, api, [])
    with pytest.raises(RH56LinuxTransportError, match="not contiguous"):
        transport.open()
    assert holder.contexts == []
    with pytest.raises(RH56LinuxTransportError, match="already failed"):
        transport.open()


def test_hand_constructor_failure_closes_entered_serial_context():
    clock = _Clock()
    api, holder = _fake_api(
        clock,
        hand_constructor_failure=RuntimeError("synthetic hand failure"),
    )
    transport = _transport(clock, api, [])
    with pytest.raises(RH56LinuxTransportError, match="synthetic hand failure"):
        transport.open()
    assert holder.contexts[0].exit_count == 1


def test_close_failure_still_latches_closed_and_is_not_retried():
    clock = _Clock()
    api, holder = _fake_api(
        clock,
        close_failure=OSError("synthetic close failure"),
    )
    transport = _transport(clock, api, []).open()
    with pytest.raises(RH56LinuxTransportError, match="synthetic close failure"):
        transport.close()
    assert transport.is_open is False
    assert holder.contexts[0].exit_count == 1
    transport.close()
    assert holder.contexts[0].exit_count == 1


@pytest.mark.parametrize(
    "values",
    [
        (1, 2, 3, 4, 5),
        (1, 2, 3, 4, 5, 1001),
        (1, 2, 3, 4, 5, True),
    ],
)
def test_invalid_angle_targets_are_rejected_without_io(values):
    clock = _Clock()
    api, holder = _fake_api(clock)
    transport = _transport(clock, api, []).open()
    with pytest.raises(ValueError, match="ANGLE_SET"):
        transport.write_angle_set(values, deadline_monotonic_s=11.0)
    assert holder.exchange_count == 0
    transport.close()


def test_operations_require_an_explicit_open_context_and_fail_after_close():
    clock = _Clock()
    api, holder = _fake_api(clock)
    transport = _transport(clock, api, [])
    with pytest.raises(RH56LinuxTransportError, match="not open"):
        transport.read_angle_set(deadline_monotonic_s=11.0)
    assert holder.events == []
    transport.open()
    transport.close()
    with pytest.raises(RH56LinuxTransportError, match="closed"):
        transport.read_safety_feedback(deadline_monotonic_s=11.0)
