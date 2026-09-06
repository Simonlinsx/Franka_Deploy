from __future__ import annotations

import copy

import pytest

from examples import inspire_rh56_test as rh56


def _snapshot() -> dict[str, object]:
    return {
        "angle_targets": (-1, -1, -1, -1, -1, -1),
        "angles": (327, 705, 853, 857, 988, 267),
        "positions": (1281, 648, 382, 364, 71, 1467),
        "forces": (-37, -22, -2, 52, -3, -56),
        "currents": (0, 0, 0, 0, 0, 0),
        "temperatures": (38, 38, 38, 38, 36, 38),
        "errors": (0, 0, 0, 0, 0, 0),
        "statuses": (2, 2, 2, 2, 2, 7),
        "speeds": (600, 600, 600, 600, 600, 600),
        "force_limits": (80, 80, 80, 80, 80, 80),
        "current_limits": (1400, 1400, 1400, 1400, 1400, 1400),
    }


class _Hand:
    def __init__(self, final_snapshot: dict[str, object]) -> None:
        self.final_snapshot = final_snapshot
        self.writes: list[tuple[int, bytes, int]] = []

    def write(self, address: int, data: bytes, retries: int = 1) -> None:
        self.writes.append((address, data, retries))

    def snapshot(self) -> dict[str, object]:
        return copy.deepcopy(self.final_snapshot)


def test_clear_latched_error_matches_official_register_write_and_stays_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = _snapshot()
    final = copy.deepcopy(initial)
    final["statuses"] = (2, 2, 2, 2, 2, 2)
    hand = _Hand(final)
    monkeypatch.setattr(rh56.time, "sleep", lambda _: None)

    observed = rh56.clear_latched_error(hand, initial)

    assert hand.writes == [(rh56.REG_CLEAR_ERROR, b"\x01", 0)]
    assert observed == final
    assert rh56.build_write_frame(1, rh56.REG_CLEAR_ERROR, b"\x01") == bytes.fromhex(
        "EB 90 01 04 12 EC 03 01 07"
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("angle_targets", (-1, -1, -1, -1, -1, 267), "ANGLE_SET"),
        ("currents", (0, 0, 0, 0, 0, 101), "idle current"),
        ("temperatures", (38, 38, 38, 38, 36, 60), "temperature"),
        ("errors", (0, 0, 0, 0, 0, 1), "ERROR bits"),
        ("statuses", (2, 2, 2, 2, 2, 1), "moving"),
        ("statuses", (2, 2, 2, 2, 2, 2), "no latched"),
    ],
)
def test_clear_latched_error_rejects_unsafe_preconditions_without_writing(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: tuple[int, ...],
    match: str,
) -> None:
    initial = _snapshot()
    initial[field] = value
    hand = _Hand(initial)
    monkeypatch.setattr(rh56.time, "sleep", lambda _: None)

    with pytest.raises(rh56.RH56Error, match=match):
        rh56.clear_latched_error(hand, initial)

    assert hand.writes == []


def test_clear_latched_error_fails_closed_if_status_does_not_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = _snapshot()
    hand = _Hand(initial)
    monkeypatch.setattr(rh56.time, "sleep", lambda _: None)

    with pytest.raises(rh56.RH56Error, match="STATUS remains non-idle"):
        rh56.clear_latched_error(hand, initial)

    assert hand.writes == [(rh56.REG_CLEAR_ERROR, b"\x01", 0)]


def test_clear_latched_error_rejects_unexpected_post_write_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = _snapshot()
    final = copy.deepcopy(initial)
    final["statuses"] = (2, 2, 2, 2, 2, 2)
    final["angles"] = (327, 705, 853, 857, 988, 271)
    hand = _Hand(final)
    monkeypatch.setattr(rh56.time, "sleep", lambda _: None)

    with pytest.raises(rh56.RH56Error, match="ANGLE_ACT movement"):
        rh56.clear_latched_error(hand, initial)


class _StaleQ6Hand:
    def __init__(self, *, clear_status_on_hold: bool = True) -> None:
        self.state = _snapshot()
        self.state["angles"] = (327, 705, 853, 857, 987, 585)
        self.state["positions"] = (1280, 648, 382, 361, 73, 995)
        self.clear_status_on_hold = clear_status_on_hold
        self.short_registers = {
            rh56.REG_SPEED_SET + 10: 600,
            rh56.REG_FORCE_SET + 10: 80,
            rh56.REG_CURRENT_LIMIT + 10: 1400,
        }
        self.writes: list[tuple[object, ...]] = []

    def snapshot(self) -> dict[str, object]:
        output = copy.deepcopy(self.state)
        output["speeds"] = (600, 600, 600, 600, 600, self.short_registers[rh56.REG_SPEED_SET + 10])
        output["force_limits"] = (80, 80, 80, 80, 80, self.short_registers[rh56.REG_FORCE_SET + 10])
        output["current_limits"] = (1400, 1400, 1400, 1400, 1400, self.short_registers[rh56.REG_CURRENT_LIMIT + 10])
        return output

    def write(self, address: int, data: bytes, retries: int = 1) -> None:
        self.writes.append(("write", address, data, retries))

    def write_short(self, address: int, value: int) -> None:
        self.short_registers[address] = value
        self.writes.append(("short", address, value))

    def read_short(self, address: int) -> int:
        return self.short_registers[address]

    def write_six_shorts(
        self, address: int, values: tuple[int, ...], retries: int = 1
    ) -> None:
        values = tuple(values)
        self.writes.append(("six", address, values, retries))
        self.state["angle_targets"] = values
        if values[-1] != -1 and self.clear_status_on_hold:
            self.state["statuses"] = (2, 2, 2, 2, 2, 2)

    def read_six_shorts(self, address: int, retries: int = 2) -> tuple[int, ...]:
        assert address == rh56.REG_ANGLE_SET
        return tuple(self.state["angle_targets"])


def test_stale_q6_recovery_relatches_only_feedback_equivalent_hold_and_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hand = _StaleQ6Hand()
    initial = hand.snapshot()
    monkeypatch.setattr(rh56.time, "sleep", lambda _: None)

    final = rh56.recover_stale_thumb_rotate_status(hand, initial)

    hold = (-1, -1, -1, -1, -1, 600)
    assert ("six", rh56.REG_ANGLE_SET, hold, 1) in hand.writes
    assert tuple(final["angle_targets"]) == (-1,) * 6
    assert tuple(final["statuses"]) == (2,) * 6
    assert hand.short_registers[rh56.REG_SPEED_SET + 10] == 600
    assert hand.short_registers[rh56.REG_FORCE_SET + 10] == 80
    assert hand.short_registers[rh56.REG_CURRENT_LIMIT + 10] == 1400


def test_stale_q6_recovery_fails_closed_when_status_does_not_relatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hand = _StaleQ6Hand(clear_status_on_hold=False)
    initial = hand.snapshot()
    monkeypatch.setattr(rh56.time, "sleep", lambda _: None)
    monkeypatch.setattr(rh56, "STALE_Q6_RECOVERY_TIMEOUT_S", 0.0)

    with pytest.raises(rh56.RH56Error, match="timed out"):
        rh56.recover_stale_thumb_rotate_status(hand, initial)

    assert tuple(hand.state["angle_targets"]) == (-1,) * 6
    assert hand.short_registers[rh56.REG_SPEED_SET + 10] == 600
    assert hand.short_registers[rh56.REG_CURRENT_LIMIT + 10] == 1400


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("statuses", (2, 2, 2, 2, 2, 2), "only thumb_rotate STATUS=7"),
        ("angles", (327, 705, 853, 799, 987, 585), "index clearance"),
        ("angles", (327, 705, 853, 857, 949, 585), "thumb-bend clearance"),
        ("forces", (-37, -22, -2, 201, -3, -56), "contact-free"),
    ],
)
def test_stale_q6_recovery_rejects_unreviewed_seed_without_writing(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: tuple[int, ...],
    match: str,
) -> None:
    hand = _StaleQ6Hand()
    initial = hand.snapshot()
    initial[field] = value
    monkeypatch.setattr(rh56.time, "sleep", lambda _: None)

    with pytest.raises(rh56.RH56Error, match=match):
        rh56.recover_stale_thumb_rotate_status(hand, initial)

    assert hand.writes == []
