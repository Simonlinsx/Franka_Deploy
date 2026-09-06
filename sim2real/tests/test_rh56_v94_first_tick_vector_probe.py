from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from sim2real.commissioning.rh56_v94_first_tick_vector_probe import (
    EXACT_FIRST_TICK_VECTOR,
    MASKED_PHASE_PREFIX,
    MIDDLE_COMMISSION_PATH,
    MIDDLE_MANUFACTURER_AXIS,
    THUMB_BEND_COMMISSION_PATH,
    THUMB_BEND_MANUFACTURER_AXIS,
    VECTOR_PHASE,
    V94FirstTickVectorProbeDriver,
    build_parser,
    derive_first_tick_contract,
    main,
    run_hardware_session,
)
from sim2real.commissioning.rh56_v94_microprobe import (
    DISABLED_TARGETS,
    OPEN_TARGETS,
    load_v94_profile,
)

from dexgrasp.tests.test_inspire_sequence_driver import (  # noqa: E402
    FakeAPI,
    FakeClock,
    FakeHand,
)


WORKSPACE = Path(__file__).resolve().parents[2]
PROFILE = WORKSPACE / "dexgrasp/configs/fr3_rh56_v94_commissioning.json"


def _snapshot():
    return {
        "angle_targets": DISABLED_TARGETS,
        "angles": OPEN_TARGETS,
        "positions": (0,) * 6,
        "forces": (0,) * 6,
        "currents": (0,) * 6,
        "errors": (0,) * 6,
        "statuses": (2,) * 6,
        "temperatures": (25,) * 6,
        "contact_axes": (),
    }


def _telemetry(phase, targets, angles, currents=None, statuses=None):
    return {
        "phase": phase,
        "elapsed_s": 0.1,
        "angle_targets": tuple(targets),
        "angles": tuple(angles),
        "positions": (0,) * 6,
        "forces": (0,) * 6,
        "currents": (0,) * 6 if currents is None else tuple(currents),
        "errors": (0,) * 6,
        "statuses": (2,) * 6 if statuses is None else tuple(statuses),
        "temperatures": (25,) * 6,
    }


def _recovery_rows():
    rows = []
    for axis in range(5):
        targets = [-1] * 6
        targets[axis] = 1000
        for _ in range(3):
            rows.append(
                _telemetry(
                    f"reset_open_bend_m{axis}_1000", targets, OPEN_TARGETS
                )
            )
    final = _telemetry("reset_open_final", DISABLED_TARGETS, OPEN_TARGETS)
    rows.append(final)
    return rows


class _FakeRobot:
    def __init__(self, q):
        self.q = tuple(q)

    def read_once(self):
        return SimpleNamespace(q=self.q, dq=(0.0,) * 7, robot_mode="Idle")


def _validator(_state, *, require_idle, enforce_success):
    assert require_idle is True
    assert enforce_success is False


class _FakeSessionDriver:
    def __init__(self, *, interrupt=False):
        self.calls = []
        self.telemetry = []
        self.external_gate = None
        self.interrupt = interrupt

    def request_stop(self):
        self.calls.append(("request_stop",))

    def adopt_disabled_state_and_verify(self):
        self.calls.append(("adopt",))

    def read_state_snapshot(self):
        self.calls.append(("snapshot",))
        if self.external_gate:
            self.external_gate()
        return _snapshot()

    def install_external_safety_check(self, callback):
        self.external_gate = callback
        self.calls.append(("install_gate",))

    def commission_masked_axis_path(self, axis, path):
        axis = int(axis)
        path = tuple(path)
        self.calls.append(("masked", axis, path))
        self.external_gate()
        if self.interrupt:
            raise KeyboardInterrupt
        for waypoint in path:
            targets = [-1] * 6
            targets[axis] = waypoint
            angles = list(OPEN_TARGETS)
            angles[axis] = waypoint
            phase = f"{MASKED_PHASE_PREFIX}_m{axis}_{waypoint:04d}"
            self.telemetry.extend(
                _telemetry(phase, targets, angles) for _ in range(3)
            )
        return path

    def commission_exact_first_tick_vector(self):
        self.calls.append(("vector", EXACT_FIRST_TICK_VECTOR))
        self.external_gate()
        self.telemetry.extend(
            _telemetry(VECTOR_PHASE, EXACT_FIRST_TICK_VECTOR, EXACT_FIRST_TICK_VECTOR)
            for _ in range(3)
        )
        return EXACT_FIRST_TICK_VECTOR

    def reset_to_open(self, **kwargs):
        self.calls.append(("reset", kwargs))
        self.external_gate()
        self.telemetry.extend(_recovery_rows())
        return (920, 1000)

    def disable_and_verify(self):
        self.calls.append(("disable",))

    def close(self):
        self.calls.append(("close",))


def _make_register_driver():
    clock = FakeClock()
    hand = FakeHand()
    driver = V94FirstTickVectorProbeDriver(
        hand,
        FakeAPI,
        thumb_rotate_range=(900, 1000),
        motion_timeout_s=5.0,
        poll_interval_s=1.0,
        stop_verify_samples=3,
        stop_verify_interval_s=0.1,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    driver.adopt_disabled_state_and_verify()
    return driver, hand


def test_shadow_contract_is_exact_and_hash_bound():
    contract = derive_first_tick_contract()
    assert contract["exact_first_tick_vector_manufacturer_order"] == list(
        EXACT_FIRST_TICK_VECTOR
    )
    assert contract["delta_from_first_angle_act_units"] == [0, -8, -53, 0, -83, 2]
    assert contract["maximum_abs_delta_from_first_angle_act_units"] == 83
    assert contract["raw_executed_linf"] == 0.0
    assert len(contract["shadow21_sha256"]) == len(contract["bundle_sha256"]) == 64


def test_default_cli_is_dry_run_and_has_no_bypass(capsys):
    calls = []
    assert (
        main(
            ["--config", str(PROFILE)],
            run_id_factory=lambda: "dry-first-tick",
            hardware_session=lambda *args, **kwargs: calls.append((args, kwargs)),
        )
        == 0
    )
    assert calls == []
    assert "no device was opened" in capsys.readouterr().out
    destinations = {action.dest for action in build_parser()._actions}
    assert not {"force", "bypass", "yes", "run_id"} & destinations


def test_execute_requires_tty_and_exact_phrase(tmp_path, capsys):
    args = [
        "--config",
        str(PROFILE),
        "--execute",
        "--output",
        str(tmp_path / "evidence.json"),
    ]
    calls = []
    assert main(
        args,
        stdin_isatty=lambda: False,
        run_id_factory=lambda: "no-tty",
        hardware_session=lambda *a, **k: calls.append((a, k)),
    ) == 1
    assert "interactive TTY" in capsys.readouterr().err
    assert main(
        args,
        stdin_isatty=lambda: True,
        input_fn=lambda _prompt: "wrong",
        run_id_factory=lambda: "wrong-phrase",
        hardware_session=lambda *a, **k: calls.append((a, k)),
    ) == 1
    assert calls == []


def test_register_driver_writes_only_masked_middle_path():
    driver, hand = _make_register_driver()
    hand.queue(angles=[1000] * 6, statuses=[2] * 6)
    for waypoint in MIDDLE_COMMISSION_PATH:
        angles = [1000] * 6
        angles[MIDDLE_MANUFACTURER_AXIS] = waypoint
        for _ in range(3):
            hand.queue(angles=angles, statuses=[2] * 6)
    before = len(hand.log)
    returned = driver.commission_masked_axis_path(
        MIDDLE_MANUFACTURER_AXIS, MIDDLE_COMMISSION_PATH
    )
    assert returned == MIDDLE_COMMISSION_PATH
    numeric = [
        item[2]
        for item in hand.log[before:]
        if item[:2] == ("write", FakeAPI.REG_ANGLE_SET)
        and item[2] != DISABLED_TARGETS
    ]
    assert numeric == [
        tuple(value if index == 2 else -1 for index in range(6))
        for value in MIDDLE_COMMISSION_PATH
    ]
    driver.disable_and_verify()


def test_register_driver_exact_vector_is_one_batch_write():
    driver, hand = _make_register_driver()
    hand.queue(angles=[1000] * 6, statuses=[2] * 6)
    for _ in range(3):
        hand.queue(angles=EXACT_FIRST_TICK_VECTOR, statuses=[2] * 6)
    before = len(hand.log)
    assert driver.commission_exact_first_tick_vector() == EXACT_FIRST_TICK_VECTOR
    numeric = [
        item[2]
        for item in hand.log[before:]
        if item[:2] == ("write", FakeAPI.REG_ANGLE_SET)
        and item[2] != DISABLED_TARGETS
    ]
    assert numeric == [EXACT_FIRST_TICK_VECTOR]
    driver.disable_and_verify()


def test_partial_readback_failure_latches_double_disable():
    driver, hand = _make_register_driver()
    hand.queue(angles=[1000] * 6, statuses=[2] * 6)
    calls = 0

    def readback():
        nonlocal calls
        calls += 1
        if calls == 2:
            return (-1, -1, 969, -1, -1, -1)
        return tuple(hand.angle_targets)

    hand.readback_override[FakeAPI.REG_ANGLE_SET] = readback
    before = len(hand.log)
    with pytest.raises(Exception, match="readback mismatch"):
        driver.commission_masked_axis_path(
            MIDDLE_MANUFACTURER_AXIS, MIDDLE_COMMISSION_PATH
        )
    disable_writes = [
        item
        for item in hand.log[before:]
        if item[:2] == ("write", FakeAPI.REG_ANGLE_SET)
        and item[2] == DISABLED_TARGETS
    ]
    assert len(disable_writes) >= 2
    assert tuple(hand.angle_targets) == DISABLED_TARGETS
    assert driver.stop_requested


def test_overcurrent_latches_disable_before_endpoint_acceptance():
    driver, hand = _make_register_driver()
    hand.queue(angles=[1000] * 6, statuses=[2] * 6)
    currents = [0] * 6
    currents[MIDDLE_MANUFACTURER_AXIS] = 101
    hand.queue(
        angles=[1000, 1000, 970, 1000, 1000, 1000],
        currents=currents,
        statuses=[2] * 6,
    )
    with pytest.raises(Exception, match="current"):
        driver.commission_masked_axis_path(
            MIDDLE_MANUFACTURER_AXIS, MIDDLE_COMMISSION_PATH
        )
    assert tuple(hand.angle_targets) == DISABLED_TARGETS
    assert driver.stop_requested


def test_full_fake_session_passes_k1_only_and_restores_open():
    binding = load_v94_profile(PROFILE)
    q = binding.payload["franka"]["default_q_rad"]
    driver = _FakeSessionDriver()
    evidence = run_hardware_session(
        binding,
        "fake-first-tick-pass",
        hand_connector=lambda _profile: driver,
        franka_connector=lambda _profile: (_FakeRobot(q), _validator),
        sleep=lambda _seconds: None,
    )
    assert evidence["result"] == "PASS"
    assert evidence["eligible_for_supervised_k1_hand_envelope"] is True
    assert evidence["authorizes_c2"] is False
    assert [item["name"] for item in evidence["stage_records"]] == [
        "middle_amplitude",
        "thumb_bend_amplitude",
        "exact_first_tick_vector",
    ]
    assert evidence["cleanup"]["double_disable_and_stationary_stop_verified"]
    assert driver.calls[-1] == ("close",)


def test_keyboard_interrupt_fails_and_still_cleans_up():
    binding = load_v94_profile(PROFILE)
    q = binding.payload["franka"]["default_q_rad"]
    driver = _FakeSessionDriver(interrupt=True)
    evidence = run_hardware_session(
        binding,
        "fake-first-tick-interrupt",
        hand_connector=lambda _profile: driver,
        franka_connector=lambda _profile: (_FakeRobot(q), _validator),
        sleep=lambda _seconds: None,
    )
    assert evidence["result"] == "FAIL"
    assert evidence["interrupted"] is True
    assert evidence["eligible_for_supervised_k1_hand_envelope"] is False
    assert ("disable",) in driver.calls
    assert driver.calls[-1] == ("close",)
