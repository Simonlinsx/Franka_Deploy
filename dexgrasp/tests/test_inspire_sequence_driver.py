from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from anydex_pipeline.inspire_sequence_driver import (
    DISABLED_TARGETS,
    OPEN_TARGETS,
    RH56MotionStopped,
    RH56SequenceDriver,
    RH56SequenceDriverError,
    RH56StopUnconfirmed,
    RH56ValidatedFeedbackObservation,
    RH56ValidatedFeedbackObserverError,
)
from anydex_pipeline.rh56_hand_path import (
    build_rh56_no_contact_execution_path,
    rh56_feedback_envelope_policy,
)
from anydex_pipeline.rh56_interrupted_recovery import (
    InterruptedRecoveryDriver,
    InterruptedRecoveryPlan,
    RECOVERY_MODE_Q6_RETURN,
    RECOVERY_ROUTE_COUPLED_CLOSE_PREFIX,
    RECOVERY_ROUTE_NEAR_OPEN_RESET,
    RECOVERY_ROUTE_SEALED_Q6_RETURN,
)
from anydex_pipeline.rh56_reset_open import (
    DEFAULT_FORCES,
    DEFAULT_SPEEDS,
    RH56ResetOpenDriver,
    RH56ResetSettingsRestoreError,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.now += float(duration)


class FakeAPI:
    REG_CURRENT_LIMIT = 1020
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

    @staticmethod
    def open_hand(
        hand,
        speed,
        force_limit,
        motion_timeout,
        include_thumb_rotate=False,
        simultaneous_bend_open=False,
        safety_check=None,
        max_axis_current_ma=None,
        endpoint_stable_samples=None,
        max_inactive_drift_units=None,
        feedback_callback=None,
    ):
        if safety_check is not None:
            safety_check()
        logged = (
                "api.open_hand",
                speed,
                force_limit,
                motion_timeout,
                include_thumb_rotate,
            )
        if max_axis_current_ma is not None:
            logged = logged + (int(max_axis_current_ma),)
        if simultaneous_bend_open:
            logged = logged + ("simultaneous_bends",)
        hand.log.append(logged)
        if hand.fail_open:
            raise RuntimeError("open helper failed")
        if include_thumb_rotate and hand.angles[5] < 885:
            raise RuntimeError("reviewed q6-open helper refuses starts below 885")
        hand.angle_targets = list(DISABLED_TARGETS)
        if include_thumb_rotate:
            hand.angles = [1000] * 6
        else:
            hand.angles = [1000] * 5 + [hand.angles[5]]
        hand.positions = [100] * 6
        hand.statuses = [2] * 6
        hand.errors = [0] * 6
        hand.temperatures = [30] * 6
        if feedback_callback is not None:
            stable = 1 if endpoint_stable_samples is None else int(
                endpoint_stable_samples
            )
            if simultaneous_bend_open:
                for _ in range(stable):
                    feedback_callback(
                        {
                            "phase": "reset_open_bends_all_1000",
                            "angle_targets": (1000,) * 5 + (-1,),
                            "angles": tuple(hand.angles),
                            "positions": tuple(hand.positions),
                            "currents": tuple(hand.currents),
                            "errors": tuple(hand.errors),
                            "statuses": tuple(hand.statuses),
                            "temperatures": tuple(hand.temperatures),
                        }
                    )
            else:
                for axis in range(5):
                    targets = [-1] * 6
                    targets[axis] = 1000
                    for _ in range(stable):
                        feedback_callback(
                            {
                                "phase": f"reset_open_bend_m{axis}_1000",
                                "angle_targets": tuple(targets),
                                "angles": tuple(hand.angles),
                                "positions": tuple(hand.positions),
                                "currents": tuple(hand.currents),
                                "errors": tuple(hand.errors),
                                "statuses": tuple(hand.statuses),
                                "temperatures": tuple(hand.temperatures),
                            }
                        )
        if safety_check is not None:
            safety_check()


class FakeHand:
    def __init__(self):
        self.log = []
        self.angle_targets = list(DISABLED_TARGETS)
        self.angles = [1000] * 6
        self.positions = [100] * 6
        self.forces = [0] * 6
        self.currents = [0] * 6
        self.current_limits = [1400] * 6
        self.errors = [0] * 6
        self.statuses = [2] * 6
        self.temperatures = [30] * 6
        self.speeds = [1000] * 6
        self.force_limits = [500] * 6
        self.feedback_frames = []
        self.fail_open = False
        self.readback_override = {}
        self.write_failures = {}

    def queue(
        self,
        *,
        angles=None,
        currents=None,
        statuses=None,
        errors=None,
        temperatures=None,
    ):
        self.feedback_frames.append(
            dict(
                angles=list(self.angles if angles is None else angles),
                currents=list(self.currents if currents is None else currents),
                statuses=list(self.statuses if statuses is None else statuses),
                errors=list(self.errors if errors is None else errors),
                temperatures=list(
                    self.temperatures if temperatures is None else temperatures
                ),
            )
        )

    def _advance_feedback(self):
        if not self.feedback_frames:
            return
        frame = self.feedback_frames.pop(0)
        self.angles = frame["angles"]
        self.currents = frame["currents"]
        self.statuses = frame["statuses"]
        self.errors = frame["errors"]
        self.temperatures = frame["temperatures"]

    def write_six_shorts(self, address, values, retries=1):
        values = tuple(int(value) for value in values)
        self.log.append(("write", address, values, retries))
        remaining = self.write_failures.get(address, 0)
        if remaining:
            self.write_failures[address] = remaining - 1
            raise RuntimeError(f"write {address} failed")
        if address == FakeAPI.REG_ANGLE_SET:
            self.angle_targets = list(values)
        elif address == FakeAPI.REG_SPEED_SET:
            self.speeds = list(values)
        elif address == FakeAPI.REG_FORCE_SET:
            self.force_limits = list(values)

    def read_six_shorts(self, address, retries=0):
        self.log.append(("read6", address, retries))
        if address == FakeAPI.REG_ANGLE_ACT:
            self._advance_feedback()
        if address in self.readback_override:
            override = self.readback_override[address]
            return tuple(override() if callable(override) else override)
        values = {
            FakeAPI.REG_ANGLE_SET: self.angle_targets,
            FakeAPI.REG_SPEED_SET: self.speeds,
            FakeAPI.REG_FORCE_SET: self.force_limits,
            FakeAPI.REG_ANGLE_ACT: self.angles,
            FakeAPI.REG_POS_ACT: self.positions,
            FakeAPI.REG_FORCE_ACT: self.forces,
            FakeAPI.REG_CURRENT: self.currents,
            FakeAPI.REG_CURRENT_LIMIT: self.current_limits,
        }[address]
        return tuple(values)

    def read(self, address, length, retries=0):
        self.log.append(("read", address, length, retries))
        values = {
            FakeAPI.REG_ERROR: self.errors,
            FakeAPI.REG_STATUS: self.statuses,
            FakeAPI.REG_TEMP: self.temperatures,
        }[address]
        return bytes(values)


def make_driver(
    hand=None,
    *,
    thumb_range=(900, 1000),
    open_min_angle=980,
    timeout=5.0,
    external_safety_check=None,
):
    clock = FakeClock()
    active_hand = FakeHand() if hand is None else hand
    driver = RH56SequenceDriver(
        active_hand,
        FakeAPI,
        thumb_rotate_range=thumb_range,
        open_min_angle=open_min_angle,
        motion_timeout_s=timeout,
        poll_interval_s=1.0,
        stop_verify_samples=3,
        stop_verify_interval_s=0.1,
        external_safety_check=external_safety_check,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    return driver, active_hand, clock


def make_reset_driver(
    hand=None,
    *,
    thumb_range=(0, 1000),
    q6_open_min_angle=None,
    q6_feedback_recovery_min_angle=None,
    angle_tolerance=20,
    external_safety_check=None,
):
    clock = FakeClock()
    active_hand = FakeHand() if hand is None else hand
    driver = RH56ResetOpenDriver(
        active_hand,
        FakeAPI,
        thumb_rotate_range=thumb_range,
        q6_open_min_angle=q6_open_min_angle,
        q6_feedback_recovery_min_angle=q6_feedback_recovery_min_angle,
        angle_tolerance=angle_tolerance,
        motion_timeout_s=5.0,
        poll_interval_s=1.0,
        stop_verify_samples=3,
        stop_verify_interval_s=0.1,
        external_safety_check=external_safety_check,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    return driver, active_hand, clock


def open_driver(driver):
    driver.open_and_verify(OPEN_TARGETS)


def preshape(driver, hand, target=900):
    hand.queue(angles=[1000] * 6, statuses=[2] * 6)
    step = driver.thumb_preshape_step_units
    waypoints = list(range(1000 - step, int(target), -step))
    if not waypoints or waypoints[-1] != int(target):
        waypoints.append(int(target))
    for waypoint in waypoints:
        for _ in range(2):
            hand.queue(
                angles=[1000, 1000, 1000, 1000, 1000, waypoint],
                statuses=[2] * 6,
            )
    driver.preshape_thumb(target)


def bind_audited_path(driver, target):
    driver.bind_audited_no_contact_execution_path(
        build_rh56_no_contact_execution_path(target).as_dict(),
        feedback_envelope_policy=rh56_feedback_envelope_policy(
            driver.angle_tolerance
        ),
    )


def coupled_waypoints(bends, q6, step=25):
    current = [1000] * 5
    output = []
    for axis, target in enumerate(bends):
        while current[axis] != int(target):
            current[axis] = max(int(target), current[axis] - int(step))
            output.append(tuple(current) + (int(q6),))
    return output


def interrupted_recovery_plan():
    interrupted = (0, 1000, 1000, 1000, 1000, 646)
    path = build_rh56_no_contact_execution_path(interrupted, step_units=25)
    return InterruptedRecoveryPlan(
        failed_evidence_path=Path("/tmp/failed-commissioning.json"),
        failed_file_sha256="a" * 64,
        failed_payload_sha256="b" * 64,
        failed_run_id="interrupted-test",
        target_q6=646,
        step_units=25,
        coupled_targets=(0, 358, 799, 911, 922, 646),
        interrupted_targets=interrupted,
        q6_forward_waypoints=tuple(list(range(975, 646, -25)) + [646]),
        bend_return_waypoints=tuple(
            item.command_targets
            for item in path.waypoints
            if item.phase.startswith("bend_reverse_")
        ),
        q6_return_waypoints=tuple(
            item.command_targets[5]
            for item in path.waypoints
            if item.phase.startswith("q6_reverse_")
        ),
        original_speeds=(1000, 1000, 1000, 1000, 1000, 80),
        original_forces=(500, 500, 500, 500, 500, 80),
    )


def interrupted_stage2_q6_return_plan():
    return InterruptedRecoveryPlan(
        failed_evidence_path=Path("/tmp/stage2-interrupted.json"),
        failed_file_sha256="c" * 64,
        failed_payload_sha256="d" * 64,
        failed_run_id="stage2-interrupted-test",
        target_q6=450,
        step_units=25,
        coupled_targets=(798, 798, 798, 798, 978, 450),
        interrupted_targets=(1000, 1000, 1000, 1000, 1000, 648),
        q6_forward_waypoints=tuple(range(975, 450, -25)) + (450,),
        bend_return_waypoints=(),
        q6_return_waypoints=(),
        original_speeds=(1000, 1000, 40, 40, 40, 40),
        original_forces=(500, 500, 80, 80, 80, 80),
        recovery_mode=RECOVERY_MODE_Q6_RETURN,
        permitted_live_q6_range=(640, 656),
    )


def test_constructor_does_not_lazy_import_examples(monkeypatch):
    calls = []

    def forbidden_import(name):
        calls.append(name)
        raise AssertionError("constructor imported hardware module")

    monkeypatch.setattr(importlib, "import_module", forbidden_import)
    driver, _, _ = make_driver()
    assert driver.stop_requested is False
    assert calls == []


def test_initial_adoption_disables_all_axes_before_any_setting_write():
    driver, hand, _ = make_driver()
    hand.angle_targets = [500, 500, 500, 500, 500, 500]
    for _ in range(3):
        hand.queue(angles=[900] * 6, statuses=[2] * 6)

    driver.adopt_disabled_state_and_verify()

    writes = [item for item in hand.log if item[0] == "write"]
    assert writes[0][1:] == (FakeAPI.REG_ANGLE_SET, DISABLED_TARGETS, 1)
    assert writes[1][1:] == (FakeAPI.REG_ANGLE_SET, DISABLED_TARGETS, 1)
    assert all(
        item[1] not in (FakeAPI.REG_SPEED_SET, FakeAPI.REG_FORCE_SET)
        for item in writes[:2]
    )


def test_open_uses_reviewed_helper_and_strictly_verifies_feedback():
    driver, hand, _ = make_driver()

    open_driver(driver)

    assert (
        "api.open_hand",
        40,
        80,
        5.0,
        True,
    ) in hand.log
    assert tuple(hand.angle_targets) == DISABLED_TARGETS
    assert tuple(hand.speeds) == (40,) * 6
    assert tuple(hand.force_limits) == (80,) * 6
    assert driver.telemetry[-1].phase == "open_verify"


def test_open_rejects_feedback_below_980_and_latches_stop():
    driver, hand, _ = make_driver()
    original_open = FakeAPI.open_hand

    def incomplete_open(*args, **kwargs):
        original_open(*args, **kwargs)
        hand.angles[2] = 979

    driver.api = type("IncompleteAPI", (), {
        **{name: getattr(FakeAPI, name) for name in dir(FakeAPI) if name.startswith("REG_")},
        "open_hand": staticmethod(incomplete_open),
    })

    with pytest.raises(RH56SequenceDriverError, match="every ANGLE_ACT"):
        open_driver(driver)
    assert driver.stop_requested


def test_open_helper_franka_gate_fault_immediately_disables_all_targets():
    calls = 0

    def gate():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("Franka left Idle while opening")

    driver, hand, _ = make_driver(external_safety_check=gate)

    with pytest.raises(RH56SequenceDriverError, match="left Idle while opening"):
        open_driver(driver)

    angle_writes = [
        entry[2]
        for entry in hand.log
        if entry[0] == "write" and entry[1] == FakeAPI.REG_ANGLE_SET
    ]
    assert angle_writes
    assert all(target == DISABLED_TARGETS for target in angle_writes)
    assert tuple(hand.angle_targets) == DISABLED_TARGETS
    assert driver.stop_requested


def test_external_gate_refuses_an_open_helper_without_loop_callback_support():
    class LegacyAPI(FakeAPI):
        @staticmethod
        def open_hand(
            hand, speed, force_limit, motion_timeout, include_thumb_rotate=False
        ):
            raise AssertionError("legacy helper must be rejected before invocation")

    driver, hand, _ = make_driver(external_safety_check=lambda: None)
    driver.api = LegacyAPI

    with pytest.raises(RH56SequenceDriverError, match="lacks the required"):
        open_driver(driver)

    assert tuple(hand.angle_targets) == DISABLED_TARGETS
    assert driver.stop_requested


def test_thumb_target_must_stay_inside_configured_range():
    driver, hand, _ = make_driver(thumb_range=(900, 1000))
    open_driver(driver)
    writes_before = len([item for item in hand.log if item[0] == "write"])

    with pytest.raises(ValueError, match="configured q6 range"):
        driver.preshape_thumb(899)

    writes_after = len([item for item in hand.log if item[0] == "write"])
    assert writes_after == writes_before


def test_formal_thumb_preshape_to_646_uses_only_25_unit_batch_steps():
    driver, hand, _ = make_driver(thumb_range=(646, 1000))
    open_driver(driver)
    target6 = (0, 358, 799, 911, 922, 646)
    bind_audited_path(driver, target6)
    hand.queue(angles=[1000] * 6, statuses=[2] * 6)
    expected = list(range(975, 646, -25)) + [646]
    for waypoint in expected:
        for _ in range(2):
            hand.queue(
                angles=[1000, 1000, 1000, 1000, 1000, waypoint],
                statuses=[2] * 6,
            )

    driver.preshape_thumb(646)

    numeric_q6 = [
        item[2][5]
        for item in hand.log
        if item[0] == "write"
        and item[1] == FakeAPI.REG_ANGLE_SET
        and item[2][5] != -1
    ]
    assert numeric_q6 == expected
    assert all(
        0 < before - after <= 25
        for before, after in zip([1000] + expected, expected)
    )


def test_audited_no_contact_q6_mismatch_is_rejected_before_any_hardware_io():
    driver, hand, _ = make_driver(thumb_range=(646, 1000))
    open_driver(driver)
    bind_audited_path(driver, (0, 358, 799, 911, 922, 646))
    log_before = list(hand.log)

    with pytest.raises(
        RH56SequenceDriverError,
        match="preshape target/path differs from bound installed audit",
    ):
        driver.preshape_thumb(700)

    assert hand.log == log_before
    assert tuple(hand.angle_targets) == DISABLED_TARGETS


def test_audited_no_contact_path_must_fit_configured_q6_range():
    driver, hand, _ = make_driver(thumb_range=(900, 1000))
    open_driver(driver)
    log_before = list(hand.log)

    with pytest.raises(
        RH56SequenceDriverError,
        match="outside the configured realtime range",
    ):
        driver.bind_audited_no_contact_execution_path(
            build_rh56_no_contact_execution_path(
                (0, 358, 799, 911, 922, 646)
            ).as_dict(),
            feedback_envelope_policy=rh56_feedback_envelope_policy(
                driver.angle_tolerance
            ),
        )

    assert hand.log == log_before
    assert tuple(hand.angle_targets) == DISABLED_TARGETS


def test_audited_path_rejects_tampered_feedback_envelope_before_hardware_io():
    driver, hand, _ = make_driver(thumb_range=(646, 1000))
    open_driver(driver)
    path = build_rh56_no_contact_execution_path(
        (0, 358, 799, 911, 922, 646)
    ).as_dict()
    policy = rh56_feedback_envelope_policy(driver.angle_tolerance)
    policy["sha256"] = "0" * 64
    log_before = list(hand.log)

    with pytest.raises(ValueError, match="policy/hash differs"):
        driver.bind_audited_no_contact_execution_path(
            path, feedback_envelope_policy=policy
        )

    assert hand.log == log_before
    assert tuple(hand.angle_targets) == DISABLED_TARGETS


def test_formal_thumb_preshape_rejects_stale_low_q6_before_numeric_motion():
    driver, hand, _ = make_driver(thumb_range=(646, 1000))
    open_driver(driver)
    writes_before = len(hand.log)
    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 646], statuses=[2] * 6
    )

    with pytest.raises(RH56SequenceDriverError, match="all six axes fully open"):
        driver.preshape_thumb(646)

    later_numeric_targets = [
        item[2]
        for item in hand.log[writes_before:]
        if item[0] == "write"
        and item[1] == FakeAPI.REG_ANGLE_SET
        and item[2] != DISABLED_TARGETS
    ]
    assert later_numeric_targets == []
    assert tuple(hand.angle_targets) == DISABLED_TARGETS


def test_external_franka_gate_failure_stops_after_first_numeric_q6_step():
    gate = {"armed": False, "calls": 0}

    def external_gate():
        if not gate["armed"]:
            return
        gate["calls"] += 1
        if gate["calls"] == 3:
            raise RuntimeError("Franka read-only gate lost Idle")

    driver, hand, _ = make_driver(thumb_range=(646, 1000))
    open_driver(driver)
    driver.install_external_safety_check(external_gate)
    writes_before = len(hand.log)
    hand.queue(angles=[1000] * 6, statuses=[2] * 6)
    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 975], statuses=[2] * 6
    )
    gate["armed"] = True

    with pytest.raises(RH56SequenceDriverError, match="Franka read-only gate lost Idle"):
        driver.preshape_thumb(646)

    later_angle_writes = [
        item[2]
        for item in hand.log[writes_before:]
        if item[0] == "write" and item[1] == FakeAPI.REG_ANGLE_SET
    ]
    assert later_angle_writes[0] == (-1, -1, -1, -1, -1, 975)
    assert all(targets == DISABLED_TARGETS for targets in later_angle_writes[1:])
    assert driver.stop_requested
    assert tuple(hand.angle_targets) == DISABLED_TARGETS


def test_external_gate_setter_is_one_shot_and_only_allowed_while_disabled():
    driver, hand, _ = make_driver()
    with pytest.raises(RH56SequenceDriverError, match="only be installed while disabled"):
        driver.install_external_safety_check(lambda: None)

    open_driver(driver)
    driver.install_external_safety_check(lambda: None)
    with pytest.raises(RH56SequenceDriverError, match="already installed"):
        driver.install_external_safety_check(lambda: None)


def test_bends_cannot_close_before_successful_thumb_preshape():
    driver, hand, _ = make_driver()
    open_driver(driver)

    with pytest.raises(RH56SequenceDriverError, match="not been successfully preshaped"):
        driver.close_bends_and_hold((700, 700, 700, 700, 800, 900))

    assert tuple(hand.angle_targets) == DISABLED_TARGETS


def test_status3_is_accepted_as_grasp_contact_and_numeric_hold_remains():
    driver, hand, _ = make_driver()
    open_driver(driver)
    preshape(driver, hand, 900)
    target = (700, 710, 720, 730, 800, 900)
    hand.current_limits = [250, 260, 270, 280, 290, 300]
    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 900], statuses=[2] * 6
    )
    hand.queue(
        angles=[820, 830, 720, 850, 810, 900],
        statuses=[3, 3, 2, 3, 2, 2],
    )

    driver.close_bends_and_hold(target)

    assert driver.numeric_hold_targets == target
    assert driver.numeric_hold_current_caps == (250, 260, 270, 280, 290, 300)
    assert tuple(hand.angle_targets) == target
    assert driver.last_contact_axes == ("pinky", "ring", "index")
    assert driver.telemetry[-1].currents == (0, 0, 0, 0, 0, 0)

    writes_before = len([item for item in hand.log if item[0] == "write"])
    driver.verify_bounded_hold(target, no_contact=False)
    writes_after = len([item for item in hand.log if item[0] == "write"])

    assert writes_after == writes_before
    assert driver.last_contact_axes == ("pinky", "ring", "index")
    assert driver.telemetry[-1].phase == "bounded_hold_verify"


@pytest.mark.parametrize(
    ("no_contact", "statuses", "angles", "currents", "message"),
    [
        (
            True,
            (3, 2, 2, 2, 2, 2),
            (850, 710, 720, 730, 800, 900),
            (0, 0, 0, 0, 0, 0),
            "all-six STATUS=2",
        ),
        (
            True,
            (1, 2, 2, 2, 2, 2),
            (700, 710, 720, 730, 800, 900),
            (0, 0, 0, 0, 0, 0),
            "all-six STATUS=2",
        ),
        (
            False,
            (1, 2, 2, 2, 2, 2),
            (700, 710, 720, 730, 800, 900),
            (0, 0, 0, 0, 0, 0),
            "neither settled nor in contact",
        ),
        (
            False,
            (2, 2, 2, 2, 2, 2),
            (721, 710, 720, 730, 800, 900),
            (0, 0, 0, 0, 0, 0),
            "settled ANGLE_ACT is outside tolerance",
        ),
        (
            False,
            (3, 2, 2, 2, 2, 3),
            (850, 710, 720, 730, 800, 921),
            (0, 0, 0, 0, 0, 0),
            "thumb rotation is outside target tolerance",
        ),
        (
            False,
            (3, 2, 2, 2, 2, 2),
            (650, 710, 720, 730, 800, 900),
            (0, 0, 0, 0, 0, 0),
            "ANGLE_ACT",
        ),
        (
            False,
            (3, 2, 2, 2, 2, 2),
            (850, 710, 720, 730, 800, 900),
            (401, 0, 0, 0, 0, 0),
            "exceeded strict cap 400mA",
        ),
    ],
)
def test_bounded_hold_read_only_verifier_rejects_invalid_live_sample(
    no_contact, statuses, angles, currents, message
):
    driver, hand, _ = make_driver()
    target = (700, 710, 720, 730, 800, 900)
    driver._numeric_hold_targets = target
    driver._numeric_hold_current_caps = (400, 400, 400, 400, 400, 400)
    hand.angle_targets = list(target)
    hand.statuses = list(statuses)
    hand.angles = list(angles)
    hand.currents = list(currents)
    writes_before = len([item for item in hand.log if item[0] == "write"])

    with pytest.raises(RH56SequenceDriverError, match=message):
        driver.verify_bounded_hold(target, no_contact=no_contact)

    assert len([item for item in hand.log if item[0] == "write"]) == writes_before
    assert tuple(hand.angle_targets) == target
    assert driver.stop_requested is False


def test_bounded_hold_target_or_cap_contract_rejects_before_register_reads():
    driver, hand, _ = make_driver()
    target = (700, 710, 720, 730, 800, 900)
    driver._numeric_hold_targets = target
    reads_before = len([item for item in hand.log if item[0].startswith("read")])

    with pytest.raises(RH56SequenceDriverError, match="current-cap contract"):
        driver.verify_bounded_hold(target, no_contact=False)
    assert len([item for item in hand.log if item[0].startswith("read")]) == reads_before

    driver._numeric_hold_current_caps = (400,) * 6
    with pytest.raises(RH56SequenceDriverError, match="does not match"):
        driver.verify_bounded_hold(
            (701, 710, 720, 730, 800, 900), no_contact=False
        )
    assert len([item for item in hand.log if item[0].startswith("read")]) == reads_before


def test_bounded_hold_read_runs_external_gate_once_before_and_after_sample():
    checks = []
    driver, hand, _ = make_driver(
        external_safety_check=lambda: checks.append("checked")
    )
    target = (700, 710, 720, 730, 800, 900)
    driver._numeric_hold_targets = target
    driver._numeric_hold_current_caps = (400,) * 6
    hand.angle_targets = list(target)
    hand.angles = list(target)
    hand.statuses = [2] * 6

    driver.verify_bounded_hold(target, no_contact=True)

    assert checks == ["checked", "checked"]


def test_loaded_hold_verification_is_read_only_and_refreshes_contacts():
    driver, hand, _ = make_driver()
    open_driver(driver)
    preshape(driver, hand, 900)
    target = (700, 710, 720, 730, 800, 900)
    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 900], statuses=[2] * 6
    )
    hand.queue(
        angles=[820, 830, 720, 850, 810, 900],
        statuses=[3, 3, 2, 3, 2, 2],
    )
    driver.close_bends_and_hold(target)
    writes_before = len([item for item in hand.log if item[0] == "write"])

    driver.verify_loaded_hold(target, minimum_contact_axes=2)

    writes_after = len([item for item in hand.log if item[0] == "write"])
    assert writes_after == writes_before
    assert driver.numeric_hold_targets == target
    assert driver.last_contact_axes == ("pinky", "ring", "index")
    assert driver.telemetry[-1].phase == "loaded_hold_verify"


def test_loaded_hold_contact_loss_raises_without_disabling_numeric_target():
    driver, hand, _ = make_driver()
    open_driver(driver)
    preshape(driver, hand, 900)
    target = (700, 710, 720, 730, 800, 900)
    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 900], statuses=[2] * 6
    )
    hand.queue(
        angles=[820, 830, 720, 850, 810, 900],
        statuses=[3, 3, 2, 3, 2, 2],
    )
    driver.close_bends_and_hold(target)
    hand.angles = list(target)
    hand.statuses = [2] * 6
    writes_before = len([item for item in hand.log if item[0] == "write"])

    with pytest.raises(RH56SequenceDriverError, match="contact count"):
        driver.verify_loaded_hold(target, minimum_contact_axes=1)

    writes_after = len([item for item in hand.log if item[0] == "write"])
    assert writes_after == writes_before
    assert not driver.stop_requested
    assert driver.numeric_hold_targets == target
    assert tuple(hand.angle_targets) == target


def test_loaded_hold_reuses_active_numeric_current_caps_without_disabling():
    driver, hand, _ = make_driver()
    target = (700, 710, 720, 730, 800, 900)
    driver._numeric_hold_targets = target
    driver._numeric_hold_current_caps = (250, 250, 250, 250, 250, 250)
    hand.angle_targets = list(target)
    hand.angles = [820, 710, 720, 730, 800, 900]
    hand.statuses = [3, 2, 2, 2, 2, 2]
    hand.currents = [251, 0, 0, 0, 0, 0]
    writes_before = len([item for item in hand.log if item[0] == "write"])

    with pytest.raises(RH56SequenceDriverError, match="strict cap 250mA"):
        driver.verify_loaded_hold(target, minimum_contact_axes=1)

    assert len([item for item in hand.log if item[0] == "write"]) == writes_before
    assert driver.numeric_hold_targets == target
    assert tuple(hand.angle_targets) == target
    assert driver.stop_requested is False


def test_loaded_hold_rejects_different_artifact_target_before_register_reads():
    driver, hand, _ = make_driver()
    driver._numeric_hold_targets = (700, 710, 720, 730, 800, 900)
    reads_before = len([item for item in hand.log if item[0].startswith("read")])

    with pytest.raises(RH56SequenceDriverError, match="does not match"):
        driver.verify_loaded_hold((701, 710, 720, 730, 800, 900), 1)

    reads_after = len([item for item in hand.log if item[0].startswith("read")])
    assert reads_after == reads_before


def test_boundary_state_snapshot_is_read_only_and_allowed_after_disable_latch():
    safety_checks = []
    driver, hand, _ = make_driver(
        external_safety_check=lambda: safety_checks.append("checked")
    )
    hand.angle_targets = list(DISABLED_TARGETS)
    hand.angles = [1000, 990, 980, 970, 960, 950]
    hand.statuses = [2, 3, 2, 3, 2, 2]
    driver._stop_requested.set()
    writes_before = len([item for item in hand.log if item[0] == "write"])

    snapshot = driver.read_state_snapshot()

    assert snapshot.angle_targets == DISABLED_TARGETS
    assert snapshot.angles == (1000, 990, 980, 970, 960, 950)
    assert snapshot.contact_axes == ("ring", "index")
    assert safety_checks == ["checked", "checked"]
    assert len([item for item in hand.log if item[0] == "write"]) == writes_before
    assert driver.telemetry[-1].phase == "boundary_state_snapshot"


def test_status3_is_rejected_for_no_contact_air_close_and_latches_stop():
    driver, hand, _ = make_driver()
    open_driver(driver)
    target = (700, 710, 720, 730, 800, 900)
    bind_audited_path(driver, target)
    preshape(driver, hand, 900)
    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 900], statuses=[2] * 6
    )
    hand.queue(angles=[975, 1000, 1000, 1000, 1000, 900], statuses=[3, 2, 2, 2, 2, 2])

    with pytest.raises(
        RH56SequenceDriverError,
        match="no-contact.*pinky",
    ):
        driver.close_bends_no_contact_and_hold(target)

    assert driver.stop_requested
    # The standalone driver latches first; the state-machine coordinator (or
    # an application's finally block) then performs the verified disable.
    for _ in range(3):
        hand.queue(
            angles=[820, 830, 720, 850, 810, 900], statuses=[2] * 6
        )
    driver.disable_and_verify()
    assert tuple(hand.angle_targets) == DISABLED_TARGETS


@pytest.mark.parametrize(
    "frame, message",
    [
        ({"errors": [0, 0, 1, 0, 0, 0]}, "ERROR is nonzero"),
        ({"statuses": [2, 2, 5, 2, 2, 2]}, "fault status 5"),
        ({"temperatures": [30, 30, 60, 30, 30, 30]}, "temperature reached"),
    ],
)
def test_close_fault_feedback_is_rejected_and_stop_is_latched(frame, message):
    driver, hand, _ = make_driver()
    open_driver(driver)
    preshape(driver, hand, 900)
    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 900], statuses=[2] * 6
    )
    hand.queue(
        angles=[900, 900, 900, 900, 900, 900],
        statuses=frame.get("statuses", [0, 0, 0, 0, 0, 2]),
        errors=frame.get("errors", [0] * 6),
        temperatures=frame.get("temperatures", [30] * 6),
    )

    with pytest.raises(RH56SequenceDriverError, match=message):
        driver.close_bends_and_hold((700, 700, 700, 700, 800, 900))
    assert driver.stop_requested


def test_thumb_preshape_timeout_is_reported_without_current_threshold_fault():
    driver, hand, clock = make_driver(timeout=5.0)
    open_driver(driver)
    hand.currents = [500, 500, 500, 500, 500, 500]
    hand.queue(angles=[1000] * 6, statuses=[2] * 6)
    for _ in range(7):
        hand.queue(
            angles=[1000, 1000, 1000, 1000, 1000, 960],
            statuses=[2, 2, 2, 2, 2, 0],
        )

    with pytest.raises(RH56SequenceDriverError, match="timed out"):
        driver.preshape_thumb(900)

    assert clock.now >= 5.0
    assert driver.telemetry[-1].currents == (500,) * 6


def test_commission_thumb_sweep_uses_exact_small_steps_and_stable_feedback():
    driver, hand, _ = make_driver(thumb_range=(0, 1000))
    open_driver(driver)
    hand.queue(angles=[1000] * 6, statuses=[2] * 6)  # preflight
    for angles, statuses in (
        ([1000, 1000, 1000, 1000, 1000, 990], [2, 2, 2, 2, 2, 0]),
        ([1000, 1000, 1000, 1000, 1000, 976], [2] * 6),
        ([1000, 1000, 1000, 1000, 1000, 976], [2] * 6),
        ([1000, 1000, 1000, 1000, 1000, 976], [2] * 6),
        ([1000, 1000, 1000, 1000, 1000, 960], [2, 2, 2, 2, 2, 0]),
        ([1000, 1000, 1000, 1000, 1000, 951], [2] * 6),
        ([1000, 1000, 1000, 1000, 1000, 951], [2] * 6),
        ([1000, 1000, 1000, 1000, 1000, 951], [2] * 6),
    ):
        hand.queue(angles=angles, statuses=statuses)

    waypoints = driver.commission_thumb_sweep(
        950,
        step_units=25,
        max_axis_current_ma=400,
        endpoint_stable_samples=3,
    )

    assert waypoints == (975, 950)
    angle_writes = [
        item[2]
        for item in hand.log
        if item[0] == "write" and item[1] == FakeAPI.REG_ANGLE_SET
    ]
    assert angle_writes[-2:] == [
        (-1, -1, -1, -1, -1, 975),
        (-1, -1, -1, -1, -1, 950),
    ]
    assert [sample.phase for sample in driver.telemetry].count("q6_step_0975") == 4
    assert [sample.phase for sample in driver.telemetry].count("q6_step_0950") == 4


def test_commission_thumb_sweep_accepts_exact_open_q6_without_substituting_999():
    driver, hand, _ = make_driver(thumb_range=(0, 1000))
    open_driver(driver)
    hand.queue(angles=[1000] * 6, statuses=[2] * 6)  # preflight
    for _ in range(3):
        hand.queue(angles=[1000] * 6, statuses=[2] * 6)

    waypoints = driver.commission_thumb_sweep(
        1000,
        step_units=25,
        max_axis_current_ma=400,
        endpoint_stable_samples=3,
    )

    assert waypoints == (1000,)
    angle_writes = [
        item[2]
        for item in hand.log
        if item[0] == "write" and item[1] == FakeAPI.REG_ANGLE_SET
    ]
    assert angle_writes[-1] == (-1, -1, -1, -1, -1, 1000)
    assert (-1, -1, -1, -1, -1, 999) not in angle_writes


def test_commission_thumb_sweep_enforces_per_axis_current_cap_and_latches_stop():
    driver, hand, _ = make_driver(thumb_range=(0, 1000))
    open_driver(driver)
    hand.currents[5] = 401
    hand.queue(angles=[1000] * 6, statuses=[2] * 6)

    with pytest.raises(RH56SequenceDriverError, match="strict cap 400mA"):
        driver.commission_thumb_sweep(950)

    assert driver.stop_requested


def test_q6_feedback_escape_beyond_adjacent_endpoint_hull_disables_immediately():
    driver, hand, _ = make_driver(thumb_range=(0, 1000))
    open_driver(driver)
    hand.queue(angles=[1000] * 6, statuses=[2] * 6)  # preflight
    # 949 is below min(1000, 975) - arrival_tolerance(20) = 955.
    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 949],
        statuses=[2, 2, 2, 2, 2, 0],
    )

    with pytest.raises(RH56SequenceDriverError, match="escaped"):
        driver.commission_thumb_sweep(975)

    assert driver.stop_requested
    assert tuple(hand.angle_targets) == DISABLED_TARGETS
    angle_writes = [
        item[2]
        for item in hand.log
        if item[0] == "write" and item[1] == FakeAPI.REG_ANGLE_SET
    ]
    assert (-1, -1, -1, -1, -1, 975) in angle_writes
    assert angle_writes[-1] == DISABLED_TARGETS


def test_candidate51_q6_646_uses_small_steps_and_only_a_short_final_remainder():
    driver, hand, _ = make_driver(thumb_range=(0, 1000))
    open_driver(driver)
    hand.queue(angles=[1000] * 6, statuses=[2] * 6)
    expected = list(range(975, 646, -25)) + [646]
    for target in expected:
        for _ in range(3):
            hand.queue(
                angles=[1000, 1000, 1000, 1000, 1000, target],
                statuses=[2] * 6,
            )

    waypoints = driver.commission_thumb_sweep(
        646, step_units=25, endpoint_stable_samples=3
    )

    assert list(waypoints) == expected
    assert waypoints[0] == 975
    assert waypoints[-2:] == (650, 646)
    assert all(0 < a - b <= 25 for a, b in zip(waypoints, waypoints[1:]))


def _commission_fake_q6_to_850(driver, hand):
    hand.queue(angles=[1000] * 6, statuses=[2] * 6)
    for waypoint in (950, 900, 850):
        for _ in range(3):
            hand.queue(
                angles=[1000, 1000, 1000, 1000, 1000, waypoint],
                statuses=[2] * 6,
            )
    assert driver.commission_thumb_sweep(
        850, step_units=50, endpoint_stable_samples=3
    ) == (950, 900, 850)


def test_wide_q6_uses_monitored_reverse_because_reviewed_open_helper_refuses_850():
    driver, hand, _ = make_driver(thumb_range=(0, 1000))
    open_driver(driver)
    _commission_fake_q6_to_850(driver, hand)

    with pytest.raises(RuntimeError, match="refuses starts below 885"):
        FakeAPI.open_hand(hand, 40, 80, 5.0, include_thumb_rotate=True)

    for _ in range(3):
        hand.queue(
            angles=[1000, 1000, 1000, 1000, 1000, 850], statuses=[2] * 6
        )
    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 850], statuses=[2] * 6
    )
    expected_return = (900, 925, 950, 975, 1000)
    for waypoint in expected_return:
        for _ in range(3):
            hand.queue(
                angles=[1000, 1000, 1000, 1000, 1000, waypoint],
                statuses=[2] * 6,
            )
    for _ in range(4):
        hand.queue(angles=[1000] * 6, statuses=[2] * 6)

    returned = driver.return_commissioned_thumb_to_open(
        endpoint_stable_samples=3
    )

    assert returned == expected_return
    assert tuple(hand.angle_targets) == DISABLED_TARGETS
    assert driver.stop_requested is False
    return_phases = [
        sample.phase for sample in driver.telemetry if sample.phase.startswith("q6_return_")
    ]
    assert "q6_return_0900" in return_phases
    assert "q6_return_1000" in return_phases
    assert return_phases[-1] == "q6_return_open_verify"


def test_q6_reverse_v3_accepts_observed_hysteresis_and_direct_stage1_open_985():
    driver, hand, _ = make_driver(thumb_range=(0, 1000))
    open_driver(driver)
    hand.queue(angles=[1000] * 6, statuses=[2] * 6)
    for actual in (950, 900):
        for _ in range(3):
            reported = 895 if actual == 900 else actual
            hand.queue(
                angles=[1000, 1000, 1000, 1000, 1000, reported],
                statuses=[2] * 6,
            )
    assert driver.commission_thumb_sweep(
        900, step_units=50, endpoint_stable_samples=3
    ) == (950, 900)

    # disable verification + fresh return preflight at the observed 895 endpoint
    for _ in range(4):
        hand.queue(
            angles=[1000, 1000, 1000, 1000, 1000, 895], statuses=[2] * 6
        )
    # Direct Stage1 return accepts the proven firmware open endpoint 985.
    for _ in range(7):
        hand.queue(
            angles=[1000, 1000, 1000, 1000, 1000, 985], statuses=[2] * 6
        )
    returned = driver.return_commissioned_thumb_to_open(
        endpoint_stable_samples=3, direct_q6_return_to_open=True
    )
    assert returned == (1000,)
    assert tuple(hand.angle_targets) == DISABLED_TARGETS


def test_q6_reverse_v3_empirical_895_to_923_is_accepted():
    driver, hand, _ = make_driver(thumb_range=(0, 1000))
    open_driver(driver)
    hand.queue(angles=[1000] * 6, statuses=[2] * 6)
    for actual in (950, 895):
        for _ in range(3):
            hand.queue(
                angles=[1000, 1000, 1000, 1000, 1000, actual], statuses=[2] * 6
            )
    driver.commission_thumb_sweep(900, step_units=50, endpoint_stable_samples=3)
    for _ in range(4):
        hand.queue(
            angles=[1000, 1000, 1000, 1000, 1000, 895], statuses=[2] * 6
        )
    # 923 is 27 below target950 but shows +28 progress; v3 accepts this
    # commissioned hysteresis endpoint while retaining idle/current gates.
    for _ in range(3):
        hand.queue(
            angles=[1000, 1000, 1000, 1000, 1000, 923],
            statuses=[2] * 6,
        )
    for actual in (950, 985):
        for _ in range(3):
            hand.queue(
                angles=[1000, 1000, 1000, 1000, 1000, actual], statuses=[2] * 6
            )
    for _ in range(4):
        hand.queue(
            angles=[1000, 1000, 1000, 1000, 1000, 985], statuses=[2] * 6
        )
    assert driver.return_commissioned_thumb_to_open(
        endpoint_stable_samples=3
    ) == (950, 975, 1000)


def test_q6_reverse_contact_fails_and_immediately_disables_all_six_targets():
    driver, hand, _ = make_driver(thumb_range=(0, 1000))
    open_driver(driver)
    _commission_fake_q6_to_850(driver, hand)
    for _ in range(3):
        hand.queue(
            angles=[1000, 1000, 1000, 1000, 1000, 850], statuses=[2] * 6
        )
    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 850], statuses=[2] * 6
    )
    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 875],
        statuses=[2, 2, 2, 2, 2, 3],
    )

    with pytest.raises(RH56SequenceDriverError, match="force contact"):
        driver.return_commissioned_thumb_to_open(endpoint_stable_samples=3)

    assert driver.stop_requested
    assert tuple(hand.angle_targets) == DISABLED_TARGETS
    assert driver.last_emergency_disable_errors == ()


def test_q6_step_cannot_pass_merely_because_start_is_inside_angle_tolerance():
    driver, hand, clock = make_driver(thumb_range=(0, 1000), timeout=5.0)
    open_driver(driver)
    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 986], statuses=[2] * 6
    )
    for _ in range(7):
        hand.queue(
            angles=[1000, 1000, 1000, 1000, 1000, 986], statuses=[2] * 6
        )

    with pytest.raises(RH56SequenceDriverError, match="timed out"):
        driver.commission_thumb_sweep(
            975, step_units=25, endpoint_stable_samples=3
        )

    assert clock.now >= 5.0
    assert driver.stop_requested


def test_stage1_q6_sweep_accepts_physical_978_open_then_proves_later_steps():
    driver, hand, _ = make_driver(
        thumb_range=(0, 1000), open_min_angle=975
    )
    open_driver(driver)
    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 978],
        statuses=[2] * 6,
    )
    for actual in (978, 950, 925, 900):
        for _ in range(3):
            hand.queue(
                angles=[1000, 1000, 1000, 1000, 1000, actual],
                statuses=[2] * 6,
            )

    assert driver.commission_thumb_sweep(
        900, step_units=25, endpoint_stable_samples=3
    ) == (975, 950, 925, 900)


def test_commission_coupled_air_close_requires_no_contact_stable_endpoint():
    driver, hand, _ = make_driver(thumb_range=(0, 1000))
    open_driver(driver)
    hand.queue(angles=[1000] * 6, statuses=[2] * 6)
    for _ in range(3):
        hand.queue(
            angles=[1000, 1000, 1000, 1000, 1000, 950],
            statuses=[2] * 6,
        )
    driver.commission_thumb_sweep(
        950, step_units=50, endpoint_stable_samples=3
    )

    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 950], statuses=[2] * 6
    )
    expected_waypoints = coupled_waypoints((900,) * 5, 950, 25)
    for waypoint in expected_waypoints:
        for _ in range(3):
            hand.queue(angles=waypoint, statuses=[2] * 6)

    reached = driver.commission_coupled_air_close(
        (900, 900, 900, 900, 900), endpoint_stable_samples=3
    )

    assert reached == (900, 900, 900, 900, 900, 950)
    assert driver.numeric_hold_targets == reached
    assert driver.numeric_hold_current_caps == (400, 400, 400, 400, 400, 400)
    numeric_writes = [
        item[2]
        for item in hand.log
        if item[0] == "write"
        and item[1] == FakeAPI.REG_ANGLE_SET
        and item[2][0] != -1
    ]
    assert numeric_writes[-len(expected_waypoints):] == expected_waypoints
    assert all(
        sum(a != b for a, b in zip(before[:5], after[:5])) == 1
        and max(abs(a - b) for a, b in zip(before[:5], after[:5])) <= 25
        for before, after in zip(
            [(1000, 1000, 1000, 1000, 1000, 950)] + expected_waypoints,
            expected_waypoints,
        )
    )
    writes_before = len([item for item in hand.log if item[0] == "write"])
    driver.verify_bounded_hold(reached, no_contact=True)
    assert len([item for item in hand.log if item[0] == "write"]) == writes_before


def test_successful_exact_air_close_reopens_bends_then_q6_over_reverse_path():
    driver, hand, _ = make_driver(thumb_range=(0, 1000))
    open_driver(driver)
    hand.queue(angles=[1000] * 6, statuses=[2] * 6)
    for _ in range(3):
        hand.queue(
            angles=[1000, 1000, 1000, 1000, 1000, 950],
            statuses=[2] * 6,
        )
    driver.commission_thumb_sweep(
        950, step_units=50, endpoint_stable_samples=3
    )

    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 950], statuses=[2] * 6
    )
    forward_bends = coupled_waypoints((950,) * 5, 950, 25)
    for waypoint in forward_bends:
        for _ in range(3):
            hand.queue(angles=waypoint, statuses=[2] * 6)
    driver.commission_coupled_air_close(
        (950, 950, 950, 950, 950), endpoint_stable_samples=3
    )

    closed = (950, 950, 950, 950, 950, 950)
    for _ in range(3):
        hand.queue(angles=closed, statuses=[2] * 6)
    reverse_bends = tuple(
        item.command_targets
        for item in build_rh56_no_contact_execution_path((950,) * 6).waypoints
        if item.phase.startswith("bend_reverse_")
    )
    for waypoint in reverse_bends:
        for _ in range(3):
            hand.queue(angles=waypoint, statuses=[2] * 6)
    # Three samples prove the post-bend disable, then one fresh preflight
    # sample binds the q6 reverse sweep to the same endpoint.
    for _ in range(4):
        hand.queue(angles=reverse_bends[-1], statuses=[2] * 6)
    for _ in range(3):
        hand.queue(angles=[1000] * 6, statuses=[2] * 6)
    for _ in range(3):
        hand.queue(angles=[1000] * 6, statuses=[2] * 6)

    writes_before = len(hand.log)
    returned = driver.return_commissioned_thumb_to_open(
        endpoint_stable_samples=3
    )

    assert returned == (1000,)
    numeric = [
        entry[2]
        for entry in hand.log[writes_before:]
        if entry[0] == "write"
        and entry[1] == FakeAPI.REG_ANGLE_SET
        and entry[2] != DISABLED_TARGETS
    ]
    assert numeric == list(reverse_bends) + [(-1, -1, -1, -1, -1, 1000)]
    assert driver.numeric_hold_targets is None
    assert tuple(hand.angle_targets) == DISABLED_TARGETS


def test_commission_coupled_air_close_rejects_contact_and_latches_stop():
    driver, hand, _ = make_driver(thumb_range=(0, 1000))
    open_driver(driver)
    hand.queue(angles=[1000] * 6, statuses=[2] * 6)
    for _ in range(3):
        hand.queue(
            angles=[1000, 1000, 1000, 1000, 1000, 950], statuses=[2] * 6
        )
    driver.commission_thumb_sweep(
        950, step_units=50, endpoint_stable_samples=3
    )
    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 950], statuses=[2] * 6
    )
    hand.queue(
        angles=[975, 1000, 1000, 1000, 1000, 950],
        statuses=[3, 0, 0, 0, 0, 2],
    )

    with pytest.raises(RH56SequenceDriverError, match="force contact"):
        driver.commission_coupled_air_close((900, 900, 900, 900, 900))

    assert driver.stop_requested


def test_numeric_batch_write_requires_exact_readback():
    driver, hand, _ = make_driver()
    open_driver(driver)
    hand.queue(angles=[1000] * 6, statuses=[2] * 6)
    hand.readback_override[FakeAPI.REG_ANGLE_SET] = DISABLED_TARGETS

    with pytest.raises(RH56SequenceDriverError, match="readback mismatch"):
        driver.preshape_thumb(900)
    assert driver.stop_requested


def test_disable_writes_twice_and_verifies_stable_idle_feedback():
    driver, hand, _ = make_driver()
    for _ in range(3):
        hand.queue(angles=[900] * 6, statuses=[2] * 6)
    writes_before = len(hand.log)

    driver.disable_and_verify()

    disable_writes = [
        item
        for item in hand.log[writes_before:]
        if item[:2] == ("write", FakeAPI.REG_ANGLE_SET)
    ]
    assert len(disable_writes) == 2
    assert all(item[2] == DISABLED_TARGETS for item in disable_writes)


def test_disable_waits_through_transient_status_then_requires_fresh_idle_tail():
    driver, hand, clock = make_driver()
    hand.queue(angles=[900] * 6, statuses=[2, 2, 2, 2, 2, 0])
    hand.queue(angles=[900] * 5 + [901], statuses=[2, 2, 2, 2, 2, 1])
    for _ in range(3):
        hand.queue(angles=[900] * 5 + [902], statuses=[2] * 6)

    driver.disable_and_verify()

    assert clock.now == pytest.approx(0.4)
    assert driver.telemetry[-1].statuses == (2,) * 6
    assert tuple(hand.angle_targets) == DISABLED_TARGETS


def test_disable_never_accepts_persistent_transitional_motion():
    driver, hand, clock = make_driver()
    for index in range(20):
        hand.queue(
            angles=[900] * 5 + [900 + index],
            statuses=[2, 2, 2, 2, 2, index % 2],
        )

    with pytest.raises(
        RH56StopUnconfirmed,
        match="did not reach a stable idle tail",
    ):
        driver.disable_and_verify()

    assert clock.now == pytest.approx(driver.stop_verify_timeout_s)
    assert driver._disabled_verified is False
    assert tuple(hand.angle_targets) == DISABLED_TARGETS


@pytest.mark.parametrize(
    ("feedback", "message"),
    [
        ({"statuses": [2, 2, 2, 2, 2, 5]}, "fault status"),
        ({"errors": [0, 0, 0, 0, 0, 1]}, "ERROR remains"),
        ({"currents": [0, 0, 0, 0, 0, 101]}, "current did not return"),
    ],
)
def test_disable_transitional_window_still_rejects_fault_feedback(
    feedback, message
):
    driver, hand, _ = make_driver()
    hand.queue(
        angles=[900] * 6,
        statuses=[2, 2, 2, 2, 2, 1],
    )
    hand.queue(angles=[900] * 6, **feedback)

    with pytest.raises(RH56StopUnconfirmed, match=message):
        driver.disable_and_verify()

    assert driver._disabled_verified is False
    assert tuple(hand.angle_targets) == DISABLED_TARGETS


@pytest.mark.parametrize("franka_mode", ["move", "reflex"])
def test_cleanup_disable_verification_survives_nonidle_franka_gate_and_keeps_numeric_motion_blocked(
    franka_mode,
):
    gate_calls = []

    def nonidle_franka_gate():
        gate_calls.append(franka_mode)
        raise RuntimeError("Franka is in {}".format(franka_mode))

    driver, hand, _ = make_driver()
    open_driver(driver)
    driver.install_external_safety_check(nonidle_franka_gate)
    hand.angle_targets = [900] * 6
    for _ in range(3):
        hand.queue(angles=[900] * 6, statuses=[2] * 6)

    driver.disable_and_verify()

    # Cleanup first writes and reads back all-six -1.  Only then may its
    # physical-idle reads bypass a Franka gate that cannot report Idle during
    # Move/Reflex recovery.
    assert gate_calls == []
    assert tuple(hand.angle_targets) == DISABLED_TARGETS
    assert driver.stop_requested is True

    writes_before_retry = len(hand.log)
    with pytest.raises(RH56MotionStopped, match="stop has been requested"):
        driver.preshape_thumb(900)
    retry_angle_writes = [
        item[2]
        for item in hand.log[writes_before_retry:]
        if item[:2] == ("write", FakeAPI.REG_ANGLE_SET)
    ]
    assert retry_angle_writes
    assert all(targets == DISABLED_TARGETS for targets in retry_angle_writes)


def test_cleanup_gate_bypass_never_extends_past_a_non_disabled_target_readback():
    gate_calls = []

    def nonidle_franka_gate():
        gate_calls.append("called")
        raise RuntimeError("Franka is in reflex")

    driver, hand, _ = make_driver()
    open_driver(driver)
    driver.install_external_safety_check(nonidle_franka_gate)
    hand.readback_override[FakeAPI.REG_ANGLE_SET] = (0, 0, 0, 0, 0, 0)
    log_start = len(hand.log)

    with pytest.raises(
        RH56StopUnconfirmed,
        match="external safety gate bypass requires all-six ANGLE_SET=-1",
    ):
        driver.disable_and_verify()

    cleanup_log = hand.log[log_start:]
    assert gate_calls == []
    # Only the ANGLE_SET reads needed to establish the disabled state are
    # allowed before the bypass.  No ANGLE_ACT/current/status telemetry is
    # consumed after that proof fails.
    assert not any(
        item[0] == "read6" and item[1] == FakeAPI.REG_ANGLE_ACT
        for item in cleanup_log
    )


def test_disable_restores_temporary_speed_and_force_settings():
    driver, hand, _ = make_driver()
    hand.speeds = [910, 920, 930, 940, 950, 960]
    hand.force_limits = [210, 220, 230, 240, 250, 260]
    original_speeds = tuple(hand.speeds)
    original_forces = tuple(hand.force_limits)
    open_driver(driver)
    assert tuple(hand.speeds) == (40,) * 6
    assert tuple(hand.force_limits) == (80,) * 6

    driver.disable_and_verify()

    assert tuple(hand.speeds) == original_speeds
    assert tuple(hand.force_limits) == original_forces


def test_disable_readback_failure_is_stop_unconfirmed_but_second_pass_runs():
    driver, hand, _ = make_driver()
    hand.readback_override[FakeAPI.REG_ANGLE_SET] = (0, 0, 0, 0, 0, 0)
    writes_before = len(hand.log)

    with pytest.raises(RH56StopUnconfirmed, match="STOP UNCONFIRMED"):
        driver.disable_and_verify()

    disable_writes = [
        item
        for item in hand.log[writes_before:]
        if item[:2] == ("write", FakeAPI.REG_ANGLE_SET)
    ]
    assert len(disable_writes) == 2


class FakeSerialContext:
    def __init__(self, *args):
        self.args = args
        self.exited = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exited = True


def test_connect_is_explicit_and_context_close_closes_serial():
    api = type(
        "ConnectAPI",
        (),
        {
            **{name: getattr(FakeAPI, name) for name in dir(FakeAPI) if name.startswith("REG_")},
            "open_hand": staticmethod(FakeAPI.open_hand),
            "find_serial_port": staticmethod(lambda: "/dev/fake"),
            "LinuxSerial": FakeSerialContext,
            "RH56Hand": staticmethod(lambda serial, hand_id: FakeHand()),
        },
    )

    driver = RH56SequenceDriver.connect(
        api=api,
        motion_timeout_s=5.0,
        poll_interval_s=0.01,
        stop_verify_samples=2,
        stop_verify_interval_s=0.0,
        sleep=lambda _: None,
    )
    serial_context = driver._serial_context
    with driver:
        assert serial_context is not None
        assert not serial_context.exited
    assert serial_context.exited


def test_interrupted_recovery_reopens_only_recorded_bend_prefix_then_q6():
    plan = interrupted_recovery_plan()
    clock = FakeClock()
    hand = FakeHand()
    hand.angles = [0, 1000, 997, 1000, 1000, 644]
    driver = InterruptedRecoveryDriver(
        hand,
        FakeAPI,
        thumb_rotate_range=(0, 1000),
        motion_timeout_s=5.0,
        poll_interval_s=1.0,
        thumb_preshape_step_units=25,
        stop_verify_samples=3,
        stop_verify_interval_s=0.1,
        external_safety_check=lambda: None,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    start = [0, 1000, 997, 1000, 1000, 644]
    for _ in range(3):  # initial disable adoption
        hand.queue(angles=start, statuses=[2] * 6)
    driver.adopt_disabled_state_and_verify()

    hand.queue(angles=start, statuses=[2] * 6)  # recovery live-state preflight
    for _ in range(3):  # inherited return's disable adoption
        hand.queue(angles=start, statuses=[2] * 6)
    for waypoint in plan.bend_return_waypoints:
        for _ in range(3):
            hand.queue(angles=waypoint, statuses=[2] * 6)
    opened_bends = [1000] * 5 + [646]
    for _ in range(3):  # post-bend disable proof
        hand.queue(angles=opened_bends, statuses=[2] * 6)
    hand.queue(angles=opened_bends, statuses=[2] * 6)  # q6 preflight
    for waypoint in plan.q6_return_waypoints:
        for _ in range(3):
            hand.queue(angles=[1000] * 5 + [waypoint], statuses=[2] * 6)
    for _ in range(3):  # final disable proof
        hand.queue(angles=[1000] * 6, statuses=[2] * 6)
    hand.queue(angles=[1000] * 6, statuses=[2] * 6)  # final open proof

    writes_before = len(hand.log)
    returned = driver.recover_interrupted_coupled_air_close_to_open(
        plan, endpoint_stable_samples=3
    )

    assert returned == plan.q6_return_waypoints
    numeric = [
        item[2]
        for item in hand.log[writes_before:]
        if item[:2] == ("write", FakeAPI.REG_ANGLE_SET)
        and item[2] != DISABLED_TARGETS
    ]
    assert numeric == list(plan.bend_return_waypoints) + [
        (-1, -1, -1, -1, -1, value) for value in plan.q6_return_waypoints
    ]
    bend_count = len(plan.bend_return_waypoints)
    assert all(
        item[1:5] == (1000, 1000, 1000, 1000)
        for item in numeric[:bend_count]
    )
    assert all(item[5] == 646 for item in numeric[:bend_count])
    assert tuple(hand.angle_targets) == DISABLED_TARGETS
    assert (
        driver.last_recovery_route
        == RECOVERY_ROUTE_COUPLED_CLOSE_PREFIX
    )
    assert driver.last_recovery_initial_q6 == 644

    # The CLI always performs this final verified disable; it must restore the
    # settings sealed before the interrupted commissioning run, not preserve
    # the temporary all-axis 40/80 recovery settings.
    for _ in range(3):
        hand.queue(angles=[1000] * 6, statuses=[2] * 6)
    driver.disable_and_verify()
    assert tuple(hand.speeds) == plan.original_speeds
    assert tuple(hand.force_limits) == plan.original_forces


def test_interrupted_stage2_recovery_reacquires_live_q6_and_uses_canonical_return():
    plan = interrupted_stage2_q6_return_plan()
    clock = FakeClock()
    hand = FakeHand()
    live = [1000, 1000, 1000, 1000, 1000, 649]
    hand.angles = list(live)
    safety_checks = []
    driver = InterruptedRecoveryDriver(
        hand,
        FakeAPI,
        thumb_rotate_range=(0, 1000),
        motion_timeout_s=5.0,
        poll_interval_s=1.0,
        thumb_preshape_step_units=25,
        stop_verify_samples=3,
        stop_verify_interval_s=0.1,
        external_safety_check=lambda: safety_checks.append(True),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    for _ in range(3):
        hand.queue(angles=live, statuses=[2] * 6)
    driver.adopt_disabled_state_and_verify()

    hand.queue(angles=live, statuses=[2] * 6)  # evidence reacquisition
    for _ in range(3):
        hand.queue(angles=live, statuses=[2] * 6)  # return disable adoption
    reacquired = [1000, 1000, 1000, 1000, 1000, 652]
    hand.queue(angles=reacquired, statuses=[2] * 6)  # final pre-write sample
    expected_q6 = (
        702, 727, 752, 777, 802, 827, 852,
        877, 902, 927, 952, 977, 1000,
    )
    for waypoint in expected_q6:
        for _ in range(3):
            hand.queue(
                angles=[1000, 1000, 1000, 1000, 1000, waypoint],
                statuses=[2] * 6,
            )
    for _ in range(3):
        hand.queue(angles=[1000] * 6, statuses=[2] * 6)
    hand.queue(angles=[1000] * 6, statuses=[2] * 6)

    writes_before = len(hand.log)
    returned = driver.recover_interrupted_coupled_air_close_to_open(
        plan, endpoint_stable_samples=3
    )

    assert returned == expected_q6
    numeric = [
        item[2]
        for item in hand.log[writes_before:]
        if item[:2] == ("write", FakeAPI.REG_ANGLE_SET)
        and item[2] != DISABLED_TARGETS
    ]
    assert numeric == [(-1, -1, -1, -1, -1, value) for value in expected_q6]
    assert tuple(hand.angle_targets) == DISABLED_TARGETS
    assert safety_checks
    assert driver.last_recovery_route == RECOVERY_ROUTE_SEALED_Q6_RETURN


def test_interrupted_stage2_recovery_rejects_drift_after_outer_preflight_before_write():
    plan = interrupted_stage2_q6_return_plan()
    clock = FakeClock()
    hand = FakeHand()
    outer = [1000, 1000, 1000, 1000, 1000, 649]
    hand.angles = list(outer)
    driver = InterruptedRecoveryDriver(
        hand,
        FakeAPI,
        thumb_rotate_range=(0, 1000),
        motion_timeout_s=5.0,
        poll_interval_s=1.0,
        thumb_preshape_step_units=25,
        stop_verify_samples=3,
        stop_verify_interval_s=0.1,
        external_safety_check=lambda: None,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    for _ in range(3):
        hand.queue(angles=outer, statuses=[2] * 6)
    driver.adopt_disabled_state_and_verify()
    hand.queue(angles=outer, statuses=[2] * 6)
    for _ in range(3):
        hand.queue(angles=outer, statuses=[2] * 6)
    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 669],
        statuses=[2] * 6,
    )
    writes_before = len(hand.log)

    with pytest.raises(RH56SequenceDriverError, match="sealed final reacquisition"):
        driver.recover_interrupted_coupled_air_close_to_open(plan)

    angle_writes = [
        item[2]
        for item in hand.log[writes_before:]
        if item[:2] == ("write", FakeAPI.REG_ANGLE_SET)
    ]
    assert angle_writes
    assert all(item == DISABLED_TARGETS for item in angle_writes)


@pytest.mark.parametrize("boundary_q6", [640, 656])
def test_interrupted_stage2_recovery_accepts_both_sealed_reacquisition_boundaries(
    boundary_q6,
):
    plan = interrupted_stage2_q6_return_plan()
    clock = FakeClock()
    hand = FakeHand()
    live = [1000, 1000, 1000, 1000, 1000, boundary_q6]
    hand.angles = list(live)
    driver = InterruptedRecoveryDriver(
        hand,
        FakeAPI,
        thumb_rotate_range=(0, 1000),
        motion_timeout_s=5.0,
        poll_interval_s=1.0,
        thumb_preshape_step_units=25,
        stop_verify_samples=3,
        stop_verify_interval_s=0.1,
        external_safety_check=lambda: None,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    for _ in range(3):
        hand.queue(angles=live, statuses=[2] * 6)
    driver.adopt_disabled_state_and_verify()
    hand.queue(angles=live, statuses=[2] * 6)
    for _ in range(3):
        hand.queue(angles=live, statuses=[2] * 6)
    hand.queue(angles=live, statuses=[2] * 6)
    expected = tuple(
        item.command_targets[5]
        for item in build_rh56_no_contact_execution_path(
            live, step_units=25
        ).waypoints
        if item.phase.startswith("q6_reverse_")
    )
    for waypoint in expected:
        for _ in range(3):
            hand.queue(
                angles=[1000, 1000, 1000, 1000, 1000, waypoint],
                statuses=[2] * 6,
            )
    for _ in range(3):
        hand.queue(angles=[1000] * 6, statuses=[2] * 6)
    hand.queue(angles=[1000] * 6, statuses=[2] * 6)

    assert driver.recover_interrupted_coupled_air_close_to_open(plan) == expected


def test_interrupted_recovery_checks_external_franka_gate_before_numeric_write():
    hand = FakeHand()

    def not_idle():
        raise RuntimeError("Franka left Idle")

    driver = InterruptedRecoveryDriver(
        hand,
        FakeAPI,
        thumb_rotate_range=(0, 1000),
        external_safety_check=not_idle,
    )
    writes_before = len(hand.log)

    with pytest.raises(RH56SequenceDriverError, match="external safety check failed"):
        driver._write_six_verified(
            FakeAPI.REG_ANGLE_SET,
            (-1, -1, -1, -1, -1, 700),
            numeric_motion=True,
        )

    assert hand.log[writes_before:] == []


def test_interrupted_stage2_recovery_without_franka_gate_fails_before_io():
    hand = FakeHand()
    driver = InterruptedRecoveryDriver(
        hand,
        FakeAPI,
        thumb_rotate_range=(0, 1000),
        thumb_preshape_step_units=25,
    )

    with pytest.raises(RH56SequenceDriverError, match="Franka read-only Idle gate"):
        driver.recover_interrupted_coupled_air_close_to_open(
            interrupted_stage2_q6_return_plan()
        )

    assert hand.log == []


@pytest.mark.parametrize(
    "outside",
    [
        [1000, 1000, 1000, 1000, 1000, 639],
        [1000, 1000, 1000, 1000, 1000, 657],
        [979, 1000, 1000, 1000, 1000, 648],
    ],
)
def test_interrupted_stage2_recovery_refuses_live_pose_outside_sealed_open_gate(
    outside,
):
    plan = interrupted_stage2_q6_return_plan()
    clock = FakeClock()
    hand = FakeHand()
    hand.angles = list(outside)
    driver = InterruptedRecoveryDriver(
        hand,
        FakeAPI,
        thumb_rotate_range=(0, 1000),
        motion_timeout_s=5.0,
        poll_interval_s=1.0,
        thumb_preshape_step_units=25,
        stop_verify_samples=3,
        stop_verify_interval_s=0.1,
        external_safety_check=lambda: None,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    for _ in range(3):
        hand.queue(angles=outside, statuses=[2] * 6)
    driver.adopt_disabled_state_and_verify()
    hand.queue(angles=outside, statuses=[2] * 6)
    writes_before = len(hand.log)

    with pytest.raises(RH56SequenceDriverError, match="recovery route"):
        driver.recover_interrupted_coupled_air_close_to_open(plan)

    writes = [
        item[2]
        for item in hand.log[writes_before:]
        if item[:2] == ("write", FakeAPI.REG_ANGLE_SET)
    ]
    assert writes
    assert all(item == DISABLED_TARGETS for item in writes)


def _make_near_open_recovery_driver(hand, safety_checks):
    clock = FakeClock()
    return InterruptedRecoveryDriver(
        hand,
        FakeAPI,
        thumb_rotate_range=(0, 1000),
        angle_tolerance=25,
        motion_timeout_s=5.0,
        poll_interval_s=1.0,
        thumb_preshape_step_units=25,
        stop_verify_samples=3,
        stop_verify_interval_s=0.1,
        external_safety_check=lambda: safety_checks.append(True),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )


def _adopt_near_open_start(driver, hand, live):
    hand.angles = list(live)
    for _ in range(3):
        hand.queue(angles=live, statuses=[2] * 6)
    driver.adopt_disabled_state_and_verify()


def _queue_near_open_968_reset(hand, *, final_q6=975):
    live = [1000, 998, 1000, 1000, 995, 968]
    hand.queue(angles=live, statuses=[2] * 6)  # route snapshot
    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 968],
        statuses=[2] * 6,
    )  # reviewed reset preflight
    backoff = [1000, 1000, 1000, 1000, 1000, 918]
    for _ in range(3):
        hand.queue(angles=backoff, statuses=[2] * 6)
    for _ in range(3):
        hand.queue(angles=backoff, statuses=[2] * 6)
    hand.queue(angles=backoff, statuses=[2] * 6)  # fresh forward anchor
    endpoint = [1000, 1000, 1000, 1000, 1000, int(final_q6)]
    for _ in range(8 if final_q6 < 975 else 3):
        hand.queue(angles=endpoint, statuses=[2] * 6)
    if final_q6 >= 975:
        for _ in range(3):
            hand.queue(angles=endpoint, statuses=[2] * 6)
        hand.queue(angles=endpoint, statuses=[2] * 6)


def test_interrupted_stage2_q6_968_uses_reviewed_near_open_deadband_reset():
    plan = interrupted_stage2_q6_return_plan()
    hand = FakeHand()
    live = [1000, 998, 1000, 1000, 995, 968]
    safety_checks = []
    driver = _make_near_open_recovery_driver(hand, safety_checks)
    _adopt_near_open_start(driver, hand, live)
    _queue_near_open_968_reset(hand)
    writes_before = len(hand.log)

    returned = driver.recover_interrupted_coupled_air_close_to_open(plan)

    numeric_angle_writes = [
        item[2]
        for item in hand.log[writes_before:]
        if item[:2] == ("write", FakeAPI.REG_ANGLE_SET)
        and item[2] != DISABLED_TARGETS
    ]
    assert returned == (918, 1000)
    assert numeric_angle_writes == [
        (-1, -1, -1, -1, -1, 918),
        (-1, -1, -1, -1, -1, 1000),
    ]
    assert all(900 <= target[5] <= 1000 for target in numeric_angle_writes)
    assert driver.last_recovery_route == RECOVERY_ROUTE_NEAR_OPEN_RESET
    assert driver.last_recovery_initial_q6 == 968
    assert driver.thumb_rotate_range == (0, 1000)
    assert safety_checks


def test_interrupted_stage2_q6_978_is_already_near_open_without_angle_motion():
    plan = interrupted_stage2_q6_return_plan()
    hand = FakeHand()
    live = [1000, 1000, 1000, 1000, 1000, 978]
    safety_checks = []
    driver = _make_near_open_recovery_driver(hand, safety_checks)
    _adopt_near_open_start(driver, hand, live)
    hand.queue(angles=live, statuses=[2] * 6)  # route snapshot
    hand.queue(angles=live, statuses=[2] * 6)  # reset preflight
    for _ in range(3):
        hand.queue(angles=live, statuses=[2] * 6)
    hand.queue(angles=live, statuses=[2] * 6)  # reset final
    writes_before = len(hand.log)

    returned = driver.recover_interrupted_coupled_air_close_to_open(plan)

    numeric_angle_writes = [
        item
        for item in hand.log[writes_before:]
        if item[:2] == ("write", FakeAPI.REG_ANGLE_SET)
        and item[2] != DISABLED_TARGETS
    ]
    assert returned == ()
    assert numeric_angle_writes == []
    assert driver.last_recovery_route == RECOVERY_ROUTE_NEAR_OPEN_RESET
    assert safety_checks


@pytest.mark.parametrize("q6", [657, 700, 899])
def test_interrupted_stage2_gap_refuses_before_any_numeric_write(q6):
    plan = interrupted_stage2_q6_return_plan()
    hand = FakeHand()
    live = [1000, 1000, 1000, 1000, 1000, q6]
    driver = _make_near_open_recovery_driver(hand, [])
    _adopt_near_open_start(driver, hand, live)
    hand.queue(angles=live, statuses=[2] * 6)
    writes_before = len(hand.log)

    with pytest.raises(RH56SequenceDriverError, match="neither recovery route"):
        driver.recover_interrupted_coupled_air_close_to_open(plan)

    numeric_writes = [
        item
        for item in hand.log[writes_before:]
        if item[0] == "write"
        and item[1] in (
            FakeAPI.REG_ANGLE_SET,
            FakeAPI.REG_SPEED_SET,
            FakeAPI.REG_FORCE_SET,
        )
        and not (
            item[1] == FakeAPI.REG_ANGLE_SET
            and item[2] == DISABLED_TARGETS
        )
    ]
    assert numeric_writes == []
    assert driver.last_recovery_route is None
    assert driver.last_recovery_initial_q6 is None


@pytest.mark.parametrize(
    "feedback",
    [
        {"angles": [979, 1000, 1000, 1000, 1000, 968]},
        {"statuses": [2, 2, 2, 2, 2, 1]},
        {"currents": [0, 0, 0, 0, 0, 101]},
        {"errors": [0, 0, 0, 0, 0, 1]},
        {"temperatures": [30, 30, 30, 30, 30, 50]},
    ],
)
def test_interrupted_stage2_near_open_unsafe_feedback_refuses_before_numeric_write(
    feedback,
):
    plan = interrupted_stage2_q6_return_plan()
    hand = FakeHand()
    safe = [1000, 1000, 1000, 1000, 1000, 968]
    driver = _make_near_open_recovery_driver(hand, [])
    _adopt_near_open_start(driver, hand, safe)
    hand.queue(
        angles=feedback.get("angles", safe),
        statuses=feedback.get("statuses", [2] * 6),
        currents=feedback.get("currents", [0] * 6),
        errors=feedback.get("errors", [0] * 6),
        temperatures=feedback.get("temperatures", [30] * 6),
    )
    writes_before = len(hand.log)

    with pytest.raises(RH56SequenceDriverError):
        driver.recover_interrupted_coupled_air_close_to_open(plan)

    numeric_writes = [
        item
        for item in hand.log[writes_before:]
        if item[0] == "write"
        and item[1] in (
            FakeAPI.REG_ANGLE_SET,
            FakeAPI.REG_SPEED_SET,
            FakeAPI.REG_FORCE_SET,
        )
        and not (
            item[1] == FakeAPI.REG_ANGLE_SET
            and item[2] == DISABLED_TARGETS
        )
    ]
    assert numeric_writes == []


@pytest.mark.parametrize(
    ("final_q6", "passes"),
    [(975, True), (974, True), (969, False)],
)
def test_interrupted_stage2_near_open_profile_endpoint_boundary(
    final_q6, passes
):
    plan = interrupted_stage2_q6_return_plan()
    hand = FakeHand()
    live = [1000, 998, 1000, 1000, 995, 968]
    driver = _make_near_open_recovery_driver(hand, [])
    _adopt_near_open_start(driver, hand, live)
    _queue_near_open_968_reset(hand, final_q6=final_q6)

    if passes:
        assert driver.recover_interrupted_coupled_air_close_to_open(plan) == (
            918,
            1000,
        )
    else:
        with pytest.raises(RH56SequenceDriverError, match="timed out"):
            driver.recover_interrupted_coupled_air_close_to_open(plan)


def test_near_open_cleanup_restore_failure_keeps_verified_stop_classification():
    hand = FakeHand()
    driver = _make_near_open_recovery_driver(hand, [])
    driver.last_recovery_route = RECOVERY_ROUTE_NEAR_OPEN_RESET
    driver._disabled_verified = True
    driver._original_speeds = DEFAULT_SPEEDS
    driver._original_forces = DEFAULT_FORCES
    hand.speeds = [1000, 1000, 1000, 1000, 1000, 40]
    hand.write_failures[FakeAPI.REG_SPEED_SET] = 20
    for _ in range(3):
        hand.queue(angles=[1000, 1000, 1000, 1000, 1000, 975], statuses=[2] * 6)

    with pytest.raises(
        RH56ResetSettingsRestoreError,
        match="all-six disable/idle is verified",
    ) as captured:
        driver.disable_and_verify()

    assert not isinstance(captured.value, RH56StopUnconfirmed)
    assert driver._disabled_verified is True
    assert tuple(hand.angle_targets) == DISABLED_TARGETS


def test_sealed_cleanup_restore_failure_retains_strict_stop_unconfirmed_policy():
    hand = FakeHand()
    driver = _make_near_open_recovery_driver(hand, [])
    driver.last_recovery_route = RECOVERY_ROUTE_SEALED_Q6_RETURN
    driver._disabled_verified = True
    driver._original_speeds = DEFAULT_SPEEDS
    driver._original_forces = DEFAULT_FORCES
    hand.speeds = [1000, 1000, 1000, 1000, 1000, 40]
    hand.write_failures[FakeAPI.REG_SPEED_SET] = 20
    for _ in range(3):
        hand.queue(angles=[1000] * 6, statuses=[2] * 6)

    with pytest.raises(RH56StopUnconfirmed, match="temporary setting restore"):
        driver.disable_and_verify()

    assert driver._disabled_verified is False


def test_interrupted_recovery_refuses_live_pose_mismatch_before_numeric_write():
    plan = interrupted_recovery_plan()
    clock = FakeClock()
    hand = FakeHand()
    hand.angles = [100, 1000, 1000, 1000, 1000, 644]
    driver = InterruptedRecoveryDriver(
        hand,
        FakeAPI,
        thumb_rotate_range=(0, 1000),
        motion_timeout_s=5.0,
        poll_interval_s=1.0,
        thumb_preshape_step_units=25,
        stop_verify_samples=3,
        stop_verify_interval_s=0.1,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    for _ in range(3):
        hand.queue(angles=hand.angles, statuses=[2] * 6)
    driver.adopt_disabled_state_and_verify()
    hand.queue(angles=hand.angles, statuses=[2] * 6)
    writes_before = len(hand.log)

    with pytest.raises(RH56SequenceDriverError, match="does not match"):
        driver.recover_interrupted_coupled_air_close_to_open(plan)

    writes = [
        item[2]
        for item in hand.log[writes_before:]
        if item[:2] == ("write", FakeAPI.REG_ANGLE_SET)
    ]
    assert writes
    assert all(item == DISABLED_TARGETS for item in writes)


def test_interrupted_bend_recovery_polls_latched_stop_before_feedback_io():
    clock = FakeClock()
    hand = FakeHand()
    driver = InterruptedRecoveryDriver(
        hand,
        FakeAPI,
        thumb_rotate_range=(0, 1000),
        motion_timeout_s=5.0,
        poll_interval_s=1.0,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    driver._stop_requested.set()

    with pytest.raises(RH56MotionStopped, match="bend recovery stopped"):
        driver._read_feedback("coupled_air_return_step_0000", 0.0)

    assert hand.log == []


def test_generic_reset_open_moves_only_q6_from_live_644_and_restores_defaults():
    hand = FakeHand()
    hand.angles = [988, 1000, 997, 1000, 1000, 644]
    hand.speeds = [1000, 1000, 1000, 1000, 1000, 40]
    hand.force_limits = [500, 500, 500, 500, 500, 80]
    safety_checks = []
    driver, hand, _clock = make_reset_driver(
        hand, external_safety_check=lambda: safety_checks.append(True)
    )
    driver.adopt_disabled_state_and_verify()

    expected_q6 = [
        694, 719, 744, 769, 794, 819, 844,
        869, 894, 919, 944, 969, 994, 1000,
    ]
    hand.queue(angles=[1000, 1000, 1000, 1000, 1000, 644], statuses=[2] * 6)
    for target in expected_q6:
        # Real q6 feedback may trail a command by about 24 units.  The next
        # command must still advance by 25 from the previous command; deriving
        # it as ACT+25 would issue only +1 and fall inside actuator deadband.
        actual = 985 if target == 1000 else target - 24
        for _ in range(3):
            hand.queue(
                angles=[1000, 1000, 1000, 1000, 1000, actual],
                statuses=[2] * 6,
            )

    writes_before = len(hand.log)
    returned = driver.reset_to_open()
    numeric = [
        item[2]
        for item in hand.log[writes_before:]
        if item[:2] == ("write", FakeAPI.REG_ANGLE_SET)
        and item[2] != DISABLED_TARGETS
    ]

    assert returned == tuple(expected_q6)
    assert numeric == [(-1, -1, -1, -1, -1, value) for value in expected_q6]
    assert numeric[0][5] - 644 == 50
    assert all(
        following[5] - previous[5] <= 25
        for previous, following in zip(numeric, numeric[1:])
    )
    assert tuple(hand.angle_targets) == DISABLED_TARGETS
    assert safety_checks

    driver.disable_and_verify()
    assert tuple(hand.speeds) == DEFAULT_SPEEDS
    assert tuple(hand.force_limits) == DEFAULT_FORCES
    assert tuple(hand.angle_targets) == DISABLED_TARGETS


def test_execution_reset_opens_q6_with_one_direct_endpoint_target():
    hand = FakeHand()
    hand.angles = [900, 920, 940, 960, 980, 644]
    driver, hand, _clock = make_reset_driver(
        hand, external_safety_check=lambda: None
    )
    driver.adopt_disabled_state_and_verify()

    # One post-bend preflight sample followed by the three stable endpoint
    # samples required by the execution reset.
    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 644],
        statuses=[2] * 6,
    )
    for _ in range(3):
        hand.queue(
            angles=[1000, 1000, 1000, 1000, 1000, 985],
            statuses=[2] * 6,
        )

    writes_before = len(hand.log)
    returned = driver.reset_to_open(
        simultaneous_bend_open=True,
        direct_q6_endpoint_open=True,
    )
    new_writes = hand.log[writes_before:]
    numeric_q6_targets = [
        item[2]
        for item in new_writes
        if item[:2] == ("write", FakeAPI.REG_ANGLE_SET)
        and item[2] != DISABLED_TARGETS
    ]
    q6_speed_settings = [
        item[2][5]
        for item in new_writes
        if item[:2] == ("write", FakeAPI.REG_SPEED_SET)
    ]

    assert returned == (1000,)
    assert numeric_q6_targets == [(-1, -1, -1, -1, -1, 1000)]
    assert q6_speed_settings[0] == 330
    assert tuple(hand.angle_targets) == DISABLED_TARGETS
    assert tuple(hand.angles[:5]) == (1000,) * 5
    assert hand.angles[5] == 985


def test_execution_reset_accepts_nonmonotonic_q6_endpoint_settling():
    hand = FakeHand()
    hand.angles = [900, 920, 940, 960, 980, 615]
    driver, hand, _clock = make_reset_driver(
        hand, external_safety_check=lambda: None
    )
    driver.adopt_disabled_state_and_verify()

    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 615],
        statuses=[2] * 6,
    )
    # The installed q6 has produced this exact pattern: one transient sample
    # at 1000 followed by a lower but valid physical endpoint.  Arrival is
    # decided from consecutive healthy idle samples, not sample monotonicity.
    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 1000],
        statuses=[2, 2, 2, 2, 2, 1],
    )
    for _ in range(3):
        hand.queue(
            angles=[1000, 1000, 1000, 1000, 1000, 986],
            statuses=[2] * 6,
        )

    assert driver.reset_to_open(
        simultaneous_bend_open=True,
        direct_q6_endpoint_open=True,
    ) == (1000,)
    assert tuple(hand.angle_targets) == DISABLED_TARGETS
    assert hand.angles[5] == 986


def test_execution_reset_accepts_post_disable_q6_release_drift():
    hand = FakeHand()
    hand.angles = [1000, 1000, 1000, 1000, 1000, 900]
    driver, hand, _clock = make_reset_driver(
        hand,
        q6_open_min_angle=975,
        external_safety_check=lambda: None,
    )
    driver.adopt_disabled_state_and_verify()

    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 900],
        statuses=[2] * 6,
    )
    for _ in range(3):
        hand.queue(
            angles=[1000, 1000, 1000, 1000, 1000, 974],
            statuses=[2] * 6,
        )
    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 973],
        statuses=[2] * 6,
    )

    assert driver.reset_to_open(
        simultaneous_bend_open=True,
        direct_q6_endpoint_open=True,
    ) == (1000,)
    assert tuple(hand.angle_targets) == DISABLED_TARGETS
    assert hand.angles[5] == 973


def test_execution_open_helper_starts_all_five_bends_in_one_batch(monkeypatch):
    api = importlib.import_module("examples.inspire_rh56_test")

    class BatchHand:
        def __init__(self):
            self.targets = [-1] * 6
            self.angles = [800, 820, 840, 860, 880, 640]
            self.positions = [400, 380, 360, 340, 320, 900]
            self.speeds = [1000] * 6
            self.forces = [500] * 6
            self.currents = [0] * 6
            self.errors = [0] * 6
            self.statuses = [2] * 6
            self.temperatures = [30] * 6
            self.writes = []

        def snapshot(self):
            return {
                "angle_targets": tuple(self.targets),
                "angles": tuple(self.angles),
                "positions": tuple(self.positions),
                "speeds": tuple(self.speeds),
                "force_limits": tuple(self.forces),
                "currents": tuple(self.currents),
                "errors": tuple(self.errors),
                "statuses": tuple(self.statuses),
                "temperatures": tuple(self.temperatures),
            }

        def write_six_shorts(self, address, values):
            values = tuple(int(value) for value in values)
            self.writes.append((int(address), values))
            if address == api.REG_ANGLE_SET:
                self.targets = list(values)
                if values == (1000, 1000, 1000, 1000, 1000, -1):
                    self.angles[:5] = [1000] * 5
                    self.positions[:5] = [100] * 5
            elif address == api.REG_SPEED_SET:
                self.speeds = list(values)
            elif address == api.REG_FORCE_SET:
                self.forces = list(values)

        def read_six_shorts(self, address):
            values = {
                api.REG_ANGLE_SET: self.targets,
                api.REG_ANGLE_ACT: self.angles,
                api.REG_POS_ACT: self.positions,
                api.REG_CURRENT: self.currents,
            }[address]
            return list(values)

        def read(self, address, length):
            values = {
                api.REG_ERROR: self.errors,
                api.REG_STATUS: self.statuses,
                api.REG_TEMP: self.temperatures,
            }[address]
            return bytes(values[:length])

    monkeypatch.setattr(api.time, "sleep", lambda _duration: None)
    hand = BatchHand()
    q6_angle_before = hand.angles[5]
    feedback_phases = []

    api.open_hand(
        hand,
        speed=40,
        force_limit=80,
        motion_timeout=1.0,
        simultaneous_bend_open=True,
        max_axis_current_ma=100,
        endpoint_stable_samples=3,
        max_inactive_drift_units=8,
        feedback_callback=lambda payload: feedback_phases.append(
            payload["phase"]
        ),
    )

    numeric_targets = [
        values
        for address, values in hand.writes
        if address == api.REG_ANGLE_SET and values != (-1,) * 6
    ]
    assert numeric_targets == [(1000, 1000, 1000, 1000, 1000, -1)]
    assert tuple(hand.targets) == (-1,) * 6
    assert tuple(hand.angles[:5]) == (1000,) * 5
    assert hand.angles[5] == q6_angle_before
    assert tuple(hand.speeds) == (1000,) * 6
    assert tuple(hand.forces) == (500,) * 6
    assert feedback_phases == ["reset_open_bends_all_1000"] * 3


def test_generic_reset_open_escapes_physical_q6_zero_endpoint_then_uses_dense_path():
    hand = FakeHand()
    hand.angles = [1000, 1000, 1000, 1000, 1000, 0]
    driver, hand, _clock = make_reset_driver(hand, thumb_range=(0, 1000))
    driver.adopt_disabled_state_and_verify()

    hand.queue(angles=[1000, 1000, 1000, 1000, 1000, 0], statuses=[2] * 6)
    expected_q6 = [400] + list(range(425, 1000, 25)) + [1000]
    for target in expected_q6:
        actual = 985 if target == 1000 else target
        for _ in range(3):
            hand.queue(
                angles=[1000, 1000, 1000, 1000, 1000, actual],
                statuses=[2] * 6,
            )

    writes_before = len(hand.log)
    returned = driver.reset_to_open()
    numeric_q6 = [
        item[2][5]
        for item in hand.log[writes_before:]
        if item[:2] == ("write", FakeAPI.REG_ANGLE_SET)
        and item[2] != DISABLED_TARGETS
    ]

    assert returned == tuple(expected_q6)
    assert numeric_q6 == expected_q6
    assert numeric_q6[0] == 400
    assert all(
        following - previous == 25
        for previous, following in zip(numeric_q6[1:-1], numeric_q6[2:-1])
    )
    assert tuple(hand.angle_targets) == DISABLED_TARGETS


def test_generic_reset_open_escapes_973_deadband_with_backoff_disable_and_fresh_anchor():
    hand = FakeHand()
    hand.angles = [988, 1000, 997, 1000, 1000, 973]
    safety_checks = []
    driver, hand, _clock = make_reset_driver(
        hand,
        thumb_range=(900, 1000),
        external_safety_check=lambda: safety_checks.append(tuple(hand.angle_targets)),
    )
    driver.adopt_disabled_state_and_verify()

    # Five-bend open helper leaves q6 at 973.  Backoff target 920 is a
    # 53-unit descending command; accepted ACT=949 preserves a >=51-unit
    # direct-open span.  Three disabled/idle samples and one fresh anchor sit
    # between the two numeric directions.
    hand.queue(angles=[1000, 1000, 1000, 1000, 1000, 973], statuses=[2] * 6)
    for _ in range(3):
        hand.queue(
            angles=[1000, 1000, 1000, 1000, 1000, 949],
            statuses=[2] * 6,
        )
    for _ in range(3):
        hand.queue(
            angles=[1000, 1000, 1000, 1000, 1000, 949],
            statuses=[2] * 6,
        )
    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 949],
        statuses=[2] * 6,
    )
    for _ in range(3):
        hand.queue(
            angles=[1000, 1000, 1000, 1000, 1000, 985],
            statuses=[2] * 6,
        )

    writes_before = len(hand.log)
    returned = driver.reset_to_open()
    angle_writes = [
        item[2]
        for item in hand.log[writes_before:]
        if item[:2] == ("write", FakeAPI.REG_ANGLE_SET)
    ]
    numeric_indices = [
        index
        for index, values in enumerate(angle_writes)
        if values != DISABLED_TARGETS
    ]

    assert returned == (920, 1000)
    assert [angle_writes[index][5] for index in numeric_indices] == [920, 1000]
    assert 973 - angle_writes[numeric_indices[0]][5] >= 50
    assert all(
        900 <= angle_writes[index][5] <= 1000 for index in numeric_indices
    )
    assert angle_writes[numeric_indices[0] + 1:numeric_indices[1]].count(
        DISABLED_TARGETS
    ) >= 2
    assert tuple(hand.angle_targets) == DISABLED_TARGETS
    assert safety_checks
    assert any(targets[-1] == 920 for targets in safety_checks)
    assert any(targets[-1] == 1000 for targets in safety_checks)


def test_generic_reset_accepts_installed_idle_q6_endpoint_without_numeric_motion():
    hand = FakeHand()
    hand.angles = [1000, 998, 954, 1000, 920, 979]
    driver, hand, _clock = make_reset_driver(
        hand,
        thumb_range=(900, 1000),
        q6_open_min_angle=975,
        angle_tolerance=25,
    )
    driver.adopt_disabled_state_and_verify()

    # The five bend axes are opened by the reviewed helper.  q6 then reports
    # its normal installed idle endpoint one unit lower than the initial read.
    # It is already within the profile's 1000 +/- 25 feedback band.
    hand.queue(
        angles=[1000, 998, 1000, 1000, 996, 978],
        statuses=[2] * 6,
    )
    writes_before = len(hand.log)

    assert driver.reset_to_open() == ()

    numeric_q6_writes = [
        item[2]
        for item in hand.log[writes_before:]
        if item[:2] == ("write", FakeAPI.REG_ANGLE_SET)
        and item[2][5] != -1
    ]
    assert numeric_q6_writes == []
    assert tuple(hand.angles) == (1000, 998, 1000, 1000, 996, 978)
    assert tuple(hand.angle_targets) == DISABLED_TARGETS


def test_generic_reset_recovers_small_q6_feedback_undershoot_without_out_of_range_command():
    hand = FakeHand()
    hand.angles = [1000, 1000, 1000, 1000, 1000, 896]
    driver, hand, _clock = make_reset_driver(
        hand,
        thumb_range=(900, 1000),
        q6_open_min_angle=975,
        q6_feedback_recovery_min_angle=885,
        angle_tolerance=25,
    )
    driver.adopt_disabled_state_and_verify()

    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 896],
        statuses=[2] * 6,
    )
    expected = ((946, 930), (971, 955), (996, 975), (1000, 975))
    for _target, actual in expected:
        for _ in range(3):
            hand.queue(
                angles=[1000, 1000, 1000, 1000, 1000, actual],
                statuses=[2] * 6,
            )

    writes_before = len(hand.log)
    returned = driver.reset_to_open(max_axis_current_ma=100)
    numeric = [
        item[2]
        for item in hand.log[writes_before:]
        if item[:2] == ("write", FakeAPI.REG_ANGLE_SET)
        and item[2] != DISABLED_TARGETS
    ]

    assert returned == tuple(target for target, _actual in expected)
    assert [values[5] for values in numeric] == [946, 971, 996, 1000]
    assert all(900 <= values[5] <= 1000 for values in numeric)
    assert tuple(hand.angle_targets) == DISABLED_TARGETS
    assert hand.angles[5] == 975


def test_generic_reset_recovers_observed_q6_880_after_policy_release():
    hand = FakeHand()
    hand.angles = [1000, 1000, 1000, 1000, 1000, 880]
    driver, hand, _clock = make_reset_driver(
        hand,
        thumb_range=(900, 1000),
        q6_open_min_angle=975,
        q6_feedback_recovery_min_angle=880,
        angle_tolerance=25,
    )
    driver.adopt_disabled_state_and_verify()

    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 880],
        statuses=[2] * 6,
    )
    expected = ((930, 915), (955, 940), (980, 965), (1000, 975))
    for _target, actual in expected:
        for _ in range(3):
            hand.queue(
                angles=[1000, 1000, 1000, 1000, 1000, actual],
                statuses=[2] * 6,
            )

    writes_before = len(hand.log)
    returned = driver.reset_to_open(max_axis_current_ma=100)
    numeric = [
        item[2]
        for item in hand.log[writes_before:]
        if item[:2] == ("write", FakeAPI.REG_ANGLE_SET)
        and item[2] != DISABLED_TARGETS
    ]

    assert returned == (930, 955, 980, 1000)
    assert [values[5] for values in numeric] == [930, 955, 980, 1000]
    assert all(900 <= values[5] <= 1000 for values in numeric)
    assert tuple(hand.angle_targets) == DISABLED_TARGETS
    assert hand.angles[5] == 975


def test_generic_reset_recovers_evidence_bound_q6_870_without_out_of_range_command():
    hand = FakeHand()
    hand.angles = [1000, 1000, 1000, 1000, 1000, 870]
    driver, hand, _clock = make_reset_driver(
        hand,
        thumb_range=(900, 1000),
        q6_open_min_angle=975,
        q6_feedback_recovery_min_angle=870,
        angle_tolerance=25,
    )
    driver.adopt_disabled_state_and_verify()

    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 870],
        statuses=[2] * 6,
    )
    expected = (
        (920, 895),
        (945, 920),
        (970, 945),
        (995, 970),
        (1000, 975),
    )
    for _target, actual in expected:
        for _ in range(3):
            hand.queue(
                angles=[1000, 1000, 1000, 1000, 1000, actual],
                statuses=[2] * 6,
            )

    writes_before = len(hand.log)
    returned = driver.reset_to_open(max_axis_current_ma=100)
    numeric = [
        item[2]
        for item in hand.log[writes_before:]
        if item[:2] == ("write", FakeAPI.REG_ANGLE_SET)
        and item[2] != DISABLED_TARGETS
    ]

    assert returned == (920, 945, 970, 995, 1000)
    assert [values[5] for values in numeric] == [920, 945, 970, 995, 1000]
    assert all(900 <= values[5] <= 1000 for values in numeric)
    assert tuple(hand.angle_targets) == DISABLED_TARGETS
    assert hand.angles[5] == 975


def test_generic_reset_rejects_q6_869_before_numeric_target_write():
    hand = FakeHand()
    hand.angles = [1000, 1000, 1000, 1000, 1000, 869]
    driver, hand, _clock = make_reset_driver(
        hand,
        thumb_range=(900, 1000),
        q6_open_min_angle=975,
        q6_feedback_recovery_min_angle=870,
        angle_tolerance=25,
    )
    driver.adopt_disabled_state_and_verify()
    hand.queue(angles=hand.angles, statuses=[2] * 6)
    writes_before = len(hand.log)

    with pytest.raises(
        RH56SequenceDriverError,
        match=r"bounded recovery range 870\.\.1000: 869",
    ):
        driver.reset_to_open(max_axis_current_ma=100)

    numeric = [
        item[2]
        for item in hand.log[writes_before:]
        if item[:2] == ("write", FakeAPI.REG_ANGLE_SET)
        and item[2] != DISABLED_TARGETS
    ]
    assert numeric == []
    assert tuple(hand.angle_targets) == DISABLED_TARGETS


@pytest.mark.parametrize("q6_open_min_angle", [969, 1001])
def test_generic_reset_rejects_unsafe_q6_open_feedback_band(q6_open_min_angle):
    with pytest.raises(ValueError, match="q6_open_min_angle"):
        make_reset_driver(
            thumb_range=(900, 1000),
            q6_open_min_angle=q6_open_min_angle,
        )


@pytest.mark.parametrize("recovery_min", [869, 901])
def test_generic_reset_rejects_overbroad_q6_feedback_recovery_band(recovery_min):
    with pytest.raises(ValueError, match="q6_feedback_recovery_min_angle"):
        make_reset_driver(
            thumb_range=(900, 1000),
            q6_open_min_angle=975,
            q6_feedback_recovery_min_angle=recovery_min,
        )


def test_generic_reset_open_refuses_deadband_backoff_outside_configured_range():
    hand = FakeHand()
    hand.angles = [1000, 1000, 1000, 1000, 1000, 973]
    driver, hand, _clock = make_reset_driver(hand, thumb_range=(940, 1000))
    driver.adopt_disabled_state_and_verify()
    hand.queue(angles=hand.angles, statuses=[2] * 6)
    writes_before = len(hand.log)

    with pytest.raises(
        RH56SequenceDriverError,
        match="cannot escape the q6 endpoint deadband inside configured range",
    ):
        driver.reset_to_open()

    numeric = [
        item[2]
        for item in hand.log[writes_before:]
        if item[:2] == ("write", FakeAPI.REG_ANGLE_SET)
        and item[2] != DISABLED_TARGETS
    ]
    assert numeric == []
    assert tuple(hand.angle_targets) == DISABLED_TARGETS


def test_generic_reset_open_backoff_rejects_current_before_accepting_sample():
    hand = FakeHand()
    hand.angles = [1000, 1000, 1000, 1000, 1000, 973]
    driver, hand, _clock = make_reset_driver(hand, thumb_range=(900, 1000))
    driver.adopt_disabled_state_and_verify()
    hand.queue(angles=hand.angles, statuses=[2] * 6)
    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 949],
        currents=[0, 0, 0, 0, 0, 401],
        statuses=[2] * 6,
    )

    with pytest.raises(RH56SequenceDriverError, match="strict cap 400mA"):
        driver.reset_to_open()

    assert tuple(hand.angle_targets) == DISABLED_TARGETS


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_axis_current_ma": True}, "max_axis_current_ma"),
        ({"max_axis_current_ma": 100.5}, "max_axis_current_ma"),
        ({"endpoint_stable_samples": 2.5}, "endpoint_stable_samples"),
        ({"max_inactive_drift_units": False}, "max_inactive_drift_units"),
    ],
)
def test_generic_reset_rejects_nonintegral_safety_limits_before_numeric_motion(
    kwargs, message
):
    driver, hand, _clock = make_reset_driver()
    driver.adopt_disabled_state_and_verify()
    writes_before = len(hand.log)

    with pytest.raises(ValueError, match=message):
        driver.reset_to_open(**kwargs)

    numeric_angle_writes = [
        item[2]
        for item in hand.log[writes_before:]
        if item[:2] == ("write", FakeAPI.REG_ANGLE_SET)
        and item[2] != DISABLED_TARGETS
    ]
    assert numeric_angle_writes == []
    assert not any(item[0] == "api.open_hand" for item in hand.log[writes_before:])
    assert tuple(hand.angle_targets) == DISABLED_TARGETS


def test_generic_reset_open_backoff_rejects_opposite_direction_and_contact():
    for angles, statuses, message in (
        ([1000, 1000, 1000, 1000, 1000, 990], [2] * 6, "opposite"),
        (
            [1000, 1000, 1000, 1000, 1000, 949],
            [2, 2, 2, 2, 2, 3],
            "force contact",
        ),
    ):
        hand = FakeHand()
        hand.angles = [1000, 1000, 1000, 1000, 1000, 973]
        driver, hand, _clock = make_reset_driver(hand, thumb_range=(900, 1000))
        driver.adopt_disabled_state_and_verify()
        hand.queue(angles=hand.angles, statuses=[2] * 6)
        hand.queue(angles=angles, statuses=statuses)

        with pytest.raises(RH56SequenceDriverError, match=message):
            driver.reset_to_open()

        assert tuple(hand.angle_targets) == DISABLED_TARGETS


def test_generic_reset_open_latches_disable_when_inactive_bend_moves():
    hand = FakeHand()
    hand.angles = [988, 1000, 997, 1000, 1000, 644]
    driver, hand, _clock = make_reset_driver(hand)
    driver.adopt_disabled_state_and_verify()
    hand.queue(angles=[1000, 1000, 1000, 1000, 1000, 644], statuses=[2] * 6)
    hand.queue(angles=[970, 1000, 1000, 1000, 1000, 660], statuses=[2] * 6)

    with pytest.raises(RH56SequenceDriverError, match="inactive bend axis moved"):
        driver.reset_to_open()

    assert tuple(hand.angle_targets) == DISABLED_TARGETS
    driver.disable_and_verify()
    assert tuple(hand.speeds) == DEFAULT_SPEEDS
    assert tuple(hand.force_limits) == DEFAULT_FORCES


def test_generic_reset_distinguishes_default_restore_failure_from_unconfirmed_stop():
    hand = FakeHand()
    hand.angles = [1000] * 6
    driver, hand, _clock = make_reset_driver(hand)
    driver.adopt_disabled_state_and_verify()
    hand.queue(angles=[1000] * 6, statuses=[2] * 6)
    driver.reset_to_open()
    hand.write_failures[FakeAPI.REG_SPEED_SET] = 1

    with pytest.raises(
        RH56ResetSettingsRestoreError, match="disable/idle is verified"
    ):
        driver.disable_and_verify()

    assert tuple(hand.angle_targets) == DISABLED_TARGETS
    assert driver._disabled_verified is True


def test_validated_feedback_observer_is_identity_bound_and_adds_no_reads():
    baseline, baseline_hand, _ = make_driver()
    baseline_hand.angles = [1000, 990, 980, 970, 960, 950]
    baseline_hand.statuses = [2, 3, 2, 3, 2, 2]
    baseline.read_state_snapshot()
    baseline_io = list(baseline_hand.log)

    driver, hand, _ = make_driver()
    hand.angles = list(baseline_hand.angles)
    hand.statuses = list(baseline_hand.statuses)
    observations = []
    driver.install_validated_feedback_observer(
        observations.append, identity="run-observer-001"
    )
    driver.read_state_snapshot()

    assert hand.log == baseline_io
    assert len(observations) == 1
    observation = observations[0]
    assert isinstance(observation, RH56ValidatedFeedbackObservation)
    assert observation.observer_identity == "run-observer-001"
    assert observation.phase == "boundary_state_snapshot"
    assert observation.angle_targets == DISABLED_TARGETS
    assert observation.angles == (1000, 990, 980, 970, 960, 950)
    assert observation.positions == (100, 100, 100, 100, 100, 100)
    assert observation.forces == (0, 0, 0, 0, 0, 0)
    assert observation.currents == (0, 0, 0, 0, 0, 0)
    assert observation.errors == (0, 0, 0, 0, 0, 0)
    assert observation.statuses == (2, 3, 2, 3, 2, 2)
    assert observation.temperatures == (30, 30, 30, 30, 30, 30)
    assert observation.timestamp_unix_ns > 0
    assert observation.timestamp_monotonic_ns > 0

    with pytest.raises(RH56SequenceDriverError, match="identity does not match"):
        driver.remove_validated_feedback_observer(identity="another-run")
    with pytest.raises(RH56SequenceDriverError, match="already installed"):
        driver.install_validated_feedback_observer(
            lambda _event: None, identity="replacement-run"
        )

    driver.remove_validated_feedback_observer(identity="run-observer-001")
    driver.read_state_snapshot()
    assert len(observations) == 1


def test_feedback_timestamp_is_register_read_complete_time_and_adds_no_io():
    baseline, baseline_hand, _ = make_driver()
    baseline.read_state_snapshot()
    baseline_io = list(baseline_hand.log)

    hand = FakeHand()
    clock = FakeClock()
    events = hand.log

    def external_gate():
        events.append(("external_gate",))

    def unix_clock():
        events.append(("clock_unix_ns",))
        return 1_234_567_890

    def monotonic_clock():
        events.append(("clock_monotonic_ns",))
        return 987_654_321

    observations = []

    def observer(observation):
        events.append(("observer",))
        observations.append(observation)

    driver = RH56SequenceDriver(
        hand,
        FakeAPI,
        motion_timeout_s=5.0,
        poll_interval_s=1.0,
        stop_verify_samples=3,
        stop_verify_interval_s=0.1,
        external_safety_check=external_gate,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        time_ns=unix_clock,
        monotonic_ns=monotonic_clock,
    )
    driver.install_validated_feedback_observer(observer, identity="capture-time")
    driver.read_state_snapshot()

    # Filtering out the passive gate/clock/observer log proves that timestamp
    # provenance did not add or reorder any RH56 register transaction.
    actual_io = [
        item
        for item in hand.log
        if item[0] in ("read", "read6", "write", "api.open_hand")
    ]
    assert actual_io == baseline_io
    last_payload_read = max(
        index
        for index, item in enumerate(events)
        if item[:2] == ("read", FakeAPI.REG_TEMP)
    )
    unix_index = events.index(("clock_unix_ns",))
    monotonic_index = events.index(("clock_monotonic_ns",))
    gate_indices = [
        index for index, item in enumerate(events) if item == ("external_gate",)
    ]
    observer_index = events.index(("observer",))
    assert last_payload_read < unix_index < monotonic_index < gate_indices[-1]
    assert gate_indices[-1] < observer_index
    assert len(observations) == 1
    assert observations[0].timestamp_unix_ns == 1_234_567_890
    assert observations[0].timestamp_monotonic_ns == 987_654_321


def test_feedback_without_observer_skips_unused_nanosecond_clocks():
    hand = FakeHand()
    clock = FakeClock()
    clock_calls = []

    def unused_unix_clock():
        clock_calls.append("unix")
        return 1

    def unused_monotonic_clock():
        clock_calls.append("monotonic")
        return 2

    driver = RH56SequenceDriver(
        hand,
        FakeAPI,
        motion_timeout_s=5.0,
        poll_interval_s=1.0,
        stop_verify_samples=3,
        stop_verify_interval_s=0.1,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        time_ns=unused_unix_clock,
        monotonic_ns=unused_monotonic_clock,
    )
    driver.read_state_snapshot()

    assert clock_calls == []


def test_observer_cannot_remove_itself_during_boundary_operation_and_latches_stop():
    driver, hand, _ = make_driver()

    def observer(_observation):
        driver.remove_validated_feedback_observer(identity="immutable-run")

    driver.install_validated_feedback_observer(observer, identity="immutable-run")
    writes_before = len([item for item in hand.log if item[0] == "write"])

    with pytest.raises(
        RH56ValidatedFeedbackObserverError,
        match="cannot be removed during an operation",
    ):
        driver.read_state_snapshot()

    assert driver.stop_requested
    assert len([item for item in hand.log if item[0] == "write"]) == writes_before


def test_observer_reentry_cannot_trigger_an_extra_register_read():
    baseline, baseline_hand, _ = make_driver()
    baseline.read_state_snapshot()
    expected_reads = [item for item in baseline_hand.log if item[0].startswith("read")]

    driver, hand, _ = make_driver()

    def observer(_observation):
        driver.read_state_snapshot()

    driver.install_validated_feedback_observer(observer, identity="passive-run")
    with pytest.raises(
        RH56ValidatedFeedbackObserverError,
        match="cannot enter a driver operation",
    ):
        driver.read_state_snapshot()

    actual_reads = [item for item in hand.log if item[0].startswith("read")]
    assert actual_reads == expected_reads
    assert driver.stop_requested


def test_unloaded_observer_failure_uses_existing_fail_and_disable_path():
    driver, hand, _ = make_driver()
    observed_phases = []

    def observer(observation):
        observed_phases.append(observation.phase)
        if observation.phase == "open_verify":
            raise RuntimeError("publisher unavailable")

    driver.install_validated_feedback_observer(observer, identity="open-run")

    with pytest.raises(
        RH56ValidatedFeedbackObserverError, match="publisher unavailable"
    ):
        open_driver(driver)

    assert observed_phases.count("adopt_disable_verify") == 3
    assert observed_phases[-1] == "open_verify"
    assert driver.stop_requested
    assert tuple(hand.angle_targets) == DISABLED_TARGETS
    assert driver.last_emergency_disable_errors == ()


def test_loaded_hold_observer_failure_latches_but_preserves_numeric_hold():
    driver, hand, _ = make_driver()
    open_driver(driver)
    preshape(driver, hand, 900)
    target = (700, 710, 720, 730, 800, 900)
    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 900], statuses=[2] * 6
    )
    hand.queue(
        angles=[820, 830, 720, 850, 810, 900],
        statuses=[3, 3, 2, 3, 2, 2],
    )
    driver.close_bends_and_hold(target)
    writes_before = len([item for item in hand.log if item[0] == "write"])

    def observer(_observation):
        raise RuntimeError("viewer stopped")

    driver.install_validated_feedback_observer(observer, identity="loaded-run")
    with pytest.raises(RH56ValidatedFeedbackObserverError, match="viewer stopped"):
        driver.verify_loaded_hold(target, minimum_contact_axes=2)

    assert driver.stop_requested
    assert driver.numeric_hold_targets == target
    assert tuple(hand.angle_targets) == target
    assert len([item for item in hand.log if item[0] == "write"]) == writes_before


def test_rejected_feedback_is_never_delivered_to_observer():
    driver, hand, _ = make_driver()
    phases = []
    driver.install_validated_feedback_observer(
        lambda observation: phases.append(observation.phase),
        identity="reject-run",
    )
    open_driver(driver)
    target = (975, 975, 975, 975, 975, 975)
    bind_audited_path(driver, target)
    preshape(driver, hand, 975)
    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 975], statuses=[2] * 6
    )
    hand.queue(
        angles=[975, 1000, 1000, 1000, 1000, 975],
        statuses=[3, 2, 2, 2, 2, 2],
    )

    with pytest.raises(RH56SequenceDriverError, match="force contact"):
        driver.close_bends_no_contact_and_hold(target)

    assert "coupled_air_close_preflight" in phases
    assert "coupled_air_close_step_0000" not in phases
    assert driver.telemetry[-1].phase == "coupled_air_close_step_0000"
    assert driver.stop_requested


def test_disable_samples_are_observed_only_after_whole_idle_set_is_stable():
    driver, hand, _ = make_driver()
    phases = []
    driver.install_validated_feedback_observer(
        lambda observation: phases.append(observation.phase),
        identity="disable-reject-run",
    )
    for index in range(20):
        hand.queue(
            angles=[
                996 + 4 * (index % 2),
                1000,
                1000,
                1000,
                1000,
                1000,
            ],
            statuses=[2] * 6,
        )

    with pytest.raises(RH56StopUnconfirmed, match="still moving"):
        driver.disable_and_verify()

    assert phases == []
    assert driver.stop_requested
    assert tuple(hand.angle_targets) == DISABLED_TARGETS


def test_observer_covers_formal_no_contact_close_and_reverse_paths():
    driver, hand, _ = make_driver()
    observations = []
    driver.install_validated_feedback_observer(
        observations.append, identity="air-path-run"
    )
    open_driver(driver)
    target = (975, 975, 975, 975, 975, 975)
    bind_audited_path(driver, target)
    preshape(driver, hand, 975)

    hand.queue(
        angles=[1000, 1000, 1000, 1000, 1000, 975], statuses=[2] * 6
    )
    forward_bends = coupled_waypoints(target[:5], target[5], 25)
    for waypoint in forward_bends:
        for _ in range(3):
            hand.queue(angles=waypoint, statuses=[2] * 6)
    driver.close_bends_no_contact_and_hold(target)

    closed = tuple(target)
    for _ in range(3):
        hand.queue(angles=closed, statuses=[2] * 6)
    reverse_bends = tuple(
        item.command_targets
        for item in build_rh56_no_contact_execution_path(target).waypoints
        if item.phase.startswith("bend_reverse_")
    )
    for waypoint in reverse_bends:
        for _ in range(3):
            hand.queue(angles=waypoint, statuses=[2] * 6)
    for _ in range(4):
        hand.queue(angles=reverse_bends[-1], statuses=[2] * 6)
    for _ in range(3):
        hand.queue(angles=[1000] * 6, statuses=[2] * 6)
    for _ in range(4):
        hand.queue(angles=[1000] * 6, statuses=[2] * 6)

    driver.return_no_contact_hand_to_open()
    phases = [observation.phase for observation in observations]

    assert "open_verify" in phases
    assert "thumb_preflight" in phases
    assert "thumb_preshape_0975" in phases
    assert "coupled_air_close_preflight" in phases
    assert any(phase.startswith("coupled_air_close_step_") for phase in phases)
    assert "q6_return_adopt_verify" in phases
    assert any(phase.startswith("coupled_air_return_step_") for phase in phases)
    assert "bend_return_disable_verify" in phases
    assert "q6_return_preflight" in phases
    assert "q6_return_1000" in phases
    assert "q6_return_disable_verify" in phases
    assert "q6_return_open_verify" in phases
    assert all(
        observation.observer_identity == "air-path-run"
        for observation in observations
    )
    assert tuple(hand.angle_targets) == DISABLED_TARGETS
