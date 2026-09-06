from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from sim2real.commissioning.rh56_v94_all_axis_microprobe import (
    EXPECTED_POLICY_ORDER,
    MicroprobeError,
    PROBE_ENDPOINT,
    PROBE_PHASE_PREFIX,
    V94MaskedAxisProbeDriver,
    _verify_axis_probe,
    build_parser,
    derive_all_axis_mapper_contract,
    main,
    run_hardware_session,
)
from sim2real.commissioning.rh56_v94_microprobe import (
    DISABLED_TARGETS,
    OPEN_TARGETS,
)

# Reuse the mature register-level fake already used to exercise the production
# RH56 sequence driver.  Importing the module above first installs dexgrasp/src
# on sys.path; neither import performs device discovery or hardware I/O.
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


def _row(axis: int, actual: int, status: int = 2):
    targets = [-1] * 6
    targets[axis] = PROBE_ENDPOINT
    angles = list(OPEN_TARGETS)
    angles[axis] = int(actual)
    currents = [0] * 6
    currents[axis] = 40
    statuses = [2] * 6
    statuses[axis] = int(status)
    return {
        "phase": f"{PROBE_PHASE_PREFIX}_m{axis}_{PROBE_ENDPOINT:04d}",
        "elapsed_s": 0.1,
        "angle_targets": tuple(targets),
        "angles": tuple(angles),
        "positions": (0,) * 6,
        "forces": (0,) * 6,
        "currents": tuple(currents),
        "errors": (0,) * 6,
        "statuses": tuple(statuses),
        "temperatures": (25,) * 6,
    }


def _recovery_row():
    row = _snapshot()
    row.update({"phase": "reset_open_final", "elapsed_s": 0.5})
    return row


def _bend_recovery_row(axis: int):
    row = _snapshot()
    targets = [-1] * 6
    targets[axis] = 1000
    row.update(
        {
            "phase": f"reset_open_bend_m{axis}_1000",
            "elapsed_s": 0.2,
            "angle_targets": tuple(targets),
        }
    )
    return row


class _FakeRobot:
    def __init__(self, q):
        self.q = tuple(q)
        self.reads = 0

    def read_once(self):
        self.reads += 1
        return SimpleNamespace(q=self.q, dq=(0.0,) * 7, robot_mode="Idle")


def _validator(_state, *, require_idle, enforce_success):
    assert require_idle is True
    assert enforce_success is False


class _FakeAllAxisDriver:
    def __init__(self, *, fail_axis=None):
        self.calls = []
        self.telemetry = []
        self.external_gate = None
        self.selected_axis = None
        self.fail_axis = fail_axis

    def request_stop(self):
        self.calls.append(("request_stop",))

    def adopt_disabled_state_and_verify(self):
        self.calls.append(("adopt_disabled",))

    def read_state_snapshot(self):
        self.calls.append(("snapshot",))
        if self.external_gate is not None:
            self.external_gate()
        return _snapshot()

    def install_external_safety_check(self, callback):
        self.calls.append(("install_franka_gate",))
        self.external_gate = callback

    def commission_masked_axis_960(self, axis, **kwargs):
        axis = int(axis)
        self.calls.append(("probe", axis, kwargs))
        assert self.external_gate is not None
        self.external_gate()
        if axis == self.fail_axis:
            raise RuntimeError(f"scripted failure on manufacturer axis {axis}")
        self.selected_axis = axis
        self.telemetry.extend(
            [
                _row(axis, 980, 1),
                _row(axis, 960),
                _row(axis, 960),
                _row(axis, 960),
            ]
        )
        target = [-1] * 6
        target[axis] = PROBE_ENDPOINT
        return tuple(target)

    def reset_to_open(self, **kwargs):
        self.calls.append(("reset_to_open", self.selected_axis, kwargs))
        assert self.external_gate is not None
        self.external_gate()
        for bend_axis in range(5):
            self.telemetry.extend(
                _bend_recovery_row(bend_axis) for _ in range(3)
            )
        self.telemetry.append(_recovery_row())
        if self.selected_axis == 5:
            return (920, 1000)
        return ()

    def disable_and_verify(self):
        self.calls.append(("disable_and_verify",))

    def close(self):
        self.calls.append(("close",))


def test_mapper_contract_is_exact_reverse_permutation_and_single_register():
    contract = derive_all_axis_mapper_contract()
    assert tuple(item["policy_axis"] for item in contract["axes"]) == (
        EXPECTED_POLICY_ORDER
    )
    assert [
        item["manufacturer_register_index_zero_based"]
        for item in contract["axes"]
    ] == [5, 4, 3, 2, 1, 0]
    for policy_axis, item in enumerate(contract["axes"]):
        assert item["policy_action_index"] == 7 + policy_axis
        assert item["policy_action13"][7 + policy_axis] == pytest.approx(-0.6)
        target = item["physical_masked_angle_set"]
        assert target.count(PROBE_ENDPOINT) == 1
        assert target.count(-1) == 5
        assert target[item["manufacturer_register_index_zero_based"]] == 960


def test_default_cli_is_dry_run_and_has_no_bypass(capsys):
    calls = []
    result = main(
        ["--config", str(PROFILE)],
        run_id_factory=lambda: "dry-run-all6",
        hardware_session=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    assert result == 0
    assert calls == []
    assert "no device was opened" in capsys.readouterr().out
    destinations = {action.dest for action in build_parser()._actions}
    assert "execute" in destinations
    assert not {"force", "bypass", "yes", "run_id"} & destinations


def test_execute_requires_tty_and_run_scoped_exact_phrase(tmp_path, capsys):
    calls = []
    common = [
        "--config",
        str(PROFILE),
        "--execute",
        "--output",
        str(tmp_path / "evidence.json"),
    ]
    assert (
        main(
            common,
            stdin_isatty=lambda: False,
            run_id_factory=lambda: "all6-no-tty",
            hardware_session=lambda *args, **kwargs: calls.append((args, kwargs)),
        )
        == 1
    )
    assert "interactive TTY" in capsys.readouterr().err
    assert (
        main(
            common,
            stdin_isatty=lambda: True,
            input_fn=lambda _prompt: "wrong phrase",
            run_id_factory=lambda: "all6-wrong-phrase",
            hardware_session=lambda *args, **kwargs: calls.append((args, kwargs)),
        )
        == 1
    )
    assert calls == []
    assert "confirmation did not match" in capsys.readouterr().err


def test_full_fake_session_proves_all_six_and_keeps_franka_read_only():
    from sim2real.commissioning.rh56_v94_microprobe import load_v94_profile

    binding = load_v94_profile(PROFILE)
    q = binding.payload["franka"]["default_q_rad"]
    robot = _FakeRobot(q)
    driver = _FakeAllAxisDriver()
    evidence = run_hardware_session(
        binding,
        "fake-all6-success",
        hand_connector=lambda _profile: driver,
        franka_connector=lambda _profile: (robot, _validator),
        sleep=lambda _seconds: None,
    )
    assert evidence["result"] == "PASS"
    assert evidence["eligible_as_full_six_axis_physical_mapping_evidence"] is True
    assert evidence["authorizes_closed_loop"] is False
    assert evidence["urdf_12d_independent_measurement_claimed"] is False
    assert len(evidence["axis_records"]) == 6
    assert [
        record["probe_proof"]["manufacturer_register_index_zero_based"]
        for record in evidence["axis_records"]
    ] == [5, 4, 3, 2, 1, 0]
    for record in evidence["axis_records"]:
        physical = record["probe_proof"]["physical_masked_angle_set"]
        assert physical.count(960) == 1
        assert physical.count(-1) == 5
        assert record["recovery_proof"]["policy_tick"] is False
        assert record["recovery_proof"][
            "bend_open_helper_exact_masked_transcript_confirmed"
        ] is True
    probe_calls = [call for call in driver.calls if call[0] == "probe"]
    reset_calls = [call for call in driver.calls if call[0] == "reset_to_open"]
    assert len(probe_calls) == len(reset_calls) == 6
    assert all(call[2]["max_axis_current_ma"] == 100 for call in probe_calls)
    assert all(call[2]["max_axis_current_ma"] == 100 for call in reset_calls)
    assert evidence["franka_read_only"]["connection"].startswith("read_once_only")
    assert evidence["franka_read_only"]["check_count"] >= 15
    assert driver.calls[0] == ("adopt_disabled",)
    assert ("disable_and_verify",) in driver.calls
    assert driver.calls[-1] == ("close",)


def test_fake_mid_sequence_failure_fails_closed_and_preserves_partial_evidence():
    from sim2real.commissioning.rh56_v94_microprobe import load_v94_profile

    binding = load_v94_profile(PROFILE)
    q = binding.payload["franka"]["default_q_rad"]
    driver = _FakeAllAxisDriver(fail_axis=3)
    evidence = run_hardware_session(
        binding,
        "fake-all6-failure",
        hand_connector=lambda _profile: driver,
        franka_connector=lambda _profile: (_FakeRobot(q), _validator),
        sleep=lambda _seconds: None,
    )
    assert evidence["result"] == "FAIL"
    assert evidence["eligible_as_full_six_axis_physical_mapping_evidence"] is False
    assert "scripted failure on manufacturer axis 3" in evidence["operation_error"]
    assert 0 < len(evidence["axis_records"]) < 6
    assert ("disable_and_verify",) in driver.calls
    assert driver.calls[-1] == ("close",)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("target", "masked readback"),
        ("inactive_drift", "inactive-axis drift"),
        ("current", "exceeded 100mA"),
        ("wrong_direction", "direction/endpoint"),
        ("fault", "fault/temperature"),
    ],
)
def test_evidence_verifier_rejects_unsafe_feedback(mutation, message):
    derivation = derive_all_axis_mapper_contract()["axes"][2]
    axis = derivation["manufacturer_register_index_zero_based"]
    rows = [
        _row(axis, 980, 1),
        _row(axis, 960),
        _row(axis, 960),
        _row(axis, 960),
    ]
    if mutation == "target":
        rows[1]["angle_targets"] = (960, -1, -1, 960, -1, -1)
    elif mutation == "inactive_drift":
        angles = list(rows[1]["angles"])
        angles[0] = 990
        rows[1]["angles"] = tuple(angles)
    elif mutation == "current":
        currents = list(rows[1]["currents"])
        currents[axis] = 101
        rows[1]["currents"] = tuple(currents)
    elif mutation == "wrong_direction":
        angles = list(rows[-1]["angles"])
        angles[axis] = 1000
        rows[-1]["angles"] = tuple(angles)
    elif mutation == "fault":
        errors = list(rows[1]["errors"])
        errors[axis] = 1
        rows[1]["errors"] = tuple(errors)
    with pytest.raises(MicroprobeError, match=message):
        _verify_axis_probe(rows, derivation=derivation, initial_angles=OPEN_TARGETS)


@pytest.mark.parametrize("axis", range(6))
def test_register_level_driver_writes_only_selected_numeric_angle_target(axis):
    clock = FakeClock()
    hand = FakeHand()
    driver = V94MaskedAxisProbeDriver(
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
    hand.queue(angles=[1000] * 6, statuses=[2] * 6)
    endpoint = [1000] * 6
    endpoint[axis] = PROBE_ENDPOINT
    for _ in range(3):
        hand.queue(angles=endpoint, statuses=[2] * 6, currents=[0] * 6)

    writes_before = len(hand.log)
    returned = driver.commission_masked_axis_960(axis)
    expected = tuple(PROBE_ENDPOINT if index == axis else -1 for index in range(6))
    assert returned == expected
    angle_writes = [
        item[2]
        for item in hand.log[writes_before:]
        if item[:2] == ("write", FakeAPI.REG_ANGLE_SET)
    ]
    numeric_writes = [item for item in angle_writes if item != DISABLED_TARGETS]
    assert numeric_writes == [expected]
    assert expected.count(PROBE_ENDPOINT) == 1
    assert expected.count(-1) == 5

    driver.disable_and_verify()
    assert tuple(hand.angle_targets) == DISABLED_TARGETS


def test_register_level_overcurrent_fault_latches_all_six_disable():
    clock = FakeClock()
    hand = FakeHand()
    driver = V94MaskedAxisProbeDriver(
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
    hand.queue(angles=[1000] * 6, statuses=[2] * 6)
    currents = [0] * 6
    currents[2] = 101
    hand.queue(angles=[1000, 1000, 980, 1000, 1000, 1000], currents=currents)

    with pytest.raises(Exception, match="current"):
        driver.commission_masked_axis_960(2)

    assert tuple(hand.angle_targets) == DISABLED_TARGETS
    assert driver.stop_requested
