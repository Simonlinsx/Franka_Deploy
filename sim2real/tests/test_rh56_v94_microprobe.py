from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sim2real.commissioning.rh56_v94_microprobe import (
    DISABLED_TARGETS,
    FrankaReadOnlyStationaryGate,
    MicroprobeError,
    OPEN_TARGETS,
    ProfileBinding,
    _verify_probe_telemetry,
    build_parser,
    derive_v94_mapper_probe_endpoint,
    load_v94_profile,
    main,
    run_hardware_session,
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


@dataclass(frozen=True)
class _Telemetry:
    phase: str
    elapsed_s: float
    angle_targets: tuple[int, ...]
    angles: tuple[int, ...]
    positions: tuple[int, ...]
    forces: tuple[int, ...]
    currents: tuple[int, ...]
    errors: tuple[int, ...]
    statuses: tuple[int, ...]
    temperatures: tuple[int, ...]


def _sample(phase, target, actual, status):
    return _Telemetry(
        phase=phase,
        elapsed_s=0.1,
        angle_targets=DISABLED_TARGETS[:5] + (target,),
        angles=OPEN_TARGETS[:5] + (actual,),
        positions=(0,) * 6,
        forces=(0,) * 6,
        currents=(0, 0, 0, 0, 0, 43),
        errors=(0,) * 6,
        statuses=(2, 2, 2, 2, 2, status),
        temperatures=(25,) * 6,
    )


class _FakeDriver:
    def __init__(self, *, fail_forward=False, fail_recovery=False):
        self.calls = []
        self.telemetry = []
        self.fail_forward = fail_forward
        self.fail_recovery = fail_recovery
        self.external_gate = None

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

    def commission_thumb_sweep(self, target, **kwargs):
        self.calls.append(("forward", target, kwargs))
        assert self.external_gate is not None
        self.external_gate()
        if self.fail_forward:
            raise RuntimeError("scripted forward failure")
        self.telemetry.extend(
            [
                _sample("q6_step_0960", 960, 978, 1),
                _sample("q6_step_0960", 960, 962, 2),
                _sample("q6_step_0960", 960, 960, 2),
            ]
        )
        return (960,)

    def reset_to_open(self, **kwargs):
        self.calls.append(("reset_to_open", kwargs))
        assert self.external_gate is not None
        self.external_gate()
        if self.fail_recovery:
            raise RuntimeError("scripted recovery failure")
        self.telemetry.extend(
            [
                _sample("reset_open_q6_backoff_0910", 910, 930, 1),
                _sample("reset_open_q6_backoff_0910", 910, 910, 2),
                _sample("reset_open_q6_0001_1000", 1000, 985, 2),
                _Telemetry(
                    phase="reset_open_final",
                    elapsed_s=0.5,
                    angle_targets=DISABLED_TARGETS,
                    angles=(1000, 1000, 1000, 1000, 1000, 985),
                    positions=(0,) * 6,
                    forces=(0,) * 6,
                    currents=(0,) * 6,
                    errors=(0,) * 6,
                    statuses=(2,) * 6,
                    temperatures=(25,) * 6,
                ),
            ]
        )
        return (910, 1000)

    def disable_and_verify(self):
        self.calls.append(("disable_and_verify",))

    def close(self):
        self.calls.append(("close",))


class _FakeRobot:
    def __init__(self, q, dq=None):
        self.q = tuple(q)
        self.dq = (0.0,) * 7 if dq is None else tuple(dq)
        self.reads = 0

    def read_once(self):
        self.reads += 1
        return SimpleNamespace(
            q=self.q,
            dq=self.dq,
            robot_mode="Idle",
        )


def _validator(_state, *, require_idle, enforce_success):
    assert require_idle is True
    assert enforce_success is False


def test_default_cli_is_dry_run_and_never_calls_hardware(capsys):
    calls = []
    result = main(
        ["--config", str(PROFILE)],
        run_id_factory=lambda: "dry-run-id",
        hardware_session=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    assert result == 0
    assert calls == []
    output = capsys.readouterr().out
    assert '"mode": "DRY_RUN_DISARMED"' in output
    assert "no device was opened" in output


def test_cli_has_no_force_bypass_or_noninteractive_yes_option():
    destinations = {action.dest for action in build_parser()._actions}
    assert "execute" in destinations
    assert "force" not in destinations
    assert "bypass" not in destinations
    assert "yes" not in destinations
    assert "run_id" not in destinations


def test_execute_requires_tty_and_exact_run_scoped_phrase(tmp_path, capsys):
    calls = []
    common = [
        "--config",
        str(PROFILE),
        "--execute",
        "--output",
        str(tmp_path / "evidence.json"),
    ]
    result = main(
        common,
        stdin_isatty=lambda: False,
        run_id_factory=lambda: "run-one",
        hardware_session=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    assert result == 1
    assert calls == []
    assert "interactive TTY" in capsys.readouterr().err

    result = main(
        common,
        stdin_isatty=lambda: True,
        input_fn=lambda _prompt: "wrong phrase",
        run_id_factory=lambda: "run-two",
        hardware_session=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    assert result == 1
    assert calls == []
    assert "confirmation did not match" in capsys.readouterr().err


def test_profile_validation_rejects_unvalidated_thumb_range(tmp_path):
    payload = json.loads(PROFILE.read_text(encoding="utf-8"))
    payload["inspire"]["thumb_rotate_validated_realtime_range"] = [975, 1000]
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MicroprobeError, match="outside the validated"):
        load_v94_profile(changed)


def test_production_mapper_derives_exact_axis5_endpoint_without_cross_coupling():
    derivation = derive_v94_mapper_probe_endpoint()
    assert derivation["policy_action13"] == pytest.approx(
        [0.0] * 7 + [-0.6] + [-1.0] * 5
    )
    assert derivation["selected_tick"] == 1
    assert derivation["selected_register_vector_manufacturer_order"] == [
        1000,
        1000,
        1000,
        1000,
        1000,
        960,
    ]
    assert "masks manufacturer axes 0..4" in derivation["important_scope"]


def test_franka_gate_rejects_motion_and_reset_pose_error():
    expected = (0.0,) * 7
    moving = _FakeRobot(expected, dq=(0.021,) + (0.0,) * 6)
    with pytest.raises(MicroprobeError, match="not stationary"):
        FrankaReadOnlyStationaryGate(moving, _validator, expected).check()

    outside = _FakeRobot((0.051,) + (0.0,) * 6)
    with pytest.raises(MicroprobeError, match="reset envelope"):
        FrankaReadOnlyStationaryGate(outside, _validator, expected).check()


def test_fake_success_uses_exact_single_axis_path_and_cleanup():
    binding = load_v94_profile(PROFILE)
    q = tuple(float(value) for value in binding.payload["franka"]["default_q_rad"])
    robot = _FakeRobot(q)
    driver = _FakeDriver()
    evidence = run_hardware_session(
        binding,
        "fake-success",
        hand_connector=lambda _profile: driver,
        franka_connector=lambda _profile: (robot, _validator),
        sleep=lambda _seconds: None,
    )
    assert evidence["result"] == "PASS"
    assert evidence["mapping_proof"]["physical_direction_confirmed"] is True
    assert evidence["scope"]["policy_tick_command_path"] == [1000, 960]
    assert evidence["mapper_derivation"][
        "selected_register_vector_manufacturer_order"
    ] == [1000, 1000, 1000, 1000, 1000, 960]
    forward = next(call for call in driver.calls if call[0] == "forward")
    assert forward[1] == 960
    assert forward[2] == {
        "step_units": 40,
        "max_axis_current_ma": 100,
        "endpoint_stable_samples": 3,
        "max_inactive_drift_units": 8,
    }
    recovery = next(call for call in driver.calls if call[0] == "reset_to_open")
    assert recovery[1]["max_axis_current_ma"] == 100
    assert evidence["recovery_proof"]["policy_tick"] is False
    assert evidence["recovery_proof"]["deadband_escape_backoff_used"] is True
    assert driver.calls[0] == ("adopt_disabled",)
    assert ("disable_and_verify",) in driver.calls
    assert driver.calls[-1] == ("close",)
    assert evidence["cleanup"]["temporary_speed_force_settings_restored"] is True
    assert evidence["franka_read_only"]["check_count"] >= 5


def test_fake_runtime_failure_still_disables_and_refuses_pass():
    binding = load_v94_profile(PROFILE)
    q = tuple(float(value) for value in binding.payload["franka"]["default_q_rad"])
    driver = _FakeDriver(fail_forward=True)
    evidence = run_hardware_session(
        binding,
        "fake-failure",
        hand_connector=lambda _profile: driver,
        franka_connector=lambda _profile: (_FakeRobot(q), _validator),
        sleep=lambda _seconds: None,
    )
    assert evidence["result"] == "FAIL"
    assert "scripted forward failure" in evidence["operation_error"]
    assert ("disable_and_verify",) in driver.calls
    assert driver.calls[-1] == ("close",)


def test_recovery_failure_keeps_outbound_mapping_proof_but_cannot_pass():
    binding = load_v94_profile(PROFILE)
    q = tuple(float(value) for value in binding.payload["franka"]["default_q_rad"])
    driver = _FakeDriver(fail_recovery=True)
    evidence = run_hardware_session(
        binding,
        "fake-recovery-failure",
        hand_connector=lambda _profile: driver,
        franka_connector=lambda _profile: (_FakeRobot(q), _validator),
        sleep=lambda _seconds: None,
    )
    assert evidence["result"] == "FAIL"
    assert evidence["mapping_proof"]["physical_direction_confirmed"] is True
    assert evidence["recovery_proof"] is None
    assert "scripted recovery failure" in evidence["operation_error"]
    assert evidence["cleanup"]["double_disable_and_stationary_stop_verified"] is True


def test_telemetry_verifier_rejects_inactive_axis_drift():
    samples = [
        _sample("q6_step_0960", 960, 978, 1),
        _sample("q6_step_0960", 960, 962, 2),
        _sample("q6_step_0960", 960, 960, 2),
    ]
    values = [dict(item.__dict__) for item in samples]
    values[1]["angles"] = (990, 1000, 1000, 1000, 1000, 962)
    with pytest.raises(MicroprobeError, match="inactive RH56 axis"):
        _verify_probe_telemetry(values, OPEN_TARGETS)
