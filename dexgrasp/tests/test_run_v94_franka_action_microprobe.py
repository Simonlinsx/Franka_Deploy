from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/fr3_rh56_v7_commissioning.json"
BUNDLE = ROOT.parent / "data/test_fixtures/sim2real/deploy.zip"


def _load(name="test_run_v94_franka_action_microprobe_app"):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "apps/run_v94_franka_action_microprobe.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _snapshot(**changes):
    value = {
        "hand_id": 1,
        "angles": (1000, 1000, 997, 1000, 1000, 985),
        "angle_targets": (-1,) * 6,
        "currents": (0,) * 6,
        "errors": (0,) * 6,
        "statuses": (2,) * 6,
        "speeds": (1000,) * 6,
        "force_limits": (500,) * 6,
    }
    value.update(changes)
    return value


def _hardware(
    app, *, behavior="success", stop_fails=False, initial_q_override=None
):
    calls = []
    snapshots = [_snapshot(), _snapshot(), _snapshot(), _snapshot()]
    initial_q = np.asarray(
        [0.0, -0.569, 0.0, -2.81, 0.0, 3.037, 0.741]
        if initial_q_override is None
        else initial_q_override,
        dtype=np.float64,
    )

    class Serial:
        def __init__(self, *args):
            calls.append(("serial-init", args))

        def __enter__(self):
            calls.append(("serial-enter",))
            return self

        def __exit__(self, *_args):
            calls.append(("serial-exit",))

    class Hand:
        def __init__(self, serial, hand_id):
            calls.append(("hand", serial, hand_id))

        def snapshot(self):
            calls.append(("snapshot",))
            return snapshots.pop(0)

    class Limits:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            for key, value in kwargs.items():
                setattr(self, key, value)

    class Arm:
        def __init__(self, limits):
            self.limits = limits
            self.robot = self
            self.q = initial_q.copy()
            self.dq = np.zeros(7)
            self.last_control_loop_telemetry = None
            self.move_count = 0

        def read_once(self):
            return SimpleNamespace(q=self.q.copy(), dq=self.dq.copy())

        def _validate_state(self, _state, **kwargs):
            calls.append(("validate", kwargs))
            return 1.0

        def move_joints(self, target):
            wanted = np.asarray(target, dtype=np.float64).copy()
            calls.append(("move", wanted))
            self.move_count += 1
            if behavior == "interrupt" and self.move_count == 1:
                raise KeyboardInterrupt()
            if behavior == "swallowed" and self.move_count == 1:
                # Deliberately report a handle while leaving q at the origin;
                # the strict fresh-q proof must still reject it.
                self.last_control_loop_telemetry = SimpleNamespace(
                    kind="joint",
                    samples=200,
                    max_read_to_write_us=20.0,
                    read_to_write_overruns=0,
                )
                return self.q.copy()
            self.q = wanted
            if behavior == "recorded-residual" and self.move_count == 1:
                self.q[0] -= 0.000101378
            if behavior == "over-tolerance" and self.move_count == 1:
                self.q[0] -= app.MICROPROBE_ARRIVAL_TOLERANCE_RAD + 1.0e-8
            self.last_control_loop_telemetry = SimpleNamespace(
                kind="joint",
                samples=100 + self.move_count,
                max_read_to_write_us=20.0 + self.move_count,
                read_to_write_overruns=0,
            )
            return self.q.copy()

        def stop(self):
            calls.append(("stop",))
            if stop_fails:
                raise RuntimeError("synthetic stop proof failure")

    class ArmType:
        @classmethod
        def connect(cls, ip, limits, enforce_realtime):
            calls.append(("connect", ip, enforce_realtime, limits))
            return Arm(limits)

    types = {
        "rh56_api": SimpleNamespace(LinuxSerial=Serial, RH56Hand=Hand),
        "FrankaMotionLimits": Limits,
        "FrankaSequenceDriver": ArmType,
    }
    return types, calls, snapshots, initial_q


def _config(app):
    return app.load_control_config(CONFIG)[0]


def test_exact_mapper_uses_positive_j1_and_changes_no_other_arm_axis():
    app = _load("test_v94_j1_probe_mapper")
    contract = app.V94Contract.from_bundle(app.DeployBundle(BUNDLE))
    q = np.asarray([0.0, -0.569, 0.0, -2.81, 0.0, 3.037, 0.741])
    mapped = app._map_live_j1_target(q, contract=contract)

    delta = mapped.peak_q_rad - mapped.origin_q_rad
    assert mapped.action13[0] == 1.0
    assert np.all(mapped.action13[1:7] == 0.0)
    assert np.all(mapped.action13[7:] == -1.0)
    assert mapped.target_delta_rad == pytest.approx(0.003, abs=2.0e-6)
    assert mapped.target_delta_rad > 0.0
    assert np.max(np.abs(delta[1:])) <= 2.0e-6
    assert mapped.reset_linf_error_rad <= app.MICROPROBE_RESET_ENVELOPE_RAD


def test_mapper_rejects_fresh_q_outside_v94_reset_envelope():
    app = _load("test_v94_j1_probe_reset_envelope")
    contract = app.V94Contract.from_bundle(app.DeployBundle(BUNDLE))
    q = contract.q_home_rad.astype(np.float64)
    q[3] += app.MICROPROBE_RESET_ENVELOPE_RAD + 1.0e-3

    with pytest.raises(RuntimeError, match="outside the conservative V94 reset"):
        app._map_live_j1_target(q, contract=contract)


def test_mapper_requires_tight_qhome_alignment_before_probe():
    app = _load("test_v94_j1_probe_start_alignment")
    contract = app.V94Contract.from_bundle(app.DeployBundle(BUNDLE))
    q = contract.q_home_rad.astype(np.float64)
    q[0] += app.MICROPROBE_START_ALIGNMENT_TOLERANCE_RAD + 1.0e-6

    with pytest.raises(RuntimeError, match="run --recover-to-v94-home"):
        app._map_live_j1_target(q, contract=contract)


def test_confirmed_fake_run_sends_microstep_and_returns_to_exact_qhome():
    app = _load("test_v94_j1_probe_success")
    types, calls, snapshots, initial_q = _hardware(app)
    result = app.run_microprobe(
        _config(app),
        bundle_path=BUNDLE,
        run_id="probe-test-001",
        hardware_types=types,
        network_preflight=lambda ip: calls.append(("network", ip)),
        sleep=lambda _seconds: None,
    )

    moves = [item[1] for item in calls if item[0] == "move"]
    assert len(moves) == 2
    assert moves[0][0] - initial_q.astype(np.float32)[0] == pytest.approx(
        0.003, abs=2.0e-6
    )
    assert np.max(np.abs(moves[0][1:] - initial_q.astype(np.float32)[1:])) <= 2.0e-6
    contract = app.V94Contract.from_bundle(app.DeployBundle(BUNDLE))
    assert np.array_equal(moves[1], contract.q_home_rad.astype(np.float64))
    assert result.outbound_measured_delta_rad == pytest.approx(0.003, abs=2.0e-4)
    assert result.outbound_target_error_rad <= 2.0e-4
    assert result.return_target_error_rad <= 2.0e-4
    assert result.outbound_control_samples > 1
    assert result.return_control_samples > 1
    assert result.franka_stop_verified
    assert result.rh56_final_disabled_verified
    assert result.rh56_command_writes == 0
    assert result.evidence["action13"] == [1.0] + [0.0] * 6 + [-1.0] * 6
    assert len(result.evidence["outbound_proof"]["q_samples_rad"]) == 3
    assert len(result.evidence["return_proof"]["dq_samples_rad_s"]) == 3
    assert result.evidence["franka_stop"]["verified"] is True
    assert result.evidence["rh56_final_disabled_verified"] is True
    assert result.evidence["return_to_v94_home_proven"] is True
    assert [item[0] for item in calls].count("snapshot") == 4
    assert [item[0] for item in calls].count("stop") == 1
    assert not snapshots

    limits = next(item[3] for item in calls if item[0] == "connect")
    assert limits.kwargs["joint_arrival_tolerance_rad"] == pytest.approx(2.0e-4)
    assert (
        limits.kwargs["joint_arrival_tolerance_rad"]
        / app.MICROPROBE_NOMINAL_DELTA_RAD
        <= 1.0 / 15.0 + 1.0e-12
    )
    assert limits.kwargs["joint_arrival_tolerance_rad"] < 0.001
    assert limits.kwargs["max_joint_speed_rad_s"] <= 0.05
    assert limits.kwargs["max_joint_segment_rad"] == pytest.approx(0.004)


def test_swallowed_microstep_fails_closed_even_when_fake_reports_control_samples():
    app = _load("test_v94_j1_probe_swallowed")
    types, calls, _snapshots, _initial_q = _hardware(app, behavior="swallowed")

    with pytest.raises(RuntimeError, match="outbound target error"):
        app.run_microprobe(
            _config(app),
            bundle_path=BUNDLE,
            run_id="probe-test-002",
            hardware_types=types,
            network_preflight=lambda _ip: None,
            sleep=lambda _seconds: None,
        )
    assert [item[0] for item in calls].count("move") == 1
    assert [item[0] for item in calls].count("stop") == 1
    assert [item[0] for item in calls].count("snapshot") == 4


def test_recorded_101urad_residual_is_accepted_then_exact_home_is_proven():
    app = _load("test_v94_j1_probe_recorded_residual")
    types, calls, _snapshots, _initial_q = _hardware(
        app, behavior="recorded-residual"
    )

    result = app.run_microprobe(
        _config(app),
        bundle_path=BUNDLE,
        run_id="probe-test-residual",
        hardware_types=types,
        network_preflight=lambda _ip: None,
        sleep=lambda _seconds: None,
    )

    assert result.outbound_target_error_rad == pytest.approx(0.000101378)
    assert result.outbound_target_error_rad < app.MICROPROBE_ARRIVAL_TOLERANCE_RAD
    assert result.evidence["return_to_v94_home_proven"] is True
    assert [item[0] for item in calls].count("move") == 2


def test_just_over_200urad_fails_with_partial_and_terminal_evidence():
    app = _load("test_v94_j1_probe_over_tolerance")
    types, calls, _snapshots, _initial_q = _hardware(
        app, behavior="over-tolerance"
    )

    with pytest.raises(RuntimeError, match="outbound target error") as caught:
        app.run_microprobe(
            _config(app),
            bundle_path=BUNDLE,
            run_id="probe-test-boundary",
            hardware_types=types,
            network_preflight=lambda _ip: None,
            sleep=lambda _seconds: None,
        )

    evidence = caught.value.microprobe_evidence
    assert evidence["failure_phase"] == "outbound_proof"
    assert evidence["outbound_command_completed"] is True
    assert evidence["outbound_telemetry"]["samples"] > 1
    assert len(evidence["outbound_proof"]["q_samples_rad"]) == 1
    assert evidence["return_command_started"] is False
    assert evidence["return_to_v94_home_proven"] is False
    assert evidence["franka_stop"]["verified"] is True
    assert evidence["rh56_final_disabled_verified"] is True
    assert [item[0] for item in calls].count("move") == 1


def test_keyboard_interrupt_still_runs_stop_and_final_hand_proof():
    app = _load("test_v94_j1_probe_interrupt")
    types, calls, _snapshots, _initial_q = _hardware(app, behavior="interrupt")

    with pytest.raises(KeyboardInterrupt):
        app.run_microprobe(
            _config(app),
            bundle_path=BUNDLE,
            run_id="probe-test-003",
            hardware_types=types,
            network_preflight=lambda _ip: None,
            sleep=lambda _seconds: None,
        )
    assert [item[0] for item in calls].count("stop") == 1
    assert [item[0] for item in calls].count("snapshot") == 4


def test_stop_verification_failure_is_never_hidden_by_original_error():
    app = _load("test_v94_j1_probe_stop_failure")
    types, _calls, _snapshots, _initial_q = _hardware(
        app, behavior="interrupt", stop_fails=True
    )

    with pytest.raises(app.MicroprobeStopUnconfirmed, match="STOP UNCONFIRMED"):
        app.run_microprobe(
            _config(app),
            bundle_path=BUNDLE,
            run_id="probe-test-004",
            hardware_types=types,
            network_preflight=lambda _ip: None,
            sleep=lambda _seconds: None,
        )


def test_recovery_moves_only_to_exact_qhome_and_proves_terminal_state():
    app = _load("test_v94_j1_recovery_success")
    contract = app.V94Contract.from_bundle(app.DeployBundle(BUNDLE))
    displaced = contract.q_home_rad.astype(np.float64)
    displaced[0] += 0.0029
    types, calls, _snapshots, _initial_q = _hardware(
        app, initial_q_override=displaced
    )

    result = app.run_recovery_to_v94_home(
        _config(app),
        bundle_path=BUNDLE,
        run_id="recovery-test-001",
        hardware_types=types,
        network_preflight=lambda _ip: None,
        sleep=lambda _seconds: None,
    )

    moves = [item[1] for item in calls if item[0] == "move"]
    assert len(moves) == 1
    assert np.array_equal(moves[0], contract.q_home_rad.astype(np.float64))
    assert result.initial_home_error_rad == pytest.approx(0.0029)
    assert result.final_home_error_rad <= app.MICROPROBE_ARRIVAL_TOLERANCE_RAD
    assert result.evidence["return_to_v94_home_proven"] is True
    assert result.evidence["franka_stop"]["verified"] is True
    assert result.evidence["rh56_final_disabled_verified"] is True
    assert result.rh56_command_writes == 0


def test_recovery_refuses_motion_outside_safe_envelope_but_still_stops():
    app = _load("test_v94_j1_recovery_envelope")
    contract = app.V94Contract.from_bundle(app.DeployBundle(BUNDLE))
    displaced = contract.q_home_rad.astype(np.float64)
    displaced[0] += app.MICROPROBE_RESET_ENVELOPE_RAD + 1.0e-3
    types, calls, _snapshots, _initial_q = _hardware(
        app, initial_q_override=displaced
    )

    with pytest.raises(RuntimeError, match="recovery refused before motion") as caught:
        app.run_recovery_to_v94_home(
            _config(app),
            bundle_path=BUNDLE,
            run_id="recovery-test-002",
            hardware_types=types,
            network_preflight=lambda _ip: None,
            sleep=lambda _seconds: None,
        )

    evidence = caught.value.microprobe_evidence
    assert evidence["recovery_command_started"] is False
    assert evidence["return_to_v94_home_proven"] is False
    assert evidence["franka_stop"]["verified"] is True
    assert not [item for item in calls if item[0] == "move"]


def test_recovery_stop_failure_is_explicit_and_evidence_is_json_safe():
    app = _load("test_v94_j1_recovery_stop_failure")
    contract = app.V94Contract.from_bundle(app.DeployBundle(BUNDLE))
    displaced = contract.q_home_rad.astype(np.float64)
    displaced[0] += 0.0029
    types, _calls, _snapshots, _initial_q = _hardware(
        app, initial_q_override=displaced, stop_fails=True
    )

    with pytest.raises(app.MicroprobeStopUnconfirmed) as caught:
        app.run_recovery_to_v94_home(
            _config(app),
            bundle_path=BUNDLE,
            run_id="recovery-test-stop",
            hardware_types=types,
            network_preflight=lambda _ip: None,
            sleep=lambda _seconds: None,
        )

    evidence = caught.value.microprobe_evidence
    assert evidence["return_to_v94_home_proven"] is True
    assert evidence["franka_stop"]["attempted"] is True
    assert evidence["franka_stop"]["verified"] is False
    assert evidence["rh56_final_disabled_verified"] is True
    import json

    json.dumps(evidence, allow_nan=False)


def test_execute_requires_run_scoped_authorization_and_all_exact_tokens():
    app = _load("test_v94_j1_probe_tokens")
    parser = app.build_parser()
    missing = parser.parse_args(["--execute"])
    with pytest.raises(ValueError, match="run-id"):
        app._require_execute_authorization(missing)

    run_id = "probe-test-005"
    complete = parser.parse_args(
        [
            "--execute",
            "--run-id",
            run_id,
            "--authorization-token",
            app.AUTHORIZATION_PREFIX + run_id,
            "--confirm-installed",
            app.INSTALLED_TOKEN,
            "--confirm-hand-open",
            app.HAND_OPEN_TOKEN,
            "--confirm-workspace-clear",
            app.WORKSPACE_TOKEN,
            "--confirm-stop-ready",
            app.STOP_TOKEN,
            "--confirm-pla-low-speed",
            app.PLA_TOKEN,
            "--confirm-mapping",
            app.MAPPING_TOKEN,
            "--confirm-c1-scope",
            app.C1_SCOPE_TOKEN,
        ]
    )
    assert app._require_execute_authorization(complete) == run_id


def test_recovery_requires_distinct_run_scoped_authorization():
    app = _load("test_v94_j1_recovery_tokens")
    parser = app.build_parser()
    run_id = "recovery-test-003"
    complete = parser.parse_args(
        [
            "--recover-to-v94-home",
            "--run-id",
            run_id,
            "--authorization-token",
            app.RECOVERY_AUTHORIZATION_PREFIX + run_id,
            "--confirm-installed",
            app.INSTALLED_TOKEN,
            "--confirm-hand-open",
            app.HAND_OPEN_TOKEN,
            "--confirm-workspace-clear",
            app.RECOVERY_WORKSPACE_TOKEN,
            "--confirm-stop-ready",
            app.STOP_TOKEN,
            "--confirm-pla-low-speed",
            app.PLA_TOKEN,
            "--confirm-recovery-target",
            app.RECOVERY_TARGET_TOKEN,
        ]
    )
    assert app._require_recovery_authorization(complete) == run_id


def test_default_cli_is_dry_run_and_never_imports_hardware(monkeypatch, capsys):
    app = _load("test_v94_j1_probe_dry_run")

    def forbidden():
        raise AssertionError("dry-run imported hardware")

    monkeypatch.setattr(app, "_load_hardware_types", forbidden)
    assert app.main(["--config", str(CONFIG), "--bundle", str(BUNDLE)]) == 0
    output = capsys.readouterr().out
    assert "[DRY RUN]" in output
    assert "No hardware driver was imported" in output
    assert "effective_gain=" in output


def test_evidence_file_is_exclusive_and_cannot_be_overwritten(tmp_path):
    app = _load("test_v94_j1_probe_evidence_exclusive")
    output = tmp_path / "probe.json"
    writer = app._ExclusiveEvidence(output)
    writer.write({"result": "PASS", "value": 1})
    writer.close()

    assert output.stat().st_mode & 0o777 == 0o600
    assert output.read_text(encoding="utf-8") == '{"result": "PASS", "value": 1}\n'
    with pytest.raises(FileExistsError):
        app._ExclusiveEvidence(output)


def test_existing_run_artifact_refuses_before_hardware_and_is_not_overwritten(
    monkeypatch, tmp_path
):
    app = _load("test_v94_j1_probe_existing_artifact")
    output = tmp_path / "existing.json"
    output.write_text("immutable-old-evidence\n", encoding="utf-8")
    run_id = "probe-existing-run"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("existing evidence path reached hardware")

    monkeypatch.setattr(app, "run_microprobe", forbidden)
    rc = app.main(
        [
            "--execute",
            "--run-id",
            run_id,
            "--output",
            str(output),
            "--authorization-token",
            app.AUTHORIZATION_PREFIX + run_id,
            "--confirm-installed",
            app.INSTALLED_TOKEN,
            "--confirm-hand-open",
            app.HAND_OPEN_TOKEN,
            "--confirm-workspace-clear",
            app.WORKSPACE_TOKEN,
            "--confirm-stop-ready",
            app.STOP_TOKEN,
            "--confirm-pla-low-speed",
            app.PLA_TOKEN,
            "--confirm-mapping",
            app.MAPPING_TOKEN,
            "--confirm-c1-scope",
            app.C1_SCOPE_TOKEN,
        ]
    )
    assert rc == 1
    assert output.read_text(encoding="utf-8") == "immutable-old-evidence\n"


def test_main_failure_json_preserves_partial_cleanup_and_position_status(
    monkeypatch, tmp_path
):
    app = _load("test_v94_j1_probe_main_failure_evidence")
    output = tmp_path / "failed-probe.json"
    run_id = "probe-test-json"
    partial = {
        "failure_phase": "outbound_proof",
        "action13": [1.0] + [0.0] * 6 + [-1.0] * 6,
        "outbound_proof": {"q_samples_rad": [[0.0] * 7]},
        "outbound_telemetry": {"samples": 100},
        "return_command_started": False,
        "return_to_v94_home_proven": False,
        "franka_stop": {"verified": True},
        "rh56_final_disabled_verified": True,
    }

    def fail_without_hardware(*_args, **_kwargs):
        exc = RuntimeError("synthetic outbound proof failure")
        raise app._attach_microprobe_evidence(exc, partial)

    monkeypatch.setattr(app, "run_microprobe", fail_without_hardware)
    rc = app.main(
        [
            "--execute",
            "--run-id",
            run_id,
            "--output",
            str(output),
            "--authorization-token",
            app.AUTHORIZATION_PREFIX + run_id,
            "--confirm-installed",
            app.INSTALLED_TOKEN,
            "--confirm-hand-open",
            app.HAND_OPEN_TOKEN,
            "--confirm-workspace-clear",
            app.WORKSPACE_TOKEN,
            "--confirm-stop-ready",
            app.STOP_TOKEN,
            "--confirm-pla-low-speed",
            app.PLA_TOKEN,
            "--confirm-mapping",
            app.MAPPING_TOKEN,
            "--confirm-c1-scope",
            app.C1_SCOPE_TOKEN,
        ]
    )
    assert rc == 1
    import json

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["result"] == "FAIL"
    assert payload["terminal_proof_available"] is True
    assert payload["return_to_v94_home_proven"] is False
    assert payload["position_statement"] == (
        "unknown_after_verified_stop_do_not_assume_returned"
    )
    assert payload["partial_session"] == partial
