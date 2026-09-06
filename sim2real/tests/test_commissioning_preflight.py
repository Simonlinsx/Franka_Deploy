from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import sim2real.deployment.preflight as preflight


@dataclass(frozen=True)
class _Verification:
    bundle_contract: str = "v94"
    bundle_status: str = preflight.EXPECTED_BUNDLE_STATUS
    hardware_writes: bool = False


class _Contract:
    bundle_status = preflight.EXPECTED_BUNDLE_STATUS
    q_home_rad = np.asarray([0.0, 0.1, 0.2, -1.5, 0.0, 1.5, 0.7])
    maximum_nominal_arm_velocity_rad_s = 0.18


class _ContractFactory:
    @classmethod
    def from_bundle(cls, _bundle):
        return _Contract()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shadow_report(*, replay=True, reset=True, writes=False, result="PASS"):
    return {
        "result": result,
        "failed_checks": [] if result == "PASS" else ["synthetic_failure"],
        "checks": [
            {
                "name": "offline_audit_and_policy_replay_pass",
                "passed": replay,
            },
            {"name": "recorded_reset_aligned", "passed": reset},
            {"name": "hardware_writes_false", "passed": not writes},
            {"name": "robot_command_writes_false", "passed": not writes},
        ],
    }


def _profile():
    return {
        "schema_version": 1,
        "mode": "c2_bounded_closed_loop_commissioned",
        "franka": {
            "default_q_rad": _Contract.q_home_rad.tolist(),
            "default_max_joint_velocity_rad_s": 0.18,
            "online_max_joint_acceleration_rad_s2": [0.5] * 7,
            "online_max_joint_jerk_rad_s3": [2.0] * 7,
            "online_max_tracking_error_rad": [0.02] * 7,
            "default_path_collision_verified": True,
        },
        "inspire": {
            "six_axis_coupled_closure_commissioned": True,
            "thumb_rotate_validated_realtime_range": [0, 1000],
        },
        "tool": {
            "installed_collision_model_verified": True,
            "low_speed_unloaded_commissioning_only": False,
        },
    }


def _config(bundle: Path, profile: Path):
    execution = {name: True for name in preflight.REQUIRED_EXECUTION_FLAGS}
    execution.update(hardware_writes_enabled=False, full_dr_stress_accepted=False)
    return {
        "schema_version": 1,
        "bundle": str(bundle),
        "commissioning_profile": str(profile),
        "execution": execution,
    }


def _write_json(path: Path, value) -> Path:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _install_offline_fakes(monkeypatch, shadow_report=None):
    monkeypatch.setattr(preflight, "verify_v94_bundle", lambda _path: _Verification())
    monkeypatch.setattr(preflight, "DeployBundle", lambda _path: object())
    monkeypatch.setattr(preflight, "V94Contract", _ContractFactory)
    report = _shadow_report() if shadow_report is None else shadow_report
    monkeypatch.setattr(
        preflight,
        "assess_v94_shadow_readiness",
        lambda _shadow, bundle_path: report,
    )


def _case(tmp_path: Path, monkeypatch):
    _install_offline_fakes(monkeypatch)
    bundle = tmp_path / "deploy.zip"
    bundle.write_bytes(b"offline-test-bundle")
    shadow = tmp_path / "shadow.npz"
    shadow.write_bytes(b"offline-test-shadow")
    profile = _write_json(tmp_path / "profile.json", _profile())
    config = _write_json(tmp_path / "config.json", _config(bundle, profile))
    artifact = _write_json(
        tmp_path / "commissioning-report.json",
        {"result": "PASS", "scope": "synthetic-hardware-free-test"},
    )
    evidence = {
        "schema_version": preflight.EVIDENCE_SCHEMA_VERSION,
        "kind": preflight.EVIDENCE_KIND,
        "run_id": "commissioning-run-001",
        "bindings": {
            "shadow_npz_sha256": _sha(shadow),
            "bundle_sha256": _sha(bundle),
            "deploy_config_sha256": _sha(config),
            "commissioning_profile_sha256": _sha(profile),
        },
        "items": {
            name: {
                "result": "PASS",
                "artifact": artifact.name,
                "sha256": _sha(artifact),
                "reviewed_by": "commissioning-reviewer",
            }
            for name, _category in preflight.REQUIRED_EVIDENCE_ITEMS
        },
        "metrics": {
            "franka_control_loop_rate_hz": 1000.0,
            "rh56_sustained_command_rate_hz": 60.0,
            "dual_device_sustained_ack_rate_hz": 60.0,
            "dual_device_max_interaction_gap_s": 1.0 / 30.0,
            "policy_command_watchdog_timeout_s": 0.050,
        },
    }
    evidence_path = _write_json(tmp_path / "evidence.json", evidence)
    return shadow, config, profile, evidence_path, evidence


def _by_code(report, code):
    return next(item for item in report["checks"] if item["code"] == code)


def test_complete_evidence_only_yields_disarmed_authorization_eligibility(
    tmp_path, monkeypatch
):
    shadow, config, _profile_path, evidence, _payload = _case(tmp_path, monkeypatch)
    report = preflight.assess_v94_commissioning_preflight(
        shadow, config_path=config, evidence_path=evidence
    )

    assert report["result"] == "PASS"
    assert report["eligible_for_operator_authorization"] is True
    assert report["arming_state"] == "DISARMED"
    assert report["physical_motion_authorized"] is False
    assert report["motion_authorization_created"] is False
    assert report["future_authorization_scope_required"] == preflight.AUTHORIZATION_SCOPE
    assert report["device_access"] is False
    assert report["hardware_writes"] is False
    assert report["robot_command_writes"] is False
    assert report["c3_task_or_production_readiness_claimed"] is False
    assert report["failed_checks"] == []


def test_missing_evidence_is_explicit_fail_with_named_hardware_blockers(
    tmp_path, monkeypatch
):
    shadow, config, _profile_path, _evidence, _payload = _case(tmp_path, monkeypatch)
    report = preflight.assess_v94_commissioning_preflight(
        shadow, config_path=config
    )

    assert report["result"] == "FAIL"
    assert report["eligible_for_operator_authorization"] is False
    assert _by_code(report, "commissioning_evidence_manifest_present")["passed"] is False
    for code in (
        "evidence_physical_deadman_acceptance",
        "evidence_physical_emergency_stop_acceptance",
        "evidence_franka_persistent_fci_1khz_session",
        "evidence_rh56_60hz_write_readback",
        "evidence_installed_fr3_adapter_rh56_collision_model",
    ):
        assert _by_code(report, code)["passed"] is False


def test_failed_or_unreviewed_physical_interlock_artifact_fails_closed(
    tmp_path, monkeypatch
):
    shadow, config, _profile_path, evidence_path, evidence = _case(
        tmp_path, monkeypatch
    )
    evidence["items"]["physical_deadman_acceptance"]["result"] = "FAIL"
    evidence["items"]["physical_emergency_stop_acceptance"]["reviewed_by"] = ""
    _write_json(evidence_path, evidence)

    # Rebind the evidence to its unchanged inputs; only the item evidence fails.
    report = preflight.assess_v94_commissioning_preflight(
        shadow, config_path=config, evidence_path=evidence_path
    )
    assert report["result"] == "FAIL"
    assert _by_code(report, "evidence_physical_deadman_acceptance")["passed"] is False
    assert _by_code(report, "evidence_physical_emergency_stop_acceptance")["passed"] is False
    assert report["physical_motion_authorized"] is False


def test_artifact_hash_and_exact_input_binding_are_mandatory(tmp_path, monkeypatch):
    shadow, config, _profile_path, evidence_path, evidence = _case(
        tmp_path, monkeypatch
    )
    evidence["bindings"]["shadow_npz_sha256"] = "0" * 64
    evidence["items"]["rh56_fault_disable_and_verified_stop"]["sha256"] = "f" * 64
    _write_json(evidence_path, evidence)
    report = preflight.assess_v94_commissioning_preflight(
        shadow, config_path=config, evidence_path=evidence_path
    )

    assert report["result"] == "FAIL"
    assert _by_code(report, "evidence_binding_shadow_npz_sha256")["passed"] is False
    assert _by_code(report, "evidence_rh56_fault_disable_and_verified_stop")[
        "passed"
    ] is False


@pytest.mark.parametrize(
    ("metric", "value"),
    [
        ("franka_control_loop_rate_hz", 999.9),
        ("rh56_sustained_command_rate_hz", 59.9),
        ("dual_device_sustained_ack_rate_hz", 59.9),
        ("dual_device_max_interaction_gap_s", 0.034),
        ("policy_command_watchdog_timeout_s", 0.051),
        ("rh56_sustained_command_rate_hz", True),
        ("franka_control_loop_rate_hz", None),
    ],
)
def test_commissioning_metrics_cannot_be_omitted_or_weakened(
    tmp_path, monkeypatch, metric, value
):
    shadow, config, _profile_path, evidence_path, evidence = _case(
        tmp_path, monkeypatch
    )
    evidence["metrics"][metric] = value
    _write_json(evidence_path, evidence)
    report = preflight.assess_v94_commissioning_preflight(
        shadow, config_path=config, evidence_path=evidence_path
    )

    assert report["result"] == "FAIL"
    assert _by_code(report, f"evidence_metric_{metric}")["passed"] is False


def test_current_style_locked_profile_velocity_and_thumb_range_are_blockers(
    tmp_path, monkeypatch
):
    shadow, config, profile_path, evidence_path, _evidence = _case(
        tmp_path, monkeypatch
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["mode"] = "commissioning_locked"
    profile["franka"]["default_max_joint_velocity_rad_s"] = 0.05
    profile["franka"]["default_q_rad"][0] += 0.2
    profile["franka"].pop("online_max_joint_jerk_rad_s3")
    profile["inspire"]["thumb_rotate_validated_realtime_range"] = [900, 1000]
    _write_json(profile_path, profile)

    # The profile edit also invalidates the manifest binding, as intended.
    report = preflight.assess_v94_commissioning_preflight(
        shadow, config_path=config, evidence_path=evidence_path
    )
    for code in (
        "profile_mode_c2_commissioned",
        "commissioning_profile_reset_matches_training",
        "policy_rate_within_commissioned_franka_envelope",
        "franka_online_max_joint_jerk_rad_s3",
        "rh56_full_thumb_rotation_profile",
        "evidence_binding_commissioning_profile_sha256",
    ):
        assert _by_code(report, code)["passed"] is False


def test_shadow_replay_reset_and_write_claims_remain_independent_gates(
    tmp_path, monkeypatch
):
    bad_shadow = _shadow_report(
        replay=False, reset=False, writes=True, result="FAIL"
    )
    _install_offline_fakes(monkeypatch, bad_shadow)
    bundle = tmp_path / "deploy.zip"
    bundle.write_bytes(b"bundle")
    shadow = tmp_path / "shadow.npz"
    shadow.write_bytes(b"shadow")
    profile = _write_json(tmp_path / "profile.json", _profile())
    config = _write_json(tmp_path / "config.json", _config(bundle, profile))

    report = preflight.assess_v94_commissioning_preflight(
        shadow, config_path=config
    )
    assert _by_code(report, "observation_shadow_readiness")["passed"] is False
    assert _by_code(report, "shadow_action_replay")["passed"] is False
    assert _by_code(report, "shadow_reset_alignment")["passed"] is False
    assert _by_code(report, "shadow_contains_no_robot_writes")["passed"] is False


def test_cli_always_emits_machine_readable_disarmed_failure(
    tmp_path, monkeypatch, capsys
):
    missing = tmp_path / "missing-shadow.npz"
    exit_code = preflight.main([str(missing)])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["result"] == "FAIL"
    assert payload["arming_state"] == "DISARMED"
    assert payload["physical_motion_authorized"] is False
    assert payload["hardware_writes"] is False
    assert payload["blockers"][0]["code"] == "preflight_input_or_verification_error"


def test_cli_exposes_no_execute_force_or_threshold_override():
    parser = preflight.build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--execute" not in option_strings
    assert "--force" not in option_strings
    assert "--minimum-action-rate-hz" not in option_strings
    assert "--maximum-camera-age-s" not in option_strings


def test_offline_preflight_rejects_hardware_write_enable_even_with_evidence(
    tmp_path, monkeypatch
):
    shadow, config_path, _profile_path, evidence_path, _evidence = _case(
        tmp_path, monkeypatch
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["execution"]["hardware_writes_enabled"] = True
    _write_json(config_path, config)
    report = preflight.assess_v94_commissioning_preflight(
        shadow, config_path=config_path, evidence_path=evidence_path
    )

    assert report["result"] == "FAIL"
    assert _by_code(report, "offline_preflight_hardware_writes_disabled")[
        "passed"
    ] is False
    assert report["hardware_writes"] is False
    assert report["physical_motion_authorized"] is False


def test_duplicate_json_keys_are_rejected(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version": 1, "schema_version": 1}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        preflight._load_json(duplicate, label="test")
