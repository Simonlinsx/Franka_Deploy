from __future__ import annotations

import hashlib
import json
from pathlib import Path
import signal

import numpy as np
import pytest

from sim2real.runtime.bounded_c2_orchestrator import (
    BoundedC2Admission,
    BoundedC2AdmissionError,
    BoundedC2Orchestrator,
    BoundedC2RunError,
    BoundedC2State,
    BoundedC2StopProof,
    CLI_STATIC_BLOCKERS,
    build_disarmed_cli_report,
    build_parser,
    seal_c2_admission,
)
from sim2real.closed_loop_core import (
    AUTHORIZATION_SCOPE,
    ClosedLoopSafetyGate,
    MotionAuthorization,
    SafetyState,
)


class _Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


class _FakeRuntime:
    physical_interlocks_configured = True
    independent_command_watchdogs_configured = True
    verified_dual_device_stop_supported = True

    def __init__(
        self,
        *,
        completed_steps: int = 3,
        raise_sigint: bool = False,
        stop_proof: BoundedC2StopProof | None = None,
    ) -> None:
        self.completed_steps = int(completed_steps)
        self.raise_sigint = bool(raise_sigint)
        self.stop_proof = stop_proof or BoundedC2StopProof(True, True)
        self.run_calls = []
        self.stop_reasons = []
        self.stop_verify_calls = 0

    def run(
        self,
        *,
        maximum_policy_steps,
        hard_deadline_monotonic_s,
        stop_requested,
    ):
        self.run_calls.append(
            (
                int(maximum_policy_steps),
                float(hard_deadline_monotonic_s),
                stop_requested,
            )
        )
        if self.raise_sigint:
            signal.raise_signal(signal.SIGINT)
            assert stop_requested.is_set()
        return {"completed_policy_steps": self.completed_steps}

    def request_stop(self, reason):
        self.stop_reasons.append(str(reason))

    def stop_and_verify(self):
        self.stop_verify_calls += 1
        return self.stop_proof


def _profile_payload():
    return {
        "schema_version": 1,
        "franka": {
            "joint_limits_rad": [[-2.0, 2.0]] * 7,
            "joint_limit_margin_rad": 0.1,
            "default_max_joint_velocity_rad_s": 0.2,
            "online_max_joint_acceleration_rad_s2": 100.0,
            "online_max_joint_jerk_rad_s3": 100000.0,
            "online_max_tracking_error_rad": 0.1,
            "expected_F_T_EE": np.eye(4).tolist(),
            "expected_end_effector": {
                "mass_kg": 0.6,
                "F_x_Cee_m": [0.0, 0.0, 0.05],
                "inertia_kg_m2": np.diag([0.01, 0.01, 0.005]).tolist(),
            },
            "persistent_session": {
                "initial_target_tolerance_rad": 0.01,
                "target_reached_tolerance_rad": 0.005,
                "control_period_min_s": 0.0005,
                "control_period_max_s": 0.002,
                "policy_command_max_age_s": 0.05,
                "read_to_write_deadline_s": 0.0004,
                "minimum_control_success_rate": 0.9,
                "maximum_session_duration_s": 1.0,
                "stop_maximum_velocity_rad_s": 0.01,
                "stop_consecutive_samples": 2,
                "stop_maximum_samples": 3,
                "pose_ring_capacity": 8,
                "allow_one_initial_zero_period": True,
                "F_T_EE_tolerance": 1.0e-8,
                "mass_tolerance_kg": 1.0e-5,
                "center_of_mass_tolerance_m": 1.0e-6,
                "inertia_tolerance_kg_m2": 1.0e-6,
                "expected_external_load": {
                    "mass_kg": 0.0,
                    "F_x_Cload_m": [0.0, 0.0, 0.0],
                    "inertia_kg_m2": np.zeros((3, 3)).tolist(),
                },
            },
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_REQUIRED_CHECKS = (
    "evidence_installed_payload_dynamics",
    "evidence_installed_fr3_adapter_rh56_collision_model",
    "evidence_physical_deadman_acceptance",
    "evidence_physical_emergency_stop_acceptance",
    "evidence_franka_persistent_fci_1khz_session",
    "evidence_franka_single_owner_state_source",
    "evidence_franka_online_velocity_acceleration_jerk_limits",
    "evidence_franka_command_watchdog_and_fault_stop",
    "evidence_franka_tracking_and_collision_monitoring",
    "evidence_rh56_single_owner_serial_session",
    "evidence_rh56_full_six_axis_and_fingertip_fk",
    "evidence_rh56_full_thumb_rotation_range",
    "evidence_rh56_60hz_write_readback",
    "evidence_rh56_fault_disable_and_verified_stop",
    "evidence_dual_device_same_sequence_ack_transaction",
    "evidence_dual_device_rate_and_max_gap",
    "evidence_metric_franka_control_loop_rate_hz",
    "evidence_metric_rh56_sustained_command_rate_hz",
    "evidence_metric_dual_device_sustained_ack_rate_hz",
    "evidence_metric_dual_device_max_interaction_gap_s",
    "evidence_metric_policy_command_watchdog_timeout_s",
)


def _passing_report(tmp_path: Path):
    workspace = Path(__file__).resolve().parents[2]
    bundle = workspace / "data/test_fixtures/sim2real/deploy.zip"
    profile = tmp_path / "profile.json"
    shadow = tmp_path / "shadow.npz"
    config = tmp_path / "deploy.json"
    profile.write_text(json.dumps(_profile_payload()), encoding="utf-8")
    shadow.write_bytes(b"synthetic bound shadow")
    config.write_text('{"schema_version":1}', encoding="utf-8")
    metrics = {
        "evidence_metric_franka_control_loop_rate_hz": 1000.0,
        "evidence_metric_rh56_sustained_command_rate_hz": 60.0,
        "evidence_metric_dual_device_sustained_ack_rate_hz": 60.0,
        "evidence_metric_dual_device_max_interaction_gap_s": 1.0 / 30.0,
        "evidence_metric_policy_command_watchdog_timeout_s": 0.05,
    }
    report = {
        "result": "PASS",
        "readiness_level": "C2_BOUNDED_CLOSED_LOOP",
        "offline_only": True,
        "device_access": False,
        "hardware_writes": False,
        "robot_command_writes": False,
        "arming_state": "DISARMED",
        "motion_authorization_created": False,
        "physical_motion_authorized": False,
        "future_authorization_scope_required": AUTHORIZATION_SCOPE,
        "eligible_for_operator_authorization": True,
        "run_id": "commissioning-evidence-run",
        "failed_checks": [],
        "blockers": [],
        "inputs": {
            "shadow_npz": str(shadow),
            "bundle": str(bundle),
            "deploy_config": str(config),
            "commissioning_profile": str(profile),
            "sha256": {
                "shadow_npz_sha256": _sha256(shadow),
                "bundle_sha256": _sha256(bundle),
                "deploy_config_sha256": _sha256(config),
                "commissioning_profile_sha256": _sha256(profile),
            },
        },
        "checks": [
            {
                "code": code,
                "passed": True,
                **({"actual": metrics[code]} if code in metrics else {}),
            }
            for code in _REQUIRED_CHECKS
        ],
    }
    return report, shadow


def _authorization():
    return MotionAuthorization(
        run_id="execution-run",
        authorization_id="explicit-run-scoped-approval",
        issued_monotonic_s=9.0,
        expires_monotonic_s=20.0,
    )


def _armed_gate(authorization):
    gate = ClosedLoopSafetyGate()
    gate.update_interlocks(deadman_asserted=True, estop_healthy=True)
    gate.arm(
        authorization,
        run_id="execution-run",
        now_monotonic_s=10.0,
        franka_rest_verified=True,
        rh56_disabled_verified=True,
    )
    return gate


def _admission(tmp_path: Path, *, steps=3):
    report, shadow = _passing_report(tmp_path)
    authorization = _authorization()
    gate = _armed_gate(authorization)
    admission = seal_c2_admission(
        report,
        run_id="execution-run",
        requested_policy_steps=steps,
        authorization=authorization,
        safety_gate=gate,
        now_monotonic_s=10.0,
    )
    return admission, gate, shadow


def test_cli_has_no_execute_force_or_bypass_option():
    destinations = {action.dest for action in build_parser()._actions}
    assert "execute" not in destinations
    assert "force" not in destinations
    assert "bypass" not in destinations


def test_offline_cli_report_is_always_disarmed_and_names_missing_wiring():
    report = build_disarmed_cli_report(
        requested_policy_steps=5,
        preflight_report={
            "result": "FAIL",
            "failed_checks": ["observation_shadow_readiness"],
            "shadow_result": "FAIL",
        },
        envelope_error="persistent_session must be an object",
    )
    assert report["result"] == "BLOCKED"
    assert report["arming_state"] == "DISARMED"
    assert report["device_access"] is False
    assert report["backend_factory_called"] is False
    assert "preflight:observation_shadow_readiness" in report["blockers"]
    assert all(code in report["blockers"] for code in CLI_STATIC_BLOCKERS)


def test_admission_cannot_be_constructed_directly():
    with pytest.raises(TypeError, match="seal_c2_admission"):
        BoundedC2Admission(_seal=object())


def test_failed_report_cannot_be_sealed(tmp_path):
    report, _shadow = _passing_report(tmp_path)
    report["result"] = "FAIL"
    authorization = _authorization()
    with pytest.raises(ValueError, match="eligible|result"):
        seal_c2_admission(
            report,
            run_id="execution-run",
            requested_policy_steps=1,
            authorization=authorization,
            safety_gate=_armed_gate(authorization),
            now_monotonic_s=10.0,
        )


def test_requested_k_must_fit_commissioned_session_duration(tmp_path):
    report, _shadow = _passing_report(tmp_path)
    authorization = _authorization()
    with pytest.raises(BoundedC2AdmissionError, match="session duration"):
        seal_c2_admission(
            report,
            run_id="execution-run",
            requested_policy_steps=61,
            authorization=authorization,
            safety_gate=_armed_gate(authorization),
            now_monotonic_s=10.0,
        )


def test_factory_is_called_only_after_active_sealed_admission(tmp_path):
    admission, gate, _shadow = _admission(tmp_path)
    runtime = _FakeRuntime(completed_steps=3)
    calls = []

    def factory(value):
        calls.append(value)
        return runtime

    orchestrator = BoundedC2Orchestrator(
        admission,
        runtime_factory=factory,
        monotonic=_Clock(),
    )
    result = orchestrator.run()
    assert calls == [admission]
    assert result["completed_policy_steps"] == 3
    assert runtime.run_calls[0][0] == 3
    assert runtime.stop_verify_calls == 1
    assert orchestrator.state is BoundedC2State.STOPPED_VERIFIED
    assert gate.state is SafetyState.DISARMED


def test_expired_admission_refuses_before_factory(tmp_path):
    admission, _gate, _shadow = _admission(tmp_path)
    calls = []
    clock = _Clock(admission.hard_deadline_monotonic_s)
    orchestrator = BoundedC2Orchestrator(
        admission,
        runtime_factory=lambda value: calls.append(value),
        monotonic=clock,
    )
    with pytest.raises(BoundedC2AdmissionError, match="deadline|expired"):
        orchestrator.run()
    assert calls == []


def test_bound_artifact_change_refuses_before_factory(tmp_path):
    admission, gate, shadow = _admission(tmp_path)
    shadow.write_bytes(b"changed after admission")
    calls = []
    orchestrator = BoundedC2Orchestrator(
        admission,
        runtime_factory=lambda value: calls.append(value),
        monotonic=_Clock(),
    )
    with pytest.raises(BoundedC2AdmissionError, match="artifact changed"):
        orchestrator.run()
    assert calls == []
    assert gate.state is SafetyState.FAULT_LATCHED


def test_runtime_requires_independent_safety_contract(tmp_path):
    admission, gate, _shadow = _admission(tmp_path)
    runtime = _FakeRuntime()
    runtime.independent_command_watchdogs_configured = False
    orchestrator = BoundedC2Orchestrator(
        admission,
        runtime_factory=lambda _value: runtime,
        monotonic=_Clock(),
    )
    with pytest.raises(BoundedC2RunError, match="watchdogs"):
        orchestrator.run()
    assert runtime.run_calls == []
    assert runtime.stop_verify_calls == 1
    assert gate.state is SafetyState.FAULT_LATCHED


def test_sigint_requests_stop_but_still_requires_verified_stop(tmp_path):
    admission, gate, _shadow = _admission(tmp_path)
    runtime = _FakeRuntime(completed_steps=0, raise_sigint=True)
    orchestrator = BoundedC2Orchestrator(
        admission,
        runtime_factory=lambda _value: runtime,
        monotonic=_Clock(),
    )
    result = orchestrator.run()
    assert result["completed_policy_steps"] == 0
    assert "SIGINT" in runtime.stop_reasons
    assert runtime.stop_verify_calls == 1
    assert gate.state is SafetyState.DISARMED


def test_unverified_dual_stop_is_terminal(tmp_path):
    admission, gate, _shadow = _admission(tmp_path)
    runtime = _FakeRuntime(
        stop_proof=BoundedC2StopProof(
            franka_stop_verified=True,
            rh56_disabled_verified=False,
        )
    )
    orchestrator = BoundedC2Orchestrator(
        admission,
        runtime_factory=lambda _value: runtime,
        monotonic=_Clock(),
    )
    with pytest.raises(BoundedC2RunError, match="stop was not verified"):
        orchestrator.run()
    assert orchestrator.state is BoundedC2State.FAULT_LATCHED
    assert gate.state is SafetyState.FAULT_LATCHED
