from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from anydex_pipeline.rh56_commissioning import (
    EVIDENCE_KIND,
    SCHEMA_VERSION,
    atomic_write_evidence,
    build_snapshot_candidate_binding,
    build_stage1_prerequisite_binding,
    json_sha256,
    seal_evidence,
    sha256_file,
    verify_applied_config,
    verify_evidence,
)
from anydex_pipeline.snapshot import (
    GraspCandidates,
    VisualizationSnapshot,
    save_snapshot_npz,
)
from anydex_pipeline.rh56_hand_path import (
    build_rh56_no_contact_execution_path,
    rh56_feedback_envelope_policy,
)


ROOT = Path(__file__).resolve().parents[1]


def _feedback(phase, targets, angles):
    return {
        "phase": phase,
        "elapsed_s": 0.1,
        "angle_targets": list(targets),
        "angles": list(angles),
        "positions": [100] * 6,
        "forces": [0] * 6,
        "currents": [10] * 6,
        "errors": [0] * 6,
        "statuses": [2] * 6,
        "temperatures": [30] * 6,
    }


def _official_snapshot(path, config, *, target_q6=950, score=0.9):
    pose = np.eye(4, dtype=np.float64)
    hand_pose = pose.copy()
    hand_pose[2, 3] = 0.02
    snapshot = VisualizationSnapshot(
        scene_points=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        scene_colors=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        object_points=np.asarray([[0.5, 0.0, 0.1]], dtype=np.float32),
        object_colors=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        grasps=GraspCandidates(
            canonical_poses=pose[None, ...],
            scores=np.asarray([score], dtype=np.float32),
            type_ids=np.asarray([2], dtype=np.int32),
            collision_free=np.asarray([False]),
            collision_checked=np.asarray([False]),
            selected_index=0,
            hand_poses=hand_pose[None, ...],
            hand_angles=np.asarray(
                [[900, 900, 900, 900, 900, target_q6]], dtype=np.float32
            ),
            source_indices=np.asarray([51], dtype=np.int64),
        ),
        reference_frame=config["reference_frame"],
        T_reference_camera=np.eye(4, dtype=np.float64),
        frame_id=8,
        timestamp_s=100.0,
        calibration_id=config["calibration"]["id"],
        camera_serial=config["calibration"]["camera_serial"],
        model_name="AnyDexGrasp official representation + Inspire obj140 decision",
        representation_checkpoint_sha256="a" * 64,
        decision_checkpoint_sha256s=tuple(
            f"{index + 1:064x}" for index in range(8)
        ),
        official_source_commit="b" * 40,
    )
    save_snapshot_npz(path, snapshot)
    return snapshot


def _passing_evidence(
    tmp_path,
    *,
    target_q6=950,
    coupled=True,
    config_path=None,
    source=None,
    evidence_name="evidence.json",
    snapshot_binding=None,
):
    if config_path is None:
        config = json.loads(
            (ROOT / "configs/fr3_rh56_v7_commissioning.json").read_text(
                encoding="utf-8"
            )
        )
        config_path = tmp_path / "profile.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    else:
        config_path = Path(config_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
    if source is None:
        source = tmp_path / "source.bin"
        source.write_bytes(b"bound commissioning source")
    else:
        source = Path(source)
    waypoints = list(range(975, int(target_q6), -25))
    if not waypoints or waypoints[-1] != int(target_q6):
        waypoints.append(int(target_q6))
    canonical_path = build_rh56_no_contact_execution_path(
        (900, 900, 900, 900, 900, int(target_q6)), step_units=25
    )
    if int(target_q6) == 900:
        return_waypoints = [1000]
        return_strategy = "direct_stage1_v1"
    else:
        return_waypoints = [
            int(item.command_targets[5])
            for item in canonical_path.waypoints
            if item.phase.startswith("q6_reverse_")
        ]
        return_strategy = "canonical_reverse_bootstrap_v3"
    q6_groups = []
    for waypoint in waypoints:
        sample = _feedback(
            f"q6_step_{waypoint:04d}",
            [-1, -1, -1, -1, -1, waypoint],
            [1000] * 5 + [waypoint],
        )
        q6_groups.append({"target_q6": waypoint, "feedback": [sample, sample]})
    q6_return_groups = []
    for waypoint in return_waypoints:
        sample = _feedback(
            f"q6_return_{waypoint:04d}",
            [-1, -1, -1, -1, -1, waypoint],
            [1000] * 5 + [waypoint],
        )
        q6_return_groups.append(
            {"target_q6": waypoint, "feedback": [sample, sample]}
        )
    coupled_target = (
        [900, 900, 900, 900, 900, int(target_q6)] if coupled else None
    )
    current = [1000] * 5
    coupled_waypoints = []
    for axis, target in enumerate([] if coupled_target is None else coupled_target[:5]):
        while current[axis] != target:
            current[axis] = max(target, current[axis] - 25)
            coupled_waypoints.append(list(current) + [int(target_q6)])
    coupled_return_waypoints = (
        [
            list(item.command_targets)
            for item in canonical_path.waypoints
            if item.phase.startswith("bend_reverse_")
        ]
        if coupled
        else []
    )
    coupled_groups = []
    for index, waypoint in enumerate(coupled_waypoints):
        sample = _feedback(
            f"coupled_air_close_step_{index:04d}", waypoint, waypoint
        )
        coupled_groups.append({"target": waypoint, "feedback": [sample, sample]})
    coupled_return_groups = []
    for index, waypoint in enumerate(coupled_return_waypoints):
        sample = _feedback(
            f"coupled_air_return_step_{index:04d}", waypoint, waypoint
        )
        coupled_return_groups.append(
            {"target": waypoint, "feedback": [sample, sample]}
        )
    opened = _feedback("q6_return_open_verify", [-1] * 6, [1000] * 6)
    disabled = _feedback("disable_verify", [-1] * 6, [1000] * 6)
    adopted = _feedback("adopt_disable_verify", [-1] * 6, [1000] * 6)
    sweep_preflight = _feedback("q6_sweep_preflight", [-1] * 6, [1000] * 6)
    return_preflight = _feedback(
        "q6_return_preflight", [-1] * 6, [1000] * 5 + [int(target_q6)]
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": EVIDENCE_KIND,
        "run_id": "test-run",
        "started_at_utc": "2026-07-18T00:00:00Z",
        "completed_at_utc": "2026-07-18T00:01:00Z",
        "motion_authorized": False,
        "control_profile": {
            "path": str(config_path.resolve()),
            "file_sha256": sha256_file(config_path),
            "parsed_sha256": json_sha256(config),
            "snapshot": config,
        },
        "source_bindings": [
            {
                "name": name,
                "path": str(source.resolve()),
                "sha256": sha256_file(source),
            }
            for name in (
                "commission_cli",
                "commission_evidence_module",
                "rh56_hand_path",
                "rh56_reset_open",
                "rh56_sequence_driver",
                "rh56_register_api",
                "franka_sequence_driver",
                "control_config_module",
                "adapter_mesh",
                "adapter_provenance",
                "actuator_to_joint_xlsx",
                "driver_to_angle_xls",
                "actuator_to_urdf_generator",
            )
        ],
        "operator_confirmations": {
            "installed_on_fr3": True,
            "24v_cutoff_ready": True,
            "franka_stop_ready": True,
            "workspace_clear": True,
            "no_contact_PLA_scope": True,
            "wide_q6_confirmed": int(target_q6) < 900,
            "coupled_air_close_confirmed": bool(coupled),
            "exact_air_target_confirmed": False,
        },
        "request": {
            "safety_scope": "installed_on_FR3_PLA_low_speed_unloaded_free_air_only",
            "target_q6": int(target_q6),
            "requested_q6_range": [int(target_q6), 1000],
            "step_units": 25,
            "q6_waypoints": waypoints,
            "q6_return_waypoints": return_waypoints,
            "q6_return_strategy": return_strategy,
            "speed": 40,
            "force_limit_g": 80,
            "motion_timeout_s_per_step": 20.0,
            "angle_tolerance_units": 25,
            "q6_open_min_angle": 975,
            "feedback_envelope_policy": rh56_feedback_envelope_policy(25),
            "q6_endpoint_tolerance_units": 20,
            "q6_reverse_hysteresis_tolerance_units": 30,
            "endpoint_stable_samples": 2,
            "max_axis_current_ma": 400,
            "stop_max_axis_current_ma": 100,
            "max_inactive_drift_units": 8,
            "aggregate_current_policy": "telemetry_only_device_limits_remain_active",
            "target_source": (
                "official_snapshot_candidate"
                if snapshot_binding is not None
                else "manual_cli"
            ),
            "coupled_closure_requested": bool(coupled),
            "coupled_targets": coupled_target,
            "coupled_step_units": 25 if coupled else None,
            "coupled_command_waypoints": coupled_waypoints if coupled else None,
            "coupled_return_command_waypoints": (
                coupled_return_waypoints if coupled else None
            ),
        },
        "franka_read_only": {
            "connection": "read_once_only_no_controller_no_robot_write",
            "verified": True,
            "continuous_gate": {
                "check_count": 42,
                "failure": None,
                "last_success": {
                    "robot_mode": "RobotMode.Idle",
                    "F_T_EE": config["franka"]["expected_F_T_EE"],
                    "m_ee_kg": 0.607,
                    "F_x_Cee_m": [0.0, 0.0, 0.076],
                    "I_ee_kg_m2": [
                        0.00151, 0.0, 0.0,
                        0.0, 0.00169, 0.0,
                        0.0, 0.0, 0.000442,
                    ],
                    "m_load_kg": 0.0,
                    "m_total_kg": 0.607,
                },
            },
            "initial": {
                "robot_mode": "RobotMode.Idle",
                "F_T_EE": config["franka"]["expected_F_T_EE"],
                "m_ee_kg": 0.607,
                "F_x_Cee_m": [0.0, 0.0, 0.076],
                "I_ee_kg_m2": [
                    0.00151, 0.0, 0.0,
                    0.0, 0.00169, 0.0,
                    0.0, 0.0, 0.000442,
                ],
                "m_load_kg": 0.0,
                "m_total_kg": 0.607,
            },
            "final": {
                "robot_mode": "RobotMode.Idle",
                "F_T_EE": config["franka"]["expected_F_T_EE"],
                "m_ee_kg": 0.607,
                "F_x_Cee_m": [0.0, 0.0, 0.076],
                "I_ee_kg_m2": [
                    0.00151, 0.0, 0.0,
                    0.0, 0.00169, 0.0,
                    0.0, 0.0, 0.000442,
                ],
                "m_load_kg": 0.0,
                "m_total_kg": 0.607,
            },
        },
        "rh56_device": {
            "hand_id": 1,
            "initial_snapshot": {
                "angles": [1000] * 6,
                "current_limits": [1400] * 6,
            },
        },
        "result": {
            "status": "pass",
            "adopted_disabled_verified": True,
            "q6_sweep_pass": True,
            "q6_return_pass": True,
            "coupled_closure_pass": bool(coupled),
            "reopened_and_verified": True,
            "disabled_verified": True,
            "feedback_envelope_enforced": True,
            "feedback_envelope_policy_sha256": (
                rh56_feedback_envelope_policy(25)["sha256"]
            ),
            "operation_error": None,
            "stop_error": None,
        },
        "observations": {
            "preopen_reset_q6_waypoints": [],
            "all_feedback": [
                adopted,
                adopted,
                sweep_preflight,
                *[
                    sample
                    for group in q6_groups
                    for sample in group["feedback"]
                ],
                *[
                    sample
                    for group in coupled_groups
                    for sample in group["feedback"]
                ],
                *[
                    sample
                    for group in coupled_return_groups
                    for sample in group["feedback"]
                ],
                return_preflight,
                *[
                    sample
                    for group in q6_return_groups
                    for sample in group["feedback"]
                ],
                opened,
                disabled,
                disabled,
            ],
            "adopt_disable_feedback": [adopted, adopted],
            "q6_steps": q6_groups,
            "q6_return_steps": q6_return_groups,
            "coupled_air_close_steps": coupled_groups,
            "coupled_air_return_steps": coupled_return_groups,
            "coupled_air_close_feedback": [
                sample
                for group in coupled_groups
                for sample in group["feedback"]
            ],
            "final_open_feedback": opened,
            "disable_feedback": [disabled, disabled],
            "actual_q6_range": [min(waypoints), max(waypoints)],
            "actual_q6_return_range": [
                min(return_waypoints),
                max(return_waypoints),
            ],
        },
        "final": {
            "angle_targets": [-1] * 6,
            "angles": [1000] * 6,
            "currents": [10] * 6,
            "errors": [0] * 6,
            "statuses": [2] * 6,
            "temperatures": [30] * 6,
            "reopened_and_verified": True,
            "disabled_verified": True,
            "snapshot_after_disable": {},
        },
        "snapshot_candidate": snapshot_binding,
    }
    payload["stage1_prerequisite"] = None
    if int(target_q6) < 900:
        _, stage1_path, _, _, _ = _passing_evidence(
            tmp_path,
            target_q6=900,
            coupled=False,
            config_path=config_path,
            source=source,
            evidence_name="stage1.json",
        )
        payload["stage1_prerequisite"] = build_stage1_prerequisite_binding(
            stage1_path,
            expected_config=config,
            expected_config_path=config_path,
        )
        payload["started_at_utc"] = "2026-07-18T00:02:00Z"
        payload["completed_at_utc"] = "2026-07-18T00:03:00Z"
    evidence = seal_evidence(payload)
    evidence_path = tmp_path / evidence_name
    atomic_write_evidence(evidence_path, evidence)
    return evidence, evidence_path, config, config_path, source


def test_passing_evidence_verifies_and_proposes_exact_target_gate(tmp_path):
    _, evidence_path, _, config_path, _ = _passing_evidence(tmp_path)

    result = verify_evidence(
        evidence_path, config_path=config_path, require_coupled=True
    )

    assert result.passed
    assert result.proposal == {
        "inspire.thumb_rotate_validated_realtime_range": [900, 1000],
        "inspire.six_axis_coupled_closure_commissioned": True,
        "inspire.commissioned_air_closure_targets": [
            [900, 900, 900, 900, 900, 950]
        ],
        "scope": "low_speed_unloaded_no_contact_PLA",
        "evidence_payload_sha256": json.loads(
            evidence_path.read_text(encoding="utf-8")
        )["integrity"]["payload_sha256"],
    }


def test_snapshot_bound_evidence_reloads_candidate_and_cross_checks_request(tmp_path):
    config = json.loads(
        (ROOT / "configs/fr3_rh56_v7_commissioning.json").read_text(
            encoding="utf-8"
        )
    )
    config_path = tmp_path / "profile.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    snapshot_path = tmp_path / "official.npz"
    _official_snapshot(snapshot_path, config, target_q6=950)
    binding = build_snapshot_candidate_binding(
        snapshot_path, config=config, candidate_index=None
    )
    _, evidence_path, _, _, _ = _passing_evidence(
        tmp_path,
        target_q6=950,
        coupled=True,
        config_path=config_path,
        snapshot_binding=binding,
    )

    result = verify_evidence(
        evidence_path, config_path=config_path, require_coupled=True
    )

    assert result.passed, result.blockers


def test_snapshot_bound_evidence_locks_if_snapshot_changes(tmp_path):
    config = json.loads(
        (ROOT / "configs/fr3_rh56_v7_commissioning.json").read_text(
            encoding="utf-8"
        )
    )
    config_path = tmp_path / "profile.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    snapshot_path = tmp_path / "official.npz"
    _official_snapshot(snapshot_path, config, target_q6=950)
    binding = build_snapshot_candidate_binding(snapshot_path, config=config)
    _, evidence_path, _, _, _ = _passing_evidence(
        tmp_path,
        target_q6=950,
        coupled=True,
        config_path=config_path,
        snapshot_binding=binding,
    )
    _official_snapshot(snapshot_path, config, target_q6=951)

    result = verify_evidence(
        evidence_path, config_path=config_path, require_coupled=True
    )

    assert not result.passed
    assert any("snapshot/candidate data changed" in item for item in result.blockers)


def test_snapshot_bound_evidence_locks_if_request_targets_are_resealed(tmp_path):
    config = json.loads(
        (ROOT / "configs/fr3_rh56_v7_commissioning.json").read_text(
            encoding="utf-8"
        )
    )
    config_path = tmp_path / "profile.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    snapshot_path = tmp_path / "official.npz"
    _official_snapshot(snapshot_path, config, target_q6=950)
    binding = build_snapshot_candidate_binding(snapshot_path, config=config)
    evidence, _, _, _, _ = _passing_evidence(
        tmp_path,
        target_q6=950,
        coupled=True,
        config_path=config_path,
        snapshot_binding=binding,
    )
    payload = copy.deepcopy(evidence)
    payload.pop("integrity")
    payload["request"]["target_q6"] = 951
    tampered = tmp_path / "rebound.json"
    atomic_write_evidence(tampered, seal_evidence(payload))

    result = verify_evidence(tampered, config_path=config_path, require_coupled=True)

    assert not result.passed
    assert any(
        "request.target_q6 differs from the bound official candidate q6" in item
        for item in result.blockers
    )


def test_candidate51_q6_646_evidence_expands_exact_lower_bound(tmp_path):
    _, evidence_path, _, config_path, _ = _passing_evidence(
        tmp_path, target_q6=646
    )

    result = verify_evidence(
        evidence_path, config_path=config_path, require_coupled=True
    )

    assert result.passed
    assert result.proposal["inspire.thumb_rotate_validated_realtime_range"] == [
        646,
        1000,
    ]
    assert result.proposal[
        "inspire.six_axis_coupled_closure_commissioned"
    ] is True


def test_wide_evidence_without_or_with_tampered_stage1_binding_is_locked(tmp_path):
    evidence, _, _, config_path, _ = _passing_evidence(
        tmp_path, target_q6=646
    )
    payload = {
        key: copy.deepcopy(value)
        for key, value in evidence.items()
        if key != "integrity"
    }
    payload["stage1_prerequisite"] = None
    missing = tmp_path / "wide-missing-stage1.json"
    atomic_write_evidence(missing, seal_evidence(payload))
    result = verify_evidence(missing, config_path=config_path)
    assert not result.passed
    assert any("requires a bound formal Stage1 PASS" in x for x in result.blockers)

    payload["stage1_prerequisite"] = copy.deepcopy(
        evidence["stage1_prerequisite"]
    )
    payload["stage1_prerequisite"]["payload_sha256"] = "0" * 64
    tampered = tmp_path / "wide-tampered-stage1-binding.json"
    atomic_write_evidence(tampered, seal_evidence(payload))
    result = verify_evidence(tampered, config_path=config_path)
    assert not result.passed
    assert any("binding fields/hash" in x for x in result.blockers)


def test_stage1_dependency_rejects_wrong_target_failed_result_and_source_drift(
    tmp_path,
):
    wrong_dir = tmp_path / "wrong"
    wrong_dir.mkdir()
    _, wrong_path, config, config_path, _ = _passing_evidence(
        wrong_dir,
        target_q6=950,
        coupled=False,
        evidence_name="wrong-target.json",
    )
    with pytest.raises(ValueError, match="target_q6 must be exactly 900"):
        build_stage1_prerequisite_binding(
            wrong_path,
            expected_config=config,
            expected_config_path=config_path,
        )

    failed_dir = tmp_path / "failed"
    failed_dir.mkdir()
    stage1, _, config, config_path, _ = _passing_evidence(
        failed_dir,
        target_q6=900,
        coupled=False,
        evidence_name="stage1-pass.json",
    )
    failed_payload = {
        key: copy.deepcopy(value)
        for key, value in stage1.items()
        if key != "integrity"
    }
    failed_payload["result"]["status"] = "fail"
    failed_path = failed_dir / "stage1-failed.json"
    atomic_write_evidence(failed_path, seal_evidence(failed_payload))
    with pytest.raises(ValueError, match="commissioning result is not pass"):
        build_stage1_prerequisite_binding(
            failed_path,
            expected_config=config,
            expected_config_path=config_path,
        )

    stale_dir = tmp_path / "stale"
    stale_dir.mkdir()
    _, stage1_path, config, config_path, source = _passing_evidence(
        stale_dir,
        target_q6=900,
        coupled=False,
        evidence_name="stage1.json",
    )
    source.write_bytes(b"source changed after Stage1")
    with pytest.raises(ValueError, match="bound source file changed"):
        build_stage1_prerequisite_binding(
            stage1_path,
            expected_config=config,
            expected_config_path=config_path,
        )


def test_resealed_zero_stable_empty_feedback_cannot_expand_q6_range(tmp_path):
    evidence, _, _, config_path, _ = _passing_evidence(
        tmp_path, target_q6=646
    )
    payload = {
        key: copy.deepcopy(value)
        for key, value in evidence.items()
        if key != "integrity"
    }
    payload["request"]["endpoint_stable_samples"] = 0
    payload["observations"]["q6_steps"] = [
        {"target_q6": group["target_q6"], "feedback": []}
        for group in payload["observations"]["q6_steps"]
    ]
    payload["observations"]["actual_q6_range"] = None
    forged = tmp_path / "resealed-zero-stable.json"
    atomic_write_evidence(forged, seal_evidence(payload))

    result = verify_evidence(forged, config_path=config_path)

    assert not result.passed
    assert result.proposal is None
    assert any("endpoint_stable_samples must be in 2..10" in x for x in result.blockers)
    applied = verify_applied_config(forged, config_path)
    assert not applied.passed
    assert applied.proposal is None


def test_resealed_group_not_identical_to_all_feedback_is_rejected(tmp_path):
    evidence, _, _, config_path, _ = _passing_evidence(tmp_path)
    payload = {
        key: copy.deepcopy(value)
        for key, value in evidence.items()
        if key != "integrity"
    }
    replacement = copy.deepcopy(
        payload["observations"]["q6_steps"][0]["feedback"][0]
    )
    replacement["elapsed_s"] = 0.2
    payload["observations"]["q6_steps"][0]["feedback"][0] = replacement
    forged = tmp_path / "resealed-divergent-group.json"
    atomic_write_evidence(forged, seal_evidence(payload))

    result = verify_evidence(forged, config_path=config_path)

    assert not result.passed
    assert result.proposal is None
    assert any("differs from all_feedback" in x for x in result.blockers)


def test_resealed_continuous_franka_gate_failure_cannot_verify(tmp_path):
    evidence, _, _, config_path, _ = _passing_evidence(tmp_path)
    payload = {
        key: copy.deepcopy(value)
        for key, value in evidence.items()
        if key != "integrity"
    }
    payload["franka_read_only"]["continuous_gate"]["failure"] = (
        "RuntimeError: Franka left Idle"
    )
    forged = tmp_path / "resealed-franka-gate-failure.json"
    atomic_write_evidence(forged, seal_evidence(payload))

    result = verify_evidence(forged, config_path=config_path)

    assert not result.passed
    assert result.proposal is None
    assert "continuous Franka gate recorded a failure" in result.blockers


def test_resealed_feedback_envelope_policy_tamper_is_locked(tmp_path):
    evidence, _, _, config_path, _ = _passing_evidence(tmp_path)
    payload = {
        key: copy.deepcopy(value)
        for key, value in evidence.items()
        if key != "integrity"
    }
    payload["request"]["feedback_envelope_policy"]["sha256"] = "0" * 64
    forged = tmp_path / "resealed-feedback-policy-tamper.json"
    atomic_write_evidence(forged, seal_evidence(payload))

    result = verify_evidence(forged, config_path=config_path)

    assert not result.passed
    assert result.proposal is None
    assert any(
        "feedback-envelope policy/hash differs" in blocker
        for blocker in result.blockers
    )


def test_resealed_feedback_outside_adjacent_endpoint_hull_is_locked(tmp_path):
    evidence, _, _, config_path, _ = _passing_evidence(tmp_path)
    payload = {
        key: copy.deepcopy(value)
        for key, value in evidence.items()
        if key != "integrity"
    }
    phase = "q6_step_0975"
    for sample in payload["observations"]["all_feedback"]:
        if sample["phase"] == phase:
            sample["angles"][5] = 949
    for sample in payload["observations"]["q6_steps"][0]["feedback"]:
        sample["angles"][5] = 949
    payload["observations"]["actual_q6_range"] = [949, 950]
    forged = tmp_path / "resealed-feedback-envelope-escape.json"
    atomic_write_evidence(forged, seal_evidence(payload))

    result = verify_evidence(forged, config_path=config_path)

    assert not result.passed
    assert result.proposal is None
    assert any("escaped rh56_post_write" in x for x in result.blockers)


def test_resealed_evidence_without_runtime_enforcement_is_locked(tmp_path):
    evidence, _, _, config_path, _ = _passing_evidence(tmp_path)
    payload = {
        key: copy.deepcopy(value)
        for key, value in evidence.items()
        if key != "integrity"
    }
    payload["result"]["feedback_envelope_enforced"] = False
    forged = tmp_path / "resealed-feedback-runtime-not-enforced.json"
    atomic_write_evidence(forged, seal_evidence(payload))

    result = verify_evidence(forged, config_path=config_path)

    assert not result.passed
    assert result.proposal is None
    assert "runtime feedback envelope was not enforced" in result.blockers


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("step_units", 9, "step_units must be in 10..50"),
        ("max_axis_current_ma", 1401, "max_axis_current_ma must be in 1..1400"),
        ("max_inactive_drift_units", 0, "max_inactive_drift_units must be in 1..20"),
        ("motion_timeout_s_per_step", 31.0, "motion_timeout_s_per_step must be in 5..30"),
    ],
)
def test_resealed_out_of_range_runtime_parameters_are_rejected(
    tmp_path, field, value, message
):
    evidence, _, _, config_path, _ = _passing_evidence(tmp_path)
    payload = {
        key: copy.deepcopy(item)
        for key, item in evidence.items()
        if key != "integrity"
    }
    payload["request"][field] = value
    forged = tmp_path / f"resealed-bad-{field}.json"
    atomic_write_evidence(forged, seal_evidence(payload))

    result = verify_evidence(forged, config_path=config_path)

    assert not result.passed
    assert result.proposal is None
    assert any(message in blocker for blocker in result.blockers)


def test_tampered_evidence_and_changed_source_fail_closed(tmp_path):
    evidence, evidence_path, _, config_path, source = _passing_evidence(tmp_path)
    source.chmod(0o644)
    source.write_bytes(b"changed")
    result = verify_evidence(evidence_path, config_path=config_path)
    assert not result.passed
    assert any("bound source file changed" in item for item in result.blockers)

    tampered = copy.deepcopy(evidence)
    tampered["result"]["disabled_verified"] = False
    evidence_path.chmod(0o644)
    evidence_path.write_text(json.dumps(tampered), encoding="utf-8")
    result = verify_evidence(evidence_path, config_path=config_path)
    assert any("canonical payload SHA-256 mismatch" in item for item in result.blockers)


def test_atomic_evidence_never_overwrites(tmp_path):
    evidence, evidence_path, _, _, _ = _passing_evidence(tmp_path)
    before = evidence_path.read_bytes()

    with pytest.raises(FileExistsError):
        atomic_write_evidence(evidence_path, evidence)

    assert evidence_path.read_bytes() == before


def test_verify_applied_requires_exact_three_field_update(tmp_path):
    _, evidence_path, config, _, _ = _passing_evidence(tmp_path)
    applied = copy.deepcopy(config)
    applied["inspire"]["six_axis_coupled_closure_commissioned"] = True
    applied["inspire"]["commissioned_air_closure_targets"] = [
        [900, 900, 900, 900, 900, 950]
    ]
    applied_path = tmp_path / "applied.json"
    applied_path.write_text(json.dumps(applied, indent=2), encoding="utf-8")

    result = verify_applied_config(
        evidence_path, applied_path, require_coupled=True
    )
    assert result.passed

    applied["inspire"]["force_limit_g"] = 81
    applied_path.write_text(json.dumps(applied, indent=2), encoding="utf-8")
    result = verify_applied_config(evidence_path, applied_path)
    assert not result.passed
    assert any("exact evidence-derived three-field update" in x for x in result.blockers)


def test_failed_result_cannot_be_promoted_by_final_fields(tmp_path):
    evidence, evidence_path, _, config_path, _ = _passing_evidence(tmp_path)
    evidence["result"]["status"] = "fail"
    evidence["integrity"]["payload_sha256"] = json_sha256(
        {key: value for key, value in evidence.items() if key != "integrity"}
    )
    evidence_path.chmod(0o644)
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    result = verify_evidence(evidence_path, config_path=config_path)

    assert not result.passed
    assert "commissioning result is not pass" in result.blockers
