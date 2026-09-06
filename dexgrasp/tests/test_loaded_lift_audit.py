from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import anydex_pipeline.loaded_lift_audit as loaded_lift
from anydex_pipeline.installed_tool_audit import (
    ARTIFACT_TYPE as INSTALLED_TOOL_ARTIFACT_TYPE,
)
from anydex_pipeline.joint_path_sampling import (
    CANONICAL_JOINT_SAMPLING_ALGORITHM,
    canonical_joint_path_samples,
)
from anydex_pipeline.loaded_lift_audit import (
    ARTIFACT_TYPE,
    LOADED_LIFT_CHECK_SPECS,
    MODE,
    bind_loaded_lift_audit,
    loaded_lift_profile_blockers,
    validate_loaded_lift_audit,
)


def _array_sha256(value):
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(
        ("dtype=<f8;shape={};".format(",".join(map(str, array.shape)))).encode(
            "ascii"
        )
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _json_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reseal(value):
    result = json.loads(json.dumps(value))
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = _json_sha256(result)
    return result


def _write(path, value=b"fixture"):
    path = Path(path)
    path.write_bytes(value)
    return path


def _hand_model_binding(targets, q12):
    return {
        "urdf_sha256": "1" * 64,
        "mapping_sha256": "2" * 64,
        "mesh_resolution": "full",
        "link_count": 13,
        "links": [
            {"name": "Link{}".format(index), "path": "/mesh/{}.stl".format(index), "sha256": "3" * 64}
            for index in range(13)
        ],
        "closure_joint_positions_rad": q12,
        "closure_fk_sha256": "4" * 64,
    }


def _make_artifact(tmp_path, *, config_path=None, base_path=None):
    config_path = config_path or _write(tmp_path / "control.json", b"{}\n")
    base_path = base_path or _write(tmp_path / "base-loaded.json", b"base\n")
    material_path = _write(tmp_path / "material.json", b"material\n")
    payload_path = _write(tmp_path / "payload.stl", b"solid payload\nendsolid\n")
    retention_path = _write(tmp_path / "retention.json", b"retention\n")
    mass_properties_path = _write(
        tmp_path / "mass-properties.json", b"mass properties\n"
    )
    fk_path = _write(tmp_path / "fk.json", b"fk\n")
    fr3_path = _write(tmp_path / "fr3.urdf", b"fr3\n")
    scene_path = _write(tmp_path / "scene.npz", b"scene\n")
    generator_path = _write(tmp_path / "loaded_lift_generator.py", b"# reviewed\n")
    collision_report_path = tmp_path / "collision-report.json"

    grasp_q = np.asarray([0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0])
    transit_q = grasp_q.copy()
    transit_q[0] = 0.025
    lift_q = grasp_q.copy()
    lift_q[0] = 0.05
    names = ["grasp", "lift_transit_0", "lift"]
    values = [grasp_q, transit_q, lift_q]
    maximum_step = 0.01
    outbound, intervals = canonical_joint_path_samples(values, maximum_step)
    round_trip = np.concatenate((outbound, outbound[-2::-1]), axis=0)
    cursor = 0
    segments = []
    for left, right, count in zip(names[:-1], names[1:], intervals):
        start = cursor
        cursor += count
        segments.append(
            {
                "name": "{}_to_{}".format(left, right),
                "start_index": start,
                "end_index": cursor,
            }
        )

    grasp_pose = np.eye(4)
    lift_pose = np.eye(4)
    lift_pose[2, 3] = 0.1
    fk_poses = np.repeat(np.eye(4)[None, :, :], len(outbound), axis=0)
    fk_poses[:, 2, 3] = np.linspace(0.0, 0.1, len(outbound))
    targets = np.asarray([500, 510, 520, 530, 540, 950], dtype=np.float64)
    q12 = np.linspace(0.1, 1.2, 12)
    feedback_lower = q12 - 0.01
    feedback_upper = q12 + 0.01
    feedback_sha = _array_sha256(
        np.stack((feedback_lower, feedback_upper), axis=0)
    )
    hand_manifest = _hand_model_binding(targets.tolist(), q12.tolist())
    hand_manifest_sha = _json_sha256(hand_manifest)
    adapter_sha = "a" * 64
    material_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": loaded_lift.MATERIAL_APPROVAL_ARTIFACT_TYPE,
                "approval_id": "material-test",
                "adapter_sha256": adapter_sha,
                "material": "Al-6061-T6",
                "approved_payload_mass_kg": 0.5,
                "approved_motion_scope": MODE,
                "max_joint_velocity_rad_s": 0.05,
                "max_joint_acceleration_rad_s2": 0.1,
                "load_rated": True,
                "approved": True,
                "motion_authorized": False,
            }
        ),
        encoding="utf-8",
    )

    geometry = {
        "path": str(payload_path.resolve()),
        "sha256": _file_sha256(payload_path),
        "representation": "watertight_triangle_mesh",
        "frame": "payload",
        "watertight": True,
    }
    dynamics = {
        "mass_kg": 0.2,
        "F_x_Cload_m": [0.0, 0.0, 0.03],
        "inertia_kg_m2": [
            [0.001, 0.0, 0.0],
            [0.0, 0.0011, 0.0],
            [0.0, 0.0, 0.0012],
        ],
    }
    attachment_pose = np.eye(4)
    attachment = {
        "convention": "T_F_payload=F_T_EE@T_EE_hand@T_hand_payload",
        "F_T_EE": attachment_pose.tolist(),
        "F_T_EE_sha256": _array_sha256(attachment_pose),
        "T_EE_hand": attachment_pose.tolist(),
        "T_EE_hand_sha256": _array_sha256(attachment_pose),
        "T_hand_payload": attachment_pose.tolist(),
        "T_hand_payload_sha256": _array_sha256(attachment_pose),
        "T_F_payload": attachment_pose.tolist(),
        "T_F_payload_sha256": _array_sha256(attachment_pose),
        "rigid_attachment_verified": True,
    }
    mass_properties_file = {
        "schema_version": 1,
        "artifact_type": loaded_lift.MASS_PROPERTIES_ARTIFACT_TYPE,
        "computation_id": "mass-properties-test",
        "payload_geometry_sha256": geometry["sha256"],
        "T_F_payload_sha256": _array_sha256(attachment_pose),
        "mass_kg": dynamics["mass_kg"],
        "F_x_Cload_m": dynamics["F_x_Cload_m"],
        "inertia_at_com_F_kg_m2": dynamics["inertia_kg_m2"],
        "backend_name": "fake-mesh-integrator",
        "backend_version": "1",
        "implementation_sha256": "9" * 64,
        "watertight_verified": True,
        "mass_properties_verified": True,
        "authoritative": True,
        "motion_authorized": False,
    }
    mass_properties_path.write_text(
        json.dumps(mass_properties_file), encoding="utf-8"
    )
    mass_properties = {
        key: value
        for key, value in mass_properties_file.items()
        if key not in ("schema_version", "artifact_type", "motion_authorized")
    }
    mass_properties.update(
        {
            "path": str(mass_properties_path.resolve()),
            "sha256": _file_sha256(mass_properties_path),
        }
    )
    retention_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": loaded_lift.RETENTION_APPROVAL_ARTIFACT_TYPE,
                "approval_id": "retention-test",
                "payload_geometry_sha256": geometry["sha256"],
                "closed_hand_targets_sha256": _array_sha256(targets),
                "minimum_contact_axes": 3,
                "validated_payload_mass_kg": 0.25,
                "retention_verified": True,
                "loaded_lift_round_trip_approved": True,
                "motion_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    retention = {
        "path": str(retention_path.resolve()),
        "sha256": _file_sha256(retention_path),
        "approval_id": "retention-test",
        "payload_geometry_sha256": geometry["sha256"],
        "closed_hand_targets_sha256": _array_sha256(targets),
        "minimum_contact_axes": 3,
        "validated_payload_mass_kg": 0.25,
        "retention_verified": True,
    }
    payload_unsigned = {
        "geometry": geometry,
        "dynamics": dynamics,
        "attachment": attachment,
        "mass_properties": mass_properties,
        "retention_approval": retention,
    }
    policies = {
        "max_joint_step_rad": maximum_step,
        "max_q_tracking_error_rad": 0.002,
        "robot_clearance_margin_m": 0.002,
        "scene_clearance_margin_m": 0.005,
        "retention_contact_max_separation_m": 0.002,
        "retention_max_penetration_m": 0.003,
        "payload_translation_uncertainty_m": 0.002,
        "payload_rotation_uncertainty_rad": 0.02,
        "hand_arrival_tolerance_units": 25,
        "settle": {
            "position_m": 0.005,
            "orientation_rad": 0.04,
            "linear_speed_m_s": 0.01,
            "angular_speed_rad_s": 0.05,
            "stable_seconds": 0.3,
        },
        "unknown_space_policy": "occupied",
        "round_trip_setdown_required": True,
        "load_applied_after_grasp_settle": True,
        "load_cleared_only_after_setdown_settle_and_release": True,
    }
    details = {
        "max_q_tracking_error_rad": 0.002,
        "joint_tracking_uncertainty_applied": True,
        "continuous_segment_envelope_verified": True,
        "conservative_motion_bound_m": 0.001,
        "minimum_distance_is_after_motion_bound": True,
        "closed_hand_feedback_envelope_applied": True,
        "closed_hand_feedback_envelope_sha256": feedback_sha,
        "payload_pose_uncertainty_applied": True,
        "payload_geometry_included": True,
        "adapter_geometry_included": True,
        "fr3_geometry_included": True,
        "closed_hand_geometry_included": True,
        "unknown_space_policy": "occupied",
        "unknown_space_policy_applied": True,
    }
    checks = []
    for spec in LOADED_LIFT_CHECK_SPECS:
        contact = spec.expectation == "retained_contact"
        checks.append(
            {
                "check_id": spec.check_id,
                "scope": spec.scope,
                "expectation": spec.expectation,
                "coverage": "loaded_round_trip_dense_path",
                "authoritative": True,
                "tested_sample_indices": list(range(len(round_trip))),
                "minimum_signed_distance_m": 0.0 if contact else 0.02,
                "observed_pairs": ["Link11 / payload"] if contact else [],
                "details": dict(details),
                "passed": True,
                "failures": [],
            }
        )
    artifact = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "mode": MODE,
        "created_at_s": 1.0,
        "audit_generator": {
            "name": "reviewed-loaded-lift-generator",
            "version": "1",
            "implementation_path": str(generator_path.resolve()),
            "implementation_sha256": _file_sha256(generator_path),
        },
        "bindings": {
            "base_loaded_grasp": {
                "path": str(base_path.resolve()),
                "file_sha256": _file_sha256(base_path),
                "artifact_sha256": "b" * 64,
                "schema_version": 2,
                "artifact_type": INSTALLED_TOOL_ARTIFACT_TYPE,
                "mode": "loaded_grasp",
                "decision_passed": True,
                "selected_candidate_index": 51,
                "grasp_q_sha256": _array_sha256(grasp_q),
                "grasp_pose_sha256": _array_sha256(grasp_pose),
                "closed_hand_targets_sha256": _array_sha256(targets),
                "closed_hand_q12_sha256": _array_sha256(q12),
                "closure_fk_sha256": "4" * 64,
                "adapter_sha256": adapter_sha,
                "hand_model_binding_sha256": hand_manifest_sha,
            },
            "control_profile": {
                "path": str(config_path.resolve()),
                "sha256": _file_sha256(config_path),
            },
            "adapter_material_approval": {
                "path": str(material_path.resolve()),
                "sha256": _file_sha256(material_path),
                "approval_id": "material-test",
                "adapter_sha256": adapter_sha,
                "material": "Al-6061-T6",
                "approved_payload_mass_kg": 0.5,
                "max_joint_velocity_rad_s": 0.05,
                "max_joint_acceleration_rad_s2": 0.1,
                "load_rated": True,
                "approved_motion_scope": MODE,
            },
            "payload": dict(payload_unsigned, binding_sha256=_json_sha256(payload_unsigned)),
            "closed_hand": {
                "actuator_targets": targets.tolist(),
                "actuator_targets_sha256": _array_sha256(targets),
                "joint_positions_rad": q12.tolist(),
                "joint_positions_sha256": _array_sha256(q12),
                "closure_fk_sha256": "4" * 64,
                "link_count": 13,
                "link_mesh_manifest_sha256": hand_manifest_sha,
                "feedback_envelope_method": "official_xls_all_six_arrival_box_v1",
                "feedback_q12_lower_rad": feedback_lower.tolist(),
                "feedback_q12_upper_rad": feedback_upper.tolist(),
                "feedback_envelope_sha256": feedback_sha,
                "feedback_envelope_applied": True,
            },
            "lift_endpoint": {
                "q_rad": lift_q.tolist(),
                "q_sha256": _array_sha256(lift_q),
                "T_base_EE": lift_pose.tolist(),
                "pose_sha256": _array_sha256(lift_pose),
            },
            "joint_path": {
                "trajectory_contract": "loaded_lift_joint_waypoint_round_trip_v1",
                "sampling_algorithm": CANONICAL_JOINT_SAMPLING_ALGORITHM,
                "waypoints": [
                    {
                        "name": name,
                        "q_rad": value.tolist(),
                        "q_sha256": _array_sha256(value),
                    }
                    for name, value in zip(names, values)
                ],
                "segments": segments,
                "outbound_sample_count": len(outbound),
                "outbound_samples_rad": outbound.tolist(),
                "outbound_sha256": _array_sha256(outbound),
                "round_trip_sample_count": len(round_trip),
                "round_trip_samples_rad": round_trip.tolist(),
                "round_trip_sha256": _array_sha256(round_trip),
                "maximum_observed_joint_step_rad": float(
                    np.max(np.abs(np.diff(round_trip, axis=0)))
                ),
                "round_trip_setdown_required": True,
                "execution_time_law": {
                    "profile": "cosine_stop_to_stop_dynamic_segments_v1",
                    "max_joint_velocity_rad_s": 0.05,
                    "max_joint_acceleration_rad_s2": 0.1,
                    "max_dynamic_segment_rad": 0.02,
                    "min_segment_duration_s": 1.0,
                    "velocity_peak_factor": loaded_lift.COSINE_PEAK_VELOCITY_FACTOR,
                    "acceleration_peak_factor": loaded_lift.COSINE_PEAK_ACCELERATION_FACTOR,
                    "material_approval_limits_applied": True,
                    "stop_at_every_waypoint": True,
                    "reverse_reuses_same_limits": True,
                },
            },
            "fk_manifest": {
                "path": str(fk_path.resolve()),
                "sha256": _file_sha256(fk_path),
                "backend_name": "fake-pinocchio",
                "backend_version": "1",
                "implementation_sha256": "c" * 64,
                "joint_samples_sha256": _array_sha256(outbound),
                "sample_count": len(outbound),
                "outbound_ee_poses_base": fk_poses.tolist(),
                "outbound_ee_poses_sha256": _array_sha256(fk_poses),
                "grasp_pose_sha256": _array_sha256(grasp_pose),
                "lift_pose_sha256": _array_sha256(lift_pose),
                "all_samples_verified": True,
                "motion_authorized": False,
            },
            "collision_geometry": {
                "fr3_model": {
                    "path": str(fr3_path.resolve()),
                    "sha256": _file_sha256(fr3_path),
                },
                "adapter_mesh_sha256": adapter_sha,
                "closed_hand_link_count": 13,
                "closed_hand_link_manifest_sha256": hand_manifest_sha,
                "payload_mesh_sha256": geometry["sha256"],
                "scene": {
                    "path": str(scene_path.resolve()),
                    "sha256": _file_sha256(scene_path),
                    "reference_frame": "robot_base",
                    "payload_removed": True,
                    "support_contact_region_excluded": True,
                },
                "required_components": [
                    "fr3",
                    "adapter",
                    "closed_rh56_13_links",
                    "payload",
                    "scene",
                ],
                "all_geometry_included": True,
            },
            "collision_report": {
                "path": str(collision_report_path.resolve()),
                "sha256": "0" * 64,
                "report_id": "collision-report-test",
            },
        },
        "policies": policies,
        "collision_backend": {
            "name": "fake-hppfcl",
            "version": "1",
            "implementation_sha256": "d" * 64,
            "configuration_sha256": "e" * 64,
            "error": "",
        },
        "checks": checks,
        "decision": {
            "passed": True,
            "all_required_checks_passed": True,
            "precondition_failures": [],
            "reasons": [],
            "round_trip_loaded_lift_eligible": True,
            "runtime_conditions_satisfied_in_artifact": False,
            "motion_authorized": False,
            "meaning": "offline loaded-lift round-trip evidence only; never a motion command",
        },
    }
    report = {
        "schema_version": 1,
        "artifact_type": loaded_lift.COLLISION_REPORT_ARTIFACT_TYPE,
        "report_id": "collision-report-test",
        "audit_generator_implementation_sha256": artifact["audit_generator"][
            "implementation_sha256"
        ],
        "collision_backend": artifact["collision_backend"],
        "round_trip_samples_sha256": artifact["bindings"]["joint_path"][
            "round_trip_sha256"
        ],
        "joint_path_binding_sha256": _json_sha256(
            artifact["bindings"]["joint_path"]
        ),
        "fk_manifest_binding_sha256": _json_sha256(
            artifact["bindings"]["fk_manifest"]
        ),
        "payload_binding_sha256": artifact["bindings"]["payload"][
            "binding_sha256"
        ],
        "collision_geometry_binding_sha256": _json_sha256(
            artifact["bindings"]["collision_geometry"]
        ),
        "closed_hand_feedback_envelope_sha256": artifact["bindings"][
            "closed_hand"
        ]["feedback_envelope_sha256"],
        "policies_sha256": _json_sha256(artifact["policies"]),
        "checks": artifact["checks"],
        "authoritative": True,
        "motion_authorized": False,
    }
    collision_report_path.write_text(json.dumps(report), encoding="utf-8")
    artifact["bindings"]["collision_report"]["sha256"] = _file_sha256(
        collision_report_path
    )
    return _reseal(artifact), hand_manifest, grasp_q, grasp_pose, targets, q12


def _commissioned_config(artifact, targets):
    approval = artifact["bindings"]["adapter_material_approval"]
    retention = artifact["bindings"]["payload"]["retention_approval"]
    mass_properties = artifact["bindings"]["payload"]["mass_properties"]
    collision_report = artifact["bindings"]["collision_report"]
    audit_generator = artifact["audit_generator"]
    fk_manifest = artifact["bindings"]["fk_manifest"]
    collision_backend = artifact["collision_backend"]
    fixture_dir = Path(artifact["bindings"]["control_profile"]["path"]).parent
    recovery_executable = _write(
        fixture_dir / "loaded_lift_recovery",
        b"#!/bin/sh\n# Offline test fixture; never invoked.\nexit 0\n",
    )
    recovery_executable.chmod(0o755)
    recovery_procedure_path = fixture_dir / "loaded-lift-recovery.json"
    recovery_procedure_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": (
                    loaded_lift.LOADED_RECOVERY_PROCEDURE_ARTIFACT_TYPE
                ),
                "procedure_id": "reviewed-recovery-test",
                "mode": MODE,
                "executable": {
                    "path": str(recovery_executable.resolve()),
                    "sha256": _file_sha256(recovery_executable),
                    "interface": "loaded_lift_recovery_cli_v1",
                },
                "required_stage_order": [
                    "read_and_verify_loaded_state",
                    "preserve_numeric_closed_hand_hold",
                    "stop_franka_in_place",
                    "reverse_exact_audited_setdown_path",
                    "verify_joint_and_eef_settle",
                    "release_payload",
                    "clear_payload_dynamics",
                ],
                "arm_fault_action": "stop_in_place_without_hand_release",
                "hand_fault_action": (
                    "preserve_last_verified_numeric_closed_hold"
                ),
                "setdown_path": "exact_audited_reverse_loaded_lift_path",
                "release_gate": (
                    "joint_and_eef_settle_verified_after_setdown"
                ),
                "operator_immediate_stop_required": True,
                "procedure_reviewed": True,
                "motion_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    return {
        "franka": {"expected_F_T_EE": np.eye(4).tolist()},
        "tool": {
            "low_speed_unloaded_commissioning_only": False,
            "installed_collision_model_verified": True,
            "material": "Al-6061-T6",
        },
        "inspire": {"six_axis_coupled_closure_commissioned": True},
        "loaded_lift": {
            "commissioned": True,
            "round_trip_setdown_commissioned": True,
            "load_rated_adapter_approved": True,
            "material_approval_path": approval["path"],
            "material_approval_sha256": approval["sha256"],
            "retention_approval_path": retention["path"],
            "retention_approval_sha256": retention["sha256"],
            "mass_properties_approval_path": mass_properties["path"],
            "mass_properties_approval_sha256": mass_properties["sha256"],
            "fk_manifest_path": fk_manifest["path"],
            "fk_manifest_sha256": fk_manifest["sha256"],
            "collision_report_path": collision_report["path"],
            "collision_report_sha256": collision_report["sha256"],
            "loaded_recovery_procedure_path": str(
                recovery_procedure_path.resolve()
            ),
            "loaded_recovery_procedure_sha256": _file_sha256(
                recovery_procedure_path
            ),
            "audit_generator": dict(audit_generator),
            "fk_backend": {
                "name": fk_manifest["backend_name"],
                "version": fk_manifest["backend_version"],
                "implementation_sha256": fk_manifest[
                    "implementation_sha256"
                ],
            },
            "collision_backend": {
                key: collision_backend[key]
                for key in (
                    "name",
                    "version",
                    "implementation_sha256",
                    "configuration_sha256",
                )
            },
            "mass_properties_backend": {
                "name": mass_properties["backend_name"],
                "version": mass_properties["backend_version"],
                "implementation_sha256": mass_properties[
                    "implementation_sha256"
                ],
            },
            "approved_payload_mass_limit_kg": 0.5,
            "commissioned_closed_hand_targets": [
                targets.astype(int).tolist()
            ],
        },
    }


def _write_bound_artifact(tmp_path, artifact, config):
    config_path = Path(artifact["bindings"]["control_profile"]["path"])
    config_path.write_text(json.dumps(config), encoding="utf-8")
    artifact["bindings"]["control_profile"]["sha256"] = _file_sha256(
        config_path
    )
    artifact = _reseal(artifact)
    audit_path = tmp_path / "lift.json"
    audit_path.write_text(json.dumps(artifact), encoding="utf-8")
    return artifact, audit_path, config_path


def _install_binding_replay_fakes(
    monkeypatch,
    tmp_path,
    *,
    hand_manifest,
    grasp_q,
    grasp_pose,
    targets,
    q12,
):
    base = {
        "schema_version": 2,
        "artifact_type": INSTALLED_TOOL_ARTIFACT_TYPE,
        "mode": "loaded_grasp",
        "artifact_sha256": "b" * 64,
        "decision": {"passed": True},
        "bindings": {
            "snapshot": {"selected_candidate": {"index": 51}},
            "joint_path": {
                "waypoints": [{"name": "grasp", "q_rad": grasp_q.tolist()}]
            },
            "execution_plan": {"grasp_pose_base_EE": grasp_pose.tolist()},
            "mount_transform": {"T_EE_hand": np.eye(4).tolist()},
            "hand_model": dict(
                hand_manifest,
                closure_actuator_targets=targets.tolist(),
                actuator_mapping_provenance={
                    "driver_workbook_path": str(tmp_path / "driver.xls")
                },
            ),
            "adapter": {"sha256": "a" * 64},
        },
    }
    monkeypatch.setattr(
        loaded_lift,
        "load_installed_tool_audit",
        lambda *_args, **_kwargs: base,
    )
    import anydex_pipeline.rh56_actuator_mapping as actuator_mapping

    class FakeMapper:
        def __init__(self, _path):
            pass

        def to_joint_positions_rad(self, registers):
            result = q12.copy()
            for axis, value in enumerate(registers):
                result[2 * axis : 2 * axis + 2] += (
                    float(value) - float(targets[axis])
                ) * 0.0004
            return result

    monkeypatch.setattr(actuator_mapping, "OfficialRH56ActuatorMapper", FakeMapper)
    return base


def _bind_fixture(
    monkeypatch,
    tmp_path,
    *,
    artifact,
    audit_path,
    config,
    config_path,
    hand_manifest,
    grasp_q,
    grasp_pose,
    targets,
    q12,
    verify_files=True,
):
    _install_binding_replay_fakes(
        monkeypatch,
        tmp_path,
        hand_manifest=hand_manifest,
        grasp_q=grasp_q,
        grasp_pose=grasp_pose,
        targets=targets,
        q12=q12,
    )
    return bind_loaded_lift_audit(
        audit_path,
        base_loaded_grasp_audit_path=Path(
            artifact["bindings"]["base_loaded_grasp"]["path"]
        ),
        config=config,
        config_path=config_path,
        expected_selected_candidate_index=51,
        expected_grasp_q_rad=grasp_q,
        expected_grasp_pose_base_EE=grasp_pose,
        expected_closed_hand_targets=targets,
        verify_files=verify_files,
    )


def test_strict_loaded_lift_artifact_validates_and_exposes_contract(tmp_path):
    artifact, _hand, _q, _pose, _targets, _q12 = _make_artifact(tmp_path)
    assert validate_loaded_lift_audit(
        artifact, verify_files=True, require_pass=True
    ) is artifact


def test_ordinary_installed_tool_artifact_can_never_unlock_loaded_lift():
    ordinary = {
        "schema_version": 2,
        "artifact_type": INSTALLED_TOOL_ARTIFACT_TYPE,
        "mode": "loaded_grasp",
        "created_at_s": 1.0,
        "bindings": {},
        "policies": {},
        "collision_backend": {},
        "checks": [],
        "decision": {},
        "artifact_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="schema_version|artifact_type|audit_generator"):
        validate_loaded_lift_audit(ordinary)


def test_round_trip_must_be_exact_reverse_of_lift_path(tmp_path):
    artifact, *_ = _make_artifact(tmp_path)
    artifact["bindings"]["joint_path"]["round_trip_samples_rad"][-1][0] += 0.001
    artifact = _reseal(artifact)
    with pytest.raises(ValueError, match="exact audited reverse path"):
        validate_loaded_lift_audit(artifact)


def test_non_authoritative_or_incomplete_check_fails_closed(tmp_path):
    artifact, *_ = _make_artifact(tmp_path)
    artifact["checks"][0]["authoritative"] = False
    artifact = _reseal(artifact)
    with pytest.raises(ValueError, match="failures are inconsistent"):
        validate_loaded_lift_audit(artifact)


@pytest.mark.parametrize("replacement", [True, "0.0"])
def test_joint_vectors_reject_boolean_and_string_coercion(tmp_path, replacement):
    artifact, *_ = _make_artifact(tmp_path)
    artifact["bindings"]["joint_path"]["waypoints"][0]["q_rad"][0] = replacement
    artifact = _reseal(artifact)
    with pytest.raises(ValueError, match="JSON numbers"):
        validate_loaded_lift_audit(artifact)


def test_sha_fields_must_be_strings(tmp_path):
    artifact, *_ = _make_artifact(tmp_path)
    artifact["bindings"]["base_loaded_grasp"]["artifact_sha256"] = int("1" * 64)
    artifact = _reseal(artifact)
    with pytest.raises(ValueError, match="SHA-256 hex string"):
        validate_loaded_lift_audit(artifact)


def test_boolean_sample_indices_cannot_equal_integer_indices(tmp_path):
    artifact, *_ = _make_artifact(tmp_path)
    artifact["checks"][0]["tested_sample_indices"][0:2] = [False, True]
    artifact = _reseal(artifact)
    with pytest.raises(ValueError, match="failures are inconsistent"):
        validate_loaded_lift_audit(artifact)


def test_base_file_hash_is_checked_in_verify_files_mode(tmp_path):
    artifact, *_ = _make_artifact(tmp_path)
    Path(artifact["bindings"]["base_loaded_grasp"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="base loaded-grasp file"):
        validate_loaded_lift_audit(artifact, verify_files=True)


def test_closed_hand_fk_must_equal_base_fk_binding(tmp_path):
    artifact, *_ = _make_artifact(tmp_path)
    artifact["bindings"]["closed_hand"]["closure_fk_sha256"] = "8" * 64
    artifact = _reseal(artifact)
    with pytest.raises(ValueError, match="closed-hand binding"):
        validate_loaded_lift_audit(artifact)


def test_payload_attachment_chain_is_recomputed(tmp_path):
    artifact, *_ = _make_artifact(tmp_path)
    attachment = artifact["bindings"]["payload"]["attachment"]
    attachment["T_F_payload"][0][3] = 0.01
    attachment["T_F_payload_sha256"] = _array_sha256(attachment["T_F_payload"])
    payload = artifact["bindings"]["payload"]
    payload_unsigned = {
        key: payload[key]
        for key in (
            "geometry",
            "dynamics",
            "attachment",
            "mass_properties",
            "retention_approval",
        )
    }
    payload["binding_sha256"] = _json_sha256(payload_unsigned)
    artifact = _reseal(artifact)
    with pytest.raises(ValueError, match="attachment transform binding"):
        validate_loaded_lift_audit(artifact)


def test_mass_properties_must_bind_payload_dynamics(tmp_path):
    artifact, *_ = _make_artifact(tmp_path)
    payload = artifact["bindings"]["payload"]
    payload["mass_properties"]["mass_kg"] = 0.21
    payload_unsigned = {
        key: payload[key]
        for key in (
            "geometry",
            "dynamics",
            "attachment",
            "mass_properties",
            "retention_approval",
        )
    }
    payload["binding_sha256"] = _json_sha256(payload_unsigned)
    artifact = _reseal(artifact)
    with pytest.raises(ValueError, match="mass-properties binding"):
        validate_loaded_lift_audit(artifact)


def test_time_law_must_respect_material_velocity_limit(tmp_path):
    artifact, *_ = _make_artifact(tmp_path)
    law = artifact["bindings"]["joint_path"]["execution_time_law"]
    law["max_joint_velocity_rad_s"] = 0.06
    artifact = _reseal(artifact)
    with pytest.raises(ValueError, match="time law exceeds"):
        validate_loaded_lift_audit(artifact)


def test_time_law_duration_must_bound_peak_acceleration(tmp_path):
    artifact, *_ = _make_artifact(tmp_path)
    law = artifact["bindings"]["joint_path"]["execution_time_law"]
    law["min_segment_duration_s"] = 0.2
    artifact = _reseal(artifact)
    with pytest.raises(ValueError, match="velocity/acceleration caps"):
        validate_loaded_lift_audit(artifact)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("position_m", 0.011),
        ("orientation_rad", 0.101),
        ("linear_speed_m_s", 0.021),
        ("angular_speed_rad_s", 0.101),
        ("stable_seconds", 0.199),
        ("stable_seconds", 2.001),
    ],
)
def test_loaded_settle_tolerances_are_conservatively_bounded(
    tmp_path, field, value
):
    artifact, *_ = _make_artifact(tmp_path)
    artifact["policies"]["settle"][field] = value
    artifact = _reseal(artifact)
    with pytest.raises(ValueError, match="settle"):
        validate_loaded_lift_audit(artifact)


def test_non_string_bound_path_cannot_be_coerced(tmp_path):
    artifact, *_ = _make_artifact(tmp_path)
    payload = artifact["bindings"]["payload"]
    payload["geometry"]["path"] = True
    payload_unsigned = {
        key: payload[key]
        for key in (
            "geometry",
            "dynamics",
            "attachment",
            "mass_properties",
            "retention_approval",
        )
    }
    payload["binding_sha256"] = _json_sha256(payload_unsigned)
    artifact = _reseal(artifact)
    with pytest.raises(ValueError, match="path must be a non-empty string"):
        validate_loaded_lift_audit(artifact)


def test_collision_report_binds_inline_fk_manifest(tmp_path):
    artifact, *_ = _make_artifact(tmp_path)
    fk = artifact["bindings"]["fk_manifest"]
    fk["outbound_ee_poses_base"][1][0][3] += 0.001
    fk["outbound_ee_poses_sha256"] = _array_sha256(
        fk["outbound_ee_poses_base"]
    )
    artifact = _reseal(artifact)
    with pytest.raises(ValueError, match="collision report differs"):
        validate_loaded_lift_audit(
            artifact, verify_files=True, require_pass=True
        )


def test_collision_report_binds_collision_policies(tmp_path):
    artifact, *_ = _make_artifact(tmp_path)
    artifact["policies"]["robot_clearance_margin_m"] = 0.003
    artifact = _reseal(artifact)
    with pytest.raises(ValueError, match="collision report differs"):
        validate_loaded_lift_audit(
            artifact, verify_files=True, require_pass=True
        )


@pytest.mark.parametrize(
    "field",
    [
        "material_approval_path",
        "material_approval_sha256",
        "retention_approval_path",
        "retention_approval_sha256",
        "mass_properties_approval_path",
        "mass_properties_approval_sha256",
        "fk_manifest_path",
        "fk_manifest_sha256",
        "collision_report_path",
        "collision_report_sha256",
        "loaded_recovery_procedure_path",
        "loaded_recovery_procedure_sha256",
        "audit_generator",
        "fk_backend",
        "collision_backend",
        "mass_properties_backend",
    ],
)
def test_every_external_authority_pin_is_mandatory(tmp_path, field):
    artifact, _hand, _q, _pose, targets, _q12 = _make_artifact(tmp_path)
    config = _commissioned_config(artifact, targets)
    del config["loaded_lift"][field]
    blockers = loaded_lift_profile_blockers(
        config, Path(artifact["bindings"]["control_profile"]["path"])
    )
    assert blockers, "missing {!r} unexpectedly unlocked loaded lift".format(field)


def test_recovery_contract_cannot_release_before_verified_setdown(tmp_path):
    artifact, _hand, _q, _pose, targets, _q12 = _make_artifact(tmp_path)
    config = _commissioned_config(artifact, targets)
    lift = config["loaded_lift"]
    recovery_path = Path(lift["loaded_recovery_procedure_path"])
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    recovery["hand_fault_action"] = "release_payload_immediately"
    recovery_path.write_text(json.dumps(recovery), encoding="utf-8")
    lift["loaded_recovery_procedure_sha256"] = _file_sha256(recovery_path)
    blockers = loaded_lift_profile_blockers(
        config, Path(artifact["bindings"]["control_profile"]["path"])
    )
    assert any("recovery procedure replay failed" in item for item in blockers)


def test_checked_in_pla_profile_is_explicitly_locked():
    root = Path(__file__).resolve().parents[1]
    path = root / "configs/fr3_rh56_v7_commissioning.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    blockers = loaded_lift_profile_blockers(config, path)
    assert any("unloaded" in item for item in blockers)
    assert any("loaded_lift" in item for item in blockers)
    binding = bind_loaded_lift_audit(
        None,
        base_loaded_grasp_audit_path=None,
        config=config,
        config_path=path,
    )
    assert not binding.passed
    assert any("loaded-lift round-trip audit" in item for item in binding.blockers)


def test_binding_replays_base_and_exposes_executor_inputs(tmp_path, monkeypatch):
    config_path = _write(tmp_path / "control.json", b"{}\n")
    base_path = _write(tmp_path / "base-loaded.json", b"base bytes\n")
    artifact, hand_manifest, grasp_q, grasp_pose, targets, q12 = _make_artifact(
        tmp_path, config_path=config_path, base_path=base_path
    )
    config = _commissioned_config(artifact, targets)
    artifact, audit_path, config_path = _write_bound_artifact(
        tmp_path, artifact, config
    )
    binding = _bind_fixture(
        monkeypatch,
        tmp_path,
        artifact=artifact,
        audit_path=audit_path,
        config=config,
        config_path=config_path,
        hand_manifest=hand_manifest,
        grasp_q=grasp_q,
        grasp_pose=grasp_pose,
        targets=targets,
        q12=q12,
        verify_files=False,
    )
    assert binding.passed, binding.blockers
    assert binding.artifact_sha256 == artifact["artifact_sha256"]
    assert [name for name, _q in binding.lift_waypoints] == [
        "grasp",
        "lift_transit_0",
        "lift",
    ]
    assert binding.payload_mass_kg == pytest.approx(0.2)
    assert binding.payload_F_x_Cload_m.tolist() == [0.0, 0.0, 0.03]
    assert binding.payload_inertia_kg_m2.shape == (3, 3)
    assert binding.minimum_contact_axes == 3
    assert binding.max_q_tracking_error_rad == pytest.approx(0.002)
    assert binding.settle_tolerances["stable_seconds"] == pytest.approx(0.3)


def test_unpinned_self_declared_authority_cannot_unlock_binding(
    tmp_path, monkeypatch
):
    config_path = _write(tmp_path / "control.json", b"{}\n")
    base_path = _write(tmp_path / "base-loaded.json", b"base bytes\n")
    artifact, hand_manifest, grasp_q, grasp_pose, targets, q12 = _make_artifact(
        tmp_path, config_path=config_path, base_path=base_path
    )
    config = _commissioned_config(artifact, targets)
    for field in (
        "collision_report_path",
        "collision_report_sha256",
        "audit_generator",
        "fk_backend",
        "collision_backend",
        "mass_properties_backend",
    ):
        del config["loaded_lift"][field]
    artifact, audit_path, config_path = _write_bound_artifact(
        tmp_path, artifact, config
    )
    binding = _bind_fixture(
        monkeypatch,
        tmp_path,
        artifact=artifact,
        audit_path=audit_path,
        config=config,
        config_path=config_path,
        hand_manifest=hand_manifest,
        grasp_q=grasp_q,
        grasp_pose=grasp_pose,
        targets=targets,
        q12=q12,
        verify_files=False,
    )
    assert not binding.passed
    assert any("collision report" in item for item in binding.blockers)
    assert any("backend" in item for item in binding.blockers)


def test_collision_report_file_tamper_fails_binding(tmp_path):
    config_path = _write(tmp_path / "control.json", b"{}\n")
    base_path = _write(tmp_path / "base-loaded.json", b"base bytes\n")
    artifact, _hand, _q, _pose, targets, _q12 = _make_artifact(
        tmp_path, config_path=config_path, base_path=base_path
    )
    config = _commissioned_config(artifact, targets)
    artifact, audit_path, config_path = _write_bound_artifact(
        tmp_path, artifact, config
    )
    report_path = Path(artifact["bindings"]["collision_report"]["path"])
    report_path.write_bytes(report_path.read_bytes() + b"\n")
    binding = bind_loaded_lift_audit(
        audit_path,
        base_loaded_grasp_audit_path=base_path,
        config=config,
        config_path=config_path,
        verify_files=True,
    )
    assert not binding.passed
    assert any(
        "collision report file no longer matches" in item
        for item in binding.blockers
    )


def test_self_consistent_fake_collision_backend_stays_profile_locked(
    tmp_path, monkeypatch
):
    config_path = _write(tmp_path / "control.json", b"{}\n")
    base_path = _write(tmp_path / "base-loaded.json", b"base bytes\n")
    artifact, hand_manifest, grasp_q, grasp_pose, targets, q12 = _make_artifact(
        tmp_path, config_path=config_path, base_path=base_path
    )
    config = _commissioned_config(artifact, targets)
    artifact["collision_backend"]["name"] = "self-declared-backend"
    artifact["collision_backend"]["implementation_sha256"] = "f" * 64
    report_path = Path(artifact["bindings"]["collision_report"]["path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["collision_backend"] = artifact["collision_backend"]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    artifact["bindings"]["collision_report"]["sha256"] = _file_sha256(
        report_path
    )
    config["loaded_lift"]["collision_report_sha256"] = _file_sha256(
        report_path
    )
    artifact, audit_path, config_path = _write_bound_artifact(
        tmp_path, artifact, config
    )
    binding = _bind_fixture(
        monkeypatch,
        tmp_path,
        artifact=artifact,
        audit_path=audit_path,
        config=config,
        config_path=config_path,
        hand_manifest=hand_manifest,
        grasp_q=grasp_q,
        grasp_pose=grasp_pose,
        targets=targets,
        q12=q12,
        verify_files=True,
    )
    assert not binding.passed
    assert any(
        "collision backend identity differs" in item
        for item in binding.blockers
    )


def test_audit_generator_identity_mismatch_stays_profile_locked(
    tmp_path, monkeypatch
):
    config_path = _write(tmp_path / "control.json", b"{}\n")
    base_path = _write(tmp_path / "base-loaded.json", b"base bytes\n")
    artifact, hand_manifest, grasp_q, grasp_pose, targets, q12 = _make_artifact(
        tmp_path, config_path=config_path, base_path=base_path
    )
    config = _commissioned_config(artifact, targets)
    artifact["audit_generator"]["name"] = "unreviewed-generator-name"
    artifact, audit_path, config_path = _write_bound_artifact(
        tmp_path, artifact, config
    )
    binding = _bind_fixture(
        monkeypatch,
        tmp_path,
        artifact=artifact,
        audit_path=audit_path,
        config=config,
        config_path=config_path,
        hand_manifest=hand_manifest,
        grasp_q=grasp_q,
        grasp_pose=grasp_pose,
        targets=targets,
        q12=q12,
        verify_files=True,
    )
    assert not binding.passed
    assert any(
        "audit generator identity differs" in item for item in binding.blockers
    )
