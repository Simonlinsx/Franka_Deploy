from __future__ import annotations

import builtins
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "move_calibration_pose.py"
SPEC = importlib.util.spec_from_file_location("joint_recovery_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
motion = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = motion
SPEC.loader.exec_module(motion)


def _rotation_x(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    result = np.eye(4)
    result[:3, :3] = [
        [1.0, 0.0, 0.0],
        [0.0, math.cos(angle), -math.sin(angle)],
        [0.0, math.sin(angle), math.cos(angle)],
    ]
    return result


def _base_plan(*, authorized: bool = True) -> dict:
    start_q = np.asarray(
        [0.0980, 0.2184, 0.2079, -0.9902, 0.1859, 1.1607, -0.1429]
    )
    delta = np.deg2rad(
        [0.872804, -2.957283, 0.0, -4.027727, -4.732793, 2.077167, -1.008233]
    )
    start_T = np.eye(4)
    start_T[:3, 3] = [0.500, 0.170, 0.731]
    target_T = _rotation_x(5.0)
    target_T[:3, 3] = [0.508, 0.170, 0.731]
    plan = {
        "schema_version": 1,
        "kind": motion.JOINT_RECOVERY_PLAN_KIND,
        "status": (
            motion.JOINT_RECOVERY_RUN_STATUS
            if authorized
            else "awaiting_explicit_motion_authorization"
        ),
        "session_slug": "fr3-test-bounded-joint-recovery",
        "topology": {"type": "eye_to_hand", "camera_mount": "fixed_external"},
        "hardware": {
            "robot": {
                "ip": "172.16.0.2",
                "expected_F_T_EE": np.eye(4).tolist(),
                "expected_m_ee_kg": 0.607,
                "expected_F_x_Cee_m": [0.0, 0.0, 0.076],
                "expected_I_ee_kg_m2": np.diag(
                    [0.00151, 0.00169, 0.000442]
                ).tolist(),
                "expected_external_load": {
                    "m_load_kg": 0.0,
                    "F_x_Cload_m": [0.0, 0.0, 0.0],
                    "I_load_kg_m2": np.zeros((3, 3)).tolist(),
                },
            }
        },
        "safety": {
            "motion_authorized": authorized,
            "emergency_stop_reachable": True,
            "full_board_bracket_and_cable_swept_volume_clear": True,
            "target_rigid_confirmed": True,
            "camera_fixed_confirmed": True,
            "workspace_bounds_base_m": {
                "x": [0.430, 0.550],
                "y": [0.110, 0.230],
                "z": [0.696, 0.790],
            },
            "minimum_eef_z_m": 0.701,
            "expected_start_q_tolerance_rad": motion.START_Q_TOLERANCE_RAD,
            "maximum_start_dq_rad_s": motion.MAX_START_DQ_RAD_S,
            "joint_arrival_tolerance_rad": motion.JOINT_ENDPOINT_TOLERANCE_RAD,
            "start_pose_translation_tolerance_m": (
                motion.START_POSE_TRANSLATION_TOLERANCE_M
            ),
            "start_pose_rotation_tolerance_deg": math.degrees(
                motion.START_POSE_ROTATION_TOLERANCE_RAD
            ),
            "endpoint_pose_translation_tolerance_m": (
                motion.ENDPOINT_POSE_TRANSLATION_TOLERANCE_M
            ),
            "endpoint_pose_rotation_tolerance_deg": math.degrees(
                motion.ENDPOINT_POSE_ROTATION_TOLERANCE_RAD
            ),
            "endpoint_max_dq_rad_s": motion.ENDPOINT_MAX_DQ_RAD_S,
            "endpoint_consecutive_samples": motion.ENDPOINT_CONSECUTIVE_SAMPLES,
            "endpoint_poll_s": motion.ENDPOINT_POLL_S,
            "maximum_dynamic_segments": motion.JOINT_RECOVERY_MAX_DYNAMIC_SEGMENTS,
            "joint_time_law": motion.JOINT_RECOVERY_TIME_LAW,
            "maximum_per_joint_delta_rad": math.radians(5.0),
            "maximum_joint_delta_norm_rad": math.radians(9.0),
            "maximum_joint_velocity_rad_s": 0.020,
            "maximum_joint_acceleration_rad_s2": 0.010,
            "minimum_joint_duration_s": 8.0,
            "board_max_extent_from_eef_m": 0.30,
            "conservative_board_sweep_displacement_m": 0.036,
        },
        "recovery": {
            "id": "JR01",
            "expected_start_q_rad": start_q.tolist(),
            "commanded_delta_rad": delta.tolist(),
            "target_q_rad": (start_q + delta).tolist(),
            "expected_start_T_base_ee": start_T.tolist(),
            "expected_target_T_base_ee": target_T.tolist(),
            "kinematics_provenance": {
                "urdf_path": "franka_description/robots/fr3/fr3.urdf",
                "urdf_sha256": "a" * 64,
                "fk_implementation": "offline_fr3_fk.py",
                "fk_implementation_sha256": "b" * 64,
                "expected_start_fk_matches_plan": True,
                "expected_target_fk_matches_plan": True,
                "interpolation_sample_count": 101,
                "all_interpolation_samples_within_workspace": True,
                "minimum_interpolation_eef_z_m": 0.731,
                "maximum_interpolation_eef_center_displacement_m": 0.009,
                "maximum_interpolation_eef_rotation_deg": 5.0,
            },
        },
    }
    if authorized:
        plan["motion_authorization"] = {
            "explicit_user_authorization_recorded": True,
            "scope": motion.JOINT_RECOVERY_AUTHORIZATION_SCOPE,
            "recovery_id": "JR01",
            "source_text": "continue bounded reviewed calibration recovery",
            "consumed": False,
        }
    return plan


def _write_plan(tmp_path: Path, plan: dict) -> Path:
    path = tmp_path / "joint-recovery.yaml"
    path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    return path


class _FakeJointArm:
    def __init__(
        self,
        start_q: np.ndarray,
        start_pose: np.ndarray,
        target_pose: np.ndarray,
        *,
        validation_error: Exception | None = None,
        stop_error: Exception | None = None,
    ) -> None:
        self.q = np.asarray(start_q, dtype=np.float64).copy()
        self.pose = np.asarray(start_pose, dtype=np.float64).copy()
        self.target_pose = np.asarray(target_pose, dtype=np.float64).copy()
        self.validation_error = validation_error
        self.stop_error = stop_error
        self.validate_calls = 0
        self.move_targets: list[np.ndarray] = []
        self.stop_calls = 0
        self.last_control_loop_telemetry = None
        self.robot = SimpleNamespace(read_once=self._read_once)

    def _read_once(self):
        return SimpleNamespace(
            q=self.q.copy(),
            dq=np.zeros(7),
            O_T_EE=self.pose.copy(),
        )

    def _validate_state(self, _state, *, require_idle, enforce_success):
        assert require_idle is True
        assert enforce_success is False
        self.validate_calls += 1
        if self.validation_error is not None:
            raise self.validation_error

    def move_joints(self, target):
        self.move_targets.append(np.asarray(target, dtype=np.float64).copy())
        self.q = np.asarray(target, dtype=np.float64).copy()
        self.pose = self.target_pose.copy()

    def stop(self):
        self.stop_calls += 1
        if self.stop_error is not None:
            raise self.stop_error


def test_preview_is_hardware_free_and_exposes_one_use_receipt(tmp_path, monkeypatch):
    path = _write_plan(tmp_path, _base_plan())
    imported_hardware = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "pylibfranka" or name.startswith("anydex_pipeline"):
            imported_hardware.append(name)
            raise AssertionError("offline preview imported hardware")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    preview = motion.load_joint_recovery_preview(path)

    assert preview.run_authorized is True
    assert preview.authorization_receipt_exists is False
    assert imported_hardware == []
    assert not preview.authorization_receipt_path.exists()


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda plan: (
                plan["recovery"].update(
                    commanded_delta_rad=np.deg2rad([5.01, 0, 0, 0, 0, 0, 0]).tolist()
                ),
                plan["recovery"].update(
                    target_q_rad=(
                        np.asarray(plan["recovery"]["expected_start_q_rad"])
                        + np.deg2rad([5.01, 0, 0, 0, 0, 0, 0])
                    ).tolist()
                ),
            ),
            "per-joint delta",
        ),
        (
            lambda plan: (
                plan["recovery"].update(
                    commanded_delta_rad=np.deg2rad([4, 4, 4, 4, 4, 4, 0]).tolist()
                ),
                plan["recovery"].update(
                    target_q_rad=(
                        np.asarray(plan["recovery"]["expected_start_q_rad"])
                        + np.deg2rad([4, 4, 4, 4, 4, 4, 0])
                    ).tolist()
                ),
            ),
            "delta norm",
        ),
        (
            lambda plan: plan["recovery"].update(target_q_rad=[0.0] * 7),
            "must exactly equal",
        ),
        (
            lambda plan: plan["recovery"]["expected_target_T_base_ee"][2].__setitem__(3, 0.690),
            "outside hard workspace|below minimum",
        ),
        (
            lambda plan: plan["recovery"]["kinematics_provenance"].update(
                minimum_interpolation_eef_z_m=0.699
            ),
            "descends below",
        ),
        (
            lambda plan: plan["safety"].update(
                conservative_board_sweep_displacement_m=0.010
            ),
            "below computed",
        ),
        (
            lambda plan: plan["recovery"]["kinematics_provenance"].update(
                urdf_sha256="not-a-sha"
            ),
            "lowercase SHA-256",
        ),
        (
            lambda plan: plan["safety"].update(maximum_dynamic_segments=2),
            "must be exactly 1",
        ),
    ],
)
def test_joint_recovery_plan_mutations_fail_closed(tmp_path, mutate, match):
    plan = _base_plan()
    mutate(plan)
    path = _write_plan(tmp_path, plan)
    with pytest.raises(ValueError, match=match):
        motion.load_joint_recovery_preview(path)


def test_execution_moves_exact_vector_once_and_receipt_prevents_replay(tmp_path):
    plan = _base_plan()
    path = _write_plan(tmp_path, plan)
    preview = motion.load_joint_recovery_preview(path, for_run=True)
    arm = _FakeJointArm(
        preview.expected_start_q_rad,
        preview.expected_start_T_base_ee,
        preview.expected_target_T_base_ee,
    )

    result = motion.execute_joint_recovery(
        preview,
        preview.robot_ip,
        connector=lambda _preview, _ip: arm,
        link_preflight=lambda _ip: None,
        sleep=lambda _seconds: None,
    )

    assert len(arm.move_targets) == 1
    np.testing.assert_allclose(
        arm.move_targets[0],
        preview.expected_start_q_rad + preview.commanded_delta_rad,
        atol=0.0,
        rtol=0.0,
    )
    assert arm.validate_calls == 1 + motion.ENDPOINT_CONSECUTIVE_SAMPLES
    assert arm.stop_calls == 1
    assert result.authorization_receipt_path.exists()
    receipt = json.loads(result.authorization_receipt_path.read_text(encoding="utf-8"))
    assert receipt["authorization_consumed"] is True
    assert receipt["status"] == "succeeded"

    with pytest.raises(ValueError, match="already consumed"):
        motion.load_joint_recovery_preview(path, for_run=True)


def test_live_start_inside_tolerance_still_commands_hash_bound_absolute_target(tmp_path):
    path = _write_plan(tmp_path, _base_plan())
    preview = motion.load_joint_recovery_preview(path, for_run=True)
    live_q = preview.expected_start_q_rad.copy()
    live_q[0] += 0.5 * motion.START_Q_TOLERANCE_RAD
    arm = _FakeJointArm(
        live_q,
        preview.expected_start_T_base_ee,
        preview.expected_target_T_base_ee,
    )

    motion.execute_joint_recovery(
        preview,
        preview.robot_ip,
        connector=lambda _preview, _ip: arm,
        link_preflight=lambda _ip: None,
        sleep=lambda _seconds: None,
    )

    np.testing.assert_allclose(
        arm.move_targets[0], preview.planned_target_q_rad, atol=0.0, rtol=0.0
    )
    assert not np.allclose(
        arm.move_targets[0], live_q + preview.commanded_delta_rad, atol=1.0e-12
    )


def test_exact_start_q_mismatch_blocks_motion_but_consumes_authorization(tmp_path):
    path = _write_plan(tmp_path, _base_plan())
    preview = motion.load_joint_recovery_preview(path, for_run=True)
    wrong_q = preview.expected_start_q_rad.copy()
    wrong_q[0] += motion.START_Q_TOLERANCE_RAD + 1.0e-4
    arm = _FakeJointArm(
        wrong_q,
        preview.expected_start_T_base_ee,
        preview.expected_target_T_base_ee,
    )

    with pytest.raises(RuntimeError, match="exact authorized start"):
        motion.execute_joint_recovery(
            preview,
            preview.robot_ip,
            connector=lambda _preview, _ip: arm,
            link_preflight=lambda _ip: None,
            sleep=lambda _seconds: None,
        )

    assert arm.move_targets == []
    assert arm.stop_calls == 1
    receipt = json.loads(preview.authorization_receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["authorization_consumed"] is True


def test_idle_error_contact_gate_runs_before_motion_and_consumes(tmp_path):
    path = _write_plan(tmp_path, _base_plan())
    preview = motion.load_joint_recovery_preview(path, for_run=True)
    arm = _FakeJointArm(
        preview.expected_start_q_rad,
        preview.expected_start_T_base_ee,
        preview.expected_target_T_base_ee,
        validation_error=RuntimeError("Franka contact gate active"),
    )

    with pytest.raises(RuntimeError, match="contact gate"):
        motion.execute_joint_recovery(
            preview,
            preview.robot_ip,
            connector=lambda _preview, _ip: arm,
            link_preflight=lambda _ip: None,
            sleep=lambda _seconds: None,
        )

    assert arm.move_targets == []
    assert arm.stop_calls == 1
    assert preview.authorization_receipt_path.exists()


def test_plan_change_after_claim_blocks_motion_and_receipt_stays_consumed(tmp_path):
    plan = _base_plan()
    path = _write_plan(tmp_path, plan)
    preview = motion.load_joint_recovery_preview(path, for_run=True)
    arm = _FakeJointArm(
        preview.expected_start_q_rad,
        preview.expected_start_T_base_ee,
        preview.expected_target_T_base_ee,
    )

    def mutating_connector(_preview, _ip):
        changed = _base_plan()
        changed["session_slug"] = "changed-after-claim"
        path.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
        return arm

    with pytest.raises(RuntimeError, match="changed after FCI connection"):
        motion.execute_joint_recovery(
            preview,
            preview.robot_ip,
            connector=mutating_connector,
            link_preflight=lambda _ip: None,
            sleep=lambda _seconds: None,
        )

    assert arm.move_targets == []
    assert arm.stop_calls == 1
    assert preview.authorization_receipt_path.exists()


def test_consumed_field_blocks_run_even_without_receipt(tmp_path):
    plan = _base_plan()
    plan["motion_authorization"]["consumed"] = True
    path = _write_plan(tmp_path, plan)
    with pytest.raises(ValueError, match="consumed must be exactly false"):
        motion.load_joint_recovery_preview(path, for_run=True)
