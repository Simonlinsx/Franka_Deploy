from __future__ import annotations

import builtins
import importlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "move_calibration_pose.py"
SPEC = importlib.util.spec_from_file_location("move_calibration_pose_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
motion = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = motion
SPEC.loader.exec_module(motion)


def _base_plan(
    *,
    authorized: bool = False,
    pose_id: str = "T02",
    start_pose_id: str = "T01",
) -> dict:
    anchor = np.eye(4)
    anchor[:3, 3] = [0.5558, -0.0796, 0.6826]
    plan = {
        "schema_version": 1,
        "kind": "franka_eye_to_hand_commissioning_plan",
        "status": (
            motion.RUN_STATUS
            if authorized
            else "awaiting_explicit_motion_authorization"
        ),
        "session_slug": "fr3-test-session",
        "topology": {"type": "eye_to_hand", "camera_mount": "fixed_external"},
        "hardware": {
            "robot": {
                "ip": "172.16.0.2",
                "expected_F_T_EE": np.eye(4).tolist(),
                "expected_m_ee_kg": 0.607,
                "expected_F_x_Cee_m": [0.0, 0.0, 0.076],
                "expected_I_ee_kg_m2": np.diag([0.00151, 0.00169, 0.000442]).tolist(),
                "expected_external_load": {
                    "m_load_kg": 0.0,
                    "F_x_Cload_m": [0.0, 0.0, 0.0],
                    "I_load_kg_m2": np.zeros((3, 3)).tolist(),
                },
            }
        },
        "reference_pose": {
            "xyz_base_m": anchor[:3, 3].tolist(),
            "T_base_ee": anchor.tolist(),
        },
        "safety": {
            "motion_authorized": authorized,
            "workspace_bounds_base_m": {
                "x": [0.495, 0.617],
                "y": [-0.141, -0.018],
                "z": [0.620, 0.744],
            },
            "minimum_eef_z_m": 0.620,
            "maximum_translation_step_m": 0.075,
            "maximum_rotation_step_deg": 20.0,
            "maximum_translation_norm_from_anchor_m": 0.050,
            "maximum_rotation_angle_from_anchor_deg": 16.0,
            "maximum_translation_velocity_m_s": 0.015,
            "maximum_translation_acceleration_m_s2": 0.030,
            "maximum_angular_velocity_rad_s": 0.030,
            "maximum_angular_acceleration_rad_s2": 0.060,
            "cartesian_endpoint_hold_s": motion.CARTESIAN_ENDPOINT_HOLD_S,
            "cartesian_pose_controller_mode": motion.CARTESIAN_POSE_CONTROLLER_MODE,
            "settle_time_s": 1.0,
        },
        "training_poses": [
            {
                "id": "T01",
                "xyz_offset_base_m": [0.0, 0.0, 0.0],
                "rotation_vector_eef_deg": [0.0, 0.0, 0.0],
            },
            {
                "id": "T02",
                "xyz_offset_base_m": [0.02, 0.0, 0.0],
                "rotation_vector_eef_deg": [0.0, 0.0, 10.0],
            },
        ],
        "holdout_poses": [
            {
                "id": "H01",
                "xyz_offset_base_m": [-0.02, 0.02, 0.03],
                "rotation_vector_eef_deg": [10.0, 0.0, 0.0],
            }
        ],
    }
    if authorized:
        plan["motion_authorization"] = {
            "explicit_user_authorization_recorded": True,
            "scope": motion.AUTHORIZATION_SCOPE,
            "pose_id": pose_id,
            "start_pose_id": start_pose_id,
        }
    return plan


def _write_plan(tmp_path: Path, plan: dict, name: str = "plan.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    return path


def _joint_recovery_plan(tmp_path: Path, *, authorized: bool = True) -> dict:
    artifacts = {}
    for name, content in (
        ("fr3.urdf", "reviewed urdf\n"),
        ("fk.py", "# reviewed offline FK implementation\n"),
        ("audit.json", '{"status":"pass"}\n'),
    ):
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        artifacts[name] = (path, motion._sha256_file(path))

    start_q = np.asarray([0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])
    delta = np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    start_T = np.eye(4)
    start_T[:3, 3] = [0.50, 0.17, 0.73]
    target_T = start_T.copy()
    target_T[0, 3] += 0.001
    return {
        "schema_version": 1,
        "kind": motion.JOINT_RECOVERY_PLAN_KIND,
        "status": (
            motion.JOINT_RECOVERY_RUN_STATUS
            if authorized
            else "awaiting_explicit_motion_authorization"
        ),
        "session_slug": "joint-recovery-test",
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
                "x": [0.43, 0.55],
                "y": [0.11, 0.23],
                "z": [0.70, 0.79],
            },
            "minimum_eef_z_m": 0.70,
            "expected_start_q_tolerance_rad": motion.START_Q_TOLERANCE_RAD,
            "maximum_start_dq_rad_s": motion.MAX_START_DQ_RAD_S,
            "joint_arrival_tolerance_rad": motion.JOINT_ENDPOINT_TOLERANCE_RAD,
            "start_pose_translation_tolerance_m": motion.START_POSE_TRANSLATION_TOLERANCE_M,
            "start_pose_rotation_tolerance_deg": float(
                np.degrees(motion.START_POSE_ROTATION_TOLERANCE_RAD)
            ),
            "endpoint_pose_translation_tolerance_m": motion.ENDPOINT_POSE_TRANSLATION_TOLERANCE_M,
            "endpoint_pose_rotation_tolerance_deg": float(
                np.degrees(motion.ENDPOINT_POSE_ROTATION_TOLERANCE_RAD)
            ),
            "endpoint_max_dq_rad_s": motion.ENDPOINT_MAX_DQ_RAD_S,
            "endpoint_poll_s": motion.ENDPOINT_POLL_S,
            "endpoint_consecutive_samples": motion.ENDPOINT_CONSECUTIVE_SAMPLES,
            "maximum_dynamic_segments": motion.JOINT_RECOVERY_MAX_DYNAMIC_SEGMENTS,
            "joint_time_law": motion.JOINT_RECOVERY_TIME_LAW,
            "maximum_per_joint_delta_rad": motion.MAX_SINGLE_JOINT_DELTA_RAD,
            "maximum_joint_delta_norm_rad": motion.MAX_JOINT_DELTA_NORM_RAD,
            "maximum_joint_velocity_rad_s": motion.MAX_JOINT_RECOVERY_SPEED_RAD_S,
            "maximum_joint_acceleration_rad_s2": motion.MAX_JOINT_RECOVERY_ACCELERATION_RAD_S2,
            "minimum_joint_duration_s": motion.MIN_JOINT_RECOVERY_DURATION_S,
            "board_max_extent_from_eef_m": 0.30,
            "conservative_board_sweep_displacement_m": 0.002,
        },
        "recovery": {
            "id": "JR01",
            "expected_start_q_rad": start_q.tolist(),
            "target_q_rad": (start_q + delta).tolist(),
            "commanded_delta_rad": delta.tolist(),
            "expected_start_T_base_ee": start_T.tolist(),
            "expected_target_T_base_ee": target_T.tolist(),
            "kinematics_provenance": {
                "urdf_path": str(artifacts["fr3.urdf"][0]),
                "urdf_sha256": artifacts["fr3.urdf"][1],
                "fk_implementation": str(artifacts["fk.py"][0]),
                "fk_implementation_sha256": artifacts["fk.py"][1],
                "audit_artifact_path": str(artifacts["audit.json"][0]),
                "audit_artifact_sha256": artifacts["audit.json"][1],
                "expected_start_fk_matches_plan": True,
                "expected_target_fk_matches_plan": True,
                "interpolation_sample_count": 101,
                "all_interpolation_samples_within_workspace": True,
                "minimum_interpolation_eef_z_m": 0.73,
                "maximum_interpolation_eef_center_displacement_m": 0.001,
                "maximum_interpolation_eef_rotation_deg": 0.0,
            },
        },
        "motion_authorization": {
            "explicit_user_authorization_recorded": authorized,
            "scope": motion.JOINT_RECOVERY_AUTHORIZATION_SCOPE,
            "recovery_id": "JR01",
            "source_text": "test authorization",
            "consumed": False,
        },
    }


def _confirmation_argv(path: Path, pose_id: str, digest: str) -> list[str]:
    return [
        "run",
        "--plan",
        str(path),
        "--pose-id",
        pose_id,
        "--confirm-plan-sha256",
        digest,
        "--confirm-pose-id",
        pose_id,
        "--confirm-e-stop",
        motion.E_STOP_TOKEN,
        "--confirm-swept-volume",
        motion.SWEPT_VOLUME_TOKEN,
        "--confirm-target-rigid",
        motion.TARGET_RIGID_TOKEN,
        "--confirm-camera-fixed",
        motion.CAMERA_FIXED_TOKEN,
    ]


def test_missing_confirmations_do_not_import_franka_or_pylibfranka(
    tmp_path, monkeypatch, capsys
):
    path = _write_plan(tmp_path, _base_plan(authorized=True))
    imported_hardware = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "pylibfranka" or name.startswith("anydex_pipeline"):
            imported_hardware.append(name)
            raise AssertionError("hardware import occurred before confirmations")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    result = motion.main(["run", "--plan", str(path), "--pose-id", "T02"])

    assert result == 1
    assert imported_hardware == []
    assert "exact confirmations required" in capsys.readouterr().err


def test_wrong_plan_sha_fails_before_hardware_import(tmp_path, monkeypatch, capsys):
    path = _write_plan(tmp_path, _base_plan(authorized=True))
    argv = _confirmation_argv(path, "T02", "0" * 64)

    def forbidden_import(name, *args, **kwargs):
        if name == "pylibfranka" or name.startswith("anydex_pipeline"):
            raise AssertionError("hardware import occurred after a bad SHA")
        return original_import(name, *args, **kwargs)

    original_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", forbidden_import)
    assert motion.main(argv) == 1
    assert "exact confirmations required" in capsys.readouterr().err


@pytest.mark.parametrize(
    "mutate, match",
    [
        (
            lambda plan: plan["training_poses"][1].update(
                xyz_offset_base_m=[0.050001, 0.0, 0.0]
            ),
            "translation norm",
        ),
        (
            lambda plan: plan["training_poses"][1].update(
                rotation_vector_eef_deg=[16.001, 0.0, 0.0]
            ),
            "rotation angle",
        ),
        (
            lambda plan: plan["safety"]["workspace_bounds_base_m"].update(
                x=[0.495, 0.56]
            ),
            "outside hard workspace",
        ),
        (
            lambda plan: plan["training_poses"][1].update(
                xyz_offset_base_m=[0.0, 0.0, -0.063]
            ),
            "translation norm",
        ),
    ],
)
def test_pose_boundaries_fail_closed(tmp_path, mutate, match):
    plan = _base_plan()
    mutate(plan)
    path = _write_plan(tmp_path, plan)
    with pytest.raises(ValueError, match=match):
        motion.load_pose_preview(path, "T02")


def test_minimum_z_is_checked_independently_of_workspace(tmp_path):
    plan = _base_plan()
    plan["safety"]["minimum_eef_z_m"] = 0.670
    plan["training_poses"][1]["xyz_offset_base_m"] = [0.0, 0.0, -0.02]
    path = _write_plan(tmp_path, plan)
    with pytest.raises(ValueError, match="below minimum"):
        motion.load_pose_preview(path, "T02")


@pytest.mark.parametrize(
    "field,value,match",
    [
        (
            "cartesian_endpoint_hold_s",
            motion.CARTESIAN_ENDPOINT_HOLD_S - 0.1,
            "endpoint_hold_s",
        ),
        (
            "cartesian_pose_controller_mode",
            "cartesian_impedance",
            "controller_mode",
        ),
    ],
)
def test_calibration_motion_locks_endpoint_gate_and_joint_impedance(
    tmp_path, field, value, match
):
    plan = _base_plan()
    plan["safety"][field] = value
    path = _write_plan(tmp_path, plan)

    with pytest.raises(ValueError, match=match):
        motion.load_pose_preview(path, "T02")


def test_target_uses_base_translation_and_eef_right_multiplied_rotation(tmp_path):
    plan = _base_plan()
    rz90 = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    anchor = np.eye(4)
    anchor[:3, :3] = rz90
    anchor[:3, 3] = [0.5558, -0.0796, 0.6826]
    plan["reference_pose"] = {
        "xyz_base_m": anchor[:3, 3].tolist(),
        "T_base_ee": anchor.tolist(),
    }
    plan["training_poses"][1] = {
        "id": "T02",
        "xyz_offset_base_m": [0.02, 0.0, 0.0],
        "rotation_vector_eef_deg": [10.0, 0.0, 0.0],
    }
    path = _write_plan(tmp_path, plan)
    preview = motion.load_pose_preview(path, "T02")

    expected_delta = motion._so3_exp(np.deg2rad([10.0, 0.0, 0.0]))
    np.testing.assert_allclose(
        preview.T_target[:3, 3], anchor[:3, 3] + [0.02, 0.0, 0.0]
    )
    np.testing.assert_allclose(preview.T_target[:3, :3], rz90 @ expected_delta)
    assert not np.allclose(preview.T_target[:3, :3], expected_delta @ rz90)


class _FakeArm:
    def __init__(self, live_pose: np.ndarray, *, settled: bool = True, stop_error=None):
        self.robot = SimpleNamespace(read_once=lambda: SimpleNamespace(O_T_EE=live_pose))
        self.settled = settled
        self.stop_error = stop_error
        self.validate_calls = 0
        self.move_targets = []
        self.hold_targets = []
        self.settle_targets = []
        self.stop_calls = 0
        self.last_control_loop_telemetry = None

    def _validate_state(self, state, *, require_idle, enforce_success):
        assert require_idle is True
        assert enforce_success is False
        self.validate_calls += 1

    def move_pose(self, target):
        self.move_targets.append(np.asarray(target).copy())

    def hold_pose_control(self, target, duration_s):
        self.hold_targets.append((np.asarray(target).copy(), float(duration_s)))

    def verify_settled(self, target):
        self.settle_targets.append(np.asarray(target).copy())
        return self.settled

    def stop(self):
        self.stop_calls += 1
        if self.stop_error is not None:
            raise self.stop_error


def test_execute_moves_exactly_one_selected_pose_and_stops(tmp_path):
    path = _write_plan(tmp_path, _base_plan(authorized=True))
    preview = motion.load_pose_preview(path, "T02", for_run=True)
    arm = _FakeArm(preview.T_anchor)

    result = motion.execute_one_pose(
        preview, "172.16.0.2", connector=lambda selected, ip: arm
    )

    assert result.capture_ready is True
    assert result.camera_capture_performed is False
    assert len(arm.move_targets) == 1
    assert len(arm.settle_targets) == 1
    assert arm.stop_calls == 1
    np.testing.assert_allclose(arm.move_targets[0], preview.T_target)


def test_execute_returns_post_handle_control_qualification_evidence(tmp_path):
    path = _write_plan(tmp_path, _base_plan(authorized=True))
    preview = motion.load_pose_preview(path, "T02", for_run=True)
    arm = _FakeArm(preview.T_anchor)
    arm.last_control_loop_telemetry = SimpleNamespace(
        kind="cartesian",
        samples=4101,
        max_read_to_write_ns=321000,
        read_to_write_overruns=0,
        success_qualified=True,
        success_qualification_positive_writes=101,
        success_qualification_control_time_s=0.101,
        success_qualification_wall_time_s=0.102,
        success_qualification_rate=1.0,
    )

    result = motion.execute_one_pose(
        preview, "172.16.0.2", connector=lambda selected, ip: arm
    )

    assert result.control_loop_telemetry == {
        "kind": "cartesian",
        "samples": 4101,
        "max_read_to_write_ns": 321000,
        "read_to_write_overruns": 0,
        "success_qualified": True,
        "success_qualification_positive_writes": 101,
        "success_qualification_control_time_s": 0.101,
        "success_qualification_wall_time_s": 0.102,
        "success_qualification_rate": 1.0,
    }


def test_execute_exact_start_hold_probe_never_calls_move_pose(tmp_path):
    plan = _base_plan(authorized=True, pose_id="T01", start_pose_id="T01")
    path = _write_plan(tmp_path, plan)
    preview = motion.load_pose_preview(path, "T01", for_run=True)
    arm = _FakeArm(preview.T_anchor)

    result = motion.execute_one_pose(
        preview,
        "172.16.0.2",
        connector=lambda selected, ip: arm,
        hold_probe_duration_s=10.0,
    )

    assert result.capture_ready is False
    assert result.diagnostic_hold_completed is True
    assert arm.move_targets == []
    assert len(arm.hold_targets) == 1
    np.testing.assert_allclose(arm.hold_targets[0][0], preview.T_target)
    assert arm.hold_targets[0][1] == 10.0
    assert arm.stop_calls == 1


def test_execute_rechecks_link_immediately_before_motion_and_stops(tmp_path):
    path = _write_plan(tmp_path, _base_plan(authorized=True))
    preview = motion.load_pose_preview(path, "T02", for_run=True)
    arm = _FakeArm(preview.T_anchor)
    events = []

    def blocked_preflight(robot_ip):
        events.append(("preflight", robot_ip))
        raise RuntimeError("Desk connected after FCI opened")

    with pytest.raises(RuntimeError, match="Desk connected"):
        motion.execute_one_pose(
            preview,
            "172.16.0.2",
            connector=lambda selected, ip: arm,
            link_preflight=blocked_preflight,
        )

    assert events == [("preflight", "172.16.0.2")]
    assert arm.move_targets == []
    assert arm.stop_calls == 1


def _patch_offline_franka_connection(
    monkeypatch,
    *,
    preflight,
    robot_factory,
):
    """Install read-only/fake connection dependencies for ``_connect_franka``."""

    dexgrasp_src = SCRIPT.parents[3] / "dexgrasp" / "src"
    monkeypatch.syspath_prepend(str(dexgrasp_src))
    preflight_module = importlib.import_module(
        "anydex_pipeline.host_network_preflight"
    )
    monkeypatch.setattr(
        preflight_module,
        "require_uncontended_franka_https_link",
        preflight,
    )

    class RealtimeConfig:
        kEnforce = object()
        kIgnore = object()

    fake_pylibfranka = SimpleNamespace(
        RealtimeConfig=RealtimeConfig,
        Robot=robot_factory,
    )
    monkeypatch.setitem(sys.modules, "pylibfranka", fake_pylibfranka)
    return RealtimeConfig


def test_connect_preflights_before_fci_and_requests_k_enforce(
    tmp_path, monkeypatch
):
    path = _write_plan(tmp_path, _base_plan(authorized=True))
    preview = motion.load_pose_preview(path, "T02", for_run=True)
    events = []
    fake_robot = SimpleNamespace()

    def preflight(robot_ip):
        events.append(("preflight", robot_ip))

    def robot_factory(robot_ip, realtime_config):
        events.append(("open_fci", robot_ip, realtime_config))
        return fake_robot

    realtime = _patch_offline_franka_connection(
        monkeypatch,
        preflight=preflight,
        robot_factory=robot_factory,
    )

    arm = motion._connect_franka(preview, "172.16.0.2")

    assert events == [
        ("preflight", "172.16.0.2"),
        ("open_fci", "172.16.0.2", realtime.kEnforce),
    ]
    assert arm.robot is fake_robot


def test_connect_fails_closed_when_network_preflight_rejects(
    tmp_path, monkeypatch
):
    path = _write_plan(tmp_path, _base_plan(authorized=True))
    preview = motion.load_pose_preview(path, "T02", for_run=True)
    events = []

    def blocked_preflight(robot_ip):
        events.append(("preflight", robot_ip))
        raise RuntimeError("Desk HTTPS connection occupies the Franka link")

    def forbidden_robot_factory(_robot_ip, _realtime_config):
        events.append(("open_fci",))
        raise AssertionError("FCI opened after a failed host-network preflight")

    _patch_offline_franka_connection(
        monkeypatch,
        preflight=blocked_preflight,
        robot_factory=forbidden_robot_factory,
    )

    with pytest.raises(RuntimeError, match="Desk HTTPS"):
        motion._connect_franka(preview, "172.16.0.2")

    assert events == [("preflight", "172.16.0.2")]


def test_connect_does_not_fallback_to_k_ignore_when_realtime_enforcement_fails(
    tmp_path, monkeypatch
):
    path = _write_plan(tmp_path, _base_plan(authorized=True))
    preview = motion.load_pose_preview(path, "T02", for_run=True)
    events = []

    def preflight(robot_ip):
        events.append(("preflight", robot_ip))

    def robot_factory(robot_ip, realtime_config):
        events.append(("open_fci", robot_ip, realtime_config))
        raise RuntimeError("real-time scheduling unavailable")

    realtime = _patch_offline_franka_connection(
        monkeypatch,
        preflight=preflight,
        robot_factory=robot_factory,
    )

    with pytest.raises(RuntimeError, match="real-time scheduling unavailable"):
        motion._connect_franka(preview, "172.16.0.2")

    assert events == [
        ("preflight", "172.16.0.2"),
        ("open_fci", "172.16.0.2", realtime.kEnforce),
    ]


def test_verify_settled_failure_never_returns_capture_ready(tmp_path):
    path = _write_plan(tmp_path, _base_plan(authorized=True))
    preview = motion.load_pose_preview(path, "T02", for_run=True)
    arm = _FakeArm(preview.T_anchor, settled=False)

    with pytest.raises(RuntimeError, match="settle gate"):
        motion.execute_one_pose(
            preview, "172.16.0.2", connector=lambda selected, ip: arm
        )

    assert len(arm.move_targets) == 1
    assert arm.stop_calls == 1


def test_every_pose_requires_authorized_planned_live_start(tmp_path):
    displaced = np.eye(4)
    displaced[:3, 3] = [0.57, -0.0796, 0.6826]

    t01_plan = _base_plan(authorized=True, pose_id="T01")
    t01_path = _write_plan(tmp_path, t01_plan, "t01.yaml")
    t01 = motion.load_pose_preview(t01_path, "T01", for_run=True)
    t01_arm = _FakeArm(displaced)
    with pytest.raises(RuntimeError, match="authorized planned start T01"):
        motion.execute_one_pose(
            t01, "172.16.0.2", connector=lambda selected, ip: t01_arm
        )
    assert t01_arm.move_targets == []
    assert t01_arm.stop_calls == 1

    later_plan = _base_plan(authorized=True, pose_id="T02")
    later_path = _write_plan(tmp_path, later_plan, "later.yaml")
    later = motion.load_pose_preview(later_path, "T02", for_run=True)
    later_arm = _FakeArm(displaced)
    with pytest.raises(RuntimeError, match="authorized planned start T01"):
        motion.execute_one_pose(
            later, "172.16.0.2", connector=lambda selected, ip: later_arm
        )
    assert later_arm.move_targets == []
    assert later_arm.stop_calls == 1


def test_authorized_start_pose_must_exist_in_plan(tmp_path):
    path = _write_plan(
        tmp_path,
        _base_plan(authorized=True, start_pose_id="NOT_A_PLANNED_POSE"),
    )
    with pytest.raises(ValueError, match="is not a planned pose"):
        motion.load_pose_preview(path, "T02", for_run=True)


def test_observed_recovery_pose_can_bind_the_authorized_start(tmp_path):
    plan = _base_plan(authorized=True, start_pose_id="R01")
    plan["recovery_poses"] = [
        {
            "id": "R01",
            "xyz_offset_base_m": [0.01, 0.005, -0.005],
            "rotation_vector_eef_deg": [3.0, 0.5, 0.5],
            "state": "observed_abort_pose_do_not_capture",
        }
    ]
    path = _write_plan(tmp_path, plan)

    preview = motion.load_pose_preview(path, "T02", for_run=True)

    assert preview.start_pose_id == "R01"
    assert preview.pose_set == "training"
    assert preview.T_start is not None


def test_consumed_single_pose_authorization_cannot_be_replayed(tmp_path):
    plan = _base_plan(authorized=True)
    plan["motion_authorization"]["consumed"] = True
    path = _write_plan(tmp_path, plan)

    with pytest.raises(ValueError, match="already been consumed"):
        motion.load_pose_preview(path, "T02", for_run=True)


@pytest.mark.parametrize(
    "target_offset,target_rotation,start_offset,start_rotation,match",
    [
        (
            [0.05, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-0.05, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            "edge T03->T02 translation",
        ),
        (
            [0.0, 0.0, 0.0],
            [16.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-16.0, 0.0, 0.0],
            "edge T03->T02 rotation",
        ),
    ],
)
def test_authorized_start_to_target_edge_is_bounded(
    tmp_path,
    target_offset,
    target_rotation,
    start_offset,
    start_rotation,
    match,
):
    plan = _base_plan(authorized=True, start_pose_id="T03")
    plan["training_poses"][1].update(
        xyz_offset_base_m=target_offset,
        rotation_vector_eef_deg=target_rotation,
    )
    plan["training_poses"].append(
        {
            "id": "T03",
            "xyz_offset_base_m": start_offset,
            "rotation_vector_eef_deg": start_rotation,
        }
    )
    path = _write_plan(tmp_path, plan)
    with pytest.raises(ValueError, match=match):
        motion.load_pose_preview(path, "T02", for_run=True)


def test_missing_dynamics_is_reported_offline_and_blocks_run(tmp_path, capsys):
    plan = _base_plan(authorized=True)
    del plan["hardware"]["robot"]["expected_m_ee_kg"]
    path = _write_plan(tmp_path, plan)

    preview = motion.load_pose_preview(path, "T02")
    assert preview.dynamics is None
    motion._print_preview(preview, as_json=False)
    assert "RUN BLOCKED: missing/invalid strict dynamics" in capsys.readouterr().out
    with pytest.raises(ValueError, match="strict dynamics"):
        motion.load_pose_preview(path, "T02", for_run=True)


def test_direct_execute_cannot_bypass_plan_authorization(tmp_path):
    path = _write_plan(tmp_path, _base_plan(authorized=False))
    preview = motion.load_pose_preview(path, "T02")
    arm = _FakeArm(preview.T_anchor)
    with pytest.raises(ValueError, match="explicit single-pose authorization"):
        motion.execute_one_pose(
            preview, "172.16.0.2", connector=lambda selected, ip: arm
        )
    assert arm.stop_calls == 0


class _FakeJointRecoveryArm:
    def __init__(self, q, start_pose, target_pose):
        self.state = SimpleNamespace(
            q=np.asarray(q, dtype=np.float64).tolist(),
            dq=np.zeros(7).tolist(),
            O_T_EE=np.asarray(start_pose, dtype=np.float64).copy(),
        )
        self.target_pose = np.asarray(target_pose, dtype=np.float64).copy()
        self.robot = SimpleNamespace(read_once=lambda: self.state)
        self.move_targets = []
        self.stop_calls = 0
        self.last_control_loop_telemetry = None

    def _validate_state(self, state, *, require_idle, enforce_success):
        assert state is self.state
        assert require_idle is True
        assert enforce_success is False

    def move_joints(self, target):
        target = np.asarray(target, dtype=np.float64).copy()
        self.move_targets.append(target)
        self.state.q = target.tolist()
        self.state.dq = np.zeros(7).tolist()
        self.state.O_T_EE = self.target_pose.copy()

    def stop(self):
        self.stop_calls += 1


def test_joint_recovery_preview_binds_artifacts_and_absolute_target(tmp_path):
    plan = _joint_recovery_plan(tmp_path, authorized=True)
    path = _write_plan(tmp_path, plan, "joint-plan.yaml")

    preview = motion.load_joint_recovery_preview(path, for_run=True)

    np.testing.assert_allclose(
        preview.planned_target_q_rad,
        preview.expected_start_q_rad + preview.commanded_delta_rad,
    )
    assert preview.maximum_live_to_target_delta_rad == pytest.approx(
        np.max(np.abs(preview.commanded_delta_rad))
        + motion.START_Q_TOLERANCE_RAD
    )
    assert preview.run_authorized is True


def test_joint_recovery_rejects_tampered_provenance_artifact(tmp_path):
    plan = _joint_recovery_plan(tmp_path, authorized=True)
    path = _write_plan(tmp_path, plan, "joint-plan.yaml")
    audit_path = Path(
        plan["recovery"]["kinematics_provenance"]["audit_artifact_path"]
    )
    audit_path.write_text('{"status":"tampered"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="audit_artifact SHA-256 mismatch"):
        motion.load_joint_recovery_preview(path, for_run=True)


def test_joint_recovery_commands_sha_bound_target_not_live_relative_target(tmp_path):
    plan = _joint_recovery_plan(tmp_path, authorized=True)
    path = _write_plan(tmp_path, plan, "joint-plan.yaml")
    preview = motion.load_joint_recovery_preview(path, for_run=True)
    live_q = preview.expected_start_q_rad.copy()
    live_q[0] += 0.5 * motion.START_Q_TOLERANCE_RAD
    arm = _FakeJointRecoveryArm(
        live_q,
        preview.expected_start_T_base_ee,
        preview.expected_target_T_base_ee,
    )

    result = motion.execute_joint_recovery(
        preview,
        "172.16.0.2",
        connector=lambda selected, ip: arm,
        link_preflight=lambda ip: None,
        sleep=lambda seconds: None,
    )

    assert len(arm.move_targets) == 1
    np.testing.assert_array_equal(arm.move_targets[0], preview.planned_target_q_rad)
    assert not np.allclose(
        arm.move_targets[0], live_q + preview.commanded_delta_rad, atol=1.0e-12
    )
    np.testing.assert_array_equal(
        result.effective_target_q_rad, preview.planned_target_q_rad
    )
    assert arm.stop_calls == 1
    receipt = yaml.safe_load(result.authorization_receipt_path.read_text())
    assert receipt["authorization_consumed"] is True
    assert receipt["status"] == "succeeded"

    with pytest.raises(RuntimeError, match="already consumed"):
        motion.execute_joint_recovery(
            preview,
            "172.16.0.2",
            connector=lambda selected, ip: arm,
            link_preflight=lambda ip: None,
            sleep=lambda seconds: None,
        )


def test_joint_recovery_without_explicit_authorization_never_claims_receipt(tmp_path):
    plan = _joint_recovery_plan(tmp_path, authorized=False)
    path = _write_plan(tmp_path, plan, "joint-plan.yaml")
    preview = motion.load_joint_recovery_preview(path)
    assert preview.run_authorized is False

    with pytest.raises(ValueError, match="unconsumed bounded joint"):
        motion.execute_joint_recovery(preview, "172.16.0.2")

    assert preview.authorization_receipt_path.exists() is False


def test_motion_and_stop_failures_keep_stop_unconfirmed_marker(tmp_path):
    path = _write_plan(tmp_path, _base_plan(authorized=True))
    preview = motion.load_pose_preview(path, "T02", for_run=True)
    arm = _FakeArm(
        preview.T_anchor,
        settled=False,
        stop_error=RuntimeError("readback unavailable"),
    )
    with pytest.raises(RuntimeError, match="STOP UNCONFIRMED") as caught:
        motion.execute_one_pose(
            preview, "172.16.0.2", connector=lambda selected, ip: arm
        )
    assert "settle gate" in str(caught.value)
    assert hasattr(caught.value, "motion_error")
    assert hasattr(caught.value, "stop_error")
