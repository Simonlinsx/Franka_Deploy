from pathlib import Path

import numpy as np
import pytest

from sim2real.diagnostics.audit_v94_preview import audit_v94_preview
from sim2real.deployment.bundle import DeployBundle, load_checkpoint_safely
from sim2real.policy import RollingStudentPolicy
from sim2real.contracts.actions import V94ActionMapper
from sim2real.contracts.v94 import V94Contract
from sim2real.observation.model import pose_from_position_quaternion_wxyz

BUNDLE = Path(__file__).resolve().parents[2] / "data/test_fixtures/sim2real/deploy.zip"
INITIAL_NPZ = "alignment/reset_idle_open/initial_observations_and_student_response.npz"
CLOSED_NPZ = "alignment/student_closed_loop/closed_loop_observation_action_pairs.npz"


def _rolling_history(values: np.ndarray, index: int) -> np.ndarray:
    start = max(0, index - 3)
    selected = values[start : index + 1]
    if selected.shape[0] < 4:
        selected = np.concatenate(
            [np.repeat(selected[:1], 4 - selected.shape[0], axis=0), selected],
            axis=0,
        )
    return selected


def _write_minimal_audit(
    path: Path, *, steps: int = 2, use_initial: bool = False, **updates
) -> Path:
    bundle = DeployBundle(BUNDLE)
    source = bundle.load_npz(INITIAL_NPZ if use_initial else CLOSED_NPZ)
    policy = RollingStudentPolicy(load_checkpoint_safely(bundle.checkpoint_bytes()))
    if use_initial:
        point_frames = source["pointcloud_xyzrgb_palm"][:steps]
        valid_frames = source["pointcloud_valid"][:steps]
        proprio_frames = source["proprio67"][:steps]
        points = np.stack(
            [_rolling_history(point_frames, index) for index in range(steps)]
        )
        valid = np.stack(
            [_rolling_history(valid_frames, index) for index in range(steps)]
        )
        proprio = np.stack(
            [_rolling_history(proprio_frames, index) for index in range(steps)]
        )
    else:
        points = source["pointcloud_history_xyzrgb_palm"][:steps].copy()
        valid = source["pointcloud_valid_history"][:steps].copy()
        proprio = source["proprio_history67"][:steps].copy()
    outputs = [policy.act(points[i], valid[i], proprio[i]) for i in range(steps)]
    timestamp = 100.0 + np.arange(steps, dtype=np.float64) / 60.0
    capture_poses = np.stack(
        [
            pose_from_position_quaternion_wxyz(
                proprio[i, -1, 26:29], proprio[i, -1, 29:33]
            )
            for i in range(steps)
        ]
    )
    payload = {
        "audit_schema_version": np.asarray(1, dtype=np.int32),
        "hardware_writes": np.asarray(False),
        "pointcloud_history_xyzrgb_palm": points,
        "pointcloud_valid_history": valid,
        "proprio_history67": proprio,
        "pointcloud_current_xyzrgb_palm": points[:, -1].copy(),
        "pointcloud_current_valid": valid[:, -1].copy(),
        "proprio_current67": proprio[:, -1].copy(),
        "raw_policy_action13": np.stack([item.action13 for item in outputs]),
        "predicted_privileged32": np.stack(
            [item.predicted_privileged32 for item in outputs]
        ),
        "predicted_future_motion24": np.stack(
            [item.future_motion24 for item in outputs]
        ),
        "predicted_hold6": np.stack([item.predicted_hold6 for item in outputs]),
        "predicted_hold_logit": np.asarray(
            [item.predicted_hold_logit for item in outputs]
        ),
        "timestamp_s": timestamp,
        "franka_timestamp_s": timestamp.copy(),
        "pointcloud_timestamp_s": timestamp.copy(),
        "rh56_timestamp_s": timestamp.copy() - 0.005,
        "T_base_palm_at_pointcloud_capture": capture_poses,
    }
    payload.update(updates)
    np.savez_compressed(path, **payload)
    return path


def test_bundle_closed_loop_minimal_audit_replays_and_reports_reset_geometry(tmp_path):
    path = _write_minimal_audit(tmp_path / "audit.npz")
    report = audit_v94_preview(path, bundle_path=BUNDLE)
    assert report["result"] == "PASS"
    assert report["steps"] == 2
    assert report["robot_command_writes"] is False
    assert report["camera_configuration_writes"] is None
    assert report["policy_replay"]["worst_max_abs_error"] == 0.0
    assert report["pointcloud_valid"]["current_counts"]["min"] >= 1
    alignment = report["reset_reference_alignment"]
    assert alignment["available"]
    assert alignment["reset_reference_aligned"]
    assert alignment["status"] == "compatible"
    assert alignment["alignment_semantics"] == (
        "bundle_derived_possible_surface_observation_compatibility"
    )
    assert alignment["visible_partial_cloud_base"][
        "coordinatewise_median_is_object_center"
    ] is False
    assert alignment["sphere_center_alignment"]["status"] == "unknown"
    assert alignment["sphere_center_alignment"]["estimated_center_base_m"] is None
    surface = alignment["possible_surface_union_aabb"]
    assert surface["allowed_outside_fraction"] == pytest.approx(0.01)
    assert surface["compatibility_passed"]
    assert report["reset_reference_aligned"] is True
    assert alignment["semantic_action_correctness_claimed"] is False
    assert report["semantic_action_correctness_claimed"] is False
    comparison = report["packaged_reference_comparison"]
    assert comparison["diagnostic_only"]
    assert comparison["initial_or_closed_action_linf_nearest"]["max"] < 5.0e-4
    assert comparison[
        "initial_arm_increment_action7_cosine_at_linf_nearest"
    ] is not None
    assert comparison[
        "initial_base_frame_pointcloud_xyz_symmetric_chamfer_m_nearest"
    ] is not None
    assert report["robot_reset_alignment"]["available"] is False


def test_audit_reports_recorded_robot_reset_alignment(tmp_path):
    bundle = DeployBundle(BUNDLE)
    initial = bundle.load_npz(INITIAL_NPZ)
    steps = 2
    path = _write_minimal_audit(
        tmp_path / "reset_audit.npz",
        steps=steps,
        use_initial=True,
        franka_q_rad=initial["franka_measured_q_rad"][:steps].astype(np.float64),
        rh56_angle_act_register_order=np.full((steps, 6), 1000, dtype=np.int32),
        rh56_angle_set_register_order=np.full((steps, 6), -1, dtype=np.int32),
        rh56_virtual_q_policy_order_rad=initial[
            "rh56_virtual_q_policy_order_rad"
        ][:steps],
    )
    report = audit_v94_preview(path, bundle_path=BUNDLE)
    reset = report["robot_reset_alignment"]
    assert reset["available"]
    assert reset["aligned"]
    assert reset["franka_aligned"]
    assert reset["rh56_open"]
    assert reset["rh56_targets_all_disabled"]
    assert reset["previous_action_reset"]
    object_reset = report["reset_reference_alignment"]
    assert object_reset["reset_reference_aligned"]
    assert object_reset["sphere_center_alignment"]["status"] == "unknown"


def test_object_reset_is_unknown_without_capture_pose_and_never_falls_back_to_z(
    tmp_path,
):
    path = _write_minimal_audit(tmp_path / "no_pose.npz", steps=1)
    with np.load(path, allow_pickle=False) as archive:
        payload = {
            name: archive[name].copy()
            for name in archive.files
            if name != "T_base_palm_at_pointcloud_capture"
        }
    np.savez_compressed(path, **payload)

    report = audit_v94_preview(path, bundle_path=BUNDLE)
    reset = report["reset_reference_alignment"]
    assert reset["status"] == "unknown"
    assert reset["reset_reference_aligned"] is None
    assert report["reset_reference_aligned"] is None
    assert "height" not in reset
    assert report["semantic_action_correctness_claimed"] is False


def test_object_surface_gate_uses_only_bundle_outlier_probability(tmp_path):
    path = _write_minimal_audit(tmp_path / "outside_surface.npz", steps=1)
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}

    assert np.all(payload["pointcloud_current_valid"][0, :2] == 1.0)
    pose = payload["T_base_palm_at_pointcloud_capture"][0]
    # Two of 128 points is 0.015625, above the bundle's explicit 0.01
    # point-outlier probability.  The point locations are intentionally far
    # outside the bundle-derived possible sphere-surface union.
    outside_base = np.asarray(
        [[0.40, 0.05, 0.06], [0.41, 0.05, 0.06]], dtype=np.float64
    )
    outside_palm = (outside_base - pose[:3, 3]) @ pose[:3, :3]
    payload["pointcloud_current_xyzrgb_palm"][0, :2, :3] = outside_palm.astype(
        np.float32
    )
    payload["pointcloud_history_xyzrgb_palm"][0, -1, :2, :3] = (
        outside_palm.astype(np.float32)
    )

    bundle = DeployBundle(BUNDLE)
    policy = RollingStudentPolicy(load_checkpoint_safely(bundle.checkpoint_bytes()))
    output = policy.act(
        payload["pointcloud_history_xyzrgb_palm"][0],
        payload["pointcloud_valid_history"][0],
        payload["proprio_history67"][0],
    )
    payload["raw_policy_action13"][0] = output.action13
    payload["predicted_privileged32"][0] = output.predicted_privileged32
    payload["predicted_future_motion24"][0] = output.future_motion24
    payload["predicted_hold6"][0] = output.predicted_hold6
    payload["predicted_hold_logit"][0] = output.predicted_hold_logit
    np.savez_compressed(path, **payload)

    report = audit_v94_preview(path, bundle_path=BUNDLE)
    reset = report["reset_reference_alignment"]
    surface = reset["possible_surface_union_aabb"]
    assert surface["allowed_outside_fraction"] == pytest.approx(0.01)
    assert surface["maximum_step_outside_fraction"] >= 2.0 / 128.0
    assert surface["compatibility_passed"] is False
    assert reset["status"] == "incompatible"
    assert reset["reset_reference_aligned"] is False
    assert reset["sphere_center_alignment"]["status"] == "unknown"
    assert report["semantic_action_correctness_claimed"] is False


def test_audit_recomputes_one_step_action_target_mapping(tmp_path):
    bundle = DeployBundle(BUNDLE)
    contract = V94Contract.from_bundle(bundle)
    initial = bundle.load_npz(INITIAL_NPZ)
    steps = 2
    path = _write_minimal_audit(
        tmp_path / "mapping_audit.npz",
        steps=steps,
        use_initial=True,
        franka_q_rad=initial["franka_measured_q_rad"][:steps].astype(np.float64),
        rh56_virtual_q_policy_order_rad=initial[
            "rh56_virtual_q_policy_order_rad"
        ][:steps],
    )
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    mapped = []
    for index in range(steps):
        mapper = V94ActionMapper(
            initial_arm_target_q_rad=payload["franka_q_rad"][index],
            initial_hand_target_q_policy_order_rad=payload[
                "rh56_virtual_q_policy_order_rad"
            ][index],
            joint_limits_rad=contract.joint_limits_rad,
            q_hand_close_rad=contract.q_hand_close_rad,
        )
        mapped.append(
            mapper.map(
                payload["raw_policy_action13"][index],
                measured_q_rad=payload["franka_q_rad"][index],
            )
        )
    payload.update(
        {
            "executed_policy_action13": np.stack(
                [item.executed_policy_action13 for item in mapped]
            ),
            "franka_target_q_rad": np.stack(
                [item.franka_target_q_rad for item in mapped]
            ),
            "rh56_target_q_policy_order_rad": np.stack(
                [item.rh56_target_q_policy_order_rad for item in mapped]
            ),
            "rh56_proposed_angle_set_register_order": np.stack(
                [item.rh56_angle_set_register_order for item in mapped]
            ),
            "proposal_mode": np.asarray(
                ["one_step_from_measured_idle"] * steps
            ),
        }
    )
    np.savez_compressed(path, **payload)

    report = audit_v94_preview(path, bundle_path=BUNDLE)
    mapping = report["action_target_mapping"]
    assert mapping["available"] and mapping["passed"]
    assert mapping["worst_float_max_abs_error"] == 0.0
    assert mapping["rh56_register_targets_exact"]

    payload["rh56_proposed_angle_set_register_order"][0, 0] += 1
    bad = tmp_path / "bad_mapping.npz"
    np.savez_compressed(bad, **payload)
    with pytest.raises(ValueError, match="register target mapping mismatch"):
        audit_v94_preview(bad, bundle_path=BUNDLE)


def test_audit_rejects_current_history_mismatch(tmp_path):
    path = _write_minimal_audit(tmp_path / "audit.npz", steps=1)
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    payload["pointcloud_current_xyzrgb_palm"][0, 0, 0] += np.float32(0.01)
    bad = tmp_path / "bad_current.npz"
    np.savez_compressed(bad, **payload)
    with pytest.raises(ValueError, match="history"):
        audit_v94_preview(bad, bundle_path=BUNDLE)


def test_audit_rejects_policy_output_that_does_not_replay(tmp_path):
    path = _write_minimal_audit(tmp_path / "audit.npz", steps=1)
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    payload["raw_policy_action13"][0, 0] -= np.float32(0.1)
    bad = tmp_path / "bad_action.npz"
    np.savez_compressed(bad, **payload)
    with pytest.raises(ValueError, match="policy replay error"):
        audit_v94_preview(bad, bundle_path=BUNDLE)


def test_audit_reports_camera_configuration_but_rejects_robot_writes(tmp_path):
    path = _write_minimal_audit(
        tmp_path / "camera_configured.npz",
        steps=1,
        hardware_writes_semantics=np.asarray(
            "robot_actuator_or_register_commands_only"
        ),
        robot_command_writes=np.asarray(False),
        camera_configuration_writes=np.asarray(True),
    )
    report = audit_v94_preview(path, bundle_path=BUNDLE)
    assert report["robot_command_writes"] is False
    assert report["camera_configuration_writes"] is True

    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    payload["robot_command_writes"] = np.asarray(True)
    bad = tmp_path / "robot_write.npz"
    np.savez_compressed(bad, **payload)
    with pytest.raises(ValueError, match="robot command writes"):
        audit_v94_preview(bad, bundle_path=BUNDLE)
