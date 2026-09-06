from pathlib import Path

import numpy as np
import pytest

from sim2real.deployment.shadow_readiness import (
    assess_v94_shadow_readiness,
    main,
)
from sim2real.deployment.bundle import DeployBundle, load_checkpoint_safely
from sim2real.policy import RollingStudentPolicy
from sim2real.observation.model import pose_from_position_quaternion_wxyz


BUNDLE = Path(__file__).resolve().parents[2] / "data/test_fixtures/sim2real/deploy.zip"
INITIAL_NPZ = "alignment/reset_idle_open/initial_observations_and_student_response.npz"
RUNS = Path(__file__).resolve().parents[2] / "dexgrasp" / "runs"
CAPTURE_08 = RUNS / "v94_live_readonly_phase_wakeup_sam_off_20260722_08.npz"
CAPTURE_09 = RUNS / "v94_live_readonly_zero_copy_phase_wakeup_sam_off_20260722_09.npz"


def _rolling_history(values: np.ndarray, index: int) -> np.ndarray:
    selected = values[max(0, index - 3) : index + 1]
    if selected.shape[0] < 4:
        selected = np.concatenate(
            [np.repeat(selected[:1], 4 - selected.shape[0], axis=0), selected],
            axis=0,
        )
    return selected


def _write_passing_shadow(path: Path, *, steps: int = 6) -> Path:
    bundle = DeployBundle(BUNDLE)
    source = bundle.load_npz(INITIAL_NPZ)
    policy = RollingStudentPolicy(load_checkpoint_safely(bundle.checkpoint_bytes()))
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
    pointcloud_frame_ids = np.repeat(
        np.arange(1, (steps // 2) + 2, dtype=np.int64), 2
    )[:steps]
    action_use_indices = np.tile(np.asarray([1, 2], dtype=np.int32), steps // 2 + 1)[
        :steps
    ]
    payload = {
        "audit_schema_version": np.asarray(1, dtype=np.int32),
        "hardware_writes": np.asarray(False),
        "robot_command_writes": np.asarray(False),
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
        "pointcloud_timestamp_s": timestamp - 0.070,
        "rh56_timestamp_s": timestamp - 0.005,
        "T_base_palm_at_pointcloud_capture": capture_poses,
        "franka_q_rad": source["franka_measured_q_rad"][:steps].astype(
            np.float64
        ),
        "rh56_angle_act_register_order": np.full(
            (steps, 6), 1000, dtype=np.int32
        ),
        "rh56_angle_set_register_order": np.full(
            (steps, 6), -1, dtype=np.int32
        ),
        "rh56_virtual_q_policy_order_rad": source[
            "rh56_virtual_q_policy_order_rad"
        ][:steps],
        "configured_max_camera_age_s": np.asarray(0.100),
        "pointcloud_age_at_action_s": np.full(steps, 0.090),
        "policy_rebootstrap_count": np.asarray(0, dtype=np.int64),
        "requested_capture_duration_s": np.asarray(steps / 60.0),
        "pointcloud_frame_id": pointcloud_frame_ids,
        "policy_action_use_index_for_camera_frame": action_use_indices,
        "maximum_policy_actions_per_camera_frame": np.asarray(2, dtype=np.int32),
        "policy_action_evidence_count": np.asarray(steps, dtype=np.int64),
        "policy_reuse_guard_hold_count": np.asarray(0, dtype=np.int64),
        "policy_action_or_reuse_hold_opportunity_count": np.asarray(
            steps, dtype=np.int64
        ),
        "policy_reuse_guard_semantics": np.asarray(
            "third_eligible_tick_holds_without_policy_state_or_action_evidence"
        ),
        "policy_hold_schema_version": np.asarray(1, dtype=np.int32),
        "policy_hold_count": np.asarray(0, dtype=np.int64),
    }
    np.savez_compressed(path, **payload)
    return path


def _rewrite(path: Path, **updates) -> Path:
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    payload.update(updates)
    np.savez_compressed(path, **payload)
    return path


def _check(report, name):
    return next(item for item in report["checks"] if item["name"] == name)


def test_synthetic_shadow_passes_but_never_claims_or_authorizes_motion(tmp_path):
    path = _write_passing_shadow(tmp_path / "passing_shadow.npz")
    report = assess_v94_shadow_readiness(path, bundle_path=BUNDLE)

    assert report["result"] == "PASS"
    assert report["failed_checks"] == []
    assert report["semantic_action_correctness_claimed"] is False
    assert report["physical_motion_authorized"] is False
    assert report["summary"]["effective_action_rate_hz"] == pytest.approx(60.0)
    reuse = report["summary"]["camera_frame_reuse"]
    assert reuse["policy_uses_per_unique_camera_frame"] == pytest.approx(2.0)
    assert reuse["maximum_policy_uses_for_one_camera_frame"] == 2
    guard = report["summary"]["camera_policy_reuse_guard"]
    assert guard["passed"]
    assert guard["policy_hold_count"] == 0
    assert _check(report, "camera_policy_reuse_guard_evidence")["passed"]
    assert "engineering commissioning" in report[
        "minimum_action_rate_threshold_semantics"
    ]


def test_explicit_third_tick_hold_reconciles_without_becoming_an_action(tmp_path):
    path = _write_passing_shadow(tmp_path / "one_hold.npz")
    _rewrite(
        path,
        policy_reuse_guard_hold_count=np.asarray(1, dtype=np.int64),
        policy_action_or_reuse_hold_opportunity_count=np.asarray(
            7, dtype=np.int64
        ),
        policy_hold_count=np.asarray(1, dtype=np.int64),
        policy_hold_diagnostic_index=np.asarray([0], dtype=np.int64),
        policy_hold_reason=np.asarray(["camera_frame_policy_reuse_limit"]),
        policy_hold_camera_frame_id=np.asarray([2], dtype=np.int64),
        policy_hold_pointcloud_frame_id=np.asarray([2], dtype=np.int64),
        policy_hold_action_uses_before_hold=np.asarray([2], dtype=np.int32),
        policy_hold_maximum_policy_actions_per_camera_frame=np.asarray(
            [2], dtype=np.int32
        ),
        policy_hold_new_policy_action_evidence=np.asarray([False]),
        policy_hold_policy_state_mutated=np.asarray([False]),
        policy_hold_write=np.asarray([False]),
        policy_hold_waits_for_new_camera_frame=np.asarray([True]),
        policy_hold_control_tick_started_monotonic_s=np.asarray([10.000]),
        policy_hold_control_tick_finished_monotonic_s=np.asarray([10.010]),
        policy_hold_control_tick_interval_from_previous_action_s=np.asarray(
            [1.0 / 60.0]
        ),
    )

    report = assess_v94_shadow_readiness(path, bundle_path=BUNDLE)
    guard_check = _check(report, "camera_policy_reuse_guard_evidence")
    assert report["result"] == "PASS"
    assert guard_check["passed"]
    assert guard_check["policy_action_evidence_count"] == 6
    assert guard_check["policy_hold_count"] == 1
    assert guard_check["policy_action_or_reuse_hold_opportunity_count"] == 7
    assert guard_check["hold_action_counts_match"]
    assert guard_check["hold_flags_passed"]

    # A skipped tick claiming either a model action or state mutation is not
    # acceptable hold evidence and fails closed.
    _rewrite(
        path,
        policy_hold_new_policy_action_evidence=np.asarray([True]),
    )
    invalid = assess_v94_shadow_readiness(path, bundle_path=BUNDLE)
    assert invalid["result"] == "FAIL"
    assert not _check(invalid, "camera_policy_reuse_guard_evidence")["passed"]


@pytest.mark.parametrize(
    ("updates", "failed_check"),
    [
        (
            {"configured_max_camera_age_s": np.asarray(0.101)},
            "configured_camera_age_within_gate",
        ),
        (
            {"pointcloud_age_at_action_s": np.full(6, 0.100001)},
            "every_pointcloud_age_at_action_within_gate",
        ),
        (
            {"policy_rebootstrap_count": np.asarray(1, dtype=np.int64)},
            "no_policy_rebootstrap",
        ),
        (
            {"requested_capture_duration_s": np.asarray(6.0 / 56.99)},
            "commissioning_effective_action_rate",
        ),
        (
            {"pointcloud_frame_id": np.asarray([1, 1, 1, 2, 2, 2])},
            "camera_30hz_to_policy_60hz_reuse_contract",
        ),
    ],
)
def test_strict_timing_and_reuse_gates_fail_closed(tmp_path, updates, failed_check):
    path = _write_passing_shadow(tmp_path / "shadow.npz")
    _rewrite(path, **updates)
    report = assess_v94_shadow_readiness(path, bundle_path=BUNDLE)
    assert report["result"] == "FAIL"
    assert _check(report, failed_check)["passed"] is False
    assert report["physical_motion_authorized"] is False


def test_replay_and_reset_are_independent_required_checks(tmp_path):
    replay_path = _write_passing_shadow(tmp_path / "bad_replay.npz")
    with np.load(replay_path, allow_pickle=False) as archive:
        action = archive["raw_policy_action13"].copy()
    action[0, 0] += np.float32(0.1)
    _rewrite(replay_path, raw_policy_action13=action)
    replay_report = assess_v94_shadow_readiness(replay_path, bundle_path=BUNDLE)
    assert _check(
        replay_report, "offline_audit_and_policy_replay_pass"
    )["passed"] is False

    reset_path = _write_passing_shadow(tmp_path / "bad_reset.npz")
    with np.load(reset_path, allow_pickle=False) as archive:
        franka_q = archive["franka_q_rad"].copy()
    franka_q[:, 0] += 0.1
    _rewrite(reset_path, franka_q_rad=franka_q)
    reset_report = assess_v94_shadow_readiness(reset_path, bundle_path=BUNDLE)
    assert _check(reset_report, "offline_audit_and_policy_replay_pass")[
        "passed"
    ]
    assert _check(reset_report, "recorded_reset_aligned")["passed"] is False


def test_no_robot_write_fields_must_both_be_explicitly_false(tmp_path):
    path = _write_passing_shadow(tmp_path / "writes.npz")
    _rewrite(path, robot_command_writes=np.asarray(True))
    report = assess_v94_shadow_readiness(path, bundle_path=BUNDLE)
    assert report["result"] == "FAIL"
    assert _check(report, "robot_command_writes_false")["passed"] is False


def test_cli_reports_object_dtype_as_json_fail(tmp_path, capsys):
    path = tmp_path / "object.npz"
    np.savez(path, unsafe=np.asarray([{"not": "allowed"}], dtype=object))
    assert main([str(path)]) == 2
    output = capsys.readouterr().out
    assert '"result": "FAIL"' in output
    assert '"physical_motion_authorized": false' in output
    assert "Object arrays cannot be loaded" in output


@pytest.mark.parametrize(
    ("path", "expected_rate", "also_rebootstrap"),
    [(CAPTURE_08, 54.7, False), (CAPTURE_09, 41.9, True)],
)
def test_recorded_08_and_09_fail_strict_default_gate(
    path, expected_rate, also_rebootstrap
):
    if not path.is_file():
        pytest.skip(f"optional recorded fixture is absent: {path}")
    report = assess_v94_shadow_readiness(path, bundle_path=BUNDLE)
    assert report["result"] == "FAIL"
    assert report["summary"]["effective_action_rate_hz"] == pytest.approx(
        expected_rate
    )
    expected_failures = [
        "commissioning_effective_action_rate",
        "camera_policy_reuse_guard_evidence",
    ]
    if also_rebootstrap:
        expected_failures.insert(0, "no_policy_rebootstrap")
    assert report["failed_checks"] == expected_failures
    assert _check(report, "commissioning_effective_action_rate")["passed"] is False
    assert _check(report, "no_policy_rebootstrap")["passed"] is (not also_rebootstrap)
    assert _check(report, "camera_30hz_to_policy_60hz_reuse_contract")["passed"]
    assert not _check(report, "camera_policy_reuse_guard_evidence")["passed"]
    assert _check(report, "offline_audit_and_policy_replay_pass")["passed"]
    assert _check(report, "recorded_reset_aligned")["passed"]
    assert report["semantic_action_correctness_claimed"] is False
    assert report["physical_motion_authorized"] is False
