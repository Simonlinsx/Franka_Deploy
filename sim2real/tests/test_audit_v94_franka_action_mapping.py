import json
import sys

import pytest

from sim2real.deployment.franka_action_audit import (
    DEFAULT_PROFILE,
    audit_v94_franka_action_mapping,
)


def test_independent_bundle_recurrence_and_previous_action_semantics_are_exact():
    report = audit_v94_franka_action_mapping()

    assert report.packaged_reference_mapping_confirmed
    assert report.axis_order_identity
    assert report.sign_transform == "identity (no negation)"
    assert report.command_unit == "joint position target in radians"
    assert report.control_dt_s == pytest.approx(1.0 / 60.0)
    assert report.effective_gain_rad_per_tick == pytest.approx(0.003)
    assert report.nominal_full_scale_target_rate_rad_s == pytest.approx(0.18)
    assert report.simulation_and_profile_joint_limits_exact
    assert report.simulation_and_profile_reset_exact

    assert report.reset_stream.independent_target_max_abs_error_rad == 0.0
    assert report.closed_loop_stream.independent_target_max_abs_error_rad == 0.0
    assert report.reset_stream.action_delta_sign_mismatches == 0
    assert report.closed_loop_stream.action_delta_sign_mismatches == 0
    assert report.closed_loop_stream.action_delta_sign_checks == 11 * 7
    assert report.reset_stream.per_tick_step_clip_count == 0
    assert report.closed_loop_stream.per_tick_step_clip_count == 0
    assert report.reset_stream.measured_q_or_joint_limit_clip_count == 0
    assert report.closed_loop_stream.measured_q_or_joint_limit_clip_count == 0
    assert report.initial_previous_action_exact
    assert report.reset_idle_previous_action_exact
    assert report.closed_loop_previous_action_exact
    assert report.hardware_accessed is False
    assert "pylibfranka" not in sys.modules


def test_mapping_pass_does_not_claim_uncommissioned_physical_tracking():
    report = audit_v94_franka_action_mapping()

    assert not report.physical_tracking_equivalence_confirmed
    assert report.current_profile_mode == "commissioning_locked"
    assert not report.current_profile_persistent_session_configured
    assert report.current_profile_max_velocity_rad_s == pytest.approx((0.05,) * 7)
    assert report.closed_loop_stream.rows_exceeding_profile_velocity == 11
    assert set(report.current_profile_missing_dynamic_limits) == {
        "online_max_joint_acceleration_rad_s2",
        "online_max_joint_jerk_rad_s3",
        "online_max_tracking_error_rad",
    }
    assert any("0.18 rad/s" in blocker for blocker in report.blockers)
    assert any("not commissioned" in blocker for blocker in report.blockers)


def test_task_owned_reset_can_replace_bundle_home_without_changing_mapping(
    tmp_path,
):
    profile = json.loads(DEFAULT_PROFILE.read_text(encoding="utf-8"))
    target = [0.0, -1.2, 0.0, -2.2, 0.0, 1.4, 0.7853981852531433]
    profile["franka"]["default_q_rad"] = target
    task_profile = tmp_path / "v57-profile.json"
    task_profile.write_text(json.dumps(profile), encoding="utf-8")

    report = audit_v94_franka_action_mapping(
        profile_path=task_profile,
        expected_reset_q_rad=target,
    )

    assert report.packaged_reference_mapping_confirmed
    assert report.simulation_and_profile_reset_exact
    assert report.axis_order_identity


def test_task_owned_joint_limits_can_replace_bundle_envelope_without_changing_mapping(
    tmp_path,
):
    profile = json.loads(DEFAULT_PROFILE.read_text(encoding="utf-8"))
    limits = [
        [-2.9007, 2.9007],
        [-1.8361, 1.8361],
        [-2.9007, 2.9007],
        [-3.077, -0.1169],
        [-2.8763, 2.8763],
        [0.4398, 4.6216],
        [-3.0508, 3.0508],
    ]
    profile["franka"]["joint_limits_rad"] = limits
    task_profile = tmp_path / "task-limits-profile.json"
    task_profile.write_text(json.dumps(profile), encoding="utf-8")

    without_task_contract = audit_v94_franka_action_mapping(
        profile_path=task_profile,
    )
    with_task_contract = audit_v94_franka_action_mapping(
        profile_path=task_profile,
        expected_joint_limits_rad=limits,
    )

    assert not without_task_contract.packaged_reference_mapping_confirmed
    assert not without_task_contract.simulation_and_profile_joint_limits_exact
    assert with_task_contract.packaged_reference_mapping_confirmed
    assert with_task_contract.simulation_and_profile_joint_limits_exact
    assert with_task_contract.axis_order_identity
