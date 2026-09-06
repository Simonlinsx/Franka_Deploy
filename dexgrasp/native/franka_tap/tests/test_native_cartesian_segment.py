#!/usr/bin/env python3
"""Offline safety/trajectory tests for the native 0.21.2 control loop."""

from __future__ import annotations

import math

import _anydex_franka_telemetry as native


def _pose(x: float = 0.5, y: float = 0.0, z: float = 0.3) -> list[float]:
    # Franka's public pose representation is column-major.
    return [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        x,
        y,
        z,
        1.0,
    ]


def _config(**updates: object) -> dict:
    result = {
        "workspace_min_m": [0.4, -0.1, 0.2],
        "workspace_max_m": [0.6, 0.1, 0.4],
        "joint_lower_rad": [
            -2.9007,
            -1.8361,
            -2.9007,
            -3.0770,
            -2.8763,
            0.4398,
            -3.0508,
        ],
        "joint_upper_rad": [
            2.9007,
            1.8361,
            2.9007,
            -0.1169,
            2.8763,
            4.6216,
            3.0508,
        ],
        "joint_limit_margin_rad": 0.05,
        "duration_s": 0.010,
        "min_cartesian_duration_s": 0.010,
        "max_cartesian_speed_m_s": 1.0,
        "max_angular_speed_rad_s": 1.0,
        "max_segment_translation_m": 0.030,
        "max_segment_rotation_rad": 0.10,
        "endpoint_timeout_s": 0.020,
        "settle_time_s": 0.003,
        "translation_arrival_tolerance_m": 0.0001,
        "rotation_arrival_tolerance_rad": 0.001,
        "settle_max_dq_rad_s": 0.015,
        "min_control_success_rate": 0.95,
        "control_success_hard_floor": 0.80,
        "control_success_evaluation_window_s": 0.50,
        "startup_deadline_s": 0.50,
        "startup_min_positive_writes": 100,
        "min_control_period_s": 1.0e-6,
        "max_control_period_s": 0.020,
        "read_to_write_budget_ns": 500_000,
        "wall_deadline_slack_s": 2.0,
        "wall_deadline_fraction": 0.25,
    }
    result.update(updates)
    return result


def _snapshot(control) -> dict:
    return native.fake_cartesian_control_snapshot(control)


def _expect_failure(control, start, target, config, name: str) -> dict:
    try:
        native.run_bounded_cartesian_segment(control, start, target, config)
    except native.NativeCartesianSegmentError:
        telemetry = native.last_native_cartesian_segment_telemetry()
        assert telemetry is not None
        assert telemetry["failure_name"] == name, telemetry
        assert not telemetry["motion_finished_written"]
        return telemetry
    raise AssertionError(f"expected native failure {name}")


def test_happy_path_and_minimum_jerk() -> None:
    start = _pose()
    target = _pose(0.501)
    control = native.make_fake_cartesian_control_for_offline_test(start)
    telemetry = native.run_bounded_cartesian_segment(
        control, start, target, _config()
    )
    snapshot = _snapshot(control)
    assert telemetry["failure_name"] == "none"
    assert telemetry["success_qualified"]
    assert telemetry["positive_period_writes"] >= 101
    assert telemetry["qualification_rate"] == 1.0
    assert telemetry["motion_finished_written"]
    assert snapshot["motion_finished"].count(True) == 1
    assert snapshot["motion_finished"][-1]

    commands = snapshot["commands"]
    # The rolling-success sample on the qualifying read covers the preceding
    # 100 positive-period holds.  That read receives one final exact hold.
    assert len(commands) > 112
    assert all(command == start for command in commands[:101])
    moving_x = [command[12] for command in commands[101:]]
    assert all(
        moving_x[index] <= moving_x[index + 1] + 1.0e-15
        for index in range(len(moving_x) - 1)
    )
    assert math.isclose(moving_x[-1], target[12], abs_tol=1.0e-15)


def test_exact_hold_probe_never_changes_pose() -> None:
    start = _pose()
    control = native.make_fake_cartesian_control_for_offline_test(start)
    telemetry = native.run_bounded_cartesian_segment(
        control, start, start, _config(duration_s=0.020)
    )
    snapshot = _snapshot(control)
    assert telemetry["success_qualified"]
    assert telemetry["motion_finished_written"]
    assert all(command == start for command in snapshot["commands"])


def test_startup_never_arms_below_quality_threshold() -> None:
    start = _pose()
    control = native.make_fake_cartesian_control_for_offline_test(
        start, initial_success=0.79
    )
    telemetry = _expect_failure(
        control, start, _pose(0.501), _config(), "startup_qualification"
    )
    snapshot = _snapshot(control)
    assert not telemetry["success_qualified"]
    assert telemetry["qualification_control_time_s"] >= 0.5
    assert all(command == start for command in snapshot["commands"])


def test_postqualification_hard_floor_stops_before_next_write() -> None:
    start = _pose()
    control = native.make_fake_cartesian_control_for_offline_test(
        start,
        degrade_after_reads=105,
        degraded_success=0.79,
    )
    telemetry = _expect_failure(
        control, start, _pose(0.501), _config(), "control_success_hard_floor"
    )
    assert telemetry["success_qualified"]
    assert telemetry["latest_success_rate"] == 0.79


def test_complete_half_second_window_is_fail_closed() -> None:
    start = _pose()
    control = native.make_fake_cartesian_control_for_offline_test(
        start,
        degrade_after_reads=102,
        degraded_success=0.94,
    )
    telemetry = _expect_failure(
        control,
        start,
        start,
        _config(duration_s=0.70),
        "control_success_window",
    )
    assert telemetry["success_qualified"]
    assert telemetry["complete_success_windows"] >= 1
    assert telemetry["latest_complete_window_average"] < 0.95


def test_contact_and_endpoint_failures_are_visible() -> None:
    start = _pose()
    contact_control = native.make_fake_cartesian_control_for_offline_test(
        start, fault="contact", fault_after_reads=105
    )
    contact = _expect_failure(
        contact_control,
        start,
        _pose(0.501),
        _config(),
        "contact_or_collision",
    )
    assert contact["success_qualified"]

    frozen_control = native.make_fake_cartesian_control_for_offline_test(
        start, follow_commands=False
    )
    endpoint = _expect_failure(
        frozen_control,
        start,
        _pose(0.501),
        _config(endpoint_timeout_s=0.005, settle_time_s=0.003),
        "endpoint_convergence",
    )
    assert endpoint["final_translation_error_m"] > 0.0009


def test_contract_cannot_weaken_fixed_communication_gates() -> None:
    start = _pose()
    weakened = (
        {"min_control_success_rate": 0.94},
        {"control_success_hard_floor": 0.79},
        {"control_success_evaluation_window_s": 0.51},
        {"startup_deadline_s": 0.51},
        {"startup_min_positive_writes": 99},
        {"read_to_write_budget_ns": 500_001},
    )
    for update in weakened:
        control = native.make_fake_cartesian_control_for_offline_test(start)
        try:
            native.run_bounded_cartesian_segment(
                control, start, start, _config(**update)
            )
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError(f"weakened contract was accepted: {update}")
        assert _snapshot(control)["reads"] == 0


def main() -> None:
    test_happy_path_and_minimum_jerk()
    test_exact_hold_probe_never_changes_pose()
    test_startup_never_arms_below_quality_threshold()
    test_postqualification_hard_floor_stops_before_next_write()
    test_complete_half_second_window_is_fail_closed()
    test_contact_and_endpoint_failures_are_visible()
    test_contract_cannot_weaken_fixed_communication_gates()
    print("offline native Cartesian segment safety tests passed")


if __name__ == "__main__":
    main()
