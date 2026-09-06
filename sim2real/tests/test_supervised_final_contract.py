"""Final safety-contract tests for the short supervised V94 CLI.

These tests deliberately inject the real runner.  They must remain hardware
inert and prove that the device-facing factory boundary can be crossed only
with the explicit execute and operator-supervision flags.
"""

from __future__ import annotations

import pytest

from sim2real.deployment import runner as cli
from sim2real.runtime import supervised_v94_runtime as runtime
from robot_control.franka.session import (
    load_experimental_supervised_franka_envelope,
)


def test_speed_cap_is_actual_supervised_contract_value() -> None:
    request = cli.SupervisedRequest(
        run_id="DRY_RUN",
        steps=1,
        bundle=cli.DEFAULT_BUNDLE,
        profile=cli.DEFAULT_PROFILE,
        pcd_config=cli.DEFAULT_PCD_CONFIG,
        execute=False,
    )

    summary = cli._summary(request)

    assert cli.FRANKA_MAX_COMMAND_SPEED_RAD_S == pytest.approx(0.5)
    assert cli.FRANKA_MAX_MEASURED_VELOCITY_RAD_S == pytest.approx(0.7)
    assert summary["hard_guards"]["franka_command_speed_rad_s"] == pytest.approx(
        0.5
    )
    assert summary["hard_guards"][
        "franka_command_acceleration_rad_s2"
    ] == pytest.approx(5.0)
    assert summary["hard_guards"]["franka_command_jerk_rad_s3"] == pytest.approx(
        250.0
    )
    assert summary["hard_guards"][
        "franka_measured_velocity_fault_rad_s"
    ] == pytest.approx(0.7)
    assert summary["hard_guards"][
        "franka_collision_behavior_applied_before_control"
    ] is True
    assert summary["hard_guards"]["franka_contact_torque_thresholds_nm"] == [
        20.0,
        20.0,
        18.0,
        18.0,
        16.0,
        14.0,
        12.0,
    ]
    assert summary["hard_guards"]["franka_contact_force_thresholds_n"] == [
        20.0,
        20.0,
        20.0,
        25.0,
        25.0,
        25.0,
    ]
    assert summary["hard_guards"]["franka_collision_torque_thresholds_nm"] == [
        40.0,
        40.0,
        36.0,
        36.0,
        32.0,
        28.0,
        24.0,
    ]
    assert summary["hard_guards"]["franka_collision_force_thresholds_n"] == [
        40.0,
        40.0,
        40.0,
        50.0,
        50.0,
        50.0,
    ]
    assert cli.FRANKA_OBSERVATION_ACTION_MAX_AGE_S == pytest.approx(0.040)
    assert cli.FRANKA_OBSERVATION_HARD_MAX_AGE_S == pytest.approx(0.05)
    assert summary["hard_guards"][
        "franka_observation_action_max_age_s"
    ] == pytest.approx(0.040)
    assert summary["hard_guards"][
        "franka_observation_hard_max_age_s"
    ] == pytest.approx(0.05)
    assert summary["hard_guards"]["rh56_stop_timeout_s"] == pytest.approx(5.0)
    assert runtime.FRANKA_OBSERVATION_ACTION_MAX_AGE_S == pytest.approx(
        cli.FRANKA_OBSERVATION_ACTION_MAX_AGE_S
    )
    assert runtime.FRANKA_OBSERVATION_HARD_MAX_AGE_S == pytest.approx(
        cli.FRANKA_OBSERVATION_HARD_MAX_AGE_S
    )
    assert runtime.RH56_STOP_TIMEOUT_S == pytest.approx(
        cli.RH56_STOP_TIMEOUT_S
    )
    assert cli.RH56_SPEED_SET == (600, 600, 600, 600, 600, 600)
    assert runtime.RH56_SPEED_SET == cli.RH56_SPEED_SET
    assert summary["sim_control_alignment"]["source_sha256"] == (
        "9934815923ba758d9a089fb0605b5473822fe30454718456b0a96c2bf6a0610a"
    )
    assert summary["sim_control_alignment"]["rh56"]["response_profile"] == (
        "uniform_speed_600_identified_free_space"
    )
    expected_rh56_contract_envelope = (137, 143, 158, 158, 251, 120)
    assert cli.DEFAULT_POLICY_RATE_MODE.rh56_max_register_delta_per_update == (
        expected_rh56_contract_envelope
    )
    assert tuple(
        summary["hard_guards"]["rh56_target_slew_units_per_update"]
    ) == expected_rh56_contract_envelope
    assert summary["hard_guards"]["franka_episode_delta_rad"] == pytest.approx(
        1.21
    )
    assert summary["hard_step_cap"] == 720
    assert summary["hard_guards"][
        "franka_maximum_session_duration_s"
    ] == pytest.approx(15.0)

    envelope = load_experimental_supervised_franka_envelope(cli.DEFAULT_PROFILE)
    assert tuple(envelope.maximum_velocity_rad_s) == pytest.approx((0.5,) * 7)
    assert tuple(envelope.maximum_target_rate_rad_s) == pytest.approx((0.5,) * 7)
    assert summary["hard_guards"][
        "franka_first_bootstrap_read_to_write_s"
    ] == pytest.approx(envelope.read_to_write_deadline_s)
    assert summary["hard_guards"]["franka_steady_read_to_write_s"] == pytest.approx(
        envelope.read_to_write_deadline_s
    )
    assert envelope.maximum_session_duration_s == pytest.approx(15.0)


@pytest.mark.parametrize("steps", ("1", "720"))
def test_steps_boundary_values_are_accepted_without_device_access(steps: str) -> None:
    calls: list[object] = []

    result = cli.main(
        ["--steps", steps],
        real_runner=lambda request: calls.append(request) or {},
    )

    assert result == 0
    assert calls == []


@pytest.mark.parametrize("steps", ("0", "721", "1.0", "-1", "True"))
def test_invalid_step_count_is_refused_before_device_access(steps: str) -> None:
    calls: list[object] = []

    result = cli.main(
        [
            "--execute",
            "--yes-i-am-supervising",
            "--run-id",
            "boundary-test",
            "--steps",
            steps,
        ],
        real_runner=lambda request: calls.append(request) or {},
    )

    assert result == 2
    assert calls == []


def test_missing_supervision_flag_never_crosses_real_runner_boundary() -> None:
    calls: list[object] = []

    result = cli.main(
        [
            "--execute",
            "--run-id",
            "confirm-test",
            "--steps",
            "10",
        ],
        real_runner=lambda request: calls.append(request) or {},
    )

    assert result == 2
    assert calls == []


def test_execute_and_supervision_flags_cross_real_runner_boundary() -> None:
    calls: list[cli.SupervisedRequest] = []

    def runner(request: cli.SupervisedRequest) -> dict[str, object]:
        calls.append(request)
        return {"result": "PASS"}

    result = cli.main(
        [
            "--execute",
            "--yes-i-am-supervising",
            "--run-id",
            "confirm-test",
            "--steps",
            "1",
        ],
        real_runner=runner,
    )

    assert result == 0
    assert len(calls) == 1
    assert calls[0].execute is True
    assert calls[0].run_id == "confirm-test"
    assert calls[0].steps == 1
