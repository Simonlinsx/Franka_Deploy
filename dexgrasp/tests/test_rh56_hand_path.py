from anydex_pipeline.rh56_hand_path import (
    build_rh56_no_contact_execution_path,
    dense_hand_interval_sha256,
    dense_hand_interval_tubes_sha256,
    feedback_q12_envelopes_sha256,
    require_rh56_feedback_in_interval,
    rh56_feedback_envelope_policy,
    rh56_feedback_interval_bounds,
    rh56_hand_interval_feedback_q12_envelopes,
    rh56_hand_interval_q12_paths_and_feedback_tubes,
    validate_rh56_hand_execution_path,
)
import pytest
from anydex_pipeline.rh56_actuator_mapping import OfficialRH56ActuatorMapper
from pathlib import Path


def test_path_is_driver_exact_q6_then_five_axes_and_reverse():
    path = build_rh56_no_contact_execution_path((950, 925, 900, 875, 850, 900))
    values = [item.actuator_configuration for item in path.waypoints]
    commands = [item.command_targets for item in path.waypoints]
    assert values[1:5] == [
        (1000, 1000, 1000, 1000, 1000, 975),
        (1000, 1000, 1000, 1000, 1000, 950),
        (1000, 1000, 1000, 1000, 1000, 925),
        (1000, 1000, 1000, 1000, 1000, 900),
    ]
    assert commands[0] == (-1, -1, -1, -1, -1, -1)
    assert commands[1] == (-1, -1, -1, -1, -1, 975)
    assert values[-1] == (1000, 1000, 1000, 1000, 1000, 1000)
    assert commands[-1] == (-1, -1, -1, -1, -1, 1000)
    assert path.waypoints[5].phase == "bend_forward_0000"
    q6_reverse = [
        item.actuator_configuration[5]
        for item in path.waypoints
        if item.phase.startswith("q6_reverse_")
    ]
    assert q6_reverse == [950, 975, 1000]
    assert path.as_dict()["direction_reversal_bootstrap_min_units"] == 50


def test_path_hash_covers_command_configuration_order_and_roundtrips():
    path = build_rh56_no_contact_execution_path((800, 820, 840, 860, 880, 900))
    payload = path.as_dict()
    rebuilt = validate_rh56_hand_execution_path(payload)
    assert rebuilt.sha256 == path.sha256
    payload["waypoints"][0]["command_targets"][5] -= 1
    try:
        validate_rh56_hand_execution_path(payload)
    except ValueError:
        pass
    else:
        raise AssertionError("modified hand command path was accepted")


def test_non_divisible_targets_end_exactly_and_reverse_to_open():
    path = build_rh56_no_contact_execution_path((997, 991, 983, 977, 969, 913))
    forward = [item for item in path.waypoints if item.phase.startswith("bend_forward_")]
    assert forward[-1].actuator_configuration == (997, 991, 983, 977, 969, 913)
    assert path.waypoints[-1].actuator_configuration == (1000,) * 6


def test_target900_reversal_skips_deadband_first_command_and_dense_tube_is_bound():
    path = build_rh56_no_contact_execution_path((900, 900, 900, 900, 900, 900))
    reverse = [
        item.command_targets[5]
        for item in path.waypoints
        if item.phase.startswith("q6_reverse_")
    ]
    assert reverse == [950, 975, 1000]
    assert 925 not in reverse
    root = Path(__file__).parents[1] / "third_party/AnyDexGrasp"
    mapper = OfficialRH56ActuatorMapper.from_anydex_root(root)
    intervals, tubes = rh56_hand_interval_q12_paths_and_feedback_tubes(
        path, mapper, 20
    )
    assert len(intervals) == len(path.waypoints) - 1 == len(tubes)
    assert all(item.shape[1:] == (12,) for item in intervals)
    assert all(item.shape == (12,) and (item >= 0).all() for item in tubes)
    assert any((item > 0).any() for item in tubes)
    assert len(dense_hand_interval_sha256(intervals)) == 64
    assert len(dense_hand_interval_tubes_sha256(tubes)) == 64


def test_feedback_q12_envelope_enumerates_tolerance_and_open_discontinuity():
    path = build_rh56_no_contact_execution_path((900, 900, 900, 900, 900, 900))
    root = Path(__file__).parents[1] / "third_party/AnyDexGrasp"
    mapper = OfficialRH56ActuatorMapper.from_anydex_root(root)
    lowers, uppers = rh56_hand_interval_feedback_q12_envelopes(
        path, mapper, 25
    )
    assert len(lowers) == len(path.waypoints) - 1 == len(uppers)
    assert len(feedback_q12_envelopes_sha256(lowers, uppers)) == 64
    interval = next(
        index
        for index, (first, second) in enumerate(
            zip(path.waypoints[:-1], path.waypoints[1:])
        )
        if first.actuator_configuration[2] == 1000
        and second.actuator_configuration[2] == 975
    )
    lower = lowers[interval]
    upper = uppers[interval]
    assert lower.shape == upper.shape == (12,)
    assert (lower <= upper).all()
    # The official source has an explicit [1000] -> [0, 0] special case;
    # register 999 has a non-zero distal angle.  Both must be retained.
    at_open = mapper.to_joint_positions_rad((900, 900, 1000, 1000, 1000, 900))
    at_999 = mapper.to_joint_positions_rad((900, 900, 999, 1000, 1000, 900))
    assert lower[3] <= min(at_open[3], at_999[3])
    assert upper[3] >= max(at_open[3], at_999[3])
    # Independent arrival-error extremes of the unchanged index axis and the
    # moving middle axis are also inside the conservative component box.
    for middle, index in ((950, 975), (950, 1000), (1000, 975), (1000, 1000)):
        q12 = mapper.to_joint_positions_rad(
            (900, 900, middle, index, 1000, 900)
        )
        assert (q12 >= lower - 1.0e-15).all()
        assert (q12 <= upper + 1.0e-15).all()


def test_feedback_envelope_accepts_normal_lag_but_rejects_either_side_escape():
    previous = (1000, 1000, 1000, 1000, 1000, 950)
    current = (975, 1000, 1000, 1000, 1000, 950)
    lower, upper = rh56_feedback_interval_bounds(
        previous,
        current,
        20,
        previous_phase="q6_forward_0013",
        current_phase="bend_forward_0000",
    )
    assert lower == (955, 980, 980, 980, 980, 930)
    assert upper == (1000, 1000, 1000, 1000, 1000, 970)
    # Immediately after the write, the actuator may still be at the preceding
    # endpoint.  That is normal lag and must not be mistaken for an escape.
    require_rh56_feedback_in_interval(
        previous,
        previous,
        current,
        20,
        previous_phase="q6_forward_0013",
        current_phase="bend_forward_0000",
    )
    with pytest.raises(ValueError, match="escaped"):
        require_rh56_feedback_in_interval(
            (954, 1000, 1000, 1000, 1000, 950),
            previous,
            current,
            20,
            previous_phase="q6_forward_0013",
            current_phase="bend_forward_0000",
        )
    with pytest.raises(ValueError, match="escaped"):
        require_rh56_feedback_in_interval(
            (1000, 1000, 1000, 1000, 1000, 971),
            previous,
            current,
            20,
            previous_phase="q6_forward_0013",
            current_phase="bend_forward_0000",
        )


def test_feedback_envelope_binds_q6_reverse_hysteresis_and_policy_hash():
    policy = rh56_feedback_envelope_policy(20)
    assert policy["arrival_tolerance_units"] == 20
    assert policy["q6_reverse_hysteresis_tolerance_units"] == 30
    assert len(policy["sha256"]) == 64
    # The prior accepted 923 feedback for target 950 remains legal while the
    # next 975 command starts: the q6 reverse endpoint band is 30 units.
    lower, upper = rh56_feedback_interval_bounds(
        (1000, 1000, 1000, 1000, 1000, 950),
        (1000, 1000, 1000, 1000, 1000, 975),
        20,
        previous_phase="q6_reverse_0000",
        current_phase="q6_reverse_0001",
    )
    assert lower[5] == 920
    assert upper[5] == 1000
    require_rh56_feedback_in_interval(
        (1000, 1000, 1000, 1000, 1000, 923),
        (1000, 1000, 1000, 1000, 1000, 950),
        (1000, 1000, 1000, 1000, 1000, 975),
        20,
        previous_phase="q6_reverse_0000",
        current_phase="q6_reverse_0001",
    )
