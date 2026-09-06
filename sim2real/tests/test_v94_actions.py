import numpy as np
import pytest

from sim2real.contracts.actions import (
    QD_G015_FRANKA_ACTION_CONTRACT_ID,
    TransactionalRH56HardwareCommandShaper,
    V94ActionMapper,
)
from sim2real.contracts.v94 import (
    POLICY_HAND_ORDER,
    Q_HAND_SEMANTIC_CLOSE_RAD,
    REGISTER_HAND_ORDER,
)

LIMITS = np.asarray([[-3.0, 3.0]] * 7, dtype=np.float32)
FRANKA_SAFE_LIMITS_F64 = np.asarray(
    [
        [-2.6937, 2.6937],
        [-1.7337, 1.7337],
        [-2.8507, 2.8507],
        [-2.9921, -0.2018],
        [-2.7565, 2.7565],
        [0.5945, 4.4669],
        [-2.9659, 2.9659],
    ],
    dtype=np.float64,
)


def test_three_policy_tick_register_envelope_matches_v94_contract():
    by_name = dict(zip(POLICY_HAND_ORDER, Q_HAND_SEMANTIC_CLOSE_RAD.tolist()))
    q_close_register_order = np.asarray(
        [by_name[name] for name in REGISTER_HAND_ORDER],
        dtype=np.float64,
    )
    envelope = np.ceil(
        3.0 * 1000.0 * 0.05 / q_close_register_order
    ).astype(np.int32)

    np.testing.assert_array_equal(
        envelope,
        np.asarray([137, 143, 158, 158, 251, 120], dtype=np.int32),
    )


def test_exact_nominal_arm_and_hand_mapping():
    mapper = V94ActionMapper(
        initial_arm_target_q_rad=np.zeros(7),
        initial_hand_target_q_policy_order_rad=np.zeros(6),
        joint_limits_rad=LIMITS,
    )
    action = np.asarray([1.0] * 7 + [1.0] * 6, dtype=np.float32)
    mapped = mapper.map(action, measured_q_rad=np.zeros(7))
    np.testing.assert_allclose(mapped.franka_target_q_rad, 0.003, atol=1.0e-8)
    np.testing.assert_allclose(
        mapped.rh56_target_q_policy_order_rad,
        np.minimum(0.2 * Q_HAND_SEMANTIC_CLOSE_RAD, 0.05),
        atol=1.0e-8,
    )
    expected_registers = np.rint(
        1000 * (1 - mapped.rh56_target_q_policy_order_rad / Q_HAND_SEMANTIC_CLOSE_RAD)
    ).astype(np.int32)[::-1]
    np.testing.assert_array_equal(
        mapped.rh56_angle_set_register_order, expected_registers
    )


def test_external_hold_gate_zeroes_only_arm_action():
    mapper = V94ActionMapper(
        initial_arm_target_q_rad=np.zeros(7),
        initial_hand_target_q_policy_order_rad=np.zeros(6),
        joint_limits_rad=LIMITS,
    )
    mapped = mapper.map(
        np.ones(13, dtype=np.float32),
        measured_q_rad=np.zeros(7),
        hold_arm_target=True,
    )
    np.testing.assert_array_equal(mapped.franka_target_q_rad, np.zeros(7))
    np.testing.assert_array_equal(mapped.executed_policy_action13[:7], np.zeros(7))
    np.testing.assert_array_equal(mapped.executed_policy_action13[7:], np.ones(6))
    assert any("hold gate" in reason for reason in mapped.reasons)


def test_measured_q_envelope_is_fail_closed():
    mapper = V94ActionMapper(
        initial_arm_target_q_rad=np.full(7, 0.2),
        initial_hand_target_q_policy_order_rad=np.zeros(6),
        joint_limits_rad=LIMITS,
    )
    mapped = mapper.map(np.zeros(13), measured_q_rad=np.zeros(7))
    np.testing.assert_allclose(mapped.franka_target_q_rad, 0.05)
    assert any("measured-q" in reason for reason in mapped.reasons)


def test_qd_g015_arm_mapping_uses_current_shaper_qd_without_old_envelopes():
    mapper = V94ActionMapper(
        initial_arm_target_q_rad=np.full(7, -0.7),
        initial_hand_target_q_policy_order_rad=np.zeros(6),
        joint_limits_rad=LIMITS,
        franka_action_contract_id=QD_G015_FRANKA_ACTION_CONTRACT_ID,
    )
    action = np.asarray([1.0] * 7 + [-1.0] * 6, dtype=np.float32)
    mapped = mapper.map(
        action,
        measured_q_rad=np.full(7, -2.0),
        shaper_q_d_rad=np.full(7, 0.1),
    )

    np.testing.assert_allclose(mapped.franka_target_q_rad, 0.16, atol=1.0e-7)
    assert not any("measured-q" in reason for reason in mapped.reasons)
    assert not any("per-tick target step" in reason for reason in mapped.reasons)


def test_qd_g015_arm_mapping_does_not_accumulate_from_previous_target():
    mapper = V94ActionMapper(
        initial_arm_target_q_rad=np.zeros(7),
        initial_hand_target_q_policy_order_rad=np.zeros(6),
        joint_limits_rad=LIMITS,
        franka_action_contract_id=QD_G015_FRANKA_ACTION_CONTRACT_ID,
    )
    action = np.asarray([1.0] * 7 + [-1.0] * 6, dtype=np.float32)
    first = mapper.map(
        action,
        measured_q_rad=np.zeros(7),
        shaper_q_d_rad=np.zeros(7),
    )
    second = mapper.map(
        action,
        measured_q_rad=np.zeros(7),
        shaper_q_d_rad=np.full(7, 0.004285),
    )

    np.testing.assert_allclose(first.franka_target_q_rad, 0.06, atol=1.0e-7)
    np.testing.assert_allclose(second.franka_target_q_rad, 0.064285, atol=1.0e-7)
    assert np.all(second.franka_target_q_rad < 0.12)


def test_qd_g015_arm_mapping_requires_explicit_shaper_state():
    mapper = V94ActionMapper(
        initial_arm_target_q_rad=np.zeros(7),
        initial_hand_target_q_policy_order_rad=np.zeros(6),
        joint_limits_rad=LIMITS,
        franka_action_contract_id=QD_G015_FRANKA_ACTION_CONTRACT_ID,
    )
    with pytest.raises(ValueError, match="requires shaper_q_d_rad"):
        mapper.map(np.zeros(13), measured_q_rad=np.zeros(7))


def test_float32_joint_limit_targets_stay_inside_all_original_double_bounds():
    midpoint = np.mean(FRANKA_SAFE_LIMITS_F64, axis=1)
    mapper = V94ActionMapper(
        initial_arm_target_q_rad=midpoint,
        initial_hand_target_q_policy_order_rad=np.zeros(6),
        joint_limits_rad=FRANKA_SAFE_LIMITS_F64,
        franka_action_contract_id=QD_G015_FRANKA_ACTION_CONTRACT_ID,
    )

    rounded = FRANKA_SAFE_LIMITS_F64.astype(np.float32)
    expected_lower = rounded[:, 0].copy()
    expected_upper = rounded[:, 1].copy()
    lower_outward = (
        expected_lower.astype(np.float64) < FRANKA_SAFE_LIMITS_F64[:, 0]
    )
    upper_outward = (
        expected_upper.astype(np.float64) > FRANKA_SAFE_LIMITS_F64[:, 1]
    )
    expected_lower[lower_outward] = np.nextafter(
        expected_lower[lower_outward], np.float32(np.inf)
    )
    expected_upper[upper_outward] = np.nextafter(
        expected_upper[upper_outward], np.float32(-np.inf)
    )

    np.testing.assert_array_equal(mapper.joint_limits[:, 0], expected_lower)
    np.testing.assert_array_equal(mapper.joint_limits[:, 1], expected_upper)
    assert np.all(
        mapper.joint_limits[:, 0].astype(np.float64)
        >= FRANKA_SAFE_LIMITS_F64[:, 0]
    )
    assert np.all(
        mapper.joint_limits[:, 1].astype(np.float64)
        <= FRANKA_SAFE_LIMITS_F64[:, 1]
    )

    lower = mapper.map(
        np.asarray([-1.0] * 7 + [-1.0] * 6, dtype=np.float32),
        measured_q_rad=midpoint,
        shaper_q_d_rad=FRANKA_SAFE_LIMITS_F64[:, 0] - 1.0,
    ).franka_target_q_rad
    upper = mapper.map(
        np.asarray([1.0] * 7 + [-1.0] * 6, dtype=np.float32),
        measured_q_rad=midpoint,
        shaper_q_d_rad=FRANKA_SAFE_LIMITS_F64[:, 1] + 1.0,
    ).franka_target_q_rad

    np.testing.assert_array_equal(lower, expected_lower)
    np.testing.assert_array_equal(upper, expected_upper)
    assert np.all(lower.astype(np.float64) >= FRANKA_SAFE_LIMITS_F64[:, 0])
    assert np.all(upper.astype(np.float64) <= FRANKA_SAFE_LIMITS_F64[:, 1])

    bad_j4_float32 = np.float32(FRANKA_SAFE_LIMITS_F64[3, 0])
    assert float(bad_j4_float32) < FRANKA_SAFE_LIMITS_F64[3, 0]
    assert lower[3] != bad_j4_float32
    assert lower[3] == np.nextafter(bad_j4_float32, np.float32(np.inf))


def test_float32_joint_limit_canonicalization_does_not_widen_or_change_normal_map():
    mapper = V94ActionMapper(
        initial_arm_target_q_rad=np.zeros(7),
        initial_hand_target_q_policy_order_rad=np.zeros(6),
        joint_limits_rad=LIMITS.astype(np.float64),
        franka_action_contract_id=QD_G015_FRANKA_ACTION_CONTRACT_ID,
    )
    np.testing.assert_array_equal(mapper.joint_limits, LIMITS)
    mapped = mapper.map(
        np.asarray([0.5] * 7 + [-1.0] * 6, dtype=np.float32),
        measured_q_rad=np.zeros(7),
        shaper_q_d_rad=np.full(7, 0.1, dtype=np.float32),
    )
    np.testing.assert_allclose(mapped.franka_target_q_rad, 0.13, atol=1.0e-7)

    invalid = FRANKA_SAFE_LIMITS_F64.copy()
    invalid[4] = [0.5, 0.5]
    with pytest.raises(ValueError, match="lower bounds must be below upper bounds"):
        V94ActionMapper(
            initial_arm_target_q_rad=np.zeros(7),
            initial_hand_target_q_policy_order_rad=np.zeros(6),
            joint_limits_rad=invalid,
        )


def test_rh56_hardware_shaper_is_transactional_20hz_and_register_rate_limited():
    shaper = TransactionalRH56HardwareCommandShaper(
        initial_angle_set_register_order=np.full(6, 1000, dtype=np.int32),
        policy_rate_hz=60.0,
        hardware_rate_hz=20.0,
        maximum_register_delta_per_update=np.full(6, 30, dtype=np.int32),
    )

    discarded = shaper.propose(1, np.zeros(6, dtype=np.int32))
    np.testing.assert_array_equal(
        discarded.angle_set_register_order,
        np.full(6, 970, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        shaper.committed_target(),
        np.full(6, 1000, dtype=np.int32),
    )
    shaper.discard_unstaged(discarded)

    first = shaper.propose(1, np.zeros(6, dtype=np.int32))
    shaper.commit(first)
    np.testing.assert_array_equal(
        shaper.committed_target(),
        np.full(6, 970, dtype=np.int32),
    )


    # Sequences 2 and 3 retain the committed physical target even though the
    # newest desired target changes.  Their proposals still commit sequence
    # continuity, but cannot consume a hardware update or register slew.
    second = shaper.propose(2, np.full(6, 1000, dtype=np.int32))
    assert second.hardware_update_due is False
    np.testing.assert_array_equal(
        second.angle_set_register_order,
        first.angle_set_register_order,
    )
    shaper.commit(second)
    third = shaper.propose(3, np.zeros(6, dtype=np.int32))
    assert third.hardware_update_due is False
    np.testing.assert_array_equal(
        third.angle_set_register_order,
        first.angle_set_register_order,
    )
    shaper.commit(third)

    fourth = shaper.propose(4, np.zeros(6, dtype=np.int32))
    assert fourth.hardware_update_due is True
    np.testing.assert_array_equal(
        fourth.angle_set_register_order,
        np.full(6, 940, dtype=np.int32),
    )
    shaper.commit(fourth)


def test_20hz_policy_releases_one_bounded_rh56_target_per_tick() -> None:
    maximum_delta = np.asarray([46, 48, 53, 53, 84, 40], dtype=np.int32)
    shaper = TransactionalRH56HardwareCommandShaper(
        initial_angle_set_register_order=np.full(6, 1000, dtype=np.int32),
        policy_rate_hz=20.0,
        hardware_rate_hz=20.0,
        maximum_register_delta_per_update=maximum_delta,
    )

    first = shaper.propose(1, np.zeros(6, dtype=np.int32))
    assert first.hardware_update_due is True
    np.testing.assert_array_equal(
        first.angle_set_register_order,
        np.full(6, 1000, dtype=np.int32) - maximum_delta,
    )
    shaper.commit(first)

    second = shaper.propose(2, np.zeros(6, dtype=np.int32))
    assert second.hardware_update_due is True
    np.testing.assert_array_equal(
        second.angle_set_register_order,
        np.full(6, 1000, dtype=np.int32) - 2 * maximum_delta,
    )


def test_rh56_hardware_shaper_uses_commissioned_q6_416_floor() -> None:
    shaper = TransactionalRH56HardwareCommandShaper(
        initial_angle_set_register_order=np.full(6, 1000, dtype=np.int32),
        policy_rate_hz=60.0,
        hardware_rate_hz=20.0,
        maximum_register_delta_per_update=np.full(6, 1000, dtype=np.int32),
        minimum_angle_set_register_order=(0, 0, 0, 0, 0, 416),
        maximum_angle_set_register_order=(1000,) * 6,
    )
    desired = np.asarray([61, 17, 524, 758, 422, 400], dtype=np.int32)
    proposal = shaper.propose(1, desired)

    np.testing.assert_array_equal(
        proposal.bounded_desired_angle_set_register_order,
        np.asarray([61, 17, 524, 758, 422, 416], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        proposal.angle_set_register_order,
        np.asarray([61, 17, 524, 758, 422, 416], dtype=np.int32),
    )
