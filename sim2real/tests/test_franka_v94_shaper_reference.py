import numpy as np

from sim2real.diagnostics.franka_shaper_demo import run_demo
from robot_control.reference.franka_v225_interpolator_reference import (
    MAX_COMMAND_ACCELERATION_RAD_S2,
    MAX_COMMAND_JERK_RAD_S3,
    MAX_COMMAND_VELOCITY_RAD_S,
    SERVO_DT_S,
    ServoSubstepAccumulator,
    V225FrankaCommandInterpolator,
    map_qd_g015_20hz_arm_action,
    map_v225_20hz_arm_action,
)
from robot_control.reference.franka_v94_shaper_reference import (
    map_current_v94_arm_action,
)


def test_legacy_reference_modules_are_exact_aliases():
    import robot_control.reference.franka_v225_interpolator_reference as legacy_interpolator
    import robot_control.reference.franka_v94_shaper_reference as legacy_shaper

    from robot_control.reference import (
        franka_v225_interpolator_reference as canonical_interpolator,
    )
    from robot_control.reference import (
        franka_v94_shaper_reference as canonical_shaper,
    )

    assert legacy_interpolator is canonical_interpolator
    assert legacy_shaper is canonical_shaper


def test_action_mapping_preserves_legacy_and_v212_contracts():
    q_home = np.asarray(
        [0.0, -0.569, 0.0, -2.81, 0.0, 3.037, 0.741], dtype=np.float32
    )
    ones = np.ones(7, dtype=np.float32)
    np.testing.assert_allclose(
        map_current_v94_arm_action(q_home, q_home, ones) - q_home,
        0.003,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        map_v225_20hz_arm_action(q_home, q_home, ones) - q_home,
        0.018,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        map_qd_g015_20hz_arm_action(q_home, ones) - q_home,
        0.06,
        atol=2.0e-7,
    )


def test_qd_g015_mapping_uses_current_shaper_state_not_previous_or_measured_q():
    q_d = np.asarray(
        [0.2, -0.8, 0.1, -2.4, -0.3, 2.7, 0.5], dtype=np.float32
    )
    action = np.asarray([1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.0], np.float32)
    expected = q_d + np.float32(0.06) * np.clip(action, -1.0, 1.0)
    np.testing.assert_array_equal(
        map_qd_g015_20hz_arm_action(q_d, action), expected
    )


def test_120hz_substep_schedule_exactly_covers_one_20hz_period():
    schedule = ServoSubstepAccumulator()
    counts = [schedule.substeps_for(1.0 / 120.0) for _ in range(6)]
    assert counts == [8, 8, 9, 8, 8, 9]
    assert sum(counts) == 50


def test_demo_stays_inside_native_derivative_envelope():
    trace = run_demo(policy_steps=20)
    dq = trace["servo_dq_rad_s"]
    ddq = trace["servo_ddq_rad_s2"]
    jerk = np.diff(
        np.concatenate((np.zeros((1, 7)), ddq), axis=0), axis=0
    ) / SERVO_DT_S
    assert np.max(np.abs(dq)) <= MAX_COMMAND_VELOCITY_RAD_S + 1.0e-5
    assert np.max(np.abs(ddq)) <= MAX_COMMAND_ACCELERATION_RAD_S2 + 1.0e-5
    assert np.max(np.abs(jerk)) <= MAX_COMMAND_JERK_RAD_S3 + 1.0e-5
    assert trace["physics_command_q_rad"].shape == (20 * 6, 7)
    assert trace["servo_q_rad"].shape == (20 * 50, 7)


def test_v225_frozen_18_mrad_step_response():
    generator = V225FrankaCommandInterpolator(np.zeros(7))
    target = np.zeros(7)
    target[0] = float(np.float32(0.045) * np.float32(0.40))
    samples = {}
    for packet in range(1, 301):
        state = generator.step(target)
        if packet in (1, 50, 100, 300):
            samples[packet] = state.q_rad[0]
    np.testing.assert_allclose(
        samples[1], 0.00000025, rtol=0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        samples[50], 0.004284888723940, rtol=0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        samples[100], 0.015174792163919, rtol=0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        samples[300], 0.018001712903260, rtol=0.0, atol=1.0e-12
    )


def test_persistent_generator_clamps_velocity_on_long_reversing_trajectory():
    generator = V225FrankaCommandInterpolator(np.zeros(7))
    emitted_q = [generator.state.q_rad.copy()]
    for policy_tick in range(80):
        direction = 1.0 if (policy_tick // 7) % 2 == 0 else -1.0
        target = np.full(7, 0.5 * direction, dtype=np.float64)
        for _ in range(50):
            state = generator.step(target)
            emitted_q.append(state.q_rad.copy())
    emitted_q = np.asarray(emitted_q)
    emitted_dq = np.diff(emitted_q, axis=0) / SERVO_DT_S
    emitted_ddq = np.diff(
        np.concatenate((np.zeros((1, 7)), emitted_dq), axis=0), axis=0
    ) / SERVO_DT_S
    emitted_jerk = np.diff(
        np.concatenate((np.zeros((1, 7)), emitted_ddq), axis=0), axis=0
    ) / SERVO_DT_S
    assert np.max(np.abs(emitted_dq)) <= MAX_COMMAND_VELOCITY_RAD_S + 1.0e-9
    assert (
        np.max(np.abs(emitted_ddq))
        <= MAX_COMMAND_ACCELERATION_RAD_S2 + 1.0e-6
    )
    assert np.max(np.abs(emitted_jerk)) <= MAX_COMMAND_JERK_RAD_S3 + 1.0e-3
