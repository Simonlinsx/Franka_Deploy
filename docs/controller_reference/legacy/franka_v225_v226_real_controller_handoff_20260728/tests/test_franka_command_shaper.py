"""Regression tests for the batched real Franka command shaper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TORCH_SHAPER_PATH = ROOT / (
    "source/simtoolreal_lab/simtoolreal_lab/tasks/dynamic_dexterous_grasp/"
    "franka_command_shaper.py"
)
NUMPY_SHAPER_PATH = ROOT / "shaper/franka_v94_shaper_reference.py"
INTERPOLATED_REFERENCE_PATH = ROOT / "shaper/franka_v225_interpolated_reference.py"
CFG_PATH = TORCH_SHAPER_PATH.with_name("dynamic_dexterous_grasp_env_cfg.py")
ENV_PATH = TORCH_SHAPER_PATH.with_name("dynamic_dexterous_grasp_env.py")
TASK_INIT_PATH = TORCH_SHAPER_PATH.with_name("__init__.py")
TRAIN_SCRIPT_PATH = ROOT / "scripts/train_rl_games.py"
EVAL_SCRIPT_PATH = ROOT / "scripts/evaluate_rl_games.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


torch_shaper = _load_module("torch_franka_command_shaper", TORCH_SHAPER_PATH)
numpy_shaper = _load_module("numpy_franka_command_shaper", NUMPY_SHAPER_PATH)
interpolated_reference = _load_module(
    "numpy_franka_interpolated_reference", INTERPOLATED_REFERENCE_PATH
)


Q_HOME = np.asarray(
    [0.0, -0.569000006, 0.0, -2.809999943, 0.0, 3.036999941, 0.740999997],
    dtype=np.float64,
)


def test_120hz_schedule_is_exactly_8_8_9_and_50_ticks_per_policy_step():
    accumulator = numpy_shaper.ServoSubstepAccumulator()
    counts = [accumulator.substeps_for(1.0 / 120.0) for _ in range(12)]
    assert counts == [8, 8, 9] * 4
    assert sum(counts[:6]) == 50
    assert torch_shaper.SERVO_TICKS_PER_120HZ_STEP == (8, 8, 9)


def test_torch_action_mapper_matches_float32_numpy_reference():
    rng = np.random.default_rng(314159)
    previous = np.repeat(Q_HOME[None, :], 32, axis=0).astype(np.float32)
    measured = previous + rng.uniform(-0.015, 0.015, size=previous.shape).astype(np.float32)
    actions = rng.uniform(-1.5, 1.5, size=previous.shape).astype(np.float32)
    expected = np.stack(
        [
            numpy_shaper.map_v212_20hz_arm_action(previous[i], measured[i], actions[i])
            for i in range(previous.shape[0])
        ]
    )
    actual = torch_shaper.map_v212_20hz_arm_action_torch(
        torch.from_numpy(previous),
        torch.from_numpy(measured),
        torch.from_numpy(actions),
    ).numpy()
    np.testing.assert_array_equal(actual, expected)
    # Float32 subtraction near a 3 rad joint target has roughly 1e-7 error.
    assert np.max(np.abs(actual - previous)) <= 0.0180002


def test_batched_float64_shaper_regresses_against_numpy_packet_by_packet():
    num_envs = 4
    starts = np.repeat(Q_HOME[None, :], num_envs, axis=0)
    starts += np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.05, -0.02, 0.03, 0.01, -0.04, 0.02, -0.03],
            [-0.04, 0.03, -0.02, 0.02, 0.03, -0.01, 0.04],
            [0.02, 0.01, -0.04, -0.01, 0.02, 0.03, -0.02],
        ],
        dtype=np.float64,
    )
    batch = torch_shaper.BatchedFrankaCommandShaper(
        num_envs, "cpu", dtype=torch.float64, strict=True
    )
    batch.reset(torch.from_numpy(starts))
    references = [numpy_shaper.V94FrankaCommandShaper(start) for start in starts]
    held = starts.astype(np.float32)
    rng = np.random.default_rng(271828)
    schedule = numpy_shaper.ServoSubstepAccumulator()

    max_q_error = 0.0
    max_dq_error = 0.0
    max_ddq_error = 0.0
    for _ in range(24):
        actions = rng.uniform(-1.0, 1.0, size=(num_envs, 7)).astype(np.float32)
        measured = np.stack([reference.state.q_rad for reference in references]).astype(np.float32)
        held = np.stack(
            [
                numpy_shaper.map_v212_20hz_arm_action(held[i], measured[i], actions[i])
                for i in range(num_envs)
            ]
        )
        for _ in range(6):
            servo_ticks = schedule.substeps_for(1.0 / 120.0)
            for _ in range(servo_ticks):
                expected_states = [reference.step(held[i]) for i, reference in enumerate(references)]
                batch.step(torch.from_numpy(held).to(dtype=torch.float64))
                expected_q = np.stack([state.q_rad for state in expected_states])
                expected_dq = np.stack([state.dq_rad_s for state in expected_states])
                expected_ddq = np.stack([state.ddq_rad_s2 for state in expected_states])
                max_q_error = max(max_q_error, float(np.max(np.abs(batch.q_d_rad.numpy() - expected_q))))
                max_dq_error = max(max_dq_error, float(np.max(np.abs(batch.dq_d_rad_s.numpy() - expected_dq))))
                max_ddq_error = max(
                    max_ddq_error,
                    float(np.max(np.abs(batch.ddq_d_rad_s2.numpy() - expected_ddq))),
                )

    assert max_q_error < 1.0e-12
    assert max_dq_error < 1.0e-9
    assert max_ddq_error < 1.0e-6
    assert float(torch.max(torch.abs(batch.dq_d_rad_s))) <= 0.50001
    assert float(torch.max(torch.abs(batch.ddq_d_rad_s2))) <= 4.00001


def test_subset_reset_reinitializes_only_requested_desired_histories():
    batch = torch_shaper.BatchedFrankaCommandShaper(3, "cpu", dtype=torch.float64)
    starts = torch.from_numpy(np.repeat(Q_HOME[None, :], 3, axis=0))
    batch.reset(starts)
    targets = starts + 0.1
    batch.advance(targets, 25)
    untouched = batch.q_d_rad[0].clone()
    reset_q = starts[1:2] - 0.03
    batch.reset(reset_q, torch.tensor([1]))
    torch.testing.assert_close(batch.q_d_rad[0], untouched)
    torch.testing.assert_close(batch.q_d_rad[1], reset_q[0])
    torch.testing.assert_close(batch.dq_d_rad_s[1], torch.zeros(7, dtype=torch.float64))
    torch.testing.assert_close(batch.ddq_d_rad_s2[1], torch.zeros(7, dtype=torch.float64))


def _libfranka_joint_position_reference_step(
    q_d: np.ndarray,
    dq_d: np.ndarray,
    ddq_d: np.ndarray,
    filtered_target: np.ndarray,
    held_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scalar NumPy transcription of libfranka 0.15 joint-position limiting."""

    dt = torch_shaper.SERVO_DT_S
    cutoff = torch_shaper.LIBFRANKA_DEFAULT_CUTOFF_FREQUENCY_HZ
    gain = dt / (dt + 1.0 / (2.0 * np.pi * cutoff))
    lower_q = np.asarray(torch_shaper.SAFE_JOINT_LOWER_RAD, dtype=np.float64)
    upper_q = np.asarray(torch_shaper.SAFE_JOINT_UPPER_RAD, dtype=np.float64)
    target = np.clip(held_target, lower_q, upper_q)
    filtered_target = gain * target + (1.0 - gain) * filtered_target

    max_velocity = np.asarray(
        torch_shaper.LIBFRANKA_MAX_JOINT_VELOCITY_RAD_S,
        dtype=np.float64,
    )
    upper_velocity = np.asarray(
        [
            min(2.62, max(0.0, -0.30 + np.sqrt(max(0.0, 12.0 * (2.75010 - q_d[0]))))),
            min(2.62, max(0.0, -0.20 + np.sqrt(max(0.0, 5.17 * (1.79180 - q_d[1]))))),
            min(2.62, max(0.0, -0.20 + np.sqrt(max(0.0, 7.00 * (2.90650 - q_d[2]))))),
            min(2.62, max(0.0, -0.30 + np.sqrt(max(0.0, 8.00 * (-0.1458 - q_d[3]))))),
            min(5.26, max(0.0, -0.35 + np.sqrt(max(0.0, 34.0 * (2.81010 - q_d[4]))))),
            min(4.18, max(0.0, -0.35 + np.sqrt(max(0.0, 11.0 * (4.52050 - q_d[5]))))),
            min(5.26, max(0.0, -0.35 + np.sqrt(max(0.0, 34.0 * (3.01960 - q_d[6]))))),
        ],
        dtype=np.float64,
    ) - torch_shaper.LIBFRANKA_LIMIT_EPS
    lower_velocity = np.asarray(
        [
            max(-2.62, min(0.0, 0.30 - np.sqrt(max(0.0, 12.0 * (2.75010 + q_d[0]))))),
            max(-2.62, min(0.0, 0.20 - np.sqrt(max(0.0, 5.17 * (1.79180 + q_d[1]))))),
            max(-2.62, min(0.0, 0.20 - np.sqrt(max(0.0, 7.00 * (2.90650 + q_d[2]))))),
            max(-2.62, min(0.0, 0.30 - np.sqrt(max(0.0, 8.00 * (3.04810 + q_d[3]))))),
            max(-5.26, min(0.0, 0.35 - np.sqrt(max(0.0, 34.0 * (2.81010 + q_d[4]))))),
            max(-4.18, min(0.0, 0.35 - np.sqrt(max(0.0, 11.0 * (-0.54092 + q_d[5]))))),
            max(-5.26, min(0.0, 0.35 - np.sqrt(max(0.0, 34.0 * (3.01960 + q_d[6]))))),
        ],
        dtype=np.float64,
    ) + torch_shaper.LIBFRANKA_LIMIT_EPS
    assert np.all(max_velocity >= np.maximum(np.abs(lower_velocity), upper_velocity))

    commanded_velocity = (filtered_target - q_d) / dt
    commanded_jerk = ((commanded_velocity - dq_d) / dt - ddq_d) / dt
    commanded_acceleration = ddq_d + np.clip(
        commanded_jerk,
        -torch_shaper.LIBFRANKA_MAX_JOINT_JERK_RAD_S3,
        torch_shaper.LIBFRANKA_MAX_JOINT_JERK_RAD_S3,
    ) * dt
    ratio = (
        torch_shaper.LIBFRANKA_MAX_JOINT_JERK_RAD_S3
        / torch_shaper.LIBFRANKA_MAX_JOINT_ACCELERATION_RAD_S2
    )
    safe_max_acceleration = np.minimum(
        ratio * (upper_velocity - dq_d),
        torch_shaper.LIBFRANKA_MAX_JOINT_ACCELERATION_RAD_S2,
    )
    safe_min_acceleration = np.maximum(
        ratio * (lower_velocity - dq_d),
        -torch_shaper.LIBFRANKA_MAX_JOINT_ACCELERATION_RAD_S2,
    )
    acceleration = np.clip(
        commanded_acceleration,
        safe_min_acceleration,
        safe_max_acceleration,
    )
    velocity = dq_d + acceleration * dt
    position = q_d + velocity * dt
    return position, velocity, acceleration, filtered_target


def test_libfranka_style_limiter_matches_public_equations_packet_by_packet():
    batch = torch_shaper.BatchedLibfrankaJointPositionRateLimiter(
        2,
        "cpu",
        dtype=torch.float64,
    )
    starts = np.repeat(Q_HOME[None, :], 2, axis=0)
    starts[1] += np.asarray([0.2, -0.1, 0.15, 0.1, -0.2, 0.1, -0.15])
    batch.reset(torch.from_numpy(starts))
    q_d = starts.copy()
    dq_d = np.zeros_like(starts)
    ddq_d = np.zeros_like(starts)
    filtered = starts.copy()
    rng = np.random.default_rng(20260728)

    for packet in range(250):
        if packet % 25 == 0:
            held = starts + rng.uniform(-0.30, 0.30, size=starts.shape)
        expected = [
            _libfranka_joint_position_reference_step(
                q_d[index],
                dq_d[index],
                ddq_d[index],
                filtered[index],
                held[index],
            )
            for index in range(2)
        ]
        q_d = np.stack([value[0] for value in expected])
        dq_d = np.stack([value[1] for value in expected])
        ddq_d = np.stack([value[2] for value in expected])
        filtered = np.stack([value[3] for value in expected])
        batch.step(torch.from_numpy(held))

        np.testing.assert_allclose(batch.q_d_rad.numpy(), q_d, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(batch.dq_d_rad_s.numpy(), dq_d, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(batch.ddq_d_rad_s2.numpy(), ddq_d, rtol=0.0, atol=1.0e-9)
        np.testing.assert_allclose(
            batch.filtered_target_rad.numpy(), filtered, rtol=0.0, atol=1.0e-12
        )


def test_libfranka_style_limiter_respects_acceleration_and_jerk_limits():
    batch = torch_shaper.BatchedLibfrankaJointPositionRateLimiter(
        1,
        "cpu",
        dtype=torch.float64,
    )
    start = torch.from_numpy(Q_HOME[None, :])
    batch.reset(start)
    target = start + 0.30
    previous_acceleration = batch.ddq_d_rad_s2.clone()

    for _ in range(500):
        batch.step(target)
        jerk = (batch.ddq_d_rad_s2 - previous_acceleration) / batch.servo_dt_s
        assert float(torch.max(torch.abs(jerk))) <= batch.max_jerk_rad_s3 + 1.0e-8
        assert (
            float(torch.max(torch.abs(batch.ddq_d_rad_s2)))
            <= batch.max_acceleration_rad_s2 + 1.0e-8
        )
        previous_acceleration = batch.ddq_d_rad_s2.clone()

    assert torch.isfinite(batch.q_d_rad).all()
    assert torch.all(batch.q_d_rad > start)


def test_v224_changes_only_the_franka_execution_limiter_contract():
    cfg_source = CFG_PATH.read_text(encoding="utf-8")
    env_source = ENV_PATH.read_text(encoding="utf-8")
    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")

    assert (
        "class InspireUnifiedRollingJointDeltaV224LibfrankaRateLimiter20HzTeacherEnvCfg"
        in cfg_source
    )
    assert 'franka_command_shaper_mode = "libfranka_joint_position"' in cfg_source
    assert "franka_command_shaper_lowpass_enabled = True" in cfg_source
    assert "franka_command_shaper_lowpass_cutoff_frequency_hz = 100.0" in cfg_source
    assert "BatchedLibfrankaJointPositionRateLimiter" in env_source
    assert (
        "UnifiedRollingJointDeltaV224LibfrankaRateLimiter20HzLSTM-Teacher-Direct-v0"
        in task_source
    )


def test_interpolated_libfranka_commands_are_stable_and_limiter_safe():
    batch = torch_shaper.BatchedLibfrankaJointPositionInterpolator(
        1,
        "cpu",
        dtype=torch.float64,
        tracking_natural_frequency_hz=6.0,
        tracking_damping_ratio=1.0,
    )
    start = torch.from_numpy(Q_HOME[None, :])
    target = start + 0.018
    batch.reset(start)
    previous_acceleration = batch.ddq_d_rad_s2.clone()
    positions = []

    for _ in range(300):
        batch.step(target)
        positions.append(batch.q_d_rad.clone())
        jerk = (batch.ddq_d_rad_s2 - previous_acceleration) / batch.servo_dt_s
        assert float(torch.max(torch.abs(jerk))) <= batch.max_jerk_rad_s3 + 1.0e-8
        assert (
            float(torch.max(torch.abs(batch.ddq_d_rad_s2)))
            <= batch.max_acceleration_rad_s2 + 1.0e-8
        )
        assert float(torch.max(torch.abs(batch.last_limiter_correction_rad))) < 1.0e-12
        assert float(torch.max(torch.abs(batch.max_limiter_correction_rad))) < 1.0e-12
        previous_acceleration = batch.ddq_d_rad_s2.clone()

    trajectory = torch.stack(positions)[:, 0]
    assert torch.all(trajectory[1:] >= trajectory[:-1] - 1.0e-12)
    assert torch.all(trajectory <= target + 1.0e-12)
    torch.testing.assert_close(
        trajectory[-1],
        target[0],
        rtol=0.0,
        atol=6.0e-6,
    )


def test_interpolated_numpy_reference_matches_batched_runtime_packet_by_packet():
    batch = torch_shaper.BatchedLibfrankaJointPositionInterpolator(
        1,
        "cpu",
        dtype=torch.float64,
        tracking_natural_frequency_hz=6.0,
        tracking_damping_ratio=1.0,
    )
    start = torch.from_numpy(Q_HOME[None, :])
    batch.reset(start)
    reference = interpolated_reference.InterpolatedJointPositionGenerator(Q_HOME)
    held = Q_HOME.astype(np.float32)
    rng = np.random.default_rng(225226)

    for packet in range(1000):
        if packet % 50 == 0:
            action = rng.uniform(-1.0, 1.0, size=7).astype(np.float32)
            held = interpolated_reference.map_policy_action_to_held_target(
                held,
                batch.q_d_rad[0].numpy().astype(np.float32),
                action,
            )
        expected = reference.step(held)
        batch.step(torch.from_numpy(held[None, :]).to(dtype=torch.float64))
        np.testing.assert_allclose(
            batch.q_d_rad[0].numpy(), expected.q_rad, rtol=0.0, atol=1.0e-10
        )
        np.testing.assert_allclose(
            batch.dq_d_rad_s[0].numpy(),
            expected.dq_rad_s,
            rtol=0.0,
            atol=1.0e-9,
        )
        np.testing.assert_allclose(
            batch.ddq_d_rad_s2[0].numpy(),
            expected.ddq_rad_s2,
            rtol=0.0,
            atol=1.0e-7,
        )
        assert float(torch.max(torch.abs(batch.last_limiter_correction_rad))) < 1.0e-12


def test_v225_adds_continuous_interpolation_before_the_libfranka_guard():
    cfg_source = CFG_PATH.read_text(encoding="utf-8")
    env_source = ENV_PATH.read_text(encoding="utf-8")
    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")

    assert (
        "class InspireUnifiedRollingJointDeltaV225InterpolatedLibfranka20HzTeacherEnvCfg"
        in cfg_source
    )
    assert (
        'franka_command_shaper_mode = "libfranka_joint_position_interpolated"'
        in cfg_source
    )
    assert "franka_command_interpolator_natural_frequency_hz = 6.0" in cfg_source
    assert "franka_command_interpolator_damping_ratio = 1.0" in cfg_source
    assert "BatchedLibfrankaJointPositionInterpolator" in env_source
    assert (
        "UnifiedRollingJointDeltaV225InterpolatedLibfranka20HzLSTM-Teacher-Direct-v0"
        in task_source
    )


def test_v213_is_an_independent_real_shaper_task_without_changing_v212():
    cfg_source = CFG_PATH.read_text(encoding="utf-8")
    env_source = ENV_PATH.read_text(encoding="utf-8")
    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")

    assert "class InspireUnifiedRollingJointDeltaV213RealFrankaShaper20HzTeacherEnvCfg" in cfg_source
    assert "franka_command_shaper_enabled = True" in cfg_source
    assert 'franka_command_shaper_runtime_dtype = "float64"' in cfg_source
    assert "franka_command_shaper_servo_ticks_per_physics_step = (8, 8, 9)" in cfg_source
    assert "franka_command_shaper_velocity_limit_rad_s = 0.5" in cfg_source
    assert "franka_command_shaper_acceleration_limit_rad_s2 = 4.0" in cfg_source
    assert "franka_command_shaper_jerk_limit_rad_s3 = 120.0" in cfg_source
    assert "self._franka_command_shaper.reset(" in env_source
    assert "self._apply_franka_command_shaper_action()" in env_source
    assert "UnifiedRollingJointDeltaV213RealFrankaShaper20HzLSTM-Teacher-Direct-v0" in task_source


def test_v215_curriculum_has_an_exact_shaper_endpoint_and_is_auditable():
    cfg_source = CFG_PATH.read_text(encoding="utf-8")
    env_source = ENV_PATH.read_text(encoding="utf-8")
    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")
    train_source = TRAIN_SCRIPT_PATH.read_text(encoding="utf-8")
    eval_source = EVAL_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "class InspireUnifiedRollingJointDeltaV215V94RewardFrankaShaperCurriculum20HzTeacherEnvCfg" in cfg_source
    assert "franka_command_shaper_execution_start_alpha = 0.0" in cfg_source
    assert "franka_command_shaper_execution_end_alpha = 1.0" in cfg_source
    assert "franka_command_shaper_execution_end_frames = 8_000_000" in cfg_source
    assert "executed_arm_target = torch.lerp(" in env_source
    assert 'self.extras["franka_shaper_execution_alpha"] = execution_alpha' in env_source
    assert "UnifiedRollingJointDeltaV215V94RewardFrankaShaperCurriculum20HzLSTM-Teacher-Direct-v0" in task_source
    assert '"franka_command_shaper": {' in train_source
    assert '"execution_curriculum": {' in train_source
    assert '"--franka-shaper-execution-alpha"' in train_source
    assert "env_cfg.franka_command_shaper_execution_start_alpha = alpha" in train_source
    assert "env_cfg.franka_command_shaper_execution_end_alpha = alpha" in train_source
    assert '"--franka-shaper-execution-alpha"' in eval_source
    assert "env_cfg.franka_command_shaper_execution_start_alpha = alpha" in eval_source
    assert "env_cfg.franka_command_shaper_execution_end_alpha = alpha" in eval_source


def test_train_and_eval_expose_a_staged_lift_threshold_without_changing_tasks():
    train_source = TRAIN_SCRIPT_PATH.read_text(encoding="utf-8")
    eval_source = EVAL_SCRIPT_PATH.read_text(encoding="utf-8")

    for source in (train_source, eval_source):
        assert '"--tabletop-success-lift-height"' in source
        assert "lift_height <= 0.0" in source
        assert "env_cfg.tabletop_success_lift_height = lift_height" in source


def test_v216_changes_only_the_hold_continuity_reward_contract():
    cfg_source = CFG_PATH.read_text(encoding="utf-8")
    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")

    assert (
        "class InspireUnifiedRollingJointDeltaV216FrankaShaperHoldContinuity20HzTeacherEnvCfg"
        in cfg_source
    )
    assert "canonical_hold_streak_loss_penalty_scale = 3000.0" in cfg_source
    assert (
        "UnifiedRollingJointDeltaV216FrankaShaperHoldContinuity20HzLSTM-Teacher-Direct-v0"
        in task_source
    )


def test_v217_replaces_avoidable_streak_loss_with_terminal_task_credit():
    cfg_source = CFG_PATH.read_text(encoding="utf-8")
    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")

    assert (
        "class InspireUnifiedRollingJointDeltaV217FrankaShaperTaskSuccess20HzTeacherEnvCfg"
        in cfg_source
    )
    assert "canonical_hold_streak_loss_penalty_scale = 0.0" in cfg_source
    assert "canonical_success_bonus = 3000.0" in cfg_source
    assert (
        "UnifiedRollingJointDeltaV217FrankaShaperTaskSuccess20HzLSTM-Teacher-Direct-v0"
        in task_source
    )


def test_v218_adds_only_the_calibrated_shaper_braking_reward():
    cfg_source = CFG_PATH.read_text(encoding="utf-8")
    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")

    assert (
        "class InspireUnifiedRollingJointDeltaV218FrankaShaperBraking20HzTeacherEnvCfg"
        in cfg_source
    )
    assert "canonical_hover_post_latch_speed_penalty_scale = 20.0" in cfg_source
    assert (
        "UnifiedRollingJointDeltaV218FrankaShaperBraking20HzLSTM-Teacher-Direct-v0"
        in task_source
    )


def test_v219_exposes_deployable_shaper_state_without_changing_actions():
    cfg_source = CFG_PATH.read_text(encoding="utf-8")
    env_source = ENV_PATH.read_text(encoding="utf-8")
    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")

    assert (
        "class InspireUnifiedRollingJointDeltaV219FrankaShaperState20HzTeacherEnvCfg"
        in cfg_source
    )
    assert "franka_command_shaper_state_obs_enabled = True" in cfg_source
    assert "observation_space = 153" in cfg_source
    assert "def _franka_command_shaper_state_observation" in env_source
    assert "shaper.dq_d_rad_s" in env_source
    assert "shaper.ddq_d_rad_s2" in env_source
    assert (
        "UnifiedRollingJointDeltaV219FrankaShaperState20HzLSTM-Teacher-Direct-v0"
        in task_source
    )


def test_v220_runs_exact_shaper_with_a_progressive_no_rake_contract():
    cfg_source = CFG_PATH.read_text(encoding="utf-8")
    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")

    assert (
        "class InspireUnifiedRollingJointDeltaV220FullShaperNoRakeControl20HzTeacherEnvCfg"
        in cfg_source
    )
    assert "franka_command_shaper_execution_curriculum_enabled = False" in cfg_source
    assert "tabletop_pregrasp_static_displacement_curriculum_enabled = True" in cfg_source
    assert "tabletop_success_max_pregrasp_static_object_xy_displacement = 0.020" in cfg_source
    assert "tabletop_pregrasp_static_displacement_terminate_threshold = 0.030" in cfg_source
    assert "dynamic_success_hold_steps = 20" in cfg_source
    assert "tabletop_success_streak_effective_arm_hold_enabled = False" in cfg_source
    assert (
        "UnifiedRollingJointDeltaV220FullShaperNoRakeControl20HzLSTM-Teacher-Direct-v0"
        in task_source
    )


def test_v221_uses_compact_physical_reward_and_same_execution_contract():
    cfg_source = CFG_PATH.read_text(encoding="utf-8")
    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")

    assert (
        "class InspireUnifiedRollingJointDeltaV221FullShaperMinimalTask20HzTeacherEnvCfg"
        in cfg_source
    )
    assert "canonical_force_closure_rew_scale = 8.0" in cfg_source
    assert "canonical_lift_progress_rew_scale = 45.0" in cfg_source
    assert "canonical_post_lift_height_drop_penalty_scale = 30.0" in cfg_source
    assert "canonical_post_lift_relative_speed_penalty_scale = 4.0" in cfg_source
    assert "canonical_success_bonus = 3000.0" in cfg_source
    assert "canonical_hold_streak_loss_penalty_scale = 0.0" in cfg_source
    assert (
        "UnifiedRollingJointDeltaV221FullShaperMinimalTask20HzLSTM-Teacher-Direct-v0"
        in task_source
    )


def test_v222_keeps_v213_acquisition_with_the_compact_reward_contract():
    cfg_source = CFG_PATH.read_text(encoding="utf-8")
    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")

    assert "class _ExactFullShaperNoRake20HzContract" in cfg_source
    assert "class _MinimalPhysicalGraspRewardContract" in cfg_source
    assert (
        "class InspireUnifiedRollingJointDeltaV222V213AcquisitionMinimalTask20HzTeacherEnvCfg"
        in cfg_source
    )
    assert "InspireUnifiedRollingJointDeltaV213RealFrankaShaper20HzTeacherEnvCfg" in cfg_source
    assert (
        "UnifiedRollingJointDeltaV222V213AcquisitionMinimalTask20HzLSTM-Teacher-Direct-v0"
        in task_source
    )


def test_v226_freezes_the_final_static_sphere_cube_asset_contract():
    cfg_source = CFG_PATH.read_text(encoding="utf-8")
    task_source = TASK_INIT_PATH.read_text(encoding="utf-8")

    assert "SIM2REAL_FOAM_STATIC_SPHERE_CUBE_NOMINAL_OBJECT_SPECS" in cfg_source
    assert "SIM2REAL_FOAM_SPHERE_60MM_SPEC" in cfg_source
    assert "SIM2REAL_FOAM_CUBE_50MM_SPEC" in cfg_source
    assert (
        "class InspireUnifiedStaticSphereCubeJointDeltaV226InterpolatedLibfranka20HzTeacherEnvCfg"
        in cfg_source
    )
    assert "tabletop_asset_curriculum_start_count = 2" in cfg_source
    assert "dynamic_tabletop_static_only_shape_codes = (0, 1)" in cfg_source
    assert "static_sim2real_validation_shape_codes = (0, 1)" in cfg_source
    assert (
        "UnifiedStaticSphereCubeJointDeltaV226InterpolatedLibfranka20HzLSTM-Teacher-Direct-v0"
        in task_source
    )
