from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sim2real.perception import _fail_closed_visualization_sample
from sim2real.tasks.launcher import build_task_command
from sim2real.tasks.thrown_shadow import (
    _ShadowPolicyRollout,
    _read_only_controller_state29,
)
from sim2real.observation.model import PolicyPointFrame


class _FakePolicy:
    history_length = 8
    point_feature_dim = 3
    proprio_dim = 96
    point_mean = np.zeros((1, 1, 1, 3), dtype=np.float32)
    point_std = np.ones((1, 1, 1, 3), dtype=np.float32)
    proprio_mean = np.zeros((1, 1, 96), dtype=np.float32)
    proprio_std = np.ones((1, 1, 96), dtype=np.float32)

    def __init__(self) -> None:
        self.calls = 0

    def act(self, points, valid, proprio):
        self.calls += 1
        assert points.shape == (8, 128, 3)
        assert valid.shape == (8, 128)
        assert proprio.shape == (8, 96)
        return SimpleNamespace(
            action13=np.linspace(-1.0, 1.0, 13, dtype=np.float32),
            predicted_hold_logit=0.25,
        )


class _Clock:
    def __init__(self) -> None:
        self.value = 10.0

    def __call__(self) -> float:
        return self.value


def _point_frame(frame_id: int) -> PolicyPointFrame:
    return PolicyPointFrame(
        xyzrgb_palm=np.zeros((128, 3), dtype=np.float32),
        valid=np.ones(128, dtype=np.float32),
        captured_at_s=1000.0 + frame_id / 60.0,
        frame_id=frame_id,
        source_valid_points=128,
        status="fresh",
    )


def test_shadow_rollout_starts_only_after_trigger_and_runs_at_policy_rate():
    clock = _Clock()
    policy = _FakePolicy()
    rollout = _ShadowPolicyRollout(
        policy,
        control_dt_s=0.05,
        monotonic=clock,
        perf_counter=clock,
    )
    kwargs = {
        "proprio67": np.zeros(67, dtype=np.float32),
        "controller_state29": np.r_[
            np.zeros(28, dtype=np.float32), np.ones(1, dtype=np.float32)
        ],
        "camera_timestamp_s": 1000.0,
    }
    assert rollout.maybe_infer(
        point_frame=_point_frame(1),
        camera_frame_id=1,
        trigger_result=None,
        trigger_detected_monotonic_s=None,
        **kwargs,
    ) is None
    assert policy.calls == 0

    trigger = {"detected": True, "frame_id": 2}
    first = rollout.maybe_infer(
        point_frame=_point_frame(2),
        camera_frame_id=2,
        trigger_result=trigger,
        trigger_detected_monotonic_s=10.0,
        **kwargs,
    )
    assert first is not None
    assert first["logical_index"] == 0
    assert first["robot_hardware_writes"] is False
    assert policy.calls == 1

    clock.value = 10.02
    assert rollout.maybe_infer(
        point_frame=_point_frame(3),
        camera_frame_id=3,
        trigger_result=trigger,
        trigger_detected_monotonic_s=10.0,
        **kwargs,
    ) is None
    assert policy.calls == 1

    clock.value = 10.05
    second = rollout.maybe_infer(
        point_frame=_point_frame(4),
        camera_frame_id=4,
        trigger_result=trigger,
        trigger_detected_monotonic_s=10.0,
        **kwargs,
    )
    assert second is not None
    assert second["logical_index"] == 1
    assert policy.calls == 2


def test_read_only_controller_state_is_finite_bounded_and_has_no_command_error():
    franka = SimpleNamespace(
        q=np.zeros(7, dtype=np.float32),
        q_desired=np.asarray([0.01] * 7, dtype=np.float32),
        dq_desired=np.asarray([0.05] * 7, dtype=np.float32),
    )
    state = _read_only_controller_state29(franka)
    assert state.shape == (29,)
    assert np.all(np.isfinite(state))
    np.testing.assert_array_equal(state[:7], np.zeros(7, dtype=np.float32))
    np.testing.assert_allclose(state[7:14], 0.2)
    np.testing.assert_allclose(state[14:21], 0.1)
    np.testing.assert_array_equal(state[21:28], np.zeros(7, dtype=np.float32))
    assert state[28] == 1.0


def test_task_launcher_shadow_is_explicit_checkpoint_and_read_only_entrypoint():
    checkpoint = "/tmp/v57.pt"
    profile = "/tmp/read-only-profile.json"
    command, _config, _metadata = build_task_command(
        "thrown_object",
        "shadow",
        [
            "--checkpoint",
            checkpoint,
            "--profile",
            profile,
            "--test-rollout-trigger",
            "--object-text",
            "small red ball",
        ],
    )
    assert "sim2real.tasks.thrown_shadow" in command
    assert "--execute" not in command
    assert checkpoint in command
    assert profile in command


def test_v60_task_launcher_reuses_the_same_read_only_shadow_core():
    checkpoint = "/tmp/v60.pt"
    profile = "/tmp/v60-shadow.json"
    command, _config, metadata = build_task_command(
        "thrown_object_v60",
        "shadow",
        [
            "--checkpoint",
            checkpoint,
            "--profile",
            profile,
            "--test-rollout-trigger",
        ],
    )
    assert metadata["commissioning_status"] == "accepted"
    assert metadata["robot_execution_enabled"] is True
    assert metadata["maximum_supervised_execute_steps"] == 72
    assert "sim2real.tasks.thrown_shadow" in command
    assert "--execute" not in command
    assert checkpoint in command
    assert profile in command


def test_shadow_module_has_no_action_mapper_or_actuator_import():
    from pathlib import Path
    import sim2real.tasks.thrown_shadow as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "V94ActionMapper" not in source
    assert "TransactionalV94ActionMapper" not in source
    assert "RH56HardwareCommand" not in source
    assert "--execute" not in source


def test_shadow_fail_closed_visualization_uses_supported_policy_palm_frame():
    camera = SimpleNamespace(
        frame_id=8,
        timestamp_s=100.0,
        color_bgr=np.zeros((3, 4, 3), dtype=np.uint8),
        mask=np.zeros((3, 4), dtype=bool),
        requested_object_mask_mode="guarded_v2",
        effective_object_mask_mode="guarded_v2",
        effective_mask_publication_mode="guarded_sam2_primary",
        effective_recovery_publication_mode="unified_three_evidence",
        provider_published_mask_source="none",
        provider_published_mask_message="empty",
        provider_online_sam2_status="complete occlusion",
    )
    palm = np.eye(4, dtype=np.float64)
    palm[0, 3] = 0.2
    sample = _fail_closed_visualization_sample(
        sequence=1,
        camera=camera,
        point_feature_dim=3,
        T_base_palm_at_capture=palm,
        point_coordinate_frame="policy_palm",
    )
    assert sample.point_coordinate_frame == "policy_palm"
    np.testing.assert_array_equal(sample.T_base_palm_at_capture, palm)
