from __future__ import annotations

import numpy as np
import pytest

from sim2real.baseline import ActionLimiter, BaselineAction, BaselinePolicy

LIMITS = np.asarray(
    [
        [-2.9, 2.9],
        [-1.8, 1.8],
        [-2.9, 2.9],
        [-3.0, -0.1],
        [-2.8, 2.8],
        [0.4, 4.6],
        [-3.0, 3.0],
    ]
)
INITIAL_Q = np.asarray([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0])


def _limiter():
    return ActionLimiter(
        initial_q_rad=INITIAL_Q,
        joint_limits_rad=LIMITS,
        joint_limit_margin_rad=0.05,
        max_arm_step_rad=0.005,
        max_arm_episode_delta_rad=0.02,
        initial_hand_angles=[1000] * 6,
        max_hand_step_units=10,
    )


def test_hold_policy_is_no_command():
    action = BaselinePolicy("hold", policy_hz=10).act(None, elapsed_s=1.0)
    np.testing.assert_array_equal(action.arm_joint_delta_rad, np.zeros(7))
    np.testing.assert_array_equal(action.hand_targets, [-1] * 6)


def test_combined_preview_has_fixed_rh56_axis_order_and_q6_open():
    policy = BaselinePolicy("preview-combined", policy_hz=10)
    action = policy.act(None, elapsed_s=1.0)
    assert action.hand_targets.shape == (6,)
    assert np.all(action.hand_targets[:5] == 900)
    assert action.hand_targets[5] == 1000


def test_action_limiter_clips_arm_step_episode_and_hand_rate():
    limiter = _limiter()
    action = BaselineAction(
        arm_joint_delta_rad=np.asarray([0.1] * 7),
        hand_targets=np.asarray([0, 900, -1, -1, -1, 1000]),
        source="test",
    )
    limited = limiter.apply(action)
    assert limited.clipped
    np.testing.assert_allclose(limited.arm_joint_delta_rad, [0.005] * 7)
    np.testing.assert_array_equal(limited.hand_targets, [990, 990, -1, -1, -1, 1000])

    for _ in range(10):
        limited = limiter.apply(action)
    np.testing.assert_allclose(limited.arm_target_q_rad, INITIAL_Q + 0.02, atol=1.0e-12)
    assert any("episode" in reason for reason in limited.reasons)


@pytest.mark.parametrize(
    "arm,hand,match",
    [
        ([0.0] * 6, [-1] * 6, "seven finite"),
        ([0.0] * 6 + [float("nan")], [-1] * 6, "seven finite"),
        ([0.0] * 7, [-2] * 6, "-1 or lie"),
    ],
)
def test_action_limiter_rejects_malformed_actions(arm, hand, match):
    with pytest.raises(ValueError, match=match):
        _limiter().apply(
            BaselineAction(np.asarray(arm), np.asarray(hand), source="bad")
        )
