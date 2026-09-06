from pathlib import Path

import numpy as np

from sim2real.deployment.bundle import DeployBundle
from sim2real.contracts.v94 import V94Contract
from sim2real.observation.kinematics import (
    KinematicVelocityTracker,
    RH56FeedbackMapper,
    RH56FingertipKinematics,
    T_base_policy_palm_from_franka,
)
from sim2real.observation.model import pose_from_position_quaternion_wxyz

BUNDLE = Path(__file__).resolve().parents[2] / "data/test_fixtures/sim2real/deploy.zip"


def test_rh56_register_feedback_maps_by_names_not_implicit_units():
    mapper = RH56FeedbackMapper()
    feedback = mapper.map([1000, 800, 600, 400, 200, 0])
    np.testing.assert_allclose(
        feedback.close_fraction_policy_order, [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    )


def test_franka_configured_ee_is_removed_before_flange_to_palm_transform():
    base_ee = np.eye(4)
    flange_ee = np.eye(4)
    flange_ee[0, 3] = 0.1
    base_ee[0, 3] = 1.1
    flange_palm = np.eye(4)
    flange_palm[2, 3] = 0.2
    result = T_base_policy_palm_from_franka(
        T_base_ee=base_ee,
        F_T_EE=flange_ee,
        T_flange_policy_palm=flange_palm,
    )
    np.testing.assert_allclose(result[:3, 3], [1.0, 0.0, 0.2])


def test_velocity_tracker_reports_base_translation_and_rotation():
    tracker = KinematicVelocityTracker()
    first = np.eye(4)
    tracker.update(
        captured_at_s=1.0,
        T_base_palm=first,
        hand_q_policy_order_rad=np.zeros(6),
    )
    second = pose_from_position_quaternion_wxyz(
        [0.1, 0.0, 0.0], [np.cos(0.05), 0.0, 0.0, np.sin(0.05)]
    )
    velocity = tracker.update(
        captured_at_s=1.1,
        T_base_palm=second,
        hand_q_policy_order_rad=np.full(6, 0.01),
    )
    np.testing.assert_allclose(velocity.linear_base_m_s, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(velocity.angular_base_rad_s, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(velocity.hand_policy_rad_s, 0.1)


def test_official_hand_fk_reproduces_packaged_open_reference():
    contract = V94Contract.from_bundle(DeployBundle(BUNDLE))
    model = RH56FingertipKinematics(contract)
    palm = pose_from_position_quaternion_wxyz(
        contract.reference_palm_position_base_m,
        contract.reference_palm_quaternion_base_wxyz,
    )
    actual = model.positions_base(
        angle_act_register_order=np.full(6, 1000), T_base_palm=palm
    )
    np.testing.assert_allclose(
        actual, contract.reference_fingertip_positions_base_m, atol=1.0e-9
    )
