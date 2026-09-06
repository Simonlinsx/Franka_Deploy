from __future__ import annotations

import unittest

import numpy as np

from sim2real.commissioning.identify_fr3_rh56_static_load import (
    PoseAggregate,
    StaticLoadIdentificationError,
    _validate_static_state,
    gravity_parameter_regressor,
    identify,
)


class _FakeState:
    def __init__(self, mass: float, center: np.ndarray) -> None:
        self.m_total = float(mass)
        self.F_x_Ctotal = center.tolist()


class _FakeStaticRobotState:
    def __init__(self, mode: str) -> None:
        self.robot_mode = mode
        self.current_errors = []
        self.joint_contact = np.zeros(7)
        self.joint_collision = np.zeros(7)
        self.cartesian_contact = np.zeros(6)
        self.cartesian_collision = np.zeros(6)
        self.dq = np.zeros(7)


class _FakeGravityModel:
    def __init__(self, robot_only: np.ndarray, regressor: np.ndarray) -> None:
        self._robot_only = robot_only
        self._regressor = regressor

    def gravity(self, state: _FakeState) -> np.ndarray:
        mass = float(state.m_total)
        center = np.asarray(state.F_x_Ctotal, dtype=np.float64)
        theta = np.r_[mass, mass * center]
        return self._robot_only + self._regressor @ theta


class StaticLoadIdentificationTest(unittest.TestCase):
    def test_idle_and_user_stopped_are_valid_static_modes(self) -> None:
        _validate_static_state(_FakeStaticRobotState("RobotMode.Idle"), 0.008)
        _validate_static_state(
            _FakeStaticRobotState("RobotMode.UserStopped"), 0.008
        )
        with self.assertRaises(StaticLoadIdentificationError):
            _validate_static_state(_FakeStaticRobotState("RobotMode.Guiding"), 0.008)

    def test_gravity_regressor_is_exact_and_restores_state(self) -> None:
        rng = np.random.default_rng(7)
        expected = rng.normal(size=(7, 4))
        state = _FakeState(0.607, np.array([0.0, 0.0, 0.076]))
        model = _FakeGravityModel(rng.normal(size=7), expected)

        actual = gravity_parameter_regressor(model, state)

        np.testing.assert_allclose(actual, expected, atol=1.0e-12, rtol=0.0)
        self.assertEqual(state.m_total, 0.607)
        np.testing.assert_allclose(state.F_x_Ctotal, [0.0, 0.0, 0.076])

    def test_recovers_mass_and_center_with_bias_and_noise(self) -> None:
        rng = np.random.default_rng(19)
        current_mass = 0.607
        current_center = np.array([0.0, 0.0, 0.076])
        actual_mass = 0.665
        actual_center = np.array([-0.004, 0.006, 0.087])
        current_theta = np.r_[current_mass, current_mass * current_center]
        actual_theta = np.r_[actual_mass, actual_mass * actual_center]
        delta_theta = actual_theta - current_theta
        joint_bias = np.array([0.03, -0.02, 0.01, 0.04, -0.03, 0.02, -0.01])

        poses = []
        for index in range(10):
            regressor = rng.normal(scale=[2.0, 8.0, 8.0, 8.0], size=(7, 4))
            tau = regressor @ delta_theta + joint_bias
            tau += rng.normal(scale=0.003, size=7)
            if index == 3:
                tau[4] += 0.25
            poses.append(
                PoseAggregate(
                    index=index + 1,
                    q_rad=np.zeros(7),
                    dq_abs_max_rad_s=0.0,
                    tau_ext_median_nm=tau,
                    tau_ext_mad_nm=np.full(7, 0.002),
                    O_F_ext_median_n_nm=np.zeros(6),
                    gravity_regressor=regressor,
                    sample_count=80,
                    robot_mode="RobotMode.Idle",
                )
            )

        result = identify(
            poses,
            {
                "m_total_kg": current_mass,
                "F_x_Ctotal_m": current_center,
            },
        )

        identified = result["identified_total_load"]
        self.assertAlmostEqual(identified["mass_kg"], actual_mass, delta=0.003)
        np.testing.assert_allclose(
            identified["F_x_Ctotal_m"], actual_center, atol=0.002, rtol=0.0
        )
        self.assertLess(result["fit"]["residual_rmse_nm"], 0.04)
        self.assertTrue(result["accepted_for_desk_candidate"])


if __name__ == "__main__":
    unittest.main()
