import unittest

import numpy as np

from sim2real.commissioning.calibrate_rh56_force_baseline import (
    PoseAggregate,
    analyze_poses,
)


class ForceBaselineAnalysisTest(unittest.TestCase):
    def test_recovers_gravity_linear_baseline_and_return_delta(self) -> None:
        gravity = np.array(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )
        gravity /= np.linalg.norm(gravity, axis=1, keepdims=True)
        intercept = np.array([10, -20, 5, 0, 15, -5], dtype=np.float64)
        coefficients = np.array(
            [
                [10, 5, 3, 2, 1, 4],
                [8, 4, 2, 1, 3, 5],
                [6, 3, 1, 4, 2, 7],
            ],
            dtype=np.float64,
        )
        force = intercept[None, :] + gravity @ coefficients

        poses = []
        for index in range(len(gravity)):
            poses.append(
                PoseAggregate(
                    index=index + 1,
                    label=f"pose_{index + 1}",
                    robot_mode="userstopped",
                    force_median_gf=force[index],
                    force_mean_gf=force[index],
                    force_std_gf=np.zeros(6),
                    force_mad_gf=np.zeros(6),
                    force_min_gf=force[index],
                    force_max_gf=force[index],
                    angles_median=np.full(6, 1000.0),
                    positions_median=np.zeros(6),
                    temperatures_median_c=np.full(6, 30.0),
                    q_median_rad=np.zeros(7),
                    gravity_ee_median=gravity[index],
                    maximum_abs_dq_rad_s=0.0,
                    within_pose_orientation_span_deg=0.0,
                    sample_count=40,
                )
            )

        result = analyze_poses(poses)
        self.assertEqual(result["gravity_design_rank"], 4)
        self.assertEqual(result["overall_classification"], "low_pose_dependence")
        self.assertTrue(result["quality"]["valid_for_gravity_compensation"])
        self.assertEqual(result["quality"]["reasons"], [])
        self.assertAlmostEqual(
            result["return_reference"]["orientation_error_deg"], 0.0
        )
        np.testing.assert_allclose(
            result["return_reference"]["force_delta_gf_hardware_order"],
            np.zeros(6),
            atol=1.0e-10,
        )
        for axis in range(6):
            model = result["axes"][axis]["gravity_linear_model"]
            self.assertAlmostEqual(model["intercept_gf"], intercept[axis])
            np.testing.assert_allclose(
                model["gravity_ee_coefficients_gf"],
                coefficients[:, axis],
                atol=1.0e-10,
            )
            self.assertAlmostEqual(model["residual_rmse_gf"], 0.0, places=10)
            self.assertAlmostEqual(model["r_squared"], 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
