from __future__ import annotations

import numpy as np
import unittest

from sim2real.commissioning.identify_rh56_dynamics import (
    _simulate_normalized_step,
    fit_step_response,
)


class RH56DynamicsFitTest(unittest.TestCase):
    def test_recovers_synthetic_velocity_limited_second_order_response(self) -> None:
        times = np.arange(0.0, 1.5, 0.02)
        measured = _simulate_normalized_step(
            times,
            omega_n_rad_s=18.0,
            damping_ratio=0.9,
            velocity_limit_steps_s=5.0,
            delay_s=0.025,
        )
        samples = [
            {
                "t_s": float(t),
                "start_angle": 800,
                "target_angle": 650,
                "angle_actual": float(800.0 - 150.0 * q),
            }
            for t, q in zip(times, measured)
        ]

        fit = fit_step_response(samples)

        self.assertGreater(fit["r_squared"], 0.999)
        self.assertAlmostEqual(fit["omega_n_rad_s"], 18.0, delta=0.5)
        self.assertAlmostEqual(fit["damping_ratio"], 0.9, delta=0.06)
        self.assertAlmostEqual(fit["command_to_actual_dc_gain"], 1.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
