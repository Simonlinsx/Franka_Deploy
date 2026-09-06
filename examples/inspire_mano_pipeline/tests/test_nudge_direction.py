from __future__ import annotations

import unittest

from inspire_rh56_test import nudge_target


class NudgeDirectionTests(unittest.TestCase):
    def test_signed_delta_selects_explicit_direction(self) -> None:
        self.assertEqual(nudge_target(887, -20), 867)
        self.assertEqual(nudge_target(887, 20), 907)

    def test_target_never_silently_reverses_at_an_endpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "opposite signed delta"):
            nudge_target(990, 20)
        with self.assertRaisesRegex(ValueError, "opposite signed delta"):
            nudge_target(10, -20)

    def test_thumb_rotate_commissioning_soft_range_rejects_907(self) -> None:
        self.assertEqual(
            nudge_target(887, -20, minimum_angle=800, maximum_angle=900),
            867,
        )
        with self.assertRaisesRegex(ValueError, "outside 800..900"):
            nudge_target(887, 20, minimum_angle=800, maximum_angle=900)

    def test_delta_and_original_angle_are_bounded(self) -> None:
        for delta in (-101, -19, 0, 19, 101):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                nudge_target(500, delta)
        with self.assertRaises(ValueError):
            nudge_target(-1, 20)
        with self.assertRaises(ValueError):
            nudge_target(1001, -20)


if __name__ == "__main__":
    unittest.main()
