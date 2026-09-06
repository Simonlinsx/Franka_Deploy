from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path

from inspire_mano_pipeline.calibration import PipelineCalibration
from inspire_mano_pipeline.model import HARDWARE_JOINTS
from inspire_mano_pipeline.retargeting import GeometricInspireRetargeter
from realsense_mano_inspire import (
    build_parser,
    is_safe_commissioning_profile,
    validate_args,
)


ROOT = Path(__file__).resolve().parents[3]
PROFILE_DIR = ROOT / "examples/inspire_mano_pipeline"
PROFILE_PATHS = {
    "commissioning_120": PROFILE_DIR
    / "inspire_rh56bfx_right_five_finger_commissioning_open1000.json",
    "fullrange_200": PROFILE_DIR
    / "inspire_rh56bfx_right_five_finger_fullrange_realtime200.json",
    "fullrange_300": PROFILE_DIR
    / "inspire_rh56bfx_right_five_finger_fullrange_realtime.json",
}
FIVE_BEND_AXES = HARDWARE_JOINTS[:5]
AXIS_Q_ENDPOINTS = {
    "pinky": (0.05, 0.55),
    "ring": (0.05, 0.58),
    "middle": (0.05, 0.47),
    "index": (0.08, 0.40),
    "thumb_bend": (0.0, 0.60),
    "thumb_rotate": (0.0, 1.308),
}


class FiveFingerProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profiles = {
            name: PipelineCalibration.load(path)
            for name, path in PROFILE_PATHS.items()
        }

    def test_exactly_five_bend_axes_are_enabled(self) -> None:
        for name, calibration in self.profiles.items():
            with self.subTest(profile=name):
                enabled = tuple(
                    axis
                    for axis in HARDWARE_JOINTS
                    if calibration.axes[axis].enabled
                )
                self.assertEqual(enabled, FIVE_BEND_AXES)
                self.assertFalse(calibration.axes["thumb_rotate"].enabled)

    def test_q_and_command_endpoints_are_exact(self) -> None:
        expected_closed = {
            "commissioning_120": (800, 800, 800, 800, 850, -1),
            "fullrange_200": (0, 0, 0, 0, 0, -1),
            "fullrange_300": (0, 0, 0, 0, 0, -1),
        }
        for name, calibration in self.profiles.items():
            with self.subTest(profile=name):
                for axis_name, endpoints in AXIS_Q_ENDPOINTS.items():
                    axis = calibration.axes[axis_name]
                    self.assertEqual((axis.q_open, axis.q_closed), endpoints)
                    self.assertEqual(
                        axis.command_open,
                        700 if axis_name == "thumb_rotate" else 1000,
                    )
                    if axis_name == "thumb_rotate":
                        self.assertEqual(axis.command_closed, 300)

                q_open = tuple(
                    calibration.axes[axis].q_open for axis in HARDWARE_JOINTS
                )
                q_closed = tuple(
                    calibration.axes[axis].q_closed for axis in HARDWARE_JOINTS
                )
                self.assertEqual(
                    calibration.map_qpos(q_open),
                    (1000, 1000, 1000, 1000, 1000, -1),
                )
                self.assertEqual(
                    calibration.map_qpos(q_closed), expected_closed[name]
                )

    def test_temporal_filter_covers_five_bend_axes_only(self) -> None:
        for name, calibration in self.profiles.items():
            with self.subTest(profile=name):
                self.assertEqual(calibration.temporal_filter_axes, FIVE_BEND_AXES)
                self.assertEqual(calibration.temporal_median_window, 3)
                self.assertEqual(calibration.temporal_ema_alpha, 0.65)
                retargeter = GeometricInspireRetargeter(calibration)
                self.assertEqual(retargeter._filter.active_indices, (0, 1, 2, 3, 4))

    def test_narrow_then_200_then_300_motion_tiers_are_exact(self) -> None:
        expected = {
            "commissioning_120": {
                "speed": 120,
                "rates": (120.0, 120.0, 120.0, 120.0, 80.0),
                "spans": (200, 200, 200, 200, 150),
            },
            "fullrange_200": {
                "speed": 200,
                "rates": (200.0, 200.0, 200.0, 200.0, 160.0),
                "spans": (1000, 1000, 1000, 1000, 1000),
            },
            "fullrange_300": {
                "speed": 300,
                "rates": (300.0, 300.0, 300.0, 300.0, 200.0),
                "spans": (1000, 1000, 1000, 1000, 1000),
            },
        }
        for name, calibration in self.profiles.items():
            with self.subTest(profile=name):
                case = expected[name]
                rates = tuple(
                    calibration.axes[axis].max_rate_units_per_second
                    for axis in FIVE_BEND_AXES
                )
                spans = tuple(
                    calibration.axes[axis].command_open
                    - calibration.axes[axis].command_closed
                    for axis in FIVE_BEND_AXES
                )
                self.assertEqual(calibration.speed, case["speed"])
                self.assertEqual(rates, case["rates"])
                self.assertEqual(spans, case["spans"])
                self.assertEqual(calibration.force_limit, 80)

    def test_total_current_limits_are_profile_bounded(self) -> None:
        expected_limits = {
            "commissioning_120": (500, 600),
            "fullrange_200": (600, 600),
            "fullrange_300": (600, 600),
        }
        for name, calibration in self.profiles.items():
            with self.subTest(profile=name):
                self.assertEqual(calibration.active_current_policy, "fault")
                per_axis, total = expected_limits[name]
                self.assertEqual(calibration.stream_max_current_ma, per_axis)
                self.assertEqual(
                    calibration.stream_max_total_current_ma,
                    total,
                )
                self.assertLessEqual(
                    calibration.stream_max_total_current_ma,
                    5 * calibration.stream_max_current_ma,
                )

    @staticmethod
    def _hardware_args(profile: Path, confirmation: str | None):
        arguments = [
            "--source",
            "realsense",
            "--camera-serial",
            "fake-camera",
            "--device",
            "cuda:0",
            "--calibration",
            str(profile),
            "--retargeter",
            "geometric",
            "--enable-hardware",
            "--confirm-hardware-motion",
            "--port",
            "/dev/fake-hand",
            "--axes",
            ",".join(FIVE_BEND_AXES),
            "--duration",
            "10",
        ]
        if confirmation is not None:
            arguments.extend(("--confirm-wide-range", confirmation))
        parser = build_parser()
        return parser, parser.parse_args(arguments)

    def test_all_five_finger_profiles_require_exact_wide_range_token(self) -> None:
        for name, path in PROFILE_PATHS.items():
            calibration = self.profiles[name]
            self.assertFalse(
                is_safe_commissioning_profile(calibration, FIVE_BEND_AXES)
            )
            for confirmation in (None, "wrong"):
                with self.subTest(profile=name, confirmation=confirmation):
                    parser, args = self._hardware_args(path, confirmation)
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
                        SystemExit
                    ):
                        validate_args(args, parser)

            with self.subTest(profile=name, confirmation="exact"):
                parser, args = self._hardware_args(path, "RH56_WIDE_RANGE")
                validate_args(args, parser)
                self.assertEqual(args.axes, FIVE_BEND_AXES)
                self.assertEqual(args.no_hand_policy, "disable")


if __name__ == "__main__":
    unittest.main()
