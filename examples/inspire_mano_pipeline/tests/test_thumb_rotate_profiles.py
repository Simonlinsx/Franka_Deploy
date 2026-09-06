from __future__ import annotations

import contextlib
import io
import json
import unittest
from dataclasses import replace
from pathlib import Path

from inspire_mano_pipeline.calibration import AxisCalibration, PipelineCalibration
from inspire_mano_pipeline.model import HARDWARE_JOINTS
from inspire_mano_pipeline.retargeting import GeometricInspireRetargeter
from realsense_mano_inspire import (
    build_parser,
    is_safe_thumb_rotate_commissioning_profile,
    validate_args,
)


ROOT = Path(__file__).resolve().parents[3]
PROFILE_DIR = ROOT / "examples/inspire_mano_pipeline"
ROTATE_ONLY_PATH = (
    PROFILE_DIR
    / "inspire_rh56bfx_right_thumb_rotate_commissioning_900_800.json"
)
SIX_DOF_PATH = (
    PROFILE_DIR
    / "inspire_rh56bfx_right_six_dof_commissioning_900_800.json"
)
SIX_DOF_OPEN1000_PATH = (
    PROFILE_DIR
    / "inspire_rh56bfx_right_six_dof_open1000_realtime.json"
)

EXPECTED_COMMAND_RANGES = {
    "pinky": (1000, 800),
    "ring": (1000, 800),
    "middle": (1000, 800),
    "index": (1000, 800),
    "thumb_bend": (1000, 850),
    "thumb_rotate": (900, 800),
}


class ThumbRotateProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rotate_only = PipelineCalibration.load(ROTATE_ONLY_PATH)
        cls.six_dof = PipelineCalibration.load(SIX_DOF_PATH)

    @staticmethod
    def _enabled_axes(calibration: PipelineCalibration) -> tuple[str, ...]:
        return tuple(
            name for name in HARDWARE_JOINTS if calibration.axes[name].enabled
        )

    def test_profiles_enable_exact_intended_axes(self) -> None:
        self.assertEqual(self._enabled_axes(self.rotate_only), ("thumb_rotate",))
        self.assertEqual(self._enabled_axes(self.six_dof), HARDWARE_JOINTS)

    def test_both_profiles_use_exact_narrow_command_ranges(self) -> None:
        for profile_name, calibration in (
            ("rotate_only", self.rotate_only),
            ("six_dof", self.six_dof),
        ):
            with self.subTest(profile=profile_name):
                actual = {
                    name: (
                        calibration.axes[name].command_open,
                        calibration.axes[name].command_closed,
                    )
                    for name in HARDWARE_JOINTS
                }
                self.assertEqual(actual, EXPECTED_COMMAND_RANGES)
                rotate = calibration.axes["thumb_rotate"]
                self.assertEqual((rotate.q_open, rotate.q_closed), (0.0, 1.308))
                self.assertEqual(
                    rotate.command_open - rotate.command_closed,
                    100,
                )
                self.assertEqual(rotate.feedback_to_command_offset_units, 15)
                self.assertEqual(
                    (
                        rotate.feedback_to_command_valid_min,
                        rotate.feedback_to_command_valid_max,
                    ),
                    (840, 870),
                )

    def test_nonzero_feedback_offsets_exist_only_on_three_axis6_profiles(self) -> None:
        observed = set()
        for path in PROFILE_DIR.glob("inspire_rh56bfx_right*.json"):
            calibration = PipelineCalibration.load(path)
            for name, axis in calibration.axes.items():
                if axis.feedback_to_command_offset_units:
                    observed.add((path.name, name, axis.feedback_to_command_offset_units))
                elif path not in (ROTATE_ONLY_PATH, SIX_DOF_PATH) or name != "thumb_rotate":
                    self.assertIsNone(axis.feedback_to_command_valid_min)
                    self.assertIsNone(axis.feedback_to_command_valid_max)
        self.assertEqual(
            observed,
            {
                (ROTATE_ONLY_PATH.name, "thumb_rotate", 15),
                (SIX_DOF_PATH.name, "thumb_rotate", 15),
                (SIX_DOF_OPEN1000_PATH.name, "thumb_rotate", 15),
            },
        )

    def test_feedback_offset_loader_is_strict_and_bounded(self) -> None:
        raw = json.loads(ROTATE_ONLY_PATH.read_text(encoding="utf-8"))
        base = raw["axes"]["thumb_rotate"]
        for invalid in (True, 15.0, "15", 101, -101):
            changed = dict(base)
            changed["feedback_to_command_offset_units"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                AxisCalibration.from_mapping(changed)

        missing_bound = dict(base)
        missing_bound.pop("feedback_to_command_valid_max")
        with self.assertRaisesRegex(ValueError, "provided together"):
            AxisCalibration.from_mapping(missing_bound)

    def test_rotate_only_motion_and_safety_values_are_exact(self) -> None:
        calibration = self.rotate_only
        self.assertEqual(calibration.speed, 60)
        self.assertEqual(calibration.force_limit, 80)
        self.assertEqual(
            calibration.axes["thumb_rotate"].max_rate_units_per_second,
            40.0,
        )
        self.assertEqual(calibration.arming_max_target_delta_units, 15)
        self.assertEqual(calibration.preflight_max_idle_current_ma, 200)
        self.assertEqual(calibration.active_current_policy, "fault")
        self.assertEqual(calibration.stream_max_current_ma, 400)
        self.assertEqual(calibration.stream_max_total_current_ma, 400)
        self.assertEqual(calibration.temporal_filter_axes, ("thumb_rotate",))
        self.assertEqual(calibration.temporal_median_window, 3)
        self.assertEqual(calibration.temporal_ema_alpha, 0.65)
        self.assertEqual(
            GeometricInspireRetargeter(calibration)._filter.active_indices,
            (5,),
        )

    def test_six_dof_motion_and_safety_values_are_exact(self) -> None:
        calibration = self.six_dof
        self.assertEqual(calibration.speed, 80)
        self.assertEqual(calibration.force_limit, 80)
        self.assertEqual(
            tuple(
                calibration.axes[name].max_rate_units_per_second
                for name in HARDWARE_JOINTS
            ),
            (80.0, 80.0, 80.0, 80.0, 60.0, 40.0),
        )
        self.assertEqual(calibration.arming_max_target_delta_units, 25)
        self.assertEqual(calibration.preflight_max_idle_current_ma, 300)
        self.assertEqual(calibration.active_current_policy, "fault")
        self.assertEqual(calibration.stream_max_current_ma, 500)
        self.assertEqual(calibration.stream_max_total_current_ma, 600)
        self.assertEqual(calibration.temporal_filter_axes, HARDWARE_JOINTS)
        self.assertEqual(calibration.temporal_median_window, 3)
        self.assertEqual(calibration.temporal_ema_alpha, 0.65)
        self.assertEqual(
            GeometricInspireRetargeter(calibration)._filter.active_indices,
            (0, 1, 2, 3, 4, 5),
        )

    def test_common_tracking_safety_values_remain_bounded(self) -> None:
        for profile_name, calibration in (
            ("rotate_only", self.rotate_only),
            ("six_dof", self.six_dof),
        ):
            with self.subTest(profile=profile_name):
                self.assertEqual(calibration.detector_confidence, 0.5)
                self.assertEqual(calibration.tracking_timeout_seconds, 0.75)
                self.assertEqual(calibration.valid_frames_to_arm, 5)
                self.assertEqual(calibration.arming_max_frame_gap_seconds, 0.25)
                self.assertEqual(calibration.control_hz, 20.0)
                self.assertEqual(calibration.feedback_hz, 5.0)
                self.assertTrue(calibration.require_depth_for_hardware)
                self.assertEqual(calibration.min_palm_depth_m, 0.15)
                self.assertEqual(calibration.max_palm_depth_m, 1.50)
                self.assertEqual(calibration.preflight_stability_seconds, 0.15)
                self.assertEqual(calibration.preflight_max_angle_delta, 5)

    def test_q6_maps_downward_and_only_changes_sixth_target(self) -> None:
        for profile_name, calibration in (
            ("rotate_only", self.rotate_only),
            ("six_dof", self.six_dof),
        ):
            with self.subTest(profile=profile_name):
                qpos = [
                    calibration.axes[name].q_open for name in HARDWARE_JOINTS
                ]
                open_targets = calibration.map_qpos(qpos)
                qpos[5] = 0.654
                middle_targets = calibration.map_qpos(qpos)
                qpos[5] = 1.308
                closed_targets = calibration.map_qpos(qpos)

                self.assertEqual(open_targets[:5], middle_targets[:5])
                self.assertEqual(middle_targets[:5], closed_targets[:5])
                self.assertEqual(
                    (open_targets[5], middle_targets[5], closed_targets[5]),
                    (900, 850, 800),
                )
                self.assertGreater(open_targets[5], middle_targets[5])
                self.assertGreater(middle_targets[5], closed_targets[5])

    def test_disabled_bend_qpos_cannot_leak_into_rotate_only_targets(self) -> None:
        rotate_q = 0.654
        bend_open = [
            self.rotate_only.axes[name].q_open for name in HARDWARE_JOINTS
        ]
        bend_closed = [
            self.rotate_only.axes[name].q_closed for name in HARDWARE_JOINTS
        ]
        bend_open[5] = rotate_q
        bend_closed[5] = rotate_q
        self.assertEqual(
            self.rotate_only.map_qpos(bend_open),
            (-1, -1, -1, -1, -1, 850),
        )
        self.assertEqual(
            self.rotate_only.map_qpos(bend_closed),
            (-1, -1, -1, -1, -1, 850),
        )

    def test_exact_rotate_only_profile_passes_content_gate(self) -> None:
        self.assertTrue(
            is_safe_thumb_rotate_commissioning_profile(
                self.rotate_only, ("thumb_rotate",)
            )
        )
        self.assertFalse(
            is_safe_thumb_rotate_commissioning_profile(
                self.rotate_only, ("thumb_rotate", "index")
            )
        )
        self.assertFalse(
            is_safe_thumb_rotate_commissioning_profile(
                self.six_dof, ("thumb_rotate",)
            )
        )

    def test_mutated_rotate_only_content_fails_gate(self) -> None:
        rotate_axis = self.rotate_only.axes["thumb_rotate"]
        widened_axes = dict(self.rotate_only.axes)
        widened_axes["thumb_rotate"] = replace(
            rotate_axis,
            command_closed=799,
        )
        wrong_offset_axes = dict(self.rotate_only.axes)
        wrong_offset_axes["thumb_rotate"] = replace(
            rotate_axis,
            feedback_to_command_offset_units=14,
        )
        leaked_offset_axes = dict(self.rotate_only.axes)
        leaked_offset_axes["index"] = replace(
            leaked_offset_axes["index"],
            feedback_to_command_offset_units=1,
            feedback_to_command_valid_min=900,
            feedback_to_command_valid_max=1000,
        )
        for changed in (
            replace(self.rotate_only, speed=81),
            replace(self.rotate_only, force_limit=81),
            replace(self.rotate_only, stream_max_total_current_ma=401),
            replace(self.rotate_only, temporal_filter_axes=HARDWARE_JOINTS),
            replace(self.rotate_only, axes=widened_axes),
            replace(self.rotate_only, axes=wrong_offset_axes),
            replace(self.rotate_only, axes=leaked_offset_axes),
        ):
            self.assertFalse(
                is_safe_thumb_rotate_commissioning_profile(
                    changed, ("thumb_rotate",)
                )
            )

    @staticmethod
    def _hardware_args(profile: Path, axes: str):
        parser = build_parser()
        args = parser.parse_args(
            [
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
                axes,
                "--duration",
                "10",
            ]
        )
        return parser, args

    def test_cli_accepts_only_exact_rotate_only_commissioning_without_token(self) -> None:
        parser, args = self._hardware_args(ROTATE_ONLY_PATH, "thumb_rotate")
        validate_args(args, parser)
        self.assertEqual(args.axes, ("thumb_rotate",))
        self.assertEqual(args.no_hand_policy, "disable")

        parser, args = self._hardware_args(
            ROTATE_ONLY_PATH,
            "thumb_rotate,index",
        )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            validate_args(args, parser)

        parser, args = self._hardware_args(SIX_DOF_PATH, "thumb_rotate")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            validate_args(args, parser)


if __name__ == "__main__":
    unittest.main()
