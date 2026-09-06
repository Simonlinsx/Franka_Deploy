from __future__ import annotations

import contextlib
import io
import json
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from inspire_mano_pipeline.calibration import PipelineCalibration
from inspire_mano_pipeline.model import CameraFrame
from realsense_mano_inspire import (
    NO_HAND_OPEN_CONFIRMATION,
    NoHandOpenController,
    build_parser,
    open_fallback_targets,
    preview_control_source,
    selected_axes_are_calibrated_open,
    validate_args,
    write_record,
)


ROOT = Path(__file__).resolve().parents[3]
COMMISSIONING = ROOT / (
    "examples/inspire_mano_pipeline/inspire_rh56bfx_right_commissioning.json"
)
WIDE = ROOT / "examples/inspire_mano_pipeline/inspire_rh56bfx_right.json"


class NoHandPolicyCliTests(unittest.TestCase):
    @staticmethod
    def _validate(arguments):
        parser = build_parser()
        args = parser.parse_args(arguments)
        validate_args(args, parser)
        return args

    @staticmethod
    def _hardware_arguments():
        return [
            "--enable-hardware",
            "--confirm-hardware-motion",
            "--camera-serial",
            "fake-camera",
            "--device",
            "cuda:0",
            "--port",
            "/dev/fake-hand",
            "--calibration",
            str(COMMISSIONING),
            "--retargeter",
            "geometric",
            "--axes",
            "index",
            "--duration",
            "10",
        ]

    def test_default_policy_is_disable_and_preview_open_needs_no_confirmation(self):
        self.assertEqual(self._validate([]).no_hand_policy, "disable")
        preview = self._validate(["--no-hand-policy", "open"])
        self.assertEqual(preview.no_hand_policy, "open")

    def test_hardware_open_requires_exact_confirmation(self):
        arguments = self._hardware_arguments() + ["--no-hand-policy", "open"]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self._validate(arguments)
        args = self._validate(
            arguments
            + ["--confirm-no-hand-open", NO_HAND_OPEN_CONFIRMATION]
        )
        self.assertEqual(args.no_hand_policy, "open")

    def test_hardware_open_rejects_non_commissioning_content(self):
        arguments = self._hardware_arguments()
        calibration_index = arguments.index(str(COMMISSIONING))
        arguments[calibration_index] = str(WIDE)
        arguments.extend(
            [
                "--no-hand-policy",
                "open",
                "--confirm-no-hand-open",
                NO_HAND_OPEN_CONFIRMATION,
                "--confirm-wide-range",
                "RH56_WIDE_RANGE",
            ]
        )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self._validate(arguments)


class NoHandOpenControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calibration = replace(
            PipelineCalibration.load(COMMISSIONING),
            valid_frames_to_arm=3,
            arming_max_target_delta_units=10,
        )
        self.targets = (-1, -1, -1, 800, -1, -1)

    def _controller(self):
        return NoHandOpenController(
            self.calibration,
            ("index",),
            "open",
            fallback_delay_seconds=0.30,
        )

    def test_fallback_cannot_arm_from_zero_mano(self):
        controller = self._controller()
        self.assertEqual(
            controller.choose(
                frame_captured_at_monotonic=10.0,
                detection_present=False,
                depth_in_calibrated_range=False,
                mano_targets=None,
                stream_ever_active=False,
            ),
            "none",
        )
        self.assertEqual(
            controller.choose(
                frame_captured_at_monotonic=20.0,
                detection_present=False,
                depth_in_calibrated_range=False,
                mano_targets=None,
                stream_ever_active=False,
            ),
            "none",
        )
        self.assertFalse(controller.fallback_active)

    def test_fallback_waits_for_loss_delay_after_mano_was_active(self):
        controller = self._controller()
        self.assertEqual(
            controller.choose(
                frame_captured_at_monotonic=100.0,
                detection_present=True,
                depth_in_calibrated_range=True,
                mano_targets=self.targets,
                stream_ever_active=True,
            ),
            "mano",
        )
        self.assertEqual(
            controller.choose(
                frame_captured_at_monotonic=100.29,
                detection_present=False,
                depth_in_calibrated_range=False,
                mano_targets=None,
                stream_ever_active=True,
            ),
            "none",
        )
        self.assertEqual(
            controller.choose(
                frame_captured_at_monotonic=100.31,
                detection_present=False,
                depth_in_calibrated_range=False,
                mano_targets=None,
                stream_ever_active=True,
            ),
            "fallback_open",
        )

    def test_stable_reacquisition_is_required_before_returning_to_mano(self):
        controller = self._controller()
        controller.last_valid_mano_capture = 10.0
        self.assertEqual(
            controller.choose(
                frame_captured_at_monotonic=10.31,
                detection_present=False,
                depth_in_calibrated_range=False,
                mano_targets=None,
                stream_ever_active=True,
            ),
            "fallback_open",
        )
        for index, target in enumerate((800, 805), start=1):
            values = (-1, -1, -1, target, -1, -1)
            self.assertEqual(
                controller.choose(
                    frame_captured_at_monotonic=10.31 + index * 0.03,
                    detection_present=True,
                    depth_in_calibrated_range=True,
                    mano_targets=values,
                    stream_ever_active=True,
                ),
                "fallback_open",
            )
        self.assertEqual(
            controller.choose(
                frame_captured_at_monotonic=10.40,
                detection_present=True,
                depth_in_calibrated_range=True,
                mano_targets=(-1, -1, -1, 810, -1, -1),
                stream_ever_active=True,
            ),
            "mano",
        )
        self.assertTrue(controller.fallback_active)
        controller.note_mano_submit_result(True)
        self.assertFalse(controller.fallback_active)

    def test_reacquisition_release_requires_stream_acceptance(self):
        controller = self._controller()
        controller.last_valid_mano_capture = 5.0
        controller.choose(
            frame_captured_at_monotonic=5.31,
            detection_present=False,
            depth_in_calibrated_range=False,
            mano_targets=None,
            stream_ever_active=True,
        )
        for index, target in enumerate((800, 805, 810), start=1):
            source = controller.choose(
                frame_captured_at_monotonic=5.31 + index * 0.03,
                detection_present=True,
                depth_in_calibrated_range=True,
                mano_targets=(-1, -1, -1, target, -1, -1),
                stream_ever_active=True,
            )
        self.assertEqual(source, "mano")
        controller.note_mano_submit_result(False)
        self.assertTrue(controller.fallback_active)
        self.assertEqual(
            controller.choose(
                frame_captured_at_monotonic=5.43,
                detection_present=True,
                depth_in_calibrated_range=True,
                mano_targets=(-1, -1, -1, 812, -1, -1),
                stream_ever_active=True,
            ),
            "fallback_open",
        )

    def test_large_reacquisition_jump_resets_the_count(self):
        controller = self._controller()
        controller.last_valid_mano_capture = 1.0
        controller.choose(
            frame_captured_at_monotonic=1.31,
            detection_present=False,
            depth_in_calibrated_range=False,
            mano_targets=None,
            stream_ever_active=True,
        )
        sources = []
        for timestamp, target in ((1.34, 800), (1.37, 850), (1.40, 855)):
            sources.append(
                controller.choose(
                    frame_captured_at_monotonic=timestamp,
                    detection_present=True,
                    depth_in_calibrated_range=True,
                    mano_targets=(-1, -1, -1, target, -1, -1),
                    stream_ever_active=True,
                )
            )
        self.assertEqual(sources, ["fallback_open"] * 3)
        self.assertTrue(controller.fallback_active)

    def test_preview_can_show_calibrated_open_without_hardware(self):
        self.assertEqual(preview_control_source("open", False, False), "fallback_open")
        self.assertEqual(preview_control_source("open", True, True), "mano")
        self.assertEqual(preview_control_source("disable", False, False), "none")


class NoHandOpenEvidenceTests(unittest.TestCase):
    def test_targets_and_start_tolerance_use_calibrated_open(self):
        calibration = PipelineCalibration.load(COMMISSIONING)
        targets = open_fallback_targets(calibration, ("index",))
        self.assertEqual(targets, (-1, -1, -1, 900, -1, -1))
        self.assertTrue(
            selected_axes_are_calibrated_open(
                calibration, ("index",), (0, 0, 0, 875, 0, 0)
            )
        )
        self.assertFalse(
            selected_axes_are_calibrated_open(
                calibration, ("index",), (0, 0, 0, 869, 0, 0)
            )
        )

    def test_fallback_log_has_no_synthetic_mano_output(self):
        frame = CameraFrame(
            color_bgr=np.zeros((4, 4, 3), dtype=np.uint8),
            depth_m=np.zeros((4, 4), dtype=np.float32),
            captured_at_monotonic=12.0,
            frame_number=7,
        )
        targets = (-1, -1, -1, 900, -1, -1)
        handle = io.StringIO()
        write_record(
            handle,
            frame,
            detection=None,
            output=None,
            inference_seconds=0.01,
            masked_targets=targets,
            accepted=True,
            control_source="fallback_open",
            mano_submit_count=5,
            fallback_submit_count=2,
        )
        row = json.loads(handle.getvalue())
        self.assertEqual(row["control_source"], "fallback_open")
        self.assertEqual(row["masked_hardware_targets"], list(targets))
        self.assertIsNone(row["qpos"])
        self.assertIsNone(row["hardware_targets"])
        self.assertEqual(row["accepted_mano_target_count"], 5)
        self.assertEqual(row["accepted_fallback_open_target_count"], 2)


if __name__ == "__main__":
    unittest.main()
