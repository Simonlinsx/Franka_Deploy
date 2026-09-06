from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from inspire_mano_pipeline.calibration import PipelineCalibration
from realsense_mano_inspire import (
    CURRENT_MONITOR_ONLY_CONFIRMATION,
    build_parser,
    is_safe_commissioning_profile,
    validate_args,
)


ROOT = Path(__file__).resolve().parents[3]
COMMISSIONING_PATH = ROOT / (
    "examples/inspire_mano_pipeline/inspire_rh56bfx_right_commissioning.json"
)


class CommissioningContentGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calibration = PipelineCalibration.load(COMMISSIONING_PATH)

    def test_official_content_allows_only_index(self) -> None:
        self.assertTrue(
            is_safe_commissioning_profile(self.calibration, ("index",))
        )
        self.assertFalse(
            is_safe_commissioning_profile(
                self.calibration, ("index", "middle")
            )
        )

    def test_mutated_high_energy_content_is_not_commissioning(self) -> None:
        for changed in (
            replace(self.calibration, speed=81),
            replace(self.calibration, force_limit=81),
            replace(self.calibration, active_current_policy="monitor_only"),
            replace(self.calibration, stream_max_current_ma=601),
        ):
            self.assertFalse(
                is_safe_commissioning_profile(changed, ("index",))
            )

    def test_mutated_wide_index_span_is_not_commissioning(self) -> None:
        index = replace(
            self.calibration.axes["index"], command_closed=600
        )
        axes = dict(self.calibration.axes)
        axes["index"] = index
        changed = replace(self.calibration, axes=axes)
        self.assertFalse(
            is_safe_commissioning_profile(changed, ("index",))
        )

    def test_feedback_offset_cannot_leak_into_index_commissioning(self) -> None:
        axes = dict(self.calibration.axes)
        axes["index"] = replace(
            axes["index"],
            feedback_to_command_offset_units=1,
            feedback_to_command_valid_min=700,
            feedback_to_command_valid_max=900,
        )
        changed = replace(self.calibration, axes=axes)
        self.assertFalse(is_safe_commissioning_profile(changed, ("index",)))

    def test_hardware_dex_is_refused_even_with_wide_range_token(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--enable-hardware",
                "--confirm-hardware-motion",
                "--port",
                "/dev/fake",
                "--camera-serial",
                "fake-camera",
                "--device",
                "cuda:0",
                "--calibration",
                str(COMMISSIONING_PATH),
                "--retargeter",
                "dex",
                "--axes",
                "index",
                "--duration",
                "10",
                "--confirm-wide-range",
                "RH56_WIDE_RANGE",
            ]
        )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
            SystemExit
        ):
            validate_args(args, parser)

    def test_monitor_only_current_policy_is_rejected_for_arbitrary_profile(
        self,
    ) -> None:
        raw = json.loads(COMMISSIONING_PATH.read_text(encoding="utf-8"))
        raw["safety"]["active_current_policy"] = "monitor_only"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "monitor.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
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
                    str(path),
                    "--retargeter",
                    "geometric",
                    "--enable-hardware",
                    "--confirm-hardware-motion",
                    "--confirm-current-monitor-only",
                    CURRENT_MONITOR_ONLY_CONFIRMATION,
                    "--port",
                    "/dev/fake",
                    "--axes",
                    "index",
                    "--duration",
                    "10",
                ]
            )
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
                SystemExit
            ):
                validate_args(args, parser)


if __name__ == "__main__":
    unittest.main()
