from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from analyze_inspire_mano_log import LogFormatError, analyze_records, main


def row(
    frame_number: int,
    timestamp: float,
    detected: bool,
    latency_seconds: float,
    *,
    depth=None,
    depth_in_range=None,
    qpos=None,
    targets=None,
):
    return {
        "log_schema_version": 2,
        "session_id": "test-session",
        "frame_number": frame_number,
        "captured_at_monotonic": timestamp,
        "right_hand_detected": detected,
        "is_right": True if detected else None,
        "palm_depth_m": depth,
        "palm_depth_in_calibrated_range": depth_in_range,
        "qpos": qpos,
        "hardware_targets": targets,
        "inference_seconds": latency_seconds,
    }


class LogAnalysisTests(unittest.TestCase):
    def test_core_statistics(self) -> None:
        records = [
            row(1, 0.0, False, 0.01),
            row(
                2,
                0.1,
                True,
                0.02,
                depth=0.4,
                depth_in_range=True,
                qpos=[0, 1, 2, 3, 4, 5],
                targets=[900, 800, 700, 600, 500, -1],
            ),
            row(3, 0.2, False, 0.03),
            row(4, 0.5, False, 0.04),
            row(
                5,
                0.7,
                True,
                0.05,
                depth=0.8,
                depth_in_range=False,
                qpos=[10, 11, 12, 13, 14, 15],
                targets=[700, 600, 500, 400, 300, -1],
            ),
            row(
                6,
                0.8,
                True,
                0.06,
                depth=None,
                depth_in_range=False,
                qpos=[13, 14, 15, 16, 17, 18],
                targets=[690, 590, 490, 390, 290, -1],
            ),
        ]

        summary = analyze_records(records)

        self.assertEqual(summary["log_schema"], "per_frame_v2")
        self.assertEqual(summary["total_frames"], 6)
        self.assertEqual(summary["right_hand"]["detected_frames"], 3)
        self.assertAlmostEqual(summary["right_hand"]["detection_rate"], 0.5)
        self.assertAlmostEqual(summary["inference_latency"]["p50_ms"], 35.0)
        self.assertAlmostEqual(summary["inference_latency"]["p95_ms"], 57.5)
        detected_latency = summary["inference_latency"]["detected_hand"]
        self.assertEqual(detected_latency["sample_count"], 3)
        self.assertAlmostEqual(detected_latency["p50_ms"], 50.0)
        self.assertAlmostEqual(detected_latency["p95_ms"], 59.0)
        missing_latency = summary["inference_latency"]["no_detected_hand"]
        self.assertEqual(missing_latency["sample_count"], 3)
        self.assertAlmostEqual(missing_latency["p50_ms"], 30.0)
        self.assertAlmostEqual(missing_latency["p95_ms"], 39.0)
        self.assertEqual(summary["depth"]["valid_frames"], 2)
        self.assertAlmostEqual(
            summary["depth"]["valid_rate_among_detections"], 2 / 3
        )
        self.assertEqual(summary["depth"]["min_m"], 0.4)
        self.assertEqual(summary["depth"]["max_m"], 0.8)
        self.assertAlmostEqual(summary["depth"]["in_calibrated_range_rate"], 1 / 3)

        longest = summary["longest_missed_detection"]
        self.assertEqual(longest["frames"], 2)
        self.assertAlmostEqual(longest["sample_span_seconds"], 0.3)
        self.assertEqual(longest["start_frame_number"], 3)
        self.assertEqual(longest["end_frame_number"], 4)
        self.assertFalse(longest["leading_censored"])
        self.assertFalse(longest["trailing_censored"])

        pinky_qpos = summary["qpos"]["axes"]["pinky"]
        self.assertEqual(pinky_qpos["min"], 0.0)
        self.assertEqual(pinky_qpos["max"], 13.0)
        # Missing rows break adjacency: the 0 -> 10 reacquisition is not called
        # an adjacent-frame jump, while frame 5 -> 6 is.
        self.assertEqual(pinky_qpos["max_adjacent_frame_jump"], 3.0)
        pinky_target = summary["hardware_targets"]["axes"]["pinky"]
        self.assertEqual(pinky_target["min"], 690.0)
        self.assertEqual(pinky_target["max"], 900.0)
        self.assertEqual(pinky_target["max_adjacent_frame_jump"], 10.0)

    def test_legacy_detection_only_log_does_not_claim_detection_rate(self) -> None:
        legacy = [
            {
                "frame_number": 10,
                "captured_at_monotonic": 1.0,
                "is_right": True,
                "inference_seconds": 0.02,
                "palm_depth_m": None,
                "qpos": [0, 0, 0, 0, 0, 0],
                "hardware_targets": [900, 900, 900, 900, 900, -1],
            },
            {
                "frame_number": 20,
                "captured_at_monotonic": 2.0,
                "is_right": True,
                "inference_seconds": 0.03,
                "palm_depth_m": None,
                "qpos": [1, 1, 1, 1, 1, 1],
                "hardware_targets": [800, 800, 800, 800, 800, -1],
            },
        ]
        summary = analyze_records(legacy)
        self.assertEqual(summary["status"], "partial")
        self.assertIsNone(summary["total_frames"])
        self.assertIsNone(summary["right_hand"]["detection_rate"])
        self.assertIsNone(summary["longest_missed_detection"])

    def test_json_cli_and_strict_boolean_validation(self) -> None:
        record = row(1, 1.0, True, 0.01, depth=0.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main([str(path), "--json"]), 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["total_frames"], 1)
            self.assertEqual(payload["input_metadata"]["session_ids"], ["test-session"])

        bad = dict(record)
        bad["right_hand_detected"] = "false"
        with self.assertRaisesRegex(LogFormatError, "JSON boolean"):
            analyze_records([bad])


if __name__ == "__main__":
    unittest.main()
