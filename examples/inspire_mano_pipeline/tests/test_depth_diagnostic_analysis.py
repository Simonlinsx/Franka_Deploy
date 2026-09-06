from __future__ import annotations

import unittest

from analyze_inspire_mano_log import (
    LogFormatError,
    analyze_records,
    render_text,
)


def row(
    frame_number: int,
    *,
    detected: bool = True,
    raw_depth=None,
    control_depth=None,
    source=None,
    reason=None,
    include_control_field: bool = True,
) -> dict:
    result = {
        "log_schema_version": 2,
        "session_id": "depth-diagnostics",
        "frame_number": frame_number,
        "captured_at_monotonic": float(frame_number),
        "right_hand_detected": detected,
        "is_right": True if detected else None,
        "palm_depth_m": raw_depth,
        "palm_depth_source": source,
        "palm_depth_reason": reason,
        "inference_seconds": 0.04,
    }
    if include_control_field:
        result["control_palm_depth_m"] = control_depth
    return result


class DepthDiagnosticAnalysisTests(unittest.TestCase):
    def test_summarizes_measured_held_missing_and_control_depth(self) -> None:
        records = [
            row(
                1,
                raw_depth=0.42,
                control_depth=0.42,
                source="measured",
                reason="measured_consensus",
            ),
            row(
                2,
                raw_depth=None,
                control_depth=0.42,
                source="held",
                reason="held_after_insufficient_valid_palm_rois",
            ),
            row(
                3,
                raw_depth=None,
                control_depth=None,
                source="missing",
                reason="insufficient_valid_palm_rois",
            ),
            row(4, detected=False),
        ]

        summary = analyze_records(records)
        depth = summary["depth"]

        self.assertEqual(depth["valid_frames"], 1)
        self.assertEqual(depth["control_valid_frames"], 2)
        self.assertAlmostEqual(depth["control_valid_rate_among_detections"], 2 / 3)
        self.assertEqual(depth["control_min_m"], 0.42)
        self.assertEqual(depth["control_max_m"], 0.42)
        self.assertEqual(depth["source_sample_count"], 3)
        self.assertEqual(
            depth["source_counts"],
            {"measured": 1, "held": 1, "missing": 1},
        )
        self.assertEqual(depth["reason_sample_count"], 3)
        self.assertEqual(
            depth["reason_counts"],
            {
                "held_after_insufficient_valid_palm_rois": 1,
                "insufficient_valid_palm_rois": 1,
                "measured_consensus": 1,
            },
        )

        text = render_text(summary)
        self.assertIn("有效控制掌深: 2 / 3", text)
        self.assertIn("source_samples=3", text)
        self.assertIn("'measured': 1", text)
        self.assertIn("'held': 1", text)
        self.assertIn("'missing': 1", text)
        self.assertIn("reason_samples=3", text)

    def test_legacy_raw_depth_is_treated_as_control_depth(self) -> None:
        legacy = row(
            1,
            raw_depth=0.55,
            include_control_field=False,
        )
        legacy.pop("palm_depth_source")
        legacy.pop("palm_depth_reason")

        summary = analyze_records([legacy])
        depth = summary["depth"]

        self.assertEqual(depth["valid_frames"], 1)
        self.assertEqual(depth["control_valid_frames"], 1)
        self.assertEqual(depth["control_min_m"], 0.55)
        self.assertEqual(depth["control_max_m"], 0.55)
        self.assertEqual(depth["source_sample_count"], 0)
        self.assertEqual(
            depth["source_counts"],
            {"measured": 0, "held": 0, "missing": 0},
        )
        self.assertEqual(depth["reason_counts"], {})
        self.assertIn("source_samples=0", render_text(summary))

    def test_rejects_invalid_new_depth_diagnostic_fields(self) -> None:
        invalid_source = row(
            1,
            raw_depth=0.4,
            control_depth=0.4,
            source="synthetic",
            reason="measured_consensus",
        )
        with self.assertRaisesRegex(LogFormatError, "palm_depth_source"):
            analyze_records([invalid_source])

        invalid_reason = row(
            1,
            raw_depth=0.4,
            control_depth=0.4,
            source="measured",
            reason=7,
        )
        with self.assertRaisesRegex(LogFormatError, "palm_depth_reason"):
            analyze_records([invalid_reason])

        invalid_control = row(
            1,
            raw_depth=0.4,
            control_depth="0.4",
            source="measured",
            reason="measured_consensus",
        )
        with self.assertRaisesRegex(LogFormatError, "control_palm_depth_m"):
            analyze_records([invalid_control])


if __name__ == "__main__":
    unittest.main()
