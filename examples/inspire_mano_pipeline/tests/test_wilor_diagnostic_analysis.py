from __future__ import annotations

import unittest

from analyze_inspire_mano_log import LogFormatError, analyze_records


def row(frame: int, diagnostics) -> dict:
    return {
        "log_schema_version": 2,
        "session_id": "diagnostics-test",
        "frame_number": frame,
        "captured_at_monotonic": float(frame),
        "right_hand_detected": diagnostics["result"] == "accepted",
        "is_right": diagnostics["result"] == "accepted" or None,
        "inference_seconds": 0.01,
        "wilor_diagnostics": diagnostics,
    }


class WiLoRDiagnosticAnalysisTests(unittest.TestCase):
    def test_candidate_labels_and_rejection_reasons_are_counted(self) -> None:
        summary = analyze_records(
            [
                row(
                    1,
                    {
                        "candidate_count": 0,
                        "detector_labels": [],
                        "result": "no_candidates",
                    },
                ),
                row(
                    2,
                    {
                        "candidate_count": 1,
                        "detector_labels": ["right"],
                        "result": "handedness_mismatch",
                    },
                ),
                row(
                    3,
                    {
                        "candidate_count": 1,
                        "detector_labels": ["left"],
                        "result": "accepted",
                    },
                ),
            ]
        )
        self.assertEqual(summary["wilor"]["diagnostic_frames"], 3)
        self.assertEqual(summary["wilor"]["frames_with_candidates"], 2)
        self.assertEqual(
            summary["wilor"]["result_counts"],
            {
                "accepted": 1,
                "handedness_mismatch": 1,
                "no_candidates": 1,
            },
        )
        self.assertEqual(
            summary["wilor"]["detector_label_counts"],
            {"left": 1, "right": 1},
        )

    def test_malformed_diagnostics_are_rejected(self) -> None:
        bad = row(
            1,
            {
                "candidate_count": True,
                "detector_labels": [],
                "result": "no_candidates",
            },
        )
        with self.assertRaisesRegex(LogFormatError, "candidate_count"):
            analyze_records([bad])


if __name__ == "__main__":
    unittest.main()
