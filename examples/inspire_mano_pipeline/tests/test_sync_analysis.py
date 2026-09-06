from __future__ import annotations

import unittest

from analyze_inspire_mano_log import LogFormatError, analyze_records


def row(frame: int, *, pipeline_age: float, copy_align: float) -> dict:
    return {
        "log_schema_version": 2,
        "session_id": "sync-test",
        "frame_number": frame,
        "depth_frame_number": frame - 1,
        "captured_at_monotonic": float(frame),
        "right_hand_detected": False,
        "is_right": None,
        "inference_seconds": 0.04,
        "pipeline_age_seconds": pipeline_age,
        "frame_copy_align_seconds": copy_align,
        "color_sensor_timestamp_ms": frame * 33.0,
        "depth_sensor_timestamp_ms": frame * 33.0 - 2.0,
    }


class SyncAnalysisTests(unittest.TestCase):
    def test_pipeline_age_and_rgbd_sync_statistics(self) -> None:
        summary = analyze_records(
            [
                row(10, pipeline_age=0.05, copy_align=0.004),
                row(11, pipeline_age=0.07, copy_align=0.006),
            ]
        )
        self.assertEqual(summary["pipeline_age"]["sample_count"], 2)
        self.assertAlmostEqual(summary["pipeline_age"]["p50_ms"], 60.0)
        self.assertAlmostEqual(summary["frame_copy_align"]["p95_ms"], 5.9)
        self.assertEqual(
            summary["rgbd_sync"]["sensor_timestamp_abs_delta_p95_ms"],
            2.0,
        )
        self.assertEqual(
            summary["rgbd_sync"]["frame_number_abs_delta_max"], 1
        )
        self.assertEqual(summary["rgbd_sync"]["frame_number_offset_min"], 1)
        self.assertEqual(summary["rgbd_sync"]["frame_number_offset_max"], 1)
        self.assertEqual(summary["rgbd_sync"]["frame_number_offset_span"], 0)

    def test_negative_pipeline_age_is_rejected(self) -> None:
        bad = row(10, pipeline_age=-0.01, copy_align=0.004)
        with self.assertRaisesRegex(LogFormatError, "cannot be negative"):
            analyze_records([bad])


if __name__ == "__main__":
    unittest.main()
