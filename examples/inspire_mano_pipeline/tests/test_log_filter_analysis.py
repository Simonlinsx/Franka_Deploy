from __future__ import annotations

import unittest

from analyze_inspire_mano_log import (
    AXES,
    LogFormatError,
    analyze_records,
    render_text,
)


def filter_row(
    frame_number: int,
    timestamp: float,
    *,
    raw_index_qpos: float,
    filtered_index_qpos: float,
    raw_index_target: int,
    filtered_index_target: int,
) -> dict:
    raw_qpos = [float(frame_number - 1 + index) for index in range(len(AXES))]
    filtered_qpos = list(raw_qpos)
    raw_qpos[3] = raw_index_qpos
    filtered_qpos[3] = filtered_index_qpos

    raw_targets = [900 - 10 * (frame_number - 1) - index for index in range(6)]
    filtered_targets = list(raw_targets)
    raw_targets[3] = raw_index_target
    filtered_targets[3] = filtered_index_target
    raw_targets[5] = -1
    filtered_targets[5] = -1
    return {
        "log_schema_version": 2,
        "session_id": "filter-analysis",
        "frame_number": frame_number,
        "captured_at_monotonic": timestamp,
        "right_hand_detected": True,
        "is_right": True,
        "qpos": filtered_qpos,
        "raw_qpos": raw_qpos,
        "hardware_targets": filtered_targets,
        "raw_hardware_targets": raw_targets,
        "inference_seconds": 0.04,
    }


class RawToFilteredLogAnalysisTests(unittest.TestCase):
    def test_reports_all_axis_ranges_jitter_and_filter_adjustment(self) -> None:
        rows = [
            filter_row(
                1,
                1.0,
                raw_index_qpos=0.0,
                filtered_index_qpos=0.0,
                raw_index_target=1000,
                filtered_index_target=1000,
            ),
            filter_row(
                2,
                1.1,
                raw_index_qpos=1.0,
                filtered_index_qpos=0.5,
                raw_index_target=0,
                filtered_index_target=500,
            ),
            filter_row(
                3,
                1.2,
                raw_index_qpos=0.0,
                filtered_index_qpos=0.0,
                raw_index_target=1000,
                filtered_index_target=1000,
            ),
        ]

        summary = analyze_records(rows)

        self.assertEqual(summary["analyzer_schema_version"], 4)
        filtering = summary["filtering"]
        self.assertTrue(filtering["available"])

        qpos = filtering["qpos"]
        self.assertTrue(qpos["available"])
        self.assertEqual(qpos["raw_sample_frames"], 3)
        self.assertEqual(qpos["filtered_sample_frames"], 3)
        self.assertEqual(qpos["paired_sample_frames"], 3)
        self.assertEqual(qpos["adjacent_pair_count"], 2)
        self.assertEqual(set(qpos["axes"]), set(AXES))

        index_qpos = qpos["axes"]["index"]
        self.assertEqual(index_qpos["raw_min"], 0.0)
        self.assertEqual(index_qpos["raw_max"], 1.0)
        self.assertEqual(index_qpos["raw_span"], 1.0)
        self.assertEqual(index_qpos["filtered_min"], 0.0)
        self.assertEqual(index_qpos["filtered_max"], 0.5)
        self.assertEqual(index_qpos["filtered_span"], 0.5)
        self.assertEqual(index_qpos["raw_adjacent_jump_p95"], 1.0)
        self.assertEqual(index_qpos["raw_adjacent_jump_max"], 1.0)
        self.assertEqual(index_qpos["filtered_adjacent_jump_p95"], 0.5)
        self.assertEqual(index_qpos["filtered_adjacent_jump_max"], 0.5)
        self.assertEqual(index_qpos["adjacent_jump_p95_reduction"], 0.5)
        self.assertAlmostEqual(
            index_qpos["mean_abs_raw_to_filtered_delta"], 1.0 / 6.0
        )
        self.assertEqual(index_qpos["max_abs_raw_to_filtered_delta"], 0.5)

        # The report covers all six axes and exposes accidental cross-axis
        # filtering: an untouched axis must have zero raw->filtered delta.
        pinky_qpos = qpos["axes"]["pinky"]
        self.assertEqual(pinky_qpos["raw_span"], 2.0)
        self.assertEqual(pinky_qpos["filtered_span"], 2.0)
        self.assertEqual(pinky_qpos["raw_adjacent_jump_p95"], 1.0)
        self.assertEqual(pinky_qpos["filtered_adjacent_jump_p95"], 1.0)
        self.assertEqual(pinky_qpos["max_abs_raw_to_filtered_delta"], 0.0)

        targets = filtering["hardware_targets"]
        self.assertTrue(targets["available"])
        index_target = targets["axes"]["index"]
        self.assertEqual(index_target["raw_span"], 1000.0)
        self.assertEqual(index_target["filtered_span"], 500.0)
        self.assertEqual(index_target["raw_adjacent_jump_p95"], 1000.0)
        self.assertEqual(index_target["filtered_adjacent_jump_p95"], 500.0)
        self.assertEqual(index_target["adjacent_jump_p95_reduction"], 500.0)
        self.assertEqual(index_target["max_abs_raw_to_filtered_delta"], 500.0)

        self.assertEqual(summary["raw_qpos"]["sample_frames"], 3)
        self.assertEqual(summary["raw_hardware_targets"]["sample_frames"], 3)
        text = render_text(summary)
        self.assertIn("滤波对照 qpos", text)
        self.assertIn("滤波对照 未屏蔽硬件目标", text)
        self.assertIn("index", text)

    def test_old_log_without_raw_vectors_remains_compatible(self) -> None:
        row = filter_row(
            1,
            1.0,
            raw_index_qpos=0.0,
            filtered_index_qpos=0.0,
            raw_index_target=1000,
            filtered_index_target=1000,
        )
        row.pop("raw_qpos")
        row.pop("raw_hardware_targets")

        summary = analyze_records([row])

        self.assertFalse(summary["filtering"]["available"])
        self.assertFalse(summary["filtering"]["qpos"]["available"])
        self.assertFalse(summary["filtering"]["hardware_targets"]["available"])
        self.assertEqual(summary["raw_qpos"]["sample_frames"], 0)
        self.assertEqual(summary["raw_hardware_targets"]["sample_frames"], 0)
        self.assertNotIn("滤波对照", render_text(summary))

    def test_missing_pair_breaks_jitter_adjacency(self) -> None:
        first = filter_row(
            1,
            1.0,
            raw_index_qpos=0.0,
            filtered_index_qpos=0.0,
            raw_index_target=1000,
            filtered_index_target=1000,
        )
        missing = filter_row(
            2,
            1.1,
            raw_index_qpos=0.5,
            filtered_index_qpos=0.25,
            raw_index_target=500,
            filtered_index_target=750,
        )
        missing["raw_qpos"] = None
        missing["raw_hardware_targets"] = None
        third = filter_row(
            3,
            1.2,
            raw_index_qpos=1.0,
            filtered_index_qpos=0.5,
            raw_index_target=0,
            filtered_index_target=500,
        )

        comparison = analyze_records([first, missing, third])["filtering"]
        self.assertEqual(comparison["qpos"]["paired_sample_frames"], 2)
        self.assertEqual(comparison["qpos"]["adjacent_pair_count"], 0)
        self.assertIsNone(
            comparison["qpos"]["axes"]["index"]["raw_adjacent_jump_p95"]
        )
        self.assertEqual(
            comparison["qpos"]["axes"]["index"]["raw_span"], 1.0
        )

    def test_malformed_raw_vector_is_rejected(self) -> None:
        row = filter_row(
            1,
            1.0,
            raw_index_qpos=0.0,
            filtered_index_qpos=0.0,
            raw_index_target=1000,
            filtered_index_target=1000,
        )
        row["raw_qpos"] = [0, 0, 0, 0, 0]
        with self.assertRaisesRegex(
            LogFormatError, "raw_qpos.*six-number array"
        ):
            analyze_records([row])


if __name__ == "__main__":
    unittest.main()
