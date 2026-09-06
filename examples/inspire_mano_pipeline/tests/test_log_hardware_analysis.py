from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analyze_inspire_mano_log import (
    LogFormatError,
    analyze_jsonl,
    analyze_records,
    render_text,
)


def hardware_row(
    frame_number: int,
    timestamp: float,
    *,
    state: str,
    accepted,
    masked,
    sent,
    session_id: str = "hardware-session",
) -> dict:
    return {
        "log_schema_version": 2,
        "session_id": session_id,
        "frame_number": frame_number,
        "captured_at_monotonic": timestamp,
        "right_hand_detected": True,
        "is_right": True,
        "palm_depth_m": 0.5,
        "palm_depth_in_calibrated_range": True,
        "inference_seconds": 0.04,
        "qpos": [0.0, 0.0, 0.0, 0.2, 0.0, 0.0],
        "hardware_targets": [900, 900, 900, 800, 900, -1],
        "masked_hardware_targets": masked,
        "hardware_submit_accepted": accepted,
        "hardware_state": state,
        "last_sent_targets": sent,
    }


class HardwareLogAnalysisTests(unittest.TestCase):
    @staticmethod
    def _summary_with_shutdown(payload: dict) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            jsonl = run_dir / "mano_retarget.jsonl"
            row = hardware_row(
                1,
                1.0,
                state="active",
                accepted=True,
                masked=[-1, -1, -1, 850, -1, -1],
                sent=[-1, -1, -1, 890, -1, -1],
            )
            jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")
            (run_dir / "shutdown_status.json").write_text(
                json.dumps(payload) + "\n", encoding="utf-8"
            )
            return analyze_jsonl(jsonl)

    @staticmethod
    def _safe_shutdown_payload() -> dict:
        return {
            "session_id": "hardware-session",
            "stop_confirmed": True,
            "physical_stop_verified": True,
            "shutdown_feedback": {
                "angles": [1000, 1000, 1000, 1000, 1000, 990],
                "positions": [100, 100, 100, 100, 100, 120],
                "currents": [0, 0, 0, 0, 0, 0],
                "errors": [0, 0, 0, 0, 0, 0],
                "statuses": [2, 2, 2, 2, 2, 255],
                "temperatures": [30, 30, 30, 30, 30, 30],
            },
            "final_angle_targets": [-1, -1, -1, -1, -1, -1],
        }

    def test_preview_targets_do_not_imply_hardware_activity(self) -> None:
        preview = hardware_row(
            1,
            1.0,
            state=None,
            accepted=None,
            masked=[900, 900, 900, 900, 900, -1],
            sent=None,
        )
        summary = analyze_records([preview])
        self.assertFalse(summary["hardware"]["evidence_present"])
        feedback = summary["hardware"]["feedback"]
        self.assertFalse(feedback["available"])
        self.assertEqual(
            feedback["sample_frames"],
            {"actual_angles": 0, "currents": 0, "temperatures": 0, "errors": 0},
        )
        self.assertIsNone(feedback["axes"]["index"]["actual_angle_min"])
        self.assertIsNone(feedback["axes"]["index"]["nonzero_error_frames"])

    def test_per_axis_hardware_feedback_statistics(self) -> None:
        first = hardware_row(
            1,
            1.0,
            state="active",
            accepted=True,
            masked=[-1, -1, -1, 850, -1, -1],
            sent=[-1, -1, -1, 890, -1, -1],
        )
        first.update(
            {
                "actual_angles": [1000, 990, 980, 970, 960, 888],
                "currents": [-40, 10, 20, 100, 0, -5],
                "temperatures": [30, 31, 32, 33, 29, 28],
                "errors": [0, 1, 0, 2, 0, 0],
            }
        )
        second = hardware_row(
            2,
            1.1,
            state="active",
            accepted=True,
            masked=[-1, -1, -1, 840, -1, -1],
            sent=[-1, -1, -1, 880, -1, -1],
        )
        second.update(
            {
                "actual_angles": [980, 995, 975, 940, 965, 888],
                "currents": [30, -20, 25, -125, 1, 0],
                "temperatures": [32, 30, 34, 35, 30, 29],
                "errors": [0, 1, 0, 0, 0, 4],
            }
        )

        summary = analyze_records([first, second])

        feedback = summary["hardware"]["feedback"]
        self.assertTrue(feedback["available"])
        self.assertEqual(
            feedback["sample_frames"],
            {"actual_angles": 2, "currents": 2, "temperatures": 2, "errors": 2},
        )
        self.assertEqual(feedback["frames_with_any_nonzero_error"], 2)
        pinky = feedback["axes"]["pinky"]
        self.assertEqual(pinky["actual_angle_min"], 980.0)
        self.assertEqual(pinky["actual_angle_max"], 1000.0)
        self.assertEqual(pinky["max_abs_current_ma"], 40.0)
        self.assertEqual(pinky["max_temperature_c"], 32.0)
        self.assertEqual(pinky["nonzero_error_frames"], 0)
        index = feedback["axes"]["index"]
        self.assertEqual(index["actual_angle_min"], 940.0)
        self.assertEqual(index["actual_angle_max"], 970.0)
        self.assertEqual(index["max_abs_current_ma"], 125.0)
        self.assertEqual(index["max_temperature_c"], 35.0)
        self.assertEqual(index["nonzero_error_frames"], 1)
        self.assertEqual(feedback["axes"]["ring"]["nonzero_error_frames"], 2)
        self.assertEqual(feedback["axes"]["thumb_rotate"]["nonzero_error_frames"], 1)
        self.assertIn("|电流|max", render_text(summary))

    def test_malformed_feedback_vector_is_rejected(self) -> None:
        row = hardware_row(
            1,
            1.0,
            state="active",
            accepted=True,
            masked=[-1, -1, -1, 850, -1, -1],
            sent=[-1, -1, -1, 890, -1, -1],
        )
        row["currents"] = [0, 0, 0, 0, 0]
        with self.assertRaisesRegex(LogFormatError, "currents.*six-number array"):
            analyze_records([row])

    def test_actual_and_masked_targets_and_motion_counts(self) -> None:
        rows = [
            hardware_row(
                1,
                1.0,
                state="waiting_for_tracking",
                accepted=True,
                masked=[-1, -1, -1, 850, -1, -1],
                sent=None,
            ),
            hardware_row(
                2,
                1.1,
                state="active",
                accepted=True,
                masked=[-1, -1, -1, 840, -1, -1],
                sent=[-1, -1, -1, 890, -1, -1],
            ),
            hardware_row(
                3,
                1.2,
                state="active",
                accepted=False,
                masked=[-1, -1, -1, 830, -1, -1],
                sent=[-1, -1, -1, 890, -1, -1],
            ),
            hardware_row(
                4,
                1.3,
                state="active",
                accepted=None,
                masked=None,
                sent=[-1, -1, -1, 880, -1, -1],
            ),
        ]

        summary = analyze_records(rows)

        self.assertTrue(summary["hardware"]["evidence_present"])
        self.assertTrue(summary["hardware"]["ever_active"])
        self.assertEqual(summary["hardware"]["active_frames"], 3)
        self.assertEqual(
            summary["hardware"]["states_seen"],
            ["active", "waiting_for_tracking"],
        )
        submission = summary["hardware"]["submission"]
        self.assertEqual(submission["attempted_frames"], 3)
        self.assertEqual(submission["accepted_frames"], 2)
        self.assertEqual(submission["rejected_frames"], 1)
        self.assertAlmostEqual(submission["acceptance_rate"], 2 / 3)
        motion = summary["hardware"]["motion"]
        self.assertEqual(motion["observed_frames"], 3)
        self.assertEqual(motion["distinct_target_vectors"], 2)
        self.assertEqual(motion["observed_target_changes"], 1)

        masked_index = summary["masked_hardware_targets"]["axes"]["index"]
        self.assertEqual(masked_index["min"], 830.0)
        self.assertEqual(masked_index["max"], 850.0)
        self.assertEqual(masked_index["max_adjacent_frame_jump"], 10.0)
        sent_index = summary["last_sent_targets"]["axes"]["index"]
        self.assertEqual(sent_index["min"], 880.0)
        self.assertEqual(sent_index["max"], 890.0)
        self.assertEqual(sent_index["max_adjacent_frame_jump"], 10.0)

    def test_matching_shutdown_file_verifies_final_all_minus_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            jsonl = run_dir / "mano_retarget.jsonl"
            row = hardware_row(
                1,
                1.0,
                state="active",
                accepted=True,
                masked=[-1, -1, -1, 850, -1, -1],
                sent=[-1, -1, -1, 890, -1, -1],
            )
            jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")
            (run_dir / "shutdown_status.json").write_text(
                json.dumps(
                    {
                        "session_id": "hardware-session",
                        "stop_confirmed": True,
                        "physical_stop_verified": True,
                        "shutdown_feedback": {
                            "angles": [1000, 1000, 1000, 1000, 1000, 990],
                            "positions": [100, 100, 100, 100, 100, 120],
                            "currents": [0, 0, 0, 0, 0, 0],
                            "errors": [0, 0, 0, 0, 0, 0],
                            "statuses": [2, 2, 2, 2, 2, 255],
                            "temperatures": [30, 30, 30, 30, 30, 30],
                        },
                        "final_angle_targets": [-1, -1, -1, -1, -1, -1],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = analyze_jsonl(jsonl)

        shutdown = summary["shutdown"]
        self.assertTrue(shutdown["checked"])
        self.assertTrue(shutdown["file_present"])
        self.assertTrue(shutdown["session_matches_log"])
        self.assertTrue(shutdown["physical_stop_verified"])
        self.assertTrue(shutdown["shutdown_feedback_idle"])
        self.assertTrue(shutdown["final_all_minus_one"])
        self.assertTrue(shutdown["safe_stop_verified"])
        self.assertEqual(shutdown["verification_status"], "safe")
        self.assertIn("status=safe", render_text(summary))

    def test_legacy_target_release_only_is_accepted_but_not_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            jsonl = run_dir / "mano_retarget.jsonl"
            row = hardware_row(
                1,
                1.0,
                state="active",
                accepted=True,
                masked=[-1, -1, -1, 850, -1, -1],
                sent=[-1, -1, -1, 890, -1, -1],
            )
            jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")
            (run_dir / "shutdown_status.json").write_text(
                json.dumps(
                    {
                        "session_id": "hardware-session",
                        "stop_confirmed": True,
                        "final_angle_targets": [-1, -1, -1, -1, -1, -1],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = analyze_jsonl(jsonl)

        shutdown = summary["shutdown"]
        self.assertIsNone(shutdown["physical_stop_verified"])
        self.assertIsNone(shutdown["shutdown_feedback"])
        self.assertIsNone(shutdown["shutdown_feedback_idle"])
        self.assertFalse(shutdown["safe_stop_verified"])
        self.assertEqual(shutdown["verification_status"], "unsafe")

    def test_monitor_only_current_breaches_do_not_override_safe_shutdown(
        self,
    ) -> None:
        payload = self._safe_shutdown_payload()
        payload.update(
            {
                "active_current_policy": "monitor_only",
                "host_current_per_axis_threshold_ma": 400,
                "host_current_selected_total_threshold_ma": 600,
                "active_current_over_limit_sample_count": 3,
                "active_current_warning_event_count": 1,
                "active_peak_abs_currents": [166, 220, 99, 111, 64, 23],
                "active_max_selected_total_current_ma": 619,
                "verified_device_current_limits": [1400] * 6,
            }
        )

        summary = self._summary_with_shutdown(payload)

        shutdown = summary["shutdown"]
        self.assertEqual(shutdown["active_current_policy"], "monitor_only")
        self.assertEqual(shutdown["host_current_per_axis_threshold_ma"], 400)
        self.assertEqual(
            shutdown["host_current_selected_total_threshold_ma"], 600
        )
        self.assertEqual(shutdown["active_current_over_limit_sample_count"], 3)
        self.assertEqual(shutdown["active_current_warning_event_count"], 1)
        self.assertEqual(
            shutdown["active_peak_abs_currents"],
            [166.0, 220.0, 99.0, 111.0, 64.0, 23.0],
        )
        self.assertEqual(shutdown["active_max_selected_total_current_ma"], 619.0)
        self.assertEqual(
            shutdown["verified_device_current_limits"], [1400.0] * 6
        )
        self.assertTrue(shutdown["safe_stop_verified"])
        self.assertEqual(shutdown["verification_status"], "safe")
        self.assertTrue(
            any(
                "monitor_only" in note and "thresholds were crossed" in note
                for note in summary["notes"]
            )
        )
        rendered = render_text(summary)
        self.assertIn("policy=monitor_only", rendered)
        self.assertIn("over_limit_samples/events=3/1", rendered)

    def test_legacy_safe_shutdown_without_current_policy_fields_still_parses(
        self,
    ) -> None:
        summary = self._summary_with_shutdown(self._safe_shutdown_payload())

        shutdown = summary["shutdown"]
        self.assertIsNone(shutdown["active_current_policy"])
        self.assertIsNone(shutdown["host_current_per_axis_threshold_ma"])
        self.assertIsNone(shutdown["active_current_over_limit_sample_count"])
        self.assertIsNone(shutdown["active_peak_abs_currents"])
        self.assertIsNone(shutdown["verified_device_current_limits"])
        self.assertTrue(shutdown["safe_stop_verified"])

    def test_malformed_shutdown_current_monitor_fields_are_rejected(self) -> None:
        invalid_fields = {
            "active_current_policy": "observe",
            "host_current_per_axis_threshold_ma": True,
            "host_current_selected_total_threshold_ma": -1,
            "active_current_over_limit_sample_count": 1.5,
            "active_current_warning_event_count": -1,
            "active_peak_abs_currents": [1, 2, 3, 4, 5],
            "active_max_selected_total_current_ma": -1,
            "verified_device_current_limits": [1400, 1400, 1400, 1400, 1400, -1],
        }
        for field, invalid in invalid_fields.items():
            payload = self._safe_shutdown_payload()
            payload[field] = invalid
            with self.subTest(field=field), self.assertRaisesRegex(
                LogFormatError, field
            ):
                self._summary_with_shutdown(payload)

    def test_non_idle_shutdown_feedback_cannot_be_marked_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            jsonl = run_dir / "mano_retarget.jsonl"
            row = hardware_row(
                1,
                1.0,
                state="active",
                accepted=True,
                masked=[-1, -1, -1, 850, -1, -1],
                sent=[-1, -1, -1, 890, -1, -1],
            )
            jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")
            (run_dir / "shutdown_status.json").write_text(
                json.dumps(
                    {
                        "session_id": "hardware-session",
                        "stop_confirmed": True,
                        "physical_stop_verified": True,
                        "shutdown_feedback": {
                            "angles": [1000, 1000, 1000, 1000, 1000, 990],
                            "positions": [100, 100, 100, 100, 100, 120],
                            "currents": [101, 101, 0, 0, 0, 0],
                            "errors": [0, 0, 0, 0, 0, 0],
                            "statuses": [2, 2, 2, 2, 2, 2],
                            "temperatures": [30, 30, 30, 30, 30, 30],
                        },
                        "final_angle_targets": [-1, -1, -1, -1, -1, -1],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = analyze_jsonl(jsonl)

        shutdown = summary["shutdown"]
        self.assertTrue(shutdown["physical_stop_verified"])
        self.assertFalse(shutdown["shutdown_feedback_idle"])
        self.assertFalse(shutdown["safe_stop_verified"])
        self.assertEqual(shutdown["verification_status"], "unsafe")

    def test_malformed_physical_stop_flag_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            jsonl = run_dir / "mano_retarget.jsonl"
            row = hardware_row(
                1,
                1.0,
                state="active",
                accepted=True,
                masked=[-1, -1, -1, 850, -1, -1],
                sent=[-1, -1, -1, 890, -1, -1],
            )
            jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")
            (run_dir / "shutdown_status.json").write_text(
                json.dumps(
                    {
                        "session_id": "hardware-session",
                        "stop_confirmed": True,
                        "physical_stop_verified": "true",
                        "final_angle_targets": [-1, -1, -1, -1, -1, -1],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                LogFormatError, "physical_stop_verified.*JSON boolean"
            ):
                analyze_jsonl(jsonl)

    def test_hardware_log_without_safe_shutdown_is_not_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            jsonl = run_dir / "mano_retarget.jsonl"
            row = hardware_row(
                1,
                1.0,
                state="active",
                accepted=True,
                masked=[-1, -1, -1, 850, -1, -1],
                sent=[-1, -1, -1, 890, -1, -1],
            )
            jsonl.write_text(json.dumps(row) + "\n", encoding="utf-8")
            (run_dir / "shutdown_status.json").write_text(
                json.dumps(
                    {
                        "session_id": "hardware-session",
                        "stop_confirmed": True,
                        "final_angle_targets": [-1, -1, -1, -1, -1, 0],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = analyze_jsonl(jsonl)

        self.assertFalse(summary["shutdown"]["final_all_minus_one"])
        self.assertFalse(summary["shutdown"]["safe_stop_verified"])
        self.assertEqual(summary["shutdown"]["verification_status"], "unsafe")
        self.assertTrue(any("does not prove" in note for note in summary["notes"]))

    def test_submit_acceptance_must_be_a_json_boolean(self) -> None:
        row = hardware_row(
            1,
            1.0,
            state="active",
            accepted="true",
            masked=[-1, -1, -1, 850, -1, -1],
            sent=[-1, -1, -1, 890, -1, -1],
        )
        with self.assertRaisesRegex(LogFormatError, "JSON boolean"):
            analyze_records([row])


if __name__ == "__main__":
    unittest.main()
