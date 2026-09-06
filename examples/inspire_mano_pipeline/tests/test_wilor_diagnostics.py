from __future__ import annotations

import unittest
from types import SimpleNamespace

from inspire_mano_pipeline.wilor_backend import WiLoRBackend
from realsense_mano_inspire import tracking_display_status


def prediction(detector_is_right: bool, bbox=(0.0, 0.0, 10.0, 10.0)) -> dict:
    return {
        "is_right": 1.0 if detector_is_right else 0.0,
        "hand_bbox": list(bbox),
        "wilor_preds": {},
    }


def backend(
    *, expected: str = "right", strict: bool = True, operator_roi=None
) -> WiLoRBackend:
    result = object.__new__(WiLoRBackend)
    result.handedness = "right"
    result.detector_handedness = expected
    result.strict_single_hand = strict
    result.operator_roi = operator_roi
    return result


class WiLoRDiagnosticsTests(unittest.TestCase):
    def test_no_candidate_and_wrong_hand_are_distinct(self) -> None:
        model = backend()
        self.assertEqual(
            model._diagnose_predictions([])["result"], "no_candidates"
        )
        diagnostics = model._diagnose_predictions([prediction(False)])
        self.assertEqual(diagnostics["result"], "handedness_mismatch")
        self.assertEqual(diagnostics["detector_labels"], ["left"])
        self.assertEqual(diagnostics["expected_detector_label"], "right")

    def test_multiple_hands_are_reported_before_acceptance(self) -> None:
        model = backend(strict=True)
        diagnostics = model._diagnose_predictions(
            [prediction(False), prediction(True)]
        )
        self.assertEqual(diagnostics["candidate_count"], 2)
        self.assertEqual(diagnostics["matching_candidate_count"], 1)
        self.assertEqual(diagnostics["result"], "multiple_hands")

    def test_matching_single_candidate_is_selected(self) -> None:
        diagnostics = backend()._diagnose_predictions([prediction(True)])
        self.assertEqual(diagnostics["result"], "candidate_selected")

    def test_operator_roi_excludes_robot_hand_candidate(self) -> None:
        model = backend(operator_roi=(0.35, 0.10, 0.98, 0.98))
        robot = prediction(True, (5.0, 40.0, 25.0, 80.0))
        operator = prediction(True, (45.0, 15.0, 85.0, 90.0))
        diagnostics = model._diagnose_predictions(
            [robot, operator], image_shape=(100, 100)
        )
        self.assertEqual(diagnostics["candidate_count"], 2)
        self.assertEqual(diagnostics["roi_candidate_count"], 1)
        self.assertEqual(diagnostics["result"], "candidate_selected")
        self.assertIs(
            model._select_prediction([robot, operator], (100, 100)), operator
        )

    def test_near_duplicate_box_is_suppressed_but_separate_hand_is_not(self) -> None:
        model = backend()
        first = prediction(True, (10.0, 10.0, 60.0, 80.0))
        duplicate = prediction(True, (11.0, 11.0, 59.0, 79.0))
        separate = prediction(True, (65.0, 10.0, 95.0, 70.0))
        duplicate_diagnostics = model._diagnose_predictions([first, duplicate])
        self.assertEqual(duplicate_diagnostics["distinct_candidate_count"], 1)
        self.assertEqual(duplicate_diagnostics["duplicate_candidate_count"], 1)
        self.assertEqual(duplicate_diagnostics["result"], "candidate_selected")
        separate_diagnostics = model._diagnose_predictions([first, separate])
        self.assertEqual(separate_diagnostics["distinct_candidate_count"], 2)
        self.assertEqual(separate_diagnostics["result"], "multiple_hands")

    def test_operator_status_distinguishes_detection_and_depth_failures(self) -> None:
        calibration = SimpleNamespace(
            min_palm_depth_m=0.15,
            max_palm_depth_m=1.50,
        )
        self.assertEqual(
            tracking_display_status(
                None, {"result": "no_candidates"}, calibration
            ),
            "NO HAND",
        )
        self.assertEqual(
            tracking_display_status(
                None, {"result": "handedness_mismatch"}, calibration
            ),
            "WRONG HAND / RIGHT REQUIRED",
        )
        self.assertEqual(
            tracking_display_status(
                None, {"result": "multiple_hands"}, calibration
            ),
            "MULTIPLE HANDS",
        )
        self.assertEqual(
            tracking_display_status(
                SimpleNamespace(palm_depth_m=0.40), {}, calibration
            ),
            "TRACKING OK",
        )
        self.assertIn(
            "TOO FAR",
            tracking_display_status(
                SimpleNamespace(palm_depth_m=1.80), {}, calibration
            ),
        )
        self.assertEqual(
            tracking_display_status(
                SimpleNamespace(palm_depth_m=None), {}, calibration
            ),
            "DEPTH INVALID",
        )


if __name__ == "__main__":
    unittest.main()
