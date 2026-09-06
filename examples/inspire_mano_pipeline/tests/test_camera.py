from __future__ import annotations

import unittest

import numpy as np

from inspire_mano_pipeline.camera import (
    PALM_KEYPOINT_INDICES,
    PalmDepthSpatialEstimate,
    PalmDepthStabilizer,
    estimate_palm_depth,
    robust_palm_depth,
)


class PalmDepthTests(unittest.TestCase):
    @staticmethod
    def _keypoints() -> np.ndarray:
        points = np.zeros((21, 2), dtype=np.float32)
        points[[0, 5, 9, 13, 17]] = np.asarray(
            [(60, 95), (20, 30), (45, 22), (75, 22), (100, 30)],
            dtype=np.float32,
        )
        return points

    @classmethod
    def _anchor_pixels(cls) -> list[tuple[int, int]]:
        palm = cls._keypoints()[np.asarray(PALM_KEYPOINT_INDICES)]
        center = np.mean(palm, axis=0)
        anchors = palm + 0.35 * (center - palm)
        anchors = np.concatenate((anchors, center[None, :]), axis=0)
        return [
            (int(round(float(x))), int(round(float(y)))) for x, y in anchors
        ]

    @staticmethod
    def _paint_patch(
        depth: np.ndarray,
        center: tuple[int, int],
        value: float,
        radius: int = 2,
    ) -> None:
        x, y = center
        depth[y - radius : y + radius + 1, x - radius : x + radius + 1] = value

    def test_compact_palm_surface_returns_depth(self) -> None:
        depth = np.full((120, 120), 0.72, dtype=np.float32)
        self.assertAlmostEqual(
            robust_palm_depth(depth, self._keypoints(), radius=2),
            0.72,
            places=5,
        )

    def test_dominant_consensus_ignores_one_background_roi(self) -> None:
        depth = np.full((120, 120), 0.72, dtype=np.float32)
        self._paint_patch(depth, self._anchor_pixels()[0], 1.05)

        estimate = estimate_palm_depth(depth, self._keypoints(), radius=2)

        self.assertEqual(estimate.reason, "measured_consensus")
        self.assertEqual(estimate.roi_sample_count, 6)
        self.assertEqual(estimate.inlier_count, 5)
        self.assertAlmostEqual(estimate.depth_m, 0.72, places=5)

    def test_ambiguous_foreground_background_split_is_rejected(self) -> None:
        depth = np.full((120, 120), 0.72, dtype=np.float32)
        anchors = self._anchor_pixels()
        for index in (0, 1, 4):
            self._paint_patch(depth, anchors[index], 1.05)

        estimate = estimate_palm_depth(depth, self._keypoints(), radius=2)

        self.assertIsNone(estimate.depth_m)
        self.assertEqual(estimate.reason, "inconsistent_palm_depth_rois")
        self.assertEqual(estimate.roi_sample_count, 6)
        self.assertEqual(estimate.inlier_count, 3)

    def test_too_few_valid_palm_samples_is_rejected(self) -> None:
        depth = np.zeros((120, 120), dtype=np.float32)
        for anchor in self._anchor_pixels()[:2]:
            self._paint_patch(depth, anchor, 0.72)

        estimate = estimate_palm_depth(depth, self._keypoints(), radius=2)

        self.assertIsNone(estimate.depth_m)
        self.assertEqual(estimate.reason, "insufficient_valid_palm_rois")
        self.assertEqual(estimate.roi_sample_count, 2)

    def test_palm_center_at_image_edge_is_rejected(self) -> None:
        keypoints = self._keypoints()
        keypoints[np.asarray(PALM_KEYPOINT_INDICES), 0] += 55.0
        depth = np.full((120, 120), 0.72, dtype=np.float32)

        estimate = estimate_palm_depth(depth, keypoints, radius=2)

        self.assertIsNone(estimate.depth_m)
        self.assertEqual(estimate.reason, "palm_outside_safe_roi")


class PalmDepthStabilizerTests(unittest.TestCase):
    @staticmethod
    def _measured(depth_m: float = 0.42) -> PalmDepthSpatialEstimate:
        return PalmDepthSpatialEstimate(
            depth_m=depth_m,
            reason="measured_consensus",
            roi_sample_count=6,
            inlier_count=6,
            valid_pixel_count=150,
            radius_px=6,
            palm_center_xy=(50.0, 40.0),
            palm_width_px=40.0,
        )

    @staticmethod
    def _missing() -> PalmDepthSpatialEstimate:
        return PalmDepthSpatialEstimate(
            depth_m=None,
            reason="insufficient_valid_palm_rois",
            roi_sample_count=2,
            inlier_count=0,
            valid_pixel_count=35,
            radius_px=6,
            palm_center_xy=(50.0, 40.0),
            palm_width_px=40.0,
        )

    def test_measured_then_held_then_expired_without_refreshing_evidence(self) -> None:
        stabilizer = PalmDepthStabilizer(max_hold_seconds=0.12)
        bbox = np.asarray([20.0, 10.0, 80.0, 90.0], dtype=np.float32)

        measured = stabilizer.apply(self._measured(), bbox, 10.0)
        held = stabilizer.apply(self._missing(), bbox, 10.05)
        expired = stabilizer.apply(self._missing(), bbox, 10.13)

        self.assertEqual(measured.source, "measured")
        self.assertAlmostEqual(measured.depth_m, 0.42)
        self.assertAlmostEqual(measured.raw_depth_m, 0.42)
        self.assertEqual(measured.evidence_at_monotonic, 10.0)
        self.assertEqual(measured.age_seconds, 0.0)

        self.assertEqual(held.source, "held")
        self.assertAlmostEqual(held.depth_m, 0.42)
        self.assertIsNone(held.raw_depth_m)
        self.assertEqual(held.evidence_at_monotonic, 10.0)
        self.assertAlmostEqual(held.age_seconds, 0.05)

        self.assertEqual(expired.source, "missing")
        self.assertIsNone(expired.depth_m)
        self.assertIsNone(expired.evidence_at_monotonic)
        self.assertIsNone(expired.age_seconds)

    def test_different_track_is_not_held(self) -> None:
        stabilizer = PalmDepthStabilizer(max_hold_seconds=0.12)
        first_bbox = np.asarray([10.0, 10.0, 50.0, 80.0], dtype=np.float32)
        other_bbox = np.asarray([70.0, 10.0, 110.0, 80.0], dtype=np.float32)
        stabilizer.apply(self._measured(), first_bbox, 20.0)

        observation = stabilizer.apply(self._missing(), other_bbox, 20.05)

        self.assertEqual(observation.source, "missing")
        self.assertIsNone(observation.depth_m)
        self.assertIsNone(observation.evidence_at_monotonic)


if __name__ == "__main__":
    unittest.main()
