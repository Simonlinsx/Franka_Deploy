from pathlib import Path
import unittest

import numpy as np

from inspire_mano_pipeline.calibration import PipelineCalibration
from inspire_mano_pipeline.retargeting import (
    EndpointHysteresisFilter,
    TemporalQposFilter,
)
from realsense_mano_inspire import is_safe_commissioning_profile


ROOT = Path(__file__).resolve().parents[3]
FULL_RANGE_PATH = ROOT / (
    "examples/inspire_mano_pipeline/"
    "inspire_rh56bfx_right_index_fullrange_open1000.json"
)
REALTIME_PATH = ROOT / (
    "examples/inspire_mano_pipeline/"
    "inspire_rh56bfx_right_index_fullrange_realtime.json"
)


class IndexFullRangeOpen1000Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.calibration = PipelineCalibration.load(FULL_RANGE_PATH)

    def test_full_human_curl_maps_to_full_rh56_range(self) -> None:
        index = self.calibration.axes["index"]
        self.assertEqual((index.q_open, index.q_closed), (0.05, 0.65))
        self.assertEqual((index.command_open, index.command_closed), (1000, 0))
        self.assertEqual(index.map_qpos(0.0), 1000)
        self.assertEqual(index.map_qpos(0.05), 1000)
        self.assertEqual(index.map_qpos(0.35), 500)
        self.assertEqual(index.map_qpos(0.65), 0)
        self.assertEqual(index.map_qpos(1.47), 0)

    def test_full_range_keeps_low_energy_limits(self) -> None:
        index = self.calibration.axes["index"]
        self.assertEqual(index.max_rate_units_per_second, 80.0)
        self.assertEqual(self.calibration.speed, 80)
        self.assertEqual(self.calibration.force_limit, 80)
        self.assertEqual(self.calibration.stream_max_current_ma, 600)

    def test_full_range_is_not_tokenless_commissioning(self) -> None:
        self.assertFalse(
            is_safe_commissioning_profile(self.calibration, ("index",))
        )

    def test_realtime_profile_uses_measured_curl_endpoint(self) -> None:
        calibration = PipelineCalibration.load(REALTIME_PATH)
        index = calibration.axes["index"]
        self.assertEqual((index.q_open, index.q_closed), (0.08, 0.40))
        self.assertEqual(index.map_qpos(0.08), 1000)
        self.assertEqual(index.map_qpos(0.24), 500)
        self.assertEqual(index.map_qpos(0.40), 0)
        self.assertEqual(calibration.speed, 200)
        self.assertEqual(index.max_rate_units_per_second, 200.0)
        self.assertEqual(calibration.force_limit, 80)
        self.assertEqual(calibration.temporal_median_window, 3)
        self.assertEqual(calibration.temporal_ema_alpha, 0.5)
        self.assertEqual(calibration.temporal_filter_axes, ("index",))

    def test_temporal_filter_rejects_one_frame_spike_and_resets(self) -> None:
        temporal = TemporalQposFilter(median_window=3, ema_alpha=0.5)
        opened = np.zeros(6, dtype=np.float32)
        closed = np.ones(6, dtype=np.float32)
        np.testing.assert_allclose(temporal.apply(opened), opened)
        np.testing.assert_allclose(temporal.apply(closed), opened)
        np.testing.assert_allclose(temporal.apply(closed), 0.5 * closed)
        temporal.reset()
        np.testing.assert_allclose(temporal.apply(closed), closed)

    def test_index_only_temporal_filter_leaves_other_axes_bit_identical(self) -> None:
        temporal = TemporalQposFilter(
            median_window=3, ema_alpha=0.5, active_indices=(3,)
        )
        first = np.asarray([0.1, 0.2, 0.3, 0.0, 0.5, 0.6], dtype=np.float32)
        second = np.asarray([1.1, 1.2, 1.3, 1.0, 1.5, 1.6], dtype=np.float32)
        temporal.apply(first)
        filtered = temporal.apply(second)
        np.testing.assert_array_equal(filtered[[0, 1, 2, 4, 5]], second[[0, 1, 2, 4, 5]])
        self.assertEqual(float(filtered[3]), 0.0)

    def test_endpoint_hysteresis_latches_open_and_closed(self) -> None:
        calibration = PipelineCalibration.load(REALTIME_PATH)
        endpoint = EndpointHysteresisFilter(calibration)

        def apply_index(value: float) -> float:
            qpos = np.zeros(6, dtype=np.float32)
            qpos[3] = value
            return float(endpoint.apply(qpos)[3])

        self.assertAlmostEqual(apply_index(0.05), 0.08)
        self.assertAlmostEqual(apply_index(0.10), 0.08)
        self.assertAlmostEqual(apply_index(0.13), 0.13)
        self.assertAlmostEqual(apply_index(0.40), 0.40)
        self.assertAlmostEqual(apply_index(0.35), 0.40)
        self.assertAlmostEqual(apply_index(0.31), 0.31)


if __name__ == "__main__":
    unittest.main()
