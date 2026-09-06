from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

import numpy as np

from inspire_mano_pipeline.model import CameraFrame
from realsense_mano_inspire import write_record


class FrameTimingRecordTests(unittest.TestCase):
    def test_rgbd_sensor_and_host_timing_are_recorded(self) -> None:
        frame = CameraFrame(
            color_bgr=np.zeros((4, 4, 3), dtype=np.uint8),
            depth_m=np.ones((4, 4), dtype=np.float32),
            captured_at_monotonic=10.0,
            frame_number=42,
            depth_frame_number=41,
            color_sensor_timestamp_ms=1234.5,
            depth_sensor_timestamp_ms=1234.0,
            sensor_timestamp_domain="hardware_clock",
            ready_at_monotonic=10.012,
        )
        handle = io.StringIO()
        with patch("realsense_mano_inspire.time.monotonic", return_value=10.050):
            write_record(
                handle,
                frame,
                detection=None,
                output=None,
                inference_seconds=0.02,
                session_id="timing-test",
            )

        row = json.loads(handle.getvalue())
        self.assertEqual(row["frame_number"], 42)
        self.assertEqual(row["depth_frame_number"], 41)
        self.assertEqual(row["color_sensor_timestamp_ms"], 1234.5)
        self.assertEqual(row["depth_sensor_timestamp_ms"], 1234.0)
        self.assertEqual(row["sensor_timestamp_domain"], "hardware_clock")
        self.assertAlmostEqual(row["frame_copy_align_seconds"], 0.012)
        self.assertAlmostEqual(row["pipeline_age_seconds"], 0.050)


if __name__ == "__main__":
    unittest.main()
