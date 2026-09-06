from unittest.mock import patch

import numpy as np
import pytest

from dynamic_pcd.pointcloud.extractor import ObjectPointCloudExtractor
from dynamic_pcd.types import CameraIntrinsics, RGBDFrame
from dynamic_pcd.utils.fps import DeadlineRateGate, FPSMeter, PacketRateMonitor


def test_deadline_rate_gate_preserves_fractional_frame_time():
    """A near-30 Hz input must not be accidentally divided down to 15 Hz."""

    gate = DeadlineRateGate(30.0)
    arrivals = np.arange(301, dtype=np.float64) * 0.0332
    admitted = sum(gate.ready(float(now)) for now in arrivals)

    elapsed = float(arrivals[-1] - arrivals[0])
    measured_hz = (admitted - 1) / elapsed
    assert admitted >= 299
    assert 29.5 <= measured_hz <= 30.1


def test_deadline_rate_gate_caps_fast_input_without_bursting_after_stall():
    gate = DeadlineRateGate(20.0)
    arrivals = np.arange(501, dtype=np.float64) * 0.01
    admitted = sum(gate.ready(float(now)) for now in arrivals)

    assert 100 <= admitted <= 101
    assert gate.ready(10.0)
    # Missing several deadlines admits only one unit, never a catch-up burst.
    assert not gate.ready(10.0001)


@pytest.mark.parametrize("rate_hz", [0.0, -1.0, np.nan, np.inf])
def test_deadline_rate_gate_rejects_invalid_rate(rate_hz):
    with pytest.raises(ValueError, match="rate_hz"):
        DeadlineRateGate(rate_hz)


def test_fps_meter_interval_mode_reports_end_to_end_cadence():
    meter = FPSMeter(window=4)
    samples = iter([1.0, 1.04, 1.08, 1.12])
    with patch("dynamic_pcd.utils.fps.time.perf_counter", side_effect=samples):
        meter.tick()
        meter.tick()
        meter.tick()
        meter.tick()

    assert meter.avg_fps == pytest.approx(25.0)
    assert meter.avg_ms == pytest.approx(40.0)


def test_packet_rate_monitor_reports_valid_pcd_rate_not_total_packet_rate():
    clock_value = [0.0]
    monitor = PacketRateMonitor(
        window_s=2.0,
        lost_timeout_s=0.25,
        clock=lambda: clock_value[0],
    )
    for index in range(61):
        clock_value[0] = index / 30.0
        monitor.tick(valid=(index % 2 == 0), now_s=clock_value[0])

    assert monitor.publish_hz == pytest.approx(30.0, rel=0.02)
    assert monitor.valid_hz == pytest.approx(15.0, rel=0.05)
    # The newest event above is valid, but effective valid PCD is only 15 Hz.
    assert monitor.status(20.0, warmup_events=30) == "LOW"


def test_packet_rate_monitor_fails_closed_on_current_or_sustained_loss():
    clock_value = [0.0]
    monitor = PacketRateMonitor(
        window_s=2.0,
        lost_timeout_s=0.20,
        clock=lambda: clock_value[0],
    )
    for index in range(35):
        clock_value[0] = index / 30.0
        monitor.tick(valid=True, now_s=clock_value[0])
    assert monitor.status(20.0, warmup_events=30) == "PASS"

    clock_value[0] += 1.0 / 30.0
    monitor.tick(valid=False, now_s=clock_value[0])
    assert monitor.status(20.0, warmup_events=30) == "LOST"

    clock_value[0] += 0.25
    assert monitor.status(20.0, warmup_events=30) == "NO_VALID"

    never_valid = PacketRateMonitor(clock=lambda: 0.0)
    never_valid.tick(False, now_s=0.0)
    assert never_valid.status(20.0, warmup_events=1, now_s=0.0) == "NO_VALID"


def test_object_extraction_does_not_allocate_full_frame_index_grids():
    height, width = 480, 848
    frame = RGBDFrame(
        color_bgr=np.zeros((height, width, 3), dtype=np.uint8),
        depth_raw=np.full((height, width), 700, dtype=np.uint16),
        depth_scale=0.001,
        intrinsics=CameraIntrinsics(
            width=width,
            height=height,
            fx=610.0,
            fy=610.0,
            ppx=width / 2.0,
            ppy=height / 2.0,
        ),
        timestamp=1.0,
        frame_id=1,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[200:280, 380:468] = 1
    extractor = ObjectPointCloudExtractor(
        {
            "num_points": 1024,
            "use_rgb": False,
            "center_policy_points": True,
            "erode_kernel": 0,
            "stride": 1,
            "voxel_size": 0.0,
            "remove_outliers": False,
        }
    )

    # np.indices was the former multi-megabyte per-frame allocation.  Sparse
    # np.nonzero coordinates should now be sufficient for an object mask.
    class _NumpyWithoutIndices:
        def __getattr__(self, name):
            if name == "indices":
                raise AssertionError("full-frame index allocation is forbidden")
            return getattr(np, name)

    with patch(
        "dynamic_pcd.pointcloud.extractor.np", new=_NumpyWithoutIndices()
    ):
        result = extractor.extract(frame, mask)

    assert result.valid
    assert result.points.shape == (80 * 88, 3)
    assert result.policy_points.shape == (1024, 3)
