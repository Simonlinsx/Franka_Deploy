import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

import dynamic_pcd.camera.realsense_camera as realsense_module
from dynamic_pcd.camera.realsense_camera import (
    RealSenseCamera,
    RetryableCameraTimeout,
    _CaptureTimestampFutureSkewError,
    _CaptureTimestampStaleError,
    _StartupReadinessGate,
    _validated_epoch_capture_timestamp,
)


class _FakeFrame:
    def __init__(
        self,
        *,
        number,
        timestamp_s,
        sensor_timestamp_us,
        domain="timestamp_domain.global_time",
        metadata_overrides=None,
    ):
        self.number = number
        self.timestamp_s = timestamp_s
        self.sensor_timestamp_us = sensor_timestamp_us
        self.domain = domain
        self.metadata = {
            "frame_timestamp": sensor_timestamp_us,
            "sensor_timestamp": (
                None
                if sensor_timestamp_us is None
                else sensor_timestamp_us + 10.0
            ),
            "backend_timestamp": timestamp_s * 1000.0 + 1.0,
            "time_of_arrival": timestamp_s * 1000.0 + 2.0,
        }
        if metadata_overrides:
            self.metadata.update(metadata_overrides)
        self.profile = _FakeVideoProfile()

    def get_frame_number(self):
        return self.number

    def get_timestamp(self):
        return self.timestamp_s * 1000.0

    def get_frame_timestamp_domain(self):
        return self.domain

    def supports_frame_metadata(self, metadata):
        return self.metadata.get(metadata) is not None

    def get_frame_metadata(self, metadata):
        return self.metadata[metadata]

    def get_data(self):
        return np.zeros((2, 3), dtype=np.uint16)


class _FakeVideoProfile:
    def as_video_stream_profile(self):
        return self

    @staticmethod
    def get_intrinsics():
        return SimpleNamespace(
            width=3,
            height=2,
            fx=100.0,
            fy=101.0,
            ppx=1.0,
            ppy=0.5,
            model="none",
            coeffs=[0.0] * 5,
        )


class _FakeFrameset:
    def __init__(self, color, depth):
        self.color = color
        self.depth = depth

    def as_frameset(self):
        return self

    def get_color_frame(self):
        return self.color

    def get_depth_frame(self):
        return self.depth


class _FakeFrameQueue:
    def __init__(self, framesets):
        self.framesets = list(framesets)
        self.wait_timeouts_ms = []

    def wait_for_frame(self, timeout_ms):
        self.wait_timeouts_ms.append(timeout_ms)
        if not self.framesets:
            raise RuntimeError("no scripted frame")
        return self.framesets.pop(0)


class _FakeDrainableFrameQueue(_FakeFrameQueue):
    def __init__(self, framesets):
        super().__init__(framesets)
        self.poll_calls = 0

    def poll_for_frame(self):
        self.poll_calls += 1
        if not self.framesets:
            return None
        return self.framesets.pop(0)


class _FakeRS:
    class frame_metadata_value:
        frame_timestamp = "frame_timestamp"
        sensor_timestamp = "sensor_timestamp"
        backend_timestamp = "backend_timestamp"
        time_of_arrival = "time_of_arrival"


class _PassthroughAlign:
    @staticmethod
    def process(frames):
        return frames


class _FakeHostClock:
    def __init__(self, *, monotonic_step_s=0.01, realtime_offset_s=1000.0):
        self.monotonic_s = 0.0
        self.monotonic_step_s = monotonic_step_s
        self.realtime_offset_s = realtime_offset_s

    def monotonic(self):
        self.monotonic_s += self.monotonic_step_s
        return self.monotonic_s

    def realtime(self):
        return self.realtime_offset_s + self.monotonic_s


class _FakeConfig:
    def enable_device(self, _serial):
        pass

    def enable_stream(self, *_args):
        pass


class _FakeDepthSensor:
    def get_depth_scale(self):
        return 0.001

    def supports(self, _option):
        return False


class _FakeDevice:
    def __init__(self):
        self.depth_sensor = _FakeDepthSensor()

    def first_depth_sensor(self):
        return self.depth_sensor

    def get_info(self, info):
        if info == "serial_number":
            return "fake-serial"
        return "fake-camera"


class _FakeProfile:
    def __init__(self):
        self.device = _FakeDevice()

    def get_device(self):
        return self.device


class _FakePipeline:
    def __init__(self):
        self.profile = _FakeProfile()
        self.started_with = None
        self.stop_calls = 0

    def start(self, config, queue):
        self.started_with = (config, queue)
        return self.profile

    def stop(self):
        self.stop_calls += 1


def _fake_frameset(
    color_number,
    depth_number,
    capture_s,
    *,
    sensor_skew_us=0.0,
    domain="timestamp_domain.global_time",
):
    sensor_time_us = 500_000.0 + 33_333.0 * color_number
    return _FakeFrameset(
        _FakeFrame(
            number=color_number,
            timestamp_s=capture_s,
            sensor_timestamp_us=sensor_time_us,
            domain=domain,
        ),
        _FakeFrame(
            number=depth_number,
            timestamp_s=capture_s,
            sensor_timestamp_us=sensor_time_us + sensor_skew_us,
            domain=domain,
        ),
    )


def _startup_settings(camera, **overrides):
    settings = camera._validated_startup_settings()
    settings.update(overrides)
    return settings


def _runtime_camera(framesets, **cfg):
    camera = RealSenseCamera(
        {"max_capture_to_retrieval_s": 0.10, **cfg}
    )
    camera.started = True
    camera.pipeline = object()
    camera.frame_queue = _FakeFrameQueue(framesets)
    camera.align = _PassthroughAlign()
    camera.rs = _FakeRS()
    return camera


def test_global_timestamp_is_used_as_epoch_capture_time():
    capture_s, skew_s, epoch_skew_s = _validated_epoch_capture_timestamp(
        color_timestamp_ms=1_784_707_812_157.474,
        depth_timestamp_ms=1_784_707_812_157.456,
        color_domain="timestamp_domain.global_time",
        depth_domain="timestamp_domain.global_time",
        retrieved_at_s=1_784_707_812.183529,
    )
    assert capture_s == pytest.approx(1_784_707_812.157474)
    # Epoch-scale float64 subtraction has sub-microsecond quantization.
    assert skew_s == pytest.approx(0.000018, abs=5.0e-7)
    assert epoch_skew_s == pytest.approx(0.000018, abs=5.0e-7)


def test_sensor_clock_checks_rgbd_pair_when_global_mappings_differ():
    _capture_s, sensor_skew_s, epoch_skew_s = (
        _validated_epoch_capture_timestamp(
            color_timestamp_ms=1_784_707_812_157.474,
            depth_timestamp_ms=1_784_707_812_151.626,
            color_domain="timestamp_domain.global_time",
            depth_domain="timestamp_domain.global_time",
            retrieved_at_s=1_784_707_812.183529,
            color_sensor_timestamp_us=838_992_225.0,
            depth_sensor_timestamp_us=838_992_207.0,
        )
    )
    assert sensor_skew_s == pytest.approx(0.000018)
    assert epoch_skew_s == pytest.approx(0.005848, abs=5.0e-7)


@pytest.mark.parametrize(
    ("color_domain", "depth_domain"),
    [
        ("timestamp_domain.hardware_clock", "timestamp_domain.hardware_clock"),
        ("timestamp_domain.global_time", "timestamp_domain.hardware_clock"),
    ],
)
def test_non_epoch_or_mixed_timestamp_domains_are_rejected(
    color_domain, depth_domain
):
    with pytest.raises(RuntimeError, match="global/system-time"):
        _validated_epoch_capture_timestamp(
            color_timestamp_ms=1000.0,
            depth_timestamp_ms=1000.0,
            color_domain=color_domain,
            depth_domain=depth_domain,
            retrieved_at_s=1.01,
        )


def test_rgbd_skew_and_transport_age_fail_closed():
    _capture_s, accepted_skew_s, _epoch_skew_s = (
        _validated_epoch_capture_timestamp(
            color_timestamp_ms=1000.0,
            depth_timestamp_ms=1000.0,
            color_domain="timestamp_domain.system_time",
            depth_domain="timestamp_domain.system_time",
            retrieved_at_s=1.01,
            color_sensor_timestamp_us=500_000.0,
            depth_sensor_timestamp_us=505_476.0,
        )
    )
    assert accepted_skew_s == pytest.approx(0.005476)
    with pytest.raises(RuntimeError, match="color/depth capture skew"):
        _validated_epoch_capture_timestamp(
            color_timestamp_ms=1000.0,
            depth_timestamp_ms=990.0,
            color_domain="timestamp_domain.system_time",
            depth_domain="timestamp_domain.system_time",
            retrieved_at_s=1.01,
        )
    with pytest.raises(_CaptureTimestampStaleError, match="already stale"):
        _validated_epoch_capture_timestamp(
            color_timestamp_ms=1000.0,
            depth_timestamp_ms=1000.0,
            color_domain="timestamp_domain.system_time",
            depth_domain="timestamp_domain.system_time",
            retrieved_at_s=2.01,
        )


def test_runtime_discards_transport_stale_frame_then_returns_fresh_frame():
    now_s = time.time()
    camera = _runtime_camera(
        [
            _fake_frameset(11, 111, now_s - 0.25),
            _fake_frameset(12, 112, now_s),
        ]
    )

    frame = camera.get_frame(timeout_ms=250)

    assert frame.sensor_frame_number == 12
    assert frame.depth_sensor_frame_number == 112
    assert frame.rejected_transport_stale_frames == 1
    assert frame.last_rejected_transport_age_s > 0.10
    assert camera.rejected_transport_stale_frames == 1
    assert len(camera.frame_queue.wait_timeouts_ms) == 2
    diagnostics = camera.recent_frame_dequeue_diagnostics()
    assert [item["outcome"] for item in diagnostics] == [
        "rejected_transport_stale",
        "accepted",
    ]
    accepted = diagnostics[-1]
    assert accepted["color_frame_number"] == 12
    assert accepted["depth_frame_number"] == 112
    assert accepted["color_frame_delta"] == 1
    assert accepted["depth_frame_delta"] == 1
    assert accepted["stream_progress"] == "both_advanced"
    assert accepted["color_depth_sensor_timestamp_skew_s"] == pytest.approx(
        0.0
    )
    assert accepted["color_depth_epoch_timestamp_skew_s"] == pytest.approx(
        0.0
    )
    assert accepted["color_timestamp_ms"] == pytest.approx(now_s * 1000.0)
    assert accepted["color_frame_timestamp_metadata"] is not None
    assert accepted["color_sensor_timestamp_metadata"] is not None
    assert accepted["color_backend_timestamp_metadata"] is not None
    assert accepted["color_time_of_arrival_metadata"] is not None
    assert accepted["host_clock_pair_span_s"] >= 0.0
    assert frame.capture_diagnostic["outcome"] == "accepted"


def test_runtime_drains_pending_queue_to_freshest_candidate_before_validation():
    now_s = time.time()
    camera = _runtime_camera([])
    camera.frame_queue = _FakeDrainableFrameQueue(
        [
            _fake_frameset(11, 111, now_s - 0.40),
            _fake_frameset(12, 112, now_s - 0.20),
            _fake_frameset(13, 113, now_s),
        ]
    )

    frame = camera.get_frame(timeout_ms=250)

    assert frame.sensor_frame_number == 13
    assert frame.depth_sensor_frame_number == 113
    assert camera.rejected_transport_stale_frames == 0
    assert camera.drained_pending_framesets == 2
    assert camera.last_drained_pending_framesets == 2
    assert camera.frame_queue.wait_timeouts_ms
    assert camera.frame_queue.poll_calls == 3
    assert frame.capture_diagnostic[
        "drained_pending_framesets_before_candidate"
    ] == 2
    assert frame.capture_diagnostic["drained_pending_framesets_total"] == 2
    assert frame.capture_diagnostic["outcome"] == "accepted"


def test_latest_sink_drain_never_falls_back_from_skewed_new_pair():
    now_s = time.time()
    camera = _runtime_camera([])
    camera.frame_queue = _FakeDrainableFrameQueue(
        [
            _fake_frameset(11, 111, now_s),
            _fake_frameset(
                12,
                112,
                now_s,
                sensor_skew_us=9_000.0,
            ),
        ]
    )

    with pytest.raises(
        RetryableCameraTimeout, match="runtime frame wait failed"
    ):
        camera.get_frame(timeout_ms=250)

    assert camera.frame_id == 0
    assert camera.drained_pending_framesets == 1
    assert camera.rejected_timestamp_skew_frames == 1
    diagnostics = camera.recent_frame_dequeue_diagnostics()
    assert len(diagnostics) == 1
    assert diagnostics[0]["color_frame_number"] == 12
    assert diagnostics[0]["depth_frame_number"] == 112
    assert diagnostics[0]["outcome"] == "rejected_rgbd_skew"


def test_runtime_rejects_duplicate_stream_then_strictly_resynchronizes():
    now_s = time.time()
    camera = _runtime_camera(
        [
            _fake_frameset(10, 110, now_s),
            # The SDK may emit a composite while one stream is recovering.
            # Even with valid timestamps, a repeated color frame is not a new
            # RGB-D observation and must not advance the acceptance anchor.
            _fake_frameset(10, 111, now_s),
            _fake_frameset(11, 111, now_s),
        ]
    )

    first = camera.get_frame(timeout_ms=250)
    second = camera.get_frame(timeout_ms=250)

    assert first.sensor_frame_number == 10
    assert first.depth_sensor_frame_number == 110
    assert second.sensor_frame_number == 11
    assert second.depth_sensor_frame_number == 111
    assert camera.rejected_non_increasing_frames == 1
    assert camera.last_accepted_sensor_frame_number == 11
    assert camera.last_accepted_depth_sensor_frame_number == 111
    assert "color=10 after 10" in camera.last_rejected_frame_sequence_reason
    diagnostics = camera.recent_frame_dequeue_diagnostics()
    assert [item["outcome"] for item in diagnostics] == [
        "accepted",
        "rejected_non_increasing_sequence",
        "accepted",
    ]
    assert diagnostics[1]["stream_progress"] == "depth_only"
    assert diagnostics[2]["rejected_non_increasing_frames_total"] == 1
    tail = camera._diagnostic_tail_text()
    assert "recent_pair_progress=" in tail
    assert "2:10/111:d0/1:depth_only:rejected_non_increasing_sequence" in tail


def test_runtime_stalled_depth_skew_streak_is_exposed_not_repaired():
    now_s = time.time()
    camera = _runtime_camera(
        [
            _fake_frameset(138, 137, now_s),
            _fake_frameset(
                139, 137, now_s, sensor_skew_us=-33_333.0
            ),
            _fake_frameset(
                140, 137, now_s, sensor_skew_us=-66_666.0
            ),
        ]
    )

    accepted = camera.get_frame(timeout_ms=250)
    assert accepted.sensor_frame_number == 138
    assert accepted.depth_sensor_frame_number == 137

    with pytest.raises(
        RetryableCameraTimeout, match="runtime frame wait failed"
    ) as exc_info:
        camera.get_frame(timeout_ms=250)

    message = str(exc_info.value)
    assert "rgbd_skew_rejected_this_call=2" in message
    assert "last_accepted_sensor_pair=138/137" in message
    assert "color_only:rejected_rgbd_skew" in message
    assert camera.frame_id == 1
    assert camera.rejected_timestamp_skew_frames == 2
    assert camera.last_accepted_sensor_frame_number == 138
    assert camera.last_accepted_depth_sensor_frame_number == 137
    diagnostics = camera.recent_frame_dequeue_diagnostics()
    assert [item["outcome"] for item in diagnostics] == [
        "accepted",
        "rejected_rgbd_skew",
        "rejected_rgbd_skew",
    ]
    assert diagnostics[-1]["stream_progress"] == "color_only"
    assert diagnostics[-1]["color_depth_sensor_timestamp_skew_s"] == (
        pytest.approx(0.066666)
    )


def test_runtime_only_stale_frames_exhaust_original_deadline(monkeypatch):
    clock = _FakeHostClock(monotonic_step_s=0.01, realtime_offset_s=1000.0)
    monkeypatch.setattr(
        realsense_module,
        "time",
        SimpleNamespace(monotonic=clock.monotonic, time=clock.realtime),
    )
    camera = _runtime_camera(
        [
            _fake_frameset(1, 101, 900.0),
            _fake_frameset(2, 102, 900.1),
        ]
    )

    with pytest.raises(
        RetryableCameraTimeout, match="no transport-fresh"
    ) as exc_info:
        camera.get_frame(timeout_ms=80)

    message = str(exc_info.value)
    assert "fixed 0.080s deadline" in message
    assert "without changing the configured freshness limit" in message
    assert "outcome=rejected_transport_stale" in message
    assert camera.rejected_transport_stale_frames == 2
    assert len(camera.frame_queue.wait_timeouts_ms) == 2
    assert all(
        item["outcome"] == "rejected_transport_stale"
        for item in camera.recent_frame_dequeue_diagnostics()
    )


@pytest.mark.parametrize(
    ("capture_offset_s", "domain", "expected"),
    [
        (1.0, "timestamp_domain.global_time", "unexpectedly in the future"),
        (0.0, "timestamp_domain.hardware_clock", "global/system-time"),
    ],
)
def test_runtime_future_and_domain_errors_are_fatal_without_retry(
    capture_offset_s, domain, expected
):
    now_s = time.time()
    camera = _runtime_camera(
        [
            _fake_frameset(
                21,
                121,
                now_s + capture_offset_s,
                domain=domain,
            ),
            _fake_frameset(22, 122, now_s),
        ]
    )

    with pytest.raises(RuntimeError, match="not retryable") as exc_info:
        camera.get_frame(timeout_ms=250)

    assert expected in str(exc_info.value)
    assert not bool(
        getattr(exc_info.value, "retryable_camera_timeout", False)
    )
    assert len(camera.frame_queue.wait_timeouts_ms) == 1
    assert camera.rejected_transport_stale_frames == 0
    diagnostics = camera.recent_frame_dequeue_diagnostics()
    assert len(diagnostics) == 1
    assert diagnostics[0]["outcome"] == "fatal_timestamp_validation"


def test_runtime_discards_modest_future_skew_then_accepts_newer_normal_frame():
    now_s = time.time()
    camera = _runtime_camera(
        [
            _fake_frameset(21, 121, now_s + 0.087),
            _fake_frameset(22, 122, now_s),
        ]
    )

    frame = camera.get_frame(timeout_ms=250)

    assert frame.sensor_frame_number == 22
    assert frame.depth_sensor_frame_number == 122
    assert camera.rejected_future_skew_frames == 1
    assert camera.last_rejected_future_skew_s == pytest.approx(0.087, abs=0.01)
    assert len(camera.frame_queue.wait_timeouts_ms) == 2
    diagnostics = camera.recent_frame_dequeue_diagnostics()
    assert [item["outcome"] for item in diagnostics] == [
        "rejected_future_skew",
        "accepted",
    ]
    assert diagnostics[0]["transport_age_s"] < -0.05
    assert frame.timestamp == pytest.approx(now_s, abs=0.01)


def test_modest_future_skew_is_never_returned_or_clamped():
    with pytest.raises(
        _CaptureTimestampFutureSkewError,
        match="retryable future skew",
    ) as exc_info:
        _validated_epoch_capture_timestamp(
            color_timestamp_ms=1087.0,
            depth_timestamp_ms=1087.0,
            color_domain="timestamp_domain.system_time",
            depth_domain="timestamp_domain.system_time",
            retrieved_at_s=1.0,
            maximum_future_skew_s=0.05,
            maximum_retryable_future_skew_s=0.10,
        )
    assert exc_info.value.future_skew_s == pytest.approx(0.087)


def test_runtime_diagnostic_ring_is_fixed_length():
    now_s = time.time()
    camera = _runtime_camera(
        [_fake_frameset(n, n + 100, now_s) for n in range(1, 21)]
    )

    for _ in range(20):
        camera.get_frame(timeout_ms=250)

    diagnostics = camera.recent_frame_dequeue_diagnostics()
    assert camera.frame_dequeue_count == 20
    assert len(diagnostics) == 16
    assert diagnostics[0]["dequeue_index"] == 5
    assert diagnostics[-1]["dequeue_index"] == 20


def test_startup_gate_requires_one_consecutive_strictly_increasing_run():
    gate = _StartupReadinessGate(
        required_consecutive_frames=3,
        maximum_clock_offset_jitter_s=0.005,
    )
    assert not gate.observe_valid(
        sensor_frame_number=10,
        depth_sensor_frame_number=100,
        retrieved_at_s=1000.0,
        retrieved_monotonic_s=10.0,
    )
    assert gate.consecutive_valid_frames == 1
    assert not gate.observe_valid(
        sensor_frame_number=11,
        depth_sensor_frame_number=101,
        retrieved_at_s=1000.01,
        retrieved_monotonic_s=10.01,
    )
    assert gate.consecutive_valid_frames == 2

    # A repeated color frame is not part of either adjacent run. It clears
    # the count to zero; the following frame starts again at one.
    assert not gate.observe_valid(
        sensor_frame_number=11,
        depth_sensor_frame_number=102,
        retrieved_at_s=1000.02,
        retrieved_monotonic_s=10.02,
    )
    assert gate.consecutive_valid_frames == 0
    assert "strictly increase" in gate.last_rejection_reason
    assert not gate.observe_valid(
        sensor_frame_number=12,
        depth_sensor_frame_number=103,
        retrieved_at_s=1000.03,
        retrieved_monotonic_s=10.03,
    )
    assert gate.consecutive_valid_frames == 1
    assert not gate.observe_valid(
        sensor_frame_number=14,
        depth_sensor_frame_number=105,
        retrieved_at_s=1000.04,
        retrieved_monotonic_s=10.04,
    )
    assert gate.skipped_sensor_frames == 1
    assert gate.observe_valid(
        sensor_frame_number=15,
        depth_sensor_frame_number=106,
        retrieved_at_s=1000.05,
        retrieved_monotonic_s=10.05,
    )
    assert gate.ready
    assert gate.consecutive_valid_frames == 3


def test_startup_gate_clock_jump_clears_count_and_jump_frame_is_not_counted():
    gate = _StartupReadinessGate(
        required_consecutive_frames=2,
        maximum_clock_offset_jitter_s=0.005,
    )
    assert not gate.observe_valid(
        sensor_frame_number=1,
        depth_sensor_frame_number=20,
        retrieved_at_s=1010.0,
        retrieved_monotonic_s=10.0,
    )
    assert not gate.observe_valid(
        sensor_frame_number=2,
        depth_sensor_frame_number=21,
        retrieved_at_s=1010.02,
        retrieved_monotonic_s=10.01,
    )
    assert gate.consecutive_valid_frames == 0
    assert "offset jumped" in gate.last_rejection_reason

    # The new offset must remain stable for a complete two-frame run.
    assert not gate.observe_valid(
        sensor_frame_number=3,
        depth_sensor_frame_number=22,
        retrieved_at_s=1010.03,
        retrieved_monotonic_s=10.02,
    )
    assert gate.observe_valid(
        sensor_frame_number=4,
        depth_sensor_frame_number=23,
        retrieved_at_s=1010.04,
        retrieved_monotonic_s=10.03,
    )


def test_bad_startup_frame_clears_the_entire_candidate_run():
    gate = _StartupReadinessGate(
        required_consecutive_frames=3,
        maximum_clock_offset_jitter_s=0.005,
    )
    for number in (1, 2):
        assert not gate.observe_valid(
            sensor_frame_number=number,
            depth_sensor_frame_number=number + 100,
            retrieved_at_s=1000.0 + number * 0.01,
            retrieved_monotonic_s=10.0 + number * 0.01,
        )
    gate.reject(
        "stale frame",
        sensor_frame_number=3,
        depth_sensor_frame_number=103,
        retrieved_at_s=1000.03,
        retrieved_monotonic_s=10.03,
    )
    assert gate.consecutive_valid_frames == 0
    assert gate.best_consecutive_valid_frames == 2
    assert gate.rejected_frames == 1


def test_startup_wait_restarts_after_stale_frame_then_passes():
    camera = RealSenseCamera(
        {
            "startup_required_consecutive_frames": 3,
            "startup_timeout_s": 0.5,
            "startup_max_clock_offset_jitter_s": 0.005,
            "max_capture_to_retrieval_s": 0.10,
        }
    )
    camera.rs = _FakeRS()
    # Fake retrieval realtime values are 1000.03, .05, .07, .09 and .11.
    # Frame 2 is stale and therefore splits the two candidate runs.
    camera.frame_queue = _FakeFrameQueue(
        [
            _fake_frameset(1, 101, 1000.02),
            _fake_frameset(2, 102, 999.00),
            _fake_frameset(3, 103, 1000.06),
            _fake_frameset(4, 104, 1000.08),
            _fake_frameset(5, 105, 1000.10),
        ]
    )
    clock = _FakeHostClock()
    camera._wait_for_startup_readiness(
        _startup_settings(camera),
        monotonic_fn=clock.monotonic,
        realtime_fn=clock.realtime,
    )
    assert camera.startup_frames_observed == 5
    assert camera.startup_rejected_frames == 1
    assert camera.startup_consecutive_valid_frames == 3
    assert camera.startup_best_consecutive_valid_frames == 3
    assert camera.last_sensor_frame_number == 5
    assert camera.last_depth_sensor_frame_number == 105
    assert camera.last_host_realtime_minus_monotonic_s == pytest.approx(1000.0)


def test_startup_wait_skewed_pair_resets_run_and_is_counted():
    camera = RealSenseCamera(
        {
            "startup_required_consecutive_frames": 2,
            "startup_timeout_s": 0.5,
            "startup_max_clock_offset_jitter_s": 0.005,
            "max_capture_to_retrieval_s": 0.10,
        }
    )
    camera.rs = _FakeRS()
    camera.frame_queue = _FakeFrameQueue(
        [
            _fake_frameset(1, 101, 1000.02),
            _fake_frameset(2, 102, 1000.04, sensor_skew_us=9000.0),
            _fake_frameset(3, 103, 1000.06),
            _fake_frameset(4, 104, 1000.08),
        ]
    )
    clock = _FakeHostClock()
    camera._wait_for_startup_readiness(
        _startup_settings(camera),
        monotonic_fn=clock.monotonic,
        realtime_fn=clock.realtime,
    )
    assert camera.startup_frames_observed == 4
    assert camera.startup_rejected_frames == 1
    assert camera.rejected_timestamp_skew_frames == 1
    assert camera.last_rejected_color_depth_skew_s == pytest.approx(0.009)


def test_startup_wait_repeated_bad_frames_hits_total_timeout_fail_closed():
    camera = RealSenseCamera(
        {
            "startup_required_consecutive_frames": 3,
            "startup_timeout_s": 0.11,
            "startup_max_clock_offset_jitter_s": 0.005,
            "max_capture_to_retrieval_s": 0.10,
        }
    )
    camera.rs = _FakeRS()
    camera.frame_queue = _FakeFrameQueue(
        [_fake_frameset(n, n + 100, 900.0) for n in range(1, 10)]
    )
    clock = _FakeHostClock(monotonic_step_s=0.02)
    with pytest.raises(
        RuntimeError, match="startup readiness gate timed out"
    ) as exc_info:
        camera._wait_for_startup_readiness(
            _startup_settings(camera),
            monotonic_fn=clock.monotonic,
            realtime_fn=clock.realtime,
        )
    assert camera.startup_consecutive_valid_frames == 0
    assert camera.startup_rejected_frames >= 1
    assert camera.startup_best_consecutive_valid_frames == 0
    assert "last rejection" in str(exc_info.value)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("startup_required_consecutive_frames", 1, ">= 2"),
        ("startup_required_consecutive_frames", 2.5, "must be an integer"),
        ("startup_timeout_s", 0.0, "startup_timeout_s"),
        ("startup_max_clock_offset_jitter_s", -0.1, "clock_offset"),
        ("max_capture_to_retrieval_s", 0.0, "timestamp limits"),
    ],
)
def test_invalid_startup_gate_configuration_is_rejected_before_capture(
    key, value, message
):
    with pytest.raises(ValueError, match=message):
        RealSenseCamera({key: value})._validated_startup_settings()


def test_startup_defaults_keep_capacity_one_age_budget_contract():
    settings = RealSenseCamera({})._validated_startup_settings()
    assert settings["required_consecutive_frames"] == 30
    assert settings["startup_timeout_s"] == pytest.approx(5.0)
    assert settings["maximum_transport_age_s"] == pytest.approx(0.10)


class _ManualDepthSensor:
    def __init__(self):
        self.values = {
            "emitter": 1.0,
            "laser": 150.0,
            "preset": 0.0,
            "auto": 1.0,
            "exposure": 8500.0,
            "gain": 16.0,
        }

    def supports(self, option):
        return option in self.values

    def set_option(self, option, value):
        self.values[option] = float(value)
        if option == "preset":
            # D435 visual presets reset advanced controls.  The production
            # configuration must therefore write laser power afterwards.
            self.values["laser"] = 150.0

    def get_option(self, option):
        return self.values[option]

    def get_option_range(self, option):
        ranges = {
            "laser": (0.0, 360.0, 30.0, 150.0),
            "exposure": (1.0, 200000.0, 1.0, 8500.0),
            "gain": (16.0, 248.0, 1.0, 16.0),
        }
        minimum, maximum, step, default = ranges[option]
        return SimpleNamespace(
            min=minimum, max=maximum, step=step, default=default
        )


def _manual_depth_camera(config):
    camera = RealSenseCamera.__new__(RealSenseCamera)
    camera.cfg = dict(config)
    camera.rs = SimpleNamespace(
        option=SimpleNamespace(
            emitter_enabled="emitter",
            laser_power="laser",
            visual_preset="preset",
            enable_auto_exposure="auto",
            exposure="exposure",
            gain="gain",
        )
    )
    camera.depth_sensor = _ManualDepthSensor()
    return camera


def test_manual_depth_exposure_contract_is_applied_exactly():
    camera = _manual_depth_camera(
        {
            "depth_auto_exposure": False,
            "depth_exposure": 2000,
            "depth_gain": 32,
            "laser_power": 360,
        }
    )
    camera._configure_depth_sensor()
    assert camera.depth_sensor.values["auto"] == 0.0
    assert camera.depth_sensor.values["exposure"] == 2000.0
    assert camera.depth_sensor.values["gain"] == 32.0
    assert camera.depth_sensor.values["laser"] == 360.0


def test_manual_depth_values_cannot_be_combined_with_auto_exposure():
    camera = _manual_depth_camera(
        {"depth_auto_exposure": True, "depth_exposure": 2000}
    )
    with pytest.raises(ValueError, match="incompatible"):
        camera._configure_depth_sensor()


def test_manual_depth_exposure_out_of_device_range_fails_closed():
    camera = _manual_depth_camera(
        {"depth_auto_exposure": False, "depth_exposure": 300000}
    )
    with pytest.raises(ValueError, match="outside"):
        camera._configure_depth_sensor()


def test_start_failure_stops_pipeline_and_keeps_capacity_one_queue(monkeypatch):
    pipeline = _FakePipeline()
    queue_calls = []

    def make_queue(capacity, keep_frames):
        queue_calls.append((capacity, keep_frames))
        return _FakeFrameQueue([])

    fake_rs = SimpleNamespace(
        pipeline=lambda: pipeline,
        config=_FakeConfig,
        frame_queue=make_queue,
        stream=SimpleNamespace(depth="depth", color="color"),
        format=SimpleNamespace(z16="z16", bgr8="bgr8"),
        camera_info=SimpleNamespace(
            serial_number="serial_number", name="name"
        ),
        option=SimpleNamespace(
            emitter_enabled="emitter_enabled",
            laser_power="laser_power",
            visual_preset="visual_preset",
            enable_auto_exposure="enable_auto_exposure",
        ),
        align=lambda _stream: object(),
        spatial_filter=lambda: object(),
        temporal_filter=lambda: object(),
        hole_filling_filter=lambda: object(),
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)
    camera = RealSenseCamera(
        {
            "startup_required_consecutive_frames": 2,
            "startup_timeout_s": 0.5,
        }
    )

    def fail_gate(_settings):
        raise RuntimeError("scripted readiness failure")

    monkeypatch.setattr(camera, "_wait_for_startup_readiness", fail_gate)
    with pytest.raises(RuntimeError, match="scripted readiness failure"):
        camera.start()

    assert queue_calls == [(1, False)]
    assert pipeline.stop_calls == 1
    assert not camera.started


def test_runtime_frame_wait_failure_has_fail_closed_context():
    class TimeoutQueue:
        @staticmethod
        def wait_for_frame(_timeout_ms):
            raise RuntimeError("Frame did not arrive in time!")

    camera = RealSenseCamera({})
    camera.started = True
    camera.pipeline = object()
    camera.frame_queue = TimeoutQueue()
    camera.align = object()
    camera.frame_id = 7
    camera.last_sensor_frame_number = 41
    camera.last_retrieved_monotonic_s = time.monotonic()

    with pytest.raises(
        RetryableCameraTimeout, match="runtime frame wait failed"
    ) as exc_info:
        camera.get_frame(timeout_ms=25)

    message = str(exc_info.value)
    assert exc_info.value.retryable_camera_timeout is True
    assert "last_frame_id=7" in message
    assert "last_sensor_frame=41" in message
    assert "timeouts=1" in message
    assert "Frame did not arrive in time" in message
    assert camera.runtime_frame_wait_timeouts == 1
    assert camera.last_frame_wait_failed_monotonic_s is not None


def test_runtime_missing_rgbd_stream_is_fatal_not_retryable():
    now_s = time.time()
    color = _fake_frameset(1, 101, now_s).get_color_frame()
    camera = _runtime_camera([_FakeFrameset(color=color, depth=None)])

    with pytest.raises(
        RuntimeError, match="failed to get synchronized depth/color frame"
    ) as exc_info:
        camera.get_frame(timeout_ms=100)

    assert not isinstance(exc_info.value, RetryableCameraTimeout)
    assert not bool(
        getattr(exc_info.value, "retryable_camera_timeout", False)
    )
    assert camera.recent_frame_dequeue_diagnostics()[-1]["outcome"] == (
        "missing_rgbd_stream"
    )
