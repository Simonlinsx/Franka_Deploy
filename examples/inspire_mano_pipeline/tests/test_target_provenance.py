from __future__ import annotations

import contextlib
import time
import unittest
from dataclasses import replace
from pathlib import Path

from inspire_mano_pipeline.calibration import PipelineCalibration
from inspire_mano_pipeline.rh56_stream import (
    REG_ANGLE_ACT,
    REG_ANGLE_SET,
    REG_CURRENT,
    REG_ERROR,
    REG_FORCE_SET,
    REG_POS_ACT,
    REG_SPEED_SET,
    REG_STATUS,
    REG_TEMP,
    SafeRH56Stream,
    StreamState,
    TargetFrame,
    TrackingTimeout,
)


ROOT = Path(__file__).resolve().parents[3]
CALIBRATION_PATH = ROOT / (
    "examples/inspire_mano_pipeline/inspire_rh56bfx_right_commissioning.json"
)


class _FakeHand:
    def __init__(self) -> None:
        self.angle_targets = (-1,) * 6
        self.angles = (800,) * 6
        self.positions = (500,) * 6
        self.speeds = (1000,) * 6
        self.force_limits = (500,) * 6
        self.write_log: list[tuple[int, tuple[int, ...]]] = []

    def snapshot(self):
        return {
            "angle_targets": self.angle_targets,
            "errors": (0,) * 6,
            "temperatures": (25,) * 6,
            "statuses": (2,) * 6,
            "angles": self.angles,
            "currents": (0,) * 6,
            "speeds": self.speeds,
            "force_limits": self.force_limits,
        }

    def write_six_shorts(self, address, values, retries=0):
        del retries
        values = tuple(int(value) for value in values)
        self.write_log.append((address, values))
        if address == REG_ANGLE_SET:
            self.angle_targets = values
            self.angles = tuple(
                old if target == -1 else target
                for old, target in zip(self.angles, values)
            )
        elif address == REG_SPEED_SET:
            self.speeds = values
        elif address == REG_FORCE_SET:
            self.force_limits = values
        else:  # pragma: no cover - catches future unexpected stream writes.
            raise AssertionError(f"unexpected write address: {address}")

    def read(self, address, length, retries=0):
        del retries
        if length != 6:
            raise AssertionError(f"unexpected read length: {length}")
        if address == REG_ERROR:
            return bytes((0,) * 6)
        if address == REG_STATUS:
            return bytes((2,) * 6)
        if address == REG_TEMP:
            return bytes((25,) * 6)
        raise AssertionError(f"unexpected read address: {address}")

    def read_six_shorts(self, address, retries=0):
        del retries
        if address == REG_ANGLE_ACT:
            return self.angles
        if address == REG_POS_ACT:
            return self.positions
        if address == REG_ANGLE_SET:
            return self.angle_targets
        if address == REG_SPEED_SET:
            return self.speeds
        if address == REG_FORCE_SET:
            return self.force_limits
        if address == REG_CURRENT:
            return (0,) * 6
        raise AssertionError(f"unexpected six-short address: {address}")


class TargetProvenanceTests(unittest.TestCase):
    def _calibration(self, tracking_timeout_seconds=0.40):
        return replace(
            PipelineCalibration.load(CALIBRATION_PATH),
            valid_frames_to_arm=1,
            tracking_timeout_seconds=tracking_timeout_seconds,
            arming_max_frame_gap_seconds=0.10,
            preflight_stability_seconds=0.01,
            control_hz=100.0,
            feedback_hz=25.0,
        )

    def _start_stream(self, tracking_timeout_seconds=0.40):
        hand = _FakeHand()

        @contextlib.contextmanager
        def factory():
            yield hand

        stream = SafeRH56Stream(
            self._calibration(tracking_timeout_seconds),
            selected_axes=("index",),
            hand_context_factory=factory,
        )
        stream.start()
        stream.wait_until_ready(1.0)
        return stream, hand

    @staticmethod
    def _frame(
        frame_number,
        target,
        source="mano",
        captured_at=None,
        *,
        depth_source="measured",
        depth_evidence_at=None,
    ):
        return TargetFrame(
            targets=(-1, -1, -1, target, -1, -1),
            captured_at_monotonic=(
                time.monotonic() if captured_at is None else captured_at
            ),
            frame_number=frame_number,
            source=source,
            depth_source=depth_source,
            depth_evidence_at_monotonic=depth_evidence_at,
        )

    @staticmethod
    def _wait_until(predicate, timeout=1.0):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.002)
        return predicate()

    def test_zero_mano_fallback_cannot_arm_or_write_motion(self):
        stream, hand = self._start_stream()
        try:
            for frame_number in range(1, 4):
                self.assertFalse(
                    stream.submit(
                        self._frame(frame_number, 900, source="fallback_open")
                    )
                )
                time.sleep(0.01)

            self.assertEqual(stream.state, StreamState.WAITING_FOR_TRACKING)
            self.assertFalse(stream.ever_active)
            self.assertEqual(stream.accepted_target_count, 0)
            self.assertEqual(stream.accepted_mano_target_count, 0)
            self.assertEqual(stream.accepted_fallback_open_target_count, 0)
            self.assertEqual(stream.motion_write_count, 0)
            self.assertFalse(
                any(
                    address == REG_ANGLE_SET and values != (-1,) * 6
                    for address, values in hand.write_log
                )
            )
        finally:
            stream.close()

    def test_mano_arms_then_fallback_moves_toward_calibrated_open(self):
        stream, _ = self._start_stream()
        try:
            self.assertTrue(stream.submit(self._frame(1, 750)))
            self.assertTrue(
                self._wait_until(
                    lambda: stream.state == StreamState.ACTIVE
                    and stream.last_sent_targets is not None
                    and stream.last_sent_targets[3] < 800
                )
            )
            before_open = stream.last_sent_targets[3]

            self.assertTrue(
                stream.submit(self._frame(2, 900, source="fallback_open"))
            )
            self.assertTrue(
                self._wait_until(
                    lambda: stream.last_sent_targets is not None
                    and stream.last_sent_targets[3] > before_open
                )
            )
            self.assertLess(stream.last_sent_targets[3], 900)
            self.assertEqual(stream.accepted_target_count, 2)
            self.assertEqual(stream.accepted_mano_target_count, 1)
            self.assertEqual(stream.accepted_fallback_open_target_count, 1)
        finally:
            stream.close()

    def test_stale_fallback_is_rejected_and_does_not_refresh_watchdog(self):
        timeout = 0.12
        stream, _ = self._start_stream(tracking_timeout_seconds=timeout)
        try:
            self.assertTrue(stream.submit(self._frame(1, 780)))
            self.assertTrue(
                self._wait_until(lambda: stream.state == StreamState.ACTIVE)
            )
            stale_capture = time.monotonic() - timeout - 0.05
            self.assertFalse(
                stream.submit(
                    self._frame(
                        2,
                        900,
                        source="fallback_open",
                        captured_at=stale_capture,
                    )
                )
            )
            self.assertEqual(stream.accepted_fallback_open_target_count, 0)
            self.assertTrue(
                self._wait_until(
                    lambda: stream.state
                    in (StreamState.FAULT_LATCHED, StreamState.STOP_UNCONFIRMED),
                    timeout=1.0,
                )
            )
            self.assertEqual(stream.state, StreamState.FAULT_LATCHED)
            self.assertIsInstance(stream.error, TrackingTimeout)
        finally:
            stream.close()

    def test_held_depth_cannot_arm_hardware(self):
        stream, hand = self._start_stream(tracking_timeout_seconds=0.20)
        try:
            now = time.monotonic()
            self.assertTrue(
                stream.submit(
                    self._frame(
                        1,
                        780,
                        captured_at=now,
                        depth_source="held",
                        depth_evidence_at=now - 0.05,
                    )
                )
            )
            time.sleep(0.05)
            self.assertEqual(stream.state, StreamState.WAITING_FOR_TRACKING)
            self.assertFalse(stream.ever_active)
            self.assertEqual(stream.motion_write_count, 0)
            self.assertFalse(
                any(
                    address == REG_ANGLE_SET and values != (-1,) * 6
                    for address, values in hand.write_log
                )
            )
        finally:
            stream.close()

    def test_held_depth_cannot_invent_a_fresh_evidence_timestamp(self):
        stream, _ = self._start_stream(tracking_timeout_seconds=0.20)
        try:
            with self.assertRaisesRegex(ValueError, "original measured-depth"):
                stream.submit(
                    self._frame(
                        1,
                        780,
                        depth_source="held",
                        depth_evidence_at=None,
                    )
                )
            with self.assertRaisesRegex(ValueError, "measured or held"):
                stream.submit(
                    self._frame(
                        2,
                        780,
                        depth_source="not_required",
                    )
                )
        finally:
            stream.close()

    def test_held_depth_does_not_refresh_active_watchdog(self):
        timeout = 0.12
        stream, _ = self._start_stream(tracking_timeout_seconds=timeout)
        try:
            evidence_at = time.monotonic()
            self.assertTrue(
                stream.submit(
                    self._frame(
                        1,
                        780,
                        captured_at=evidence_at,
                        depth_source="measured",
                        depth_evidence_at=evidence_at,
                    )
                )
            )
            self.assertTrue(
                self._wait_until(lambda: stream.state == StreamState.ACTIVE)
            )
            frame_number = 2
            while time.monotonic() - evidence_at < timeout * 0.75:
                self.assertTrue(
                    stream.submit(
                        self._frame(
                            frame_number,
                            760,
                            captured_at=time.monotonic(),
                            depth_source="held",
                            depth_evidence_at=evidence_at,
                        )
                    )
                )
                frame_number += 1
                time.sleep(0.015)
            self.assertTrue(
                self._wait_until(
                    lambda: stream.state
                    in (StreamState.FAULT_LATCHED, StreamState.STOP_UNCONFIRMED),
                    timeout=1.0,
                )
            )
            self.assertEqual(stream.state, StreamState.FAULT_LATCHED)
            self.assertIsInstance(stream.error, TrackingTimeout)
        finally:
            stream.close()

    def test_unknown_source_is_rejected(self):
        stream = SafeRH56Stream(
            self._calibration(),
            selected_axes=("index",),
            hand_context_factory=lambda: None,
        )
        with self.assertRaisesRegex(ValueError, "source"):
            stream.submit(self._frame(1, 800, source="unknown"))


if __name__ == "__main__":
    unittest.main()
