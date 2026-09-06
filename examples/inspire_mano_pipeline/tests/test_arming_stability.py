from __future__ import annotations

import contextlib
import json
import tempfile
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
        else:  # pragma: no cover - catches unexpected stream IO in future changes.
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


class ArmingTargetStabilityTests(unittest.TestCase):
    def _calibration(self):
        return replace(
            PipelineCalibration.load(CALIBRATION_PATH),
            valid_frames_to_arm=3,
            arming_max_target_delta_units=10,
            tracking_timeout_seconds=0.50,
            arming_max_frame_gap_seconds=0.20,
            preflight_stability_seconds=0.05,
            control_hz=100.0,
            feedback_hz=20.0,
        )

    def _start_stream(self):
        hand = _FakeHand()

        @contextlib.contextmanager
        def factory():
            yield hand

        stream = SafeRH56Stream(
            self._calibration(),
            selected_axes=("index",),
            hand_context_factory=factory,
        )
        stream.start()
        stream.wait_until_ready(1.0)
        return stream

    @staticmethod
    def _submit_index(stream, frame_number: int, target: int) -> None:
        accepted = stream.submit(
            TargetFrame(
                targets=(-1, -1, -1, target, -1, -1),
                captured_at_monotonic=time.monotonic(),
                frame_number=frame_number,
            )
        )
        if not accepted:
            raise AssertionError(f"frame {frame_number} was unexpectedly rejected")
        # The serial owner must observe each generation; publishing all frames in
        # a burst would intentionally collapse them into one latest-target slot.
        time.sleep(0.03)

    @staticmethod
    def _wait_for_state(stream, expected, timeout=0.50):
        deadline = time.monotonic() + timeout
        while stream.state != expected and time.monotonic() < deadline:
            time.sleep(0.005)
        return stream.state

    def test_stable_selected_axis_targets_arm(self) -> None:
        stream = self._start_stream()
        try:
            self._submit_index(stream, 1, 800)
            self._submit_index(stream, 2, 805)
            self._submit_index(stream, 3, 810)
            self.assertEqual(
                self._wait_for_state(stream, StreamState.ACTIVE),
                StreamState.ACTIVE,
            )
        finally:
            stream.close()

    def test_repeated_large_target_jumps_never_arm(self) -> None:
        stream = self._start_stream()
        try:
            for frame_number, target in enumerate((800, 850, 800, 850, 800), start=1):
                self._submit_index(stream, frame_number, target)
            self.assertEqual(stream.state, StreamState.WAITING_FOR_TRACKING)
            self.assertEqual(stream.motion_write_count, 0)
        finally:
            stream.close()

    def test_calibration_rejects_out_of_range_target_delta(self) -> None:
        data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        data["safety"]["arming_max_target_delta_units"] = 1001
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "arming_max_target_delta_units"):
                PipelineCalibration.load(path)

    def test_calibration_rejects_non_integer_target_delta(self) -> None:
        data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        data["safety"]["arming_max_target_delta_units"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON integer"):
                PipelineCalibration.load(path)


if __name__ == "__main__":
    unittest.main()
