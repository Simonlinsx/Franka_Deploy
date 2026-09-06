from __future__ import annotations

import contextlib
import threading
import unittest
from pathlib import Path
from unittest import mock

from inspire_mano_pipeline.calibration import PipelineCalibration
from inspire_mano_pipeline.rh56_stream import RH56Error, SafeRH56Stream
from inspire_rh56_test import (
    REG_ANGLE_ACT,
    REG_ANGLE_SET,
    REG_CURRENT,
    REG_ERROR,
    REG_FORCE_SET,
    REG_POS_ACT,
    REG_SPEED_SET,
    REG_STATUS,
    REG_TEMP,
)


ROOT = Path(__file__).resolve().parents[3]
ROTATE_ONLY_PATH = ROOT / (
    "examples/inspire_mano_pipeline/"
    "inspire_rh56bfx_right_thumb_rotate_commissioning_900_800.json"
)
SIX_DOF_PATH = ROOT / (
    "examples/inspire_mano_pipeline/"
    "inspire_rh56bfx_right_six_dof_commissioning_900_800.json"
)


class _OffsetHand:
    def __init__(self, thumb_actual: int = 857) -> None:
        self.angle_targets = (-1,) * 6
        self.angles = (1000, 1000, 1000, 1000, 1000, thumb_actual)
        self.positions = (500,) * 6
        self.speeds = (1000,) * 6
        self.force_limits = (500,) * 6
        self.currents = (0,) * 6
        self.write_log: list[tuple[int, tuple[int, ...]]] = []
        self.thread_ids: set[int] = set()

    def snapshot(self):
        self.thread_ids.add(threading.get_ident())
        return {
            "angle_targets": self.angle_targets,
            "errors": (0,) * 6,
            "temperatures": (30,) * 6,
            "statuses": (2,) * 6,
            "angles": self.angles,
            "currents": self.currents,
            "speeds": self.speeds,
            "force_limits": self.force_limits,
        }

    def write_six_shorts(self, address, values, retries=0):
        del retries
        self.thread_ids.add(threading.get_ident())
        values = tuple(int(value) for value in values)
        self.write_log.append((address, values))
        if address == REG_ANGLE_SET:
            self.angle_targets = values
        elif address == REG_SPEED_SET:
            self.speeds = values
        elif address == REG_FORCE_SET:
            self.force_limits = values

    def read_six_shorts(self, address, retries=0):
        del retries
        self.thread_ids.add(threading.get_ident())
        if address == REG_ANGLE_ACT:
            return self.angles
        if address == REG_POS_ACT:
            return self.positions
        if address == REG_ANGLE_SET:
            return self.angle_targets
        if address == REG_CURRENT:
            return self.currents
        if address == REG_SPEED_SET:
            return self.speeds
        if address == REG_FORCE_SET:
            return self.force_limits
        raise AssertionError(f"unexpected six-short read: {address}")

    def read(self, address, length, retries=0):
        del retries
        self.thread_ids.add(threading.get_ident())
        if length != 6:
            raise AssertionError(f"unexpected byte-read length: {length}")
        if address in (REG_ERROR,):
            return bytes((0,) * 6)
        if address in (REG_STATUS,):
            return bytes((2,) * 6)
        if address in (REG_TEMP,):
            return bytes((30,) * 6)
        raise AssertionError(f"unexpected byte read: {address}")


class ThumbRotateStreamOffsetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rotate_only = PipelineCalibration.load(ROTATE_ONLY_PATH)

    def _stream(self) -> SafeRH56Stream:
        return SafeRH56Stream(
            self.rotate_only,
            selected_axes=("thumb_rotate",),
            hand_context_factory=lambda: None,
        )

    def test_feedback_seed_uses_plus_fifteen_without_clipping(self) -> None:
        stream = self._stream()
        self.assertEqual(
            stream._selected_commands_from_actual(
                (1000, 1000, 1000, 1000, 1000, 857)
            ),
            (-1, -1, -1, -1, -1, 872),
        )
        with self.assertRaisesRegex(RH56Error, "validated.*range"):
            stream._selected_commands_from_actual(
                (1000, 1000, 1000, 1000, 1000, 839)
            )
        with self.assertRaisesRegex(RH56Error, "validated.*range"):
            stream._selected_commands_from_actual(
                (1000, 1000, 1000, 1000, 1000, 871)
            )

    def test_worker_initial_seed_is_virtual_and_writes_no_motion_target(self) -> None:
        hand = _OffsetHand(thumb_actual=857)

        @contextlib.contextmanager
        def factory():
            yield hand

        stream = SafeRH56Stream(
            self.rotate_only,
            selected_axes=("thumb_rotate",),
            hand_context_factory=factory,
        )
        stream.start()
        stream.wait_until_ready(1.0)
        self.assertEqual(
            stream.initial_command_seed,
            (-1, -1, -1, -1, -1, 872),
        )
        stream.close()
        numeric_targets = [
            values
            for address, values in hand.write_log
            if address == REG_ANGLE_SET and values != (-1,) * 6
        ]
        self.assertEqual(numeric_targets, [])

    def test_normal_shutdown_holds_with_offset_then_disables_twice(self) -> None:
        hand = _OffsetHand(thumb_actual=857)
        stream = self._stream()
        with mock.patch("inspire_mano_pipeline.rh56_stream.time.sleep"):
            stream._disable(hand, stop_in_place=True)

        angle_writes = [
            values for address, values in hand.write_log if address == REG_ANGLE_SET
        ]
        self.assertEqual(angle_writes[0], (-1, -1, -1, -1, -1, 872))
        self.assertEqual(angle_writes[1:], [(-1,) * 6, (-1,) * 6])
        self.assertEqual(
            stream.shutdown_hold_targets,
            (-1, -1, -1, -1, -1, 872),
        )
        self.assertEqual(stream.final_angle_targets, (-1,) * 6)

    def test_out_of_evidence_range_and_fault_path_send_no_numeric_hold(self) -> None:
        for hand, stop_in_place in (
            (_OffsetHand(thumb_actual=871), True),
            (_OffsetHand(thumb_actual=857), False),
        ):
            with self.subTest(
                thumb_actual=hand.angles[5], stop_in_place=stop_in_place
            ):
                stream = self._stream()
                with mock.patch("inspire_mano_pipeline.rh56_stream.time.sleep"):
                    stream._disable(hand, stop_in_place=stop_in_place)
                numeric_targets = [
                    values
                    for address, values in hand.write_log
                    if address == REG_ANGLE_SET and values != (-1,) * 6
                ]
                self.assertEqual(numeric_targets, [])
                self.assertEqual(stream.final_angle_targets, (-1,) * 6)

    def test_offset_does_not_leak_to_unselected_thumb_axis(self) -> None:
        calibration = PipelineCalibration.load(SIX_DOF_PATH)
        stream = SafeRH56Stream(
            calibration,
            selected_axes=("index",),
            hand_context_factory=lambda: None,
        )
        self.assertEqual(
            stream._selected_commands_from_actual(
                (900, 900, 900, 900, 900, 857)
            ),
            (-1, -1, -1, 900, -1, -1),
        )


if __name__ == "__main__":
    unittest.main()
