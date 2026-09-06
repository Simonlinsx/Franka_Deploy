from __future__ import annotations

import sys
import unittest
from unittest import mock

import inspire_rh56_test as rh56


class _FakeHand:
    """In-memory angle-mode hand; it never constructs a serial transport."""

    def __init__(
        self,
        *,
        thumb_angle: int = 885,
        bend_angles=(1000, 1000, 1000, 1000, 1000),
        angle_targets=(-1, -1, -1, -1, -1, -1),
        actual_after_target: dict[int, int] | None = None,
    ) -> None:
        self.angles = [*map(int, bend_angles), int(thumb_angle)]
        self.angle_targets = list(map(int, angle_targets))
        self.position_targets = [500] * 6
        self.positions = [500] * 6
        self.forces = [0] * 6
        self.currents = [0] * 6
        self.speeds = [1000] * 6
        self.force_limits = [259] * 6
        self.statuses = [2] * 6
        self.errors = [0] * 6
        self.temperatures = [30] * 6
        self.actual_after_target = dict(actual_after_target or {})
        self.write_log: list[tuple[int, int]] = []
        self.batch_write_log: list[tuple[int, tuple[int, ...]]] = []

    def snapshot(self):
        return {
            "angle_targets": tuple(self.angle_targets),
            "position_targets": tuple(self.position_targets),
            "positions": tuple(self.positions),
            "forces": tuple(self.forces),
            "currents": tuple(self.currents),
            "angles": tuple(self.angles),
            "statuses": tuple(self.statuses),
            "errors": tuple(self.errors),
            "temperatures": tuple(self.temperatures),
            "speeds": tuple(self.speeds),
            "force_limits": tuple(self.force_limits),
        }

    @staticmethod
    def _axis(address: int, base: int) -> int:
        offset = address - base
        if offset < 0 or offset % 2 or offset >= 2 * len(rh56.JOINTS):
            raise AssertionError(f"unexpected register address: {address}")
        return offset // 2

    def read_short(self, address: int) -> int:
        if rh56.REG_ANGLE_ACT <= address < rh56.REG_ANGLE_ACT + 12:
            return self.angles[self._axis(address, rh56.REG_ANGLE_ACT)]
        if rh56.REG_ANGLE_SET <= address < rh56.REG_ANGLE_SET + 12:
            return self.angle_targets[self._axis(address, rh56.REG_ANGLE_SET)]
        if rh56.REG_SPEED_SET <= address < rh56.REG_SPEED_SET + 12:
            return self.speeds[self._axis(address, rh56.REG_SPEED_SET)]
        if rh56.REG_FORCE_SET <= address < rh56.REG_FORCE_SET + 12:
            return self.force_limits[self._axis(address, rh56.REG_FORCE_SET)]
        if rh56.REG_POS_SET <= address < rh56.REG_POS_SET + 12:
            return self.position_targets[self._axis(address, rh56.REG_POS_SET)]
        if rh56.REG_POS_ACT <= address < rh56.REG_POS_ACT + 12:
            return self.positions[self._axis(address, rh56.REG_POS_ACT)]
        if rh56.REG_FORCE_ACT <= address < rh56.REG_FORCE_ACT + 12:
            return self.forces[self._axis(address, rh56.REG_FORCE_ACT)]
        raise AssertionError(f"unexpected short read: {address}")

    def write_short(self, address: int, value: int) -> None:
        value = int(value)
        self.write_log.append((address, value))
        if rh56.REG_ANGLE_SET <= address < rh56.REG_ANGLE_SET + 12:
            axis = self._axis(address, rh56.REG_ANGLE_SET)
            self.angle_targets[axis] = value
            if value != -1:
                self.angles[axis] = self.actual_after_target.get(value, value)
            return
        if rh56.REG_SPEED_SET <= address < rh56.REG_SPEED_SET + 12:
            self.speeds[self._axis(address, rh56.REG_SPEED_SET)] = value
            return
        if rh56.REG_FORCE_SET <= address < rh56.REG_FORCE_SET + 12:
            self.force_limits[self._axis(address, rh56.REG_FORCE_SET)] = value
            return
        raise AssertionError(f"unexpected short write: {address}")

    def read_six_shorts(self, address: int, retries: int = 2):
        del retries
        if address == rh56.REG_ANGLE_SET:
            return tuple(self.angle_targets)
        if address == rh56.REG_ANGLE_ACT:
            return tuple(self.angles)
        if address == rh56.REG_POS_ACT:
            return tuple(self.positions)
        if address == rh56.REG_FORCE_ACT:
            return tuple(self.forces)
        if address == rh56.REG_CURRENT:
            return tuple(self.currents)
        raise AssertionError(f"unexpected six-short read: {address}")

    def write_six_shorts(self, address: int, values, retries: int = 1) -> None:
        del retries
        values = tuple(int(value) for value in values)
        if len(values) != 6:
            raise AssertionError(f"expected six values, got {values}")
        self.batch_write_log.append((address, values))
        if address != rh56.REG_ANGLE_SET:
            raise AssertionError(f"unexpected six-short write: {address}")
        self.angle_targets[:] = values
        for axis, value in enumerate(values):
            if value != -1:
                self.angles[axis] = self.actual_after_target.get(value, value)

    def read(self, address: int, length: int, retries: int = 2) -> bytes:
        del retries
        if address == rh56.REG_ERROR and length == 6:
            return bytes(self.errors)
        if address == rh56.REG_STATUS and length == 6:
            return bytes(self.statuses)
        if address == rh56.REG_TEMP and length == 6:
            return bytes(self.temperatures)
        raise AssertionError(f"unexpected byte read: address={address}, length={length}")


class ThumbRotateRealtimeOpenTests(unittest.TestCase):
    def test_single_way_open_reaches_983_and_releases_all_axes(self) -> None:
        hand = _FakeHand(
            thumb_angle=927,
            actual_after_target={1000: 983},
        )

        with mock.patch.object(rh56.time, "sleep", return_value=None):
            rh56.open_thumb_rotate_to_realtime_start(
                hand,
                speed=40,
                force_limit=80,
                motion_timeout=5,
            )

        numeric_targets = [
            values
            for address, values in hand.batch_write_log
            if address == rh56.REG_ANGLE_SET and any(value != -1 for value in values)
        ]
        self.assertEqual(
            numeric_targets,
            [(-1, -1, -1, -1, -1, 1000)],
        )
        self.assertEqual(tuple(hand.angle_targets), (-1,) * len(rh56.JOINTS))
        self.assertEqual(hand.angles, [1000, 1000, 1000, 1000, 1000, 983])
        self.assertEqual(hand.speeds[-1], 1000)
        self.assertEqual(hand.force_limits[-1], 259)

    def test_already_open_releases_without_numeric_target(self) -> None:
        hand = _FakeHand(thumb_angle=983)

        with mock.patch.object(rh56.time, "sleep", return_value=None):
            rh56.open_thumb_rotate_to_realtime_start(
                hand,
                speed=40,
                force_limit=80,
                motion_timeout=5,
            )

        self.assertFalse(
            any(
                any(value != -1 for value in values)
                for address, values in hand.batch_write_log
                if address == rh56.REG_ANGLE_SET
            )
        )
        self.assertEqual(tuple(hand.angle_targets), (-1,) * len(rh56.JOINTS))

    def test_preflight_rejects_uncleared_bend_without_writes(self) -> None:
        hand = _FakeHand(
            thumb_angle=927,
            bend_angles=(1000, 1000, 979, 1000, 1000),
        )

        with mock.patch.object(rh56.time, "sleep", return_value=None), self.assertRaisesRegex(
            rh56.RH56Error, "five bend axes completely open"
        ):
            rh56.open_thumb_rotate_to_realtime_start(
                hand,
                speed=40,
                force_limit=80,
                motion_timeout=5,
            )

        self.assertEqual(hand.write_log, [])
        self.assertEqual(hand.batch_write_log, [])

    def test_external_safety_fault_during_motion_disables_all_targets(self) -> None:
        hand = _FakeHand(
            thumb_angle=927,
            actual_after_target={1000: 950},
        )
        calls = 0

        def safety_check() -> None:
            nonlocal calls
            calls += 1
            if calls == 5:
                raise RuntimeError("Franka watchdog left Idle")

        with mock.patch.object(rh56.time, "sleep", return_value=None), self.assertRaisesRegex(
            RuntimeError, "Franka watchdog left Idle"
        ):
            rh56.open_thumb_rotate_to_realtime_start(
                hand,
                speed=40,
                force_limit=80,
                motion_timeout=5,
                safety_check=safety_check,
            )

        self.assertTrue(
            any(
                values[-1] == 1000
                for address, values in hand.batch_write_log
                if address == rh56.REG_ANGLE_SET
            )
        )
        self.assertEqual(tuple(hand.angle_targets), (-1,) * len(rh56.JOINTS))


class ThumbRotateNudgeRecoveryTests(unittest.TestCase):
    THUMB = rh56.JOINTS.index("thumb_rotate")
    THUMB_ANGLE_SET = rh56.REG_ANGLE_SET + 2 * THUMB

    @staticmethod
    def _nudge(
        hand: _FakeHand, delta: int, *, batch_angle_write: bool = False
    ) -> None:
        rh56.nudge(
            hand,
            ThumbRotateNudgeRecoveryTests.THUMB,
            delta,
            speed=40,
            force_limit=80,
            motion_timeout=0.01,
            batch_angle_write=batch_angle_write,
        )

    @classmethod
    def _thumb_angle_writes(cls, hand: _FakeHand) -> list[int]:
        return [
            value
            for address, value in hand.write_log
            if address == cls.THUMB_ANGLE_SET
        ]

    @staticmethod
    def _assert_pos_set_was_never_written(hand: _FakeHand) -> None:
        pos_addresses = range(rh56.REG_POS_SET, rh56.REG_POS_SET + 12, 2)
        assert not any(address in pos_addresses for address, _ in hand.write_log)
        assert not any(
            address == rh56.REG_POS_SET for address, _ in hand.batch_write_log
        )

    def test_thumb_nudge_requires_open_bends_and_all_six_disabled(self) -> None:
        cases = (
            (_FakeHand(bend_angles=(1000, 1000, 949, 1000, 1000)), "five bend"),
            (
                _FakeHand(angle_targets=(-1, -1, -1, 900, -1, -1)),
                "all six ANGLE_SET",
            ),
        )
        for hand, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                rh56.RH56Error, message
            ):
                self._nudge(hand, -20)
            self.assertEqual(hand.write_log, [])

    def test_thumb_nudge_rejects_non_idle_current_before_writes(self) -> None:
        selected_over = _FakeHand()
        selected_over.currents[self.THUMB] = 101
        total_over = _FakeHand()
        total_over.currents[:] = [40] * 6
        for hand, message in (
            (selected_over, "each idle current"),
            (total_over, "total idle current"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(
                rh56.RH56Error, message
            ):
                self._nudge(hand, -20)
            self.assertEqual(hand.write_log, [])

    def test_negative_path_is_885_to_865_to_885_to_minus_one(self) -> None:
        hand = _FakeHand(thumb_angle=885)
        with mock.patch.object(rh56.time, "sleep"):
            self._nudge(hand, -20)

        self.assertEqual(self._thumb_angle_writes(hand), [865, 885, -1])
        self.assertEqual(tuple(hand.angle_targets), (-1,) * 6)
        self.assertEqual(hand.angles[self.THUMB], 885)
        self.assertEqual(hand.speeds[self.THUMB], 1000)
        self.assertEqual(hand.force_limits[self.THUMB], 259)
        self._assert_pos_set_was_never_written(hand)

    def test_positive_path_from_859_to_879_is_legal(self) -> None:
        hand = _FakeHand(thumb_angle=859)
        with mock.patch.object(rh56.time, "sleep"):
            self._nudge(hand, 20)

        self.assertEqual(self._thumb_angle_writes(hand), [879, 859, -1])
        self.assertEqual(tuple(hand.angle_targets), (-1,) * 6)
        self._assert_pos_set_was_never_written(hand)

    def test_batch_path_matches_realtime_six_register_write(self) -> None:
        hand = _FakeHand(thumb_angle=855)
        with mock.patch.object(rh56.time, "sleep"):
            self._nudge(hand, 20, batch_angle_write=True)

        numeric_batches = [
            values
            for address, values in hand.batch_write_log
            if address == rh56.REG_ANGLE_SET and values[self.THUMB] != -1
        ]
        self.assertEqual(
            numeric_batches,
            [(-1, -1, -1, -1, -1, 875), (-1, -1, -1, -1, -1, 855)],
        )
        self.assertFalse(self._thumb_angle_writes(hand))
        self.assertEqual(tuple(hand.angle_targets), (-1,) * 6)
        self._assert_pos_set_was_never_written(hand)

    def test_nudge_does_not_accept_only_half_of_a_twenty_unit_step(self) -> None:
        hand = _FakeHand(thumb_angle=855, actual_after_target={875: 865})
        with mock.patch.object(rh56.time, "sleep"), self.assertRaisesRegex(
            rh56.RH56Error, "did not reach 875"
        ):
            self._nudge(hand, 20, batch_angle_write=True)

        self.assertEqual(tuple(hand.angle_targets), (-1,) * 6)

    def test_batch_nudge_rejects_inactive_axis_feedback_drift(self) -> None:
        hand = _FakeHand(thumb_angle=855)
        original_write = hand.write_six_shorts

        def write_with_drift(address, values, retries=1):
            original_write(address, values, retries=retries)
            if address == rh56.REG_ANGLE_SET and values[self.THUMB] != -1:
                hand.angles[0] += rh56.NUDGE_MAX_INACTIVE_ANGLE_DRIFT + 1

        hand.write_six_shorts = write_with_drift
        with mock.patch.object(rh56.time, "sleep"), self.assertRaisesRegex(
            rh56.RH56Error, "inactive joint.*ANGLE_ACT drifted"
        ):
            self._nudge(hand, 20, batch_angle_write=True)

        self.assertEqual(tuple(hand.angle_targets), (-1,) * 6)

    def test_recovery_never_rewrites_actual_angle_as_a_hold_target(self) -> None:
        hand = _FakeHand(thumb_angle=885, actual_after_target={865: 870})
        timeout = rh56.RH56Error("injected motion timeout")
        with mock.patch.object(
            rh56, "wait_for_angle", side_effect=timeout
        ), mock.patch.object(rh56.time, "sleep"), self.assertRaisesRegex(
            rh56.RH56Error, "injected motion timeout"
        ):
            self._nudge(hand, -20)

        self.assertEqual(self._thumb_angle_writes(hand), [865])
        self.assertNotIn(870, self._thumb_angle_writes(hand))
        self.assertEqual(tuple(hand.angle_targets), (-1,) * 6)
        self.assertTrue(
            any(
                address == rh56.REG_ANGLE_SET and values == (-1,) * 6
                for address, values in hand.batch_write_log
            )
        )

    def test_target_907_is_rejected_before_any_motion_write(self) -> None:
        hand = _FakeHand(thumb_angle=887)
        with self.assertRaisesRegex(ValueError, "907.*outside 800\.\.900"):
            self._nudge(hand, 20)

        self.assertEqual(hand.write_log, [])
        self.assertEqual(tuple(hand.angle_targets), (-1,) * 6)
        self._assert_pos_set_was_never_written(hand)

    def test_first_leg_timeout_stops_in_place_without_commanding_return(self) -> None:
        # The failed 865 target only reaches 870. Recovery must hold 870 and
        # disable it, rather than issuing the original 885 return target.
        hand = _FakeHand(thumb_angle=885, actual_after_target={865: 870})
        timeout = rh56.RH56Error("injected first-leg timeout")
        with mock.patch.object(rh56, "wait_for_angle", side_effect=timeout), mock.patch.object(
            rh56.time, "sleep"
        ), self.assertRaisesRegex(rh56.RH56Error, "first-leg timeout"):
            self._nudge(hand, -20)

        self.assertEqual(self._thumb_angle_writes(hand), [865])
        self.assertNotIn(885, self._thumb_angle_writes(hand)[1:])
        self.assertEqual(tuple(hand.angle_targets), (-1,) * 6)
        self.assertEqual(hand.speeds[self.THUMB], 1000)
        self.assertEqual(hand.force_limits[self.THUMB], 259)
        self._assert_pos_set_was_never_written(hand)

    def test_return_leg_timeout_stops_at_current_angle_without_retrying_return(self) -> None:
        # The normal return command 885 only reaches 880 before timing out.
        # Recovery may hold 880, but must not issue a second 885 command.
        hand = _FakeHand(thumb_angle=885, actual_after_target={885: 880})
        timeout = rh56.RH56Error("injected return-leg timeout")
        with mock.patch.object(
            rh56,
            "wait_for_angle",
            side_effect=(865, timeout),
        ), mock.patch.object(rh56.time, "sleep"), self.assertRaisesRegex(
            rh56.RH56Error, "return-leg timeout"
        ):
            self._nudge(hand, -20)

        self.assertEqual(self._thumb_angle_writes(hand), [865, 885])
        self.assertEqual(self._thumb_angle_writes(hand).count(885), 1)
        self.assertEqual(tuple(hand.angle_targets), (-1,) * 6)
        self.assertEqual(hand.speeds[self.THUMB], 1000)
        self.assertEqual(hand.force_limits[self.THUMB], 259)
        self._assert_pos_set_was_never_written(hand)


class ThumbRotateOneWayProbeTests(unittest.TestCase):
    THUMB = rh56.JOINTS.index("thumb_rotate")

    @staticmethod
    def _probe(hand: _FakeHand, timeout: float = 0.01) -> None:
        rh56.probe_thumb_rotate(
            hand,
            delta=30,
            speed=40,
            force_limit=80,
            observation_timeout=timeout,
        )

    def test_probe_detects_two_samples_then_disables_without_return(self) -> None:
        hand = _FakeHand(
            thumb_angle=850,
            actual_after_target={880: 854, 869: 854},
        )
        with mock.patch.object(rh56.time, "sleep"):
            self._probe(hand)

        numeric_batches = [
            values
            for address, values in hand.batch_write_log
            if address == rh56.REG_ANGLE_SET and values[self.THUMB] != -1
        ]
        self.assertEqual(
            numeric_batches,
            [
                (-1, -1, -1, -1, -1, 880),
                (-1, -1, -1, -1, -1, 869),
            ],
        )
        self.assertEqual(tuple(hand.angle_targets), (-1,) * 6)
        self.assertEqual(hand.angles[self.THUMB], 854)
        self.assertEqual(hand.speeds[self.THUMB], 1000)
        self.assertEqual(hand.force_limits[self.THUMB], 259)

    def test_probe_timeout_disables_directly_without_hold_or_return(self) -> None:
        hand = _FakeHand(thumb_angle=850, actual_after_target={880: 850})
        with mock.patch.object(rh56.time, "sleep"), self.assertRaisesRegex(
            rh56.RH56Error, "no measurable motion"
        ):
            self._probe(hand)

        numeric_batches = [
            values
            for address, values in hand.batch_write_log
            if address == rh56.REG_ANGLE_SET and values[self.THUMB] != -1
        ]
        self.assertEqual(numeric_batches, [(-1, -1, -1, -1, -1, 880)])
        self.assertEqual(tuple(hand.angle_targets), (-1,) * 6)

    def test_probe_rejects_wrong_direction_and_disables(self) -> None:
        hand = _FakeHand(thumb_angle=850, actual_after_target={880: 846})
        with mock.patch.object(rh56.time, "sleep"), self.assertRaisesRegex(
            rh56.RH56Error, "wrong direction"
        ):
            self._probe(hand)

        self.assertEqual(tuple(hand.angle_targets), (-1,) * 6)


class ThumbRotateVisualCycleTests(unittest.TestCase):
    THUMB = rh56.JOINTS.index("thumb_rotate")

    @staticmethod
    def _visual(hand: _FakeHand, timeout: float = 0.1) -> None:
        rh56.visual_cycle_thumb_rotate(
            hand,
            speed=40,
            force_limit=80,
            motion_timeout=timeout,
        )

    def test_visible_cycle_uses_bounded_commands_and_returns_to_start(self) -> None:
        hand = _FakeHand(
            thumb_angle=857,
            actual_after_target={855: 840, 885: 870, 872: 857},
        )
        with mock.patch.object(rh56.time, "sleep"):
            self._visual(hand)

        numeric_batches = [
            values
            for address, values in hand.batch_write_log
            if address == rh56.REG_ANGLE_SET and values[self.THUMB] != -1
        ]
        self.assertEqual(
            numeric_batches,
            [
                (-1, -1, -1, -1, -1, 855),
                (-1, -1, -1, -1, -1, 885),
                (-1, -1, -1, -1, -1, 855),
                (-1, -1, -1, -1, -1, 885),
                (-1, -1, -1, -1, -1, 872),
                (-1, -1, -1, -1, -1, 872),
            ],
        )
        self.assertEqual(hand.angles[self.THUMB], 857)
        self.assertEqual(tuple(hand.angle_targets), (-1,) * 6)
        self.assertEqual(hand.speeds[self.THUMB], 1000)
        self.assertEqual(hand.force_limits[self.THUMB], 259)

    def test_wrong_direction_skips_remaining_numeric_waypoints(self) -> None:
        hand = _FakeHand(thumb_angle=857, actual_after_target={855: 860})
        with mock.patch.object(rh56.time, "sleep"), self.assertRaisesRegex(
            rh56.RH56Error, "wrong direction"
        ):
            self._visual(hand)

        numeric_batches = [
            values
            for address, values in hand.batch_write_log
            if address == rh56.REG_ANGLE_SET and values[self.THUMB] != -1
        ]
        self.assertEqual(numeric_batches, [(-1, -1, -1, -1, -1, 855)])
        self.assertEqual(tuple(hand.angle_targets), (-1,) * 6)

    def test_no_response_timeout_skips_return_and_disables(self) -> None:
        hand = _FakeHand(thumb_angle=857, actual_after_target={855: 857})
        with mock.patch.object(rh56.time, "sleep"), self.assertRaisesRegex(
            rh56.RH56Error, "did not settle"
        ):
            self._visual(hand, timeout=0.01)

        numeric_batches = [
            values
            for address, values in hand.batch_write_log
            if address == rh56.REG_ANGLE_SET and values[self.THUMB] != -1
        ]
        self.assertEqual(numeric_batches, [(-1, -1, -1, -1, -1, 855)])
        self.assertEqual(tuple(hand.angle_targets), (-1,) * 6)

    def test_preflight_rejections_make_no_motion_write(self) -> None:
        cases = []
        bend_closed = _FakeHand(
            thumb_angle=857,
            bend_angles=(1000, 1000, 949, 1000, 1000),
        )
        cases.append((bend_closed, "five bend"))
        numeric_target = _FakeHand(
            thumb_angle=857,
            angle_targets=(-1, -1, -1, -1, -1, 872),
        )
        cases.append((numeric_target, "all six ANGLE_SET"))
        non_idle = _FakeHand(thumb_angle=857)
        non_idle.statuses[self.THUMB] = 1
        cases.append((non_idle, "STATUS"))
        hot = _FakeHand(thumb_angle=857)
        hot.temperatures[self.THUMB] = 50
        cases.append((hot, "50"))
        outside = _FakeHand(thumb_angle=839)
        cases.append((outside, "outside the validated"))

        for hand, message in cases:
            with self.subTest(message=message), mock.patch.object(
                rh56.time, "sleep"
            ), self.assertRaisesRegex(rh56.RH56Error, message):
                self._visual(hand)
            numeric_batches = [
                values
                for address, values in hand.batch_write_log
                if address == rh56.REG_ANGLE_SET and values[self.THUMB] != -1
            ]
            self.assertEqual(numeric_batches, [])


class ThumbRotateFullCycleTests(unittest.TestCase):
    THUMB = rh56.JOINTS.index("thumb_rotate")

    @staticmethod
    def _full(hand: _FakeHand) -> None:
        rh56.full_cycle_thumb_rotate(
            hand,
            speed=40,
            force_limit=80,
            motion_timeout=30,
        )

    @classmethod
    def _numeric_batches(cls, hand: _FakeHand):
        return [
            values
            for address, values in hand.batch_write_log
            if address == rh56.REG_ANGLE_SET and values[cls.THUMB] != -1
        ]

    def test_rated_cycle_is_open_close_open_and_finishes_open(self) -> None:
        hand = _FakeHand(
            thumb_angle=857,
            actual_after_target={1000: 985, 0: 0, 872: 857},
        )
        with mock.patch.object(rh56.time, "sleep"):
            self._full(hand)

        self.assertEqual(
            self._numeric_batches(hand),
            [
                (-1, -1, -1, -1, -1, 1000),
                (-1, -1, -1, -1, -1, 750),
                (-1, -1, -1, -1, -1, 500),
                (-1, -1, -1, -1, -1, 250),
                (-1, -1, -1, -1, -1, 0),
                (-1, -1, -1, -1, -1, 250),
                (-1, -1, -1, -1, -1, 500),
                (-1, -1, -1, -1, -1, 750),
                (-1, -1, -1, -1, -1, 1000),
            ],
        )
        self.assertEqual(hand.angles[self.THUMB], 985)
        self.assertEqual(tuple(hand.angle_targets), (-1,) * 6)
        self.assertEqual(hand.speeds[self.THUMB], 1000)
        self.assertEqual(hand.force_limits[self.THUMB], 259)
        angle_addresses = range(rh56.REG_ANGLE_SET, rh56.REG_ANGLE_SET + 12, 2)
        position_addresses = range(rh56.REG_POS_SET, rh56.REG_POS_SET + 12, 2)
        self.assertFalse(
            any(address in angle_addresses for address, _ in hand.write_log)
        )
        self.assertFalse(
            any(address in position_addresses for address, _ in hand.write_log)
        )
        self.assertFalse(
            any(
                address == rh56.REG_POS_SET
                for address, _ in hand.batch_write_log
            )
        )

    def test_function_enforces_fixed_low_speed_force_and_timeout(self) -> None:
        cases = (
            ({"speed": 41, "force_limit": 80, "motion_timeout": 30}, "speed=40"),
            ({"speed": 40, "force_limit": 81, "motion_timeout": 30}, "80g"),
            ({"speed": 40, "force_limit": 80, "motion_timeout": 29}, "30..180"),
        )
        for arguments, message in cases:
            hand = _FakeHand(thumb_angle=857)
            with self.subTest(arguments=arguments), self.assertRaisesRegex(
                rh56.RH56Error, message
            ):
                rh56.full_cycle_thumb_rotate(hand, **arguments)
            self.assertEqual(hand.write_log, [])
            self.assertEqual(hand.batch_write_log, [])

    def test_preflight_requires_all_five_bends_completely_open(self) -> None:
        hand = _FakeHand(
            thumb_angle=857,
            bend_angles=(1000, 1000, 979, 1000, 1000),
        )
        with mock.patch.object(rh56.time, "sleep"), self.assertRaisesRegex(
            rh56.RH56Error, "five bend axes completely open"
        ):
            self._full(hand)
        self.assertEqual(self._numeric_batches(hand), [])

    def test_wrong_direction_aborts_before_the_close_leg(self) -> None:
        hand = _FakeHand(thumb_angle=857, actual_after_target={1000: 850})
        with mock.patch.object(rh56.time, "sleep"), self.assertRaisesRegex(
            rh56.RH56Error, "wrong direction"
        ):
            self._full(hand)
        self.assertEqual(
            self._numeric_batches(hand),
            [(-1, -1, -1, -1, -1, 1000)],
        )
        self.assertEqual(tuple(hand.angle_targets), (-1,) * 6)

    def test_motion_current_limit_aborts_without_an_automatic_return(self) -> None:
        hand = _FakeHand(thumb_angle=857, actual_after_target={1000: 985})
        original_write = hand.write_six_shorts

        def write_and_inject_current(address, values, retries=1):
            original_write(address, values, retries=retries)
            if address == rh56.REG_ANGLE_SET and values[self.THUMB] == 1000:
                hand.currents[self.THUMB] = 401

        hand.write_six_shorts = write_and_inject_current
        with mock.patch.object(rh56.time, "sleep"), self.assertRaisesRegex(
            rh56.RH56Error, "MOTION STOP UNCONFIRMED"
        ):
            self._full(hand)
        self.assertEqual(
            self._numeric_batches(hand),
            [(-1, -1, -1, -1, -1, 1000)],
        )
        self.assertEqual(tuple(hand.angle_targets), (-1,) * 6)
        self.assertEqual(hand.speeds[self.THUMB], 40)
        self.assertEqual(hand.force_limits[self.THUMB], 80)

    def test_release_readback_is_not_accepted_while_feedback_keeps_moving(self) -> None:
        class MovingAfterReleaseHand(_FakeHand):
            def __init__(self):
                super().__init__(
                    thumb_angle=857,
                    actual_after_target={1000: 985, 0: 0, 872: 857},
                )
                self.saw_numeric = False
                self.released_after_numeric = False

            def write_six_shorts(self, address, values, retries=1):
                super().write_six_shorts(address, values, retries=retries)
                if address != rh56.REG_ANGLE_SET:
                    return
                if values[ThumbRotateFullCycleTests.THUMB] != -1:
                    self.saw_numeric = True
                elif self.saw_numeric:
                    self.released_after_numeric = True

            def read_six_shorts(self, address, retries=2):
                if address == rh56.REG_ANGLE_ACT and self.released_after_numeric:
                    self.angles[self.THUMB] += 2
                return super().read_six_shorts(address, retries=retries)

        hand = MovingAfterReleaseHand()
        with mock.patch.object(rh56.time, "sleep"), self.assertRaisesRegex(
            rh56.RH56Error, "MOTION STOP UNCONFIRMED"
        ):
            self._full(hand)
        self.assertEqual(tuple(hand.angle_targets), (-1,) * 6)
        self.assertEqual(hand.speeds[self.THUMB], 40)
        self.assertEqual(hand.force_limits[self.THUMB], 80)


class ThumbRotateFullCycleCliTests(unittest.TestCase):
    @staticmethod
    def _arguments(token: str):
        return [
            "inspire_rh56_test.py",
            "thumb-full-cycle",
            "--port",
            "/dev/fake-rh56",
            "--joint",
            "thumb_rotate",
            "--include-thumb-rotate",
            "--speed",
            "40",
            "--force-limit",
            "80",
            "--motion-timeout",
            "120",
            "--confirm-movement",
            "--confirm-rated-thumb-cycle",
            token,
        ]

    def test_complete_command_passes_cli_gates_and_reaches_device_open(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            self._arguments(rh56.THUMB_ROTATE_FULL_CYCLE_TOKEN),
        ), mock.patch.object(
            rh56.LinuxSerial,
            "__init__",
            side_effect=OSError("device-open-marker"),
        ) as serial_init:
            self.assertEqual(rh56.main(), 1)
        serial_init.assert_called_once()

    def test_wrong_exact_confirmation_token_is_rejected_before_device_open(self) -> None:
        with mock.patch.object(
            sys, "argv", self._arguments("WRONG_TOKEN")
        ), mock.patch.object(rh56.LinuxSerial, "__init__") as serial_init:
            self.assertEqual(rh56.main(), 2)
        serial_init.assert_not_called()


if __name__ == "__main__":
    unittest.main()
