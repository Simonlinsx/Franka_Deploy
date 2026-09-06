"""Reusable, live-state-anchored reset-to-open motion for an installed RH56.

This module is intentionally independent of the evidence-bound commissioning
and interrupted-recovery implementations.  Importing it performs no device
discovery and no hardware I/O.
"""

from __future__ import annotations

import math
from typing import Any, Tuple

from .inspire_sequence_driver import (
    DISABLED_TARGETS,
    RH56MotionStopped,
    RH56SequenceDriver,
    RH56SequenceDriverError,
    RH56StopUnconfirmed,
)


DEFAULT_SPEEDS: Tuple[int, ...] = (1000,) * 6
DEFAULT_FORCES: Tuple[int, ...] = (500,) * 6
RESET_BEND_OPEN_MIN_ANGLE = 980
RESET_Q6_SPEED = 40
# Normal execution reset uses the same installed-hand q6 speed validated for
# the 20 Hz deployment.  The slower value above remains the default for
# commissioning/recovery paths that require intermediate q6 evidence.
RESET_Q6_DIRECT_SPEED = 330
RESET_Q6_FORCE_G = 80
RESET_Q6_FIRST_STEP_UNITS = 50
RESET_Q6_STEP_UNITS = 25
RESET_Q6_ARRIVAL_TOLERANCE_UNITS = 30
RESET_Q6_DEADBAND_ESCAPE_UNITS = 50
# The installed q6 can snap to the physical lower endpoint 0 and ignore both
# +50 and +125 commands while idle/current-free.  A low-speed target of 400 is
# still inside the official register range and gives enough span to leave the
# endpoint; subsequent targets return to the ordinary 25-unit path.
RESET_Q6_LOWER_ENDPOINT_ESCAPE_UNITS = 400
# ANGLE_ACT can undershoot an in-range q6 command by the installed hand's
# measured command/feedback offset and can settle farther after ANGLE_SET is
# released.  Installed recovery evidence includes valid target 924 with
# feedback 874/877 and normal current, temperature, and error state.  This
# margin matches the reviewed arrival tolerance and is reset-only recovery
# admission: every numeric q6 target remains inside thumb_rotate_range.
RESET_Q6_FEEDBACK_RECOVERY_MARGIN_UNITS = 30


class RH56ResetSettingsRestoreError(RH56SequenceDriverError):
    """Motion is proven stopped, but default speed/force restore failed."""


def _bounded_integer(value: Any, name: str, lower: int, upper: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be an integer in {lower}..{upper}"
        ) from exc
    if (
        not math.isfinite(numeric)
        or numeric != round(numeric)
        or not lower <= int(numeric) <= upper
    ):
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return int(numeric)


class RH56ResetOpenDriver(RH56SequenceDriver):
    """Reset five bend axes and q6 to open from verified live feedback."""

    def __init__(
        self,
        hand: Any,
        api: Any,
        *,
        q6_open_min_angle: Any = None,
        q6_feedback_recovery_min_angle: Any = None,
        **driver_kwargs: Any,
    ) -> None:
        """Create a reset driver with a q6-specific physical-open band.

        ``ANGLE_SET=1000`` remains the commanded/simulation open endpoint.  On
        the installed RH56, however, the idle q6 ``ANGLE_ACT`` endpoint has an
        observed offset while the five bend axes reach the ordinary 980-unit
        open band.  Keeping the feedback thresholds separate prevents an idle
        q6 at its real endpoint from being moved merely to chase an unattainable
        feedback value.
        """

        driver_kwargs.setdefault("open_min_angle", RESET_BEND_OPEN_MIN_ANGLE)
        super().__init__(hand, api, **driver_kwargs)
        q6_lower, q6_upper = self.thumb_rotate_range
        configured = (
            self.open_min_angle
            if q6_open_min_angle is None
            else q6_open_min_angle
        )
        q6_open_min = _bounded_integer(
            configured,
            "q6_open_min_angle",
            q6_lower,
            q6_upper,
        )
        conservative_minimum = max(
            q6_lower,
            q6_upper - RESET_Q6_ARRIVAL_TOLERANCE_UNITS,
        )
        if q6_open_min < conservative_minimum:
            raise ValueError(
                "q6_open_min_angle exceeds the reset q6 arrival tolerance: "
                f"minimum={conservative_minimum}, actual={q6_open_min}"
            )
        self.q6_open_min_angle = q6_open_min
        recovery_configured = (
            q6_lower
            if q6_feedback_recovery_min_angle is None
            else q6_feedback_recovery_min_angle
        )
        self.q6_feedback_recovery_min_angle = _bounded_integer(
            recovery_configured,
            "q6_feedback_recovery_min_angle",
            max(0, q6_lower - RESET_Q6_FEEDBACK_RECOVERY_MARGIN_UNITS),
            q6_lower,
        )

    def reset_to_open(
        self,
        *,
        max_axis_current_ma: int = 400,
        endpoint_stable_samples: int = 3,
        max_inactive_drift_units: int = 8,
        simultaneous_bend_open: bool = False,
        direct_q6_endpoint_open: bool = False,
    ) -> Tuple[int, ...]:
        """Open all axes and leave them disabled.

        The reviewed five-bend helper runs first without touching q6.  Its
        default commissioning behavior is sequential; execution reset may set
        ``simultaneous_bend_open`` to issue one all-five endpoint transaction.
        Execution reset may also set ``direct_q6_endpoint_open`` so q6 receives
        exactly one endpoint target after the bends are clear.  Otherwise the
        evidence-oriented segmented q6 path below remains unchanged.  If q6 is
        below its configured physical-feedback open band, only the q6 slot
        receives a numeric target.  An idle q6 already inside that band remains
        disabled and is not moved merely to make ``ANGLE_ACT`` equal the
        commanded endpoint.
        In the segmented evidence path, a live q6 less than 50 units from 1000
        first backs off to escape the observed endpoint deadband, then returns
        from a fresh actual anchor using +25 commands.  The direct execution
        path instead accepts the commissioned physical endpoint tolerance.
        Every accepted sample remains range-, direction-, current-, fault-,
        inactive-axis-, and external-gate checked.

        The caller must invoke :meth:`disable_and_verify` in ``finally``.  That
        method also restores the all-six default speed/force settings declared
        above, including cleaning up a stale q6 40/80 commissioning setting.
        """

        stable_samples = _bounded_integer(
            endpoint_stable_samples, "endpoint_stable_samples", 2, 10
        )
        if not isinstance(simultaneous_bend_open, bool):
            raise ValueError("simultaneous_bend_open must be boolean")
        if not isinstance(direct_q6_endpoint_open, bool):
            raise ValueError("direct_q6_endpoint_open must be boolean")
        inactive_drift = _bounded_integer(
            max_inactive_drift_units, "max_inactive_drift_units", 1, 20
        )

        with self._operation_lock:
            try:
                self._ensure_motion_allowed()
                if not self._disabled_verified:
                    self.adopt_disabled_state_and_verify()

                # This is a reset-to-default operation, not a preserve-current-
                # settings operation.  In particular q6 may still contain the
                # temporary 40/80 values left by an interrupted acceptance run.
                self._original_speeds = DEFAULT_SPEEDS
                self._original_forces = DEFAULT_FORCES

                # Strictly validate the caller value and bind it to the live
                # device CURRENT_LIMIT registers before any recovery target is
                # written.  In particular, do not int() an unchecked bool or
                # fractional value before this point.
                current_caps = self._commissioning_current_caps(
                    max_axis_current_ma
                )

                # The reviewed helper opens only the five bend axes.  It never
                # enters the q6 helper, whose historical start gate is q6>=885.
                self._run_reviewed_open_helper(
                    include_thumb_rotate=False,
                    simultaneous_bend_open=simultaneous_bend_open,
                    # The helper accepts one scalar, so use the most
                    # conservative live per-axis cap.
                    max_axis_current_ma=min(current_caps),
                    endpoint_stable_samples=stable_samples,
                    max_inactive_drift_units=inactive_drift,
                )

                # Keep inactive bends at their normal defaults while q6 alone
                # uses the conservative no-contact reset settings.
                self._write_six_verified(
                    self._constant("REG_SPEED_SET"),
                    DEFAULT_SPEEDS[:5]
                    + (
                        RESET_Q6_DIRECT_SPEED
                        if direct_q6_endpoint_open
                        else RESET_Q6_SPEED,
                    ),
                    numeric_motion=True,
                )
                self._write_six_verified(
                    self._constant("REG_FORCE_SET"),
                    DEFAULT_FORCES[:5] + (RESET_Q6_FORCE_G,),
                    numeric_motion=True,
                )
                started = self._monotonic()
                preflight = self._read_feedback("reset_open_preflight", started)
                self._check_fault_feedback(preflight, "reset-open preflight")
                self._check_commissioning_currents(
                    preflight, current_caps, "reset-open preflight"
                )
                if preflight.angle_targets != DISABLED_TARGETS:
                    raise RH56SequenceDriverError(
                        "reset-open requires all-six ANGLE_SET=-1 before q6 motion"
                    )
                if any(angle < self.open_min_angle for angle in preflight.angles[:5]):
                    raise RH56SequenceDriverError(
                        "reset-open requires all five bend axes open; "
                        f"actual={preflight.angles[:5]}"
                    )
                if not all(status in (2, 0xFF) for status in preflight.statuses):
                    raise RH56SequenceDriverError(
                        "reset-open preflight requires all axes idle; "
                        f"actual={preflight.statuses}"
                    )

                inactive_reference = tuple(int(value) for value in preflight.angles[:5])
                previous_actual = int(preflight.angles[5])
                q6_lower, q6_upper = self.thumb_rotate_range
                q6_endpoint_arrival_min = max(
                    q6_lower,
                    q6_upper - RESET_Q6_ARRIVAL_TOLERANCE_UNITS,
                )
                # InterruptedRecoveryDriver deliberately reuses this method
                # without inheriting this class's constructor.  Preserve its
                # commissioned-range behavior unless it explicitly opts into
                # the narrow reset-only feedback recovery band.
                q6_feedback_recovery_min = int(
                    getattr(
                        self,
                        "q6_feedback_recovery_min_angle",
                        q6_lower,
                    )
                )
                if q6_upper != 1000:
                    raise RH56SequenceDriverError(
                        "reset-open requires a configured q6 range whose open "
                        f"endpoint is 1000; actual={q6_lower}..{q6_upper}"
                    )
                if not (
                    q6_feedback_recovery_min
                    <= previous_actual
                    <= q6_upper
                ):
                    raise RH56SequenceDriverError(
                        "reset-open q6 feedback is outside the bounded "
                        "recovery range "
                        f"{q6_feedback_recovery_min}..{q6_upper}: "
                        f"{previous_actual}"
                    )

                def check_motion_feedback(
                    feedback: Any,
                    phase: str,
                    expected: Tuple[int, ...],
                ) -> Tuple[int, int]:
                    self._check_fault_feedback(feedback, phase)
                    self._check_commissioning_currents(feedback, current_caps, phase)
                    if feedback.angle_targets != expected:
                        raise RH56SequenceDriverError(
                            f"{phase}: ANGLE_SET changed unexpectedly: "
                            f"{feedback.angle_targets}"
                        )
                    bend_moved = any(
                        angle < self.open_min_angle
                        or abs(int(angle) - reference)
                        > inactive_drift
                        for angle, reference in zip(
                            feedback.angles[:5], inactive_reference
                        )
                    )
                    bend_not_idle = not all(
                        status in (2, 0xFF) for status in feedback.statuses[:5]
                    )
                    if bend_moved or bend_not_idle:
                        raise RH56SequenceDriverError(
                            f"{phase}: an inactive bend axis moved; "
                            f"angles={feedback.angles[:5]}, "
                            f"statuses={feedback.statuses[:5]}"
                        )
                    q6_status = int(feedback.statuses[5])
                    if q6_status == 3:
                        raise RH56SequenceDriverError(
                            f"{phase}: q6 reported force contact in free air"
                        )
                    if q6_status not in (0, 1, 2):
                        raise RH56SequenceDriverError(
                            f"{phase}: unsupported q6 status {q6_status}"
                        )
                    q6_actual = int(feedback.angles[5])
                    if not (
                        q6_feedback_recovery_min
                        <= q6_actual
                        <= q6_upper
                    ):
                        raise RH56SequenceDriverError(
                            f"{phase}: q6 ANGLE_ACT={q6_actual} escaped bounded "
                            "recovery range "
                            f"{q6_feedback_recovery_min}..{q6_upper}"
                        )
                    return q6_actual, q6_status

                commanded = []
                first = True
                needs_q6_motion = previous_actual < self.q6_open_min_angle
                previous_command = previous_actual
                force_direct_endpoint = False

                if direct_q6_endpoint_open and needs_q6_motion:
                    target = q6_upper
                    expected_targets = (-1, -1, -1, -1, -1, target)
                    phase = "reset_open_q6_direct_1000"
                    self._run_external_safety_check(
                        phase, "before numeric target write"
                    )
                    self._write_six_verified(
                        self._constant("REG_ANGLE_SET"),
                        expected_targets,
                        numeric_motion=True,
                    )
                    commanded.append(target)
                    deadline = self._monotonic() + self.motion_timeout_s
                    stable = 0
                    while True:
                        if self._stop_requested.is_set():
                            raise RH56MotionStopped(
                                "direct q6 reset-open was interrupted by a "
                                "stop request"
                            )
                        feedback = self._read_feedback(phase, started)
                        q6_actual, q6_status = check_motion_feedback(
                            feedback, phase, expected_targets
                        )
                        # ANGLE_ACT is not monotonic while q6 is moving: the
                        # installed hand can briefly report the endpoint and
                        # then settle back by tens of units because of sensor
                        # quantization/backlash.  A one-sample direction test
                        # therefore rejects healthy direct-open motion.  The
                        # execution reset instead relies on the stronger
                        # conditions below: exact target ownership, bounded
                        # feedback, healthy current/error/status, timeout, and
                        # consecutive idle samples inside the open band.
                        reached = (
                            q6_status == 2
                            and q6_actual >= q6_endpoint_arrival_min
                        )
                        stable = stable + 1 if reached else 0
                        if stable >= stable_samples:
                            previous_actual = q6_actual
                            break
                        if self._monotonic() >= deadline:
                            raise RH56SequenceDriverError(
                                f"{phase}: timed out at q6 "
                                f"ANGLE_ACT={q6_actual}, target={target}, "
                                f"status={q6_status}"
                            )
                        self._sleep(self.poll_interval_s)
                    # The direct execution path has completed the only q6
                    # numeric command.  Leave the legacy segmented loop to the
                    # commissioning/recovery mode below.
                    needs_q6_motion = False

                # When the open-side residual is smaller than the known
                # effective command span, a direct 1000 command can remain
                # physically idle (the observed 973 -> 1000 case).  Back off
                # just far enough that both the descending command and the
                # later direct-open command retain at least a 50-unit span,
                # even at the accepted +30 feedback tolerance.
                if (
                    needs_q6_motion
                    and q6_upper - previous_actual
                    < RESET_Q6_DEADBAND_ESCAPE_UNITS
                ):
                    backoff_target = max(
                        q6_lower,
                        min(
                            previous_actual - RESET_Q6_DEADBAND_ESCAPE_UNITS,
                            q6_upper
                            - RESET_Q6_DEADBAND_ESCAPE_UNITS
                            - RESET_Q6_ARRIVAL_TOLERANCE_UNITS,
                        ),
                    )
                    backoff_span = previous_actual - backoff_target
                    if backoff_span < RESET_Q6_DEADBAND_ESCAPE_UNITS:
                        raise RH56SequenceDriverError(
                            "reset-open cannot escape the q6 endpoint deadband "
                            f"inside configured range {q6_lower}..{q6_upper}: "
                            f"ANGLE_ACT={previous_actual}, candidate="
                            f"{backoff_target}, required_span="
                            f"{RESET_Q6_DEADBAND_ESCAPE_UNITS}"
                        )
                    expected_targets = (
                        -1,
                        -1,
                        -1,
                        -1,
                        -1,
                        backoff_target,
                    )
                    phase = f"reset_open_q6_backoff_{backoff_target:04d}"
                    self._run_external_safety_check(
                        phase, "before numeric target write"
                    )
                    self._write_six_verified(
                        self._constant("REG_ANGLE_SET"),
                        expected_targets,
                        numeric_motion=True,
                    )
                    commanded.append(backoff_target)
                    deadline = self._monotonic() + self.motion_timeout_s
                    stable = 0
                    last_actual = previous_actual
                    minimum_progress = max(
                        3,
                        backoff_span - RESET_Q6_ARRIVAL_TOLERANCE_UNITS,
                    )
                    opposite_slack = max(4, min(12, self.angle_tolerance // 2))
                    while True:
                        if self._stop_requested.is_set():
                            raise RH56MotionStopped(
                                "q6 reset-open backoff was interrupted by a stop request"
                            )
                        feedback = self._read_feedback(phase, started)
                        q6_actual, q6_status = check_motion_feedback(
                            feedback, phase, expected_targets
                        )
                        if (
                            q6_actual > previous_actual + opposite_slack
                            or q6_actual > last_actual + opposite_slack
                        ):
                            raise RH56SequenceDriverError(
                                f"{phase}: q6 moved opposite the descending "
                                f"backoff command; previous={last_actual}, "
                                f"actual={q6_actual}"
                            )
                        last_actual = q6_actual
                        reached = (
                            q6_status == 2
                            and abs(q6_actual - backoff_target)
                            <= RESET_Q6_ARRIVAL_TOLERANCE_UNITS
                            and previous_actual - q6_actual >= minimum_progress
                            and q6_upper - q6_actual
                            >= RESET_Q6_DEADBAND_ESCAPE_UNITS
                        )
                        stable = stable + 1 if reached else 0
                        if stable >= stable_samples:
                            break
                        if self._monotonic() >= deadline:
                            raise RH56SequenceDriverError(
                                f"{phase}: timed out at q6 ANGLE_ACT={q6_actual}, "
                                f"target={backoff_target}, status={q6_status}"
                            )
                        self._sleep(self.poll_interval_s)

                    # Direction reversal is never made from a stale numeric
                    # hold.  Prove all six targets disabled/idle, then take a
                    # fresh external-gated feedback anchor for the open sweep.
                    for _pass_index in (1, 2):
                        self._write_disable_pass()
                    self._verify_disabled_feedback(
                        phase="reset_open_q6_backoff_disable_verify"
                    )
                    anchor_phase = "reset_open_q6_forward_anchor"
                    anchor = self._read_feedback(anchor_phase, started)
                    self._check_fault_feedback(anchor, anchor_phase)
                    self._check_commissioning_currents(
                        anchor, current_caps, anchor_phase
                    )
                    if anchor.angle_targets != DISABLED_TARGETS:
                        raise RH56SequenceDriverError(
                            f"{anchor_phase}: all-six ANGLE_SET is not disabled: "
                            f"{anchor.angle_targets}"
                        )
                    if any(
                        angle < self.open_min_angle
                        or abs(int(angle) - reference)
                        > inactive_drift
                        for angle, reference in zip(
                            anchor.angles[:5], inactive_reference
                        )
                    ) or not all(
                        status in (2, 0xFF) for status in anchor.statuses[:5]
                    ):
                        raise RH56SequenceDriverError(
                            f"{anchor_phase}: an inactive bend axis moved or "
                            "left idle"
                        )
                    if int(anchor.statuses[5]) not in (2, 0xFF):
                        raise RH56SequenceDriverError(
                            f"{anchor_phase}: q6 is not idle after backoff disable; "
                            f"status={anchor.statuses[5]}"
                        )
                    previous_actual = int(anchor.angles[5])
                    if not (
                        q6_feedback_recovery_min
                        <= previous_actual
                        <= q6_upper
                    ):
                        raise RH56SequenceDriverError(
                            f"{anchor_phase}: q6 ANGLE_ACT={previous_actual} "
                            "escaped bounded recovery range "
                            f"{q6_feedback_recovery_min}..{q6_upper}"
                        )
                    if (
                        abs(previous_actual - backoff_target)
                        > RESET_Q6_ARRIVAL_TOLERANCE_UNITS
                        or q6_upper - previous_actual
                        < RESET_Q6_DEADBAND_ESCAPE_UNITS
                    ):
                        raise RH56SequenceDriverError(
                            f"{anchor_phase}: q6 no longer has a verified "
                            "deadband-escape anchor; ANGLE_ACT="
                            f"{previous_actual}, target={backoff_target}"
                        )
                    previous_command = backoff_target
                    first = False
                    force_direct_endpoint = True

                while needs_q6_motion:
                    if force_direct_endpoint:
                        target = q6_upper
                    else:
                        if (
                            first
                            and q6_lower == 0
                            and previous_actual
                            <= q6_lower + RESET_Q6_ARRIVAL_TOLERANCE_UNITS
                        ):
                            increment = RESET_Q6_LOWER_ENDPOINT_ESCAPE_UNITS
                        else:
                            increment = (
                                RESET_Q6_FIRST_STEP_UNITS
                                if first
                                else RESET_Q6_STEP_UNITS
                            )
                        target = min(q6_upper, previous_command + increment)
                    expected_targets = (-1, -1, -1, -1, -1, target)
                    phase = f"reset_open_q6_{len(commanded):04d}_{target:04d}"
                    if not q6_lower <= target <= q6_upper:
                        raise RH56SequenceDriverError(
                            f"{phase}: target escaped configured q6 range "
                            f"{q6_lower}..{q6_upper}"
                        )
                    if (
                        force_direct_endpoint
                        and target - previous_actual
                        < RESET_Q6_DEADBAND_ESCAPE_UNITS
                    ):
                        raise RH56SequenceDriverError(
                            f"{phase}: fresh q6 opening span is below "
                            f"{RESET_Q6_DEADBAND_ESCAPE_UNITS}; "
                            f"ANGLE_ACT={previous_actual}, target={target}"
                        )
                    self._run_external_safety_check(
                        phase, "before numeric target write"
                    )
                    self._write_six_verified(
                        self._constant("REG_ANGLE_SET"),
                        expected_targets,
                        numeric_motion=True,
                    )
                    commanded.append(target)
                    deadline = self._monotonic() + self.motion_timeout_s
                    stable = 0
                    last_feedback = None
                    last_actual = previous_actual
                    opposite_slack = max(4, min(12, self.angle_tolerance // 2))
                    minimum_progress = max(
                        3,
                        target
                        - previous_actual
                        - RESET_Q6_ARRIVAL_TOLERANCE_UNITS,
                    )
                    while True:
                        if self._stop_requested.is_set():
                            raise RH56MotionStopped(
                                "q6 reset-open was interrupted by a stop request"
                            )
                        feedback = self._read_feedback(phase, started)
                        last_feedback = feedback
                        q6_actual, q6_status = check_motion_feedback(
                            feedback, phase, expected_targets
                        )
                        lower = max(
                            q6_feedback_recovery_min,
                            previous_actual - 12,
                        )
                        upper = min(
                            q6_upper,
                            target + RESET_Q6_ARRIVAL_TOLERANCE_UNITS,
                        )
                        if not lower <= q6_actual <= upper:
                            raise RH56SequenceDriverError(
                                f"{phase}: q6 ANGLE_ACT={q6_actual} escaped "
                                f"actual-anchored interval {lower}..{upper}"
                            )
                        if q6_actual < last_actual - opposite_slack:
                            raise RH56SequenceDriverError(
                                f"{phase}: q6 moved opposite the ascending "
                                f"command; previous={last_actual}, "
                                f"actual={q6_actual}"
                            )
                        last_actual = q6_actual
                        if target == q6_upper:
                            # Use the reset driver's existing endpoint
                            # tolerance consistently.  Reapplying the profile
                            # 975 floor here made healthy status-2 feedback at
                            # 974 time out after an exact 1000 write.
                            reached = (
                                q6_status == 2
                                and q6_actual >= q6_endpoint_arrival_min
                            )
                        else:
                            reached = (
                                q6_status == 2
                                and abs(q6_actual - target)
                                <= RESET_Q6_ARRIVAL_TOLERANCE_UNITS
                                and q6_actual - previous_actual >= minimum_progress
                            )
                        stable = stable + 1 if reached else 0
                        if stable >= stable_samples:
                            previous_actual = q6_actual
                            break
                        if self._monotonic() >= deadline:
                            raise RH56SequenceDriverError(
                                f"{phase}: timed out at q6 ANGLE_ACT={q6_actual}, "
                                f"target={target}, status={q6_status}"
                            )
                        self._sleep(self.poll_interval_s)
                    if last_feedback is None:
                        raise RH56SequenceDriverError(
                            f"{phase}: no feedback sample was received"
                        )
                    first = False
                    force_direct_endpoint = False
                    previous_command = target
                    if target == q6_upper:
                        break

                # Even an already-open q6 gets an explicit endpoint command only
                # when it is below the verified open threshold.  Always finish
                # with two all-six disable writes and stable idle verification.
                for _pass_index in (1, 2):
                    self._write_disable_pass()
                self._verify_disabled_feedback(phase="reset_open_disable_verify")
                final = self._read_feedback("reset_open_final", started)
                self._check_fault_feedback(final, "reset-open final")
                self._check_commissioning_currents(
                    final, current_caps, "reset-open final"
                )
                if final.angle_targets != DISABLED_TARGETS:
                    raise RH56SequenceDriverError(
                        f"reset-open final targets are not disabled: {final.angle_targets}"
                    )
                bend_not_open = any(
                    angle < self.open_min_angle for angle in final.angles[:5]
                )
                # q6 has already passed either the active endpoint proof or
                # the live already-open preflight.  After ANGLE_SET=-1, do
                # not repeat an absolute endpoint floor: encoder release and
                # backlash are not a failed reset.  Reject only unexpected
                # movement relative to the freshly proven q6 state, using the
                # same inactive-axis drift bound as the bend axes.
                q6_release_drift = abs(
                    int(final.angles[5]) - int(previous_actual)
                )
                if bend_not_open or q6_release_drift > inactive_drift:
                    raise RH56SequenceDriverError(
                        "reset-open final angles are not open: "
                        f"actual={final.angles}, bend_min={self.open_min_angle}, "
                        f"q6_release_drift={q6_release_drift}, "
                        f"allowed_drift={inactive_drift}"
                    )
                if not all(status in (2, 0xFF) for status in final.statuses):
                    raise RH56SequenceDriverError(
                        f"reset-open final statuses are not idle: {final.statuses}"
                    )
                self._numeric_hold_targets = None
                self._disabled_verified = True
                self.last_contact_axes = ()
                return tuple(commanded)
            except BaseException as exc:
                self._fail_and_latch(exc)
                raise

    def disable_and_verify(self) -> None:
        """Stop first, then restore defaults without conflating the two proofs.

        The base driver deliberately treats any cleanup failure as
        ``STOP UNCONFIRMED``.  For this reusable reset command we retain the
        stronger distinction operators need: a failed settings write after a
        verified all-six disable is an error, but it is not a reason by itself
        to cut 24 V.
        """

        self.request_stop()
        with self._operation_lock:
            stop_failures = []
            for pass_index in (1, 2):
                try:
                    self._write_disable_pass()
                except BaseException as exc:
                    stop_failures.append(f"disable pass {pass_index}: {exc}")
            try:
                self._verify_disabled_feedback(
                    allow_external_gate_bypass_after_disabled_readback=True
                )
            except BaseException as exc:
                stop_failures.append(f"physical stop verification: {exc}")
            if stop_failures:
                self._disabled_verified = False
                raise RH56StopUnconfirmed(
                    "STOP UNCONFIRMED: " + "; ".join(stop_failures)
                )

            self._preshaped_q6 = None
            self._commissioned_q6_waypoints = None
            self._commissioned_bend_waypoints = None
            self._numeric_hold_targets = None
            self._disabled_verified = True
            self.last_contact_axes = ()
            try:
                self._restore_original_settings()
            except BaseException as exc:
                raise RH56ResetSettingsRestoreError(
                    "all-six disable/idle is verified, but default speed/force "
                    f"restore failed: {exc}"
                ) from exc

    def close(self) -> None:
        """Close while preserving the verified-stop/cleanup distinction."""

        if self._closed:
            return
        stop_error = None
        cleanup_failures = []
        try:
            self.disable_and_verify()
        except RH56StopUnconfirmed as exc:
            stop_error = exc
        except BaseException as exc:
            cleanup_failures.append(str(exc))
        try:
            if self._serial_context is not None:
                self._serial_context.__exit__(None, None, None)
        except BaseException as exc:
            cleanup_failures.append(f"serial close failed: {exc}")
        finally:
            self._closed = True
        if stop_error is not None:
            raise stop_error
        if cleanup_failures:
            raise RH56SequenceDriverError(
                "physical stop is verified, but cleanup failed: "
                + "; ".join(cleanup_failures)
            )


__all__ = [
    "DEFAULT_FORCES",
    "DEFAULT_SPEEDS",
    "RESET_Q6_ARRIVAL_TOLERANCE_UNITS",
    "RESET_Q6_DEADBAND_ESCAPE_UNITS",
    "RESET_Q6_FIRST_STEP_UNITS",
    "RESET_Q6_FORCE_G",
    "RESET_Q6_SPEED",
    "RESET_Q6_STEP_UNITS",
    "RH56ResetOpenDriver",
    "RH56ResetSettingsRestoreError",
]
